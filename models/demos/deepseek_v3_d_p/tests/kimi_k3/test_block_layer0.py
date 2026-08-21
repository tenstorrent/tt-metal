# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 layer 0 on device against the torch reference, on real checkpoint weights.

Layer 0 is the cheapest real-weight gate in the model: it is dense (``first_k_dense_replace=1``)
and KDA, so every weight it touches is stored unquantized and no MXFP4 dequantizer is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_k3_mla import KimiRMSNorm
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.utils import assert_accurate, reconstruct_sp_tp_tensor
from models.demos.deepseek_v3_d_p.tests.kimi_k3.harness import (
    SP_AXIS,
    TP_AXIS,
    build_layer_0,
    kda_sequence_length,
    layer_cache_path,
    shard_activation,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.weights import load_dense_block_state_dict
from models.demos.deepseek_v3_d_p.tt.kda.state_store import KdaStateStore
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
    pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True),
]


def residual_stream(seq_len: int) -> torch.Tensor:
    """A deterministic stand-in for the embedding output entering layer 0."""
    return torch.randn(
        1,
        1,
        seq_len,
        KimiK3Config.EMB_SIZE,
        generator=torch.Generator().manual_seed(20260821),
        dtype=torch.bfloat16,
    )


def test_kimi_k3_layer_0_kda_matches_torch_reference(
    mesh_device: ttnn.MeshDevice,
    kimi_k3_checkpoint_dir: Path,
    kimi_k3_tt_cache_root: Path,
) -> None:
    """PCC the block's KDA attention output against ``norm -> kda_forward_reference`` in torch.

    This drives ``attn_norm`` and ``_kda_path`` rather than ``forward``, because ``forward`` folds
    the attention output into the residual and on through the FFN. It is still the whole
    integration under test: the tensor-parallel gather into KDA's full-width input projection, the
    recurrent carry the store hands over, and the reduce-scattered residual-shaped result.
    """
    seq_len = kda_sequence_length(mesh_device)
    hidden = residual_stream(seq_len)

    block, state_dict = build_layer_0(mesh_device, kimi_k3_checkpoint_dir, kimi_k3_tt_cache_root, seq_len)
    assert block.is_kda, "layer 0 must be a KDA layer"

    norm = KimiRMSNorm(KimiK3Config.EMB_SIZE, eps=KimiK3Config.RMS_NORM_EPS)
    with torch.no_grad():
        norm.weight.copy_(state_dict["attn_norm_weight"])
        normed = norm(hidden.reshape(1, seq_len, KimiK3Config.EMB_SIZE))
    golden, _ = kda_forward_reference(normed, state_dict["kda_weights"], kimi_k3_kda_config())

    kda_states = KdaStateStore({0: block.kda})
    attn_norm_out = block.attn_norm(shard_activation(mesh_device, hidden))
    attn_out = block._kda_path(attn_norm_out, kda_states)

    actual = reconstruct_sp_tp_tensor(attn_out, mesh_device, SP_AXIS, TP_AXIS, tp_dim=2, sp_dim=1)
    assert_accurate(golden, actual.to(golden.dtype), name="Kimi-K3 layer 0 attn output", pcc_threshold=0.98)


def test_kimi_k3_single_layer_transformer_forward(
    mesh_device: ttnn.MeshDevice,
    kimi_k3_checkpoint_dir: Path,
    kimi_k3_tt_cache_root: Path,
) -> None:
    """Run a one-layer ``TtPrefillTransformer`` over layer 0 and check what only it owns.

    Not an accuracy gate: the block's dense FFN applies SiLU where Kimi-K3 applies ``situ``, so the
    layer output is not comparable to anything yet. What this does exercise is the wiring the
    transformer adds around a KDA layer — the MLA-relative KV-slot map, and the state store that
    makes the recurrence something the caller carries across chunks.

    Built as a middle pipeline rank so the tail is skipped: the embedding and LM head are 2.4 GB of
    vocab weights each and neither is on the path being checked.
    """
    seq_len = kda_sequence_length(mesh_device)
    config = kimi_k3_hf_config(max_seq=seq_len)
    state_dict = {"layers": [load_dense_block_state_dict(kimi_k3_checkpoint_dir, layer_idx=0)]}
    mesh_shape = tuple(mesh_device.shape)

    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=KimiK3Config,
        state_dict=state_dict,
        num_layers=1,
        seq_len=seq_len,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        weight_cache_path=layer_cache_path(mesh_device, kimi_k3_tt_cache_root),
        first_layer_idx=0,
        is_first_rank=False,
        is_last_rank=False,
    )
    assert transformer.kv_slot_of_layer == [None], "layer 0 is KDA and owns no KV slot"
    assert transformer.num_mla_layers == 0
    assert transformer.kda_states is not None

    # Layer 0 writes no KV, but forward() takes a cache unconditionally; one slot is the smallest
    # allocation that is still a cache.
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=1,
    )

    x = shard_activation(mesh_device, residual_stream(seq_len))
    output = transformer.forward(x, kvpe_cache, actual_isl=seq_len)
    assert tuple(output.shape) == (
        1,
        1,
        seq_len // mesh_shape[SP_AXIS],
        KimiK3Config.EMB_SIZE // mesh_shape[TP_AXIS],
    )
    host_output = reconstruct_sp_tp_tensor(output, mesh_device, SP_AXIS, TP_AXIS, tp_dim=2, sp_dim=1)
    assert torch.isfinite(host_output.float()).all(), "layer output is not finite"

    # The recurrence has to have advanced: a store that was never written back still holds the
    # zeros ttKDA allocated, and every later chunk would restart the layer's history.
    recurrent = ttnn.to_torch(ttnn.get_device_tensors(transformer.kda_states.get(0).recurrent)[0])
    assert recurrent.float().abs().sum() > 0, "KDA recurrent carry was not written back to the store"
