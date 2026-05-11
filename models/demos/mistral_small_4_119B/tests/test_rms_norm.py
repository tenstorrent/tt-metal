# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm parity tests for Mistral Small 4.

Validates both :class:`DistributedRMSNorm` (pre_all_gather + all_gather + post_all_gather)
and :class:`RMSNorm` (single-device ``ttnn.rms_norm``) against torch reference.

On single-device meshes, the all_gather in ``DistributedRMSNorm`` is a no-op
(mesh extent ≤ 1), so both classes are exercisable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[4]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
try:
    from tests.scripts.ompi_singleton_env import apply_ompi_singleton_workaround_env

    apply_ompi_singleton_workaround_env()
except ImportError:
    if os.environ.get("TT_METAL_OMPI_SINGLETON_WORKAROUND", "1") != "0":
        os.environ.setdefault("OMPI_MCA_plm", "isolated")
        os.environ.setdefault("PRTE_MCA_plm", "isolated")

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_small_4_119B.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.mistral_small_4_119B.tt.rms_norm.rms_norm import RMSNorm
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.run_config import create_run_config, deallocate_weight_config_tensors

# ── Helpers ──────────────────────────────────────────────────────────────


def _tiny_mistral4_config():
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=128,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_rope_head_dim=32,
        v_head_dim=32,
        qk_nope_head_dim=32,
        rms_norm_eps=1e-6,
    )


def _assert_pcc(tt_output: torch.Tensor, reference_output: torch.Tensor, *, pcc_required: float) -> None:
    """PCC comparison matching hidden dimension, tolerating leading singleton dims."""
    tt_out = tt_output.cpu().float()
    ref_out = reference_output.cpu().float()

    while tt_out.ndim < ref_out.ndim:
        tt_out = tt_out.unsqueeze(0)
    while ref_out.ndim < tt_out.ndim:
        ref_out = ref_out.unsqueeze(0)

    # Align dimensions
    hidden = min(tt_out.shape[-1], ref_out.shape[-1])
    seq = min(tt_out.shape[-2], ref_out.shape[-2])
    tt_out = tt_out[..., :seq, :hidden]
    ref_out = ref_out[..., :seq, :hidden]

    passing, pcc = comp_pcc(tt_out, ref_out, pcc_required)
    logger.info(f"RMSNorm PCC: {pcc}")
    assert passing, f"PCC {pcc} < required {pcc_required}"


def _run_rms_norm_test(
    *,
    RMSNormClass,
    hidden_size: int,
    mode: str,
    seq_len: int,
    batch_size_per_row: int,
    hf_config,
    tmp_path: Path,
    mesh_device,
):
    """Core test logic shared across parametrized tests."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    num_module_layers = mesh_device.shape[0]

    # ── Reference model ──────────────────────────────────────────────
    reference_model = Mistral4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps).eval()
    state_dict = reference_model.to(torch.bfloat16).state_dict()

    torch_input = torch.randn(num_module_layers, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    reference_output = reference_model.float()(torch_input.float())

    # ── Weight conversion ────────────────────────────────────────────
    state_dicts = (state_dict,) * num_module_layers
    weight_config = RMSNormClass.convert_weights(
        hf_config,
        state_dicts,
        tmp_path / f"rms_norm_{RMSNormClass.__name__}_{mode}",
        mesh_device,
    )

    # ── Model config ─────────────────────────────────────────────────
    if mode == "decode":
        model_config = RMSNormClass.decode_model_config(hf_config, mesh_device, batch_size_per_row=batch_size_per_row)
    else:
        model_config = RMSNormClass.prefill_model_config(hf_config, mesh_device)

    # ── Model state ──────────────────────────────────────────────────
    if RMSNormClass is DistributedRMSNorm:
        ccl = CCL(mesh_device)
        model_state = DistributedRMSNorm.create_state(hf_config, mesh_device, ccl)
    else:
        model_state = {"mesh_device": mesh_device}

    # ── Run config ───────────────────────────────────────────────────
    run_config = create_run_config(model_config, weight_config, model_state)

    # ── Send input to device ─────────────────────────────────────────
    if RMSNormClass is DistributedRMSNorm:
        shard_dims = (0, -1)
    else:
        shard_dims = (0, None)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # ── Forward pass ─────────────────────────────────────────────────
    try:
        if mode == "decode":
            tt_output = RMSNormClass.forward_decode(tt_input, run_config)
        else:
            tt_output = RMSNormClass.forward_prefill(tt_input, run_config)

        # ── Convert back and compare ─────────────────────────────────
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
        )

        # For non-distributed RMSNorm, the last dim is not sharded across columns
        if RMSNormClass is RMSNorm:
            tt_output_torch = tt_output_torch[..., :hidden_size]

        _assert_pcc(tt_output_torch, reference_output, pcc_required=0.98)
    finally:
        ttnn.deallocate(tt_input)
        if "tt_output" in dir() and tt_output is not None:
            ttnn.deallocate(tt_output)
        deallocate_weight_config_tensors(weight_config)


# ── Tests: DistributedRMSNorm ────────────────────────────────────────────


@pytest.mark.parametrize("mode, seq_len", [("decode", 8), ("prefill", 64)])
def test_distributed_rms_norm_hidden_size(device, tmp_path, mode, seq_len):
    """``DistributedRMSNorm`` (hidden_size) matches ``Mistral4RMSNorm`` for decode and prefill."""
    hf_config = _tiny_mistral4_config()
    _run_rms_norm_test(
        RMSNormClass=DistributedRMSNorm,
        hidden_size=hf_config.hidden_size,
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=8,
        hf_config=hf_config,
        tmp_path=tmp_path,
        mesh_device=device,
    )


# ── Tests: RMSNorm (non-distributed, used for kv/q layernorm in MLA) ────


@pytest.mark.parametrize("mode, seq_len", [("decode", 8), ("prefill", 64)])
def test_rms_norm_kv_lora_rank(device, tmp_path, mode, seq_len):
    """``RMSNorm`` (kv_lora_rank dim) matches ``Mistral4RMSNorm`` for decode and prefill."""
    hf_config = _tiny_mistral4_config()
    _run_rms_norm_test(
        RMSNormClass=RMSNorm,
        hidden_size=hf_config.kv_lora_rank,
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=8,
        hf_config=hf_config,
        tmp_path=tmp_path,
        mesh_device=device,
    )


@pytest.mark.parametrize("mode, seq_len", [("decode", 8), ("prefill", 64)])
def test_rms_norm_q_lora_rank(device, tmp_path, mode, seq_len):
    """``RMSNorm`` (q_lora_rank dim) matches ``Mistral4RMSNorm`` for decode and prefill."""
    hf_config = _tiny_mistral4_config()
    _run_rms_norm_test(
        RMSNormClass=RMSNorm,
        hidden_size=hf_config.q_lora_rank,
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=8,
        hf_config=hf_config,
        tmp_path=tmp_path,
        mesh_device=device,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
