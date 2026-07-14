# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC regression test for device proj_in at 61f vs 189f sequence lengths.

device proj_in produced clean output at 61f but noisy output at 189f. This
test pins that regression: both N_gen sizes must pass the same PCC threshold
against an equivalent host reference.

Mesh: full 4×8 BH Galaxy → (2, 8) submesh (SP=2 axis-0, TP=8 axis-1).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor
from models.tt_dit.utils.test import line_params

_HIDDEN = 5120
_HEAD_DIM = 128
_NUM_Q_HEADS = 64
_NUM_KV_HEADS = 8
_INTERMEDIATE = 13824
_PATCH_LATENT_DIM = 192
_N_UND = 128
_N_CLEAN = 512  # leading clean-token rows (conditioning frame)

# Both must be multiples of 256 (= k_chunk_size × sp_factor=2).
_N_GEN_SMALL = 9216  # 36 × 256 ≈ 61f 720p token count
_N_GEN_LARGE = 43008  # 168 × 256 ≈ 189f 720p token count (padded)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 8), line_params, id="4x8_full")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "n_gen",
    [
        pytest.param(_N_GEN_SMALL, id="N9216_61f"),
        pytest.param(_N_GEN_LARGE, id="N43008_189f"),
    ],
)
@pytest.mark.timeout(600)
def test_device_proj_in_pcc(mesh_device: ttnn.MeshDevice, n_gen: int) -> None:
    """TT trunk (device proj_in, L=1) vs host reference (host proj_in, L=1)."""
    from torch import nn

    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMoTDecoderLayer as RefDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

    torch.manual_seed(0)

    submesh = mesh_device.create_submesh(ttnn.MeshShape(2, 8))
    sp_factor, tp_factor = 2, 8

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Linear)

    # TT trunk: 1 layer + proj_in.
    tt_trunk = Cosmos3OmniTransformer(
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_attention_heads=_NUM_Q_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=1,
        patch_latent_dim=_PATCH_LATENT_DIM,
        enable_proj_in=True,
        enable_proj_out=False,
        mesh_device=submesh,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    # Host reference components sharing the same random weights.
    ref_proj_in = nn.Linear(_PATCH_LATENT_DIM, _HIDDEN, bias=False).to(torch.bfloat16)
    ref_layer = (
        RefDecoderLayer(
            hidden_size=_HIDDEN,
            head_dim=_HEAD_DIM,
            num_attention_heads=_NUM_Q_HEADS,
            num_key_value_heads=_NUM_KV_HEADS,
            intermediate_size=_INTERMEDIATE,
            attention_bias=False,
            rms_norm_eps=1e-6,
        )
        .eval()
        .to(torch.bfloat16)
    )
    import torch.nn as _nn

    ref_norm = _nn.RMSNorm(_HIDDEN, eps=1e-6, elementwise_affine=True).to(torch.bfloat16)

    # Sync weights: TT trunk ← reference random weights.
    state = {
        "proj_in.weight": ref_proj_in.weight,
        **{f"layers.0.{k}": v for k, v in ref_layer.state_dict().items()},
        "norm_moe_gen.weight": ref_norm.weight,
        "norm.weight": ref_norm.weight,  # und norm (output unused in I2V)
    }
    tt_trunk.load_torch_state_dict(state)

    # Inputs.
    rope = Cosmos3VLTextRotaryEmbedding(
        head_dim=_HEAD_DIM,
        rope_theta=10000.0,
        rope_axes_dim=[16, 56, 56],
    )
    pos_ids = torch.arange(_N_UND + n_gen).unsqueeze(0)
    cos_all, sin_all = rope(pos_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all, sin_all = cos_all.squeeze(0), sin_all.squeeze(0)
    cos_und, sin_und = cos_all[:_N_UND], sin_all[:_N_UND]
    cos_gen, sin_gen = cos_all[_N_UND : _N_UND + n_gen], sin_all[_N_UND : _N_UND + n_gen]

    und_seq = torch.randn(_N_UND, _HIDDEN, dtype=torch.bfloat16)
    raw_patches = torch.randn(n_gen, _PATCH_LATENT_DIM, dtype=torch.bfloat16)
    time_embed = torch.randn(1, _HIDDEN, dtype=torch.bfloat16)
    noisy_mask = torch.zeros(n_gen, 1, dtype=torch.bfloat16)
    noisy_mask[_N_CLEAN:] = 1.0

    # Host reference: proj_in + time_embed + 1 decoder layer + final norm.
    with torch.no_grad():
        gen_proj = ref_proj_in(raw_patches)
        gen_hidden = gen_proj + noisy_mask * time_embed
        _, ref_gen_layer = ref_layer(und_seq, gen_hidden, (cos_und, sin_und, cos_gen, sin_gen))
        ref_gen_out = ref_norm(ref_gen_layer)

    # TT: upload raw patches + mask + time_embed.
    gen_seq_multiple = 128 * sp_factor
    pad_n = (-n_gen) % gen_seq_multiple

    def _pad(t: torch.Tensor, pad_rows: int) -> torch.Tensor:
        return t if pad_rows == 0 else torch.cat([t, t.new_zeros(pad_rows, t.shape[-1])])

    raw_pad = _pad(raw_patches, pad_n)
    mask_pad = _pad(noisy_mask, pad_n)
    cos_gen_pad = _pad(cos_gen, pad_n)
    sin_gen_pad = _pad(sin_gen, pad_n)

    N_pad = n_gen + pad_n

    gen_tt = bf16_tensor(raw_pad.reshape(1, 1, N_pad, _PATCH_LATENT_DIM), device=submesh, mesh_axis=0, shard_dim=2)
    mask_tt = bf16_tensor(mask_pad.reshape(1, 1, N_pad, 1), device=submesh, mesh_axis=0, shard_dim=2)
    time_tt = bf16_tensor(time_embed.reshape(1, 1, 1, _HIDDEN), device=submesh)
    und_tt = bf16_tensor(und_seq.reshape(1, 1, _N_UND, _HIDDEN), device=submesh)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, _N_UND, _HEAD_DIM), device=submesh)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, _N_UND, _HEAD_DIM), device=submesh)
    cos_gen_tt = bf16_tensor(cos_gen_pad.reshape(1, 1, N_pad, _HEAD_DIM), device=submesh, mesh_axis=0, shard_dim=2)
    sin_gen_tt = bf16_tensor(sin_gen_pad.reshape(1, 1, N_pad, _HEAD_DIM), device=submesh, mesh_axis=0, shard_dim=2)

    _und_tt_out, gen_tt_out = tt_trunk(
        und_tt,
        gen_tt,
        cos_und_tt,
        sin_und_tt,
        cos_gen_tt,
        sin_gen_tt,
        logical_n_gen=n_gen,
        time_embed=time_tt,
        noisy_mask_gen=mask_tt,
    )

    # Gather SP-sharded output, strip padding.
    devs = ttnn.get_device_tensors(gen_tt_out)
    chunks = [ttnn.to_torch(devs[i]).reshape(-1, _HIDDEN) for i in range(sp_factor)]
    gen_tt_full = torch.cat(chunks, dim=0)[:n_gen]

    # PCC must hold at both sequence lengths. If 189f fails while 61f passes,
    # the bug is in the proj_in or time_embed computation at large M.
    assert_quality(ref_gen_out, gen_tt_full, pcc=0.90)
