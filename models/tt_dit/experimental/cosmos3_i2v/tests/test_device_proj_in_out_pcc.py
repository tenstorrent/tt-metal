# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive PCC regression tests for device proj_in + proj_out.

The generate pipeline enables BOTH proj_in and proj_out by default. The
original test_device_proj_in_pcc.py only tested proj_in alone (proj_out=False),
which does not match production. This test covers:

  1. proj_in + proj_out (production config) at 61f and 189f token counts
  2. _build_noisy_mask_gen correctness (host-only, no device needed)

The 189f noisy-output bug was found while proj_in+proj_out were both enabled.
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
_N_CLEAN = 512

# 61f 720p: 16 × 23 × 40 = 14720 (k_chunk_size × sp_factor = 256; 14720/256=57.5 ✗)
# Adjust to nearest valid: 14848 = 58 × 256
# 189f 720p: 48 × 23 × 40 = 44160 (44160/256=172.5 ✗; pad to 44288)
# Use clean multiples for the test.
_N_GEN_SMALL = 14720  # ≈ 61f production (14720/128=115 ✓)
_N_GEN_LARGE = 44160  # ≈ 189f production (44160/128=345 ✓)


# ── Host-only tests (no device) ──────────────────────────────────────────────


class TestBuildNoisyMaskGen:
    """Verify _build_noisy_mask_gen produces the correct 0/1 mask."""

    def _make_mask(self, n_total: int, t_total: int, spatial: int, n_noisy_frames: int) -> torch.Tensor:
        from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_native import _build_noisy_mask_gen

        packed = torch.zeros(n_total, _PATCH_LATENT_DIM)
        # First frame is clean, rest are noisy.
        n_clean_frames = t_total - n_noisy_frames
        noisy_idxs = torch.arange(n_clean_frames, t_total)
        shape = (t_total, spatial, 1)  # (T, H*W, 1) simplified
        return _build_noisy_mask_gen(packed, [noisy_idxs], [(t_total, spatial, 1)])

    def test_small_sequence(self):
        n_total = _N_GEN_SMALL
        spatial = n_total // 16  # 16 temporal frames
        mask = self._make_mask(n_total, 16, spatial, n_noisy_frames=15)
        clean_rows = n_total // 16  # 1 clean frame
        assert mask[:clean_rows].sum() == 0.0, "clean rows must be 0"
        assert mask[clean_rows:].sum() == float(n_total - clean_rows), "noisy rows must be 1"

    def test_large_sequence(self):
        n_total = _N_GEN_LARGE
        spatial = n_total // 48  # 48 temporal frames
        mask = self._make_mask(n_total, 48, spatial, n_noisy_frames=47)
        clean_rows = n_total // 48  # 1 clean frame
        assert mask[:clean_rows].sum() == 0.0
        assert mask[clean_rows:].sum() == float(n_total - clean_rows)


# ── Device PCC tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 8), line_params, id="4x8_full")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "n_gen",
    [
        pytest.param(_N_GEN_SMALL, id="N14720_61f"),
        pytest.param(_N_GEN_LARGE, id="N44160_189f"),
    ],
)
@pytest.mark.timeout(600)
def test_proj_in_out_pcc(mesh_device: ttnn.MeshDevice, n_gen: int) -> None:
    """TT trunk (proj_in + proj_out, L=1) vs host reference at production dims.

    Production pipeline enables both proj_in and proj_out. The proj_out
    matmul [N/sp, 5120] × [5120, 192] at 189f was not tested by the
    proj_in-only PCC test.
    """
    import torch.nn as nn

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

    tt_trunk = Cosmos3OmniTransformer(
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_attention_heads=_NUM_Q_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=1,
        patch_latent_dim=_PATCH_LATENT_DIM,
        enable_proj_in=True,
        enable_proj_out=True,
        mesh_device=submesh,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    ref_proj_in = nn.Linear(_PATCH_LATENT_DIM, _HIDDEN, bias=True).to(torch.bfloat16)
    ref_proj_out = nn.Linear(_HIDDEN, _PATCH_LATENT_DIM, bias=True).to(torch.bfloat16)
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
    ref_norm = nn.RMSNorm(_HIDDEN, eps=1e-6, elementwise_affine=True).to(torch.bfloat16)

    state = {
        "proj_in.weight": ref_proj_in.weight,
        "proj_in.bias": ref_proj_in.bias,
        "proj_out.weight": ref_proj_out.weight,
        "proj_out.bias": ref_proj_out.bias,
        **{f"layers.0.{k}": v for k, v in ref_layer.state_dict().items()},
        "norm_moe_gen.weight": ref_norm.weight,
        "norm.weight": ref_norm.weight,
    }
    tt_trunk.load_torch_state_dict(state)

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

    # Host reference: proj_in → time_embed → layer → norm → proj_out.
    with torch.no_grad():
        gen_proj = ref_proj_in(raw_patches)
        gen_hidden = gen_proj + noisy_mask * time_embed
        _, ref_gen_layer = ref_layer(und_seq, gen_hidden, (cos_und, sin_und, cos_gen, sin_gen))
        ref_gen_normed = ref_norm(ref_gen_layer)
        ref_gen_out = ref_proj_out(ref_gen_normed)  # [N_gen, 192]

    # TT path.
    gen_seq_multiple = 128 * sp_factor
    pad_n = (-n_gen) % gen_seq_multiple

    def _pad(t: torch.Tensor, rows: int) -> torch.Tensor:
        return t if rows == 0 else torch.cat([t, t.new_zeros(rows, t.shape[-1])])

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

    # Gather output — proj_out produces [N_gen, 192], replicated after final gather.
    devs = ttnn.get_device_tensors(gen_tt_out)
    # With proj_out enabled, output is replicated [1,1,N_gen,192] after all-gather.
    gen_tt_full = ttnn.to_torch(devs[0]).reshape(-1, _PATCH_LATENT_DIM)[:n_gen]

    assert_quality(ref_gen_out, gen_tt_full, pcc=0.90)
