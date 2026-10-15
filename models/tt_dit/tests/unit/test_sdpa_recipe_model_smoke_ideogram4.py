# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: Ideogram4 transformer block (D256) under the SDPA recipes.

Random weights, no checkpoint. A reduced-width block (4 heads x head_dim 256, hidden 1024) is built from the
official reference module (modeling_ideogram4.Ideogram4TransformerBlock, run in fp32 as the ground truth) and the
tt block is run once per variant of sdpa_smoke.py with the SAME state dict and inputs.

Gates (sdpa_smoke.gate_sdpa_core), per recipe variant:
- SDPA core: the SDPA op call inside the block is captured (its inputs after QK-norm/RoPE and, for FAST,
  prepare_sdpa_input, and its output) and compared against exact fp64 attention on those inputs (keys/queries
  limited to logical_n): relative L2 <= the bound. For FAST the same L2 against the pre-preparation bf16 Q/K/V
  (incl. the bfp8 KV quantization) is recorded as `*_sdpa_core_l2_incl_input_prep`.
- end to end vs torch: the block's residual update (out - x) vs the fp32 reference's update:
  L2 <= legacy L2 + margin (no regression). The block output itself is dominated by bf16 matmul /
  norm error shared by all variants, so the SDPA-core gate is the one that isolates the recipe.

Paths: dense SDPA (sp_factor 1, unmasked) on 1x1; ring joint SDPA (SP=2 on the size-2 axis, TP=1) on 1x2 with a
logical_n pad tail; segment-masked dense SDPA (two packed samples, additive attn_mask) on 1x1, gated against fp64
masked attention on the captured inputs.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn

from ...models.transformers.transformer_ideogram4 import Ideogram4TransformerBlock, rope_halfsplit_to_interleaved
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...reference.ideogram4 import modeling_ideogram4
from ...utils import tensor
from ...utils.tensor import bf16_tensor
from .sdpa_smoke import LINE_1D, SdpaCapture, gate_sdpa_core, run_variants

HEAD_DIM = 256
NUM_HEADS = 4
HIDDEN = NUM_HEADS * HEAD_DIM
INTERMEDIATE = 2048
ADALN_DIM = 512
NORM_EPS = 1e-5

BOUNDS = {"default": (3.0, 1.0), "FAST_bfp8": (3.0, 1.5)}  # variant -> (SDPA core bound %, margin % points)


def _reference_block():
    block = modeling_ideogram4.Ideogram4TransformerBlock(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adanln_dim=ADALN_DIM,
    ).eval()
    # Non-trivial norm affines / adaLN so the recipe sees realistic score ranges.
    with torch.no_grad():
        for name, p in block.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            if "adaln_modulation" in name:
                p.mul_(4.0)
    return block


def _inputs(seq_len):
    x = torch.randn(1, seq_len, HIDDEN)
    adaln_input = torch.randn(1, 1, ADALN_DIM)
    theta = torch.rand(1, seq_len, HEAD_DIM // 2) * 2 * math.pi
    cos = torch.cat([theta.cos(), theta.cos()], dim=-1)  # half-split rotate-half tables
    sin = torch.cat([theta.sin(), theta.sin()], dim=-1)
    return x, adaln_input, cos, sin


def _run_variants(
    mesh_device, seq_len, sp_axis, tp_axis, monkeypatch, record_property, tag, expect_path, segment_split=None
):
    torch.manual_seed(0)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    torch_block = _reference_block()
    x, adaln_input, cos, sin = _inputs(seq_len)
    segment_ids = torch.zeros(1, seq_len, dtype=torch.long)
    if segment_split is not None:
        segment_ids[:, segment_split:] = 1
    with torch.no_grad():
        torch_out = torch_block(x, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)
    torch_delta = torch_out - x
    state = torch_block.state_dict()

    alignment = ttnn.TILE_SIZE * sp_factor
    padded = math.ceil(seq_len / alignment) * alignment
    cos4, sin4 = rope_halfsplit_to_interleaved(cos.unsqueeze(1), sin.unsqueeze(1), HEAD_DIM)
    pad = padded - seq_len
    x_p = F.pad(x, (0, 0, 0, pad))
    cos4 = torch.cat([cos4, torch.ones(1, 1, pad, HEAD_DIM)], dim=2)
    sin4 = torch.cat([sin4, torch.zeros(1, 1, pad, HEAD_DIM)], dim=2)

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    shard = dict(mesh_axis=sp_axis) if sp_factor > 1 else {}
    tt_x = bf16_tensor(x_p, device=mesh_device, **shard, **(dict(shard_dim=1) if shard else {}))
    tt_cos = bf16_tensor(cos4, device=mesh_device, **shard, **(dict(shard_dim=2) if shard else {}))
    tt_sin = bf16_tensor(sin4, device=mesh_device, **shard, **(dict(shard_dim=2) if shard else {}))
    tt_adaln = bf16_tensor(adaln_input, device=mesh_device)
    tt_mask = None
    if segment_split is not None:
        assert sp_factor == 1 and pad == 0
        same = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
        mask = torch.zeros(1, 1, seq_len, seq_len).masked_fill(~same.unsqueeze(1), float("-inf"))
        tt_mask = bf16_tensor(mask, device=mesh_device)

    default = Ideogram4TransformerBlock.sdpa_precision_default

    def run_tt(sdpa_precision, kv_dtype):
        capture = SdpaCapture(monkeypatch, mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
        tt_block = Ideogram4TransformerBlock(
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_heads=NUM_HEADS,
            norm_eps=NORM_EPS,
            adaln_dim=ADALN_DIM,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            sdpa_precision=sdpa_precision,
            sdpa_kv_dtype=kv_dtype,
        )
        tt_block.load_torch_state_dict({k: v.clone() for k, v in state.items()})
        tt_out = tt_block(
            tt_x, cos=tt_cos, sin=tt_sin, adaln_input=tt_adaln, attn_mask=tt_mask, spatial_sequence_length=seq_len
        )
        out = tensor.to_torch(tt_out, mesh_axes=[None, sp_axis if sp_factor > 1 else None, None])[:, :seq_len]
        monkeypatch.undo()

        assert len(capture.calls) == 1, f"{tag}: expected one SDPA call, got {len(capture.calls)}"
        call = capture.calls[0]
        assert (call["logical_n"] is not None) == (expect_path == "ring")
        assert (call["mask"] is not None) == (segment_split is not None)
        # The block always names its recipe; LEGACY strips it below this capture (sdpa_smoke.py).
        expected = default if sdpa_precision is None else sdpa_precision
        assert call["kwargs"] == {"precision": expected}
        if expected == ttnn.SDPAPrecision.FAST:
            assert call["dtypes"][1:] == (kv_dtype, kv_dtype)
        return out.float() - x, (capture.core_l2(), capture.core_l2(unprepared=True))

    results = run_variants(run_tt)
    gate_sdpa_core(
        record_property,
        tag,
        torch_delta,
        {vid: delta for vid, (delta, _) in results.items()},
        {vid: core for vid, (_, core) in results.items()},
        BOUNDS,
    )


@pytest.mark.parametrize("device_params", [{}], indirect=True)  # no fabric on a single device
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1088], ids=["n1088"])
def test_ideogram4_block_dense_recipes(mesh_device, seq_len, record_property, monkeypatch):
    _run_variants(mesh_device, seq_len, 0, 1, monkeypatch, record_property, f"dense_n{seq_len}", "dense")


@pytest.mark.parametrize("device_params", [LINE_1D], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("seq_len", [1000], ids=["n1000_pad1024_local512"])  # the logical_n pad tail
def test_ideogram4_block_ring_sp2_recipes(mesh_device, seq_len, record_property, monkeypatch):
    _run_variants(mesh_device, seq_len, 1, 0, monkeypatch, record_property, f"ring_n{seq_len}", "ring")


@pytest.mark.parametrize("device_params", [{}], indirect=True)  # no fabric on a single device
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1088], ids=["n1088"])
def test_ideogram4_block_dense_segment_mask_recipes(mesh_device, seq_len, record_property, monkeypatch):
    # Two packed samples with a non-tile-aligned boundary: the recipe gets the additive segment mask.
    _run_variants(
        mesh_device,
        seq_len,
        0,
        1,
        monkeypatch,
        record_property,
        f"masked_n{seq_len}",
        "dense",
        segment_split=seq_len // 3,
    )
