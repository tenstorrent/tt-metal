# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""K/V retention: with a single K chunk, non-causal SDPA keeps K/V in a core's CBs across its consecutive Q chunks of
one KV head and takes it from the previous core in its chain only once. This is a subset that fits Wormhole and
Blackhole grids; the full sweep (including 12x10) is
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_kv_retention_sweep.py."""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.kv_retention_test_utils import check_kv_retention
from tests.ttnn.utils_for_testing import assert_with_pcc


# (b, nh, nkv, s, d): GQA 32/8 at batch 4 and a group of 8 at a smaller head dim
@pytest.mark.parametrize("b, nh, nkv, s, d", [(4, 32, 8, 512, 128), (2, 16, 2, 256, 64)])
# 8x7: a core's Q chunks span two KV heads; 8x4: whole KV heads per core
@pytest.mark.parametrize("grid", [(8, 7), (8, 4)])
# 192 does not divide s: at batch 4 on 8x4 the Q chunks also cross Q-head boundaries within a GQA group
@pytest.mark.parametrize("q_chunk", [256, 128, 192])
@pytest.mark.parametrize("concat", [False, True], ids=["head_major", "concat"])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention(device, b, nh, nkv, s, d, grid, q_chunk, concat):
    check_kv_retention(device, b, nh, nkv, s, d, grid, q_chunk, concat, torch_reference=not concat)


@pytest.mark.parametrize("kind", ["mask", "sink"])
@pytest.mark.parametrize("b", [1, 4])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_mask_and_sink(device, kind, b):
    """A mask or a sink does not change which K/V a core needs, so retention stays on for them."""
    grid = (8, 4)
    nh, nkv, s, d = 32, 8, 512, 128
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=128, k_chunk_size=s)
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    kr, vr = k.repeat_interleave(nh // nkv, dim=1), v.repeat_interleave(nh // nkv, dim=1)
    if kind == "mask":
        mask = torch.zeros(b, 1, s, s)
        mask[..., 3 * s // 4 :] = -1e9  # serving pad mask: the last quarter of the keys is padding
        tmask = ttnn.from_torch(mask, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, attn_mask=tmask, program_config=pc, compute_kernel_config=ck
        )
        ref = torch.nn.functional.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)
    else:
        sink = torch.rand(1, nh, 1, 1) * 4.0
        tsink = ttnn.from_torch(sink, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, attention_sink=tsink, program_config=pc, compute_kernel_config=ck
        )
        # the sink joins the softmax denominator, scaled like the scores (test_sdpa_prefill's reference)
        scores = torch.cat([(q @ kr.transpose(-1, -2)) / d**0.5, sink.expand(b, nh, s, 1) / d**0.5], dim=-1)
        ref = torch.softmax(scores, dim=-1)[..., :s] @ vr
    assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.99)
