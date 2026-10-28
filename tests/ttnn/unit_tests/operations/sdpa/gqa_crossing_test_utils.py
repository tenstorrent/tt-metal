# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared checks for the Q-head crossing tests (unit subset and nightly sweep).

Non-causal GQA SDPA lets a Q chunk run on into the next query head of its group when ``q_chunk_size`` does not
divide S and that lowers the most Q chunks any core runs. The schedule is then that of the plain op on Q viewed as
``[B, NKH, (NQH / NKH) * S, d]`` (the same memory for a tile-aligned S, and no GQA, so no crossing decision), so the
output must be bit-identical to it, and with ``output_concat_heads`` to ``nlp_concat_heads`` of the head-major output.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# (b, nh, nkv, s, d, q_chunk, k_chunk, grid): q_chunk does not divide s, and crossing lowers the per-core chunk count
CROSSING_CASES = [
    (1, 32, 8, 512, 128, 192, 512, (11, 8)),
    (1, 32, 8, 512, 128, 160, 256, (12, 10)),
    (8, 32, 8, 512, 128, 192, 512, (12, 10)),
    (8, 32, 8, 512, 128, 160, 256, (8, 8)),
    (2, 16, 4, 256, 64, 192, 256, (6, 8)),
    (2, 16, 4, 256, 64, 160, 256, (7, 8)),
]


def _inputs(device, b, nh, nkv, s, d, dtype, grid):
    device_grid = device.compute_with_storage_grid_size()
    if grid[0] > device_grid.x or grid[1] > device_grid.y:
        pytest.skip(f"grid {grid} exceeds the device grid {device_grid.x}x{device_grid.y}")
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    tq, tk, tv = (ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    return (q, k, v), (tq, tk, tv)


def _torch_ref(q, k, v, nh, nkv):
    g = nh // nkv
    return torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1), is_causal=False
    )


def check_gqa_crossing(device, b, nh, nkv, s, d, q_chunk, k_chunk, grid, dtype):
    (q, k, v), (tq, tk, tv) = _inputs(device, b, nh, nkv, s, d, dtype, grid)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    kwargs = dict(is_causal=False, program_config=pc, compute_kernel_config=ck)
    g = nh // nkv

    ref = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(ttnn.reshape(tq, [b, nkv, g * s, d]), tk, tv, **kwargs)
    ).reshape(b, nh, s, d)

    out = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, **kwargs)
    assert tuple(out.shape) == (b, nh, s, d)
    got = ttnn.to_torch(out)
    assert torch.equal(got, ref), "GQA call must be bit-identical to the plain op on the [B, NKH, G*S, d] view of Q"

    out_cat = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, output_concat_heads=True, **kwargs)
    assert tuple(out_cat.shape) == (b, 1, s, nh * d)
    assert torch.equal(
        ttnn.to_torch(out_cat), ttnn.to_torch(ttnn.experimental.nlp_concat_heads(out))
    ), "crossing concat layout must be bit-identical to nlp_concat_heads of the head-major output"

    assert_with_pcc(_torch_ref(q, k, v, nh, nkv), got.float(), 0.99 if dtype == ttnn.bfloat8_b else 0.999)


def check_concat_wrap_non_streaming(device, b, nh, nkv, s, d, q_chunk, k_chunk, grid):
    """fp32_dest_acc_en=True runs the non-streaming kernel, whose writer drains a whole Q chunk with write_block
    instead of write_block_row_grouped. A chunk's rows run into the next query head of the group, so write_block must
    wrap them too."""
    (q, k, v), (tq, tk, tv) = _inputs(device, b, nh, nkv, s, d, ttnn.bfloat16, grid)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    kwargs = dict(is_causal=False, program_config=pc, compute_kernel_config=ck)
    out = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, **kwargs)
    out_cat = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, output_concat_heads=True, **kwargs)
    assert torch.equal(
        ttnn.to_torch(out_cat), ttnn.to_torch(ttnn.experimental.nlp_concat_heads(out))
    ), "crossing concat layout must be bit-identical to nlp_concat_heads of the head-major output"
    assert_with_pcc(_torch_ref(q, k, v, nh, nkv), ttnn.to_torch(out).float(), 0.999)


def check_ineligible_call(device, kind):
    """A mask, an attention sink or causal attention rules crossing out; the same chunk and grid must still give the
    right answer through the per-head schedule."""
    b, nh, nkv, s, d, q_chunk, k_chunk, grid = CROSSING_CASES[0]
    (q, k, v), (tq, tk, tv) = _inputs(device, b, nh, nkv, s, d, ttnn.bfloat16, grid)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=q_chunk, k_chunk_size=k_chunk
    )
    g = nh // nkv
    kr, vr = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    if kind == "mask":
        mask = torch.zeros(b, 1, s, s)
        mask[..., 3 * s // 4 :] = -1e9
        tmask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, attn_mask=tmask, program_config=pc
        )
        ref = torch.nn.functional.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)
    elif kind == "sink":
        sink = torch.rand(1, nh, 1, 1) * 4.0
        tsink = ttnn.from_torch(sink, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, attention_sink=tsink, program_config=pc
        )
        # the sink joins the softmax denominator, scaled like the scores (test_sdpa_prefill's reference)
        scores = (q @ kr.transpose(-1, -2)) / d**0.5
        scores = torch.cat([scores, sink.expand(b, nh, s, 1) / d**0.5], dim=-1)
        ref = torch.softmax(scores, dim=-1)[..., :s] @ vr
    else:
        out = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, is_causal=True, program_config=pc)
        ref = torch.nn.functional.scaled_dot_product_attention(q, kr, vr, is_causal=True)
    assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.999)
