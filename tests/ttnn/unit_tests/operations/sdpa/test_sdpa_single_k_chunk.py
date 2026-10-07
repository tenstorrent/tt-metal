# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""With one K chunk (``k_chunk_size`` covering all of Sk) there is no online-softmax correction, so on Blackhole the
streaming kernel (``fp32_dest_acc_en=False``) takes each softmax row's denominator from the exp'd scores with a matmul
on the math thread, a whole normalize row group at a time, instead of accumulating it on the pack thread. Every masking
mode, a padded sequence and Q chunks whose tile count is not a multiple of the normalize row group go through that
path here, checked against torch and against the same call split into several K chunks (the per-chunk sum path)."""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

TILE = 32


def _round_up(x, m):
    return (x + m - 1) // m * m


def _windows(s):
    return [0, s // 3, 2 * s // 3 + 5, s]


def _torch_mask(mode, b, s, user_mask):
    if mode == "mask":
        return user_mask
    if mode.startswith("windowed"):
        cu = _windows(s)
        mask = torch.full((s, s), float("-inf"))
        for lo, hi in zip(cu, cu[1:]):
            mask[lo:hi, lo:hi] = 0.0
        if mode == "windowed_causal":
            mask = mask + torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
        return mask.expand(b, 1, s, s)
    return None


def _run(device, mode, q, k, v, user_mask, q_chunk, k_chunk):
    b, _, s, _ = q.shape
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    kwargs = dict(program_config=pc, compute_kernel_config=ck, is_causal=mode in ("causal", "windowed_causal"))
    if mode == "mask":
        kwargs["attn_mask"] = tt(user_mask)
    if mode.startswith("windowed"):
        kwargs["cu_window_seqlens"] = ttnn.from_torch(
            torch.tensor(_windows(s), dtype=torch.int32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
        )
    out = ttnn.transformer.scaled_dot_product_attention(tt(q), tt(k), tt(v), **kwargs)
    return ttnn.to_torch(out)[:, :, :s, :].float()


@pytest.mark.parametrize("mode", ["none", "causal", "mask", "windowed", "windowed_causal"])
# a GQA group, MHA at d=128 with batch 2, and an S that is not a tile multiple (padded last Q and K tile)
@pytest.mark.parametrize("b, nh, nkv, s, d", [(1, 8, 2, 256, 64), (2, 4, 4, 512, 128), (1, 4, 1, 200, 64)])
# 32: one-tile row groups. 96: an odd tile count gets 2-tile row groups plus a remainder (streaming_qktv_h), which
# keeps the per-row sum path. 128: whole multi-tile row groups.
@pytest.mark.parametrize("q_chunk", [32, 96, 128])
@pytest.mark.timeout(300)
def test_sdpa_single_k_chunk(device, mode, b, nh, nkv, s, d, q_chunk):
    if mode == "mask" and s % TILE:
        pytest.skip("a user mask must cover the padded K columns; padded S is covered by the other modes")
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    user_mask = torch.bernoulli(torch.full((b, 1, s, s), 0.25)) * -1e9 if mode == "mask" else None

    single = _run(device, mode, q, k, v, user_mask, q_chunk, _round_up(s, TILE))
    chunked = _run(device, mode, q, k, v, user_mask, q_chunk, 128)

    g = nh // nkv
    gt = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(g, dim=1),
        v.repeat_interleave(g, dim=1),
        attn_mask=_torch_mask(mode, b, s, user_mask),
        is_causal=mode == "causal",
    )
    assert torch.isfinite(single).all(), "non-finite output rows"
    ok, pcc = comp_pcc(gt, single, 0.995)
    assert ok, f"single K chunk vs torch: {pcc}"
    # PCC ignores a per-row scale, which is what a wrong softmax denominator produces: the least-squares scale of each
    # output row onto the reference must be 1 up to bf16 noise, which averages out over the row's d elements
    row_scale = (single * gt).sum(-1) / (gt * gt).sum(-1)
    worst = (row_scale - 1).abs().max().item()
    assert worst < 0.1, f"single K chunk vs torch: an output row is off by a factor {1 + worst:.4f}"
    ok, pcc = comp_pcc(chunked, single, 0.999)
    assert ok, f"single K chunk vs {(_round_up(s, TILE) + 127) // 128} K chunks: {pcc}"
