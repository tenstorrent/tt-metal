# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Selection accuracy of topk_large_indices on DSA-indexer-like scores, measured against an FP32 reference.

The sparse-SDPA indexer computes its logits in FP32 and today hands them to top-k as BF16, so near-ties
around the k-th score collapse and the op keeps some keys the FP32 ranking would drop. This test builds
FP32 logits shaped like the indexer's (sum over heads of w_h * relu(q_h . k_t)), offers them to the op as
an FP32 tensor, and falls back to BF16 when the op rejects FP32 input.

Tie-aware metric: a returned index is a miss when its FP32 score is below the FP32 k-th largest score of
its row (indices tied at the threshold are all correct). Logged per case:
    DREAM_METRIC miss_pct=<% of returned indices that are misses> mass_gap_ppm=<lost score mass> fp32_input=<0|1>
The op is called WARMUP + MEASURED times per case (the Dream perf adapter reads the device time).
"""

import pytest
import torch
import ttnn
from loguru import logger

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="topk_large_indices is Blackhole-only"),
]

WARMUP, MEASURED = 3, 10
HEADS, HEAD_DIM = 16, 128  # local indexer heads per chip (64 heads / TP 4), index head dim
SENTINEL = 0xFFFFFFFF


def _indexer_logits(rows: int, width: int, seed: int) -> torch.Tensor:
    """FP32 [1, 1, rows, width] logits: sum_h w[r, h] * relu(q[r, h] . k[t]) * HEAD_DIM**-0.5."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(rows, HEADS, HEAD_DIM, generator=g)
    k = torch.randn(width, HEAD_DIM, generator=g)
    w = torch.randn(rows, HEADS, generator=g) * HEADS**-0.5
    out = torch.zeros(rows, width)
    for h in range(HEADS):
        out += w[:, h : h + 1] * torch.relu(q[:, h] @ k.T) * HEAD_DIM**-0.5
    return out.reshape(1, 1, rows, width)


def _run(device, logits: torch.Tensor, k: int, valid_length):
    """Call the op WARMUP + MEASURED times; FP32 input when the op accepts it, else BF16."""
    kwargs = {"k": k}
    if valid_length is not None:
        kwargs["valid_length"] = valid_length
    fp32 = True
    tt_in = ttnn.from_torch(logits, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    try:
        out = ttnn.experimental.topk_large_indices(tt_in, **kwargs)
    except RuntimeError as e:
        logger.info(f"FP32 input rejected, using BF16: {str(e).splitlines()[0][:200]}")
        fp32 = False
        tt_in = ttnn.from_torch(logits, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = ttnn.experimental.topk_large_indices(tt_in, **kwargs)
    for _ in range(WARMUP + MEASURED - 1):
        out = ttnn.experimental.topk_large_indices(tt_in, **kwargs)
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out, dtype=torch.uint32).to(torch.int64), fp32


@pytest.mark.parametrize(
    "rows,width,k,valid_length",
    [
        (256, 8192, 2048, None),
        (128, 32768, 2048, None),
        (640, 51200, 1536, None),
        (128, 65536, 2048, 40960),
    ],
    ids=["r256_t8k_k2048", "r128_t32k_k2048", "r640_t51k_k1536", "r128_t64k_v40k_k2048"],
)
def test_topk_large_indices_accuracy(device, rows, width, k, valid_length):
    logits = _indexer_logits(rows, width, seed=rows * 7919 + width)
    idx, fp32 = _run(device, logits, k, valid_length)

    n = valid_length or width
    ref = logits.reshape(rows, width)[:, :n]
    assert list(idx.shape) == [1, 1, rows, k], f"unexpected output shape {list(idx.shape)}"
    idx = idx.reshape(rows, k)
    assert not (idx == SENTINEL).any(), "sentinel returned for finite scores"
    assert int(idx.min()) >= 0 and int(idx.max()) < n, f"index out of range [0, {n}): {idx.min()}..{idx.max()}"
    assert all(len(torch.unique(r)) == k for r in idx), "duplicate indices within a row"

    top = torch.topk(ref, k, dim=-1).values
    kth = top[:, -1:]
    picked = torch.gather(ref, 1, idx)
    misses = int((picked < kth).sum())
    miss_pct = 100.0 * misses / (rows * k)
    ideal = top.sum(dim=-1)
    mass_gap_ppm = float(((ideal - picked.sum(dim=-1)) / ideal.abs().clamp_min(1e-30)).mean()) * 1e6

    logger.info(f"rows={rows} width={width} valid={n} k={k} fp32_input={fp32} misses={misses}")
    print(f"DREAM_METRIC miss_pct={miss_pct:.6f} mass_gap_ppm={mass_gap_ppm:.4f} fp32_input={int(fp32)}")
