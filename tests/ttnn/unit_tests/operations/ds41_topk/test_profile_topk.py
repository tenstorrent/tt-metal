# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler bench (run_safe_pytest.sh --profile), one card: ttnn.bringup.topk_large_indices (TOPK_OP=stock:
ttnn.experimental.topk_large_indices) at the
DeepSeek-V4.1 indexer's calls. 2x2 per chip (512 queries, the 55296 target): the 512 selection over 27648 / 55296
keys, the 2048 candidate blocks over 6912 blocks; 8x4 per chip (160 queries, 1M context): 512 over 262144 keys, 2048
over 32768 blocks. bf16 ROW_MAJOR scores from randn. TOPK_SAVE=<path> saves the indices for an A/B compare."""

import os

import pytest
import torch

import ttnn

REPS = 5
OP = ttnn.experimental.topk_large_indices if os.environ.get("TOPK_OP") == "stock" else ttnn.bringup.topk_large_indices
OUT = {}


@pytest.mark.parametrize(
    "sq, n, k",
    [(512, 27648, 512), (512, 55296, 512), (512, 6912, 2048), (160, 262144, 512), (160, 32768, 2048)]
    + [(512, 16384, 512), (512, 27648, 1024), (512, 55296, 1024), (160, 262144, 1024)],
    ids=["2x2-k512-27648", "2x2-k512-55296", "2x2-cand-6912", "8x4-k512-262144", "8x4-cand-32768"]
    + ["2x2-k512-16384", "2x2-k1024-27648", "2x2-k1024-55296", "8x4-k1024-262144"],
)
def test_topk(device, sq, n, k):
    g = torch.Generator().manual_seed(n + k)
    x = torch.randn(1, 1, sq, n, generator=g).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    for i in range(1 + REPS):
        o = OP(t, k=k, valid_length=n)
        if i == 0:
            idx = ttnn.to_torch(o).to(torch.int64) & 0xFFFFFFFF
        o.deallocate()
    ttnn.synchronize_device(device)
    ref = x.float().topk(k, dim=-1).values
    assert torch.equal(torch.gather(x.float(), -1, idx).sort(-1, descending=True).values, ref)  # the k best values
    path = os.environ.get("TOPK_SAVE")
    if path:
        OUT[f"{sq}-{n}-{k}"] = idx
        torch.save(OUT, path)
