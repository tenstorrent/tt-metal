# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tt/pf_tune.fp4_fast (bf16 slices) vs the fp32 chain of the prefill indexer (``DSV41DecodeIndexer._fp4_blocks`` on the [N, 32] view): must be bit-identical."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import pf_tune
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer


@pytest.mark.parametrize("scale", [0.05, 1.0, 30.0])
def test_fp4_fast_matches(device, scale):
    torch.manual_seed(0)
    x = (torch.randn(2, 8, 256, 128) * scale).to(torch.bfloat16)
    x[0, 0, 0] = 0  # all-zero blocks
    x[0, 1, 1, :32] = 6.0 * 2.0**3  # amax exactly 6 * 2^k
    t = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    n = 2 * 8 * 256
    f = ttnn.reshape(ttnn.typecast(t, ttnn.float32), (1, 1, n * 4, 32))
    ref = ttnn.typecast(ttnn.reshape(DSV41DecodeIndexer._fp4_blocks(f), list(x.shape)), ttnn.bfloat16)
    got = pf_tune.fp4_fast(t)
    r, g = ttnn.to_torch(ref), ttnn.to_torch(got)
    ndiff = int((r != g).sum())
    print(f"FP4 scale {scale}: {ndiff} of {r.numel()} elements differ", flush=True)
    assert ndiff == 0
