# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full K/V retention sweep: every schedule regime, head-major and heads-concat output. Each batched call must be
bit-identical row by row to the same rows run alone (a different chain and retention layout). Grids larger than the
device are skipped, so Wormhole runs everything but 12x10."""

import pytest

from tests.ttnn.unit_tests.operations.sdpa.kv_retention_test_utils import check_kv_retention


# (b, nh, nkv, s, d): the pplx-embed-4b bs16 attention, a smaller batch, a group of 8 at a smaller head dim
@pytest.mark.parametrize("b, nh, nkv, s, d", [(16, 32, 8, 512, 128), (4, 32, 8, 512, 128), (2, 16, 2, 256, 64)])
# 12x10 (Blackhole) and 8x7: a core's Q chunks span one or two KV heads; 8x8 / 8x4: whole KV heads per core
@pytest.mark.parametrize("grid", [(12, 10), (8, 8), (8, 7), (8, 4)])
@pytest.mark.parametrize("q_chunk", [512, 256, 128])
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.timeout(900)
def test_sdpa_kv_retention_sweep(device, b, nh, nkv, s, d, grid, q_chunk, concat):
    check_kv_retention(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        grid,
        q_chunk,
        concat,
        torch_reference=not concat and grid == (8, 8) and q_chunk == 512,
    )
