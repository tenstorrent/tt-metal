# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full K/V retention sweep: every schedule regime, head-major and heads-concat output, masks, sinks, causal,
sliding-window and windowed calls. Each batched call must be bit-identical row by row to the same rows run alone (a
different chain and retention layout). Grids larger than the device are skipped, so Wormhole runs everything but
12x10."""

import pytest

from tests.ttnn.unit_tests.operations.sdpa.kv_retention_test_utils import (
    check_kv_retention,
    check_kv_retention_causal_and_sliding,
    check_kv_retention_mask_and_sink,
    check_kv_retention_windowed,
)


# (b, nh, nkv, s, d): the pplx-embed-4b bs16 attention, a smaller batch, a group of 8 at a smaller head dim
@pytest.mark.parametrize("b, nh, nkv, s, d", [(16, 32, 8, 512, 128), (4, 32, 8, 512, 128), (2, 16, 2, 256, 64)])
# 12x10 (Blackhole) and 8x7: a core's Q chunks span one or two KV heads; 8x8 / 8x4: whole KV heads per core
@pytest.mark.parametrize("grid", [(12, 10), (8, 8), (8, 7), (8, 4)])
# 192 does not divide s: on 8x4 the Q chunks also cross Q-head boundaries within a GQA group
@pytest.mark.parametrize("q_chunk", [512, 256, 192, 128])
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


@pytest.mark.parametrize("kind", ["mask", "sink"])
@pytest.mark.parametrize("b", [1, 4])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_mask_and_sink(device, kind, b):
    check_kv_retention_mask_and_sink(device, kind, b)


@pytest.mark.parametrize("kind", ["causal", "sliding", "sliding_causal"])
@pytest.mark.parametrize("grid", [(8, 7), (8, 4), (8, 8)])
@pytest.mark.parametrize("q_chunk", [64, 128])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_causal_and_sliding(device, kind, grid, q_chunk):
    check_kv_retention_causal_and_sliding(device, kind, grid, q_chunk)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("q_chunk", [64, 128])
@pytest.mark.timeout(600)
def test_sdpa_kv_retention_windowed(device, is_causal, q_chunk):
    check_kv_retention_windowed(device, is_causal, q_chunk)
