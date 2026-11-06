# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Two-stage segmented QSA top-k (qsa_topk_block_ids, QWEN38_TOPK_SEG=1) against the single-stage topk_large_indices.

(a) distinct scores: ids equal the single-stage ids (set and order); (b) ties and masked tails: a row's result is
bitwise the same in a call of 1, 4 or 8 rows (and at another row position); (c) the selected set equals the
single-stage set when no tie straddles rank 512; (d) traced timing, single vs two-stage.
"""

import os
import time

import numpy as np
import pytest
import torch
import ttnn

from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module

K = 512
MASK = -3.3895e38

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="topk_large_indices is Blackhole-only"),
]


def _bf16_from_u16(hi16: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((hi16.astype(np.uint32) << 16).view(np.float32).copy()).to(torch.bfloat16)


def _distinct_top_input(rows: int, width: int, seed: int) -> torch.Tensor:
    """Per row: 1024 random positions hold distinct high values, the rest tie among low values."""
    gen = np.random.default_rng(seed)
    out = []
    for _ in range(rows):
        low = gen.integers(0x3000, 0x3400, size=width)
        pos = gen.permutation(width)[:1024]
        low[pos] = 0x4000 + gen.permutation(1024)
        out.append(_bf16_from_u16(low))
    return torch.stack(out)


def _boundary_distinct_input(rows: int, width: int, seed: int) -> torch.Tensor:
    """Ties among the top 300 (triples), distinct ranks 301..1024, no tie at rank 512."""
    gen = np.random.default_rng(seed)
    out = []
    for _ in range(rows):
        low = gen.integers(0x3000, 0x3400, size=width)
        pos = gen.permutation(width)[:1024]
        vals = np.concatenate([0x4800 + np.arange(300) // 3, 0x4000 + np.arange(724)])
        low[pos] = vals
        out.append(_bf16_from_u16(low))
    return torch.stack(out)


def _tied_masked_input(rows: int, width: int, seed: int) -> torch.Tensor:
    """Many ties (4 distinct values) and a masked tail of a different length per row."""
    gen = np.random.default_rng(seed)
    out = []
    for r in range(rows):
        vals = _bf16_from_u16(0x3F00 + gen.integers(0, 4, size=width) * 0x10)
        live = width // 2 + (r * 997) % (width // 4)
        vals[live:] = MASK
        out.append(vals)
    return torch.stack(out)


def _upload(t: torch.Tensor, device):
    rows, width = t.shape
    return ttnn.from_torch(
        t.reshape(1, 1, rows, width), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _single(x):
    os.environ["QWEN38_TOPK_SEG"] = "0"
    return qsa_module.qsa_topk_block_ids(x, k=K)


def _two_stage(x):
    os.environ["QWEN38_TOPK_SEG"] = "1"
    os.environ.pop("QWEN38_TOPK_SEG_W", None)
    try:
        return qsa_module.qsa_topk_block_ids(x, k=K)
    finally:
        os.environ["QWEN38_TOPK_SEG"] = "0"


def _host(t) -> torch.Tensor:
    return ttnn.to_torch(t).reshape(-1, K).to(torch.int64)


@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("width", [32768, 65536])
def test_two_stage_ids_equal_single_stage_distinct(device, rows, width):
    data = _distinct_top_input(rows, width, seed=rows * 7 + width)
    x = _upload(data, device)
    one = _host(_single(x))
    two = _host(_two_stage(x))
    assert two.shape == (rows, K)
    # the top 512 are all distinct values: set and order are fixed by the scores
    assert torch.equal(one, two)


@pytest.mark.parametrize("width", [32768, 65536])
def test_two_stage_row_is_independent_of_the_call_rows(device, width):
    data = _tied_masked_input(8, width, seed=3)
    results = {}
    for n in (1, 4, 8):
        results[n] = _host(_two_stage(_upload(data[:n], device)))
    assert torch.equal(results[4][:1], results[1])
    assert torch.equal(results[8][:4], results[4])
    assert torch.equal(results[8][:1], results[1])
    # another position in another batch
    order = [5, 0, 1, 2]
    moved = _host(_two_stage(_upload(data[order], device)))
    alone = _host(_two_stage(_upload(data[5:6], device)))
    assert torch.equal(moved[:1], alone)
    again = _host(_two_stage(_upload(data, device)))
    assert torch.equal(again[5:6], alone)


@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("width", [32768, 65536])
def test_two_stage_set_equals_single_stage_without_tie_at_rank_512(device, rows, width):
    data = _boundary_distinct_input(rows, width, seed=rows + width)
    x = _upload(data, device)
    one = _host(_single(x))
    two = _host(_two_stage(x))
    for r in range(rows):
        assert sorted(one[r].tolist()) == sorted(two[r].tolist())
        # and the values read at the ids are the same descending sequence
        v1 = data[r][one[r]]
        v2 = data[r][two[r]]
        assert torch.equal(v1, v2)


def _trace_us(device, run, iters=50):
    run()
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(iters):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        return (time.perf_counter() - start) / iters * 1e6
    finally:
        ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize("rows,width", [(1, 65536), (4, 65536), (8, 32768), (8, 16384)])
def test_two_stage_traced_timing(device, rows, width):
    x = _upload(_distinct_top_input(rows, width, seed=11), device)
    single_us = _trace_us(device, lambda: _single(x))
    two_us = _trace_us(device, lambda: _two_stage(x))
    print(f"TOPK_SEG_TIMING rows={rows} W={width} single_us={single_us:.1f} two_stage_us={two_us:.1f}")
