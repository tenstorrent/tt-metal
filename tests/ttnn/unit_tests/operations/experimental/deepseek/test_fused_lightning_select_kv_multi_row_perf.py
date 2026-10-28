# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device-time profile of ``fused_lightning_select_kv`` with ``Sq`` query rows, against ``Sq`` single-row calls.

For each context length, ``test_multi_row_profile`` runs one single-row call, then the ``Sq``-row op for
every ``Sq`` in ``SEQ_LENS``. The rows sit at consecutive positions ending at the context length, as in a
speculative-decode step. Each configuration first runs once to fill the program cache, then
``ITERATIONS`` times after its ``<keys>-sq<Sq>`` signpost (``<keys>-single`` for the single-row call).
Run::

    python -m tracy -p -r -v -m pytest tests/ttnn/unit_tests/operations/experimental/deepseek/test_fused_lightning_select_kv_multi_row_perf.py

then summarize the ops report (median device kernel time per signpost, against ``Sq`` looped single calls)::

    python tests/ttnn/unit_tests/operations/experimental/deepseek/test_fused_lightning_select_kv_multi_row_perf.py \
        generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
"""

from __future__ import annotations

import csv
import statistics
import sys

import pytest
import torch

import ttnn
from tracy import signpost

from tests.ttnn.unit_tests.operations.experimental.deepseek.test_fused_lightning_select_kv import (
    COMPRESS_RATE,
    V4_HEAD_DIM,
    V4_INDEX_HEAD_DIM,
    V4_INDEX_HEADS,
    V4_INDEX_TOPK,
    V4_KV_HEADS,
    _paged_inputs,
    _replicated_rm_hs,
    _rm,
    _tile,
)

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

PAGE_BLOCK_SIZE = 64
SEQ_LENS = list(range(1, 33))
ITERATIONS = 5


@pytest.mark.parametrize("num_keys", [8 * 1024, 32 * 1024], ids=["ctx32k", "ctx128k"])
def test_multi_row_profile(device, num_keys) -> None:
    torch.manual_seed(0)
    heads, dim, k = V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, V4_INDEX_TOPK
    kv_cache, page_table = _paged_inputs(1, num_keys, PAGE_BLOCK_SIZE, V4_KV_HEADS, V4_HEAD_DIM, seed=0)
    # The model's caches: interleaved TILE index keys, row-major interleaved kv rows.
    key_tt = _tile(torch.randn(page_table.numel(), 1, PAGE_BLOCK_SIZE, dim), device)
    kv_tt = _rm(kv_cache, device, ttnn.bfloat16)
    page_table_tt = _rm(page_table, device, ttnn.int32)
    last_pos = num_keys * COMPRESS_RATE - 1

    configs = [("single", 1)] + [(f"sq{sq}", sq) for sq in SEQ_LENS]
    for name, num_queries in configs:
        query = _replicated_rm_hs(torch.randn(1, num_queries * heads, 1, dim), device)
        weights = _replicated_rm_hs(torch.randn(1, 1, num_queries, heads) * dim**-0.5, device)
        cur_pos = _rm(torch.arange(last_pos - num_queries + 1, last_pos + 1, dtype=torch.int32), device, ttnn.int32)
        # The rows go to DRAM: at Sq = 32 the query replica and its tilized copy fill most of L1.
        ttnn.experimental.deepseek.fused_lightning_select_kv(
            query, key_tt, weights, kv_tt, page_table_tt, cur_pos, k, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.synchronize_device(device)
        signpost(f"{num_keys}-{name}")
        for _ in range(ITERATIONS):
            ttnn.experimental.deepseek.fused_lightning_select_kv(
                query, key_tt, weights, kv_tt, page_table_tt, cur_pos, k, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.synchronize_device(device)
        # Each configuration's replicas would otherwise stay in L1 under the next one's buffers.
        ttnn.deallocate(query)
        ttnn.deallocate(weights)


def summarize(csv_path: str) -> None:
    """Median device kernel time of the select under every signpost, against ``Sq`` looped single calls."""
    durations: dict[str, list[float]] = {}
    section = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["OP TYPE"] == "signpost":
                section = row["OP CODE"]
            elif section is not None and "LightningSelectKv" in row["OP CODE"].replace("_", ""):
                durations.setdefault(section, []).append(float(row["DEVICE KERNEL DURATION [ns]"]) / 1000)
    lengths = sorted({s.split("-")[0] for s in durations}, key=int)
    for keys in lengths:
        single = statistics.median(durations[f"{keys}-single"])
        print(f"\n{keys} keys ({int(keys) * COMPRESS_RATE} tokens): single-row call {single:.1f} us")
        print(f"{'Sq':>4} {'Sq rows us':>11} {'us / row':>9} {'looped us':>10} {'speedup':>8}")
        for sq in SEQ_LENS:
            times = durations.get(f"{keys}-sq{sq}")
            if not times:
                continue
            t = statistics.median(times)
            print(f"{sq:>4} {t:>11.1f} {t / sq:>9.1f} {single * sq:>10.1f} {single * sq / t:>7.2f}x")


if __name__ == "__main__":
    summarize(sys.argv[1])
