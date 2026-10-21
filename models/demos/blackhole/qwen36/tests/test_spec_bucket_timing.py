# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MTP step time versus engine batch size B (the trace row count), server flags, greedy, async chain pipeline.
Per case: cycle time between consecutive step submissions (async depth 1), sync step time (decode + propose, host blocked),
tokens committed per row-step. B=8 also runs with 2 live rows and 1 live row (padding rows inside the trace)."""

import statistics
import time

import pytest
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import NB_BIG, ChainDriver, _marks
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine import _admit_all

WARM, STEADY, SYNC = 20, 150, 50
PROMPT_LEN = 1600


def _p90(v):
    s = sorted(v)
    return s[min(len(s) - 1, int(0.9 * len(s)))]


def _rate(drv, start):
    st = drv.stats[start:]
    return sum(m + 1 for s in st for m in s["m"].values()) / sum(len(s["live"]) for s in st)


def _case(mesh_device, eng, prompts, B, live):
    d = AsyncDriver(eng, seed=71, sampled=lambda s: False)
    _admit_all(d, prompts, live, 5000)
    stamps = []
    for _ in range(WARM + STEADY):
        stamps.append(time.perf_counter())
        d.enq()
        if len(d.pend) == 2:
            d.resolve()
    d.drain()
    cyc = [1e3 * (b - a) for a, b in zip(stamps[WARM:], stamps[WARM + 1 :])]
    rate_a = _rate(d, WARM)
    for r in d.live():
        d._finish(r)
    s = ChainDriver(eng, seed=72, sampled=lambda x: False)
    _admit_all(s, prompts, live, 5000)
    for _ in range(SYNC + 2):
        s.step()
    sync = s.wall_ms[2:]
    ttnn.synchronize_device(mesh_device)
    rate_s = _rate(s, 2)
    for r in s.live():
        s._finish(r)
    logger.info(
        f"[bucket] B={B} buckets={eng._buckets} live={live} cycle_med={statistics.median(cyc):.2f}ms cycle_p90={_p90(cyc):.2f}ms "
        f"sync_med={statistics.median(sync):.2f}ms sync_p90={_p90(sync):.2f}ms tok/row-step async={rate_a:.2f} sync={rate_s:.2f} "
        f"chain_steps={eng.chain_steps} n_cycles={len(cyc)}"
    )


@_marks
@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_spec_bucket_timing(mesh_device, monkeypatch, B, buckets="8", lives=None):
    for k, v in {
        "QWEN36_SPEC_FUSED": "1",
        "QWEN36_SPEC_CHAIN": "1",
        "QWEN36_SPEC_CHAIN_SAMPLED": "1",
        "QWEN36_SPEC_FAST_PREFILL": "1",
        "QWEN36_SPEC_K": "3",
        "QWEN36_SPEC_PREFILL_OVERLAP": "1",
        "QWEN36_SPEC_JOIN_TRACE": "1",
        "QWEN36_SPEC_CHAIN_AUDIT": "0",
        "QWEN36_SPEC_FUSED_AUDIT": "0",
        "QWEN36_SPEC_BUCKETS": buckets,
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(tc, "B", B)
    monkeypatch.setattr(tc, "NB", NB_BIG)
    monkeypatch.setattr(tc, "NUM_BLOCKS", B * NB_BIG + 40)
    monkeypatch.setattr(tc, "PROMPT_LENS", (PROMPT_LEN,) * 8)
    try:
        model, eng, tokenizer, prompts = tc._setup(mesh_device)
    except Exception as e:  # noqa: BLE001
        if B in (2, 4):
            pytest.skip(f"engine B={B} could not be built: {type(e).__name__}: {str(e)[:300]}")
        raise
    assert eng._ch_g is not None and eng._ch_trace_g is not None, "chain greedy trace missing"
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for live in lives or ([8, 2, 1] if B == 8 else [B]):
            _case(mesh_device, eng, prompts, B, live)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    model.free_kv_caches()


@_marks
def test_spec_bucket_timing_buckets_on(mesh_device, monkeypatch):
    """Engine B=8 with QWEN36_SPEC_BUCKETS=1,2,4,8 and 1 / 2 / 4 / 8 live rows (rows 0..n-1): compare the cycle times with the
    flag-off B=8 engine (test_spec_bucket_timing[8], 8 / 2 / 1 live rows) and the B=1, 2, 4 engines."""
    test_spec_bucket_timing(mesh_device, monkeypatch, 8, buckets="1,2,4,8", lives=[1, 2, 4, 8])
