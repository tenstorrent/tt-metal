# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill and traced warmed decode windows of the QB2 multichip decoder (Tracy signposts).

python -m tracy -r -p -v -m pytest "models/autoports/google_gemma_4_12b_it/tests/test_multichip_perf.py::test_multichip_perf[sliding]"

Same windows as ``test_optimized_perf.py`` (single chip), so the tt-perf-report tables compare directly:
- PERF_PREFILL: one warmed 4096-token prefill (hidden-sharded residual input);
- PERF_PREFILL_SHORT: one warmed 1000-token prefill (non-aligned; tail padded to 1024 rows, 2D-matmul path);
- PERF_DECODE: one replay of the warmed batch-1 decode trace at position 4096;
- PERF_DECODE_B32: one replay of a batch-32 trace (users at positions 4065..4096).
Every window covers all four chips (tt-perf-report merges devices). Run one test per Tracy session.
"""

import time

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests import mc_cases
from models.autoports.google_gemma_4_12b_it.tests.harness import Pages, record
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import (
    real_layer_state_dict,
    synthetic_hidden,
    text_config,
)

try:
    from tracy import signpost
except ImportError:  # profiler not built in

    def signpost(*args, **kwargs):
        pass


PREFILL_LEN = 4096
SHORT_PREFILL_LEN = 1000
PERF_LAYERS = {"sliding": 0, "full": 5, "sliding_mlp4": 13, "full_mlp4": 17}


def _profile_flush(device):
    ttnn.ReadDeviceProfiler(device)


def _decode_window(be, dec, batch, x, pos, table, name, reps=50):
    trace = mc_cases.TracedDecode(be, dec, batch, table)
    trace.set_inputs(x, pos, table)
    trace.capture()
    for _ in range(3):
        trace.replay()
    be.sync()
    _profile_flush(be.device)
    signpost(name)
    trace.replay()
    be.sync()
    signpost(name + "_END")
    _profile_flush(be.device)
    total = 0.0
    for _ in range(reps // 10):
        t0 = time.perf_counter()
        for _ in range(10):
            trace.replay()
        be.sync()
        total += time.perf_counter() - t0
        _profile_flush(be.device)
    trace.release()
    return total * 1e3 / (reps // 10 * 10)


@pytest.mark.parametrize("kind", list(PERF_LAYERS))
def test_multichip_perf(qb2, kind):
    cfg = text_config()
    be = mc_cases.Multichip(qb2)
    layer_idx = PERF_LAYERS[kind]
    sd = real_layer_state_dict(layer_idx)
    batch = 32
    max_pages = (PREFILL_LEN + 64) // 64
    dec = be.build(cfg, layer_idx, sd, num_blocks=batch * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=batch)
    for u in range(batch):
        pages.ensure(u, PREFILL_LEN + 1)
    table = be.ints(pages.table)
    x = synthetic_hidden(mc_cases.stats_layer(cfg, layer_idx), PREFILL_LEN + 1)
    tx = be.prefill_in(dec, x[:, :PREFILL_LEN])

    dec.prefill_forward(tx, page_table=table)  # compile
    be.sync()
    _profile_flush(qb2)
    signpost("PERF_PREFILL")
    t0 = time.perf_counter()
    out = dec.prefill_forward(tx, page_table=table)
    be.sync()
    prefill_ms = (time.perf_counter() - t0) * 1e3
    signpost("PERF_PREFILL_END")
    out.deallocate(True)
    short_table = be.ints(pages.table[1:2])
    ts = be.prefill_in(dec, x[:, :SHORT_PREFILL_LEN])
    dec.prefill_forward(ts, page_table=short_table).deallocate(True)
    be.sync()
    _profile_flush(qb2)
    signpost("PERF_PREFILL_SHORT")
    t0 = time.perf_counter()
    out = dec.prefill_forward(ts, page_table=short_table)
    be.sync()
    short_ms = (time.perf_counter() - t0) * 1e3
    signpost("PERF_PREFILL_SHORT_END")
    out.deallocate(True)
    decode_b1 = _decode_window(
        be, dec, 1, x[:, PREFILL_LEN:], torch.tensor([PREFILL_LEN]), pages.table[:1], "PERF_DECODE"
    )
    xb = torch.cat([synthetic_hidden(mc_cases.stats_layer(cfg, layer_idx), 1, seed=10 + u) for u in range(batch)], 1)
    pos = torch.arange(PREFILL_LEN - batch + 1, PREFILL_LEN + 1)
    decode_b32 = _decode_window(be, dec, batch, xb, pos, pages.table, "PERF_DECODE_B32")
    _profile_flush(qb2)
    record(
        f"multichip_perf[{kind}]",
        layer=layer_idx,
        prefill_tokens=PREFILL_LEN,
        prefill_wall_ms=round(prefill_ms, 3),
        short_prefill_tokens=SHORT_PREFILL_LEN,
        short_prefill_wall_ms=round(short_ms, 3),
        decode_pos=PREFILL_LEN,
        decode_b1_trace_wall_avg50_ms=round(decode_b1, 4),
        decode_b32_trace_wall_avg50_ms=round(decode_b32, 4),
    )
