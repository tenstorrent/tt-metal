# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill and traced warmed decode measurement windows (run under Tracy for device times).

python -m tracy -r -p -v -m pytest models/autoports/google_gemma_4_12b_it/tests/test_perf.py -k <kind>

Signposts: PERF_PREFILL..PERF_PREFILL_END wraps one warmed 4096-token prefill (two 2048-row chunks);
PERF_DECODE..PERF_DECODE_END wraps one warmed replay of the captured batch-1 decode trace at position
4096. Host wall times (synchronized) are recorded as well.
"""

import time

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests.harness import Pages, build_decoder, record, to_tt
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import (
    LAYER_KINDS,
    real_layer_state_dict,
    synthetic_hidden,
    text_config,
)
from models.autoports.google_gemma_4_12b_it.tests.test_functional_decoder import TracedDecode

try:
    from tracy import signpost
except ImportError:  # profiler not built in

    def signpost(*args, **kwargs):
        pass


PREFILL_LEN = 4096


@pytest.mark.parametrize("kind", list(LAYER_KINDS))
def test_perf(device_1x1, kind):
    cfg = text_config()
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    max_pages = 8192 // 64
    dec = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=1)
    pages.ensure(0, PREFILL_LEN + 1)
    table = pages.tt(device_1x1)
    x = synthetic_hidden(layer_idx, PREFILL_LEN + 1)
    tx = to_tt(x[:, :PREFILL_LEN], device_1x1)

    dec.prefill_forward(tx, page_table=table)  # compile
    ttnn.synchronize_device(device_1x1)
    signpost("PERF_PREFILL")
    t0 = time.perf_counter()
    out = dec.prefill_forward(tx, page_table=table)
    ttnn.synchronize_device(device_1x1)
    prefill_ms = (time.perf_counter() - t0) * 1e3
    signpost("PERF_PREFILL_END")
    out.deallocate(True)

    trace = TracedDecode(dec, device_1x1, 1)
    trace.set_inputs(x[:, PREFILL_LEN:], torch.tensor([PREFILL_LEN]), pages.table)
    trace.capture()
    for _ in range(3):
        trace.replay()
    ttnn.synchronize_device(device_1x1)
    signpost("PERF_DECODE")
    t0 = time.perf_counter()
    trace.replay()
    ttnn.synchronize_device(device_1x1)
    decode_ms = (time.perf_counter() - t0) * 1e3
    signpost("PERF_DECODE_END")
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        trace.replay()
    ttnn.synchronize_device(device_1x1)
    decode_avg_ms = (time.perf_counter() - t0) * 1e3 / n
    trace.release()
    ttnn.ReadDeviceProfiler(device_1x1)
    record(
        f"perf[{kind}]",
        prefill_tokens=PREFILL_LEN,
        prefill_wall_ms=round(prefill_ms, 3),
        decode_pos=PREFILL_LEN,
        decode_batch=1,
        decode_trace_wall_ms=round(decode_ms, 3),
        decode_trace_wall_avg20_ms=round(decode_avg_ms, 3),
    )
