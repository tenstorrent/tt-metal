# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill and traced warmed decode windows: functional (before) vs fused (after) decoder.

python -m tracy -r -p -v -m pytest "models/autoports/google_gemma_4_12b_it/tests/test_fused_perf.py::test_fused_perf[fused-sliding]"

Each test builds one implementation (each at its default prefill chunk) for one layer kind, real
weights. Signposts:
- PERF_PREFILL..PERF_PREFILL_END wraps one warmed 4096-token prefill;
- PERF_DECODE..PERF_DECODE_END wraps one replay of the warmed, captured batch-1 decode trace at
  position 4096;
- PERF_DECODE_B32..PERF_DECODE_B32_END wraps one replay of a batch-32 trace (users at positions
  4096 - 31 .. 4096).
Host wall times (synchronized; decode averaged over 50 replays) are recorded too. Run one test per
Tracy session so the signpost windows are unique.
"""

import time

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests.harness import Pages, record, to_tt
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import (
    LAYER_KINDS,
    real_layer_state_dict,
    synthetic_hidden,
    text_config,
)
from models.autoports.google_gemma_4_12b_it.tests.test_functional_decoder import TracedDecode
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_12b_it.tt.fused_decoder import FusedDecoder

try:
    from tracy import signpost
except ImportError:  # profiler not built in

    def signpost(*args, **kwargs):
        pass


PREFILL_LEN = 4096
IMPLS = {"functional": FunctionalDecoder, "fused": FusedDecoder}


def _decode_window(device, dec, batch, x, pos, table, name, reps=50):
    trace = TracedDecode(dec, device, batch)
    trace.set_inputs(x, pos, table)
    trace.capture()
    for _ in range(3):
        trace.replay()
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # flush the device profiler buffers before each window
    signpost(name)
    trace.replay()
    ttnn.synchronize_device(device)
    signpost(name + "_END")
    ttnn.ReadDeviceProfiler(device)
    total = 0.0
    for _ in range(reps // 10):
        t0 = time.perf_counter()
        for _ in range(10):
            trace.replay()
        ttnn.synchronize_device(device)
        total += time.perf_counter() - t0
        ttnn.ReadDeviceProfiler(device)  # outside the timed region; keeps the profiler buffers from overflowing
    wall = total * 1e3 / (reps // 10 * 10)
    trace.release()
    return wall


@pytest.mark.parametrize("kind", list(LAYER_KINDS))
@pytest.mark.parametrize("impl", list(IMPLS))
def test_fused_perf(device_1x1, impl, kind):
    cfg = text_config()
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    batch = 32
    max_pages = (PREFILL_LEN + 64) // 64
    dec = IMPLS[impl].from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=device_1x1, num_blocks=batch * max_pages + 1
    )
    assert type(dec) is IMPLS[impl]
    pages = Pages(dec.num_blocks, max_pages, num_users=batch)
    for u in range(batch):
        pages.ensure(u, PREFILL_LEN + 1)
    table = pages.tt(device_1x1)
    x = synthetic_hidden(layer_idx, PREFILL_LEN + 1)
    tx = to_tt(x[:, :PREFILL_LEN], device_1x1)

    dec.prefill_forward(tx, page_table=table)  # compile
    ttnn.synchronize_device(device_1x1)
    ttnn.ReadDeviceProfiler(device_1x1)
    signpost("PERF_PREFILL")
    t0 = time.perf_counter()
    out = dec.prefill_forward(tx, page_table=table)
    ttnn.synchronize_device(device_1x1)
    prefill_ms = (time.perf_counter() - t0) * 1e3
    signpost("PERF_PREFILL_END")
    out.deallocate(True)
    # The other users' caches hold zeros below their positions: decode latency does not depend on
    # the cached values, only on positions (attended length) and batch.
    decode_b1 = _decode_window(
        device_1x1, dec, 1, x[:, PREFILL_LEN:], torch.tensor([PREFILL_LEN]), pages.table[:1], "PERF_DECODE"
    )
    xb = torch.cat([synthetic_hidden(layer_idx, 1, seed=10 + u) for u in range(batch)], 1)
    pos = torch.arange(PREFILL_LEN - batch + 1, PREFILL_LEN + 1)
    decode_b32 = _decode_window(device_1x1, dec, batch, xb, pos, pages.table, "PERF_DECODE_B32")
    ttnn.ReadDeviceProfiler(device_1x1)
    record(
        f"fused_perf[{impl},{kind}]",
        impl=impl,
        prefill_tokens=PREFILL_LEN,
        prefill_chunk=dec.prefill_chunk_size,
        prefill_wall_ms=round(prefill_ms, 3),
        decode_pos=PREFILL_LEN,
        decode_b1_trace_wall_avg50_ms=round(decode_b1, 4),
        decode_b32_trace_wall_avg50_ms=round(decode_b32, 4),
    )
