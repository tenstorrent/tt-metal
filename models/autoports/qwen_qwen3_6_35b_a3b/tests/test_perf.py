# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Warmed prefill / traced-warmed decode performance for the functional decoder.

Driven under the Tracy profiler by ``tests/run_perf.sh``; the signposts below bound the
measured window that ``tt-perf-report`` filters on:

    prefill: PERF_PREFILL      .. PERF_PREFILL_END
    decode:  PERF_DECODE       .. PERF_DECODE_END

Shape/iteration knobs (env): ``QWEN36_PERF_PREFILL_SEQ`` (2048),
``QWEN36_PERF_DECODE_BATCH`` (32), ``QWEN36_PERF_DECODE_POS`` (4095),
``QWEN36_PERF_PREFILL_ITERS`` (2), ``QWEN36_PERF_DECODE_ITERS`` (8).

Decode is timed at ``QWEN36_PERF_DECODE_POS`` rather than immediately after the seeding
prefill: paged-SDPA cost grows with the context length, so a position right after a 512-token
prefill would report an unrepresentatively cheap decode. Flash attention is data-independent,
so the cache rows between the prefill length and the timed position holding zeros does not
change the timing (only the values, which this test does not check — ``test_functional_decoder``
and ``test_long_context`` own correctness).

Never run these with ``TT_METAL_WATCHER`` set — watcher and the device profiler contend for
the same debug resources.
"""

import os
import time

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.harness import TracedDecode, build_layer_pair, record, to_tt_prefill
from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref

KINDS = ["linear", "full"]

PREFILL_SEQ = int(os.environ.get("QWEN36_PERF_PREFILL_SEQ", 2048))
DECODE_BATCH = int(os.environ.get("QWEN36_PERF_DECODE_BATCH", 32))
DECODE_POS = int(os.environ.get("QWEN36_PERF_DECODE_POS", 4095))
#: Iteration counts are deliberately small: the device profiler records every op, and one warmed
#: prefill of this layer already issues ~450 (the 32-step delta-rule scan plus 4 MoE chunks).
#: tt-perf-report aggregates per-op device time over the signposted window, so a couple of warmed
#: iterations is a complete measurement.
PREFILL_ITERS = int(os.environ.get("QWEN36_PERF_PREFILL_ITERS", 2))
DECODE_ITERS = int(os.environ.get("QWEN36_PERF_DECODE_ITERS", 8))
#: The layer is built at this `supported_context`, **not** the advertised 262144. Two reasons, both
#: worth knowing when reading the numbers: the paged K/V for batch 32 at the full context is 16 GiB
#: (see `test_long_context.py::test_max_batch_full_context_capacity`), which leaves no room for a
#: profiler buffer; and decode cost grows with `cur_pos`, so a fixed 4095 is what makes the
#: prefill/decode rows comparable across runs. It does mean the decode rows are *not* the
#: advertised-context latency: the decode SDPA alone is 11.5 ms/call at 262144 keys against ~1 ms
#: here (`logs/diag_sdpa_decode.txt`), so the advertised-context step is ~10 ms slower than the
#: table shows. Override with QWEN36_PERF_CONTEXT / QWEN36_PERF_DECODE_POS to measure elsewhere.
PERF_CONTEXT = int(os.environ.get("QWEN36_PERF_CONTEXT", 8192))


def _signpost(name):
    from tracy import signpost

    signpost(name)


def _write_summary(payload):
    """Host-side wall-clock provenance row alongside the tt-perf-report device numbers."""
    record([payload], "perf_host_summary")


@pytest.fixture(scope="module")
def perf_device():
    if os.environ.get("TT_METAL_WATCHER"):
        pytest.fail("do not profile with TT_METAL_WATCHER set; run watcher separately")
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield device
    # Flush the device profiler before close. Without this, closing a device that still holds a
    # layer's ~1.5 GiB of weights plus a full profiler buffer segfaults inside close_mesh_device.
    ttnn.ReadDeviceProfiler(device)
    ttnn.synchronize_device(device)
    ttnn.close_mesh_device(device)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(7200)
def test_perf_prefill(perf_device, kind):
    """Warmed prefill: compile + warm up, synchronize, then a signposted measured window."""
    pair = build_layer_pair(perf_device, kind=kind, max_batch_size=1, supported_context=PERF_CONTEXT, build_hf=False)
    x = to_tt_prefill(perf_device, ref.synthetic_hidden_states(pair.hf_config, 1, PREFILL_SEQ, seed=1))

    for _ in range(2):  # compile every program, warm the caches
        pair.tt.reset_state()
        ttnn.deallocate(pair.tt.prefill_forward(x, user_id=0, page_table=pair.page_table))
    ttnn.synchronize_device(perf_device)

    _signpost("PERF_PREFILL")
    started = time.perf_counter()
    for _ in range(PREFILL_ITERS):
        out = pair.tt.prefill_forward(x, user_id=0, page_table=pair.page_table)
        ttnn.deallocate(out)
    ttnn.synchronize_device(perf_device)
    elapsed = time.perf_counter() - started
    _signpost("PERF_PREFILL_END")

    ttnn.deallocate(x)
    pair.tt.release()
    _write_summary(
        {
            "mode": "prefill",
            "kind": kind,
            "seq_len": PREFILL_SEQ,
            "batch": 1,
            "iters": PREFILL_ITERS,
            "host_wall_s_total": round(elapsed, 6),
            "host_wall_ms_per_iter": round(1e3 * elapsed / PREFILL_ITERS, 3),
            "tokens_per_s_host": round(PREFILL_ITERS * PREFILL_SEQ / elapsed, 1),
            "supported_context": PERF_CONTEXT,
        }
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(7200)
def test_perf_decode(perf_device, kind):
    """Traced warmed decode: capture once, warm the replay, then a signposted window."""
    pair = build_layer_pair(
        perf_device,
        kind=kind,
        max_batch_size=DECODE_BATCH,
        supported_context=PERF_CONTEXT,
        build_hf=False,
    )
    pair.tt.reset_state()
    prefill_len = 512
    for user_id in range(DECODE_BATCH):
        tt_x = to_tt_prefill(perf_device, ref.synthetic_hidden_states(pair.hf_config, 1, prefill_len, seed=user_id))
        ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table))
        ttnn.deallocate(tt_x)

    traced = TracedDecode(pair)  # compiles, then captures
    try:
        tokens = ref.synthetic_hidden_states(pair.hf_config, DECODE_BATCH, 1, seed=77).reshape(DECODE_BATCH, 1, -1)
        positions = torch.full((DECODE_BATCH,), DECODE_POS, dtype=torch.int32)
        for _ in range(3):  # warm the replay path
            traced.run(tokens, positions)
        ttnn.synchronize_device(perf_device)

        _signpost("PERF_DECODE")
        started = time.perf_counter()
        for _ in range(DECODE_ITERS):
            ttnn.execute_trace(perf_device, traced.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(perf_device)
        elapsed = time.perf_counter() - started
        _signpost("PERF_DECODE_END")
    finally:
        traced.release()
    pair.tt.release()

    _write_summary(
        {
            "mode": "decode",
            "kind": kind,
            "seq_len": 1,
            "current_pos": DECODE_POS,
            "batch": DECODE_BATCH,
            "iters": DECODE_ITERS,
            "traced": True,
            "host_wall_s_total": round(elapsed, 6),
            "host_wall_ms_per_iter": round(1e3 * elapsed / DECODE_ITERS, 3),
            "tokens_per_s_host": round(DECODE_ITERS * DECODE_BATCH / elapsed, 1),
            "supported_context": PERF_CONTEXT,
        }
    )
