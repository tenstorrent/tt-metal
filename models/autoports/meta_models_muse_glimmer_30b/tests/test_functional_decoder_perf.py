# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Warmed prefill and traced warmed decode performance for the Muse-Glimmer decoder layer.

Each test follows the same shape: build the layer, fill a paged cache, run the path once
to compile and populate the program cache, synchronize, emit a Tracy signpost, run the
*warmed* measured window, synchronize once, emit the closing signpost. Device-time tables
come from ``tt-perf-report`` over the signposted window; wall-clock latency over the
measured iterations is written to ``doc/functional_decoder/perf/perf_summary.json``.

Collect a Tracy ops CSV with, for example::

    python -m tracy -r -p -v -m pytest \
        models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder_perf.py \
        -k "decode and sliding_rope"

Signposts: ``PERF_PREFILL`` / ``PERF_PREFILL_END`` and ``PERF_DECODE`` / ``PERF_DECODE_END``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.reference import hf_reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests import decoder_test_utils as U
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import _round_up

KIND_IDS = ["sliding_rope", "full_nope"]
PERF_DIR = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder" / "perf"
PREFILL_ITERS = int(os.environ.get("MUSE_GLIMMER_PREFILL_ITERS", "5"))
# Under the profiler the per-core buffers hold a bounded number of ops; 32 replays of a
# ~31-op decode step overflowed them for some configurations and silently dropped rows from
# the report, so the profiled window is shorter. Unprofiled runs use the full count.
DECODE_ITERS = int(os.environ.get("MUSE_GLIMMER_DECODE_ITERS", "32"))
PROFILED_DECODE_ITERS = int(os.environ.get("MUSE_GLIMMER_PROFILED_DECODE_ITERS", "12"))


def _signpost(name: str) -> None:
    from tracy import signpost

    signpost(name)


def _profiled() -> bool:
    """True when running under ``python -m tracy`` / the device profiler.

    Profiling inflates wall clock substantially, so profiled and unprofiled runs are kept as
    separate records: the profiled run is what the ``tt-perf-report`` device-time tables come
    from, the unprofiled run is the honest end-to-end latency.
    """
    return bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))


def _drain_profiler(mesh_device) -> None:
    """Flush device profiler buffers before entering a measured window.

    Setup work (filling 32 users' caches with 32 sequential prefills) emits enough ops to
    fill the per-core profiler buffers, after which ``process_ops_logs.py`` aborts with
    "Unexpected FW start ... Other cores might have their profiler buffer filled up ...
    run read device profiler more often". Draining between setup and the measured window
    keeps the measured ops intact. No-op when not profiling.
    """
    if _profiled():
        ttnn.ReadDeviceProfiler(mesh_device)


def _record(entry: dict) -> None:
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    path = PERF_DIR / "perf_summary.json"
    records = json.loads(path.read_text())["records"] if path.is_file() else []

    def key(record):
        return (
            record["measurement"],
            record["kind"],
            record.get("seq_len"),
            record.get("batch"),
            record.get("profiled"),
        )

    by_key = {key(r): r for r in records}
    by_key[key(entry)] = entry
    path.write_text(
        json.dumps(
            {"records": sorted(by_key.values(), key=lambda r: (r["measurement"], r["kind"], str(key(r))))}, indent=2
        )
        + "\n"
    )


def _setup(decoder, mesh_device, *, batch, total_tokens, block_size, page_seed=4242):
    blocks_per_seq = _round_up(total_tokens, block_size) // block_size
    total_blocks = blocks_per_seq * batch + 2
    page_table = U.build_page_table(
        batch=batch, blocks_per_seq=blocks_per_seq, total_blocks=total_blocks, seed=page_seed
    )
    kv_cache = decoder.allocate_kv_cache(
        batch_size=batch, max_seq_len=blocks_per_seq * block_size, num_blocks=total_blocks
    )
    return kv_cache, U.page_table_to_device(page_table, mesh_device)


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize("seq_len", [4096, 8192])
def test_prefill_perf(
    kind_id, seq_len, kinds, text_config, synthetic_state_dicts, build_cached_decoder, mg_mesh_device
):
    """Warmed (program-cache hot) paged prefill, measured between Tracy signposts."""
    kind = kinds[kind_id]
    decoder = build_cached_decoder(kind.layer_idx, synthetic_state_dicts[kind.layer_idx], block_size=64)
    kv_cache, page_table_tt = _setup(decoder, mg_mesh_device, batch=1, total_tokens=seq_len + 1, block_size=64)

    hidden = R.unit_rms_hidden_states((1, seq_len, text_config.hidden_size), seed=61)
    hidden_tt = U.prefill_input(hidden, mg_mesh_device)

    out = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
    out.deallocate(True)
    ttnn.synchronize_device(mg_mesh_device)
    _drain_profiler(mg_mesh_device)

    _signpost("PERF_PREFILL")
    start = time.perf_counter()
    for _ in range(PREFILL_ITERS):
        out = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
        out.deallocate(True)
    ttnn.synchronize_device(mg_mesh_device)
    elapsed = time.perf_counter() - start
    _signpost("PERF_PREFILL_END")

    _record(
        {
            "measurement": "prefill_warmed",
            "kind": kind_id,
            "layer_idx": kind.layer_idx,
            "seq_len": seq_len,
            "batch": 1,
            "iterations": PREFILL_ITERS,
            "wall_clock_ms_per_iter": 1000.0 * elapsed / PREFILL_ITERS,
            "tokens_per_second": PREFILL_ITERS * seq_len / elapsed,
            "signposts": ["PERF_PREFILL", "PERF_PREFILL_END"],
            "profiled": _profiled(),
        }
    )


@pytest.mark.parametrize("kind_id", KIND_IDS)
@pytest.mark.parametrize("batch", [1, 32])
def test_decode_perf(kind_id, batch, kinds, text_config, synthetic_state_dicts, build_cached_decoder, mg_mesh_device):
    """Traced warmed decode: capture once, then replay the trace inside the signposts.

    The measured window contains only ``ttnn.execute_trace`` calls — no tensor allocation,
    no host input updates — so the reported device time is the decode step itself. The
    per-user position is held constant across the replays (a decode step is idempotent in
    the cache for a fixed input and position), which keeps host work out of the window.
    """
    kind = kinds[kind_id]
    decoder = build_cached_decoder(kind.layer_idx, synthetic_state_dicts[kind.layer_idx], block_size=64)
    context = 4096
    kv_cache, page_table_tt = _setup(decoder, mg_mesh_device, batch=batch, total_tokens=context + 1, block_size=64)

    # Fill each user's cache with its own batch-1 prefill: a 32 x 4096 batched prefill
    # would allocate a ~10 GB SwiGLU intermediate, and this test only needs the cache
    # populated, not a batched-prefill measurement.
    prompt = R.unit_rms_hidden_states((1, context, text_config.hidden_size), seed=62)
    prompt_tt = U.prefill_input(prompt, mg_mesh_device)
    for user in range(batch):
        decoder.prefill_forward(prompt_tt, kv_cache=kv_cache, page_table=page_table_tt, user_ids=[user]).deallocate(
            True
        )
    prompt_tt.deallocate(True)
    ttnn.synchronize_device(mg_mesh_device)
    _drain_profiler(mg_mesh_device)

    hidden_d = R.unit_rms_hidden_states((batch, 1, text_config.hidden_size), seed=63)
    x_dev = U.decode_input(hidden_d, mg_mesh_device)
    pos_dev, rope_dev = U.position_tensors([context] * batch, mg_mesh_device)

    decoder.decode_forward(
        x_dev, kv_cache=kv_cache, page_table=page_table_tt, current_pos=pos_dev, rope_idxs=rope_dev
    ).deallocate(True)
    ttnn.synchronize_device(mg_mesh_device)

    trace_id = ttnn.begin_trace_capture(mg_mesh_device, cq_id=0)
    out_dev = decoder.decode_forward(
        x_dev, kv_cache=kv_cache, page_table=page_table_tt, current_pos=pos_dev, rope_idxs=rope_dev
    )
    ttnn.end_trace_capture(mg_mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mg_mesh_device)

    ttnn.execute_trace(mg_mesh_device, trace_id, cq_id=0, blocking=True)  # warm replay
    ttnn.synchronize_device(mg_mesh_device)
    _drain_profiler(mg_mesh_device)

    iterations = PROFILED_DECODE_ITERS if _profiled() else DECODE_ITERS
    _signpost("PERF_DECODE")
    start = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(mg_mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mg_mesh_device)
    elapsed = time.perf_counter() - start
    _signpost("PERF_DECODE_END")

    assert out_dev.shape[-2] == batch
    _record(
        {
            "measurement": "decode_traced_warmed",
            "kind": kind_id,
            "layer_idx": kind.layer_idx,
            "batch": batch,
            "context": context,
            "iterations": iterations,
            "wall_clock_ms_per_iter": 1000.0 * elapsed / iterations,
            "tokens_per_second": iterations * batch / elapsed,
            "signposts": ["PERF_DECODE", "PERF_DECODE_END"],
            "profiled": _profiled(),
        }
    )
    ttnn.release_trace(mg_mesh_device, trace_id)
