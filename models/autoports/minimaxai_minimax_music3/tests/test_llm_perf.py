# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance evidence for ``MusicLLM``: warmed traced decode and warmed prefill, Tracy-signposted.

Pattern (functional-decoder skill): compile/warm, synchronize, signpost start, run the *warmed*
measured window, synchronize once, signpost end. Under ``python -m tracy -r -p -v -m pytest ...``
the signposts bound the window ``tt-perf-report`` sums; without Tracy the tests still record
wall-clock numbers to ``doc/llm/perf/*.json``.

    scripts/collect_llm_perf.sh decode
    scripts/collect_llm_perf.sh prefill
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.tt.constants import LLM_HIDDEN

pytestmark = [pytest.mark.hardware, pytest.mark.timeout(3600)]

DECODE_STEPS = int(os.environ.get("MM3_PERF_DECODE_STEPS", 20))
# One full-model step is ~1000 device ops, about the per-core capacity of the device profiler
# buffer, so profiled windows contain very few steps with a profiler drain between them.
PROFILED_DECODE_STEPS = int(os.environ.get("MM3_PERF_PROFILED_DECODE_STEPS", 2))


def _profiled() -> bool:
    """True under ``python -m tracy`` / ``TT_METAL_DEVICE_PROFILER=1`` (profiling inflates wall clock)."""
    return bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))


def _drain_profiler(mesh_device) -> None:
    """Flush the device profiler buffers right before a measured window (no-op when not profiling).

    Setup (model build, prefill, trace capture) emits thousands of ops; draining keeps the
    per-core profiler buffers from overflowing and makes the trace-replay ops the first device
    records after the start signpost.
    """
    if _profiled():
        ttnn.ReadDeviceProfiler(mesh_device)


def _signpost(name: str) -> None:
    try:
        from tracy import signpost
    except ImportError:  # plain pytest run without the profiler
        return
    signpost(name)


def _write(evidence_dir, name, payload):
    out = evidence_dir / "perf"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def _board_id() -> str:
    return os.environ.get("MM3_BOARD_ID", "000004613193411b (p300c, tt-smi -s)")


@pytest.fixture
def drained_llm(music_llm):
    """The session model with the device profiler drained after every prefill row / chunk while profiling.

    A 36-layer prefill row is ~1000 device ops (the per-core profiler buffer capacity), so without
    these drains process_ops_logs aborts with "Unexpected FW start ... profiler buffer filled up".
    """
    if _profiled():
        music_llm.after_prefill_chunk = lambda: ttnn.ReadDeviceProfiler(music_llm.mesh_device)
    _drain_profiler(music_llm.mesh_device)
    yield music_llm
    music_llm.after_prefill_chunk = None


def test_decode_perf(drained_llm, golden, evidence_dir):
    """Warmed traced decode step time at batch 2 (trace replay only inside the signposts)."""
    music_llm = drained_llm
    text_ids = golden["text_ids"]
    seq_len = text_ids.shape[1]
    music_llm.reset_cache()
    music_llm.prefill(music_llm.embed_tokens(text_ids))
    torch.manual_seed(0)
    x = torch.randn(music_llm.max_batch_size, LLM_HIDDEN) * 0.02

    steps = PROFILED_DECODE_STEPS if _profiled() else DECODE_STEPS
    _drain_profiler(music_llm.mesh_device)

    # Warm: captures the trace on the first call, replays on the second.
    music_llm.decode(x, seq_len)
    _drain_profiler(music_llm.mesh_device)
    music_llm.decode(x, seq_len + 1)
    ttnn.synchronize_device(music_llm.mesh_device)
    _drain_profiler(music_llm.mesh_device)
    stats_before = dict(music_llm.decode_stats)

    # Measured window: pure trace replays with device-advanced positions, one sync at the end.
    # Under the profiler each replay is followed by a buffer drain (host work, device time unaffected).
    _signpost("PERF_DECODE")
    t0 = time.perf_counter()
    for _ in range(steps):
        music_llm.decode_replay_only()
        if _profiled():
            ttnn.synchronize_device(music_llm.mesh_device)
            _drain_profiler(music_llm.mesh_device)
    ttnn.synchronize_device(music_llm.mesh_device)
    replay_s = (time.perf_counter() - t0) / steps
    _signpost("PERF_DECODE_END")
    _drain_profiler(music_llm.mesh_device)

    # Caller-visible step: refresh the input from host, replay, read hidden + logits back.
    t0 = time.perf_counter()
    for i in range(steps):
        music_llm.decode(x, seq_len + 2 + steps + i)
        _drain_profiler(music_llm.mesh_device)
    ttnn.synchronize_device(music_llm.mesh_device)
    e2e_s = (time.perf_counter() - t0) / steps
    stats_after = dict(music_llm.decode_stats)

    payload = {
        "batch": music_llm.max_batch_size,
        "steps": steps,
        "profiled": _profiled(),
        "traced_replay_ms_per_step": replay_s * 1e3,
        "end_to_end_ms_per_step_with_host_input_and_readback": e2e_s * 1e3,
        "tokens_per_s_per_user": 1.0 / e2e_s,
        "start_position": seq_len,
        "signposts": ["PERF_DECODE", "PERF_DECODE_END"],
        "host_counters_measured_window": {k: stats_after[k] - stats_before[k] for k in stats_after},
        "board": _board_id(),
    }
    logger.info(f"decode perf: {payload}")
    _write(evidence_dir, "decode_profiled" if _profiled() else "decode", payload)
    assert payload["host_counters_measured_window"]["trace_captures"] == 0


@pytest.mark.parametrize("seq_len", [104, 5000])
def test_prefill_perf(drained_llm, golden, seq_len, evidence_dir):
    """Warmed prefill wall time for the golden prompt length and the maximum prompt (chunked)."""
    music_llm = drained_llm
    if seq_len == 104:
        text_ids = golden["text_ids"]
        emb = ttnn.to_torch(music_llm.embed_tokens(text_ids))[0].float()
    else:
        torch.manual_seed(1)
        emb = torch.randn(music_llm.max_batch_size, seq_len, LLM_HIDDEN) * 0.02
    music_llm.reset_cache()
    music_llm.prefill(emb)  # warm / compile
    ttnn.synchronize_device(music_llm.mesh_device)
    _drain_profiler(music_llm.mesh_device)

    # Under the profiler the drain hook also runs after each row / chunk inside this window
    # (host work only; per-op device durations are what tt-perf-report sums).
    _signpost("PERF_PREFILL")
    t0 = time.perf_counter()
    music_llm.prefill(emb)
    ttnn.synchronize_device(music_llm.mesh_device)
    dt = time.perf_counter() - t0
    _signpost("PERF_PREFILL_END")
    _drain_profiler(music_llm.mesh_device)

    payload = {
        "batch_rows": music_llm.max_batch_size,
        "seq_len": seq_len,
        "profiled": _profiled(),
        "warmed_wall_s_both_rows": dt,
        "warmed_wall_s_per_row": dt / music_llm.max_batch_size,
        "tokens_per_s": seq_len * music_llm.max_batch_size / dt,
        "chunked": seq_len > music_llm.args.max_prefill_chunk_size,
        "max_prefill_chunk_size": music_llm.args.max_prefill_chunk_size,
        "signposts": ["PERF_PREFILL", "PERF_PREFILL_END"],
        "board": _board_id(),
    }
    logger.info(f"prefill perf: {payload}")
    _write(evidence_dir, f"prefill_{seq_len}" + ("_profiled" if _profiled() else ""), payload)
