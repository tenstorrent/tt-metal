# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-op perf report for the windows the demo optimisation targets: Talker
prefill, one autoregressive decode frame, and the ECAPA speaker encoder.

Every window here **replays the Metal trace the demo replays**, exactly once,
between a ``start`` / ``stop`` signpost pair. That is the difference from the
older ``test_qwen3_tts_profile_*`` tests, which profile untraced passes of the
same bodies: an untraced pass has the same kernel graph, but every op waits on
the host, so the report's Op-to-Op Gap column is host dispatch rather than
anything you can optimise. On the AR-step-0 capture that gap totalled 80 s
against 46 ms of device time.

Windows (one Tracy capture each — ``-k`` selects one)::

    test_prefill[32]        Talker prefill trace, bucket 32     ── prefill ms
    test_prefill[64]        Talker prefill trace, bucket 64
    test_prefill[128]       Talker prefill trace, bucket 128
    test_prefill[demo]      whichever bucket the demo prompt takes

    test_decode_talker      Talker decode trace (28 layers + codec_head)
    test_decode_cp          fused CodePredictor frame trace           ── decode
    test_decode_frame       CP frame + Talker decode = one AR frame   ── throughput
    test_speaker_encoder    ECAPA forward trace (jim_reference mel)   ── TTFA

Splitting them is deliberate. The whole AR frame is ~4,200 device ops and the
profiler's DRAM buffer defaults to **1,000 programs**; past that it silently
drops markers and the report is partial — which is what a report full of
``TilizeDeviceOperation`` and no signposts means. Run the driver, which sets
``--op-support-count`` and fails the run if any marker was dropped::

    ./models/demos/qwen3_tts/tests/qwen3_tts_perf_report.sh

Or one window by hand (N300 device 1; see the driver for the full env)::

    python -m tracy -p -v -r --op-support-count 20000 -m pytest -s -q \\
      models/demos/qwen3_tts/tests/test_qwen3_tts_perf_report.py -k test_decode_frame

    CSV=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
    tt-perf-report --start-signpost start --end-signpost stop --no-stacked-report "$CSV"

``test_decode_frame`` also carries inner signposts (``cp_frame_start`` /
``cp_frame_stop`` / ``talker_decode_start`` / ``talker_decode_stop``) so the two
halves can be sliced out of the same capture.

Each test also prints a wall-clock median over ``QWEN3_TTS_PERF_REPS`` replays
taken *outside* the profiled window — that is the ms/frame number to track across
optimisation attempts, while the Tracy report says where it goes. It defaults to
10 normally and to **0 under the device profiler**, because every replay writes
markers for every op on every core and ten of them bury post-processing in
gigabytes. So the driver runs each window twice: a plain pass for the milliseconds
and a Tracy pass for the ops.
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tests import qwen3_tts_profile_demo_common as _common
from models.demos.qwen3_tts.tests.qwen3_tts_profile_demo_common import (
    allocate_talker_kv,
    build_fused_cp_profile_buffers,
    build_or_load_demo_icl_state,
    build_talker_decode_profile_buffers,
    capture_speaker_encoder_forward_trace,
    capture_talker_decode_trace,
    capture_talker_prefill_trace,
    demo_target_text,
    flush_profiler,
    pad_inputs_to_demo_bucket,
    run_talker_decode_untraced,
    upload_prefill_embeds,
)

# Rebound rather than imported by name. These two are pytest fixtures, referenced only
# as test parameters, so `from ... import profile_device, demo_model` looks like a pair
# of unused imports and the repo's autoflake hook (--remove-all-unused-imports over
# models/) deletes them on commit — leaving "fixture 'profile_device' not found" at run
# time. An assignment is not an import, so it survives.
profile_device = _common.profile_device
demo_model = _common.demo_model

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_a, **_k):
        pass


# Device ops per window, for the profiler-budget check below. Measured on N300 TP=2
# (per chip, as ops_list.md counts them) and rounded up; N150 differs mainly by losing
# the collectives. Only the order of magnitude matters — this gates the budget check.
_WINDOW_OPS = {
    "prefill": 800,
    "decode_talker": 1600,
    "decode_cp": 4400,
    "decode_frame": 6000,
    "speaker_encoder": 500,
}
_DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT = 1000


def _reps() -> int:
    """Wall-clock replays to time. Zero under the device profiler.

    Every replay is recorded by the device profiler, and a replay of the AR frame is
    ~4,200 ops x 64 cores x 5 RISCs of markers. Ten of them push
    ``profile_log_device.csv`` past a gigabyte and post-processing past 9 GB of RSS,
    for a number the same test already produces in a quarter of the time when it is
    not being profiled. The driver runs the timing pass separately.
    """
    default = "0" if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "0" else "10"
    return int(os.environ.get("QWEN3_TTS_PERF_REPS", default))


def _check_profiler_budget(window: str) -> None:
    """Warn when the profiler's DRAM buffer is too small for this window.

    Over the budget the device drops markers, logs "Profiler DRAM buffers were full",
    and the CSV comes back partial *without* failing anything — the failure mode this
    file exists to avoid. ``--op-support-count`` on ``python -m tracy`` sets it.
    """
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "0":
        return
    budget = int(os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", _DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT))
    need = _WINDOW_OPS[window]
    if budget < need:
        pytest.fail(
            f"profiler budget {budget} programs < ~{need} device ops in the '{window}' window. "
            f"Markers would be dropped and the report would be silently partial. "
            f"Re-run with: python -m tracy ... --op-support-count {max(need * 4, 20000)}"
        )
    print(f"[perf_report] window={window} ~{need} ops, profiler budget {budget} programs")


def _ms(median_ms: float) -> str:
    if not median_ms:
        return "wall clock not measured (skipped under the profiler; the driver times it separately)"
    return f"{median_ms:.2f} ms (median of {_reps()} traced replays)"


def _replay(device, trace_ids, reps: int) -> float:
    """Median wall-clock ms over ``reps`` replays of ``trace_ids``, in order.

    ``reps == 0`` still replays once — the profiled replay must not be the first one
    after capture, which pays a cold trace-dispatch path worth 10-15 ms of variance.
    """
    if reps <= 0:
        for tid in trace_ids:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        return 0.0
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for tid in trace_ids:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _prefill_state(device, model, main_weights, bucket=None):
    """ICL embeds → bucket → KV caches → captured prefill trace, replayed once.

    Returns the state every decode window needs: a filled Talker KV cache and the
    prefill hidden the first CodePredictor frame reads.
    """
    talker_h = model.talker_config.hidden_size
    icl = build_or_load_demo_icl_state(device, model, main_weights, use_cache=True)
    flush_profiler(device)

    real_seq_len = icl["real_seq_len"]
    padded_inputs, demo_bucket = pad_inputs_to_demo_bucket(device, icl["inputs_embeds_tt"], real_seq_len, talker_h)
    if bucket is None:
        bucket = demo_bucket
    elif bucket != demo_bucket:
        # Profiling a bucket the prompt does not fill: pad/crop to the trace's width.
        # Prefill cost is set by the bucket, not by how much of it is real text.
        padded = torch.zeros(1, 1, bucket, talker_h, dtype=torch.bfloat16)
        from models.demos.qwen3_tts.tt.mesh_utils import to_torch as mesh_to_torch

        src = mesh_to_torch(padded_inputs).to(torch.bfloat16)
        n = min(bucket, src.shape[2])
        padded[:, :, :n, :] = src[:, :, :n, :]
        padded_inputs = ttnn.from_torch(
            padded,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    talker_kv_caches, max_talker_seq_len = allocate_talker_kv(device, model, bucket)
    prefill_trace = capture_talker_prefill_trace(device, model, bucket, talker_kv_caches)
    upload_prefill_embeds(device, prefill_trace, padded_inputs)
    ttnn.execute_trace(device, prefill_trace["trace_id"], cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    return {
        "icl": icl,
        "real_seq_len": real_seq_len,
        "demo_bucket": demo_bucket,
        "bucket": bucket,
        "prefill_trace": prefill_trace,
        "talker_kv_caches": talker_kv_caches,
        "max_talker_seq_len": max_talker_seq_len,
        "padded_inputs": padded_inputs,
    }


# ── Prefill ───────────────────────────────────────────────────────────────────


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("bucket", [None, 32, 64, 128], ids=["demo", "32", "64", "128"])
def test_prefill(profile_device, demo_model, bucket):
    """One replay of the Talker prefill trace the demo replays for this bucket."""
    device = profile_device
    model, main_weights = demo_model
    _check_profiler_budget("prefill")
    flush_profiler(device)

    st = _prefill_state(device, model, main_weights, bucket=bucket)
    tid = st["prefill_trace"]["trace_id"]

    median_ms = _replay(device, [tid], _reps())
    ttnn.synchronize_device(device)
    flush_profiler(device)

    signpost("start")
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("stop")

    print(
        f"\n[perf_report] prefill bucket={st['bucket']} "
        f"(demo prompt: {st['real_seq_len']} tok -> bucket {st['demo_bucket']}, "
        f"{len(demo_target_text())} chars): {_ms(median_ms)}"
    )


# ── Decode ────────────────────────────────────────────────────────────────────


def _decode_state(device, model, main_weights, *, need_cp: bool):
    """Prefill state + the decode traces the demo captures, in the demo's capture order.

    Order is load-bearing: a trace replay writes to the addresses it wrote to during
    capture, so every buffer a trace reads must be allocated before that trace is
    captured (see the unsafe-allocation notes in ``server.generate_codes_ttnn``).
    """
    st = _prefill_state(device, model, main_weights)
    ttnn.synchronize_device(device)
    flush_profiler(device)

    decode_bufs = build_talker_decode_profile_buffers(
        device,
        model,
        real_seq_len=st["real_seq_len"],
        max_talker_seq_len=st["max_talker_seq_len"],
        talker_kv_caches=st["talker_kv_caches"],
    )
    cp_bufs = None
    if need_cp:
        cp_bufs = build_fused_cp_profile_buffers(
            device,
            model,
            st["icl"]["config"],
            talker_hidden_tt=st["prefill_trace"]["hidden_out"],
            trailing_text_hidden=st["icl"]["trailing_text_hidden"],
            tts_pad_embed=st["icl"]["tts_pad_embed"],
            code_pred_embeds=st["icl"]["code_pred_embeds"],
            talker_embed_dst_tt=decode_bufs["trace_embed_tt"],
        )

    run_talker_decode_untraced(decode_bufs)  # compiles the kernels + creates the CCL semaphores
    ttnn.synchronize_device(device)
    talker_trace = capture_talker_decode_trace(device, model, decode_bufs)

    cp_trace = None
    if need_cp:
        from models.demos.qwen3_tts.tt.server import capture_fused_cp_trace

        cp_trace = capture_fused_cp_trace(
            device,
            model,
            st["icl"]["config"],
            cp_trans_mat=cp_bufs["cp_trans_mat"],
            cp_kv_caches_persistent=cp_bufs["cp_kv_caches"],
            cp_kv_zero_hosts=cp_bufs["cp_kv_zero_hosts"],
            cp_prefill_embed_tt=cp_bufs["cp_prefill_embed_tt"],
            cp_prefill_mask_tt=cp_bufs["cp_prefill_mask_tt"],
            cp_prefill_cos_tt=cp_bufs["cp_prefill_cos_tt"],
            cp_prefill_sin_tt=cp_bufs["cp_prefill_sin_tt"],
            cp_prefill_mask_src=cp_bufs["cp_prefill_mask_src"],
            cp_prefill_cos_src=cp_bufs["cp_prefill_cos_src"],
            cp_prefill_sin_src=cp_bufs["cp_prefill_sin_src"],
            cp_decode_embed_tt=cp_bufs["cp_decode_embed_tt"],
            cp_decode_cos_tts=cp_bufs["cp_decode_cos_tts"],
            cp_decode_sin_tts=cp_bufs["cp_decode_sin_tts"],
            cp_decode_mask_tts=cp_bufs["cp_decode_mask_tts"],
            talker_hidden_src_tt=cp_bufs["talker_hidden_src_tt"],
            talker_embed_dst_tt=cp_bufs["talker_embed_dst_tt"],
            codec_embed_tt=cp_bufs["codec_embed_tt"],
            cp_embed_tts=cp_bufs["cp_embed_tts"],
            talker_h=cp_bufs["talker_h"],
            sampler=cp_bufs["sampler"],
            tok_bufs=cp_bufs["tok_bufs"],
            trail_row_tt=cp_bufs["trail_row_tt"],
            trail_row_h2d=cp_bufs["trail_row_h2d"],
        )
        ttnn.execute_trace(device, cp_trace.trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(device)

    st.update({"decode_bufs": decode_bufs, "cp_bufs": cp_bufs, "talker_trace": talker_trace, "cp_trace": cp_trace})
    return st


def _talker_h2d(decode_ctx) -> None:
    """The four per-frame H2Ds the AR loop issues before the Talker decode trace."""
    ttnn.copy_host_to_device_tensor(decode_ctx["cos_h2d"], decode_ctx["trace_cos_tt"])
    ttnn.copy_host_to_device_tensor(decode_ctx["sin_h2d"], decode_ctx["trace_sin_tt"])
    ttnn.copy_host_to_device_tensor(decode_ctx["cur_pos_h2d"], decode_ctx["trace_cur_pos_tt"])
    ttnn.copy_host_to_device_tensor(decode_ctx["mask_h2d"], decode_ctx["trace_mask_tt"])


def _cp_h2d(device, cp_trace, token_0: int = 0) -> None:
    """The per-frame host work the AR loop does before the fused CP trace.

    Fresh Gumbel noise, the code-0 token id from the Talker, and this frame's
    trailing-text row. ``token_0`` only picks an embedding row, so any valid id
    profiles the same; the real one is not worth a prefill-logits D2H here.
    """
    from models.demos.qwen3_tts.tt.mesh_utils import is_mesh_device
    from models.demos.qwen3_tts.tt.server import _replicate_mapper

    cp_trace.sampler.refresh_noise()
    mapper = _replicate_mapper(device) if is_mesh_device(device) else None
    tok_host = ttnn.from_torch(
        torch.tensor([[token_0]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mapper,
    )
    ttnn.copy_host_to_device_tensor(tok_host, cp_trace.tok_bufs[0])
    ttnn.copy_host_to_device_tensor(cp_trace.trail_row_h2d[0], cp_trace.trail_row_tt)


@pytest.mark.timeout(2400)
def test_decode_talker(profile_device, demo_model):
    """One replay of the Talker decode trace (28 layers + codec_head)."""
    device = profile_device
    model, main_weights = demo_model
    _check_profiler_budget("decode_talker")
    flush_profiler(device)

    st = _decode_state(device, model, main_weights, need_cp=False)
    tid = st["talker_trace"]["trace_id"]

    _talker_h2d(st["decode_bufs"])
    median_ms = _replay(device, [tid], _reps())
    ttnn.synchronize_device(device)
    flush_profiler(device)

    signpost("start")
    _talker_h2d(st["decode_bufs"])
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("stop")

    print(f"\n[perf_report] talker decode: {_ms(median_ms)}")


@pytest.mark.timeout(2400)
def test_decode_cp(profile_device, demo_model):
    """One replay of the fused CodePredictor frame trace.

    That single trace is CP prefill (seq=2) + 13 CP decode steps + 15 sampling chains
    + the accumulated Talker input embedding — 75 CP layer evaluations, and the larger
    half of the AR frame.
    """
    device = profile_device
    model, main_weights = demo_model
    _check_profiler_budget("decode_cp")
    flush_profiler(device)

    st = _decode_state(device, model, main_weights, need_cp=True)
    cp_trace = st["cp_trace"]

    _cp_h2d(device, cp_trace)
    median_ms = _replay(device, [cp_trace.trace_id], _reps())
    ttnn.synchronize_device(device)
    flush_profiler(device)

    signpost("start")
    _cp_h2d(device, cp_trace)
    ttnn.execute_trace(device, cp_trace.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("stop")

    print(f"\n[perf_report] fused CP frame: {_ms(median_ms)}")


@pytest.mark.timeout(2400)
def test_decode_frame(profile_device, demo_model):
    """One full AR frame: fused CP frame then Talker decode — the throughput unit.

    Inner signposts let the two halves be sliced out of this one capture::

        tt-perf-report --start-signpost cp_frame_start --end-signpost cp_frame_stop ...
        tt-perf-report --start-signpost talker_decode_start --end-signpost talker_decode_stop ...
    """
    device = profile_device
    model, main_weights = demo_model
    _check_profiler_budget("decode_frame")
    flush_profiler(device)

    st = _decode_state(device, model, main_weights, need_cp=True)
    cp_trace = st["cp_trace"]
    talker_tid = st["talker_trace"]["trace_id"]

    _cp_h2d(device, cp_trace)
    _talker_h2d(st["decode_bufs"])
    median_ms = _replay(device, [cp_trace.trace_id, talker_tid], _reps())
    ttnn.synchronize_device(device)
    flush_profiler(device)

    signpost("start")
    signpost("cp_frame_start")
    _cp_h2d(device, cp_trace)
    ttnn.execute_trace(device, cp_trace.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("cp_frame_stop")
    signpost("talker_decode_start")
    _talker_h2d(st["decode_bufs"])
    ttnn.execute_trace(device, talker_tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("talker_decode_stop")
    signpost("stop")

    rate = ""
    if median_ms:
        fps = 1e3 / median_ms
        rate = f" = {fps:.2f} frames/s = {fps / 12.5:.2f}x realtime at 12.5 fps"
    print(f"\n[perf_report] AR frame (CP + Talker decode): {_ms(median_ms)}{rate}")


# ── Speaker encoder ───────────────────────────────────────────────────────────


@pytest.mark.timeout(2400)
def test_speaker_encoder(profile_device, demo_model):
    """One replay of the ECAPA forward trace (``QWEN3_TTS_SE_TRACE=1`` path).

    Mel length follows ``jim_reference.wav`` (~384). This is the whole encoder on
    device — entry TDNN, 3× SERes2Net, MFA, ASP, FC — not the partial
    ``speaker_tdnn`` / ``speaker_block`` single-layer slices (those profile
    untraced subgraphs with synthetic weights and miss most conv work).
    """
    device = profile_device
    model, main_weights = demo_model
    _check_profiler_budget("speaker_encoder")
    flush_profiler(device)

    st = capture_speaker_encoder_forward_trace(device, model, main_weights)
    tid = st["trace_id"]

    median_ms = _replay(device, [tid], _reps())
    ttnn.synchronize_device(device)
    flush_profiler(device)

    signpost("start")
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("stop")

    print(f"\n[perf_report] speaker_encoder mel_T={st['mel_len']}: {_ms(median_ms)}")
