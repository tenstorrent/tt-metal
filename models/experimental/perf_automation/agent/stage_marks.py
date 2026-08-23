# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage boundaries for the TRACY run, which is the only run that records per-op data.

THE TOOL MEASURES TWICE, FOR TWO DIFFERENT THINGS:

    A  tracy eager profile     PROFILER=1, TT_PERF_TRACE=0, coverage depth (2 layers)
       -> per-op records: op_code, shape, FIDELITY, cores, grid, memory, bytes, device_ms
       -> the slice is sized to hold every distinct op, so op PROPERTIES are authoritative here

    B  full-depth stopwatch    profiler popped, all layers, trace+1cq
       -> TRACE_STAGE_MS / TRACE_STAGE_BYTES: times and totals. No per-op anything.

A traced replay runs as one fused program and emits NO per-op device data, which is why A is eager
and why fidelity exists ONLY in A. Five earlier attempts at per-stage fidelity were written into
trace_replay -- B's machinery -- which never produces a fidelity field at all.

WHAT A CANNOT DO BY ITSELF. Its measured call is `pipe.run_head(...)`: encode, prefill and every
decode step inside one opaque call. The ops arrive unlabelled, so the report takes the dominant
fidelity across the whole pile and applies it to all three stacks -- right only while they agree.

WHAT THIS ADDS. A SECOND, MARKED PASS after the measured one: each stage the model declares, run
once through its own `<stage>_trace_step`, bracketed by tracy signposts. tt-perf-report already
slices a capture between two signposts, so the report can then price each stack at its own peak.

WHY A SECOND PASS RATHER THAN REPLACING run_head. 94 of the 112 ops in a real capture carry no
parseable shape, so there is no way to prove by inspection that per-stage steps cover the same op
set -- and an op that only run_head reaches would vanish from the ladder's view. The measured region
keeps its exact op set; the marked pass is additive and is used for fidelity rollup only. The two
are kept apart by the conventional start/stop pair, which resolve_signposts already looks for and
refine() already slices on, so the main report sees exactly what it saw before.
"""
from __future__ import annotations

import sys


def signpost(name: str) -> None:
    """Emit one tracy signpost. Best-effort: a mark that cannot be written costs the split, not the run.

    tracy.signpost goes out through ttnn.tracy_message -- the same channel the op records travel --
    and process_ops_logs writes it as a row whose OP TYPE is "signpost", which is what
    tt-perf-report slices on. So a mark is a real row in the op stream, not a host annotation.
    """
    try:
        from tracy import signpost as _sp

        _sp(name)
    except Exception as exc:  # noqa: BLE001
        print(
            "  [stage-marks] could not emit signpost %r (%s: %s) -- the capture will carry no stage "
            "boundary, so every stack shares one math-fidelity peak." % (name, type(exc).__name__, str(exc)[:140]),
            file=sys.stderr,
            flush=True,
        )


def mark_stages(adapter, device) -> int:
    """Run each declared stage once, eagerly, between marks. Returns how many stages were marked.

    Eager by necessity and by policy: the profiler attributes per-op time from eager dispatch, and
    synchronising inside a trace capture is fatal ("Event Synchronization is not supported during
    trace capture"). Nothing here opens a capture.

    Zero is a real answer -- a pipeline that declares no stages, or whose steps will not run one at a
    time, simply gets no split, and every consumer keeps the whole-profile figure it already had.
    """
    try:
        import ttnn
    except Exception:  # noqa: BLE001
        return 0
    stages = list(getattr(adapter, "stages", None) or [])
    if not stages:
        return 0
    n = 0
    for st in stages:
        name = str(getattr(st, "name", "") or "").strip()
        step = getattr(st, "step", None)
        if not name or not callable(step):
            continue
        signpost("stage:%s" % name)
        try:
            step()
            ttnn.synchronize_device(device)
            n += 1
        except Exception as exc:  # noqa: BLE001
            # One stage that will not run alone must not cost the others their boundary, nor the run.
            print(
                "  [stage-marks] stage %r could not be run on its own (%s: %s); no boundary for it"
                % (name, type(exc).__name__, str(exc)[:140]),
                file=sys.stderr,
                flush=True,
            )
        finally:
            signpost("stage:%s:end" % name)
    return n
