# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two closure checks the serving stage owes, in one place.

**Fallback audit.** The serving decode path is only as good as the paths it does
*not* take. This reads the server log for every marker that would mean the
measured path silently degraded — an eager decode instead of a traced one, host
sampling instead of on-device sampling, a prefill-trace capture failure, a
KV-cache rebind forcing a recapture — and reports which appeared.

It does **not** read `sampling_tests.log`; an earlier version of this docstring
said it did, which the stage review caught. What a server log can and cannot
settle is recorded explicitly in `conditions_evidenced_elsewhere`, and every
`degraded` marker is checked to be a string some source file can actually emit
(`verify_markers_live`) — two of them once were not, which made `clean: true`
vacuous for the two conditions that matter most.

**Process cleanup.** vLLM's launcher forks `EngineCore` workers that can outlive
`SIGTERM` to the parent and keep holding the chips, which is the failure
`$vllm-integration` calls out by name. This lists any surviving vLLM/EngineCore
process and any process holding `/dev/tenstorrent/*`.

Usage::

    python doc/vllm_integration/bench/audit_serving.py \
        --server-log <model_dir>/readiness_vllm/server.log \
        --out doc/vllm_integration/serving_audit.json
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess

#: (marker, severity, what it would mean). Severity ``degraded`` means the
#: measured path was not the intended one; ``event`` means it happened and is
#: expected to be classified in the work log.
MARKERS: tuple[tuple[str, str, str], ...] = (
    ("DEGRADED PATH untraced_eager_decode", "degraded", "eager decode instead of the captured trace"),
    (
        "DEGRADED PATH host_argmax_fallback",
        "degraded",
        "generate(host_sampling=True) standalone compatibility mode; the serving adapter never "
        "calls it, so on a serving log this marker is expected to be absent",
    ),
    (
        "DEGRADED PATH serving_full_logits_readback",
        "degraded",
        "the SERVING host-sampling route -- gathered full-vocab logits read to host instead of "
        "an on-device sampled token. This is the marker that matters on a serving log",
    ),
    ("capturing the prefill trace", "event", "prefill trace capture attempted"),
    (
        "DEGRADED PATH prefill_traces_released_for_sampling_capture",
        "degraded",
        "a sampling mode with no warmed trace arrived while prefill traces were resident, so they "
        "were released and serving prefill fell back to eager; the sampling-mode warmup list is "
        "incomplete for this workload",
    ),
    # Contiguous substring on purpose: the emitted sentence is concatenated from two
    # source literals, so the full phrase is never findable in the source and the
    # liveness check below would call a live marker dead.
    ("tracing is disabled for this generator", "degraded", "prefill trace capture failed"),
    ("the KV cache was rebound to different buffers", "event", "cache rebind forced a trace recapture"),
    ("failed to release the decode trace", "degraded", "a trace was orphaned"),
    ("failed to release the sampling trace", "degraded", "a sampling trace was orphaned"),
    ("deferring the free", "event", "a free was deferred behind an unreleased sampling trace"),
    ("teardown() leaves", "degraded", "traces or tensors outlived teardown"),
    ("builds a REDUCED", "degraded", "a reduced bring-up target was served"),
    ("Disabling async scheduling", "degraded", "the plugin refused the async-decode capability"),
    ("Prefix caching is not supported", "event", "prefix caching disabled, as declared"),
    ("Chunked prefill is not yet supported", "event", "chunked prefill disabled by the platform"),
    ("EngineCore encountered a fatal error", "degraded", "engine crash"),
    ("EngineDeadError", "degraded", "engine death"),
    ("adopted an externally owned paged KV cache", "event", "vLLM-owned cache bound"),
    ("captured decode trace", "event", "decode trace captured"),
    ("captured sampling trace", "event", "sampling trace captured"),
)

#: Files that may emit a marker. A ``degraded`` marker that no source file can emit
#: makes this audit worse than useless: it reports "clean" for a condition it cannot
#: observe. The stage review caught exactly that -- two markers were grepped for that
#: existed only in docstrings -- so marker liveness is now checked rather than assumed.
MARKER_SOURCES: tuple[str, ...] = (
    "models/autoports/meta_models_muse_glimmer_30b/tt/generator.py",
    "models/autoports/meta_models_muse_glimmer_30b/tt/model.py",
    "models/autoports/meta_models_muse_glimmer_30b/tt/generator_vllm.py",
)
#: Markers emitted by vLLM or the TT plugin rather than by this port; their liveness
#: is not checked against this repo.
EXTERNAL_MARKERS: frozenset[str] = frozenset(
    {
        "Disabling async scheduling",
        "Prefix caching is not supported",
        "Chunked prefill is not yet supported",
        "EngineCore encountered a fatal error",
        "EngineDeadError",
    }
)

#: Conditions a server log cannot settle on its own, and what does settle them.
#: Recorded in the report so a future reader is not left believing the log proved
#: more than it did.
EVIDENCED_ELSEWHERE: dict[str, str] = {
    "every measured decode step was a trace replay": (
        "probe_full_fixed.json -> multi_request.decode_counters.trace_replays equals the step "
        "count with synchronizations 0; and vllm_benchmark.json ITL p50 23.015 ms / p99 23.641 ms "
        "over 128 steps, which leaves no room for a step that gathered the full 202048-wide vocab"
    ),
    "on-device sampling was active for the benchmarked run": (
        "server.log records sample_on_device_mode=all and 'captured sampling trace'; the adapter "
        "refuses to start if the plugin requests device sampling it does not support"
    ),
}


def _emittable_strings(path: pathlib.Path) -> list[str]:
    """String literals a module could actually pass to a logger, docstrings excluded.

    Searching raw source text is not good enough: it is satisfied by a marker that
    appears only in a docstring, which is precisely the bug this check exists to
    prevent -- two markers once did exactly that. Docstrings are the ``Constant``
    expression-statements at the head of a module/class/function, so the AST can
    identify and drop them instead of guessing.
    """
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (OSError, SyntaxError):
        return []
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


def verify_markers_live() -> dict:
    """Check every ``degraded`` marker is a string some source file can actually emit."""
    # bench -> vllm_integration -> doc -> <model> -> autoports -> models -> repo root
    repo = pathlib.Path(__file__).resolve().parents[6]
    literals = []
    for rel in MARKER_SOURCES:
        literals.extend(_emittable_strings(repo / rel))
    joined = "\n".join(literals)
    dead = []
    for marker, severity, _meaning in MARKERS:
        if severity != "degraded" or marker in EXTERNAL_MARKERS:
            continue
        if marker not in joined:
            dead.append(marker)
    return {
        "sources_checked": list(MARKER_SOURCES),
        "method": "non-docstring string literals only, collected via ast",
        "external_markers_not_checked": sorted(EXTERNAL_MARKERS),
        "dead_markers": dead,
        "all_degraded_markers_live": not dead,
    }


def marker_provenance(log_paths) -> dict:
    """Say plainly whether each log predates the markers it is scanned for.

    A marker introduced after a log was written cannot appear in it, so its absence
    proves nothing about that run. Reporting both mtimes keeps a reader from
    over-reading a green result.
    """
    repo = pathlib.Path(__file__).resolve().parents[6]
    newest_source = 0.0
    for rel in MARKER_SOURCES:
        f = repo / rel
        if f.is_file():
            newest_source = max(newest_source, f.stat().st_mtime)
    rows = []
    for path in log_paths:
        if path.is_file():
            mtime = path.stat().st_mtime
            rows.append({"log": str(path), "log_mtime": mtime, "predates_current_markers": mtime < newest_source})
    return {
        "newest_marker_source_mtime": newest_source,
        "logs": rows,
        "caveat": (
            "for any log with predates_current_markers=true, the absence of a marker added after "
            "it was written is not evidence about that run; see conditions_evidenced_elsewhere"
        ),
    }


#: End of the engine's warmup window in the server log.  vLLM prints this once, after
#: ``warmup_model()`` returns and before the API server accepts a request.
WARMUP_END_MARKER = "init engine (profile, create kv cache, warmup model) took"

#: Degraded markers the *warmup* window is expected to emit, with why.  The plugin's
#: warmup is two-phase by design (``model_runner.py::warmup_model``): phase 1 compiles
#: every op variant with ``enable_trace=False`` so that phase 2 can capture traces with
#: nothing left to compile.  Phase 1 therefore drives one genuinely untraced decode
#: step, and this port announces untraced decode -- correctly.  Counting that against
#: the *serving* path would make the audit unable to ever report clean, which is worse
#: than useless: it would train a reader to ignore the field that matters.
EXPECTED_WARMUP_DEGRADED: dict[str, str] = {
    "DEGRADED PATH untraced_eager_decode": (
        "the TT plugin's phase-1 warmup compiles the decode path with enable_trace=False before "
        "phase 2 captures the trace, so exactly one untraced decode step is expected here and is "
        "not a serving-path degradation"
    ),
}


#: Degraded markers a *post-benchmark check* window may legitimately emit, with why.
#: These are only tolerated after ``--benchmark-window-end-bytes``: inside the measured
#: window they still fail the audit, because there they would mean the reported numbers
#: were produced by a degraded path.
EXPECTED_CHECKS_DEGRADED: dict[str, str] = {
    "DEGRADED PATH serving_full_logits_readback": (
        "the plugin sampling suite asks for logprobs, allowed_token_ids and structured outputs, and "
        "vLLM routes those to host sampling on a mesh whose device count is not 8 or 32. The route is "
        "the declared compatibility mode, exercised on purpose by those tests. Inside the benchmark "
        "window the same marker still fails the audit, because there it would mean the reported decode "
        "numbers came from a gathered full-vocab readback"
    ),
    "DEGRADED PATH prefill_traces_released_for_sampling_capture": (
        "the sampling suite sends explicit per-request seeds, which make the shared sampler bypass "
        "its trace and allocate; the generator gives up its prefill traces first, by design, rather "
        "than let a buffer be allocated under a live trace. Expected once per sweep that runs seeded "
        "sampling, and it means every check after it ran on the eager prefill path"
    ),
}


def scan(
    path: pathlib.Path,
    benchmark_window_end_bytes: int | None = None,
    shutdown_window_start_bytes: int | None = None,
) -> dict:
    """Marker census for one log, split into warmup / benchmark / checks windows.

    The split is the whole point.  A marker's meaning depends on *when* it was emitted:
    an untraced decode step during the plugin's compile phase is the design, the same
    marker during serving is a defect; a prefill-trace release during a seeded sampling
    test is the design, the same release during the benchmark would mean the reported
    TTFT was not measured on the path being reported.  Before the split this audit
    reported the union and could only ever say ``clean: false``.

    ``benchmark_window_end_bytes`` is the byte offset the sweep recorded after its last
    benchmark invocation.  Without it there is no benchmark window and everything after
    warmup is judged as one serving window, which is the conservative reading.
    """
    if not path.is_file():
        return {"path": str(path), "present": False}
    # Read as BYTES and decode per window, because the offsets the sweep records are
    # ``wc -c`` byte counts and these logs are not pure ASCII -- vLLM's banner uses
    # box-drawing characters, and a corrupted arm's log carries U+FFFD. Slicing a
    # decoded str by a byte offset silently shifts every window boundary.
    raw = path.read_bytes()

    def decode(chunk: bytes) -> str:
        return chunk.decode("utf-8", errors="replace")

    text = decode(raw)
    # Teardown is not serving.  SIGTERM to the launcher makes the API server log an
    # EngineDeadError traceback about the EngineCore it was talking to, which is the
    # shutdown working, not a serving-path degradation.  Everything from the offset the
    # sweep recorded when it sent the signal is reported separately and not judged.
    shutdown_text = ""
    if shutdown_window_start_bytes is not None and 0 < shutdown_window_start_bytes < len(raw):
        shutdown_text = decode(raw[shutdown_window_start_bytes:])
        raw = raw[:shutdown_window_start_bytes]
        text = decode(raw)
    cut_bytes = raw.find(WARMUP_END_MARKER.encode())
    cut = cut_bytes
    if cut < 0:
        # No boundary: treat the whole log as serving, which is the conservative
        # reading -- it can only over-report degradation, never hide it.
        warmup_text, serving_text, split = "", text, False
    else:
        warmup_text, serving_text, split = decode(raw[:cut]), decode(raw[cut:]), True

    bench_split = benchmark_window_end_bytes is not None and cut >= 0 and benchmark_window_end_bytes > cut
    if bench_split:
        benchmark_text = decode(raw[cut:benchmark_window_end_bytes])
        checks_text = decode(raw[benchmark_window_end_bytes:])
    else:
        benchmark_text, checks_text = serving_text, ""

    def census(chunk: str) -> dict:
        found = {}
        for marker, severity, meaning in MARKERS:
            count = chunk.count(marker)
            if count:
                found[marker] = {"count": count, "severity": severity, "meaning": meaning}
        return found

    def degraded(chunk: dict) -> list:
        return sorted(m for m, v in chunk.items() if v["severity"] == "degraded")

    found = census(text)
    warmup = census(warmup_text)
    serving = census(serving_text)
    benchmark = census(benchmark_text)
    checks = census(checks_text)
    shutdown = census(shutdown_text)
    warmup_degraded = degraded(warmup)
    checks_degraded = degraded(checks)
    return {
        "path": str(path),
        "present": True,
        "bytes": len(text),
        "warmup_window_split": split,
        "warmup_window_split_marker": WARMUP_END_MARKER,
        "benchmark_window_split": bench_split,
        "benchmark_window_end_bytes": benchmark_window_end_bytes,
        "markers": found,
        "markers_warmup_window": warmup,
        "markers_serving_window": serving,
        "markers_benchmark_window": benchmark,
        "markers_checks_window": checks,
        "markers_shutdown_window": shutdown,
        "shutdown_window_start_bytes": shutdown_window_start_bytes,
        "degraded_shutdown_window": sorted(m for m, v in shutdown.items() if v["severity"] == "degraded"),
        # Kept as the union for backward compatibility with the vLLM-integration
        # stage's committed report, which had no split and no degraded markers at all.
        "degraded": sorted(m for m, v in found.items() if v["severity"] == "degraded"),
        "degraded_serving_window": degraded(serving),
        "degraded_benchmark_window": degraded(benchmark),
        "degraded_checks_window": checks_degraded,
        "degraded_checks_window_expected": {
            m: EXPECTED_CHECKS_DEGRADED[m] for m in checks_degraded if m in EXPECTED_CHECKS_DEGRADED
        },
        "degraded_checks_window_unexpected": [m for m in checks_degraded if m not in EXPECTED_CHECKS_DEGRADED],
        "degraded_warmup_window": warmup_degraded,
        "degraded_warmup_window_expected": {
            m: EXPECTED_WARMUP_DEGRADED[m] for m in warmup_degraded if m in EXPECTED_WARMUP_DEGRADED
        },
        "degraded_warmup_window_unexpected": [m for m in warmup_degraded if m not in EXPECTED_WARMUP_DEGRADED],
    }


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}: {exc}>"


def processes() -> dict:
    ps = _run(["ps", "-eo", "pid,etime,args", "--no-headers"])
    pattern = re.compile(r"(EngineCore|vllm\.entrypoints|run_vllm_server|vllm bench|VLLM::)")
    survivors = [line.strip() for line in ps.splitlines() if pattern.search(line) and "audit_serving" not in line]
    holders = _run(["bash", "-lc", "ls -l /proc/*/fd 2>/dev/null | grep -l tenstorrent || true"])
    fuser = _run(["bash", "-lc", 'for d in /dev/tenstorrent/*; do echo -n "$d "; fuser $d 2>/dev/null; echo; done'])
    return {
        "surviving_vllm_processes": survivors,
        "device_holders_raw": fuser.strip().splitlines(),
        "proc_fd_scan": holders.strip()[:2000],
        "clean": not survivors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--shutdown-window-start-bytes",
        type=int,
        default=None,
        help=(
            "byte offset in the server log at which the sweep sent SIGTERM. Markers after it are "
            "teardown noise -- vLLM logs an EngineDeadError when the EngineCore it was talking to "
            "goes away -- and are reported without being judged."
        ),
    )
    parser.add_argument(
        "--benchmark-window-end-bytes",
        type=int,
        default=None,
        help=(
            "byte offset in the server log at which the last benchmark invocation finished. "
            "Splits the serving window into the measured benchmark window and the later check "
            "window, so a by-design degradation taken by a check stage does not read as one "
            "taken while the reported numbers were being produced."
        ),
    )
    args = parser.parse_args()

    marker_health = verify_markers_live()
    report = {
        "logs": [
            scan(path, args.benchmark_window_end_bytes, args.shutdown_window_start_bytes) for path in args.server_log
        ],
        "processes": processes(),
        "marker_health": marker_health,
        "marker_provenance": marker_provenance(list(args.server_log)),
        "conditions_evidenced_elsewhere": EVIDENCED_ELSEWHERE,
    }
    degraded = sorted({marker for log in report["logs"] for marker in log.get("degraded", [])})
    serving_degraded = sorted({m for log in report["logs"] for m in log.get("degraded_serving_window", [])})
    bench_degraded = sorted({m for log in report["logs"] for m in log.get("degraded_benchmark_window", [])})
    checks_unexpected = sorted({m for log in report["logs"] for m in log.get("degraded_checks_window_unexpected", [])})
    warmup_unexpected = sorted({m for log in report["logs"] for m in log.get("degraded_warmup_window_unexpected", [])})
    split = any(log.get("benchmark_window_split") for log in report["logs"])
    report["degraded_markers"] = degraded
    report["degraded_markers_serving_window"] = serving_degraded
    report["degraded_markers_benchmark_window"] = bench_degraded
    report["degraded_markers_checks_window_unexpected"] = checks_unexpected
    report["degraded_markers_checks_window_expected"] = {
        m: why for log in report["logs"] for m, why in log.get("degraded_checks_window_expected", {}).items()
    }
    report["degraded_markers_shutdown_window"] = sorted(
        {m for log in report["logs"] for m in log.get("degraded_shutdown_window", [])}
    )
    report["degraded_markers_warmup_window_unexpected"] = warmup_unexpected
    report["degraded_markers_warmup_window_expected"] = {
        m: why for log in report["logs"] for m, why in log.get("degraded_warmup_window_expected", {}).items()
    }
    # A dead degraded marker means this audit cannot see the condition it claims to
    # check, so it must not report clean.
    #
    # ``clean`` is judged on the window whose result is being reported -- the benchmark
    # window when the sweep recorded one, the whole serving window when it did not --
    # plus any warmup- or checks-window degradation that is not on a by-design list. A
    # marker the plugin's compile phase, or a seeded sampling test, is documented to
    # produce is reported, explained, and does not fail the audit.
    report["clean"] = (
        not (bench_degraded if split else serving_degraded)
        and not warmup_unexpected
        and not checks_unexpected
        and report["processes"]["clean"]
        and marker_health["all_degraded_markers_live"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
