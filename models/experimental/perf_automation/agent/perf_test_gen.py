# SPDX-License-Identifier: Apache-2.0
"""Generate a bounded, profiler-safe perf test for a pipeline FROM its demo, when none exists.

emit-e2e emits demos (demo/demo_<task>.py) but no perf test; some tt-metal demos lack one too.
Discovery calls generate_perf_test() for any pipeline whose perf_test resolved to None: an LLM lifts
the build+run from the demo and wraps it in a fixed profiler-safe skeleton (bounded work + periodic
ttnn.ReadDeviceProfiler drain, NO PCC asserts). The perf test is an OUTPUT we manufacture from the
demo (the reliable input), not something we require to pre-exist. Idempotent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Structural reference handed to the LLM (the seamless bounded-perf pattern, generic-ized).
_SKELETON_REF = """
import os
import time
import pytest
import ttnn
# from <model>.tt.<generator> import <Generator>   # lift the import from the demo

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    # Reserve the trace + 2-CQ budget at device-open, ONCE, for baseline and every candidate: the
    # second queue and the trace region exist before any candidate runs, so trace+2CQ is the fixed
    # measurement mode (never a per-candidate downgrade for lack of a queue). A device/config that
    # genuinely can't open 2 CQs still degrades gracefully in measure_adapter; override with TT_PERF_NUM_CQ.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))

@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_<task>_perf(device_params, device):
    # 1) build the pipeline EXACTLY as demo/demo_<task>.py does
    # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
    #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
    #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
    #    transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
    #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
    counter = [0]
    _orig = []
    def _draining(fn):
        def inner(*a, **k):
            r = fn(*a, **k); counter[0] += 1
            if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                try: ttnn.ReadDeviceProfiler(device)   # 'device' = mesh_device on multi-chip
                except Exception: pass
            return r
        return inner
    _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
    for _mod in [_m for _m in _mods if _m is not None]:
        for _n in dir(_mod):
            _op = getattr(_mod, _n, None)
            if type(_op).__name__ == "FastOperation":     # every dispatched ttnn op, by type
                _orig.append((_mod, _n, _op)); setattr(_mod, _n, _draining(_op))
    _fw0 = time.monotonic()
    try:
        out = ...  # run the pipeline BOUNDED (cap decode via PERF_MAX_NEW_TOKENS, or one forward)
        try: ttnn.ReadDeviceProfiler(device)
        except Exception: pass
    finally:
        for _mod, _n, _f in _orig: setattr(_mod, _n, _f)
    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None   # perf only — NO PCC

    if _PERF_TRACE:
        try:
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

            def _build_for_perf(dev):
                from <model>.tt.pipeline import build_pipeline   # lift the real import
                return build_pipeline(dev)                        # + the same build args the demo uses
            _prompt_ids = ...
            # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
            # traced (+2CQ where the stage stages its inputs). Falls back to the single decode
            # contract for pipelines that expose only decode_step.
            _adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1)
            measure_adapter(_adapter, device, mode="auto")
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
"""


_SELF_TRACED_SKELETON_REF = """
import os
import time
import pytest
import ttnn
# from <model>.tt.<module> import build_pipeline, <self_traced_fn>   # lift both from the demo

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))

@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_<task>_perf(device_params, device):
    # SELF-RECORDING PIPELINE: the model's own <self_traced_fn> already records its trace (trace+2CQ)
    # internally. Do NOT re-record its trace here (no adapter, no manual capture calls) — a nested capture
    # fatals + hangs. Build EXACTLY as the demo does, WARM UP once, then TIME steady-state calls of that
    # SAME function; that native latency IS the trace+2CQ number. Print the markers verbatim.
    pipe = ...        # build EXACTLY as demo/demo_<task>.py does, on `device`
    _inp = ...        # a SMALL representative input (lift from the demo)
    <self_traced_fn>(pipe, _inp)                       # warm up (its own internal capture runs here)
    _iters = int(os.environ.get("TT_PERF_REPLAY_ITERS", "16"))
    _t0 = time.monotonic()
    for _ in range(_iters):
        out = <self_traced_fn>(pipe, _inp)             # its own trace+2CQ path, timed
    ttnn.synchronize_device(device)
    _ms = (time.monotonic() - _t0) * 1000.0 / _iters
    assert out is not None                              # perf only — NO PCC
    print("FORWARD_WALL_MS=%.4f" % _ms)
    print("TRACE_PER_TOKEN_MS=%.4f" % _ms)
    print("TRACE_REPLAY_PATH=trace+2cq native batch=1")
"""


def _inline_inprocess_sources(src_text: str, root: Path) -> str:
    """When a source orchestrates the forward by launching pytest node-ids in SUBPROCESSES (a
    union gate), tracy cannot see those device ops — profiling yields an empty CSV. Pull the REAL
    in-process forwards: find the referenced `<path>.py::test_*` node-ids, resolve each file
    (under the model root), and return their bodies so the LLM can lift the build+forward directly.
    Model-agnostic: any model whose gate/demo shells out to per-module nodes gets them inlined."""
    if not any(tok in src_text for tok in ("subprocess", "Popen", "os.system", "os.popen")):
        return ""
    blocks, seen = [], set()
    for m in re.finditer(r"([\w./\-]+\.py)::\w+", src_text):
        rel = m.group(1)
        if rel in seen:
            continue
        seen.add(rel)
        cand = None
        candidates = [root / rel, root / Path(rel).name]
        marker = f"{root.name}/"
        if marker in rel:
            candidates.append(root / rel.split(marker, 1)[1])
        for c in candidates:
            if c.is_file():
                cand = c
                break
        if cand is None:
            hits = list(root.rglob(Path(rel).name))
            cand = hits[0] if hits else None
        if cand and cand.is_file():
            blocks.append(f"<inprocess_source path='{rel}'>\n{cand.read_text(errors='ignore')}\n</inprocess_source>")
    return "\n\n".join(blocks)


def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    # The model sometimes wraps the file in PROSE + a ```python fence (e.g. "here is the file: ```python
    # ..."). Extract the first fenced code block when present, so the prose preamble never reaches disk.
    m = re.search(r"```(?:python|py)?[^\n]*\n(.*?)```", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def _claude(prompt: str, timeout_s: int = 600) -> str | None:
    """One headless `claude` CLI call returning the generated file text (None on failure). Uses the
    CLI's native auth (real key or login), not the perf_automation LiteLLM proxy."""
    env = dict(os.environ)
    # native auth: drop proxy vars; restore the native key stashed by config (else fall back to login)
    for _k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(_k, None)
    _native = env.pop("PERF_NATIVE_ANTHROPIC_API_KEY", "")
    if _native:
        env["ANTHROPIC_API_KEY"] = _native
    else:
        env.pop("ANTHROPIC_API_KEY", None)
    try:
        from .agent_bin import resolve_claude_bin

        r = subprocess.run(
            [resolve_claude_bin(), "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except Exception:  # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


_DEVICE_UNAVAILABLE = (
    "no devices",
    "no available devices",
    "failed to open device",
    "cannot open device",
    "no such device",
    "no module named 'ttnn'",
    "no module named ttnn",
)


def _parse_trace_path(text: str) -> str | None:
    m = re.search(r"TRACE_REPLAY_PATH=(\S+)", text or "")
    return m.group(1) if m else None


_TRACE_REGION_NEED_RE = re.compile(r"Creating trace buffers of size (\d+)B.*?only (\d+)B is allocated")
_TRACE_REGION_GROW_ROUNDS = 6


def _needed_trace_region(text: str):
    # a multi-stage trace grows the region cumulatively, so each failing capture reports a bigger need;
    # take the MAX over-allocation seen in this run so the next attempt jumps past it.
    over = [int(n) for n, alloc in _TRACE_REGION_NEED_RE.findall(text or "") if int(n) > int(alloc)]
    return int(max(over) * 1.25) if over else None


_DEVICE_DISRUPTION_RE = re.compile(
    r"AICLK failed to settle|clamped by max-arbiter|Sysmem mapped at unexpected NOC|"
    r"pin_or_map_sysmem_to_device|failed to open device|could not open device|GetPCIeDeviceID|"
    r"GetNumPCIeDevices",
    re.IGNORECASE,
)


_TRACE_RAN_MARKERS = ("[perf_test_gen] WEDGE", "FORWARD_WALL_MS=", "TRACE_PER_TOKEN_MS=")


def _is_device_disruption(rc, out: str) -> bool:
    """A board-level disruption that happens BEFORE the test body runs (device open / PCIe enumeration /
    clock / stale sysmem) — the test file is fine, the DEVICE is wedged, so reset + cooldown + retry the
    SAME test is the right response. Kept narrow so (a) an ordinary assertion / import error still flows
    to the correction loop, and (b) a TRACE HANG is NOT retried here: if the test already RAN (it emitted
    a wall-time marker, or the tracy hang marker), the trace capture wedged mid-test — that already got
    one reset and must return to the caller as a WEDGE (-> eager fallback), NOT loop reset+retry (which
    just re-hangs). The post-hang reset re-init prints 'AICLK failed to settle', which would otherwise
    look like a fresh board disruption — this guard prevents that misclassification."""
    if not out:
        return False
    if any(m in out for m in _TRACE_RAN_MARKERS):
        return False
    if _DEVICE_DISRUPTION_RE.search(out):
        return True
    if "unordered_map::at" in out and re.search(
        r"GetPCIeDeviceID|open_device|CreateDevice|MeshDevice|conftest\.py|device_params", out
    ):
        return True
    return False


def _run_perf_node(node_abs: str, extra_env: dict, timeout_s: int = 2400):
    def _once(ev):
        env = dict(os.environ)
        env.setdefault("TT_PERF_TRACE", "1")
        env.setdefault("TT_PERF_MAX_NEW_TOKENS", "4")
        env.pop("TT_METAL_DEVICE_PROFILER", None)
        env.update(ev)
        cmd = [sys.executable, "-m", "pytest", "-o", "timeout=0", "-s", node_abs]
        from . import probes as _pr

        log = Path(tempfile.mkdtemp(prefix="perf_node_")) / "run.log"
        stall = int(os.environ.get("PERF_MCP_VALIDATE_STALL_SEC", "300") or "300")
        try:
            rc = _pr._execute([str(c) for c in cmd], Path.cwd(), env, timeout_s, log, stall_timeout_s=stall)
            return rc, (log.read_text(errors="ignore") if log.exists() else "")
        except _pr.TracyHangError as exc:
            out = log.read_text(errors="ignore") if log.exists() else ""
            ok = _pr._device_reset()
            return 124, out + "\n[perf_test_gen] WEDGE: %s; killed process group + tt-smi -r (reset_ok=%s)\n" % (
                exc,
                ok,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"run failed: {str(exc)[-300:]}"
        finally:
            shutil.rmtree(log.parent, ignore_errors=True)

    ev = dict(extra_env)
    max_disrupt = int(os.environ.get("PERF_MCP_DEVICE_DISRUPT_RETRIES", "3") or "3")
    disruptions = 0
    while True:
        rc, out = _once(ev)
        # model/hardware-agnostic trace region: never a fixed guess. The device reports the EXACT bytes a
        # capture needs when the region is too small; grow to that (doubling to cover a multi-stage trace's
        # cumulative growth) and re-run, until every stage's capture fits or the grow budget is exhausted.
        for _ in range(_TRACE_REGION_GROW_ROUNDS):
            need = _needed_trace_region(out)
            if need is None:
                break
            cur = int(ev.get("TT_PERF_TRACE_REGION") or os.environ.get("TT_PERF_TRACE_REGION") or 0)
            target = max(need, cur * 2)
            if target <= cur:
                break
            ev["TT_PERF_TRACE_REGION"] = str(target)
            rc, out = _once(ev)
        if disruptions < max_disrupt and _is_device_disruption(rc, out):
            from . import probes as _pr

            ok = _pr._device_reset()
            try:
                _pr._await_cool()
            except Exception:  # noqa: BLE001
                pass
            disruptions += 1
            print(
                "      · device disruption detected (board wedge, not a test bug) — self-heal "
                f"tt-smi reset (ok={ok}) + cooldown, retry {disruptions}/{max_disrupt}",
                file=sys.stderr,
                flush=True,
            )
            continue
        return rc, out


def _write_trace_caps(out_path: Path, caps: dict) -> None:
    try:
        (out_path.parent / (out_path.name + ".trace_caps.json")).write_text(json.dumps(caps, indent=2))
    except Exception:  # noqa: BLE001
        pass


# The correction loop keeps regenerating until the test is trace+2cq-acceptable (or a legitimate eager
# terminal). It has NO fixed attempt budget — only a STALL guard: if the LLM fails to make forward
# progress this many consecutive times, give up rather than spin forever on a pipeline it can't fix.
_STALL_LIMIT = 6

_TRACE_WEDGE_LIMIT = int(os.environ.get("PERF_MCP_TRACE_WEDGE_LIMIT", "3") or "3")

_COMPONENT_WEDGE_REASON = (
    "your trace capture HUNG the device (execute_trace blocked) — the timed forward contains HOST work "
    "that a trace cannot record. The tool already builds the module + its inputs ONCE before the capture "
    "and gives you a resident-buffer skeleton; your job is to make sure NOTHING between "
    "ttnn.begin_trace_capture and ttnn.end_trace_capture touches the host: NO ttnn.from_torch / "
    "ttnn.to_torch / .item() / .cpu() / torch tensor construction / python shape or control-flow decisions "
    "inside _forward(). Move any mask / rope / position / scale construction ABOVE the capture (build once, "
    "keep the ttnn tensor resident) and have _forward() call ONLY the module on the already-resident inputs. "
    "If one op inside the captured region is the host bit, move it OUT of the region (before/after the "
    "capture) or replace it with a device-only equivalent. The goal is a REAL trace (TRACE_PER_TOKEN_MS): "
    "do NOT fall back to eager timing and do NOT print TRACE_NOT_TRACE_CAPABLE — keep fixing until it traces."
)


# Lines from a failing pytest run that are NOISE, not the real error: nanobind/UMD teardown chatter,
# raw backtraces, python-internal frames. Feeding these back to the LLM as "the error" wastes the
# correction (verified: a failing run's last 1800 chars were "leaked function" x10, zero error lines).
_ERR_NOISE = re.compile(
    r"leaked|nanobind|Backtrace|^\s*\[0x[0-9a-f]+\]|\.so[)\s]|_PyEval|Py_|"
    r"ttnn/deprecated|site-packages/torch|^\s*File \"|^\s*self\.|^\s*return ",
    re.IGNORECASE,
)


def _extract_error(out: str) -> str:
    """Surface the REAL failure from a pytest run so the correction feedback is actionable. Anchor on
    pytest's own error lines ('E   ...', 'ERROR collecting', assertion/exception summaries) and DROP the
    teardown/backtrace noise — the LLM fixes the file, not the harness's leaked-function chatter."""
    if not out:
        return ""
    lines = out.splitlines()
    picked = []
    for ln in lines:
        s = ln.strip()
        if not s or _ERR_NOISE.search(ln):
            continue
        if (
            ln.lstrip().startswith("E ")
            or ln.lstrip().startswith("E\t")
            or "ERROR collecting" in ln
            or re.match(r"^\s*[A-Za-z_][\w.]*Error\b", ln)
            or "Traceback (most recent call last)" in ln
            or "FATAL" in ln
            or "WEDGE" in ln
            or "assert" in s
            or "cannot import" in s
            or "has no attribute" in s
        ):
            picked.append(s)
    tail = "\n".join(picked[-25:]) if picked else ""
    if not tail:
        # nothing anchored — fall back to the last few non-noise lines so feedback is never empty.
        clean = [l.strip() for l in lines if l.strip() and not _ERR_NOISE.search(l)]
        tail = "\n".join(clean[-12:])
    return tail[-2000:]


def _is_eager_terminal(out: str) -> bool:
    """A pipeline that GENUINELY cannot be trace-replayed (repeat-prefill / no decode_step) emits the
    authoritative TRACE_NOT_TRACE_CAPABLE=1 marker (from measure_adapter). That is the ONE legitimate
    reason a test stays on FORWARD_WALL_MS instead of trace+2cq — accept it, don't keep correcting."""
    return "TRACE_NOT_TRACE_CAPABLE=1" in (out or "")


def _pipeline_api_hint(root: Path, demo_src: str) -> str:
    """Feed the model's REAL pipeline API (the build_pipeline factory signature + PIPELINE_STAGES) into
    the prompt so the LLM fills `_build_for_perf` with the actual call, not a guess. Model-agnostic:
    discovered by scanning the model's own tt/ pipeline modules; empty when there's nothing to surface."""
    try:
        sigs, stages_seen = [], False
        for py in sorted(root.rglob("*.py")):
            if "/tests/" in py.as_posix() or py.name.startswith("test_"):
                continue
            try:
                txt = py.read_text(errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            for m in re.finditer(r"^def build_pipeline\s*\([^)]*\)", txt, re.MULTILINE):
                rel = py.relative_to(root).as_posix()
                sigs.append(f"# {rel}\n{m.group(0)}")
            if "PIPELINE_STAGES" in txt and not stages_seen:
                sm = re.search(r"PIPELINE_STAGES\s*=\s*[\[(][^\])]*[\])]", txt)
                if sm:
                    sigs.append(sm.group(0))
                    stages_seen = True
            if len(sigs) >= 4:
                break
        if not sigs:
            return ""
        return (
            "\n\nMODEL PIPELINE API (use the ACTUAL factory below in `_build_for_perf` — do not invent a "
            "signature; import it from the module shown and pass `dev` + the same build args):\n"
            + "\n".join(sigs[:4])
            + "\n"
        )
    except Exception:  # noqa: BLE001
        return ""


def _correction_feedback(reason: str, failure: str, prev_draft: str | None) -> str:
    """Build the correction addendum appended to the prompt for the NEXT attempt: the reason the last
    draft was rejected, the REAL extracted error, and the LLM's own previous draft so it EDITS the
    failing file instead of rewriting blind. This is what makes the loop converge rather than churn."""
    parts = [
        "\n\n=== CORRECTION — your previous draft was REJECTED. Fix it; do not start over blindly. ===",
        f"REASON: {reason}",
    ]
    err = _extract_error(failure)
    if err:
        parts.append(f"REAL ERROR (fix THIS, ignore any leaked-function/backtrace teardown noise):\n{err}")
    if prev_draft:
        parts.append(
            "YOUR PREVIOUS DRAFT (edit it to fix the error above; keep the parts that worked):\n"
            f"```python\n{prev_draft[-6000:]}\n```"
        )
    parts.append("Return ONLY the corrected complete python file content — no prose, no markdown fences.")
    return "\n".join(parts)


def validate_generated_perf_test(out_path: Path, task: str, component: bool = False) -> tuple[str, str]:
    """Execute the freshly-generated perf test and JUDGE it, model- and hardware-agnostically:
      skip      device/ttnn unavailable at generation time -> soft-accept (never a false rejection)
      ok_2cq    the 2-CQ probe genuinely engaged (TRACE_REPLAY_PATH=trace+2cq) -> ship it
      ok_marker the pipeline GENUINELY cannot trace (TRACE_NOT_TRACE_CAPABLE=1) -> the one legit eager
                terminal, ship it on FORWARD_WALL_MS rather than loop forever chasing a trace it can't do
      invalid   ran but produced no full-pipeline marker, OR is trace-capable yet only degraded to 1cq
                -> NOT shipped; the caller keeps correcting until it reaches trace+2cq.
    The 2-CQ run must run TWICE across the optimize loop (baseline + final bookend) WITHOUT degrading, so
    a test that can't hold trace+2cq here is rejected NOW rather than silently downgraded later. Records
    what it saw in the trace_caps sidecar either way. Second return value is the failure detail fed back."""
    node_abs = f"{out_path}::test_{task}_perf"
    vt = int(os.environ.get("PERF_MCP_VALIDATE_TIMEOUT", "900") or "900")
    if component:
        rc1, out1 = _run_perf_node(node_abs, {"TT_PERF_TRACE": "1", "TT_PERF_NUM_CQ": "1"}, timeout_s=vt)
        if rc1 is None:
            return "skip", out1
        low = out1.lower()
        if any(s in low for s in _DEVICE_UNAVAILABLE):
            return "skip", "device/ttnn unavailable during generation-time validation"
        has_eager = "FORWARD_WALL_MS=" in out1
        traced = ("TRACE_PER_TOKEN_MS=" in out1) and bool(_parse_trace_path(out1))
        if rc1 == 0 and traced:
            _write_trace_caps(
                out_path,
                {
                    "trace_1cq": True,
                    "trace_1cq_path": _parse_trace_path(out1),
                    "trace_2cq": False,
                    "trace_2cq_path": None,
                    "eager_terminal": False,
                },
            )
            return "ok_1cq", ""
        if rc1 == 124 or "WEDGE" in out1:
            return "invalid", "WEDGE: " + (_extract_error(out1) or "device hung capturing the module's forward")
        return "invalid", (
            _extract_error(out1)
            or "module perf test produced no TRACE_PER_TOKEN_MS (a real device trace is required; eager "
            "timing is not accepted)"
        )
    rc1, out1 = _run_perf_node(node_abs, {"TT_PERF_NUM_CQ": "1"}, timeout_s=vt)
    if rc1 is None:
        return "skip", out1
    low = out1.lower()
    if any(s in low for s in _DEVICE_UNAVAILABLE):
        return "skip", "device/ttnn unavailable during generation-time validation"
    has_marker = ("TRACE_PER_TOKEN_MS=" in out1) or ("FORWARD_WALL_MS=" in out1)
    if rc1 != 0 or not has_marker:
        return "invalid", (
            _extract_error(out1)
            or "perf test did not run the full pipeline (no TRACE_PER_TOKEN_MS / FORWARD_WALL_MS marker)"
        )
    if os.environ.get("TT_PERF_TRACE") == "0" and "FORWARD_WALL_MS=" in out1:
        _write_trace_caps(
            out_path,
            {
                "trace_1cq": False,
                "trace_1cq_path": None,
                "trace_2cq": False,
                "trace_2cq_path": None,
                "eager_terminal": True,
            },
        )
        return "ok_marker", ""
    eager = _is_eager_terminal(out1)
    caps = {
        "trace_1cq": "TRACE_PER_TOKEN_MS=" in out1,
        "trace_1cq_path": _parse_trace_path(out1),
        "trace_2cq": False,
        "trace_2cq_path": None,
        "eager_terminal": eager,
    }
    if eager:
        _write_trace_caps(out_path, caps)
        return "ok_marker", ""
    rc2, out2 = _run_perf_node(node_abs, {"TT_PERF_NUM_CQ": "2"}, timeout_s=vt)
    path2 = _parse_trace_path(out2) if rc2 == 0 and "TRACE_PER_TOKEN_MS=" in out2 else None
    caps["trace_2cq_path"] = path2
    caps["trace_2cq"] = path2 == "trace+2cq"
    if _is_eager_terminal(out2):
        caps["eager_terminal"] = True
        _write_trace_caps(out_path, caps)
        return "ok_marker", ""
    _write_trace_caps(out_path, caps)
    if caps["trace_2cq"]:
        return "ok_2cq", ""
    if caps["trace_1cq"]:
        if os.environ.get("TT_PERF_MODULE_LEVEL", "") not in ("", "0", "false", "False"):
            return "ok_1cq", ""
        return "invalid", "trace-capable but degraded to 1cq (never held trace+2cq)"
    return "invalid", (
        f"pipeline could not trace at all (path={path2 or _parse_trace_path(out1)}); neither 1cq nor "
        "2cq trace engaged. " + (_extract_error(out2) or "")
    )


def _self_traced_prompt(out_rel: str, task: str, src_label: str, demo_src: str, fns: list) -> str:
    """Dedicated prompt for a SELF-RECORDING pipeline: no measure_adapter instructions anywhere (they'd
    mandate a second, freezing capture). Just: build like the demo, TIME the model's own self-recording
    function, print the markers. Its native path already runs trace+2CQ."""
    _fns = ", ".join("`%s`" % f for f in fns)
    return (
        f"Write a pytest PERFORMANCE test file `{out_rel}` for the '{task}' pipeline of this TTNN model.\n"
        f"CRITICAL — SELF-RECORDING PIPELINE: this pipeline's function(s) {_fns} ALREADY capture their own "
        f"trace (trace+2CQ) INTERNALLY (they call ttnn.begin_trace_capture). You must NOT record a SECOND "
        f"time — a nested capture fatals and hangs the device. So do NOT import or call measure_adapter, "
        f"PipelineStageAdapter, or ttnn.begin_trace_capture ANYWHERE in this test.\n"
        f"Instead, MEASURE the native path by TIMING it: build the pipeline EXACTLY as the demo does, warm "
        f"up once, then time steady-state calls of the model's own function; that latency IS the trace+2CQ "
        f"number.\n"
        f"<demo path='{src_label}'>\n{demo_src}\n</demo>\n\n"
        "Requirements:\n"
        f"- a pytest function named `test_{task}_perf`.\n"
        "- DEVICE OPEN: open the device the SAME way the demo/source does (lift its open, or use the "
        "device_params fixture) with trace_region_size + num_command_queues=2 when TT_PERF_TRACE is set, so "
        "the model's own capture + 2-CQ replay have the budget. Pass that device to the build + the function.\n"
        "- Run the device work IN-PROCESS (never subprocess/os.system/python -m pytest).\n"
        "- Cap the profiled work SMALL on whatever axis drives this model's dispatch count (tokens for an "
        "LLM, phonemes/audio-frames for TTS, timesteps for diffusion, frames for video). When size comes "
        "from the RAW INPUT (e.g. a phoneme string sets the audio-frame count), TRIM THE RAW INPUT ITSELF — "
        "a SHORT phoneme string / few timesteps — do NOT copy the demo's full-length input.\n"
        "- NO PCC / correctness asserts. Print, verbatim, `FORWARD_WALL_MS=<ms>`, `TRACE_PER_TOKEN_MS=<ms>` "
        "(the per-call latency), and `TRACE_REPLAY_PATH=trace+2cq`.\n"
        "- Do NOT use measure_adapter / PipelineStageAdapter / begin_trace_capture — the model self-records.\n\n"
        f"Use this skeleton (adapt build + the function name to the demo):\n{_SELF_TRACED_SKELETON_REF}\n\n"
        "Do NOT use any tools and do NOT write the file yourself — the caller writes it. Respond with ONLY "
        "the complete python file content — no prose, no markdown fences."
    )


def _self_tracing_fns(root: Path) -> set:
    """MODEL-AGNOSTIC: the model's OWN callables that ALREADY capture a trace themselves — their body
    calls ttnn.begin_trace_capture. Instrumenting one under the tool's own trace/measure would nest two
    captures on the device -> TT_FATAL + teardown hang. Derived by scanning the model's source (no
    per-model names); empty for models that don't self-trace. Lets the generator emit a time-it-directly
    test for a self-recording pipeline instead of a re-recording one that freezes.

    Covers BOTH shapes the demo can call: a top-level function (`run_tts_fast(...)`) AND a class method
    (`pipe.generate(...)`). An indent-tracked scope stack attributes a begin_trace_capture to the OUTERMOST
    enclosing callable — the public entry the demo invokes — so nested private helpers roll up to it."""
    fns = set()
    try:
        for py in sorted(root.rglob("*.py")):
            p = py.as_posix()
            if "/tests/" in p or py.name.startswith("test_"):
                continue
            try:
                txt = py.read_text(errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            if "begin_trace_capture" not in txt:
                continue
            stack = []
            for raw in txt.splitlines():
                stripped = raw.strip()
                if not stripped:
                    continue
                indent = len(raw) - len(raw.lstrip())
                while stack and stack[-1][0] >= indent:
                    stack.pop()
                m = re.match(r"(class|def)\s+([A-Za-z_]\w*)", stripped)
                if m:
                    stack.append((indent, m.group(1), m.group(2)))
                elif "begin_trace_capture" in raw:
                    for _ind, kind, name in stack:
                        if kind == "def":
                            fns.add(name)
                            break
    except Exception:  # noqa: BLE001
        pass
    return fns


def _invoked_as_pipeline_op(fn: str, demo_src: str) -> bool:
    """True only when the demo calls `fn` as a PIPELINE OPERATION — attribute-accessed (`P.fn(...)`) or
    called WITH arguments (`fn(pipe, ...)`). This excludes a bare launcher like `main()`, which every demo
    ends with: a self-recording `main` must NOT make a task whose actual pipeline function (e.g. run_tts)
    does NOT self-record get the time-it-directly treatment — that would time the eager path and mislabel
    it trace+2CQ."""
    esc = re.escape(fn)
    return bool(re.search(r"\.%s\s*\(" % esc, demo_src) or re.search(r"\b%s\s*\(\s*[^)\s]" % esc, demo_src))


_SKELETON_COMPONENT = """
import os
import time
import pytest
import ttnn

PERF_ITERS = int(os.environ.get("TT_PERF_ITERS", "5"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": _TRACE_REGION}], indirect=True)
def test_<task>_perf(device_params, device):
    ...  # build the one module + its input(s), lifted VERBATIM from the source PCC test

    def _forward():
        ...  # return the source's ttnn forward call for the built module

    counter = [0]
    _orig = []

    def _draining(fn):
        def inner(*a, **k):
            r = fn(*a, **k)
            counter[0] += 1
            if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            return r

        return inner

    for _mod in [ttnn] + [getattr(ttnn, _s, None) for _s in ("transformer", "experimental")]:
        if _mod is None:
            continue
        for _n in dir(_mod):
            _op = getattr(_mod, _n, None)
            if type(_op).__name__ == "FastOperation":
                _orig.append((_mod, _n, _op))
                setattr(_mod, _n, _draining(_op))
    _forward()
    ttnn.synchronize_device(device)
    _t0 = time.monotonic()
    try:
        for _ in range(PERF_ITERS):
            out = _forward()
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
    ttnn.synchronize_device(device)
    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _t0) * 1000.0 / PERF_ITERS))
    assert out is not None

    if os.environ.get("TT_PERF_TRACE") != "0":
        try:
            _forward()
            ttnn.synchronize_device(device)
            _tid = ttnn.begin_trace_capture(device, cq_id=0)
            _forward()
            ttnn.end_trace_capture(device, _tid, cq_id=0)
            ttnn.execute_trace(device, _tid, cq_id=0, blocking=True)
            _tt0 = time.monotonic()
            for _ in range(PERF_ITERS):
                ttnn.execute_trace(device, _tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, _tid)
            print("TRACE_PER_TOKEN_MS=%.4f" % ((time.monotonic() - _tt0) * 1000.0 / PERF_ITERS))
            print("TRACE_REPLAY_PATH=trace 1cq module-forward")
        except Exception as _te:  # noqa: BLE001
            print("TRACE_NOT_TRACE_CAPABLE=1", flush=True)
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
"""


def _component_prompt(
    out_rel: str, src_label: str, demo_src: str, task: str, cache_instr: str = "", agentic: bool = False
) -> str:
    """LLM prompt for a single-component perf test — the GENERAL path (covers any module/model type).
    Mirrors the demo path's proven 'lift the build+run from a complete source' recipe, but the source is
    the component's per-component PCC test and the target is one module timed in isolation. Carries the
    golden-cache fast path (so candidates never reload the full model) and the resident-buffer trace rules
    (so the isolated-module trace capture does not hang the device). agentic=True drops the one-shot
    'respond with only the file content' tail, because the agentic builder writes + runs the file itself."""
    tail = (
        ""
        if agentic
        else (
            "Do NOT use any tools and do NOT write the file yourself — respond with ONLY the complete python "
            "file content as your message text, no prose, no markdown fences."
        )
    )
    return (
        f"Write a pytest PERFORMANCE test file `{out_rel}` that times ONE component of this TTNN model in "
        f"ISOLATION. The source below is that component's per-component CORRECTNESS (PCC) test — it ALREADY "
        f"builds the module correctly and runs its forward.\n"
        + cache_instr
        + f"LIFT ITS SETUP: reproduce its ttnn module build and input-tensor construction so the module has "
        f"the SAME weights and inputs; on a golden-cache MISS reproduce the source's reference-model load / "
        f"submodule resolution VERBATIM (do NOT substitute AutoModel/from_pretrained on the miss path; if it "
        f"loads a model-local `_reference_loader`, load it the SAME way it does). Build the module + ALL its "
        f"inputs ONCE before timing. DROP ONLY the final comp_pcc / assert_with_pcc correctness comparison, "
        f"then time the module's forward per the skeleton. This is NOT a pipeline: do NOT use build_pipeline, "
        f"run_tts, run_main, generate, PipelineStageAdapter, measure_adapter, or PIPELINE_STAGES.\n"
        f"<pcc_test path='{src_label}'>\n{demo_src}\n</pcc_test>\n\n"
        f"Fill this structural skeleton — keep the drain, the eager timing, AND the trace-replay block "
        f"VERBATIM; your ONLY edits are replacing the two `...` placeholders (the build/input in step 1 and "
        f"the `_forward()` body in step 2) with code lifted from the source. TRACE-CAPTURE RULE (critical — "
        f"a violation HANGS the device): build the module and ALL its inputs/constants (masks, rope, "
        f"positions, scales) in step 1, ONCE, before the capture, and keep them resident on device; "
        f"`_forward()` must call ONLY the module on those already-built ttnn tensors — NO ttnn.from_torch / "
        f"ttnn.to_torch / .item() / .cpu() / torch tensor construction / python shape or control-flow "
        f"decisions inside `_forward()`. If one op inside the captured region is the host bit, move it OUT "
        f"(before/after the capture) or replace it with a device-only equivalent. The result MUST be a real "
        f"device trace (TRACE_PER_TOKEN_MS) — do NOT fall back to eager timing:\n"
        f"{_SKELETON_COMPONENT}\n\n" + tail
    )


def generate_perf_test(
    model_root: str | Path,
    task: str,
    demo_rel: str | None,
    *,
    runner=None,
    force: bool = False,
    source_abs: str | Path | None = None,
    source_kind: str = "demo",
    validate: bool | None = None,
) -> str | None:
    """Write tests/e2e/test_<task>_perf.py by lifting build+run from a source — the WHOLE pipeline
    forward (prefill + a capped decode loop when the source has one). Returns the node id
    'tests/e2e/test_<task>_perf.py::test_<task>_perf' on success, else None. `runner` (prompt->str)
    overrides the default claude call (for tests).

    Source: source_kind='demo' (default) lifts from `demo_rel` (under model_root); source_kind='pcc'
    lifts from `source_abs` (the e2e PCC test, which may live outside model_root) and DROPS the
    reference build + correctness asserts, keeping only the TTNN forward.

    force=False keeps the old idempotent behavior (return an existing file unchanged). force=True
    REGENERATES from scratch every time and overwrites — used by discovery so a stale/partial
    (e.g. prefill-only) perf test is NEVER reused; the pipeline's perf workload is recomputed each run."""
    root = Path(model_root)
    _component = source_kind == "pcc" and os.environ.get("TT_PERF_MODULE_LEVEL", "") not in ("", "0", "false", "False")
    out_rel = f"tests/pcc/test_{task}_perf.py" if _component else f"tests/e2e/test_{task}_perf.py"
    out_path = root / out_rel
    node = f"{out_rel}::test_{task}_perf"
    if out_path.exists() and not force:
        return node
    if source_kind == "pcc":
        src_file = Path(source_abs) if source_abs else None
        if src_file is None or not src_file.is_file():
            return None
        src_label = str(src_file)
    else:
        src_file = root / demo_rel if demo_rel else None
        if src_file is None or not src_file.is_file():
            return None
        src_label = demo_rel
    demo_src = src_file.read_text(errors="ignore")
    if _component:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    _selfrec_set = _self_tracing_fns(root)
    self_traced = sorted(f for f in _selfrec_set if _invoked_as_pipeline_op(f, demo_src))
    if source_kind == "pcc":
        _cache_instr = ""
        if _component:
            _cache_instr = (
                f"MODULE-LEVEL FAST PATH (MANDATORY): this is a single-component perf test. Loading the "
                f"multi-GB full model on every candidate is the dominant cost this must avoid. The "
                f"correctness (PCC) run caches this component's torch submodule + inputs on disk; READ it and "
                f"build the ttnn port from that submodule, falling back to the full reference build ONLY on a "
                f"cache miss:\n"
                f"    from models.common.golden_cache import golden_cache_path, load_golden_cache\n"
                f"    _hit = load_golden_cache(golden_cache_path(__file__, {task!r}))\n"
                f"    if _hit is not None:\n"
                f"        torch_module, sample_kwargs = _hit[0], _hit[1]\n"
                f"    else:\n"
                f"        <build the reference EXACTLY as the pcc source does to get torch_module + "
                f"sample_kwargs (from_pretrained / _reference_loader / submodule resolve)>\n"
                f"Do NOT write the cache from this perf test (the PCC test owns it). Then build the ttnn port "
                f"from torch_module and time ONLY the on-device forward — no PCC/comp_pcc comparison. The "
                f"from_pretrained fallback must run at most once (the cache miss), never per candidate.\n"
            )
        prompt = (
            f"Write a pytest PERFORMANCE test file `{out_rel}` for the '{task}' pipeline of this TTNN model.\n"
            f"This source is a CORRECTNESS (PCC) test — build and run the TTNN model EXACTLY as it does, but "
            f"KEEP ONLY the on-device TTNN forward: DROP the reference/torch model construction and DROP every "
            f"PCC / comp_pcc / allclose / assert_with_pcc correctness comparison.\n"
            f"<pcc_test path='{src_label}'>\n{demo_src}\n</pcc_test>\n\n" + _cache_instr + "Requirements:\n"
        )
    else:
        prompt = (
            f"Write a pytest PERFORMANCE test file `{out_rel}` for the '{task}' pipeline of this TTNN model.\n"
            f"Build and run the pipeline EXACTLY as this demo does:\n<demo path='{src_label}'>\n{demo_src}\n</demo>\n\n"
            "Requirements:\n"
        )
    prompt += (
        "- CRITICAL — run the device forward IN-PROCESS inside the test. NEVER shell out: no "
        "subprocess, os.system, os.popen, Popen, or launching `pytest` / `python -m`. Tracy profiles "
        "ONLY the current process, so any TTNN op executed in a child process is INVISIBLE to the "
        "profiler and produces an EMPTY ops-perf CSV (the run aborts with TracyRunError). If the "
        "source orchestrates work by launching pytest node-ids in subprocesses, do NOT replicate "
        "that — inline / call those modules' build+forward directly so every device op runs here.\n"
        f"- a pytest function named `test_{task}_perf`.\n"
        "- DEVICE OPEN — MATCH THE SOURCE'S TOPOLOGY EXACTLY (this is critical for sharded models). If the "
        "source SELF-OPENS its device (calls open_pipeline_mesh / open_mesh_device / ttnn.open_mesh_device, "
        "or builds a MeshShape), your test MUST open + close the device the SAME way — lift that exact "
        "open call into the test body, close it in a finally — and pass that device object to "
        "build_pipeline / the forward. Do NOT substitute a pytest `device` / `device_params` fixture: a "
        "single `device` fixture silently DISABLES the pipeline's sharding (shard_active becomes False) and "
        "profiles the WRONG single-chip config for a model built to run tensor-parallel on a mesh. Use the "
        "pytest `device`/`device_params` fixture ONLY when the source itself uses that fixture (genuine "
        "single-device pipelines). When TT_PERF_TRACE is set and the source's open function accepts "
        "trace_region_size / num_command_queues, pass them through that open; otherwise open exactly as the "
        "source does (the trace block stays guarded and simply falls back).\n"
        "- BOUNDED + profiler-safe so tracy's 12000-marker buffer never overflows: cap the work (decode "
        "loop via env TT_PERF_MAX_NEW_TOKENS default 4, or a SINGLE forward if there's no loop), AND drain "
        "the profiler every TT_PERF_FLUSH_EVERY ops (default 32) + a final ttnn.ReadDeviceProfiler. DRAIN "
        "MUST BE MODEL-AGNOSTIC — wrap EVERY ttnn op by TYPE, not a curated list: iterate ttnn (and its op "
        "submodules ttnn.transformer / ttnn.experimental) and wrap every attribute whose "
        "type(obj).__name__ == 'FastOperation' with a counter that drains every TT_PERF_FLUSH_EVERY calls. "
        "A curated list (matmul/linear/conv only) UNDER-counts — sdpa/eltwise/transpose/reduction slip "
        "through, the buffer overflows on some device, ops get dropped, and device_ms becomes "
        "non-reproducible. Wrapping by type can never miss an op. Restore all originals in a finally. "
        "(Use the generic wrap loop from the skeleton below verbatim — do NOT hand-pick op names.)\n"
        "- CAP THE PROFILED WORK SMALL — on WHATEVER axis drives THIS model's op/dispatch count, not just "
        "sequence length. The heavy axis is model-specific: TOKENS for an LLM (decode length + prompt), "
        "PHONEMES / AUDIO FRAMES for a TTS or audio model, TIMESTEPS for diffusion, FRAMES for video, "
        "PIXELS / PATCHES for vision. Reduce THAT axis to a SMALL representative size for every forward. Do "
        "NOT reuse the model's production / maximum shapes (max_position_embeddings, max_seq, full-length "
        "audio, the full timestep schedule) even if the source/PCC test or demo does — those are correctness "
        "stress sizes. CRUCIAL: when the model's size comes from the RAW INPUT rather than a constant — a TTS "
        "phoneme string whose length sets the number of audio frames, a diffusion step count, a video frame "
        "count — TRIM THE RAW INPUT ITSELF (a SHORT phoneme string / few timesteps / few frames); do NOT copy "
        "the demo's full-length input, because 'run exactly as the demo' does NOT mean 'at the demo's full "
        "input length'. Under tracy EVERY device op is instrumented, so a full-length forward runs orders of "
        "magnitude slower and the host blocks in ttnn.synchronize_device for many minutes, stalling the run. "
        "Make the small size env-overridable with a SMALL default. A perf profile only needs a "
        "representative dispatch-dense pass, not the full-length run.\n"
        '- KEEP the skeleton\'s `os.environ.setdefault("TT_PERF_LAYERS", ...)` line VERBATIM near the top. '
        "It caps profiled depth for deep (many-layer) models so the device profiler's marker buffer does "
        "not overflow (worse on a multi-chip mesh, where markers scale x chips). It is set in-process so "
        "ONLY this perf run is capped; a pipeline that does not read TT_PERF_LAYERS simply ignores it. Do "
        "NOT hard-require it and do NOT gate on it — just carry it through.\n"
        "- NO PCC / correctness assertions (this is perf only) — just assert the pipeline produced output.\n"
        "- TIME THE FORWARD: keep the skeleton's time.monotonic() bracket around the bounded forward and "
        'the final print("FORWARD_WALL_MS=...") VERBATIM — the harness reads it as an independent '
        "end-to-end check on the profiler capture. Do not remove or rename it.\n"
        "- KEEP the skeleton's trace-replay block VERBATIM in structure: the `_PERF_TRACE`/`_DEV_PARAMS` "
        "device-param gate near the top AND the trailing `if _PERF_TRACE:` measure_adapter block. This is a "
        "MODEL-AGNOSTIC, GPU-comparable latency (TRACE_PER_TOKEN_MS + per-stage TRACE_STAGE_MS). Do NOT "
        "write a per-model adapter class — the tool ships the generic PipelineStageAdapter, which profiles "
        "WHATEVER emit-e2e emitted: every `PIPELINE_STAGES` entry is traced (+2CQ where the stage exposes "
        "`<stage>_write_inputs`), falling back to the single decode contract for decode-only pipelines. Your "
        "ONLY job in that block is to fill `_build_for_perf(dev)` so it RETURNS THE RESIDENT, STAGE-EXPOSING "
        "PIPELINE OBJECT — the one carrying PIPELINE_STAGES + the per-stage trace hooks (or a trace-capturable "
        "`decode_step(state)`). Call the model's module-level `build_pipeline(device, ...)` factory that "
        "emit-e2e emits (import it from the demo's tt/pipeline module, pass `dev` + the same build args). Do "
        "NOT return the demo's run_tts()/generate() RESULT or a closure that runs the pipeline — that object "
        "has no stage hooks, so the adapter raises and trace+2CQ silently falls back to FORWARD_WALL_MS. Set "
        "`_prompt_ids` to a SMALL prompt. Leave everything else in the block verbatim. The clean numbers are "
        "emitted automatically once `_build_for_perf` returns that object; a genuine repeat-prefill pipeline "
        "with no stage hooks and no decode_step legitimately falls back to FORWARD_WALL_MS, which is fine. "
        "Never delete the block, never let it fail the test.\n"
        "- TRACE BLOCK + SELF-OPEN: if (per the DEVICE OPEN rule) the test self-opens a mesh, pass the "
        "device the test actually opened to `measure_adapter(...)` (NOT a fixture `device`), and put "
        "`trace_region_size`/`num_command_queues` on that self-open call when TT_PERF_TRACE (drop the "
        "`_DEV_PARAMS`/`device_params` fixture entirely). Keep `_build_for_perf(dev)` building the pipeline "
        "on the passed-in `dev` so both the eager forward and the trace run the SAME sharded topology.\n"
        "- MESH SHAPE — honor the tool's topology: when you self-open a mesh, derive the (rows, cols) via "
        "`from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape` and "
        "`rows, cols = resolve_mesh_shape(default_rows=<source rows>, default_cols=<source cols>)`, then "
        "open `MeshShape(rows, cols)`. This lets --devices/--mesh reshape the run (single->1x1, N chips->the "
        "planned TP x DP); with the env unset it falls back to the source's own shape. Do NOT hardcode the shape.\n"
        "- Lift the imports + build args straight from the demo above.\n\n"
        f"Use this structural skeleton (adapt the build+run to the demo):\n{_SKELETON_REF}\n"
    )
    inproc_ctx = _inline_inprocess_sources(demo_src, root)
    if inproc_ctx:
        prompt += (
            "\n\nNOTE: the source above is a SUBPROCESS-UNION — it launches per-module pytest node-ids "
            "in child processes, whose device ops tracy CANNOT profile. Below are those modules' ACTUAL "
            f"in-process build+forward bodies. Lift the build and on-device TTNN forward from THESE and run "
            f"them directly in test_{task}_perf (one process, same modules covered), dropping every "
            "PCC/correctness assert. CRITICAL: use the SAME pytest device fixtures and the SAME "
            "@pytest.mark.parametrize decorator these modules use (e.g. `mesh_device`, `device_params`, "
            "`reset_seeds` with their MESH_DEVICE_PARAMETRIZE_* marker) — do NOT substitute a plain single "
            "`device` fixture; the lifted builds run on whatever device/mesh object these modules take. "
            "Reuse their imports/constants (config builders, MESH_DEVICE_PARAMETRIZE_*, helpers) verbatim:\n"
            f"{inproc_ctx}\n"
        )
    prompt += _pipeline_api_hint(root, demo_src)
    prompt += (
        "\n\nDo NOT use any tools and do NOT try to write the file yourself — the caller writes it. "
        "Respond with ONLY the complete python file content as your message text — no prose, no markdown fences."
    )
    if _component:
        prompt = _component_prompt(out_rel, src_label, demo_src, task, cache_instr=_cache_instr)
    if self_traced and not _component:
        prompt = _self_traced_prompt(out_rel, task, src_label, demo_src, self_traced)
    if (
        _component
        and runner is None
        and validate is not False
        and os.environ.get("TT_PERF_NO_AGENTIC_BUILDER", "") in ("", "0", "false", "False")
    ):
        try:
            from .perf_test_agent import build_component_perf_test

            _body = _component_prompt(out_rel, src_label, demo_src, task, cache_instr=_cache_instr, agentic=True)
            if build_component_perf_test(root, task, out_rel, _body):
                _verdict, _ = validate_generated_perf_test(out_path, task, component=True)
                if _verdict in ("ok_2cq", "ok_1cq", "skip"):
                    print(f"      auto-gen perf from pcc (agentic) -> {node}", file=sys.stderr, flush=True)
                    return node
            print(
                "      · agentic builder did not converge; falling back to one-shot generator",
                file=sys.stderr,
                flush=True,
            )
        except Exception as _exc:  # noqa: BLE001
            print(
                f"      · agentic builder unavailable ({str(_exc)[:100]}); using one-shot", file=sys.stderr, flush=True
            )
    # A generative demo's perf test must exercise the (capped) decode loop, not a prefill-only slice.
    demo_is_generative = any(
        k in demo_src.lower()
        for k in ("max_new_tokens", "generate(", ".generate", "next_token", "decode_step", "for _ in range")
    )
    gen = runner or _claude
    if validate is None:
        validate = runner is None
    feedback = ""
    prev_draft = None
    stall = 0
    trace_wedges = 0
    while stall < _STALL_LIMIT:
        content = _strip_fence(gen(prompt + feedback) or "")
        if "def test_" not in content or "ttnn" not in content:
            stall += 1
            feedback = _correction_feedback(
                "draft was not a complete python perf test (missing `def test_` or `ttnn`)", "", prev_draft
            )
            continue
        if re.search(
            r"import\s+subprocess|subprocess\.|\bPopen\s*\(|os\.system\s*\(|os\.popen\s*\(|"
            r"-m['\"]\s*,\s*['\"]pytest|python\s+-m\s+pytest",
            content,
        ):
            stall += 1
            feedback = _correction_feedback(
                "draft shelled out (subprocess/Popen/os.system/python -m pytest) — tracy can't profile "
                "child-process ops. Run the device forward IN-PROCESS.",
                "",
                content,
            )
            prev_draft = content
            continue
        _code = "\n".join(re.sub(r"#.*$", "", ln) for ln in content.splitlines())
        _times_selfrec = sorted(f for f in _selfrec_set if _invoked_as_pipeline_op(f, _code))
        _external_capture = "measure_adapter(" in _code or "begin_trace_capture(" in _code
        _claims_trace = "trace+2cq" in content.lower()
        if _times_selfrec and _external_capture:
            stall += 1
            feedback = _correction_feedback(
                "the timed function (%s) ALREADY records its own trace+2CQ internally — do NOT re-record it with "
                "measure_adapter or begin_trace_capture (a nested capture fatals + hangs the device). Just TIME "
                "it directly and print TRACE_PER_TOKEN_MS + TRACE_REPLAY_PATH=trace+2cq." % ", ".join(_times_selfrec),
                "",
                content,
            )
            prev_draft = content
            continue
        if _claims_trace and not _external_capture and not _times_selfrec:
            stall += 1
            feedback = _correction_feedback(
                "the test prints TRACE_REPLAY_PATH=trace+2cq but the timed function does NOT record a trace "
                "(it is not one of the model's self-recording functions) and you did not call measure_adapter — "
                "so this is TIMING THE EAGER PATH and mislabelling it. Wrap the timed forward in measure_adapter "
                "to actually capture + replay a trace+2CQ, or time a function that self-records.",
                "",
                content,
            )
            prev_draft = content
            continue
        if demo_is_generative and "TT_PERF_MAX_NEW_TOKENS" not in content:
            stall += 1
            feedback = _correction_feedback(
                "generative pipeline but the test omits the decode-loop cap (TT_PERF_MAX_NEW_TOKENS) — it "
                "would profile a prefill-only slice. Add the capped decode loop.",
                "",
                content,
            )
            prev_draft = content
            continue
        prev_draft = content
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        if not validate:
            return node
        verdict, failure = validate_generated_perf_test(out_path, task, component=_component)
        if verdict in ("ok_2cq", "ok_1cq", "ok_marker", "skip"):
            return node
        stall += 1
        if _component and "WEDGE" in failure:
            trace_wedges += 1
            print(
                f"      · perf-test regen {stall}/{_STALL_LIMIT} (trace wedge {trace_wedges}/{_TRACE_WEDGE_LIMIT}): "
                "device hung capturing this module's forward — reset + regenerating",
                file=sys.stderr,
                flush=True,
            )
            feedback = _correction_feedback(_COMPONENT_WEDGE_REASON, failure, prev_draft)
            continue
        if "WEDGE" in failure:
            _why = "device wedged on a non-capturable step — reset + regenerating"
        elif "degraded to 1cq" in failure:
            _why = "held only trace+1cq — regenerating to reach trace+2cq"
        else:
            _why = ((_extract_error(failure).splitlines() or [""])[0] or "did not run the full pipeline").strip()[:80]
        print(f"      · perf-test regen {stall}/{_STALL_LIMIT}: {_why}", file=sys.stderr, flush=True)
        reason = (
            "the test ran but never held trace+2cq (it degraded to 1cq); the 2-CQ input overlap must "
            "engage so the optimize bookend doesn't silently downgrade"
            if "degraded to 1cq" in failure
            else (
                "the module perf test produced no TRACE_PER_TOKEN_MS — implement the trace-replay block so "
                "it captures a REAL device trace (a trace is required; eager timing is not accepted)"
                if _component
                else "the test did not run the full pipeline / errored"
            )
        )
        feedback = _correction_feedback(reason, failure, prev_draft)
    return None
