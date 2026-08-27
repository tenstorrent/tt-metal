# SPDX-License-Identifier: Apache-2.0
"""Generate a bounded, profiler-safe perf test for a pipeline FROM its demo, when none exists.

emit-e2e emits demos (demo/demo_<task>.py) but no perf test; some tt-metal demos lack one too.
Discovery calls generate_perf_test() for any pipeline whose perf_test resolved to None: an LLM lifts
the build+run from the demo and wraps it in a fixed profiler-safe skeleton (bounded work + periodic
ttnn.ReadDeviceProfiler drain, NO PCC asserts). The perf test is an OUTPUT we manufacture from the
demo (the reliable input), not something we require to pre-exist. Idempotent.
"""

from __future__ import annotations

# The seam names live in ONE module -- see stage_seams. This file is also loaded BY FILE LOCATION
# (spec_from_file_location, no parent package) by the generator tests, where the relative form
# cannot resolve; those loaders put perf_automation on sys.path, so the flat spelling answers.
try:
    from . import stage_seams as _seams
except ImportError:  # loaded as a bare module, with perf_automation on sys.path
    from agent import stage_seams as _seams

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

# ONE VARIABLE FOR ONE THING. There were two: PERF_OSL_TOKENS, which the test PRINTED, and
# PERF_OSL_TOKENS, which the loop actually RAN -- with its own default of 4, from the generator's
# first commit. So a declared OSL of 128 was reported while 4 executed, and every profile sampled a
# thirty-second of the request it claimed to measure. Two names for one quantity is how a setting can
# be honoured and ignored at the same time. TT_PERF_OSL_TOKENS is GONE; a probe that wants a
# cheaper unit sets TT_PERF_OSL_TOKENS, and then the declared unit and the executed one still agree.
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS, and they default to a REALISTIC operating point rather
# than to whatever example prompt reads naturally. Left unspecified, a generated perf test used the
# shortest prompt that proves the pipeline runs -- on llama3_1_8b_p150 that was "The capital of
# France is", six tokens, and nothing recorded that the throughput number was a six-token one.
# Decode is weight-bandwidth bound so ISL barely moves tok/s/u (measured: 0.5% from ISL 6 to 128),
# but TTFT, prefill cost and any long-context claim all depend on it, so the default must be a
# figure someone would actually quote. 128 in / 128 out is the industry-standard short-context
# benchmark point. Both are env-overridable; the markers below record what actually ran, so a
# reader never has to guess the conditions.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# BATCH BELONGS TO THE MODEL, not to this generator. It was written into the generated test as a
# literal `batch=1`, so a pipeline emit-e2e built to serve 8 users was measured serving one, and its
# aggregate throughput under-reported by 8x. 0 means "ask the pipeline" -- it already knows, via
# max_batch_size / batch_size / batch, whichever it exposes -- and any positive value overrides, for
# sweeping batch without rebuilding the demo. Unlike ISL and OSL, which are the TOOL's choice of
# measurement condition, batch is a property of the artifact under test.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window so a deep model's marker stream (x mesh
# chips) does not overflow the profiler; the tool sends that number for tracy runs. The variable being
# ABSENT means ALL LAYERS -- the tool expresses "whole model" by REMOVING the cap, never by sending a
# sentinel, because "0" arrives as a truthy string and gets read as "build zero layers".
# Pass PERF_LAYERS straight to the builder: None is every builder's own all-layers value. Do NOT
# default it to a number here -- that would silently cap the full-depth gate.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# TOPOLOGY. --devices/--mesh are planned by the tool and exported as TT_PERF_MESH_ROWS/COLS;
# resolve_mesh_shape is how a run honours them. Give it the SOURCE's own shape as the default, so an
# unset env behaves exactly as the demo does. If the source uses a `mesh_device` FIXTURE, keep that
# fixture + its parametrize and feed this tuple in; if it SELF-OPENS, pass it to MeshShape().
# A copied MESH_DEVICE board table on its own cannot see --devices/--mesh.
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape
_MESH_SHAPE = resolve_mesh_shape(default_rows=<source rows>, default_cols=<source cols>)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
# FABRIC only when the resolved mesh spans MORE THAN ONE chip. A demo that targets multi-chip shapes
# hardcodes fabric_config=True; carried into a 1x1 run it still trains ethernet across every VISIBLE
# chip, and on a partly-idle board that times out ("Fabric Router Sync: Timeout ... ethernet handshake
# failed") before the model is built. Copy the source's other device_params verbatim; gate ONLY fabric.
if _MESH_SHAPE[0] * _MESH_SHAPE[1] > 1:
    _DEV_PARAMS["fabric_config"] = True   # only if the SOURCE sets it
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, for baseline and every candidate. The tool
    # measures trace+1cq end to end, so the device opens with a single command queue.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))
    _DEV_PARAMS["num_command_queues"] = 1

@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_<task>_perf(device_params, device):
    # 1) build the pipeline EXACTLY as demo/demo_<task>.py does
    # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
    #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
    #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
    #    transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
    #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
    def _eager_forward():
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
            out = ...  # run the pipeline BOUNDED (PERF_OSL_TOKENS decode steps, or one forward)
            try: ttnn.ReadDeviceProfiler(device)
            except Exception: pass
        finally:
            for _mod, _n, _f in _orig: setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None   # perf only — NO PCC

    def _traced_forward():
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

        def _build_for_perf(dev):
            from <model>.tt.pipeline import build_pipeline   # lift the real import
            return build_pipeline(dev)                        # + the same build args the demo uses
        # ISL: build the prompt to EXACTLY PERF_ISL_TOKENS tokens rather than writing an example
        # sentence, so the measurement condition is the tool's choice and not the generator's.
        _prompt_ids = prompt_ids_for_isl(<tokenizer>, PERF_ISL_TOKENS)
        print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
        # traced. Falls back to the single decode contract for pipelines that expose only decode_step.
        measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH), device)

    def _try_traced():
        try:
            _traced_forward(); return True
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
            return False

    # MEASUREMENT ORDER — two consumers, two different needs, and running both is not free.
    #   TRACY PROFILING RUN (TT_METAL_DEVICE_PROFILER=1, layer-capped): needs BOTH products. The
    #     op-wrapped eager forward IS the per-op capture; the trace pass supplies
    #     TRACE_PER_TOKEN_MS for throughput. Two different measurements, so both run.
    #   FULL-PIPELINE GATE (no tracy, TT_PERF_LAYERS=0, FULL depth): needs exactly ONE whole-model
    #     latency. Running both builds the model TWICE at full depth on one device -- the second
    #     build has no memory left for its KV cache and dies before any marker is printed.
    # So the gate runs TRACE FIRST and only falls back to the eager forward when trace genuinely
    # could not be measured. That is the designed contract: trace by default, eager as the fallback.
    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if _PERF_TRACE and not _PROFILING:
        if not _try_traced():
            print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working — timing eagerly", flush=True)
            _eager_forward()
    else:
        _eager_forward()
        if _PERF_TRACE:
            _try_traced()
"""


_SELF_TRACED_SKELETON_REF = """
import os
import time
import pytest
import ttnn
# from <model>.tt.<module> import build_pipeline, <self_traced_fn>   # lift both from the demo

# TOPOLOGY. --devices/--mesh are planned by the tool and exported as TT_PERF_MESH_ROWS/COLS;
# resolve_mesh_shape is how a run honours them. Give it the SOURCE's own shape as the default, so an
# unset env behaves exactly as the demo does. If the source uses a `mesh_device` FIXTURE, keep that
# fixture + its parametrize and feed this tuple in; if it SELF-OPENS, pass it to MeshShape().
# A copied MESH_DEVICE board table on its own cannot see --devices/--mesh.
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape
_MESH_SHAPE = resolve_mesh_shape(default_rows=<source rows>, default_cols=<source cols>)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
# FABRIC only when the resolved mesh spans MORE THAN ONE chip. A demo that targets multi-chip shapes
# hardcodes fabric_config=True; carried into a 1x1 run it still trains ethernet across every VISIBLE
# chip, and on a partly-idle board that times out ("Fabric Router Sync: Timeout ... ethernet handshake
# failed") before the model is built. Copy the source's other device_params verbatim; gate ONLY fabric.
if _MESH_SHAPE[0] * _MESH_SHAPE[1] > 1:
    _DEV_PARAMS["fabric_config"] = True   # only if the SOURCE sets it
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))
    _DEV_PARAMS["num_command_queues"] = 1

@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_<task>_perf(device_params, device):
    # SELF-RECORDING PIPELINE: the model's own <self_traced_fn> already records its trace (trace+1CQ)
    # internally. Do NOT re-record its trace here (no adapter, no manual capture calls) — a nested capture
    # fatals + hangs. Build EXACTLY as the demo does, WARM UP once, then TIME steady-state calls of that
    # SAME function; that native latency IS the trace+1CQ number. Print the markers verbatim.
    pipe = ...        # build EXACTLY as demo/demo_<task>.py does, on `device`
    _inp = ...        # a SMALL representative input (lift from the demo)
    <self_traced_fn>(pipe, _inp)                       # warm up (its own internal capture runs here)
    _iters = int(os.environ.get("TT_PERF_REPLAY_ITERS", "16"))
    _t0 = time.monotonic()
    for _ in range(_iters):
        out = <self_traced_fn>(pipe, _inp)             # its own trace+1cq path, timed
    ttnn.synchronize_device(device)
    _ms = (time.monotonic() - _t0) * 1000.0 / _iters
    assert out is not None                              # perf only — NO PCC
    print("FORWARD_WALL_MS=%.4f" % _ms)
    print("TRACE_PER_TOKEN_MS=%.4f" % _ms)
    # resolve_batch, not PERF_BATCH: 0 means "ask the pipeline", and printing 0 would tell the gate
    # this run served nobody. This path has the built pipeline in hand, so it can ask.
    print("TRACE_REPLAY_PATH=trace+1cq native batch=%d" % resolve_batch(pipe, PERF_BATCH))
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


_NON_TRACE_PATHS = {"eager", "none", "skipped", "off", "n/a", "na", "false", "0", "trace"}


def _parse_trace_path(text: str) -> str | None:
    """The declared replay path, or None when it declares NO trace.

    Callers do `bool(_parse_trace_path(out))` to decide `traced`, and bool("eager") is True -- so
    TRACE_REPLAY_PATH=eager/none/skipped was banked as a trace measurement. The old non-whitespace
    pattern also truncated "trace 1cq" (the spelling this module's own skeleton emits) to the bare
    word "trace", which is likewise not a real trace+Ncq path.
    """
    m = re.search(r"TRACE_REPLAY_PATH=([^\r\n]+)", text or "")
    if not m:
        return None
    val = m.group(1).strip().strip("'\"").lower()
    val = re.sub(r"\s+", "+", val)
    if not val or val.split("+")[0] in _NON_TRACE_PATHS and "cq" not in val:
        return None
    return val


_TRACE_REGION_NEED_RE = re.compile(r"Creating trace buffers of size (\d+)B.*?only (\d+)B is allocated")
_TRACE_REGION_GROW_ROUNDS = 6
# A single-device overflow reports the exact bytes (parsed above). A MeshDevice overflow is a BARE
# assertion (get_trace_buffers_size() <= trace_region_size, mesh_trace.cpp) with NO byte count, so the
# parse can't size it -- detect it by marker and grow by DOUBLING instead, mirroring emit-e2e's
# trace_gate.overflow_fix_loop (already mesh-safe). Matches the region assertion, not the separate
# write-during-capture fatal (fd_mesh_command_queue.cpp).
_MESH_TRACE_OVERFLOW_RE = re.compile(r"get_trace_buffers_size|mesh_trace\.cpp", re.I)


def _skeleton_default(var: str) -> int:
    """The default for `var` READ OUT OF THE EMITTED TEST, not written again here.

    The generated test is standalone -- it cannot import from its generator -- so its defaults have
    to be literals in `_SKELETON_REF`. That makes the skeleton the one place these numbers are
    stated, and anything else that needs them (the report prices prefill as 2 x params x ISL) must
    READ them from there rather than keep a second copy: two copies of a number is how the report
    came to price a run's arithmetic against a length that run never used.

    0 when the skeleton does not state one, which withholds rather than invents.
    """
    m = re.search(r'os\.environ\.get\(\s*"%s"\s*,\s*"(\d+)"\s*\)' % re.escape(var), _SKELETON_REF)
    return int(m.group(1)) if m else 0


DEFAULT_ISL_TOKENS = _skeleton_default("TT_PERF_ISL_TOKENS")
DEFAULT_OSL_TOKENS = _skeleton_default("TT_PERF_OSL_TOKENS")

# ROOM FOR EVERY STAGE'S TRACE, NOT JUST ONE. 23887872 fit a decode trace alone. A pipeline that
# also traces its prefill needs both to co-reside -- gemma-3 asks for 24420352 -- and the shortfall
# does not degrade, it TT_FATALs inside end_trace_capture. The failure reads as "tracing this stage
# does not work" and is really "the region was sized for one trace", which is how traced prefill came
# to be disabled in a model config rather than given room.
#
# 40 MiB, so a second stage trace is covered without anyone passing an env var. Still overridable;
# the perf test prints what it reserved.
_DEFAULT_TRACE_REGION_BYTES = 41943040


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


_CAPTURE_DRIVER = re.compile(
    r"\.(run_[a-z0-9_]+|generate|synthesize|infer|forward)\s*\(|\b(run_[a-z0-9_]+|generate)\s*\("
)
# The seam suffix is interpolated from stage_seams rather than spelled again; `decode_step` beside
# it is the LEGACY contract name, which is a different thing and not a per-stage seam.
_CAPTURE_HOSTFREE = re.compile(r"\b\w*(?:%s|decode_step)\s*\(" % re.escape(_seams.STEP))


def _handrolled_capture_violation(src: str) -> str | None:
    """A generated perf test MAY hand-roll ttnn.begin_trace_capture (the emit-e2e trace test does), but
    ONLY around a host-free step: a *_trace_step()/decode_step() hook whose inputs were staged BEFORE the
    capture. If it instead captures the demo's host-writing driver (run_*/generate/…), the trace records a
    host->device write — legal on a single chip but a TT_FATAL on a MeshDevice ('Writes are not supported
    during trace capture') that hangs the device. Detect it statically so BOTH the agentic and one-shot
    generators reject it BEFORE launching on the device. Returns guidance to feed back, else None."""
    code = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
    if "begin_trace_capture" not in code:
        return None
    for win in re.findall(r"begin_trace_capture\s*\(.*?end_trace_capture\s*\(", code, re.S):
        if _CAPTURE_DRIVER.search(win) and not _CAPTURE_HOSTFREE.search(win):
            return (
                "HANDROLLED_TRACE_CAPTURE: the test wraps the pipeline's host-writing driver (run_*/generate) "
                "in ttnn.begin_trace_capture. A trace captures ONLY pure on-device compute; a host->device "
                "write inside the capture TT_FATALs on a multi-chip mesh ('Writes are not supported during "
                "trace capture') and hangs. Either capture a host-free *_trace_step()/decode_step() hook "
                "(stage ALL inputs BEFORE begin_trace_capture, as the emit-e2e trace test does), or trace via "
                "measure_adapter(PipelineStageAdapter(...)); NEVER wrap run_*/generate in begin_trace_capture."
            )
    return None


def _run_perf_node(node_abs: str, extra_env: dict, timeout_s: int = 2400):
    _test_file = Path(str(node_abs).partition("::")[0])
    try:
        _viol = _handrolled_capture_violation(_test_file.read_text(errors="ignore")) if _test_file.is_file() else None
    except OSError:
        _viol = None
    if _viol:
        return 2, _viol

    def _once(ev):
        env = dict(os.environ)
        env.setdefault("TT_PERF_TRACE", "1")
        # VALIDATION, not measurement. This answers "does the generated test run at all", which four
        # tokens establish as well as a hundred and twenty-eight, and the cap keeps a generative loop
        # from running forever. It sets the SAME variable everything else uses -- there is no longer a
        # second one -- so the test still reports the unit it executed. An earlier revision raised it
        # to the declared OSL believing the PROFILE came through here; it does not (profile_model ->
        # measure_runs -> probes.make_run_profiled), so that only made validation 32x slower.
        env.setdefault("TT_PERF_OSL_TOKENS", "4")
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
            ok = _pr._device_reset(error_text=out)
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
            cur = (
                int(ev.get("TT_PERF_TRACE_REGION") or os.environ.get("TT_PERF_TRACE_REGION") or 0)
                or _DEFAULT_TRACE_REGION_BYTES
            )
            need = _needed_trace_region(out)
            if need is None:
                if _MESH_TRACE_OVERFLOW_RE.search(out or ""):
                    need = cur * 2
                else:
                    break
            target = max(need, cur * 2)
            if target <= cur:
                break
            ev["TT_PERF_TRACE_REGION"] = str(target)
            rc, out = _once(ev)
        if disruptions < max_disrupt and _is_device_disruption(rc, out):
            from . import probes as _pr

            ok = _pr._device_reset(error_text=out)
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


# The correction loop keeps regenerating until the test is trace+1cq-acceptable (or a legitimate eager
# terminal). It has NO fixed attempt budget — only a STALL guard: if the LLM fails to make forward
# progress this many consecutive times, give up rather than spin forever on a pipeline it can't fix.
# Env-overridable (PERF_MCP_STALL_LIMIT) for hard-to-trace models that need more regen attempts.
_STALL_LIMIT = int(os.environ.get("PERF_MCP_STALL_LIMIT", "3") or "3")

_TRACE_WEDGE_LIMIT = int(os.environ.get("PERF_MCP_TRACE_WEDGE_LIMIT", "10") or "10")

_COMPONENT_WEDGE_REASON = (
    "your trace capture HUNG the device (execute_trace blocked) — the timed forward contains HOST work "
    "that a trace cannot record. The tool already builds the module + its inputs ONCE before the capture "
    "and gives you a resident-buffer skeleton; your job is to make sure NOTHING between "
    "ttnn.begin_trace_capture and ttnn.end_trace_capture touches the host: NO ttnn.from_torch / "
    "ttnn.to_torch / .item() / .cpu() / torch tensor construction / python shape or control-flow decisions "
    "inside _forward(). Move any mask / rope / position / scale construction ABOVE the capture (build once, "
    "keep the ttnn tensor resident) and have _forward() call ONLY the module on the already-resident inputs. "
    "If the module's OWN forward has an irreducible host op that cannot be removed, print "
    "TRACE_NOT_TRACE_CAPABLE=1 and skip the trace block so it falls back to the eager FORWARD_WALL_MS number."
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
            or "TRACE_REPLAY_SKIPPED" in ln
            or "TRACE_NOT_TRACE_CAPABLE" in ln
            or "TRACE_REPLAY_PATH" in ln
            or "HANDROLLED_TRACE_CAPTURE" in ln
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
    reason a test stays on FORWARD_WALL_MS instead of trace+1cq — accept it, don't keep correcting."""
    return "TRACE_NOT_TRACE_CAPABLE=1" in (out or "")


def has_pipeline_stages(root: Path) -> bool:
    """Does this model expose emit-e2e's PIPELINE_STAGES trace hooks?

    Decides WHICH adapter the generated perf test should use. PipelineStageAdapter is built for
    models the bring-up tool assembled (it looks for `<stage>_trace_step` hooks); a hand-carved
    general model has none of that and only exposes the decode_step contract. The skeleton used to
    hardcode the stage adapter for BOTH, so a general model got an adapter written for a shape it
    does not have -- and its comment told the authoring agent this was about emit-e2e stages, which
    is misleading when there are none.
    """
    try:
        for py in list((root / "tt").rglob("*.py")) + list(root.rglob("pipeline*.py")):
            try:
                if "PIPELINE_STAGES" in py.read_text(errors="ignore"):
                    return True
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return False


def skeleton_for(root: Path) -> str:
    """The structural skeleton with the ADAPTER matched to what this model actually exposes."""
    if has_pipeline_stages(root):
        return _SKELETON_REF
    return (
        _SKELETON_REF.replace(
            "from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter",
            "from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter",
        )
        .replace(
            "            # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets\n"
            "            # traced. Falls back to the single decode contract for pipelines that expose only\n"
            "            # decode_step.\n",
            "            # This model exposes NO PIPELINE_STAGES (it was not assembled by emit-e2e), so profile\n"
            "            # the single decode contract it does expose: decode_prefill(ids) then decode_step(state).\n",
        )
        .replace(
            "_adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH)",
            "_adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH)",
        )
    )


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


# The workload knobs the TOOL owns. Anything else that shrinks the work is the generator inventing a
# measurement condition nobody asked for.
_KNOWN_WORKLOAD_VARS = ("TT_PERF_ISL_TOKENS", "TT_PERF_OSL_TOKENS", "TT_PERF_BATCH", "TT_PERF_LAYERS", "TT_PERF_TRACE",
                        "TT_PERF_HEADS", "TT_PERF_SEQ_LEN", "TT_PERF_TRACE_REGION", "TT_PERF_PREFILL_TRACE")  # fmt: skip
_ENV_INT_DEFAULT_RE = re.compile(r'os\.environ\.get\(\s*"(TT_PERF_[A-Z0-9_]+)"\s*,\s*"(\d+)"\s*\)')


def _stage_layers_var(stage) -> str:
    """layer_depth owns this spelling. Imported lazily, not at module scope: this module is loaded
    BY PATH in places (spec_from_file_location), where a relative import has no parent package --
    the same failure that stopped a refusal being reported from before_loop on 2026-08-17."""
    try:
        from .layer_depth import stage_layers_var

        return stage_layers_var(stage)
    except Exception:  # noqa: BLE001
        from layer_depth import stage_layers_var  # type: ignore[no-redef]

        return stage_layers_var(stage)


def invented_workload_vars(src: str, stages=()) -> list:
    """Env-driven counts the generated test invented, with a literal default that SHRINKS the work.

    BATCH BELONGS TO THE MODEL and the skeleton says so: the tool sends TT_PERF_BATCH=0 meaning "ask
    the pipeline", and build_pipeline answers with its own DECODE_BATCH. That channel works. What
    defeats it is a SECOND axis the tool knows nothing about.

    Measured on Voxtral-Mini-3B: the generated test defined

        PERF_AUDIO_STREAMS = int(os.environ.get("TT_PERF_AUDIO_STREAMS", "2"))

    which appears nowhere in this repo -- the agent writing the test decided the model's heaviest
    trimmable input was the clip count and picked 2. The pipeline was still BUILT for its declared
    batch of 8, and then handed 2 clips, so prefill measured a quarter of the real workload and was
    printed beside a full-batch roofline: 148.76 ms against a 7.48 ms theoretical. Nothing read the
    variable back, so nothing could notice.

    Trimming input for a tracy run is legitimate -- it is how a deep model stays under the profiler's
    marker limit. Inventing the axis, defaulting it below the model's own batch, and never reporting
    it is not. A per-stage depth variable is not flagged: those names come from PIPELINE_STAGES and
    the tool sets them.

    REPORTED, NOT REFUSED. Not every invented knob caps input -- the same test also defines
    TT_PERF_FLUSH_EVERY=32, a profiler flush cadence that changes no workload -- and no static rule
    separates the two reliably. Refusing on a heuristic would reject working tests; the defect worth
    fixing is that nothing ever surfaced these, so a 2-clip measurement sat beside an 8-clip ceiling
    for a week. Naming them where the run can see them is the fix.
    """
    allowed = (
        set(_KNOWN_WORKLOAD_VARS)
        | {_stage_layers_var(st) for st in (stages or ())}
        # The per-stage trace knob, derived from the same declaration as the depth knob beside it.
        | {"TT_PERF_%s_TRACE" % str(st).upper() for st in (stages or ())}
    )
    out = []
    for var, default in _ENV_INT_DEFAULT_RE.findall(src or ""):
        if var in allowed:
            continue
        if int(default) > 0:  # a positive literal is a CAP; 0 means "ask the pipeline"
            out.append((var, int(default)))
    return sorted(set(out))


def validate_generated_perf_test(out_path: Path, task: str, component: bool = False) -> tuple[str, str]:
    """Execute the freshly-generated perf test and JUDGE it, model- and hardware-agnostically:
      skip      device/ttnn unavailable at generation time -> soft-accept (never a false rejection)
      ok_1cq    the trace probe engaged at 1 CQ (a real TRACE_PER_TOKEN_MS / trace+1cq path) -> ship it
      ok_marker the pipeline GENUINELY cannot trace (TRACE_NOT_TRACE_CAPABLE=1) -> the one legit eager
                terminal, ship it on FORWARD_WALL_MS rather than loop forever chasing a trace it can't do
      invalid   ran but produced no full-pipeline marker, or could not trace at all -> NOT shipped; the
                caller keeps correcting.
    Measurement is trace+1cq end to end.
    Records what it saw in the trace_caps sidecar either way. Second return value is the failure detail."""
    node_abs = f"{out_path}::test_{task}_perf"
    vt = int(os.environ.get("PERF_MCP_VALIDATE_TIMEOUT", "900") or "900")
    if component:
        rc1, out1 = _run_perf_node(node_abs, {"TT_PERF_TRACE": "1"}, timeout_s=vt)
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
                    "eager_terminal": False,
                },
            )
            return "ok_1cq", ""
        if os.environ.get("TT_PERF_TRACE") == "0" and rc1 == 0 and has_eager:
            _write_trace_caps(
                out_path,
                {
                    "trace_1cq": False,
                    "trace_1cq_path": None,
                    "eager_terminal": True,
                },
            )
            return "ok_marker", ""
        if rc1 == 124 or "WEDGE" in out1:
            return "invalid", "WEDGE: " + (_extract_error(out1) or "device hung capturing the module's forward")
        return "invalid", (
            _extract_error(out1)
            or "module perf test produced no TRACE_PER_TOKEN_MS (trace required; eager only via TT_PERF_TRACE=0)"
        )
    rc1, out1 = _run_perf_node(node_abs, {}, timeout_s=vt)
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
                "eager_terminal": True,
            },
        )
        return "ok_marker", ""
    eager = _is_eager_terminal(out1)
    caps = {
        "trace_1cq": "TRACE_PER_TOKEN_MS=" in out1,
        "trace_1cq_path": _parse_trace_path(out1),
        "eager_terminal": eager,
    }
    if eager:
        _write_trace_caps(out_path, caps)
        return "ok_marker", ""
    # A trace-capable pipeline that traces at 1 CQ ships now.
    _write_trace_caps(out_path, caps)
    if caps["trace_1cq"]:
        return "ok_1cq", ""
    return "invalid", (
        f"pipeline could not trace at all (path={_parse_trace_path(out1)}); no trace engaged. "
        + (_extract_error(out1) or "")
    )


def _self_traced_prompt(
    out_rel: str, task: str, src_label: str, demo_src: str, fns: list, agentic: bool = False
) -> str:
    """Dedicated prompt for a SELF-RECORDING pipeline: no measure_adapter instructions anywhere (they'd
    mandate a second, freezing capture). Just: build like the demo, TIME the model's own self-recording
    function, print the markers. Its native path already runs its own trace (measured trace+1cq).
    agentic=True drops the one-shot 'respond with only the file content' tail, because the agentic builder
    writes + runs the file itself."""
    _fns = ", ".join("`%s`" % f for f in fns)
    tail = (
        ""
        if agentic
        else (
            "Do NOT use any tools and do NOT write the file yourself — the caller writes it. Respond with ONLY "
            "the complete python file content — no prose, no markdown fences."
        )
    )
    return (
        f"Write a pytest PERFORMANCE test file `{out_rel}` for the '{task}' pipeline of this TTNN model.\n"
        f"CRITICAL — SELF-RECORDING PIPELINE: this pipeline's function(s) {_fns} ALREADY capture their own "
        f"trace INTERNALLY (they call ttnn.begin_trace_capture). You must NOT record a SECOND "
        f"time — a nested capture fatals and hangs the device. So do NOT import or call measure_adapter, "
        f"PipelineStageAdapter, or ttnn.begin_trace_capture ANYWHERE in this test.\n"
        f"Instead, MEASURE the native path by TIMING it: build the pipeline EXACTLY as the demo does, warm "
        f"up once, then time steady-state calls of the model's own function; that latency IS the trace+1cq "
        f"number (the tool measures trace+1cq end to end).\n"
        f"<demo path='{src_label}'>\n{demo_src}\n</demo>\n\n"
        "Requirements:\n"
        f"- a pytest function named `test_{task}_perf`.\n"
        "- DEVICE OPEN: open the device the SAME way the demo/source does (lift its open, or use the "
        "device_params fixture) with trace_region_size + num_command_queues=1 "
        "when TT_PERF_TRACE is set, so the model's own capture + replay have the trace budget. Pass that "
        "device to the build + the function.\n"
        "- Run the device work IN-PROCESS (never subprocess/os.system/python -m pytest).\n"
        "- Cap the profiled work SMALL on whatever axis drives this model's dispatch count (tokens for an "
        "LLM, phonemes/audio-frames for TTS, timesteps for diffusion, frames for video). When size comes "
        "from the RAW INPUT (e.g. a phoneme string sets the audio-frame count), TRIM THE RAW INPUT ITSELF — "
        "a SHORT phoneme string / few timesteps — do NOT copy the demo's full-length input.\n"
        "- NO PCC / correctness asserts. Print, verbatim, `FORWARD_WALL_MS=<ms>`, `TRACE_PER_TOKEN_MS=<ms>` "
        "(the per-call latency), and `TRACE_REPLAY_PATH=trace+1cq`.\n"
        "- Do NOT use measure_adapter / PipelineStageAdapter / begin_trace_capture — the model self-records.\n\n"
        f"Use this skeleton (adapt build + the function name to the demo):\n{_SELF_TRACED_SKELETON_REF}\n\n" + tail
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
    it trace+1CQ."""
    esc = re.escape(fn)
    return bool(re.search(r"\.%s\s*\(" % esc, demo_src) or re.search(r"\b%s\s*\(\s*[^)\s]" % esc, demo_src))


_SKELETON_COMPONENT = """
import os
import time
import pytest
import ttnn

PERF_ITERS = int(os.environ.get("TT_PERF_ITERS", "5"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))


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


def _sibling_component_perf_ref(root, task: str) -> str:
    """A WORKING per-component perf test for a sibling module of the SAME model, to anchor the
    device-open / fixture / trace-capture / CCL handling the isolated generator otherwise has to guess.
    Prefer one proven trace-capable (a .trace_caps.json sidecar next to it). Empty string if none."""
    try:
        pdir = root / "tests" / "pcc"
        cands = [p for p in sorted(pdir.glob("test_*_perf.py")) if p.stem != "test_%s_perf" % task]
        if not cands:
            return ""
        proven = [p for p in cands if (p.parent / (p.name + ".trace_caps.json")).is_file()]
        ref = (proven or cands)[0]
        body = ref.read_text(errors="ignore")[:8000]
        return (
            "\n\nREFERENCE — a WORKING per-component perf test for a SIBLING module of THIS SAME model, on "
            "THIS board. Its device-open, `device_params`/`device` fixture params, trace-capture bracket, and "
            "cross-device-collective handling are CORRECT for this model + hardware. COPY THOSE VERBATIM: use "
            "the SAME `device_params` dict (do NOT add `fabric_config` and do NOT open a mesh — a component "
            "runs single-device), and if this module calls a collective (all_reduce / all_gather) reuse the "
            "sibling's single-device no-op wrapper for it. Change ONLY the module-specific build and the "
            "`_forward()` body.\n"
            "<working_sibling path='%s'>\n%s\n</working_sibling>" % (ref.name, body)
        )
    except Exception:
        return ""


def _shard_slice_directive(root, task: str) -> str:
    """When the module graduated TP-sharded, per-module optimize measures ONE rank on a single chip.
    Build every sharded weight/input at its per-chip 1/degree slice along the module's shard dim (NOT
    the full unsharded shape, which may not fit on one chip) and keep the collective a single-device
    no-op. The shard dim is encoded in the module's graduated sharded stub (cluster_axis /
    ShardTensorToMesh(dim=...) / mesh_shape); include it so the generator slices the right axis. Empty
    for a non-sharded (degree<=1) module."""
    try:
        degree = int(os.environ.get("TT_PERF_SHARD_DEGREE", "1") or "1")
    except Exception:
        degree = 1
    if degree <= 1 or root is None:
        return ""
    snap = ""
    try:
        p = root / "_stubs" / ("%s.py.last_good_sharded" % task)
        if p.is_file():
            snap = p.read_text(errors="ignore")[:8000]
    except Exception:
        snap = ""
    out = (
        "\n\nTP-SHARDED MODULE (shard degree=%d): this module runs tensor-parallel across %d chips in the "
        "real pipeline, so its FULL unsharded shape may not fit on one chip. For this SINGLE-CHIP "
        "per-module measurement build ONE rank's slice: size every sharded weight/input to 1/%d along the "
        "module's shard dim, keep the collective (all_reduce/all_gather) a single-device no-op, and stay on "
        "the single `device` fixture (NO mesh, NO fabric_config). Optimizing this per-chip slice yields the "
        "levers that transfer to the sharded deployment." % (degree, degree, degree)
    )
    if snap:
        out += (
            "\nThe module's graduated TP-sharded stub below shows the shard dim (cluster_axis / "
            "ShardTensorToMesh(dim=...) / mesh_shape) — slice that SAME axis:\n"
            "<sharded_stub path='%s.py.last_good_sharded'>\n%s\n</sharded_stub>" % (task, snap)
        )
    return out


def _component_prompt(
    out_rel: str, src_label: str, demo_src: str, task: str, cache_instr: str = "", agentic: bool = False, root=None
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
        + f"LIFT ITS SETUP VERBATIM: reproduce its ttnn module build AND its input-tensor construction EXACTLY "
        f"as the source does — INCLUDING any model-faithful sample-input hook (e.g. a `_special` / "
        f"`_acestep_sample_inputs` / captured-inputs / bespoke-inputs branch in its `_build_torch_reference`). "
        f"The source's inputs are the ONLY correct inputs for this module: they match its precomputed shapes "
        f"(RoPE tables, masks, sequence length). NEVER substitute a generic `torch.randn` / `_make_arg_for` "
        f"synthetic input — a wrong-shaped input crashes the forward before its compute ops and yields a "
        f"degenerate, structural-op-free capture. On a golden-cache MISS reproduce the source's reference-model "
        f"load / submodule resolution VERBATIM (do NOT substitute AutoModel/from_pretrained on the miss path; if "
        f"it loads a model-local `_reference_loader`, load it the SAME way it does). Build the module + ALL its "
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
        f"decisions inside `_forward()`. If the module's own forward has an irreducible host op, print "
        f"TRACE_NOT_TRACE_CAPABLE=1 and skip the trace so it falls back to the eager number:\n"
        f"{_SKELETON_COMPONENT}\n\n"
        + _sibling_component_perf_ref(root, task)
        + _shard_slice_directive(root, task)
        + tail
    )


def _pipeline_is_generative(model_root, demo_src: str) -> bool:
    """Whether this model retires tokens one at a time, from the contract its pipeline keeps.

    The decode contract is the signal: decode_step(state) advances exactly one token per call, which
    is what the perf test's capped loop is for. A declared PIPELINE_STAGES entry that is
    autoregressive counts too -- that is the same test model_contract._c_decode_contract applies
    before demanding the hooks.

    Falls back to the demo text ONLY for the two markers that are generation APIs rather than
    incidental words, and never for "for _ in range", which classed almost every demo generative.
    """
    try:
        from .model_contract import Source

        src = Source.load(model_root)
        if src.mentions("def decode_step"):
            return True
        for _p, node in src.assigns("PIPELINE_STAGES"):
            import ast as _ast

            if isinstance(node.value, (_ast.List, _ast.Tuple)):
                names = [e.value for e in node.value.elts if isinstance(e, _ast.Constant)]
                # An AR stage is one the pipeline also gives a decode contract to; absent that, a
                # declared stage set alone does not make a model generative -- an encoder-decoder
                # translation model declares stages and retires one output per call.
                if any(isinstance(n, str) and src.mentions("def %s_trace_step" % n) for n in names):
                    return bool(src.mentions("def decode_step"))
    except Exception:  # noqa: BLE001 -- an unreadable source falls back to the text below
        pass
    low = (demo_src or "").lower()
    return ("max_new_tokens" in low) or ("decode_step" in low)


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
    stacks: list | None = None,
) -> str | None:
    """Write tests/e2e/test_<task>_perf.py by lifting build+run from a source — the WHOLE pipeline
    forward (prefill + a capped decode loop when the source has one). Returns the node id
    'tests/e2e/test_<task>_perf.py::test_<task>_perf' on success, else None. `runner` (prompt->str)
    overrides the default claude call (for tests).

    stacks: optional list of StackInfo objects discovered by find_all_stacks(). When provided and
    len > 1, per-stack depth variable instructions (TT_PERF_STACK0_LAYERS, TT_PERF_STACK1_LAYERS,
    ...) are included in the LLM prompt instead of the single-stack TT_PERF_LAYERS instruction.

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
        # A REUSED TEST STILL NEEDS THE MARKS. Injection at the write point only fires when a test is
        # GENERATED, and a test that already exists returns here first -- which is how run 23 ran an
        # unmarked test written three hours earlier and reported "no stage signposts" for the seventh
        # time. The injector is idempotent and refuses when it cannot place the block, so applying it
        # to an existing file is safe and is the only way the marks reach a run that regenerates
        # nothing.
        try:
            from .stage_marks import inject_stage_marks as _inject_marks

            _cur = out_path.read_text()
            _new, _why = _inject_marks(_cur)
            if _new != _cur:
                out_path.write_text(_new)
                print("      · stage marks (existing test): %s" % _why, file=sys.stderr, flush=True)
        except Exception as _mi:  # noqa: BLE001
            print("      · stage marks (existing test): not injected (%s)" % _mi, file=sys.stderr, flush=True)
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
        "- ISL / OSL ARE NOT YOURS TO CHOOSE. Build the prompt to EXACTLY PERF_ISL_TOKENS tokens with "
        "agent.perf_test_gen.prompt_ids_for_isl(tokenizer, PERF_ISL_TOKENS) -- do NOT write an example "
        "sentence and do NOT pick a length. Left to a model, this became a six-token prompt whose length "
        "nothing recorded, so the reported throughput silently described a six-token context. Echo "
        "PERF_ISL_TOKENS= and PERF_OSL_TOKENS= so the conditions are in the log.\n"
        "- BOUNDED + profiler-safe so tracy's 12000-marker buffer never overflows: cap the work (decode "
        "loop via PERF_OSL_TOKENS -- the SAME value the test declares and prints, never a second variable "
        "unit is the one the test reports -- never a smaller literal, or the profile samples a fraction "
        "of the request and every recurring op is under-counted; or a SINGLE forward if there's no "
        "loop), AND drain "
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
        "- The WORKLOAD SIZE comes from the model, never from a literal you choose. The skeleton sends "
        "TT_PERF_BATCH=0, which means ASK THE PIPELINE -- build_pipeline answers with its own declared "
        "batch. Do NOT introduce another environment variable that caps how much input is fed (clip "
        "count, stream count, image count, chunk count). If the shape of the input genuinely needs a "
        "count, DERIVE it from the batch the pipeline was built with, and let the environment override "
        "it DOWNWARD only. A test that builds for the model's batch and then feeds a fraction of it "
        "measures a workload nobody asked for, and its number is later compared against a full-batch "
        "ceiling.\n"
        "- KEEP the skeleton's `_pl` / `PERF_LAYERS` depth lines VERBATIM near the top. A POSITIVE "
        "TT_PERF_LAYERS caps profiled depth for deep models so the device profiler's marker buffer does "
        "not overflow (worse on a multi-chip mesh, where markers scale x chips); the tool sends that "
        "number for tracy runs. The variable being ABSENT means ALL LAYERS -- the tool expresses 'whole "
        "model' by REMOVING the cap, so `PERF_LAYERS` is None in that case. Do NOT add a numeric default "
        "and do NOT call os.environ.setdefault on it: defaulting absent-to-a-number silently caps the "
        "full-depth gate, which then measures a fraction of the model and calls it whole.\n"
        "- NEVER pass a literal 0 as a depth. PREFERRED: hand the builder `PERF_LAYERS` directly, since "
        "None is the all-layers value for essentially every builder; or omit the depth argument entirely "
        "and let the builder read the env itself. Passing 0 to a builder whose 'all' sentinel is None "
        "yields a ZERO-LAYER model: no KV cache, and the first prefill dies on kv_cache[0][0] before any "
        "timing marker, so the gate can only report 'no markers'. If the builder REQUIRES an explicit "
        "depth and its 'all' sentinel is something else, read its source and emit THAT -- never guess.\n"
        "- NO PCC / correctness assertions (this is perf only) — just assert the pipeline produced output.\n"
        "- TIME THE FORWARD: keep the skeleton's time.monotonic() bracket around the bounded forward and "
        'the final print("FORWARD_WALL_MS=...") VERBATIM inside `_eager_forward()` — the harness reads it '
        "as an independent end-to-end check on the profiler capture. Do not remove or rename it.\n"
        "- KEEP THE MEASUREMENT ORDER VERBATIM: `_eager_forward()` and `_traced_forward()` are two separate "
        "functions, and the `_PROFILING` branch at the bottom decides which run. Do NOT inline them back "
        "into one straight-line body and do NOT call both unconditionally. Under the full-pipeline gate the "
        "model is built at FULL depth, and calling both means building it TWICE on one device: the second "
        "build has no memory left for its KV cache and dies before printing any marker, which the gate can "
        "only report as 'no markers'. Trace is the default path; eager is its FALLBACK.\n"
        "- KEEP the skeleton's trace-replay block VERBATIM in structure: the `_PERF_TRACE`/`_DEV_PARAMS` "
        "device-param gate near the top AND the `_traced_forward()` measure_adapter body. This is a "
        "MODEL-AGNOSTIC, GPU-comparable latency (TRACE_PER_TOKEN_MS + per-stage TRACE_STAGE_MS). Do NOT "
        "write a per-model adapter class — the tool ships the generic PipelineStageAdapter, which profiles "
        "WHATEVER emit-e2e emitted: every `PIPELINE_STAGES` entry is traced, "
        "falling back to the single decode contract for decode-only pipelines. Your "
        "ONLY job in that block is to fill `_build_for_perf(dev)` so it RETURNS THE RESIDENT, STAGE-EXPOSING "
        "PIPELINE OBJECT — the one carrying PIPELINE_STAGES + the per-stage trace hooks (or a trace-capturable "
        "`decode_step(state)`). Call the model's module-level `build_pipeline(device, ...)` factory that "
        "emit-e2e emits (import it from the demo's tt/pipeline module, pass `dev` + the same build args). Do "
        "NOT return the demo's run_tts()/generate() RESULT or a closure that runs the pipeline — that object "
        "has no stage hooks, so the adapter raises and the trace silently falls back to FORWARD_WALL_MS. Set "
        "`_prompt_ids` to a SMALL prompt. Leave everything else in the block verbatim. The clean numbers are "
        "emitted automatically once `_build_for_perf` returns that object; a genuine repeat-prefill pipeline "
        "with no stage hooks and no decode_step legitimately falls back to FORWARD_WALL_MS, which is fine. "
        "Never delete the block, never let it fail the test.\n"
        "- TRACE BLOCK + SELF-OPEN: if (per the DEVICE OPEN rule) the test self-opens a mesh, pass the "
        "device the test actually opened to `measure_adapter(...)` (NOT a fixture `device`), and put "
        "`trace_region_size`/`num_command_queues` on that self-open call when TT_PERF_TRACE (drop the "
        "`_DEV_PARAMS`/`device_params` fixture entirely). Keep `_build_for_perf(dev)` building the pipeline "
        "on the passed-in `dev` so both the eager forward and the trace run the SAME sharded topology.\n"
        "- MESH SHAPE — honor the tool's topology, HOWEVER this test obtains its device. The shape MUST come "
        "from `from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape` / "
        "`rows, cols = resolve_mesh_shape(default_rows=<source rows>, default_cols=<source cols>)`. That is "
        "what lets --devices/--mesh reshape the run (single->1x1, N chips->the planned TP x DP); with the env "
        "unset it returns the source's own shape, so a bare manual run behaves exactly as before. It applies "
        "to BOTH shapes a test can take:\n"
        "    SELF-OPENED mesh -> open `MeshShape(rows, cols)`.\n"
        "    `mesh_device` FIXTURE + @pytest.mark.parametrize (what most tt-metal demos use, and what the "
        "SAME-FIXTURES rule tells you to reuse) -> keep the fixture and the decorator, but the tuple inside "
        "the parametrize must be resolve_mesh_shape's result, with the source's own lookup as its DEFAULT:\n"
        "        _DEMO_MESH = {<the source's board table>}.get(os.environ.get('MESH_DEVICE'), <source default>)\n"
        "        _MESH_SHAPE = resolve_mesh_shape(default_rows=_DEMO_MESH[0], default_cols=_DEMO_MESH[1])\n"
        "        @pytest.mark.parametrize('mesh_device', [_MESH_SHAPE], indirect=True)\n"
        "  NEVER hardcode the shape, and NEVER leave a copied board-name table as the ONLY source of it: a "
        "lookup keyed on MESH_DEVICE cannot see --devices/--mesh, so the planned topology is silently ignored "
        "and the operator must hand-set MESH_DEVICE. Measured on llama3_1_8b_p150 (2026-07-26): the copied "
        "table meant --devices single never reached the device open.\n"
        "- Lift the imports + build args straight from the demo above.\n\n"
        f"Use this structural skeleton (adapt the build+run to the demo):\n{skeleton_for(root)}\n"
    )
    # Multi-stack depth variable instructions: when multiple block stacks are present, override
    # the single TT_PERF_LAYERS instruction with per-stack variables so the LLM wires each
    # stack's depth cap separately.
    if stacks and len(stacks) > 1:
        # NAMES COME FROM THE MODEL, NOT FROM A LOOP COUNTER.
        #
        # This used to hand the generator a positional id and a stack path -- "PERF_STACK0_LAYERS ->
        # audio_tower.layers (32 layers)" -- and let it choose builder argument names from that. It
        # chose `audio_layers` and `text_layers`, while the knob repair (which reads PIPELINE_STAGES
        # and writes `<stage>_layers`) created encode_layers / prefill_layers / decode_layers. The
        # test then passed kwargs the factory does not accept, they vanished into **kwargs, and the
        # model built every layer: measured 18729 -> 18729 dispatched ops, a cap that capped nothing.
        #
        # PIPELINE_STAGES is declared in the model's own source, needs no build and no device, and is
        # already what the repair names its parameters after. Deriving from it makes the two ends
        # agree by construction instead of by luck. Falls back to the positional form only when the
        # model declares no stages, which is the case that has no shared vocabulary to use.
        # Guarded because this module is also loaded BY PATH (tests do it, and so does any caller
        # without the package on sys.path): a bare relative import raises "attempted relative import
        # with no known parent package" and would take generation down with it.
        try:
            from .stack_knob_repair import stage_names as _stage_names
        except ImportError:  # loaded as a standalone module
            import importlib.util as _ilu

            _spec = _ilu.spec_from_file_location(
                "_stack_knob_repair", str(Path(__file__).resolve().parent / "stack_knob_repair.py")
            )
            _skr = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_skr)
            _stage_names = _skr.stage_names

        _stages = [st for st in (_stage_names(root) or []) if str(st).isidentifier()]
        if _stages:
            _stack_lines = "\n".join(f"  TT_PERF_{st.upper()}_LAYERS -> build argument `{st}_layers`" for st in _stages)
            _stack_var_examples = "\n".join(
                f'_pl_{st} = (os.environ.get("TT_PERF_{st.upper()}_LAYERS") or "").strip()\n'
                f"PERF_{st.upper()}_LAYERS = int(_pl_{st}) if (_pl_{st}.isdigit() and int(_pl_{st}) > 0) else None"
                for st in _stages
            )
            prompt += (
                f"\n\nPER-STAGE DEPTH OVERRIDE: this model runs {len(stacks)} repeating block stacks "
                f"across the stages it declares in PIPELINE_STAGES ({', '.join(_stages)}). REPLACE the "
                "single `_pl` / `PERF_LAYERS` lines from the skeleton with one variable per STAGE "
                "(same None-means-all-layers contract):\n"
                f"{_stack_var_examples}\n"
                "Pass each one to the builder using EXACTLY these argument names -- they are the names "
                "the tool adds to build_pipeline, and any other spelling is silently swallowed by "
                f"**kwargs and caps nothing:\n{_stack_lines}\n"
                "Also keep passing `layers=PERF_LAYERS` when the builder accepts it: it is the default "
                "depth for any stack a per-stage argument does not name.\n"
            )
        else:
            _stack_lines = "\n".join(
                f"  PERF_STACK{i}_LAYERS -> {s.path} ({s.count} layers)" for i, s in enumerate(stacks)
            )
            _stack_var_examples = "\n".join(
                f'_pl{i} = (os.environ.get("TT_PERF_STACK{i}_LAYERS") or "").strip()\n'
                f"PERF_STACK{i}_LAYERS = int(_pl{i}) if (_pl{i}.isdigit() and int(_pl{i}) > 0) else None"
                f"  # {s.path}: {s.count} total"
                for i, s in enumerate(stacks)
            )
            prompt += (
                f"\n\nMULTI-STACK DEPTH OVERRIDE: this model has {len(stacks)} repeating block stacks. "
                "REPLACE the single `_pl` / `PERF_LAYERS` lines from the skeleton with these per-stack "
                "variables (one env var per stack, same None-means-all-layers contract):\n"
                f"{_stack_var_examples}\n"
                f"Pass each depth cap to the builder:\n{_stack_lines}\n"
                "None means all layers for that stack. Do NOT emit `PERF_LAYERS` for a multi-stack model.\n"
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
    _main_agentic_body = (
        _self_traced_prompt(out_rel, task, src_label, demo_src, self_traced, agentic=True) if self_traced else prompt
    )
    prompt += (
        "\n\nDo NOT use any tools and do NOT try to write the file yourself — the caller writes it. "
        "Respond with ONLY the complete python file content as your message text — no prose, no markdown fences."
    )
    if _component:
        prompt = _component_prompt(out_rel, src_label, demo_src, task, cache_instr=_cache_instr, root=root)
    if self_traced and not _component:
        prompt = _self_traced_prompt(out_rel, task, src_label, demo_src, self_traced)
    if (
        runner is None
        and validate is not False
        and os.environ.get("TT_PERF_NO_AGENTIC_BUILDER", "") in ("", "0", "false", "False")
    ):
        try:
            from .perf_test_agent import build_component_perf_test

            _body = (
                _component_prompt(out_rel, src_label, demo_src, task, cache_instr=_cache_instr, agentic=True, root=root)
                if _component
                else _main_agentic_body
            )
            if build_component_perf_test(root, task, out_rel, _body):
                _verdict, _ = validate_generated_perf_test(out_path, task, component=_component)
                if _verdict in ("ok_1cq", "ok_marker", "skip"):
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
    #
    # ASKED OF THE PIPELINE CONTRACT, NOT OF A BAG OF SUBSTRINGS. This matched any of six strings in
    # the demo source, one of which was "for _ in range" -- present in very nearly every Python file
    # ever written. So almost every demo was classed generative, and its perf test was then REJECTED
    # and regenerated until it contained PERF_OSL_TOKENS: a decode-loop cap forced onto models with
    # no decode loop, burning correction rounds on a requirement that could not be satisfied
    # honestly. The remaining five are the same guess in kinder clothes -- ".generate" is a method
    # any wrapper may expose, "next_token" is a variable name.
    #
    # What makes a pipeline generative is the contract it keeps: a decode_step(state) hook (one token
    # per call, by definition -- PipelineDecodeAdapter raises NotTraceCapable without it), or a
    # declared PIPELINE_STAGES set containing an autoregressive stage. model_contract already reads
    # both, from the pipeline's own source, and _c_decode_contract already reports on them.
    demo_is_generative = _pipeline_is_generative(model_root, demo_src)
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
        _claims_trace = "trace+" in content.lower()
        if _times_selfrec and _external_capture:
            stall += 1
            feedback = _correction_feedback(
                "the timed function (%s) ALREADY records its own trace internally — do NOT re-record it with "
                "measure_adapter or begin_trace_capture (a nested capture fatals + hangs the device). Just TIME "
                "it directly and print TRACE_PER_TOKEN_MS + TRACE_REPLAY_PATH=trace+1cq." % ", ".join(_times_selfrec),
                "",
                content,
            )
            prev_draft = content
            continue
        if _claims_trace and not _external_capture and not _times_selfrec:
            stall += 1
            feedback = _correction_feedback(
                "the test prints a traced TRACE_REPLAY_PATH but the timed function does NOT record a trace "
                "(it is not one of the model's self-recording functions) and you did not call measure_adapter — "
                "so this is TIMING THE EAGER PATH and mislabelling it. Wrap the timed forward in measure_adapter "
                "to actually capture + replay a trace, or time a function that self-records.",
                "",
                content,
            )
            prev_draft = content
            continue
        if demo_is_generative and "PERF_OSL_TOKENS" not in content:
            stall += 1
            feedback = _correction_feedback(
                "generative pipeline but the test omits the decode-loop cap (PERF_OSL_TOKENS) — it "
                "would profile a prefill-only slice. Add the capped decode loop.",
                "",
                content,
            )
            prev_draft = content
            continue
        prev_draft = content
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # STAGE MARKS ARE INJECTED, NOT REQUESTED. The skeleton above is a "structural reference
        # handed to the LLM", so anything added to it is a suggestion -- and the generated test for
        # voxtral came back with zero references to the marked pass, leaving five commits' worth of
        # downstream machinery starved behind an emission that never ran. This puts it in after the
        # model-specific content is written and before it is validated, so what ships is what runs.
        try:
            from .stage_marks import inject_stage_marks as _inject_marks

            content, _why_marks = _inject_marks(content)
            print("      · stage marks: %s" % _why_marks, file=sys.stderr, flush=True)
        except Exception as _mi:  # noqa: BLE001
            print("      · stage marks: not injected (%s)" % _mi, file=sys.stderr, flush=True)
        out_path.write_text(content)
        if not validate:
            return node
        verdict, failure = validate_generated_perf_test(out_path, task, component=_component)
        if verdict in ("ok_1cq", "ok_marker", "skip"):
            return node
        if _component and "WEDGE" in failure:
            trace_wedges += 1
            print(
                f"      · trace wedge {trace_wedges}/{_TRACE_WEDGE_LIMIT}: device hung capturing this "
                "module's forward — reset + retrying the TRACE",
                file=sys.stderr,
                flush=True,
            )
            if trace_wedges >= _TRACE_WEDGE_LIMIT:
                return None
            feedback = _correction_feedback(_COMPONENT_WEDGE_REASON, failure, prev_draft)
            continue
        stall += 1
        if "WEDGE" in failure:
            _why = "device wedged on a non-capturable step — reset + regenerating"
        else:
            _why = ((_extract_error(failure).splitlines() or [""])[0] or "did not run the full pipeline").strip()[:80]
        print(f"      · perf-test regen {stall}/{_STALL_LIMIT}: {_why}", file=sys.stderr, flush=True)
        reason = (
            "the module perf test produced no TRACE_PER_TOKEN_MS — implement the trace-replay block so "
            "it captures a REAL device trace at 1 CQ (trace required; eager is only enabled by the operator "
            "via TT_PERF_TRACE=0, never as a fallback)"
            if _component
            else "the test did not run the full pipeline / errored"
        )
        feedback = _correction_feedback(reason, failure, prev_draft)
    return None
