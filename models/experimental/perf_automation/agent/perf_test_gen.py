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
        r = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
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


def _run_perf_node(node_abs: str, extra_env: dict, timeout_s: int = 2400):
    def _once(ev):
        env = dict(os.environ)
        env["TT_PERF_TRACE"] = "1"
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


def validate_generated_perf_test(out_path: Path, task: str) -> tuple[str, str]:
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
    return "invalid", (
        f"trace-capable pipeline degraded to 1cq (path={path2 or _parse_trace_path(out1)}); the 2-CQ "
        "overlap never engaged, so it would silently downgrade at the optimize bookend. " + (_extract_error(out2) or "")
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
    out_rel = f"tests/e2e/test_{task}_perf.py"
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
    if source_kind == "pcc":
        prompt = (
            f"Write a pytest PERFORMANCE test file `{out_rel}` for the '{task}' pipeline of this TTNN model.\n"
            f"This source is a CORRECTNESS (PCC) test — build and run the TTNN model EXACTLY as it does, but "
            f"KEEP ONLY the on-device TTNN forward: DROP the reference/torch model construction and DROP every "
            f"PCC / comp_pcc / allclose / assert_with_pcc correctness comparison.\n"
            f"<pcc_test path='{src_label}'>\n{demo_src}\n</pcc_test>\n\n"
            "Requirements:\n"
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
        "- CAP THE INPUT SIZE SMALL: use a SMALL fixed sequence length / token count (e.g. 128) for every "
        "forward. Do NOT reuse the model's production / maximum shapes (max_position_embeddings, max_seq, "
        "max_enc_seq, etc.) even if the source/PCC test does — those are correctness stress sizes. Under "
        "tracy EVERY device op is instrumented, so one max-seq forward runs orders of magnitude slower and "
        "the host blocks in ttnn.synchronize_device for many minutes, stalling the run. If the source "
        "defines a large seq constant, OVERRIDE it with a small value here (env-overridable, small default). "
        "A perf profile only needs a representative dispatch-dense pass, not the max shape.\n"
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
    # A generative demo's perf test must exercise the (capped) decode loop, not a prefill-only slice.
    demo_is_generative = any(
        k in demo_src.lower()
        for k in ("max_new_tokens", "generate(", ".generate", "next_token", "decode_step", "for _ in range")
    )
    gen = runner or _claude
    if validate is None:
        validate = runner is None
    # CORRECTION LOOP — no fixed attempt budget. Keep regenerating until the test is trace+2cq-acceptable
    # (or a legitimate eager terminal), feeding the REAL error + the LLM's own previous draft back each
    # round so it EDITS the failing file rather than rewriting blind. Terminate only on acceptance or on
    # a STALL (no forward progress _STALL_LIMIT times running) — never ship a test that fails 2cq. This is
    # fully model- and hardware-agnostic: the verdicts come from what the pipeline actually did on device.
    feedback = ""
    prev_draft = None
    stall = 0
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
        verdict, failure = validate_generated_perf_test(out_path, task)
        if verdict in ("ok_2cq", "ok_marker", "skip"):
            return node
        stall += 1
        print(f"[perf_test_gen] draft rejected (correcting, stall {stall}/{_STALL_LIMIT}): {failure[:200]}", flush=True)
        reason = (
            "the test ran but never held trace+2cq (it degraded to 1cq); the 2-CQ input overlap must "
            "engage so the optimize bookend doesn't silently downgrade"
            if "degraded to 1cq" in failure
            else "the test did not run the full pipeline / errored"
        )
        feedback = _correction_feedback(reason, failure, prev_draft)
    return None
