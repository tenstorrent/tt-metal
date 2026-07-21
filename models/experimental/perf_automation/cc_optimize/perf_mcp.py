"""perf-mcp — an EXTERNAL stdio MCP server that exposes the perf_automation tool's DETERMINISTIC
core (profile / measure-with-integrity-verdict / pcc / git) to a free-roaming Claude Code agent.

This is the "deterministic gate" half of the Claude-Code-native optimize design: Claude Code drives
the loop (what to try, when to stop) by REASONING; this server owns the parts that must be
guaranteed, not judged — the measurement, the integrity guards, the keep criterion's inputs, and
git. Every tool here REUSES the exact functions the FSM uses (measure.measure_runs,
pcc_runner.run_pcc, roofline.*, remeasure._comparable, gitio) via the same ctx-shim the isolated
kernel test proved — so "valid + faster here" == "the FSM would agree".

ADDITIVE: imports the existing agent package, touches none of it. Removing this file fully reverts.

Config via env (set in .mcp.json):
  PERF_MCP_MANIFEST   path to a discovered runs/<ts>/manifest.json (gives perf_test, pcc, env, config)
  PERF_MCP_MODEL_ROOT optional override of the model dir (default: manifest config.model_root)
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import signal
import statistics
import sys
import tempfile
from pathlib import Path

# import the EXISTING deterministic core (no reimplementation)
_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG.parent.parent.parent))  # repo root, so `models...` imports resolve
sys.path.insert(0, str(_PKG))  # the perf_automation dir, so `agent` imports resolve

from agent import gitio, promote, roofline, router  # noqa: E402
from agent.handlers import remeasure as _rm  # noqa: E402
from agent.measure import measure_runs  # noqa: E402
from agent.pcc_runner import run_pcc  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("perf-mcp")

_MANIFEST_PATH = os.environ.get("PERF_MCP_MANIFEST", "")
_MANIFEST = json.load(open(_MANIFEST_PATH)) if _MANIFEST_PATH else {}

# Per-mode retargeting (the whole-pipeline runner sets these per head, so one manifest covers every
# mode of a multi-modal model): override which perf test profile_model/measure_candidate run, and
# which e2e PCC test check_pcc runs, without minting a new manifest.
if os.environ.get("PERF_MCP_PERF_TEST") and _MANIFEST:
    _MANIFEST.setdefault("perf_test_resolved", {})["path"] = os.environ["PERF_MCP_PERF_TEST"]
    if os.environ.get("PERF_MCP_PERF_CASE") is not None:
        _MANIFEST["perf_test_resolved"]["case"] = os.environ["PERF_MCP_PERF_CASE"]
if os.environ.get("PERF_MCP_PCC_TEST") and _MANIFEST:
    _MANIFEST.setdefault("pathmap", {}).setdefault("pcc", {}).setdefault("end_to_end", {})
    _MANIFEST["pathmap"]["pcc"]["end_to_end"]["path"] = os.environ["PERF_MCP_PCC_TEST"]
_MODEL_ROOT = Path(os.environ.get("PERF_MCP_MODEL_ROOT") or _MANIFEST.get("config", {}).get("model_root", "."))
_ENV = _MANIFEST.get("env", {})
# where profile_model stashes the current baseline so measure_candidate can compare structurally
_BASELINE_PATH = Path(tempfile.gettempdir()) / "perf_mcp_baseline.json"
_FULLPIPE_BASELINE_PATH = Path(tempfile.gettempdir()) / "perf_mcp_full_pipeline_baseline.json"
_FULLPIPE_BASELINE_1CQ_PATH = Path(tempfile.gettempdir()) / "perf_mcp_full_pipeline_baseline_1cq.json"
_FULLPIPE_TOL = float(os.environ.get("PERF_MCP_FULLPIPE_TOL", "0.08"))
_FULLPIPE_SAMPLES = max(1, int(os.environ.get("PERF_MCP_FULLPIPE_SAMPLES", "1")))
_FULLPIPE_TARGET_MS = float(os.environ.get("PERF_MCP_TARGET_MS", "0") or "0")

# C++-kernel SAFETY: a bad Metalium kernel can WEDGE a device core (tt-lang/ttnn fail gracefully; raw
# C++ can deadlock the NoC). Device runs are already subprocess-isolated+timeout-bounded (so the loop
# survives a hang), but a wedged core can persist across runs -> recover with tt-smi reset. Only after
# TWO consecutive crashes (not a single transient) so it's a RARE fallback, not routine.
import shutil as _shutil  # noqa: E402
import subprocess as _sp  # noqa: E402

_CONSEC_CRASH = {"n": 0}
_TT_SMI = _shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"


def _device_recover(where: str) -> None:
    """Best-effort board reset after a likely wedge (2+ consecutive device crashes)."""
    try:
        _sp.run([_TT_SMI, "-r", "0"], capture_output=True, text=True, timeout=180)
        sys.stderr.write(f"[perf-mcp] device recovered via tt-smi -r after wedge at {where}\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[perf-mcp] tt-smi reset failed at {where}: {exc}\n")


def _note_device_crash(where: str) -> None:
    _CONSEC_CRASH["n"] += 1
    if _CONSEC_CRASH["n"] >= 2:  # repeated crash -> likely a wedged core -> recover
        _device_recover(where)
        _CONSEC_CRASH["n"] = 0


def _note_device_ok() -> None:
    _CONSEC_CRASH["n"] = 0


def _l1_sig(text) -> bool:
    s = (str(text) or "").lower()
    return ("circular buffer" in s or "max l1" in s or "l1 size" in s) and (
        "beyond max l1" in s or "grow to" in s or "l1 size of" in s
    )


def _is_l1_overflow(msg) -> bool:
    if _l1_sig(msg):
        return True
    import re as _re2

    for lp in _re2.findall(r"(/\S+run\d+_tracy\.log)", str(msg) or ""):
        try:
            if _l1_sig(Path(lp).read_text(errors="ignore")):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _reclaim_mesh(where: str) -> None:
    try:
        _sp.run([_TT_SMI, "-r"], capture_output=True, text=True, timeout=420)
        sys.stderr.write(f"[perf-mcp] full-mesh reset (L1 overflow) at {where}\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[perf-mcp] full-mesh reset failed at {where}: {exc}\n")
    _CONSEC_CRASH["n"] = 0


_L1_OVERFLOW_MSG = (
    "L1_OVERFLOW: this config's circular buffers exceed the per-core L1 budget (~1.5MB on Wormhole) "
    "and crashed the run; the mesh was reset. Reduce the L1 footprint (smaller in0_block_w / per_core_N, "
    "or spread the matmul over more cores) and retry — do NOT keep this config."
)

_DRAM_TRACE_SIG_A = ("trace region", "trace_region", "trace buffer", "trace_buffer")
_DRAM_TRACE_SIG_B = ("full", "exceed", "not enough", "out of space", "too small", "ran out", "grow the trace")
_TRACE_REGION_DEFAULT = 23887872
_TRACE_REGION_MAX = 512 * 1024 * 1024


def _dram_trace_sig(text) -> bool:
    s = (str(text) or "").lower()
    return any(a in s for a in _DRAM_TRACE_SIG_A) and any(b in s for b in _DRAM_TRACE_SIG_B)


def _is_dram_trace_overflow(msg) -> bool:
    if _dram_trace_sig(msg):
        return True
    import re as _re3

    for lp in _re3.findall(r"(/\S+run\d+_tracy\.log)", str(msg) or ""):
        try:
            if _dram_trace_sig(Path(lp).read_text(errors="ignore")):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _grow_trace_region() -> int:
    cur = int(os.environ.get("TT_PERF_TRACE_REGION", str(_TRACE_REGION_DEFAULT)) or str(_TRACE_REGION_DEFAULT))
    new = min(cur * 2, _TRACE_REGION_MAX)
    os.environ["TT_PERF_TRACE_REGION"] = str(new)
    return new


def _dram_overflow_msg(new_region: int) -> str:
    return (
        "DRAM_TRACE_OVERFLOW: the captured trace command stream exceeded the reserved trace region; grew "
        "TT_PERF_TRACE_REGION to %d bytes and reset the mesh — retry (auto-healed, not a config to abandon; "
        "if it recurs at the %d-byte cap the trace genuinely does not fit)." % (new_region, _TRACE_REGION_MAX)
    )


# ---------------------------------------------------------------------------
# DETERMINISTIC TERMINATION GATE — the "no reasoning off-ramp" guard
# ---------------------------------------------------------------------------
# The agent must NOT be able to declare DONE by REASONING that a kernel won't help an open op while
# reachable roofline gap remains. The ONLY ways to retire a material-gap open op are: (a) the model
# is at_floor, or (b) a REAL kernel was authored AND measured for it — a measured attempt that fails
# to beat ttnn STILL counts as 'tried' (that's the empirical validation, not an assertion). This is
# made binding by termination_check(), which refuses can_stop while any material op lacks a measured
# kernel attempt. The log PERSISTS across server restarts so a driver can re-invoke claude -p multiple
# rounds on the SAME pipeline and the ladder state carries over; the driver clears it (env path or rm)
# at the START of each pipeline. Override the path per-pipeline via PERF_MCP_KERNEL_LOG.
_KERNEL_LOG_PATH = Path(
    os.environ.get("PERF_MCP_KERNEL_LOG") or (Path(tempfile.gettempdir()) / "perf_mcp_kernel_attempts.json")
)
_MATERIAL_GAP_MS = float(os.environ.get("PERF_MCP_MATERIAL_GAP_MS", "0.25"))
_MAX_KNOB_RETRIES = int(os.environ.get("PERF_MCP_MAX_KNOB_RETRIES", "2"))
_MAX_TRACE_FIX_RETRIES = int(os.environ.get("PERF_MCP_MAX_TRACE_FIX_RETRIES", "5"))
_TRACE_SAFE_HINT = (
    "the kernel WEDGED trace capture — a TRACE-COMPATIBILITY defect in the kernel's LIFECYCLE, NOT a "
    "math error (check_pcc validates math). The compute body is usually fine; the wedge is that the op "
    "recompiles or re-allocates INSIDE trace capture. Fix the lifecycle: (1) build the generic_op "
    "ProgramDescriptor / ttl op ONCE per shape and cache+reuse it — do NOT rebuild or call generic_op "
    "fresh each call; (2) allocate the output buffer ONCE and reuse the same handle — never ttnn.zeros a "
    "new output per call; (3) use override_runtime_args on the cached program instead of baking "
    "buffer_address() into a freshly-built descriptor; (4) warm up the op once BEFORE begin_trace_capture "
    "so compilation never lands in the traced region. VALIDATE IN ISOLATION FIRST: author a single-op "
    "trace test (build inputs once -> warm-up -> begin/end_trace_capture -> execute_trace -> assert PCC vs "
    "the stock op), run it STANDALONE and fix until it traces clean + PCC-passes, THEN wire it into the "
    "model and call measure_candidate ONCE. This is a fixable implementation issue, not an exhausted rung"
)
_ISOLATE_FIRST = (
    " Before integrating, VALIDATE IN ISOLATION: author a standalone single-op trace test (warm-up + "
    "begin/end_trace_capture + execute_trace + PCC vs the stock op), run it STANDALONE, and fix it there "
    "cheaply until it traces clean; only then wire it into the model and measure_candidate ONCE."
)

# kernel-authoring evidence markers, searched in the model source tree (grounds a recorded attempt)
_KERNEL_MARKERS = ("generic_op", "ProgramDescriptor", "KernelDescriptor", "@ttl.", "ttl.operation", "import ttl")
_TP_SHARD_MARKERS = ("ShardTensorToMesh", "shard_tensor_to_mesh")
_CCL_MARKERS = ("all_gather", "reduce_scatter", "all_reduce")


def _scan_kernel_evidence() -> dict:
    """Look for real custom-kernel authoring in the model source so a recorded attempt can't be a
    phantom. Returns {markers, cpp_files, tp_shard, ccl} — empty/False if nothing is present."""
    found, cpp = set(), []
    tp_shard = ccl = False
    try:
        for p in _MODEL_ROOT.rglob("*"):
            if p.is_dir() or p.suffix not in (".py", ".cpp", ".cc", ".h", ".hpp"):
                continue
            if p.suffix in (".cpp", ".cc"):
                cpp.append(str(p.relative_to(_MODEL_ROOT)))
            try:
                txt = p.read_text(errors="ignore")
            except OSError:
                continue
            for m in _KERNEL_MARKERS:
                if m in txt:
                    found.add(m)
            if any(m in txt for m in _TP_SHARD_MARKERS):
                tp_shard = True
            if any(m in txt for m in _CCL_MARKERS):
                ccl = True
    except Exception:  # noqa: BLE001
        pass
    return {"markers": sorted(found), "cpp_files": cpp, "tp_shard": tp_shard, "ccl": ccl}


def _load_attempts() -> list:
    if _KERNEL_LOG_PATH.exists():
        try:
            return json.loads(_KERNEL_LOG_PATH.read_text())
        except Exception:  # noqa: BLE001
            return []
    return []


def _save_attempts(a: list) -> None:
    _KERNEL_LOG_PATH.write_text(json.dumps(a))


_LAST_TARGET_PATH = Path(str(_KERNEL_LOG_PATH) + ".target")
_LAST_TARGET: dict = {}


def _persist_target(t) -> None:
    _LAST_TARGET.clear()
    if isinstance(t, dict):
        _LAST_TARGET.update(t)
    try:
        _LAST_TARGET_PATH.write_text(json.dumps(_LAST_TARGET))
    except Exception:  # noqa: BLE001
        pass


def _load_target() -> dict:
    if _LAST_TARGET:
        return dict(_LAST_TARGET)
    try:
        return json.loads(_LAST_TARGET_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _append_attempt(rec: dict) -> list:
    attempts = _load_attempts()
    sig, kind, note = rec.get("op_signature"), rec.get("kernel_kind"), rec.get("note") or ""
    if rec.get("wedged"):
        attempts = [
            a
            for a in attempts
            if not (a.get("wedged") and a.get("op_signature") == sig and a.get("kernel_kind") == kind)
        ]
    else:
        attempts = [
            a
            for a in attempts
            if not (
                not a.get("wedged")
                and a.get("op_signature") == sig
                and a.get("kernel_kind") == kind
                and (a.get("note") or "") == note
            )
        ]
    attempts.append(rec)
    _save_attempts(attempts)
    _rebuild_optimize_report()
    return attempts


def _autorecord_wedge(reason: str) -> None:
    t = _load_target()
    rec = {
        "op_signature": t.get("op") or "candidate config",
        "kernel_kind": t.get("rung") or "knob",
        "measured_ms": None,
        "beat_baseline": False,
        "note": reason,
        "stages": [],
        "kernel_detected_in_source": False,
        "wedged": True,
        "evidence": {},
        "diff": "",
    }
    try:
        _append_attempt(rec)
    except Exception:  # noqa: BLE001
        pass


def _summary_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cc_summary", str(Path(__file__).parent / "summary.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report_baseline_ms():
    try:
        if _BASELINE_PATH.exists():
            return round(float(json.loads(_BASELINE_PATH.read_text()).get("device_ms", 0.0)), 4)
    except Exception:  # noqa: BLE001
        pass
    return None


def _read_baseline_profile():
    try:
        if _BASELINE_PATH.exists():
            return json.loads(_BASELINE_PATH.read_text())
    except Exception:  # noqa: BLE001
        pass
    return None


def _original_baseline_path():
    model = _MODEL_ROOT.name if _MODEL_ROOT else "model"
    task = os.environ.get("PERF_MCP_TASK", "main")
    return Path(tempfile.gettempdir()) / ("perf_mcp_orig_baseline_%s_%s.json" % (model, task))


def _report_original_baseline_ms():
    try:
        p = _original_baseline_path()
        if p.exists():
            return round(float(json.loads(p.read_text()).get("device_ms", 0.0)), 4)
    except Exception:  # noqa: BLE001
        pass
    return _report_baseline_ms()


def _merge_cumulative(cum_path, attempts) -> list:
    try:
        prior = json.loads(cum_path.read_text())
        if not isinstance(prior, list):
            prior = []
    except Exception:  # noqa: BLE001
        prior = []
    seen, out = set(), []
    for a in list(prior) + list(attempts or []):
        if not isinstance(a, dict):
            continue
        key = (
            a.get("op_signature") or a.get("op_code") or "",
            a.get("kernel_kind") or "",
            (a.get("note") or "")[:200],
            bool(a.get("wedged")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    try:
        cum_path.write_text(json.dumps(out))
    except Exception:  # noqa: BLE001
        pass
    return out


def _rebuild_optimize_report(model_root=None) -> None:
    import time as _t

    attempts = _load_attempts()
    cum_path = Path(str(_KERNEL_LOG_PATH) + ".cumulative")
    merged = _merge_cumulative(cum_path, attempts)
    if not merged:
        return
    root = model_root if model_root is not None else _MODEL_ROOT
    render_path = cum_path
    n_attempts = len(merged)
    try:
        mod = _summary_mod()
        perf_test = (_MANIFEST.get("perf_test_resolved") or {}).get("path") or ""
        text = mod.render_summary(
            render_path,
            _report_baseline_ms(),
            model=Path(root).name,
            task=os.environ.get("PERF_MCP_TASK", "main"),
            metric=os.environ.get("PERF_MCP_METRIC", "device_ms"),
            perf_test=perf_test,
            baseline_profile=_read_baseline_profile(),
            finalized=False,
        )
        when = (
            f"Updated live: {_t.strftime('%Y-%m-%d %H:%M:%S %Z')} · {n_attempts} lever attempt(s) so far — "
            "each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed."
        )
        _key = os.environ.get("PERF_MCP_REPORT_KEY", "optimize")
        _module = os.environ.get("PERF_MCP_REPORT_MODULE")
        if _module:
            _block = mod.module_optimize_block(
                root,
                len(attempts),
                text,
                when,
                module=_module,
                index=os.environ.get("PERF_MCP_REPORT_INDEX", ""),
                pcc_gate=os.environ.get("PERF_MCP_REPORT_PCC", ""),
                outcome="optimizing…",
            )
        else:
            _block = mod.optimize_block(root, len(attempts), text, when)
        mod.upsert_report_section(root, _key, _block)
    except Exception as exc:  # noqa: BLE001
        print(f"  [perf-report] render failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


import re as _re  # noqa: E402


def _op_key(s: str):
    """(op-class token, shape-dims tuple) — e.g. 'MatmulDeviceOperation 32 x 237568 x 2688' ->
    ('matmuldeviceoperation', ('32','237568','2688'))."""
    n = _norm(s)
    parts = n.split()
    cls = parts[0] if parts else n
    return cls, tuple(_re.findall(r"\d+", n))


def _op_match(op_code: str, attempt: dict) -> bool:
    """Does this recorded attempt target this open op? The op CLASS must match AND, if the op carries
    a shape (matmul dims), the attempt MUST carry the SAME shape — so an attempt on one matmul does
    NOT clear a different-shape matmul (the bug that let an expert-matmul attempt clear the
    dispatch-bound mamba 32x32x32). Shapeless ops (BinaryNg/LayerNorm) match on class alone."""
    ocls, onums = _op_key(op_code)
    scls, snums = _op_key(attempt.get("op_signature", ""))
    if not scls or not (scls in ocls or ocls in scls):
        return False
    return (snums == onums) if onums else True


def _op_has_attempt(op_code: str, attempts: list):
    for a in attempts:
        if _op_match(op_code, a):
            return a
    return None


_SEALED_OP_MARKERS = (
    "conv",
    "untilize",
    "tilize",
    "slice",
    "allgather",
    "all_gather",
    "allreduce",
    "all_reduce",
    "reducescatter",
    "reduce_scatter",
    "scaleddotproduct",
    "sdpa",
    "layernorm",
    "rmsnorm",
    "groupnorm",
    "softmax",
    "embedding",
    "pool",
)


def _is_kernel_able(op_code: str) -> bool:
    oc = (op_code or "").lower()
    return not any(m in oc for m in _SEALED_OP_MARKERS)


def _ttl_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("ttl") is not None


_TP_REGIME = os.environ.get("TT_PERF_TP_REGIME", "0") == "1"


def set_tp_regime(enabled: bool) -> None:
    global _TP_REGIME
    _TP_REGIME = bool(enabled)


def _tp_candidate(open_op: dict, op_code: str) -> bool:
    if not _TP_REGIME:
        return False
    oc = (op_code or "").lower()
    if "matmul" not in oc and "linear" not in oc:
        return False
    return (open_op.get("bound_by") or "").lower() in ("memory", "dram", "both")


def _rung_state(matches, kind):
    clean = any((a.get("kernel_kind") or "").lower() == kind and not a.get("wedged") for a in matches)
    wedged = sum(1 for a in matches if (a.get("kernel_kind") or "").lower() == kind and a.get("wedged"))
    return clean, wedged


def _trace_compat_feedback(raw_reason: str) -> str:
    rung = (_load_target().get("rung") or "").lower()
    if rung not in ("tt-lang", "cpp"):
        return raw_reason
    return "%s\n%s" % (raw_reason, _TRACE_SAFE_HINT)


def _op_ladder_status(open_op: dict, op_code: str, attempts: list) -> tuple[bool, str, str]:
    """DETERMINISTIC ladder gate for ONE open op. Returns (done, rung, reason).

    The optimize ladder is knob -> fusion -> tt-lang -> C++. This enforces the climb ORDER from the
    op's OWN profile tags + the recorded kernel attempts — the agent CANNOT skip a rung, and a kernel
    attempt does NOT clear an op while a cheaper lever is still untried. An op is DONE only when the
    WHOLE ladder is exhausted: cheap levers gone (grid=full) AND both kernel rungs measured. That is
    the genuine irreducible residual (e.g. a memory-bound matmul already at full grid + bandwidth).
    There is NO 'kernel was tried, so stop' shortcut and NO OR-with-at_floor escape — the FINAL gate
    decides purely from this per-op ladder, which is driven by deterministic measurement, not by what
    any other gate 'fired'."""
    matches = [a for a in attempts if _op_match(op_code, a)]
    kinds = {(a.get("kernel_kind") or "").lower() for a in matches}
    grid_tries = sum(1 for a in matches if (a.get("kernel_kind") or "").lower() == "grid")
    dtype_tries = sum(1 for a in matches if (a.get("kernel_kind") or "").lower() == "dtype")
    grid = (open_op.get("grid") or "").lower()
    wdtype = (open_op.get("weight_dtype") or "").lower()
    bound = (open_op.get("bound_by") or "").lower()
    is_matmul = "matmul" in (op_code or "").lower()
    # BOX (0) HOST / dispatch bucket (GAP-A) — NOT a device op, so the matmul ladder (grid/dtype/
    # tt-lang/C++) is meaningless. Its lever is a STRUCTURAL host-loop transform (trace capture /
    # 2-CQ). Routed via recall_knobs(op_class=host_fallback). Cleared once a measured 'structural'
    # attempt is on file — a trace that doesn't help still counts as tried (bounded, can't hang).
    if bound == "host" or (open_op.get("bucket") or "").lower() == "host_fallback":
        if not (kinds & {"structural", "trace", "2cq"}):
            return (
                False,
                "structural",
                "host/dispatch-bound: recall_knobs(op_class='host_fallback') and apply a "
                "trace-capture / 2-CQ structural lever to the generation loop; record_kernel_attempt(...,'structural',...)",
            )
        return (
            True,
            "done",
            "host lever tried (trace/2-CQ) -> remaining dispatch residual is irreducible by the loop transform",
        )
    # BOX (1) KNOBS — exhaust the cheap levers IN ORDER before any kernel. A knob is satisfied when
    # the profile tag shows it applied, OR a record_kernel_attempt of that knob-kind is on file (so a
    # PCC-failed/ineffective knob can be marked 'tried' and not loop forever).
    #  (1a) full core grid — applying a full-grid program_config sets grid=full
    if grid and grid != "full" and grid_tries < _MAX_KNOB_RETRIES:
        return (False, "knob:grid", f"occupy the FULL core grid (grid={grid}) via a full-grid program_config")
    #  (1b) lower the weight dtype on a memory-bound matmul (the dominant cheap lever there)
    if is_matmul and bound == "memory" and wdtype in ("fp32", "bf16", "fp16") and dtype_tries < _MAX_KNOB_RETRIES:
        return (
            False,
            "knob:dtype",
            f"lower the weight dtype (now {wdtype}) to bf8_b/bf4_b; if PCC fails, record_kernel_attempt(...,'dtype',...) to mark it tried",
        )
    if _is_kernel_able(op_code):
        _tl_clean, _tl_wedged = _rung_state(matches, "tt-lang")
        _cpp_clean, _cpp_wedged = _rung_state(matches, "cpp")
        if not _tl_clean and _ttl_available() and _tl_wedged < _MAX_TRACE_FIX_RETRIES:
            if _tl_wedged:
                return (
                    False,
                    "tt-lang",
                    "HOLD the tt-lang rung (do NOT switch to cpp — it wedges identically): %s (trace-fix %d/%d)"
                    % (_TRACE_SAFE_HINT, _tl_wedged, _MAX_TRACE_FIX_RETRIES),
                )
            return (
                False,
                "tt-lang",
                "knobs exhausted (grid+dtype); author a tt-lang kernel (GUIDELINES/11) and record it." + _ISOLATE_FIRST,
            )
        if (_tl_clean or not _ttl_available()) and not _cpp_clean and _cpp_wedged < _MAX_TRACE_FIX_RETRIES:
            if _cpp_wedged:
                return (
                    False,
                    "cpp",
                    "HOLD the cpp rung (fix it in isolation, do NOT bounce rungs): %s (trace-fix %d/%d)"
                    % (_TRACE_SAFE_HINT, _cpp_wedged, _MAX_TRACE_FIX_RETRIES),
                )
            return (
                False,
                "cpp",
                "tt-lang measured; author a C++ Metalium kernel via ttnn.generic_op (GUIDELINES/12) and record it." + _ISOLATE_FIRST,
            )
    if _tp_candidate(open_op, op_code) and "tp-fracture" not in kinds:
        return (
            False,
            "tp-fracture",
            "single-chip levers + both kernels exhausted and this dense matmul is still memory-bound on "
            "a mesh -> tp_pick_degree(M,K,N) to MEASURE the fastest TP degree (best_tp=1 means keep it "
            "single-chip). If best_tp>1, fracture the weight across the TP axis + insert the matching CCL "
            "(GUIDELINES/08 §7), verify_tp_fracture(M,K,N,best_tp) to PROVE PCC, then commit; ALWAYS "
            "record_kernel_attempt(...,'tp-fracture',...) even on a no-gain result",
        )
    # BOX (4) STRUCTURAL (ALWAYS ON, GENERAL — no model/architecture knowledge) — the per-op ladder
    # above (grid+dtype+tt-lang+C++) is exhausted but a MATERIAL GAP REMAINS (this op is in the
    # blocking set, so its gap >= the material threshold). That is the universal, config-free signal
    # that the op-level levers PROVABLY can't reach the floor -> the residual bottleneck is STRUCTURAL.
    # The gate does NOT name the fix (no MoE/gather/fusion hardcoding); it REQUIRES the LLM to
    # investigate THIS model's architecture for reducible work and record an attempt. The LLM finds +
    # improvises the restructure (gather / fusion / cache / ... ) for whatever architecture it is; a
    # recorded 'structural' attempt (with evidence) clears the op (measured=tried, bounded).
    _STRUCTURAL = {"structural", "gather", "fusion", "fuse", "sparse", "cache", "kv-cache"}
    if not (kinds & _STRUCTURAL):
        return (
            False,
            "structural",
            "per-op ladder exhausted but a gap remains -> the residual bottleneck is STRUCTURAL. "
            "INVESTIGATE this model's architecture/source for REDUCIBLE WORK (use bound_by as a HINT, "
            "not a rule: memory-bound -> are these bytes reducible? e.g. a sparse/MoE matmul loading "
            "experts that don't fire -> gather; recompute across steps -> cache. dispatch-bound -> fuse "
            "adjacent ops / trace the region). Improvise the restructure for THIS model, measure it, and "
            "record_kernel_attempt(...,'structural', note=<what you found + did>). If there is genuinely no "
            "reducible work, record 'structural' with note='none: <evidence you checked>' to mark it tried.",
        )
    # every box ticked -> genuine irreducible residual -> DONE for this op
    return (True, "done", "checklist complete (grid+dtype+tt-lang+C++ + structural assessment) -> irreducible")


class _Run:
    def __init__(self, d):
        self.profiles_dir = Path(d)
        self.dir = Path(d)


class _Ctx:
    """Minimal LoopContext shim: just enough for measure_runs / run_pcc (proven in kernel_test.py)."""

    def __init__(self):
        self.manifest = _MANIFEST
        _d = tempfile.mkdtemp(prefix="perf_mcp_")
        _TMP_DIRS.add(_d)
        self.run = _Run(_d)
        self.deps = {}

    def model_root(self):
        return _MODEL_ROOT


def _reap_measurement_dir(path) -> bool:
    p = Path(path)
    if not p.name.startswith("perf_mcp_"):
        return False
    if not str(p.resolve()).startswith(str(Path(tempfile.gettempdir()).resolve())):
        return False
    _shutil.rmtree(p, ignore_errors=True)
    return True


_TMP_DIRS = set()


def _reap_tracked_tmp():
    for _d in list(_TMP_DIRS):
        try:
            _reap_measurement_dir(_d)
        except Exception:
            pass


atexit.register(_reap_tracked_tmp)
try:
    signal.signal(signal.SIGTERM, lambda *_a: (_reap_tracked_tmp(), os._exit(143)))
except Exception:
    pass


_STABLE_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "perf_mcp_last_profile"


def _persist_artifacts(prof: dict) -> dict:
    """Copy prof['artifacts'] CSVs out of the about-to-be-reaped tmpdir into one fixed dir
    (overwritten each call) and repoint the paths there, so they stay readable after the
    reap. Best-effort: never raises, so profiling is unaffected if persistence fails."""
    arts = prof.get("artifacts")
    if not isinstance(arts, dict):
        return prof
    try:
        _shutil.rmtree(_STABLE_ARTIFACT_DIR, ignore_errors=True)
        _STABLE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        repointed = {}
        for key, src in arts.items():
            sp = Path(str(src))
            try:
                if sp.is_file():
                    dst = _STABLE_ARTIFACT_DIR / sp.name
                    _shutil.copy2(sp, dst)
                    repointed[key] = str(dst)
                else:
                    repointed[key] = src
            except Exception:
                repointed[key] = src
        prof["artifacts"] = repointed
    except Exception:
        pass
    return prof


def _detect_partial_capture(profiles_dir) -> str | None:
    try:
        d = Path(profiles_dir)
        for sc in sorted(d.glob("*.partial")):
            txt = sc.read_text().strip()
            if txt:
                return txt
        from agent.probes import detect_marker_drop

        for log in sorted(d.glob("*_tracy.log")):
            hit = detect_marker_drop(log.read_text())
            if hit:
                return hit
    except Exception:
        return None
    return None


_PROFILE_CACHE_DIR = Path(tempfile.gettempdir()) / "perf_mcp_profile_cache"


def _model_source_fingerprint() -> str:
    h = hashlib.sha256()
    try:
        root = _MODEL_ROOT
        files = sorted(list((root / "_stubs").glob("*.py")) + list((root / "tt").glob("*.py")))
        if not files:
            return ""
        for f in files:
            h.update(f.name.encode())
            h.update(f.read_bytes())
    except Exception:
        return ""
    return h.hexdigest()


def _profile_cache_get(fp: str):
    if not fp:
        return None
    p = _PROFILE_CACHE_DIR / (fp + ".json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _profile_cache_put(fp: str, prof: dict) -> None:
    if not fp:
        return
    try:
        _PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_PROFILE_CACHE_DIR / (fp + ".json")).write_text(json.dumps(prof))
    except Exception:
        pass


def _profile_once(cq=None) -> dict:
    _cache_on = os.environ.get("PERF_MCP_NO_PROFILE_CACHE") != "1"
    _fp = _model_source_fingerprint() if _cache_on else ""
    if _fp:
        _hit = _profile_cache_get(_fp)
        if _hit is not None:
            return _hit
    ctx = _Ctx()
    tmpdir = ctx.run.dir
    _saved_cq = os.environ.get("TT_PERF_NUM_CQ")
    if cq is not None:
        os.environ["TT_PERF_NUM_CQ"] = str(cq)
    try:
        profiles = measure_runs(ctx)
        prof = profiles[0]
        try:
            prof = roofline.annotate_profile(prof, _ENV)
        except Exception:  # annotation is best-effort; raw profile still usable
            pass
        partial = _detect_partial_capture(ctx.run.profiles_dir)
        if partial:
            prof["capture_partial"] = partial
        prof = _persist_artifacts(prof)
        if _fp and not prof.get("capture_partial"):
            _profile_cache_put(_fp, prof)
        return prof
    finally:
        if cq is not None:
            if _saved_cq is None:
                os.environ.pop("TT_PERF_NUM_CQ", None)
            else:
                os.environ["TT_PERF_NUM_CQ"] = _saved_cq
        _reap_measurement_dir(tmpdir)


def _buckets_view(prof: dict) -> list[dict]:
    rows = []
    for b in prof.get("buckets", []):
        if b.get("id") == "host_overhead":
            continue
        ops = b.get("top_ops") or []
        # bound_by/shape/fidelity/grid live PER-OP; surface them from the bucket's dominant op
        # (largest modeled gap, else largest device_ms) — that's the op a lever must target.
        dom = max(ops, key=lambda o: (o.get("gap_ms") or 0, o.get("device_ms") or 0), default={})
        rows.append(
            {
                "bucket": b.get("id"),
                "device_ms": round(float(b.get("device_ms", 0.0)), 4),
                "count": b.get("count"),
                "gap_ms": b.get("gap_ms"),
                "bound_by": dom.get("bound_by"),
                "dominant_op": {
                    "op_code": dom.get("op_code"),
                    "shape": dom.get("shape"),
                    "fidelity": dom.get("fidelity"),
                    "grid": dom.get("grid"),
                    "memory": dom.get("memory"),
                    "ideal_ms": dom.get("ideal_ms"),
                    "gap_ms": dom.get("gap_ms"),
                },
            }
        )
    return sorted(rows, key=lambda r: -(r.get("gap_ms") or r.get("device_ms") or 0))


@mcp.tool()
def profile_model() -> dict:
    """Profile the model on-device (real tracy run) and return its device_ms, the per-bucket
    breakdown (matmul/datamove/reduction/eltwise) with roofline gap + bound_by tags, and the
    roofline target (the achievable floor). Records this as the baseline for measure_candidate.
    Call this first, and again whenever you want a fresh picture."""
    try:
        prof = _profile_once(cq=2)
    except Exception as exc:  # noqa: BLE001
        _msg = str(exc)
        if _is_dram_trace_overflow(_msg):
            _new = _grow_trace_region()
            _reclaim_mesh("profile_model:dram_trace")
            return {"ok": False, "error": _dram_overflow_msg(_new)}
        if _is_l1_overflow(_msg):
            _reclaim_mesh("profile_model")
            return {"ok": False, "error": _L1_OVERFLOW_MSG}
        return {"ok": False, "error": _msg[-800:]}
    if prof.get("capture_partial"):
        return {
            "ok": False,
            "error": (
                f"partial capture (profiler dropped markers: {prof['capture_partial']}); baseline NOT "
                f"recorded — auto-heal could not get a clean run. Re-profile a smaller/signposted region."
            ),
        }
    _BASELINE_PATH.write_text(json.dumps(prof))
    _orig = _original_baseline_path()
    if not _orig.exists():
        try:
            _orig.write_text(json.dumps(prof))
        except Exception:  # noqa: BLE001
            pass
    dev = round(float(prof.get("device_ms", 0.0)), 4)
    target, at_floor, residual_gap, open_ops = None, None, None, []
    try:
        rep = roofline.residual_report(prof, _ENV)
        target = rep.get("modeled_floor_ms") or rep.get("ideal_ms")
        at_floor = rep.get("at_floor")  # True == nothing ttnn-reachable left (every modeled op at its floor)
        residual_gap = rep.get("residual_gap_ms")  # ms still ABOVE the floor = reachable headroom
        # the ops still on the table (biggest reachable gap first) — these are what's NOT done yet
        open_ops = [
            {"op": o.get("op_code") or o.get("bucket"), "gap_ms": o.get("gap_ms"), "bound_by": o.get("bound_by")}
            for o in (rep.get("open_ops") or [])[:8]
        ]
    except Exception:
        pass
    # OBJECTIVE termination signal: you are NOT done while residual_gap is material and open_ops remain.
    return {
        "ok": True,
        "device_ms": dev,
        "per_token_ms": prof.get("per_token_ms"),
        "tokens_per_sec_per_user": prof.get("tokens_per_sec_per_user"),
        "tokens_per_sec": prof.get("tokens_per_sec"),
        "decode_status": prof.get("decode_status"),
        # repeat_prefill (AR decode, no cached decode_step/KV-cache) -> propose the
        # conditional structural-decode lever; null otherwise so it never fires elsewhere.
        "suggested_lever": ("structural-decode" if prof.get("decode_status") == "repeat_prefill" else None),
        "roofline_target_ms": target,
        "at_floor": at_floor,
        "residual_gap_ms": residual_gap,
        "open_ops": open_ops,
        "buckets": _buckets_view(prof),
    }


@mcp.tool()
def measure_candidate() -> dict:
    """Profile the CURRENT (edited) model and judge it against the recorded baseline. Returns
    device_ms, a delta vs baseline, AND a deterministic integrity verdict: 'valid', or
    'REJECTED' with the reason (crashed/partial capture: op-count collapsed; or inflated capture).
    A REJECTED measurement is NEVER a win no matter how fast it looks — do not keep it. Call this
    after every edit; only a 'valid' result that is faster than baseline is a real gain."""
    try:
        prof = _profile_once(cq=1)
    except Exception as exc:  # noqa: BLE001
        _msg = str(exc)
        if _is_dram_trace_overflow(_msg):
            _new = _grow_trace_region()
            _reclaim_mesh("measure_candidate:dram_trace")
            _autorecord_wedge(_dram_overflow_msg(_new))
            return {"verdict": "REJECTED", "reason": _dram_overflow_msg(_new)}
        if _is_l1_overflow(_msg):
            _reclaim_mesh("measure_candidate")
            _autorecord_wedge(_L1_OVERFLOW_MSG)
            return {"verdict": "REJECTED", "reason": _L1_OVERFLOW_MSG}
        _note_device_crash("measure_candidate")  # may tt-smi reset if this is a repeat (wedge)
        _autorecord_wedge(_trace_compat_feedback(f"wedged/crashed when tried: {_msg[-300:]}"))
        return {"verdict": "REJECTED", "reason": _trace_compat_feedback(f"profiler crashed: {_msg[-600:]}")}
    _note_device_ok()
    dev = round(float(prof.get("device_ms", 0.0)), 4)
    if prof.get("capture_partial"):
        return {
            "verdict": "REJECTED",
            "reason": _trace_compat_feedback(f"partial_capture: profiler dropped markers ({prof['capture_partial']})"),
            "device_ms": dev,
        }
    if not _BASELINE_PATH.exists():
        return {"verdict": "valid", "device_ms": dev, "note": "no baseline recorded; call profile_model first"}
    baseline = json.loads(_BASELINE_PATH.read_text())
    base_dev = round(float(baseline.get("device_ms", 0.0)), 4)
    # DETERMINISTIC integrity guard (the exact check REMEASURE uses) — GENERALIZED to physics: pass
    # the model's roofline floor so a below-floor (impossible) reading is rejected as a crashed
    # capture, while a legitimate op-reducing fusion ABOVE the floor is accepted (no op-count veto).
    floor_ms = None
    try:
        floor_ms = roofline.residual_report(baseline, _ENV).get("modeled_floor_ms")
    except Exception:
        pass
    ok, reason = _rm._comparable(baseline, prof, floor_ms=floor_ms, tp_regime=_TP_REGIME)
    if not ok:
        return {"verdict": "REJECTED", "reason": reason, "device_ms": dev, "baseline_ms": base_dev}
    delta = round(base_dev - dev, 4)
    pct = round((delta / base_dev) * 100.0, 2) if base_dev else 0.0
    faster = delta > 0.05  # noise floor
    pt_ms = prof.get("per_token_ms")
    base_pt = baseline.get("per_token_ms")
    return {
        "verdict": "valid",
        "device_ms": dev,
        "baseline_ms": base_dev,
        "delta_ms": delta,
        "pct_faster": pct,
        "is_real_gain": faster,
        "per_token_ms": pt_ms,
        "baseline_per_token_ms": base_pt,
        "per_token_delta_ms": round(base_pt - pt_ms, 6) if (pt_ms and base_pt) else None,
        "tokens_per_sec_per_user": prof.get("tokens_per_sec_per_user"),
        "tokens_per_sec": prof.get("tokens_per_sec"),
        "note": "FASTER — real gain" if faster else ("SLOWER" if delta < -0.05 else "no gain (within noise)"),
    }


@mcp.tool()
def check_pcc() -> dict:
    """Run the model's end-to-end PCC correctness test on-device (the SAME gate the FSM uses).
    Returns {status: ok|pcc_low|crash, pcc?}. An edit is only acceptable if status==ok. A crash
    or pcc_low means the edit broke correctness — fix or revert it; never keep it."""
    try:
        res = run_pcc(_Ctx())
    except Exception as exc:  # noqa: BLE001
        _note_device_crash("check_pcc")
        return {"status": "crash", "error": str(exc)[-800:]}
    if res.get("status") == "crash":
        _note_device_crash("check_pcc")
    else:
        _note_device_ok()
    return res


def _pg_cpu_jiffies(pgid):
    total = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    target = str(pgid)
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry) as fh:
                data = fh.read()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        rp = data.rfind(")")
        if rp == -1:
            continue
        fields = data[rp + 2 :].split()
        if len(fields) > 12 and fields[2] == target:
            try:
                total += int(fields[11]) + int(fields[12])
            except ValueError:
                pass
    return total


class _AdaptiveResult:
    def __init__(self, returncode, stdout):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def _adaptive_run(cmd, cwd, env, label="device run", stall_s=None, backstop=None):
    import threading as _th
    import time as _t

    stall_s = int(stall_s if stall_s is not None else os.environ.get("PERF_MCP_MEASURE_STALL_SEC", "600") or "600")
    if backstop is None:
        from agent.probes import adaptive_backstop as _abs

        backstop = _abs(3600)
    else:
        backstop = int(backstop)
    proc = _sp.Popen(
        list(cmd), cwd=str(cwd), env=env, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, start_new_session=True
    )
    buf = []
    act = [_t.monotonic()]

    def _pump():
        try:
            for ln in proc.stdout:
                buf.append(ln)
                act[0] = _t.monotonic()
        except Exception:  # noqa: BLE001
            pass

    pt = _th.Thread(target=_pump, daemon=True)
    pt.start()
    pgid = proc.pid
    start = _t.monotonic()
    last_progress = start
    last_cpu = _pg_cpu_jiffies(pgid)
    max_gap = 0.0
    while proc.poll() is None:
        _t.sleep(5)
        now = _t.monotonic()
        cpu = _pg_cpu_jiffies(pgid)
        moved = cpu > last_cpu + 10 or act[0] > last_progress
        last_cpu = cpu
        if moved:
            max_gap = max(max_gap, now - last_progress)
            last_progress = now
        limit = max(stall_s, int(3 * max_gap))
        if now - last_progress >= limit or now - start >= backstop:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                proc.kill()
            try:
                proc.communicate(timeout=30)
            except Exception:  # noqa: BLE001
                pass
            raise _sp.TimeoutExpired(cmd, limit if now - last_progress >= limit else backstop)
    rc = proc.returncode
    pt.join(timeout=30)
    return _AdaptiveResult(rc, "".join(buf))


def _run_full_pipeline_ms():
    ptr = _MANIFEST.get("perf_test_resolved", {}) or {}
    node = ptr.get("path")
    if not node:
        return None, None, "no perf test in manifest"
    case = ptr.get("case")
    repo = str(Path(_PKG).parent.parent.parent)
    env = dict(os.environ)
    env["TT_METAL_HOME"] = repo
    env["PYTHONPATH"] = repo
    env["TT_PERF_LAYERS"] = "0"
    env["TT_PERF_MAX_NEW_TOKENS"] = os.environ.get("PERF_MCP_FULLPIPE_TOKENS", "1")
    env.setdefault("TT_PERF_TRACE", "1")
    env["TT_PERF_PREFILL_TRACE"] = "1"
    _prof = os.environ.get("PERF_MCP_PROFILE_ENV")
    if _prof:
        try:
            env.update(json.loads(_prof))
        except (ValueError, TypeError):
            pass
    _cq = os.environ.get("PERF_MCP_FULLPIPE_CQ")
    if _cq and _cq.isdigit():
        env["TT_PERF_NUM_CQ"] = _cq
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    cmd = [sys.executable, "-m", "pytest", "-o", "timeout=0", "-s", node]
    if case:
        cmd += ["-k", case]
    per_tokens = []
    walls = []
    prefills = []
    tp = dp = 1
    shard = False
    batch = 1
    decode_path = prefill_path = "n/a"
    last_err = None
    for _ in range(_FULLPIPE_SAMPLES):
        try:
            r = _adaptive_run(cmd, repo, env, "full-pipeline")
        except Exception as exc:  # noqa: BLE001
            last_err = f"run failed: {str(exc)[-400:]}"
            continue
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        for line in out.splitlines():
            if "TRACE_PER_TOKEN_MS=" in line:
                try:
                    per_tokens.append(float(line.split("TRACE_PER_TOKEN_MS=", 1)[1].split()[0]))
                except Exception:  # noqa: BLE001
                    pass
            elif "TRACE_PREFILL_MS=" in line:
                try:
                    prefills.append(float(line.split("TRACE_PREFILL_MS=", 1)[1].split()[0]))
                except Exception:  # noqa: BLE001
                    pass
            elif "FORWARD_WALL_MS=" in line:
                try:
                    walls.append(float(line.split("FORWARD_WALL_MS=", 1)[1].split()[0]))
                except Exception:  # noqa: BLE001
                    pass
            m = _re.search(r"DP=(\d+)\s+TP=(\d+)", line)
            if m:
                dp, tp = int(m.group(1)), int(m.group(2))
            if "shard_active=True" in line:
                shard = True
            if "TRACE_REPLAY_PATH=" in line:
                mb = _re.search(r"batch=(\d+)", line)
                if mb:
                    batch = int(mb.group(1))
                try:
                    decode_path = line.split("TRACE_REPLAY_PATH=", 1)[1].split()[0]
                except Exception:  # noqa: BLE001
                    pass
            if "TRACE_PREFILL_PATH=" in line:
                try:
                    prefill_path = line.split("TRACE_PREFILL_PATH=", 1)[1].split()[0]
                except Exception:  # noqa: BLE001
                    pass
    dec = statistics.median(per_tokens) if per_tokens else None
    pf = statistics.median(prefills) if prefills else None
    if dec is not None or pf is not None:
        isl = env.get("TT_PERF_SEQ_LEN", os.environ.get("TT_PERF_SEQ_LEN", "128"))
        osl = env.get("TT_PERF_MAX_NEW_TOKENS", "1")
        tsu = (1000.0 / dec) if dec else 0.0
        sys.stderr.write(
            "[full-pipeline-gate] PERF_SCORECARD mesh=%dx%d TP=%d DP=%d shard=%s on_device=%s "
            "ISL=%s OSL=%s batch=%d TTFT_ms=%s prefill_path=%s decode_ms=%s decode_path=%s TSU=%.2f TS=%.2f\n"
            % (
                dp,
                tp,
                tp,
                dp,
                shard,
                (dec is not None or pf is not None),
                isl,
                osl,
                batch,
                ("%.2f" % pf) if pf is not None else "NA",
                prefill_path,
                ("%.4f" % dec) if dec is not None else "NA",
                decode_path,
                tsu,
                tsu * batch,
            )
        )
        sys.stderr.flush()
    if per_tokens:
        return statistics.median(per_tokens), "trace", None, decode_path
    if walls:
        return statistics.median(walls), "eager", None, None
    if last_err:
        return None, None, last_err, None
    return None, None, "no TRACE_PER_TOKEN_MS or FORWARD_WALL_MS in output (workload did not run full-pipeline)", None


_FULLPIPE_GATE_LOG = Path(tempfile.gettempdir()) / "perf_mcp_fullpipe_gate.log"


def _emit_fullpipe(result: dict) -> dict:
    m = result.get("method")
    src = "trace_replay" if m == "trace" else ("eager_wall" if m == "eager" else "n/a")
    parts = [
        "[full-pipeline-gate]",
        "status=%s" % result.get("status"),
        "end_to_end_ms=%s" % result.get("full_pipeline_ms"),
        "via=%s" % src,
    ]
    if result.get("best_ms") is not None:
        parts.append("best_ms=%s" % result.get("best_ms"))
    if result.get("delta_pct") is not None:
        parts.append("delta_pct=%s" % result.get("delta_pct"))
    if result.get("target_ms") is not None:
        parts.append("target_ms=%s gap_ms=%s" % (result.get("target_ms"), result.get("gap_to_target_ms")))
    if result.get("error"):
        parts.append("error=%s" % str(result.get("error"))[:140])
    line = " ".join(parts)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
    try:
        with open(_FULLPIPE_GATE_LOG, "a") as _f:
            _f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass
    return result


_FULLPIPE_MODE_RANK = {"eager": 0, "trace": 1, "trace+1cq": 1, "trace+2cq": 2}


def _fullpipe_mode(method: str, path: str | None) -> str:
    if method != "trace":
        return "eager"
    p = (path or "").strip()
    return p if p in ("trace+2cq", "trace+1cq") else "trace"


def _mode_rank(mode: str) -> int:
    return _FULLPIPE_MODE_RANK.get(mode, 1)


_SIGNPOST_PREFIX = "PERF_BLOCK_SIGNPOST:"


def _infer_block_count(counts: dict) -> int:
    vals = [c for c in counts.values() if c > 1]
    if not vals:
        return 1
    from collections import Counter as _C

    return _C(vals).most_common(1)[0][0]


def _signpost_blocks(seq: list) -> int:
    m = -1
    for s in seq or []:
        if isinstance(s, str) and s.startswith(_SIGNPOST_PREFIX):
            try:
                m = max(m, int(s.split(":", 1)[1]))
            except (ValueError, IndexError):
                pass
    return m + 1


def _signposts_agree(seq: list) -> bool:
    sp = _signpost_blocks(seq)
    op = _infer_block_count(
        _C_counts([s for s in seq or [] if not (isinstance(s, str) and s.startswith(_SIGNPOST_PREFIX))])
    )
    if sp <= 1 or op <= 1:
        return False
    lo, hi = sorted((sp, op))
    return lo / hi >= float(os.environ.get("PERF_MCP_SIGNPOST_AGREE_RATIO", "0.8"))


def _block_starts(sequence: list, n_blocks: int | None = None) -> tuple:
    seq = sequence or []
    sp = [i for i, s in enumerate(seq) if isinstance(s, str) and s.startswith(_SIGNPOST_PREFIX)]
    if sp and _signposts_agree(seq):
        return sp, "signposts"
    if n_blocks is None:
        n_blocks = _infer_block_count(_C_counts(seq))
    from collections import Counter as _C

    c = _C(seq)
    anchor = next((s for s in seq if c.get(s) == n_blocks), None)
    if anchor is None:
        return [], "none"
    return [i for i, s in enumerate(seq) if s == anchor], "inferred"


def _C_counts(seq):
    from collections import Counter as _C

    return dict(_C(seq))


def _block_of(pos: int, starts: list) -> int:
    import bisect

    return bisect.bisect_right(starts, pos) - 1


def compute_lever_coverage(
    counts: dict, sequence: list, op_match: str, stale_dtype: str = "", new_dtype: str = ""
) -> dict:
    matching = {s: n for s, n in (counts or {}).items() if op_match and op_match in s}
    if not matching:
        return {"status": "not_found", "note": "no op signature matched '%s' at full depth" % op_match}
    total = sum(matching.values())
    stale = sum(n for s, n in matching.items() if stale_dtype and stale_dtype in s)
    fresh = sum(n for s, n in matching.items() if new_dtype and new_dtype in s)
    starts, block_source = _block_starts(sequence or [], _infer_block_count(counts or {}))
    n_blocks = len(starts) if starts else _infer_block_count(counts or {})
    missed_blocks = []
    if stale_dtype:
        stale_sigs = {s for s in matching if stale_dtype in s}
        seen = set()
        for i, s in enumerate(sequence or []):
            if s in stale_sigs:
                b = _block_of(i, starts)
                if b >= 0 and b not in seen:
                    seen.add(b)
                    missed_blocks.append(b)
    fully = (stale == 0 and fresh > 0) if (stale_dtype and new_dtype) else None
    if fully:
        note = "lever reached ALL %d instances of this op" % total
    elif fully is False:
        note = (
            "PARTIAL: %d of %d instances still carry the OLD signature (blocks %s) — the edit is on an "
            "instance-specific path; move it to the SHARED block definition and reapply so every layer changes"
            % (stale, total, sorted(missed_blocks))
        )
    else:
        note = "signature-visible check only: pass stale_dtype+new_dtype for a dtype lever; grid/program_config levers are not tensor-visible and rely on shared-definition propagation"
    return {
        "status": "ok",
        "op_match": op_match,
        "total_instances": total,
        "applied": fresh if new_dtype else None,
        "stale_remaining": stale if stale_dtype else None,
        "fully_applied": fully,
        "n_blocks": n_blocks,
        "block_source": block_source,
        "missed_blocks": sorted(missed_blocks),
        "note": note,
    }


def _full_depth_op_probe():
    ptr = _MANIFEST.get("perf_test_resolved", {}) or {}
    node = ptr.get("path")
    if not node:
        return None, None
    case = ptr.get("case")
    repo = str(Path(_PKG).parent.parent.parent)
    env = dict(os.environ)
    env["TT_METAL_HOME"] = repo
    env["PYTHONPATH"] = repo
    env["TT_PERF_LAYERS"] = "0"
    env["TT_PERF_MAX_NEW_TOKENS"] = "1"
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    cmd = [sys.executable, str(Path(__file__).parent / "_op_sig_probe.py"), node]
    if case:
        cmd.append(case)
    try:
        r = _adaptive_run(cmd, repo, env, "op-sig probe")
    except Exception as exc:  # noqa: BLE001
        return None, "probe failed: %s" % str(exc)[-300:]
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    counts, seq = {}, []
    for line in out.splitlines():
        if line.startswith("PERF_OP_SIG_COUNTS="):
            try:
                counts = json.loads(line.split("=", 1)[1])
            except Exception:  # noqa: BLE001
                counts = {}
        elif line.startswith("PERF_OP_SIG_SEQUENCE="):
            try:
                seq = json.loads(line.split("=", 1)[1])
            except Exception:  # noqa: BLE001
                seq = []
    return counts, seq


@mcp.tool()
def check_lever_coverage(op_match: str, stale_dtype: str = "", new_dtype: str = "") -> dict:
    """After applying a lever (dtype knob / kernel swap) to an op, VERIFY it reached EVERY layer instance,
    not just the profiled representative slice. Runs an ALL-LAYERS op-signature probe (TT_PERF_LAYERS=0,
    NO tracy -> no marker buffer -> overflow-safe) and checks whether the op still appears with its OLD
    signature anywhere. op_match: a substring identifying the op (name + a shape dim, e.g. 'linear(1, 4096').
    stale_dtype/new_dtype: the OLD and NEW dtype markers the lever changed (e.g. 'BFLOAT16','BFLOAT8_B') —
    supply both for a dtype lever so coverage is exact. Returns fully_applied + missed_blocks: if PARTIAL,
    the edit is on an instance-specific path (e.g. layers[0]) — move it to the SHARED block definition and
    reapply so all N layers change. A repeated block is ONE class instantiated N times, so a lever on the
    shared definition propagates to every instance; this catches the case where it did not."""
    counts, seq = _full_depth_op_probe()
    if not counts:
        return {
            "status": "skip",
            "note": "all-layers op-signature probe produced no counts (%s)" % (seq or "no output"),
        }
    return compute_lever_coverage(counts, seq, op_match, stale_dtype, new_dtype)


@mcp.tool()
def check_full_pipeline_latency() -> dict:
    """Measure end-to-end latency and gate it as a CONVERGENCE gate toward the target (a GPU number if
    set via PERF_MCP_TARGET_MS, else just best-so-far). Measurement source is trace_replay when the
    pipeline exposes a trace-capturable decode step: method 'trace' reports the clean, GPU-comparable
    per-token wall (TRACE_PER_TOKEN_MS via agent/trace_replay); otherwise it falls back to method 'eager'
    (the whole-model FORWARD_WALL_MS, layer cap OFF, no tracy). The best-so-far baseline is keyed by
    method, so a switch from eager to trace (e.g. once a decode_step is added) re-baselines instead of
    cross-comparing incomparable numbers. This is NOT a fixed-threshold gate: a kept edit only has to
    move TOWARD the target (faster / not slower), and is NEVER rejected for failing to REACH the target.
    status 'ok' = moved toward target or held (accept); status 'diverged' = got slower than best-so-far
    by more than the tolerance (reject — revert it). E.g. target=1ms: 10->8 is ok, 10->12 is diverged; 8
    is accepted even though it is not 1. Best-so-far ratchets down on every improvement. Run alongside
    check_pcc before banking any win. Each check prints a `[full-pipeline-gate]` line (status,
    end_to_end_ms, via=trace_replay|eager_wall, best/delta/target) to stderr and appends it to
    $TMPDIR/perf_mcp_fullpipe_gate.log so the gated end-to-end time is visible every iteration.
    Returns {status, full_pipeline_ms, method, metric, best_ms?, delta_pct?, target_ms?,
    gap_to_target_ms?, reached_target?}."""
    _cq_env = os.environ.get("PERF_MCP_FULLPIPE_CQ")
    cq = int(_cq_env) if (_cq_env and _cq_env.isdigit()) else 1
    ms, method, err, path = _run_full_pipeline_ms()
    if ms is None:
        return _emit_fullpipe({"status": "crash", "error": err, "cq": cq})
    metric = "trace_per_token_ms" if method == "trace" else "eager_full_pipeline_ms"
    mode = _fullpipe_mode(method, path)
    base_path = _FULLPIPE_BASELINE_PATH if cq >= 2 else _FULLPIPE_BASELINE_1CQ_PATH
    cq_note = (
        "trace+1cq (robust per-iteration signal): validate/bank compute-op wins here — it always engages "
        "(no 2-CQ reservation, so no OOM/downgrade). The trace+2cq production number is confirmed at the "
        "start/end bookend; only 2-CQ-overlap / L1-headroom levers require that bookend to judge."
        if cq < 2
        else "trace+2cq (production ship metric): start/end bookend anchor."
    )
    tgt = _FULLPIPE_TARGET_MS if _FULLPIPE_TARGET_MS > 0 else None
    tgt_fields = {}
    if tgt is not None:
        tgt_fields = {
            "target_ms": round(tgt, 4),
            "gap_to_target_ms": round(ms - tgt, 4),
            "reached_target": ms <= tgt,
        }
    base = {}
    if base_path.exists():
        try:
            base = json.loads(base_path.read_text())
        except Exception:  # noqa: BLE001
            base = {}
    best = float(base.get("full_pipeline_ms", 0.0) or 0.0)
    base_mode = base.get("mode") or _fullpipe_mode(base.get("method", "eager"), None)
    if best > 0 and _mode_rank(mode) < _mode_rank(base_mode):
        return _emit_fullpipe(
            {
                "status": "degraded",
                "full_pipeline_ms": round(ms, 4),
                "method": method,
                "metric": metric,
                "mode": mode,
                "cq": cq,
                "baseline_mode": base_mode,
                "error": (
                    "trace fidelity degraded %s -> %s in the %d-CQ track; delta NOT banked, baseline NOT "
                    "downgraded — the workload fell back below the expected trace mode (fix or revert)"
                    % (base_mode, mode, cq)
                ),
                **tgt_fields,
            }
        )
    if base_mode != mode or best <= 0:
        base_path.write_text(json.dumps({"full_pipeline_ms": ms, "method": method, "mode": mode}))
        return _emit_fullpipe(
            {
                "status": "ok",
                "full_pipeline_ms": round(ms, 4),
                "method": method,
                "metric": metric,
                "mode": mode,
                "cq": cq,
                "note": "best-so-far recorded · " + cq_note,
                **tgt_fields,
            }
        )
    delta_pct = round((ms - best) / best * 100.0, 2) if best > 0 else None
    diverged = ms > best * (1.0 + _FULLPIPE_TOL)
    if ms < best:
        base_path.write_text(json.dumps({"full_pipeline_ms": ms, "method": method, "mode": mode}))
    return _emit_fullpipe(
        {
            "status": "diverged" if diverged else "ok",
            "full_pipeline_ms": round(ms, 4),
            "best_ms": round(best, 4),
            "delta_pct": delta_pct,
            "method": method,
            "metric": metric,
            "mode": mode,
            "cq": cq,
            "note": cq_note,
            **tgt_fields,
        }
    )


_HITL_STEP = {"n": 0}


@mcp.tool()
def hitl_gate(
    tried_op: str,
    tried_lever: str,
    why_tried: str,
    is_win: bool,
    why_not: str = "",
    next_target: str = "",
    next_why: str = "",
    before_ms: float = 0.0,
    after_ms: float = 0.0,
    stages_json: str = "",
) -> dict:
    """HUMAN-IN-THE-LOOP gate (--hitl only). After applying ONE lever and measuring it, call this
    INSTEAD of git_commit/git_revert. It shows the operator a block-level timing + rationale pause
    screen and returns their decision {action: 'commit'|'revert'|'try', note, knob}: on 'commit'/'revert'
    the orchestrator performs the git action for you; on 'try' apply the operator's `knob` next. Pass
    what you tried + why, the win flag + why_not, the next planned target + why, the before/after
    full-pipeline ms, and stages_json = the per-stage trace timings you just measured as a JSON list of
    {"name","ms"} (and optional "dominant"). Blocks until the operator answers."""
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("cc_hitl", str(Path(__file__).parent / "hitl.py"))
    hitl = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(hitl)
    hdir = os.environ.get("PERF_MCP_HITL_DIR")
    if not hdir:
        return {"action": "commit", "note": "hitl not wired (no PERF_MCP_HITL_DIR) — proceeding without gate"}
    try:
        stages = json.loads(stages_json) if stages_json else []
    except ValueError:
        stages = []
    _HITL_STEP["n"] += 1
    proposal = {
        "model": Path(_MODEL_ROOT).name,
        "step": _HITL_STEP["n"],
        "stages": stages,
        "tried": {"op": tried_op, "lever": tried_lever, "why": why_tried},
        "result": {
            "win": bool(is_win),
            "before_ms": before_ms or None,
            "after_ms": after_ms or None,
            "why_not": why_not,
        },
        "next": {"target": next_target, "why": next_why},
    }
    hitl.post_proposal(hdir, proposal)
    _to = float(os.environ.get("PERF_MCP_HITL_TIMEOUT", "0") or "0") or None
    return hitl.await_decision(hdir, timeout=_to)


@mcp.tool()
def git_head() -> dict:
    """Return the current git HEAD sha of the model repo (your clean checkpoint / revert target)."""
    repo = gitio.repo_root(_MODEL_ROOT)
    return {"sha": gitio.head_sha(repo)}


@mcp.tool()
def git_commit(message: str) -> dict:
    """Commit the current model-dir changes (scoped to the model dir only — unrelated repo changes
    are left untouched). Use this to BANK a verified win: valid measure + ok pcc (check_pcc) + faster
    + full-pipeline NOT regressed (check_full_pipeline_latency status == 'ok'). If check_pcc OR
    check_full_pipeline_latency is not ok, revert — never commit. Returns the new sha."""
    repo = gitio.repo_root(_MODEL_ROOT)
    try:
        pathspec = _MODEL_ROOT.relative_to(repo)
    except ValueError:
        pathspec = None
    sha = gitio.commit(repo, message, pathspec)
    return {"committed": bool(sha), "sha": sha}


@mcp.tool()
def git_revert(sha: str) -> dict:
    """Revert the model dir to a given sha (scoped checkout — unrelated repo changes untouched).
    Use this to discard a rejected/slower/incorrect edit and return to the clean checkpoint."""
    repo = gitio.repo_root(_MODEL_ROOT)
    try:
        pathspec = _MODEL_ROOT.relative_to(repo)
    except ValueError:
        pathspec = None
    gitio.checkout(repo, sha, pathspec)
    return {"reverted_to": sha}


def _capture_attempt_diff(max_lines: int = 40) -> str:
    """Best-effort snapshot of the SOURCE change THIS attempt made, for the RUN_REPORT code-change log.
    A rejected (no-gain) attempt is still in the working tree at record time (revert happens AFTER) ->
    `git diff HEAD`; a banked win is already committed -> `git show HEAD`. Scoped to the model dir,
    truncated to max_lines. Never raises."""
    try:
        repo = gitio.repo_root(_MODEL_ROOT)
        try:
            pathspec = str(_MODEL_ROOT.relative_to(repo))
        except ValueError:
            pathspec = "."
        # Exclude tool-generated artifacts so the captured "code change" is the
        # ACTUAL source edit — not the report quoting itself (RUN_REPORT.md lives
        # in the model dir, so without this the diff embeds its own live-update
        # churn -> the optimize table nests/duplicates) nor scaffold/state files.
        _excludes = [
            ":(exclude,glob)**/RUN_REPORT.md",
            ":(exclude,glob)**/bringup_status.json",
            ":(exclude,glob)**/.bringup_cc_state.json",
            ":(exclude,glob)**/*.opplan.json",
        ]
        out = ""
        for cmd in (
            ["git", "-C", str(repo), "diff", "HEAD", "--", pathspec, *_excludes],
            ["git", "-C", str(repo), "show", "--format=", "HEAD", "--", pathspec, *_excludes],
        ):
            r = _sp.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                out = r.stdout
                break
        if not out.strip():
            return ""
        all_lines = out.splitlines()
        if len(all_lines) > max_lines:
            all_lines = all_lines[:max_lines] + [f"... (truncated, {len(out.splitlines()) - max_lines} more lines)"]
        return "\n".join(all_lines)
    except Exception:  # noqa: BLE001
        return ""


@mcp.tool()
def record_kernel_attempt(
    op_signature: str, kernel_kind: str, measured_ms: float, beat_baseline: bool, note: str = "", stages_json: str = ""
) -> dict:
    """Record that you AUTHORED and MEASURED a real custom kernel for an open op (tt-lang or C++).
    REQUIRED before termination_check() will let you stop on any op with material roofline gap — a
    measured attempt is the EMPIRICAL validation that REPLACES 'I reasoned a kernel won't help'.
    Even a measured kernel that does NOT beat ttnn clears the op as 'tried' (that's the proof it
    can't be improved, not an assertion). op_signature: enough of the profiler op_code to match it
    (e.g. 'MatmulDeviceOperation 32 x 32 x 32' or 'LayerNorm'). kernel_kind: 'tt-lang' | 'cpp'.
    measured_ms: the device_ms measure_candidate reported with the kernel in place. This verifies a
    kernel actually exists in the model source — a record with no kernel present is flagged and will
    NOT clear the op. stages_json: OPTIONAL per-stage trace timings for this lever (the same
    JSON list of {"name","ms","dominant?"} you would pass hitl_gate) — rendered as the block-level
    timing table in RUN_REPORT.md so BOTH hitl and non-hitl runs surface where device time went."""
    # CONFIG/STRUCTURAL kinds — changes with NO custom-kernel source marker: program-config knobs
    # (grid/dtype), trace/2-CQ host-loop transforms, and dataflow restructures (gather/fusion/cache).
    # Accepted as detected; the re-profile + PCC verify the real effect. Only true kernel kinds
    # (tt-lang/cpp) require a generic_op/ttl marker in the model source.
    _KNOB_KINDS = {
        "grid",
        "dtype",
        "knob",
        "fusion",
        "fuse",
        "structural",
        "gather",
        "sparse",
        "cache",
        "kv-cache",
        "trace",
        "2cq",
    }
    is_knob = (kernel_kind or "").lower() in _KNOB_KINDS
    is_tp = (kernel_kind or "").lower() == "tp-fracture"
    ev = _scan_kernel_evidence()
    if is_tp:
        detected = bool(ev.get("tp_shard") and ev.get("ccl"))
    else:
        detected = True if is_knob else bool(ev["markers"] or ev["cpp_files"])
    ttl_absent = (kernel_kind or "").lower() == "tt-lang" and not _ttl_available()
    if ttl_absent:
        detected = False
    try:
        stages = json.loads(stages_json) if stages_json else []
        stages = [s for s in stages if isinstance(s, dict)] if isinstance(stages, list) else []
    except Exception:  # noqa: BLE001
        stages = []
    rec = {
        "op_signature": op_signature,
        "kernel_kind": kernel_kind,
        "measured_ms": round(float(measured_ms), 4),
        "beat_baseline": bool(beat_baseline),
        "note": note,
        "stages": stages,
        "kernel_detected_in_source": detected,
        "wedged": False,
        "evidence": ev,
        "diff": _capture_attempt_diff(),
    }
    _append_attempt(rec)
    return {
        "recorded": True,
        "attempt": rec,
        "warning": (
            "tt-lang toolchain (ttl) not installed — kernel cannot run or be measured; attempt NOT credited."
            if ttl_absent
            else (
                None
                if detected
                else (
                    "TP attempt needs BOTH a ShardTensorToMesh AND a CCL (all_gather/reduce_scatter) in model "
                    "source — not found; attempt UNSUPPORTED and will NOT clear the op."
                    if is_tp
                    else "NO kernel markers (generic_op/@ttl/.cpp/ProgramDescriptor) found in model source — this "
                    "attempt is UNSUPPORTED and will NOT clear the op in termination_check. Author a real kernel first."
                )
            )
        ),
    }


@mcp.tool()
def _trace_budget_facts():
    if os.environ.get("TT_PERF_TRACE", "1") != "1":
        return None
    try:
        ncq = int(os.environ.get("TT_PERF_NUM_CQ", "2") or "2")
    except ValueError:
        ncq = 2
    if ncq < 2:
        return None
    try:
        tr = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872") or "23887872")
    except ValueError:
        tr = 23887872
    return {
        "num_command_queues": ncq,
        "trace_region_size": tr,
        "note": (
            "2-CQ + trace region are reserved at device-open. Size L1 shards/grids to leave headroom so "
            "the candidate fits 2 command queues; filling all of L1 OOMs under 2 CQs and forces a "
            "trace+1cq fallback — a fidelity downgrade check_full_pipeline_latency flags (delta not banked)."
        ),
    }


def recall_knobs(op_class: str, grid: str = "", bound_by: str = "") -> dict:
    """REUSE-FIRST: return the tested/known knobs already catalogued for this op_class, so you
    APPLY/ADAPT a proven one BEFORE improvising from scratch. Routed deterministically from the
    GUIDELINES catalog (numbered guides + LEARNED_*/GRADUATED_* learned levers) by op_class. CALL
    THIS for next_target.op_class before every rung's edit.

    ADVISORY ONLY — a recalled knob SEEDS the attempt (tested block/shard/fidelity values, and the
    NEGATIVE knowledge of what NOT to do, e.g. 'don't bf16 Q/K/V', 'packer_l1_acc must be True'); it
    NEVER lets you skip a rung or stop early. You must still check_pcc + measure_candidate +
    record_kernel_attempt for the rung exactly as termination_check requires. If nothing matches,
    improvise from principles (cc does not yet auto-distill the win back into the catalog — that
    write-back currently happens only on the FSM path).

    op_class: one of matmul|attention|reduction|eltwise|datamove|embedding|conv_pool|ccl|
    host_fallback|other (pass next_target.op_class). grid + bound_by (pass next_target.grid +
    next_target.bound_by) NARROW the result to the knobs relevant to THIS op — broad guidance that
    declares a mismatching grid/bound is pruned, tuned LEARNED_/GRADUATED_ levers (wildcard on those
    dims) are kept and ranked FIRST. Falls back to the op_class-wide set if narrowing would starve.
    Returns {op_class, narrowed_by, known_knobs:[{id,title,source,status,lever_type,text}], count};
    known_knobs is [] on no match OR any lookup failure (this tool can never block the loop)."""
    oc = (op_class or "").strip()
    if not oc:
        return {"op_class": op_class, "known_knobs": [], "count": 0}
    _BOUND_MAP = {
        "memory": "dram",
        "dram": "dram",
        "bandwidth": "dram",
        "compute": "flop",
        "flop": "flop",
        "both": "both",
        "dispatch": "slow",
        "slow": "slow",
        "host": "host",
        "host_fallback": "host",
    }
    _GRID_VOCAB = {"full", "partial", "tiny"}
    try:
        gdir = str(_PKG / "GUIDELINES")
        index = router.build_index(gdir)
        q = {"op_class": oc}
        g = (grid or "").strip().lower()
        if g in _GRID_VOCAB:
            q["grid"] = g
        b = _BOUND_MAP.get((bound_by or "").strip().lower())
        if b:
            q["bound"] = b
        try:
            hits = router.route(index, q)  # raises on out-of-vocab -> fall back below
        except Exception:  # noqa: BLE001
            hits = []
        if not hits and len(q) > 1:  # narrowing starved -> never return empty wrongly; op_class-only
            hits = router.route(index, {"op_class": oc})
        # tuned learned levers first (most specific to this op), then baseline guidance
        rank = {"GRADUATED_": 0, "LEARNED_": 1}
        hits = sorted(hits, key=lambda h: next((v for k, v in rank.items() if (h.get("file") or "").startswith(k)), 2))
        out = []
        for h in hits:
            fname = h.get("file", "") or ""
            status = (
                "trusted"
                if fname.startswith("GRADUATED_")
                else "provisional"
                if fname.startswith("LEARNED_")
                else "baseline-guideline"
            )
            try:
                text = router.read_section(h["id"], gdir)
            except Exception:  # noqa: BLE001
                text = ""
            out.append(
                {
                    "id": h.get("id"),
                    "title": h.get("title"),
                    "source": fname,
                    "status": status,
                    "lever_type": h.get("lever_type"),
                    "text": text,
                }
            )
        return {
            "op_class": oc,
            "narrowed_by": {k: v for k, v in q.items() if k != "op_class"},
            "known_knobs": out,
            "count": len(out),
            "budget": _trace_budget_facts(),
        }
    except Exception as exc:  # noqa: BLE001 — advisory tool: never raise into the loop
        return {"op_class": op_class, "known_knobs": [], "count": 0, "error": str(exc)[-200:]}


@mcp.tool()
def distill_knob(
    op_class: str, title: str, fires_when: str, recipe: str, reused_lever_id: str = "", bucket: str = ""
) -> dict:
    """WRITE-BACK (closes the learn loop): persist a verified IMPROVISED win into the catalog as a
    reusable provisional learned lever (LEARNED_<bucket>-coherence-<model>.md), so future runs/models
    REUSE it via recall_knobs instead of re-deriving it. Reuses the FSM promote.write_provisional_lever
    so the format + graduation path are IDENTICAL to the FSM's. Call this AFTER you COMMITTED a win
    that you IMPROVISED (recall_knobs returned no match). Write the GENERAL technique, not this model's
    code, so it transfers.

    GRADUATION (cross-model validation): if THIS win re-used a PROVISIONAL lever that was learned on a
    DIFFERENT model, pass its anchor as reused_lever_id and it graduates to trusted (renamed
    GRADUATED_*). Safe to pass whenever you reused a provisional knob and won — it only graduates when
    that lever's learned_on != this model.

    op_class: matmul|attention|reduction|eltwise|datamove|embedding|conv_pool|ccl|host_fallback|other.
    title: short technique name. fires_when: ONE general sentence (the bottleneck signature it targets).
    recipe: 2-6 lines, the general TTNN technique abstracted from your edit (not model-specific code).
    bucket: profile bucket id (defaults to op_class). Returns {written, graduated, error?} — never
    raises into the loop."""
    try:
        gdir = _PKG / "GUIDELINES"
        model = _MODEL_ROOT.name or "model"
        result: dict = {"written": None, "graduated": None}
        if (fires_when or recipe) and op_class:
            bkt = (bucket or op_class).strip()
            slug = promote._slug(bkt, model)
            section = (
                f"## Learned: {bkt} coherence {{#{slug}}}\n"
                "<!-- route\n"
                f"op_class: {op_class.strip()}\n"
                "lever_type: structural\n"
                "-->\n\n"
                f"**Fires when:** {(fires_when or '').strip()}\n\n"
                f"{(recipe or '').strip()}\n"
            )
            result["written"] = str(promote.write_provisional_lever(section, slug, gdir, model))
        if reused_lever_id:
            import types as _types

            shim = _types.SimpleNamespace(model_root=lambda: _MODEL_ROOT)
            grad = promote.maybe_graduate(shim, reused_lever_id.strip(), gdir)
            result["graduated"] = str(grad) if grad else None
        return result
    except Exception as exc:  # noqa: BLE001 — write-back must never break the loop
        return {"written": None, "graduated": None, "error": str(exc)[-200:]}


@mcp.tool()
def _host_gate(prof: dict, blocking: list, attempts: list) -> dict | None:
    if blocking:
        return None
    for b in prof.get("buckets") or []:
        if b.get("id") != "host_overhead":
            continue
        hms = round(float(b.get("device_ms") or 0.0), 4)
        src = (b.get("tags") or {}).get("source")
        if hms < _MATERIAL_GAP_MS or src != "op_gap":
            return None
        host_op = {
            "op_code": "host_overhead",
            "bucket": "host_fallback",
            "bound_by": "host",
            "gap_ms": hms,
            "grid": "",
            "weight_dtype": "",
        }
        done, rung, reason = _op_ladder_status(host_op, "host_overhead", attempts)
        if done:
            return None
        return {
            "op": "host_overhead",
            "op_class": "host_fallback",
            "gap_ms": hms,
            "bound_by": "host",
            "grid": None,
            "weight_dtype": None,
            "next_rung": rung,
            "reason": reason,
        }
    return None


def kv_cache_needed_by_scaling(ms_at_c, ms_at_2c, ratio_threshold=1.6):
    if not isinstance(ms_at_c, (int, float)) or not isinstance(ms_at_2c, (int, float)):
        return None
    if ms_at_c <= 0:
        return None
    return (ms_at_2c / ms_at_c) >= ratio_threshold


def _decode_is_recompute(model_root) -> bool:
    try:
        src = (Path(model_root) / "tt" / "pipeline.py").read_text(errors="ignore")
    except Exception:
        return False
    s = "".join(src.split()).lower()
    no_kv = ("use_cache=false" in s) or ("past_key_value=none" in s)
    kv_write = any(
        k in s for k in ("update_cache", "paged_update", "fill_cache", "kv_cache=", "cache_k[", "self.cache")
    )
    return no_kv and not kv_write


def _decode_gate(prof: dict, attempts: list) -> dict | None:
    if os.environ.get("TT_PERF_MODULE_LEVEL") == "1":
        return None
    repeat = prof.get("decode_status") == "repeat_prefill"
    scale = kv_cache_needed_by_scaling(prof.get("decode_ms_at_c"), prof.get("decode_ms_at_2c"))
    recompute = bool(scale) if scale is not None else _decode_is_recompute(_MODEL_ROOT)
    if not (repeat or recompute):
        return None
    if any((a.get("kernel_kind") or "").lower() in ("structural-decode", "kv-cache") for a in attempts):
        return None
    reason = (
        "repeat_prefill: decode re-runs the full prefill every token (no cached decode_step / "
        "KV-cache); add a KV-cache + single-token decode_step (recall_knobs(op_class='decode'))"
        if repeat
        else "recompute decode: per-token cost scales with capacity (use_cache=False, no KV-cache write) "
        "-> O(capacity) recompute every token EVEN THOUGH it traces; add a KV-cache + single-token "
        "decode_step (recall_knobs(op_class='decode'))"
    )
    return {
        "op": "generation_loop",
        "op_class": "decode",
        "gap_ms": round(float(prof.get("per_token_ms") or _MATERIAL_GAP_MS), 4),
        "bound_by": "host",
        "grid": None,
        "weight_dtype": None,
        "next_rung": "structural-decode",
        "reason": reason,
    }


@mcp.tool()
def termination_check() -> dict:
    """THE BINDING STOP GATE and SOLE authority on 'optimize more or not' — you may declare DONE ONLY
    when this returns can_stop=true. It decides PURELY from its own deterministic measurement (the
    roofline profile + per-op tags), NOT from whether any other gate 'fired'. For EVERY open op with
    material gap (>= PERF_MCP_MATERIAL_GAP_MS) it runs the DETERMINISTIC LADDER (knob -> fusion ->
    tt-lang -> C++) IN ORDER from the op's tags + recorded kernel attempts: a non-full grid blocks on
    the full-grid knob FIRST; only when cheap levers are exhausted (grid=full) does it require a
    tt-lang kernel, then a C++ kernel. An op is DONE only when the WHOLE ladder is exhausted
    (grid=full + tt-lang + C++ all measured) = genuine irreducible residual. NO 'kernel was tried so
    stop' shortcut; NO OR-with-at_floor escape. can_stop is true iff no material op has a reachable
    rung left. Obey can_stop; for each blocking_op do the rung named in its 'next_rung'."""
    try:
        prof = _profile_once(cq=1)
    except Exception as exc:  # noqa: BLE001
        _note_device_crash("termination_check")
        return {"can_stop": False, "error": f"profiler crashed: {str(exc)[-500:]}"}
    _note_device_ok()
    dev = round(float(prof.get("device_ms", 0.0)), 4)
    try:
        rep = roofline.residual_report(prof, _ENV)
    except Exception as exc:  # noqa: BLE001
        return {"can_stop": False, "error": f"residual_report failed: {str(exc)[-400:]}"}
    at_floor = bool(rep.get("at_floor"))
    attempts = [a for a in _load_attempts() if a.get("kernel_detected_in_source")]
    blocking, cleared = [], []
    for o in rep.get("open_ops") or []:
        gap = o.get("gap_ms") or 0.0
        if gap < _MATERIAL_GAP_MS:
            continue
        op_code = o.get("op_code") or o.get("bucket") or ""
        entry = {
            "op": op_code,
            "op_class": o.get("bucket"),
            "gap_ms": round(float(gap), 4),
            "bound_by": o.get("bound_by"),
            "grid": o.get("grid"),
            "weight_dtype": o.get("weight_dtype"),
        }
        done, rung, reason = _op_ladder_status(o, op_code, attempts)
        if done:
            cleared.append({**entry, "verdict": reason})
        else:
            blocking.append({**entry, "next_rung": rung, "reason": reason})
    # SOLE-AUTHORITY decision: stop iff no material op has a reachable rung left. This is driven only
    # by the gate's own ladder analysis — there is no "OR a kernel was attempted" escape, and the
    # at_floor field is informational evidence, not an independent stop license.
    host_block = _host_gate(prof, blocking, attempts)
    if host_block:
        blocking.append(host_block)
    decode_block = _decode_gate(prof, attempts)
    if decode_block:
        blocking.append(decode_block)
    can_stop = not blocking
    halt = next((b for b in blocking if b.get("next_rung") == "tt-lang:install-required"), None)
    # DETERMINISTIC SELECTION: the single op+rung the agent must work next (largest-gap blocking op).
    next_target = (
        {
            "op": blocking[0]["op"],
            "op_class": blocking[0]["op_class"],
            "grid": blocking[0]["grid"],
            "bound_by": blocking[0]["bound_by"],
            "rung": blocking[0]["next_rung"],
            "gap_ms": blocking[0]["gap_ms"],
            "reason": blocking[0]["reason"],
        }
        if blocking
        else None
    )
    _persist_target(next_target)
    return {
        "can_stop": can_stop,
        "halt": bool(halt),
        "halt_reason": halt.get("reason") if halt else None,
        "device_ms": dev,
        "at_floor": at_floor,
        "residual_gap_ms": rep.get("residual_gap_ms"),
        "material_gap_threshold_ms": _MATERIAL_GAP_MS,
        "next_target": next_target,
        "blocking_ops": blocking,
        "cleared_ops": cleared,
        "directive": (
            "DONE — every material-gap op has its full checklist ticked (grid + dtype knobs + tt-lang "
            "+ C++). No reachable rung remains."
            if can_stop
            else "NOT DONE — work next_target (the largest-gap blocking op) at its rung. REUSE-FIRST: "
            "BEFORE editing, call recall_knobs(next_target.op_class, next_target.grid, "
            "next_target.bound_by) and APPLY/ADAPT any matching "
            "catalogued knob (heed its negative knowledge); improvise from scratch ONLY if nothing "
            "matches — a recalled knob still requires check_pcc + measure + record_kernel_attempt (it "
            "never skips a rung). Ladder ORDER: "
            "knob:grid -> knob:dtype -> tt-lang -> cpp. record_kernel_attempt for EACH rung (knobs too: "
            "kind='grid'/'dtype'; kernels: 'tt-lang'/'cpp'). A later rung does NOT clear an op while an "
            "earlier rung is untried. WRITE-BACK: after you COMMIT a win you IMPROVISED (recall_knobs "
            "had no match), call distill_knob to persist it for reuse; if the win re-used a provisional "
            "lever from another model, pass its id to distill_knob to graduate it. Re-run "
            "termination_check after each rung."
        ),
    }


@mcp.tool()
def tp_pick_degree(m: int, k: int, n: int) -> dict:
    """Decide the tensor-parallel degree for a dense matmul (M x K x N) by MEASUREMENT: sweep each
    feasible degree (>= the model's TP floor) on the mesh and return the fastest. Returns
    {best_tp, timings_ms, floor}. best_tp=1 means TP did not help this matmul (keep it single-chip).
    Call this on the tp-fracture rung to pick the level, then apply it and verify_tp_fracture."""
    if os.environ.get("PERF_MCP_ENABLE_TP_SWEEP") != "1":
        return {
            "best_tp": 1,
            "skipped": (
                "on-mesh TP sweep disabled by default: it opens a NESTED mesh device and toggles the fabric "
                "config while a mesh is already resident, which can wedge the inter-chip fabric on ANY "
                "multi-chip system (recovery needs a board reset). Base TP is already applied by the pipeline "
                "sharding. Set PERF_MCP_ENABLE_TP_SWEEP=1 to force this sweep."
            ),
        }
    try:
        import ttnn

        from cc_optimize.tp_fracture import sweep_degrees

        floor = int(os.environ.get("TT_PERF_TP_FLOOR", "1") or "1")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
        try:
            num = mesh.shape[0] * mesh.shape[1]
            candidates = [d for d in (floor, num) if d >= floor]
            r = sweep_degrees(mesh, m=m, k=k, n=n, candidates=candidates)
        finally:
            ttnn.close_mesh_device(mesh)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        r["floor"] = floor
        return r
    except Exception as exc:  # noqa: BLE001
        return {"best_tp": 1, "error": str(exc)[-400:]}


@mcp.tool()
def verify_tp_fracture(m: int, k: int, n: int, tp: int = 4) -> dict:
    """Validate the tensor-parallel fracture of a dense matmul (M x K x N) on the real mesh: shard the
    weight across the mesh, matmul per-chip, all_gather, and compare to the dense single-chip result.
    Returns {ok, pcc, ...}: ok=True when pcc>0.99 (the fracture is mathematically correct). Call this
    on the tp-fracture rung to PROVE a fracture is correct before committing it."""
    if os.environ.get("PERF_MCP_ENABLE_TP_SWEEP") != "1":
        return {
            "ok": False,
            "skipped": (
                "on-mesh TP fracture verify disabled by default: it opens a NESTED mesh device and toggles "
                "the fabric config while a mesh is already resident, which can wedge the inter-chip fabric on "
                "ANY multi-chip system. Set PERF_MCP_ENABLE_TP_SWEEP=1 to force it."
            ),
        }
    try:
        import ttnn

        from cc_optimize.tp_fracture import verify_fracture

        rows = cols = 2
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
        try:
            r = verify_fracture(mesh, m=m, k=k, n=n, tp=tp)
        finally:
            ttnn.close_mesh_device(mesh)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        r["ok"] = bool(r.get("pcc", 0.0) > 0.99)
        return r
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[-400:]}


if __name__ == "__main__":
    try:
        _rebuild_optimize_report()
    except Exception:  # noqa: BLE001
        pass
    mcp.run()
