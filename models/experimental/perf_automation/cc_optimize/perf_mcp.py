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

import json
import os
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

# kernel-authoring evidence markers, searched in the model source tree (grounds a recorded attempt)
_KERNEL_MARKERS = ("generic_op", "ProgramDescriptor", "KernelDescriptor", "@ttl.", "ttl.operation", "import ttl")


def _scan_kernel_evidence() -> dict:
    """Look for real custom-kernel authoring in the model source so a recorded attempt can't be a
    phantom. Returns {markers, cpp_files} — empty if no custom kernel is present."""
    found, cpp = set(), []
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
    except Exception:  # noqa: BLE001
        pass
    return {"markers": sorted(found), "cpp_files": cpp}


def _load_attempts() -> list:
    if _KERNEL_LOG_PATH.exists():
        try:
            return json.loads(_KERNEL_LOG_PATH.read_text())
        except Exception:  # noqa: BLE001
            return []
    return []


def _save_attempts(a: list) -> None:
    _KERNEL_LOG_PATH.write_text(json.dumps(a))


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
        if "tt-lang" not in kinds:
            if not _ttl_available():
                return (
                    False,
                    "tt-lang:install-required",
                    "this op needs a tt-lang kernel but the ttl toolchain is not installed — install "
                    "tt-lang first (e.g. pip install tt-lang==1.0.1 --no-deps, matching your ttnn)",
                )
            return (
                False,
                "tt-lang",
                "knobs exhausted (grid+dtype); author a tt-lang kernel (GUIDELINES/11) and record it",
            )
        if "cpp" not in kinds:
            return (
                False,
                "cpp",
                "tt-lang tried; author a C++ Metalium kernel via ttnn.generic_op (GUIDELINES/12) and record it",
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
        self.run = _Run(tempfile.mkdtemp(prefix="perf_mcp_"))
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


def _profile_once() -> dict:
    ctx = _Ctx()
    tmpdir = ctx.run.dir
    try:
        profiles = measure_runs(ctx)
        prof = profiles[0]
        try:
            prof = roofline.annotate_profile(prof, _ENV)
        except Exception:  # annotation is best-effort; raw profile still usable
            pass
        return prof
    finally:
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
        prof = _profile_once()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[-800:]}
    _BASELINE_PATH.write_text(json.dumps(prof))
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
        prof = _profile_once()
    except Exception as exc:  # noqa: BLE001
        _note_device_crash("measure_candidate")  # may tt-smi reset if this is a repeat (wedge)
        return {"verdict": "REJECTED", "reason": f"profiler crashed: {str(exc)[-600:]}"}
    _note_device_ok()
    dev = round(float(prof.get("device_ms", 0.0)), 4)
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
    ok, reason = _rm._comparable(baseline, prof, floor_ms=floor_ms)
    if not ok:
        return {"verdict": "REJECTED", "reason": reason, "device_ms": dev, "baseline_ms": base_dev}
    delta = round(base_dev - dev, 4)
    pct = round((delta / base_dev) * 100.0, 2) if base_dev else 0.0
    faster = delta > 0.05  # noise floor
    return {
        "verdict": "valid",
        "device_ms": dev,
        "baseline_ms": base_dev,
        "delta_ms": delta,
        "pct_faster": pct,
        "is_real_gain": faster,
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


@mcp.tool()
def git_head() -> dict:
    """Return the current git HEAD sha of the model repo (your clean checkpoint / revert target)."""
    repo = gitio.repo_root(_MODEL_ROOT)
    return {"sha": gitio.head_sha(repo)}


@mcp.tool()
def git_commit(message: str) -> dict:
    """Commit the current model-dir changes (scoped to the model dir only — unrelated repo changes
    are left untouched). Use this to BANK a verified win (valid measure + ok pcc + faster). Returns
    the new sha."""
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


@mcp.tool()
def record_kernel_attempt(
    op_signature: str, kernel_kind: str, measured_ms: float, beat_baseline: bool, note: str = ""
) -> dict:
    """Record that you AUTHORED and MEASURED a real custom kernel for an open op (tt-lang or C++).
    REQUIRED before termination_check() will let you stop on any op with material roofline gap — a
    measured attempt is the EMPIRICAL validation that REPLACES 'I reasoned a kernel won't help'.
    Even a measured kernel that does NOT beat ttnn clears the op as 'tried' (that's the proof it
    can't be improved, not an assertion). op_signature: enough of the profiler op_code to match it
    (e.g. 'MatmulDeviceOperation 32 x 32 x 32' or 'LayerNorm'). kernel_kind: 'tt-lang' | 'cpp'.
    measured_ms: the device_ms measure_candidate reported with the kernel in place. This verifies a
    kernel actually exists in the model source — a record with no kernel present is flagged and will
    NOT clear the op."""
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
    ev = _scan_kernel_evidence()
    detected = True if is_knob else bool(ev["markers"] or ev["cpp_files"])
    ttl_absent = (kernel_kind or "").lower() == "tt-lang" and not _ttl_available()
    if ttl_absent:
        detected = False
    attempts = _load_attempts()
    rec = {
        "op_signature": op_signature,
        "kernel_kind": kernel_kind,
        "measured_ms": round(float(measured_ms), 4),
        "beat_baseline": bool(beat_baseline),
        "note": note,
        "kernel_detected_in_source": detected,
        "evidence": ev,
    }
    attempts.append(rec)
    _save_attempts(attempts)
    return {
        "recorded": True,
        "attempt": rec,
        "warning": (
            "tt-lang toolchain (ttl) not installed — kernel cannot run or be measured; attempt NOT credited."
            if ttl_absent
            else None
            if detected
            else "NO kernel markers (generic_op/@ttl/.cpp/ProgramDescriptor) found in model source — this "
            "attempt is UNSUPPORTED and will NOT clear the op in termination_check. Author a real kernel first."
        ),
    }


@mcp.tool()
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
        prof = _profile_once()
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


if __name__ == "__main__":
    mcp.run()
