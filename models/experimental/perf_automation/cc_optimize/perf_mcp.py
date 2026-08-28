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
import time
import tempfile
from pathlib import Path

# import the EXISTING deterministic core (no reimplementation)
_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG.parent.parent.parent))  # repo root, so `models...` imports resolve
sys.path.insert(0, str(_PKG))  # the perf_automation dir, so `agent` imports resolve

from agent import gitio, perf_target, promote, roofline, router  # noqa: E402
from agent import integrity as _integrity  # noqa: E402
from agent.layer_depth import set_depth as _set_depth  # noqa: E402

_DEPTH_GUARD = "models.experimental.perf_automation.agent.depth_guard_plugin"
from agent.handlers import remeasure as _rm  # noqa: E402
from agent.measure import measure_runs  # noqa: E402
from agent.pcc_runner import run_pcc  # noqa: E402

# ONE state directory for every durable temp artifact -- see cc_optimize/tmpstate.py. Loaded by path
# because cc_optimize is not a package: these modules run both as scripts and as plain imports.
import importlib.util as _ilu_ts

_ts_spec = _ilu_ts.spec_from_file_location("_tmpstate", str(Path(__file__).resolve().parent / "tmpstate.py"))
_tmpstate = _ilu_ts.module_from_spec(_ts_spec)
_ts_spec.loader.exec_module(_tmpstate)
state_dir = _tmpstate.state_dir

try:  # mcp < 2.0
    from mcp.server.fastmcp import FastMCP  # noqa: E402
except ModuleNotFoundError:  # mcp >= 2.0 renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP  # noqa: E402

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
_MODEL_ROOT_CONFIGURED = bool(
    (os.environ.get("PERF_MCP_MODEL_ROOT") or "").strip()
    or str(_MANIFEST.get("config", {}).get("model_root") or "").strip()
)
_MODEL_ROOT = Path(os.environ.get("PERF_MCP_MODEL_ROOT") or _MANIFEST.get("config", {}).get("model_root", "."))
# WAS THE MODEL DIRECTORY ACTUALLY STATED, or is this the "." default? The two are not the same
# question as "does the path exist", and reading a model FACT from an unstated root means resolving
# it against the WORKING DIRECTORY -- which is how the gemma-3-12b report came to price a 32-layer,
# hidden-1280, 30 MB model: a stray perf_target_inputs.json in the repo root, describing the vision
# tower, sitting where the loop happened to be cd'd. It produced a prefill memory ceiling of 0.061 ms
# against a 100 ms measurement, no param count, therefore no compute roof, therefore no fidelity
# ladder -- three broken sections from one silently-adopted file.
#
# A fact about the model has to come from the model's own directory. Unstated is UNKNOWN, and unknown
# renders as "not measured", which is true -- rather than as a number, which was not.
_MODEL_ROOT_STATED = bool(
    str(os.environ.get("PERF_MCP_MODEL_ROOT") or "").strip()
    or str(_MANIFEST.get("config", {}).get("model_root") or "").strip()
)
# PUBLISH the key every per-run artifact is named after. perf_mcp derived it from _MODEL_ROOT while
# run.py and the ledger read PERF_MCP_MODEL_NAME -- which nothing ever set, so those fell back to the
# literal "model". Reader and writer then pointed at different files (perf_mcp_baseline_model_main.json
# appeared beside the real one) and every model would have shared one "model" ledger: the unkeyed bug
# under a new name. One authoritative source, exported once, read by all three processes.
os.environ.setdefault("PERF_MCP_MODEL_NAME", _MODEL_ROOT.name or "model")
_ENV = _MANIFEST.get("env", {})


# where profile_model stashes the current baseline so measure_candidate can compare structurally
def _baseline_path():
    """Per-(model, task) device_ms baseline. Was a single global file, unlike the already-keyed
    the measurement ledger / _throughput_path(), and nothing reset it at task start -- so the
    baseline a candidate was compared against could belong to a previous model, a previous
    module, or a concurrent optimize on the same box. A leftover SLOWER baseline books the
    first candidate of the new run as a large fake win."""
    model = _MODEL_ROOT.name if _MODEL_ROOT else "model"
    task = os.environ.get("PERF_MCP_TASK", "main")
    return state_dir() / ("perf_mcp_baseline_%s_%s.json" % (model, task))


def baseline_exists() -> bool:
    """Is there already a usable device_ms baseline for this (model, task)?

    The bar is established ONCE, the first time a model is optimized, and then only ever moves down
    on a win. Callers use this to skip re-measuring it -- a re-measure is not free (250 s on
    gemma-3-12b-it) and, worse, it silently redefines what every later verdict is graded against.
    """
    try:
        return float(json.loads(_baseline_path().read_text()).get("device_ms") or 0.0) > 0
    except Exception:  # noqa: BLE001
        return False


def _promote_baseline(prof: dict) -> dict:
    """Store the profile, but move the device_ms BAR only DOWNWARD. Returns what was stored.

    One file was doing two jobs. profile_model's own docstring invites re-profiling ("call this
    again whenever you want a fresh picture"), because the agent needs current per-op buckets and
    tags to choose its next target -- and that write was unconditional, so refreshing the PICTURE
    also redefined the BAR. On gemma-3-12b-it profile_model ran ~44 times in one run.

    The failure that produces: the agent applies an edit that makes the model slower, re-profiles
    for a fresh picture, and the slower number becomes the baseline. measure_candidate now grades
    against it, so REVERTING the bad edit reads as a win. It also drifts the baseline_at_record
    stamps (381.186 / 381.222 / 381.263 / 381.291 / 381.311), which the resume filter compares with
    exact equality -- a different subset of attempt history survives each run, which is the upstream
    cause of the 38% repeat rate.

    The full-pipeline bar has ratcheted since 9358229fa8; this is the same rule for the steering
    metric. Refresh the picture freely, move the bar only on a real improvement.

    A shape change (perf_layers) re-baselines outright: a 4-layer profile and a 48-layer profile are
    different units, so "slower" is not a regression, it is a different measurement.
    """
    path = _baseline_path()
    new_ms = 0.0
    try:
        new_ms = float(prof.get("device_ms") or 0.0)
    except (TypeError, ValueError):
        new_ms = 0.0
    kept_ms = None
    try:
        cur = json.loads(path.read_text())
        cur_ms = float(cur.get("device_ms") or 0.0)
        same_shape = str(cur.get("perf_layers") or "") == str(prof.get("perf_layers") or "")
        if cur_ms > 0 and new_ms > 0 and same_shape and new_ms > cur_ms:
            kept_ms = cur_ms
    except Exception:  # noqa: BLE001
        kept_ms = None
    out = dict(prof)
    if kept_ms is not None:
        out["device_ms"] = kept_ms
        out["observed_device_ms"] = new_ms
        print(
            "  [baseline-ratchet] profile read %.4f ms, slower than the committed baseline %.4f ms; "
            "picture refreshed, BAR unchanged (a re-profile must not redefine what wins are graded "
            "against)" % (new_ms, kept_ms),
            file=sys.stderr,
            flush=True,
        )
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(out))
        os.replace(str(tmp), str(path))
    except Exception:  # noqa: BLE001
        path.write_text(json.dumps(out))
    return out


# The tool is trace+1cq end to end; this is the ONLY full-pipeline baseline (no 2-CQ twin).
# KEYED by (model, task), like _baseline_path() and the measurement ledger. It was a single global
# file, so anything on the box could overwrite a live run's AFTER number: on 2026-07-27 a unit test
# writing a fixture value of 100.0 to the real path landed in a 10-hour optimize run whose every
# actual reading was ~23.9 ms, and two concurrent optimize runs would have done the same to each
# other. A scoreboard that any other process can write is not a scoreboard.
def _fullpipe_baseline_1cq_path():
    model = _MODEL_ROOT.name if _MODEL_ROOT else "model"
    task = os.environ.get("PERF_MCP_TASK", "main")
    return state_dir() / ("perf_mcp_full_pipeline_baseline_1cq_%s_%s.json" % (model, task))


_FULLPIPE_BASELINE_1CQ_PATH = _fullpipe_baseline_1cq_path()
# Divergence tolerance for the end-to-end gate. This is NOT a licence to commit a regression:
# it only decides when to shout "diverged". A reading that is slower than the committed best
# is reported as a regression regardless (see the regressed branch below), because "within 8%
# of the best ever" used to read as ok and let real latency ratchet upward unnoticed.
_FULLPIPE_TOL = float(os.environ.get("PERF_MCP_FULLPIPE_TOL", "0.08"))
# Must match the BEFORE bookend (run.py sets 3): AFTER = min over 1-sample readings vs
# BEFORE = median of 3 manufactured the full noise range as a gain on every run.
_FULLPIPE_SAMPLES = max(1, int(os.environ.get("PERF_MCP_FULLPIPE_SAMPLES", "3")))
# How far two readings of one stage's read set may differ and still be treated as the same
# measurement. The working set is a property of the build, so agreeing samples are the evidence that
# the number is an observation rather than instrumentation noise -- and only an observation may be
# pinned as a permanent ceiling divisor.
_STAGE_BYTES_AGREE_TOL = 0.01
_FULLPIPE_TARGET_MS = float(os.environ.get("PERF_MCP_TARGET_MS", "0") or "0")

# C++-kernel SAFETY: a bad Metalium kernel can WEDGE a device core (tt-lang/ttnn fail gracefully; raw
# C++ can deadlock the NoC). Device runs are already subprocess-isolated+timeout-bounded (so the loop
# survives a hang), but a wedged core can persist across runs -> recover with tt-smi reset. Only after
# TWO consecutive crashes (not a single transient) so it's a RARE fallback, not routine.
import shutil as _shutil  # noqa: E402
import subprocess as _sp  # noqa: E402


def _dr():
    """The shared device-recovery primitive (agent/device_recovery.py). Imported lazily and by path
    so this server keeps working when it is launched with a bare sys.path, as the MCP client does."""
    global _DR_MOD
    try:
        return _DR_MOD
    except NameError:
        pass
    try:
        from agent import device_recovery as _m
    except Exception:  # noqa: BLE001
        import importlib.util as _ilu

        _p = Path(__file__).resolve().parents[1] / "agent" / "device_recovery.py"
        _spec = _ilu.spec_from_file_location("tt_device_recovery", str(_p))
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
    globals()["_DR_MOD"] = _m
    return _m


_CONSEC_CRASH = _dr().CONSEC_CRASH
_TT_SMI = _shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"


_RUN_MOD = None


def _run_module():
    """Load cc_optimize/run.py by path (stdlib-only import, same as optimize.py does) and cache it, so
    device recovery reuses run.py's board-aware _reset_devices — the SAME per-board reset (whole
    board(s) of PERF_MCP_DEVICES) run.py's own watchdog uses. None if it can't be loaded."""
    global _RUN_MOD
    if _RUN_MOD is None:
        try:
            import importlib.util

            _p = Path(__file__).with_name("run.py")
            _spec = importlib.util.spec_from_file_location("cc_optimize_run_reset", str(_p))
            _m = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
            _RUN_MOD = _m
        except Exception:  # noqa: BLE001
            _RUN_MOD = False
    return _RUN_MOD or None


def _board_reset(where: str, note: str, target: str = "") -> bool:
    """Reset ``target`` (default: the configured devices) and REPORT WHETHER THE RESET RAN.

    ``target`` is the chip/board spec to reset. It used to be absent: callers computed which chip had
    died and passed it only inside ``note``, i.e. into the log line, while this function always reset
    ``PERF_MCP_DEVICES``. The evidence was printed and discarded — the message read ``target=3`` while
    board 0 was reset. A target that does not reach the reset command is not a target.

    Recover via run.py's board-aware _reset_devices (whole board(s) — a single-chip `-r 0`
    half-resets a p300c and breaks its enumeration, and does not reset a Galaxy at all). If that module
    is unavailable, fall back to agent.probes._reset_arg_sets — the SAME galaxy/board-aware invocations
    (glx_reset on a Galaxy, full enumerated `-r` elsewhere) — and never a bare single-chip `-r 0`."""
    spec = (target or "").strip() or (os.environ.get("PERF_MCP_DEVICES", "").strip() or "all")
    _mod = _run_module()
    if _mod is not None:
        try:
            status = _mod._reset_devices(spec)
            sys.stderr.write(f"[perf-mcp] {note}: {status} at {where}\n")
            return True
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[perf-mcp] board-aware reset unavailable ({exc}) at {where}; probes fallback\n")
    try:
        from agent import probes as _pr

        arg_sets = _pr._reset_arg_sets()
    except Exception:  # noqa: BLE001
        arg_sets = [["-r"]]
    for args in arg_sets:
        try:
            r = _sp.run([_TT_SMI, *args], capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                sys.stderr.write(f"[perf-mcp] {note} via tt-smi {' '.join(args)} (fallback) at {where}\n")
                return True
        except Exception:  # noqa: BLE001
            continue
    sys.stderr.write(f"[perf-mcp] tt-smi reset failed at {where}\n")
    return False


def _device_recover(where: str) -> bool:
    """Reset after a likely wedge, through the VERIFIED path.

    Kept as a named entry point, but it no longer calls _board_reset directly: an unverified reset
    helper sitting next to a verified one is a trap, since the next caller cannot tell them apart.
    """
    return _recover_device(where, "")


def _note_device_crash(where: str, error_text: str = "") -> None:
    """Record a device crash and recover when the evidence justifies it (shared policy)."""
    _dr().note_crash(
        where,
        lambda tgt: _board_reset(where, "recover (target=%s)" % tgt, target=tgt),
        error_text=error_text,
        config_target=os.environ.get("PERF_MCP_DEVICES", "") or "",
        log=lambda m: sys.stderr.write("[perf-mcp] %s\n" % m),
    )


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


# --- device recovery: decide from EVIDENCE, fall back to the config guess -----------------------

_RESET_FAIL_LIMIT = _dr().RESET_FAIL_LIMIT
_RESET_FAILS = _dr().RESET_FAILS


def _is_dead_board(text) -> bool:
    return _dr().is_dead_board(text)


def _dead_chip_from_error(text):
    return _dr().dead_chip_from_error(text)


def _recover_device(where: str, error_text: str = "") -> bool:
    """Reset the board and REPORT WHETHER IT WORKED, via the shared primitive.

    Only the ISSUING of the reset is local (run.py's board-aware _reset_devices); which board, how
    many tries, whether it came back and when to give up are decided in one place for every caller.
    """
    return _dr().recover(
        where,
        lambda tgt: _board_reset(where, "recover (target=%s)" % tgt, target=tgt),
        error_text=error_text,
        config_target=os.environ.get("PERF_MCP_DEVICES", "") or "",
        log=lambda m: sys.stderr.write("[perf-mcp] %s\n" % m),
    )


def _device_is_healthy() -> bool:
    return _dr().device_is_healthy()


def _recovery_exhausted() -> bool:
    """Have resets failed enough times that the run should stop rather than poll a dead board?"""
    return _dr().recovery_exhausted()


def _reclaim_mesh(where: str) -> bool:
    """Reclaim the mesh after an L1 overflow, VERIFY it came back, and report the outcome.

    This path is reached from profile_model / measure_candidate — the hot loop. It used to reset
    blindly and then clear the crash counter unconditionally, so a reclaim that failed was recorded as
    a success and erased the very history that would have escalated. Route it through _recover_device
    so it gets the same verification and the same escalation budget as every other reset; there is no
    chip id in an L1 overflow, so the target falls to the config guess and then to every board.
    """
    ok = _recover_device(where, "")
    if ok:
        _CONSEC_CRASH["n"] = 0
    return ok


_L1_OVERFLOW_MSG = (
    "L1_OVERFLOW: this config's circular buffers exceed the per-core L1 budget (~1.5MB on Wormhole) "
    "and crashed the run; the mesh was reset. Reduce the L1 footprint (smaller in0_block_w / per_core_N, "
    "or spread the matmul over more cores) and retry — do NOT keep this config."
)

_DRAM_TRACE_SIG_A = ("trace region", "trace_region", "trace buffer", "trace_buffer")
_DRAM_TRACE_SIG_B = ("full", "exceed", "not enough", "out of space", "too small", "ran out", "grow the trace")


def _dram_capacity_per_chip() -> int:
    """Per-chip DRAM the tool already detected (ARCH_FACTS -> manifest env: 12 GB wormhole, 32 GB
    blackhole). 0 when the env probe was unavailable."""
    try:
        v = int(_ENV.get("dram_capacity_bytes") or 0)
        return v if v > 0 else 0
    except (TypeError, ValueError):
        return 0


# Derive the trace-region start/ceiling from the board's DRAM (the value the tool already reads at
# Step 1), NOT a hardcoded byte count, via the SINGLE source of truth shared with the tracy-baseline
# path (agent.environment.default_trace_region_bytes). Falls back to the legacy constants when the
# manifest carries no capacity (e.g. env probe unavailable), so nothing regresses.
try:
    from agent.environment import default_trace_region_bytes as _default_trace_region_bytes

    _TRACE_REGION_DEFAULT, _TRACE_REGION_MAX = _default_trace_region_bytes(_dram_capacity_per_chip())
except Exception:  # noqa: BLE001
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
_KERNEL_LOG_PATH = Path(os.environ.get("PERF_MCP_KERNEL_LOG") or (state_dir() / "perf_mcp_kernel_attempts.json"))
_MATERIAL_GAP_MS = float(os.environ.get("PERF_MCP_MATERIAL_GAP_MS", "0.25"))
_MATERIAL_GAP_ENV_SET = "PERF_MCP_MATERIAL_GAP_MS" in os.environ
_MATERIAL_GAP_FRAC = float(os.environ.get("PERF_MCP_MATERIAL_GAP_FRAC", "0.03"))
_MATERIAL_GAP_FLOOR = float(os.environ.get("PERF_MCP_MATERIAL_GAP_FLOOR", "0.05"))
_MAX_KNOB_RETRIES = int(os.environ.get("PERF_MCP_MAX_KNOB_RETRIES", "2"))
_STRUCTURAL_RUNGS = {"structural", "gather", "fusion", "fuse", "sparse", "cache", "kv-cache"}
# THE ladder, in climb order, for each roofline binding. ONE table, because the module used to hold
# two orderings of the same rungs and only one of them knew what the op was waiting on:
# `_KNOB_ORDER` (below, now derived from here) steered the per-op gate correctly, while a separate
# `_LADDER_ORDER` literal -- written fresh from memory when the closed-rung refusal needed an
# ordered list -- fixed the climb at grid -> fidelity -> dtype -> shard for every model, and dropped
# `host` entirely.
#
# The cost of each is specific. Dropping `host`: the refusal returns `rungs_still_open` to redirect
# the agent, so on gemma-3-12b-it -- where EVERY top op reads bound_by=dispatch and host_overhead is
# 62.4 ms, the second-largest bucket -- it named every remaining rung EXCEPT the one that addresses
# the binding constraint. Across 158 attempts the dispatch axis was tried once. Fixing the order:
# fidelity speeds the MATH ENGINE, so on a model whose ops wait on DRAM bytes it leads with a lever
# that cannot move the number, ahead of the two (dtype, shard) that cut bytes directly. gemma-3-12b
# is memory-bound in both stages -- decode compute runs at 0.1% of peak.
#
# So the binding sets PRIORITY, never MEMBERSHIP: every rung appears in every row. bound_by is a
# roofline ESTIMATE, ops are rarely purely one-bound, and `compute` is only ever computed for
# matmuls -- used as a filter it silently deletes levers for a whole run (see _op_ladder_status).
_RUNG_PRIORITY = {
    "memory": ("grid", "dtype", "shard", "fidelity", "host", "structural", "tt-lang", "cpp"),
    "compute": ("grid", "fidelity", "dtype", "shard", "host", "structural", "tt-lang", "cpp"),
    # Nothing on the knob rungs addresses dispatch: the op is waiting on the host loop that launches
    # it, so trace capture / 2-CQ leads and the knobs follow as the completeness sweep.
    "dispatch": ("host", "grid", "fidelity", "dtype", "shard", "structural", "tt-lang", "cpp"),
    "": ("grid", "dtype", "shard", "fidelity", "host", "structural", "tt-lang", "cpp"),
}
_KNOBS = ("grid", "fidelity", "dtype", "shard")


def ladder_order(bound_by: str = "") -> list:
    """The canonical climb order for a binding. Import this rather than restating it.

    An unknown or missing bound_by gets the "" row, which leads with the byte-cutting levers: an op
    whose binding could not be established is more often waiting on memory than on the math engine,
    and grid -- first either way -- is the cheapest thing to try while that is still open."""
    return list(_RUNG_PRIORITY.get((bound_by or "").strip().lower()) or _RUNG_PRIORITY[""])


# Derived, so the two can no longer disagree: same table, filtered to the four cheap knobs.
_KNOB_ORDER = {b: tuple(r for r in order if r in _KNOBS) for b, order in _RUNG_PRIORITY.items()}

_KNOB_REASON = {
    "grid": lambda g, f, w: "occupy the FULL core grid (grid=%s) via a full-grid program_config; "
    "record_kernel_attempt(...,'grid',...) even on a no-gain" % (g or "unknown"),
    "fidelity": lambda g, f, w: "lower the math fidelity (now %s) HiFi4->HiFi2->LoFi; "
    "record_kernel_attempt(...,'fidelity',...) to mark it tried (even on a PCC revert / no-gain)" % (f or "unknown"),
    "dtype": lambda g, f, w: "lower the weight dtype (now %s) to bf8_b/bf4_b; "
    "record_kernel_attempt(...,'dtype',...) even on a PCC revert / no-gain" % (w or "unknown"),
    "shard": lambda g, f, w: "shard this op's weights/activations into L1 (height/width shard) to cut DRAM "
    "reads; record_kernel_attempt(...,'shard',...) to mark it tried (even on a no-gain)",
}
_MAX_KERNEL_WEDGES = int(os.environ.get("PERF_MCP_MAX_KERNEL_WEDGES", "3"))


def _material_gap_ms(device_ms: float) -> float:
    """Smallest op gap worth optimizing, scale-relative to the workload's total device_ms.

    A fixed 0.25ms threshold is right for a whole-model run (tens of ms, where 0.25ms is
    noise) but wrong for per-module optimize (device_ms of a few ms), where real matmul
    gaps of ~0.1-0.2ms — reachable by grid/dtype levers — fall under it and get discarded
    as noise, so the op is never optimized. Scale to a fraction of total device_ms, clamped
    to a small absolute floor and never exceeding the 0.25ms whole-model default (so large
    runs are unchanged). An explicit PERF_MCP_MATERIAL_GAP_MS override is honored verbatim."""
    if _MATERIAL_GAP_ENV_SET:
        return _MATERIAL_GAP_MS
    return min(_MATERIAL_GAP_MS, max(_MATERIAL_GAP_FLOOR, _MATERIAL_GAP_FRAC * float(device_ms or 0.0)))


_TRACE_SAFE_HINT = (
    "custom generic_op/ttl kernels ARE trace-capturable on this build — verified on device: a "
    "cached-descriptor + persistent-buffer generic_op traces clean at PCC 1.0. A wedge or wrong-PCC-on-"
    "replay means the PROVEN trace-safe RECIPE was not applied, NOT that it is impossible or a math error "
    "(check_pcc validates math separately). Apply the recipe: (1) build the generic_op ProgramDescriptor "
    "/ ttl op ONCE per shape and CACHE it — never rebuild or call generic_op fresh each call; (2) use a "
    "PERSISTENT output buffer — allocate once, reuse the same handle, never ttnn.zeros a new output per "
    "call; (3) use a PERSISTENT input buffer — copy the fresh input into the SAME fixed buffer each call, "
    "since trace replay re-reads the captured address; (4) warm up once before begin_trace_capture. With "
    "this recipe generic_op records and replays cleanly under trace; a hang or stale replay means one of "
    "(1)-(3) is violated — fix the recipe application"
)
_ISOLATE_FIRST = (
    " Optionally smoke-test the kernel in ISOLATION first — ONE eager+trace pass (build persistent inputs "
    "once; eager run + PCC vs the stock op; then begin/end_trace_capture + execute_trace + PCC vs eager) — "
    "to catch a wiring mistake in seconds before the full-pipeline run; then wire it into the model with a "
    "PERSISTENT input buffer and measure_candidate."
)
_EAGER_NOTE = (
    " TT_PERF_TRACE=0 (eager): no trace-safety recipe needed — the kernel runs op-by-op, so just author "
    "it, record it, and measure_candidate returns the eager number directly."
)


def _trace_on():
    return os.environ.get("TT_PERF_TRACE", "1") != "0"


# kernel-authoring evidence markers, searched in the model source tree (grounds a recorded attempt)
_KERNEL_MARKERS = ("generic_op", "ProgramDescriptor", "KernelDescriptor", "@ttl.", "ttl.operation", "import ttl")
_TP_SHARD_MARKERS = ("ShardTensorToMesh", "shard_tensor_to_mesh")
_CCL_MARKERS = ("all_gather", "reduce_scatter", "all_reduce")


def _dirty_model_files() -> set:
    """Files under the model dir changed since HEAD (staged, unstaged or untracked).

    Kernel evidence has to come from what THIS attempt touched. A repo-wide substring scan meant
    one orphaned file from a partial revert, or an unrelated all_gather already in the model, made
    every subsequent attempt read as "kernel detected" -- so ops retired on phantom kernels and
    can_stop went true without the work being done.
    """
    try:
        repo = gitio.repo_root(_MODEL_ROOT)
        try:
            spec = str(_MODEL_ROOT.relative_to(repo))
        except ValueError:
            spec = "."
        out = set()
        r = gitio._git(["status", "--porcelain", "--", spec], repo)
        for ln in (r.stdout or "").splitlines():
            rel = ln[3:].strip().split(" -> ")[-1]
            if rel:
                out.add((Path(repo) / rel).resolve())
        return out
    except Exception:  # noqa: BLE001
        return set()


def _scan_kernel_evidence() -> dict:
    """Look for real custom-kernel authoring in the model source so a recorded attempt can't be a
    phantom. Returns {markers, cpp_files, tp_shard, ccl, scope} — empty/False if nothing is
    present. Scoped to files changed since HEAD when any are; falls back to the whole model dir
    only when the tree is clean (nothing to attribute)."""
    found, cpp = set(), []
    tp_shard = ccl = False
    _dirty = _dirty_model_files()
    try:
        for p in _MODEL_ROOT.rglob("*"):
            if _dirty and p.resolve() not in _dirty:
                continue
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
    return {
        "markers": sorted(found),
        "cpp_files": cpp,
        "tp_shard": tp_shard,
        "ccl": ccl,
        "scope": "changed-since-HEAD" if _dirty else "whole-model-dir (clean tree)",
    }


def _load_attempts() -> list:
    if _KERNEL_LOG_PATH.exists():
        try:
            return json.loads(_KERNEL_LOG_PATH.read_text())
        except Exception:  # noqa: BLE001
            return []
    return []


def _load_attempts_all() -> list:
    """Every attempt ever recorded for this model: the archive UNION the live log.

    "Was this rung tried?" is a fact about history and must not depend on which baseline happens to
    be current. The resume filter (run.py) keeps only rows whose baseline_at_record equals the
    baseline measured now and REWRITES the live log with that subset, so _load_attempts() answers a
    baseline-scoped question. Feeding it to the ladder made attempts that genuinely happened read as
    untried the moment the baseline moved -- and it moves every run (381.186 / 381.222 / 381.263 /
    381.291 / 381.311 on gemma-3-12b-it, compared with exact equality). Run 25 went 179 rows -> 121,
    dropping the structural/tt-lang/cpp rows that were capping an op's knob retries, and the same
    matmul came back at `grid` in four consecutive runs.

    The archive is otherwise read in exactly one place -- _rebuild_optimize_report, to draw the
    report -- so nothing that DECIDES anything was consulting it.

    Verdicts keep using _load_attempts(): a "no gain" earned against another baseline still has not
    been shown to hold now. Only the tried/not-tried question changes.
    """
    out, seen = [], set()
    for src in (Path(str(_KERNEL_LOG_PATH) + ".cumulative"), _KERNEL_LOG_PATH):
        try:
            rows = json.loads(src.read_text())
        except Exception:  # noqa: BLE001 -- absent on a first run, and a bad archive must not erase
            continue  # the live rows; this runs on every termination_check
        if not isinstance(rows, list):
            continue
        for a in rows:
            if not isinstance(a, dict):
                continue
            # _fold_cumulative copies live rows into the archive, so overlap is normal. Key on the
            # identity of the ATTEMPT -- op, rung, measurement, note -- so a duplicated row collapses
            # while two genuinely different variants of the same rung both survive and each spends
            # its own retry.
            k = (
                a.get("op_signature"),
                (a.get("kernel_kind") or "").lower(),
                a.get("measured_ms"),
                (a.get("note") or "")[:400],
            )
            if k in seen:
                continue
            seen.add(k)
            out.append(a)
    return out


def _save_attempts(a: list) -> None:
    _KERNEL_LOG_PATH.write_text(json.dumps(a))


_LAST_TARGET_PATH = Path(str(_KERNEL_LOG_PATH) + ".target")
_LAST_TARGET: dict = {}


def _normalise_rung(rung) -> str:
    """`knob:grid` -> `grid`. Rungs are minted prefixed but counted bare, so without this the
    retry counters never saturate and the ladder re-issues the same rung forever."""
    return (str(rung or "").strip().lower().split(":")[-1] or "knob").strip()


_MATMUL_SHAPE_PAT = r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)"


def _warm_start_for(model_root, op_code: str):
    """The pre-pass's PCC-verified (fidelity, dtype) for this op's shape, or None.

    `--matmul-sweep` writes matmul_sweep.json: a table of the best fidelity x dtype per matmul
    shape, each entry already PCC-gated on device. The optimize loop was told to use it only by
    prose in run.py::_PROMPT ("Glob for it once ... look up next_target's shape"). Nothing made that
    happen and nothing recorded whether it did, so a table the agent never opened looked exactly
    like a table with no matching shape -- the pre-pass could be paid for and silently wasted.

    Resolving it HERE makes the recommendation data on next_target instead of a file the agent has
    to remember to read. Deliberately narrow: no file, no seeds/table, or no matching shape returns
    None and the caller behaves exactly as it does today.
    """
    import re as _re  # perf_mcp does not import re at module scope

    if not op_code or "matmul" not in str(op_code).lower():
        return None
    m = _re.search(_MATMUL_SHAPE_PAT, str(op_code))
    if not m:
        return None
    want = tuple(int(g) for g in m.groups())
    try:
        data = json.loads((Path(model_root) / "matmul_sweep.json").read_text())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None

    # matmul_sweep.py writes summarize()'s output: `seeds` (pre-picked winners, shape NESTED under
    # row["shape"], fidelity/dtype at top level) plus the full `table` (shape FLAT on the row, winner
    # under row["best"]). It never writes an `entries` key -- reading that made every lookup return None
    # and silently discarded the whole PCC-gated pre-pass. Read seeds first, fall back to the table row.
    def _shape_of(row):
        sh = row.get("shape") if isinstance(row.get("shape"), dict) else row
        try:
            return (int(sh["m"]), int(sh["k"]), int(sh["n"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _config_of(row):
        src = row.get("best") if isinstance(row.get("best"), dict) else row
        out = {k: src[k] for k in ("fidelity", "dtype") if isinstance(src, dict) and src.get(k)}
        return out or None

    def _rows(key):
        """Only a LIST is iterable here.

        ``data.get(k) or []`` keeps ANY truthy value, so a corrupt file whose `seeds` is a string
        made this ``"nope" + []`` -- a TypeError out of a lookup whose entire contract is to degrade
        to None. A damaged cache would crash the MCP tool instead of starting cold.
        """
        v = data.get(key)
        return v if isinstance(v, list) else []

    for row in _rows("seeds") + _rows("table"):
        if isinstance(row, dict) and _shape_of(row) == want:
            cfg = _config_of(row)
            if cfg:
                return cfg
    return None


def _warm_start_applied(profile, op_code: str, warm_start):
    """Did the op ACTUALLY run at the recommended fidelity? True / False / None (unknown).

    Ground truth, not self-report: tracy records MATH FIDELITY per op (tracy_tool.py:487), so the
    profile says what ran regardless of what anyone claims. record_kernel_attempt takes the agent's
    word for the config, which is exactly the kind of claim this replaces.

    Returns None -- never False -- when the answer is genuinely unknown: the op is absent from the
    profile, or the recommendation is dtype-only. Per-op DTYPE is not captured (roofline.py:232
    reads weight_dtype, which nothing populates), so a dtype recommendation is delivered but cannot
    be verified this way. Absent evidence must not render as "the agent ignored it".
    """
    if not isinstance(profile, dict) or not isinstance(warm_start, dict):
        return None
    want = warm_start.get("fidelity")
    if not want:
        return None
    for o in profile.get("top_ops") or []:
        if not isinstance(o, dict) or o.get("op_code") != op_code:
            continue
        got = o.get("fidelity")
        if not got:
            return None
        return str(got).strip().lower() == str(want).strip().lower()
    return None


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


FAULT_KIND_WEDGED = "wedged"
FAULT_KIND_CRASHED = "crashed"
FAULT_KIND_TIMEOUT = "timeout"

# An op REFUSING A CONFIG, raised by validation before any device work. classify_failure lists
# tt_fatal/tt_throw/tt_assert as device-fault markers -- correct for a runtime abort, wrong for
# these, and that is why an illegal dtype or shard spec reset healthy silicon. The distinguishing
# feature is the SHAPE of the claim: validation states a requirement about the inputs.
_OP_VALIDATION_MARKERS = (
    "must have dtype",
    "must be tile",
    "must be equal",
    "only support",
    "only supports",
    "is not supported",
    "unsupported",
    "must be divisible",
    "must match",
    "invalid shape",
    "shape mismatch",
    "programconfig",
    "must be a multiple",
    "does not fit",
    "out of memory",
    "l1 allocation",
)


def _is_op_validation_reject(reason) -> bool:
    """Did an op reject the CONFIG (device untouched), rather than the device faulting?

    Requires BOTH an assert-family marker and a validation-shaped claim, so a bare TT_FATAL with no
    stated requirement still falls through to the conservative device-fault path.
    """
    s = str(reason or "").lower()
    if not any(m in s for m in ("tt_fatal", "tt_throw", "tt_assert")):
        return False
    return any(m in s for m in _OP_VALIDATION_MARKERS)


def fault_kind_for(reason, killed_by_watchdog: bool = False) -> str:
    """WHICH of the three things went wrong -- they were all called "wedged" (BUG 1).

        timeout  WE killed it (watchdog). The workload never finished, so the lever was never
                 measured and nothing was learned about it. Known at the CALL SITE, no parsing.
        crashed  the op rejected the config -- "TT_FATAL: All input tensors must have dtype =
                 bfloat16". A clean assert from validation; the board is untouched and the answer
                 is real: this lever does not apply to this op.
        wedged   the runtime says the HARDWARE is gone ("Read 0xffffffff over PCIe ID N"), which is
                 what device_recovery.is_dead_board already recognises.

    UNKNOWN stays `wedged` deliberately: absent a signature, under-reacting to a genuinely dead
    board costs more than one unnecessary reset. This only refines what we CALL the failure --
    classify_failure still does the judging.
    """
    if killed_by_watchdog:
        return FAULT_KIND_TIMEOUT
    if _is_op_validation_reject(reason):
        # An op refusing an illegal CONFIG, before it touches the device.
        return FAULT_KIND_CRASHED
    if classify_failure(reason) == FAULT_MEASUREMENT:
        # Positive evidence the host could not take a reading while the device stayed healthy
        # (profiler / CSV / launcher). The board is fine.
        return FAULT_KIND_CRASHED
    # FAULT_DEVICE and FAULT_UNKNOWN both stay `wedged`. Without positive evidence that the device
    # survived, assuming it did is the expensive direction to be wrong in.
    return FAULT_KIND_WEDGED


def _autorecord_wedge(reason: str, killed_by_watchdog: bool = False) -> None:
    t = _load_target()
    kind = fault_kind_for(reason, killed_by_watchdog=killed_by_watchdog)
    rec = {
        "op_signature": t.get("op") or "candidate config",
        "kernel_kind": _normalise_rung(t.get("rung")),
        # An unmeasured attempt is not evidence about the lever. Without this the *_tries
        # counters treated a host-side measurement failure as "tried and lost on merit".
        # A watchdog timeout is unmeasured BY CONSTRUCTION -- we killed it before it could finish.
        "measurement_failed": kind == FAULT_KIND_TIMEOUT or _is_confirmed_measurement_failure(reason),
        "measured_ms": None,
        "beat_baseline": False,
        "note": reason,
        "stages": [],
        "kernel_detected_in_source": False,
        # KEPT TRUTHY for every non-clean outcome: _rung_state, the report renderer and
        # termination_check all read this field, and silently changing what it means would be a
        # worse bug than the one being fixed. fault_kind carries the detail.
        "wedged": True,
        "fault_kind": kind,
        # A lever we killed ourselves deserves another go; an illegal config would fail identically.
        "retryable": kind == FAULT_KIND_TIMEOUT,
        # ONLY a real device fault should reset the board. Resetting on an op-validation assert or
        # on our own watchdog is what produced the "killed holders none" resets on healthy silicon.
        "needs_device_recovery": kind == FAULT_KIND_WEDGED,
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
        if _baseline_path().exists():
            return round(float(json.loads(_baseline_path().read_text()).get("device_ms", 0.0)), 4)
    except Exception:  # noqa: BLE001
        pass
    return None


def _read_baseline_profile():
    try:
        if _baseline_path().exists():
            return json.loads(_baseline_path().read_text())
    except Exception:  # noqa: BLE001
        pass
    return None


def _throughput_path():
    """Per-(model, task) path for the STATIC roofline-target snapshot the report renders. Keyed like
    the baseline path so a per-module run never reads another module's target."""
    model = _MODEL_ROOT.name if _MODEL_ROOT else "model"
    task = os.environ.get("PERF_MCP_TASK", "main")
    return state_dir() / ("perf_mcp_throughput_%s_%s.json" % (model, task))


_MIN_CREDIBLE_DEVICE_MS = float(os.environ.get("PERF_MCP_MIN_CREDIBLE_MS", "1.0") or "1.0")
# A profile whose total is far below the sum of its own buckets recorded only a fragment of the run.
_BUCKET_AGREEMENT_FRAC = 0.5


def _is_credible_profile(prof: dict) -> bool:
    """Is this profile a believable measurement of the model, or an empty capture?

    The only quality check before this was capture_partial -- did the profiler DROP markers. A run
    that drops nothing but RECORDS almost nothing looks clean and is accepted. On llama3_1_8b_p150
    one such capture reported device_ms=0.0612 with its op buckets totalling 2.657 ms, against a real
    profile of 2464 ms. That is 61 microseconds for an 8B model.

    It mattered because of WHERE it landed. The rolling baseline is overwritten by the next profile,
    so a bad one self-corrects; the ORIGINAL baseline is written once and never refreshed, so the
    garbage becomes the permanent anchor for every future headline -- it would have printed
    "0.06 ms -> 714.94 ms" as the result of a run that actually achieved 2464 -> 715.

    Comparability is not plausibility: that value carried a correct depth stamp and a correct method,
    and passed both guards. This one asks whether the number can be true at all.

    Two ways a profile fails: it reports no time, or the total disagrees with the sum of its own
    parts (a capture that recorded a handful of ops and called it the whole model).
    """
    try:
        dev = float(prof.get("device_ms") or 0.0)
    except (TypeError, ValueError):
        return False
    if dev < _MIN_CREDIBLE_DEVICE_MS:
        print(
            "  [perf-mcp] original baseline NOT pinned: device_ms=%.4f is not a credible measurement "
            "(a profile this small is an empty capture, not a fast model)" % dev,
            file=sys.stderr,
            flush=True,
        )
        return False
    buckets = prof.get("buckets") or []
    if buckets:
        total = 0.0
        for b in buckets:
            if not isinstance(b, dict):
                continue
            try:
                total += float(b.get("device_ms") or 0.0)
            except (TypeError, ValueError):
                continue
        if total > 0 and dev < total * _BUCKET_AGREEMENT_FRAC:
            print(
                "  [perf-mcp] original baseline NOT pinned: device_ms=%.4f is far below the sum of its "
                "own op buckets (%.4f) -- the capture is incomplete" % (dev, total),
                file=sys.stderr,
                flush=True,
            )
            return False
    return True


def _ledger():
    """The measurement ledger (cc_optimize/measurements.py), loaded by path so the MCP server keeps
    working under a bare sys.path."""
    global _LEDGER_MOD
    try:
        return _LEDGER_MOD
    except NameError:
        pass
    import importlib.util as _ilu

    _p = Path(__file__).with_name("measurements.py")
    _spec = _ilu.spec_from_file_location("tt_measurements", str(_p))
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    globals()["_LEDGER_MOD"] = _m
    return _m


def _ledger_record(prof: dict) -> None:
    """Append this profile's eager per-op total to the ledger, with the depth it was taken at.

    PHASE is decided by the ledger's own history, not by a caller flag: the first eager reading ever
    taken for this (model, task) is the BEFORE, everything after is an AFTER. That is what makes the
    original survive a rerun -- a second optimize on an already-optimized model appends an 'after',
    it cannot overwrite the original 'before'.
    """
    try:
        led = _ledger()
        ms = prof.get("device_ms")
        depth = str(prof.get("perf_layers") or "all")
        _mname = _MODEL_ROOT.name if _MODEL_ROOT else ""
        seen = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model=_mname)
        phase = led.PHASE_AFTER if seen else led.PHASE_BEFORE
        if phase == led.PHASE_BEFORE and not _is_credible_profile(prof):
            return
        led.record(led.KIND_EAGER, phase, ms, depth=depth, mode="eager", source="profile_model", model=_mname)
        _tr = led.trace_ms_from_profile(prof)
        if _tr:
            led.record(
                led.KIND_TRACE_PASS, phase, _tr, depth=depth, mode="tracy-trace", source="profile_model", model=_mname
            )
    except Exception:  # noqa: BLE001
        pass


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
        # THE MEASUREMENT IS PART OF THE IDENTITY. Keying on the note's first 200 characters alone
        # collapsed distinct attempts whose rationale shared a long prefix -- the agent writes
        # "Hypothesis: ..." openings that routinely match for 200 chars -- while re-measurements that
        # differed only past char 200 survived as separate rows. A restart merges prior attempts into
        # this file, so both directions were visible in one report: the same trace lever appearing
        # several times, its win counted twice across two columns. Hash the WHOLE note and include the
        # measured value, so two rows collapse only when they are genuinely the same attempt.
        key = (
            a.get("op_signature") or a.get("op_code") or "",
            a.get("kernel_kind") or "",
            hashlib.sha1((a.get("note") or "").encode("utf-8", "replace")).hexdigest()[:16],
            a.get("measured_ms"),
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
            throughput=_read_throughput(),
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
        if not _MODEL_ROOT_CONFIGURED:
            # NEVER write a report into whatever directory we happen to be started from. The model
            # root falls back to "." when nothing configured one, so an unconfigured import wrote
            # RUN_REPORT.md into the working tree -- once into the repo itself, where a broad
            # `git add` committed a generated artifact as if it were tool code. The report belongs to
            # a model; with no model there is nothing to report and nowhere it belongs.
            print(
                "  [perf-report] skipped: no model root configured "
                "(set PERF_MCP_MODEL_ROOT or manifest config.model_root)",
                file=sys.stderr,
            )
            return
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
    return (open_op.get("bound_by") or "").lower() == "memory"


def _untried_material_ops(blocking, attempts) -> list:
    """Ops with a material gap that have no recorded attempt at all, in gap order.

    Shape-aware via _op_match, so an attempt on one matmul does not clear a different-shape matmul.
    A `wedged` attempt counts as tried -- the candidate crashed the device, which is a result, and
    demanding a clean one would loop forever on a shape that cannot be built.
    """
    out = []
    for b in blocking or []:
        op = (b or {}).get("op") or ""
        if not op:
            continue
        if not any(_op_match(op, a) for a in (attempts or []) if isinstance(a, dict)):
            out.append(op)
    return out


def _rung_state(matches, kind):
    clean = any((a.get("kernel_kind") or "").lower() == kind and not a.get("wedged") for a in matches)
    wedged = sum(1 for a in matches if (a.get("kernel_kind") or "").lower() == kind and a.get("wedged"))
    return clean, wedged


def _trace_compat_feedback(raw_reason: str) -> str:
    rung = (_load_target().get("rung") or "").lower()
    if rung not in ("tt-lang", "cpp") or not _trace_on():
        return raw_reason
    return "%s\n%s" % (raw_reason, _TRACE_SAFE_HINT)


def _op_ladder_status(open_op: dict, op_code: str, attempts: list) -> tuple[bool, str, str]:
    """DETERMINISTIC ladder gate for ONE open op. Returns (done, rung, reason).

    bound_by sets PRIORITY, never MEMBERSHIP. The bound-conditional gates below decide which rung is
    tried FIRST -- fidelity speeds the math engine, so it cannot help an op waiting on DRAM bytes and
    should not lead there. They must not decide which rungs are tried AT ALL: bound_by is a roofline
    ESTIMATE, ops are rarely purely one-bound, and `compute` is only ever computed for matmuls, so no
    reduction/eltwise op can be compute-bound however it behaves. Used as a filter, that silently
    deleted levers for a whole run -- llama3_1_8b_p150 recorded 0 fidelity attempts across 133, and
    its two costliest ops (TopK, 631ms + 489ms at HiFi2) were structurally ineligible for the rung.
    It also dropped every knob for dispatch-bound ops, which matched no gate at all.

    So after the bound-appropriate rungs are exhausted, the completeness sweep offers each remaining
    knob ONCE (not _MAX_KNOB_RETRIES -- a floor, not a second search) before the op may clear. This
    is the rule record_kernel_attempt already applies to the expensive kernel rungs: a measured
    attempt REPLACES "I reasoned it won't help". The cheapest knob on the ladder should not be the
    only one exempt from it. Host-bucket entries stay exempt: they are not device ops.

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
    grid_tries = sum(
        1 for a in matches if _normalise_rung(a.get("kernel_kind")) == "grid" and not a.get("measurement_failed")
    )
    dtype_tries = sum(
        1 for a in matches if _normalise_rung(a.get("kernel_kind")) == "dtype" and not a.get("measurement_failed")
    )
    fidelity_tries = sum(
        1 for a in matches if _normalise_rung(a.get("kernel_kind")) == "fidelity" and not a.get("measurement_failed")
    )
    shard_tries = sum(
        1 for a in matches if _normalise_rung(a.get("kernel_kind")) == "shard" and not a.get("measurement_failed")
    )
    grid = (open_op.get("grid") or "").lower()
    wdtype = (open_op.get("weight_dtype") or "").lower()
    fidelity = (open_op.get("fidelity") or "").lower()
    bound = (open_op.get("bound_by") or "").lower()
    is_matmul = "matmul" in (op_code or "").lower()
    # BOX (0) HOST / dispatch bucket (GAP-A) — NOT a device op, so the matmul ladder (grid/dtype/
    # tt-lang/C++) is meaningless. Its lever is a STRUCTURAL host-loop transform (trace capture /
    # 2-CQ). Routed via recall_knobs(op_class=host_fallback).
    #
    # MEASUREMENT-GATED, not presence-gated. This used to clear the moment ANY attempt of these kinds
    # existed -- "a trace that doesn't help still counts as tried" -- so one failed attempt sealed the
    # whole dispatch axis permanently. On gemma-3-12b-it that is exactly what happened: run 20 recorded
    # trace / 400.44 ms / beat_baseline=False, and across 158 later attempts dispatch was never offered
    # again while 20.92 ms of host_overhead stayed in every profile. Attempted was being read as
    # resolved.
    #
    # _decode_gate already solves this for the KV-cache lever, in this same file, and its comment names
    # the failure directly: it "clears ONLY when a KV-cache attempt actually reduced cost". Same rule
    # here -- clear on a MEASURED win, or after PERF_MCP_MAX_HOST_ATTEMPTS real tries so a genuinely
    # irreducible dispatch residual cannot loop forever. Wedged attempts count toward the cap, matching
    # how _decode_gate treats a cache that crashes every time.
    if bound == "host" or (open_op.get("bucket") or "").lower() == "host_fallback":
        _host_kinds = {"structural", "trace", "trace-capture"}
        _host_tried = [a for a in matches if (a.get("kernel_kind") or "").lower() in _host_kinds]
        # _ledger().is_win OWNS "is a win"; re-deriving it from beat_baseline here would be a second
        # definition of the same fact (test_single_source_of_truth enforces that). If the ledger is
        # unreadable, treat it as NOT won -- that keeps the rung open, which is the safe direction.
        try:
            _host_won = any(_ledger().is_win(a) for a in _host_tried)
        except Exception:  # noqa: BLE001
            _host_won = False
        try:
            _host_wedged = sum(
                1 for a in _load_attempts() if (a.get("kernel_kind") or "").lower() in _host_kinds and a.get("wedged")
            )
        except Exception:  # noqa: BLE001
            _host_wedged = 0
        _max_host = int(os.environ.get("PERF_MCP_MAX_HOST_ATTEMPTS", "3") or "3")
        if not (_host_won or (len(_host_tried) + _host_wedged) >= _max_host):
            return (
                False,
                "trace-capture",
                "host/dispatch-bound: recall_knobs(op_class='host_fallback') and apply the "
                "TRACE-CAPTURE lever to the generation loop; record_kernel_attempt(...,'trace-capture',...). "
                "This tool measures trace+1CQ end to end (cq is fixed at 1, there is no 2-CQ track), so do NOT "
                "spend this rung on a second command queue — it cannot be measured here. "
                "NOTE: trace removes DISPATCH gaps only. If decode is repeat_prefill (no KV-cache), the bulk of "
                "this bucket is REDUNDANT RECOMPUTE, which trace does NOT remove — that is handled by the SEPARATE "
                "generation_loop 'kv-cache' target, not here.",
            )
        return (
            True,
            "done",
            "DISPATCH lever %s -> remaining DISPATCH residual is bounded by the loop transform. "
            "This does NOT clear a repeat_prefill RECOMPUTE gap: if the generation_loop 'kv-cache' target is still "
            "blocking, the residual here is redundant recompute, reducible ONLY by a KV-cache (NOT irreducible)."
            % ("WON a measured reduction" if _host_won else "tried %d time(s), the cap" % len(_host_tried)),
        )
    tries = {"grid": grid_tries, "fidelity": fidelity_tries, "dtype": dtype_tries, "shard": shard_tries}
    # A DEEPER RUNG ON FILE SPENDS THE SECOND-VARIANT ALLOWANCE. _MAX_KNOB_RETRIES exists so a
    # preferred knob can be tried twice -- the first attempt reads the profile, the second acts on what
    # it learned. That is for an op still ON the knob rungs. Counted per-knob with no reference to how
    # far the op has climbed, it also owes a second grid try to an op that already has a C++ kernel:
    # NLPConcatHeads carried grid=374.56 and cpp=366.85 from earlier runs, run 22 resumed, saw
    # grid_tries==1 < 2, and handed out grid a third time for 377.08 -- slower than both, a full round
    # (edit + PCC + end-to-end) spent walking back down while untried ops waited. Spending the RETRY
    # allowance is not sealing the rung: a knob with ZERO attempts has not been tried at all and is
    # still offered once, which is what the completeness sweep guarantees.
    _went_deeper = bool(kinds & (_STRUCTURAL_RUNGS | {"tt-lang", "cpp", "tp-fracture"}))
    applicable = {
        "grid": grid != "full",
        "fidelity": True,
        "dtype": is_matmul,
        "shard": True,
    }
    preferred = {
        "grid": True,
        "fidelity": bound == "compute",
        "dtype": bound == "memory" and is_matmul and wdtype not in ("bf8_b", "bf4_b"),
        "shard": bound == "memory",
    }
    order = _KNOB_ORDER.get(bound) or _KNOB_ORDER[""]
    for want_preferred in (True, False):
        for knob in order:
            if not applicable[knob] or preferred[knob] is not want_preferred:
                continue
            if tries[knob] >= (_MAX_KNOB_RETRIES if (want_preferred and not _went_deeper) else 1):
                continue
            lead = "" if want_preferred else "LOW-PRIORITY SWEEP (bound_by=%s): " % (bound or "unknown")
            return (False, "knob:" + knob, lead + _KNOB_REASON[knob](grid, fidelity, wdtype))

    if _tp_candidate(open_op, op_code) and "tp-fracture" not in kinds:
        return (
            False,
            "tp-fracture",
            "single-chip levers exhausted and this dense matmul is still memory-bound on "
            "a mesh -> tp_pick_degree(M,K,N) to MEASURE the fastest TP degree (best_tp=1 means keep it "
            "single-chip). If best_tp>1, fracture the weight across the TP axis + insert the matching CCL "
            "(GUIDELINES/08 §7), verify_tp_fracture(M,K,N,best_tp) to PROVE PCC, then commit; ALWAYS "
            "record_kernel_attempt(...,'tp-fracture',...) even on a no-gain result",
        )
    # BOX (2.5) STRUCTURAL / ALGORITHMIC — moved AHEAD of the kernel rungs. The cheap knobs are
    # exhausted but a MATERIAL GAP REMAINS. An algorithmic restructure (KV-cache for a recompute
    # decode, gather for a sparse/MoE matmul that loads experts it never fires, fusion/trace for a
    # dispatch-bound chain) is usually a BIGGER and CHEAPER win than a hand-written kernel -> try it
    # BEFORE tt-lang/C++, so a long ladder can't burn the whole iteration budget on kernels and never
    # reach it. Cleared once a measured 'structural' attempt is on file (a 'none: <evidence>' counts).
    if not (kinds & _STRUCTURAL_RUNGS):
        return (
            False,
            "structural",
            "knobs exhausted (grid+fidelity+dtype+shard) but a gap remains -> try an ALGORITHMIC "
            "restructure BEFORE authoring a kernel. INVESTIGATE this model's architecture/source for "
            "REDUCIBLE WORK (bound_by is a HINT: recompute across steps -> cache, e.g. a KV-cache + "
            "single-token decode_step; a sparse/MoE matmul loading experts that never fire -> gather; "
            "dispatch-bound -> fuse adjacent ops / trace the region). Improvise the restructure for THIS "
            "model, measure it, and record_kernel_attempt(...,'structural', note=<what you found + did>). "
            "If there is genuinely no reducible work, record 'structural' with note='none: <evidence>'.",
        )
    if _is_kernel_able(op_code):
        _tr = _trace_on()
        _suffix = _ISOLATE_FIRST if _tr else _EAGER_NOTE
        _tl_clean, _tl_wedged = _rung_state(matches, "tt-lang")
        _cpp_clean, _cpp_wedged = _rung_state(matches, "cpp")
        _tl_done = _tl_clean or _tl_wedged >= _MAX_KERNEL_WEDGES
        _cpp_done = _cpp_clean or _cpp_wedged >= _MAX_KERNEL_WEDGES
        if not _tl_done and _ttl_available():
            if _tl_wedged:
                if _tr:
                    _r = (
                        "apply the PROVEN trace-safe recipe (do NOT switch to cpp — the same recipe applies): %s (attempt %d)"
                        % (
                            _TRACE_SAFE_HINT,
                            _tl_wedged + 1,
                        )
                    )
                else:
                    _r = (
                        "the tt-lang kernel crashed in EAGER (TT_PERF_TRACE=0) — a real math/runtime error, not a trace issue; fix it from the traceback/check_pcc and retry (attempt %d)"
                        % (_tl_wedged + 1)
                    )
                return (False, "tt-lang", _r)
            return (
                False,
                "tt-lang",
                "knobs exhausted (grid+dtype); author a tt-lang kernel (GUIDELINES/11) and record it." + _suffix,
            )
        if (_tl_done or not _ttl_available()) and not _cpp_done:
            if _cpp_wedged:
                if _tr:
                    _r = (
                        "apply the PROVEN trace-safe recipe (fix the recipe application, do NOT bounce rungs): %s (attempt %d)"
                        % (
                            _TRACE_SAFE_HINT,
                            _cpp_wedged + 1,
                        )
                    )
                else:
                    _r = (
                        "the C++ kernel crashed in EAGER (TT_PERF_TRACE=0) — a real math/runtime error, not a trace issue; fix it from the traceback/check_pcc and retry (attempt %d)"
                        % (_cpp_wedged + 1)
                    )
                return (False, "cpp", _r)
            return (
                False,
                "cpp",
                "tt-lang measured; author a C++ Metalium kernel via ttnn.generic_op (GUIDELINES/12) and record it."
                + _suffix,
            )
    # every box ticked (knobs + structural + tt-lang + C++) -> genuine irreducible residual -> DONE.
    # STRUCTURAL is asserted ABOVE (before the kernel rungs), so reaching here means it is already tried.
    return (
        True,
        "done",
        "checklist complete (grid+fidelity+dtype+shard+structural+tt-lang+C++) -> irreducible",
    )


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


def _stable_artifact_dir():
    """Resolved per call: a module constant freezes the path at import, before any redirect."""
    return state_dir() / "perf_mcp_last_profile"


def _persist_artifacts(prof: dict) -> dict:
    """Copy prof['artifacts'] CSVs out of the about-to-be-reaped tmpdir into one fixed dir
    (overwritten each call) and repoint the paths there, so they stay readable after the
    reap. Best-effort: never raises, so profiling is unaffected if persistence fails."""
    arts = prof.get("artifacts")
    if not isinstance(arts, dict):
        return prof
    try:
        _shutil.rmtree(_stable_artifact_dir(), ignore_errors=True)
        _stable_artifact_dir().mkdir(parents=True, exist_ok=True)
        repointed = {}
        for key, src in arts.items():
            sp = Path(str(src))
            try:
                if sp.is_file():
                    dst = _stable_artifact_dir() / sp.name
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

        _logs = sorted(d.glob("*_tracy.log"))
        _readable = 0
        for log in _logs:
            try:
                _txt = log.read_text(errors="ignore")
            except OSError:
                continue
            if not _txt.strip():
                continue
            _readable += 1
            hit = detect_marker_drop(_txt)
            if hit:
                return hit
        if not _readable:
            return "no readable tracy log to check for marker drops -- capture integrity UNKNOWN"
    except Exception as exc:  # noqa: BLE001
        # A detector that FAILED did not observe a clean capture. Returning None here (and on a
        # single ImportError, for the whole session) silently disabled partial-capture detection,
        # which is what let a truncated capture through as a low device_ms and a false win.
        return "partial-capture detection FAILED (%s) -- capture integrity UNKNOWN" % (str(exc)[:120],)
    return None


def _profile_cache_dir():
    """Resolved per call: a module constant freezes the path at import, before any redirect."""
    return state_dir() / "perf_mcp_profile_cache"


def _win_threshold(base_ms: float, spread_ms=None) -> float:
    """Smallest delta worth calling a win. SINGLE source of truth: agent.integrity.win_threshold.

    A bare 0.05 ms was unit-agnostic -- on a 2266 ms llama baseline that accepts three thousandths
    of one percent, far inside this board's thermal drift. It also lived in two modules, so fixing
    one left the optimize path still banking noise; hence one implementation, imported.
    """
    return _integrity.win_threshold(base_ms, spread_ms)


_SOURCE_EXTS = (".py", ".cpp", ".hpp", ".h", ".cc", ".cu")


def _authored_source_files(root: Path) -> list:
    """Source files a human (or the agent) authored under `root`, per git.

    Tracked files plus untracked-and-not-ignored ones. Generated artifacts are gitignored, so they
    are excluded by construction rather than by a list of patterns this code would have to keep
    guessing at. Falls back to a non-recursive glob only when git cannot answer.
    """
    try:
        repo = gitio.repo_root(root)
        rel = str(root.relative_to(repo))
        out = set()
        for args in (
            ["ls-files", "--", rel],
            ["ls-files", "--others", "--exclude-standard", "--", rel],
        ):
            r = gitio._git(args, repo)
            if r.returncode != 0:
                continue
            for line in (r.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                f = Path(repo) / line
                if f.is_file() and f.suffix in _SOURCE_EXTS and "__pycache__" not in f.parts:
                    out.add(f)
        if out:
            return list(out)
    except Exception:  # noqa: BLE001
        pass
    return [f for d in ("_stubs", "tt") for f in (root / d).glob("*.py") if f.is_file()]


def _model_source_fingerprint() -> str:
    """Cache key for a profiling run: hashes the model's stub/tt source AND the identity
    of the module + perf-test being profiled.

    Module-level optimize runs every module against the SAME source tree but a DIFFERENT
    per-module perf test, so a source-only key made all modules collide on one entry — a
    module could be handed another module's cached profile (e.g. di_t_model's matmuls
    returned for ace_step_attention). Folding PERF_MCP_TASK + the resolved perf-test node
    into the key scopes the cache per module. Empty string disables caching."""
    h = hashlib.sha256()
    try:
        root = _MODEL_ROOT
        # AUTHORED INPUTS ONLY. A plain recursive walk hashes whatever is on disk, including files
        # the profiling run itself writes, so the key changed as a side effect of measuring and the
        # cache could never hit. git already distinguishes authored source from generated artifacts:
        # tracked files, plus untracked ones that .gitignore does not exclude. That also keeps a
        # hand-authored .cpp / tt-lang kernel IN the key -- the omission that made the custom-kernel
        # rung unmeasurable, because an authored kernel produced a byte-identical fingerprint and
        # measure_candidate returned the PRE-kernel cached profile.
        files = sorted(_authored_source_files(root))
        if not files:
            return ""
        for f in files:
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    except Exception:
        return ""
    ptr = _MANIFEST.get("perf_test_resolved", {}) or {}
    # Env that changes WHAT is measured must be in the key: without TT_PERF_TRACE/TT_PERF_LAYERS
    # a later eager or different-depth run was served the earlier trace-mode profile as its own.
    _env_keys = (
        "PERF_MCP_TASK",
        "TT_PERF_TRACE",
        "TT_PERF_LAYERS",
        "TT_PERF_SEQ_LEN",
        "PERF_MCP_DEVICES",
        "PERF_MCP_PROFILE_ENV",
    )
    for part in [os.environ.get(k, "") for k in _env_keys] + [
        ptr.get("path", ""),
        ptr.get("case", ""),
    ]:
        h.update(b"\x00")
        h.update(str(part).encode())
    return h.hexdigest()


def _profile_cache_get(fp: str):
    if not fp:
        return None
    p = _profile_cache_dir() / (fp + ".json")
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
        _profile_cache_dir().mkdir(parents=True, exist_ok=True)
        (_profile_cache_dir() / (fp + ".json")).write_text(json.dumps(prof))
    except Exception:
        pass


# --- BUG 1/BUG 5 (2026-07-25) -------------------------------------------------
# A profile can fail to produce a MEASUREMENT without anything being wrong with the
# device. Upstream tt-metal writes a HEADERLESS ops csv (exactly b"\r\n") and logs
# success when the report has zero op rows, so the harness sees "unexpected CSV header
# ... '\n'". That is a host-side extraction failure: it must be retried and reported as
# unmeasured, never recorded as a device wedge (which burned the lever and reset the
# board after two occurrences).
_MEASUREMENT_FAILURE_MARKERS = (
    "unexpected csv header",
    "ops csv missing/empty",
    "no ops_perf_results",
    "could not read the test list",
)
_ZERO_ROW_RETRIES = int(os.environ.get("PERF_MCP_ZERO_ROW_RETRIES", "2") or "2")


_DEVICE_FAULT_MARKERS = (
    "segmentation fault",
    "core dumped",
    "aborted",
    "terminate called",
    "tt_fatal",
    "tt_throw",
    "tt_assert",
    "device watchdog",
    "hang detected",
    "fabric router sync: timeout",
    "harvesting",
    "umd",
    "pcie",
    "eth link",
    "unrecoverable",
    "dma",
)


FAULT_DEVICE = "device_fault"
FAULT_MEASUREMENT = "measurement_failure"
FAULT_UNKNOWN = "unknown"


def classify_failure(msg) -> str:
    """Was this a DEVICE fault or a failed MEASUREMENT? Judged by an agent, not by substrings.

    Three states (FAULT_DEVICE / FAULT_MEASUREMENT / FAULT_UNKNOWN) because the two decisions this
    feeds need opposite conservatism: resetting the board needs positive device evidence (a false
    reset is expensive), while forgiving a lever needs positive measurement-failure evidence
    (forgiving by default makes every wedge invisible to the ladder).

    The substring lists are the OFFLINE fallback only. They are why this was broken twice already:
    an allow-list of four markers sent every other host-side fault down the device-wedge path, and
    the next TT_FATAL / launcher / report-tool failure is phrased a way no list anticipated. The
    agent reads the actual text; the answer is cached by content hash so a repeat costs nothing.
    """
    s = str(msg or "").lower()
    if not s.strip():
        return FAULT_UNKNOWN

    verdict = _integrity.classify(
        s[:400],
        {FAULT_DEVICE, FAULT_MEASUREMENT},
        what="failure",
        evidence=(
            "device_fault = the accelerator itself faulted, hung or was reset (the board needs "
            "recovery). measurement_failure = the device was fine but the host could not extract a "
            "reading (profiler/report/CSV/launcher/parse problem), so the edit was simply not "
            "measured. Text follows:\n" + s[:400]
        ),
    )
    if verdict in (FAULT_DEVICE, FAULT_MEASUREMENT):
        return verdict

    # offline / agent-unavailable fallback: positive evidence only, never a default
    if any(m in s for m in _DEVICE_FAULT_MARKERS):
        return FAULT_DEVICE
    if any(m in s for m in _MEASUREMENT_FAILURE_MARKERS):
        return FAULT_MEASUREMENT
    return FAULT_UNKNOWN


def _is_measurement_failure(msg) -> bool:
    """Should this be reported as an unmeasured attempt rather than a device wedge?

    True for a known measurement failure AND for UNKNOWN: absent a device-fault signature the board
    is fine, so calling it a wedge burns the lever, bumps the crash counter and resets the board on
    repeat (the original BUG 1). This answers the REPORTING question only -- lever accounting uses
    `_is_confirmed_measurement_failure`, which does not forgive UNKNOWN.
    """
    return classify_failure(msg) != FAULT_DEVICE


def _is_confirmed_measurement_failure(msg) -> bool:
    """Positive evidence that no measurement happened. Used for ladder accounting, where forgiving
    an UNKNOWN attempt would hide every wedge from the retry counters."""
    return classify_failure(msg) == FAULT_MEASUREMENT


def _measurement_failed_result(msg) -> dict:
    reason = (
        "MEASUREMENT_FAILED: the profiler returned no readable op rows, so your edit was NOT measured "
        "(this says nothing about whether the edit is good). The harness already retried. "
        f"Detail: {str(msg)[-300:]}"
    )
    return {"verdict": "MEASUREMENT_FAILED", "measured": False, "retryable": True, "reason": reason}


def _profile_with_zero_row_retry(retries: int = None) -> dict:
    """Profile, retrying when the run yields no readable op rows.

    A zero-row profile is transient far more often than not; one retry would have saved
    all 9 llama attempts on 2026-07-25. Raises the original error once retries are spent
    so the caller can classify it via _is_measurement_failure().
    """
    n = _ZERO_ROW_RETRIES if retries is None else retries
    last = None
    for attempt in range(n + 1):
        try:
            return _profile_once()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt >= n or not _is_measurement_failure(exc):
                raise
            print(
                f"  [perf-mcp] profile produced no readable op rows "
                f"({str(exc)[-120:]}); re-profiling {attempt + 1}/{n}",
                file=sys.stderr,
                flush=True,
            )
    raise last


def _profile_once() -> dict:
    _cache_on = os.environ.get("PERF_MCP_NO_PROFILE_CACHE") != "1"
    _fp = _model_source_fingerprint() if _cache_on else ""
    if _fp:
        _hit = _profile_cache_get(_fp)
        if _hit is not None:
            return _hit
    ctx = _Ctx()
    tmpdir = ctx.run.dir
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
        prof = _profile_with_zero_row_retry()
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
    prof.setdefault("perf_layers", _depth_in_force())
    prof = _promote_baseline(prof)
    _ledger_record(prof)
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
            for o in (rep.get("open_ops") or [])
        ]
        _persist_throughput(rep, prof)  # fresh static target for the RUN_REPORT roofline table (non-stale)
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
        prof = _profile_with_zero_row_retry()
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
        if _is_measurement_failure(_msg):
            # host-side extraction failure: NOT a device wedge. Do not mark a device crash
            # (2 in a row triggers a board reset), do not record a wedge, do not burn the lever.
            return _measurement_failed_result(_msg)
        _note_device_crash("measure_candidate", _msg)  # evidence-driven: resets the chip the error names
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
    if not _baseline_path().exists():
        return {
            "verdict": "NO_BASELINE",
            "measured": True,
            "device_ms": dev,
            "is_real_gain": False,
            "reason": (
                "no baseline recorded, so this reading cannot be compared to anything and is NOT a "
                "win: call profile_model to establish the baseline, then re-measure."
            ),
        }
    baseline = json.loads(_baseline_path().read_text())
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
    faster = delta > _win_threshold(base_dev)
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
        # same threshold in both directions -- a 4th hardcoded 0.05 lived here, so a change could be
        # labelled SLOWER while an equal-sized gain was correctly dismissed as noise.
        "note": (
            "FASTER — real gain"
            if faster
            else ("SLOWER" if delta < -_win_threshold(base_dev) else "no gain (within noise)")
        ),
    }


@mcp.tool()
def check_pcc() -> dict:
    """Run the model's end-to-end PCC correctness test on-device (the SAME gate the FSM uses).
    Returns {status: ok|pcc_low|crash, pcc?}. An edit is only acceptable if status==ok. A crash
    or pcc_low means the edit broke correctness — fix or revert it; never keep it."""
    try:
        res = run_pcc(_Ctx())
    except Exception as exc:  # noqa: BLE001
        _msg = str(exc)[-800:]
        if _is_measurement_failure(_msg):
            out = _measurement_failed_result(_msg)
            out["status"] = "measurement_failed"
            record_gate_verdict("pcc", "measurement_failed")
            return out
        _note_device_crash("check_pcc", _msg)
        record_gate_verdict("pcc", "crash")
        return {"status": "crash", "error": _msg}
    if res.get("status") == "crash":
        _note_device_crash("check_pcc", str(res.get("error") or ""))
    else:
        _note_device_ok()
    record_gate_verdict("pcc", res.get("status"), pcc=res.get("pcc"))
    return res


def _pg_progress_watch(pgid, stall_s=0.0):
    """probes owns the arithmetic; see ProgressWatch there.

    THIS LOOP USED CPU, AND CPU IS NOT PROGRESS. It read `cpu > last_cpu + 10` as "still working",
    which is the signal that let run 12 spin for nine hours: a livelock burns CPU by definition. The
    backstop below bounded this loop so it could not hang outright, but the stall check was blind for
    the whole hour leading up to it. The signature moves only when work does.

    The fallback answers "moved" every poll: an unreadable signature must not be read as a wedge.
    The backstop still bounds the loop.
    """
    try:
        from agent.probes import ProgressWatch

        return ProgressWatch(pgid, None, stall_s)
    except Exception:  # noqa: BLE001

        class _Blind:
            def moved(self, *_a, **_k):
                return True

        return _Blind()


class _AdaptiveResult:
    def __init__(self, returncode, stdout):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def _adaptive_run(cmd, cwd, env, label="device run", stall_s=None, backstop=None):
    import threading as _th
    import time as _t

    if stall_s is None:
        _ov = os.environ.get("PERF_MCP_MEASURE_STALL_SEC")
        # No caller-supplied budget (full-pipeline run, op-sig probe): derive it, never assume 600 s
        # of silence means dead -- weight load and kernel compile are legitimately quiet for longer.
        from agent.probes import adaptive_op_timeout as _aot

        stall_s = int(_ov) if _ov else int(_aot("profile"))
    stall_s = int(stall_s)
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
    _watch = _pg_progress_watch(pgid, stall_s)
    max_gap = 0.0
    while proc.poll() is None:
        _t.sleep(5)
        now = _t.monotonic()
        moved = _watch.moved(now, last_progress, proc.pid) or act[0] > last_progress
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


def _grow_trace_region_and_retry(cmd, repo, env, out, r):
    """Re-run with a bigger trace region until the capture fits, growing up to the DRAM-derived
    ceiling. Model- and hardware-agnostic. Fires on EITHER of the two ways a too-small region shows up:

      EXPLICIT: the device prints "Creating trace buffers of size X B ... only Y B" -> grow to X.
      SILENT:   the overflow HANGS the mesh (no message) and the run is killed with NO trace number --
                reset the (wedged) device and grow by DOUBLING anyway. A hung overflow prints nothing,
                so the byte-parse alone can't see it; this is why an earlier run wedged for 600s and was
                killed instead of grown.

    A run that produced a trace number, or that declared the pipeline genuinely un-traceable
    (TRACE_NOT_TRACE_CAPABLE), is left alone. Returns the last (output, result).
    """
    try:
        from agent.perf_test_gen import _needed_trace_region, _TRACE_REGION_GROW_ROUNDS
    except Exception:  # noqa: BLE001
        return out, r
    for _ in range(int(_TRACE_REGION_GROW_ROUNDS or 0)):
        text = out or ""
        if "TRACE_PER_TOKEN_MS=" in text or "TRACE_NOT_TRACE_CAPABLE" in text:
            break  # a real trace ran, or the pipeline genuinely cannot trace -- nothing to grow
        cur = int(env.get("TT_PERF_TRACE_REGION") or os.environ.get("TT_PERF_TRACE_REGION") or _TRACE_REGION_DEFAULT)
        need = _needed_trace_region(text)
        silent = need is None
        target = min(max(int(need), cur * 2) if need is not None else cur * 2, _TRACE_REGION_MAX)
        if target <= cur:
            break  # already at the DRAM ceiling -> the trace genuinely does not fit
        env["TT_PERF_TRACE_REGION"] = str(target)
        sys.stderr.write(
            "[perf-mcp] trace region too small (%s); growing to %d B%s and re-running\n"
            % (
                "silent overflow / mesh hang -- no size reported" if silent else "device reports %d B" % need,
                target,
                " + resetting the wedged device" if silent else "",
            )
        )
        if silent:
            # A silent overflow hangs the mesh; heal it before the retry or the re-run hangs too.
            try:
                from agent.probes import _device_reset

                _device_reset(error_text=text)
            except Exception:  # noqa: BLE001
                pass
        try:
            r = _adaptive_run(cmd, repo, env, "full-pipeline")
        except Exception:  # noqa: BLE001
            break
        out = (r.stdout or "") + "\n" + (r.stderr or "")
    return out, r


def _workload_failure_tail(out: str, keep: int = 12) -> str:
    """The lines from a failed workload that explain WHY, for attaching to the gate's error.

    Prefers real error signatures over the nanobind teardown spam that otherwise fills the tail.
    """
    if not out:
        return ""
    noise = ("nanobind", "leaked", "reference counting", "skipped remainder")
    lines = [ln.rstrip() for ln in out.splitlines() if ln.strip() and not any(n in ln.lower() for n in noise)]
    if not lines:
        return ""
    sig = [
        ln
        for ln in lines
        if any(k in ln for k in ("TT_FATAL", "TT_THROW", "Error", "error:", "Traceback", "Exception", "assert"))
    ]
    picked = sig[-keep:] if sig else lines[-keep:]
    return "\n  workload output:\n    " + "\n    ".join(x[:220] for x in picked)


def _stage_ms_path():
    """Where the MEASURED per-phase timings live, keyed like every other per-run artifact."""
    model = _MODEL_ROOT.name if _MODEL_ROOT else "model"
    task = os.environ.get("PERF_MCP_TASK", "main")
    return state_dir() / ("perf_mcp_stage_ms_%s_%s.json" % (model, task))


def _depth_in_force() -> str:
    """The one answer to "what depth is this". See layer_depth.depth_in_force."""
    try:
        from agent.layer_depth import depth_in_force

        return depth_in_force()
    except Exception:  # noqa: BLE001 -- unknown depth reads as full, exactly as it did before
        return "all"


def _stage_run_stamp() -> str:
    """Which run a stage measurement belongs to, or "" when nothing said."""
    return str(os.environ.get("PERF_MCP_RUN_ID") or "").strip()


def _read_stage_doc(state_dir_path=None, model="", task="") -> dict:
    """The stage file for (model, task), or {} when it belongs to a DIFFERENT run.

    One reader for all three of stages/paths/isl, so the freshness rule cannot be enforced in one
    place and forgotten in the next two -- which is exactly how the timings, their trace paths and
    the observed ISL came to be read back with three separate copies of the same lookup.

    A file stamped with a DIFFERENT run is refused. So is an UNSTAMPED one, which this used to accept
    on the reasoning that it predates stamping and refusing it would blank the report for anyone who
    had not re-run. That reasoning was wrong, and the cost showed up immediately: gemma-3's report
    rendered `prefill 100.46 ms` from a file written 2026-08-07T00:07 -- forty hours BEFORE the fix
    that let prefill trace at all -- beside a headline from a run that had measured nothing. Because
    the file also predated `paths`, the roofline could not mark it eager, so a pre-fix eager number
    rendered wearing a traced one's clothes and read as "prefill is still eager today".

    A blank says `not measured`, which is TRUE and prompts a measurement. This said a number, which
    was false. Absent evidence is not weak evidence -- it is the absence of a claim.
    """
    try:
        from pathlib import Path as _P

        base = _P(state_dir_path) if state_dir_path else state_dir()
        m = model or (_MODEL_ROOT.name if _MODEL_ROOT else "model")
        t = task or os.environ.get("PERF_MCP_TASK", "main")
        doc = json.loads((base / ("perf_mcp_stage_ms_%s_%s.json" % (m, t))).read_text())
        if not isinstance(doc, dict):
            return {}
        _stamp, _now = str(doc.get("run") or ""), _stage_run_stamp()
        if not _stamp or (_now and _stamp != _now):
            return {}
        return doc
    except Exception:  # noqa: BLE001
        return {}


def _parse_gathered_weights(out: str) -> dict:
    """{subtree: resident bytes read PER ITEM} from the pipeline's marker, or {} when it printed none.

    Resident is not the same as streamed. The ceiling assumes one unit of work reads every weight in
    its subtree once, which is true of a matmul weight and false of a lookup table -- voxtral's
    embed_tokens is 805 MB and a decode step reads one 6 KB row of it. Nothing in the checkpoint
    separates the two: embed_tokens and lm_head are both [131072, 3072], same size, same section.
    The pipeline holds them and says which is which.
    """
    out_map: dict = {}
    try:
        for line in (out or "").splitlines():
            if "TRACE_GATHERED_WEIGHTS=" not in line:
                continue
            for pair in line.split("TRACE_GATHERED_WEIGHTS=", 1)[1].split()[0].split(","):
                k, _, v = pair.partition(":")
                try:
                    if k.strip() and int(v) > 0:
                        out_map[k.strip()] = int(v)
                except (TypeError, ValueError):
                    continue
    except Exception:  # noqa: BLE001 -- a malformed marker leaves the old behaviour
        return {}
    return out_map


def _parse_census_sections(out: str) -> dict:
    """{attribute name: resident bytes} from the census marker, or {} when it did not print one.

    Tolerant by design: a malformed pair is skipped rather than discarding the line, because a
    partial split still prices the towers it names and the alternative is the checkpoint ratio."""
    secs: dict = {}
    try:
        for line in (out or "").splitlines():
            if "TRACE_WEIGHT_SECTIONS=" not in line:
                continue
            for pair in line.split("TRACE_WEIGHT_SECTIONS=", 1)[1].split()[0].split(","):
                name, _, raw = pair.partition(":")
                if name.strip() and raw.strip().isdigit() and int(raw) > 0:
                    secs[name.strip()] = int(raw)
    except Exception:  # noqa: BLE001 -- a census line the report can do without
        return {}
    return secs


def _persist_device_weight_bytes(
    nbytes: int,
    complete: bool,
    bytes_per_param: float = 0.0,
    sections: dict | None = None,
    gathered: dict | None = None,
) -> None:
    """Merge the census result into the model's own facts file. Best-effort; never raises.

    Written to the MODEL directory, and only when that directory was STATED -- a model fact resolved
    against the working directory is how a 31 MB, 32-layer file came to describe gemma-3-12b. The
    two keys are merged rather than the file rewritten, so a hand-tuned per-tensor list beside them
    survives."""
    if not _MODEL_ROOT_STATED or not nbytes or nbytes <= 0:
        return
    try:
        p = _MODEL_ROOT / "perf_target_inputs.json"
        doc = {}
        if p.is_file():
            try:
                doc = json.loads(p.read_text()) or {}
            except Exception:  # noqa: BLE001
                doc = {}
        if not isinstance(doc, dict):
            return
        if (
            doc.get("device_weight_bytes") == int(nbytes)
            and doc.get("device_census_complete") == bool(complete)
            and (not bytes_per_param or doc.get("bytes_per_param") == float(bytes_per_param))
            and (not sections or doc.get("device_section_bytes"))
        ):
            return
        # THE FIRST COMPLETE CENSUS WINS, and nothing later moves it.
        #
        # This is called from _run_full_pipeline_ms, so it fires on EVERY full-pipeline gate -- every
        # iteration of the loop. The census measures the model AS CURRENTLY BUILT, so the moment a
        # dtype rung lands (bf16 -> bf8 halves the resident weights) the figure drops. Left
        # overwriting, the ceiling derived from it would fall in step with the optimisation being
        # scored against it: the run would chase a target that recedes as it improves, and a report
        # written at iteration 9 could not be compared with one written at iteration 1.
        #
        # The ceiling describes the model the run STARTED with. An incomplete census may still be
        # replaced -- it is not yet an answer -- but a complete one is the answer for this run.
        if doc.get("device_weight_bytes") and doc.get("device_census_complete"):
            # SAME MEASUREMENT, MORE OF IT. An identical total is not a second census -- it is this
            # one, described further. Refusing the split here would be refusing detail that changes
            # no number already written, and it would strand every model whose facts file was pinned
            # by a tool version that had no split to record: the file is already complete, so the
            # split could never arrive and the ceiling would silently keep apportioning by the
            # checkpoint. A DIFFERENT total is a different census and is still refused.
            if (
                sections
                and not doc.get("device_section_bytes")
                and int(doc.get("device_weight_bytes") or 0) == int(nbytes)
            ):
                doc["device_section_bytes"] = {str(k): int(v) for k, v in sections.items() if int(v) > 0}
                tmp = p.with_suffix(p.suffix + ".tmp")
                tmp.write_text(json.dumps(doc, indent=2) + "\n")
                os.replace(str(tmp), str(p))
            return
        doc["device_weight_bytes"] = int(nbytes)
        doc["device_census_complete"] = bool(complete)
        # PINNED WITH THE TOTAL, by being written inside the same guard. The split describes the
        # model the run STARTED with for exactly the reason the total does: a dtype rung that halves
        # one tower changes the proportions, and a share that moves between iterations scores the
        # same run against two different ceilings.
        if sections:
            doc["device_section_bytes"] = {str(k): int(v) for k, v in sections.items() if int(v) > 0}
        # Pinned with the total for the same reason the split is: a lever that replaces a lookup
        # table changes this, and a value that moves between iterations scores one run against two
        # ceilings.
        if gathered:
            doc["gathered_weight_bytes"] = {str(k): int(v) for k, v in gathered.items() if int(v) > 0}
        if bytes_per_param and bytes_per_param > 0:
            doc["bytes_per_param"] = float(bytes_per_param)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        os.replace(str(tmp), str(p))
    except Exception:  # noqa: BLE001
        pass


def _persist_stage_ms(
    stage_ms: dict,
    stage_paths: dict | None = None,
    stage_isl: dict | None = None,
    stage_ops: dict | None = None,
    stage_batch: int = 0,
    prompt_tokens: int = 0,
    stage_isl_per_request: dict | None = None,
    stage_bytes: dict | None = None,
) -> None:
    """Record trace_replay's per-stage timings so the report can show a MEASURED phase split.

    Written to a file rather than threaded through _run_full_pipeline_ms's return, because that
    function has five return sites and the report is rendered by a different process.

    Why this matters: the only per-stage rows the report could show came from the agent's
    stages_json -- free text it writes, which nothing validates. Decode ms and prefill ms are not
    the same currency (decode recurs per token, prefill once per request), so a phase split guessed
    from prose can put time in the wrong pool and be acted on. These names come from the
    PIPELINE_STAGES the model declares, measured by the harness.
    """
    if not stage_ms:
        return
    try:
        p = _stage_ms_path()
        tmp = p.with_suffix(p.suffix + ".tmp")
        # STAMPED WITH THE RUN THAT MEASURED IT. The file is keyed by (model, task), which outlives
        # the run, and nothing recorded which run wrote it -- so a report read whatever was there
        # last, however old. That is not hypothetical: a prefill of 91.33 ms from a previous run was
        # rendered beside a fresh decode for hours, and the same stage appeared with two different
        # values in one report because the headline came from this run and the split from another.
        # Same lifetime problem, and same fix, as the device-recovery counters.
        tmp.write_text(
            json.dumps(
                {
                    "run": _stage_run_stamp(),
                    "stages": stage_ms,
                    "paths": stage_paths or {},
                    "isl": stage_isl or {},
                    # The prompt this run served, kept beside the per-stage counts rather than inside
                    # them: it belongs to the request, and every stage sees it.
                    "prompt_tokens": int(prompt_tokens or 0),
                    # PER-REQUEST counts, from the legacy marker. "isl" above holds TOTALS a stage
                    # stated for one call; these must be multiplied by the batch and those must not.
                    "isl_per_request": stage_isl_per_request or {},
                    # THE READ SET, MEASURED. Distinct device tensors each stage's ops touched, from
                    # trace_replay's dispatch hook. The roofline's memory floor divides by this;
                    # every other source for it is an inference from the checkpoint plus a naming
                    # convention, and the one that matters most -- what a TOKEN reads -- had no
                    # measurement at all until this.
                    "bytes": stage_bytes or {},
                    # The doc tracks the CURRENT build by design -- the report needs both numbers.
                    # The pinned baseline lives in the ledger, written just below.
                    # THE BATCH THE RUN ACTUALLY SERVED. Parsed off TRACE_REPLAY_PATH for the
                    # scorecard and then dropped, so the report had to fall back to TT_PERF_BATCH --
                    # which carries 0, the "ask the pipeline" sentinel, and reads as 1. Every ceiling
                    # was then priced for one user against a measurement of eight. Recorded here
                    # because it belongs to the same measurement as the stage timings beside it.
                    "batch": int(stage_batch or 0),
                    # The op-dispatch count each path label was DERIVED from -- the evidence, kept
                    # beside the conclusion, so "prefill is traced" can be checked rather than
                    # believed.
                    "ops": stage_ops or {},
                }
            )
        )
        os.replace(str(tmp), str(p))
    except Exception:  # noqa: BLE001
        pass


def read_stage_ms(state_dir_path=None, model="", task="") -> dict:
    """The measured per-phase split, or {}. Read by the report renderer."""
    try:
        return {
            k: float(v)
            for k, v in (_read_stage_doc(state_dir_path, model, task).get("stages") or {}).items()
            if float(v) > 0
        }
    except Exception:  # noqa: BLE001
        return {}


def read_stage_bytes(state_dir_path=None, model="", task="") -> dict:
    """The measured per-stage read set in BYTES, or {}. Read by the report renderer.

    The distinct device tensors each stage's ops touched, observed by trace_replay's dispatch hook
    while the stage ran in isolation. This is the roofline's memory-floor divisor, MEASURED -- the
    alternative sources are the checkpoint's total (which counts towers a token never reads), a name
    list (`audio_tower|vision_tower|...`, which a model can spell differently), and a stage_roots
    section map (which drops lm_head on any untied model).
    """
    try:
        return {
            k: int(v)
            for k, v in (_read_stage_doc(state_dir_path, model, task).get("bytes") or {}).items()
            if int(v) > 0
        }
    except Exception:  # noqa: BLE001
        return {}


# The key this tool used for the prompt length before `prompt_tokens` existed. An on-disk schema
# name from an older version of THIS file, read only when the modern field is absent.
_LEGACY_PROMPT_KEY = "prefill"


def read_stage_isl(state_dir_path=None, model="", task="") -> int:
    """The prompt length the run ACTUALLY profiled, or 0.

    Read back from the same file as the stage timings, because it was measured in the same run: the
    generated test prints it after tokenizing, so it is the length that reached the model rather than
    the length the environment asked for."""
    try:
        _doc = _read_stage_doc(state_dir_path, model, task)
        # THE WORKLOAD'S PROMPT LENGTH, NOT A STAGE'S. Every stage in a run sees the same prompt --
        # the byte model takes it as `seq_len` for all of them -- so it is a property of the request.
        # It was read out of the per-stage map under the key "prefill", which made a workload fact
        # reachable only through one stage's name, and only for models that have a stage so called.
        # The map keeps its old entry for docs written before this key existed.
        _pt = int(_doc.get("prompt_tokens") or 0)
        if _pt > 0:
            return _pt
        # A KEY IN AN OLD FILE FORMAT, not a stage of this model. Before `prompt_tokens` existed,
        # THIS TOOL wrote the prompt length under the literal "prefill", so reading it back is how a
        # doc from that version is understood -- the same kind of constant as any other on-disk
        # schema name, and unlike the writer above it cannot mis-price a model, because a doc that
        # does not have the key simply yields 0. Every doc written since carries prompt_tokens.
        return int((_doc.get("isl") or {}).get(_LEGACY_PROMPT_KEY) or 0)
    except Exception:  # noqa: BLE001
        return 0


def read_stage_isl_per_request_map(state_dir_path=None, model="", task="") -> dict:
    """{stage: items PER REQUEST}, from the legacy prompt-length marker. {} when none recorded.

    The unit is what separates this from read_stage_isl_map: these are multiplied by the batch to
    reach one unit of work, and the totals in that map must not be.
    """
    try:
        got = _read_stage_doc(state_dir_path, model, task).get("isl_per_request") or {}
        return {str(k): int(v) for k, v in got.items() if int(v or 0) > 0}
    except Exception:  # noqa: BLE001
        return {}


def read_stage_isl_map(state_dir_path=None, model="", task="") -> dict:
    """{stage: items it processes in one unit}, as the run recorded them. {} when nothing was recorded.

    A stage that consumes the prompt retires every prompt token; a recurring stage retires one. That
    cannot be inferred from the byte model -- decode READS a long context while processing a single
    token, so "does the read set grow with the prompt" says yes for both -- and it must not be
    inferred from the stage's NAME. The run measured it, so the run states it.
    """
    try:
        got = _read_stage_doc(state_dir_path, model, task).get("isl") or {}
        return {str(k): int(v) for k, v in got.items() if int(v or 0) > 0}
    except Exception:  # noqa: BLE001
        return {}


def read_stage_batch(state_dir_path=None, model="", task="") -> int:
    """The batch the profiled run actually served, or 0 when it was not recorded.

    Read back from the same file as the stage timings because it was measured in the same run: a
    ceiling priced for one user against an eight-user measurement is not a comparison.
    """
    try:
        return int(_read_stage_doc(state_dir_path, model, task).get("batch") or 0)
    except Exception:  # noqa: BLE001
        return 0


def read_stage_paths(state_dir_path=None, model="", task="") -> dict:
    """How each stage was MEASURED -- "trace+1cq", "eager", or "unknown".

    Separate from read_stage_ms because a caller wanting the split does not always want the paths,
    and because an older state file has no `paths` key at all: absent means unknown, which withholds
    a verdict rather than asserting one."""
    try:
        return {
            str(k): str(v) for k, v in (_read_stage_doc(state_dir_path, model, task).get("paths") or {}).items() if v
        }
    except Exception:  # noqa: BLE001
        return {}


def read_stage_ops(state_dir_path=None, model="", task="") -> dict:
    """Op dispatches observed per stage during the last warmup -- the EVIDENCE for the path label.

    A traced stage issues one dispatch; an eager one issues hundreds. trace_replay prints the count
    and derived "trace+1cq"/"eager" from it, but only the label was kept, so a doubted label could
    be checked only by re-running the workload by hand. gemma-3's prefill was diagnosed as eager
    twice, once wrongly, for exactly that reason."""
    try:
        return {
            str(k): int(v)
            for k, v in (_read_stage_doc(state_dir_path, model, task).get("ops") or {}).items()
            if isinstance(v, (int, float)) and int(v) >= 0
        }
    except Exception:  # noqa: BLE001
        return {}


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
    # WHOLE model: the depth cap is REMOVED, not set to 0. "0" arrives as a truthy string and was
    # read by model builders as "build zero layers", so this gate measured nothing and could only
    # report "no markers". See agent/layer_depth.py.
    _set_depth(env, None)
    # OSL: the DECLARED unit, not 1. This defaulted to "1" with no comment, and one decode step is not
    # a decode measurement -- it makes TTFT unmeasurable (no first-token boundary), forces TS == TSU,
    # and blends prefill into a number the report then calls "per token". gemma-3-12b-it's 33.98 ms is
    # prefill of 128 tokens PLUS one decode step, so the 29.4 tok/s/u it implies is not a decode rate
    # and prefill work cannot be scored against anything.
    #
    # The test already declares the unit (TT_PERF_ISL_TOKENS / TT_PERF_OSL_TOKENS, both 128). Honour it:
    # take OSL from the declared value so the measured unit and the reported unit are the same thing.
    # PERF_MCP_FULLPIPE_TOKENS still overrides for a cheap steering measurement.
    env["TT_PERF_OSL_TOKENS"] = os.environ.get("PERF_MCP_FULLPIPE_TOKENS") or os.environ.get(
        "TT_PERF_OSL_TOKENS", "128"
    )
    env.setdefault("TT_PERF_TRACE", "1")
    # TRACE EVERY STAGE THE MODEL DECLARES, not the one an LLM happens to have. This set a single
    # TT_PERF_PREFILL_TRACE for every model measured, so a pipeline whose stages are encode/vocode
    # had no way to be told to trace them -- the flag names the stage, and only one stage had a name
    # the tool knew. The per-stage depth knobs (TT_PERF_<STAGE>_LAYERS) have been derived from
    # PIPELINE_STAGES for a while; this is the same derivation, applied to the same question.
    #
    # TT_PERF_PREFILL_TRACE is still set, because it is what a generated test written before this
    # would read, and an old test meeting a new harness must not lose its trace.
    env["TT_PERF_PREFILL_TRACE"] = "1"
    for _st in _declared_stages_for_env():
        env["TT_PERF_%s_TRACE" % str(_st).upper()] = "1"
    # Start the trace region at the DRAM-derived size, not the perf test's hardcoded default. This gate
    # runs the whole pipeline at FULL depth (a much bigger capture than the coverage-depth default the
    # builder baked in), so starting big avoids the overflow entirely; the grow below is the fallback.
    _cur_reg = int(env.get("TT_PERF_TRACE_REGION") or 0)
    if _TRACE_REGION_DEFAULT > _cur_reg:
        env["TT_PERF_TRACE_REGION"] = str(_TRACE_REGION_DEFAULT)
    _prof = os.environ.get("PERF_MCP_PROFILE_ENV")
    if _prof:
        try:
            env.update(json.loads(_prof))
        except (ValueError, TypeError):
            pass
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    # AND THE DEPTH CAP, WHICH THIS GATE HAS BEEN INHERITING BY ACCIDENT.
    #
    # The comment above says this runs the whole pipeline at FULL depth, and it does not. `env` is a
    # copy of os.environ, the MCP config hands this process TT_PERF_STACK<i>_LAYERS=2 as LOOSE
    # variables beside the PERF_MCP_PROFILE_ENV json that is the intended channel, and the generated
    # perf test reads exactly those names and passes them to build_pipeline. So the gate has been
    # timing a 2-layer model while reporting it as the whole one.
    #
    # THE CAP IS FOR TRACY, AND TRACY IS OFF HERE. An uncapped tracy capture overflows the 12000
    # marker buffer, which is why the profiling runs are capped and must stay capped. This path pops
    # TT_METAL_DEVICE_PROFILER one line above: it is trace_replay, a stopwatch that prints three
    # numbers, and no buffer can overflow. The cap has no purpose here and never did.
    #
    # Measured on Voxtral, 2026-08-21: the gate reported 2.47 ms/token where a full-depth capture
    # recorded 55.68 tok/s (17.96 ms). The roofline then compared that 2-layer time against a
    # 62-layer ceiling and printed decode at 539% of peak -- and, worse, every win was banked against
    # a 7x-optimistic proxy of the model that actually ships.
    #
    # Names come from layer_depth, which owns the spelling, and from the stages the MODEL declared --
    # not a hardcoded list, so a model with other stack names is covered too.
    try:
        from agent.layer_depth import stack_layers_var, stage_layers_var
        from agent.stack_knob_repair import stage_names as _stage_names

        _depth_vars = {"TT_PERF_LAYERS"}
        for _s in _stage_names(_MODEL_ROOT) or []:
            _depth_vars.add(stage_layers_var(_s))
        # HOW MANY STACKS IS THE MODEL'S ANSWER, not a number picked here. _declared_stack_count
        # reads it from the checkpoint; 0 means unreadable, and then the stage names above are the
        # only spelling we can offer.
        from agent.layer_depth import _declared_stack_count

        for _i in range(max(0, int(_declared_stack_count(_MODEL_ROOT) or 0))):
            _depth_vars.add(stack_layers_var(_i))
        for _v in (os.environ.get("PERF_MCP_DEPTH_VARS") or "").split(","):
            if _v.strip():
                _depth_vars.add(_v.strip())
    except Exception:  # noqa: BLE001 -- a name we cannot derive is one we cannot strip; the cap stays
        _depth_vars = {"TT_PERF_LAYERS"}
    _dropped = sorted(k for k in _depth_vars if env.pop(k, None) is not None)
    if _dropped:
        print(
            "  [full-pipeline-gate] measuring at FULL depth: dropped the profiling depth cap (%s)"
            % ", ".join(_dropped),
            flush=True,
        )
    # -p depth_guard: this gate asks for ALL layers by removing the cap, and a perf test can fill
    # it back in at import via setdefault. The guard drops it again before the test body builds.
    cmd = [sys.executable, "-m", "pytest", "-p", _DEPTH_GUARD, "-o", "timeout=0", "-s", node]
    if case:
        cmd += ["-k", case]
    per_tokens = []
    stage_ms = {}
    # EVERY SAMPLE, NOT THE LAST ONE. The headline is the median of _FULLPIPE_SAMPLES readings
    # (`dec` below) precisely because one reading is noise; the per-stage split was a plain
    # overwrite, so the report showed decode as the median beside the SAME stage as whatever the
    # final sample happened to be -- 12.87 ms and 20.49 ms for one quantity in one report. Worse
    # than cosmetic: `stage_win` is computed from these numbers and gates a candidate record, so a
    # single slow or fast sample could fabricate a stage win or bury a real one while the headline
    # it sits beside was properly filtered. Collect them all and take the same median.
    stage_ms_samples: dict = {}
    stage_bytes: dict = {}
    # Every sample's reading, not just the last -- the read set is pinned WRITE-ONCE, so the value
    # that lands here is the ceiling divisor for the rest of the model's life.
    stage_bytes_samples: dict = {}
    stage_paths = {}
    stage_isl = {}
    # {stage: items PER REQUEST}, the legacy marker's unit. Kept apart from stage_isl, which holds
    # the TOTAL a stage states for one call.
    stage_isl_per_request = {}
    # The legacy PERF_ISL_TOKENS marker: one prompt length for the whole request, not a stage's count.
    prompt_tokens_seen = 0
    stage_ops = {}
    headline_units = []
    walls = []
    prefills = []
    # Topology comes ONLY from the run's own DP=/TP=/shard_active marker (parsed below), which
    # trace_replay.measure_adapter emits from the mesh the pipeline ACTUALLY opened. No default and no
    # guess: if the run reports no topology the scorecard prints 'unknown' rather than fabricating a mesh
    # (the old hardcoded 1x1 silently mislabelled a genuine multi-chip trace as single-chip).
    dp = tp = None
    shard = None
    batch = 1
    decode_path = prefill_path = "n/a"
    last_err = None
    try:
        from agent.perf_test_gen import _TRACE_REGION_GROW_ROUNDS as _STALL_GROW_ROUNDS
    except Exception:  # noqa: BLE001
        _STALL_GROW_ROUNDS = 6
    for _ in range(_FULLPIPE_SAMPLES):
        # A stall with NO trace is the silent-overflow signature: a too-small region can HANG the mesh,
        # and _adaptive_run then SIGKILLs it and RAISES -- so it never reaches the grow below. Handle it
        # HERE: reset the wedged device, double the region up to the DRAM ceiling, and retry the SAME
        # sample (without spending the measurement budget), so a hang that printed nothing self-corrects
        # instead of just burning samples. Only a timeout-type stall triggers this; a clean crash still
        # returns and takes the message-based grow path below.
        r = None
        for _stall_try in range(int(_STALL_GROW_ROUNDS) + 1):
            try:
                r = _adaptive_run(cmd, repo, env, "full-pipeline")
                break
            except Exception as exc:  # noqa: BLE001
                last_err = f"run failed: {str(exc)[-400:]}"
                cur = int(env.get("TT_PERF_TRACE_REGION") or _TRACE_REGION_DEFAULT)
                if not (isinstance(exc, _sp.TimeoutExpired) and cur < _TRACE_REGION_MAX):
                    break
                env["TT_PERF_TRACE_REGION"] = str(min(cur * 2, _TRACE_REGION_MAX))
                sys.stderr.write(
                    "[full-pipeline-gate] run stalled with no trace (silent overflow); resetting device "
                    "and growing trace region to %s B, retrying\n" % env["TT_PERF_TRACE_REGION"]
                )
                try:
                    from agent.probes import _device_reset

                    _device_reset(error_text=str(exc))
                except Exception:  # noqa: BLE001
                    pass
        if r is None:
            continue
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        # GROW THE TRACE REGION, same as the builder does. The generated test carries a
        # TT_PERF_TRACE_REGION default that the BUILDER derived while validating at the coverage
        # depth (2 layers for llama -> 23887872 B). This gate runs the SAME test at
        # TT_PERF_LAYERS=0, so the capture is ~26x larger and no longer fits -- the run then emits
        # no TRACE_PER_TOKEN_MS and the whole end-to-end gate reads as a crash. The device states
        # the exact bytes it needs; grow to that and re-run rather than reusing a value derived for
        # a different depth. perf_test_gen has had this since a5aa6a96af ("no fixed magic number")
        # -- it was simply never wired into this path.
        out, r = _grow_trace_region_and_retry(cmd, repo, env, out, r)
        # UMD prints the clamp itself, so the run tells us whether its own clock was valid. Cheaper
        # and more reliable than sampling telemetry alongside, which aliases against short runs.
        if _run_reported_clamp(out):
            globals()["_LAST_RUN_CLAMPED"] = True
        # READ AHEAD, because the census is PINNED on the line before this one. The total and the
        # split are printed as two lines and the total is written first; persisting it there would
        # take the pin, and the split arriving one line later would be refused by the very guard that
        # stops the ceiling drifting mid-run. They are one measurement, so they are written together.
        _census_sections = _parse_census_sections(out)
        _gathered_weights = _parse_gathered_weights(out)
        for line in out.splitlines():
            # PHASE TIMINGS, MEASURED. trace_replay prints one of these per stage it traced, derived
            # from the PIPELINE_STAGES the model itself declares -- so "prefill"/"decode" here are
            # tool measurements, not names a model typed. They were printed and discarded; the report
            # then had no phase accounting at all and the only per-stage rows it could show were the
            # agent's own stages_json prose, which nothing validates.
            #
            # Decode ms and prefill ms are not the same currency (decode recurs per token, prefill
            # once per request), so a phase split guessed from free text can put time in the wrong
            # pool and be acted on. Measured, it cannot.
            # THE BYTES A STAGE ACTUALLY READ. Parsed here beside its time because they are one
            # measurement of one stage: trace_replay observes the distinct device tensors the stage's
            # ops touched, which is the quantity the roofline has always had to infer from the
            # checkpoint and a tower name list.
            if "TRACE_STAGE_BYTES[" in line:
                try:
                    _bn = line.split("TRACE_STAGE_BYTES[", 1)[1].split("]", 1)[0].strip()
                    _bv = int(float(line.split("]=", 1)[1].split()[0]))
                    if _bn and _bv > 0:
                        stage_bytes[_bn] = _bv
                        stage_bytes_samples.setdefault(_bn, []).append(_bv)
                except (ValueError, IndexError):
                    pass
            if "TRACE_STAGE_MS[" in line:
                try:
                    _nm = line.split("TRACE_STAGE_MS[", 1)[1].split("]", 1)[0].strip()
                    _sv = float(line.split("]=", 1)[1].split()[0])
                    if _nm and _sv > 0:
                        stage_ms[_nm] = _sv
                        stage_ms_samples.setdefault(_nm, []).append(_sv)
                        # THE PATH TRAVELS WITH THE NUMBER. It was printed on this same line and
                        # dropped, so a stage that fell back to EAGER arrived at the report
                        # indistinguishable from a traced one -- and was then scored against a band
                        # that assumes trace. gemma-3's prefill is exactly that case.
                        if "path=" in line:
                            stage_paths[_nm] = line.split("path=", 1)[1].split()[0].strip()
                except Exception:  # noqa: BLE001
                    pass
            # THE EVIDENCE BEHIND THE PATH LABEL. trace_replay derives "trace+1cq" vs "eager" by
            # COUNTING op dispatches during the last warmup -- a traced stage issues one -- and it
            # prints that count. Only the verdict was read; the count itself was discarded, so a
            # reader who doubted a label had nothing to check it against, and "prefill is traced"
            # could only be confirmed by re-running the workload by hand. It is the difference
            # between a claim and a measurement, and it costs one line to keep.
            # THE MEASURED RESIDENT BYTE COUNT. Written into the model's facts so the ceiling
            # divides by what is ON THE CHIP rather than by a rule about what might be. See
            # agent/weight_census and perf_target.compute_target's census branch.
            if "TRACE_WEIGHT_BYTES=" in line:
                try:
                    _wb = int(line.split("TRACE_WEIGHT_BYTES=", 1)[1].split()[0])
                    _ok = "complete=1" in line
                    # THE WIDTH IS THE PART THE CEILING CAN USE. The byte TOTAL counts everything
                    # resident -- on gemma-3, 15.49 GB of which 6.85 GB is KV cache, which
                    # active_bytes already prices from seq_len -- so dividing by it double-counts.
                    # The RATIO is the average width of what the loader actually produced, and
                    # multiplied by a param count it gives the weights figure the placeholder
                    # 1.0 B/param was standing in for. Parsed here because the marker is the only
                    # place it crosses the process boundary.
                    _bpp = 0.0
                    if "bytes_per_param=" in line:
                        _bpp = float(line.split("bytes_per_param=", 1)[1].split()[0])
                    # A CAPPED CENSUS IS NOT THE MODEL. The census runs inside measure_adapter,
                    # which executes both in the capped profiling runs and in the uncapped
                    # full-pipeline gate -- and device_weight_bytes is pinned by the FIRST complete
                    # census to arrive. The capped one arrives first.
                    #
                    # Voxtral run 10: 1.299 B parameters recorded against the checkpoint's 4.676 B,
                    # 27.8%, because the build had 2 of 30 language layers. Every ceiling in that
                    # report divided a number missing three quarters of the model, and nothing could
                    # tell, because a capped census is shaped exactly like a whole one.
                    #
                    # The marker now states the depth it was taken at. Anything but "all" is refused
                    # here -- including "unknown", which is not a claim of full depth -- so the pin
                    # waits for the full-depth gate rather than being taken by whoever ran first.
                    _depth = ""
                    if "depth=" in line:
                        _depth = line.split("depth=", 1)[1].split()[0].strip()
                    if _wb > 0 and _depth and _depth != "all":
                        print(
                            "  [perf-mcp] census ignored: taken at depth=%s, not the whole model "
                            "(%.3g GB). Waiting for the full-depth measurement." % (_depth, _wb / 1e9),
                            file=sys.stderr,
                            flush=True,
                        )
                    elif _wb > 0:
                        _persist_device_weight_bytes(_wb, _ok, _bpp, _census_sections, _gathered_weights)
                except Exception:  # noqa: BLE001
                    pass
            if "TRACE_STAGE_OPS[" in line:
                try:
                    _nm = line.split("TRACE_STAGE_OPS[", 1)[1].split("]", 1)[0].strip()
                    _ov = int(line.split("]=", 1)[1].split()[0])
                    if _nm and _ov >= 0:
                        stage_ops[_nm] = _ov
                except Exception:  # noqa: BLE001
                    pass
            # THE PROMPT LENGTH THE RUN ACTUALLY PROFILED. The generated test prints it from
            # `_prompt_ids.shape[-1]` -- after tokenization, so it is the length that reached the
            # model, not the length anyone asked for -- and nothing read it. The report needs it to
            # price prefill's arithmetic (2 x params x ISL), and an optimize run exports no ISL
            # variable, so the renderer was left guessing and withheld the whole PREFILL stage.
            # Observed beats declared beats defaulted, and this is the observed one.
            # THE COUNT THE STAGE ITSELF STATED, for whatever stage stated it. TRACE_STAGE_ITEMS is
            # printed beside TRACE_STAGE_MS by the same loop that measured the stage, so a third
            # tower can carry a real item count instead of inheriting the fallback of 1.
            if "TRACE_STAGE_ITEMS[" in line:
                try:
                    _nm = line.split("TRACE_STAGE_ITEMS[", 1)[1].split("]", 1)[0].strip()
                    _nv = int(line.split("]=", 1)[1].split()[0])
                    if _nm and _nv > 0:
                        stage_isl[_nm] = _nv
                except Exception:  # noqa: BLE001
                    pass
            if "PERF_ISL_TOKENS=" in line:
                try:
                    _iv = int(line.split("PERF_ISL_TOKENS=", 1)[1].split()[0])
                    if _iv > 0:
                        # A DIFFERENT UNIT, KEPT IN A DIFFERENT MAP. This marker is the prompt
                        # length PER REQUEST -- the reader multiplies it by the batch to get what one
                        # unit of work retires. A count a stage states through <stage>_trace_items()
                        # is already the TOTAL for one call: voxtral's prefill_trace_items returns
                        # PREFILL_C * B, and its encode traces at batch 1 whatever the pipeline
                        # serves. Merging the two into one map made the reader multiply a total by
                        # the batch again -- prefill counted 8x its real work, and encode 8x for a
                        # batch it does not have.
                        # THE REQUEST PROPERTY, kept as one -- the prompt length reaches
                        # _persist_stage_ms as a scalar now, instead of being dug back out from under
                        # a stage key.
                        if not prompt_tokens_seen:
                            prompt_tokens_seen = _iv
                        # AND FILED AGAINST NO STAGE. This marker is printed at tokenisation, before
                        # any stage exists, so it cannot say which stage consumes it -- and it used to
                        # be filed under the literal "prefill" anyway. That made a workload fact
                        # reachable through one typed name: exactly one stage per model could be
                        # sized, and only if it happened to be called that, while every other stage
                        # fell back to ONE item. Voxtral's encoder was priced at 1 instead of 1500 and
                        # reported memory-bound when it is compute-bound.
                        #
                        # Dropping the write was previously refused because it repriced the named
                        # stage at one item, which was the worse failure. That is no longer the
                        # trade: summary._stage_items_observed reads the count back off the matmuls
                        # each stage actually ran, so every stage -- named anything -- is sized from
                        # what it did. The prompt length stays what it is, a property of the request,
                        # carried as the scalar below.
                except Exception:  # noqa: BLE001
                    pass
            if "TRACE_PER_TOKEN_MS=" in line:
                try:
                    per_tokens.append(float(line.split("TRACE_PER_TOKEN_MS=", 1)[1].split()[0]))
                except Exception:  # noqa: BLE001
                    pass
            elif "TRACE_HEADLINE_UNIT=" in line:
                # WHICH unit that reading measures. trace_replay picks the decode stage when there is
                # one (per token), a step/denoise stage next (per step), else the whole pipeline sum
                # (one inference). The marker name TRACE_PER_TOKEN_MS is historical and says "token"
                # for all three, so without this the roofline band scored a diffusion model's per-step
                # ceiling against a whole-pipeline number and called it a verdict.
                headline_units.append(line.split("TRACE_HEADLINE_UNIT=", 1)[1].split()[0].strip())
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
            elif "shard_active=False" in line:
                shard = False
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
        osl = env.get("TT_PERF_OSL_TOKENS") or os.environ.get("TT_PERF_OSL_TOKENS", "128")
        tsu = (1000.0 / dec) if dec else 0.0
        # mesh/TP/DP/shard come solely from the run's marker; when it is absent they stay None and we
        # print 'unknown' rather than a fabricated topology.
        _mesh_s = ("%dx%d" % (dp, tp)) if (dp is not None and tp is not None) else "unknown"
        _tp_s = ("%d" % tp) if tp is not None else "unknown"
        _dp_s = ("%d" % dp) if dp is not None else "unknown"
        _shard_s = "unknown" if shard is None else str(shard)
        sys.stderr.write(
            "[full-pipeline-gate] PERF_SCORECARD mesh=%s TP=%s DP=%s shard=%s on_device=%s "
            "ISL=%s OSL=%s batch=%d TTFT_ms=%s prefill_path=%s decode_ms=%s decode_path=%s TSU=%.2f TS=%.2f\n"
            % (
                _mesh_s,
                _tp_s,
                _dp_s,
                _shard_s,
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
        if headline_units:
            os.environ["PERF_MCP_LAST_HEADLINE_UNIT"] = headline_units[-1]
        # The same filter the headline gets, applied to each stage independently: a stage missing
        # from a sample simply has fewer readings, and median() of what it did report is still the
        # right answer for it.
        for _sn, _svals in stage_ms_samples.items():
            if _svals:
                stage_ms[_sn] = float(statistics.median(_svals))
        # The read set has the same last-write-wins defect and a worse consequence: it is pinned
        # write-once. Take the median here, before the doc is written, so the recorded number and the
        # pinned number are the same one.
        for _bn2, _bvals in stage_bytes_samples.items():
            if _bvals:
                stage_bytes[_bn2] = int(round(statistics.median(_bvals)))
        _persist_stage_ms(
            stage_ms,
            stage_paths,
            stage_isl,
            stage_ops,
            batch,
            int(prompt_tokens_seen or 0),
            stage_isl_per_request,
            stage_bytes,
        )
        # PIN WHAT ONE CALL OF EACH STAGE RETIRES, beside the read set and for the same reason: it
        # is a ceiling input (the compute floor is 2 x params x tokens), and the report's THEORETICAL
        # column must not move while the run works. Observed per run, so unpinned it would follow a
        # change in prefill chunking straight into the ceiling.
        try:
            for _tn, _tv in (stage_isl or {}).items():
                if _tn and int(_tv or 0) > 0:
                    _ledger().anchor(
                        _ledger().KIND_STAGE_TOKENS,
                        float(int(_tv)),
                        depth=str(_tn).strip().lower(),
                        mode="items",
                        source="trace_replay observed item count",
                        model=_MODEL_ROOT.name if _MODEL_ROOT else "",
                    )
        except Exception:  # noqa: BLE001 -- a pin that cannot be written must not cost a measurement
            pass
        # PIN THE BASELINE READ SET, where it is produced. Measuring the bytes made them right; it did
        # not stop them moving, and the dtype rung moves them by construction: bf16 -> bf8_b halves a
        # weight, the observed bytes halve, and a ceiling recomputed from them retreats ahead of the
        # measurement -- never reached, which is the defect KIND_FLOOR exists to prevent. Write-once
        # per (model, task, stage); the doc above keeps tracking the current build so the report can
        # tell a real dtype win from a stale reading.
        # A READING THAT DOES NOT REPRODUCE IS NOT A CEILING. This pin is write-once, so the first
        # value becomes the memory roof's divisor permanently -- there is no second chance to correct
        # it, and every later "% of ceiling" for that stage inherits it. The gate already takes
        # _FULLPIPE_SAMPLES readings, so agreement across them is free evidence: a stage's working set
        # is a property of the build and must come out the same every time. One that varies is
        # instrumentation noise, and pinning it would freeze the noise.
        #
        # Deliberately not a flag or an env switch: the run has the evidence in hand and can decide.
        try:
            for _st, _svals in (stage_bytes_samples or {}).items():
                if not _st or not _svals:
                    continue
                _med = float(statistics.median(_svals))
                _spread = (max(_svals) - min(_svals)) / _med if _med > 0 else 1.0
                if len(_svals) < 2 or _spread > _STAGE_BYTES_AGREE_TOL:
                    sys.stderr.write(
                        "[full-pipeline-gate] read set for %s NOT pinned: %d reading(s), spread %.1f%% "
                        "(need <= %.1f%%) -- values %s\n"
                        % (_st, len(_svals), _spread * 100.0, _STAGE_BYTES_AGREE_TOL * 100.0, _svals)
                    )
                    continue
                _ledger().anchor(
                    _ledger().KIND_STAGE_BYTES,
                    _med / 1e6,
                    depth=str(_st).strip().lower(),
                    mode="bytes_mb",
                    source="trace_replay observed read set (%d agreeing samples)" % len(_svals),
                    model=_MODEL_ROOT.name if _MODEL_ROOT else "",
                )
        except Exception:  # noqa: BLE001 -- a pin that cannot be written must not cost a measurement
            pass
        return statistics.median(per_tokens), "trace", None, decode_path
    if walls:
        return statistics.median(walls), "eager", None, None
    if last_err:
        return None, None, last_err, None
    # ATTACH THE EVIDENCE. `out` holds the workload's full stdout+stderr and was being discarded, so
    # this gate could only ever say "no markers" -- the actual reason (a TT_FATAL, an import error, a
    # crash before the first print) was written nowhere. Every full-pipeline failure was therefore
    # undiagnosable without patching the tool, which cost several wrong diagnoses on 2026-07-25/26.
    return (
        None,
        None,
        "no TRACE_PER_TOKEN_MS or FORWARD_WALL_MS in output (workload did not run full-pipeline)"
        + _workload_failure_tail(locals().get("out") or ""),
        None,
    )


_THERMAL_GATE = str(os.environ.get("PERF_MCP_THERMAL_GATE", "1")).lower() not in ("0", "false", "no")
_THERMAL_WAIT_S = float(os.environ.get("PERF_MCP_THERMAL_WAIT_S", "900"))
_THERMAL_POLL_S = float(os.environ.get("PERF_MCP_THERMAL_POLL_S", "15"))
_THERMAL_RETRIES = int(os.environ.get("PERF_MCP_THERMAL_RETRIES", "3"))
# THE TEMPERATURE A MEASUREMENT MAY START AT. Stated, not learned -- see _clamp_threshold_c for the
# 177 samples showing that a start temperature does not predict a clamp. Above the 60C the
# post-clamp cooldown holds to, so a board cooled after a clamp is always clear to start again.
_START_TEMP_C = 65.0
# AFTER A CLAMPED READING, COOL PROPERLY BEFORE TRYING AGAIN. The headroom wait is bounded and
# GIVES UP -- it runs anyway -- so on a board that cannot reach the threshold in 900 s every retry
# starts hotter than the last: measured on Voxtral 2026-08-14, the board went 79C -> 96C across a
# run while each attempt "waited" and then added its own heat. Four such attempts cost an hour and
# produced nothing. This target is ABSOLUTE and the wait is not abandoned on a timer: a retry is
# only worth taking from a genuinely cold board, and if the board never gets there the run should
# say so rather than burn attempts at 800 MHz.
# WAITING FOR PHYSICS IS NOT WORK. The parent watchdog kills a device subprocess on wall clock, and
# a thermal wait happens INSIDE that subprocess -- so on 2026-08-14 the tool cooled the board, the
# cooling consumed the budget, and it killed itself at 1716 s for "likely a device wedge". The board
# was not wedged; the reset that kill triggered is what actually broke it. These markers bracket
# every thermal wait so the watchdog can stop its clock instead of counting a cooldown as a hang.
_COOL_BEGIN = "PERF_MCP_COOLING_BEGIN"
_COOL_END = "PERF_MCP_COOLING_END"


def _cooling_marker(which):
    print(which, file=sys.stderr, flush=True)


_COOLDOWN_TO_C = float(os.environ.get("PERF_MCP_COOLDOWN_TO_C", "60"))
# NO TIMER ON COOLING. A board cools at the rate it cools; a deadline on that is a guess about
# physics, and the only thing it can do is cut the wait short and hand back a hot board. What IS
# evidence is the trend: a board still dropping will get there, and a board that has not moved in
# _COOLDOWN_PLATEAU_S is not cooling -- it has found its floor (this chassis idles at 79C) and no
# amount of further waiting changes that. So: wait as long as progress continues, stop when it stops.
_COOLDOWN_PLATEAU_S = float(os.environ.get("PERF_MCP_COOLDOWN_PLATEAU_S", "600"))
_COOLDOWN_POLL_S = float(os.environ.get("PERF_MCP_COOLDOWN_POLL_S", "20"))

# Fallback only. The real detector is agent.probes.detect_overheat, which already existed for the
# tracy path; this list is used if that import fails, so a broken import degrades to a weaker check
# rather than to no check.
_CLAMP_MARKERS = ("AICLK failed to settle", "clamped by max-arbiter", "AICLK clamped")


def _run_reported_clamp(out):
    """Did the device tell us its own clock was clamped during this run?

    agent.probes.detect_overheat is the tool's existing signal for exactly this and already guards
    the tracy profiling path (probes.py:1055). What it did NOT cover is the full-pipeline gate --
    _run_full_pipeline_ms drives _adaptive_run directly and never reaches that runner -- which is
    why a 68.3 ms clamped reading became a ledger anchor while the profiling path was protected.
    Reuse it here rather than growing a second detector.
    """
    if not out:
        return False
    # OR, not delegate-and-trust. `agent` resolves off sys.path, so WHICH copy of probes answers
    # depends on how the process was launched -- and a copy predating "AICLK failed to settle" in
    # _DEVICE_OVERHEAT_RE returns False for a genuinely clamped run. Checking the current wording
    # locally as well means a stale import can only ADD coverage, never silently remove it.
    if any(m in out for m in _CLAMP_MARKERS):
        return True
    try:
        from agent.probes import detect_overheat

        return bool(detect_overheat(out))
    except Exception:  # noqa: BLE001
        return False


_LAST_RUN_CLAMPED = False


def _board_state_dir():
    """Where facts about the MACHINE live, as opposed to facts about this model's run.

    state_dir() is per-model (~/.perf_mcp/<model>), which is right for a baseline or a knob cache and
    wrong for the clamp point: the temperature at which the driver drops AICLK to 800 MHz belongs to
    the board and its cooling, not to whatever is being optimized on it. Kept per-model it was learned
    from scratch every time -- gemma3 had 140 observations of this board while voxtral had 4 and no
    clean reading at all, so the same hardware answered "what is too hot" two different ways.

    PERF_MCP_BOARD_STATE_DIR names it outright; otherwise it is the parent of the per-model state
    dir. The explicit form exists because CLIMBING OUT OF A SANDBOX DEFEATS IT: the test suite gives
    each test a private state dir and expects nothing to escape, and a blind `.parent` landed in the
    shared pytest root where every other test's box lives. Tests then inherited clamp observations
    from each other, a fully mocked test found a learned threshold, read the real die temperature of
    a hot board, and sat in a 900 s thermal wait -- which is how the tool's own preflight suite went
    from 190 s to hanging.

    Only climbs when the state dir was pointed at explicitly. On the tempdir default the parent is /,
    which is not somewhere to write.
    """
    explicit = os.environ.get("PERF_MCP_BOARD_STATE_DIR")
    if explicit:
        return Path(explicit)
    return state_dir().parent if os.environ.get("PERF_MCP_STATE_DIR") else state_dir()


def _thermal_profile_path():
    return _board_state_dir() / "perf_mcp_thermal_profile.json"


def _adopt_per_model_profiles(doc):
    """Fold any per-model profiles beside the board file into it, once.

    The observations are not wrong -- they were taken on this board -- they were merely filed under
    the model that happened to be running. Dropping them would mean re-learning by clamping again,
    which costs real runs, so they are adopted rather than discarded. A heavier model reaches a given
    start temperature with more heat already in the package, so mixing them makes the threshold more
    conservative, never less: the tool waits slightly longer than one model alone would demand.
    """
    if doc.get("clamped_at") or doc.get("clean_at"):
        return doc
    root = _board_state_dir()
    merged = {"clamped_at": [], "clean_at": []}
    try:
        found = sorted(root.glob("*/perf_mcp_thermal_profile.json"))
    except OSError:
        return doc
    for f in found:
        try:
            old = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        for k in ("clamped_at", "clean_at"):
            merged[k] += [float(v) for v in (old.get(k) or []) if isinstance(v, (int, float))]
    if not (merged["clamped_at"] or merged["clean_at"]):
        return doc
    return {k: sorted(v)[-200:] for k, v in merged.items() if v}


def _load_thermal_profile():
    try:
        doc = json.loads(_thermal_profile_path().read_text())
        doc = doc if isinstance(doc, dict) else {}
    except Exception:  # noqa: BLE001
        doc = {}
    return _adopt_per_model_profiles(doc)


def _cooldown_after_clamp(target_c: float = 0.0) -> tuple:
    """Hold until the board is genuinely cold, after the hardware has been observed throttling.

    THE HEADROOM WAIT GIVES UP; THIS ONE DOES NOT (until its own bound). _wait_for_thermal_headroom
    is bounded at 900 s and then runs regardless, which is right before ordinary device work -- a
    capped profile compared against other capped profiles survives a clamp. It is wrong after a
    reading has ALREADY been discarded for clamping: retrying from a hot board reproduces the clamp,
    and the retry itself adds heat.

    Measured on Voxtral 2026-08-14: the board climbed 79C -> 96C over one run, each attempt waiting
    its full 900 s, giving up, measuring at 800 MHz instead of 1350, and having its reading thrown
    away. Four attempts, one hour, nothing measured, and the board hotter at the end than the start.

    The target is ABSOLUTE (default 60C) rather than "entry minus a margin", because a relative
    target on a board sitting at 96C just asks for 91C -- still clamped. Progress is printed every
    poll so a long cooldown is visible rather than looking like a hang, and it reports the
    temperature it reached so the caller can say why a retry was skipped.
    """
    target = float(target_c or _COOLDOWN_TO_C)
    t0 = time.time()
    last = _read_die_temp_c()
    if last is not None and last <= target:
        return True, last
    print(
        "  [thermal-gate] cooling to %.1fC before retrying (now %s) -- the previous reading was "
        "discarded for a clamped clock, and a retry from a hot board reproduces it"
        % (target, ("%.1fC" % last) if last is not None else "unknown"),
        file=sys.stderr,
        flush=True,
    )
    _cooling_marker(_COOL_BEGIN)
    try:
        best, best_t = last, time.time()
        while True:
            time.sleep(_COOLDOWN_POLL_S)
            # RE-ASSERT, EVERY POLL. The watchdog credits the gap between consecutive beats and
            # nothing beyond them, so a wait that goes quiet stops being free -- which is what keeps
            # a deadlock from buying itself unlimited time by claiming to be cooling.
            _cooling_marker(_COOL_BEGIN)
            cur = _read_die_temp_c()
            if cur is None:
                return True, None  # telemetry we cannot read is not a board we refuse to use
            if cur <= target:
                print(
                    "  [thermal-gate] cooled to %.1fC after %.0fs" % (cur, time.time() - t0),
                    file=sys.stderr,
                    flush=True,
                )
                return True, cur
            # PROGRESS, NOT A CLOCK, decides whether to keep waiting.
            if best is None or cur < best - 0.5:
                best, best_t = cur, time.time()
                print(
                    "  [thermal-gate] cooling: %.1fC (target %.1fC, %.0fs elapsed)" % (cur, target, time.time() - t0),
                    file=sys.stderr,
                    flush=True,
                )
            elif time.time() - best_t >= _COOLDOWN_PLATEAU_S:
                print(
                    "  [thermal-gate] board has sat at %.1fC for %.0fs and is no longer cooling -- that "
                    "is its floor in this chassis, so the %.1fC target is unreachable"
                    % (cur, time.time() - best_t, target),
                    file=sys.stderr,
                    flush=True,
                )
                return False, cur
    finally:
        _cooling_marker(_COOL_END)


def _record_thermal_observation(start_temp_c, clamped):
    """Remember the die temperature a reading STARTED at, and whether its clock held.

    This is what makes the gate hardware-agnostic. The clamp point is a property of the board, not
    of this tool: it was ~78C on the liquid-cooled p300c this was found on, and there is no reason
    for that number to hold on another Blackhole, on Wormhole, or on the same silicon with different
    cooling. So nothing is hardcoded -- the threshold is LEARNED from what this board actually did.
    """
    if start_temp_c is None:
        return
    doc = _load_thermal_profile()
    key = "clamped_at" if clamped else "clean_at"
    vals = [float(v) for v in (doc.get(key) or []) if isinstance(v, (int, float))]
    vals.append(round(float(start_temp_c), 2))
    doc[key] = sorted(vals)[-200:]
    try:
        p = _thermal_profile_path()
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2))
        os.replace(str(tmp), str(p))
    except Exception:  # noqa: BLE001
        pass


def _clamp_threshold_c():
    """The die temperature a measurement may start at: 60C, stated, the same figure the cooldown uses.

    IT USED TO BE LEARNED, AND THE LEARNING WAS UNSOUND. `clamped_at` holds the temperature a run
    STARTED at, recorded whenever that run clamped at some later point -- so a run beginning at
    56.75C, heating for twenty minutes and clamping at 85C wrote down 56.75. The threshold then took
    min() of that list and subtracted a 3C margin:

        min(clamped) - 3  =  56.75 - 3  =  53.8C      board idles at 53.9C

    Three defects in one line. It attributed a clamp at 85C to a start at 56.75C. It let ONE sample
    out of 39 speak for all of them -- the other 36 were 68C or higher. And min() only ever moves
    down, so no quantity of later evidence could raise it again: the board was pinned below its own
    idle temperature permanently. The gate could never pass, so it timed out at 900s every time and
    measured hot -- the precise outcome it exists to prevent, reached by way of a 15-minute wait.

    THE SIGNAL DOES NOT PREDICT ANYWAY, which is what settles it. Across 177 recorded runs:

        clamped starts   n=39   median 72.5C   range 56.8 - 87.2
        clean starts     n=138  median 70.8C   range 57.9 - 75.3

    Medians 1.7C apart on ranges that almost entirely overlap. Starting below 68C takes the clamp
    rate from 22% to 10%; starting below 60C measured WORSE than average. No threshold drawn through
    those two distributions separates them, so no amount of arithmetic on them was going to work.

    60C, stated. It is the figure _cooldown_after_clamp already holds to after a discarded reading,
    and having one number for "cool enough to measure" instead of two removes the case where the
    gate admits a board the cooldown would have refused. A stated number is also inspectable: it can
    be read here, argued with, and changed -- which the learned one could not be.

    65C, AND NOTHING READS THE PROFILE TO GET THERE. Not a percentile of it, not a reachability
    check against it -- any rule that consults the samples is a rule that can be moved by them, and
    being moved by them is what broke it. One number, in one place, that a reader can find by looking.

    The profile keeps recording. The samples are the evidence behind this docstring and behind any
    future revision of it; what stopped is a threshold deriving itself from them unsupervised.
    """
    env = os.environ.get("PERF_MCP_MAX_START_TEMP_C")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _START_TEMP_C


_LAST_KNOWN_TEMP_C = None


def _read_die_temp_c():
    """Hottest ASIC temperature across the chips this run can see, or None if unreadable.

    Every successful reading is remembered in _LAST_KNOWN_TEMP_C so a later failure has something
    better than a guess to fall back on -- see _wait_for_thermal_headroom.

    Delegates to agent.probes._read_asic_temp -- the tool already had this and a second copy would
    be one more place to fix. MAX is the right reduction on a mesh: a collective runs at the pace
    of its slowest chip, so one clamped member spoils the reading for every other one.

    Scoping to the mesh needs no index filtering: tt-smi honours TT_VISIBLE_DEVICES itself,
    verified on a 4-chip p300c -- unset reports 4 chips, "0" reports 1, "0,1" and "2,3" each report
    2. Filtering on top would be WRONG for a non-zero-based set, since tt-smi returns "2,3" at list
    positions 0 and 1. With visibility unset -- how --devices single presents -- this reduces over
    every chip on the host, deliberately: nothing says which chip the run will open, and
    over-waiting is the safe direction.

    None means "cannot tell". A host that has never produced a reading has no telemetry to wait for
    and still proceeds -- a missing sensor is not a hot board. But once a reading HAS been seen, an
    unreadable sensor falls back to it instead of proceeding: see _wait_for_thermal_headroom.
    """
    try:
        from agent.probes import _read_asic_temp

        v = _read_asic_temp()
    except Exception:  # noqa: BLE001
        return None
    if v is not None:
        globals()["_LAST_KNOWN_TEMP_C"] = float(v)
    return v


def report_board_over_clamp(label: str = "") -> bool:
    """Record, from the one place that owns the threshold, that the board is above it RIGHT NOW.

    A caller watching a subprocess cannot wait for the board -- the work is in flight -- but it can
    put a timestamp on the crossing, and on 2026-08-28 that record was the only thing missing. Chip 2
    stopped answering MID-RUN at 21:43 with all four chips at 98-103C; its telemetry went straight to
    the 0xffffffff sentinel and the kernel said nothing until 21:51, eight minutes later, when
    something tried to reopen it. The "Failed to set initial power state" line every previous
    investigation treated as the failure is the post-mortem, not the event. Temperature was the only
    warning the board ever gave.

    The clamp threshold is learned from THIS board's history, so the comparison lives here rather
    than in each watcher. Returns whether it reported, so a caller can rate-limit on real crossings.
    """
    try:
        cur = _read_die_temp_c()
        if cur is None:
            return False
        limit = _clamp_threshold_c()
        if cur <= limit:
            return False
        print(
            "  [thermal-watch] %s: board at %.1fC, above this board's clamp threshold %.1fC -- work in "
            "flight is holding it there" % (label or "device subprocess", cur, limit),
            file=sys.stderr,
            flush=True,
        )
        return True
    except Exception:  # noqa: BLE001 -- a thermometer that cannot be read must never stop the work
        return False


def _wait_for_thermal_headroom():
    """Block until the hottest die is below the AICLK clamp threshold.

    Measured on a liquid-cooled p300c running gemma-3-12b-it, IDENTICAL code each time:

        start 69.8C -> 35.83 ms   AICLK settles at 1350
        start 73.5C -> 36.75 ms   settles, but the NEXT run no longer does
        start 78.3C -> 58.06 ms   UMD: "clamped by max-arbiter index 7 at 800 MHz"
        start 79.9C -> 69.94 ms   same clamp

    A 1.9x swing with no code change. That is what wrote a 68.3 ms BEFORE-anchor into the gemma3
    ledger and made a whole run's verdicts meaningless. Note the two clamped readings differ by
    20% from each other, so "just measure everything hot" is not a valid alternative -- the
    clamped state is not a stable operating point.

    A device reset does NOT clear this; it does not cool the board. Roughly 6 minutes of idle does.
    Starting below the threshold is enough: a 17-second measurement does not generate the heat to
    cross it, so consecutive readings stay valid.
    """
    if not _THERMAL_GATE:
        return True, None
    limit = _clamp_threshold_c()
    t0 = time.time()
    cur = _read_die_temp_c()
    if cur is None:
        # UNKNOWN IS NOT COOL. This used to fall through with the cool boards, and the moment a read
        # is most likely to fail is right after a heavy run -- exactly when the board is hottest. On
        # 2026-08-15 that let a measurement start on a 93C board; it ran clamped at 800 MHz instead
        # of 1350, took ~1.7x longer, blew its 1806 s budget, and the timeout's reset bricked the
        # board. Both sources have to fail to get here, so this is rare and worth waiting out.
        #
        # The distinction that keeps it from hanging: a host that has NEVER produced a reading has no
        # telemetry to wait for, and a missing sensor is still not a hot board -- that case proceeds
        # exactly as before. Having read 93C a minute ago and being unable to read now is a different
        # thing entirely, and the last known value is the best evidence available.
        cur = _LAST_KNOWN_TEMP_C
        if cur is None:
            return True, None
        print(
            "  [thermal-gate] sensors unreadable; using the last known %.1fC rather than assuming cool" % cur,
            file=sys.stderr,
            flush=True,
        )
    if limit is None or cur <= limit:
        return True, cur
    print(
        "  [thermal-gate] die at %.1fC, this board has clamped at/above %.1fC; waiting (up to %.0fs)"
        % (cur, limit, _THERMAL_WAIT_S),
        file=sys.stderr,
        flush=True,
    )
    _cooling_marker(_COOL_BEGIN)
    try:
        return _headroom_poll(t0, limit, cur)
    finally:
        _cooling_marker(_COOL_END)


def _headroom_poll(t0, limit, cur):
    """The polling half of _wait_for_thermal_headroom, split out so the wait can be bracketed."""
    while time.time() - t0 < _THERMAL_WAIT_S:
        time.sleep(_THERMAL_POLL_S)
        _cooling_marker(_COOL_BEGIN)  # re-assert; the watchdog credits beats, not a single claim
        cur = _read_die_temp_c()
        if cur is None:
            return True, None
        if cur <= limit:
            print(
                "  [thermal-gate] cooled to %.1fC after %.0fs" % (cur, time.time() - t0),
                file=sys.stderr,
                flush=True,
            )
            return True, cur
    print(
        "  [thermal-gate] STILL %.1fC after %.0fs; measuring anyway, the clamp check decides"
        % (cur if cur is not None else -1.0, _THERMAL_WAIT_S),
        file=sys.stderr,
        flush=True,
    )
    return False, cur


def _measure_full_pipeline_guarded():
    """One reading of the pipeline; return (ms, method, err, path).

    ONE reading, not a median. Repeating the measurement was tried and removed: the board clamps
    AICLK once the die passes its threshold, and a 17-second measurement is itself what pushes it
    there -- so extra reps do not sample independent noise, they manufacture the very condition
    that invalidates them. On gemma-3-12b-it, reps 2 and 3 read 68.32 and 69.32 against a true
    35-36, and the MEDIAN of the three was therefore 68.32: taking three readings turned one bad
    number into the answer, where a single reading would have been correct.

    What the repeats were meant to fix -- that one reading cannot grade a win smaller than the
    board's spread -- turned out to be the same problem seen from the other end. That 2.28 ms
    "spread" was clamped and unclamped readings mixed together, not noise around a stable mean.
    Refusing clamped readings removes the spread at its source, which is what the thermal gate
    below does.

    Kept from that work: the clamp check and its bounded retries. A discarded reading is REPLACED,
    not averaged with, and a board that is clamped on every attempt fails loudly rather than
    anchoring the ledger to a number taken at 800 MHz instead of 1350.
    """
    discarded = 0
    for _ in range(1 + max(0, _THERMAL_RETRIES)):
        _ok, _start_c = _wait_for_thermal_headroom()
        globals()["_LAST_RUN_CLAMPED"] = False
        ms, method, err, path = _run_full_pipeline_ms()
        if ms is None:
            return None, method, err, path
        # Teach the board profile what this start temperature produced, so the threshold is derived
        # from THIS hardware rather than the p300c the gate was written on.
        _record_thermal_observation(_start_c, bool(_LAST_RUN_CLAMPED))
        if _THERMAL_GATE and _LAST_RUN_CLAMPED:
            discarded += 1
            print(
                "  [thermal-gate] DISCARDED %.4f ms: AICLK was clamped during the run, so this "
                "reading is not comparable to an unclamped one" % float(ms),
                file=sys.stderr,
                flush=True,
            )
            # A RETRY IS ONLY WORTH TAKING FROM A COLD BOARD. Going straight back in reproduces the
            # clamp and adds heat; the headroom wait ahead of the next attempt is bounded and would
            # give up again. Hold for a real cooldown instead, and stop retrying if it never comes.
            _cool_ok, _cool_c = _cooldown_after_clamp()
            if not _cool_ok:
                return (
                    None,
                    None,
                    "board stopped cooling at %s, above the %.1fC needed for an unclamped reading; "
                    "measuring again would only produce another clamped number"
                    % (("%.1fC" % _cool_c) if _cool_c else "an unknown temperature", _COOLDOWN_TO_C),
                    None,
                )
            continue
        return float(ms), method, None, path
    _lim = _clamp_threshold_c()
    return (
        None,
        None,
        "every reading was discarded: AICLK was clamped on all %d attempts (board too hot to "
        "measure; learned clamp threshold %s)"
        % (discarded, ("%.1fC" % _lim) if _lim is not None else "not yet established"),
        None,
    )


def _fullpipe_gate_log():
    """Resolved per call: a module constant freezes the path at import, before any redirect."""
    return state_dir() / "perf_mcp_fullpipe_gate.log"


def _gate_verdict_path():
    """Where every gate's LAST verdict is recorded, keyed like every other per-run artifact."""
    model = _MODEL_ROOT.name if _MODEL_ROOT else "model"
    task = os.environ.get("PERF_MCP_TASK", "main")
    return state_dir() / ("perf_mcp_gate_verdicts_%s_%s.json" % (model, task))


def record_gate_verdict(gate: str, status: str, **extra) -> None:
    """Persist a gate's verdict so the WRITE PATH can enforce it.

    Every gate returned its verdict to the agent and nothing else. git_commit's docstring says "valid
    measure + ok pcc + faster + full-pipeline NOT regressed ... never commit" and its body committed
    unconditionally; record_kernel_attempt took `beat_baseline` as a PARAMETER. So a regressed
    end-to-end reading and a banked win could coexist -- d54438bb4b and 7fac4ae685 were committed as
    wins at 13:24 and 13:34 while the gate's best had not moved since 13:14.

    A verdict that is only returned is advice. Recorded, it becomes a precondition.
    """
    try:
        p = _gate_verdict_path()
        doc = {}
        if p.is_file():
            try:
                doc = json.loads(p.read_text()) or {}
            except Exception:  # noqa: BLE001
                doc = {}
        row = {"status": str(status)}
        row.update({k: v for k, v in extra.items() if v is not None})
        row["sha"] = _git_head_sha() if "_git_head_sha" in globals() else ""
        doc[str(gate)] = row
        p.write_text(json.dumps(doc, indent=2))
    except Exception:  # noqa: BLE001
        pass


def gate_verdicts() -> dict:
    try:
        return json.loads(_gate_verdict_path().read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def gates_allow_banking() -> tuple:
    """(allowed, reason). THE one precondition for banking a win, read from recorded verdicts.

    A win is trace_replay end-to-end lower than before, with correctness intact -- so both gates must
    have run and both must be ok. Absent verdicts are refused, not assumed: an unrun gate is not a
    passed gate.
    """
    v = gate_verdicts()
    pcc, fp = v.get("pcc") or {}, v.get("full_pipeline") or {}
    if not pcc:
        return False, "check_pcc has not run since the last commit"
    if str(pcc.get("status")) != "ok":
        return False, "check_pcc status=%s" % pcc.get("status")
    if not fp:
        return False, "check_full_pipeline_latency has not run since the last commit"
    if str(fp.get("status")) != "ok":
        return False, "full-pipeline status=%s (%s ms vs best %s)" % (
            fp.get("status"),
            fp.get("full_pipeline_ms"),
            fp.get("best_ms"),
        )
    return True, "pcc ok + full-pipeline ok"


def _win_from_verdict(fp: dict, ms: float, ref) -> tuple:
    """(win, delta_ms, metric) for one full-pipeline verdict. THE win rule, stated once.

    The headline comes first: if it moved down, that is the result and the delta is stated in it.
    Otherwise a stage may still have ratcheted -- the headline measures the DECODE stage, so prefill
    levers cannot show up in it, and the delta then has to be stated in the stage that moved or the
    report prints a win beside a `+0.05 ms`. `metric` names which one, so the two can never be read
    as the same number.

    check_full_pipeline_latency sets stage_win only when a stage improved past the same tolerance
    the headline uses AND no stage regressed, so this cannot credit a lever that merely moved time
    from one stage into another."""
    if ref is None:
        return False, None, ""
    if ms < ref:
        return True, round(ms - ref, 4), "end_to_end"
    if not fp.get("stage_win"):
        return False, round(ms - ref, 4), "end_to_end"
    gains = [
        (row["ms"] - row["best"], name)
        for name, row in (fp.get("stages") or {}).items()
        if isinstance(row, dict) and row.get("improved") and row.get("best")
    ]
    if not gains:
        # stage_win with no stage to attribute it to: the verdict is internally inconsistent, so
        # fall back to the headline rather than inventing a metric.
        return False, round(ms - ref, 4), "end_to_end"
    delta, name = min(gains)
    return True, round(delta, 4), name


def gate_set_new_best() -> bool:
    """Did the last full-pipeline verdict actually RATCHET the end-to-end best down?

    This is what a win means. 'ok' alone includes holding steady, which is acceptable for keeping an
    edit but is not a win -- crediting it is how 29 device_ms new-bests became 29 ticks while the
    end-to-end best moved far fewer times.
    """
    fp = gate_verdicts().get("full_pipeline") or {}
    if str(fp.get("status")) != "ok":
        return False
    try:
        ms = float(fp.get("full_pipeline_ms"))
    except (TypeError, ValueError):
        return False
    if not ms > 0:
        return False
    prev = _fullpipe_reference_ms(fp)
    if prev is None:
        # NOTHING TO RATCHET AGAINST -> NOT A WIN. `prev is None or ms < prev` credited the FIRST
        # full-pipeline measurement of a run unconditionally: on gemma-3-12b-it attempt #0 measured
        # 87.9294 ms -- EXACTLY the baseline it was supposed to beat -- and was banked as a win that
        # improved nothing, while the attempt that really moved 87.93 -> 46.16 recorded False.
        # Failing closed is right: a run always measures its baseline before the loop starts, so a
        # missing reference means something is wrong, and a fabricated win is the one outcome that
        # cannot be corrected afterwards.
        return False
    # A STAGE THAT RATCHETED IS A WIN EVEN WHEN THE HEADLINE DID NOT MOVE. The headline is the decode
    # stage, so `ms < prev` credited decode work and nothing else -- prefill levers scored no-gain and
    # were reverted, including the one that took TTFT from 95.19 to 54.81 ms. Same rule as the
    # per-attempt verdict, from the same function, because three disagreeing answers to "is this a
    # win" is the bug this module has already had once.
    return _win_from_verdict(fp, ms, prev)[0]


def _fullpipe_reference_ms(fp: dict):
    """What this measurement must beat: the banked best, else the run's BASELINE.

    The gate only ever consulted `best_ms`, which does not exist until something has already won, so
    the first measurement had nothing to compare against. The baseline was never missing -- merely
    unread: the ledger holds `fullpipe_e2e / before` from the pre-loop measurement, keyed by
    (model, task) and durable across reruns.
    """
    prev = fp.get("best_ms")
    if prev is not None:
        try:
            return float(prev)
        except (TypeError, ValueError):
            return None
    try:
        led = _ledger()
        model = _MODEL_ROOT.name if _MODEL_ROOT else "model"
        row = led.first(led.KIND_FULLPIPE, led.PHASE_BEFORE, model=model, task=os.environ.get("PERF_MCP_TASK", "main"))
        return float(row["value_ms"]) if row else None
    except Exception:  # noqa: BLE001
        return None


def _emit_fullpipe(result: dict) -> dict:
    record_gate_verdict(
        "full_pipeline",
        result.get("status"),
        full_pipeline_ms=result.get("full_pipeline_ms"),
        best_ms=result.get("best_ms"),
        method=result.get("method"),
        # _verdict_identity keys ownership on this id and fails closed without it, so dropping it
        # here made EVERY record_kernel_attempt unownable and no rung could ever clear.
        measurement_id=result.get("measurement_id"),
        # A stage that ratcheted while the headline held flat -- gate_set_new_best reads this, so it
        # has to survive the trip through the verdict file. Every key dropped here is a fact the
        # write path cannot see.
        stage_win=result.get("stage_win"),
        stages=result.get("stages"),
    )
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
        with open(_fullpipe_gate_log(), "a") as _f:
            _f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass
    return result


_FULLPIPE_MODE_RANK = {"eager": 0, "trace": 1, "trace+1cq": 1}


def _fullpipe_pending_path() -> Path:
    """A candidate READING, not yet a committed result.

    The committed best used to be ratcheted down by any lower reading -- taken before PCC was
    known and regardless of a later revert -- and run.py reads that file for the AFTER headline.
    So a candidate that measured faster and was then reverted for pcc_low still set the run's
    reported speedup while the tree ended byte-identical to baseline. Readings now land here and
    are promoted only by an actual commit.
    """
    return _FULLPIPE_BASELINE_1CQ_PATH.with_suffix(".pending.json")


def _head_sha_quiet() -> str:
    try:
        return gitio.head_sha(gitio.repo_root(_MODEL_ROOT)) or ""
    except Exception:  # noqa: BLE001
        return ""


def _record_fullpipe_candidate(ms: float, method: str, mode: str, stages: dict | None = None) -> None:
    """Stash a reading as PENDING, stamped with the HEAD it was measured at."""
    try:
        _fullpipe_pending_path().write_text(
            json.dumps(
                {
                    "full_pipeline_ms": ms,
                    "method": method,
                    "mode": mode,
                    "sha": _head_sha_quiet(),
                    "unit": os.environ.get("PERF_MCP_LAST_HEADLINE_UNIT", ""),
                    # EVERY STAGE THE PIPELINE DECLARES, not just the one the headline measures.
                    # The headline is the decode stage (that is what tok/s/u means), so prefill work
                    # left it flat and read as no-gain -- the ratchet had no memory of prefill at
                    # all, and TTFT is a product metric too. See _bar_stages / gate_set_new_best.
                    "stages": {k: float(v) for k, v in (stages or {}).items() if isinstance(v, (int, float)) and v > 0},
                }
            )
        )
    except Exception:  # noqa: BLE001
        pass


def _measured_stages() -> dict:
    """The per-stage readings from the measurement that just ran, or {}.

    Read back from the stage file rather than threaded through _run_full_pipeline_ms's return: that
    function has five return sites, and the file is already written there, run-stamped, by the one
    reader every other consumer uses. A second channel for the same numbers is how the report came
    to show a stale prefill beside a fresh decode."""
    try:
        return {k: float(v) for k, v in (read_stage_ms() or {}).items() if isinstance(v, (int, float)) and v > 0}
    except Exception:  # noqa: BLE001
        return {}


def _bar_stages() -> dict:
    """The committed per-stage bests, or {}. Absent means no stage has ratcheted yet."""
    try:
        p = _FULLPIPE_BASELINE_1CQ_PATH
        if not p.exists():
            return {}
        doc = json.loads(p.read_text())
        return {k: float(v) for k, v in (doc.get("stages") or {}).items() if isinstance(v, (int, float)) and v > 0}
    except Exception:  # noqa: BLE001
        return {}


def _min_stages(cur: dict | None, new: dict | None) -> dict:
    """Per-stage minimum of two bars. A stage present in only one survives with its value."""
    out = {k: float(v) for k, v in (cur or {}).items() if isinstance(v, (int, float)) and v > 0}
    for k, v in (new or {}).items():
        if not isinstance(v, (int, float)) or v <= 0:
            continue
        if k not in out or float(v) < out[k]:
            out[k] = float(v)
    return out


def _stage_deltas(now: dict, bar: dict) -> dict:
    """Per stage: {ms, best, delta_pct, improved, regressed}. A stage with no bar yet is neither.

    THE TOLERANCE IS THE SAME ONE THE HEADLINE USES. A stage improving by less than the board's
    spread is not a result, and treating it as one is how a lever that moved nothing gets banked."""
    out = {}
    for name, ms in sorted(now.items()):
        prev = bar.get(name)
        row = {"ms": round(ms, 4), "best": (round(prev, 4) if prev else None)}
        if prev and prev > 0:
            row["delta_pct"] = round((ms - prev) / prev * 100.0, 2)
            row["improved"] = ms < prev * (1.0 - _FULLPIPE_TOL)
            row["regressed"] = ms > prev * (1.0 + _FULLPIPE_TOL)
        else:
            row["delta_pct"] = None
            row["improved"] = row["regressed"] = False
        out[name] = row
    return out


def _read_fullpipe_bar():
    """Return (best_ms, mode, readable). `readable` distinguishes "no baseline" from "could not read".

    The gate used to swallow a parse failure into an empty dict, so an existing-but-damaged bar was
    indistinguishable from no bar at all -- and the `best <= 0` branch responds to that by adopting
    whatever it just measured. gemma-3-12b-it went from 34.9066 to 67.2294 (28.6 -> 14.9 tok/s/u)
    through exactly that branch, right after two readings had been correctly REJECTED against the
    healthy bar. The next run would have banked anything at all as a ~20 ms win.

    readable=False means a reference exists and we failed to load it; the caller must refuse rather
    than re-baseline. An absent file is readable=True with best 0.0, which is the genuine
    nothing-to-compare-against case.
    """
    p = _FULLPIPE_BASELINE_1CQ_PATH
    if not p.exists():
        return 0.0, "", True
    try:
        doc = json.loads(p.read_text())
        ms = float(doc.get("full_pipeline_ms", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0, "", False
    if ms <= 0:
        return 0.0, "", False
    return ms, (doc.get("mode") or _fullpipe_mode(doc.get("method", "eager"), None)), True


def _write_fullpipe_bar(doc: dict) -> None:
    """Write the bar atomically. `write_text` truncates before writing, so a concurrent reader sees a
    partial file -- which is one way _read_fullpipe_bar's readable=False arises. temp + os.replace
    means a reader gets either the old contents or the new, never half of either."""
    p = _FULLPIPE_BASELINE_1CQ_PATH
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(doc))
        os.replace(str(tmp), str(p))
    except Exception:  # noqa: BLE001
        try:
            tmp.unlink()
        except Exception:  # noqa: BLE001
            pass


def _promote_fullpipe_if_committed() -> bool:
    """Promote a pending reading once HEAD has actually moved past the sha it was measured at.

    Promotion must follow the OBSERVABLE FACT that a commit happened, not the code path used to make
    it. Tying it to the git_commit tool alone meant a win committed any other way (the agent has Bash)
    would sit pending forever and never reach the AFTER headline -- under-reporting a real gain, the
    mirror of the bug this split fixes.
    """
    try:
        pend = json.loads(_fullpipe_pending_path().read_text())
    except Exception:  # noqa: BLE001
        return False
    was = str(pend.get("sha") or "")
    now = _head_sha_quiet()
    if not was or not now or was == now:
        return False
    return _promote_fullpipe_pending()


def _promote_fullpipe_pending() -> bool:
    """Called on a real commit: the pending reading becomes the committed best."""
    src = _fullpipe_pending_path()
    try:
        if not src.exists():
            return False
        # A RATCHET DOES NOT TURN BACKWARDS. This copied the pending reading in unconditionally, so
        # the bar tracked the LAST committed measurement rather than the BEST one, and one bad commit
        # replaced a good reference for every later verdict. gemma-3-12b-it: the ledger went
        # 35.3772 -> 34.9066 -> 55.0264, all tagged committed-best. 55.0264 ms is 18.2 tok/s/u on a
        # model that runs at 28.6, so the next run would have graded every attempt against 18.2 and
        # banked anything at all as a ~20 ms win, written conclusive. A mode change still
        # re-baselines: eager and trace numbers are not comparable, so the old value is meaningless
        # rather than better.
        #
        # ONE RATCHET PER METRIC. The doc carries the headline AND every declared stage, and they do
        # not move together: a prefill lever lowers TTFT while the decode headline holds flat, and
        # promoting or refusing the doc WHOLE meant the prefill best could only ever be recorded by
        # a commit that also happened to beat the headline. So each field keeps its own minimum --
        # the headline never rises, and neither does any stage.
        _keep_old = False
        _merged = None
        try:
            _new_doc = json.loads(src.read_text())
            _new_ms = float(_new_doc.get("full_pipeline_ms") or 0.0)
            _merged = dict(_new_doc)
            if _FULLPIPE_BASELINE_1CQ_PATH.exists() and _new_ms > 0:
                _cur_doc = json.loads(_FULLPIPE_BASELINE_1CQ_PATH.read_text())
                _cur_ms = float(_cur_doc.get("full_pipeline_ms") or 0.0)
                _same_mode = str(_cur_doc.get("mode") or "") == str(_new_doc.get("mode") or "")
                # A mode change re-baselines everything: eager and trace numbers are not comparable,
                # so the old values are meaningless rather than better.
                if _same_mode:
                    _merged["stages"] = _min_stages(_cur_doc.get("stages"), _new_doc.get("stages"))
                if _cur_ms > 0 and _same_mode and _new_ms > _cur_ms:
                    _merged["full_pipeline_ms"] = _cur_ms
                    print(
                        "  [full-pipeline-gate] REFUSED to move the headline backwards: %.4f ms is "
                        "slower than the committed best %.4f ms; keeping the best%s."
                        % (
                            _new_ms,
                            _cur_ms,
                            (
                                " (per-stage bests still ratchet)"
                                if _merged.get("stages") != (_cur_doc.get("stages") or {})
                                else ""
                            ),
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception:  # noqa: BLE001 -- a corrupt pending file must not take the bar with it
            _keep_old = True
        if not _keep_old:
            try:
                _write_fullpipe_bar(_merged if _merged is not None else json.loads(src.read_text()))
            except Exception:  # noqa: BLE001
                _FULLPIPE_BASELINE_1CQ_PATH.write_text(src.read_text())
        # RECORD THE RATCHET. This is the reading a win is confirmed against -- trace_replay
        # end-to-end vs trace_replay end-to-end -- and it was written only to the gate's own baseline
        # file. The ledger's fullpipe rows therefore only ever held the run's START and END bookends,
        # so the report's e2e line sat at the starting 22.79 ms while the gate had ratcheted to 17.05,
        # and the same section quoted both numbers as the current end-to-end.
        try:
            _doc = json.loads(src.read_text()) if src.exists() else json.loads(_FULLPIPE_BASELINE_1CQ_PATH.read_text())
            _ms = float(_doc.get("full_pipeline_ms") or 0.0)
            if _ms > 0:
                led = _ledger()
                _mname = _MODEL_ROOT.name if _MODEL_ROOT else ""
                _phase = (
                    led.PHASE_AFTER
                    if led.first(led.KIND_FULLPIPE, led.PHASE_BEFORE, model=_mname)
                    else led.PHASE_BEFORE
                )
                led.record(
                    led.KIND_FULLPIPE,
                    _phase,
                    _ms,
                    depth="all",
                    mode=str(_doc.get("mode") or "trace+1cq"),
                    source="fullpipe-gate:committed-best",
                    model=_mname,
                )
        except Exception:  # noqa: BLE001
            pass
        src.unlink()
        return True
    except Exception:  # noqa: BLE001
        return False


def _discard_fullpipe_pending() -> None:
    """Called on revert: the reading never became a result."""
    try:
        _fullpipe_pending_path().unlink()
    except Exception:  # noqa: BLE001
        pass


def _establish_fullpipe_baseline(ms: float, method: str, mode: str) -> None:
    """Write the committed baseline directly, and drop any pending reading.

    Used only when there is nothing to compare against -- no baseline yet, or the measurement MODE
    changed so the stored value is a different unit. That is a re-baseline, not a candidate: there
    is no commit to wait for, and keeping the old number would make every later delta meaningless.
    """
    try:
        _write_fullpipe_bar(
            {
                "full_pipeline_ms": ms,
                "method": method,
                "mode": mode,
                "unit": os.environ.get("PERF_MCP_LAST_HEADLINE_UNIT", ""),
            }
        )
    except Exception:  # noqa: BLE001
        pass
    _discard_fullpipe_pending()


def _fullpipe_verdict_for(ms: float, method: str, mode: str, best: float, base_mode: str) -> dict:
    """Verdict for a reading against the COMMITTED best, split out so it is testable.

    A mode flip cannot be differenced against the old baseline: the same TRACE_PER_TOKEN_MS
    field carries per-token decode ms in one mode and a summed pipeline wall in another, so
    subtracting across a flip fabricated gains like +98.8%. It re-establishes the baseline but
    must NOT return the agent's bank-a-win status.
    """
    if best <= 0:
        _establish_fullpipe_baseline(ms, method, mode)
        return {
            "status": "ok",
            "delta_pct": None,
            "note": "no baseline existed; established at this reading — nothing to compare against, so no delta",
        }
    if _mode_rank(mode) != _mode_rank(base_mode):
        _establish_fullpipe_baseline(ms, method, mode)
        upgrade = _mode_rank(mode) > _mode_rank(base_mode)
        return {
            "status": "ok" if upgrade else "rebaselined",
            "delta_pct": None,
            "note": (
                "measurement MODE changed %s -> %s. The two are different UNITS, so no delta is "
                "computable and none is reported. %s"
                % (
                    base_mode,
                    mode,
                    (
                        "This is a genuine fidelity UPGRADE, so it is bankable — commit it; the "
                        "baseline is re-established at the new mode."
                        if upgrade
                        else "This is NOT a win; the baseline is re-established at the new mode."
                    ),
                )
            ),
        }
    return {}


def _fullpipe_mode(method: str, path: str | None) -> str:
    if method != "trace":
        return "eager"
    p = (path or "").strip()
    return p if p == "trace+1cq" else "trace"


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


def _signposts_usable(seq: list) -> bool:
    """Are there signposts to read? See cc_optimize.run._signposts_usable -- a histogram of op
    repetition counts EXECUTIONS and cannot audit a stack tagged by identity."""
    idx = [i for i, t in enumerate(seq or []) if isinstance(t, str) and t.startswith(_SIGNPOST_PREFIX)]
    if len({seq[i] for i in idx}) <= 1:
        return False
    # DECOUPLED SIGNPOSTS ARE NOT SIGNPOSTS. A stack whose markers all land in a clump -- typically
    # trailing the ops entirely -- delimits nothing: every op would attribute to block 0. Presence is
    # necessary, interleaving is what makes them usable.
    return any(isinstance(t, str) and not t.startswith(_SIGNPOST_PREFIX) for t in (seq or [])[idx[0] :])


def _block_starts(sequence: list, n_blocks: int | None = None) -> tuple:
    seq = sequence or []
    sp = [i for i, s in enumerate(seq) if isinstance(s, str) and s.startswith(_SIGNPOST_PREFIX)]
    if sp and _signposts_usable(seq):
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
    _set_depth(env, None)  # ALL layers: cap REMOVED, never sent as 0 (see agent/layer_depth.py)
    env["TT_PERF_OSL_TOKENS"] = "1"
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
            "status": "unknown",
            "fully_applied": None,
            "note": (
                "coverage could NOT be determined -- the all-layers op-signature probe produced no "
                "counts (%s). This is NOT a pass: do not treat the lever as fully applied." % (seq or "no output")
            ),
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
    # The tool is trace+1cq end to end: one track, one baseline, no 2-CQ bookend.
    cq = 1
    ms, method, err, path = _measure_full_pipeline_guarded()
    if ms is None:
        return _emit_fullpipe({"status": "crash", "error": err, "cq": cq})
    metric = "trace_per_token_ms" if method == "trace" else "eager_full_pipeline_ms"
    mode = _fullpipe_mode(method, path)
    base_path = _FULLPIPE_BASELINE_1CQ_PATH
    cq_note = (
        "trace+1cq (the production metric): validate/bank EVERY win here — it always engages (no 2-CQ "
        "reservation, so no OOM/downgrade). The run is trace+1cq end to end; the start/end bookend is the "
        "same 1cq measure (AFTER = last committed 1cq verdict)."
    )
    tgt = _FULLPIPE_TARGET_MS if _FULLPIPE_TARGET_MS > 0 else None
    tgt_fields = {}
    if tgt is not None:
        tgt_fields = {
            "target_ms": round(tgt, 4),
            "gap_to_target_ms": round(ms - tgt, 4),
            "reached_target": ms <= tgt,
        }
    # A commit may have landed since the last reading (by the tool or by the agent's own git);
    # promote first so `best` reflects the committed tree rather than a stale pre-commit value.
    _promote_fullpipe_if_committed()
    # THREE STATES, NOT TWO. A parse failure used to collapse into "no baseline", and the
    # `best <= 0` branch answers that by adopting whatever was just measured -- which is how the bar
    # went 34.9066 -> 67.2294 (28.6 -> 14.9 tok/s/u) immediately after two readings had been
    # correctly rejected against the healthy bar.
    best, base_mode, _bar_readable = _read_fullpipe_bar()
    if not _bar_readable:
        return _emit_fullpipe(
            {
                "status": "bar_unreadable",
                "full_pipeline_ms": round(ms, 4),
                "method": method,
                "metric": metric,
                "mode": mode,
                "cq": cq,
                "error": (
                    "the committed-best file exists but could not be read (%s). REFUSING to "
                    "re-baseline from this reading -- a damaged reference is not an absent one. "
                    "Restore or delete it before continuing." % base_path
                ),
            }
        )
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
    _special = _fullpipe_verdict_for(ms, method, mode, best, base_mode)
    if _special:
        return _emit_fullpipe(
            {
                "full_pipeline_ms": round(ms, 4),
                "method": method,
                "metric": metric,
                "mode": mode,
                "cq": cq,
                **_special,
                "note": (_special.get("note") or "") + " · " + cq_note,
                **tgt_fields,
            }
        )
    delta_pct = round((ms - best) / best * 100.0, 2) if best > 0 else None
    diverged = ms > best * (1.0 + _FULLPIPE_TOL)
    # "not more than 8% slower than the best ever seen" was reported as ok, which is the agent's
    # bank-a-win signal -- so a 7%-slower lever could be committed and, repeated, ratcheted real
    # latency upward while the reported AFTER stayed at the old minimum. Slower is `regressed`.
    regressed = (not diverged) and ms > best
    # THE HEADLINE IS ONE STAGE, NOT THE WHOLE PRODUCT. TRACE_PER_TOKEN_MS is the DECODE stage --
    # that is what tok/s/u means -- so every prefill lever left it flat, read as no-gain, and was
    # reverted: TTFT went 95.19 -> 54.81 ms on this model and the ratchet had no field to put it in.
    # A stage that improved with none regressed is a real result and is stashed as a candidate, so a
    # commit can promote it. The STATUS still follows the headline: a decode regression is a
    # regression whatever prefill did, and only the headline may be compared against `best`.
    _stages_now = _measured_stages()
    _sdelta = _stage_deltas(_stages_now, _bar_stages())
    stage_win = (
        bool(_sdelta)
        and any(r["improved"] for r in _sdelta.values())
        and not any(r["regressed"] for r in _sdelta.values())
    )
    if ms < best or (not diverged and not regressed and stage_win):
        _record_fullpipe_candidate(ms, method, mode, _stages_now)
    return _emit_fullpipe(
        {
            "status": "diverged" if diverged else ("regressed" if regressed else "ok"),
            "full_pipeline_ms": round(ms, 4),
            "stages": _sdelta,
            "stage_win": stage_win,
            # MINTED HERE, where a trace replay actually ran. _verdict_identity keys ownership on it,
            # so exactly one attempt can claim this reading and every later one reports own=False.
            "measurement_id": _measurement_id(),
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


def _untracked_baseline_path() -> Path:
    model = _MODEL_ROOT.name if _MODEL_ROOT else "model"
    task = os.environ.get("PERF_MCP_TASK", "main")
    return state_dir() / ("perf_mcp_untracked_%s_%s.json" % (model, task))


def _write_untracked_baseline() -> None:
    """Snapshot which files were already untracked at the clean checkpoint, so a later revert can
    delete ONLY what an edit created and never a pre-existing generated artifact."""
    try:
        repo = gitio.repo_root(_MODEL_ROOT)
        try:
            spec = _MODEL_ROOT.relative_to(repo)
        except ValueError:
            spec = None
        _untracked_baseline_path().write_text(json.dumps(sorted(gitio.untracked_under(repo, spec))))
    except Exception:  # noqa: BLE001
        pass


def _read_untracked_baseline() -> set:
    try:
        return set(json.loads(_untracked_baseline_path().read_text()))
    except Exception:  # noqa: BLE001
        return set()


@mcp.tool()
def git_head() -> dict:
    """Return the current git HEAD sha of the model repo (your clean checkpoint / revert target)."""
    repo = gitio.repo_root(_MODEL_ROOT)
    _write_untracked_baseline()
    return {"sha": gitio.head_sha(repo)}


def _record_committed_win(message: str) -> None:
    """Log the just-committed lever as a win against the current target.

    git_commit IS the bank-a-verified-win action, but the ✓win marks in RUN_REPORT.md come
    only from record_kernel_attempt(beat_baseline=true). The agent often records the FOLLOW-UP
    re-measurements (which no longer beat the already-lowered floor, so beat_baseline=false) and
    never marks the winning moment, leaving committed wins shown as ·try. Deriving the win mark
    from the commit itself makes the report reflect what was actually banked. Fail-open: never
    raises, so it can never break the commit."""
    try:
        t = _load_target()
        op = t.get("op")
        if not op:
            return
        rung = str(t.get("rung") or t.get("next_rung") or "knob").split(":")[-1] or "knob"
        if t.get("measured_ms") is None:
            # A COMMIT IS NOT A MEASUREMENT. This marked beat_baseline=True on every successful
            # git_commit, so housekeeping commits -- "refresh the generated RUN_REPORT", "checkpoint
            # the perf test", a comment-only "record the measured dead ends" -- all rendered as ✓win.
            # On llama3_1_8b_p150 that was 47 of 73 wins in one run, and it put a ✓ in the fidelity
            # column while both real fidelity measurements showed no gain. The docstring on
            # git_commit already promises "valid measure + ok pcc + faster"; enforce the measure part
            # rather than trusting the commit as proof. The commit still succeeds; it just is not
            # claimed as a speedup.
            return
        # RECORD THE COMMITTED STATE IN THE LEDGER TOO. The ledger only saw profile_model readings,
        # but a banked win is measured by measure_candidate -- so the ledger lagged the real state and
        # the headline understated the gain: llama3_1_8b_p150 rendered "-> 664.17 ms" (the last full
        # profile) while the run had already committed 654.43 and then 615.69. A commit is the moment
        # the current state changes, so it is exactly when the ledger should learn the new number.
        try:
            _led = _ledger()
            _led.record(
                _led.KIND_EAGER,
                _led.PHASE_AFTER,
                t.get("measured_ms"),
                depth=_depth_in_force(),
                mode="eager",
                source="git_commit",
                model=_MODEL_ROOT.name if _MODEL_ROOT else "",
            )
        except Exception:  # noqa: BLE001
            pass
        _append_attempt(
            {
                "op_signature": op,
                "kernel_kind": rung,
                "measured_ms": t.get("measured_ms"),
                "fullpipe_ms": (_fullpipe_ms_now() or (None, None))[0],
                "fullpipe_best_ms": (_fullpipe_ms_now() or (None, None))[1],
                "baseline_at_record": _baseline_at_record(),
                # The baseline this verdict was reached against, so a resumed run can tell whether it still
                # applies to the same work before skipping the lever.
                "baseline_at_record": _baseline_at_record(),
                "beat_baseline": True,
                "wedged": False,
                "kernel_detected_in_source": True,
                "note": "committed: " + " ".join((message or "").split())[:140],
            }
        )
    except Exception:  # noqa: BLE001
        pass


def _perf_test_paths() -> list:
    """Every perf test this run resolved, absolute. From the manifest, never a filename pattern."""
    paths = []
    try:
        _root = gitio.repo_root(_MODEL_ROOT)
    except Exception:  # noqa: BLE001
        return paths
    _m = _MANIFEST if isinstance(_MANIFEST, dict) else {}
    cands = []
    _res = (_m.get("perf_test_resolved") or {}).get("path")
    if _res:
        cands.append(_res)
    for _c in _m.get("components") or []:
        if isinstance(_c, dict) and _c.get("perf_test"):
            cands.append(_c["perf_test"])
    for _c in cands:
        try:
            _p = Path(str(_c).split("::", 1)[0])
            paths.append(_p if _p.is_absolute() else (Path(_root) / _p))
        except Exception:  # noqa: BLE001
            continue
    return paths


def _reinject_stage_marks() -> list:
    """Re-apply the stage marks the revert just removed. Returns what changed, never raises."""
    out = []
    try:
        from agent.stage_marks import inject_stage_marks as _inject
    except Exception:  # noqa: BLE001
        return out
    seen = set()
    for _p in _perf_test_paths():
        try:
            if not _p.is_file() or str(_p) in seen:
                continue
            seen.add(str(_p))
            cur = _p.read_text()
            new, why = _inject(cur)
            if new != cur:
                _p.write_text(new)
                out.append("%s: %s" % (_p.name, why))
        except Exception:  # noqa: BLE001
            continue
    return out


@mcp.tool()
def git_commit(message: str) -> dict:
    """Commit the current model-dir changes (scoped to the model dir only — unrelated repo changes
    are left untouched). Use this to BANK a verified win: valid measure + ok pcc (check_pcc) + faster
    + full-pipeline NOT regressed (check_full_pipeline_latency status == 'ok'). If check_pcc OR
    check_full_pipeline_latency is not ok, revert — never commit. Returns the new sha."""
    # ENFORCE, do not merely instruct. The rule above was prose for the agent and the body committed
    # unconditionally: d54438bb4b and 7fac4ae685 were banked as wins at 13:24 and 13:34 while the
    # end-to-end best had not moved since 13:14, because nothing here asked the gates.
    _allowed, _why = gates_allow_banking()
    if not _allowed and os.environ.get("PERF_MCP_ALLOW_UNGATED_COMMIT") != "1":
        return {"committed": False, "refused": _why, "sha": ""}
    repo = gitio.repo_root(_MODEL_ROOT)
    try:
        pathspec = _MODEL_ROOT.relative_to(repo)
    except ValueError:
        pathspec = None
    sha = gitio.commit(repo, message, pathspec)
    if sha:
        _record_committed_win(message)
        _promote_fullpipe_pending()
        _write_untracked_baseline()
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
    # tracked files are restored above; files the edit CREATED are not, so remove those too
    _removed = []
    try:
        _removed = gitio.remove_new_untracked(repo, _read_untracked_baseline(), pathspec)
    except Exception:  # noqa: BLE001
        pass
    _discard_fullpipe_pending()
    # THE REVERT RESTORES THE MODEL, NOT THE INSTRUMENTATION. The stage marks are an UNCOMMITTED edit
    # to a tracked file inside the model dir, so the scoped checkout above -- doing exactly its job --
    # erases them along with the rejected attempt. voxtral run 25: injected 09:29, first revert
    # 11:20:47, and every later capture ran an unmarked test. Injecting earlier cannot win, because a
    # revert always comes after; re-applying here is the only place that knows they were just removed.
    # Idempotent and self-refusing, so it can never double-mark a pass.
    _remarked = _reinject_stage_marks()
    out = {"reverted_to": sha}
    if _removed:
        out["removed_created_files"] = _removed
    if _remarked:
        out["stage_marks_reapplied"] = _remarked
    return out


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


def _baseline_at_record():
    """The baseline anchor in force right now, or None -- stamped onto every attempt.

    A resumed run may skip a lever it already tried, but only if that verdict was reached against the
    SAME work. Without this stamp there is no way to tell, so a restart after the baseline changed
    (different input, seq len, layer depth) would inherit conclusions drawn about something else.
    """
    try:
        led = _ledger()
        row = led.first(
            led.KIND_EAGER,
            led.PHASE_BEFORE,
            _MODEL_ROOT.name if _MODEL_ROOT else "",
            os.environ.get("PERF_MCP_TASK", "main"),
        )
        v = float((row or {}).get("value_ms"))
        return round(v, 4) if v > 0 else None
    except (TypeError, ValueError, AttributeError):
        return None


def _consumed_verdict_path():
    return state_dir() / (
        "perf_mcp_fullpipe_consumed_%s_%s.json"
        % (_MODEL_ROOT.name if _MODEL_ROOT else "model", os.environ.get("PERF_MCP_TASK", "main"))
    )


def _measurement_id() -> str:
    """A fresh id, minted by the code that actually runs a trace replay."""
    return "fp-%d" % time.monotonic_ns()


def _verdict_identity(fp: dict):
    """What makes one end-to-end measurement distinct from the next: the MEASUREMENT's own id.

    This used to key on the verdict FILE's mtime, reasoning that ms alone is not enough (after a
    revert the pipeline legitimately measures the same number again, and the sha repeats too). But
    that file is rewritten whenever ANY gate records a verdict -- pcc, measure, commit -- not only
    when a trace replay runs. The mtime therefore moved with no measurement behind it, the identity
    looked new, and the next attempt claimed a reading it never took. On gemma-3-12b-it that produced
    fourteen attempts across five ops all carrying the identical -0.1926 win, one of them a
    PCC-rejected structural change that was 2.1x slower and had already been reverted.

    An id minted at measurement time cannot drift, because nothing else mints one. A verdict without
    an id has not been shown to correspond to a trace replay, so it is NOT ownable -- failing closed,
    since a fabricated win is the one outcome that cannot be corrected afterwards.
    """
    mid = fp.get("measurement_id")
    if not mid:
        return None
    return [str(fp.get("sha") or ""), fp.get("full_pipeline_ms"), str(mid)]


def _attempt_fullpipe_verdict() -> dict:
    """THE single end-to-end comparison for the attempt being recorded.

    One number per attempt, one subtraction, and the sign is the verdict:

        delta = this attempt's OWN end-to-end  -  the running best
        delta < 0  ->  win, and the best moves;  delta >= 0  ->  no gain, best unchanged

    Three separate implementations used to answer this, and they disagreed. `gate_set_new_best`
    banked, the report's Δ column subtracted `fullpipe_ms - fullpipe_best_ms`, and `winning_indices`
    re-derived the ✓ marks from a staircase -- so one run showed 16 raw flags, 3 ticks and 2 real
    improvements, with a "-41.77 ms" printed next to rows marked "no gain".

    The reason those numbers repeated down the page is that an end-to-end measurement is EXPENSIVE
    (a full trace replay, minutes) so it is not run per attempt: attempts that did not trigger one
    simply re-read the last verdict and inherited its value as though it were their own. So this
    attributes a verdict ONCE. The attempt that caused the measurement owns it; every later attempt
    reading the same verdict reports `own=False` and carries no delta and no win -- honest about
    having measured nothing, rather than borrowing someone else's result.
    """
    out = {"own": False, "ms": None, "ref": None, "delta": None, "win": False, "metric": ""}
    fp = gate_verdicts().get("full_pipeline") or {}
    # A `regressed` reading is a real measurement -- the candidate ran, the replay finished, the
    # number came back worse -- and record_kernel_attempt's contract is that even a measured LOSS
    # clears the op as tried. Refusing it deadlocked the ladder: three genuinely different grid
    # variants could measure, all lose, none record, and termination_check returned the same
    # next_target forever. Ownership stays one-shot and id-keyed, so this cannot resurrect the
    # borrowed-verdict bug. `diverged` stays refused -- in this harness that is nearly always the
    # degraded-device regime rather than the edit.
    if str(fp.get("status")) not in ("ok", "regressed"):
        return out
    try:
        ms = float(fp.get("full_pipeline_ms"))
    except (TypeError, ValueError):
        return out
    if not ms > 0:
        return out

    ident = _verdict_identity(fp)
    if ident is None:
        # No measurement id -> this verdict cannot be traced to a trace replay. Not ownable.
        return out
    path = _consumed_verdict_path()
    try:
        already = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        already = None
    if already == ident:
        return out  # this measurement already belongs to an earlier attempt

    ref = _fullpipe_reference_ms(fp)
    out.update(own=True, ms=round(ms, 4), ref=None if ref is None else round(ref, 4))
    if ref is not None:
        out["win"], out["delta"], out["metric"] = _win_from_verdict(fp, ms, ref)
    try:
        path.write_text(json.dumps(ident))
    except OSError:
        pass
    return out


def _fullpipe_ms_now():
    """The end-to-end full-pipeline ms behind the CURRENT gate verdict, or None.

    Stamped onto every attempt so the report can rank wins by the number the win definition is stated
    in. Without it the ranking falls back to `measured_ms`, which means a different thing depending on
    which lever was measured -- per-token for a host/dispatch lever, per-profile device_ms for an op
    lever -- and one staircase over both disqualifies the larger-scoped rows.
    """
    try:
        fp = gate_verdicts().get("full_pipeline") or {}
        v = float(fp.get("full_pipeline_ms"))
        if v <= 0:
            return None
        try:
            prev = float(fp.get("best_ms"))
            prev = prev if prev > 0 else None
        except (TypeError, ValueError):
            prev = None
        return (v, prev)
    except (TypeError, ValueError, AttributeError):
        return None


def _rung_allowance(op_signature: str, kernel_kind: str, attempts: list) -> tuple[int, int]:
    """(attempts already made, how many are permitted) for this (op, rung).

    Mirrors the policy _op_ladder_status already applies when handing rungs out, so the two cannot
    drift: knob rungs get _MAX_KNOB_RETRIES, deep rungs get one, and a knob is capped to one once the
    op has gone deep (perf_mcp.py:1338 -- a second knob search is not worth it after a structural or
    kernel rung exists). Failed measurements do not count against the allowance; nothing was learned.
    """
    rung = _normalise_rung(kernel_kind)
    matches = [a for a in attempts if _op_match(op_signature, a)]
    tries = sum(
        1
        for a in matches
        if _normalise_rung(a.get("kernel_kind")) == rung
        and not a.get("measurement_failed")
        and a.get("measured_ms") is not None
    )
    # SAME FILTER AS `tries` ABOVE, for the same reason. went_deeper cuts the knob allowance from
    # _MAX_KNOB_RETRIES to 1 because "a second knob search is not worth it after a structural or
    # kernel rung EXISTS" -- but a row with measured_ms=None is not a rung that exists, it is a rung
    # that was skipped, and this model's ledger carries 50 of them (41 deliberate "EXCLUDED BY
    # OPERATOR" entries). Counting those spent the second-variant allowance on ops that had never
    # been measured at all: the decode QKV matmul got ONE grid attempt, and since a first attempt is
    # what tells you which direction to move, the informed second attempt -- worth -1.31% device_ms
    # when finally recorded out-of-process -- was refused as a closed rung.
    kinds = {
        (a.get("kernel_kind") or "").lower()
        for a in matches
        if not a.get("measurement_failed") and a.get("measured_ms") is not None
    }
    went_deeper = bool(kinds & (_STRUCTURAL_RUNGS | {"tt-lang", "cpp", "tp-fracture"}))
    if rung in _KNOB_RUNG_NAMES:
        allowed = 1 if went_deeper else _MAX_KNOB_RETRIES
    else:
        allowed = 1
    return tries, allowed


_KNOB_RUNG_NAMES = {"grid", "dtype", "fidelity", "shard"}


@mcp.tool()
def record_kernel_attempt(
    op_signature: str, kernel_kind: str, measured_ms: float, beat_baseline: bool, note: str = "", stages_json: str = ""
) -> dict:
    """Record that you AUTHORED and MEASURED a real custom kernel for an open op (tt-lang or C++).

    A win claim is kept only when the measurement backs it, so the flag and the number can never
    disagree once written and readers need no second opinion about which to trust.
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
        "fidelity",
        "shard",
        "knob",
        "fusion",
        "fuse",
        "structural",
        "gather",
        "sparse",
        "cache",
        "kv-cache",
        "trace",
        # `trace-capture` IS a config/loop transform, not a hand-written kernel, so it belongs in
        # this set -- and it is the rung _op_ladder_status literally NAMES for the host bucket.
        # Omitting it deadlocked that bucket: the host ladder clears on a win or on
        # PERF_MCP_MAX_HOST_ATTEMPTS (3) real tries out of _host_kinds = {structural, trace,
        # trace-capture}, but _rung_allowance permits ONE attempt per non-knob rung, so only
        # `trace` and `structural` were ever recordable -- 2 of a required 3. The third try was
        # refused for having no `generic_op`/`@ttl`/`.cpp` marker in the model source, which a loop
        # transform correctly does not have, and host_overhead was re-emitted as next_target
        # forever with no recordable rung left. Same species as the three fixes in f41ac046a5.
        "trace-capture",
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
    _ms = round(float(measured_ms), 4)
    # CLOSED-RUNG CHECK RUNS FIRST, BEFORE the verdict is claimed. `_attempt_fullpipe_verdict()`
    # CONSUMES the measurement id (one replay, one attempt), so when it ran ahead of this check a
    # refusal still burned the measurement: the agent recorded a closed rung by mistake, got refused,
    # and the trace replay it had already paid minutes for was gone -- the next, legitimate kind on
    # the same op then refused too, for "owns no end-to-end measurement". Order matters because one
    # of these two refusals is free to retry and the other is not.
    if os.environ.get("PERF_MCP_ALLOW_RETRIED_RUNG") != "1":
        _all = [a for a in _load_attempts_all() if not a.get("measurement_failed")]
        _tries, _allowed = _rung_allowance(op_signature, kernel_kind, _all)
        if _tries >= _allowed:
            _done_kinds = sorted(
                {
                    _normalise_rung(a.get("kernel_kind"))
                    for a in _all
                    if _op_match(op_signature, a) and a.get("kernel_kind")
                }
            )
            _left = [r for r in ladder_order() if r not in _done_kinds]
            return {
                "recorded": False,
                "refused": (
                    "rung %r on %r is CLOSED: %d of %d permitted attempt(s) already recorded"
                    % (_normalise_rung(kernel_kind), op_signature, _tries, _allowed)
                ),
                "already_tried": _done_kinds,
                "rungs_still_open": _left,
                "next": (
                    "work a rung in rungs_still_open, or a different op from termination_check's "
                    "blocking list. Re-measuring a closed rung cannot be recorded, so it cannot "
                    "clear the op or bank a win."
                ),
            }
    _fpv = _attempt_fullpipe_verdict()
    # AN ATTEMPT NEEDS ITS OWN MEASUREMENT. gates_allow_banking already enforces this at COMMIT time
    # -- "absent verdicts are refused, not assumed: an unrun gate is not a passed gate" -- but nothing
    # enforced it here, so a row could be written with fullpipe_ms=None and render `n/m`. On
    # gemma-3-12b-it that was 79 of 94 attempts with no end-to-end number, and 13 more that inherited
    # someone else's. The device side was never the gap: measure_candidate runs for every attempt.
    # What was missing is the replay that decides whether the change moved the metric being scored.
    #
    # WEDGED is exempt because a candidate that crashed or hung the device cannot be measured, and
    # dropping that row would hide the crash and invite the next run to re-derive it. The env override
    # mirrors PERF_MCP_ALLOW_UNGATED_COMMIT for a resumed session whose verdict belongs to a previous
    # process. A refusal RETURNS -- it does not raise -- because this runs inside a live agent loop
    # and an exception would end the round instead of redirecting it.
    if (
        not _fpv["own"]
        and "wedged" not in (note or "").lower()
        and os.environ.get("PERF_MCP_ALLOW_UNMEASURED_ATTEMPT") != "1"
    ):
        return {
            "recorded": False,
            "refused": (
                "this attempt owns no end-to-end measurement: call check_full_pipeline_latency for "
                "THIS candidate, then record it. A verdict measured for an earlier attempt cannot be "
                "reused -- one replay, one attempt."
            ),
        }
    # ENFORCED, not advised: a rung that has already had its attempts does not get another one.
    #
    # termination_check hands the agent a next_target, but that is ADVICE and the agent is an LLM
    # that routinely works something else -- matmul was excluded twice by seeding and measured anyway
    # in runs 25 and 26; NLPConcatHeads/shard was measured a fourth time with three attempts already
    # on file. Across gemma-3-12b-it's history that is 30 repeats in 146 attempts (21%), and earlier
    # 62 in 162 (38%): MatmulDeviceOperation 128x15360x3840/grid three times, 32x15360x3840/shard
    # three times, BinaryNg/grid three times.
    #
    # Refusing HERE is what makes it stick. This is the one choke point every attempt passes through,
    # and a refused attempt cannot enter history, cannot clear an op, and cannot be banked as a win --
    # so re-doing a closed rung stops being merely discouraged and becomes unrecordable. The refusal
    # names the op's remaining rungs so the agent is redirected rather than just blocked.
    #
    # Permanence across runs comes free: _load_attempts_all reads archive UNION live, so a rung tried
    # in a previous optimize of this model is still closed in the next one.
    rec = {
        "op_signature": op_signature,
        "kernel_kind": kernel_kind,
        "measured_ms": _ms,
        # ONE COMPARISON, MADE ONCE, FOR THIS ATTEMPT -- see _attempt_fullpipe_verdict. This attempt's
        # OWN end-to-end minus the running best; the SIGN IS THE VERDICT. The banked flag, the delta
        # the report prints and the win mark all read this one result, so they cannot disagree about
        # the same row -- which they did: 16 beat_baseline flags from 14 measurements, 3 rendered
        # ticks, and 2 improvements that actually happened.
        # A WIN IS WHAT GOT COMMITTED. This used to be _fpv["win"] -- the end-to-end verdict, read at
        # RECORD time, before the commit-or-revert decision exists. _record_committed_win then sets the
        # same flag from the commit, which is what the report was always meant to show: "git_commit IS
        # the bank-a-verified-win action ... Deriving the win mark from the commit itself makes the
        # report reflect what was actually banked." Two writers, and the early one fires on changes
        # that never reach the tree.
        #
        # gemma-3-12b-it run 21: three ✓win marks in 80 minutes with ZERO commits. One was a shard the
        # agent applied, measured at -0.84%, and REVERTED as noise (claimed_beat_baseline=False). Two
        # were investigations that made no edit at all -- note "none: ...", device_ms unchanged at
        # 381.2266. All three "beat" the reference because the run's baseline was measured on freshly
        # reset chips (36.2548 cold vs ~35 warm) and, with nothing committed, the ratchet never moved
        # off it.
        #
        # So the flag has ONE writer now: the commit. The verdict fields below stay -- they are the
        # evidence -- they simply stop deciding a mark only a commit can earn.
        "beat_baseline": False,
        "claimed_beat_baseline": bool(beat_baseline),
        "fullpipe_ms": _fpv["ms"],
        "fullpipe_best_ms": _fpv["ref"],
        "fullpipe_delta_ms": _fpv["delta"],
        "fullpipe_measured_here": _fpv["own"],
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
def recall_knobs(op_class: str, grid: str = "", bound_by: str = "", regime: str = "") -> dict:
    """REUSE-FIRST: return the tested/known knobs already catalogued for this op_class, so you
    APPLY/ADAPT a proven one BEFORE improvising from scratch. Routed deterministically from the
    GUIDELINES catalog (numbered guides + LEARNED_*/GRADUATED_* learned levers) by op_class. CALL
    THIS for next_target.op_class before every rung's edit.

    ADVISORY ONLY — a recalled knob SEEDS the attempt (tested block/shard/fidelity values, and the
    NEGATIVE knowledge of what NOT to do, e.g. 'don't bf16 Q/K/V', 'packer_l1_acc must be True'); it
    NEVER lets you skip a rung or stop early. You must still check_pcc + measure_candidate +
    record_kernel_attempt for the rung exactly as termination_check requires. If nothing matches,
    improvise from principles, then persist the win yourself with distill_knob (the write-back is a
    manual agent call on the cc engine).

    regime: the STAGE this lever belongs to, as THE MODEL declared it in PIPELINE_STAGES -- pass
    next_target.regime when the target carries one. This axis is deliberately open (router.VOCABULARY
    validates it against the run's declared stages, not a fixed set) because a fixed
    {prefill,decode,na} could not tag a lever written for an audio encoder. Levers for one stage are
    tagged with it and nothing else reaches them: the KV-cache section is `op_class: attention,
    datamove` + `regime: decode`, so narrowing by op_class ALONE cannot find it from a decode target.

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
    # Phase/alias names the tool's OWN instructions hand to the agent ('decode' at the kv-cache
    # gate, 'generation_loop', 'host') are not router op_class vocabulary. Map them; anything
    # still unknown returns the UNNARROWED set with a visible note instead of a silent empty.
    _vocab = router.VOCABULARY.get("op_class", frozenset())
    _alias_note = ""
    if oc.lower() not in _vocab:
        # The caller's op_class is whatever the prompt or the agent used ('decode' at the kv-cache
        # gate, 'generation_loop', 'host'). Ask what it MEANS rather than keeping an alias table
        # that misses the next phrasing; UNKNOWN returns the unnarrowed set, never a silent empty.
        _mapped = _integrity.classify(oc, set(_vocab), what="op_class")
        if _mapped:
            _alias_note = "op_class %r resolved to %r" % (oc, _mapped)
            oc = _mapped
        else:
            _alias_note = "op_class %r could not be resolved; returning UNNARROWED levers" % (oc,)
            oc = ""
    try:
        gdir = str(_PKG / "GUIDELINES")
        index = router.build_index(gdir)
        q = {"op_class": oc} if oc else {}
        g = (grid or "").strip().lower()
        if g in _GRID_VOCAB:
            q["grid"] = g
        b = _BOUND_MAP.get((bound_by or "").strip().lower())
        if b:
            q["bound"] = b
        # THE STAGE AXIS, WHICH NOTHING COULD REACH. router.DIMENSIONS has carried "regime" all along
        # and route() filters on it, but no caller passed one -- so a stage-tagged lever was only ever
        # findable by whatever op_class it also happened to declare. The KV-cache section is
        # `op_class: attention,datamove` + `regime: decode`; a decode target narrowing by op_class
        # alone lands on attention levers and never sees it.
        _rg = (regime or "").strip().lower()
        if _rg and _rg != "na":
            q["regime"] = _rg
        try:
            hits = router.route(index, q) if q else router.all_entries(index)
        except Exception:  # noqa: BLE001
            hits = []
        if not hits and len(q) > 1:  # narrowing starved -> never return empty wrongly
            # Widen one axis at a time rather than collapsing straight to op_class: a target that
            # carries a regime and no usable op_class (every run-level gate) would otherwise fall to
            # `{"op_class": ""}` and lose the stage narrowing that was the only thing it had.
            for _wider in (
                {"op_class": oc, "regime": q["regime"]} if oc and "regime" in q else None,
                {"regime": q["regime"]} if "regime" in q else None,
                {"op_class": oc} if oc else None,
            ):
                if not _wider:
                    continue
                try:
                    hits = router.route(index, _wider)
                except Exception:  # noqa: BLE001
                    hits = []
                if hits:
                    break
            if not hits:
                hits = router.all_entries(index)
        if not hits and not q:
            hits = router.all_entries(index)
        # MODEL-SCOPED GUIDANCE FOLLOWS THE BOUND, NOT THE OP CLASS. GUIDELINES 13's sections declare
        # op_class: matmul, which is right for what they describe -- weight streaming is a matmul
        # concern -- but it means the agent only meets them while standing on a matmul. Run 22 on
        # gemma-3-12b-it: 1 of 52 recall_knobs calls was a matmul, and the two hours before it went to
        # reduction, eltwise, datamove and RoPE. The one body of guidance that addresses the actual
        # bottleneck was never handed over.
        #
        # The bottleneck does not belong to an op: 34.4 ms of producer wait on a GeGLU mul is the decode
        # weight stream stalling, and that op's own guidance is about eltwise. So on a MEMORY-bound
        # recall the model-level sections are appended whatever the class -- appended, not promoted, so
        # the named rung's own levers still come first. This is not a rung and makes nothing a target;
        # can_stop still does not wait for them. It only guarantees the knowledge is present when it
        # applies. Compute-bound and unstated-bound recalls are untouched: prefetch hides DRAM latency,
        # and a FLOP-bound op has none to hide.
        if b:
            _have = {h.get("id") for h in hits}
            try:
                for _m in router.route(index, {"bound": b}):  # bound only -- op_class deliberately wild
                    if str(_m.get("id", "")).startswith("model-") and _m.get("id") not in _have:
                        hits = list(hits) + [_m]
            except Exception:  # noqa: BLE001
                pass
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
                    # A model-scoped section is NOT this op's rung. Without the marker the agent files
                    # it under whichever op it was standing on -- run 20 recorded the prefetcher as
                    # `Matmul 32x3840x15360 / shard`, which is where the agent was, not what changed.
                    "scope": "model" if str(h.get("id", "")).startswith("model-") else "op",
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


def _host_gate(prof: dict, blocking: list, attempts: list) -> dict | None:
    for b in prof.get("buckets") or []:
        if b.get("id") != "host_overhead":
            continue
        hms = round(float(b.get("device_ms") or 0.0), 4)
        src = (b.get("tags") or {}).get("source")
        if hms < _material_gap_ms(prof.get("device_ms") or 0.0) or src != "op_gap":
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


def _token_stage_name() -> str:
    """The stage THE MODEL declared for the loop that retires a token, or "" if it did not say.

    _decode_gate used to label its target `op_class="decode"`. That is a stage word this tool chose,
    not one the model supplied, and it is not in the op_class vocabulary at all -- it survived only
    because recall_knobs sends unknown values through _integrity.classify, an LLM round-trip, to be
    remapped into something that is. So every firing of the gate spent a model call resolving a
    constant we wrote ourselves, and resolved it to a SINGLE op_class, which cannot reach the
    KV-cache section anyway (`op_class: attention,datamove` + `regime: decode`).

    The stage belongs on the regime axis, which router.VOCABULARY leaves open precisely so a model
    names its own stages -- its comment records that a fixed {prefill,decode,na} could not tag a
    lever written for an audio encoder. So ask the model: PIPELINE_STAGES is its own declaration.

    Returning "" is a real answer and the right one when nothing was declared: recall_knobs treats an
    absent axis as unnarrowed, which hands over the whole catalogue with a note. That is strictly
    better than narrowing on a word nobody said.
    """
    try:
        from agent.stack_knob_repair import stage_names
    except Exception:  # noqa: BLE001
        return ""
    try:
        declared = [str(n or "").strip() for n in (stage_names(_MODEL_ROOT) or []) if str(n or "").strip()]
    except Exception:  # noqa: BLE001
        return ""
    if not declared:
        return ""
    if len(declared) == 1:
        return declared[0].lower()  # one stage, and it is the one that retires the token
    # SEVERAL STAGES, AND NOTHING HERE CAN SAY WHICH ONE RETIRES THE TOKEN. The fact that would
    # settle it -- a stage whose per-call item count is 1 -- lives in the pipeline's
    # <stage>_trace_items hooks, and reading those means importing the pipeline, which imports ttnn
    # and needs a device. This gate runs without one.
    #
    # So return "" and let the catalogue come back unnarrowed with a note. The alternative is a
    # substring test for "decode" in the stage names, which is the exact guess headline_unit's
    # docstring calls "a guess wearing an observation's clothes": a pipeline whose recurring stage is
    # called `generate` reads as one-pass, and one that names any stage `decode` reads as
    # autoregressive whether it loops or not. Unnarrowed-and-honest beats narrowed-and-wrong.
    return ""


def _decode_gate(prof: dict, attempts: list) -> dict | None:
    if os.environ.get("TT_PERF_MODULE_LEVEL") == "1":
        return None
    # ONLY A MODEL THAT DECODES CAN HAVE A BROKEN DECODE. This gate BLOCKS a run until a KV-cache
    # lever lands, and it reaches that verdict from `decode_status == "repeat_prefill"` -- which
    # trace_replay emits whenever its capture was SKIPPED, for any reason, on any pipeline. A
    # classifier or an encoder-only model exposes no traceable step, skips, and is then ordered to
    # add a cached single-token decode step to a model that emits no tokens. Nothing downstream
    # could clear it: there is no decode to cache, so the attempt cap is the only way out, three
    # wasted rewrites later.
    #
    # The structural answer is already recorded. trace_replay prints TRACE_HEADLINE_UNIT from
    # headline_unit(), which reads the decode_step CONTRACT rather than any name: "token" means the
    # pipeline retires one token per call, "step" a denoise step, "inference" one whole pass. Only
    # the first has a decode loop for a KV-cache to fix.
    _unit = str(os.environ.get("PERF_MCP_LAST_HEADLINE_UNIT", "") or _reliable_forward_unit() or "").strip().lower()
    if _unit and _unit != "token":
        return None
    repeat = prof.get("decode_status") == "repeat_prefill"
    scale = kv_cache_needed_by_scaling(prof.get("decode_ms_at_c"), prof.get("decode_ms_at_2c"))
    recompute = bool(scale) if scale is not None else _decode_is_recompute(_MODEL_ROOT)
    if not (repeat or recompute):
        return None
    # MEASUREMENT-GATED EXIT: this lever clears ONLY when a KV-cache attempt actually reduced cost
    # (beat_baseline == a measured per-token reduction). A generic 'structural'/trace attempt does NOT
    # clear it — trace removes dispatch gaps, not recompute. Bounded by PERF_MCP_MAX_KV_ATTEMPTS so a
    # genuinely infeasible cache cannot loop forever: after N real kv-cache attempts the gate yields.
    _kv_kinds = ("structural-decode", "kv-cache")
    kv_clean = [a for a in attempts if (a.get("kernel_kind") or "").lower() in _kv_kinds]
    kv_won = any(_ledger().is_win(a) for a in kv_clean)
    # A KV-cache attempt that WEDGED the device is auto-recorded (_autorecord_wedge) with
    # kernel_detected_in_source=False, so termination_check's detected-filter drops it from the
    # `attempts` passed here. Count those wedges from the full log toward the cap: a KV-cache that
    # crashes every time must be treated as "tried" so this gate yields instead of ordering the
    # same wedging rewrite forever — mirroring how a wedged tt-lang/C++ kernel retires its rung
    # (_rung_state). A clean measured win still clears immediately.
    kv_wedged = sum(
        1 for a in _load_attempts() if (a.get("kernel_kind") or "").lower() in _kv_kinds and a.get("wedged")
    )
    max_kv = int(os.environ.get("PERF_MCP_MAX_KV_ATTEMPTS", "3") or "3")
    if kv_won or (len(kv_clean) + kv_wedged) >= max_kv:
        return None
    reason = (
        "MANDATORY kv-cache — a lever SEPARATE from trace. decode is repeat_prefill: it re-runs the "
        "full prefill every token (no cached decode_step / KV-cache). Trace removes DISPATCH gaps ONLY "
        "and does NOT remove this REDUNDANT RECOMPUTE, so 'trace already applied' / 'irreducible' does NOT "
        "resolve this and will NOT clear the gate. You MUST add a KV-cache + single-token decode_step "
        "(recall_knobs(op_class='attention', regime=<the stage this target names>)). Then "
        "record_kernel_attempt(op='generation_loop','kv-cache',"
        "measured_ms,beat_baseline) — this gate clears ONLY on a MEASURED per-token reduction from the cache."
        if repeat
        else "MANDATORY kv-cache — SEPARATE from trace. per-token cost scales with capacity "
        "(use_cache=False, no KV-cache write) -> O(capacity) recompute every token EVEN THOUGH it traces. "
        "Trace does NOT remove recompute; 'irreducible' is NOT accepted. Add a KV-cache + single-token "
        "decode_step (recall_knobs(op_class='attention', regime=<the stage this target names>)); "
        "record_kernel_attempt(op='generation_loop','kv-cache',"
        "measured_ms,beat_baseline) — clears ONLY on a MEASURED per-token reduction."
    )
    host_ms = 0.0
    for b in prof.get("buckets") or []:
        if b.get("id") == "host_overhead":
            host_ms = float(b.get("device_ms") or 0.0)
            break
    gap = max(host_ms, float(prof.get("per_token_ms") or 0.0), _MATERIAL_GAP_MS)
    _stage = _token_stage_name()
    return {
        "op": "generation_loop",
        # IN-VOCABULARY, and the stage carried on the axis built for it. The loop is host-bound
        # orchestration -- this gate already says bound_by="host" -- so host_fallback is what it IS;
        # WHICH stage it belongs to is the model's word, on the regime axis, or absent.
        "op_class": "host_fallback",
        "regime": _stage,
        "gap_ms": round(gap, 4),
        "bound_by": "host",
        "grid": None,
        "weight_dtype": None,
        "next_rung": "structural-decode",
        "reason": reason,
    }


def _fold_gate(prof: dict, attempts: list) -> dict | None:
    """One op fingerprint evaluated many times per layer -- fold the repeats into one wider matmul.

    THE DISCRIMINATOR IS SELF-NORMALISING, which is what makes this detectable at all. A high call
    count is not evidence: EVERY matmul in a 30-layer model runs 30 times. What distinguishes a
    foldable repeat from ordinary per-layer recurrence is running MORE than once per layer -- and the
    per-layer baseline does not have to be looked up, because the profile states it. top_ops groups
    by (op_code, shape, memory), so the modal count across fingerprints IS "once per layer per item":
    the great majority of ops run exactly that often. A fingerprint whose count is an integer multiple
    of that mode is being evaluated several times within one layer.

    Deriving the baseline from the profile rather than from depth x items matters practically: depth
    needs the checkpoint and items needs <stage>_trace_items, which means importing the pipeline,
    which needs a device this gate does not have. The mode needs neither.

    WHAT IT CATCHES. Two projections of identical shape applied to the same input share a fingerprint
    and run 2x per layer -- a SwiGLU gate/up pair is the standard case, and concatenating them into
    one wider matmul is the standard fix. A per-head or per-frame matmul runs heads x or frames x and
    folds into a batched call.

    A CANDIDATE, NOT A VERDICT. The gate can see that something recurs within a layer; it cannot see
    whether the operands are concatenable or whether a data dependency forbids it. That is what the
    attempt establishes, and 'none: <evidence>' is a legitimate outcome the cap will accept.

    Same shape as _decode_gate and _conv_gate: applicability, detector, measured-win exit, cap.
    """
    ops = [o for o in (prof.get("open_ops") or []) if int(o.get("count") or 0) > 0]
    if len(ops) < 3:
        return None  # too few fingerprints for a mode to mean anything
    counts = [int(o.get("count") or 0) for o in ops]
    baseline = max(set(counts), key=lambda c: (counts.count(c), -c))
    if baseline < 1:
        return None
    mult = max(2, int(os.environ.get("PERF_MCP_FOLD_MIN_MULTIPLE", "2") or "2"))
    cands = [
        o
        for o in ops
        if int(o["count"]) >= baseline * mult and int(o["count"]) % baseline == 0 and float(o.get("gap_ms") or 0.0) > 0
    ]
    if not cands:
        return None  # (1) not applicable: nothing recurs within a layer
    gap = sum(float(o.get("gap_ms") or 0.0) for o in cands)
    if gap < _material_gap_ms(float(prof.get("device_ms") or 0.0)):
        return None
    _kinds = ("fold", "structural-fold")
    clean = [a for a in attempts if (a.get("kernel_kind") or "").lower() in _kinds]
    if any(_ledger().is_win(a) for a in clean):  # (2) measured win clears it
        return None
    wedged = sum(1 for a in _load_attempts() if (a.get("kernel_kind") or "").lower() in _kinds and a.get("wedged"))
    if (len(clean) + wedged) >= int(os.environ.get("PERF_MCP_MAX_FOLD_ATTEMPTS", "3") or "3"):
        return None  # (3) capped
    worst = max(cands, key=lambda o: float(o.get("gap_ms") or 0.0))
    return {
        "op": str(worst.get("op_code") or "repeated_op"),
        "op_class": str(worst.get("bucket") or ""),
        "gap_ms": round(gap, 4),
        "bound_by": worst.get("bound_by"),
        "grid": worst.get("grid"),
        "weight_dtype": worst.get("weight_dtype"),
        "next_rung": "structural-fold",
        "reason": (
            "FOLD THE REPEATS -- a structural lever, and it comes BEFORE the kernel rungs. %r runs %d "
            "times while most op fingerprints in this profile run %d, so it is evaluated %dx per layer "
            "rather than once. Folding those calls into ONE wider matmul (concatenate the weights along "
            "the output dim and slice the result, or batch them into a single call) removes %d-1 "
            "dispatches per layer and gives the remaining matmul a larger, better-shaped problem. Two "
            "projections of the same shape on the same input -- a SwiGLU gate/up pair is the usual case "
            "-- are the textbook instance; a per-head or per-frame matmul is the other. Check FIRST that "
            "the operands are actually concatenable and that no data dependency orders them: if they are "
            "not, record_kernel_attempt(op=%r,'fold',measured_ms,False,note='none: <why not foldable>') "
            "and this clears. Otherwise fold it, measure, and record -- this gate clears ONLY on a "
            "MEASURED reduction."
        )
        % (
            str(worst.get("op_code") or ""),
            int(worst.get("count") or 0),
            baseline,
            int(worst.get("count") or 0) // baseline,
            int(worst.get("count") or 0) // baseline,
            str(worst.get("op_code") or "repeated_op"),
        ),
    }


def _is_datamove(op_code: str) -> bool:
    """Does this op exist to move/reshape data rather than compute on it?

    ASK THE CLASSIFIER, DO NOT KEEP A LIST. This started as
    _SLICE_MARKERS = ("slice","gather","concat","split","transpose","permute","reshape","index")
    -- a substring list matched against op names, which is both a hardcoded vocabulary and a second
    copy of a decision agent/opclass.py already owns. Every one of those ops classifies as
    `datamove` there, and a list like that misses the next op name by construction, which is the
    same reason _integrity.classify exists instead of an alias table.
    """
    try:
        from agent.opclass import classify_op

        return classify_op(op_code or "") == "datamove"
    except Exception:  # noqa: BLE001
        return False


def _order_gate(prof: dict, attempts: list) -> dict | None:
    """A projection sitting next to a slice/gather -- try the other order and measure.

    The two orders are not equivalent in cost. Projecting the whole tensor and then slicing does
    full-width math and throws part of it away; slicing first does narrower math but moves data to
    do it. Which wins depends on the ratio and on where the tensors live, and the honest answer is
    that nobody can tell from the shapes alone -- so measure both. The catalogue currently PRESCRIBES
    instead: 03 section 5 picks one of three head-split strategies "by regime", which is a rule where
    an experiment belongs.

    THE ADJACENCY IS REAL DATA, not a guess. tracy_tool._neighbours reads the capture in execution
    order (GLOBAL CALL COUNT) and records the most common op either side of each fingerprint, so
    "this projection feeds a slice" is an observation about what ran. Most-common, so a one-off
    pairing at a stack boundary does not trigger it.

    Same four parts as the other gates: applicability, detector, measured-win exit, cap.
    """
    cands = []
    for o in prof.get("open_ops") or []:
        # the op's OWN class, as the profile classified it -- not a substring of its name
        if str(o.get("bucket") or "").lower() != "matmul":
            continue
        if not (_is_datamove(o.get("next_op")) or _is_datamove(o.get("prev_op"))):
            continue
        if float(o.get("gap_ms") or 0.0) > 0:
            cands.append(o)
    if not cands:
        return None  # (1) not applicable: no projection is adjacent to a slice/gather
    gap = sum(float(o.get("gap_ms") or 0.0) for o in cands)
    if gap < _material_gap_ms(float(prof.get("device_ms") or 0.0)):
        return None
    _kinds = ("order", "structural-order")
    clean = [a for a in attempts if (a.get("kernel_kind") or "").lower() in _kinds]
    if any(_ledger().is_win(a) for a in clean):  # (2)
        return None
    wedged = sum(1 for a in _load_attempts() if (a.get("kernel_kind") or "").lower() in _kinds and a.get("wedged"))
    if (len(clean) + wedged) >= int(os.environ.get("PERF_MCP_MAX_ORDER_ATTEMPTS", "3") or "3"):
        return None  # (3)
    worst = max(cands, key=lambda o: float(o.get("gap_ms") or 0.0))
    _pair = str(worst.get("next_op") or worst.get("prev_op") or "the adjacent reshape")
    return {
        "op": str(worst.get("op_code") or "projection"),
        "op_class": str(worst.get("bucket") or ""),
        "gap_ms": round(gap, 4),
        "bound_by": worst.get("bound_by"),
        "grid": worst.get("grid"),
        "weight_dtype": worst.get("weight_dtype"),
        "next_rung": "structural-order",
        "reason": (
            "TRY BOTH ORDERS -- a structural lever, ahead of the kernel rungs. %r runs immediately "
            "beside %r in the capture. Projecting the full width and then slicing does math you throw "
            "away; slicing or gathering first does narrower math but moves data to do it. Which is "
            "cheaper depends on the width ratio and on where the tensors live, and it is NOT decidable "
            "from the shapes -- so measure both orders and keep the faster. If the order is forced (a "
            "data dependency, or the slice is what produces the projection's input), that is a real "
            "answer: record_kernel_attempt(op=%r,'order',measured_ms,False,note='none: order is "
            "forced because <reason>') and this clears. Otherwise swap them, check_pcc, measure, and "
            "record -- this gate clears ONLY on a MEASURED reduction."
        )
        % (str(worst.get("op_code") or ""), _pair, str(worst.get("op_code") or "projection")),
    }


def _conv_gate(prof: dict, attempts: list) -> dict | None:
    """MANDATORY conv weight preparation, for a model that actually has convs.

    WHY THIS IS A GATE AND NOT PLAYBOOK PROSE. `conv_pool` is a first-class op_class -- opclass.py
    maps Conv/Halo/Pool/GridSample/Upsample to it, and it is in STRUCTURAL_OP_CLASSES -- but the
    catalogue tags exactly two sections with it, `11_TT_LANG_KERNELS` and `12_CPP_METALIUM_KERNELS`:
    the bottom two rungs. So for a conv op the only thing recall_knobs can ever return is "author a
    custom kernel", and the ladder walks a conv straight down to hand-writing Metalium for work that
    `ttnn.prepare_conv_weights` does once at load. Voxtral-Mini-3B-2507 has audio_tower.conv1 and
    conv2, so this is not hypothetical for the model in hand.

    The expensive part of a conv is not the math, it is preparing weights into the layout the kernel
    wants. Done per call it is paid every forward; done once per (in_channels, out_channels, kernel,
    stride, padding, dtype, layout) it is paid once, and two call sites with identical parameters can
    share one prepared tensor rather than each building a byte-identical copy.

    Shaped exactly like _decode_gate, and for the same reasons:

      1. APPLICABILITY FIRST. No conv ops -> return None. _decode_gate's comment records what the
         alternative costs: an encoder-only model was ordered to add a KV-cache and burned three
         rewrites before the attempt cap released it. A model with no convs must never see this.
      2. MEASUREMENT-GATED EXIT. Clears on a MEASURED win, never on "attempted" -- the failure the
         host rung had, where one unhelpful attempt sealed an axis for 158 later rounds.
      3. CAPPED. A genuinely unpreparable conv yields after PERF_MCP_MAX_CONV_ATTEMPTS real tries,
         wedges included, so this cannot become a loop that never converges.
    """
    convs = [o for o in (prof.get("open_ops") or []) if str(o.get("bucket") or "").lower() == "conv_pool"]
    if not convs:
        return None  # (1) not applicable: this model has no convolution at all
    gap = sum(float(o.get("gap_ms") or 0.0) for o in convs)
    if gap < _material_gap_ms(float(prof.get("device_ms") or 0.0)):
        return None
    _kinds = ("conv-prep", "structural-conv")
    clean = [a for a in attempts if (a.get("kernel_kind") or "").lower() in _kinds]
    if any(_ledger().is_win(a) for a in clean):  # (2) measured win clears it
        return None
    wedged = sum(1 for a in _load_attempts() if (a.get("kernel_kind") or "").lower() in _kinds and a.get("wedged"))
    if (len(clean) + wedged) >= int(os.environ.get("PERF_MCP_MAX_CONV_ATTEMPTS", "3") or "3"):
        return None  # (3) capped
    _codes = ", ".join(sorted({str(o.get("op_code") or "") for o in convs})[:4])
    return {
        "op": "conv_weights",
        "op_class": "conv_pool",
        "gap_ms": round(gap, 4),
        "bound_by": "host",
        "grid": None,
        "weight_dtype": None,
        "next_rung": "structural-conv",
        "reason": (
            "MANDATORY conv weight preparation -- a lever SEPARATE from the kernel rungs, and it comes "
            "BEFORE them. This model has conv ops (%s) carrying %.3fms of gap. Prepare the weights ONCE "
            "per distinct (in_channels, out_channels, kernel_size, stride, padding, dtype, layout) with "
            "ttnn.prepare_conv_weights / prepare_conv_bias at load, hold the prepared tensors, and pass "
            "them to every ttnn.conv2d call instead of letting each call prepare its own. Two call sites "
            "whose parameters match must SHARE one prepared tensor, not build byte-identical copies. "
            "Authoring a tt-lang or C++ kernel does NOT resolve this and will NOT clear the gate -- a "
            "hand-written kernel still pays per-call preparation. Then record_kernel_attempt(op="
            "'conv_weights','conv-prep',measured_ms,beat_baseline) -- this clears ONLY on a MEASURED "
            "reduction. If the weights are provably already prepared once and shared, record it with "
            "note='none: <evidence>' and the cap will release it."
        )
        % (_codes or "conv", gap),
    }


def _reliable_forward_ms(dev: float) -> float | None:
    """The trace+1cq per-token number, or None when there isn't one.

    Used to fall back to the per-op device_ms, which is a CAPPED-WINDOW measurement (TT_PERF_LAYERS)
    scored against a full-model tok/s target -- the unit mix this function's own docstring blames
    for making every module read ABOVE_BAND. Since the caller can set can_stop from the result, a
    missing trace number must read as "cannot score", not as a number.
    """
    try:
        b = json.loads(_FULLPIPE_BASELINE_1CQ_PATH.read_text())
        ms = float(b.get("full_pipeline_ms") or 0.0)
        if ms > 0:
            return ms
    except Exception:  # noqa: BLE001
        pass
    return None


def _declared_stages_for_env() -> list:
    """This model's declared stages, for building the per-stage env knobs. [] when it declares none.

    Read from the model's own PIPELINE_STAGES via the contract's single reader, so the names the
    harness sets match the names the generated test binds and the report keys on.
    """
    try:
        from agent.model_contract import declared_stage_names

        return [str(x) for x in (declared_stage_names(_MODEL_ROOT) or []) if str(x).strip()]
    except Exception:  # noqa: BLE001 -- no declaration is no extra knobs, never a failed run
        return []


def _reliable_forward_unit() -> str:
    """The unit of work the gate's stored reading counts, or "" when it predates the marker."""
    try:
        return str(json.loads(_FULLPIPE_BASELINE_1CQ_PATH.read_text()).get("unit") or "").strip().lower()
    except Exception:  # noqa: BLE001
        return ""


# Set once the facts rebuild has been attempted in this process -- see _load_perf_target_inputs.
_PTIN_REBUILD_TRIED = False


def _load_perf_target_inputs() -> dict | None:
    """The bandwidth facts, REBUILT if the file is gone.

    perf_target_inputs.json is untracked and lives in the model directory the optimize loop reverts
    between attempts, and the producer runs once at setup and never overwrites -- so once a revert
    deleted it, nothing rebuilt it and every later report fell to the band-less ms floor for the rest
    of the run. That is why a RUN_REPORT could show "modeled floor / NO_BAND / no weight-bytes input"
    for a model whose facts had existed minutes earlier. Rebuilding here costs one safetensors header
    read and makes the ceiling survive the revert instead of depending on it.

    Read ONLY from a STATED model root. With the "." default this resolved against the working
    directory and adopted whatever perf_target_inputs.json happened to be lying there -- see
    _MODEL_ROOT_STATED. A fact about the model comes from the model's directory or not at all.
    """
    facts = None
    if _MODEL_ROOT_STATED:
        try:
            facts = json.loads((_MODEL_ROOT / "perf_target_inputs.json").read_text())
        except Exception:  # noqa: BLE001
            pass
    # The REBUILD path resolves the same relative "." and so re-adopted the working directory's file
    # even after the direct read was guarded -- one rule, two doors. Both are shut by the same test.
    # AND REBUILT IF NOBODY WHO KNOWS THE MODEL HAS WRITTEN IT. The condition was `facts is None`
    # -- the file being GONE. A file that merely cannot answer satisfies that check: the device
    # census creates perf_target_inputs.json as soon as it has walked the boards, carrying bytes and
    # nothing else, and stamps no `source` because it is not the facts producer. From then on this
    # read succeeds, the rebuild never fires, and the producer's keys -- total_params, the per-block
    # geometry, the layer counts -- never reach the file for the rest of the run.
    #
    # `source` is the file's own record of who wrote it, which is exactly the question here: absent,
    # no producer has ever written, so ask one to. The emit merges rather than replaces, so a file
    # another writer owns keeps every value it already carries.
    # ONCE PER PROCESS. The rebuild walks safetensors headers, and when it cannot produce facts (no
    # reachable checkpoint) it writes nothing -- so the file keeps no `source` and every later read
    # would pay for the same failed walk again. Retrying a rebuild that already declined to write
    # cannot succeed on the same inputs, so it is attempted once and the result stands.
    global _PTIN_REBUILD_TRIED
    _unproduced = isinstance(facts, dict) and not str(facts.get("source") or "").strip()
    if (facts is None or (_unproduced and not _PTIN_REBUILD_TRIED)) and _MODEL_ROOT_STATED:
        _PTIN_REBUILD_TRIED = True
        try:
            import importlib.util as _ilu

            _spec = _ilu.spec_from_file_location("cc_run_ptin", str(Path(__file__).parent / "run.py"))
            _run = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_run)
            # HAND IT THE MODEL ID. Setup calls this with a real hint; this path passed None, so
            # _resolve_model_id had only the demo dir to go on and `blocks` -- which needs the HF
            # config behind that id -- could not be rebuilt at all. run.py already owns the lookup.
            try:
                _mid_hint = _run._model_id_for_facts(_MODEL_ROOT) or None
            except Exception:  # noqa: BLE001
                _mid_hint = None
            _run._emit_perf_target_inputs(_MODEL_ROOT, _MODEL_ROOT, _mid_hint, _MANIFEST)
            facts = json.loads((_MODEL_ROOT / "perf_target_inputs.json").read_text())
        except Exception:  # noqa: BLE001
            # A FAILED REBUILD MUST NOT COST THE FACTS ALREADY READ, AND MUST NOT SKIP THE OVERRIDE.
            # This returned None -- right while the branch only ran with nothing in hand, but now
            # that an unproduced file also reaches here it would trade a partial answer for none.
            # Falling through instead keeps whatever was read (None included, which the single exit
            # below returns unchanged) and leaves both routes passing through the unit override.
            pass
    # THE OBSERVED UNIT OVERRIDES THE CACHED GUESS. The file's `unit` came from a table keyed on the
    # HF pipeline_tag, which names the TASK and cannot state whether a model loops: `text-to-speech`
    # covers XTTS, which emits tokens, and Kokoro, which is StyleTTS2 and produces a whole waveform in
    # one pass. trace_replay reports what the built pipeline ACTUALLY did, and once it has, that is a
    # fact rather than a lookup. Applied HERE, on both paths, because the file is written once at
    # setup -- before any trace exists -- and then read from cache for the rest of the run, so
    # correcting only the producer would never reach a run whose file already exists.
    _obs = str(os.environ.get("PERF_MCP_LAST_HEADLINE_UNIT") or "").strip().lower()
    if _obs and isinstance(facts, dict) and facts.get("unit") != _obs:
        facts["unit"] = _obs
    return facts


def _perf_target_status(rep: dict, dev: float) -> dict | None:
    """Opt-in DRAM-bandwidth band status for the target-driven stop (PERF_MCP_TARGET_BAND=1).

    The measured metric MUST match the target's unit or the band is meaningless: the config
    active_bytes ceiling (compute_target) is per-token tok/s, scored against the per-token trace
    (_reliable_forward_ms); the roofline aggregate floor (target_from_floor_ms) is a per-profile
    sum of op floors, scored against the per-profile device_ms (dev). Mixing them (per-token vs
    per-profile) made every module read ABOVE_BAND. Full-model prefers the config ceiling when
    perf_target_inputs.json is present, else the floor; per-module always uses that module's OWN
    floor (recomputed per module, never shared). Fail-open: returns None so the ladder stop is
    untouched on any issue."""
    if os.environ.get("PERF_MCP_TARGET_BAND") != "1":
        return None
    try:
        target, scope, is_llm = _select_perf_target(rep)
        measured_ms = _reliable_forward_ms(dev) if is_llm else dev
        # UNIT MATCH, OR NO VERDICT. `is_llm` is really "a config ceiling exists", which is true for a
        # diffusion (step) or classifier (inference) model too -- and _reliable_forward_ms returns
        # whatever the gate last measured. If the ceiling counts bytes per STEP and the reading counts
        # a whole pipeline pass, the ratio is arithmetic, not a comparison, and IN_BAND from it could
        # end a run at a target that was never tested. The gate now records the unit it measured
        # (TRACE_HEADLINE_UNIT, chosen where trace_replay picks its headline stage), so require the two
        # to agree. A reading with no recorded unit is accepted only for the token ceiling, which is
        # the pairing every existing run and test was built on.
        if is_llm and measured_ms is not None:
            want = str(getattr(target, "unit", "token") or "token").lower()
            got = str(_reliable_forward_unit() or "").lower()
            if got != want and not (got == "" and want == "token"):
                return {
                    "status": "UNIT_MISMATCH",
                    "scope": scope,
                    "target_unit": want,
                    "measured_unit": got or "unrecorded",
                    "note": (
                        "no verdict: the ceiling is per %s and the measurement counts per %s"
                        % (want, got or "unknown unit")
                    ),
                }
        if measured_ms is None:
            # No trace+1cq per-token number exists, so nothing here is comparable to a full-model
            # tok/s target. Return no score rather than scoring the capped-window device_ms, which
            # is what let the stop gate declare a model IN_BAND from a missing tmp file.
            return None
        s = perf_target.score(target, measured_ms)
        s["scope"] = scope
        if perf_target.unknown_dtypes():
            s["ceiling_degraded"] = "unknown dtype(s) %s fell back to the default byte width" % (
                ", ".join(perf_target.unknown_dtypes()),
            )
        return s
    except Exception:  # noqa: BLE001
        return None


def _select_perf_target(rep: dict):
    """Pick the roofline target for this pipeline, STATIC (no measurement mixed in). Returns
    ``(target, scope, has_unit_ceiling)``. Model-level config ceiling (compute_target, per-token tok/s)
    when perf_target_inputs.json exists and this is not a per-module run; otherwise the per-profile
    roofline floor (target_from_floor_ms). Shared by the stop gate and the report snapshot so both
    agree on the target."""
    module_level = os.environ.get("TT_PERF_MODULE_LEVEL") == "1"
    if not module_level:
        mf = _load_perf_target_inputs()
        anchored = _anchored_ceiling_bytes(mf)
        if not mf and anchored > 0:
            # THE FACTS ARE GONE BUT THE BASELINE SURVIVES. Losing the file must not silently downgrade
            # the gate from a bandwidth ceiling to the band-less ms floor -- that is the anchor's whole
            # purpose, and it was only wired into the report.
            mf = _anchored_ceiling_facts()
        if mf:
            tp = int(os.environ.get("TT_PERF_MESH_COLS", "1") or "1")
            # THE GATE AND THE REPORT MUST DIVIDE BY THE SAME BYTES. The report prices each stage by
            # the subtree it streams -- a decode token reads the language backbone and never the
            # audio encoder -- while this handed compute_target the WHOLE model. On a two-tower model
            # that ceiling is too LOW, so the measured rate looks closer to it than it is and the run
            # can be declared at-the-floor while the report, dividing correctly, still shows headroom.
            #
            # Applied only when the model's own facts carry both halves -- which stage runs which
            # subtree, and that subtree's MEASURED resident bytes. A checkpoint ratio is not used
            # here: it states disk precision, and a wrong ceiling in the stop gate is worse than a
            # conservative one. Absent either, the whole model is the read set, exactly as before.
            _share = _recurring_subtree_share(mf)
            mf = (
                dict(mf, device_weight_bytes=int(float(mf.get("device_weight_bytes") or 0) * _share))
                if (_share < 1.0 and mf.get("device_weight_bytes"))
                else mf
            )
            anchored = anchored * _share if (anchored and _share < 1.0) else anchored
            if _share < 1.0:
                print(
                    "  [perf-mcp] stop-gate ceiling scaled to the recurring stage's subtree "
                    "(%.1f%% of resident weights), matching the roofline table" % (100.0 * _share),
                    file=sys.stderr,
                    flush=True,
                )
            return (
                perf_target.compute_target(mf, _ENV, tp_degree=tp, bytes_per_unit=anchored),
                "model",
                True,
            )
    return perf_target.target_from_floor_ms(rep.get("modeled_floor_ms")), ("module" if module_level else "model"), False


def _recurring_subtree_share(mf: dict) -> float:
    """Fraction of resident weights the RECURRING stage streams; 1.0 when it cannot be established.

    MEASURED ONLY. device_section_bytes is the census walking the built model, so this is the served
    split; the checkpoint's ratio is deliberately not accepted as a substitute here, because it
    states disk precision and the stop gate acts on the answer. stage_roots says which subtree each
    stage runs, and the recurring stage is the one the headline unit counts.

    1.0 means "the whole model", which is both the honest answer for a single-tower model and the
    behaviour that predates this.
    """
    try:
        roots = (mf or {}).get("stage_roots") or {}
        dev = (mf or {}).get("device_section_bytes") or {}
        total = float((mf or {}).get("device_weight_bytes") or 0.0)
        if not roots or not dev or total <= 0:
            return 1.0
        # The stage the per-unit ceiling is about: the one the run measured its headline against.
        _rec = str(os.environ.get("PERF_MCP_RECURRING_STAGE", "") or "").strip()
        if not _rec:
            _isl = read_stage_isl_map() or {}
            _ones = [k for k, v in _isl.items() if int(v or 0) == 1]
            _rec = _ones[0] if len(_ones) == 1 else ""
        if not _rec or _rec not in roots:
            return 1.0
        mine = float(dev.get(str(roots[_rec])) or 0.0)
        if mine <= 0 or mine > total:
            return 1.0
        return mine / total
    except Exception:  # noqa: BLE001 -- an unestablished share is the whole model, never a failure
        return 1.0


# A depth that is not a unit of work. The byte anchor's producer used to fall back to the literal
# string "unit" when no unit had been observed yet, so legacy ledgers carry rows keyed under it. The
# fallback scans below read a row's depth AS the unit, so accepting one of these does not merely
# return the wrong bytes -- it reports the model's unit of work as "unit", and the band, the at-floor
# verdict and the headline rate all inherit that. Refused on both scans; a run whose only anchor is a
# placeholder is a run with no anchor, which lands on the documented floor fallback.
_PLACEHOLDER_DEPTHS = frozenset({"unit", "unknown", ""})


def _is_real_unit(depth) -> bool:
    return str(depth or "").strip().lower() not in _PLACEHOLDER_DEPTHS


def _anchored_ceiling_bytes(mf: dict) -> float:
    """The PINNED baseline bytes-per-unit from the ledger, or 0.0 when nothing is pinned.

    The report already read this anchor while the stop gate computed from perf_target_inputs.json --
    and that file lives in the model directory the optimize loop REVERTS between attempts, so the gate
    could judge against a different ceiling than the report printed for the same run (and against a
    moving one, since a lever that shrinks weights shrinks the facts). Both sides now read the same
    write-once anchor. Keyed exactly like the renderer's lookup: depth = the model's own unit, model =
    the root's name, task = PERF_MCP_TASK. Best-effort; 0.0 means "compute from the facts as before".
    """
    try:
        led = _ledger()
        model = Path(_MODEL_ROOT).name if _MODEL_ROOT else ""
        task = os.environ.get("PERF_MCP_TASK", "main")
        depth = str((mf or {}).get("unit") or "").strip().lower()
        if depth:
            mb = led.anchor_value(led.KIND_ACTIVE_BYTES, depth=depth, model=model, task=task)
            if mb and float(mb) > 0:
                return float(mb) * 1e6
        # NO UNIT TO KEY ON. perf_target_inputs.json is untracked, so the loop's revert can DELETE it --
        # and then the gate had no facts, skipped the ceiling entirely and fell to the band-less ms
        # floor, while the report still showed the pinned ceiling. The anchored row carries the unit it
        # was pinned under, so the baseline is recoverable without the file and without guessing.
        for r in led.rows(led.KIND_ACTIVE_BYTES, led.PHASE_BEFORE, model, task):
            if not _is_real_unit(r.get("depth")):
                continue
            try:
                v = float(r.get("value_ms"))
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v * 1e6
        return 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _anchored_ceiling_facts():
    """Minimal facts rebuilt from the byte anchor when the facts FILE is gone, or None.

    Returns None rather than defaulting the unit. A ceiling whose unit is unknown cannot be scored --
    a per-step ceiling against a per-token measurement is not a comparison -- and defaulting to
    "token" is the exact bug that once labelled every diffusion and classifier model per-token. No
    recoverable unit means no ceiling, which lands on the floor fallback: weaker, but not wrong.
    """
    try:
        led = _ledger()
        for r in led.rows(
            led.KIND_ACTIVE_BYTES,
            led.PHASE_BEFORE,
            Path(_MODEL_ROOT).name if _MODEL_ROOT else "",
            os.environ.get("PERF_MCP_TASK", "main"),
        ):
            d = str(r.get("depth") or "").strip().lower()
            if _is_real_unit(d):
                return {"unit": d}
    except Exception:  # noqa: BLE001
        pass
    return None


def _dominant_peak_flops(rep: dict) -> float:
    """Peak FLOP/s at the fidelity carrying the most FLOPs in this report, or 0.0.

    THE MODE THE MODEL ACTUALLY RUNS AT, by FLOP share -- a measurement, not the hifi4 default
    chip_peak_flops falls back to when handed nothing. Returns 0.0 rather than guessing when the
    report carries no fidelity-tagged FLOPs, so a run with nothing to measure pins nothing and the
    renderer keeps its existing behaviour.
    """
    try:
        ops = [o for o in ((rep or {}).get("open_ops") or []) if o.get("flops")]
        if not ops:
            return 0.0
        agg: dict = {}
        for o in ops:
            f = str(o.get("fidelity") or "hifi4").lower()
            agg[f] = agg.get(f, 0) + int(o["flops"])
        if not agg:
            return 0.0
        dom = max(agg.items(), key=lambda kv: kv[1])[0]
        from agent.environment import ARCH_FACTS
        from agent.perf_target import chip_peak_flops as _cpf

        _arch = str(os.environ.get("PERF_MCP_ARCH") or "blackhole").strip().lower()
        return float(_cpf(ARCH_FACTS.get(_arch) or {}, dom) or 0.0)
    except Exception:  # noqa: BLE001 -- a peak that cannot be derived is simply not pinned
        return 0.0


def _persist_throughput(rep: dict, prof_for_peak: dict | None = None) -> None:
    """Write a FRESH static roofline-target snapshot (theoretical ceiling / band / active_bytes /
    peak BW / floor) each time we profile. Deliberately stores NO measured number — the report
    computes measured tok/s + utilization from the exact ms it is reporting, so a stale measured can
    never leak in (this is the fix for the old '+0.0%'-style stale readout). Best-effort; never raises.

    The snapshot tracks the CURRENT build, so the floor in it is deliberately recomputed every time.
    The floor is ALSO pinned once into the measurement ledger here -- at the point it is produced,
    which is the only place that knows it describes the state the run started from. The report reads
    that anchor, so a later, faster build cannot lower its own target: llama3_1_8b_p150 read 83% ->
    55% at-floor while measuring FASTER, purely because the floor was recomputed each round."""
    try:
        target, scope, is_llm = _select_perf_target(rep)
        _win = _depth_in_force()
        snap = {
            "scope": scope,
            "has_unit_ceiling": bool(is_llm),
            "theoretical_rate": target.theoretical_rate,
            "band": [target.band[0], target.band[1]],
            "active_bytes": target.active_bytes,
            "peak_bw_gbps": float((_ENV or {}).get("dram_bw_gbps", 0.0)),
            "tp_degree": target.tp_degree,
            # The sustained fraction folded into theoretical_rate, so the report can label the ceiling
            # achievable-not-spec and 512 GB/s stays recoverable from it.
            "bw_fraction": getattr(target, "bw_fraction", 1.0),
            "bytes_source": getattr(target, "bytes_source", ""),
            # THE UNIT MUST TRAVEL WITH THE CEILING. Without this key the report's
            # `throughput.get("unit") or "token"` fell back to token for EVERY model, so a diffusion
            # model printed its steps/s ceiling labelled "tok/s/u" and the byte anchor was looked up
            # under the wrong depth -- the unit plumbing existed end to end except for the one line
            # that put the unit in the snapshot, which is where a test passing `unit` by hand hid it.
            "unit": getattr(target, "unit", "token"),
            "modeled_floor_ms": rep.get("modeled_floor_ms"),
            # The floor is a SUM OVER THE PROFILED OPS, so it scales with the window it was computed
            # at, and the FLOOR form is rendered against the measurement -- so the floor keeps the real
            # window (_win) below. The CONFIG/anchored CEILING (has_unit_ceiling), however, divides by
            # DEPTH-INDEPENDENT full-model bytes (weights or params): it describes the WHOLE model no
            # matter what window the profile ran, so it must read "all". Stamping the profiler window
            # onto it made the report's depth guard mismatch a full-model ceiling against the all-layer
            # measurement and blank utilization -- the coverage default runs TT_PERF_LAYERS=2, so that
            # false mismatch fired on essentially every run.
            "perf_layers": "all" if is_llm else _win,
        }
        _throughput_path().write_text(json.dumps(snap))
        # THE BYTE ANCHOR, WRITTEN WHERE THE UNIT IS FINALLY KNOWN. KIND_FLOOR and KIND_PEAK_FLOPS are
        # both pinned here; active_bytes was not, and its only writer (run.py) is guarded on `if
        # _unit:` at a point BEFORE any trace has reported one -- so it never fired. run.py's own
        # comment promises the gap is covered ("once a trace reports, the rebuild path anchors with
        # the observed unit"); nothing implemented it, and _anchored_ceiling_facts is a reader.
        #
        # Unpinned, the memory roof is re-derived from the LIVE census on every step: this function
        # runs from termination_check, which runs every step, so a dtype win (bf16 -> bf8_b halves a
        # weight) shrinks the census and the ceiling follows it down mid-run -- the target retreating
        # ahead of the measurement, which is the one thing an anchor exists to prevent. The compute
        # roof was already protected this way; the memory roof was not.
        try:
            _ab = int(getattr(target, "active_bytes", 0) or 0)
            _u = str(snap.get("unit") or "").strip().lower()
            if _ab > 0 and _is_real_unit(_u):
                _ledger().anchor(
                    _ledger().KIND_ACTIVE_BYTES,
                    float(_ab) / 1e6,
                    depth=_u,
                    mode="bytes_mb",
                    source="_persist_throughput (%s)" % str(snap.get("bytes_source") or "census")[:60],
                    model=_MODEL_ROOT.name if _MODEL_ROOT else "",
                )
        except Exception:  # noqa: BLE001 -- a pin that cannot be written must not cost the snapshot
            pass
        _f = rep.get("modeled_floor_ms") if isinstance(rep, dict) else None
        if _f:
            _ledger().anchor(
                _ledger().KIND_FLOOR,
                _f,
                depth=_win,
                source="_persist_throughput",
                model=_MODEL_ROOT.name if _MODEL_ROOT else "",
            )
        # THE COMPUTE ROOF'S DENOMINATOR, pinned here for the same reason the floor above it is.
        #
        # The snapshot tracks the current build by design, and the peak read from it moves whenever
        # the `fidelity` rung lands: _promote_baseline replaces the profile PICTURE on every
        # profile_model call while ratcheting only device_ms, and the peak is read off that picture.
        # Blackhole's modes are 4x apart, so one matmul changing mode can double the ceiling -- on
        # voxtral, hifi4 carries 4.037e12 FLOPs against hifi2's 3.299e12, a 5.0%-of-total margin, and
        # the largest single hifi4 matmul is 16.7%.
        #
        # KEYED ON THE UNIT, exactly like the byte anchor, because the renderer looks it up by the
        # unit the ceiling describes. Not on _win: a peak is a property of the math mode and the
        # silicon, not of how many layers the profiler happened to build.
        # PER-STAGE PEAKS pinned beside the per-stage bytes, when the capture marked its stages.
        try:
            for _st, _bk in ((prof_for_peak or {}).get("stage_buckets") or {}).items():
                _p = _dominant_peak_flops({"open_ops": [o for b in _bk for o in (b.get("top_ops") or [])]})
                if _st and _p > 0:
                    _ledger().anchor(
                        _ledger().KIND_PEAK_FLOPS,
                        _p,
                        depth=str(_st).strip().lower(),
                        mode="roofline",
                        source="_persist_throughput per-stage",
                        model=_MODEL_ROOT.name if _MODEL_ROOT else "",
                    )
        except Exception:  # noqa: BLE001
            pass
        _pk = _dominant_peak_flops(rep)
        if _pk > 0:
            _ledger().anchor(
                _ledger().KIND_PEAK_FLOPS,
                _pk,
                depth=str(snap.get("unit") or "token"),
                mode="roofline",
                source="_persist_throughput",
                model=_MODEL_ROOT.name if _MODEL_ROOT else "",
            )
    except Exception:  # noqa: BLE001
        pass


def _read_throughput() -> dict | None:
    try:
        p = _throughput_path()
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _ceiling_armed(target, rep: dict) -> tuple:
    """(armed, why). May an IN_BAND verdict END a run? Returns False plus the reason when not.

    IN_BAND overrides every blocking op, so it is the most consequential thing the ceiling feeds. The
    ceiling itself is now rendered for every model -- which is the point, a reader should always see the
    bound -- but "shown" and "trustworthy enough to terminate on" are different bars, and only the second
    one needs evidence:

      * the divisor must be an EXACT param count. The name-derived and file-size fallbacks are estimates,
        and a divisor that is too large lowers the band under what the model already does, which is the
        direction that stops a run early.
      * `bound_by` must match the profile's dominant bucket. XTTS is dispatch-bound at 47 tok/s against a
        bandwidth ceiling of 1724: the number is arithmetically right and describes a constraint that is
        not binding, so reaching 60% of it says nothing about being done.
      * the reading must cover the FULL model. A capped layer window streams a fraction of the bytes, so
        its ceiling describes a fraction of the work.

    Unarmed is exactly today's no-band behaviour: the run keeps optimizing and terminates on the ladder
    instead. PERF_MCP_ARM_BAND=0 disables the band stop outright.
    """
    if os.environ.get("PERF_MCP_ARM_BAND") == "0":
        return False, "band stop disabled (PERF_MCP_ARM_BAND=0)"
    src = str(getattr(target, "bytes_source", "") or "")
    if "params rule" not in src and "anchored" not in src:
        return False, "divisor is an estimate (%s), not an exact param count" % (src or "unknown")
    depth = str((_read_throughput() or {}).get("perf_layers") or "").strip().lower()
    if depth and depth not in ("all", "0", "none"):
        return False, "measurement covers a %s-layer window, not the full model" % depth
    bound = str(getattr(target, "bound_by", "") or "memory").lower()
    buckets = [b for b in (rep or {}).get("buckets") or [] if isinstance(b, dict)]
    dom = max(buckets, key=lambda b: b.get("device_ms") or 0.0, default=None)
    dom_bound = str(((dom or {}).get("tags") or {}).get("bound") or "").lower()
    dom_id = str((dom or {}).get("id") or "")
    # host_overhead dominating means the run is dispatch-bound whatever the op tags say.
    if dom_id == "host_overhead" or dom_bound in ("host", "dispatch"):
        if bound != "dispatch":
            return False, "profile is dominated by %s, but the ceiling is %s-bound" % (dom_id or dom_bound, bound)
    return True, ""


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
        prof = _profile_with_zero_row_retry()
    except Exception as exc:  # noqa: BLE001
        _note_device_crash("termination_check", str(exc))
        # BEFORE THE VERDICT, NOT AFTER IT. Run 18 halted telling the operator to reboot the host
        # while two stale processes still held /dev/tenstorrent; the supervisor reaped them on its way
        # out and a plain `tt-smi -r` then recovered all four boards in ninety seconds. The resets
        # that "failed" had run against a held device. One retry, only if reaping actually freed
        # something -- see device_recovery.retry_once_after_reaping for why that is not a loosened
        # limit.
        if _recovery_exhausted():
            try:
                if _dr().retry_once_after_reaping(
                    "termination_check",
                    lambda tgt: _board_reset("termination_check", "post-reap retry (target=%s)" % tgt, target=tgt),
                    log=lambda m: sys.stderr.write("[perf-mcp] %s\n" % m),
                ):
                    prof = _profile_with_zero_row_retry()
            except Exception:  # noqa: BLE001
                pass
        if _recovery_exhausted():
            # STOP POLLING A DEAD BOARD. can_stop=False means "work remains"; a device that cannot be
            # recovered is not work remaining, and the agent is instructed never to stop while it is
            # False -- which is how a dead card produced hours of 6-minute no-op gate calls.
            # NAME WHAT THE OPERATOR MUST DO. "unrecoverable after N attempts" is an inference from
            # repeated failure; when the kernel has reported a board-management fault it is a
            # diagnosis, and the two call for different actions. Run 39 halted with the vague form and
            # sat dead until morning -- a reboot would have taken two minutes.
            _reboot = False
            try:
                _reboot = _dr().board_needs_host_reboot()
            except Exception:  # noqa: BLE001
                pass
            _why = (
                "the driver reports a board-management fault no PCIe reset can clear "
                "('Failed to set initial power state') -- REBOOT THE HOST, then re-run: %s"
                if _reboot
                else "device could not be recovered after %d reset attempts: %s"
            ) % ((str(exc)[-300:],) if _reboot else (_RESET_FAIL_LIMIT, str(exc)[-300:]))
            return {
                "can_stop": True,
                "halt": "needs_host_reboot" if _reboot else "device_unrecoverable",
                # ONE KEY FOR THE REASON A RUN HALTED. This carried the message in `error` while the
                # OTHER halt (tt-lang) carried it in `halt_reason`, and the supervisor reads
                # halt_reason -- so a dead board printed an EMPTY reason under a message hardcoded to
                # "install tt-lang first, then re-run". The operator was told to install a toolchain
                # when the card needed a host reboot, and the run sat dead until morning. `error`
                # stays for readers that already use it; the two are the same string.
                "halt_reason": _why,
                "error": _why,
            }
        return {"can_stop": False, "error": f"profiler crashed: {str(exc)[-500:]}"}
    _note_device_ok()
    dev = round(float(prof.get("device_ms", 0.0)), 4)
    try:
        rep = roofline.residual_report(prof, _ENV)
    except Exception as exc:  # noqa: BLE001
        return {"can_stop": False, "error": f"residual_report failed: {str(exc)[-400:]}"}
    # Persist the roofline-target snapshot the RUN_REPORT reads. It used to be written ONLY by
    # profile_model, which the agent does not call in the normal loop (the engine sets the baseline
    # and the loop steers with termination_check) -- so the snapshot was often absent and the report
    # fell to NO_BAND. termination_check runs every step and already has `rep`, so writing it here
    # makes the ceiling reliably available. Best-effort; _persist_throughput never raises.
    _persist_throughput(rep)
    at_floor = bool(rep.get("at_floor"))
    # TRIED is history, not a property of the current baseline. This fed _op_ladder_status from the
    # resume-filtered live log, so a rung tried against an earlier baseline read as untried and got
    # handed out again -- the same matmul returned at `grid` in four consecutive runs. Verdicts still
    # come from _load_attempts(); only the tried/not-tried question reads the full history.
    attempts = [a for a in _load_attempts_all() if a.get("kernel_detected_in_source")]
    blocking, cleared = [], []
    material = _material_gap_ms(dev)
    for o in rep.get("open_ops") or []:
        gap = o.get("gap_ms") or 0.0
        # eff_gap is precision-aware for matmuls, == gap otherwise; keep the op if EITHER is material.
        eff_gap = o.get("eff_gap_ms")
        if eff_gap is None:
            eff_gap = gap
        if max(gap, eff_gap) < material:
            continue
        op_code = o.get("op_code") or o.get("bucket") or ""
        entry = {
            "op": op_code,
            "op_class": o.get("bucket"),
            "gap_ms": round(float(gap), 4),
            "achievable_gap_ms": o.get("achievable_gap_ms"),
            "eff_gap_ms": round(float(eff_gap), 4),
            "bound_by": o.get("bound_by"),
            "grid": o.get("grid"),
            "weight_dtype": o.get("weight_dtype"),
        }
        # THE STAGE AXIS, CARRIED. router has had "regime" among its DIMENSIONS all along,
        # declare_stages() fills its vocabulary from the model's own PIPELINE_STAGES, and
        # recall_knobs(regime=...) narrows on it -- but a per-op target never carried one, so the only
        # value ever to reach the axis was the host gate's generation_loop. A knob section written for
        # one stage was therefore unreachable from an op in that stage: the KV-cache section is
        # `op_class: attention, datamove` + `regime: decode`, and narrowing by op_class alone cannot
        # find it.
        #
        # ABSENT RATHER THAN "na" when the op does not know. "na" is a VALUE in that vocabulary -- it
        # asserts "belongs to no stage" -- so passing it would narrow the search to levers tagged
        # stage-less rather than leaving the axis open. Ops carry "na" today because their source is
        # unwired (tracy_tool: TBD(regime-source)), and an unwired source must read as silence.
        _rg = str(o.get("regime") or "").strip().lower()
        if _rg and _rg != "na":
            entry["regime"] = _rg
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
    conv_block = _conv_gate(prof, attempts)
    if conv_block:
        blocking.append(conv_block)
    fold_block = _fold_gate(prof, attempts)
    if fold_block:
        blocking.append(fold_block)
    order_block = _order_gate(prof, attempts)
    if order_block:
        blocking.append(order_block)
    blocking.sort(key=lambda b: -(b.get("eff_gap_ms") or b.get("gap_ms") or 0.0))
    can_stop = not blocking
    # AND NOTHING MATERIAL MAY BE UNTRIED. `blocking` empties as each op's checklist fills, so an op
    # that was never SELECTED never appears there and never blocks -- which is how a run ends with
    # known gaps untouched. Attempted, not won: a measured dead end clears an op, per the existing
    # contract that "even a measured kernel that does NOT beat ttnn clears the op as 'tried'". The
    # round budget still bounds the run; it lives in the loop (`while rounds < max_rounds`), not here.
    _untried = _untried_material_ops(blocking, _load_attempts_all())
    if _untried:
        can_stop = False
    pt_status = _perf_target_status(rep, dev)
    # STOPPING IS ONLY ALLOWED AGAINST A REAL BANDWIDTH BAND. 60-80% of peak DRAM bandwidth is a
    # hardware fact; 60-80% of 1000/modeled_floor is not -- the floor is a sum of per-op minimum times
    # over one profiling window, so that rate has no hardware peak behind it. The fallback target used
    # to manufacture such a band, and reaching 60% of it set can_stop=True and overrode every blocking
    # op: a run could be declared done against a range never derived from the hardware. The fallback
    # now carries no band (status NO_BAND) and the nonzero check keeps it that way.
    _band = (pt_status or {}).get("band") or (0, 0)
    _armed, _why = _ceiling_armed(_select_perf_target(rep)[0], rep)
    if pt_status and pt_status.get("status") == "IN_BAND" and _band[0] and _band[1]:
        # ...AND ONLY ON EVIDENCE THAT SUPPORTS ENDING A RUN. See _ceiling_armed: the ceiling is shown for
        # every model now, but an estimated divisor, a non-binding bound or a truncated window must not
        # override a blocking op. Unarmed behaves exactly as no-band does -- keep optimizing.
        if _armed:
            can_stop = True
        else:
            pt_status["band_stop_disarmed"] = _why
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
    if next_target:
        # DELIVER the pre-pass recommendation as data. Only when matmul_sweep.json exists AND holds
        # this exact shape; otherwise the key is simply absent and nothing downstream changes.
        _ws = _warm_start_for(_MODEL_ROOT, next_target.get("op"))
        if _ws:
            next_target["warm_start"] = _ws
            next_target["warm_start_note"] = (
                "matmul-sweep measured this shape EAGER and PCC-gated it: apply %s FIRST on the "
                "knob:fidelity/knob:dtype rung, then check_pcc + measure_candidate and commit/revert "
                "as usual -- it is a starting guess, not a verdict." % _ws
            )
    _persist_target(next_target)
    return {
        "can_stop": can_stop,
        "halt": bool(halt),
        "halt_reason": halt.get("reason") if halt else None,
        "device_ms": dev,
        "at_floor": at_floor,
        "residual_gap_ms": rep.get("residual_gap_ms"),
        "material_gap_threshold_ms": round(material, 4),
        "perf_target": pt_status,
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
            "knob:grid -> knob:fidelity -> knob:dtype -> knob:shard -> tt-lang -> cpp. record_kernel_attempt for EACH rung (knobs too: "
            "kind='grid'/'fidelity'/'dtype'/'shard'; kernels: 'tt-lang'/'cpp'). A later rung does NOT clear an op while an "
            "earlier rung is untried. WRITE-BACK: after you COMMIT a win you IMPROVISED (recall_knobs "
            "had no match), call distill_knob to persist it for reuse; if the win re-used a provisional "
            "lever from another model, pass its id to distill_knob to graduate it. Re-run "
            "termination_check after each rung."
        ),
    }


def _tp_mesh_shapes(ttnn_mod) -> list:
    """Mesh shapes to try for the TP levers, best first, taken from what the tool ALREADY KNOWS.

    Both TP entry points opened a literal MeshShape(2, 2) -- the QB2 bench machine the lever was
    written on (f08b2ce8cc: "Verified on the real (2,2) mesh"). Nothing else fits that, and it is why
    the all_gather bug survived: 2x2 was the only shape the lever ever ran on.

    Deriving a shape arithmetically is no better, because chip count does not imply topology: QB2 has
    4 chips but its fabric is 2x2 only (8 QSFP-DD, 2 links/chip), so a "sensible" 1x4 ring cannot be
    formed at all. The tool already records the answer in two places, so ask them in order:

      1. PERF_MCP_TP_MESH          an explicit pin, for debugging a fabric problem
      2. TT_PERF_MESH_ROWS/COLS    the mesh THIS RUN planned and the model actually opens
                                   (optimize._derive_topology_env -> plan_parallelism), read through
                                   perf_adapter.resolve_mesh_shape so the lever measures the same
                                   topology as everything else in the run. Asked with 0x0 defaults,
                                   so an UNPARSEABLE setting returns 0x0 and is rejected here: taking
                                   resolve_mesh_shape's own 1x1 default as a plan would run the TP
                                   sweep on ONE chip and report it as the planned topology
      3. the box registry          scripts/tt_hw_planner/hardware.HARDWARE, matched on the tt-smi
                                   board_type: default_mesh first, then the box's own canonical
                                   mesh_shapes that use every chip
      4. chip count                last resort, only when the board is unrecognised

    Every entry is a CANDIDATE; the caller opens the first that works.
    """
    override = (os.environ.get("PERF_MCP_TP_MESH") or "").strip()
    if override:
        try:
            r, c = (int(x) for x in override.lower().replace("x", ",").split(",")[:2])
            return [(r, c)]
        except Exception:  # noqa: BLE001
            pass

    planned = []
    if (os.environ.get("TT_PERF_MESH_ROWS") or os.environ.get("TT_PERF_MESH_COLS") or "").strip():
        try:
            from agent.perf_adapter import resolve_mesh_shape

            r, c = resolve_mesh_shape(0, 0)
            if r >= 1 and c >= 1:
                planned = [(int(r), int(c))]
        except Exception:  # noqa: BLE001
            planned = []

    try:
        num = int(ttnn_mod.get_num_devices())
    except Exception:  # noqa: BLE001
        num = 0

    box_shapes = []
    box = _tp_box(num)
    if box is not None:
        canonical = [tuple(m) for m in (box.mesh_shapes or []) if len(m) == 2]
        chips = int(getattr(box, "chips", 0) or 0)
        full = [m for m in canonical if m[0] * m[1] == chips] or canonical
        default = tuple(box.default_mesh) if box.default_mesh else None
        if default and default in full:
            box_shapes.append(default)
        for m in full:
            if m not in box_shapes:
                box_shapes.append(m)
        if not num:
            num = chips

    out = []
    for s in planned + box_shapes:
        if s not in out:
            out.append(s)
    if out:
        return out

    if num < 1:
        return [(1, 1)]
    pairs = []
    for r in range(2, int(num**0.5) + 1):
        if num % r == 0:
            pairs.append((r, num // r))
    pairs.reverse()
    out = [(1, num)]
    for s in pairs:
        if s not in out:
            out.append(s)
    return out


def _tp_box(num_devices: int = 0):
    """The Box this machine is, from the tt-hw-planner registry, or None.

    Matched on the tt-smi board_type the way probes.board_to_arch does, then REQUIRED to agree with
    the devices actually present -- a registry entry describes one box, and several entries describe a
    single CARD. This machine reports board_type p300 with 4 devices visible: the P300 entry is a
    2-chip dual-ASIC card, so two of them are installed. Trusting its mesh_shapes there would sweep
    2 of the 4 chips and report the result as the board's TP. A mismatch means the system is not that
    box, so the caller falls through to deriving from the real device count. An UNKNOWN device count
    returns None for the same reason: without it the box cannot be confirmed, and picking one by
    board-name similarity is a guess dressed as a lookup.

    Guarded: the registry is a sibling package, and the levers must still work without it.
    """
    try:
        from scripts.tt_hw_planner.hardware import HARDWARE
    except Exception:  # noqa: BLE001
        return None
    board = ""
    try:
        from agent.probes import tt_smi_probe

        board = str((json.loads(tt_smi_probe()) or {}).get("card") or "").strip().lower()
    except Exception:  # noqa: BLE001
        board = ""
    matches = []
    for box in HARDWARE:
        for bt in box.board_types or ():
            if bt and board.startswith(bt.lower()):
                matches.append(box)
                break
    if not matches:
        return None
    if not num_devices:
        return None
    exact = [b for b in matches if int(getattr(b, "chips", 0) or 0) == int(num_devices)]
    return exact[0] if exact else None


def _open_tp_mesh(ttnn_mod):
    """Open the widest mesh the board actually supports; raise the last error if none open."""
    last = None
    for rows, cols in _tp_mesh_shapes(ttnn_mod):
        try:
            return ttnn_mod.open_mesh_device(ttnn_mod.MeshShape(rows, cols))
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise last if last else RuntimeError("no mesh shape could be opened")


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
        mesh = _open_tp_mesh(ttnn)
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

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = _open_tp_mesh(ttnn)
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
