import ast
import json
import os
import re
import time
from pathlib import Path

# ONE OWNER FOR THE NAME. bringup_plan defines it with the note that "the emitter, the lookup and
# the already-scaffolded short-circuit must agree on the name" -- so this asks rather than spelling a
# third copy. Imported lazily at module scope through a guard because trace_gate is imported from
# inside functions in both directions and a hard import would couple the two at load time.
try:
    from .bringup_plan import BRINGUP_STATUS_FILENAME as _STATUS_FILE
except Exception:  # noqa: BLE001 -- a partial checkout must still gate
    _STATUS_FILE = "bringup_status.json"


def _component_status_dirs(demo_dir):
    """The bring-up dirs of a COMPOSITE model, discovered from what its own pipeline imports.

    A composite e2e package holds no status file of its own: its parts were brought up separately and
    each keeps its status beside its own stubs, anywhere in the checkout. Which dirs those are cannot
    be guessed from the demo path and must not be typed -- so the PIPELINE is asked, by walking the
    imports it actually makes (the same closure the correctness cache keys on) and keeping every
    directory along the way that carries a status file. A model that renames or relocates its
    components still resolves, because nothing here assumed where they were.

    This is the case the gate was failing on: with no status file in the demo dir, `trace_policy`
    reported `known: False` and `classify_trace_verdict` returned a hard FAIL -- correctly, on its
    own terms, since nothing licenses skipping a trace on no evidence. But the evidence existed; it
    was one directory away, and the agent was told to make the status readable from a dir it was
    never written to. 25 graduated modules read as 0, every round.
    """
    try:
        from .commands.emit_e2e import _import_closure
    except Exception:  # noqa: BLE001 - no walker available: no components to report
        return []
    out = []
    seen = set()
    for f in _import_closure(Path(demo_dir)):
        for d in f.parents:
            if d in seen:
                continue
            seen.add(d)
            if (d / _STATUS_FILE).is_file():
                out.append(d)
    return sorted(out)


def _graduation_in(status_dir, qualify=False):
    """The graduation state recorded in ONE bring-up dir: {module: "sharded"|"native"|None}."""
    status_dir = Path(status_dir)
    result = {}
    try:
        data = json.loads((status_dir / _STATUS_FILE).read_text())
    except Exception:
        return result
    try:
        from .bringup_loop import _safe_id, _stub_has_graduated_any
    except Exception:
        return result
    for comp in data.get("components", []):
        name = comp.get("name")
        if not name:
            continue
        stub = status_dir / "_stubs" / f"{_safe_id(name)}.py"
        native = stub.with_suffix(".py.last_good_native").is_file()
        sharded = stub.with_suffix(".py.last_good_sharded").is_file()
        try:
            graduated = bool(_stub_has_graduated_any(stub))
        except Exception:
            graduated = False
        # Qualified only when several dirs are being merged, where the same module name can occur in
        # more than one component and an unqualified key would silently drop one. A single-dir model
        # keeps the bare names it has always reported.
        key = "%s/%s" % (status_dir.name, name) if qualify else name
        if graduated and sharded:
            result[key] = "sharded"
        elif graduated and native:
            result[key] = "native"
        else:
            result[key] = None
    return result


def read_graduation(demo_dir):
    demo_dir = Path(demo_dir)
    if (demo_dir / _STATUS_FILE).is_file():
        return _graduation_in(demo_dir)
    dirs = _component_status_dirs(demo_dir)
    result = {}
    for d in dirs:
        result.update(_graduation_in(d, qualify=len(dirs) > 1))
    return result


def trace_policy(graduation):
    graduated = {n for n, k in graduation.items() if k}
    ungraduated = {n for n, k in graduation.items() if not k}
    all_graduated = bool(graduation) and not ungraduated
    return {
        "required": all_graduated,
        "all_graduated": all_graduated,
        "graduated_modules": graduated,
        "eager_eligible_modules": ungraduated,
        # Whether the graduation state is KNOWN AT ALL. An empty mapping is not the same
        # fact as "every module is ungraduated": `read_graduation` also returns {} when the
        # demo dir carries no status file (a composite's e2e dir keeps its status per
        # component, not at the top), when the JSON will not parse, and when the import it
        # needs is unavailable. Collapsing those into the eager-eligible branch let the gate
        # waive the trace requirement on NO evidence.
        "known": bool(graduation),
    }


def trace_engaged(trace_caps):
    if not isinstance(trace_caps, dict):
        return False
    return bool(trace_caps.get("trace_1cq"))


def valid_overflow_proof(proof):
    """A waiver needs a MEASURED budget, not a placeholder.

    This accepted any proof where required > budget, and overflow_fix_loop filled budget_bytes with
    0 when it gave up -- so anything exceeded it and every unfixed memory failure minted a valid
    proof. Measured on a Qwen-Image-Edit run: a VAE decode that could not allocate a 100663296 B
    DRAM buffer produced "trace waived: verified physical overflow required=191102976 > budget=0",
    G6 went green, and the gate reported the pipeline trace-ready with no trace ever captured.
    A budget of zero is not a statement about the device; it is the absence of one."""
    if not isinstance(proof, dict):
        return False
    required = proof.get("required_bytes")
    budget = proof.get("budget_bytes")
    if not isinstance(required, (int, float)) or not isinstance(budget, (int, float)):
        return False
    if budget <= 0:
        return False
    return required > budget


def classify_trace_verdict(trace_caps, policy, allow_no_trace=False, overflow_proof=None):
    if trace_engaged(trace_caps):
        return "PASS", "trace engaged"
    if policy.get("required"):
        if allow_no_trace and valid_overflow_proof(overflow_proof):
            return "EAGER_WAIVED", (
                "trace waived: verified physical overflow required=%s > budget=%s"
                % (overflow_proof.get("required_bytes"), overflow_proof.get("budget_bytes"))
            )
        return "FAIL", (
            "trace did not engage but ALL modules graduated on-device -> eager not permitted; "
            "fix pipeline/glue to trace (or supply a verified overflow proof)"
        )
    if not policy.get("known"):
        # No graduation state was readable, so nothing here licenses skipping the trace. The
        # waiver below is an argument from evidence -- "these named modules are still eager"
        # -- and with no modules to name it degenerated into waiving by default, printing a
        # bare "?" where the justification should be. Absence of evidence is not a proof.
        return "FAIL", (
            "trace did not engage and the graduation state could not be read, so eager "
            "execution cannot be justified; make the bring-up status readable from this demo "
            "dir, fix the pipeline/glue to trace, or supply a verified overflow proof"
        )
    return "EAGER_WAIVED", (
        "trace not engaged; eager permitted because ungraduated module(s) present: "
        + ", ".join(sorted(policy.get("eager_eligible_modules")))
    )


_TRACED_STEP_NAMES = re.compile(
    r"^(decode_step|prefill_step|forward_step|_forward_from_hidden|run_forward)$|_trace_step$|(?<!_write)_step$"
)

_TORCH_HOST_FNS = {"full", "zeros", "arange", "tensor", "argmax", "multinomial", "topk", "cat", "stack", "eye"}
_TTNN_HOST_FNS = {"from_torch", "to_torch", "from_device", "to_device"}
_TTNN_ALLOC_FNS = {"allocate", "zeros", "arange", "empty"}
_TTNN_CHURN_FNS = {"tilize", "untilize", "to_layout"}
_METHOD_HOST = {"item", "cpu", "tolist", "numpy"}


def _is_traced_step(name):
    return bool(_TRACED_STEP_NAMES.search(name))


def _base_name(attr_node):
    base = attr_node.value
    return base.id if isinstance(base, ast.Name) else None


def _forbidden_calls(fn_node):
    hits = []
    for n in ast.walk(fn_node):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute):
            continue
        attr = n.func.attr
        base = _base_name(n.func)
        if base == "torch" and attr in _TORCH_HOST_FNS:
            hits.append(("host-op", "torch." + attr))
        elif base == "ttnn" and attr in _TTNN_HOST_FNS:
            hits.append(("host-op", "ttnn." + attr))
        elif base == "ttnn" and attr in _TTNN_CHURN_FNS:
            hits.append(("layout-churn", "ttnn." + attr))
        elif base == "ttnn" and attr in _TTNN_ALLOC_FNS:
            hits.append(("per-call-alloc", "ttnn." + attr))
        elif attr in _METHOD_HOST:
            hits.append(("host-op", "." + attr + "()"))
    return hits


def glue_trace_violations(pipeline_src):
    violations = []
    try:
        tree = ast.parse(pipeline_src)
    except Exception:
        return violations
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_traced_step(node.name):
            continue
        for kind, tok in _forbidden_calls(node):
            violations.append("glue %s in traced step `%s`: %s" % (kind, node.name, tok))
    return violations


def decode_repin_violation(pipeline_src):
    try:
        tree = ast.parse(pipeline_src)
    except Exception:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "decode_step":
            continue
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Attribute):
                if f.attr == "_pin_hidden":
                    return _repin_reason(node.name)
                if _base_name(f) == "torch" and f.attr == "full":
                    return _repin_reason(node.name)
                if _base_name(f) == "ttnn" and f.attr == "from_torch":
                    return _repin_reason(node.name)
    return None


def _repin_reason(fn_name):
    return (
        "decode per-token host re-pin in `%s` (torch.full/from_torch) -> O(capacity) recompute; "
        "no KV-cache single-token decode step" % fn_name
    )


def _caps_path(demo_dir):
    demo_dir = Path(demo_dir)
    e2e = demo_dir / "tests" / "e2e"
    if not e2e.is_dir():
        return None
    caps = sorted(e2e.glob("*perf*.trace_caps.json")) or sorted(e2e.glob("*.trace_caps.json"))
    return caps[-1] if caps else None


def read_trace_caps(demo_dir):
    p = _caps_path(demo_dir)
    if p is None:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _perf_test(demo_dir):
    e2e = Path(demo_dir) / "tests" / "e2e"
    perf = sorted(e2e.glob("test_*_perf.py")) if e2e.is_dir() else []
    return perf[-1] if perf else None


def _task_of(perf_path):
    m = re.match(r"test_(.+)_perf\.py$", Path(perf_path).name)
    return m.group(1) if m else "main"


def caps_stale(demo_dir):
    demo_dir = Path(demo_dir)
    caps = _caps_path(demo_dir)
    if caps is None or not caps.is_file():
        return True
    pipeline = demo_dir / "tt" / "pipeline.py"
    if pipeline.is_file() and pipeline.stat().st_mtime > caps.stat().st_mtime:
        return True
    return False


# RETRY A WEDGE ON THE BOARD THE WEDGE JUST RESET.
#
# A trace hang already triggers a reset (perf_test_gen._run_perf_node catches TracyHangError, runs
# tt-smi -r, and returns the WEDGE). What it does NOT do on this path is try again: the retry loop
# lives in `generate_perf_test`, which bounds itself at _TRACE_WEDGE_LIMIT, and the gate reaches the
# capture through `validate_generated_perf_test` instead -- which has none. So the board is reset and
# the result is then discarded.
#
# That costs the whole round. Reaching the capture at all means the correctness gate has just run:
# ~3.5 h of device work on one Qwen-Image-Edit bring-up. Against that, one more ~10-minute capture on
# a board that was JUST reset is nearly free, and it is the only way to find out whether the reset
# actually restored anything. Six rounds over two days each paid the 3.5 h, wedged once, and stopped.
#
# `_is_device_disruption` states the case against retrying -- "that already got one reset and must
# return to the caller as a WEDGE, NOT loop reset+retry (which just re-hangs)". That is why this is
# BOUNDED AND SMALL (one extra attempt by default) rather than a loop: if it just re-hangs, the cost
# is one capture and the report now says so with BOTH attempts' evidence; if it does not, the round
# is saved. Every attempt's detail is kept, so the progression is visible instead of only the last.
# HOW MANY, from what an attempt costs against what it tells you. A capture runs ~5-15 min (the
# longest observed was 13); the round it is trying to save is ~3.5 h. The value decays fast, though:
# attempt 1 establishes the failure, attempt 2 answers the open question (did the reset restore the
# board?), attempt 3 separates flaky from deterministic -- and past that each 15 minutes buys
# information already in hand. generate_perf_test's _TRACE_WEDGE_LIMIT of 10 would be up to 2.5 h,
# as expensive as the round, so it is not reused as a count here.
_WEDGE_RETRY_ENV = "E2E_TRACE_WEDGE_RETRIES"
_WEDGE_RETRIES = 2

# WHAT THE CAPTURE SAW, IN THE FIELD THE AGENT ACTUALLY READS.
#
# `capture_detail` is the only model-specific evidence this gate produces: the stage markers the
# capture printed before it stopped, one line per attempt. It was returned in the result and written
# to the report -- and left OUT of `reasons`, which is the list the gate server turns into
# next_target.reason and blocking[]. So the agent was handed the verdict PROSE only ("trace did not
# engage ..."), identical every round, while the line naming where it stopped sat in a file nothing
# told it to read: five rounds of guessing, no edits. The report keeps its own copy; this puts the
# same fact into the pipe that reaches the agent.
_CAPTURE_DETAIL_CHARS = 900  # next_target.reason is capped at 2000 -- leave room for the other blockers


# WHAT A CAPTURE MAY COST, FROM WHAT ONE HAS COST.
#
# This was a literal 900 s in the signature below, and probes._execute ends a step absolutely at
# _HARD_CEILING_MULT x its budget -- so 900 became a 3600 s wall. A Qwen-Image-Edit capture needed
# longer and was SIGKILLed there twice, 3611 s apart, by the gate's own process, both times with one
# stage traced and the next mid-replay. The ceiling is right to be absolute (it is the only guard
# against work that progresses forever); what was wrong is that it multiplied a number typed for a
# much smaller step.
#
# So the budget is sized by probes.sized_budget -- operator's value, else headroom over the longest
# capture actually observed, else the old 900 as a FLOOR, so nothing gets tighter than before. The
# observation has to OUTLIVE the process: the gate runs in a fresh interpreter every round, so an
# in-memory record would never be read back. It is kept beside the demo's other gate state and
# scoped to the run that measured it, because a cost measured on another board says nothing here.
_CAPTURE_COST_FILE = ".e2e_capture_cost.json"
_CAPTURE_BUDGET_ENV = "E2E_TRACE_CAPTURE_TIMEOUT"
_CAPTURE_FLOOR_S = 900  # what the typed default used to be, kept as the floor


def _run_stamp_of_this_run() -> str:
    try:
        from .commands.emit_e2e import run_stamp

        return run_stamp()
    except Exception:  # noqa: BLE001 -- no run identity is a valid answer
        return ""


def _observed_capture_s(demo_dir) -> float:
    """The longest capture measured for this demo IN THIS RUN, or 0 when there is no such record."""
    try:
        doc = json.loads((Path(demo_dir) / _CAPTURE_COST_FILE).read_text())
    except Exception:  # noqa: BLE001
        return 0.0
    if not isinstance(doc, dict) or doc.get("run") != _run_stamp_of_this_run():
        return 0.0
    try:
        return max(0.0, float(doc.get("seconds") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _record_capture_s(demo_dir, seconds: float) -> None:
    """Keep the LONGEST observed; a budget that shrank on a lucky attempt would kill the next one."""
    try:
        if not seconds or seconds <= 0 or seconds <= _observed_capture_s(demo_dir):
            return
        (Path(demo_dir) / _CAPTURE_COST_FILE).write_text(
            json.dumps({"run": _run_stamp_of_this_run(), "seconds": round(float(seconds), 1)})
        )
    except OSError:
        pass


def _capture_budget_s(demo_dir) -> int:
    from models.experimental.perf_automation.agent import probes as _pr_budget

    override = os.environ.get(_CAPTURE_BUDGET_ENV)
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return _pr_budget.sized_budget(_observed_capture_s(demo_dir), _CAPTURE_FLOOR_S)


def _wedge_retries() -> int:
    try:
        return max(0, int(os.environ.get(_WEDGE_RETRY_ENV, str(_WEDGE_RETRIES))))
    except ValueError:
        return _WEDGE_RETRIES


def _is_wedge(status, detail) -> bool:
    """A hang, as opposed to an ordinary invalid/skip verdict the agent should fix in code."""
    return status == "invalid" and "WEDGE" in (detail or "")


def run_fresh_trace_capture(demo_dir, timeout_s=None):
    demo_dir = Path(demo_dir)
    # None -> sized from what a capture has actually cost here (see _capture_budget_s). An explicit
    # value from a caller still wins, so existing callers are unaffected.
    timeout_s = int(timeout_s) if timeout_s else _capture_budget_s(demo_dir)
    perf = _perf_test(demo_dir)
    if perf is None:
        return None, "no perf test to capture"
    task = _task_of(perf)
    try:
        from models.experimental.perf_automation.agent.perf_test_gen import validate_generated_perf_test
    except Exception as e:  # noqa: BLE001
        return read_trace_caps(demo_dir), "perf_test_gen unavailable: %s" % e
    os.environ["TT_PERF_TRACE"] = "1"
    os.environ.setdefault("PERF_MCP_VALIDATE_TIMEOUT", str(timeout_s))
    attempts = []
    for attempt in range(1 + _wedge_retries()):
        _t0 = time.monotonic()
        try:
            status, detail = validate_generated_perf_test(perf, task)
        except Exception as e:  # noqa: BLE001
            attempts.append("capture raised: %s" % e)
            break
        finally:
            # Recorded even for a FAILED attempt: how long it ran is a fact about this model's cost,
            # and a capture killed at the wall is the strongest evidence the wall was too low.
            _record_capture_s(demo_dir, time.monotonic() - _t0)
        attempts.append("%s %s" % (status, detail or ""))
        if not _is_wedge(status, detail):
            break
        if attempt < _wedge_retries():  # another attempt follows
            print(
                "  [trace] capture wedged and the board was reset; retrying the capture on it "
                "(%d/%d)" % (attempt + 1, _wedge_retries()),
                flush=True,
            )
    # Every attempt is reported, not just the last: "wedged in the same place twice" and "got further
    # the second time" need opposite fixes, and only the sequence tells them apart.
    if len(attempts) > 1:
        joined = "\n".join("attempt %d: %s" % (i + 1, a) for i, a in enumerate(attempts))
    else:
        joined = attempts[0] if attempts else "no capture attempted"
    return read_trace_caps(demo_dir), joined


def evaluate_trace_gate(demo_dir, trace_caps=None, allow_no_trace=False, overflow_proof=None, fresh=False):
    demo_dir = Path(demo_dir)
    capture_detail = None
    if trace_caps is None:
        if fresh or caps_stale(demo_dir):
            trace_caps, capture_detail = run_fresh_trace_capture(demo_dir)
        if trace_caps is None:
            trace_caps = read_trace_caps(demo_dir)
    graduation = read_graduation(demo_dir)
    policy = trace_policy(graduation)
    verdict, reason = classify_trace_verdict(
        trace_caps, policy, allow_no_trace=allow_no_trace, overflow_proof=overflow_proof
    )
    reasons = []
    pipeline = demo_dir / "tt" / "pipeline.py"
    glue = []
    repin = None
    if pipeline.is_file():
        src = pipeline.read_text(errors="ignore")
        glue = glue_trace_violations(src)
        repin = decode_repin_violation(src)
    runtime_glue = glue_from_runtime(demo_dir)
    l1_overflow = is_l1_overflow(capture_detail)
    if l1_overflow:
        reclaim_mesh()
    if verdict == "FAIL":
        reasons.append("G6 trace-gate: " + reason)
        if capture_detail:
            reasons.append("G6 trace-gate: what the capture itself reported: " + capture_detail[:_CAPTURE_DETAIL_CHARS])
        for g in glue:
            reasons.append("G6 trace-gate: " + g)
        if repin:
            reasons.append("G6 trace-gate: " + repin)
        if runtime_glue:
            reasons.append(
                "G6 trace-gate: %d runtime glue op(s) outside graduated modules: %s"
                % (len(runtime_glue), ", ".join(runtime_glue[:6]))
            )
        if l1_overflow:
            reasons.append("G6 trace-gate: " + l1_overflow_reason())
    return {
        "verdict": verdict,
        "reason": reason,
        "policy": policy,
        "graduation": graduation,
        "glue_violations": glue,
        "repin_violation": repin,
        "reasons": reasons,
        "trace_caps": trace_caps,
        "capture_detail": capture_detail,
        "runtime_glue": runtime_glue,
        "l1_overflow": l1_overflow,
    }


def glue_ops_from_stream(op_stream, module_signatures):
    owned = set()
    for sig in (module_signatures or {}).values():
        owned |= set(sig)
    return [op for op in (op_stream or []) if op not in owned]


def _read_op_stream(demo_dir):
    demo_dir = Path(demo_dir)
    root = demo_dir
    for parent in demo_dir.parents:
        if (parent / "models").is_dir():
            root = parent
            break
    runs = root / "models" / "experimental" / "perf_automation" / "runs"
    if not runs.is_dir():
        return None
    csvs = sorted(runs.rglob("*baseline*report*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return None
    ops = []
    try:
        import csv as _csv

        with open(csvs[0], newline="") as fh:
            for row in _csv.DictReader(fh):
                name = row.get("OP CODE") or row.get("op_code") or row.get("OP TYPE") or row.get("op")
                if name:
                    ops.append(name.strip())
    except Exception:
        return None
    return ops or None


def _read_module_signatures(demo_dir):
    demo_dir = Path(demo_dir)
    grad = read_graduation(demo_dir)
    try:
        from .bringup_loop import _safe_id
    except Exception:
        return {}
    sigs = {}
    for name in grad:
        probe = demo_dir / "_stubs" / (_safe_id(name) + ".py.native_probe.json")
        if not probe.is_file():
            continue
        try:
            data = json.loads(probe.read_text())
        except Exception:
            continue
        ops = data.get("ttnn_ops") or data.get("ops") or []
        if ops:
            sigs[name] = set(o.strip() for o in ops if isinstance(o, str))
    return sigs


def glue_from_runtime(demo_dir):
    op_stream = _read_op_stream(demo_dir)
    sigs = _read_module_signatures(demo_dir)
    if not op_stream or not sigs:
        return None
    return glue_ops_from_stream(op_stream, sigs)


_OVERFLOW_MARKERS = ("trace region", "trace_region", "overflow", "out of memory", "oom", "not enough space")
# THE REMEDY ONLY FITS ONE OF THESE. overflow_fix_loop's fix is to GROW the trace region, which is
# right when the trace region is what overflowed and actively harmful otherwise: a bigger trace
# region leaves LESS device memory, so on a plain buffer allocation failure the loop made the fault
# worse on every one of its three doublings and then waived the gate. Matching the allocator's own
# vocabulary, not the model's -- the same shape as _L1_MARKERS_* below.
_REGION_MARKERS = ("trace region", "trace_region")
_DEFAULT_TRACE_REGION = 23887872

_L1_MARKERS_A = ("circular buffer", "max l1", "l1 size")
_L1_MARKERS_B = ("beyond max l1", "grow to", "l1 size of")


def _is_overflow(detail):
    d = (detail or "").lower()
    return any(m in d for m in _OVERFLOW_MARKERS)


def _is_region_overflow(detail):
    """The trace REGION overflowed -- the one failure growing the region can fix."""
    d = (detail or "").lower()
    return any(m in d for m in _REGION_MARKERS)


def is_l1_overflow(detail):
    d = (detail or "").lower()
    return any(a in d for a in _L1_MARKERS_A) and any(b in d for b in _L1_MARKERS_B)


def reclaim_mesh():
    try:
        import subprocess

        # Same precondition device_recovery applies everywhere else: a chip reporting a plausible
        # die temperature has a running ARC, and resetting it is pure risk. This path reset with no
        # device list at all -- every board on the host -- and never asked.
        try:
            from models.experimental.perf_automation.agent.device_recovery import board_needs_reset

            if not board_needs_reset():
                return True
        except Exception:  # noqa: BLE001 -- cannot tell is not a reason to stop resetting
            pass
        subprocess.run(["tt-smi", "-r"], capture_output=True, text=True, timeout=420)
        return True
    except Exception:
        return False


def l1_overflow_reason():
    return (
        "L1_OVERFLOW: trace capture's circular buffers exceed the per-core L1 budget and crashed the run; "
        "the mesh was reset. Reduce the L1 footprint (smaller in0_block_w / per_core_N, or spread the op "
        "over more cores) and retry -- do NOT keep this config."
    )


def overflow_fix_loop(demo_dir, capture_fn=None, max_rounds=3, base_region=_DEFAULT_TRACE_REGION):
    capture_fn = capture_fn or run_fresh_trace_capture
    region = base_region
    caps, detail = None, None
    for _ in range(max_rounds):
        os.environ["TT_PERF_TRACE_REGION"] = str(region)
        caps, detail = capture_fn(demo_dir)
        if caps and caps.get("trace_1cq"):
            return {"resolved": True, "caps": caps, "detail": "traced at region=%d" % region, "proof": None}
        if not _is_region_overflow(detail):
            # An allocation failure that is NOT the trace region: growing the region cannot help and
            # would take memory away from the thing that just ran out of it. Report it as it is.
            return {"resolved": False, "caps": caps, "detail": detail, "proof": None}
        region *= 2
    # GIVING UP IS NOT A PROOF. Three doublings that did not help says this tool could not fix it;
    # it says nothing about what the device can physically hold, and a waiver needs the latter.
    return {
        "resolved": False,
        "caps": caps,
        "detail": "trace region overflow persists after %d rounds (region grown to %d); no physical "
        "budget was measured, so this is not a waiver -- the trace requirement stands" % (max_rounds, region),
        "proof": None,
    }


def build_fix_directive(result):
    if not result or result.get("verdict") != "FAIL":
        return None
    parts = []
    if result.get("l1_overflow"):
        parts.append(
            "Reduce the L1 footprint (smaller in0_block_w / per_core_N, or spread the op over more cores) "
            "so the circular buffers fit per-core L1 with trace headroom."
        )
    if result.get("repin_violation"):
        parts.append(
            "Add a KV-cache single-token decode_step and remove the O(capacity) host re-pin "
            "(torch.full/ttnn.from_torch) in decode_step."
        )
    for g in result.get("glue_violations") or []:
        parts.append("Port to on-device ttnn (remove from traced step): " + g)
    if not parts:
        # Nothing static to point at, so the only lead is what the capture reported. Without it this
        # fell back to the verdict prose, which is the same sentence every round and names nothing.
        parts.append(result.get("reason", "trace did not engage"))
        detail = result.get("capture_detail")
        if detail:
            parts.append("The capture's own last output was: " + detail[:_CAPTURE_DETAIL_CHARS])
    return " ".join(parts)


def record_trace_verdict(demo_dir, result):
    try:
        from .run_report import upsert_report_section
    except Exception:
        return None
    pol = result.get("policy") or {}
    lines = [
        "# Trace gate",
        "",
        "verdict: **%s**" % result.get("verdict"),
        "",
        result.get("reason", ""),
        "",
        "graduated on-device: %d, ungraduated: %d"
        % (len(pol.get("graduated_modules") or []), len(pol.get("eager_eligible_modules") or [])),
    ]
    caps = result.get("capture_detail")
    if caps:
        lines += ["", "fresh capture: %s" % caps]
    if result.get("reasons"):
        lines += ["", "blockers:"]
        lines += ["- " + r for r in result["reasons"]]
    directive = build_fix_directive(result)
    if directive:
        lines += ["", "fix directive:", directive]
    return upsert_report_section(demo_dir, "trace-gate", "\n".join(lines))
