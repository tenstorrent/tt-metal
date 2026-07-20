import ast
import json
import os
import re
from pathlib import Path


def read_graduation(demo_dir):
    demo_dir = Path(demo_dir)
    status_path = demo_dir / "bringup_status.json"
    result = {}
    if not status_path.is_file():
        return result
    try:
        data = json.loads(status_path.read_text())
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
        stub = demo_dir / "_stubs" / f"{_safe_id(name)}.py"
        native = stub.with_suffix(".py.last_good_native").is_file()
        sharded = stub.with_suffix(".py.last_good_sharded").is_file()
        try:
            graduated = bool(_stub_has_graduated_any(stub))
        except Exception:
            graduated = False
        if graduated and sharded:
            result[name] = "sharded"
        elif graduated and native:
            result[name] = "native"
        else:
            result[name] = None
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
    }


def trace_engaged(trace_caps):
    if not isinstance(trace_caps, dict):
        return False
    return bool(trace_caps.get("trace_2cq") or trace_caps.get("trace_1cq"))


def valid_overflow_proof(proof):
    if not isinstance(proof, dict):
        return False
    required = proof.get("required_bytes")
    budget = proof.get("budget_bytes")
    if not isinstance(required, (int, float)) or not isinstance(budget, (int, float)):
        return False
    return required > budget


def classify_trace_verdict(trace_caps, policy, allow_no_trace=False, overflow_proof=None):
    if trace_engaged(trace_caps):
        return "PASS", "trace+2CQ engaged"
    if policy.get("required"):
        if allow_no_trace and valid_overflow_proof(overflow_proof):
            return "EAGER_WAIVED", (
                "trace waived: verified physical overflow required=%s > budget=%s"
                % (overflow_proof.get("required_bytes"), overflow_proof.get("budget_bytes"))
            )
        return "FAIL", (
            "trace+2CQ did not engage but ALL modules graduated on-device -> eager not permitted; "
            "fix pipeline/glue to trace (or supply a verified overflow proof)"
        )
    return "EAGER_WAIVED", (
        "trace not engaged; eager permitted because ungraduated module(s) present: "
        + ", ".join(sorted(policy.get("eager_eligible_modules") or {"?"}))
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
        if node.name not in ("decode_write_inputs", "decode_step"):
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


def run_fresh_trace_capture(demo_dir, timeout_s=900):
    demo_dir = Path(demo_dir)
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
    try:
        status, detail = validate_generated_perf_test(perf, task)
    except Exception as e:  # noqa: BLE001
        return read_trace_caps(demo_dir), "capture raised: %s" % e
    return read_trace_caps(demo_dir), "%s %s" % (status, detail or "")


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
    if verdict == "FAIL":
        reasons.append("G6 trace-gate: " + reason)
        for g in glue:
            reasons.append("G6 trace-gate: " + g)
        if repin:
            reasons.append("G6 trace-gate: " + repin)
        if runtime_glue:
            reasons.append(
                "G6 trace-gate: %d runtime glue op(s) outside graduated modules: %s"
                % (len(runtime_glue), ", ".join(runtime_glue[:6]))
            )
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
_DEFAULT_TRACE_REGION = 23887872


def _is_overflow(detail):
    d = (detail or "").lower()
    return any(m in d for m in _OVERFLOW_MARKERS)


def overflow_fix_loop(demo_dir, capture_fn=None, max_rounds=3, base_region=_DEFAULT_TRACE_REGION):
    capture_fn = capture_fn or run_fresh_trace_capture
    region = base_region
    caps, detail = None, None
    for _ in range(max_rounds):
        os.environ["TT_PERF_TRACE_REGION"] = str(region)
        caps, detail = capture_fn(demo_dir)
        if caps and (caps.get("trace_2cq") or caps.get("trace_1cq")):
            return {"resolved": True, "caps": caps, "detail": "traced at region=%d" % region, "proof": None}
        if not _is_overflow(detail):
            return {"resolved": False, "caps": caps, "detail": detail, "proof": None}
        region *= 2
    return {
        "resolved": False,
        "caps": caps,
        "detail": "overflow persists after %d rounds (region grown to %d)" % (max_rounds, region),
        "proof": {"required_bytes": region, "budget_bytes": 0, "rounds": max_rounds},
    }


def build_fix_directive(result):
    if not result or result.get("verdict") != "FAIL":
        return None
    parts = []
    if result.get("repin_violation"):
        parts.append(
            "Add a KV-cache single-token decode_step and remove the O(capacity) host re-pin "
            "(torch.full/ttnn.from_torch) in decode_write_inputs."
        )
    for g in result.get("glue_violations") or []:
        parts.append("Port to on-device ttnn (remove from traced step): " + g)
    if not parts:
        parts.append(result.get("reason", "trace+2CQ did not engage"))
    return " ".join(parts)


def record_trace_verdict(demo_dir, result):
    try:
        from .run_report import upsert_report_section
    except Exception:
        return None
    pol = result.get("policy") or {}
    lines = [
        "# Trace+2CQ gate",
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
