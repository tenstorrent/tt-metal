"""Pure binding/report seams around the unchanged runtime owner and existing supervisor."""
import json
from pathlib import Path

from lifecycle import clean_lifecycle
from support import require, sha256


def validate(plan):
    require(plan.get("reviewed") is True and plan.get("dispatch_enabled") is True, "CLOSED runtime launch")
    import re

    match = re.fullmatch(r"bh-glx-(\d+)-(\w+)", str(plan.get("node", "")))
    require(match is not None and re.fullmatch(r"[0-9]+", str(plan.get("job_id", ""))), "wrong assigned endpoint")
    lock = Path(str(plan.get("lock_path", "")))
    require(
        lock.is_absolute() and lock.name == "prefill-device-" + match[1] + "-" + match[2] + ".lock",
        "wrong physical node lock",
    )
    require(
        (plan.get("payload_budget_seconds"), plan.get("outer_budget_seconds"), plan.get("recovery_reserve_seconds"))
        == (1800, 2100, 600),
        "runtime budgets changed",
    )
    require(plan.get("scope") == "2k_runtime_edges_without_manager", "wrong runtime scope")
    require(
        plan.get("launch_receipt") == plan.get("launch_receipt_sha256") == "",
        "root plan must leave node launch receipt unbound",
    )


def owner_state(report_path, log_path, plan, owner_plan_hash):
    path = Path(report_path)
    if not path.exists():
        return {}
    report = json.loads(path.read_bytes())
    require(isinstance(report, dict), "malformed owner report")
    require(
        report.get("run_nonce") == plan["run_nonce"] and report.get("scope") == plan["scope"], "foreign runtime report"
    )
    if report.get("baseline"):
        require(report["baseline"]["identity"]["plan_sha256"] == owner_plan_hash, "owner plan identity changed")
    if report.get("cleanup_errors") or report.get("recovery_required"):
        return {"recovery": report}
    if report.get("owner_cleanup_complete") is True:
        close = clean_lifecycle(Path(log_path).read_text(errors="replace"))
        if close:
            return {"result": dict(report, verified_clean_close=close)}
    # An exited child without both the owner cleanup receipt and final32 close will hold.
    return {"failure": report}


def bind_owner_plan(plan, receipt, receipt_path):
    require(
        receipt["node"] == plan["node"]
        and receipt["job_id"] == plan["job_id"]
        and receipt["run_nonce"] == plan["run_nonce"],
        "launch receipt identity changed",
    )
    bound = dict(plan, launch_receipt=str(receipt_path), launch_receipt_sha256=sha256(receipt_path))
    return bound
