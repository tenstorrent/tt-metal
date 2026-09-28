# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Human approvals, recorded with the hash of exactly what was approved.

``<bringup_dir>/approvals.yaml``: {point: {by, at, files: {path: sha256}, ledger, note}}. Approval points: intake
(spec.yaml), plan (plan.yaml, plan.md, components.yaml, and the ledger's structure), perf (opportunities.md with the
picked items). Any edit to an approved file after approval makes ``is_approved`` false again, so the gate that needs
the approval fails. The ledger is approved by structure (ids, titles, deps, gate commands and thresholds), because
freezing and runs legitimately add ``frozen`` blocks to tasks.yaml later. Findings go to findings.yaml, not here.

Op requests (F46): ``approvals.yaml`` ``op_request: {<op>: {by, at, files, note}}``, the hash of every file of
``op_requests/<op>/`` with request.yaml hashed without its lifecycle fields (status, exported, delivered), so export and
delivery keep the approval while any edit of the request voids it. op-export needs it.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import time

import yaml

from models.demos.common.bringup.core.freeze import hash_paths
from models.demos.common.bringup.core.gate import format_paths
from models.demos.common.bringup.core.ledger import Ledger

POINTS = {
    "intake": ["spec.yaml"],
    "plan": ["plan.yaml", "plan.md", "components.yaml"],
    "perf": ["opportunities.md"],
}


def _path(spec):
    return spec.bringup_dir / "approvals.yaml"


def _files(spec, point: str) -> list[str]:
    rel = spec.bringup_dir.relative_to(spec.repo)
    return [str(rel / f) for f in POINTS[point] if (spec.bringup_dir / f).exists()]


def ledger_signature(spec) -> str:
    # Picked performance items (step: perf) are approved at the perf point, so adding one keeps the plan approval.
    tasks = [t for t in Ledger(spec.bringup_dir).tasks().values() if t.get("step") != "perf"]
    keep = [{k: t.get(k) for k in ("id", "title", "deps", "gate", "paths", "tests", "device")} for t in tasks]
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def load(spec) -> dict:
    p = _path(spec)
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def approve(spec, point: str, note: str = "", by: str | None = None) -> dict:
    files = _files(spec, point)
    if not files:
        raise FileNotFoundError(f"nothing to approve for {point}: none of {POINTS[point]} in {spec.bringup_dir}")
    format_paths(spec.repo, files)  # the commit hooks must not change an approved file afterwards
    data = load(spec)
    data[point] = {
        "by": by or getpass.getuser(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": hash_paths(spec.repo, files),
        "note": note,
    }
    if point == "plan":
        data[point]["ledger"] = ledger_signature(spec)
    _path(spec).write_text(yaml.safe_dump(data, sort_keys=False))
    return data[point]


def shared_paths(spec, task: dict) -> list[str]:
    """The task's allowed paths outside the model's own directories (model_dir, plus spec ``paths.own``): shared code
    that other models use. An agent may change them only after the owner approves that task (``shared:<task id>``)."""
    own = [str(spec.model_dir.relative_to(spec.repo))] + list(spec.get("paths.own") or [])
    inside = lambda p: any(p == o or p.startswith(o.rstrip("/") + "/") for o in own)  # noqa: E731
    return sorted(p for p in task.get("paths") or [] if not inside(p))


def approve_shared(spec, task: dict, note: str = "", by: str | None = None) -> dict:
    paths = shared_paths(spec, task)
    if not paths:
        raise ValueError(f"{task['id']} changes no shared code; nothing to approve")
    data = load(spec)
    data.setdefault("shared", {})[task["id"]] = {
        "by": by or getpass.getuser(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paths": paths,
        "note": note,
    }
    _path(spec).write_text(yaml.safe_dump(data, sort_keys=False))
    return data["shared"][task["id"]]


def shared_approved(spec, task: dict) -> bool:
    rec = (load(spec).get("shared") or {}).get(task["id"])
    return bool(rec) and rec.get("paths") == shared_paths(spec, task)


def is_approved(spec, point: str) -> bool:
    rec = load(spec).get(point)
    if not rec:
        return False
    files = _files(spec, point)
    if not files or set(files) != set(rec["files"]) or hash_paths(spec.repo, files) != rec["files"]:
        return False
    return point != "plan" or rec.get("ledger") == ledger_signature(spec)


# ---------------------------------------------------------------- op requests (F46)
LIFECYCLE = ("status", "exported", "delivered")


def op_request_hashes(spec, op: str) -> dict[str, str]:
    from models.demos.common.bringup.plan import op_request as OR

    d = OR.request_dir(spec, op)
    if not (d / "request.yaml").exists():
        raise FileNotFoundError(f"no op request {op} in {OR.root(spec)}")
    rel = lambda f: str(f.relative_to(spec.repo))  # noqa: E731
    files = [f for f in sorted(d.rglob("*")) if f.is_file() and "__pycache__" not in f.parts]
    out = {rel(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.name != "request.yaml"}
    req = {k: v for k, v in OR.load(d).items() if k not in LIFECYCLE}
    out[rel(d / "request.yaml")] = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
    return out


def approve_op_request(spec, op: str, note: str = "", by: str | None = None) -> dict:
    """The owner's approval of an op request: what op-export will hand to op-gen. Marks the request approved."""
    from models.demos.common.bringup.plan import op_request as OR

    d = OR.request_dir(spec, op)
    format_paths(spec.repo, [str(d.relative_to(spec.repo))])
    if OR.load(d).get("status") == "draft":
        OR.set_status(d, "approved")
    data = load(spec)
    rec = {
        "by": by or getpass.getuser(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": op_request_hashes(spec, op),
        "note": note,
    }
    data.setdefault("op_request", {})[op] = rec
    _path(spec).write_text(yaml.safe_dump(data, sort_keys=False))
    return rec


def op_request_approved(spec, op: str) -> bool:
    rec = (load(spec).get("op_request") or {}).get(op)
    try:
        return bool(rec) and rec.get("files") == op_request_hashes(spec, op)
    except FileNotFoundError:
        return False
