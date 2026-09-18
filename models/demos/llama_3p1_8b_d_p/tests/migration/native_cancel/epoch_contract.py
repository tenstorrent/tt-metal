"""Import-free epoch bindings; a prior clean stop never authorizes a future manager."""

import copy
import hashlib
from pathlib import Path

from runner_support import require


def epoch_nonce(nonce, epoch):
    require(epoch in ("a", "b"), "Unknown cancellation epoch")
    return hashlib.sha256((nonce + ":cancel-restart:" + epoch).encode()).hexdigest()[:32]


def epoch_plan(plan, epoch):
    value = copy.deepcopy(plan)
    value["run_nonce"] = epoch_nonce(plan["run_nonce"], epoch)
    value["run_dir"] = str(Path(plan["run_dir"]) / "epochs" / epoch)
    value["owner_supervision"] = {k: plan[k] for k in ("run_dir", "run_nonce")}
    return value


def require_epoch_b_allowed(stops, role):
    require(set(stops) == {"source", "passive"}, "Both epoch-A stop receipts required")
    require(
        all(v.get("native_io_stopped") is True and v.get("manager", {}).get("exit_code") == 0 for v in stops.values()),
        "Epoch-A native stop is not proven",
    )
    require(role in stops, "Missing own native stop")


def check_role_events(events):
    expected = [
        "epoch-a-started",
        "epoch-a-drained",
        "epoch-a-stopped",
        "epoch-a-pair-stopped",
        "sentinel-ready",
        "epoch-b-started",
        "epoch-b-drained",
        "epoch-b-stopped",
        "epoch-b-pair-stopped",
        "cache-release",
    ]
    require(events == expected, "Per-role cancellation/restart lifetime order differs")


def check_fresh_identities(old, new):
    for kind in ("manager", "bridge"):
        require(set(old[kind]) == set(new[kind]) == {"source", "passive"}, "Missing exact process roles")
        for role in ("source", "passive"):
            a, b = old[kind][role], new[kind][role]
            require(
                all(type(v.get(k)) is int and v[k] > 0 for v in (a, b) for k in ("pid", "start_ticks")),
                "Missing process generation identity",
            )
            require(a["pid"] != b["pid"] and a != b, "Restart reused a process generation")
