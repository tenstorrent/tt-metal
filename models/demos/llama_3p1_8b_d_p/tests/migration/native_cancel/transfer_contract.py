"""Initial 2K native source-to-passive gate; no decoder or generation semantics."""

import re
from pathlib import Path
from urllib.parse import urlparse

from runner_support import require, sha256

CONFIGS = tuple(f"{kind}_h{i}" for kind in ("k", "v") for i in range(8))


def check_manager_ready(plan, role, health, log):
    require(health == 200, "Both-endpoint native discovery/tables not ready")
    require("All configured tables loaded successfully" in log, "Native tables not loaded")
    require("Device map loaded: 32 entries" in log, "Native map incomplete")
    label = "prefill" if role == "source" else "decode"
    expected = (
        f"KV chunk indexes installed for host '{plan[role]['host']}' from {label} tables "
        "(16 configs, 65536 read / 65536 write chunks over 32 device(s))"
    )
    require(log.count(expected) == 1, "Native local source/passive index differs")
    for prefix in ("Prefill", "Decode"):
        for name in CONFIGS:
            require(
                log.count(f"{prefix} KV chunk table loaded: config '{name}'") == 1,
                "Missing/duplicate native configuration",
            )
    for line in log.splitlines():
        if "[ERROR]" in line:
            require(
                "KV manager not ready after startup (discovery pending: peers not yet resolved)" in line,
                "Unexpected native startup error: " + line,
            )


def validate_plan(plan):
    require(
        plan.get("reviewed") is True
        and plan.get("launch_authorized") is True
        and plan.get("scope") == "2k_native_real_cancel_restart"
        and re.fullmatch("[0-9a-f]{32}", plan["run_nonce"]),
        "Closed or invalid cancellation plan",
    )
    require(
        plan["capacity"] == 2048 and re.fullmatch("[0-9a-f]{64}", str(plan.get("bridge_manifest_sha256", ""))),
        "Wrong integrated bridge contract",
    )
    require(plan["source"]["host"] != plan["passive"]["host"], "Two distinct physical endpoints required")
    for role in ("source", "passive"):
        local = plan[role]
        match = re.fullmatch(r"bh-glx-(\d+)-(\w+)", local["host"])
        require(match is not None and str(local["job_id"]).isdigit(), "Unassigned endpoint")
        require(
            local["node_lock"].endswith("/tmp/prefill-device-" + match[1] + "-" + match[2] + ".lock"),
            "Wrong physical lock",
        )
        require(
            local["health_receipt"] in plan["pins"] and "device_ids" not in local, "Unbound health or guessed UMD IDs"
        )
        for port in ("control_port", "health_port", "manager_port", "manager_control_port"):
            require(type(local[port]) is int and 1024 <= local[port] <= 65535, "Invalid task-local port")
        require(
            len({local[k] for k in ("control_port", "health_port", "manager_port", "manager_control_port")}) == 4,
            "Listener ports collide",
        )
    require(plan["source"]["endpoint_id"] != plan["passive"]["endpoint_id"], "Endpoint IDs collide")
    endpoint = urlparse(plan["etcd_endpoint"])
    require(
        endpoint.scheme == "http" and endpoint.hostname == plan["source"]["host"] and endpoint.port,
        "Wrong shared etcd endpoint",
    )
    require(plan["etcd_peer_endpoint"].startswith("http://127.0.0.1:"), "etcd raft must stay local")
    required = (
        "accepted_gate_spec",
        "accepted_source_pins",
        "input_ids",
        "restart_input_ids",
        "runtime_edge_evidence",
        "cancel_spec",
        "seed_helper",
        "environment_script",
        "manager_binary",
        "dmk_elf",
        "source_client",
        "passive_client",
        "etcd_binary",
        "native_probe",
        "startup_evidence",
        "bridge_host_validation",
        "bridge_manifest",
    )
    for key in required:
        require(plan.get(key) in plan["pins"], "Missing required frozen provenance: " + key)
    import json

    host = json.loads(Path(plan["bridge_host_validation"]).read_bytes())
    require(
        host.get("reviewed") is True
        and host.get("actual_exit") == host.get("verified_exit") == 0
        and host.get("bridge_manifest_sha256") == plan["bridge_manifest_sha256"]
        and host.get("cpp_cases_passed") == 48
        and host.get("no_metal_umd_linkage") is True,
        "Range C++ validation is still pending",
    )
    for role in ("source", "passive"):
        binary = plan[role + "_client"]
        require(host["binaries"].get(binary) == plan["pins"][binary], "Unvalidated range executable")
    require(
        set(plan["manager_libraries"]) == {"libtt_metal.so", "libtt-umd.so", "libtt_stl.so", "libtracy.so"},
        "Wrong native library set",
    )
    for pin in plan["manager_libraries"].values():
        require(plan["pins"].get(pin["path"]) == pin["sha256"], "Unpinned native library")
    for path in Path(__file__).parent.glob("*.py"):
        if not path.name.startswith(("checks_", "test_")):
            require(plan["pins"].get(str(path)) == sha256(path), "Unpinned helper " + path.name)
    require(sha256(plan["bridge_manifest"]) == plan["bridge_manifest_sha256"], "Bridge preparation changed")
    for path, digest in plan["pins"].items():
        require(sha256(path) == digest, "Changed source/binary: " + path)
