"""Pure final-evidence checks for real capacity migration; no native/device access."""

import json
from pathlib import Path

from capacity_execution import make_cases, require, resources
from capacity_observation import check_capacity_ready
from lifecycle import clean_lifecycle
from range_contract import generation
from runner_support import sha256


def verify_report(report, role, doc):
    wanted = resources(doc["capacity"])
    calls = wanted["full32_chunk_calls"]
    require(
        report.get("ok") is True
        and report.get("owner_cleanup_complete") is True
        and not report.get("errors")
        and not report.get("cleanup_errors"),
        "Owner failed or cleanup incomplete",
    )
    require(
        report.get("capacity") == doc["capacity"] and report.get("manager_exit") == report.get("bridge_exit_code") == 0,
        "Owner capacity or native exit differs",
    )
    require(
        report.get("native_transfer_tested") is True
        and report.get("model_executed") is (role == "source")
        and report.get("persistent_h2d_tested") is (role == "source"),
        "Wrong producer scope",
    )
    require(
        report.get("compile_warmup_full32_calls") == (2 if role == "source" else 0), "Unreported runtime warmup calls"
    )
    warmup = report["capacity_warmup"]
    expected_calls = [x for phase in doc["phases"] for x in phase["compute_calls"]]
    require(
        report["warmup_barrier_before_native_clients"] is True
        and warmup["full32_calls"] == calls
        and len(warmup["calls"]) == calls
        and warmup["native_acks"] == 0
        and warmup["serving_request_id_before"] == warmup["serving_request_id_after"] == -1,
        "Warmup not isolated from native clients/serving state",
    )
    from capacity_warmup import token_digest

    hashes = {}
    for phase in doc["phases"]:
        tokens = phase["source_command"]["tokens"]
        hashes[phase["source_slot"]] = (token_digest(tokens), token_digest([(x + 1) % 128256 for x in tokens]))
    for row, expected in zip(warmup["calls"], expected_calls):
        real_hash, warm_hash = hashes[expected["slot"]]
        require(
            all(row[k] == expected[k] for k in ("slot", "begin", "end"))
            and row["native_acks"] == 0
            and (row["real_prompt_sha256"], row["prompt_sha256"]) == (real_hash, warm_hash),
            "Incomplete/distinct warmup geometry",
        )
    phases = report["phases"]
    require(len(phases) == 2, "Missing request result")
    for expected, actual in zip(doc["phases"], phases):
        require(actual["uuid"] == expected["source_command"]["uuid"], "Wrong request order")
        generation(actual["terminal"], expected, role)
        require(
            actual["readback"]["exact"] is True and actual["readback"]["pages"] == 16384,
            "Destination selected readback incomplete",
        )
        require(
            generation(actual["retired"], expected, role)["retired"] is True,
            "Generation not retired after verified landing",
        )
        if role == "passive":
            sentinels = actual["readback"]["untouched_samples"]
            from capacity_execution import sentinel_keys

            require(
                sentinels["unchanged_sha256"] is True
                and sentinels["pages"] == len(sentinel_keys(doc["capacity"], expected)),
                "Adjacent/other-slot sentinel coverage incomplete",
            )
    if role == "source":
        expected_calls = [x for phase in doc["phases"] for x in phase["compute_calls"]]
        requests = report["requests"]
        require(
            len(requests) == calls
            and all(
                all(row[k] == expected[k] for k in ("slot", "request_id", "begin", "end"))
                and row["ordinal"] == i
                and row["routed_acks"] == 32
                and row["borrowed_input_preserved"] is True
                and row["all32_metadata_equal"] is True
                for i, (row, expected) in enumerate(zip(requests, expected_calls))
            ),
            "H2D/full32 coverage differs",
        )
        require(len(report["acks"]) == len(report["published"]) == calls * 32, "Ack inventory differs")
        for i, (row, pub) in enumerate(zip(report["acks"], report["published"])):
            require(
                row["request_id"] == pub["request_id"] == i // 32
                and row["layer"] == pub["layer"] == i % 32
                and row["synchronized_ns"] <= row["ack_ns"] <= pub["published_ns"],
                "Readiness was not ordered after synchronization",
            )
        require(
            report["terminal"]["successful"] is True and len(report["terminal"]["calls"]) == 104,
            "Native audited selected-range inventory differs",
        )
        require(len(report["selected_captures"]) == calls, "Missing per-chunk capture boundary")
        expected_captures = []
        for phase in doc["phases"]:
            selected = phase["source_command"]
            for call in phase["compute_calls"]:
                overlap = max(0, min(call["end"], selected["to"]) - max(call["begin"], selected["from"]))
                expected_captures.append(overlap // 32 * 16 * 32)
        for i, row in enumerate(report["selected_captures"]):
            require(
                row["request"] == i
                and row["selected_pages"] == expected_captures[i]
                and row["snapshot_complete_ns"] <= report["published"][i * 32]["published_ns"],
                "Selected snapshot coverage/order differs",
            )
        changed = [row["changed_from_pre_request"] for row in report["selected_captures"] if "selected" in row]
        require(len(changed) == 2, "Missing pre-request stale-cache comparison")
        from capacity_pages import require_changed_groups

        for row in changed:
            require_changed_groups(row["before"], row["after"])
        hashes = report["selected_source_config_hashes"]
        require(
            len(hashes) == 2 and all(hashes[0][str(c)] != hashes[1][str(c)] for c in range(16)),
            "Source slot distinction absent",
        )
    final = report["after_manager_shutdown"]
    require(
        len(final) == 2 and all(row["exact"] is True and row["pages"] == 16384 for row in final),
        "Final selected bytes not preserved",
    )
    if role == "passive":
        from capacity_execution import sentinel_keys

        count = len({k for phase in doc["phases"] for k in sentinel_keys(doc["capacity"], phase, doc["phases"])})
        require(
            report["final_untouched_samples"]["unchanged_sha256"] is True
            and report["final_untouched_samples"]["pages"] == count,
            "Final sentinels not preserved",
        )
    for key in ("manager_memory_after_tables", "manager_memory_after_transfer"):
        memory = report[key]
        require(
            0 < memory["rss_bytes"] <= memory["hwm_bytes"] <= memory["limit_bytes"],
            "Memory evidence exceeds reviewed limit",
        )
    return wanted


def verify(plan):
    run = Path(plan["run_dir"])
    doc, _ = make_cases(plan["capacity"], plan["book_manifest"], plan["pins"][plan["book_manifest"]])
    before = json.loads((run / "source-before.json").read_bytes())
    require(before == plan["pins"] == json.loads((run / "source-after.json").read_bytes()), "Source maps differ")
    dispatch = json.loads((run / "dispatch-result.json").read_bytes())
    require(dispatch["dispatch_exits"] == {"source": 0, "passive": 0}, "Dispatch did not pass")
    reports = {}
    for role in ("source", "passive"):
        directory = run / role
        report = json.loads((directory / "result.json").read_bytes())
        verify_report(report, role, doc)
        require(report["run_nonce"] == plan["run_nonce"] and report["role"] == role, "Foreign owner result")
        require(
            json.loads((directory / "native-environment.json").read_bytes())["actual_exit"] == 0,
            "Native provenance probe failed",
        )
        node = json.loads((run / (role + "-node") / "node-contract.json").read_bytes())
        require(
            node["node"] == plan[role]["host"]
            and str(node["job"]) == str(plan[role]["job_id"])
            and node["cpus_per_task"] == 1
            and len(node["affinity"]) == 1,
            "Node/CPU identity differs",
        )
        started = json.loads((directory / "manager-started.json").read_bytes())["manager"]
        for name in ("manager_memory_after_tables", "manager_memory_after_transfer"):
            observed = report[name]
            require(
                observed["limit_bytes"] == plan["manager_rss_limit_bytes"][role]
                and all(observed["process"][key] == started[key] for key in ("pid", "start_ticks")),
                "Manager memory observation has wrong generation/budget",
            )
        require(
            json.loads((directory / "bridge-exit.json").read_bytes())["exit_code"] == 0,
            "Actual native client exit differs",
        )
        from capacity_resources import admit

        admit(
            json.loads((run / (role + "-node") / "host-resource-before-native.json").read_bytes()),
            plan["minimum_host_available_bytes"],
            plan["minimum_shared_disk_free_bytes"],
        )
        require(json.loads((run / (role + "-node") / "source-on-node.json").read_bytes()) == before, "Node pins differ")
        decision = json.loads((run / (role + "-supervisor") / "decision.json").read_bytes())
        require(
            decision["phase"] == "finished" and decision["owner_exit"] == 0 and decision["release_lock"] is True,
            "Supervisor did not release cleanly",
        )
        log = (run / (role + "-supervisor") / "owner.log").read_text(errors="replace")
        require(clean_lifecycle(log) is not None, "Missing all32 final close")
        check_capacity_ready(
            plan["capacity"], role, plan[role]["host"], 200, (directory / "manager.log").read_text(errors="replace")
        )
        for row in report["selected_captures"]:
            if "selected" in row:
                receipt = row["selected"]
                require(sha256(receipt["path"]) == receipt["sha256"], "Saved selected bytes changed")
        reports[role] = dict(result_sha256=sha256(directory / "result.json"), manager_exit=report["manager_exit"])
    return dict(
        actual_exit=0,
        verified_exit=0,
        scope="real_prefill_capacity_selected_native_transfer",
        capacity=plan["capacity"],
        resources=resources(plan["capacity"]),
        roles=reports,
        model_golden_compared=False,
        performance_measured=False,
        full_model_numerical_accepted=False,
    )
