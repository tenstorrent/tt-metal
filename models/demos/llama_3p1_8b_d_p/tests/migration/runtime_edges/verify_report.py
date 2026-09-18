"""Independent inventory and receipt-order verifier; no native imports."""
import argparse
import hashlib
import json
from pathlib import Path

from edge_coverage import RUNTIME_CALLS, pages
from support import require, sha256


def verify(report, fixtures, load_snapshot):
    require(
        report.get("gate_passed") is True and report.get("owner_cleanup_complete") is True,
        "owner failed or did not close",
    )
    require(
        not report["errors"] and not report["cleanup_errors"] and report.get("recovery_required") is False,
        "hidden failure",
    )
    require(
        report.get("cleanup_attempts")
        == [
            "service.drop",
            "service.collect",
            "channel.drop",
            "producer",
            "router",
            "saved",
            "cache.k",
            "cache.v",
            "model",
            "model.collect",
            "mesh.synchronize",
            "fabric.disable",
            "mesh.close",
        ],
        "missing or repeated cleanup action",
    )
    for key in (
        "native_manager_tested",
        "native_source_pin_or_retirement_tested",
        "native_transfer_tested",
        "kv_golden_comparison_performed",
        "new_model_numerical_acceptance",
        "decoder_tested",
    ):
        require(report.get(key) is False, "scope overclaim: " + key)
    require(
        (report["mesh_devices"], report["configs"], report["table_entries"], report["page_bytes"])
        == (32, 16, 65536, 4352),
        "wrong allocation inventory",
    )
    require(report["seed_writer_calls"] == 128 and report["warmup_full32_calls"] == 2, "wrong initialization inventory")
    require(
        len(report["requests"]) == len(report["snapshots"]) == 5
        and len(report["acks"]) == len(report["published"]) == 160,
        "missing or duplicate call/ack",
    )
    base = report["baseline"]
    identity = base["identity"]
    require(identity["ordinal"] == -1 and identity["run_nonce"] == report["run_nonce"], "wrong baseline generation")
    require(base["checks"] == dict(pages=65536, nonzero_pages=65536, seed_groups=1024), "incomplete seed coverage")
    previous = base
    previous_ack = base["capture_finished_ns"]
    for ordinal, summary in enumerate([base] + report["snapshots"]):
        durable = load_snapshot(summary)
        require(
            durable
            == {
                k: v for k, v in summary.items() if k not in ("receipt_path", "receipt_sha256", "snapshot_complete_ns")
            },
            "snapshot receipt differs",
        )
        require(len(durable["files"]) == 2, "snapshot slots missing/duplicated")
        for slot, row in enumerate(durable["files"]):
            require(
                (row["slot"], row["begin"], row["end"], row["pages"], row["bytes"])
                == (slot, 0, 2048, 32768, 142606336),
                "snapshot file coverage differs",
            )
    for request, call in enumerate(RUNTIME_CALLS):
        row = report["requests"][request]
        snapshot = report["snapshots"][request]
        ident = snapshot["identity"]
        checks = snapshot["checks"]
        require(
            tuple(row[k] for k in ("request_id", "prompt", "slot", "begin", "end"))
            == (request, call.prompt, call.slot, call.begin, call.end),
            "request association differs",
        )
        require(
            row["routed_acks"] == 32
            and row["borrowed_input_preserved"] is True
            and row["all32_metadata_equal"] is True,
            "input or routed completion failed",
        )
        require(
            row["native_retirement_checked"] is False and row["runtime_only_reuse"] == (request in (3, 4)),
            "reuse scope differs",
        )
        require(
            tuple(ident[k] for k in ("ordinal", "prompt", "slot", "begin", "end"))
            == (request, call.prompt, call.slot, call.begin, call.end),
            "snapshot call association differs",
        )
        require(all(ident[k] == v for k, v in identity.items() if k != "ordinal"), "cache/source generation changed")
        require(ident["previous_snapshot_sha256"] == previous["receipt_sha256"], "snapshot chain changed")
        tokens = fixtures[call.prompt][call.begin : call.end]
        require(
            ident["token_ids_sha256"] == hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
            "snapshot tokens changed",
        )
        selected = len(pages(call.begin, call.end)) * 512
        padded = ((call.end + 31) // 32 * 32) - call.end
        expected = dict(
            pages=65536,
            changed_pages=selected,
            untouched_pages=65536 - selected,
            configs=16,
            layers=32,
            slots=2,
            valid_values=(call.end - call.begin) * 128 * 512,
            padding_values=padded * 128 * 512,
            semantic_valid_end=call.end,
            packed_end=call.end + padded,
            golden_comparison=False,
            structural_write_checked=True,
        )
        require(all(checks.get(k) == v for k, v in expected.items()), "incomplete semantic/page inventory")
        for layer in range(32):
            ack = report["acks"][request * 32 + layer]
            sent = report["published"][request * 32 + layer]
            require(
                tuple(ack[k] for k in ("request_id", "slot", "start", "end", "layer"))
                == (request, call.slot, call.begin, call.end, layer),
                "ack identity/order differs",
            )
            require((sent["request_id"], sent["layer"]) == (request, layer), "publication identity differs")
            require(
                previous_ack
                <= ack["synchronized_ns"]
                <= snapshot["capture_finished_ns"]
                <= snapshot["snapshot_complete_ns"]
                <= ack["ack_ns"]
                <= sent["published_ns"],
                "snapshot/ack precedes completion",
            )
        previous = snapshot
        previous_ack = report["published"][request * 32 + 31]["published_ns"]
    return dict(
        passed=True,
        calls=5,
        acks=160,
        snapshot_pages=6 * 65536,
        selected_pages=sum(len(pages(c.begin, c.end)) * 512 for c in RUNTIME_CALLS),
        native_transfer_tested=False,
    )


def main(path, plan_path):
    path = Path(path)
    plan = json.loads(Path(plan_path).read_bytes())
    report = json.loads(path.read_bytes())
    fixture = json.loads(Path(plan["fixtures"]).read_bytes())["tokens"]
    identity = report["baseline"]["identity"]
    require(
        report["run_nonce"] == plan["run_nonce"] and identity["plan_sha256"] == sha256(plan_path),
        "plan/generation changed",
    )
    require(identity["fixture_sha256"] == sha256(plan["fixtures"]), "fixture changed")
    require(
        identity["table_sha256"] == sha256(path.parent / "table.pb")
        and identity["device_map_sha256"] == sha256(path.parent / "device-map.json"),
        "allocation exports changed",
    )
    require(
        json.loads((path.parent / "native-environment.json").read_bytes())["actual_exit"] == 0,
        "native provenance failed",
    )
    for name, pin in plan["native_libraries"].items():
        actual = report["loaded_libraries"]["libraries"][name]
        require(
            actual["path"] == str(Path(pin["path"]).resolve()) and actual["sha256"] == pin["sha256"],
            "loaded library changed",
        )

    def load(summary):
        require(sha256(summary["receipt_path"]) == summary["receipt_sha256"], "snapshot receipt hash changed")
        row = json.loads(Path(summary["receipt_path"]).read_bytes())
        for file in row["files"]:
            require(
                Path(file["path"]).stat().st_size == file["bytes"] and sha256(file["path"]) == file["sha256"],
                "snapshot bytes changed",
            )
        return row

    print(json.dumps(verify(report, fixture, load), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    main(args.report, args.plan)
