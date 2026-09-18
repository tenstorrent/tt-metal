"""Future post-run evidence checks; no native imports or decoder/model execution."""
import argparse
import json
from pathlib import Path

from page_io import SavedPages, range_keys
from range_contract import generation, scenario, snapshot_receipt
from range_pages import PageEffect, source_progress_policy
from runner_support import require, sha256


def check_reports(source, passive, doc):
    for role, report in (("source", source), ("passive", passive)):
        require(
            report.get("role") == role
            and report.get("ok") is True
            and report.get("owner_cleanup_complete") is True
            and not report.get("errors")
            and not report.get("cleanup_errors")
            and report.get("manager_exit") == 0
            and report.get("native_transfer_tested") is True,
            "failed or incomplete owner report",
        )
        final = report.get("after_manager_shutdown", {})
        require(
            final.get("pages") == 65536 and final.get("bytes") == 285212672 and final.get("packed_bytes_equal") is True,
            "post-manager full-cache proof missing",
        )
        require(
            report.get("persistent_h2d_tested") is (role == "source")
            and report.get("model_executed") is (role == "source"),
            "wrong source/passive execution scope",
        )
        require(
            len(report["phases"]) == 6 and len(report["snapshots"]) == (7 if role == "source" else 6),
            "missing/duplicate phase/snapshot",
        )
        for phase, row in zip(doc["phases"], report["phases"]):
            require(row["uuid"] == phase["source_command"]["uuid"], "phase order/UUID differs")
            generation(row["terminal"], phase, role)
            require(generation(row["retired"], phase, role)["retired"] is True, "generation not retired")
    calls = [(phase, call) for phase in doc["phases"] for call in phase["compute_calls"]]
    require(
        len(source["requests"]) == 7 and len(source["acks"]) == len(source["published"]) == 224,
        "missing or duplicate full32 call",
    )
    for ordinal, ((phase, call), request, snapshot) in enumerate(zip(calls, source["requests"], source["snapshots"])):
        require(
            request["ordinal"] == ordinal
            and request["uuid"] == phase["source_command"]["uuid"]
            and all(request[k] == call[k] for k in ("slot", "request_id", "begin", "end"))
            and request["routed_acks"] == 32
            and request["borrowed_input_preserved"] is True
            and request["all32_metadata_equal"] is True,
            "wrong runtime request identity",
        )
        require(
            snapshot["identity"]["ordinal"] == ordinal
            and snapshot["identity"]["uuid"] == phase["source_command"]["uuid"]
            and all(snapshot["identity"][k] == call[k] for k in ("slot", "request_id", "begin", "end")),
            "wrong immutable call snapshot",
        )
        policy = source_progress_policy(phase, call)
        checks = snapshot.get("checks", {})
        require(
            all(checks.get(key) == value for key, value in policy.items())
            and checks.get("valid_change_groups") == policy["expected_valid_change_groups"],
            "missing valid-region progress evidence",
        )
        for layer in range(32):
            ack = source["acks"][ordinal * 32 + layer]
            published = source["published"][ordinal * 32 + layer]
            require(
                (ack["request_id"], ack["slot"], ack["start"], ack["end"], ack["layer"])
                == (ordinal, call["slot"], call["begin"], call["end"], layer),
                "wrong ack association/order",
            )
            require(
                ack["synchronized_ns"] <= snapshot["snapshot_complete_ns"] <= ack["ack_ns"] <= published["published_ns"]
                and (published["request_id"], published["layer"]) == (ordinal, layer),
                "ack preceded sync/snapshot or routed differently",
            )
    return dict(
        generations=6,
        runtime_calls=7,
        post_sync_acks=224,
        native_layer_commands=224,
        selected_page_visits=24576,
        selected_packed_bytes=106954752,
    )


def check_files(source, passive, doc, nonce):
    # Reconstruct destination evolution from saved source pages and each exact prior destination snapshot.
    before = SavedPages(passive["baseline"]["files"], 2048)
    verified = 0
    try:
        ordinal = 0
        for phase, row in zip(doc["phases"], passive["snapshots"]):
            ordinal += len(phase["compute_calls"])
            source_row = source["snapshots"][ordinal - 1]
            snapshot_receipt(source_row, phase, nonce, "source")
            snapshot_receipt(row, phase, nonce, "passive")
            require(
                row["identity"]["source_snapshot_sha256"] == source_row["receipt_sha256"],
                "destination used another source snapshot",
            )
            previous = passive["baseline"] if verified == 0 else passive["snapshots"][verified - 1]
            require(
                row["identity"]["previous_snapshot_sha256"] == previous["receipt_sha256"],
                "broken passive snapshot chain",
            )
            src = dst = None
            try:
                src = SavedPages(source_row["files"], 2048)
                dst = SavedPages(row["files"], 2048)
                effect = PageEffect("passive", phase=phase)
                for slot in (0, 1):
                    for key in range_keys(slot, 0, 2048):
                        expected = (
                            src.get((key[0], phase["source_slot"], key[2], key[3]))
                            if slot == effect.slot and key[3] in effect.positions
                            else None
                        )
                        effect.accept(key, before.get(key), dst.get(key), expected)
                require(effect.finish() == row["checks"], "saved readback coverage differs")
                before.close()
                before = dst
                dst = None
                verified += 1
            finally:
                if src is not None:
                    src.close()
                if dst is not None:
                    dst.close()
    finally:
        before.close()
    # Every source capture retains the other slot and all pages outside this runtime call.
    before = SavedPages(source["baseline"]["files"], 2048)
    try:
        ordinal = 0
        for phase in doc["phases"]:
            for call in phase["compute_calls"]:
                row = source["snapshots"][ordinal]
                snapshot_receipt(row, phase, nonce, "source")
                previous = source["baseline"] if ordinal == 0 else source["snapshots"][ordinal - 1]
                require(
                    row["identity"]["previous_snapshot_sha256"] == previous["receipt_sha256"],
                    "broken source snapshot chain",
                )
                current = SavedPages(row["files"], 2048)
                try:
                    selected = set(range(call["begin"], (call["end"] + 31) // 32 * 32, 32))
                    for slot in (0, 1):
                        for key in range_keys(slot, 0, 2048):
                            if slot != call["slot"] or key[3] not in selected:
                                require(before.get(key) == current.get(key), "source untouched bytes differ")
                    before.close()
                    before = current
                    current = None
                finally:
                    if current is not None:
                        current.close()
                ordinal += 1
    finally:
        before.close()
    return dict(
        exact_destination_generations=verified,
        source_untouched_calls=ordinal,
        source_structural_values="owner-checked; no new numerical golden",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    doc = scenario(plan["scenario"])
    run = Path(plan["run_dir"])
    for path, digest in plan["pins"].items():
        require(sha256(path) == digest, "changed pinned source: " + path)
    source = json.loads((run / "source/result.json").read_bytes())
    passive = json.loads((run / "passive/result.json").read_bytes())
    require(source["run_nonce"] == passive["run_nonce"] == plan["run_nonce"], "wrong run identity")
    print(
        json.dumps(
            dict(
                report=check_reports(source, passive, doc),
                bytes=check_files(source, passive, doc, plan["run_nonce"]),
                full_model_accepted=False,
                decoder_tested=False,
                controller_lifetime_verification_still_required=True,
            ),
            indent=2,
        )
    )
