"""Offline exact command/packed-byte/lifetime verifier; no native imports."""

import hashlib
import json
from pathlib import Path

from bridge_network import validate_bridge_network
from cancel_restart_contract import check_cancelled_pair, check_restart
from delayed_peer import check_delayed_passive, check_unarmed
from epoch_contract import check_fresh_identities, check_role_events, epoch_plan
from lifecycle import clean_lifecycle
from page_io import PAGE
from runner_support import require, require_clean_manager_exit, sha256
from runtime_cancel import prompt_pair, validate_real_chunk
from transfer_contract import check_manager_ready
from transfer_lifetime import peer_stopped, same_owner, started_manager


def read(path):
    return json.loads(Path(path).read_bytes())


def page_file(output, label, end=32):
    row = read(output / (label + ".json"))
    path = Path(row["path"])
    raw = path.read_bytes()
    count = 512 * (end // 32)
    require(
        path == output / (label + ".bin")
        and len(raw) == count * PAGE
        and row["pages"] == count
        and row["bytes"] == len(raw)
        and row["slot"] == (0 if output.name == "source" else 1)
        and row["begin"] == 0
        and row["end"] == end
        and sha256(path) == row["sha256"],
        "Wrong selected packed file",
    )
    groups = {}
    index = 0
    for config in range(16):
        for layer in range(32):
            for position in range(0, end, 32):
                key = f"{config}:{layer}" if end == 32 else f"{config}:{layer}:{position}"
                groups[key] = hashlib.sha256(raw[index * PAGE : (index + 1) * PAGE]).hexdigest()
                index += 1
    require(groups == row["groups"], "Per-page hashes differ from saved packed bytes")
    return raw, groups


def verify(plan):
    run = Path(plan["run_dir"])
    reports = {role: read(run / role / "result.json") for role in ("source", "passive")}
    identities = {}
    prompts = prompt_pair(read(plan["input_ids"]), read(plan["restart_input_ids"]))
    for role, report in reports.items():
        require(
            report["run_nonce"] == plan["run_nonce"]
            and report["role"] == role
            and report["ok"] is True
            and report["owner_cleanup_complete"] is True
            and not report["errors"]
            and not report["cleanup_errors"],
            "Owner failed",
        )
        require(
            report["runtime_h2d_tested"] is (role == "source")
            and report["model_executed"] is (role == "source")
            and report["decoder_tested"] is False
            and report["bytes_in_flight_at_cancel_proven"] is False,
            "Scope exceeded",
        )
        require(
            report["initial_seed_calls"] == 128
            and report["synthetic_acks"] == 0
            and report["real_layer_acks"] == (64 if role == "source" else 0),
            "Seed/ack count differs",
        )
        if role == "passive":
            require(report["restart_seed_calls"] == 128, "Missing passive sentinel rewrite")
        require(len(report["cpu_affinity"]) == 1, "Owner CPU affinity differs")
        check_role_events(report["events"])
        started = read(run / (role + "-supervisor") / "started.json")
        require(same_owner(started["owner"], report["owner"]), "Owner generation changed")
        require(
            clean_lifecycle((run / (role + "-supervisor") / "owner.log").read_text(errors="replace")),
            "No final clean32 close",
        )
        require(read(run / role / "native-environment.json")["actual_exit"] == 0, "Native probe failed")
        for epoch in ("a", "b"):
            ep = epoch_plan(plan, epoch)
            out = Path(ep["run_dir"]) / role
            require(
                read(out / "bridge-exit.json")["exit_code"] == 0
                and report["epochs"][epoch]["terminal"]["bridge_exit_code"] == 0,
                "Bridge terminal exit differs",
            )
            native = peer_stopped(ep, "passive" if role == "source" else "source")
            require(native is not None, "Epoch stop proof missing")
            identity = started_manager(ep, role, report["owner"], out)
            row = read(out / "processes.json")
            require(
                row["run_nonce"] == ep["run_nonce"] and row["role"] == role and same_owner(row["manager"], identity),
                "Epoch process receipt differs",
            )
            identities.setdefault(epoch, {k: {} for k in ("manager", "bridge")})
            for kind in ("manager", "bridge"):
                identities[epoch][kind][role] = row[kind]
            log = (out / "manager.log").read_text(errors="replace")
            require_clean_manager_exit(0, log)
            check_manager_ready(ep, role, 200, log)
            validate_bridge_network(ep, role, read(out / "bridge-config.json"), read(out / "bridge-network.json"))
            for label in ("ownership-before-transfer", "ownership-after-transfer"):
                observed = read(out / (label + ".json"))
                require(same_owner(observed["process"], identity), "Observed another manager")
                seats = observed["seats"]["seats"]
                require(len(seats) == len({(x["asic_id"], x["x"], x["y"]) for x in seats}) == 64, "Missing held seats")
                require(
                    observed["environment"] == read(out / "manager-environment.json"), "Observed environment differs"
                )
            if role == "source":
                runtime = read(out / "real-runtime.json")
                validate_real_chunk(
                    runtime, epoch=epoch, nonce=ep["run_nonce"], ids=prompts[0][:1024] if epoch == "a" else prompts[1]
                )
                require(
                    page_file(out, "selected-prewrite")[1] == runtime["source_before"]
                    and page_file(out, "selected-before")[1] == runtime["source_after"],
                    "Runtime source snapshot is not backed by saved bytes",
                )
                require(
                    page_file(out, "selected-after-stop")[0] == page_file(out, "selected-before")[0],
                    "Native stop changed actual runtime source",
                )
    a = Path(epoch_plan(plan, "a")["run_dir"])
    b = Path(epoch_plan(plan, "b")["run_dir"])
    for label in ("unarmed-complete", "arm-permitted"):
        check_unarmed(read(a / "source" / (label + ".json"))["snapshot"])
    before, after = (
        page_file(a / "passive", "selected-delay-before", 1024)[1],
        page_file(a / "passive", "selected-delay-after", 1024)[1],
    )
    delayed = read(a / "passive/delay-verified.json")
    check_delayed_passive(before, after, delayed["snapshot"])
    require(
        delayed["pages"] == after and read(a / "passive/delay-held.json")["pages"] == before,
        "Delayed receipt is not backed by saved bytes",
    )
    check_cancelled_pair(
        read(a / "source/cancel-terminal.json")["snapshot"], read(a / "passive/cancel-terminal.json")["snapshot"], 700
    )
    check_fresh_identities(identities["a"], identities["b"])
    src, src_hashes = page_file(b / "source", "selected-before")
    sentinel, sentinel_hashes = page_file(b / "passive", "selected-before")
    dst, dst_hashes = page_file(b / "passive", "selected-after")
    require(src == dst, "Restart packed bytes differ")
    for role, expected in [("source", src), ("passive", dst)]:
        require(page_file(b / role, "selected-after-stop")[0] == expected, "Native shutdown changed selected pages")
    receipt = read(b / "source/restart-terminal.json")["receipt"]
    require(
        receipt["source_before"] == src_hashes and receipt["runtime_receipt"] == read(b / "source/real-runtime.json"),
        "Restart used another source snapshot",
    )
    receipt.update(passive_sentinel=sentinel_hashes, passive_after=dst_hashes)
    check_restart(receipt)
    for role in ("source", "passive"):
        rewritten = read(b / "passive/sentinel-rewritten.json")["prior_native_stops"]
        require(
            rewritten[role] == sha256(a / role / "native-stopped.json"),
            "Rewrite was not bound to both stopped managers",
        )
    return dict(
        verified_exit=0,
        epochs=2,
        cancel_uuid=700,
        restart_uuid=701,
        synthetic_layer_acks=0,
        real_layer_acks=64,
        real_runtime_calls=2,
        delayed_peer_pages=16384,
        restart_selected_pages=512,
        restart_packed_bytes=512 * PAGE,
        exact_packed_bytes=True,
        owner_allocations_retained_across_restart=True,
        both_epoch_stop_barriers=True,
        runtime_h2d_tested=True,
        model_executed=True,
        decoder_tested=False,
        bytes_in_flight_at_cancel_proven=False,
    )
