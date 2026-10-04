"""Schema2 declared-producer storage gate; no synthetic model completion marker."""
import json
import re
from pathlib import Path

import owned_seed_copy as owned
import producer_profile as profiles
import slurm_pair_checks as checks


def input_gate(storage, protocol, snapshot):
    assert storage["schema"] == 2 and storage["snapshot_revision"] == checks.SNAPSHOT
    assert storage["producer"]["source"] == profiles.BASELINE and storage["producer"]["protocol"] == protocol
    assert storage["producer"]["limits"] == profiles.budget()
    assert storage["producer"]["scope"] == "declared-stock-baseline-only"
    assert str(snapshot.resolve(strict=True)) == storage["snapshot_canonical_path"]
    assert not any(
        k in storage for k in ("seed_markers", "seed_files", "seed_provenance")
    ), "Schema1 cache assumption refused"
    return storage


def completed(status, phase, native, payload_hash, protocol):
    assert status["complete"] and status["protocol"] == protocol and status["exit_code"] in (0, 1)
    if protocol == "chunked":
        assert status["all_three_assertions_completed"] and status["payloads_sha256"] == payload_hash
    else:
        assert protocol == "structured" and status["structured_scope"]["completed"] == 32
    assert status["source"] == (profiles.CANDIDATE if phase == "candidate" else profiles.BASELINE)
    assert status["phase"] == phase and status["native_members"] == native
    assert status["cache_production_declared"] == (phase == "producer")
    return profiles.validate_profile(status["profile"], protocol)


def seal(seed, producer_status, checkpoint=lambda: None):
    profile = profiles.validate_profile(producer_status["profile"], producer_status["protocol"])
    before = owned.inventory(seed)
    assert before and all(
        name.endswith(".tensorbin") for name in before
    ), "Only stock tensorbin producer files admitted"
    kv = {"k": set(), "v": set()}
    files = {}
    for name, stamp in before.items():
        checkpoint()
        digest = checks.sha(seed / name, checkpoint)
        assert owned.stamp((seed / name).stat()) == stamp, "Producer seed changed during seal"
        files[name] = {"bytes": stamp[2], "sha256": digest}
        match = re.fullmatch(r"empty_([kv])cache_paged_attention.+_t([0-9]+)_dtype_.+\.tensorbin", Path(name).name)
        if match:
            kv[match[1]].add(int(match[2]))
    assert kv == {"k": set(range(32)), "v": set(range(32))}, "Missing stock32-layer KV tensor seed"
    assert before == owned.inventory(seed)
    assert sum(r["bytes"] for r in files.values()) <= profiles.budget()["cache_cap_bytes"]
    return {
        "schema": 2,
        "kind": "declared-stock-baseline-seed",
        "profile": profile,
        "protocol": producer_status["protocol"],
        "producer_status": producer_status,
        "files": files,
        "producer_canonical_path": str(seed.resolve()),
        "full_byte_seal": True,
    }


def verify(seed, seal, expected_profile, checkpoint=lambda: None):
    assert seal["schema"] == 2 and seal["kind"] == "declared-stock-baseline-seed" and seal["full_byte_seal"]
    assert profiles.validate_profile(seal["profile"], seal["protocol"]) == profiles.validate_profile(
        expected_profile, seal["protocol"]
    )
    inventory = owned.inventory(seed)
    assert set(inventory) == set(seal["files"]), "Seed membership differs"
    for name, stamp in inventory.items():
        checkpoint()
        expected = seal["files"][name]
        assert (
            stamp[2] == expected["bytes"] and checks.sha(seed / name, checkpoint) == expected["sha256"]
        ), "Seed byte corruption"
        assert owned.stamp((seed / name).stat()) == stamp
    assert inventory == owned.inventory(seed), "Seed changed while verifying"
    return {
        "schema": 2,
        "verified": True,
        "bytes": sum(r["bytes"] for r in seal["files"].values()),
        "files": len(inventory),
    }
