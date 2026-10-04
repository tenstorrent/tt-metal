#!/usr/bin/env python3
"""Pure filesystem/receipt gates for the formatter Slurm diagnostic."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import subprocess
import tarfile

SNAPSHOT = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_REPO = "models--meta-llama--Llama-3.1-8B-Instruct"
MODEL = "Llama-3.1-8B-Instruct"
OWNER = "moconnor(1211402857)"


HF_ROOT = Path("/mnt/MLPerf")
HF_HUB = HF_ROOT / "huggingface/hub"
# Exact twelve assets observed at the frozen snapshot; no shared-pool fallback.
SNAPSHOT_ASSETS = {
    "config.json": {"blob": "0bb6fd75b3ad2fe988565929f329945262c2814e", "bytes": 855, "shared": None},
    "generation_config.json": {"blob": "cc7276afd599de091142c6ed3005faf8a74aa257", "bytes": 184, "shared": None},
    "model-00001-of-00004.safetensors": {
        "blob": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
        "bytes": 4976698672,
        "shared": "37/37e46c41d055c1b298309131cad034dba275c05a3215d27d33af8034753c8087",
    },
    "model-00002-of-00004.safetensors": {
        "blob": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
        "bytes": 4999802720,
        "shared": "34/34a702f221ae1d749c3b638c6e59e1590c44b7044bfffdcfd3f709633ce53be4",
    },
    "model-00003-of-00004.safetensors": {
        "blob": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
        "bytes": 4915916176,
        "shared": "b8/b85fa10bd12d4ea01487c18eae95c3261e97514648772d0b2883b67cb164c9d2",
    },
    "model-00004-of-00004.safetensors": {
        "blob": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
        "bytes": 1168138808,
        "shared": "41/41dde9eee68695933ec2df79ba83ba26d296aa922d096f3d6cba7103fab06563",
    },
    "model.safetensors.index.json": {
        "blob": "0fd8120f1c6acddc268ebc2583058efaf699a771",
        "bytes": 23950,
        "shared": None,
    },
    "special_tokens_map.json": {"blob": "02ee80b6196926a5ad790a004d9efd6ab1ba6542", "bytes": 296, "shared": None},
    "tokenizer.json": {"blob": "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8", "bytes": 9085657, "shared": None},
    "tokenizer_config.json": {"blob": "db88166e2bc4c799fd5d1ae643b75e84d03ee70e", "bytes": 55351, "shared": None},
    "original/params.json": {"blob": "f1131204e79d0c09d2bac93f11569a8a655d68ba", "bytes": 199, "shared": None},
    "original/tokenizer.model": {
        "blob": "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
        "bytes": 2183982,
        "shared": "c0/c0fff81447bf4ba4dbf0a427469a7e1f7e48a32b365aeaceea7d4760a619a3d7",
    },
}


# Official assets used only to classify the informational layout receipt.
# These names are not an acceptance allowlist; unselected source entries are ignored.
IGNORED_SNAPSHOT_ASSETS = {
    ".gitattributes": {"blob": "a6344aac8c09253b3b630fb776ae94478aa0275b", "bytes": 1519},
    "LICENSE": {"blob": "a7c3ca16cee30425ed6ad841a809590f2bcbf290", "bytes": 7627},
    "README.md": {"blob": "bbd5630a05b65c1a8b25141bd11ec44844107d58", "bytes": 44044},
    "USE_POLICY.md": {"blob": "81ebb55902285e8dd5804ccf423d17ffb2a622ee", "bytes": 4691},
    "original/consolidated.00.pth": {
        "blob": "ab33d910f405204e5d388bc3521503584800461dc96808e287821dd451c1edac",
        "bytes": 16060617592,
    },
}


def fields(text):
    return dict(part.split("=", 1) for part in text.split() if "=" in part)


def duration(text):
    assert text not in ("INVALID", "UNLIMITED", "N/A")
    days, clock = text.split("-", 1) if "-" in text else ("0", text)
    hh, mm, ss = map(int, clock.split(":"))
    assert 0 <= mm < 60 and 0 <= ss < 60
    return int(days) * 86400 + hh * 3600 + mm * 60 + ss


def assignment(text, job_id, node):
    f = fields(text)
    expected = {
        "JobId": str(job_id),
        "UserId": OWNER,
        "JobState": "RUNNING",
        "NodeList": node,
        "BatchHost": node,
        "NumNodes": "1",
        "NumTasks": "1",
        "JOB_GRES": "board:wormhole_b0-lb:4",
        "GRES": "board:wormhole_b0-lb:4(IDX:0-3)",
        "Requeue": "0",
        "Restarts": "0",
        "Reboot": "0",
    }
    assert all(f.get(k) == v for k, v in expected.items()), "Wrong exact Slurm assignment"
    assert node == "wh-lb-45" and duration(f["TimeLimit"]) == 14400
    elapsed = duration(f["RunTime"])
    assert 0 <= elapsed < 13200, "No bounded paired budget remains"
    return {
        "job_id": int(job_id),
        "node": node,
        "owner": OWNER,
        "driver_indices": [0, 1, 2, 3],
        "expected_asics": 8,
        "scheduler_elapsed_seconds": elapsed,
        "remaining_controller_seconds": 13200 - elapsed,
        "scheduler_fields": expected,
        "shared_cpus_correctness_only": True,
    }


def driver_nodes(root=Path("/dev/tenstorrent")):
    # Normal CI assigns a T3K, not the previously inventoried Slurm board IDs.
    # Bind its four root-owned driver nodes and exact relative aliases afresh.
    captured = {}

    def capture(path):
        info = path.lstat()
        stamp = (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_rdev,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if path in captured:
            assert captured[path] == stamp, "Driver namespace replaced during admission"
        else:
            captured[path] = stamp
        return info

    assert stat.S_ISDIR(capture(root).st_mode), "Linked or special driver directory"
    names = sorted(p.name for p in root.iterdir())
    assert names in (
        ["0", "1", "2", "3"],
        ["0", "1", "2", "3", "by-id"],
    ), "Need exactly four assigned PCIe driver nodes and only reviewed aliases"
    result = []
    for index in range(4):
        p = root / str(index)
        info = capture(p)
        assert stat.S_ISCHR(info.st_mode) and info.st_uid == 0, "Unexpected driver-node type or owner"
        result.append({"index": index, "path": str(p), "rdev": info.st_rdev})
    assert len({r["rdev"] for r in result}) == 4
    if "by-id" in names:
        directory = root / "by-id"
        info = capture(directory)
        assert stat.S_ISDIR(info.st_mode) and info.st_uid == 0, "Linked, special or unowned alias directory"
        import re

        aliases = {p.name: os.readlink(p) for p in directory.iterdir()}
        assert len(aliases) == 4 and set(aliases.values()) == {"../0", "../1", "../2", "../3"}
        assert all(re.fullmatch(r"wormhole-[0-9A-F]{16}", name) for name in aliases), "Wrong CI board alias"
        for name, target in aliases.items():
            p = directory / name
            info = capture(p)
            assert stat.S_ISLNK(info.st_mode) and info.st_uid == 0, "Wrong driver alias type or owner"
            assert os.readlink(p) == target, "Escaping or changed driver alias"
            assigned = root / target.removeprefix("../")
            assert p.resolve(strict=True) == assigned, "Alias does not resolve to the assigned driver"
            assert capture(assigned).st_rdev == result[int(assigned.name)]["rdev"]
        assert sorted(p.name for p in directory.iterdir()) == sorted(aliases), "Driver aliases changed during admission"
    assert sorted(p.name for p in root.iterdir()) == names, "Driver membership changed during admission"
    for path in captured:
        capture(path)
    return result


def quiet(expected, proc=Path("/proc")):
    """Read-only FD scan; no unrelated args, environ or FD paths in output."""
    rdevs = {r["rdev"] for r in expected}
    assert len(rdevs) == 4 and {r["index"] for r in expected} == {0, 1, 2, 3}
    assert {os.stat(r["path"]).st_rdev for r in expected} == rdevs
    holders, unknown = [], []
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            birth = int((p / "stat").read_text().rsplit(")", 1)[1].split()[19])
            comm = (p / "comm").read_text().strip()
            for descriptor in (p / "fd").iterdir():
                try:
                    info = descriptor.stat()
                except FileNotFoundError:
                    continue
                if stat.S_ISCHR(info.st_mode) and info.st_rdev in rdevs:
                    holders.append({"pid": int(p.name), "birth_ticks": birth, "comm": comm, "rdev": info.st_rdev})
            again = int((p / "stat").read_text().rsplit(")", 1)[1].split()[19])
            if again != birth:
                unknown.append(int(p.name))
        except FileNotFoundError:
            # A vanished process is harmless; a live PID with missing visibility
            # cannot establish an exclusive, quiet device assignment.
            if p.exists():
                unknown.append(int(p.name))
        except (OSError, ValueError, IndexError):
            unknown.append(int(p.name))
    return {
        "quiet": not holders and not unknown,
        "driver_rdevs": sorted(rdevs),
        "holders": holders,
        "incomplete_pid_visibility": sorted(set(unknown)),
        "telemetry_exclusions": [],
    }


def sha(path, checkpoint=lambda: None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checkpoint()
            digest.update(block)
    return digest.hexdigest()


def file_stamp(p):
    s = p.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def lstat_stamp(path):
    s = path.lstat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def snapshot_sha(name, path, checkpoint=lambda: None):
    # The pinned HF metadata identifies five LFS objects by SHA256 and seven
    # ordinary files by Git blob SHA1. Shared-pool filenames are not content IDs.
    asset = SNAPSHOT_ASSETS[name]
    digest = hashlib.sha256()
    git = hashlib.sha1(("blob " + str(asset["bytes"])).encode() + bytes([0])) if asset["shared"] is None else None
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checkpoint()
            digest.update(block)
            if git is not None:
                git.update(block)
    identity = git.hexdigest() if git is not None else digest.hexdigest()
    assert identity == asset["blob"], "Pinned HF content identity differs"
    return digest.hexdigest()


def snapshot_layout(path):
    """Model-relative metadata only, saved before selection can refuse a layout."""
    canonical = path.resolve(strict=True)
    hub = HF_HUB.resolve(strict=True)
    assert canonical.name == SNAPSHOT and canonical.parent.name == "snapshots"
    assert canonical.parent.parent == hub / MODEL_REPO and hub.is_relative_to(HF_ROOT.resolve(strict=True))
    entries = []
    paths = sorted(canonical.iterdir())
    original = canonical / "original"
    if original in paths and stat.S_ISDIR(original.lstat().st_mode):
        paths.extend(sorted(original.iterdir()))  # Never follow an unknown directory or link.
    for p in paths:
        info = p.lstat()
        row = {"name": p.relative_to(canonical).as_posix(), "kind": "special"}
        if stat.S_ISDIR(info.st_mode):
            row["kind"] = "directory"
        elif stat.S_ISREG(info.st_mode):
            row["kind"] = "regular"
        elif stat.S_ISLNK(info.st_mode):
            row["kind"] = "symlink"
            link = os.readlink(p)
            target = Path(os.path.normpath(str(p.parent / link)))
            canonical_link = (
                not os.path.isabs(link)
                and target.parent == canonical.parent.parent / "blobs"
                and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", target.name) is not None
            )
            row["canonical_repository_link"] = canonical_link
            if canonical_link:
                row["link"] = link
            else:
                row["noncanonical_link_sha256"] = hashlib.sha256(link.encode()).hexdigest()
        row["selection"] = (
            "required"
            if row["name"] in SNAPSHOT_ASSETS
            else "ignored_official"
            if row["name"] in IGNORED_SNAPSHOT_ASSETS
            else "directory"
            if row["name"] == "original"
            else "unknown"
        )
        entries.append(row)
    return {
        "revision": SNAPSHOT,
        "metadata_only": True,
        "entries": entries,
        "selected_assets": sorted(SNAPSHOT_ASSETS),
        "ignored_official_present": sorted(row["name"] for row in entries if row["selection"] == "ignored_official"),
    }


def snapshot_files(path):
    canonical = path.resolve(strict=True)
    assert canonical.name == SNAPSHOT and canonical.parent.name == "snapshots"
    repo = canonical.parent.parent
    hub = HF_HUB.resolve(strict=True)
    assert hub.is_relative_to(HF_ROOT.resolve(strict=True))
    assert repo == hub / MODEL_REPO and not (repo / "blobs").is_symlink(), "Wrong canonical model repository"
    original = canonical / "original"
    assert stat.S_ISDIR(original.lstat().st_mode), "Required asset directory is linked or special: original"
    rows = {}
    # Only these twelve paths enter the model namespace. Shared-cache siblings
    # are informational metadata, not inputs; never recurse or inspect them here.
    for relative, asset in SNAPSHOT_ASSETS.items():
        p = canonical / relative
        blob = repo / "blobs" / asset["blob"]
        assert p.is_symlink() and os.readlink(p) == os.path.relpath(blob, p.parent), (
            "Snapshot link escapes/replaces frozen asset: " + relative
        )
        # CI's supported HF_HUB_CACHE is a standard one-hop repository cache.
        # Required inputs cannot use shared-pool guesses or a second-hop link.
        assert stat.S_ISREG(blob.lstat().st_mode), "CI repo blob replacement/special file: " + relative
        target, second = blob, None
        assert target.resolve(strict=True) == target and stat.S_ISREG(target.lstat().st_mode), (
            "Target replacement/special file: " + relative
        )
        assert p.resolve(strict=True) == target and target.stat().st_size == asset["bytes"], (
            "Frozen asset identity/size differs: " + relative
        )
        rows[relative] = {
            "target": target,
            "stamp": file_stamp(target),
            "binding": {
                "snapshot_link": os.readlink(p),
                "snapshot_lstat": lstat_stamp(p),
                "repo_blob_link": second,
                "repo_blob_lstat": lstat_stamp(blob),
                "target_relative_to_hub": target.relative_to(hub).as_posix(),
            },
        }
    return canonical, repo, rows


def seal_snapshot(path, expected, checkpoint=lambda: None):
    canonical, repo, rows = snapshot_files(path)
    assert set(rows) == set(expected), "Snapshot file membership differs from reviewed seal"
    output = {}
    for name, row in rows.items():
        checkpoint()
        assert (
            row["stamp"] == expected[name]["stamp"] and row["binding"] == expected[name]["binding"]
        ), "Snapshot binding replaced since host seal"
        assert row["stamp"][2] == expected[name]["bytes"]
        digest = snapshot_sha(name, row["target"], checkpoint)
        assert digest == expected[name]["sha256"], "Pinned snapshot bytes differ from reviewed seal"
        assert file_stamp(row["target"]) == row["stamp"], "Snapshot changed during hashing"
        output[name] = {"bytes": row["stamp"][2], "sha256": digest, "stamp": row["stamp"], "binding": row["binding"]}
    assert snapshot_files(path)[2] == rows, "Snapshot membership/identity changed"
    return {
        "requested": str(path),
        "canonical": str(canonical),
        "repo_root": str(repo),
        "hub_root": str(repo.parent),
        "revision": SNAPSHOT,
        "files": output,
        "logical_bytes": sum(r["bytes"] for r in output.values()),
    }


def unchanged_snapshot(seal):
    canonical, repo, current = snapshot_files(Path(seal["canonical"]))
    assert str(repo.parent) == seal["hub_root"] and set(current) == set(seal["files"])
    for name, row in current.items():
        assert (
            row["stamp"] == seal["files"][name]["stamp"] and row["binding"] == seal["files"][name]["binding"]
        ), "Snapshot changed during the paired control"


def snapshot_mount_plan(seal, owned_hub):
    # The twelve byte-sealed CI assets have already been copied into an owned hub.
    unchanged_snapshot(seal)
    base = "/mnt/MLPerf/huggingface/hub"
    mounts = [
        (owned_hub, base, "ro"),
        (Path(seal["canonical"]), base + "/" + MODEL_REPO + "/snapshots/" + SNAPSHOT, "ro"),
        (Path(seal["repo_root"]) / "blobs", base + "/" + MODEL_REPO + "/blobs", "ro"),
    ]
    assert len(mounts) == 3 and all(mode == "ro" for source, target, mode in mounts)
    return mounts


def snapshot_mounts(seal, owned_hub):
    # A task-owned empty namespace makes every overlay destination explicit.
    mounts = snapshot_mount_plan(seal, owned_hub)
    assert not owned_hub.exists() and not owned_hub.is_symlink()
    owned_hub.mkdir(mode=0o700)
    assert owned_hub.stat().st_uid == os.getuid()
    model = owned_hub / MODEL_REPO
    (model / "refs").mkdir(parents=True)
    (model / "refs/main").write_text(SNAPSHOT)
    (model / "snapshots" / SNAPSHOT).mkdir(parents=True)
    (model / "blobs").mkdir()
    base = "/mnt/MLPerf/huggingface/hub/"
    for source, target, mode in mounts[3:]:
        placeholder = owned_hub / Path(target).relative_to(base)
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        with placeholder.open("xb"):
            pass
    return mounts


def topology(doc, indices):
    chips = {int(k): tuple(v) for k, v in doc["chips"].items()}
    assert len(chips) == 8 and len(doc["chip_unique_ids"]) == 8 and len(set(doc["chip_unique_ids"].values())) == 8
    assert set(map(int, doc["arch"])) == set(chips) and set(doc["arch"].values()) == {"wormhole_b0"}
    assert set(chips.values()) == {
        (x, y, 0, 0) for x in range(4) for y in range(2)
    }, "Not the original T3K physical grid"
    mmio = {int(k): int(v) for row in doc["chips_with_mmio"] for k, v in row.items()}
    assert len(mmio) == 4 and set(mmio.values()) == set(indices) and set(mmio).issubset(chips)
    assert not doc.get("ethernet_connections_to_remote_devices"), "No multihost topology is admitted"
    graph = {c: set() for c in chips}
    for edge in doc["ethernet_connections"]:
        assert len(edge) == 2
        a, b = [int(endpoint["chip"]) for endpoint in edge]
        assert a in chips and b in chips and a != b
        graph[a].add(b)
        graph[b].add(a)
    seen, pending = set(), [next(iter(chips))]
    while pending:
        c = pending.pop()
        if c not in seen:
            seen.add(c)
            pending.extend(graph[c] - seen)
    assert seen == set(chips), "Eight ASICs are not connected"
    return {
        "asics": 8,
        "pcie_driver_nodes": 4,
        "connected": True,
        "physical_grid": [4, 2],
        "mmio_to_driver_index": mmio,
    }


def inspect_native(path):
    process = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    total, count, critical = 0, 0, {}
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                p = Path(member.name)
                assert p.parts and not p.is_absolute() and ".." not in p.parts
                assert p.parts[0] in {"build", "runtime", "tt_metal", "ttnn"}
                assert member.isfile() or member.isdir() or member.issym() or member.islnk()
                if member.issym() or member.islnk():
                    target = Path(member.linkname)
                    assert not target.is_absolute()
                    resolved = Path(os.path.normpath(str(p.parent / target) if member.issym() else str(target)))
                    assert (
                        resolved.parts
                        and resolved.parts[0] in {"build", "runtime", "tt_metal", "ttnn"}
                        and ".." not in resolved.parts
                    )
                if member.isfile():
                    total += member.size
                    if p.as_posix() == "build/tools/umd/topology":
                        digest = hashlib.sha256()
                        stream = archive.extractfile(member)
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                        critical[p.as_posix()] = {"bytes": member.size, "sha256": digest.hexdigest()}
                count += 1
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    assert "build/tools/umd/topology" in critical, "No sealed topology admission tool"
    return {"regular_file_bytes": total, "members": count, "paths_validated": True, "critical_members": critical}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["quiet", "native"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = quiet(json.loads(args.input.read_text())) if args.mode == "quiet" else inspect_native(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if args.mode == "native" or result["quiet"] else 2)
