#!/usr/bin/env python3
"""Owned local copy of exactly twelve sealed HF assets; no device/Docker calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

import owned_seed_copy as owned
import slurm_pair_checks as checks


def seal_digest(seal):
    return hashlib.sha256(json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def layout(seal):
    assert seal["revision"] == checks.SNAPSHOT and set(seal["files"]) == set(checks.SNAPSHOT_ASSETS)
    regular, links = {}, {}
    for name, asset in checks.SNAPSHOT_ASSETS.items():
        row = seal["files"][name]
        assert row["bytes"] == asset["bytes"]
        relative = checks.MODEL_REPO + "/blobs/" + asset["blob"]
        assert row["binding"]["target_relative_to_hub"] == relative
        regular[relative] = name
        snapshot = checks.MODEL_REPO + "/snapshots/" + checks.SNAPSHOT + "/" + name
        repo_blob = checks.MODEL_REPO + "/blobs/" + asset["blob"]
        first = os.path.relpath(repo_blob, str(Path(snapshot).parent))
        assert row["binding"]["snapshot_link"] == first
        links[snapshot] = first
        assert row["binding"]["repo_blob_link"] is None
    assert len(regular) == 12 and len(links) == 12
    return regular, links


def directory_members(hub):
    assert hub.is_dir() and not hub.is_symlink() and hub.stat().st_uid == os.getuid()
    files, directories = set(), set()
    for parent, dirs, names in os.walk(hub, followlinks=False):
        for name in dirs:
            p = Path(parent) / name
            s = p.lstat()
            assert stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid(), "Unowned/linked staging directory"
            directories.add(p.relative_to(hub).as_posix())
        for name in names:
            files.add((Path(parent) / name).relative_to(hub).as_posix())
    return files, directories


def hash_descriptor(path, expected, checkpoint, copied=False):
    before_path = path.lstat()
    assert (
        stat.S_ISREG(before_path.st_mode) and owned.stamp(before_path) == expected["stamp"]
    ), "HF path is not the sealed regular file"
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode) and before.st_size == expected["bytes"]
        if copied:
            assert before.st_uid == os.getuid() and stat.S_IMODE(before.st_mode) == 0o444
        assert owned.stamp(before) == expected["stamp"], "HF descriptor identity changed"
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            checkpoint()
            h.update(block)
        assert owned.stamp(os.fstat(stream.fileno())) == expected["stamp"]
    assert h.hexdigest() == expected["sha256"], "HF bytes changed"
    assert owned.stamp(path.lstat()) == expected["stamp"]


def verify(seal, staged, checkpoint=lambda: None):
    assert staged["complete"] and staged["source_seal_sha256"] == seal_digest(seal)
    hub = Path(staged["hub_root"])
    assert hub.resolve(strict=True) == hub
    regular, links = layout(seal)
    assert set(staged["files"]) == set(regular) and set(staged["links"]) == set(links)
    expected_files = set(regular) | set(links)
    expected_directories = {
        str(parent) for name in expected_files for parent in Path(name).parents if str(parent) != "."
    }
    assert directory_members(hub) == (expected_files, expected_directories), "Unlisted staged asset/directory"
    for relative, name in regular.items():
        checkpoint()
        row = staged["files"][relative]
        assert row["bytes"] == seal["files"][name]["bytes"] and row["sha256"] == seal["files"][name]["sha256"]
        path = hub / relative
        assert path.resolve(strict=True) == path, "Staged regular file replaced by a link"
        hash_descriptor(path, row, checkpoint, copied=True)
    for relative, link in links.items():
        checkpoint()
        path = hub / relative
        s = path.lstat()
        assert stat.S_ISLNK(s.st_mode) and s.st_uid == os.getuid() and os.readlink(path) == link
        assert owned.stamp(s) == staged["links"][relative]["stamp"]
        assert path.resolve(strict=True).is_relative_to(hub), "Staged snapshot link escapes"
    assert staged["logical_bytes"] == seal["logical_bytes"] == sum(r["bytes"] for r in staged["files"].values())
    return {
        "complete": True,
        "assets": 12,
        "logical_bytes": staged["logical_bytes"],
        "source_seal_sha256": staged["source_seal_sha256"],
        "hub_root": str(hub),
    }


def worker(seal, hub, proof, seconds):
    cancelled = False

    def interrupt(signum, frame):
        nonlocal cancelled
        cancelled = True

    assert sys.stdin.readline() == "GO\n", "Owned parent must publish birth before copying"
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    end = time.monotonic() + seconds

    def checkpoint():
        if cancelled or time.monotonic() >= end:
            raise owned.Cancelled("HF copy cancelled/deadline; preserve partial")

    record = {
        "complete": False,
        "worker": owned.identity(os.getpid()),
        "hub_root": str(hub),
        "source_seal_sha256": seal_digest(seal),
        "files": {},
        "links": {},
        "logical_bytes": 0,
    }
    owned.save(proof, record)
    try:
        checkpoint()
        checks.unchanged_snapshot(seal)
        regular, links = layout(seal)
        assert hub.parent.is_dir() and not hub.parent.is_symlink() and hub.parent.stat().st_uid == os.getuid()
        assert not hub.exists() and not hub.is_symlink(), "Staging target must be new"
        hub.mkdir(mode=0o755)
        for relative, name in regular.items():
            checkpoint()
            expected = seal["files"][name]
            source = Path(seal["hub_root"]) / relative
            assert source.resolve(strict=True) == source
            target = hub / relative
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            src = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(src, "rb") as stream, target.open("xb") as out:
                s = os.fstat(stream.fileno())
                assert stat.S_ISREG(s.st_mode) and owned.stamp(s) == expected["stamp"]
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(4 * 1024**2), b""):
                    checkpoint()
                    h.update(block)
                    out.write(block)
                assert owned.stamp(os.fstat(stream.fileno())) == expected["stamp"]
                assert h.hexdigest() == expected["sha256"] and out.tell() == expected["bytes"]
                out.flush()
                os.fchmod(out.fileno(), 0o444)
            assert owned.stamp(source.lstat()) == expected["stamp"]
            copied = {"bytes": expected["bytes"], "sha256": expected["sha256"], "stamp": owned.stamp(target.stat())}
            hash_descriptor(target, copied, checkpoint, copied=True)
            record["files"][relative] = copied
            record["logical_bytes"] += copied["bytes"]
            owned.save(proof, record)
        for relative, link in links.items():
            checkpoint()
            target = hub / relative
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            target.symlink_to(link)
            record["links"][relative] = {"link": link, "stamp": owned.stamp(target.lstat())}
        checks.unchanged_snapshot(seal)
        record["complete"] = True
        verify(seal, record, checkpoint)
    except BaseException:
        record["complete"] = False
        raise
    finally:
        try:
            owned.save(proof, record)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def mounts(seal, staged, owned_hub, checkpoint=lambda: None):
    verify(seal, staged, checkpoint)
    original = checks.snapshot_mounts(seal, owned_hub)
    hub = Path(staged["hub_root"])
    result = [original[0]]
    for source, target, mode in original[1:]:
        assert mode == "ro"
        relative = source.relative_to(Path(seal["hub_root"]))
        local = hub / relative
        assert local.resolve(strict=True).is_relative_to(hub)
        result.append((local, target, mode))
    assert len(result) == 3 and [r[1:] for r in result] == [r[1:] for r in original]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true", required=True)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--hub", type=Path, required=True)
    p.add_argument("--proof", type=Path, required=True)
    p.add_argument("--seconds", type=float, required=True)
    a = p.parse_args()
    worker(json.loads(a.seal.read_text()), a.hub, a.proof, a.seconds)
