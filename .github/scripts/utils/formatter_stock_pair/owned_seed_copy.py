#!/usr/bin/env python3
"""Local preparation module. No scheduler, Docker, device or model operations.

The parent publishes worker PID/birth/directory before releasing a stdin gate.
Cancellation closes and reaps only that child. Incomplete copies are retained;
this module never removes an operation directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


class Cancelled(RuntimeError):
    pass


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "birth_ticks": int(fields[19]), "state": fields[0]}
    except FileNotFoundError:
        return None


def same_process(saved):
    now = identity(saved["pid"])
    return now and now["birth_ticks"] == saved["birth_ticks"]


def close_owned(process, saved, seconds=60):
    """Reap the direct child; a reused PID is never signalled."""
    if process.poll() is None:
        assert same_process(saved), "Owned child identity disappeared or changed"
        os.kill(saved["pid"], signal.SIGTERM)
    escalated = False
    try:
        code = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        assert same_process(saved), "Refuse escalation against an unverified PID"
        escalated = True
        os.kill(saved["pid"], signal.SIGKILL)
        code = process.wait(timeout=10)
    assert not same_process(saved), "Owned copy is not reaped"
    return {"reaped": True, "exit_code": code, "escalated": escalated}


def owned_command(command, receipt, operation_dir, seconds, cancel, ordinary_close_seconds=60):
    """Command must wait for GO on stdin before doing work and spawn no children."""
    receipt = Path(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "controller": identity(os.getpid()),
        "operation_dir": str(Path(operation_dir).resolve()),
        "released": False,
        "closed": False,
        "cleanup_allowed": False,
    }
    assert record["controller"]
    process = None
    failure = None
    with receipt.with_suffix(".log").open("x") as log:
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, text=True
            )
            record["worker"] = identity(process.pid)
            assert record["worker"] and same_process(record["worker"])
            # The birth/operation receipt exists BEFORE any seed read/hash/copy.
            save(receipt, record)
            if cancel():
                raise Cancelled("Cancelled before releasing copy gate")
            process.stdin.write("GO\n")
            process.stdin.flush()
            process.stdin.close()
            record["released"] = True
            save(receipt, record)
            deadline = time.monotonic() + seconds
            while process.poll() is None:
                if cancel():
                    raise Cancelled("Owned operation cancelled")
                if time.monotonic() >= deadline:
                    raise Cancelled("Owned operation deadline")
                time.sleep(0.05)
            if process.returncode != 0:
                raise RuntimeError("Seed worker did not complete")
        except BaseException as error:
            failure = error
        finally:
            if process is not None and process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process is not None and record.get("worker"):
                record["close"] = close_owned(process, record["worker"], ordinary_close_seconds)
                record["closed"] = record["close"]["reaped"]
                record["cleanup_allowed"] = record["closed"] and not record["close"]["escalated"]
            if failure:
                record["incomplete"] = type(failure).__name__
            save(receipt, record)
    if record.get("close", {}).get("escalated"):
        raise RuntimeError("Escalated owned-copy closure; no candidate or cleanup")
    if failure:
        raise failure
    return record


def stamp(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def payload_link(path, source):
    """Only pinned UMD's tracked generated-metadata link may be dangling."""
    link = os.readlink(path)
    assert not Path(link).is_absolute() and path.resolve(strict=False).is_relative_to(
        source.resolve()
    ), "Payload link escapes its capsule"
    if path.exists():
        return link
    assert (
        tuple(path.parts[-4:]) == ("tt_metal", "third_party", "umd", "compile_commands.json")
        and link == "build/compile_commands.json"
    ), "Unreviewed dangling payload link"
    argv = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-C", str(path.parent)]

    def read(*args):
        return subprocess.check_output(argv + list(args), text=True, timeout=10).strip()

    assert read("rev-parse", "HEAD") == "538094128060917a2c181436ab59b6a32be51efc", "Unpinned UMD link"
    common = Path(read("rev-parse", "--git-common-dir"))
    common = common if common.is_absolute() else path.parent / common
    assert common.resolve().is_relative_to(source.resolve()), "External UMD Git storage"
    encoded = link.encode()
    blob = hashlib.sha1(b"blob " + str(len(encoded)).encode() + b"\0" + encoded).hexdigest()
    assert (
        read("ls-files", "--stage", "--", path.name) == "120000 " + blob + " 0\t" + path.name
    ), "Untracked or changed UMD link"
    return link


def inventory(source, payload=False):
    assert source.is_dir() and not source.is_symlink(), "Missing regular seed root"
    files = {}
    for root, directories, names in os.walk(source, followlinks=False):
        for name in directories + names:
            path = Path(root) / name
            info = path.lstat()
            if payload and stat.S_ISLNK(info.st_mode):
                link = payload_link(path, source)
                files[path.relative_to(source).as_posix()] = {"link": link, "stamp": stamp(info)}
                continue
            assert stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), "Cache link/special file"
            if stat.S_ISREG(info.st_mode):
                files[path.relative_to(source).as_posix()] = stamp(info)
    assert files and (payload or any(name.endswith(".tensorbin") for name in files)), "No existing tensor seed"
    return dict(sorted(files.items()))


def payload_directories(source):
    """Preserve real owned directories, including empty packed-ref Git refs."""
    assert source.stat().st_uid == os.getuid(), "Unowned payload root"
    result = {}
    for root, directories, names in os.walk(source, followlinks=False):
        for name in directories:
            path = Path(root) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                payload_link(path, source)  # Existing contained-link policy remains.
                continue
            assert stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(), "Unowned/special payload directory"
            result[path.relative_to(source).as_posix()] = stamp(info)
    return dict(sorted(result.items()))


def worker(source, destinations, proof, seconds, payload=False, expected=None):
    """Chunked synchronous copy allows ordinary cancellation without descendants."""
    cancelled = False

    def interrupt(signum, frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    assert sys.stdin.readline() == "GO\n", "Parent did not publish/release copy ownership"
    started = time.monotonic()

    def checkpoint():
        if cancelled or time.monotonic() - started >= seconds:
            raise Cancelled("Copy cancelled/deadline; preserve partial owned directories")

    record = {
        "worker": identity(os.getpid()),
        "complete": False,
        "files": [],
        "destinations": [str(p) for p in destinations],
    }
    save(proof, record)
    try:
        checkpoint()
        assert len(destinations) == (1 if payload else 2) and len(set(destinations)) == len(
            destinations
        ), "Wrong independent destination count"
        before = inventory(source, payload)
        directories = payload_directories(source) if payload else {}
        if expected is not None:
            assert set(before) == set(expected), "Source membership differs from reviewed seal"
        for destination in destinations:
            assert not destination.exists() and not destination.is_symlink(), "Destination must be new"
            destination.mkdir(parents=True, exist_ok=False)
            for relative in directories:
                checkpoint()
                (destination / relative).mkdir(parents=True, exist_ok=True)
        record["payload_directories"] = sorted(directories)
        record["seed_bytes"] = sum(row[2] for row in before.values() if not isinstance(row, dict))
        for name, sealed in before.items():
            checkpoint()
            path = source / name
            if isinstance(sealed, dict):
                assert payload and (expected is None or expected[name].get("link") == sealed["link"])
                for destination in destinations:
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(sealed["link"])
                record["files"].append({"path": name, "link": sealed["link"]})
                continue
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                assert stamp(os.fstat(stream.fileno())) == sealed, "Seed changed before copy"
                digest = hashlib.sha256()
                targets = []
                try:
                    for destination in destinations:
                        target = destination / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        targets.append(target.open("xb"))
                    while True:
                        checkpoint()
                        block = stream.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        for target in targets:
                            target.write(block)
                    assert stamp(os.fstat(stream.fileno())) == sealed, "Seed changed during copy"
                    if payload:
                        executable = os.fstat(stream.fileno()).st_mode & 0o111
                        for target in targets:
                            os.fchmod(target.fileno(), (os.fstat(target.fileno()).st_mode & 0o777) | executable)
                finally:
                    for target in targets:
                        target.close()
            digest_value = digest.hexdigest()
            assert expected is None or (
                expected[name]["sha256"] == digest_value and expected[name]["bytes"] == sealed[2]
            ), "Source bytes differ from reviewed seal"
            for destination in destinations:
                actual = hashlib.sha256()
                with (destination / name).open("rb") as stream:
                    while True:
                        checkpoint()
                        block = stream.read(1024 * 1024)
                        if not block:
                            break
                        actual.update(block)
                assert actual.hexdigest() == digest_value, "Copied seed bytes differ"
            record["files"].append({"path": name, "bytes": sealed[2], "sha256": digest_value})
            if len(record["files"]) % 50 == 0:
                save(proof, record)
        assert inventory(source, payload) == before, "Seed membership/identity changed"
        assert not payload or payload_directories(source) == directories, "Payload directories changed during copy"
        record["complete"] = True
    finally:
        save(proof, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, action="append", required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--payload", action="store_true")
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    worker(
        args.source,
        args.destination,
        args.proof,
        args.seconds,
        args.payload,
        json.loads(args.expected.read_text()) if args.expected else None,
    )
