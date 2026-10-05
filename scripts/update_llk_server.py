#!/usr/bin/env python3
"""Fast-forward LLK sources, build Metal, then restart this checkout's JIT server."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time


def run(root, *args, capture=False):
    print(f"[{root.name}] {shlex.join(args)}", flush=True)
    return subprocess.run(args, cwd=root, check=True, text=True, stdout=subprocess.PIPE if capture else None).stdout


def git(root, *args):
    return run(root, "git", *args, capture=True).strip()


def validate_build_args(build_args):
    # build_metal.sh uses GNU getopt, which accepts abbreviated long options and
    # grouped short options. Reject those forms too, before fetching or building.
    forbidden = ("--clean", "--configure-only", "--build-packages", "--help")
    for arg in build_args:
        option = arg.split("=", 1)[0]
        if (option.startswith("--") and option != "--" and any(flag.startswith(option) for flag in forbidden)) or (
            option.startswith("-") and not option.startswith("--") and "h" in option
        ):
            raise ValueError(
                "clean, configure-only, build-packages and help options are incompatible with a live server rebuild"
            )


def find_server(root):
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [os.fsdecode(arg) for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg]
            if not args or Path(args[0]).name != "jit_compile_server":
                continue
            cwd = (proc / "cwd").resolve()
            executable = Path(args[0])
            if not executable.is_absolute():
                executable = cwd / executable
            if not executable.resolve().is_relative_to(root):
                continue
            matches.append((proc, args, cwd))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"Expected one JIT server for {root}; found {len(matches)}")
    proc, args, cwd = matches[0]
    env = dict(item.split(b"=", 1) for item in (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
    # Hold the actual log destinations open across the build and restart.
    stdout = open(proc / "fd/1", "ab", buffering=0)
    stderr = open(proc / "fd/2", "ab", buffering=0)
    identity = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    return proc, args, cwd, env, stdout, stderr, identity


def alive(proc, identity):
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == identity
    except FileNotFoundError:
        return False


def restart(root, server):
    proc, args, cwd, env, stdout, stderr, identity = server
    executable = root / "build/tools/jit_compile_server"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"Built executable missing: {executable}; old server was not stopped")
    endpoint = os.fsdecode(env.get(b"TT_METAL_JIT_SERVER_ENDPOINT", b"localhost:9876"))
    host, port = endpoint.rsplit(":", 1)
    host = host.strip("[]")
    host = {"0.0.0.0": "127.0.0.1", "::": "::1", "*": "127.0.0.1"}.get(host, host)
    port = int(port)
    if not alive(proc, identity):
        raise RuntimeError("Original server exited during the build; refusing to restart a different process")
    print(f"Stopping server PID {proc.name}", flush=True)
    os.kill(int(proc.name), signal.SIGTERM)
    deadline = time.monotonic() + 30
    while alive(proc, identity):
        if time.monotonic() >= deadline:
            raise RuntimeError("Server did not stop within 30 seconds; no replacement started")
        time.sleep(0.1)
    args[0] = str(executable.resolve())
    child = subprocess.Popen(
        args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True
    )
    print(f"Started server PID {child.pid}", flush=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Server exited with status {child.returncode}; check the existing server log")
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass
            time.sleep(1)
            if child.poll() is not None:
                raise RuntimeError("Server exited after opening its port; check the existing server log")
            print(f"Server ready at {endpoint} (PID {child.pid})", flush=True)
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server PID {child.pid} started but did not become reachable; check its log")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Pass build_metal.sh options after --. Requires an existing running server."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Update/build/restart even when the remote branch is unchanged (e.g. retry a failed build)",
    )
    parser.add_argument("build_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    build_args = options.build_args
    if build_args[:1] == ["--"]:
        build_args = build_args[1:]
    validate_build_args(build_args)
    root = Path(__file__).resolve().parents[1]
    lock_path = Path(tempfile.gettempdir()) / (
        "update-llk-server-" + hashlib.sha256(os.fsencode(root)).hexdigest()[:16] + ".lock"
    )
    with open(lock_path, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another update is already running for this checkout") from None
        sub = root / "tt_metal/third_party/tt_ops_code_gen"
        if git(root, "branch", "--show-current") != "llk_helper_library":
            raise RuntimeError("Run from a checkout on branch llk_helper_library")
        pending = Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "llk-server-update-pending"))
        remote_ref = "refs/heads/llk_helper_library"
        remote = git(root, "ls-remote", "--exit-code", "--refs", "origin", remote_ref)
        remote_tip, ref = remote.split()
        if ref != remote_ref:
            raise RuntimeError(f"Unexpected remote reference: {ref}")
        local_tip = git(root, "rev-parse", "HEAD")
        if remote_tip == local_tip and not options.force and not pending.exists():
            print("llk_helper_library is unchanged; skipping fetch, build, and restart.", flush=True)
            return
        if remote_tip != local_tip:
            print(f"llk_helper_library changed: {local_tip[:12]} -> {remote_tip[:12]}", flush=True)
        elif pending.exists():
            print("Retrying an unfinished update.", flush=True)
        if not (sub / ".git").exists():
            raise RuntimeError(f"Submodule is not initialized: {sub}")
        if git(sub, "branch", "--show-current") not in ("", "main"):
            raise RuntimeError("tt_ops_code_gen must be on main or a detached HEAD")
        for repo in (root, sub):
            if git(repo, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"):
                raise RuntimeError(f"Tracked local changes in {repo}; commit or stash them first")
        server = find_server(root)
        try:
            for repo, branch in ((root, "llk_helper_library"), (sub, "main")):
                run(repo, "git", "fetch", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
                run(repo, "git", "merge-base", "--is-ancestor", "HEAD", f"origin/{branch}")
            # Remember unfinished work before advancing HEAD, so a failed build
            # is retried on the next timer tick even if the remote is unchanged.
            pending.write_text(remote_tip + "\n")
            for repo, branch in ((root, "llk_helper_library"), (sub, "main")):
                run(repo, "git", "-c", "submodule.recurse=false", "merge", "--ff-only", f"origin/{branch}")
            run(root, "./build_metal.sh", *build_args)
            restart(root, server)
            pending.unlink()
        finally:
            server[4].close()
            server[5].close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
