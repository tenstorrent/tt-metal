#!/usr/bin/env python3
"""Advance LLK sources; rebuild Metal and restart this checkout's JIT server only when the server needs it."""

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


# Paths whose changes alter the server binary or the build outputs it compiles and links with
# (SFPI toolchain, generated linker scripts and link objects). Other changes only need the
# updated sources, which the running server reads from disk per request.
SERVER_PATHS = (
    "tt_metal/tools/jit_compile_server/",
    "tt_metal/impl/jit_server/",
    "tt_metal/jit_build/",
    "tt_metal/sfpi-version",
    "tt_metal/sfpi-info.sh",
    "tt_metal/hw/toolchain/",
    "tt_metal/hw/CMakeLists.txt",
)


def server_paths_changed(repo, old, new):
    return subprocess.run(["git", "diff", "--quiet", old, new, "--", *SERVER_PATHS], cwd=repo).returncode != 0


def is_ancestor(repo, commit, ref):
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, ref], cwd=repo).returncode == 0


def advance_command(repo, branch, had_local_commits):
    """Return the git command that moves HEAD to the freshly fetched origin/<branch>.

    A fast-forward when possible. If the remote branch was rewritten (e.g. llk_helper_library
    rebased onto main) and HEAD held no commits missing from origin before the fetch, nothing
    local can be lost, so HEAD is moved to the new tip. Otherwise the update is refused."""
    target = f"origin/{branch}"
    if is_ancestor(repo, "HEAD", target):
        return ["git", "-c", "submodule.recurse=false", "merge", "--ff-only", target]
    if had_local_commits:
        raise RuntimeError(f"{repo} has local commits that are not on {target}; refusing to discard them")
    print(f"[{repo.name}] {target} was rewritten; moving HEAD to the new tip", flush=True)
    # checkout (unlike reset --hard) refuses to overwrite untracked files.
    return ["git", "-c", "submodule.recurse=false", "checkout", "-q", "-B", branch, target]


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
        help="Build and restart even when no server-relevant path changed (e.g. retry a failed build)",
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
        # tt_ops_code_gen sets `update = none`; it follows the parent's pin only where already cloned.
        code_gen_path = "tt_metal/third_party/tt_ops_code_gen"
        if git(root, "branch", "--show-current") != "llk_helper_library":
            raise RuntimeError("Run from a checkout on branch llk_helper_library")
        pending = Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "llk-server-update-pending"))
        sources_pending = pending.with_name("llk-server-sources-pending")
        published_head = pending.with_name("llk-server-published-head")
        remote_ref = "refs/heads/llk_helper_library"
        remote = git(root, "ls-remote", "--exit-code", "--refs", "origin", remote_ref)
        remote_tip, ref = remote.split()
        if ref != remote_ref:
            raise RuntimeError(f"Unexpected remote reference: {ref}")
        local_tip = git(root, "rev-parse", "HEAD")
        if remote_tip == local_tip and not options.force and not pending.exists() and not sources_pending.exists():
            print("llk_helper_library is unchanged; skipping fetch, build, and restart.", flush=True)
            return
        if remote_tip != local_tip:
            print(f"llk_helper_library changed: {local_tip[:12]} -> {remote_tip[:12]}", flush=True)
        elif pending.exists() or sources_pending.exists():
            print("Retrying an unfinished update.", flush=True)
        # Submodules follow the commits the parent pins.
        sub_paths = git(root, "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$").split()[1::2]
        build_sub_paths = [path for path in sub_paths if path != code_gen_path]
        code_gen_initialized = (root / code_gen_path / ".git").exists()
        initialized_subs = [root / path for path in sub_paths if (root / path / ".git").exists()]
        for repo in (root, *initialized_subs):
            if git(repo, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"):
                raise RuntimeError(f"Tracked local changes in {repo}; commit or stash them first")
        branch = "llk_helper_library"
        # Checked before the fetch, while the pre-rewrite remote commits are still known.
        had_local_commits = git(root, "rev-list", "--count", "HEAD", "--not", "--remotes=origin") != "0"
        # A previous fetch may have replaced the remote history before the update
        # failed. Only trust that earlier publication check for the exact same HEAD;
        # commits added locally since then must still be protected.
        if published_head.exists() and published_head.read_text().strip() == local_tip:
            had_local_commits = False
        if not had_local_commits:
            temporary = published_head.with_suffix(".tmp")
            temporary.write_text(local_tip + "\n")
            temporary.replace(published_head)
        # Submodules are fetched by `submodule update` below.
        run(
            root,
            "git",
            "fetch",
            "--no-recurse-submodules",
            "origin",
            f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
        )
        remote_tip = git(root, "rev-parse", f"origin/{branch}")
        command = advance_command(root, branch, had_local_commits)
        # A rewritten branch (e.g. llk_helper_library rebased onto main) always rebuilds.
        rewritten = not is_ancestor(root, local_tip, remote_tip)
        rebuild = options.force or pending.exists() or rewritten or server_paths_changed(root, local_tip, remote_tip)
        server = find_server(root) if rebuild else None
        try:
            sources_pending.write_text(remote_tip + "\n")
            if rebuild:
                # Remember unfinished work before advancing HEAD, so a failed build
                # is retried on the next timer tick even if the remote is unchanged.
                pending.write_text(remote_tip + "\n")
            run(root, *command)
            if build_sub_paths:
                run(root, "git", "submodule", "update", "--init", "--recursive", "--", *build_sub_paths)
            if code_gen_initialized:
                # --checkout overrides `update = none`.
                run(root, "git", "submodule", "update", "--checkout", "--", code_gen_path)
            if not rebuild:
                sources_pending.unlink()
                published_head.unlink(missing_ok=True)
                print("No server-relevant changes; sources updated without a build or restart.", flush=True)
                return
            run(root, "./build_metal.sh", *build_args)
            restart(root, server)
            pending.unlink()
            sources_pending.unlink()
            published_head.unlink(missing_ok=True)
        finally:
            if server is not None:
                server[4].close()
                server[5].close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
