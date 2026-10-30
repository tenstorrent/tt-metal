# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Try downstream Git reads once through cachew, falling back to upstream.

Any failed cache read is retried once against upstream. Only failures of the
cache endpoint itself (connection, TLS, 3xx/5xx, stall, deadline) remove the
job's cache rewrite; answers about one repository or ref (401/403/404, missing
ref, `ls-remote --exit-code` finding nothing) keep caching on for later reads.
"""

import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefetch import TRANSPORT_ERRORS  # noqa: E402


OPTIONS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--super-prefix"}
CLONE_OPTIONS_WITH_VALUE = {
    "-o",
    "--origin",
    "-b",
    "--branch",
    "-u",
    "--upload-pack",
    "--reference",
    "--reference-if-able",
    "--separate-git-dir",
    "--depth",
    "--shallow-since",
    "--shallow-exclude",
    "-c",
    "--config",
    "--server-option",
    "-j",
    "--jobs",
    "--filter",
    "--template",
    "--bundle-uri",
    "--ref-format",
}
HARD_DEADLINE_SECONDS = 3600
TERMINATE_GRACE_SECONDS = 10
STDERR_TAIL_BYTES = 65536


def command_index(arguments):
    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if arg in OPTIONS_WITH_VALUE:
            index += 2
        elif arg.startswith("-"):
            index += 1
        else:
            return index
    return None


def config_environment(environ):
    return {key: value for key, value in environ.items() if not key.startswith(("GIT_CONFIG_", "GIT_TRACE"))}


def remove_managed(real_git, config, state, environ, rewrites_only=False):
    keys = dict.fromkeys(key for key, _ in state.get("managed", []))
    for key in keys:
        if rewrites_only and not key.lower().endswith(".insteadof"):
            continue
        # `git config` takes the file's lock, so concurrent user writes are safe.
        subprocess.run(
            [real_git, "config", "--file", str(config), "--unset-all", key],
            env=config_environment(environ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def disable_cache(real_git, session, state, environ):
    # The marker stops concurrent and later wrappers first. Only cachew's own
    # keys are removed, so settings written by the job are never lost.
    (session / "cache-disabled").touch(mode=0o600, exist_ok=True)
    remove_managed(real_git, session / "global.config", state, environ)
    state = dict(state, active=False, managed=[])
    with tempfile.NamedTemporaryFile("w", dir=session, prefix="state-", delete=False) as stream:
        json.dump(state, stream)
    os.chmod(stream.name, 0o600)
    os.replace(stream.name, session / "state.json")


class UpstreamView:
    """A private copy of the job's global configuration without the cache rewrite."""

    def __init__(self, real_git, environ, session, state):
        self.path = None
        self.environ = environ
        if (session / "cache-disabled").exists():
            return
        descriptor, name = tempfile.mkstemp(dir=session, prefix="upstream-view-", suffix=".config")
        self.path = Path(name)
        with os.fdopen(descriptor, "wb") as target, (session / "global.config").open("rb") as source:
            shutil.copyfileobj(source, target)
        remove_managed(real_git, self.path, state, environ, rewrites_only=True)
        self.environ = dict(environ, GIT_CONFIG_GLOBAL=str(self.path))

    def __enter__(self):
        return self.environ

    def __exit__(self, *_):
        if self.path:
            self.path.unlink(missing_ok=True)


def direct_network_operation(real_git, arguments, index, environ):
    """Avoid Git's internal recursive clone retries and keep pushes upstream."""
    command = arguments[index]
    tail = arguments[index + 1 :]
    if command in ("push", "pull", "submodule") or (command == "remote" and tail and tail[0] in ("update", "prune")):
        return True
    recursive = [arg for arg in tail if arg.startswith(("--recursive", "--recurse-submodules"))]
    if (command == "clone" and recursive) or any(
        arg.split("=", 1)[-1].lower() not in ("false", "no", "0") for arg in recursive
    ):
        return True
    if any(arg.lower() in ("submodule.recurse=true", "fetch.recursesubmodules=true") for arg in arguments):
        return True
    if command == "fetch" and not any(arg in ("--no-recurse-submodules", "--recurse-submodules=false") for arg in tail):
        # An ordinary fetch may recurse implicitly into initialized submodules.
        # Let upstream handle that operation; the checkout action seeds its
        # requested recursive submodules through cachew with explicit fetches.
        result = subprocess.run(
            [real_git, *arguments[:index], "rev-parse", "--path-format=absolute", "--git-path", "modules"],
            env=environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        path = result.stdout.decode(errors="replace").strip()
        if result.returncode or not Path(path).is_absolute() or Path(path).is_dir():
            return True
    return False


def working_directory(arguments, index):
    directory = Path.cwd()
    position = 0
    while position < index:
        if arguments[position] == "-C":
            directory = directory / arguments[position + 1]
            position += 2
        elif arguments[position] in OPTIONS_WITH_VALUE:
            position += 2
        else:
            position += 1
    return directory


def clone_destination(arguments, index):
    """Where `git clone` will create its repository, or None if unknown."""
    positional = []
    position = index + 1
    while position < len(arguments):
        arg = arguments[position]
        if arg == "--":
            positional.extend(arguments[position + 1 :])
            break
        if arg in CLONE_OPTIONS_WITH_VALUE:
            position += 2
            continue
        if not arg.startswith("-"):
            positional.append(arg)
        position += 1
    if not positional:
        return None
    if len(positional) > 1:
        name = positional[1]
    else:
        name = positional[0].rstrip("/")
        name = name.removesuffix("/.git").removesuffix(".git")
        name = name.replace(":", "/").rsplit("/", 1)[-1]
        if not name:
            return None
    return working_directory(arguments, index) / name


class CloneTarget:
    """Remove what a failed cache clone left behind before the upstream retry."""

    def __init__(self, arguments, index):
        self.path = clone_destination(arguments, index) if arguments[index] == "clone" else None
        self.existed = bool(self.path) and self.path.exists()
        self.was_empty = self.existed and self.path.is_dir() and not any(self.path.iterdir())

    def reset(self):
        if not self.path or not self.path.exists() or self.path.is_symlink():
            return
        if not self.existed:
            shutil.rmtree(self.path, ignore_errors=True)
        elif self.was_empty:
            for child in self.path.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)


def seconds(environ, name, default, maximum):
    try:
        value = float(environ.get(name, default))
        if not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError
        return value
    except ValueError:
        return float(default)


def cache_attempt(real_git, arguments, environ, index):
    """Run once through the cache. Returns (exit status, stderr tail)."""
    stall = seconds(environ, "CACHEW_GIT_FETCH_TIMEOUT", 120, 600)
    deadline = seconds(environ, "CACHEW_HARD_DEADLINE", HARD_DEADLINE_SECONDS, 86400)

    def terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            # Let clone remove its partially created destination before fallback.
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    command = [
        real_git,
        *arguments[:index],
        "-c",
        "http.followRedirects=false",
        "-c",
        "credential.helper=",
        # Abandon a stalled transfer, never a slow but progressing one.
        "-c",
        "http.lowSpeedLimit=1000",
        "-c",
        f"http.lowSpeedTime={max(1, math.ceil(stall))}",
        *arguments[index:],
    ]
    cache_env = dict(environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false")
    tail = bytearray()

    def relay(stream):
        output = sys.stderr.buffer
        for chunk in iter(lambda: stream.read1(8192), b""):
            output.write(chunk)
            output.flush()
            tail.extend(chunk)
            del tail[:-STDERR_TAIL_BYTES]

    with subprocess.Popen(command, env=cache_env, stderr=subprocess.PIPE, start_new_session=True) as process:
        reader = threading.Thread(target=relay, args=(process.stderr,), daemon=True)
        reader.start()
        try:
            status = process.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            terminate(process)
            status = 124
        except BaseException:
            if process.poll() is None:
                terminate(process)
            raise
        reader.join(timeout=5)
    return status, bytes(tail)


def run(arguments, environ):
    real_git = environ["CACHEW_REAL_GIT"]
    session = Path(environ["CACHEW_SESSION_DIR"])
    index = command_index(arguments)
    if index is None:
        return subprocess.call([real_git, *arguments], env=environ)
    owned_config = Path(environ.get("GIT_CONFIG_GLOBAL", "")).resolve() == (session / "global.config").resolve()
    if not owned_config:
        return subprocess.call([real_git, *arguments], env=environ)
    try:
        state = json.loads((session / "state.json").read_text())
    except (OSError, ValueError):
        # A missing session must not prevent Git from running after post cleanup.
        return subprocess.call([real_git, *arguments], env=environ)
    if not state.get("active") or (session / "cache-disabled").exists():
        return subprocess.call([real_git, *arguments], env=environ)
    command = arguments[index]
    if command in ("clone", "fetch", "ls-remote", "push", "pull", "submodule", "remote") and direct_network_operation(
        real_git, arguments, index, environ
    ):
        with UpstreamView(real_git, environ, session, state) as upstream:
            return subprocess.call([real_git, *arguments], env=upstream)
    if command not in ("clone", "fetch", "ls-remote"):
        return subprocess.call([real_git, *arguments], env=environ)
    target = CloneTarget(arguments, index)
    result, errors = cache_attempt(real_git, arguments, environ, index)
    if result == 0 or result < 0:
        return result
    target.reset()
    if result == 124 or TRANSPORT_ERRORS.search(errors):
        disable_cache(real_git, session, state, environ)
        print("Cachew is unavailable; removed the job's cache rewrite and retrying upstream once.", file=sys.stderr)
        return subprocess.call([real_git, *arguments], env=environ)
    print("Cachew read did not succeed; retrying this operation upstream once.", file=sys.stderr)
    with UpstreamView(real_git, environ, session, state) as upstream:
        return subprocess.call([real_git, *arguments], env=upstream)


def main():
    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    result = run(sys.argv[1:], dict(os.environ))
    return 128 - result if result < 0 else result


if __name__ == "__main__":
    sys.exit(main())
