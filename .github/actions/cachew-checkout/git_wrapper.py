# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Try downstream Git reads once through cachew, then disable the job's rewrite."""

import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile


OPTIONS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--super-prefix"}


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


def upstream_environment(environ, session):
    return dict(environ, GIT_CONFIG_GLOBAL=str(session / "upstream.config"))


def disable_cache(session):
    # The marker also stops concurrent/future wrappers while the atomic rewrite
    # removal is in progress. Never change the caller's original global config.
    (session / "cache-disabled").touch(mode=0o600, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=session, prefix="upstream-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with (session / "upstream.config").open("rb") as source:
                shutil.copyfileobj(source, stream)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(session / "global.config")
        finally:
            temporary.unlink(missing_ok=True)


def direct_network_operation(real_git, arguments, index, environ, session):
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
            env=upstream_environment(environ, session),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        path = result.stdout.decode(errors="replace").strip()
        if result.returncode or not Path(path).is_absolute() or Path(path).is_dir():
            return True
    return False


def cache_attempt(real_git, arguments, environ, timeout, index):
    def terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            # Let clone remove its partially created destination before fallback.
            process.wait(timeout=1)
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
        *arguments[index:],
    ]
    cache_env = dict(environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false")
    with subprocess.Popen(command, env=cache_env, start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            return 124
        except BaseException:
            if process.poll() is None:
                terminate(process)
            raise


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
    if not state.get("active"):
        return subprocess.call([real_git, *arguments], env=environ)
    if (session / "cache-disabled").exists():
        return subprocess.call([real_git, *arguments], env=upstream_environment(environ, session))
    if arguments[index] in (
        "clone",
        "fetch",
        "ls-remote",
        "push",
        "pull",
        "submodule",
        "remote",
    ) and direct_network_operation(real_git, arguments, index, environ, session):
        return subprocess.call([real_git, *arguments], env=upstream_environment(environ, session))
    if arguments[index] not in ("clone", "fetch", "ls-remote"):
        return subprocess.call([real_git, *arguments], env=environ)
    try:
        timeout = float(environ.get("CACHEW_GIT_FETCH_TIMEOUT", "120"))
        if not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError
    except ValueError:
        timeout = 120
    result = cache_attempt(real_git, arguments, environ, timeout, index)
    if result == 0 or result < 0:
        return result
    disable_cache(session)
    print("Cachew Git read failed; removed the job's cache rewrite and retrying upstream once.", file=sys.stderr)
    return subprocess.call([real_git, *arguments], env=upstream_environment(environ, session))


def main():
    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    result = run(sys.argv[1:], dict(os.environ))
    return 128 - result if result < 0 else result


if __name__ == "__main__":
    sys.exit(main())
