#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Keep Copilot's native launcher off Actions' potentially nonblocking log pipes.

The runtime writes to a private, unlinked temporary file. This process relays
its bytes to Actions, retrying temporary pipe backpressure instead of letting
the runtime panic. It does not inspect/filter output or change validation tools.
"""

import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def install():
    directory = Path(os.environ["RUNNER_TEMP"]) / "tt-copilot-stdio"
    directory.mkdir(mode=0o700, exist_ok=True)
    relay = directory / "relay.py"
    startup = directory / "bash-env.sh"
    previous = os.environ.get("BASH_ENV", "")
    shutil.copyfile(__file__, relay)
    if previous != str(startup):
        # Only the generated Processing Request step has this runtime marker
        # after user setup. Ordinary Actions steps and nested shells keep their
        # normal behavior. Re-execute the same script with its generated flags.
        content = f"""# Installed by copilot-runtime-stdio.py
if [ -n "${{COPILOT_AGENT_RUNTIME_VERSION:-}}" ] &&
   [ "${{TT_COPILOT_STDIO_SPOOLED:-}}" != 1 ] && [ -f "$0" ]; then
  export TT_COPILOT_STDIO_SPOOLED=1
  exec {shlex.quote(sys.executable)} {shlex.quote(str(relay))} -- \\
    "$BASH" --noprofile --norc -e -o pipefail "$0" "$@"
fi
"""
        if previous:
            content += f"if [ -f {shlex.quote(previous)} ]; then . {shlex.quote(previous)}; fi\n"
        startup.write_text(content)
    with open(os.environ["GITHUB_ENV"], "a") as env_file:
        env_file.write(f"BASH_ENV={startup}\n")
    print("Installed file-backed output relay for Copilot runtime steps.")


def relay_command(command):
    # Merge streams at the writer, preserving workflow-command suppression
    # markers relative to stderr diagnostics. Separate relays could emit the
    # closing marker before queued stderr. pread does not move the write cursor.
    with tempfile.TemporaryFile(dir=os.environ.get("RUNNER_TEMP")) as spool:
        child = subprocess.Popen(command, stdout=spool, stderr=subprocess.STDOUT, start_new_session=True)
        cancelled = []

        def forward_signal(signum, _frame):
            if not cancelled:
                cancelled.extend([signum, time.monotonic()])
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

        signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        previous_handlers = {sig: signal.signal(sig, forward_signal) for sig in signals}
        blocking = os.get_blocking(1)
        offset = 0
        pending = b""
        try:
            os.set_blocking(1, False)
            while True:
                exited = child.poll() is not None
                if not pending:
                    pending = os.pread(spool.fileno(), 65536, offset)
                if exited and not pending:
                    break
                if cancelled and time.monotonic() - cancelled[1] > 5:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    # Cancellation must finish even if Actions stopped reading
                    # its pipes. Any remaining output is discarded only here.
                    break
                _, writable, _ = select.select([], [1] if pending else [], [], 0.05)
                if writable:
                    try:
                        count = os.write(1, pending)
                    except (BlockingIOError, InterruptedError):
                        continue
                    offset += count
                    pending = pending[count:]
            result = child.wait()
            if cancelled:
                return 128 + cancelled[0]
            return result if result >= 0 else 128 - result
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            os.set_blocking(1, blocking)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    if sys.argv[1:] == ["--install"]:
        install()
    elif len(sys.argv) > 2 and sys.argv[1] == "--":
        sys.exit(relay_command(sys.argv[2:]))
    else:
        sys.exit("usage: copilot-runtime-stdio.py --install | -- COMMAND [ARG ...]")
