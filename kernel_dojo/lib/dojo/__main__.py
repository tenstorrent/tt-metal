# SPDX-License-Identifier: Apache-2.0
"""Entry point.

Wraps the CLI so that the dojo's own output is fully flushed before interpreter
finalization, and so that finalization's stderr noise is discarded.
"""

import os
import sys

from .cli import main


def _run() -> int:
    """Invoke the CLI, normalising every exit path to an integer code."""
    try:
        return main() or 0
    except SystemExit as exc:
        # argparse exits with an int (having already printed its message); the
        # runner raises SystemExit("some message") for bad user input.
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 1
        return exc.code or 0
    except KeyboardInterrupt:
        # Realistic: interrupting a long benchmark. Don't dump a traceback.
        print("\ninterrupted", file=sys.stderr)
        return 130


code = _run()

# ttnn leaks one module-level CoreRangeSet on import. That makes nanobind's leak
# checker dump a few hundred lines of "leaked type" / "leaked function" to
# stderr from a Py_AtExit handler, long after the dojo has finished printing. It
# is harmless, it is not caused by anything the dojo does (a bare `import ttnn`
# reproduces it), and nanobind's set_leak_warnings() switch is C++-only so it
# cannot be reached from Python.
#
# So: flush our own output, then point the underlying stderr file descriptor at
# /dev/null. Interpreter finalization still runs normally — device teardown and
# C++ destructors are unaffected — but anything it prints is discarded.
#
# Set DOJO_SHOW_SHUTDOWN_NOISE=1 to keep it, if you are debugging shutdown.
sys.stdout.flush()
sys.stderr.flush()

if not os.environ.get("DOJO_SHOW_SHUTDOWN_NOISE"):
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

sys.exit(code)
