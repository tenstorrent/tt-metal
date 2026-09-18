"""Keep the preflight, supervisor and owner on exactly one allowed CPU."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    allowed = os.sched_getaffinity(0)
    if not allowed:
        raise RuntimeError("No assigned CPU")
    os.sched_setaffinity(0, {min(allowed)})
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("CPU affinity was not narrowed")
    here = Path(__file__).parent
    subprocess.run([sys.executable, "-I", "-S", "-B", str(here / "node-preflight.py"), *sys.argv[1:]], check=True)
    owner = here
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-B",
            str(owner / "supervise_owner.py"),
            "--plan",
            sys.argv[1],
            "--plan-sha256",
            sys.argv[2],
            "--role",
            sys.argv[3],
        ],
    )


if __name__ == "__main__":
    main()
