# SPDX-License-Identifier: Apache-2.0
"""Serial stage command logger. Each invocation runs exactly one device job."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    label, *command = sys.argv[1:]
    out = Path(os.environ["FULL_ARTIFACT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    record = dict(
        label=label,
        command=command,
        start=time.time(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        environment={k: v for k, v in os.environ.items() if k.startswith(("TT_METAL_", "FULL_", "KOLIBRI_"))},
    )
    with (out / f"{label}.log").open("w") as log:
        run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=run.returncode, end=time.time())
    (out / f"{label}.command.json").write_text(json.dumps(record, indent=2) + "\n")
    print(label, run.returncode, flush=True)
    sys.exit(run.returncode)


if __name__ == "__main__":
    main()
