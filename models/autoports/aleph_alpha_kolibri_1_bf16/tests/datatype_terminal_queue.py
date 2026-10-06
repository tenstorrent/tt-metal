# SPDX-License-Identifier: Apache-2.0
"""Serialize tracked head/cache reconstruction, full trials, and the final router control."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def main():
    package = "models.autoports.aleph_alpha_kolibri_1_bf16.tests"
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
    names = json.loads((root / "configs/terminal_matrix.json").read_text())
    assert len(names) == 20 and all(name.startswith("head_") for name in names)
    for reduced in (True, False):
        label = "terminal_tracked" if reduced else "terminal_full"
        env = dict(os.environ, FULL_ARTIFACT_DIR=str(root))
        env.pop("KOLIBRI_PRECISION_CONFIG", None)
        env["TT_METAL_TRACE_ALLOC_TRACKING"] = "1" if reduced else "0"
        env["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] = "0"
        command = [sys.executable, "-m", package + ".datatype_terminal_sweep"]
        if reduced:
            command.append("--reduced")
        command += ["--configs", *names]
        print("TERMINAL_PHASE_START", label, flush=True)
        completed = subprocess.run([sys.executable, "-m", package + ".optimized_full_run", label, *command], env=env)
        # Retain the complete wrapper log and exact per-candidate excerpts.
        log_path = root / (label + ".log")
        data = log_path.read_bytes()
        parts = re.split(r"(?m)^TERMINAL_CONFIG_START ([^\n]+)\n", data.decode())
        for i in range(1, len(parts), 2):
            target = root / ("terminal_reconfigure_smoke" if reduced else "candidates") / parts[i]
            target.mkdir(parents=True, exist_ok=True)
            (target / "run.log").write_text("TERMINAL_CONFIG_START " + parts[i] + "\n" + parts[i + 1])
            (target / "log_origin.json").write_text(
                json.dumps(
                    dict(
                        parent_log=log_path.name,
                        parent_sha256=hashlib.sha256(data).hexdigest(),
                        excerpt="From this config's START marker up to the next START marker or process end",
                    ),
                    indent=2,
                )
                + "\n"
            )
        print("TERMINAL_PHASE_END", label, completed.returncode, flush=True)
        if completed.returncode:
            sys.exit(completed.returncode)
    subprocess.run([sys.executable, "-m", package + ".datatype_queue", "--configs", "router_bfp8_lofi"], check=True)


if __name__ == "__main__":
    main()
