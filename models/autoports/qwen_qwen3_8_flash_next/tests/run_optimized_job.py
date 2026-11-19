# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized bounded chip0 jobs, with exact commands and logs on disk."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--seconds", type=int, default=1800)
    p.add_argument("--tracked", action="store_true")
    p.add_argument("--watcher", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = Path("models/autoports/qwen_qwen3_8_flash_next/doc/optimized_decoder")
    holder = subprocess.run(["fuser", "/dev/tenstorrent/0"], capture_output=True, text=True)
    assert holder.returncode == 1 and not holder.stdout.strip(), holder.stdout
    env = dict(
        os.environ,
        TT_VISIBLE_DEVICES="0",
        TT_MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto",
        TT_METAL_TRACE_ALLOC_TRACKING="1" if a.tracked or a.watcher else "0",
        TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
        TT_METAL_TRACE_ALLOC_TRACEBACKS="0",
    )
    if a.watcher:
        env["TT_METAL_WATCHER"] = "10"
        env["TT_METAL_LOGS_PATH"] = str((root / "watcher" / a.name).resolve())
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_8_flash_next.tests.run_recorded",
            "--record",
            str(root / (a.name + "_provenance.json")),
            "--log",
            str(root / "logs" / (a.name + ".log")),
            "--",
            "timeout",
            str(a.seconds),
            *command,
        ],
        env=env,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
