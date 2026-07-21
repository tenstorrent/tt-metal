#!/usr/bin/env python3
"""After prepare: run ttnop batch per case, write case-list JSON for consume.

Env: OPEN_MP_NOP_OUT, TTNOP, NOP_THREAD (default math), NOP_COUNTS (or --counts).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from nop_injector.helper import (
    allowed_roots,
    item_key,
    rm_tree,
    work_dir,
)

# Same approach as
# tests/tt_metal/tt_metal/jit_build/compile_stress_ci.py.
_TTNOP = Path(__file__).resolve().parent.parent / "ttnop" / "ttnop"
_THREAD = "math"
_COUNTS = ",".join(str(c) for c in range(1, 101))  # 1..100


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodeids", required=True, type=Path)  # pytest IDs from collect
    p.add_argument("--case-list", required=True, type=Path)  # JSON for consume
    # --counts kept for CLI compat with run_nop_injector.sh but ignored for argv
    # (Cycode: must not flow into subprocess). Batch always uses _COUNTS above.
    p.add_argument("--counts", default=os.environ.get("NOP_COUNTS", ""))
    args = p.parse_args()

    if not _TTNOP.is_file():
        return 1

    roots = [os.path.abspath(str(r)) for r in allowed_roots()]

    # Path-traversal guard for Cycode
    nodeids_abs = os.path.abspath(os.path.expanduser(str(args.nodeids)))
    if not any(nodeids_abs == r or nodeids_abs.startswith(r + os.sep) for r in roots):
        return 1
    # Guard the case-list write path
    case_list_abs = os.path.abspath(os.path.expanduser(str(args.case_list)))
    if not any(
        case_list_abs == r or case_list_abs.startswith(r + os.sep) for r in roots
    ):
        return 1

    entries = []
    with open(nodeids_abs, "r") as f:
        nodeids_text = f.read()
    for nodeid in nodeids_text.splitlines():
        nodeid = nodeid.strip()
        if not nodeid:
            continue
        # Because nodeid is the same, the hash key will be the same
        key = item_key(nodeid)
        work = work_dir(key)
        # Skip if prepare never snapped this case (no meta.json)
        if not (work / "meta.json").is_file():
            continue

        # Path-traversal guard for Cycode
        work_abs = os.path.abspath(str(work))
        if not any(work_abs == r or work_abs.startswith(r + os.sep) for r in roots):
            return 1

        batch = os.path.join(work_abs, "batch")
        rm_tree(batch)
        os.makedirs(batch)

        # Command-injection guard for Cycode: every argv entry is a constant /
        # literal. Per-case dirs are selected only via cwd (not argv).
        r = subprocess.run(  # nosec B603
            [
                str(_TTNOP),
                "batch",
                "--base-dir",
                "base_elfs",
                "--out-root",
                "batch",
                "--thread",
                _THREAD,
                "--counts",
                _COUNTS,
            ],
            cwd=work_abs,
            shell=False,
            stdout=subprocess.DEVNULL,
        )
        if r.returncode == 0:
            entries.append({"nodeid": nodeid, "key": key, "work": str(work)})

    # case_list_abs validated above for cycode .
    with open(case_list_abs, "w") as f:
        f.write(json.dumps(entries, indent=2) + "\n")
    # Exit 1 when no cases were batched so run_nop_injector.sh can record a
    # failed exit_status; an empty case-list then triggers its hard stop before consume.
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
