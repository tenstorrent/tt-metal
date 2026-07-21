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
    counts_csv,
    item_key,
    nop_thread,
    rm_tree,
    work_dir,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodeids", required=True, type=Path)  # pytest IDs from collect
    p.add_argument("--case-list", required=True, type=Path)  # JSON for consume
    p.add_argument(
        "--counts", default=os.environ.get("NOP_COUNTS", "")  # e.g. 1,2,...,100
    )
    args = p.parse_args()

    # Command-injection / path-traversal guard for Cycode
    ttnop = Path(os.path.abspath(os.path.expanduser(os.environ.get("TTNOP", ""))))

    if not args.counts or not ttnop.is_file():
        return 1

    try:
        counts = counts_csv(args.counts)  # digits / ranges only
        thread = nop_thread()  # safelist: unpack|math|pack
    except ValueError:
        return 1

    # Absolute-path allowlist
    roots = [os.path.abspath(str(r)) for r in allowed_roots()]

    # Guard the nodeids read
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
        # Emit perturbed ELF sets under work/batch/n<count>/.
        batch = os.path.abspath(str(work / "batch"))
        base_dir = os.path.abspath(str(work / "base_elfs"))
        if not any(batch == r or batch.startswith(r + os.sep) for r in roots):
            return 1
        if not any(base_dir == r or base_dir.startswith(r + os.sep) for r in roots):
            return 1
        rm_tree(batch)
        os.makedirs(batch)
        # parse base once, patch all counts in parallel.
        r = subprocess.run(
            [
                str(ttnop),
                "batch",
                "--base-dir",
                base_dir,
                "--out-root",
                batch,
                "--thread",
                thread,
                "--counts",
                counts,
            ],
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
