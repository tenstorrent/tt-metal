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
    counts_csv,
    item_key,
    nop_thread,
    require_under,
    rm_tree,
    work_dir,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodeids", required=True, type=Path)  # pytest IDs from collect
    p.add_argument("--case-list", required=True, type=Path)  # JSON for consume
    p.add_argument(
        "--counts", default=os.environ.get("NOP_COUNTS", "")
    )  # e.g. 1,2,...,100
    args = p.parse_args()

    # Absolute binary path; argv list only (no shell) — counts/thread validated below.
    ttnop = Path(os.environ.get("TTNOP", "")).expanduser().resolve()
    if not args.counts or not ttnop.is_file():
        return 1

    try:
        counts = counts_csv(args.counts)  # digits / ranges only
        thread = nop_thread()  # safelist: unpack|math|pack
        nodeids_path = require_under(args.nodeids)
        case_list_path = require_under(args.case_list)
    except ValueError:
        return 1

    entries = []
    for nodeid in nodeids_path.read_text().splitlines():
        nodeid = nodeid.strip()
        if not nodeid:
            continue
        # Same key scheme as prepare (sha1 of nodeid)
        key = item_key(nodeid)
        work = work_dir(key)
        # Skip if prepare never snapped this case (no meta.json)
        if not (work / "meta.json").is_file():
            continue
        # Emit perturbed ELF sets under work/batch/n<count>/
        batch = require_under(work / "batch")
        rm_tree(batch)
        batch.mkdir(parents=True)
        base_dir = require_under(work / "base_elfs")
        # OpenMP lives inside ttnop: parse base once, patch all counts in parallel
        r = subprocess.run(
            [
                str(ttnop),
                "batch",
                "--base-dir",
                str(base_dir),
                "--out-root",
                str(batch),
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

    case_list_path.write_text(json.dumps(entries, indent=2) + "\n")
    return 0 if entries else 1  # non-zero → shell sets rc / aborts consume


if __name__ == "__main__":
    sys.exit(main())
