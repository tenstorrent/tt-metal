#!/usr/bin/env python3
"""Delete per-case work dirs listed in a case-list JSON

Each entry is {nodeid, key, work}. Removes the whole work/<key>/ tree which includes:
  - prepare snapshot (meta.json + base ELFs)
  - batch/nN/ leftovers

Does not touch fails/ or summary.log. This can be skipped by setting OPEN_MP_NOP_KEEP=1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nop_injector.helper import load_case_list, rm_tree


def main() -> int:
    """Load --case-list JSON and delete each entry's work/ directory."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-list", required=True, type=Path)
    args = p.parse_args()

    # Walk cases that made it through the program and delete their work trees.
    n = 0
    for e in load_case_list(args.case_list):
        work = Path(e["work"])
        if work.exists():
            rm_tree(work)
            n += 1
    print(f">> deleted {n} work dir(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
