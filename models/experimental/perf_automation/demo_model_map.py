"""See the deterministic model skeleton (ast) on a real run — NO key, NO hardware.

    python demo_model_map.py [runs/<id>]   # defaults to runs/latest

Prints the full skeleton + an op-class-filtered view (what the lead would see for
a matmul lever). Shows how the localization context stays small even for a
many-file model.
"""

import json
import sys
from pathlib import Path

from agent.model_map import OP_CLASS_SUBSTRINGS, build_model_map, render_skeleton


def main(argv):
    run = Path(argv[0]) if argv else Path("runs/latest")
    m = json.loads((run / "manifest.json").read_text())
    root = Path(m["config"]["model_root"])
    files = [root / f for f in m["pathmap"]["model_files"]]
    mm = build_model_map(files, root=root)

    full = render_skeleton(mm)
    print("===== FULL SKELETON =====")
    print(full)
    print(f"\n[full: {len(full.splitlines())} lines, ~{len(full)//4} tokens]\n")

    subs = OP_CLASS_SUBSTRINGS["matmul"]
    filt = render_skeleton(mm, op_substrings=subs)
    print(f"===== FILTERED for a matmul lever ({subs}) =====")
    print(filt)
    print(f"\n[matmul-filtered: {len(filt.splitlines())} lines, ~{len(filt)//4} tokens]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
