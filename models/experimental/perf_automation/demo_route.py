"""See the REAL playbook search (ROUTE) on a REAL baseline — NO key, NO hardware.

    python demo_route.py [runs/<id>]      # defaults to the newest run with a profile

Builds the GUIDELINES index, takes the top bucket from a real baseline_profile.json,
and prints the candidate levers ROUTE finds + the first lines of the top one (what
SELECT would read to choose). This is the deterministic "find the information" half.
"""

import glob
import json
import sys
from pathlib import Path

from agent import router


def main(argv):
    if argv:
        prof_path = Path(argv[0]) / "profiles" / "baseline_profile.json"
    else:
        cands = sorted(glob.glob("runs/2026-*/profiles/baseline_profile.json"))
        if not cands:
            print("no run with a baseline_profile.json — run the Before Loop first")
            return 1
        prof_path = Path(cands[-1])

    index = router.build_index()
    print(f"playbook index: {len(index)} tagged sections from GUIDELINES/")
    print(f"baseline:       {prof_path}\n")

    prof = json.loads(prof_path.read_text())
    top = max(prof["buckets"], key=lambda b: b["device_ms"])
    query = {k: v for k, v in top["tags"].items() if k in router.DIMENSIONS}
    print(f"top bucket: {top['id']}  {top['device_ms']:.3f} ms  ({top['pct']:.1f}%)")
    print(f"ROUTE query (the bucket's tag fingerprint):\n  {query}\n")

    hits = router.route(index, query)
    print(f"=== ROUTE found {len(hits)} candidate levers ===")
    for h in hits:
        print(f"  [{h['file']:36}] {h['id']:26} {h['title']}")
    if hits:
        print(f"\n--- read_section('{hits[0]['id']}') — what SELECT would read ---")
        print("\n".join(router.read_section(hits[0]["id"]).splitlines()[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
