# SPDX-License-Identifier: Apache-2.0
"""Print unreviewed completed diagnostic replies; no automatic semantic verdict."""
import json
from pathlib import Path

base = Path(__file__).resolve().parents[2]
root = Path(__file__).resolve().parent / "fixed108"
jobs = json.loads((base / "setup/swe85-precision-bisect/p19-down-bfp8-fixed108-root/candidate-cases.json").read_text())
review = json.loads((root / "root-semantic-review.json").read_text())
done = {r["index"] for r in review["rows"]}
for i, job in enumerate(jobs):
    p = root / "runs" / (job["name"] + ".json")
    if i in done or not p.exists():
        continue
    result = json.loads(p.read_text())
    if not result.get("diagnostic_total_wall_s"):
        continue
    reply = result["free_running"][0]
    old = json.loads((base / "setup/full-q32-production-validation/fixed108/runs" / p.name).read_text())[
        "free_running"
    ][0]
    same = reply["token_ids"] == old["token_ids"]
    print(
        "CASE",
        i,
        "TOKENS",
        len(reply["token_ids"]),
        "STOP",
        reply["token_ids"][-1],
        "FLAG",
        reply["flag"],
        "SAME_FULL",
        same,
    )
    print(reply["text"] if not same else "(exact reviewed full baseline)")
