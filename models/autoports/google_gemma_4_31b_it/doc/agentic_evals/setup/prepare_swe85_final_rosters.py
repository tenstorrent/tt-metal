"""Append only explicitly selected final two tasks; CPU data preparation only."""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
B = D / "anomalies/swe85-below-band"
p = B / "correct-actions/all-candidate-action-cases.json"
rows = json.loads(p.read_text())
for relative in ["psf__requests-5414/validated-pure-urls/cases.json", "sympy__sympy-14248/validated/cases.json"]:
    for row in json.loads((B / "correct-actions" / relative).read_text()):
        existing = [x for x in rows if x["case"] == row["case"]]
        if existing:
            assert existing == [row]
        else:
            rows.append(row)
assert len({x["case"] for x in rows}) == len(rows)
p.write_text(json.dumps(rows, indent=2) + "\n")
print("Correct action cases:", len(rows))
