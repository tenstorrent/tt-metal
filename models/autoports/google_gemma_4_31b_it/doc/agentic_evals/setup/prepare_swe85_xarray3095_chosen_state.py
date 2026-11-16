"""Literal audited native96 sed effect, never executes agent commands."""
import hashlib
import json
import shutil
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
B = D / "anomalies/swe85-below-band/correct-actions"
P = B / "pydata__xarray-3095"
O = B / "pydata__xarray-3095-chosen48"
assert not O.exists()
shutil.copytree(P, O)
shutil.rmtree(O / "candidate")
shutil.copytree(P / "precall", O / "candidate")
p = O / "candidate/xarray/core/variable.py"
lines = p.read_text().splitlines(True)
assert lines[182] == "    data = np.asarray(data)\n"
lines[182] = "    if not isinstance(data, PandasIndexAdapter):\n        data = np.asarray(data)\n"
p.write_text("".join(lines))
m = json.loads((O / "manifest.json").read_text())
m["correct_branch_status"] = "Chosen native48 post-state only, not a correct candidate"
m["semantic_execution_validation"] = "pending"
m["chosen_native_message"] = 96
for row in m["precall_files"]:
    row["after_sha256"] = hashlib.sha256((O / "candidate" / row["file"]).read_bytes()).hexdigest()
(O / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
