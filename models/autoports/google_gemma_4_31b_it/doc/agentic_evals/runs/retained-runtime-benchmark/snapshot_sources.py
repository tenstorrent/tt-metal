import hashlib
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[7]
paths = sorted((root / "models/autoports/google_gemma_4_31b_it/tt").glob("*.py"))
paths.append(root / "models/autoports/google_gemma_4_31b_it/tests/run_serving.sh")
commands = json.loads(pathlib.Path(__file__).with_name("commands.json").read_text())
a = commands["tti_matrix"]["argv"]
paths.append(pathlib.Path(a[a.index("--runtime-model-spec-json") + 1]))


def head(cwd):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


result = {
    "root_head": head(root),
    "tti_head": head(commands["tti_matrix"]["cwd"]),
    "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
}
with pathlib.Path(sys.argv[1]).open("x") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
