"""Host-only exact AST audit of held model configuration."""
import ast
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
worktree = Path("/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa80-c4-fastforward")
ids = json.loads((root / "cohort.json").read_text())["doc_ids"]
source = "reference_config/evals/eval_config.py"
before = ast.parse(
    subprocess.check_output(
        ["git", "-C", str(worktree), "show", "7ad6cf50d160a2e9cbe007fc4475af3c8f8bf365:" + source], text=True
    )
)
after = ast.parse((worktree / source).read_text())
changes = []


def compare(a, b, path=""):
    if type(a) != type(b):
        raise AssertionError((path, type(a), type(b)))
    if isinstance(a, ast.AST):
        for field in a._fields:
            compare(getattr(a, field), getattr(b, field), path + "." + field)
    elif isinstance(a, list):
        if len(a) != len(b):
            assert len(a) == 20 and len(b) == 80 and [ast.literal_eval(x) for x in b] == ids
            assert [ast.literal_eval(x) for x in a] == ids[:20]
            changes.append("doc_ids20to80")
            return
        for x, y in zip(a, b):
            compare(x, y, path)
    elif a != b:
        assert (a, b) in [(20, 80), (8, 4)]
        changes.append("nightly20to80" if (a, b) == (20, 80) else "clients8to4")


compare(before, after)
assert sorted(changes) == ["clients8to4", "doc_ids20to80", "nightly20to80"]
print("PASS production AST exact except one GPQA list20to80 and nightlycount20to80 and clients8to4")
