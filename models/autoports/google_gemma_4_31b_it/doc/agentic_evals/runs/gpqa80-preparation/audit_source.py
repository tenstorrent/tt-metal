"""Host-only exact AST audit of held model configuration."""
import ast
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
worktree = Path("/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa80-retained")
ids = json.loads((root / "cohort.json").read_text())["doc_ids"]
source = "reference_config/evals/eval_config.py"
before = ast.parse(
    subprocess.check_output(
        ["git", "-C", str(worktree), "show", "6efe6739e6513d0edbf49a8ef39527b80a23784d:" + source], text=True
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
            assert len(a) == 40 and len(b) == 80 and [ast.literal_eval(x) for x in b] == ids
            assert [ast.literal_eval(x) for x in a] == ids[:40]
            changes.append("doc_ids40to80")
            return
        for x, y in zip(a, b):
            compare(x, y, path)
    elif a != b:
        assert (a, b) == (40, 80)
        changes.append("nightly40to80")


compare(before, after)
assert sorted(changes) == ["doc_ids40to80", "nightly40to80"]
print("PASS production AST exact except one GPQA list40to80 and nightlycount40to80")
