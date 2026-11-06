# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prepare independent model-only patches; does not publish or dispatch."""
import ast
import difflib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXECUTION = ROOT.parents[5]
COHORTS = json.loads((ROOT.parents[1] / "cohorts.json").read_text())
WORK = Path("/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-nightly-native-preparation")
FILE = "reference_config/evals/eval_config.py"
BASE = subprocess.check_output(
    ["git", "-C", str(WORK), "show", "a087be086b6e2cd6edf215dddb4c47c48463f13a:" + FILE], text=True
)
lines = BASE.splitlines(keepends=True)
starts = [0]
for line in lines:
    starts.append(starts[-1] + len(line))


def bounds(node):
    return starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset


tree = ast.parse(BASE)
model = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "EvalConfig"
    and any(
        k.arg == "hf_model_repo" and isinstance(k.value, ast.Constant) and k.value.value == "google/gemma-4-31B-it"
        for k in node.keywords
    )
)
tasks = next(k.value.elts for k in model.keywords if k.arg == "tasks")
for chosen, cohort in COHORTS.items():
    edits = []
    for task in tasks:
        kw = {k.arg: k.value for k in task.keywords}
        name = ast.literal_eval(kw["task_name"])
        count = len(cohort["rungs"]["nightly"]) if chosen == name else 0
        value = "{EvalLimitMode.SMOKE_TEST: 2, EvalLimitMode.CI_NIGHTLY: %d}" % count
        edits.append((*bounds(kw["limit_samples_map"]), value))
        if chosen != name:
            continue
        if name == "r1_gpqa_diamond":
            edits.append((*bounds(kw["doc_ids"]), repr(cohort["rungs"]["nightly"])))
        else:
            agent = {k.arg: k.value for k in kw["agentic_eval_config"].keywords}
            value = "{EvalLimitMode.SMOKE_TEST: %r, EvalLimitMode.CI_NIGHTLY: %r}" % (
                cohort["rungs"]["smoke"],
                cohort["rungs"]["nightly"],
            )
            edits.append((*bounds(agent["task_names_map"]), value))
    source = BASE
    for a, b, value in sorted(edits, reverse=True):
        source = source[:a] + value + source[b:]
    (WORK / FILE).write_text(source)
    subprocess.run(
        [
            "/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/.pre-commit/bin/ruff",
            "format",
            str(WORK / FILE),
        ],
        check=True,
    )
    source = (WORK / FILE).read_text()
    (ROOT / (chosen + ".patch")).write_text(
        "".join(
            difflib.unified_diff(
                BASE.splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile="a/" + FILE,
                tofile="b/" + FILE,
            )
        )
    )
    subprocess.run(
        [
            "/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/.pre-commit/bin/python",
            str(ROOT / "verify.py"),
            chosen,
        ],
        cwd=WORK,
        check=True,
    )
(WORK / FILE).write_text(BASE)
