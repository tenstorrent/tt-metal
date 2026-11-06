# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Add exact frozen-cohort test expectations to each independent patch."""
import difflib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = Path("/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-nightly-preparation")
BIN = Path("/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/.pre-commit/bin")
COHORTS = json.loads((ROOT.parents[1] / "cohorts.json").read_text())
FILES = ["tests/workflows/test_gemma4_autoport_smoke_suites.py", "tests/workflows/test_eval_config_mode_reference.py"]
BASE = {name: (WORK / name).read_text() for name in FILES}
for suite, cohort in COHORTS.items():
    subprocess.run(["git", "apply", str(ROOT / (suite + ".patch"))], cwd=WORK, check=True)
    s = BASE[FILES[0]]
    gpqa = COHORTS["r1_gpqa_diamond"]["rungs"]["nightly"] if suite == "r1_gpqa_diamond" else []
    a = s.index('    assert json.loads(command[command.index("--samples") + 1])')
    b = s.index('    assert "--limit" not in command', a)
    s = (
        s[:a]
        + '    assert json.loads(command[command.index("--samples") + 1]) == {\n        "r1_gpqa_diamond": '
        + repr(gpqa)
        + ' if mode == "ci-nightly" else '
        + repr(gpqa[:3] if gpqa else [152, 189])
        + "\n    }\n"
        + s[b:]
    )
    # Per-candidate expectation is frozen data, independent of config under test.
    s = s.replace(
        "    task = TASKS[index]\n",
        '    task = TASKS[index]\n    if mode == "ci-nightly" and task.task_name == '
        + repr(suite)
        + ":\n        names = "
        + repr(cohort["rungs"]["nightly"])
        + "\n",
    )
    s = s.replace("        n_tasks=2,", "        n_tasks=len(names),").replace(
        'assert generated["datasets"][0]["n_tasks"] == 2', 'assert generated["datasets"][0]["n_tasks"] == len(names)'
    )
    s = s.replace("    EvalLimitMode,\n", "")
    (WORK / FILES[0]).write_text(s)
    s = BASE[FILES[1]]
    if suite == "r1_gpqa_diamond":
        s = s.replace("assert gpqa.doc_ids == [152, 189]", "assert gpqa.doc_ids == " + repr(gpqa))
        s = s.replace(
            "assert gpqa.limit_samples_map[mode] in (0, 2)",
            "assert gpqa.limit_samples_map[mode] == (10 if mode == EvalLimitMode.CI_NIGHTLY else 2)",
        )
    (WORK / FILES[1]).write_text(s)
    subprocess.run([str(BIN / "ruff"), "format", *FILES], cwd=WORK, check=True)
    subprocess.run([str(BIN / "ruff"), "check", *FILES], cwd=WORK, check=True)
    subprocess.run([str(BIN / "pytest"), "-q", *FILES], cwd=WORK, check=True)
    with (ROOT / (suite + ".patch")).open("a") as out:
        for name in FILES:
            out.write(
                "".join(
                    difflib.unified_diff(
                        BASE[name].splitlines(keepends=True),
                        (WORK / name).read_text().splitlines(keepends=True),
                        fromfile="a/" + name,
                        tofile="b/" + name,
                    )
                )
            )
    subprocess.run(["git", "restore", "reference_config/evals/eval_config.py", *FILES], cwd=WORK, check=True)
