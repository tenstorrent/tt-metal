# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""knowledge ids and knowledge/usage.py: every entry has a unique id; the step reports are validated, cross-checked
against what the step touched, and tallied (uses, contradictions, unused over N relevant runs, promotion)."""

import json
import subprocess
from pathlib import Path

from models.demos.common.bringup.knowledge import usage
from models.demos.common.bringup.knowledge.check import SHARED_FILE, check_ids, knowledge_files

OP = "ttnn/cpp/ttnn/operations/foo/my_op"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _entry(i: str, title: str, applies: str) -> str:
    return (
        f"- **[{i}] {title}.** Symptom: s. Cause: c. Fix: f. Applies to: {applies}. Seen at: abc1234. Found: m T.1.\n"
    )


def _repo(tmp: Path) -> Path:
    _git(tmp, "init", "-q")
    (tmp / OP / "device").mkdir(parents=True)
    (tmp / OP / "device/my_op_device_operation.cpp").write_text("int a;\n")
    (tmp / "models/demos/m/tt").mkdir(parents=True)
    (tmp / "models/demos/m/tt/mhc.py").write_text("x = 1\n")
    shared = tmp / SHARED_FILE
    shared.parent.mkdir(parents=True)
    shared.write_text(
        "# Known issues\n\n## API behavior\n"
        + _entry("K-1", "Op lesson", OP)
        + _entry("K-2", "Method lesson", "method (component tests)")
    )
    (tmp / "models/demos/m/bringup").mkdir(parents=True)
    (tmp / "models/demos/m/bringup/known_issues.md").write_text(
        "# Known issues: m\n\n" + _entry("M-1", "Model lesson", "tt/mhc.py")
    )
    _git(tmp, "add", "-A")
    _git(tmp, "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "base")
    return tmp


def test_ids_unique_present_and_next(tmp_path):
    repo = _repo(tmp_path)
    errs, nxt = check_ids(knowledge_files(repo))
    assert errs == []
    assert nxt[str(repo / SHARED_FILE)] == "K-3"
    assert nxt[str(repo / "models/demos/m/bringup/known_issues.md")] == "M-2"
    f = repo / "models/demos/m/bringup/known_issues.md"
    f.write_text(f.read_text() + _entry("K-1", "Clash", "tt/mhc.py") + "- **No id.** Symptom: s. Found: m.\n")
    errs, _ = check_ids(knowledge_files(repo))
    assert any("K-1 is used twice" in e for e in errs) and any("has no id" in e for e in errs)


def test_finalize_validates_and_traces(tmp_path):
    repo = _repo(tmp_path)
    p = usage.report_path(repo / "models/demos/m/bringup", "C.x", "implement", 1)
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "used": [
                    {"id": "K-1", "how": "a"},
                    {"id": "M-1", "how": "b"},
                    {"id": "K-2", "how": "c"},
                    {"id": "Z-9"},
                ],
                "contradicted": [{"id": "K-1", "evidence": "abc: the op now takes it"}],
                "missing": ["how to X", ""],
            }
        )
    )
    meta = {"model": "m", "run": "r1", "task": "C.x", "role": "implement", "attempt": 1, "t": "2026-10-13T10:00:00"}
    problems = usage.finalize(p, meta, [f"{OP}/device/my_op_device_operation.cpp"], ["ls"], repo)
    assert len(problems) == 1 and "Z-9" in problems[0]
    rep = json.loads(p.read_text())
    traced = {u["id"]: u["traced"] for u in rep["used"]}
    assert traced == {"K-1": True, "M-1": False, "K-2": None}  # touched / not touched / method: unchecked
    assert rep["missing"] == ["how to X"] and rep["meta"]["task"] == "C.x" and rep["meta"]["head"]
    gone = usage.report_path(repo / "models/demos/m/bringup", "C.y", "implement", 1)
    assert usage.finalize(gone, {**meta, "task": "C.y"}, [], [], repo) == [f"no knowledge report {gone.name}"]
    assert json.loads(gone.read_text())["used"] == []  # the stub still counts the step


def test_tally_flags_contradicts_promotes(tmp_path):
    repo = _repo(tmp_path)
    known = usage.entries(repo)
    anchors = {"K-1": [OP], "K-2": [], "M-1": ["models/demos/m/tt/mhc.py"]}

    def rep(run, used=(), contradicted=(), touched=(), model="m"):
        return {
            "used": [{"id": i, "how": "x", "traced": True} for i in used],
            "contradicted": [{"id": i, "evidence": "e"} for i in contradicted],
            "missing": ["topic"],
            "meta": {
                "model": model,
                "run": run,
                "task": "T",
                "t": f"2026-10-{run[1:]:0>2}T00:00:00",
                "touched": list(touched),
            },
        }

    reports = [rep("r1", used=["M-1"], touched=[OP + "/x.cpp"])]
    reports += [rep(f"r{n}", touched=[OP + "/x.cpp"]) for n in range(2, 7)]  # K-1 relevant, unused 5 runs
    reports += [rep("r7", used=["M-1"], contradicted=["K-2"])]
    reports += [rep("r8", touched=[OP + "/x.cpp"], model="other")]  # another model: not M-1's run
    t = usage.tally(known, reports, anchors, 5)
    by = {s["id"]: s for s in t["entries"]}
    # a shared entry's runs are every model's: r1-r6 and r8 touched its op (r7 did not)
    assert by["K-1"]["relevant_runs"] == 7 and by["K-1"]["unused_streak"] == 7 and "K-1" in t["flag_unused"]
    assert by["M-1"]["uses"] == 2 and "M-1" in t["promote"] and by["M-1"]["relevant_runs"] == 2
    assert t["contradicted"] == ["K-2"] and by["K-2"]["contradictions"][0]["run"] == "r7"
    assert by["K-2"]["relevant_runs"] == 8  # a shared entry without anchors is relevant to every run
    assert t["missing_topics"] == [("topic", 8)]
    assert "## Contradicted" in usage.markdown(t)


def test_transcript_touches(tmp_path):
    log = tmp_path / "a.jsonl"
    ev = lambda name, inp: json.dumps(  # noqa: E731
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "input": inp}]}}
    )
    log.write_text(
        "\n".join(
            [
                ev("Read", {"file_path": str(tmp_path / "models/demos/m/tt/mhc.py")}),
                ev("Grep", {"pattern": "x", "path": "ttnn/ttnn/bringup/sdpa"}),
                ev("Bash", {"command": "pytest x"}),
                "not json",
            ]
        )
    )
    paths, cmds = usage.transcript_touches(log, tmp_path)
    assert paths == ["models/demos/m/tt/mhc.py", "ttnn/ttnn/bringup/sdpa"] and cmds == ["pytest x"]
