# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""knowledge/stale.py: entries whose anchored code changed since `Seen at:` are listed; moved entries keep their age."""

import subprocess
from pathlib import Path

from models.demos.common.bringup.knowledge.stale import Resolver, check_entry, entries_of, expand

SHARED = "models/demos/common/bringup/knowledge/known_issues.md"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", msg)
    return _git(repo, "rev-parse", "--short=11", "HEAD")


def _entry(title: str, applies: str, seen: str = "") -> str:
    seen = f" Seen at: {seen}." if seen else ""
    return f"- **{title}.** Symptom: s. Cause: c. Fix: f. Applies to: {applies}.{seen} Found: m T.1.\n"


def test_expand_braces():
    assert expand("ttnn/ttnn/bringup/{dispatch,combine}") == ["ttnn/ttnn/bringup/dispatch", "ttnn/ttnn/bringup/combine"]


def test_changed_unchanged_unanchored_and_moved_entries(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q")
    op = repo / "ttnn/cpp/ttnn/operations/foo/my_op"
    (op / "device").mkdir(parents=True)
    (op / "device/my_op_device_operation.cpp").write_text("int a;\n")
    (op / "my_op_nanobind.cpp").write_text('ttnn::bind_function<"my_op">(mod);\n')
    tool = repo / "models/demos/common/bringup/testing/tool.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("x = 1\n")
    shared = repo / SHARED
    shared.parent.mkdir(parents=True)
    shared.write_text("# Known issues\n\n## API behavior\n" + _entry("Moved lesson", "my_op") + "\n")
    base = _commit(repo, "base")
    shared.write_text(
        "# Known issues\n\n## API behavior\n"
        + _entry("Op lesson", "my_op", base)
        + _entry("Tool lesson", "testing/tool.py", base)
        + _entry("Method lesson", "component tests", base)
    )
    model = repo / "models/demos/m/bringup/known_issues.md"
    model.parent.mkdir(parents=True)
    model.write_text(
        "# Known issues: m\n\n- **Moved lesson.** Symptom: s. Cause: c. Fix: f. Applies to: my_op. Found: m.\n"
    )
    _commit(repo, "split the knowledge")
    (op / "device/my_op_device_operation.cpp").write_text("int b;\n")
    _commit(repo, "my_op: change the validation")
    head = _git(repo, "rev-parse", "--short=11", "HEAD")

    res = Resolver(repo, head)
    got = {e["title"]: check_entry(e, res, head) for e in entries_of(shared, repo)}
    assert got["Op lesson."]["status"] == "changed"
    assert got["Op lesson."]["anchors"] == ["ttnn/cpp/ttnn/operations/foo/my_op"]
    assert "my_op: change the validation" in got["Op lesson."]["commits"][0]
    assert got["Tool lesson."]["status"] == "unchanged"  # a partial path resolves by suffix
    assert got["Method lesson."]["status"] == "unanchored"
    moved = check_entry(entries_of(model, repo)[0], res, head)
    assert moved["since"] == base  # its age is the shared file's, not the commit that moved it
    assert moved["status"] == "changed"


def test_needs_probe_method_and_own_model_paths(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q")
    for m in ("a", "b"):
        f = repo / f"models/demos/{m}/tt/mhc.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1\n")
    (repo / "models/demos/a/bringup").mkdir(parents=True)
    base = _commit(repo, "base")
    model = repo / "models/demos/a/bringup/known_issues.md"
    model.write_text(
        "# Known issues: a\n\n"
        + _entry("Own file", "tt/mhc.py", base)
        + _entry("Method", "method (component tests)", base)
        + "- **Detail only.** Symptom: s. Cause: c. Fix: f. Detail: known_issues_detail.md. Found: a T.1.\n"
        + f"- **Open.** Symptom: s. Cause: c. Fix: f. Applies to: tt/mhc.py. Seen at: {base}. Check: p. "
        f"Unverified since: {base}. Found: a T.1.\n"
    )
    _commit(repo, "lessons")
    (repo / "models/demos/b/tt/mhc.py").write_text("x = 2\n")  # another model's file of the same name
    _commit(repo, "b: change mhc")
    head = _git(repo, "rev-parse", "--short=11", "HEAD")
    res = Resolver(repo, head)
    got = {e["title"]: check_entry(e, res, head) for e in entries_of(model, repo)}
    assert got["Own file."]["anchors"] == ["models/demos/a/tt/mhc.py"]  # this model's tt/mhc.py, not b's
    assert got["Own file."]["status"] == "unchanged"
    assert got["Method."]["status"] == "method"
    assert got["Open."]["status"] == "needs-probe"
    assert got["Detail only."]["anchors"] == []  # the pointer to the detail file is not the entry's code


def test_model_entries_are_short_and_detail_sections_exist(tmp_path):
    from models.demos.common.bringup.knowledge.check import MODEL_MAX_CHARS, check_model_known_issues

    ki = tmp_path / "known_issues.md"
    ok = "- **Short.** Symptom: s. Cause: c. Fix: f. Detail: known_issues_detail.md. Found: m T.1."
    long = "- **Long.** Symptom: " + "x" * MODEL_MAX_CHARS + ". Cause: c. Fix: f. Found: m T.1."
    ki.write_text(f"# Known issues: m\n\n{ok}\n")
    errs, n = check_model_known_issues(ki)
    assert n == 1 and any("no '## Short' section" in e for e in errs)
    (tmp_path / "known_issues_detail.md").write_text("# Known issues, full text: m\n\n## Short\n- full text\n")
    assert check_model_known_issues(ki) == ([], 1)
    ki.write_text(f"# Known issues: m\n\n{ok}\n{long}\n")
    errs, _ = check_model_known_issues(ki)
    assert len(errs) == 1 and "chars (max" in errs[0]


def test_a_shortened_entry_keeps_its_age_through_the_detail_file(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q")
    shared = repo / SHARED
    shared.parent.mkdir(parents=True)
    long_title = "A long original title that the trim shortened"
    shared.write_text("# Known issues\n\n## API behavior\n" + _entry(long_title, "component tests") + "\n")
    base = _commit(repo, "base")
    shared.write_text("# Known issues\n\n## API behavior\n")
    bdir = repo / "models/demos/m/bringup"
    bdir.mkdir(parents=True)
    (bdir / "known_issues.md").write_text(
        "# Known issues: m\n\n- **Short title.** Symptom: s. Cause: c. Fix: f. Detail: known_issues_detail.md. "
        "Found: m T.1.\n"
    )
    (bdir / "known_issues_detail.md").write_text(
        "# Known issues, full text: m\n\n## Short title\n" + _entry(long_title, "component tests")
    )
    _commit(repo, "split and trim")
    (entry,) = entries_of(bdir / "known_issues.md", repo)
    assert entry["blame"] == base
