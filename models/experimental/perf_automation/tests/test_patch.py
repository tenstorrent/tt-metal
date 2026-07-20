"""patch.apply_edits — deterministic, self-validating content-anchored edits."""

from agent.patch import apply_edits


def _model(tmp_path, text="a = 1\nb = 2\n"):
    (tmp_path / "m.py").write_text(text)
    return tmp_path


def test_single_edit_applies(tmp_path):
    root = _model(tmp_path)
    changed, failures = apply_edits(root, [{"file": "m.py", "old_string": "a = 1", "new_string": "a = 42"}])
    assert changed == ["m.py"] and failures == []
    assert (root / "m.py").read_text() == "a = 42\nb = 2\n"


def test_multiple_edits_same_file_in_order(tmp_path):
    root = _model(tmp_path)
    changed, failures = apply_edits(
        root,
        [
            {"file": "m.py", "old_string": "a = 1", "new_string": "a = 9"},
            {"file": "m.py", "old_string": "b = 2", "new_string": "b = 8"},
        ],
    )
    assert failures == [] and changed == ["m.py"]
    assert (root / "m.py").read_text() == "a = 9\nb = 8\n"


def test_empty_new_string_deletes(tmp_path):
    root = _model(tmp_path)
    changed, failures = apply_edits(root, [{"file": "m.py", "old_string": "a = 1\n", "new_string": ""}])
    assert failures == [] and (root / "m.py").read_text() == "b = 2\n"


def test_missing_anchor_fails_and_writes_nothing(tmp_path):
    root = _model(tmp_path)
    changed, failures = apply_edits(root, [{"file": "m.py", "old_string": "nonexistent", "new_string": "x"}])
    assert changed == [] and failures and "not found" in failures[0]["reason"]
    assert (root / "m.py").read_text() == "a = 1\nb = 2\n"  # untouched


def test_ambiguous_anchor_fails(tmp_path):
    root = _model(tmp_path, text="x\nx\n")  # 'x' appears twice
    changed, failures = apply_edits(root, [{"file": "m.py", "old_string": "x", "new_string": "y"}])
    assert changed == [] and "not unique" in failures[0]["reason"]


def test_all_or_nothing_across_files(tmp_path):
    (tmp_path / "a.py").write_text("keep = 1\n")
    (tmp_path / "b.py").write_text("other = 2\n")
    changed, failures = apply_edits(
        tmp_path,
        [
            {"file": "a.py", "old_string": "keep = 1", "new_string": "keep = 99"},  # would succeed
            {"file": "b.py", "old_string": "MISSING", "new_string": "x"},  # fails
        ],
    )
    assert changed == [] and failures  # nothing written because one failed
    assert (tmp_path / "a.py").read_text() == "keep = 1\n"  # the good file is untouched too


def test_missing_file_fails(tmp_path):
    changed, failures = apply_edits(tmp_path, [{"file": "nope.py", "old_string": "x", "new_string": "y"}])
    assert changed == [] and "file not found" in failures[0]["reason"]
