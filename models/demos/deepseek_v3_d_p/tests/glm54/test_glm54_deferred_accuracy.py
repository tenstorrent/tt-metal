# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for deferring a device test's accuracy check past the device lock (deferred_accuracy and its checker)."""

import pytest
import torch

from models.demos.deepseek_v3_d_p.tests.glm54 import check_deferred_accuracy
from models.demos.deepseek_v3_d_p.tests.glm54.deferred_accuracy import DEFER_ENV, run_or_defer

CALLS = []


def toy_check(x: torch.Tensor, expected: float, label: str) -> None:
    """A check function as device tests write them: tensors and plain values in, AssertionError on failure."""
    CALLS.append(label)
    assert float(x.sum()) == expected, f"{label}: sum {float(x.sum())} != {expected}"


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


def test_inline_runs_the_check_and_returns_its_failure(monkeypatch):
    monkeypatch.delenv(DEFER_ENV, raising=False)
    assert run_or_defer(toy_check, "ok", x=torch.ones(3), expected=3.0, label="ok") == []
    failures = run_or_defer(toy_check, "bad", x=torch.ones(3), expected=4.0, label="bad")
    assert len(failures) == 1 and failures[0].startswith("bad: sum 3.0 != 4.0")  # pytest may append its explanation
    assert CALLS == ["ok", "bad"]


def test_deferred_check_runs_after_on_the_saved_inputs(monkeypatch, tmp_path, capsys):
    """Deferred: nothing runs in the test; the checker runs the same function on the saved tensors, deletes a bundle
    that passed, keeps one that failed and exits 1; an empty directory exits 2."""
    monkeypatch.setenv(DEFER_ENV, str(tmp_path))
    assert run_or_defer(toy_check, "ok", x=torch.ones(3), expected=3.0, label="ok") == []
    assert run_or_defer(toy_check, "bad", x=torch.ones(3), expected=4.0, label="bad") == []
    assert CALLS == [] and len(list(tmp_path.glob("*.pt"))) == 2

    assert check_deferred_accuracy.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert sorted(CALLS) == ["bad", "ok"]
    assert out.count("GLM54_ACCURACY: PASS") == 1 and out.count("GLM54_ACCURACY: FAIL") == 1
    assert "bad: sum 3.0 != 4.0" in out
    left = list(tmp_path.glob("*.pt"))
    assert len(left) == 1 and left[0].name.startswith("bad-")

    empty = tmp_path / "empty"
    empty.mkdir()
    assert check_deferred_accuracy.main([str(empty)]) == 2


def test_checker_reports_errors_and_continues(monkeypatch, tmp_path, capsys):
    """A bundle that cannot be checked (saved at another checkout HEAD, or naming a check that no longer exists) is an
    ERROR line counted as failed and kept; the checker goes on to the next bundle."""
    monkeypatch.setenv(DEFER_ENV, str(tmp_path))
    for name in ("a-stale", "b-renamed", "c-ok"):
        run_or_defer(toy_check, name, x=torch.ones(3), expected=3.0, label=name)
    stale, renamed, ok = sorted(tmp_path.glob("*.pt"))
    b = torch.load(stale, weights_only=True)
    torch.save({**b, "head": "0" * 40}, stale)
    b = torch.load(renamed, weights_only=True)
    torch.save({**b, "check": f"{__name__}:no_such_check"}, renamed)

    assert check_deferred_accuracy.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert out.count("GLM54_ACCURACY: ERROR") == 2 and "from HEAD 0000000000" in out and "AttributeError" in out
    assert out.count("GLM54_ACCURACY: PASS") == 1 and CALLS == ["c-ok"]
    assert sorted(tmp_path.glob("*.pt")) == [stale, renamed]
