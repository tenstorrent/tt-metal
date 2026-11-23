# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F303: a repo-wide hook is checked on the gate commit's own files; violations elsewhere do not refuse it."""

from pathlib import Path

import pytest

from models.demos.common.bringup.core import gate as G

CHECKER = Path(__file__).resolve().parents[5] / "scripts/check_models_do_not_import_tests.py"


def _repo(tmp_path: Path, clean: str, bad: str) -> Path:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / CHECKER.name).write_text(CHECKER.read_text())
    m = tmp_path / "models" / "x"
    m.mkdir(parents=True)
    (m / "clean.py").write_text(clean)
    (m / "old.py").write_text(bad)  # a pre-existing violation the commit does not touch
    return tmp_path


def test_violation_outside_the_commit_is_skipped(tmp_path):
    repo = _repo(tmp_path, "import os\n", "from tests.ttnn import utils\n")
    env = G.repo_wide_hook_env(repo, ["models/x/clean.py"])
    assert "check-models-do-not-import-tests" in env["SKIP"].split(",")


def test_violation_in_the_commit_refuses_it(tmp_path):
    repo = _repo(tmp_path, "from tests.ttnn import utils\n", "import os\n")
    with pytest.raises(RuntimeError, match="check-models-do-not-import-tests"):
        G.repo_wide_hook_env(repo, ["models/x/clean.py"])
