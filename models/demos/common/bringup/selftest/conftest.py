# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the framework's own tests: a throwaway git repo with a spec and a ledger. No device, no model."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.spec import Spec

PY = sys.executable


def record_cmd(**metrics) -> str:
    body = "; ".join(f"M.record({k!r}, {v!r})" for k, v in metrics.items())
    return f'{PY} -c "from models.demos.common.bringup.core import metrics as M; {body}"'


def git(repo: Path, *args) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


class Sandbox:
    def __init__(self, root: Path):
        self.repo = root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.email", "t@example.com")
        git(self.repo, "config", "user.name", "t")
        (self.repo / "README").write_text("x\n")
        git(self.repo, "add", "README")
        git(self.repo, "commit", "-q", "-m", "init")
        self.spec_path = self.repo / "spec.yaml"
        self.spec_data = {
            "model": "toyspec",
            "tag": "toy",
            "commit_trailer": "",
            "paths": {"repo": str(self.repo), "art": str(root / "art"), "bringup_dir": "bringup"},
        }
        self.write_spec()

    def write_spec(self, **extra):
        self.spec_data.update(extra)
        self.spec_path.write_text(yaml.safe_dump(self.spec_data))
        return self.spec

    @property
    def spec(self) -> Spec:
        return Spec.load(self.spec_path)

    @property
    def ledger(self) -> Ledger:
        return Ledger(self.spec.bringup_dir)

    def tasks(self, *tasks):
        self.ledger.write_tasks({"tasks": list(tasks)})
        return self.ledger

    def git(self, *args):
        return git(self.repo, *args)


@pytest.fixture
def sandbox(tmp_path):
    return Sandbox(tmp_path)


@pytest.fixture
def fx(tmp_path, monkeypatch):
    """A valid model spec whose hooks are the synthetic fixture; metrics go to tmp."""
    monkeypatch.setenv(M.RESULTS_ENV, str(tmp_path / "results"))
    monkeypatch.setenv(M.TASK_ENV, "T")

    def make(**over):
        d = {
            "model": "fixture",
            "hf_id": "none/fixture",
            "model_dir": "models/demos/fixture",
            "hooks": "models.demos.common.bringup.selftest.fixture_model",
            "num_layers": 3,
            "box": {"mesh": [1, 1]},
            "target": {"seq": 256, "chunk": 64},
            "ladder": [
                {"name": "s256", "seq": 256, "chunk": 64, "full_dumps": True},
                {"name": "s512", "seq": 512, "chunk": 128},
                {"name": "last", "seq": 512, "chunk": 128, "golden": "s512", "prefix_from_golden": True},
            ],
            "block_types": {"blk": {"layers": "0-2"}},
            "state": {"kind": "kv", "tensors": ["key", "value"]},
            "paths": {"art": str(tmp_path / "art"), "repo": str(tmp_path / "repo")},
        }
        d.update(over)
        (tmp_path / "repo").mkdir(exist_ok=True)
        p = tmp_path / "spec.yaml"
        p.write_text(yaml.safe_dump(d))
        return str(p)

    return make


def got():
    return {k: v["value"] for k, v in M.load("T").items()}


def pytest_sessionfinish(session, exitstatus):
    """Selftest gates check these counts (a gate on exit code alone would pass an empty collection)."""
    from models.demos.common.bringup.core import metrics

    rep = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = rep.stats if rep else {}
    metrics.record("selftest_passed", len(stats.get("passed", [])))
    metrics.record("selftest_failed", len(stats.get("failed", [])) + len(stats.get("error", [])))
