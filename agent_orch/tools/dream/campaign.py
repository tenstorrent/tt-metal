"""Campaign config and path conventions shared by all agent_orch tools."""

import fnmatch
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

ORCH_DIR = Path(__file__).resolve().parents[2]  # .../agent_orch


def dream_home() -> Path:
    return Path(os.environ.get("DREAM_HOME", f"/localdev/{os.environ.get('USER', 'user')}/dream"))


@dataclass
class Campaign:
    name: str
    cfg: dict
    repo: Path  # any checkout/worktree of the repo; refs are shared

    @property
    def home(self) -> Path:
        return dream_home() / self.name

    @property
    def ledger(self) -> Path:
        return self.home / "ledger"

    @property
    def eval_checkout(self) -> Path:
        return self.home / "eval"

    # refs
    def ref_root(self) -> str:
        return f"dream/{self.name}/root"

    def ref_node(self, node_id: str) -> str:
        return f"dream/{self.name}/n/{node_id}"

    def ref_branch(self, rnd: int, branch: int) -> str:
        return f"dream/{self.name}/b/r{rnd:02d}-b{branch:02d}"

    def ref_ledger(self) -> str:
        return f"dream/{self.name}/ledger"

    def worktree(self, rnd: int, branch: int) -> Path:
        return self.home / "wt" / f"r{rnd:02d}-b{branch:02d}"

    def attempts_rel(self, node_id: str = "") -> str:
        base = f"agent_orch/campaigns/{self.name}/attempts"
        return f"{base}/{node_id}" if node_id else base

    def report_dir(self, label: str) -> Path:
        return self.home / "reports" / label

    def allowed(self, path: str, node_id: str | None = None) -> bool:
        if node_id and path.startswith(self.attempts_rel(node_id) + "/"):
            return True
        return any(fnmatch.fnmatch(path, pat) for pat in self.cfg.get("allowed_paths", []))

    def git(self, *args: str, cwd: Path | None = None, check: bool = True) -> str:
        r = subprocess.run(["git", *args], cwd=cwd or self.repo, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout.strip()


def repo_root(start: Path | None = None) -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=start or Path.cwd(), capture_output=True, text=True, check=True
    )
    return Path(out.stdout.strip())


def load_campaign(name: str, repo: Path | None = None) -> Campaign:
    """Load campaign.yaml from the repo (falls back to this checkout's copy)."""
    repo = repo or repo_root(ORCH_DIR)
    path = repo / "agent_orch" / "campaigns" / name / "campaign.yaml"
    if not path.exists():
        path = ORCH_DIR / "campaigns" / name / "campaign.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return Campaign(name=name, cfg=cfg, repo=repo)


def parse_node_id(node_id: str) -> tuple[int, int, int]:
    """'r01-b03-a02' -> (1, 3, 2)"""
    r, b, a = node_id.split("-")
    assert r[0] == "r" and b[0] == "b" and a[0] == "a", node_id
    return int(r[1:]), int(b[1:]), int(a[1:])


def node_id(rnd: int, branch: int, attempt: int) -> str:
    return f"r{rnd:02d}-b{branch:02d}-a{attempt:02d}"
