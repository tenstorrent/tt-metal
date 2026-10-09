"""Campaign spec (dream.yaml), defaults, path and ref conventions shared by all agent_orch tools."""

from __future__ import annotations

import copy
import fnmatch
import getpass
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

ORCH_DIR = Path(__file__).resolve().parents[2]  # .../agent_orch of the checkout these tools run from
LIBRARY_DIR = ORCH_DIR / "policies" / "library"

DEFAULTS: dict = {
    "description": "",
    "machine": "local",  # "local", "host", "host:port" or "user@host:port"
    "repo": None,  # repo path on the machine; default: this checkout's main repo path
    "dream_home": None,  # default /localdev/$USER/dream (on the machine)
    "python_env": None,  # default <repo>/python_env
    "editable": [],
    "brief": None,
    "rules": [],
    "forbidden_patterns": [],
    "eval": {
        "command": None,
        "direction": "minimize",  # minimize | maximize
        "unit": "",
        "gates": [],  # e.g. ["pcc >= 0.99999"], checked per case against the result's extra fields
        "timeout_s": 900,
        "baseline_runs": 3,
        "min_noise_pct": 1.0,
        "drift_check": True,
        "env": {
            "TT_METAL_HOME": "${EVAL}",
            "PYTHONPATH": "${EVAL}:${EVAL}/ttnn:${EVAL}/tools",
        },
    },
    "build": {
        "command": "./build_metal.sh --release --enable-ccache",
        "skip_if_only": ["*/kernels/*"],  # changed files matching these never need a host rebuild
        "check_file": "ttnn/ttnn/_ttnn.so",  # rebuild if this is missing
        "timeout_s": 5400,
    },
    "resource": {"reset_command": "tt-smi -r"},
    "budget": {"max_attempts": 60, "max_hours": 8.0, "max_usd": 300.0},
    # round_root: origin = every round starts from the campaign root, earlier rounds are history only (as in the
    # Dream-RSI paper); best = every round starts from the best attempt so far (improvements compound)
    "search": {
        "policy": "fresh",
        "W": 4,
        "R": 4,
        "beta": 0.6,
        "max_rounds": 6,
        "max_steps": 30,
        "round_root": "origin",
    },
    "dreaming": {
        "enabled": True,
        "revisions": 5,
        "cost_per_attempt": 0.005,
        "parallel_bonus": 0.01,
        "beta_sweep": [0.2, 0.4, 0.6, 0.8, 1.0],
        "max_replay_steps": 100,
    },
    "models": {"worker": "opus", "policy_dev": "opus", "summary": "sonnet"},
    "worker": {"timeout_min": 120},
    # Workers see only this campaign: its own repo on the machine ($DREAM_HOME/<c>/repo.git, no remotes or other
    # refs), an explicit rule, and an audit of every transcript. on_violation: flag (report it) | invalidate.
    "isolation": {"enabled": True, "on_violation": "flag"},
}

REQUIRED = ["name", "editable", "eval.command"]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def validate_spec(spec: dict) -> list[str]:
    errs = []
    for k in REQUIRED:
        v = _get(spec, k)
        if v in (None, "", []):
            errs.append(f"missing required field '{k}'")
    name = spec.get("name") or ""
    if name and not all(ch.isalnum() or ch in "-_." for ch in name):
        errs.append(f"name '{name}' may only contain letters, digits, '-', '_' and '.'")
    if (spec.get("isolation") or {}).get("on_violation", "flag") not in ("flag", "invalidate"):
        errs.append("isolation.on_violation must be 'flag' or 'invalidate'")
    if spec["eval"]["direction"] not in ("minimize", "maximize"):
        errs.append("eval.direction must be 'minimize' or 'maximize'")
    for g in spec["eval"]["gates"]:
        try:
            parse_gate(g)
        except ValueError as e:
            errs.append(str(e))
    s = spec["search"]
    if s.get("round_root", "origin") not in ("origin", "best"):
        errs.append("search.round_root must be 'origin' or 'best'")
    if int(s["W"]) < 1 or int(s["R"]) < 1:
        errs.append("search.W and search.R must be >= 1")
    for k in ("max_attempts", "max_hours", "max_usd"):
        if float(spec["budget"][k]) <= 0:
            errs.append(f"budget.{k} must be > 0")
    return errs


ROUND_ROOT_MEANING = {
    "origin": (
        "every round starts from the campaign's original code (the campaign root), as in the Dream-RSI paper. "
        "A branch closed at its depth cap does NOT continue next round: its code is gone from the next tree and "
        "survives only as history that next-round workers may port by hand"
    ),
    "best": (
        "every round starts from the best attempt found so far, so a branch closed at its depth cap effectively "
        "continues next round from the round best"
    ),
}


def round_root_mode(cfg: dict) -> str:
    return (cfg.get("search") or {}).get("round_root", "origin")


def parse_gate(g: str) -> tuple[str, str, float]:
    for op in (">=", "<=", "==", ">", "<"):
        if op in g:
            k, v = g.split(op, 1)
            try:
                return k.strip(), op, float(v)
            except ValueError:
                break
    raise ValueError(f"bad gate '{g}': expected '<field> <op> <number>' with op one of >=, <=, ==, >, <")


def load_spec(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return deep_merge(DEFAULTS, raw)


def user() -> str:
    return os.environ.get("USER") or getpass.getuser()


def default_home() -> str:
    return f"/localdev/{user()}/dream"


def parse_machine(m: str | None) -> dict | None:
    """None for the local machine, else {"user", "host", "port"}."""
    if not m or m == "local":
        return None
    u, _, rest = m.rpartition("@")
    host, _, port = rest.partition(":")
    if host in ("localhost", socket.gethostname(), socket.getfqdn()):
        return None
    return {"user": u or None, "host": host, "port": int(port) if port else None}


def git_main_repo(start: Path) -> Path:
    """Main checkout of the repository that `start` belongs to (worktrees resolve to their main repo)."""
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=start,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(out).parent


def repo_root(start: Path | None = None) -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=start or Path.cwd(), capture_output=True, text=True, check=True
    )
    return Path(out.stdout.strip())


@dataclass
class Campaign:
    name: str
    cfg: dict
    repo: Path  # any checkout/worktree of the repo; refs are shared by all of them

    # ---- machine paths -----------------------------------------------------
    @property
    def home(self) -> Path:
        return Path(os.environ.get("DREAM_HOME") or self.cfg.get("dream_home") or default_home()) / self.name

    @property
    def dream_home(self) -> Path:
        return self.home.parent

    @property
    def ledger(self) -> Path:
        return self.home / "ledger"

    @property
    def eval_checkout(self) -> Path:
        return self.home / "eval"

    @property
    def ctl(self) -> Path:
        return self.home / "ctl"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def main_repo(self) -> Path:
        """The user's checkout on the machine (holds python_env); an isolated campaign repo records it in its config."""
        if self.cfg.get("repo"):
            return Path(self.cfg["repo"])
        recorded = self.git("config", "--get", "dream.mainRepo", check=False)
        return Path(recorded) if recorded else git_main_repo(self.repo)

    @property
    def isolated(self) -> bool:
        return bool((self.cfg.get("isolation") or {}).get("enabled"))

    @property
    def python_env(self) -> Path:
        return Path(self.cfg.get("python_env") or self.main_repo / "python_env")

    def worktree(self, rnd: int, branch: int) -> Path:
        return self.home / "wt" / f"r{rnd:02d}-b{branch:02d}"

    def report_dir(self, label: str) -> Path:
        return self.home / "reports" / label

    # ---- refs --------------------------------------------------------------
    @property
    def ref_prefix(self) -> str:
        return f"refs/dream/{self.name}"

    def ref_base(self) -> str:
        return f"{self.ref_prefix}/base"

    def ref_root(self) -> str:
        return f"{self.ref_prefix}/root"

    def ref_node(self, node_id: str) -> str:
        return f"{self.ref_prefix}/n/{node_id}"

    def ref_branch(self, rnd: int, branch: int) -> str:
        return f"{self.ref_prefix}/b/r{rnd:02d}-b{branch:02d}"

    def ref_ledger(self) -> str:
        return f"{self.ref_prefix}/ledger"

    def ref_best(self) -> str:
        return f"refs/heads/dream/{self.name}/best"

    # ---- repo paths --------------------------------------------------------
    @property
    def campaign_rel(self) -> str:
        return f"agent_orch/campaigns/{self.name}"

    def attempts_rel(self, node_id: str = "") -> str:
        base = f"{self.campaign_rel}/attempts"
        return f"{base}/{node_id}" if node_id else base

    def allowed(self, path: str, node_id: str | None = None) -> bool:
        if node_id and path.startswith(self.attempts_rel(node_id) + "/"):
            return True
        return any(fnmatch.fnmatch(path, pat) for pat in self.cfg.get("editable", []))

    def git(self, *args: str, cwd: Path | None = None, check: bool = True, env: dict | None = None) -> str:
        r = subprocess.run(
            ["git", *args], cwd=cwd or self.repo, capture_output=True, text=True, env={**os.environ, **(env or {})}
        )
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def ref_exists(self, ref: str) -> bool:
        return self.git("rev-parse", "-q", "--verify", f"{ref}^{{commit}}", check=False) != ""


def spec_path_in(checkout: Path, name: str) -> Path:
    return checkout / "agent_orch" / "campaigns" / name / "dream.yaml"


def load_campaign(name: str, repo: Path | None = None) -> Campaign:
    """Load a campaign by name from this checkout (or `repo`), whose agent_orch/campaigns/<name>/dream.yaml defines it."""
    candidates = [spec_path_in(repo, name)] if repo else []
    candidates.append(spec_path_in(ORCH_DIR.parent, name))
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"no dream.yaml for campaign '{name}' (looked in {', '.join(map(str, candidates))})")
    cfg = load_spec(path)
    cfg["name"] = name
    return Campaign(name=name, cfg=cfg, repo=repo or ORCH_DIR.parent)


def parse_node_id(node_id: str) -> tuple[int, int, int]:
    """'r01-b03-a02' -> (1, 3, 2)"""
    r, b, a = node_id.split("-")
    assert r[0] == "r" and b[0] == "b" and a[0] == "a", node_id
    return int(r[1:]), int(b[1:]), int(a[1:])


def node_id(rnd: int, branch: int, attempt: int) -> str:
    return f"r{rnd:02d}-b{branch:02d}-a{attempt:02d}"


def node_from_ref(ref: str) -> str | None:
    """Node id from a round-root ref ('refs/dream/c/n/r01-b04-a04' or legacy 'dream/c/n/r01-b04-a04')."""
    return ref.rsplit("/", 1)[1] if "/n/" in ref else None
