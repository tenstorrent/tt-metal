#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFB repair loop: one nightly run.

Refreshes the bot-only tt-metal checkout to origin/<base> plus the carried branches, builds unit_tests_api, runs every
Quasar DFB test on the ttsim simulator (one process per test, several in parallel), and groups the failures.

Modes:
  report  build, test and report; no model calls.
  repair  also give each new failure group to Claude Code. A fix counts as verified only when its saved patch, applied
          to a clean tree, makes the failing tests pass and a full-suite rerun shows no test getting worse. Fixes stay
          on disk under runs/<stamp>/fixes/.
  pr      repair, then push each verified fix to its own branch and open a draft PR. Without GitHub CLI login the
          branch is pushed and the summary gives a link to the PR form (choose "Create draft pull request" there). A
          verified fix that could not be published on an earlier night is re-verified on tonight's tree and published,
          without calling the model.

Exit codes: 0 no failures except known issues, 1 failures remain, 2 the run could not judge the tests (build,
simulator or the agent's service).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import fnmatch
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

EXIT_GREEN, EXIT_FAILURES, EXIT_INFRA = 0, 1, 2
OK_VERDICTS = ("PASS", "SKIP")
NOT_RUN = "not run: the run's deadline was reached"
# Errors on the service side (login, access, quota, outage): the night stops giving groups to the agent.
# Claude Code's result text when it could not work at all (no login, no access, an outage).
AGENT_DOWN_RX = re.compile(
    r"^API Error|\b401\b|\b403\b|\b429\b|\b5\d\d\b|authenticat|oauth|/login|invalid api key|does not support this "
    r"model|credit balance|usage limit|out of (extra )?usage|usage credit|you'?ve (hit|reached) your|rate.?limit|"
    r"overloaded|connection (error|refused|reset)|ECONNRESET|ETIMEDOUT|socket hang up",
    re.I,
)
# The Copilot CLI's error text for the same. Its HTTP status and error type are checked on their own, so no bare
# numbers here: every final model error reads "Failed to get response from the AI model; ... Last error: <...>".
COPILOT_DOWN_RX = re.compile(
    r"requested model is not supported|access denied by policy|invalid access to inference|no authentication "
    r"information|authentication failed|not authorized|quota exceeded|exceeded your quota|too many requests|"
    r"Copilot Requests|no AI credits pricing|run out of your|multiple GitHub Copilot licenses|payment required|"
    r"you'?ve (hit|reached) your|rate.?limit|overloaded|Failed native model HTTP request|connection (error|refused|"
    r"reset)|ECONNRESET|ETIMEDOUT|socket hang up",
    re.I,
)
COPILOT_DOWN_TYPES = ("authentication", "authorization", "quota", "rate_limit")  # session.error types of an outage
ERROR_LINE_RX = re.compile(
    r"error|TT_FATAL|TT_THROW|Failure|Expected|Which is|Value of|Actual|critical|Unimplemented|Unpredictable|"
    r"assert|abort|Segmentation|terminate|exception",
    re.I,
)
COMPILE_RX = re.compile(
    r"^(?:\S*/)?([\w.+-]+\.(?:cpp|cc|c|h|hpp|inl|ld)):(\d+):(?:\d+:)? (?:fatal )?error: (.+)$", re.M
)
LINK_RX = re.compile(
    r"\b(?:ld|collect2)[\w.-]*: (?:error: )?([^\n]*(?:overflowed|undefined reference|cannot find|"
    r"multiple definition|relocation truncated)[^\n]*)"
)
SIM_RX = re.compile(r"\b(Unimplemented\w*|UnpredictableValue\w*|NotImplemented\w*)\b\W*([^\n]{0,100})")
FATAL_RX = re.compile(r"TT_(FATAL|THROW) @ (?:\S*/)?([\w.+-]+):(\d+): ([^\n]+)")
EXPECT_RX = re.compile(r"^(?:\S*/)?([\w.+-]+\.(?:cpp|hpp|h)):(\d+): Failure\s*$", re.M)
CRITICAL_RX = re.compile(r"\| critical \|[^|\n]*\| ([^\n]+)")
STATE_FIX_KEYS = (
    "handoff_at",
    "handoff_file",
    "fixed_at",
    "fix_dir",
    "fix_verified",
    "pr_branch",
    "pr_url",
    "queued_branch",
    "fixed_by",
)
MODEL_KEYS = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "COPILOT_GITHUB_TOKEN")  # given only to the agent
GITHUB_KEYS = ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN")  # given only to gh


class Infra(Exception):
    """The run cannot judge the tests (build, simulator or agent service problem)."""


class NotPublished(Exception):
    """A verified fix could not be published; it stays on disk."""


@dataclass
class Result:
    test: str
    verdict: str  # PASS, FAIL, SKIP, TIMEOUT, CRASH, NORESULT
    rc: int
    seconds: float
    log: str
    signature: str = ""  # shown to people; keeps file:line
    key: str = ""  # used for grouping; no line numbers


@dataclass
class Group:
    gid: str
    signature: str
    kind: str  # repairable, known, sim_gap
    tests: list
    known: str = ""
    seen_before: bool = False
    prior_gid: str = ""
    status: str = ""
    attempts: list = field(default_factory=list)
    fix: dict = field(default_factory=dict)
    fixed_by: str = ""
    handed_off: bool = False
    held: bool = False  # tonight's status comes from an earlier night's fix, PR or handoff
    promote: str = ""  # a fix branch found on origin tonight, recorded as the group's PR branch


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers


def as_float(value) -> float:
    """A number from an agent's report, or 0 for anything that is not one."""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def read_text(path, limit: int | None = None) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ""
    if limit and len(data) > limit:
        data = data[-limit:]
    return data.decode("utf-8", "replace")


def tail_lines(path, n: int) -> str:
    return "\n".join(read_text(path, 1_000_000).splitlines()[-n:])


def safe_name(test: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", test)


def norm(s: str, all_digits: bool = False) -> str:
    s = re.sub(r"(?:/[\w.+-]+)+/([\w.+-]+)", r"\1", s)  # paths become basenames
    s = re.sub(r"0x[0-9a-fA-F]+", "0xN", s)
    s = re.sub(r"\d+" if all_digits else r"\b\d{2,}\b", "N", s)
    return re.sub(r"\s+", " ", s).strip()[:200]


def gid_of(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, data) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def days_since(iso) -> float:
    try:
        return (utc_now() - dt.datetime.fromisoformat(iso)).total_seconds() / 86400
    except (TypeError, ValueError):
        return 1e9


def sanitize(text: str) -> str:
    """Text going public: no @mentions that ping people, no references that would close issues."""
    text = re.sub(r"(?<![\w`.])@([A-Za-z0-9][A-Za-z0-9-]*)", r"\1", text)
    ref = r"(#\d+|[\w.-]+/[\w.-]+#\d+|https?://github\.com/[^\s/]+/[^\s/]+/(?:issues|pull)/\d+)"
    return re.sub(
        rf"\b(close[sd]?|fix(?:e[sd])?|resolve[sd]?)(\s*:?\s+){ref}",
        lambda m: f"{m.group(1)}{m.group(2)}issue {m.group(3).lstrip('#')}",
        text,
        flags=re.I,
    )


def sig_family(sig: str) -> tuple[str, str]:
    """The kind of check and its file, e.g. ('EXPECT', 'dfb_test_common.hpp'): a moved or reworded check keeps them."""
    words = sig.split()
    return (words[0] if words else "", words[1].split(":")[0] if len(words) > 1 else "")


def md_cell(text: str) -> str:
    """Text for one markdown table cell: on one line, with its pipes escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def kill_marked(marker: str) -> int:
    """Kill every process of ours whose start environment holds `marker` (KEY=value). Survives setsid."""
    want, me, n = marker.encode(), os.getpid(), 0
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(f"/proc/{d}/environ", "rb") as fh:
                if want in fh.read().split(b"\0"):
                    os.kill(int(d), signal.SIGKILL)
                    n += 1
        except (OSError, ProcessLookupError):
            pass
    return n


def parse_verdict(text: str, test: str, rc: int) -> str:
    t = re.escape(test)
    if re.search(rf"^\[\s+OK\s+\] {t} \(", text, re.M):
        # A test that passes but whose process does not exit cleanly still has a problem (teardown crash or hang).
        return "PASS" if rc == 0 else ("TIMEOUT" if rc in (124, 137, -9) else "CRASH")
    if re.search(rf"^\[\s+SKIPPED\s+\] {t}(\s|,|$)", text, re.M):
        return "SKIP"
    if re.search(rf"^\[\s+FAILED\s+\] {t}(\s|,|$)", text, re.M):
        return "FAIL"
    if rc in (124, 137, -9):
        return "TIMEOUT"
    return "CRASH" if rc != 0 else "NORESULT"


def test_section(text: str, test: str) -> str:
    """The part of a shared gtest log that belongs to one test."""
    start = text.find(f"[ RUN      ] {test}\n")
    if start < 0:
        return ""
    m = re.compile(rf"^\[\s+(OK|FAILED|SKIPPED)\s+\] {re.escape(test)}(\s|,|$).*$", re.M).search(text, start)
    return text[start : m.end()] if m else text[start:]


def expect_block(text: str, end: int) -> list[str]:
    """The lines gtest prints after '<file>:<line>: Failure', up to the next log or gtest line."""
    out = []
    for line in text[end:].lstrip("\n").splitlines()[:14]:
        s = line.strip()
        if (
            not s
            or s.startswith("[ ")
            or re.match(r"\d{4}-\d\d-\d\d ", s)
            or s.startswith("unknown file")
            or EXPECT_RX.match(line)
        ):
            break
        out.append(s)
    return out


def failure_signature(text: str, verdict: str, rc: int) -> tuple[str, str]:
    """(display, key). The display keeps file:line for people. The key leaves line numbers out, so a group keeps its
    id when lines move, and carries the failure's own message, so unrelated failures at one shared check stay apart."""
    m = COMPILE_RX.search(text)
    if m:
        f, line, msg = m.groups()
        return f"COMPILE {f}:{line} {norm(msg)}", f"COMPILE {f} {norm(msg, True)}"
    m = LINK_RX.search(text)
    if m:
        return f"LINK {norm(m.group(1))}", f"LINK {norm(m.group(1), True)}"
    m = SIM_RX.search(text)
    if m:
        return f"SIM {m.group(1)} {norm(m.group(2))}", f"SIM {m.group(1)} {norm(m.group(2), True)}"
    m = FATAL_RX.search(text)
    if m:
        kind, f, line, cond = m.groups()
        after = text[m.end() :].lstrip("\n").splitlines()[:6]
        info = ""
        for i, s in enumerate(after):
            if s.strip() == "info:" and i + 1 < len(after):
                info = after[i + 1].strip()
                break
        return (
            f"TT_{kind} {f}:{line} {norm(cond)}" + (f" :: {norm(info)}" if info else ""),
            f"TT_{kind} {f} {norm(cond, True)} :: {norm(info, True)}",
        )
    m = EXPECT_RX.search(text)
    if m:
        f, line = m.groups()
        keep = []
        for s in expect_block(text, m.end()):
            if s.startswith(("Which is:", "Actual:")) or s in ("Expected: true", "Expected: false"):
                continue
            keep.append(re.sub(r", actual: .*$", "", s))
        summary = " | ".join(keep)
        return f"EXPECT {f}:{line} {norm(summary)}", f"EXPECT {f} {norm(summary, True)}"
    m = CRITICAL_RX.search(text)
    if m:
        return f"CRITICAL {norm(m.group(1))}", f"CRITICAL {norm(m.group(1), True)}"
    if verdict == "TIMEOUT":
        return "TIMEOUT (no error before the time limit)", "TIMEOUT"
    if verdict == "CRASH":
        return f"CRASH (exit code {rc}, no error line)", f"CRASH {rc}"
    return f"{verdict} (no error line found)", verdict


def assertion_statements(text: str, rx: re.Pattern) -> Counter:
    """Every assertion statement in a source file, from its macro to the closing ';', with whitespace removed, so an
    edit anywhere in a wrapped assertion counts as a change."""
    out: Counter = Counter()
    for m in rx.finditer(text):
        i, depth, started = m.start(), 0, False
        while i < len(text):
            c = text[i]
            if c == "(":
                depth, started = depth + 1, True
            elif c == ")":
                depth -= 1
            elif c == ";" and started and depth <= 0:
                break
            i += 1
        out[re.sub(r"\s+", "", text[m.start() : i + 1])] += 1
    return out


def tree_fingerprint(paths) -> str:
    h = hashlib.sha256()
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
        for p in files:
            if "__pycache__" in p.parts:
                continue
            h.update(str(p).encode())
            try:
                h.update(p.read_bytes())
            except OSError:
                pass
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------


class Loop:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root).resolve()
        self.run_dir = Path(args.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ompi_dir = self.run_dir / "cache" / "ompi"  # Open MPI's per-process files; removed with the run's cache
        self.ompi_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.run_dir / "cache" / "tmp"  # other children's temporary files; removed with the run's cache
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        live_tool = Path(args.tool_dir).resolve() if args.tool_dir else self.root / "tool"
        # The night runs from its own copy of the tooling, so edits to tool/ during a run take effect the next night.
        self.tool_dir = self.run_dir / "tool"
        shutil.copytree(live_tool, self.tool_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        self.live_tool = live_tool
        self.tool_bytes = {p: p.read_bytes() for p in self.tool_dir.rglob("*") if p.is_file()}
        self.cfg = yaml.safe_load((self.tool_dir / "repair_budget.yaml").read_text())
        for var, section, key in (("DFB_PUSH_URL", "pr", "push_url"), ("DFB_BRANCH_PREFIX", "pr", "branch_prefix")):
            if os.environ.get(var):  # CI overrides
                self.cfg[section][key] = os.environ[var]
        self.mode = args.mode
        self.checkout = Path(args.checkout).resolve() if args.checkout else self.root / "tt-metal"
        self.bin = self.root / "bin" / "dfbloop_api_suite"  # a name the emulator runners' patterns never match
        self.marker = f"DFB_LOOP_CHILD={self.run_dir.name}"  # every child carries it, so leftovers can be found
        self.t0 = time.time()
        self.deadline = self.t0 + 60 * float(args.deadline_minutes or self.cfg["budget"]["deadline_minutes"])
        self._log = open(self.run_dir / "orchestrator.log", "a", buffering=1)
        self._lock = threading.Lock()
        self.state_path = Path(args.state).resolve() if args.state else self.root / "state.json"
        self.state = load_json(self.state_path, {"groups": {}, "runs": []})
        self.notes: list[str] = []
        self.carried: list[str] = []
        self.no_publish = ""  # set when tonight's fixes must stay on disk, with the reason
        self.repair_stop = ""  # set when repairs must stop for the night, with the reason
        self.base_sha = ""
        self.target_sha = ""
        self.sim_version = ""
        self.sim_dir = Path(args.sim_dir).resolve() if args.sim_dir else self.run_dir / "sim"
        self.sim_bytes: dict[Path, bytes] = {}
        self.git_files: dict[str, bytes] = {}
        self.manifest: dict = {"device": [], "host": []}
        self.baseline: dict[str, Result] = {}
        self.hard: dict[str, Result] = {}
        self.flaky: list[str] = []
        self.groups: list[Group] = []
        self.current_gids: set[str] = set()
        self.outbox_done = False  # open_prs wrote the outbox; otherwise run() clears it, so nothing unchecked goes out
        self.classified = False
        self.verified_full: dict[str, dict[str, Result]] = {}
        self.attempts_used = 0
        self.cost_usd = 0.0  # Claude Code reports dollars
        self.ai_credits = 0.0  # the Copilot CLI reports AI credits and premium requests
        self.premium_requests = 0.0
        self.claude_down = ""
        self.phases: dict[str, float] = {}

    # ----- plumbing --------------------------------------------------------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"{dt.datetime.now().strftime('%H:%M:%S')} {msg}"
        with self._lock:
            print(line, flush=True)
            self._log.write(line + "\n")

    def phase(self, name: str, started: float) -> None:
        self.phases[name] = round((time.time() - started) / 60, 1)

    def minutes_left(self) -> float:
        return (self.deadline - time.time()) / 60

    def env(self, cache: Path | None = None, model: bool = False, github: bool = False) -> dict:
        """A child's environment. The model credential goes only to the agent, GitHub tokens only to gh."""
        drop = (
            "TT_METAL_",
            "CLAUDE",
            "ANTHROPIC_",
            "ARCH_NAME",
            "TT_MESH_",
            "TT_VISIBLE_DEVICES",
            "DFB_",
            "ACTIONS_",
            "GITHUB_",
            "RUNNER_",
            "COPILOT_",
            "GH_",
            "XDG_",
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith(drop) and k not in MODEL_KEYS + GITHUB_KEYS}
        for k in (self.model_keys() if model else ()) + (GITHUB_KEYS if github else ()):
            if os.environ.get(k):
                env[k] = os.environ[k]
        b = self.cfg["build"]
        env.update(
            GIT_TERMINAL_PROMPT="0",
            GIT_EDITOR="true",
            # No global or system git config: nothing written there (say, a core.fsmonitor command) runs in our git.
            GIT_CONFIG_GLOBAL="/dev/null",
            GIT_CONFIG_NOSYSTEM="1",
            TT_METAL_HOME=str(self.checkout),
            TT_METAL_RUNTIME_ROOT=str(self.checkout),
            CCACHE_DIR=str(self.root / b.get("ccache_dir", "ccache")),
            CCACHE_MAXSIZE=str(b.get("ccache_maxsize", "20G")),
            # No ccache.conf is read (one planted in the cache directory could name another compiler), and the
            # settings that make ccache run a command are fixed here.
            CCACHE_CONFIGPATH=os.devnull,
            CCACHE_PREFIX="",
            CCACHE_PREFIX_CPP="",
            CCACHE_COMPILERCHECK="mtime",
        )
        if getattr(self, "tmp_dir", None):  # temporary files of builds, git and gh stay in the run's cache
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            env["TMPDIR"] = str(self.tmp_dir)
        if github:  # the bot's own gh login lives in the loop's directory, not in the user's ~/.config/gh
            env.update(GH_CONFIG_DIR=str(self.root / "gh-config"), GH_NO_UPDATE_NOTIFIER="1", GH_PROMPT_DISABLED="1")
        k, v = self.marker.split("=", 1)
        env[k] = v
        if cache is not None:
            env.update(
                TT_METAL_SIMULATOR_HOME=str(self.sim_dir),
                TT_METAL_SIMULATOR=str(self.sim_dir / "libttsim.so"),
                TT_METAL_CACHE=str(cache),
            )
            # Each test process leaves an Open MPI shared-memory file and a session directory behind; keep both in the
            # run's cache (deleted after the run) instead of /dev/shm and /tmp.
            env.update(OMPI_MCA_btl_sm_backing_directory=str(self.ompi_dir), TMPDIR=str(self.ompi_dir))
            env.update({k: str(v) for k, v in (self.cfg["sim"].get("env") or {}).items()})
        return env

    def sh(self, cmd, cwd=None, env=None, timeout: float = 600, out: Path | None = None) -> tuple[int, str]:
        """Run a command. Returns (rc, output); with `out`, output is appended to that file and its tail returned."""
        cwd = str(cwd or self.checkout)
        env = env or self.env()
        try:
            if out is not None:
                with open(out, "ab") as fh:
                    rc = subprocess.run(
                        cmd,
                        cwd=cwd,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                    ).returncode
                return rc, tail_lines(out, 60)
            p = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                text=True,
                errors="replace",
            )
            return p.returncode, p.stdout
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {int(timeout)} s: {' '.join(map(str, cmd))[:200]}"

    def git(self, *args, check: bool = True, timeout: float = 600) -> tuple[int, str]:
        """The bot's git never runs hooks, so nothing an agent leaves in .git/hooks runs with the bot's authority."""
        rc, out = self.sh(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *args], timeout=timeout
        )
        if check and rc != 0:
            raise Infra(f"git {' '.join(args)[:120]} failed (rc={rc}): {out.strip()[-600:]}")
        return rc, out.strip()

    def kill_strays(self) -> None:
        """Stop every process this run started that is still alive (they all carry the run's marker)."""
        kill_marked(self.marker)

    def clear_git_locks(self) -> int:
        """Remove lock files a killed git process left. Only safe when no bot or agent git process is running."""
        n = 0
        for p in list((self.checkout / ".git").rglob("*.lock")):
            if p.is_file():
                p.unlink(missing_ok=True)
                n += 1
        return n

    # ----- integrity of what the agent must not touch ------------------------------------------------------------

    def protected_fingerprint(self) -> str:
        return tree_fingerprint([self.tool_dir, self.sim_dir / "libttsim.so", self.sim_dir / "soc_descriptor.yaml"])

    def restore_protected(self) -> None:
        shutil.rmtree(self.tool_dir, ignore_errors=True)
        for p, data in self.tool_bytes.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            if p.name.endswith(".sh"):
                p.chmod(0o755)
        for p, data in self.sim_bytes.items():  # an install from --sim-dir may be read-only, and then needs no restore
            try:
                if p.read_bytes() != data:
                    p.write_bytes(data)
            except OSError as e:
                self.notes.append(f"could not restore {p}: {e}")

    def live_fingerprint(self) -> str:
        names = ("nightly.sh", "loop.sh", "ensure_loop.sh", "copilot_login.sh", "summarize.py", "claude.env", "HOST")
        return tree_fingerprint([self.live_tool, self.root / "logs" / "bootstrap.sh"] + [self.root / n for n in names])

    def save_git_files(self) -> None:
        g = self.checkout / ".git"
        for rel in ("config", "info/exclude", "info/attributes"):
            p = g / rel
            if p.is_file():
                self.git_files[rel] = p.read_bytes()

    def restore_git_files(self) -> None:
        g = self.checkout / ".git"
        if "info/attributes" not in self.git_files:
            (g / "info" / "attributes").unlink(missing_ok=True)
        for rel, data in self.git_files.items():
            (g / rel).parent.mkdir(parents=True, exist_ok=True)
            (g / rel).write_bytes(data)
        hooks = g / "hooks"
        if hooks.is_dir():
            for h in hooks.iterdir():
                if h.is_file() and not h.name.endswith(".sample"):
                    h.unlink()

    def cmake_cache_fingerprint(self) -> str:
        return tree_fingerprint([self.checkout / self.cfg["build"]["dir"] / "CMakeCache.txt"])

    def ninja_fingerprint(self) -> str:
        bdir = self.checkout / self.cfg["build"]["dir"]
        return tree_fingerprint([bdir / "build.ninja", bdir / "CMakeFiles" / "rules.ninja"])

    def kept_trees_changed(self, stamp: Path) -> list[str]:
        """Top directories of the toolchain and package caches (kept between attempts) with files newer than `stamp`."""
        roots = [
            self.checkout / p for p in self.cfg["build"].get("watched_ignored") or [] if (self.checkout / p).exists()
        ]
        if not roots:
            return []
        _, out = self.sh(
            [
                "find",
                *map(str, roots),
                "-newer",
                str(stamp),
                "-type",
                "f",
                "-not",
                "-path",
                "*-build/*",
                "-not",
                "-path",
                "*-subbuild/*",
            ],
            timeout=300,
        )
        tops = set()
        for line in out.splitlines()[:5000]:
            p = Path(line.strip())
            for r in roots:
                if r in p.parents:
                    rel = p.relative_to(r).parts
                    tops.add(str(r / rel[0]) if rel and r.name in (".cpmcache", "_deps") else str(r))
        return sorted(tops)

    def drop_generated(self) -> None:
        """Delete what the build generates into the tree, so the next build regenerates it from the current sources."""
        for p in self.cfg["build"].get("generated_paths") or []:
            shutil.rmtree(self.checkout / p, ignore_errors=True)

    # ----- checkout --------------------------------------------------------------------------------------------------

    def reset_tree(self, ref: str, deep: bool = False) -> None:
        self.clear_git_locks()  # callers make sure no bot or agent git process is running
        gitdir = self.checkout / ".git"
        for marker, cmd in (("CHERRY_PICK_HEAD", "cherry-pick"), ("MERGE_HEAD", "merge"), ("REVERT_HEAD", "revert")):
            if (gitdir / marker).exists():
                self.git(cmd, "--abort", check=False)
        if (gitdir / "rebase-merge").exists() or (gitdir / "rebase-apply").exists():
            self.git("rebase", "--abort", check=False)
        self.git("reset", "-q", "--hard")
        self.git("checkout", "-q", "--detach", ref)
        self.git("reset", "-q", "--hard", ref)
        if deep:  # also remove ignored files, except the build and what it keeps in the tree
            keep = []
            for p in self.cfg["build"].get("keep_ignored") or []:
                keep += ["-e", "/" + p.strip("/")]
            self.git("clean", "-q", "-ffdx", *keep)
        else:
            self.git("clean", "-q", "-fd")
        self.git(
            "submodule",
            "foreach",
            "-q",
            "--recursive",
            "git reset -q --hard && git clean -q -fd",
            check=False,
            timeout=900,
        )
        self.git("submodule", "update", "-q", "--init", "--recursive", "--force", check=False, timeout=1800)

    def prepare_checkout(self) -> None:
        tcfg = self.cfg["target"]
        base = tcfg.get("base", "main")
        gitdir = self.checkout / ".git"
        n = self.clear_git_locks()  # left by a run that was killed; the run lock rules out any other bot git process
        if n:
            self.notes.append(f"removed {n} stale git lock file(s) left by an earlier run")
        hooks = gitdir / "hooks"
        if hooks.is_dir():
            for h in hooks.iterdir():
                if h.is_file() and not h.name.endswith(".sample"):
                    h.unlink()
        exclude = gitdir / "info" / "exclude"
        have = read_text(exclude).splitlines()
        missing = [p for p in (".claude/", "CLAUDE.local.md") if p not in have]
        if missing:  # files Claude Code may create in the project are never part of a fix
            exclude.parent.mkdir(parents=True, exist_ok=True)
            with open(exclude, "a") as fh:
                fh.write("\n" + "\n".join(missing) + "\n")
        if not self.args.skip_fetch:
            self.git("fetch", "-q", "--prune", "origin", f"+refs/heads/{base}:refs/remotes/origin/{base}", timeout=1800)
        ref = self.args.ref or f"origin/{base}"
        self.reset_tree(ref)
        self.base_sha = self.git("rev-parse", "HEAD")[1]
        for branch in [] if self.args.no_carry else (tcfg.get("carry") or []):
            fetched_before = (
                self.git("rev-parse", "-q", "--verify", f"refs/remotes/origin/{branch}", check=False)[0] == 0
            )
            if not self.args.skip_fetch:
                rc, _ = self.git(
                    "fetch",
                    "-q",
                    "origin",
                    f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                    check=False,
                    timeout=900,
                )
                if rc != 0:
                    if not fetched_before:
                        self.drop_carry(branch, "it could not be fetched from origin")
                        continue
                    self.notes.append(
                        f"carried branch `{branch}` could not be fetched (deleted on origin?), so the copy "
                        f"fetched on an earlier night was applied; remove it from target.carry once it "
                        f"is merged"
                    )
            rc, out = self.git("cherry", "HEAD", f"origin/{branch}", check=False)
            if rc != 0:
                self.drop_carry(branch, "its commits could not be listed")
                continue
            before, picked = self.git("rev-parse", "HEAD")[1], []
            for line in out.splitlines():
                sign, sha = line.split()[:2]
                subject = self.git("log", "-1", "--format=%s", sha)[1][:90]
                if sign == "-":
                    self.notes.append(
                        f"carried commit {sha[:11]} ({subject}) is already on {base}; remove its branch "
                        f"from target.carry once it is merged"
                    )
                    continue
                rc, _ = self.git("cherry-pick", "-x", sha, check=False)
                if rc != 0:
                    unmerged = self.git("diff", "--name-only", "--diff-filter=U", check=False)[1]
                    if not unmerged and self.git("diff", "--cached", "--quiet", check=False)[0] == 0:
                        self.git("cherry-pick", "--skip", check=False)  # nothing left to apply: base has its changes
                        self.notes.append(
                            f"carried commit {sha[:11]} ({subject}) is already on {base} (merged with "
                            f"changes); remove its branch from target.carry"
                        )
                        continue
                if rc != 0:  # one commit of the branch no longer applies: skip the whole branch
                    self.git("cherry-pick", "--abort", check=False)
                    self.git("reset", "-q", "--hard", before)
                    self.drop_carry(branch, f"it no longer applies to {base} (at {sha[:11]} {subject})")
                    picked = []
                    break
                picked.append(f"{sha[:11]} {subject}")
            self.carried += picked
        self.git("submodule", "sync", "-q", "--recursive", check=False)
        self.git("submodule", "update", "-q", "--init", "--recursive", timeout=3600)
        self.target_sha = self.git("rev-parse", "HEAD")[1]
        # Pushes go only through pr.push_url, so a stray `git push` in the checkout fails.
        self.git("remote", "set-url", "--push", "origin", "push-disabled-by-dfb-loop")
        self.save_git_files()

    def drop_carry(self, branch: str, why: str) -> None:
        self.notes.append(f"carried branch `{branch}` was not applied tonight: {why}")
        # Its fixes are missing tonight, so the agent may re-fix the same bugs; nothing is published until it is back.
        self.no_publish = f"the carried branch {branch} was dropped tonight, so a fix could duplicate it"

    # ----- simulator and build ---------------------------------------------------------------------------------------

    def install_sim(self) -> None:
        s = self.cfg["sim"]
        self.sim_version = (self.checkout / s["version_file"]).read_text().strip()
        if self.args.sim_dir:  # an existing install (CI's setup-ttsim) is used as is; keep a copy to undo agent edits
            pinned = read_text(self.sim_dir / "ttsim-version").strip()
            if pinned and pinned != self.sim_version:
                raise Infra(
                    f"the simulator in {self.sim_dir} is ttsim {pinned}, but the tested tree pins {self.sim_version}"
                )
            self.sim_bytes = {
                p: p.read_bytes()
                for p in (self.sim_dir / "libttsim.so", self.sim_dir / "soc_descriptor.yaml")
                if p.is_file()
            }
            return
        cache = self.root / "sim" / self.sim_version
        lib = cache / s["lib"]
        if not lib.exists():
            cache.mkdir(parents=True, exist_ok=True)
            url = s["url"].format(version=self.sim_version, lib=s["lib"])
            tmp = cache / (s["lib"] + ".part")
            rc, out = self.sh(["curl", "-fsSL", "--retry", "3", "-o", str(tmp), url], cwd=self.root, timeout=900)
            if rc != 0 or not tmp.exists() or tmp.stat().st_size < 10_000:
                raise Infra(f"cannot download ttsim {self.sim_version} from {url}: {out.strip()[-300:]}")
            tmp.replace(lib)
        self.sim_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lib, self.sim_dir / "libttsim.so")
        shutil.copy2(self.checkout / s["soc_descriptor"], self.sim_dir / "soc_descriptor.yaml")
        self.sim_bytes = {
            p: p.read_bytes() for p in (self.sim_dir / "libttsim.so", self.sim_dir / "soc_descriptor.yaml")
        }

    def write_loop_env(self) -> None:
        """Shell file the agent's run_test.sh sources, so it runs tests exactly like the loop does."""
        also = (self.marker.split("=")[0], "OMPI_MCA_btl_sm_backing_directory", "TMPDIR")
        lines = [
            f"export {k}={json.dumps(v)}"
            for k, v in sorted(self.env(Path("/unused")).items())
            if (k.startswith("TT_METAL_") and k != "TT_METAL_CACHE") or k in also
        ]
        lines += [
            f"export DFB_BIN={json.dumps(str(self.bin))}",
            f"export DFB_TEST_TIMEOUT={int(self.cfg['suite']['test_timeout_seconds'])}",
        ]
        (self.run_dir / "loop_env.sh").write_text("\n".join(lines) + "\n")

    def configure(self, log_path: Path, fresh: bool = False) -> bool:
        b = self.cfg["build"]
        bdir = self.checkout / b["dir"]
        if fresh:
            (bdir / "CMakeCache.txt").unlink(missing_ok=True)
            shutil.rmtree(bdir / "CMakeFiles", ignore_errors=True)
        cmd = [c.replace("{checkout}", str(self.checkout)) for c in b["configure"]]
        rc, _ = self.sh(cmd, out=log_path, timeout=60 * float(b["timeout_minutes"]))
        link = self.checkout / "build"
        if rc == 0 and not link.exists():
            link.symlink_to(b["dir"])
        return rc == 0

    def build(self, log_path: Path) -> bool:
        b = self.cfg["build"]
        bdir = self.checkout / b["dir"]
        limit = 60 * float(b["timeout_minutes"])

        def make() -> bool:
            rc, _ = self.sh(
                ["timeout", "-k", "60", str(int(limit)), "cmake", "--build", str(bdir), "--target", b["target"]],
                out=log_path,
                timeout=limit + 120,
            )
            return rc == 0

        if not (bdir / "build.ninja").exists() and not self.configure(log_path):
            return False
        if make():
            return True
        text = read_text(log_path, 4_000_000)
        if re.search(r"SFPI packaging malfunction|GPP_CMD path .* does not exist", text):
            self.log("build: stale SFPI download; clearing it and retrying")
            self.drop_sfpi()
            return make()
        if re.search(r"CMake Error|No rule to make target|rebuilding 'build.ninja': subcommand failed", text):
            self.log("build: stale CMake state; reconfiguring from scratch")
            shutil.rmtree(bdir, ignore_errors=True)
            return self.configure(log_path) and make()
        return False

    def drop_sfpi(self) -> None:
        for stamp in (self.checkout / self.cfg["build"]["dir"] / "CMakeFiles" / "fc-stamp" / "sfpi").glob("*.stamp"):
            stamp.unlink()
        shutil.rmtree(self.checkout / "runtime" / "sfpi", ignore_errors=True)

    def ensure_bin(self) -> None:
        target = self.checkout / self.cfg["build"]["dir"] / "test" / "tt_metal" / self.cfg["build"]["target"]
        if not target.exists():
            raise Infra(f"the test binary is missing after the build: {target}")
        self.bin.parent.mkdir(parents=True, exist_ok=True)
        if self.bin.is_symlink() or self.bin.exists():
            self.bin.unlink()
        self.bin.symlink_to(target)

    # ----- manifest and test runs ------------------------------------------------------------------------------------

    def make_manifest(self) -> None:
        s = self.cfg["suite"]
        rc, out = self.sh(
            ["timeout", "-k", "10", "300", str(self.bin), "--gtest_list_tests"],
            env=self.env(self.run_dir / "cache" / "list"),
            timeout=360,
        )
        if rc != 0:
            raise Infra(f"--gtest_list_tests failed (rc={rc}): {out.strip()[-500:]}")
        names, suite = [], None
        for line in out.splitlines():
            if not line.strip():
                continue
            if not line.startswith(" "):
                cand = line.split("#")[0].strip()
                suite = cand if re.fullmatch(r"[\w/]+\.", cand) else None
            elif suite:
                names.append(suite + line.split("#")[0].strip())
        rx = re.compile(s["name_regex"])
        extra = set(s.get("extra_tests") or [])
        patterns = s.get("exclude") or []

        def wanted(t: str) -> bool:
            if "DISABLED_" in t:  # gtest's own rule: disabled tests do not run
                return False
            return bool(rx.search(t) or t in extra) and not any(fnmatch.fnmatchcase(t, p) for p in patterns)

        chosen = [t for t in names if wanted(t)]
        host = [t for t in chosen if t.split(".", 1)[1].startswith("CPU_")] if s.get("host_tests", True) else []
        device = [t for t in chosen if not t.split(".", 1)[1].startswith("CPU_")]
        self.manifest = {
            "count": len(device) + len(host),
            "device": device,
            "host": host,
            "missing_extra_tests": sorted(extra - set(names)),
            "binary_tests": len(names),
        }
        write_json(self.run_dir / "manifest.json", self.manifest)
        if self.manifest["missing_extra_tests"]:
            self.notes.append(
                f"{len(self.manifest['missing_extra_tests'])} configured extra tests are no longer in "
                f"the binary (renamed or removed): see manifest.json"
            )
        if not device:
            raise Infra("the manifest has no device tests (test binary or name filter problem)")
        self.log(f"manifest: {len(device)} device tests, {len(host)} host tests (of {len(names)} in the binary)")

    def result_of(self, test: str, text: str, rc: int, seconds: float, log: str) -> Result:
        v = parse_verdict(text, test, rc)
        if v in OK_VERDICTS:
            return Result(test, v, rc, seconds, log)
        display, key = failure_signature(text, v, rc)
        return Result(test, v, rc, seconds, log, display, key)

    def run_tests(self, tests, out_dir: Path, cache_root: Path, workers: int, label: str) -> dict[str, Result]:
        """Run each test in its own process on ttsim, `workers` at a time."""
        out_dir.mkdir(parents=True, exist_ok=True)
        limit = int(self.cfg["suite"]["test_timeout_seconds"])
        slots: queue.Queue = queue.Queue()
        for i in range(max(1, workers)):
            slots.put(i)
        done = [0]

        def one(test: str) -> Result:
            if time.time() > self.deadline:
                return Result(test, "NORESULT", 0, 0.0, "", NOT_RUN, NOT_RUN)
            w = slots.get()
            try:
                log = out_dir / (safe_name(test) + ".log")
                log.unlink(missing_ok=True)
                t1 = time.time()
                rc, _ = self.sh(
                    ["timeout", "-k", "30", str(limit), str(self.bin), f"--gtest_filter={test}"],
                    env=self.env(cache_root / f"w{w}"),
                    timeout=limit + 120,
                    out=log,
                )
                res = self.result_of(test, read_text(log, 4_000_000), rc, round(time.time() - t1, 1), str(log))
            finally:
                slots.put(w)
            with self._lock:
                done[0] += 1
                n = done[0]
            if n % 100 == 0 or n == len(tests):
                self.log(f"{label}: {n}/{len(tests)} done")
            return res

        results: dict[str, Result] = {}
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for res in ex.map(one, tests):
                results[res.test] = res
        return results

    def run_host_tests(self, tests, out_dir: Path, cache_root: Path) -> dict[str, Result]:
        """Host-only CPU_ tests run together in one process; any it did not finish run again one by one."""
        if not tests:
            return {}
        if time.time() > self.deadline:
            return {t: Result(t, "NORESULT", 0, 0.0, "", NOT_RUN, NOT_RUN) for t in tests}
        out_dir.mkdir(parents=True, exist_ok=True)
        log = out_dir / "host_tests.log"
        log.unlink(missing_ok=True)
        limit = int(self.cfg["suite"]["host_timeout_seconds"])
        self.sh(
            ["timeout", "-k", "30", str(limit), str(self.bin), "--gtest_filter=" + ":".join(tests)],
            env=self.env(cache_root / "host"),
            timeout=limit + 120,
            out=log,
        )
        text = read_text(log, 50_000_000)
        results, unfinished = {}, []
        for t in tests:
            sec = test_section(text, t)
            v = parse_verdict(sec, t, 0) if sec else "NORESULT"
            if v in ("PASS", "SKIP", "FAIL"):
                results[t] = self.result_of(t, sec, 0, 0.0, str(log))
            else:
                unfinished.append(t)
        if unfinished:
            results.update(
                self.run_tests(unfinished, out_dir / "solo", cache_root / "host-solo", 8, "host tests (one by one)")
            )
        return results

    def run_suite(self, out_dir: Path, label: str) -> dict[str, Result]:
        cache = self.run_dir / "cache" / f"{out_dir.parent.name}-{out_dir.name}"
        res = self.run_tests(self.manifest["device"], out_dir, cache, int(self.cfg["suite"]["workers"]), label)
        res.update(self.run_host_tests(self.manifest["host"], out_dir / "host", cache))
        shutil.rmtree(cache, ignore_errors=True)
        return res

    # ----- classification --------------------------------------------------------------------------------------------

    def known_entry(self, r: Result) -> dict | None:
        for k in self.cfg.get("known_failures") or []:
            if not any(fnmatch.fnmatchcase(r.test, p) for p in k.get("tests") or []):
                continue
            if k.get("signature") and not (re.search(k["signature"], r.signature) or re.search(k["signature"], r.key)):
                continue
            return k
        return None

    def classify(self) -> None:
        bad = [r for r in self.baseline.values() if r.verdict not in OK_VERDICTS and r.signature != NOT_RUN]
        not_run = [r.test for r in self.baseline.values() if r.signature == NOT_RUN]
        if not_run:
            self.notes.append(f"{len(not_run)} tests were not run because the run's deadline was reached")
        if bad:
            self.log(f"baseline: {len(bad)} tests did not pass; running each once more on its own")
            retry = self.run_tests(
                [r.test for r in bad],
                self.run_dir / "baseline_retry",
                self.run_dir / "cache" / "retry",
                min(8, len(bad)),
                "retry",
            )
            for r in bad:
                rr = retry[r.test]
                if rr.verdict in OK_VERDICTS:
                    self.flaky.append(r.test)
                else:
                    self.hard[r.test] = rr if rr.verdict != "NORESULT" else r
        if self.hard:
            n_device = max(1, len(self.manifest["device"]))
            top_key, top_n = Counter(r.key for r in self.hard.values()).most_common(1)[0]
            if len(self.hard) >= float(self.cfg["suite"]["infra_fraction"]) * n_device and top_n >= 0.5 * len(
                self.hard
            ):
                sample = next(r.signature for r in self.hard.values() if r.key == top_key)
                raise Infra(f"{len(self.hard)} of {n_device} device tests fail, most of them with: {sample}")
            buckets: dict[tuple, list] = {}
            for r in self.hard.values():
                k = self.known_entry(r)
                if k:
                    key = ("known", k["name"])
                elif r.key.startswith("SIM "):
                    key = ("sim_gap", r.key)
                else:
                    key = ("repairable", r.key)
                buckets.setdefault(key, []).append(r.test)
            ordered = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
            self.current_gids = {gid_of(f"{kind}:{key}") for (kind, key), _ in ordered}
            for (kind, key), tests in ordered:
                gid = gid_of(f"{kind}:{key}")
                display = Counter(self.hard[t].signature for t in tests).most_common(1)[0][0]
                prior_gid, _ = self.prior_state_for(gid, kind, tests, display)
                self.groups.append(
                    Group(
                        gid,
                        display,
                        kind,
                        sorted(tests),
                        known=key if kind == "known" else "",
                        seen_before=bool(prior_gid),
                        prior_gid=prior_gid,
                    )
                )
            self.log("failure groups: " + "; ".join(f"{g.gid} {g.kind} x{len(g.tests)}" for g in self.groups))
        self.classified = not not_run

    # ----- repair ----------------------------------------------------------------------------------------------------

    def remote_branch_state(self, branch: str) -> bool | None:
        """True if the branch is on origin, False if not, None if origin could not be asked."""
        rc, _ = self.sh(["git", "ls-remote", "--exit-code", "--heads", self.cfg["pr"]["push_url"], branch], timeout=120)
        return True if rc == 0 else False if rc == 2 else None

    def remote_fix_branches(self, gid: str) -> list[str] | None:
        """The bot's branches for this group on origin, from any night (None if origin could not be asked), so a lost
        or cleared state never leads to a second PR for the same failure."""
        p = self.cfg["pr"]
        rc, out = self.sh(["git", "ls-remote", "--heads", p["push_url"], f"{p['branch_prefix']}/*-{gid}"], timeout=120)
        if rc != 0:
            return None
        return [line.split("refs/heads/", 1)[1] for line in out.splitlines() if "refs/heads/" in line]

    def prior_state_for(self, gid: str, kind: str, tests, signature: str) -> tuple[str, dict]:
        """(gid, state) of this group on earlier nights: the same id, or else an open group that is not failing tonight,
        has the same kind of check in the same file, and shares at least half its tests (a moved or reworded check). A
        different failure of the same tests is a new group."""
        groups = self.state.get("groups", {})
        if gid in groups:
            return gid, groups[gid]
        if not re.fullmatch(r"[\w.+-]+\.\w+", sig_family(signature)[1]):  # crashes, timeouts...: only the same id
            return "", {}
        best, best_gid, best_overlap, now = {}, "", 0.0, set(tests)
        for old_gid, st in groups.items():
            prev = set(st.get("tests") or [])
            if (
                st.get("kind") != kind
                or st.get("resolved_at")
                or not prev
                or old_gid in self.current_gids
                or sig_family(st.get("signature", "")) != sig_family(signature)
            ):
                continue
            overlap = len(now & prev) / len(now | prev)
            if overlap > best_overlap:
                best, best_gid, best_overlap = st, old_gid, overlap
        return (best_gid, best) if best_overlap >= 0.5 else ("", {})

    def effective_state(self, g: Group) -> dict:
        """The group's prior state, followed through 'fixed by another group' to that group's fix and PR."""
        _, st = self.prior_state_for(g.gid, g.kind, g.tests, g.signature)
        if st.get("fixed_by"):
            fixer = self.state.get("groups", {}).get(st["fixed_by"], {})
            merged = dict(st)
            for k in ("fixed_at", "fix_dir", "fix_verified", "pr_branch", "pr_url", "queued_branch"):
                if fixer.get(k):
                    merged[k] = fixer[k]
            return merged
        return st

    def fix_is_history(self, st: dict) -> bool:
        """For a group that went away and came back: True when its earlier fix, PR or handoff no longer applies, because
        the fix is already in tonight's tree (it was merged) or the group was gone for longer than rehandoff_days."""
        if days_since(st.get("resolved_at")) >= float(self.cfg["repair"]["rehandoff_days"]):
            return True
        patch = Path(st.get("fix_dir") or "/nonexistent") / "patch.diff"
        return patch.is_file() and self.git("apply", "--check", "-R", str(patch), check=False)[0] == 0

    def retire(self, gid: str, st: dict, why: str) -> None:
        """Forget a group's earlier fix, PR and handoff; its bot branches stay on record so they are not taken for a
        new proposal."""
        retired = list(st.get("retired_branches") or [])
        retired += [b for b in (st.get("pr_branch"), st.get("queued_branch")) if b and b not in retired]
        found = self.remote_fix_branches(gid) or []
        retired += [b for b in found if b not in retired]
        for k in STATE_FIX_KEYS + ("resolved_at", "superseded_by"):
            st.pop(k, None)
        st["retired_branches"] = retired
        self.notes.append(f"{gid}: earlier fixes, PRs and handoffs no longer apply ({why})")

    def repair(self) -> None:
        r = self.cfg["repair"]
        days = float(r["rehandoff_days"])
        reserve = float(self.cfg["budget"]["final_suite_reserve_minutes"])
        candidates = []
        for g in self.groups:
            if g.kind == "known":
                g.status = "known issue; not given to the agent"
            elif g.kind == "sim_gap":
                g.status = "simulator gap; not given to the agent"
            elif self.mode == "report":
                g.status = "report mode; not attempted"
            else:
                candidates.append(g)
        candidates.sort(key=lambda g: (g.seen_before, -len(g.tests)))
        groups = self.state.setdefault("groups", {})
        handled_dir, retry_dir = self.root / "handled", self.root / "retry"
        handled = set(re.findall(r"[0-9a-f]{10}", os.environ.get("DFB_HANDLED", "")))  # CI: a repository variable
        retry = set(re.findall(r"[0-9a-f]{10}", os.environ.get("DFB_RETRY", "")))  # CI: a manual run's input
        if not self.args.state and self.mode != "report" and retry_dir.is_dir():  # nightly markers, used up here
            for m in sorted(retry_dir.iterdir()):
                retry.add(m.name)
                try:
                    shutil.rmtree(m) if m.is_dir() and not m.is_symlink() else m.unlink(missing_ok=True)
                except OSError as e:
                    self.notes.append(f"could not remove {m}: {e}")
        for gid in sorted(retry if self.mode != "report" else ()):  # forget the group's earlier fix, PR and handoff
            self.retire(
                gid,
                groups.setdefault(
                    gid, {"first_seen": utc_now().isoformat(timespec="seconds"), "times_seen": 0, "attempts_total": 0}
                ),
                "a person asked to retry",
            )
        tried = 0
        for g in candidates:
            fixer = next(
                (
                    gid
                    for gid, full in self.verified_full.items()
                    if all(t in full and full[t].verdict == "PASS" for t in g.tests)
                ),
                "",
            )
            if fixer:
                g.status, g.fixed_by = f"fixed by the {fixer} fix", fixer
                continue
            mine = {g.gid, g.prior_gid} - {""}  # with every earlier id of this group, through rewordings
            while more := {old for old, s in groups.items() if s.get("superseded_by") in mine} - mine:
                mine |= more
            hit = sorted(x for x in mine if x in handled or (handled_dir / x).exists())
            if hit:
                where = f"{handled_dir}/{hit[0]}" if (handled_dir / hit[0]).exists() else "DFB_REPAIR_HANDLED"
                earlier = "" if hit[0] == g.gid else f", an earlier id of this group"
                g.status, g.held = f"a person is handling it ({where}{earlier}); not attempted", True
                continue
            own = groups.get(g.gid, {})
            if own.get("resolved_at") and self.fix_is_history(own):  # it came back after its fix was merged
                self.retire(
                    g.gid, own, "the group came back after its earlier fix was merged or long after it " "went away"
                )
            st = self.effective_state(g)
            branch = st.get("pr_branch") or st.get("queued_branch")
            if branch:
                there = self.remote_branch_state(branch)
                if there is None:
                    g.status = "not attempted: origin could not be asked whether the earlier fix branch still exists"
                    continue
                if there or st.get("pr_branch"):
                    g.held = True
                    if there and not st.get("pr_branch"):
                        g.promote = branch  # queued for CI's publish job and pushed since: now it is the PR branch
                    link = st.get("pr_url") or branch
                    if there and "/compare/" in link:
                        g.status = f"fix branch pushed, but the bot opened no PR: {link}"
                    elif there:
                        g.status = f"fix already proposed: {link}"
                    else:  # a person deleted the branch, usually after closing the PR: do not propose it again
                        g.status = (
                            f"waiting for a human: the fix branch {branch} is gone from origin (deleted after its "
                            f"PR was closed or merged) and the tests still fail; mkdir -p {retry_dir} && touch "
                            f"{retry_dir / g.gid} lets the bot try again"
                        )
                    continue
                # queued on an earlier night but never pushed: verified and queued again below
            retired = set(groups.get(g.gid, {}).get("retired_branches") or [])
            found = self.remote_fix_branches(g.gid)
            if found is None:
                g.status = "not attempted: origin could not be asked whether a fix branch already exists"
                continue
            found = [b for b in found if b not in retired]
            if found:
                g.status, g.held, g.promote = f"fix already proposed: branch {found[0]} is on origin", True, found[0]
                continue
            fix_dir = Path(st.get("fix_dir") or "/nonexistent")
            if st.get("fixed_at") and (fix_dir / "patch.diff").exists():
                if self.mode == "pr" and st.get("fix_verified") and not st.get("pr_branch") and not self.no_publish:
                    if self.reverify_saved_fix(g, fix_dir):  # verified before but never published: publish it now
                        continue
                elif days_since(st["fixed_at"]) < days:
                    g.status, g.held = f"fix on disk since {st['fixed_at'][:10]}: {fix_dir}/patch.diff", True
                    continue
            if st.get("handoff_at") and days_since(st["handoff_at"]) < days:
                where = f": {st['handoff_file']}" if st.get("handoff_file") else ""
                g.status, g.held = f"handed off on {st['handoff_at'][:10]}; waiting for a human{where}", True
                continue
            if self.claude_down or self.repair_stop:
                g.status = f"not attempted: {self.claude_down or self.repair_stop}"
                continue
            if tried >= int(r["groups_per_run"]):
                g.status = "not attempted: group limit for this run"
                continue
            tried += 1
            for n in range(1, int(r["attempts_per_group"]) + 1):
                if self.attempts_used >= int(r["attempts_per_run"]):
                    g.status = g.status or "not finished: attempt limit for this run"
                    break
                if self.minutes_left() < 15 + reserve + 10:
                    g.status = g.status or "not finished: not enough time left"
                    break
                self.attempts_used += 1
                a = self.attempt(g, n)
                if a["outcome"] in ("prebuild_failed", "no_time"):
                    self.attempts_used -= 1  # the agent never ran
                    if a["outcome"] == "prebuild_failed":
                        self.repair_stop = "the unchanged tree did not rebuild before an attempt"
                    g.status = f"not finished: {a['reason']}"
                    break
                g.attempts.append(a)
                if a["outcome"] == "verified":
                    g.status = "fixed"
                    break
                if a["outcome"] in ("unverified", "out_of_time") and g.fix:
                    g.status = "candidate fix (not fully verified): " + a["reason"]
                    break
                if a["outcome"] == "agent_error":
                    self.claude_down = a["reason"]
                    break
                if a["outcome"] in ("handoff", "no_change"):
                    break  # the agent concluded it cannot fix this; another attempt rarely helps
            if g.status == "fixed" or g.status.startswith(("candidate fix", "not finished")):
                continue
            last = g.attempts[-1]["outcome"] if g.attempts else ""
            if last == "agent_error" or not g.attempts:
                g.status = g.status or f"not attempted: {self.claude_down or self.repair_stop}"
            elif last in ("handoff", "no_change") or len(g.attempts) >= int(r["attempts_per_group"]):
                self.write_handoff(g)
                g.handed_off = True
                g.status = "needs human help"
            else:
                g.status = f"not finished after {len(g.attempts)} attempt(s); retried on a later night"
        if self.target_sha:
            self.reset_tree(self.target_sha)

    # ----- the agent: Claude Code or the GitHub Copilot CLI ----------------------------------------------------------

    def copilot(self) -> bool:
        return (os.environ.get("DFB_AGENT") or self.cfg["repair"].get("agent") or "claude") == "copilot"

    def model_keys(self) -> tuple:
        """The selected agent's own credentials; the other agent's never reach it."""
        return ("COPILOT_GITHUB_TOKEN",) if self.copilot() else ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")

    def agent_label(self) -> str:
        if self.copilot():
            return f"GitHub Copilot CLI with {self.cfg['repair']['copilot']['model']}"
        return f"Claude Code with {self.cfg['repair']['model']}"

    def loop_path(self, value: str) -> str:
        """A bare program name is found on PATH; a relative path is relative to the loop root."""
        return value if "/" not in value or Path(value).is_absolute() else str(self.root / value)

    def copilot_home(self) -> Path:
        """The bot's own Copilot login (copilot_login.sh), never the user's ~/.copilot. Each attempt gets a copy."""
        home = Path(self.cfg["repair"]["copilot"].get("home") or "copilot-home")
        return home if home.is_absolute() else self.root / home

    def copilot_deny_rules(self) -> list[str]:
        """The Claude deny rules in the Copilot CLI's syntax: Bash(git push:*) -> shell(git push) (git and gh rules
        match the first-level subcommand), Bash(kill:*) -> shell(kill:*). Web tools are left out of its tool list."""
        rules = []
        for rule in self.cfg["repair"]["disallowed_tools"]:
            m = re.fullmatch(r"Bash\((.+?)(?::\*)?\)", rule)
            if m:
                rules.append(f"shell({m.group(1)})" if " " in m.group(1) else f"shell({m.group(1)}:*)")
        return rules

    def copilot_command(self, adir: Path, minutes: int) -> list[str]:
        c = self.cfg["repair"]["copilot"]
        exe = self.loop_path(os.environ.get("DFB_COPILOT_BIN") or c["bin"])
        prompt = read_text(adir / "prompt.md").replace("\0", "")
        if len(prompt.encode()) > 100_000:  # one argument holds at most 128 KiB
            prompt = f"Read {adir / 'prompt.md'} and follow it exactly."
        cmd = ["timeout", "-k", "60", str(minutes * 60), exe, "-p", prompt]
        cmd += ["--model", c["model"], "--output-format", "json", "--allow-all-tools", "--no-ask-user"]
        # No instructions from the repository, no MCP servers, no updates and no session export. Any path is allowed:
        # the test helper, the logs and earlier attempts are outside the checkout, and the loop's own checks do not
        # depend on path limits.
        cmd += ["--no-custom-instructions", "--disable-builtin-mcps", "--no-auto-update", "--allow-all-paths"]
        cmd += ["--no-remote", "--no-remote-export"]
        for server in c.get("disabled_mcp_servers") or []:
            cmd += ["--disable-mcp-server", server]
        if c.get("excluded_tools"):
            cmd += ["--excluded-tools", ",".join(c["excluded_tools"])]
        for rule in self.copilot_deny_rules():
            cmd += ["--deny-tool", rule]
        if c.get("effort"):
            cmd += ["--effort", str(c["effort"])]
        if c.get("max_ai_credits"):
            cmd += ["--max-ai-credits", str(c["max_ai_credits"])]
        cmd += ["--secret-env-vars", ",".join(MODEL_KEYS + GITHUB_KEYS)]  # not passed on to its shell commands
        cmd += ["--log-dir", str(adir / "copilot-logs"), "--log-level", "info"]
        return cmd + ["--usage-output-file", str(adir / "usage.json")]

    def agent_command(self, adir: Path, minutes: int) -> list[str]:
        if self.copilot():
            return self.copilot_command(adir, minutes)
        r = self.cfg["repair"]
        claude = os.environ.get("DFB_CLAUDE_BIN") or r["claude"]
        if "/" in claude and not Path(claude).is_absolute():  # a bare name is found on PATH; a relative path is
            claude = str(self.root / claude)  # relative to the loop root
        # Only the listed tools exist for the agent, and the user's own Claude Code settings (permissions, hooks, plugins,
        # extra directories, effort) do not apply: --setting-sources keeps only the repository's settings.
        cmd = [
            "timeout",
            "-k",
            "60",
            str(minutes * 60),
            claude,
            "-p",
            "--model",
            r["model"],
            "--max-turns",
            str(r["max_turns"]),
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--add-dir",
            str(adir),
            "--tools",
            ",".join(r.get("tools") or r["allowed_tools"]),
            "--allowedTools",
            ",".join(r["allowed_tools"]),
            "--disallowedTools",
            ",".join(r["disallowed_tools"]),
            "--setting-sources",
            str(r.get("setting_sources", "project")),
            "--strict-mcp-config",
        ]
        if r.get("effort"):
            cmd += ["--effort", str(r["effort"])]
        return cmd

    def run_agent(self, adir: Path, minutes: int, env: dict, stdin, out, err) -> int:
        """Run the agent until it exits, or until its time is up (124). After a result, Claude Code waits for any commands it
        moved to the background and then starts a new turn, so a result counts as final only when no turn follows.
        A final result with nothing in the background gets a minute to exit, one with background tasks 15 minutes;
        then whatever is still running is stopped."""
        # Read the transcript through our own handle on the file, so nothing the agent does to the path breaks this.
        reader = open(adir / "agent.jsonl", "rb")
        try:
            proc = subprocess.Popen(
                self.agent_command(adir, minutes), cwd=self.checkout, env=env, stdin=stdin, stdout=out, stderr=err
            )
        except (OSError, ValueError) as e:  # say, a prompt too long for one argument
            reader.close()
            err.write(f"the agent could not be started: {e}\n".encode())
            return 127
        deadline, finished_at, pending, partial = time.time() + minutes * 60 + 180, 0.0, False, b""
        try:
            while True:
                try:
                    return proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                lines = (partial + reader.read()).split(b"\n")
                partial = lines.pop()
                for line in lines:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    kind = (e.get("type"), e.get("subtype"))
                    if kind[0] == "result":  # Claude Code's result, or the Copilot CLI's last line
                        finished_at = time.time()
                    elif kind == ("system", "init") or kind[0] == "assistant":
                        finished_at = 0.0  # a new turn started
                    if kind == ("system", "background_tasks_changed"):
                        pending = bool(e.get("tasks"))
                grace = 900 if pending else 60
                if (finished_at and time.time() - finished_at > grace) or time.time() > deadline:
                    if finished_at:
                        self.notes.append(
                            f"the agent did not exit after its result in {adir.name}; what it left running was stopped"
                        )
                    return 0 if finished_at else 124
        finally:
            reader.close()
            if proc.poll() is None:
                kill_marked(f"DFB_ATTEMPT_DIR={adir}")
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    @staticmethod
    def agent_report(adir: Path) -> dict:
        """What an agent run reported, the same way for Claude Code (its result line) and the Copilot CLI (its events,
        then a result line with its exit code, and its usage file): last message, turns, cost, how it ended, and any
        error with its HTTP status."""
        rep = {
            "found": False,
            "said": "",
            "turns": 0,
            "cost_usd": 0.0,
            "ai_credits": 0.0,
            "premium_requests": 0.0,
            "end": "",
            "is_error": False,
            "error": "",
            "status": None,
            "model": "",
            "down": False,
            "outputs": 0,  # the model's answers and tool calls: 0 means it never answered
        }
        copilot_error, exit_code = False, None
        for line in read_text(adir / "agent.jsonl").splitlines():
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            kind, d = e.get("type"), e.get("data") if isinstance(e.get("data"), dict) else {}
            if kind == "result" and "exitCode" in e:  # the Copilot CLI's last line: its exit code and request count
                exit_code = e.get("exitCode")
                usage = e.get("usage") if isinstance(e.get("usage"), dict) else {}
                rep["found"] = True
                rep["premium_requests"] = as_float(usage.get("premiumRequests")) or rep["premium_requests"]
            elif kind == "result":  # Claude Code
                rep.update(
                    found=True,
                    said=str(e.get("result", "")),
                    turns=e.get("num_turns", 0),
                    cost_usd=as_float(e.get("total_cost_usd")),
                    end=str(e.get("subtype", "")),
                    is_error=bool(e.get("is_error")),
                    status=e.get("api_error_status"),
                    error=" ".join(str(x) for x in (e.get("errors") or [])),
                )
            elif kind == "system" and e.get("subtype") == "init":  # Claude Code's first line names its model
                rep["model"] = str(e.get("model") or "")
            elif kind == "assistant":  # a Claude Code turn
                rep["outputs"] += 1
            elif kind == "assistant.message":  # the Copilot CLI's events from here on; a tool call has no text
                rep["outputs"] += 1
                if d.get("content"):
                    rep["said"] = str(d["content"])
                    rep["model"] = str(d.get("model") or rep["model"])
            elif kind == "tool.execution_complete":
                rep["outputs"] += 1
                error = d.get("error") if isinstance(d.get("error"), dict) else {}
                if error.get("code") == "session_limits_exhausted":  # this attempt's --max-ai-credits cap
                    rep["end"] = "credits_exhausted"
            elif kind == "session.warning" and d.get("warningType") == "session_limits":  # "Session limit reached"
                rep["end"] = "credits_exhausted"
            elif kind == "assistant.turn_start":
                rep["turns"] += 1
            elif kind == "session.error":  # model.call_failure is not one: it also reports calls retried successfully
                copilot_error = True
                rep["error"] = str(d.get("message") or d.get("errorType") or "error")
                if d.get("errorCode") == "session_quota_exceeded":  # this attempt's own --max-ai-credits cap
                    rep["end"] = "credits_exhausted"
                    continue
                rep["status"] = int(as_float(d.get("statusCode"))) or rep["status"]
                rep["down"] = rep["down"] or d.get("errorType") in COPILOT_DOWN_TYPES or rep["status"] == 402
            elif kind == "session.usage_checkpoint":
                rep["ai_credits"] = as_float(d.get("totalNanoAiu")) / 1e9 or rep["ai_credits"]
                rep["premium_requests"] = as_float(d.get("totalPremiumRequests")) or rep["premium_requests"]
            elif kind == "session_limits_exhausted.requested":
                rep["end"] = "credits_exhausted"
        if exit_code not in (None, 0):
            rep["error"] = rep["error"] or f"the Copilot CLI exited with code {exit_code}"
        if copilot_error or exit_code not in (None, 0):
            rep["is_error"] = True
        usage = load_json(adir / "usage.json", {})  # the Copilot CLI's final totals
        if isinstance(usage, dict):
            rep["ai_credits"] = as_float(usage.get("totalNanoAiu")) / 1e9 or rep["ai_credits"]
            rep["premium_requests"] = as_float(usage.get("totalPremiumRequestCost")) or rep["premium_requests"]
        return rep

    def agent_down(self, agent: dict) -> bool:
        """Whether the agent could not work at all (no login, no access, no quota, an outage) rather than having
        tried. For the Copilot CLI only its error text counts: its last message is the agent's own words, which may
        hold any number. Running into this attempt's own credit cap is not an outage."""
        status = agent["status"]
        if self.copilot():
            down_text = bool(COPILOT_DOWN_RX.search(agent["error"]))
        else:
            down_text = bool(AGENT_DOWN_RX.search(agent["said"] + " " + agent["error"]))
        return (
            agent["is_error"]
            and agent["end"] != "credits_exhausted"
            and (
                agent["down"]
                or (isinstance(status, int) and (status in (401, 402, 403, 429) or status >= 500))
                or down_text
            )
        )

    def agent_env(self, adir: Path, ahome: Path) -> dict:
        """The agent's environment. It gets a fresh home in the run's cache: the agent and its tools keep their files
        there instead of in the user's home, and nothing carries over from one attempt to the next."""
        shutil.rmtree(ahome, ignore_errors=True)
        ahome.mkdir(mode=0o700, parents=True)
        env = self.env(adir / "cache-agent", model=True)
        env.update(
            HOME=str(ahome),
            DFB_LOOP_ENV=str(self.run_dir / "loop_env.sh"),
            DFB_ATTEMPT_DIR=str(adir),
            DFB_RUN_TEST=str(self.tool_dir / "run_test.sh"),
        )
        if self.copilot():
            # A fresh Copilot home with only the bot's login copied in: nothing an earlier session stored or left there
            # (its session index, hooks, MCP servers) is loaded. No second model reviews its work, and no hooks run.
            chome = ahome / ".copilot"
            chome.mkdir(mode=0o700)
            if (self.copilot_home() / "config.json").is_file():
                shutil.copy(self.copilot_home() / "config.json", chome / "config.json")
            write_json(chome / "settings.json", {"builtInAgents": {"rubberDuck": False}, "disableAllHooks": True})
            env.update(COPILOT_HOME=str(chome), COPILOT_AUTO_UPDATE="false", NO_COLOR="1", PLAIN_DIFF="true")
        else:
            env["ENABLE_CLAUDEAI_MCP_SERVERS"] = "false"  # none of the account's claude.ai connectors in its session
        # The agent's git can neither push to nor fetch from GitHub. Without fetching, a search through old file
        # contents (git log -S, blame) fails at once instead of downloading the clone's missing history for hours.
        urls = ("git@github.com:", "ssh://git@github.com/", "https://github.com/")
        rules = [("url.push-disabled-by-dfb-loop/.pushInsteadOf", u) for u in urls]
        rules += [("url.fetch-disabled-by-dfb-loop/.insteadOf", u) for u in urls]
        env["GIT_CONFIG_COUNT"] = str(len(rules))
        for i, (k, v) in enumerate(rules):
            env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"] = k, v
        return env

    def attempt(self, g: Group, n: int) -> dict:
        r = self.cfg["repair"]
        reserve = float(self.cfg["budget"]["final_suite_reserve_minutes"])
        adir = self.run_dir / "attempts" / g.gid / str(n)
        adir.mkdir(parents=True, exist_ok=True)
        info = {"attempt": n, "dir": str(adir), "outcome": "", "reason": ""}

        def done(outcome: str, reason: str = "") -> dict:
            info.update(outcome=outcome, reason=reason)
            write_json(adir / "attempt.json", info)
            self.log(f"attempt {g.gid}/{n}: {outcome}" + (f" ({reason[:160]})" if reason else ""))
            return info

        self.reset_tree(self.target_sha, deep=True)
        self.drop_generated()
        if not self.build(adir / "prebuild.log"):  # the binary must match the target before the agent starts
            return done(
                "prebuild_failed", "the unchanged tree did not rebuild: " + tail_lines(adir / "prebuild.log", 8)
            )
        self.ensure_bin()
        minutes = int(min(float(r["attempt_minutes"]), self.minutes_left() - reserve - 10))
        if minutes < 15:
            return done("no_time", "not enough time left for an attempt")
        protected, live, cache = self.protected_fingerprint(), self.live_fingerprint(), self.cmake_cache_fingerprint()
        ninja = self.ninja_fingerprint()
        stamp = adir / ".started"
        stamp.write_text("")
        (adir / "prompt.md").write_text(self.render_prompt(g, n, adir, minutes))
        env = self.agent_env(adir, self.run_dir / "cache" / "agent-home" / g.gid / str(n))
        self.log(f"attempt {g.gid}/{n}: {self.agent_label()} on {len(g.tests)} test(s), up to {minutes} min")
        t1 = time.time()
        # Claude Code reads the prompt on stdin; the Copilot CLI gets it as its -p argument.
        prompt_in = os.devnull if self.copilot() else adir / "prompt.md"
        with (
            open(prompt_in, "rb") as stdin,
            open(adir / "agent.jsonl", "wb") as out,
            open(adir / "agent.err", "wb") as err,
        ):
            rc = self.run_agent(adir, minutes, env, stdin, out, err)
        kill_marked(f"DFB_ATTEMPT_DIR={adir}")  # the agent runs each command in its own session; find leftovers anyway
        self.kill_others()
        self.clear_git_locks()
        self.restore_git_files()
        agent = self.agent_report(adir)
        write_json(adir / "agent.json", agent)
        cost = agent["cost_usd"]
        self.cost_usd += cost
        self.ai_credits += agent["ai_credits"]
        self.premium_requests += agent["premium_requests"]
        said = agent["said"]
        timed_out = rc in (124, 137, -9)  # -9: timeout's own KILL after the grace period
        info.update(
            minutes=round((time.time() - t1) / 60, 1),
            agent_rc=rc,
            agent_end=agent["end"],
            turns=agent["turns"],
            cost_usd=round(cost, 2),
            ai_credits=round(agent["ai_credits"], 2),
            agent_said=said[-1500:],
        )
        errors = agent["error"]
        agent_error = self.agent_down(agent)
        if timed_out and not agent["outputs"]:  # a stalled model endpoint, not an agent that tried
            agent_error = True
            said = errors = f"the model gave no answer in {minutes} min"
        if not agent["found"] and not timed_out and rc != 0:
            agent_error = True
            first = " ".join(read_text(adir / "agent.err").split())  # the first lines name the problem
            said = f"the agent did not run (rc={rc}): {(errors + ' ' + first).strip()[:300]}"
        if self.live_fingerprint() != live:
            self.notes.append(
                f"the live loop files ({self.live_tool}, the scripts, claude.env or HOST in {self.root}) changed "
                f"during attempt {g.gid}/{n}; if you did not edit it, review the change"
            )
        if self.protected_fingerprint() != protected:
            self.restore_protected()
            self.reset_tree(self.target_sha, deep=True)
            return done("rejected", "the agent changed the loop's own tooling or simulator files")
        if self.cmake_cache_fingerprint() != cache:
            self.reset_tree(self.target_sha, deep=True)
            self.configure(adir / "reconfigure.log", fresh=True)
            return done(
                "rejected", "the agent changed the build configuration (CMakeCache.txt), which a patch cannot " "carry"
            )
        if self.ninja_fingerprint() != ninja:  # build files changed: regenerate them from the sources before verifying
            self.configure(adir / "regenerate.log")
            info["regenerated_build_files"] = True
        touched = self.kept_trees_changed(stamp)
        if touched:  # toolchain or package-cache edits a patch cannot carry: drop them so the build fetches them again
            for t in touched:
                if Path(t) == self.checkout / "runtime" / "sfpi":
                    self.drop_sfpi()
                else:
                    shutil.rmtree(t, ignore_errors=True)
            info["dropped_trees"] = touched
            self.notes.append(
                f"attempt {g.gid}/{n} changed files under {', '.join(touched)}; they were removed so the "
                f"next build fetches clean copies"
            )
        diff = self.collect_diff(adir)
        if agent_error and not ((adir / "NOTES.md").exists() and diff.strip()):
            why = (errors or said) if self.copilot() else (said or errors)
            return done("agent_error", f"{self.agent_label()} is not reachable: {why[:300]}")
        if (adir / "HANDOFF.md").exists() and not (adir / "NOTES.md").exists():
            return done("handoff", "the agent handed off; see HANDOFF.md")
        if not diff.strip():
            if timed_out:
                return done("timeout", "the agent ran out of time without a change")
            if agent["end"] in ("error_max_turns", "credits_exhausted"):
                return done("max_turns", "the agent ran out of turns or AI credits without a change")
            return done("no_change", "the agent left no change and no handoff")
        problems, review = self.check_guards(diff)
        if problems:
            return done("rejected", "; ".join(problems))
        # Verify exactly what would be published: the saved patch on a clean tree.
        self.reset_tree(self.target_sha, deep=True)
        rc_apply, out = self.git("apply", "--index", str(adir / "diff.patch"), check=False)
        if rc_apply != 0:
            return done("rejected", f"the saved patch does not apply to a clean tree: {out[-300:]}")
        self.drop_generated()
        if not self.build(adir / "build.log"):
            return done("build_failed", tail_lines(adir / "build.log", 25))
        self.ensure_bin()
        outcome, reason, full = self.verify(g, adir)
        if outcome == "verified" and review:
            outcome, reason = "unverified", "the fix passes but needs a human look first: " + "; ".join(review)
        if outcome in ("verified", "unverified", "out_of_time"):
            self.save_fix(g, adir, diff, n, verified=outcome == "verified", full=full, why=reason)
            if outcome == "verified":
                self.verified_full[g.gid] = full
        return done(outcome, reason)

    def verify(self, g: Group, adir: Path) -> tuple[str, str, dict | None]:
        """Rerun the failing tests and then the whole suite on the current build. (outcome, reason, full results)."""
        reserve = float(self.cfg["budget"]["final_suite_reserve_minutes"])
        focused = self.run_tests(g.tests, adir / "focused", adir / "cache-focused", min(8, len(g.tests)), "focused")
        if any(x.signature == NOT_RUN for x in focused.values()):
            return "out_of_time", "the run's deadline came before the failing tests could be rerun", None
        bad = [x for x in focused.values() if x.verdict != "PASS"]
        if bad:
            return "focused_failed", "; ".join(f"{x.test}: {x.verdict} {x.signature}" for x in bad[:8]), None
        if self.minutes_left() < reserve:
            return "unverified", "the failing tests pass; no time was left for the full-suite check", None
        full = self.run_suite(adir / "full", f"full suite for {g.gid}")
        if any(x.signature == NOT_RUN for x in full.values()):
            return "unverified", "the failing tests pass; the full-suite check did not finish in time", None
        worse, unchecked = self.regressions(full, adir)
        if worse:
            return "regressed", "these tests got worse with the change: " + ", ".join(worse[:12]), None
        if unchecked:
            return "unverified", "the failing tests pass; some regression rechecks did not run in time", None
        still = [t for t in g.tests if t in full and full[t].verdict != "PASS"]
        if still:
            return "focused_failed", "failing again in the full-suite run: " + ", ".join(still[:8]), None
        return "verified", "", full

    def reverify_saved_fix(self, g: Group, fix_dir: Path) -> bool:
        """A fix verified on an earlier night that was never published: apply it to tonight's tree and verify it again,
        without the model. True when it is verified again (it is then published with tonight's fixes)."""
        adir = self.run_dir / "attempts" / g.gid / "saved"
        adir.mkdir(parents=True, exist_ok=True)
        self.reset_tree(self.target_sha, deep=True)
        rc, _ = self.git("apply", "--index", str(fix_dir / "patch.diff"), check=False)
        if rc != 0:
            self.notes.append(f"the saved fix for {g.gid} no longer applies; the group goes to the agent again")
            self.reset_tree(self.target_sha)
            return False
        problems, review = self.check_guards(read_text(fix_dir / "patch.diff"))  # tonight's guards, as for a new fix
        if problems or review:
            self.notes.append(
                f"the saved fix for {g.gid} does not pass tonight's guards ({'; '.join(problems + review)[:300]}); "
                f"the group goes to the agent again"
            )
            self.reset_tree(self.target_sha)
            return False
        self.drop_generated()
        if not self.build(adir / "build.log"):
            self.reset_tree(self.target_sha)
            return False
        self.ensure_bin()
        outcome, reason, full = self.verify(g, adir)
        self.log(f"saved fix for {g.gid} re-verified: {outcome} {reason[:120]}")
        if outcome != "verified":
            self.reset_tree(self.target_sha)
            return False
        shutil.copy2(fix_dir / "patch.diff", adir / "diff.patch")
        if (fix_dir / "NOTES.md").exists():
            shutil.copy2(fix_dir / "NOTES.md", adir / "NOTES.md")
        else:
            title = read_text(fix_dir / "title.txt").strip()
            prefix = self.cfg["pr"]["title_prefix"]
            title = title[len(prefix) :].strip() if title.startswith(prefix) else title
            (adir / "NOTES.md").write_text(
                f"# {title or 'Fix ' + g.signature[:60]}\n\n(Verified again tonight; the "
                f"original notes were not kept.)\n"
            )
        self.save_fix(g, adir, read_text(adir / "diff.patch"), 0, verified=True, full=full)
        self.verified_full[g.gid] = full
        g.status = "fixed (saved fix from an earlier night, verified again tonight)"
        self.reset_tree(self.target_sha)
        return True

    def regressions(self, full: dict[str, Result], adir: Path) -> tuple[list[str], bool]:
        """(tests whose verdict got worse than in tonight's baseline, whether some rechecks did not run). PASS to
        anything else and SKIP to a failure count as worse. Each is rerun once on its own, so a one-off flake does not
        reject a good fix."""

        def worse(t: str, now: Result) -> bool:
            before = self.baseline.get(t)
            if before is None or t in self.hard or now.signature == NOT_RUN:
                return False
            was = "PASS" if t in self.flaky else before.verdict
            return (was == "PASS" and now.verdict != "PASS") or (was == "SKIP" and now.verdict not in OK_VERDICTS)

        cand = [t for t, x in full.items() if worse(t, x)]
        if not cand:
            return [], False
        again = self.run_tests(cand, adir / "full_retry", adir / "cache-retry", min(8, len(cand)), "regression check")
        unchecked = any(again[t].signature == NOT_RUN for t in cand)
        return [t for t in cand if worse(t, again[t])], unchecked

    def collect_diff(self, adir: Path) -> str:
        """Working tree against the target, new files included (anything the agent committed is included too)."""
        self.git("add", "-A", "-N", ".")
        _, diff = self.sh(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--binary",
                self.target_sha,
            ],
            timeout=300,
        )
        self.git("reset", "-q", check=False)
        (adir / "diff.patch").write_text(diff)
        return diff

    def check_guards(self, diff: str) -> tuple[list[str], list[str]]:
        """(problems that reject the change, concerns that keep it from being published without a human look)."""
        gc = self.cfg["guards"]
        files: dict[str, dict] = {}
        cur, in_hunk = None, False
        for line in diff.splitlines():
            if line.startswith("diff --git "):
                m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
                cur, in_hunk = (m.group(2) if m else line), False
                files.setdefault(cur, {"added": [], "removed": [], "special": []})
            elif cur is None:
                continue
            elif not in_hunk:
                if line.startswith("@@"):
                    in_hunk = True
                elif line.startswith(
                    ("new file mode", "deleted file mode", "rename from", "copy from", "old mode", "new mode")
                ):
                    files[cur]["special"].append(line.split(" mode")[0].split(" from")[0])
            elif line.startswith("@@"):
                continue
            elif line.startswith("+"):
                files[cur]["added"].append(line[1:])
            elif line.startswith("-"):
                files[cur]["removed"].append(line[1:])
        problems, review = [], []
        if len(files) > int(gc["max_files"]):
            problems.append(f"touches {len(files)} files (limit {gc['max_files']})")
        changed = sum(len(f["added"]) + len(f["removed"]) for f in files.values())
        if changed > int(gc["max_changed_lines"]):
            problems.append(f"changes {changed} lines (limit {gc['max_changed_lines']})")
        if "Subproject commit" in diff:
            problems.append("changes a submodule")
        if "GIT binary patch" in diff:
            problems.append("adds or changes a binary file")
        assert_rx = re.compile(gc["assertion_regex"])
        sim_rx = re.compile(gc["simulator_regex"])
        roots = tuple(gc.get("new_file_roots") or ())
        for path, f in files.items():
            added = f["added"]
            is_test = any(fnmatch.fnmatch(path, p) for p in gc["test_paths"])
            for sp in f["special"]:
                if sp == "new file":
                    if roots and not path.startswith(roots):
                        problems.append(f"adds a file outside {', '.join(roots)}: {path}")
                else:
                    problems.append(f"{sp} on {path} (deleting, renaming or changing the mode of files is not allowed)")
            if any(fnmatch.fnmatch(path, p) for p in gc["forbidden_paths"]):
                problems.append(f"edits a protected path: {path}")
            for pat in gc["forbidden_added"]:
                hits = [ln for ln in added if re.search(pat, ln)]
                if hits:
                    problems.append(f"{path} adds a forbidden pattern ({pat}): {hits[0].strip()[:100]}")
            sim_hits = [ln for ln in added if sim_rx.search(ln)]
            if sim_hits and is_test:
                problems.append(f"{path} adds simulator-specific code to a test: {sim_hits[0].strip()[:100]}")
            elif sim_hits:
                review.append(f"{path} adds simulator-specific code: {sim_hits[0].strip()[:100]}")
            # Whole assertion statements before and after, so an edit to any line of a wrapped one counts.
            if f["removed"] and "new file" not in f["special"]:
                _, old = self.sh(["git", "show", f"{self.target_sha}:{path}"], timeout=60)
                new = read_text(self.checkout / path)
                lost = assertion_statements(old, assert_rx) - assertion_statements(new, assert_rx)
                if lost:
                    first = next(iter(lost))[:100]
                    if is_test:
                        problems.append(f"{path} removes or changes a test assertion: {first}")
                    else:
                        review.append(f"{path} removes or changes an assertion in product code: {first}")
            if is_test:
                if any(re.search(r"\breturn\s*;", ln) for ln in added):
                    problems.append(f"{path} adds a `return;` to test code")
                if any(re.match(r"\s*#\s*(if|ifdef|ifndef)\b", ln) for ln in added):
                    review.append(f"{path} adds a preprocessor condition to test code")
        return problems, review

    # ----- prompt, fixes, handoffs, PRs ------------------------------------------------------------------------------

    def excerpt(self, test: str, max_key: int = 60, max_tail: int = 40) -> str:
        r = self.hard.get(test) or self.baseline.get(test)
        log = r.log if r else ""
        text = read_text(log, 4_000_000)
        sec = test_section(text, test) or text
        lines = sec.splitlines()
        key = [ln for ln in lines if ERROR_LINE_RX.search(ln)][:max_key]
        picked, seen = [], set()
        for ln in key + (["..."] if key else []) + lines[-max_tail:]:
            if ln in seen and ln != "...":
                continue
            seen.add(ln)
            picked.append(ln[:300])
        return f"### {test}\nFull log: {log}\n```\n" + "\n".join(picked) + "\n```"

    def render_prompt(self, g: Group, n: int, adir: Path, minutes: int) -> str:
        r = self.cfg["repair"]
        prev = ""
        if g.attempts:
            parts = ["## Earlier attempts tonight (they did not work)", ""]
            for a in g.attempts:
                parts.append(f"- Attempt {a['attempt']}: {a['outcome']}. {a.get('reason', '')[:1200]}")
                parts.append(f"  Its patch: {a['dir']}/diff.patch. Its notes: {a['dir']}/NOTES.md (if any).")
            parts.append("\nDo not repeat an approach that already failed.")
            prev = "\n".join(parts)
        values = {
            "base_sha": self.base_sha[:11],
            "carried": "; ".join(self.carried) or "none",
            "signature": g.signature,
            "n_tests": str(len(g.tests)),
            "tests": "\n".join(f"- `{t}`" for t in g.tests[:40])
            + (f"\n- ... and {len(g.tests) - 40} more" if len(g.tests) > 40 else ""),
            "excerpts": "\n\n".join(self.excerpt(t) for t in g.tests[:3]),
            "previous_attempts": prev,
            "attempt_dir": str(adir),
            "run_test": str(self.tool_dir / "run_test.sh"),
            "test_timeout": str(self.cfg["suite"]["test_timeout_seconds"]),
            "attempt_minutes": str(minutes),
            "max_turns": str(r["max_turns"]),
            "attempt": str(n),
            "attempts_per_group": str(r["attempts_per_group"]),
            "build_dir": self.cfg["build"]["dir"],
            "build_target": self.cfg["build"]["target"],
            "loop_root": str(self.root),
        }
        text = (self.tool_dir / "prompt_template.md").read_text()
        for k, v in values.items():
            text = text.replace("{{" + k + "}}", v)
        return text

    def reproduce_lines(self, tests, public: bool = False) -> list[str]:
        """Commands for a tt-metal checkout of the tested commit (not the bot's clone, which the loop resets)."""
        s, b = self.cfg["sim"], self.cfg["build"]
        public = public or bool(self.args.sim_dir)  # CI's simulator path exists only inside its container
        configure = " ".join(b["configure"]).replace("{checkout}", "$PWD")
        env = " ".join(
            ["TT_METAL_SIMULATOR_HOME=$TTSIM", "TT_METAL_SIMULATOR=$TTSIM/libttsim.so"]
            + [f"{k}={v}" for k, v in sorted((s.get("env") or {}).items())]
        )
        first = (
            f"# TTSIM: a directory with ttsim {self.sim_version}'s {s['lib']} as libttsim.so and "
            f"{s['soc_descriptor']} as soc_descriptor.yaml"
            if public
            else f"TTSIM={self.sim_dir}"
        )
        tests_on = f"cmake -B {b['dir']} -DTT_METAL_BUILD_TESTS=ON"  # a checkout that was built without tests
        return [
            first,
            f"if [ -f {b['dir']}/CMakeCache.txt ]; then {tests_on}; else {configure}; fi",
            f"cmake --build {b['dir']} --target {b['target']}",
        ] + [f"{env} ./{b['dir']}/test/tt_metal/{b['target']} --gtest_filter='{t}'" for t in tests]

    def pr_title(self, notes: str, g: Group) -> str:
        first = notes.strip().splitlines()[0] if notes.strip() else ""
        title = first[2:].strip() if first.startswith("# ") else f"Fix DFB test failure: {g.signature[:60]}"
        return sanitize(f"{self.cfg['pr']['title_prefix']} {title}")[:120]

    def save_fix(self, g: Group, adir: Path, diff: str, n: int, verified: bool, full, why: str = "") -> None:
        fdir = self.run_dir / "fixes" / g.gid
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / "patch.diff").write_text(diff)
        notes = read_text(adir / "NOTES.md")
        if notes.strip():
            (fdir / "NOTES.md").write_text(notes)
        body_notes = (
            "\n".join(notes.strip().splitlines()[1:]).strip()
            if notes.strip().startswith("# ")
            else notes.strip() or "(The agent wrote no notes.)"
        )
        if full:
            c = Counter(x.verdict for x in full.values())
            suite_line = (
                f"Full DFB suite on ttsim after the fix: {c['PASS']} pass, {c['SKIP']} skip, no test got "
                f"worse ({len(full)} tests)."
            )
        else:
            suite_line = f"Not fully verified: {why}."
        lines = [
            f"> Opened automatically by the nightly DFB repair loop ({self.agent_label()}, unattended). Checked on the ttsim "
            "simulator only. Please review it before marking it ready.",
            "",
            body_notes,
            "",
            "## Evidence",
            "",
            f"- Failure signature: `{g.signature}`",
            f"- {len(g.tests)} test(s) failed on ttsim before the fix and pass after it:",
            *[f"  - `{t}`" for t in g.tests[:25]],
            f"- {suite_line}",
            f"- Tested on `origin/{self.cfg['target'].get('base', 'main')}` {self.base_sha[:11]}"
            + (f" with carried commits not yet on main: {'; '.join(self.carried)}" if self.carried else "")
            + f". ttsim {self.sim_version}. {self.agent_label()}"
            + (f", attempt {n}." if n else ", from a fix verified on an earlier night."),
            "",
            "## Reproduce on ttsim (in a tt-metal checkout of the tested commit; up to 5 tests)",
            "",
            "```",
            *self.reproduce_lines(g.tests[:5], public=True),
            "```",
            "",
        ]
        body = sanitize("\n".join(lines))
        (fdir / "pr_body.md").write_text(body)
        title = sanitize(self.pr_title(notes, g))
        (fdir / "title.txt").write_text(title + "\n")
        # What gets published is kept here, not re-read from files a later attempt's agent could have edited.
        g.fix = {
            "dir": str(fdir),
            "verified": verified,
            "title": title,
            "body": body,
            "attempt": n,
            "why_unverified": why,
            "sha256": hashlib.sha256((fdir / "patch.diff").read_bytes()).hexdigest(),
        }

    def write_handoff(self, g: Group) -> None:
        hdir = self.run_dir / "handoff"
        hdir.mkdir(exist_ok=True)
        parts = [
            f"# Needs human help: {g.signature}",
            "",
            f"{len(g.tests)} test(s) fail on ttsim with this signature:",
            *[f"- `{t}`" for t in g.tests[:40]],
            "",
            "## What the loop tried",
            "",
        ]
        for a in g.attempts:
            parts.append(f"- Attempt {a['attempt']}: **{a['outcome']}**. {a.get('reason', '')[:600]}")
            parts.append(f"  Files: {a['dir']} (prompt.md, agent.jsonl, diff.patch, NOTES.md or HANDOFF.md)")
        for a in reversed(g.attempts):
            text = next(
                (
                    read_text(Path(a["dir"]) / nm)
                    for nm in ("HANDOFF.md", "NOTES.md")
                    if read_text(Path(a["dir"]) / nm).strip()
                ),
                "",
            )
            if text:
                parts += ["", f"## The agent's notes (attempt {a['attempt']})", "", text.strip()]
                break
        parts += [
            "",
            "## Reproduce on ttsim (in your own checkout of the tested commit, not the bot's clone; up to 5 tests)",
            "",
            "```",
            *self.reproduce_lines(g.tests[:5]),
            "```",
            "",
            "## First lines of the failure",
            "",
            self.excerpt(g.tests[0], max_key=40, max_tail=20),
        ]
        (hdir / f"{g.gid}.md").write_text("\n".join(parts) + "\n")

    def clear_outbox(self) -> None:
        """Remove what the loop writes into the outbox (one directory per group); anything else there is left alone."""
        out = Path(self.args.outbox)
        out.mkdir(parents=True, exist_ok=True)
        for p in out.iterdir():
            if re.fullmatch(r"[0-9a-f]{10}", p.name):
                shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)

    def kill_others(self) -> None:
        """With --own-pid-namespace (the loop alone in a container, as in CI), stop every other process in it, so
        nothing an agent started outlives its attempt, even a process that changed its environment or session."""
        if not self.args.own_pid_namespace:
            return
        init = read_text("/proc/1/comm").strip()
        if init not in ("docker-init", "tini", "catatonit"):
            self.notes.append(f"--own-pid-namespace was given, but PID 1 is {init!r}; other processes were left alone")
            return
        try:
            os.kill(-1, signal.SIGKILL)  # every process this one may signal, except itself and PID 1
        except ProcessLookupError:
            pass

    def open_prs(self) -> None:
        self.kill_others()
        if self.args.outbox and self.mode == "pr":  # only what the loop queues below goes out
            self.clear_outbox()
        for g in self.groups:  # credit a fix with the other groups it also fixed
            if g.fixed_by:
                fixer = next((x for x in self.groups if x.gid == g.fixed_by), None)
                if fixer and fixer.fix:
                    credit = sanitize(
                        f"\nThis fix also makes these tests pass (they failed with "
                        f"`{g.signature}`): " + ", ".join(f"`{t}`" for t in g.tests[:20]) + "\n"
                    )
                    fixer.fix["body"] = fixer.fix.get("body", "") + credit
                    with open(Path(fixer.fix["dir"]) / "pr_body.md", "a") as fh:
                        fh.write(credit)
        fixed = [g for g in self.groups if g.fix.get("verified")]
        if self.mode != "pr" or not fixed:
            return
        if self.no_publish:
            for g in fixed:
                g.status = f"fixed, not published: {self.no_publish} (the patch is on disk)"
            return
        p = self.cfg["pr"]
        gh_ok = (
            shutil.which("gh") is not None
            and self.sh(["gh", "auth", "status"], env=self.env(github=True), timeout=60)[0] == 0
        )
        stamp = utc_now().strftime("%Y%m%d")
        published = 0
        for g in fixed:
            if published >= int(p["max_per_run"]):
                g.status = "fixed, not published: the PR limit for this run was reached (the patch is on disk)"
                continue
            if self.minutes_left() < 5:
                g.status = "fixed, not published: no time was left (the patch is on disk)"
                continue
            branch = f"{p['branch_prefix']}/{stamp}-{g.gid}"
            try:
                url, how = self.publish(g, branch, gh_ok)
            except (Infra, NotPublished) as e:
                g.status = f"fixed, not published: {e}"
                self.log(f"{g.gid}: {g.status}")
                continue
            published += 1
            if url:
                g.fix.update(branch=branch, url=url, how=how)
            else:  # queued in the outbox: the next night sees the branch on origin once the publish job pushed it
                g.fix.update(queued=branch, how=how)
            g.status = f"fixed: {how}"
            self.log(f"{g.gid}: {how}" + (f": {url}" if url else ""))
        self.outbox_done = True
        if self.target_sha:
            self.reset_tree(self.target_sha)

    def publish(self, g: Group, branch: str, gh_ok: bool) -> tuple[str, str]:
        p = self.cfg["pr"]
        base = self.cfg["target"].get("base", "main")
        fdir = Path(g.fix["dir"])
        found = self.remote_fix_branches(g.gid)
        if found is None:
            raise NotPublished("origin could not be asked whether a fix branch already exists")
        retired = set(self.state.get("groups", {}).get(g.gid, {}).get("retired_branches") or [])
        found = [b for b in found if b not in retired]
        if found:
            raise NotPublished(f"a fix branch for this group is already on origin: {found[0]}")
        # A later attempt's agent runs as the same user and could have edited this file since it was verified.
        if hashlib.sha256((fdir / "patch.diff").read_bytes()).hexdigest() != g.fix.get("sha256"):
            raise NotPublished("the saved patch changed after it was verified")
        self.reset_tree(self.target_sha, deep=True)
        self.git("apply", "--index", str(fdir / "patch.diff"))
        body = sanitize(g.fix.get("body") or "")
        (fdir / "pr_body.md").write_text(body)  # from memory, for gh below
        msg = fdir / "commit_message.txt"
        msg.write_text(sanitize(g.fix["title"] + "\n\n" + body.split("## Reproduce")[0].strip()) + "\n")
        self.git("commit", "-q", "--no-verify", "-F", str(msg))
        fix_sha = self.git("rev-parse", "HEAD")[1]
        main_sha = self.git("rev-parse", f"origin/{base}")[1]
        if self.carried or self.args.ref or self.base_sha != main_sha:
            # The fix was verified on another tree; publish it on plain main only after it builds and passes there.
            self.git("checkout", "-q", "-B", branch, f"origin/{base}")
            rc, _ = self.git("cherry-pick", fix_sha, check=False)
            if rc != 0:
                self.git("cherry-pick", "--abort", check=False)
                raise NotPublished(f"the fix does not apply to {base} on its own")
            self.drop_generated()
            if not self.build(fdir / "main_build.log"):
                raise NotPublished(f"the fix does not build on {base} on its own")
            self.ensure_bin()
            res = self.run_tests(
                g.tests,
                fdir / "main_check",
                self.run_dir / "cache" / f"main-{g.gid}",
                min(8, len(g.tests)),
                f"check on plain {base}",
            )
            bad = [t for t, x in res.items() if x.verdict != "PASS"]
            if bad:
                raise NotPublished(f"on plain {base} these still fail: " + ", ".join(bad[:5]))
        else:
            self.git("checkout", "-q", "-B", branch, fix_sha)
        if self.args.outbox:  # CI: a separate job without the agent pushes it and opens the draft PR
            # A plain diff and a separate message: nothing is parsed out of a mail-formatted patch on the other side.
            out_dir = Path(self.args.outbox) / g.gid
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True)
            rc, diff = self.sh(["git", "-c", "core.fsmonitor=false", "diff", "--binary", f"{branch}^", branch])
            if rc != 0 or not diff.startswith("diff --git "):
                raise NotPublished(f"git diff of the fix commit failed: {diff.strip()[-200:]}")
            (out_dir / "patch.diff").write_text(diff)
            (out_dir / "message.txt").write_text(read_text(msg))
            (out_dir / "base.txt").write_text(self.git("rev-parse", f"{branch}^")[1] + "\n")
            (out_dir / "branch.txt").write_text(branch + "\n")
            (out_dir / "title.txt").write_text(g.fix["title"] + "\n")
            (out_dir / "pr_body.md").write_text(body)
            self.git("checkout", "-q", "--detach", self.target_sha, check=False)
            self.git("branch", "-q", "-D", branch, check=False)
            return "", f"queued for the publish job as {branch}"
        dry = ["--dry-run"] if self.args.publish_dry_run else []
        rc, out = self.git("push", *dry, p["push_url"], f"{branch}:refs/heads/{branch}", check=False, timeout=900)
        self.git("checkout", "-q", "--detach", self.target_sha, check=False)
        self.git("branch", "-q", "-D", branch, check=False)
        if rc != 0:
            if re.search(r"SAML|authorized for this organization", out):
                raise NotPublished(
                    "GitHub refused the push: this machine's SSH key is not authorized for the "
                    "organization's single sign-on (GitHub > Settings > SSH and GPG keys > Configure "
                    "SSO > Authorize)"
                )
            raise NotPublished(f"git push failed: {out.strip()[-300:]}")
        if self.args.publish_dry_run:
            return f"(dry run) would push {branch} and open a draft PR", "dry run, nothing pushed"
        if gh_ok:
            rc, out = self.sh(
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    p["repo"],
                    "--draft",
                    "--base",
                    base,
                    "--head",
                    branch,
                    "--title",
                    g.fix["title"],
                    "--body-file",
                    str(fdir / "pr_body.md"),
                ],
                env=self.env(github=True),
                timeout=300,
            )
            if rc == 0 and "http" in out:
                return out.strip().splitlines()[-1], "draft PR opened"
            self.notes.append(f"gh pr create failed for {branch}: {out.strip()[-200:]}")
        return (
            f"https://github.com/{p['repo']}/compare/{base}...{branch}?expand=1",
            "branch pushed; the link opens the PR form, where you choose 'Create draft pull request'",
        )

    # ----- summary ---------------------------------------------------------------------------------------------------

    def update_state(self, headline: str, counts: Counter) -> None:
        now = utc_now().isoformat(timespec="seconds")
        groups = self.state.setdefault("groups", {})
        dry = bool(self.args.publish_dry_run)
        current = {g.gid for g in self.groups}
        for g in self.groups:
            st = groups.setdefault(g.gid, {"first_seen": now, "times_seen": 0, "attempts_total": 0})
            if g.prior_gid and g.prior_gid != g.gid and g.prior_gid in groups and g.prior_gid not in current:
                old = groups[g.prior_gid]  # the same failure under a new id
                for k in ("first_seen",) + STATE_FIX_KEYS:
                    if old.get(k) and not st.get(k):
                        st[k] = old[k]
                old.update(superseded_by=g.gid, resolved_at=now)
            if st.get("resolved_at"):  # it came back after going away
                if not g.held:  # earlier fixes and handoffs no longer apply, unless tonight's status still uses them
                    for k in STATE_FIX_KEYS:
                        st.pop(k, None)
                st.pop("resolved_at", None)
                st.pop("superseded_by", None)
            st.update(
                signature=g.signature,
                kind=g.kind,
                tests=g.tests[:30],
                last_seen=now,
                status=g.status,
                times_seen=st.get("times_seen", 0) + 1,
                attempts_total=st.get("attempts_total", 0) + len(g.attempts),
            )
            if g.handed_off:
                st.update(handoff_at=now, handoff_file=str(self.run_dir / "handoff" / f"{g.gid}.md"))
            if g.fixed_by:
                st["fixed_by"] = g.fixed_by
            if g.fix and not dry:
                st.update(fixed_at=now, fix_dir=g.fix["dir"], fix_verified=bool(g.fix.get("verified")))
            if g.fix.get("branch") and not dry:
                st.update(pr_branch=g.fix["branch"], pr_url=g.fix.get("url", ""))
            if g.fix.get("queued") and not dry:
                st["queued_branch"] = g.fix["queued"]
            if g.promote and not st.get("pr_branch"):
                st["pr_branch"] = g.promote
        if self.classified:  # only a run that judged every test can say a group went away
            current = {g.gid for g in self.groups}
            for gid, st in groups.items():
                if gid not in current and not st.get("resolved_at"):
                    st["resolved_at"] = now
        runs = self.state.setdefault("runs", [])
        runs.append(
            {
                "date": now,
                "mode": self.mode,
                "result": headline,
                "run_dir": str(self.run_dir),
                "counts": dict(counts),
                "groups": len(self.groups),
                "minutes": round((time.time() - self.t0) / 60),
            }
        )
        del runs[:-90]
        write_json(self.state_path, self.state)

    def small_patch(self, g: Group) -> str:
        """The fix's patch for the summary: only the one that was saved after verification, and only when it is no
        longer than summary.diff_max_lines; control characters are shown as '?'."""
        limit = int((self.cfg.get("summary") or {}).get("diff_max_lines", 0))
        try:
            fd = os.open(Path(g.fix["dir"]) / "patch.diff", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except (KeyError, OSError):
            return ""
        with os.fdopen(fd, "rb") as f:
            data = f.read(256 * 1024 + 1) if stat.S_ISREG(os.fstat(f.fileno()).st_mode) else b""
        if not data or len(data) > 256 * 1024 or hashlib.sha256(data).hexdigest() != g.fix.get("sha256"):
            return ""
        text = data.decode("utf-8", "replace")
        if text.count("\n") > limit:
            return ""
        return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "?", text)

    def finish(self, error: str) -> int:
        res = dict(self.baseline)
        res.update(self.hard)
        counts = Counter(r.verdict for r in res.values())
        n_fail = sum(v for k, v in counts.items() if k not in OK_VERDICTS)
        unknown = [g for g in self.groups if g.kind != "known"]
        open_unknown = [g for g in unknown if not g.status.startswith("fixed")]
        if error:
            headline, code = f"INFRASTRUCTURE PROBLEM: {error[:300]}", EXIT_INFRA
        elif not self.groups:
            headline, code = "GREEN: every DFB test passes or skips", EXIT_GREEN
        elif not open_unknown:
            headline = (
                f"NO NEW FAILURES: {len(self.groups)} known issue group(s)"
                if not unknown
                else f"FIXED: {len(unknown)} failure group(s) fixed"
            )
            code = EXIT_GREEN
        else:
            headline = f"FAILURES: {len(open_unknown)} failure group(s) not fixed"
            code = EXIT_FAILURES
        unpublished = [
            g
            for g in self.groups
            if self.mode == "pr" and g.fix.get("verified") and not g.fix.get("url") and not g.fix.get("queued")
        ]
        if unpublished:
            headline += f"; {len(unpublished)} verified fix(es) not published (see Fixes)"
        if self.claude_down:
            headline += f" (the agent was not reachable: {self.claude_down[:120]})"
            code = EXIT_INFRA
        elif self.repair_stop:
            headline += f" (repairs stopped: {self.repair_stop})"
            code = EXIT_INFRA
        headline = " ".join(headline.split())  # one line: it goes into STATUS, tables and terminal banners
        minutes = round((time.time() - self.t0) / 60)
        base = self.cfg["target"].get("base", "main")
        lines = [
            f"# DFB repair loop: {utc_now().strftime('%Y-%m-%d %H:%M UTC')} ({self.mode} mode)",
            "",
            f"**{headline}**",
            "",
            "| | |",
            "|---|---|",
            f"| Tested | `{self.args.ref or 'origin/' + base}` {self.base_sha[:11]}"
            + (f" + carried: {md_cell('; '.join(self.carried))}" if self.carried else "")
            + f" (`{self.target_sha[:11]}`) |",
            f"| Simulator | ttsim {self.sim_version or '?'}, Quasar, slow dispatch, no watcher |",
            f"| Tests | {sum(counts.values())} run ({len(self.manifest['device'])} device, "
            f"{len(self.manifest['host'])} host): {counts['PASS']} pass, {n_fail} fail, {counts['SKIP']} skip, "
            f"{len(self.flaky)} flaky |",
            f"| Time | {minutes} min ({', '.join(f'{k} {v}' for k, v in self.phases.items())}) |",
            f"| Agent | {self.agent_label()}: {self.attempts_used} attempt(s), "
            + (
                f"{self.ai_credits:.1f} AI credits, {self.premium_requests:g} premium requests |"
                if self.copilot()
                else f"${self.cost_usd:.2f} reported |"
            ),
            "",
        ]
        if self.groups:
            lines += ["## Failure groups", "", "| Group | Kind | Tests | Status |", "|---|---|---|---|"]
            for g in self.groups:
                sample = ", ".join(f"`{t.split('.', 1)[1]}`" for t in g.tests[:3]) + (
                    " ..." if len(g.tests) > 3 else ""
                )
                kind = g.kind + (f" ({g.known})" if g.known else "")
                lines.append(
                    f"| `{g.gid}` {md_cell(g.signature[:110])} | {md_cell(kind)} | {len(g.tests)}: {sample} "
                    f"| {md_cell(g.status)} |"
                )
            lines.append("")
        fixes = [g for g in self.groups if g.fix]
        if fixes:
            lines += ["## Fixes", ""]
            for g in fixes:
                where = g.fix.get("url") or f"{g.fix['dir']}/patch.diff"
                extra = "" if g.fix.get("verified") else f" (not fully verified: {g.fix.get('why_unverified', '')})"
                if g.status.startswith("fixed, not published"):
                    extra += f". {g.status[:400]}"
                lines.append(f"- `{g.gid}` {g.fix['title']}: {where}{extra}")
                diff = self.small_patch(g)
                if diff:  # a fence longer than any run of backticks in the patch, so the patch cannot end it
                    fence = "`" * max(3, 1 + max((len(m) for m in re.findall(r"`+", diff)), default=0))
                    lines += ["", f"{fence}diff", diff.rstrip("\n"), fence, ""]
            lines.append("")
        waiting = [g for g in self.groups if g.handed_off or "waiting for a human" in g.status]
        if waiting:
            lines += ["## Needs human help", ""]
            lines += [
                f"- `{g.gid}` {g.signature[:100]}: "
                + (f"{self.run_dir}/handoff/{g.gid}.md" if g.handed_off else g.status)
                for g in waiting
            ]
            lines.append("")
        if self.flaky:
            lines += ["## Flaky (failed once, passed on a rerun)", "", *[f"- `{t}`" for t in self.flaky], ""]
        if self.notes:
            lines += ["## Notes", "", *[f"- {n}" for n in self.notes], ""]
        lines += [
            "## Where things are",
            "",
            f"- Run directory: {self.run_dir} (orchestrator.log, build.log, manifest.json, baseline/, "
            f"attempts/, fixes/, handoff/)",
            "",
        ]
        summary_md = "\n".join(lines)
        summary = {
            "date": utc_now().isoformat(timespec="seconds"),
            "mode": self.mode,
            "headline": headline,
            "exit_code": code,
            "base_sha": self.base_sha,
            "target_sha": self.target_sha,
            "ref": self.args.ref,
            "carried": self.carried,
            "sim_version": self.sim_version,
            "counts": dict(counts),
            "flaky": self.flaky,
            "minutes": minutes,
            "phases": self.phases,
            "attempts": self.attempts_used,
            "agent": self.agent_label(),
            "cost_usd": round(self.cost_usd, 2),
            "ai_credits": round(self.ai_credits, 2),
            "premium_requests": self.premium_requests,
            "notes": self.notes,
            "canary": bool(self.args.state),
            "dry_run": bool(self.args.publish_dry_run),
            "groups": [
                {
                    "gid": g.gid,
                    "kind": g.kind,
                    "signature": g.signature,
                    "tests": g.tests,
                    "status": g.status,
                    "attempts": g.attempts,
                    "fix": g.fix,
                }
                for g in self.groups
            ],
        }
        for name in ("summary.json", "summary.md"):  # whatever an agent may have left under these names goes first
            p = self.run_dir / name
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)
        write_json(self.run_dir / "summary.json", summary)  # first: CI reads the queue from it
        (self.run_dir / "summary.md").write_text(summary_md)
        if not self.args.state:  # a canary or test run with its own state file leaves the nightly status alone
            status = (
                f"{utc_now().strftime('%Y-%m-%d %H:%M')} UTC {self.mode}: {headline[:160]} "
                f"({counts['PASS']} pass/{n_fail} fail/{counts['SKIP']} skip, {minutes} min) "
                f"- {self.run_dir}/summary.md"
            )
            (self.root / "STATUS").write_text(status + "\n")
        try:
            self.update_state(headline, counts)
        except OSError as e:
            self.log(f"could not save the state file: {e}")
        self.log(f"done: {headline} (exit {code})")
        return code

    # ----- main ------------------------------------------------------------------------------------------------------

    def on_sigterm(self, signum, frame) -> None:
        self.deadline = 0
        self.kill_strays()
        raise Infra("stopped by SIGTERM (the hard time limit or a manual stop)")

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.on_sigterm)
        error = ""
        try:
            self.log(f"DFB repair loop: mode {self.mode}, run dir {self.run_dir}")
            t = time.time()
            self.prepare_checkout()
            self.phase("checkout", t)
            self.log(
                f"target {self.target_sha[:11]} = {self.args.ref or 'origin/' + self.cfg['target'].get('base', 'main')} "
                f"{self.base_sha[:11]} + {len(self.carried)} carried commit(s)"
            )
            self.install_sim()
            self.write_loop_env()
            t = time.time()
            if not self.build(self.run_dir / "build.log"):
                if not self.carried:
                    raise Infra("the build failed: " + tail_lines(self.run_dir / "build.log", 12))
                self.drop_carry("(all)", "the build failed with the carried commits: " + "; ".join(self.carried))
                self.carried = []
                self.reset_tree(self.base_sha)
                self.target_sha = self.base_sha
                self.save_git_files()
                if not self.build(self.run_dir / "build_without_carry.log"):
                    raise Infra("the build failed: " + tail_lines(self.run_dir / "build_without_carry.log", 12))
            self.phase("build", t)
            self.ensure_bin()
            self.make_manifest()
            t = time.time()
            self.baseline = self.run_suite(self.run_dir / "baseline", "baseline suite")
            self.classify()
            self.phase("suite", t)
            t = time.time()
            self.repair()
            self.open_prs()
            if self.mode != "report":
                self.phase("repair", t)
        except Infra as e:
            error = str(e)
            self.log(f"INFRASTRUCTURE PROBLEM: {error}")
        except Exception as e:  # noqa: BLE001 - the summary must still be written
            error = f"the orchestrator crashed: {type(e).__name__}: {e}"
            self.log(traceback.format_exc())
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            self.kill_strays()
            self.kill_others()
            if self.args.outbox and not self.outbox_done:  # the run stopped before queueing: nothing goes out
                try:
                    self.clear_outbox()
                except OSError as e:
                    self.log(f"could not clear the outbox: {e}")
            if self.target_sha:
                try:
                    self.reset_tree(self.target_sha)
                except Infra as e:
                    self.log(f"could not reset the checkout: {e}")
            shutil.rmtree(self.run_dir / "cache", ignore_errors=True)
            for c in list(self.run_dir.glob("attempts/*/*/cache*")) + list(
                self.run_dir.glob("attempts/*/*/test_runs/cache.*")
            ):
                shutil.rmtree(c, ignore_errors=True)
        code = self.finish(error)  # the summary and state are saved before the slow part below
        if (
            self.target_sha
            and self.deadline
            and not self.args.own_pid_namespace  # in CI the checkout goes away with the container
            and (self.attempts_used or any(g.fix for g in self.groups))
        ):
            try:  # leave the checkout's binary built from the tested tree; skipped after a stop
                self.drop_generated()
                self.build(self.run_dir / "final_build.log")
            except Infra as e:
                self.log(f"the final rebuild failed: {e}")
            shutil.rmtree(self.run_dir / "cache", ignore_errors=True)
        return code


def check_patch(args) -> int:
    """--check-patch: run the guards on a patch that is already applied to --checkout on top of --base. CI's publish job
    uses it on each queued fix, from its own checkout, before pushing anything."""
    tool = Path(args.tool_dir).resolve() if args.tool_dir else Path(__file__).resolve().parent
    loop = Loop.__new__(Loop)
    loop.args, loop.cfg = args, yaml.safe_load((tool / "repair_budget.yaml").read_text())
    loop.checkout = loop.root = loop.run_dir = Path(args.checkout or ".").resolve()
    loop.marker, loop.target_sha = "DFB_LOOP_CHILD=check-patch", args.base
    problems, review = loop.check_guards(read_text(args.check_patch))
    for p in problems + review:
        print(p)
    return 1 if problems or review else 0


def agent_check(args) -> int:
    """--agent-check: ask the configured agent for one word through the same code as an attempt, from an empty
    directory, and print the model, the reply, the cost and any error. Everything it writes is removed at the end."""
    root = Path(args.root).resolve()

    def stop(signum, frame):  # a closed terminal or a kill still removes what the check wrote
        raise SystemExit(128 + signum)

    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, stop)
    work = Path(tempfile.mkdtemp(prefix=".agent-check.", dir=root))
    try:
        (work / "empty").mkdir()
        ns = argparse.Namespace(**vars(args))
        ns.run_dir, ns.checkout, ns.state = str(work / "run"), str(work / "empty"), str(work / "state.json")
        loop = Loop(ns)
        adir = loop.run_dir / "attempts" / "check" / "1"
        adir.mkdir(parents=True)
        (adir / "prompt.md").write_text("Reply with exactly one word: OK")
        env = loop.agent_env(adir, loop.run_dir / "cache" / "agent-home" / "check")
        prompt_in = os.devnull if loop.copilot() else adir / "prompt.md"
        with (
            open(prompt_in, "rb") as stdin,
            open(adir / "agent.jsonl", "wb") as out,
            open(adir / "agent.err", "wb") as err,
        ):
            rc = loop.run_agent(adir, 5, env, stdin, out, err)
        kill_marked(f"DFB_ATTEMPT_DIR={adir}")
        rep = loop.agent_report(adir)
        ok = "OK" in rep["said"]
        cost = f"{rep['ai_credits']:.2f} AI credits" if loop.copilot() else f"${rep['cost_usd']:.4f}"
        print(
            f"{'OK' if ok else 'FAILED'}: {loop.agent_label()}: exit {rc}; model {rep['model'] or '?'}; "
            f"reply {rep['said'].strip()[:200]!r}; {cost}"
        )
        want = loop.cfg["repair"]["copilot"]["model"] if loop.copilot() else loop.cfg["repair"]["model"]
        if rep["model"] and rep["model"] != want:
            print(f"note: the reply came from {rep['model']}, not {want}")
        if rep["error"]:
            print(f"error (status {rep['status']}): {rep['error'][:500]}")
        if not ok:
            print(" ".join(read_text(adir / "agent.err").split())[:1500])
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("--root", help="loop root (holds tool/, tt-metal/, runs/, state.json); required for a run")
    ap.add_argument("--run-dir", help="this run's directory; required for a run")
    ap.add_argument("--mode", choices=("report", "repair", "pr"), default="report")
    ap.add_argument("--ref", help="test this ref instead of origin/<base>, e.g. a local canary branch")
    ap.add_argument("--no-carry", action="store_true", help="do not apply the carried branches")
    ap.add_argument("--skip-fetch", action="store_true", help="do not fetch from origin")
    ap.add_argument("--checkout", help="tt-metal checkout to use (default: <root>/tt-metal)")
    ap.add_argument("--tool-dir", help="directory with repair_budget.yaml, prompt_template.md, run_test.sh")
    ap.add_argument("--sim-dir", help="existing ttsim install (libttsim.so + soc_descriptor.yaml); skips download")
    ap.add_argument("--state", help="state file (default <root>/state.json); with it, STATUS is not updated")
    ap.add_argument("--publish-dry-run", action="store_true", help="pr mode: build the branch, push with --dry-run")
    ap.add_argument(
        "--outbox",
        help="pr mode: instead of pushing, write each fix as a plain patch with its message (patch.diff, message.txt, "
        "base.txt, branch.txt, title.txt, pr_body.md) under this directory, for another job to commit and publish",
    )
    ap.add_argument(
        "--deadline-minutes",
        type=float,
        help="stop starting new work after this many minutes (default: budget.deadline_minutes); CI fits it to its job",
    )
    ap.add_argument(
        "--agent-check",
        action="store_true",
        help="only ask the configured agent for one word, exactly the way an attempt runs it (a fresh home, the same "
        "flags and credential), print what came back, and exit 1 unless it answered; needs --root",
    )
    ap.add_argument(
        "--check-patch",
        help="only run the guards on this patch, applied to --checkout on top of --base; "
        "exit 1 if it would be rejected or held for review",
    )
    ap.add_argument("--base", help="with --check-patch: the commit the patch was applied to")
    ap.add_argument(
        "--own-pid-namespace",
        action="store_true",
        help="the loop runs alone in a container (CI): after each agent attempt, stop every other process in it",
    )
    ap.add_argument(
        "--sanitize-file",
        help="only rewrite this file the way public text is written (no @mentions that ping people, no "
        "issue-closing keywords, at most 60000 characters), and exit",
    )
    args = ap.parse_args()
    if args.agent_check:
        if not args.root:
            ap.error("--agent-check needs --root")
        return agent_check(args)
    if args.sanitize_file:
        p = Path(args.sanitize_file)
        p.write_text(sanitize(read_text(p))[:60000])
        return 0
    if args.check_patch:
        if not args.base:
            ap.error("--check-patch needs --base")
        return check_patch(args)
    if not args.root or not args.run_dir:
        ap.error("--root and --run-dir are required")
    if args.outbox:
        args.outbox = str(Path(args.outbox).resolve())
    try:
        loop = Loop(args)
    except Exception as e:  # noqa: BLE001 - e.g. a broken repair_budget.yaml
        print(f"cannot start: {type(e).__name__}: {e}", flush=True)
        return EXIT_INFRA
    try:
        return loop.run()
    except Exception as e:  # noqa: BLE001 - e.g. the disk filled while writing the summary
        print(f"the run failed: {type(e).__name__}: {e}", flush=True)
        return EXIT_INFRA


if __name__ == "__main__":
    raise SystemExit(main())
