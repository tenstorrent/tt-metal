"""`dream`: launch and manage Dream-RSI campaigns.

User commands (run from any checkout; they act on the campaign's machine, locally or over ssh):
    dream init NAME [--machine HOST[:PORT]]     scaffold agent_orch/campaigns/NAME/{dream.yaml,brief.md}
    dream policies [--json]                     list the policy library
    dream check SPEC|NAME [--snapshot]          create the campaign root (from HEAD, or the working tree), set up
                                                the machine, build, measure the baseline
    dream start SPEC|NAME                       check if needed, then run the campaign in the background
    dream status NAME [--json]                  state, budget, best, last log lines
    dream watch NAME [--out FILE]               print a line (and refresh FILE) whenever the report changes
    dream report NAME [--out FILE]              regenerate the report and copy it here
    dream stop NAME | dream resume NAME
    dream fetch NAME                            bring refs/dream/NAME/* and the dream/NAME/best branch into this repo
    dream finalize NAME                         write dream/NAME/best from the best node now
    dream export-policy NAME --version vN --as POLICY [--description TEXT]
    dream delete NAME --yes
    dream install-skill                         link the /dream Claude Code skill into ~/.claude/skills

Machine-side commands (used by workers and the driver):
    dream eval --campaign C --node N            evaluate the worktree you are in
    dream commit --campaign C --node N          commit your node
    dream replay ...                            replay policies on recorded rounds (dreaming)
    dream verify|history ...                    debugging helpers
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from .campaign import (
    ORCH_DIR,
    Campaign,
    default_home,
    git_main_repo,
    load_campaign,
    load_spec,
    parse_machine,
    repo_root,
    spec_path_in,
    validate_spec,
)

REGISTRY = Path(os.environ.get("DREAM_REGISTRY", Path.home() / ".config" / "dream" / "campaigns.json"))
DONE_STATES = {"finished", "stopped", "blocked", "error", "crashed"}


def die(msg: str, code: int = 1):
    print(f"dream: {msg}", file=sys.stderr)
    sys.exit(code)


def say(msg: str):
    print(f"[dream] {msg}", flush=True)


# ================================================================ registry + machine access
def registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def register(name: str, entry: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    reg = registry()
    reg[name] = {**reg.get(name, {}), **entry}
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")


class Target:
    """Where a campaign runs: the machine, the repo path there and its DREAM_HOME."""

    def __init__(self, name: str, machine: str, repo: str, home: str, isolated: bool = True):
        self.name, self.machine, self.repo, self.home = name, machine, repo, home
        self.isolated = isolated
        self.m = parse_machine(machine)

    @property
    def campaign_repo(self) -> str:
        """Where the campaign's refs live on the machine: its own repo when isolated, else the user's repo."""
        return f"{self.home}/{self.name}/repo.git" if self.isolated else self.repo

    @property
    def remote(self) -> bool:
        return self.m is not None

    @property
    def ctl(self) -> str:
        return f"{self.home}/{self.name}/ctl"

    def ssh_base(self) -> list[str]:
        cmd = ["ssh", "-o", "BatchMode=yes"]
        if self.m["port"]:
            cmd += ["-p", str(self.m["port"])]
        return cmd + [(self.m["user"] + "@" if self.m["user"] else "") + self.m["host"]]

    def url(self, path: str | None = None) -> str:
        path = path or self.repo
        if not self.remote:
            return path
        u = (self.m["user"] + "@") if self.m["user"] else ""
        port = f":{self.m['port']}" if self.m["port"] else ""
        return f"ssh://{u}{self.m['host']}{port}{path}"

    def shell(self, script: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
        """Run a bash script on the machine."""
        env_home = f"export DREAM_HOME={shlex.quote(self.home)}; "
        if self.remote:
            args = self.ssh_base() + ["bash -lc " + shlex.quote(env_home + script)]
        else:
            args = ["bash", "-c", env_home + script]
        r = subprocess.run(args, capture_output=capture, text=True)
        if check and r.returncode != 0:
            die(
                f"command failed on {self.machine} (rc {r.returncode}): {script}"
                + (f"\n{r.stderr.strip()}" if capture else "")
            )
        return r

    def dream(self, *args: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
        """Run this campaign's pinned tools (the ctl checkout at the campaign root) on the machine."""
        return self.shell(
            f"cd {shlex.quote(self.ctl)} && agent_orch/bin/dream " + " ".join(map(shlex.quote, args)),
            capture=capture,
            check=check,
        )

    def dream_json(self, *args: str) -> dict:
        r = self.dream(*args, capture=True)
        return json.loads(r.stdout.strip().splitlines()[-1])


def target_from_spec(spec: dict, local_repo: Path) -> Target:
    return Target(
        spec["name"],
        spec.get("machine") or "local",
        spec.get("repo") or str(local_repo),
        spec.get("dream_home") or default_home(),
        bool((spec.get("isolation") or {}).get("enabled", True)),
    )


def target(name: str) -> Target:
    e = registry().get(name)
    if not e:
        p = spec_path_in(git_main_repo(Path.cwd()), name)
        if not p.exists():
            die(f"unknown campaign '{name}' (not in {REGISTRY}, no {p})")
        spec = load_spec(p)
        spec["name"] = name
        return target_from_spec(spec, git_main_repo(Path.cwd()))
    return Target(name, e["machine"], e["repo"], e["home"], e.get("isolated", False))


# ================================================================ init / policies
SPEC_TEMPLATE = """# Dream-RSI campaign spec. Only name, editable and eval.command are required.
name: {name}
description: "<one line: what is being optimized and on what hardware>"
machine: {machine}          # local, host or host:port (an IRD reservation's ssh port)
# repo: /localdev/$USER/tt-metal   # repo path on the machine (default: same path as this checkout)

editable:                  # files workers may change (fnmatch globs, repo-relative)
  - ttnn/cpp/ttnn/operations/<path to your op>/*
brief: brief.md            # what the op does and what is in scope; read by every worker
rules: []                  # told to every worker, e.g. "Keep MathFidelity::HiFi4"
forbidden_patterns: []     # regexes; code matching one is invalid, e.g. 'MathFidelity::(HiFi2|LoFi)'

eval:
  # Must write the result JSON to $DREAM_RESULT (see agent_orch/README.md). For a tt-metal op test:
  command: >-
    python agent_orch/adapters/ttmetal_op_perf.py
    --test tests/<path to your test>.py --op-code <OP CODE column in the ops perf CSV>
    --warmup 3 --measured 10
  direction: minimize      # minimize | maximize
  unit: us
  gates: []                # per-case checks on extra fields, e.g. ["pcc >= 0.99999"]
  timeout_s: 900

budget: {{max_attempts: 40, max_hours: 6, max_usd: 200}}   # hard limits
search: {{policy: fresh, W: 4, R: 4, max_rounds: 4, round_root: origin}}   # `dream policies` lists the policies;
                           # round_root: origin (each round from the campaign root) | best (from the best so far)
"""

BRIEF_TEMPLATE = """# {name}

## What the code does
<the op, its inputs and outputs, where it is used>

## Where the code is
<program factory, kernels, host op files; anything a worker must read first>

## What is measured
<the test, the cases, what the metric means>

## Known ideas and constraints
<optional: what has been tried, what must not change>
"""


def cmd_init(a):
    repo = git_main_repo(Path.cwd())
    d = repo / "agent_orch" / "campaigns" / a.name
    if (d / "dream.yaml").exists():
        die(f"{d / 'dream.yaml'} exists")
    d.mkdir(parents=True, exist_ok=True)
    (d / "dream.yaml").write_text(SPEC_TEMPLATE.format(name=a.name, machine=a.machine or "local"))
    if not (d / "brief.md").exists():
        (d / "brief.md").write_text(BRIEF_TEMPLATE.format(name=a.name))
    say(f"wrote {d / 'dream.yaml'} and {d / 'brief.md'}; fill them in, commit your test, then `dream check {a.name}`")


def cmd_policies(a):
    from .policy_lib import list_policies

    ps = list_policies()
    if a.json:
        print(json.dumps(ps, indent=2))
        return
    for p in ps:
        o = p["origin"]
        org = (
            f"dreamed on {o.get('campaign')} ({o.get('version')}, replay V {o.get('replay_V')})"
            if o.get("kind") == "dreamed"
            else o.get("kind", "")
        )
        al = f" (alias: {', '.join(p['aliases'])})" if p["aliases"] else ""
        print(f"{p['name']}{al}\n    {org}\n    {p['description']}\n")


# ================================================================ check / start
def spec_from_arg(arg: str) -> tuple[dict, Path, Path]:
    """(spec, spec dir, local repo) from a spec path or a campaign name."""
    p = Path(arg)
    if p.suffix not in (".yaml", ".yml"):
        p = spec_path_in(git_main_repo(Path.cwd()), arg)
    if not p.exists():
        die(f"no spec at {p}")
    spec = load_spec(p)
    if not spec.get("name"):
        spec["name"] = p.parent.name
    errs = validate_spec(spec)
    if errs:
        die("invalid spec:\n  " + "\n  ".join(errs))
    return spec, p.parent.resolve(), git_main_repo(p.parent)


def campaign_files(spec: dict, spec_dir: Path) -> dict[str, bytes]:
    """Files of the campaign directory that go into the root commit."""
    rel = f"agent_orch/campaigns/{spec['name']}"
    files = {}
    for f in sorted(spec_dir.rglob("*")):
        r = f.relative_to(spec_dir)
        if f.is_file() and r.parts[0] not in ("attempts", "export") and not r.name.startswith("."):
            files[f"{rel}/{r.as_posix()}"] = f.read_bytes()
    files[f"{rel}/dream.yaml"] = (
        (spec_dir / "dream.yaml").read_bytes()
        if (spec_dir / "dream.yaml").exists()
        else files.get(f"{rel}/dream.yaml", b"")
    )
    if spec.get("brief") and (spec_dir / spec["brief"]).exists():
        files[f"{rel}/brief.md"] = (spec_dir / spec["brief"]).read_bytes()
    return files


def ensure_root(spec: dict, spec_dir: Path, repo: Path, t: Target, snapshot: bool = False) -> str:
    from .gitops import git, make_root_commit, snapshot as snap_worktree

    name = spec["name"]
    files = campaign_files(spec, spec_dir)
    root_ref, base_ref = f"refs/dream/{name}/root", f"refs/dream/{name}/base"
    existing = git("rev-parse", "-q", "--verify", f"{root_ref}^{{commit}}", cwd=repo, check=False)
    if existing:
        base = git("rev-parse", f"{base_ref}^{{commit}}", cwd=repo)
        fresh = make_root_commit(repo, base, name, files)
        if git("rev-parse", f"{fresh}^{{tree}}", cwd=repo) != git("rev-parse", f"{existing}^{{tree}}", cwd=repo):
            die(
                f"campaign '{name}' already exists with different campaign files. Use a new name, "
                f"or `dream delete {name} --yes` first."
            )
        root = existing
        say(f"campaign root exists: {root[:12]}")
    else:
        dirty = [
            l for l in git("status", "--porcelain", cwd=repo).splitlines() if f"agent_orch/campaigns/{name}/" not in l
        ]
        if snapshot:
            base = snap_worktree(repo, f"agent_orch/campaigns/{name}", f"[dream:{name}] working-tree snapshot")
            say(f"base = snapshot of the working tree ({len(dirty)} uncommitted change(s) included)")
        else:
            if dirty:
                say(
                    f"note: {len(dirty)} uncommitted change(s) in {repo} are NOT part of the campaign "
                    f"(it starts from HEAD; --snapshot includes them). First: {dirty[0].strip()}"
                )
            base = git("rev-parse", "HEAD", cwd=repo)
        root = make_root_commit(repo, base, name, files)
        git("update-ref", base_ref, base, cwd=repo)
        git("update-ref", root_ref, root, cwd=repo)
        say(f"campaign root {root[:12]} = HEAD {base[:12]} + agent_orch/campaigns/{name}/")
    if t.remote:
        say(f"pushing the campaign root to {t.url()}")
        r = subprocess.run(
            ["git", "push", "-q", t.url(), f"{base_ref}:{base_ref}", f"{root_ref}:{root_ref}"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 and "already exists" not in r.stderr and "up to date" not in r.stderr:
            die(f"git push to {t.url()} failed:\n{r.stderr.strip()}")
    return root


def cmd_check(a):
    spec, spec_dir, repo = spec_from_arg(a.spec)
    t = target_from_spec(spec, repo)
    ensure_root(spec, spec_dir, repo, t, snapshot=getattr(a, "snapshot", False))
    register(
        t.name,
        {
            "machine": t.machine,
            "repo": t.repo,
            "home": t.home,
            "spec": str(spec_dir / "dream.yaml"),
            "local_repo": str(repo),
            "isolated": t.isolated,
        },
    )
    q = shlex.quote
    n, cr = t.name, t.campaign_repo
    if t.isolated:
        # the campaign's own repo: only base/root and the campaign's refs, no remotes; objects are shared with the
        # user's repo through alternates, so this is instant and uses no extra disk
        say(f"isolated campaign repo {cr} on {t.machine}")
        t.shell(
            f"cd {q(t.repo)} && mkdir -p {q(t.home + '/' + n)} && "
            f"if [ ! -d {q(cr)} ]; then git init -q --bare {q(cr)} && "
            f'echo "$(git rev-parse --path-format=absolute --git-common-dir)/objects" > {q(cr)}/objects/info/alternates'
            f" && git -C {q(cr)} config dream.mainRepo {q(t.repo)}; fi && "
            f"git -C {q(cr)} fetch -q {q(t.repo)} refs/dream/{n}/base:refs/dream/{n}/base "
            f"refs/dream/{n}/root:refs/dream/{n}/root && "
            f"( [ -d {q(t.ctl)} ] || git -C {q(cr)} worktree add -q --detach {q(t.ctl)} refs/dream/{n}/root )"
        )
    else:
        t.shell(
            f"cd {q(t.repo)} && mkdir -p {q(t.home + '/' + n)} && "
            f"( [ -d {q(t.ctl)} ] || git worktree add -q --detach {q(t.ctl)} refs/dream/{n}/root )"
        )
    say(f"setting up on {t.machine} (ledger, eval checkout)")
    t.dream("_setup", t.name)
    say(
        "measuring the baseline: the first run builds the eval checkout (30-60 min from scratch), then runs the "
        f"test {spec['eval']['baseline_runs']} times; progress lines follow"
    )
    args = ["_baseline", t.name] + (["--force"] if a.rebaseline else [])
    t.dream(*args)
    report = f"{t.home}/{t.name}/report/index.html"
    say(f"ready. The report so far is {report} on {t.machine}.")
    say(f"Start with: dream start {t.name}. For a live page that updates every step, run /dream in Claude Code")
    say(f"(it publishes the report and follows the campaign), or copy it here with: dream report {t.name} --out FILE")


def cmd_start(a, resume: bool = False):
    if resume:
        t = target(a.spec)
    else:
        spec, spec_dir, repo = spec_from_arg(a.spec)
        t = target_from_spec(spec, repo)
        st = (
            t.dream_json("_status", t.name)
            if t.shell(f"[ -d {shlex.quote(t.ctl)} ]", check=False).returncode == 0
            else {"has_baseline": False}
        )
        if not st.get("has_baseline"):
            cmd_check(argparse.Namespace(spec=a.spec, rebaseline=False, snapshot=getattr(a, "snapshot", False)))
    st = t.dream_json("_status", t.name)
    if st.get("alive"):
        die(f"{t.name} is already running (pid {st.get('pid')})")
    log = f"{t.home}/{t.name}/logs/driver.log"
    q = shlex.quote
    t.shell(
        f"mkdir -p {q(t.home + '/' + t.name + '/logs')} && cd {q(t.ctl)} && "
        f"nohup setsid agent_orch/bin/dream _run {q(t.name)} >> {q(log)} 2>&1 < /dev/null & sleep 3"
    )
    st = t.dream_json("_status", t.name)
    if not st.get("alive"):
        die(f"the driver did not stay up; see {log} on {t.machine}:\n" + "\n".join(st.get("log_tail", [])))
    register(t.name, {"started": datetime.datetime.now().isoformat(timespec="seconds")})
    say(f"{t.name} is running on {t.machine} (pid {st['pid']}). Follow it with: dream watch {t.name}")


# ================================================================ status / report / watch
def cmd_status(a):
    t = target(a.name)
    st = t.dream_json("_status", t.name)
    if a.json:
        print(json.dumps(st, indent=2))
        return
    b = st.get("budget", {})
    print(f"{t.name} on {t.machine}: {st.get('state')}{' (driver alive)' if st.get('alive') else ''}")
    if st.get("message"):
        print(f"  {st['message']}")
    if b:
        print(
            f"  attempts {b['attempts']}/{b['max_attempts']}, {b['hours']:.2f}/{b['max_hours']} h, "
            f"${b['usd']:.2f}/${b['max_usd']:.0f}"
        )
    if st.get("best"):
        print(f"  best {st['best']['node']} score {st['best']['score']:.4f}: {st['best']['mechanism']}")
    for line in st.get("log_tail", [])[-5:]:
        print(f"  | {line}")


def fetch_report(t: Target, out: Path) -> Path:
    path = t.dream_json("_report", t.name)["path"]
    if t.remote:
        r = subprocess.run(t.ssh_base() + ["cat " + shlex.quote(path)], capture_output=True, check=True)
        out.write_bytes(r.stdout)
    else:
        out.write_bytes(Path(path).read_bytes())
    return out


def cmd_report(a):
    t = target(a.name)
    out = Path(a.out or f"dream-{t.name}.html").resolve()
    print(fetch_report(t, out))


def cmd_watch(a):
    t = target(a.name)
    out = Path(a.out or f"dream-{t.name}.html").resolve()
    last = None
    while True:
        try:
            st = t.dream_json("_status", t.name)
        except SystemExit:
            print("WARN status unavailable; retrying", flush=True)
            time.sleep(a.interval)
            continue
        key = (st.get("state"), st.get("updated"), st.get("report_mtime"))
        if key != last:
            last = key
            fetch_report(t, out)
            b = st.get("budget", {})
            best = st.get("best") or {}
            tag = "FINAL" if st.get("state") in DONE_STATES else "UPDATE"
            print(
                f"{tag} state={st.get('state')} attempts={b.get('attempts')}/{b.get('max_attempts')} "
                f"usd={b.get('usd')} best={best.get('score', '-')} ({best.get('node', '-')}) "
                f"report={out} msg={st.get('message', '')!r}",
                flush=True,
            )
            if tag == "FINAL":
                if st.get("state") == "finished":
                    cmd_fetch(argparse.Namespace(name=t.name, quiet=True))
                return
        time.sleep(a.interval)


# ================================================================ stop / fetch / finalize / export / delete
def cmd_stop(a):
    t = target(a.name)
    print(json.dumps(t.dream_json("_stop", t.name)))


def cmd_fetch(a):
    t = target(a.name)
    repo = Path(registry().get(t.name, {}).get("local_repo") or git_main_repo(Path.cwd()))
    if not t.remote and not t.isolated:
        say("campaign runs in this repo: refs are already here")
        return
    src = t.url(t.campaign_repo)
    specs = [f"+refs/dream/{t.name}/*:refs/dream/{t.name}/*"]
    subprocess.run(["git", "fetch", "-q", src, *specs], cwd=repo, check=True)
    r = subprocess.run(
        ["git", "fetch", "-q", src, f"+refs/heads/dream/{t.name}/best:refs/heads/dream/{t.name}/best"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if not getattr(a, "quiet", False):
        say(
            f"fetched refs/dream/{t.name}/*"
            + (f" and branch dream/{t.name}/best" if r.returncode == 0 else " (no dream/<c>/best branch yet)")
        )


def cmd_finalize(a):
    t = target(a.name)
    res = t.dream_json("_finalize", t.name)
    print(json.dumps(res))
    if res.get("branch"):
        cmd_fetch(argparse.Namespace(name=t.name))


def cmd_export_policy(a):
    from .policy_lib import write_policy

    t = target(a.name)
    dump = t.dream_json("_policy-dump", t.name, "--version", a.version)
    rep = dump.get("replay") or {}
    meta = {
        "description": a.description or dump.get("title", ""),
        "origin": {
            "kind": "dreamed",
            "campaign": t.name,
            "version": a.version,
            "machine": t.machine,
            "rounds_replayed": rep.get("rounds"),
            "replay_V": rep.get("objective_V"),
            "default_beta": rep.get("default_beta"),
        },
        "notes": dump.get("notes", ""),
        "exported_at": datetime.date.today().isoformat(),
    }
    d = write_policy(a.as_name, dump["policy"], meta, force=a.force)
    say(f"exported {t.name} {a.version} to {d}; commit it so other campaigns can use it")


def cmd_delete(a):
    if not a.yes:
        die("this removes the campaign's refs, worktrees and $DREAM_HOME data; pass --yes")
    t = target(a.name)
    if t.shell(f"[ -d {shlex.quote(t.ctl)} ]", check=False).returncode == 0:
        st = t.dream_json("_status", t.name)
        if st.get("alive"):
            die(f"{t.name} is running; `dream stop {t.name}` first")
    q = shlex.quote
    h = f"{t.home}/{t.name}"
    unlink = (
        ""
        if t.isolated  # its worktrees belong to the campaign repo, which goes with the directory
        else f"for w in {q(h)}/wt/* {q(h)}/eval {q(h)}/ledger {q(h)}/ctl; do "
        f'[ -e "$w" ] && git worktree remove --force "$w"; done; '
    )
    t.shell(
        f"cd {q(t.repo)} && {unlink}rm -rf {q(h)}; git worktree prune; "
        f"git for-each-ref --format='%(refname)' refs/dream/{q(t.name)}/ | xargs -r -n1 git update-ref -d; "
        f"git update-ref -d refs/heads/dream/{q(t.name)}/best 2>/dev/null",
        check=False,
    )
    local = registry().get(t.name, {}).get("local_repo")
    if local and t.remote:
        subprocess.run(
            f"git for-each-ref --format='%(refname)' refs/dream/{q(t.name)}/ | xargs -r -n1 git update-ref -d",
            shell=True,
            cwd=local,
        )
    reg = registry()
    reg.pop(t.name, None)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    say(f"deleted {t.name} (the dream/{t.name}/best branch in this repo, if fetched, is kept)")


def cmd_install_skill(a):
    src = ORCH_DIR / "skill" / "dream"
    dst = Path.home() / ".claude" / "skills" / "dream"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            say(f"{dst} already links to {src}")
            return
        die(f"{dst} exists; remove it first")
    dst.symlink_to(src)
    say(f"linked {dst} -> {src}; /dream is available in new Claude Code sessions")


# ================================================================ machine-side commands
def here_campaign(name: str) -> Campaign:
    return load_campaign(name, repo_root(Path.cwd()))


def m_setup(a):
    from .gitops import forbidden_hits, ledger_init

    c = here_campaign(a.name)
    hits = forbidden_hits(c, c.ref_root())
    if hits:
        die(f"the campaign root already matches forbidden_patterns: {hits}")
    ledger_init(c, c.cfg["search"]["policy"])
    ev = c.eval_checkout
    if not ev.exists():
        c.git("worktree", "add", "-q", "--detach", str(ev), c.ref_root())
        say(f"eval checkout at {ev}; initializing submodules")
        c.git("submodule", "update", "--init", "--recursive", cwd=ev)
    from .driver import write_status

    st = json.loads((c.home / "status.json").read_text()) if (c.home / "status.json").exists() else {}
    if st.get("state") in (None, "not started"):
        write_status(c, "set up", "campaign set up; baseline next")
    say(f"ledger at {c.ledger} (active policy {(c.ledger / 'policies' / 'ACTIVE').read_text().strip()})")


def m_baseline(a):
    from .driver import write_status
    from .evaluate import load_baseline, make_baseline_runs
    from .gitops import ledger_commit
    from . import report

    c = here_campaign(a.name)
    if load_baseline(c) and not a.force:
        b = load_baseline(c)
        say(f"baseline exists (noise ±{b['noise_pct']}%); pass --rebaseline to re-measure")
    else:
        b = make_baseline_runs(c)
        ledger_commit(c, "baseline")
        write_status(c, "ready", f"baseline measured ({b['runs']} runs, noise ±{b['noise_pct']}%)")
    unit = c.cfg["eval"].get("unit", "")
    for cid, v in b["cases"].items():
        say(f"  {cid:32s} {v['value']:.6g} {unit}  (runs {v['runs']}, spread {v['spread_pct']}%)")
    say(f"noise band ±{b['noise_pct']}%; report: {report.build(c)}")


def m_run(a):
    from .driver import run

    run(here_campaign(a.name))


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def m_status(a):
    from .driver import Clock, budget, global_best, read_status
    from .evaluate import load_baseline

    c = here_campaign(a.name)
    st = read_status(c)
    alive = st.get("state") == "running" and _alive(st.get("pid"))
    if st.get("state") == "running" and not alive:
        st["state"], st["message"] = "crashed", "the driver is not running (crashed or killed); `dream resume`"
    out = {**st, "alive": alive, "has_baseline": load_baseline(c) is not None}
    if c.ledger.exists():
        out["budget"] = budget(c, Clock(c))
        best = global_best(c)
        if best:
            out["best"] = {"node": best.node_id, "score": best.score, "mechanism": best.mechanism}
    rep = c.home / "report" / "index.html"
    out["report_mtime"] = rep.stat().st_mtime if rep.exists() else None
    log = c.logs / "driver.log"
    out["log_tail"] = log.read_text(errors="replace").splitlines()[-8:] if log.exists() else []
    print(json.dumps(out))


def m_report(a):
    from . import report

    print(json.dumps({"path": str(report.build(here_campaign(a.name)))}))


def m_stop(a):
    from .driver import read_status, write_status

    c = here_campaign(a.name)
    st = read_status(c)
    killed = []
    sess = c.home / "sessions"
    pids = [int(p.stem) for p in sess.glob("*.pid")] if sess.exists() else []
    if st.get("state") == "running" and _alive(st.get("pid")):
        pids.insert(0, int(st["pid"]))
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(3)
    for pid in killed:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if sess.exists():
        for p in sess.glob("*.pid"):
            p.unlink()
    if st.get("state") not in DONE_STATES - {"crashed"}:
        write_status(c, "stopped", "stopped by the user; `dream resume` continues where it stopped")
    print(json.dumps({"stopped": True, "killed_process_groups": killed}))


def m_finalize(a):
    from .driver import finalize

    print(json.dumps(finalize(here_campaign(a.name)) or {"branch": None, "why": "no valid attempt"}))


def m_policy_dump(a):
    c = here_campaign(a.name)
    d = c.ledger / "policies" / a.version
    if not (d / "policy.py").exists():
        die(f"no policy {a.version} in {c.ledger / 'policies'}")
    notes = (d / "notes.md").read_text() if (d / "notes.md").exists() else ""
    rep = json.loads((d / "replay.json").read_text()) if (d / "replay.json").exists() else None
    if rep:
        rep = {k: rep.get(k) for k in ("objective_V", "rounds", "default_beta")}
    title = next((l.lstrip("# ").strip() for l in notes.splitlines() if l.startswith("#")), "")
    print(json.dumps({"policy": (d / "policy.py").read_text(), "notes": notes, "replay": rep, "title": title}))


def w_eval(a):
    from .evaluate import eval_node

    wt = repo_root(Path.cwd())
    c = load_campaign(a.campaign, wt)
    res = eval_node(c, wt, a.node)
    print(json.dumps({k: res.get(k) for k in ("valid", "fail_class", "score")}))


def w_commit(a):
    from .gitops import commit_node

    wt = repo_root(Path.cwd())
    try:
        print(json.dumps(commit_node(load_campaign(a.campaign, wt), wt, a.node)))
    except RuntimeError as e:
        die(str(e))


def w_verify(a):
    from .gitops import verify_node

    r = verify_node(here_campaign(a.campaign), a.node, record=a.record)
    print(json.dumps(r))
    sys.exit(0 if r["ok"] else 1)


def w_history(a):
    from .history import write_md

    c = here_campaign(a.campaign)
    write_md(c, c.home / "history.md")
    print(c.home / "history.md")


def w_replay(a):
    from .evaluate import load_baseline
    from .replay import replay_policy
    from .tree import load_fixture, load_round, recorded_rounds

    if a.campaign:
        c = here_campaign(a.campaign)
        dcfg = c.cfg["dreaming"]
        s = c.cfg["search"]
        defaults = {"W": int(s["W"]), "R": int(s["R"]), "beta": float(s["beta"])}
        wanted = [int(x) for x in a.rounds.split(",")] if a.rounds else recorded_rounds(c)
        rounds = [load_round(c, r) for r in wanted]
        b = load_baseline(c)
        noise = b["noise_pct"] if b else 2.0
    else:
        dcfg, defaults = {}, {"W": 4, "R": 4, "beta": 0.6}
        rounds, noise = load_fixture(a.fixture)
        if a.rounds:
            keep = {int(x) for x in a.rounds.split(",")}
            rounds = [r for r in rounds if r.round in keep]
        for r in rounds:
            if r.manifest.get("W"):
                defaults = {**defaults, "W": r.manifest["W"], "R": r.manifest["R"]}
    dcfg = dict(dcfg)
    if a.cost is not None:
        dcfg["cost_per_attempt"] = a.cost
    if a.bonus is not None:
        dcfg["parallel_bonus"] = a.bonus
    sweep = [float(x) for x in a.beta_sweep.split(",")] if a.beta_sweep else None
    results, traces = {}, []
    for pp in a.policy:
        res = replay_policy(pp, rounds, noise, defaults, dcfg, sweep, traces)
        results[str(pp)] = res
        at = [e for e in res["episodes"] if e["beta"] == res["default_beta"]]
        errs = [e for e in res["episodes"] if "error" in e]
        print(
            f"{pp}: objective V = {res['objective_V']:.4f} at beta={res['default_beta']}"
            + (f"  ({len(errs)} ILLEGAL episodes)" if errs else "")
        )
        for e in at:
            print(
                f"    r{e['round']:02d}: V={e['V']:.4f} best={e['best']:.3f} attempts={e['attempts']} steps={e['steps']}"
            )
        for e in errs[:3]:
            print(f"    ! r{e['round']:02d} beta={e['beta']}: {e['error']}")
    if a.out:
        a.out.write_text(json.dumps(next(iter(results.values())) if len(results) == 1 else results, indent=2) + "\n")
    if a.traces:
        a.traces.write_text("".join(json.dumps(t) + "\n" for t in traces))


# ================================================================ argparse
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="dream", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_=None):
        p = sp.add_parser(name, help=help_ or argparse.SUPPRESS)
        p.set_defaults(fn=fn)
        return p

    p = add("init", cmd_init, "scaffold a campaign spec + brief")
    p.add_argument("name")
    p.add_argument("--machine")
    p = add("policies", cmd_policies, "list the policy library")
    p.add_argument("--json", action="store_true")
    p = add("check", cmd_check, "set up the machine, build and measure the baseline")
    p.add_argument("spec")
    p.add_argument("--rebaseline", action="store_true")
    p.add_argument("--snapshot", action="store_true", help="start from the working tree instead of HEAD")
    p = add("start", cmd_start, "run the campaign in the background")
    p.add_argument("spec")
    p.add_argument("--snapshot", action="store_true", help="start from the working tree instead of HEAD")
    p = add("resume", lambda a: cmd_start(a, resume=True), "continue a stopped or crashed campaign")
    p.add_argument("spec", metavar="name")
    p = add("status", cmd_status, "campaign state")
    p.add_argument("name")
    p.add_argument("--json", action="store_true")
    p = add("watch", cmd_watch, "follow a campaign; one line per report update")
    p.add_argument("name")
    p.add_argument("--out")
    p.add_argument("--interval", type=int, default=30)
    p = add("report", cmd_report, "copy the current report here")
    p.add_argument("name")
    p.add_argument("--out")
    p = add("stop", cmd_stop, "stop the driver and its sessions")
    p.add_argument("name")
    p = add("fetch", cmd_fetch, "fetch the campaign refs and best branch into this repo")
    p.add_argument("name")
    p = add("finalize", cmd_finalize, "write dream/<name>/best from the best node now")
    p.add_argument("name")
    p = add("export-policy", cmd_export_policy, "add a campaign's policy version to the library")
    p.add_argument("name")
    p.add_argument("--version", required=True)
    p.add_argument("--as", dest="as_name", required=True)
    p.add_argument("--description")
    p.add_argument("--force", action="store_true")
    p = add("delete", cmd_delete, "remove a campaign's refs, worktrees and data")
    p.add_argument("name")
    p.add_argument("--yes", action="store_true")
    add("install-skill", cmd_install_skill, "link the /dream skill into ~/.claude/skills")

    for name, fn in (
        ("_setup", m_setup),
        ("_run", m_run),
        ("_status", m_status),
        ("_report", m_report),
        ("_stop", m_stop),
        ("_finalize", m_finalize),
    ):
        add(name, fn).add_argument("name")
    p = add("_baseline", m_baseline)
    p.add_argument("name")
    p.add_argument("--force", action="store_true")
    p = add("_policy-dump", m_policy_dump)
    p.add_argument("name")
    p.add_argument("--version", required=True)
    for name, fn in (("eval", w_eval), ("commit", w_commit)):
        p = add(name, fn, f"(worker) {name} your node")
        p.add_argument("--campaign", required=True)
        p.add_argument("--node", required=True)
    p = add("verify", w_verify)
    p.add_argument("--campaign", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--record", action="store_true")
    p = add("history", w_history)
    p.add_argument("--campaign", required=True)
    p = add("replay", w_replay, "replay policies on recorded rounds")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--campaign")
    src.add_argument("--fixture", type=Path)
    p.add_argument("--policy", type=Path, action="append", required=True)
    p.add_argument("--rounds")
    p.add_argument("--beta-sweep")
    p.add_argument("--cost", type=float)
    p.add_argument("--bonus", type=float)
    p.add_argument("--out", type=Path)
    p.add_argument("--traces", type=Path)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
