"""Git plumbing: worktrees, eval snapshots, node commits, verification, the ledger and the final branch.

Everything lives under refs/dream/<c>/ (invisible to `git branch` / `git tag`):
    base            the user's commit the campaign starts from
    root            base + the campaign directory (spec, brief); round 1 starts here
    n/<node>        one ref per attempt, immutable
    b/r<tt>-b<bb>   one ref per exploration branch, moved by commit_node
    ledger          orphan history: policies, baseline, round manifests, decisions, costs
The one visible result is refs/heads/dream/<c>/best: the best node's code change squashed onto base.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .campaign import Campaign, node_from_ref, parse_node_id

NODE_REQUIRED = ["node.json", "proposal.md", "context.md", "reflection.md", "eval/score.json"]


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def git(*args: str, cwd: Path, check: bool = True, env: dict | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} (in {cwd}) failed: {r.stderr.strip()}")
    return r.stdout.strip()


# ---------------------------------------------------------------- campaign root
def make_root_commit(repo: Path, base: str, name: str, files: dict[str, bytes]) -> str:
    """Commit `files` (repo-relative path -> content) on top of `base` without touching any worktree."""
    with tempfile.TemporaryDirectory() as td:
        env = {"GIT_INDEX_FILE": str(Path(td) / "index")}
        git("read-tree", base, cwd=repo, env=env)
        for rel, data in files.items():
            p = Path(td) / "blob"
            p.write_bytes(data)
            sha = git("hash-object", "-w", str(p), cwd=repo)
            git("update-index", "--add", "--cacheinfo", f"100644,{sha},{rel}", cwd=repo, env=env)
        tree = git("write-tree", cwd=repo, env=env)
    return git("commit-tree", tree, "-p", base, "-m", f"[dream:{name}] campaign root", cwd=repo)


# ---------------------------------------------------------------- ledger
def ledger_init(c: Campaign, policy_name: str) -> None:
    """Create refs/dream/<c>/ledger (orphan) + its worktree, seeded with the chosen library policy as v0."""
    if not c.ref_exists(c.ref_ledger()):
        empty = c.git("hash-object", "-t", "tree", "/dev/null")
        init = c.git("commit-tree", empty, "-m", f"[dream:{c.name}] ledger init")
        c.git("update-ref", c.ref_ledger(), init)
    if not c.ledger.exists():
        c.git("worktree", "add", "-q", "--detach", str(c.ledger), c.ref_ledger())
    act = c.ledger / "policies" / "ACTIVE"
    if not act.exists():
        from .policy_lib import resolve

        src = resolve(policy_name)
        v0 = c.ledger / "policies" / "v0"
        v0.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "policy.py", v0 / "policy.py")
        meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
        (v0 / "notes.md").write_text(
            f"# v0: library policy '{src.name}'\n\n{meta.get('description', '')}\n\n"
            f"Origin: {meta.get('origin', 'hand-written')}. See policy.py docstring.\n"
        )
        (c.ledger / "rounds").mkdir(exist_ok=True)
        act.write_text("v0\n")
        ledger_commit(c, f"ledger: policy v0 from library '{src.name}'")


def ledger_commit(c: Campaign, msg: str) -> bool:
    """Commit everything in the ledger worktree and move refs/dream/<c>/ledger. Returns False if nothing changed."""
    git("add", "-A", cwd=c.ledger)
    if git("status", "--porcelain", cwd=c.ledger) == "":
        return False
    git("commit", "-q", "--no-verify", "-m", f"[dream:{c.name}] {msg}", cwd=c.ledger)
    c.git("update-ref", c.ref_ledger(), git("rev-parse", "HEAD", cwd=c.ledger))
    return True


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- worktrees
def round_manifest(c: Campaign, rnd: int) -> dict:
    return json.loads((c.ledger / "rounds" / f"r{rnd:02d}" / "manifest.json").read_text())


def prepare_worker(c: Campaign, node: str, parent: str, worker: str = "") -> Path:
    """Create or reuse the branch worktree for `node`, check it sits on `parent`, write a skeleton node.json."""
    rnd, b, a = parse_node_id(node)
    wt = c.worktree(rnd, b)
    if c.ref_exists(c.ref_node(node)):
        raise RuntimeError(f"{node} already exists")
    if a == 1:
        if parent != "root":
            raise RuntimeError("a01 must start from root")
        parent_commit = round_manifest(c, rnd)["round_root_commit"]
    else:
        parent_commit = c.git("rev-parse", f"{c.ref_node(parent)}^{{commit}}")
    if wt.exists():
        if git("rev-parse", "HEAD", cwd=wt) != parent_commit:
            raise RuntimeError(f"{wt} HEAD is not at {parent} ({parent_commit[:12]})")
        git("reset", "-q", "--hard", cwd=wt)
        git("clean", "-qfd", cwd=wt)  # leftovers of an interrupted attempt
    else:
        wt.parent.mkdir(parents=True, exist_ok=True)
        c.git("worktree", "add", "-q", "--detach", str(wt), parent_commit)
    nd = wt / c.attempts_rel(node)
    nd.mkdir(parents=True, exist_ok=True)
    (nd / "node.json").write_text(
        json.dumps(
            {
                "node_id": node,
                "campaign": c.name,
                "round": rnd,
                "branch": b,
                "attempt": a,
                "parent": parent,
                "parent_commit": parent_commit,
                "worker": worker,
                "started_at": now(),
                "mechanism": "",
                "tags": [],
            },
            indent=2,
        )
        + "\n"
    )
    return wt


def changed_files(wt: Path) -> list[str]:
    tracked = git("diff", "--name-only", "HEAD", cwd=wt).split("\n")
    untracked = git("ls-files", "--others", "--exclude-standard", cwd=wt).split("\n")
    return sorted({p for p in tracked + untracked if p})


def snapshot(wt: Path, exclude: str, msg: str) -> str:
    """Commit object of the worktree's current state (committed + uncommitted), without touching its index."""
    with tempfile.TemporaryDirectory() as td:
        env = {"GIT_INDEX_FILE": str(Path(td) / "index")}
        git("read-tree", "HEAD", cwd=wt, env=env)
        git("add", "-A", "--", ".", f":!{exclude}", cwd=wt, env=env)
        tree = git("write-tree", cwd=wt, env=env)
    return git("commit-tree", tree, "-p", "HEAD", "-m", msg, cwd=wt)


def commit_node(c: Campaign, wt: Path, node: str) -> dict:
    """Validate the node directory, sync node.json with the eval, commit, move the node + branch refs."""
    rel = c.attempts_rel(node)
    nd = wt / rel
    missing = [f for f in NODE_REQUIRED if not (nd / f).exists()]
    if missing:
        raise RuntimeError(f"missing in {rel}: {missing}")
    meta = json.loads((nd / "node.json").read_text())
    score = json.loads((nd / "eval/score.json").read_text())
    if not meta.get("mechanism"):
        raise RuntimeError("node.json: fill in 'mechanism' (one line)")
    changed = changed_files(wt)
    bad = [p for p in changed if not c.allowed(p, node)]
    if bad:
        raise RuntimeError(f"files outside the editable paths changed; revert them: {bad}")
    code = [p for p in changed if not p.startswith(rel + "/")]
    snap = score.get("commit_under_test")
    if snap:
        diff = subprocess.run(
            ["git", "diff", "--name-only", snap, "--", ".", f":!{c.attempts_rel()}"],
            cwd=wt,
            capture_output=True,
            text=True,
        ).stdout.split()
        if diff:
            raise RuntimeError(f"code changed after evaluation ({diff}); re-run the eval before committing")
    meta.update(
        valid=score.get("valid", False),
        fail_class=score.get("fail_class"),
        score=score.get("score", 0.0),
        files_changed=code,
        build_seconds=score.get("build_seconds"),
        eval_seconds=score.get("eval_seconds"),
        finished_at=now(),
    )
    (nd / "node.json").write_text(json.dumps(meta, indent=2) + "\n")
    outcome = f"score {meta['score']:.4f}" if meta["valid"] else meta["fail_class"]
    msg = f"[dream:{c.name}] {node}: {meta['mechanism']} ({outcome})".replace("\n", " ")
    git("add", "-A", "--", *code, rel, cwd=wt)
    git("commit", "-q", "--no-verify", "-m", msg, cwd=wt)
    sha = git("rev-parse", "HEAD", cwd=wt)
    rnd, b, _ = parse_node_id(node)
    c.git("update-ref", c.ref_node(node), sha)
    c.git("update-ref", c.ref_branch(rnd, b), sha)
    nxt = ""
    refl = (nd / "reflection.md").read_text().splitlines()
    for i, line in enumerate(refl):
        if line.lower().startswith("## what a child"):
            nxt = next((l.strip("- ").strip() for l in refl[i + 1 :] if l.strip()), "")
            break
    return {
        "node_id": node,
        "commit": sha,
        "valid": meta["valid"],
        "fail_class": meta["fail_class"],
        "score": meta["score"],
        "mechanism": meta["mechanism"],
        "next": nxt,
    }


def verify_node(c: Campaign, node: str, record: bool = False) -> dict:
    """Check a returned node; with record=True a failing node gets an override/lost line in decisions.jsonl."""
    from .tree import read_node_file

    rnd, _, att = parse_node_id(node)
    issues, fail_class = [], "infra"
    sha = c.git("rev-parse", "-q", "--verify", f"{c.ref_node(node)}^{{commit}}", check=False) or None
    if sha is None:
        issues.append("node ref missing (worker did not commit)")
        fail_class = "lost"
    else:
        manifest = round_manifest(c, rnd)
        meta = json.loads(read_node_file(c, node, "node.json") or "{}")
        score = json.loads(read_node_file(c, node, "eval/score.json") or "{}")
        if att == 1:
            exp_parent = manifest["round_root_commit"]
        else:
            p = meta.get("parent", "")
            exp_parent = c.git("rev-parse", f"{c.ref_node(p)}^{{commit}}", check=False) if p else None
        if c.git("rev-parse", f"{sha}^") != exp_parent:
            issues.append(f"git parent is not the expected parent {meta.get('parent')}")
        if not meta or not score:
            issues.append("node.json or eval/score.json missing")
        for k in ("valid", "fail_class", "score"):
            if meta.get(k) != score.get(k):
                issues.append(f"node.json {k}={meta.get(k)} != score.json {score.get(k)}")
        files = c.git("diff", "--name-only", f"{sha}^", sha).split()
        bad = [p for p in files if not c.allowed(p, node)]
        if bad:
            issues.append(f"touches files outside the editable paths: {bad}")
            fail_class = "forbidden_edit"
        hits = forbidden_hits(c, sha)
        if hits:
            issues.append(f"code matches forbidden patterns (campaign rules): {hits}")
            fail_class = "forbidden_edit"
    ok = not issues
    if record and not ok:
        append_jsonl(
            c.ledger / "rounds" / f"r{rnd:02d}" / "decisions.jsonl",
            {
                "type": "lost" if fail_class == "lost" else "override",
                "node": node,
                "fail_class": fail_class,
                "why": "; ".join(issues),
                "time": now(),
            },
        )
    return {"node": node, "ok": ok, "issues": issues}


def forbidden_hits(c: Campaign, rev: str) -> list[str]:
    hits = []
    for pat in c.cfg.get("forbidden_patterns", []):
        for p in c.cfg.get("editable", []):
            out = c.git("grep", "-lE", pat, rev, "--", p.rstrip("*") or ".", check=False)
            hits += [f"{pat!r} in {line.split(':', 1)[-1]}" for line in out.splitlines() if line]
    return hits


# ---------------------------------------------------------------- final result
def lineage(c: Campaign, node: str) -> list[str]:
    """Commit subjects from the campaign root to `node` (oldest first)."""
    out = c.git("log", "--reverse", "--format=%s", f"{c.ref_root()}..{c.ref_node(node)}")
    return [line for line in out.splitlines() if line]


def finalize_best(c: Campaign, node: str, score: float) -> str:
    """refs/heads/dream/<c>/best = base + the best node's code change (no campaign files), one commit."""
    base = c.git("rev-parse", f"{c.ref_base()}^{{commit}}")
    diff = subprocess.run(
        ["git", "diff", "--binary", c.ref_root(), c.ref_node(node), "--", ".", f":!{c.campaign_rel}"],
        cwd=c.repo,
        capture_output=True,
        check=True,
    ).stdout
    with tempfile.TemporaryDirectory() as td:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(td) / "index")}
        subprocess.run(["git", "read-tree", base], cwd=c.repo, env=env, check=True)
        if diff:
            subprocess.run(
                ["git", "apply", "--cached", "--whitespace=nowarn"],
                cwd=c.repo,
                env=env,
                input=diff,
                check=True,
                capture_output=True,
            )
        tree = subprocess.run(
            ["git", "write-tree"], cwd=c.repo, env=env, capture_output=True, text=True, check=True
        ).stdout.strip()
    steps = "\n".join(f"- {s.split('] ', 1)[-1]}" for s in lineage(c, node))
    msg = (
        f"{c.name}: best Dream-RSI result {node} (score {score:.4f})\n\n"
        f"Squashed from refs/dream/{c.name}/n/{node}. Lineage:\n{steps}\n"
    )
    sha = c.git("commit-tree", tree, "-p", base, "-m", msg)
    c.git("update-ref", c.ref_best(), sha)
    return sha


def node_round_root(c: Campaign, rnd: int) -> str | None:
    try:
        return node_from_ref(round_manifest(c, rnd).get("round_root", ""))
    except FileNotFoundError:
        return None
