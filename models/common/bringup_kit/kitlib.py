"""Shared helpers for the bring-up kit tools (stdlib only)."""
import subprocess
import sys

KIT_PREFIX = "[bringup_kit]"


def git(repo, *args, check=True):
    """Run git in ``repo``; return stdout stripped. With check=False return None on failure."""
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        if check:
            raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {r.stderr.strip()}")
        return None
    return r.stdout.rstrip("\n").strip() if not args[0] == "status" else r.stdout.rstrip("\n")


def head(repo, short=11):
    return git(repo, "rev-parse", f"--short={short}", "HEAD")


def resolve(repo, rev):
    """Full hash of ``rev`` or None."""
    return git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False) or None


def same_commit(repo, a, b):
    ra, rb = resolve(repo, a), resolve(repo, b)
    return ra is not None and ra == rb


def dirty_files(repo, ignore_prefixes=()):
    """Tracked files modified/staged (untracked ignored, like the DeepSeek grid). Paths under ignore_prefixes are skipped."""
    out = git(repo, "status", "--porcelain", "-uno")
    files = [l[3:].split(" -> ")[-1] for l in out.splitlines() if l.strip()]
    return [f for f in files if not any(f.startswith(p) for p in ignore_prefixes)]


def is_ancestor(repo, commit, of="HEAD"):
    r = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, of], capture_output=True)
    return r.returncode == 0


def changed_paths(repo, commit, paths, to="HEAD"):
    out = git(repo, "diff", "--name-only", f"{commit}..{to}", "--", *paths, check=False)
    return out.splitlines() if out else []


def die(msg, code=2):
    print(f"{KIT_PREFIX} ERROR: {msg}", file=sys.stderr)
    sys.exit(code)
