# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""List the knowledge entries whose code changed since they were last verified (the mechanical half of a staleness
re-check; skill/rebase-bringup/staleness.md is the procedure that acts on the list).

    python -m models.demos.common.bringup.knowledge.stale [--models all|<m1,m2>] [--out DIR] [--head REV]

For every entry of the shared knowledge/known_issues.md and of each model's bringup/known_issues.md:
- since: the entry's `Seen at:` commit, else the commit that last touched its line (git blame);
- anchors: what its `Applies to:` (else its text) names, resolved to repo paths at HEAD: paths as written ({a,b}
  expanded) or partial ones by suffix (`testing/profiler.py`), and op names (`unified_routed_expert_moe`,
  `ttnn.bringup.mhc_pre`) to the folder that registers them (`bind_function<"name">`) or the fork of that name;
- status: `needs-probe` (it carries `Unverified since:`: a device check is still open), `anchor-missing` (a path that
  existed at `since` is gone), `changed` (an anchored path has commits in since..HEAD, listed), `method`
  (`Applies to: method ...`: testing / process knowledge, never stale with the code), `unanchored` (nothing resolved: a
  person or agent judges it from the text), or `unchanged`.
Partial paths in a model's entries resolve inside that model's folder first; arch copies (`quasar`) of an op are not
anchors. After a rebase, since..HEAD can list commits older than the entry (history was reordered): those whose author
date is before `since`'s are dropped (counted in `older_commits`).
Repo-map rows are checked for paths that no longer exist. Writes stale.json and stale.md to --out (default: the
current directory) and prints a summary. Read-only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from models.demos.common.bringup.core.spec import CODE_ROOT

HERE = Path(__file__).resolve().parent
PATH_RE = re.compile(r"(?:models|ttnn|tt_metal|tests|scripts)/[A-Za-z0-9_./{},*-]+")
ARCH_COPIES = ("/quasar/",)  # per-arch copies of an op: not where a lesson about the op lives


def git(*args: str, root: Path = CODE_ROOT) -> str:
    return subprocess.run(["git", "-c", "gc.auto=0", *args], cwd=root, capture_output=True, text=True).stdout


def expand(p: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", p)
    if not m:
        return [p]
    return [q for alt in m.group(1).split(",") for q in expand(p[: m.start()] + alt + p[m.end() :])]


class Resolver:
    """Turns what an entry's `Applies to:` names into repo paths at `head`."""

    def __init__(self, root: Path, head: str):
        self.root, self.head = root, head
        self.files = git("ls-tree", "-r", "--name-only", head, root=root).split()
        self.fileset = set(self.files)
        self.dirs = {str(Path(f).parent) for f in self.files}
        for d in list(self.dirs):
            while d not in ("", "."):
                d = str(Path(d).parent)
                self.dirs.add(d)
        # op name -> its folder: the Python names ops and forks register (bind_function<"name">), and fork folder names
        self.ops: dict[str, set[str]] = {}
        grep = git(
            "grep", "-o", "-E", 'bind_function<"[a-z_0-9]+"', head, "--", "ttnn/cpp", "ttnn/ttnn/bringup", root=root
        )
        for line in grep.splitlines():
            m = re.match(r"[^:]+:([^:]+):bind_function<\"([a-z_0-9]+)\"", line)
            if m and not any(a in m.group(1) for a in ARCH_COPIES):
                self.ops.setdefault(m.group(2), set()).add(self.op_dir(m.group(1)))
        for f in self.files:
            parts = f.split("/")
            if f.startswith("ttnn/ttnn/bringup/") and len(parts) > 4:
                self.ops.setdefault(parts[3], set()).add("/".join(parts[:4]))

    @staticmethod
    def op_dir(f: str) -> str:
        parts = f.split("/")
        if f.startswith("ttnn/ttnn/bringup/"):
            return "/".join(parts[:4])
        return "/".join(parts[: parts.index("device")]) if "device" in parts else str(Path(f).parent)

    def path(self, p: str, prefer: str = "") -> list[str]:
        """A repo path as written, or a partial one (`testing/profiler.py`, `sdpa_program_factory.cpp`) by suffix;
        `prefer` (a model's folder): a partial path that exists there means that one (`tt/mhc.py` of this model)."""
        if p in self.fileset:
            return [p]
        if p in self.dirs:
            return [p] if p.count("/") >= 2 else []  # `models/demos`, `ttnn/cpp`: too broad to anchor anything
        hits = [f for f in self.files if f.endswith("/" + p)] or [d for d in self.dirs if d.endswith("/" + p)]
        hits = [h for h in hits if not any(a in h for a in ARCH_COPIES)]
        own = [h for h in hits if prefer and h.startswith(prefer + "/")]
        if own:
            return own
        return hits if 0 < len(hits) <= 3 else []

    def anchors(self, text: str, prefer: str = "") -> tuple[list[str], list[str]]:
        """(paths, written paths that resolve to nothing at head: maybe gone)."""
        out, gone = [], []
        for raw in re.findall(r"[A-Za-z0-9_./{},*-]*[/.][A-Za-z0-9_./{},*-]*", text):
            for p in expand(raw.strip(".,;:()`*")):
                p = p.split(":")[0].rstrip("/.")
                if "*" in p or "/" not in p and not re.search(r"\.(py|cpp|hpp|h|md|yaml)$", p):
                    continue
                hit = self.path(p, prefer)
                out += hit
                if not hit and PATH_RE.fullmatch(p) and p not in self.fileset and p not in self.dirs:
                    gone.append(p)
        dotted = [op.split(".")[-1] for op in re.findall(r"\bttnn(?:\.[A-Za-z_][A-Za-z0-9_]*)+", text)]
        for tok in dotted + re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text):
            if tok in self.ops and len(self.ops[tok]) <= 3:
                out += sorted(self.ops[tok])
        return list(dict.fromkeys(out)), gone

    def exists(self, path: str, rev: str) -> bool:
        return bool(git("ls-tree", "--name-only", rev, "--", path.rstrip("/"), root=self.root).strip())


def moved_ages(root: Path, rel: str) -> dict[str, str]:
    """Entries a model's file got from the shared file keep their real age: title -> the commit that last touched the
    title's line in the shared known_issues.md just before this file was created."""
    created = git("log", "--diff-filter=A", "--format=%h", "--", rel, root=root).split()
    if not created:
        return {}
    shared = "models/demos/common/bringup/knowledge/known_issues.md"
    blame = git("blame", "--line-porcelain", f"{created[-1]}^", "--", shared, root=root)
    ages, commit = {}, ""
    for line in blame.splitlines():
        m = re.match(r"^([0-9a-f]{40}) \d+ \d+", line)
        if m:
            commit = m.group(1)[:11]
        elif line.startswith("\t- **"):
            t = re.match(r"\t- \*\*(.+?)\*\*", line)
            if t:
                ages[t.group(1)] = commit
    return ages


def entries_of(path: Path, root: Path) -> list[dict]:
    """Bullets of a knowledge file with their line, section and the commit that last touched the entry."""
    rel = str(path.relative_to(root))
    blame = git("blame", "--line-porcelain", "--", rel, root=root)
    line_commit = {}
    for line in blame.splitlines():
        m = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", line)
        if m:
            line_commit[int(m.group(2))] = m.group(1)[:11]
    ages = moved_ages(root, rel) if not rel.startswith("models/demos/common/") else {}
    out, section = [], ""
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("- **"):
            t = re.match(r"- \*\*(.+?)\*\*", line)
            title = t.group(1) if t else line[2:80]
            out.append(
                {
                    "file": rel,
                    "line": i,
                    "section": section,
                    "title": title,
                    "text": line,
                    "blame": ages.get(title) or line_commit.get(i, ""),
                }
            )
    return out


FIELD_END = r"(?: Seen at:| Check:| Unverified since:| Found:|$)"


def check_entry(e: dict, res: Resolver, head: str) -> dict:
    m = re.search(r"Seen at: ([0-9a-f]{7,40})", e["text"])
    since = m.group(1) if m else e["blame"]
    a = re.search(r"Applies to: (.*?)" + FIELD_END, e["text"])
    applies = a.group(1).strip().rstrip(".") if a else ""
    prefer = str(Path(e["file"]).parent.parent) if not e["file"].startswith("models/demos/common/") else ""
    method = applies.startswith("method")  # `Applies to: method (component tests)`: never stale with the code
    anchors, gone = res.anchors(applies or e["text"], prefer) if not method else ([], [])
    missing = [p for p in gone if since and res.exists(p, since)]
    e = {**e, "since": since, "seen_at_field": bool(m), "anchors": anchors, "commits": [], "missing": missing}
    e["older_commits"] = 0
    if anchors and since:
        floor = git("show", "-s", "--format=%at", since, root=res.root).strip()
        log = git("log", "--format=%h %at %ad %s", "--date=short", f"{since}..{head}", "--", *anchors, root=res.root)
        for c in (c for c in log.splitlines() if c.strip()):
            sha, at, rest = c.split(" ", 2)
            if floor and int(at) < int(floor):
                e["older_commits"] += 1  # reordered by a rebase: written before the entry was last checked
            else:
                e["commits"].append(f"{sha} {rest}")
    if "Unverified since:" in e["text"]:
        e["status"] = "needs-probe"
    elif missing:
        e["status"] = "anchor-missing"
    elif method:
        e["status"] = "method"
    elif not anchors:
        e["status"] = "unanchored"
    else:
        e["status"] = "changed" if e["commits"] else "unchanged"
    return e


def repo_map_rows(path: Path, res: Resolver, root: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("|") and not line.startswith(("| Need", "|---")):
            gone = [
                p
                for raw in PATH_RE.findall(line)
                for p in expand(raw.strip(".,;:()`*"))
                if "*" not in p and not res.path(p.split(":")[0].rstrip("/."))
            ]
            if gone:
                rows.append({"file": str(path.relative_to(root)), "line": i, "row": line[:120], "missing": gone})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all", help="'all', 'none' or a comma list of model dirs under models/demos")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    root = CODE_ROOT
    head = git("rev-parse", "--short=11", a.head, root=root).strip()
    res = Resolver(root, head)
    files = [HERE / "known_issues.md"]
    maps = [HERE / "repo_map.md"]
    if a.models != "none":
        names = (
            sorted(p.parent.parent.name for p in root.glob("models/demos/*/bringup/known_issues.md"))
            if a.models == "all"
            else a.models.split(",")
        )
        files += [root / f"models/demos/{m}/bringup/known_issues.md" for m in names]
        maps += [root / f"models/demos/{m}/bringup/repo_map.md" for m in names]
    results = [check_entry(e, res, head) for f in files if f.exists() for e in entries_of(f, root)]
    rm = [r for f in maps if f.exists() for r in repo_map_rows(f, res, root)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stale.json").write_text(json.dumps({"head": head, "entries": results, "repo_map": rm}, indent=1) + "\n")
    order = {"needs-probe": 0, "anchor-missing": 1, "changed": 2, "unanchored": 3, "method": 4, "unchanged": 5}
    md = [
        f"# Knowledge staleness at {head}",
        "",
        "| Status | Entry | Since | Commits | Anchors |",
        "|---|---|---|---|---|",
    ]
    for e in sorted(results, key=lambda e: (order[e["status"]], e["file"], e["line"])):
        md.append(
            f"| {e['status']} | `{e['file']}:{e['line']}` {e['title'][:70]} | {e['since']} | {len(e['commits'])} | "
            + ", ".join(f"`{p}`" for p in e["anchors"][:4])
            + (" ..." if len(e["anchors"]) > 4 else "")
            + " |"
        )
    if rm:
        md += ["", "## Repo-map rows with missing paths", ""] + [
            f"- `{r['file']}:{r['line']}` {r['missing']}" for r in rm
        ]
    (out / "stale.md").write_text("\n".join(md) + "\n")
    c = Counter(e["status"] for e in results)
    print(f"head {head}: {len(results)} entries: " + ", ".join(f"{k} {c[k]}" for k in order if c[k]))
    print(f"repo-map rows with missing paths: {len(rm)}; wrote {out / 'stale.json'} and {out / 'stale.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
