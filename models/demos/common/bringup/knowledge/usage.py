# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Which knowledge entries the agents use, contradict or miss: the per-step reports and their tally.

Every agent step writes `<bringup_dir>/results/knowledge_use/<task>.<role>.<attempt>.json` (the brief names it):

    {"used": [{"id": "K-12", "how": "what it changed in this step"}],
     "contradicted": [{"id": "GLM-3", "evidence": "commit, file:line or device result"}],
     "missing": ["what the agent had to work out that an entry should have said"]}

After the step the orchestrator adds `meta` (model, run, task, role, attempt, time, HEAD, the files the step read or
changed) and marks each use `traced`: the step touched the entry's `Applies to:` code (read, searched, edited, or named
it in a command); null for an entry without code anchors (method knowledge), which cannot be cross-checked. Agents
name an entry's id where it decides something (agent rule 11): a use or contradiction whose id appears in the step's
visible text (its messages, the files it wrote) is `cited`; ids cited there but missing from the report are kept as
`cited_only` (uses the agent forgot to report).

    python -m models.demos.common.bringup.knowledge.usage [--out DIR] [--unused-runs 5]

tallies every committed report: per entry its uses (traced / untraced / unchecked; cited; unreported = cited only),
last use, contradictions, and the
runs it was relevant to (a run of its model whose steps touched its code; every run of its model for an entry without
anchors; every run of any model for a shared entry without anchors) but not used. An entry unused over its last
`--unused-runs` relevant runs is flagged for the run-end curation (cut or move); a model entry used in two or more
runs is a promotion candidate; missing topics are grouped. Writes usage.json and usage.md. Read-only.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from models.demos.common.bringup.core.spec import CODE_ROOT
from models.demos.common.bringup.knowledge.check import entry_id, entry_title, knowledge_files

REPORT_DIR = "results/knowledge_use"
UNUSED_RUNS = 5
FILE_TOOLS = {"Read": "file_path", "Edit": "file_path", "MultiEdit": "file_path", "Write": "file_path"}
SEARCH_TOOLS = {"Grep": "path", "Glob": "path"}
FIELD_END = r"(?: Seen at:| Check:| Unverified since:| Found:|$)"


def report_path(bringup_dir: Path, tid: str, role: str, attempt: int) -> Path:
    return bringup_dir / REPORT_DIR / f"{tid}.{role}.{attempt}.json"


def model_of(file: str) -> str:
    """The model a knowledge file belongs to ('' for the shared file)."""
    m = re.match(r"models/demos/([^/]+)/bringup/known_issues", file)
    return m.group(1) if m else ""


def entries(root: Path = CODE_ROOT) -> dict[str, dict]:
    """id -> {file, model, title, text, applies} for the shared and every model's known_issues.md."""
    out = {}
    for f in knowledge_files(root):
        if not f.exists():
            continue
        rel = str(f.resolve().relative_to(root))
        for line in f.read_text().splitlines():
            i = entry_id(line)
            if not i:
                continue
            a = re.search(r"Applies to: (.*?)" + FIELD_END, line)
            out[i] = {
                "id": i,
                "file": rel,
                "model": model_of(rel),
                "title": entry_title(line),
                "text": line,
                "applies": a.group(1).strip().rstrip(".") if a else "",
            }
    return out


def anchors(e: dict, res) -> list[str]:
    """The repo paths an entry's code anchors resolve to (stale.Resolver); [] for method knowledge."""
    from models.demos.common.bringup.knowledge.stale import KNOWLEDGE_FILES

    if e["applies"].startswith("method"):
        return []
    prefer = str(Path(e["file"]).parent.parent) if e["model"] else ""
    text = re.sub(r"Detail: \S+\.", "", e["applies"] or e["text"])
    paths, _ = res.anchors(text, prefer)
    return [p for p in paths if not p.endswith(KNOWLEDGE_FILES)]


def transcript_touches(log: Path, repo: Path) -> tuple[list[str], list[str], str]:
    """(repo-relative paths the agent read, edited or searched; its Bash commands; its visible text: messages and the
    text it wrote into files) from a stream-json log. Thinking is not in the log."""
    paths, commands, text = set(), [], []
    for line in log.read_text(errors="replace").splitlines() if log.exists() else []:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "assistant":
            continue
        for c in (ev.get("message") or {}).get("content") or []:
            if c.get("type") == "text":
                text.append(c.get("text") or "")
            if c.get("type") != "tool_use":
                continue
            inp = c.get("input") or {}
            if c.get("name") == "Bash":
                commands.append(inp.get("command", ""))
                continue
            text += [str(inp.get(k) or "") for k in ("new_string", "content")]
            text += [str(e.get("new_string") or "") for e in inp.get("edits") or [] if isinstance(e, dict)]
            key = FILE_TOOLS.get(c.get("name")) or SEARCH_TOOLS.get(c.get("name"))
            p = inp.get(key) if key else None
            if p:
                p = Path(p)
                try:
                    paths.add(str(p.resolve().relative_to(repo)) if p.is_absolute() else str(p))
                except ValueError:
                    pass
    return sorted(paths), commands, "\n".join(text)


def touches(anchor_paths: list[str], touched: list[str], commands: list[str]) -> bool:
    for a in anchor_paths:
        if any(t == a or t.startswith(a + "/") or a.startswith(t + "/") for t in touched):
            return True
        if any(a in c for c in commands):
            return True
    return False


def cited_ids(text: str, known) -> set[str]:
    return {i for i in re.findall(r"\b[A-Z][A-Z0-9]*-\d+\b", text) if i in known}


def finalize(
    path: Path,
    meta: dict,
    touched: list[str],
    commands: list[str],
    root: Path = CODE_ROOT,
    text: str = "",
) -> list[str]:
    """Validate the agent's report, add meta and the traced / cited flags, write it back. Returns what is wrong with it
    (empty: fine); a missing report is one problem, and a stub with the meta is written so the run is still counted.
    text: the step's visible text (transcript_touches), where cited ids are looked for."""
    from models.demos.common.bringup.knowledge.stale import Resolver, git

    known = entries(root)
    problems = []
    if path.exists():
        try:
            rep = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            rep, problems = {}, [f"knowledge report {path.name} is not valid JSON ({e})"]
    else:
        rep, problems = {}, [f"no knowledge report {path.name}"]
    if not isinstance(rep, dict):
        rep, problems = {}, problems + [f"knowledge report {path.name} is not an object"]
    rep.setdefault("used", [])
    rep.setdefault("contradicted", [])
    rep.setdefault("missing", [])
    for key in ("used", "contradicted"):
        good = []
        for item in rep[key] if isinstance(rep[key], list) else []:
            if isinstance(item, dict) and item.get("id") in known:
                good.append(item)
            else:
                problems.append(f"knowledge report {key}: unknown or malformed entry {item!r}")
        rep[key] = good
    rep["missing"] = (
        [m for m in rep["missing"] if isinstance(m, str) and m.strip()] if isinstance(rep["missing"], list) else []
    )
    head = git("rev-parse", "--short=11", "HEAD", root=root).strip()
    res = Resolver(root, head) if rep["used"] else None
    cited = cited_ids(text, known)
    for item in rep["used"]:
        anc = anchors(known[item["id"]], res)
        item["traced"] = touches(anc, touched, commands) if anc else None
        item["cited"] = item["id"] in cited
    for item in rep["contradicted"]:
        item["cited"] = item["id"] in cited
    reported = {i["id"] for i in rep["used"] + rep["contradicted"]}
    rep["cited_only"] = sorted(cited - reported)
    rep["meta"] = {**meta, "head": head, "touched": touched}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1) + "\n")
    return problems


def load_reports(root: Path = CODE_ROOT) -> list[dict]:
    out = []
    for p in sorted(root.glob(f"models/demos/*/bringup/{REPORT_DIR}/*.json")):
        try:
            rep = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(rep, dict) and isinstance(rep.get("meta"), dict):
            rep["_path"] = str(p.relative_to(root))
            out.append(rep)
    return out


def tally(known: dict[str, dict], reports: list[dict], anchor_of: dict[str, list[str]], unused_runs: int) -> dict:
    runs: dict[tuple[str, str], list[dict]] = defaultdict(list)  # (model, run) -> its reports
    for r in reports:
        runs[(r["meta"].get("model", ""), r["meta"].get("run", ""))].append(r)
    order = sorted(runs, key=lambda k: min(r["meta"].get("t", "") for r in runs[k]))
    stats = {
        i: {
            "id": i,
            "file": e["file"],
            "title": e["title"],
            "uses": 0,
            "traced": 0,
            "untraced": 0,
            "unchecked": 0,
            "cited": 0,
            "unreported": 0,
            "last_used": None,
            "used_in_runs": [],
            "contradictions": [],
            "relevant_runs": 0,
            "unused_streak": 0,
        }
        for i, e in known.items()
    }
    unknown = Counter()
    for key in order:
        model, run = key
        reps = runs[key]
        used_here = set()
        touched = sorted({t for r in reps for t in r["meta"].get("touched") or []})  # what the run's steps touched
        for r in reps:
            for u in r.get("used") or []:
                s = stats.get(u.get("id"))
                if s is None:
                    unknown[u.get("id")] += 1
                    continue
                s["uses"] += 1
                s[{True: "traced", False: "untraced", None: "unchecked"}[u.get("traced")]] += 1
                s["cited"] += bool(u.get("cited"))
                s["last_used"] = {"model": model, "run": run, "task": r["meta"].get("task"), "t": r["meta"].get("t")}
                used_here.add(s["id"])
            for i in r.get("cited_only") or []:  # named while working, left out of the report: still a use
                s = stats.get(i)
                if s is None:
                    continue
                s["uses"] += 1
                s["unreported"] += 1
                s["last_used"] = {"model": model, "run": run, "task": r["meta"].get("task"), "t": r["meta"].get("t")}
                used_here.add(i)
            for c in r.get("contradicted") or []:
                s = stats.get(c.get("id"))
                if s is None:
                    unknown[c.get("id")] += 1
                    continue
                s["contradictions"].append(
                    {
                        "model": model,
                        "run": run,
                        "task": r["meta"].get("task"),
                        "t": r["meta"].get("t"),
                        "head": r["meta"].get("head"),
                        "evidence": c.get("evidence", ""),
                    }
                )
        for i, s in stats.items():
            e = known[i]
            if e["model"] and e["model"] != model:
                continue  # a model's entries are read only by that model's agents
            anc = anchor_of.get(i) or []
            relevant = touches(anc, touched, []) if anc else True
            if i in used_here:
                s["used_in_runs"].append(f"{model}/{run}")
                s["relevant_runs"] += 1
                s["unused_streak"] = 0
            elif relevant:
                s["relevant_runs"] += 1
                s["unused_streak"] += 1
    rows = list(stats.values())
    missing = Counter(m.strip().rstrip(".").lower() for r in reports for m in r.get("missing") or [])
    return {
        "runs": [f"{m}/{r}" for m, r in order],
        "unused_runs": unused_runs,
        "entries": rows,
        "flag_unused": [s["id"] for s in rows if s["unused_streak"] >= unused_runs],
        "contradicted": [s["id"] for s in rows if s["contradictions"]],
        "promote": [s["id"] for s in rows if known[s["id"]]["model"] and len(set(s["used_in_runs"])) >= 2],
        "untraced": [s["id"] for s in rows if s["untraced"] and not s["traced"]],
        "missing_topics": missing.most_common(),
        "unknown_ids": dict(unknown),
        "reports": len(reports),
    }


def markdown(t: dict) -> str:
    by = {s["id"]: s for s in t["entries"]}
    md = [
        "# Knowledge use",
        "",
        f"{t['reports']} step reports over {len(t['runs'])} runs. An entry is flagged when unused over its last "
        f"{t['unused_runs']} relevant runs.",
        "",
    ]

    def section(title: str, ids: list[str], line) -> None:
        md.extend([f"## {title} ({len(ids)})", ""])
        md.extend(line(by[i]) for i in ids)
        md.append("")

    section(
        "Contradicted (re-check: staleness.md)",
        t["contradicted"],
        lambda s: f"- `{s['id']}` {s['title']} ({s['file']}): "
        + "; ".join(f"{c['model']} {c['task']}: {c['evidence']}" for c in s["contradictions"]),
    )
    section(
        f"Unused over {t['unused_runs']}+ relevant runs (cut or move)",
        t["flag_unused"],
        lambda s: f"- `{s['id']}` {s['title']} ({s['file']}): {s['relevant_runs']} relevant runs, "
        f"{s['uses']} uses, last {(s['last_used'] or {}).get('run', 'never')}",
    )
    section(
        "Model entries used in 2+ runs (promote to the shared file?)",
        t["promote"],
        lambda s: f"- `{s['id']}` {s['title']}: {s['uses']} uses in {len(set(s['used_in_runs']))} runs",
    )
    section(
        "Claimed but never traced to the entry's code",
        t["untraced"],
        lambda s: f"- `{s['id']}` {s['title']}: {s['untraced']} untraced uses",
    )
    md.extend([f"## Missing topics ({len(t['missing_topics'])})", ""])
    md.extend(f"- {n}x {m}" for m, n in t["missing_topics"])
    md.extend(
        [
            "",
            "## All entries",
            "",
            "| Id | Uses (traced / untraced / unchecked) | Cited / unreported | Relevant runs | Unused streak | Last used |",
            "|---|---|---|---|---|---|",
        ]
    )
    for s in sorted(t["entries"], key=lambda s: (-s["uses"], s["id"])):
        last = s["last_used"] or {}
        md.append(
            f"| `{s['id']}` {s['title'][:60]} | {s['uses']} ({s['traced']} / {s['untraced']} / {s['unchecked']}) | "
            f"{s['cited']} / {s['unreported']} | {s['relevant_runs']} | {s['unused_streak']} | "
            f"{last.get('model', '')} {last.get('task', '')} |"
        )
    if t["unknown_ids"]:
        md.extend(["", f"Reports name ids that no longer exist: {t['unknown_ids']}"])
    return "\n".join(md) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".")
    ap.add_argument("--unused-runs", type=int, default=UNUSED_RUNS)
    a = ap.parse_args(argv)
    from models.demos.common.bringup.knowledge.stale import Resolver, git

    root = CODE_ROOT
    known = entries(root)
    res = Resolver(root, git("rev-parse", "--short=11", "HEAD", root=root).strip())
    t = tally(known, load_reports(root), {i: anchors(e, res) for i, e in known.items()}, a.unused_runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "usage.json").write_text(json.dumps(t, indent=1) + "\n")
    (out / "usage.md").write_text(markdown(t))
    print(
        f"{t['reports']} reports, {len(t['runs'])} runs, {len(known)} entries: "
        f"{sum(1 for s in t['entries'] if s['uses'])} used, {len(t['contradicted'])} contradicted, "
        f"{len(t['flag_unused'])} flagged unused, {len(t['promote'])} to promote, {len(t['missing_topics'])} missing "
        f"topics; wrote {out / 'usage.json'} and {out / 'usage.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
