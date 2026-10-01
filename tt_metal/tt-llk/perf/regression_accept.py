# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Accept intentional regressions: the REGRESSION ACCEPTANCE table and /accept-regression (#57871).

Run it by path: ``tt_metal/tt-llk`` is not an importable package.
"""

import argparse
import hashlib
import json
import math
import re
import sys

HEADING = "## REGRESSION ACCEPTANCE"
HEADER = ["point", "test", "run type", "max delta", "reason"]
MODULE_COL = "test_module"
BOT = "github-actions[bot]"
COMMAND = "/accept-regression"
PLACEHOLDER = "<reason>"

_ID = re.compile(r"^[0-9a-f]{12}$")
_MAX = re.compile(r"^\+?(\d+(?:\.\d+)?)\s*%$")
_APPROVAL = re.compile(r"<!-- llk-perf-accept approver=(\S+) table=([0-9a-f]{12}) -->")


def _norm(value):
    """One spelling per value, so 4, 4.0 and "4" give the same point ID."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value)
    return text[:-2] if re.fullmatch(r"-?\d+\.0", text) else text


def point_id(module, marker, run_type, config):
    """12 hex characters of a hash of module | marker | run type | sorted config."""
    pairs = sorted((k, _norm(v)) for k, v in config if k != MODULE_COL)
    key = f"{module}|{marker}|{run_type}|" + ",".join(f"{k}={v}" for k, v in pairs)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rule(n, cells):
    """One parsed row, or raise ValueError with the row number."""
    if len(cells) != len(HEADER):
        raise ValueError(f"row {n}: expected {len(HEADER)} cells, got {len(cells)}")
    point, test, run_types, max_delta, reason = cells
    if not point:
        raise ValueError(f"row {n}: point is empty")
    if not test:
        raise ValueError(f"row {n}: test is empty")
    types = [t.strip() for t in run_types.split(",") if t.strip()]
    if not types:
        raise ValueError(f"row {n}: run type is empty")
    m = _MAX.match(max_delta)
    if not m:
        raise ValueError(f"row {n}: max delta must look like +10%")
    if not reason or reason == PLACEHOLDER:
        raise ValueError(f"row {n}: reason is empty")
    rule = {
        "row": n,
        "test": test,
        "run_types": types,
        "max_pct": float(m.group(1)),
        "reason": reason,
    }
    if _ID.match(point):
        rule["id"] = point
    else:
        pairs = [p.strip() for p in point.split(",") if p.strip()]
        if not pairs or any("=" not in p for p in pairs):
            raise ValueError(
                f"row {n}: point must be a 12-character point ID or key=value pairs"
            )
        rule["filter"] = {
            k.strip(): v.strip() for k, v in (p.split("=", 1) for p in pairs)
        }
    return rule


def parse_section(body):
    """``(rules, errors)`` from a PR description; ``(None, [])`` when there is no section."""
    lines = (body or "").replace("\r\n", "\n").split("\n")
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == HEADING)
    except StopIteration:
        return None, []
    table = []
    for line in lines[start + 1 :]:
        if line.strip().startswith("|"):
            table.append(line)
        elif table or line.startswith("#"):
            break
    if len(table) < 2:
        return [], ["the section has no table"]
    if [c.lower() for c in _cells(table[0])] != HEADER:
        return [], ["the header row must be exactly | " + " | ".join(HEADER) + " |"]
    if not all(re.fullmatch(r":?-+:?", c) for c in _cells(table[1])):
        return [], ["the second row of the table must be the |---| separator"]
    rules, errors = [], []
    for n, line in enumerate(table[2:], 1):
        try:
            rules.append(_rule(n, _cells(line)))
        except ValueError as e:
            errors.append(str(e))
    if not rules and not errors:
        errors.append("the table has no rows")
    return rules, errors


def section_hash(body):
    """A hash of the section's text, valid or not; ``none`` without a section."""
    lines = (body or "").replace("\r\n", "\n").split("\n")
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == HEADING)
    except StopIteration:
        return "none"
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("#")),
        len(lines),
    )
    text = "\n".join(line.strip() for line in lines[start:end]).strip()
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def table_hash(rules):
    """What an approval is bound to: any change to a row gives another hash."""
    canon = [{k: v for k, v in r.items() if k != "row"} for r in rules]
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:12]


def approval(comments, table, approvers, pr_author):
    """The approver whose recorded approval matches this table, else None."""
    for c in reversed(comments or []):
        if (c.get("user") or {}).get("login") != BOT:
            continue
        # Only at the start: a bot comment that quotes PR text cannot carry one.
        m = _APPROVAL.match(c.get("body") or "")
        if (
            m
            and m.group(2) == table
            and m.group(1) in approvers
            and m.group(1) != pr_author
        ):
            return m.group(1)
    return None


def read(body, comments, approvers, pr_author):
    """The gate's view of a PR: the rules, the errors and the approval."""
    rules, errors = parse_section(body)
    if rules is None:
        return {"state": "none", "rules": [], "errors": [], "approved": False}
    table = table_hash(rules) if rules else None
    approver = (
        approval(comments, table, approvers, pr_author)
        if rules and not errors
        else None
    )
    return {
        "state": "invalid" if errors else "present",
        "rules": rules,
        "errors": errors,
        "table": table,
        "approved": approver is not None,
        "approver": approver,
    }


def matches(rule, point, ids):
    """``rule`` covers ``point``; an ID shared by two points covers neither."""
    if (
        rule["test"] != point.get(MODULE_COL)
        or point["run_type"] not in rule["run_types"]
    ):
        return False
    if "id" in rule:
        pid = point["point_id"]
        return pid == rule["id"] and ids.get(pid, 0) == 1
    config = {k: _norm(v) for k, v in point["config"]}
    return all(config.get(k) == v for k, v in rule["filter"].items())


def apply(result, acceptance, ids):
    """Move approved, covered regressions to ``result["accepted"]``; return the status."""
    rules = acceptance.get("rules") or []
    approved = acceptance.get("approved")
    kept, accepted, missing = [], [], []
    for r in result["regressions"]:
        rule = next(
            (
                x
                for x in rules
                if matches(x, r, ids) and r["delta"] * 100 <= x["max_pct"]
            ),
            None,
        )
        if rule and approved:
            accepted.append({**r, "reason": rule["reason"], "max_pct": rule["max_pct"]})
            continue
        kept.append(r)
        if not rule:
            missing.append(r["point_id"])
    result["regressions"], result["accepted"] = kept, accepted
    errors = list(acceptance.get("errors") or [])
    errors += [
        f"row {x['row']}: point ID {x['id']} matches {ids[x['id']]} points; "
        "use key=value pairs"
        for x in rules
        if "id" in x and ids.get(x["id"], 0) > 1
    ]
    state = acceptance.get("state", "none")
    if state == "present":
        if approved:
            state = "approved"
        elif not kept:
            state = "not needed"
        else:
            state = "incomplete" if missing else "waiting"
    return {
        "state": state,
        "approver": acceptance.get("approver"),
        "accepted": len(accepted),
        "missing": sorted(set(missing)),
        "errors": errors,
    }


def paste_table(regressions):
    """The section an author pastes into the PR description, one row per point."""
    lines = [HEADING, "", "| " + " | ".join(HEADER) + " |", "|---|---|---|--:|---|"]
    for r in sorted(regressions, key=lambda x: -x["delta"]):
        pct = math.ceil(round(r["delta"] * 100, 6))
        module = r.get(MODULE_COL) or "?"
        lines.append(
            f"| {r['point_id']} | {module} | {r['run_type']} | +{pct}% | {PLACEHOLDER} |"
        )
    return lines


def _load_comments(path):
    """A JSON list, or one JSON object per line as ``gh api --paginate --jq`` writes it."""
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except ValueError:
        out = []
        for line in filter(str.strip, text.splitlines()):
            try:
                out.append(json.loads(line))
            except ValueError:
                print(f"::warning::Skipped a comment that is not JSON: {line[:80]}")
        return out


def _load(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rd = sub.add_parser(
        "read", help="the gate: parse the table and find a valid approval"
    )
    rd.add_argument("--pr", required=True, help="the PR as JSON from GET /pulls/{n}")
    rd.add_argument("--comments", required=True, help="the PR's comments as JSON")
    rd.add_argument("--approvers", required=True)
    rd.add_argument("--out", required=True)
    ok = sub.add_parser(
        "approve", help="/accept-regression: check the commenter and the table"
    )
    ok.add_argument("--pr", required=True)
    ok.add_argument("--approvers", required=True)
    ok.add_argument("--commenter", required=True)
    sub.add_parser("hash", help="print a hash of the section of a description on stdin")
    a = ap.parse_args(argv)

    if a.cmd == "hash":
        print(section_hash(sys.stdin.read()))
        return 0

    pr = _load(a.pr, {})
    approvers = _load(a.approvers, {})
    author = (pr.get("user") or {}).get("login")
    if a.cmd == "read":
        with open(a.out, "w") as fh:
            json.dump(
                read(pr.get("body"), _load_comments(a.comments), approvers, author),
                fh,
                indent=1,
            )
        return 0

    if a.commenter not in approvers:
        names = ", ".join("@" + x for x in approvers)
        print(
            f"`{COMMAND}` from @{a.commenter} has no effect: "
            f"only perf approvers can accept ({names})."
        )
        return 2
    if a.commenter == author:
        print(
            f"`{COMMAND}` has no effect: "
            "the PR author cannot approve their own regressions."
        )
        return 2
    rules, errors = parse_section(pr.get("body"))
    if not rules or errors:
        why = "; ".join(errors) if errors else f"there is no `{HEADING}` section"
        print(f"`{COMMAND}` has no effect: {why}.")
        return 2
    table = table_hash(rules)
    print(f"<!-- llk-perf-accept approver={a.commenter} table={table} -->")
    print(
        f"@{a.commenter} accepted the {len(rules)} row(s) of the "
        f"REGRESSION ACCEPTANCE table (table `{table}`)."
    )
    print("A change to the table cancels this approval. The perf gate runs again now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
