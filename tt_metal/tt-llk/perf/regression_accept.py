# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter

LABEL = "llk-perf-intentional-regression"
HEADING = "## REGRESSION ACCEPTANCE"
HEADER = ["point", "test", "run type", "max delta", "reason"]
MODULE_COL = "test_module"
COMMAND = "/accept-regression"
PLACEHOLDER = "<reason>"
_PASTE_ROWS = 100

_ID = re.compile(r"^[0-9a-f]{12}$")
_MAX = re.compile(r"^\+?(\d+(?:\.\d+)?)\s*%$")
_APPROVAL = re.compile(r"^/accept-regression\s+([0-9a-f]{12})\s*$", re.MULTILINE)


def _norm(value):
    # 4, 4.0 and "4" must give the same point ID.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value)
    return text[:-2] if re.fullmatch(r"-?\d+\.0", text) else text


def point_id(arch, module, marker, run_type, config):
    pairs = sorted((k, _norm(v)) for k, v in config if k != MODULE_COL)
    key = f"{arch}|{module}|{marker}|{run_type}|" + ",".join(
        f"{k}={v}" for k, v in pairs
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rule(n, cells):
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


def _sections(body):
    lines = (body or "").replace("\r\n", "\n").split("\n")
    starts = [i for i, line in enumerate(lines) if line.strip() == HEADING]
    return lines, starts


def parse_section(body):
    lines, starts = _sections(body)
    if not starts:
        return None, []
    if len(starts) > 1:
        return [], [
            f"the description has {len(starts)} `{HEADING}` sections; "
            "put all rows in one table under one heading"
        ]
    table = []
    for line in lines[starts[0] + 1 :]:
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
    lines, starts = _sections(body)
    if not starts:
        return "none"
    text = []
    for start in starts:
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].startswith("#")),
            len(lines),
        )
        text += [line.strip() for line in lines[start:end]]
    return hashlib.sha256("\n".join(text).strip().encode()).hexdigest()[:12]


def table_hash(rules):
    canon = [{k: v for k, v in r.items() if k != "row"} for r in rules]
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:12]


def approved_hash(body):
    m = _APPROVAL.search(body or "")
    return m.group(1) if m else None


def approval(comments, table, approvers, pr_author):
    for c in reversed(comments or []):
        login = (c.get("user") or {}).get("login")
        # Anyone with write access can edit a comment, so an edited one does not count.
        if c.get("updated_at") != c.get("created_at"):
            continue
        if (
            login in approvers
            and login != pr_author
            and approved_hash(c.get("body")) == table
        ):
            return login
    return None


def read(body, comments, approvers, pr_author):
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


def matches(rule, point, ids, arch):
    if (
        rule["test"] != point.get(MODULE_COL)
        or point["run_type"] not in rule["run_types"]
    ):
        return False
    if "id" in rule:
        pid = point["point_id"]
        return pid == rule["id"] and ids.get(pid, 0) == 1
    config = {k: _norm(v) for k, v in point["config"]}
    config.setdefault("arch", arch)
    return all(config.get(k) == v for k, v in rule["filter"].items())


def _pct(delta):
    # Rounded so that +7% covers 1000 -> 1070 cycles (7.000000000000001%).
    return round(delta * 100, 6)


def apply(result, acceptance, arch):
    if acceptance.get("state") == "off":
        return {"state": "off"}
    for p in result["records"] + result["new_points"]:
        p["point_id"] = point_id(
            arch, p.get(MODULE_COL, ""), p["marker"], p["run_type"], p["config"]
        )
    ids = Counter(p["point_id"] for p in result["records"] + result["new_points"])
    rules = acceptance.get("rules") or []
    approved = acceptance.get("approved")
    kept, accepted, missing = [], [], []
    for r in result["regressions"]:
        rule = next(
            (
                x
                for x in rules
                if matches(x, r, ids, arch) and _pct(r["delta"]) <= x["max_pct"]
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
        "table": acceptance.get("table"),
        "accepted": len(accepted),
        "missing": sorted(set(missing)),
        "errors": errors,
    }


def paste_table(regressions, arch):
    rows = {}
    for r in sorted(regressions, key=lambda x: -x["delta"]):
        module = r.get(MODULE_COL) or "?"
        # A GitHub comment holds 65536 characters: many points get one row per test.
        point = r["point_id"] if len(regressions) <= _PASTE_ROWS else f"arch={arch}"
        rows.setdefault((point, module, r["run_type"]), math.ceil(_pct(r["delta"])))
    lines = [HEADING, "", "| " + " | ".join(HEADER) + " |", "|---|---|---|--:|---|"]
    lines += [
        f"| {point} | {module} | {run_type} | +{pct}% | {PLACEHOLDER} |"
        for (point, module, run_type), pct in rows.items()
    ]
    return lines


_STATE = {
    "none": f"no `{HEADING}` section in the PR description.",
    "invalid": f"the `{HEADING}` section does not parse, so nothing is accepted.",
    "waiting": "the table covers every regressed point. It waits for a perf approver.",
    "incomplete": "the table does not cover every regressed point, so the perf approvers are not asked yet.",
    "approved": "approved by @{approver}.",
    "not needed": "the table is present, but no point regressed.",
}


def report_lines(result, acc, arch):
    if acc["state"] == "off":
        if not result["regressions"]:
            return []
        return [
            "",
            f"If this slowdown is intentional, add the label `{LABEL}` "
            "to start an acceptance review (#57871).",
        ]
    lines = [
        "",
        "## Regression acceptance",
        "",
        "- " + _STATE[acc["state"]].format(approver=acc.get("approver")),
    ]
    if acc.get("table") and acc["state"] in ("waiting", "incomplete"):
        lines.append(
            f"- Table hash `{acc['table']}`. A perf approver who accepts this table "
            f"comments `{COMMAND} {acc['table']}`. A change to the table changes the hash."
        )
    lines += [f"- ❌ {e}" for e in acc.get("errors") or []]
    if acc.get("missing"):
        ids = ", ".join(f"`{i}`" for i in acc["missing"][:20])
        more = (
            f" and {len(acc['missing']) - 20} more" if len(acc["missing"]) > 20 else ""
        )
        lines.append(
            f"- Not covered: {ids}{more}. A row also stops covering a point that is slower than its max delta."
        )
    accepted = result.get("accepted") or []
    if accepted:
        lines += [
            "",
            f"### Accepted regressions ({len(accepted)}), not counted",
            "",
            "| point | test | marker | run type | Δ | max delta | reason |",
            "|---|---|---|---|--:|--:|---|",
        ]
        for r in sorted(accepted, key=lambda x: -x["delta"])[:_PASTE_ROWS]:
            # The reason is PR text: keep it from adding markup to the bot's comment.
            reason = html.escape(r["reason"], quote=False)
            lines.append(
                f"| `{r['point_id']}` | {r.get(MODULE_COL) or '?'} | {r['marker']} | "
                f"{r['run_type']} | {r['delta'] * 100:+.1f}% | +{r['max_pct']:g}% | {reason} |"
            )
        if len(accepted) > _PASTE_ROWS:
            lines.append(f"\n_… and {len(accepted) - _PASTE_ROWS} more._")
    if result["regressions"]:
        lines += [
            "",
            "<details><summary>To accept these regressions, paste this into the PR description "
            "and write a reason in each row</summary>",
            "",
            "```markdown",
            *paste_table(result["regressions"], arch),
            "```",
            "",
            "Point IDs differ per arch. If the other arch also has rows, add them to the same "
            "table: the description can have only one heading. A row can also name a filter "
            "instead of an ID, for example `mathop=MathOperation.Square` or "
            "`arch=blackhole, mathop=MathOperation.Square`.",
            "</details>",
        ]
    return lines


def _load_comments(path):
    with open(path) as fh:
        text = fh.read()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Accept intentional perf regressions (#57871)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    rd = sub.add_parser("read", help="the gate: parse the table, find the approval")
    rd.add_argument("--pr", required=True, help="the PR as JSON from GET /pulls/{n}")
    rd.add_argument("--comments", required=True, help="the PR's comments as JSON")
    rd.add_argument("--approvers", required=True)
    rd.add_argument("--out", required=True)
    ok = sub.add_parser("approve", help="/accept-regression: check the approval")
    ok.add_argument("--pr", required=True)
    ok.add_argument("--approvers", required=True)
    ok.add_argument("--commenter", required=True)
    ok.add_argument("--comment", required=True, help="file with the comment's body")
    sub.add_parser("hash", help="hash the section of a description on stdin")
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

    if LABEL not in [x.get("name") for x in pr.get("labels") or []]:
        print(
            f"`{COMMAND}` has no effect: the PR does not have the label `{LABEL}`. "
            "The PR author adds it to start an acceptance review."
        )
        return 2
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
    with open(a.comment) as fh:
        named = approved_hash(fh.read())
    if named is None:
        print(
            f"`{COMMAND}` has no effect: name the table hash from the gate report, "
            f"`{COMMAND} {table}`."
        )
        return 2
    if named != table:
        print(
            f"`{COMMAND} {named}` has no effect: the table changed, and its hash is "
            f"now `{table}`. Read it again, then comment `{COMMAND} {table}`."
        )
        return 2
    print(
        f"@{a.commenter} accepted the {len(rules)} row(s) of the "
        f"REGRESSION ACCEPTANCE table `{table}`."
    )
    print("A change to the table cancels this approval. The perf gate runs again now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
