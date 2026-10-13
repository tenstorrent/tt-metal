#!/usr/bin/env python3
"""Extract a results row from a demo log; compare two logs.

    results.py extract LOG [--json|--md]
    results.py compare LOG_A LOG_B [--json]

Understands the DeepSeek text_demo log format (a new model's demo should print the same lines, or edit PATTERNS below):
  GRID_ENV header (host, date, head=<commit>, variables)         -> host, commit, env
  '=== session scenario X ==='                                    -> scenario blocks
  'TTFT (whole batch of B users, ISL max N): T ms -> prefill P tok/s'
  'Decode: M ms/token @ U tok/s/user (S tok/s throughput)'
  'SPEC_RESULT... {json}', 'FIRSTTOK_IDS [..]', 'PREFILL_ONLY first tokens [..]'
  'built layer N (Ts)', PASSED / FAILED / passed|failed summary, 'Out of Memory', 'TT_FATAL', 'TT_THROW', '[hangwatch]'
compare: pairs scenarios by name and prints ratios (B/A). The comparison is flagged INVALID when the logs ran on different hosts
(per-host performance differs: one host is faulty, NFS/Engram load differs) or at different commits; differing env variables are listed.
"""
import argparse
import json
import re
import sys

PATTERNS = {
    "grid_head": re.compile(r"^GRID_ENV (\S+) (\S+) head=(\w+)(?: branch=(\S+))?"),
    "grid_kv": re.compile(r"^GRID_ENV ([A-Z0-9_]+)=(.*)$"),
    "scenario": re.compile(r"=== session scenario (\S+) ==="),
    "ttft": re.compile(r"TTFT \(whole batch of (\d+) users, ISL max (\d+)\): (\d+) ms -> prefill (\d+) tok/s"),
    "decode": re.compile(r"Decode: ([\d.]+) ms/token @ ([\d.]+) tok/s/user \(([\d.]+) tok/s throughput\)"),
    "spec": re.compile(r"(SPEC_RESULT\w*) (\{.*\})\s*$"),
    "firsttok": re.compile(r"(?:FIRSTTOK_IDS|PREFILL_ONLY first tokens) (\[[^\]]*\])"),
    "built": re.compile(r"built layer (\d+) \((\d+)s\)"),
    "ts": re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"),
    "free": re.compile(r"MEMLOG prefill end\s+allocated\s+([\d.]+) MiB/bank\s+free\s+([\d.]+)"),
}


def host_id(name):
    return name.split("-")[-1] if name else None


def extract(text):
    r = {
        "host": None,
        "commit": None,
        "branch": None,
        "env": {},
        "scenarios": [],
        "spec": [],
        "first_tokens": None,
        "built_layers": None,
        "build_s": None,
        "status": "INCOMPLETE",
        "failures": [],
        "hangwatch": False,
        "header_date": None,
    }
    cur = None

    def scen(name):
        nonlocal cur
        cur = {"scenario": name}
        r["scenarios"].append(cur)

    for line in text.splitlines():
        m = PATTERNS["grid_head"].match(line)
        if m:
            r["host"], r["header_date"], r["commit"], r["branch"] = (
                host_id(m.group(1)),
                m.group(2),
                m.group(3),
                m.group(4),
            )
            continue
        m = PATTERNS["grid_kv"].match(line)
        if m:
            r["env"][m.group(1)] = m.group(2)
            continue
        if m := PATTERNS["scenario"].search(line):
            scen(m.group(1))
        elif m := PATTERNS["ttft"].search(line):
            if cur is None or "ttft_ms" in cur:
                if cur is None:
                    scen("default")
                else:
                    continue  # a later TTFT line in the same scenario (spec phase): keep the first (plain)
            cur.update(B=int(m.group(1)), isl=int(m.group(2)), ttft_ms=int(m.group(3)), prefill_tok_s=int(m.group(4)))
        elif m := PATTERNS["decode"].search(line):
            if cur is None:
                scen("default")
            if "decode_ms_tok" not in cur:
                cur.update(decode_ms_tok=float(m.group(1)), tok_s_user=float(m.group(2)), tok_s_total=float(m.group(3)))
        elif (m := PATTERNS["free"].search(line)) and cur is not None:
            cur["free_gib_chip"] = round(
                float(m.group(2)) * 8 / 1024, 2
            )  # MiB/bank x 8 banks (Blackhole: 8 DRAM banks per chip)
        elif m := PATTERNS["spec"].search(line):
            try:
                d = json.loads(m.group(2))
                d["_kind"] = m.group(1)
                r["spec"].append(d)
            except ValueError:
                pass
        elif m := PATTERNS["firsttok"].search(line):
            r["first_tokens"] = json.loads(m.group(1)) if r["first_tokens"] is None else r["first_tokens"]
        elif m := PATTERNS["built"].search(line):
            r["built_layers"], r["build_s"] = int(m.group(1)) + 1, int(m.group(2))
        if "Out of Memory" in line or "out of memory" in line.lower() and "DRAM" in line:
            r["failures"].append("OOM")
        if "TT_FATAL" in line or "TT_THROW" in line:
            r["failures"].append(line.split("TT_", 1)[1][:160].strip() if "TT_" in line else line[:160])
        if "[hangwatch] killing" in line or "[hangwatch] pid" in line:
            r["hangwatch"] = True
        if re.match(r"^(PASSED|FAILED)\b", line) or re.search(r"=+ .*\b(passed|failed)\b.* =+", line):
            if "failed" in line.lower() or line.startswith("FAILED"):
                r["status"] = "FAILED"
            elif r["status"] != "FAILED":
                r["status"] = "PASSED"
    if r["hangwatch"]:
        r["status"] = "HANG"
    r["failures"] = sorted(set(r["failures"]))[:6]
    r["cause"] = "OOM" if "OOM" in r["failures"] else (r["failures"][0] if r["failures"] else None)
    return r


def fmt_md(r):
    head = f"host .{r['host']}  commit {r['commit']}  status {r['status']}" + (
        f"  cause {r['cause']}" if r["cause"] else ""
    )
    out = [
        head,
        "",
        "| scenario | B | ISL | TTFT ms | prefill tok/s | decode ms/tok | tok/s/user | tok/s total | free GiB/chip |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in r["scenarios"]:
        out.append(
            "| "
            + " | ".join(
                str(s.get(k, ""))
                for k in [
                    "scenario",
                    "B",
                    "isl",
                    "ttft_ms",
                    "prefill_tok_s",
                    "decode_ms_tok",
                    "tok_s_user",
                    "tok_s_total",
                    "free_gib_chip",
                ]
            )
            + " |"
        )
    if r["spec"]:
        out.append("")
        out += [
            f"{d['_kind']}: " + ", ".join(f"{k}={v}" for k, v in d.items() if k not in ("_kind", "k_hist"))
            for d in r["spec"]
        ]
    if r["first_tokens"] is not None:
        out.append(f"\nfirst tokens: {r['first_tokens']}")
    if r["built_layers"]:
        out.append(f"built layers: {r['built_layers']} (last at {r['build_s']} s)")
    return "\n".join(out)


def compare(ra, rb):
    problems = []
    if ra["host"] != rb["host"]:
        problems.append(f"different hosts (.{ra['host']} vs .{rb['host']}): cross-host comparison INVALID")
    if ra["commit"] != rb["commit"]:
        problems.append(
            f"different commits ({ra['commit']} vs {rb['commit']}): comparison measures the code change plus anything else"
        )
    envdiff = {
        k: (ra["env"].get(k), rb["env"].get(k))
        for k in sorted(set(ra["env"]) | set(rb["env"]))
        if ra["env"].get(k) != rb["env"].get(k)
    }
    if not ra["host"] or not rb["host"] or not ra["commit"] or not rb["commit"]:
        problems.append(
            "missing GRID_ENV header in a log: host/commit unknown, comparison INVALID (launch with launch.py)"
        )
    valid = not any("INVALID" in p for p in problems)
    sb = {s["scenario"]: s for s in rb["scenarios"]}
    rows = []
    for s in ra["scenarios"]:
        t = sb.get(s["scenario"])
        if not t:
            continue
        row = {"scenario": s["scenario"]}
        for k in ["ttft_ms", "prefill_tok_s", "decode_ms_tok", "tok_s_user"]:
            if k in s and k in t and s[k]:
                row[k] = {"a": s[k], "b": t[k], "b_over_a": round(t[k] / s[k], 3)}
        rows.append(row)
    return {"valid": valid, "problems": problems, "env_diff": envdiff, "rows": rows}


def fmt_cmp(c):
    out = [
        ("VALID" if c["valid"] else "INVALID COMPARISON") + (": " + "; ".join(c["problems"]) if c["problems"] else "")
    ]
    if c["env_diff"]:
        out.append("env differs: " + ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in c["env_diff"].items()))
    for r in c["rows"]:
        out.append(
            r["scenario"]
            + ": "
            + ", ".join(f"{k} {v['a']} -> {v['b']} (x{v['b_over_a']})" for k, v in r.items() if k != "scenario")
        )
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("extract")
    s.add_argument("log")
    s.add_argument("--json", action="store_true")
    s.add_argument("--md", action="store_true")
    s = sub.add_parser("compare")
    s.add_argument("log_a")
    s.add_argument("log_b")
    s.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    rd = lambda f: open(f, errors="ignore").read()  # noqa: E731
    if a.cmd == "extract":
        r = extract(rd(a.log))
        print(json.dumps(r, indent=1) if a.json else fmt_md(r))
        return 0
    c = compare(extract(rd(a.log_a)), extract(rd(a.log_b)))
    print(json.dumps(c, indent=1) if a.json else fmt_cmp(c))
    return 0 if c["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
