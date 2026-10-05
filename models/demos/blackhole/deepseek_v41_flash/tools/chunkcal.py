#!/usr/bin/env python3
"""Chunk-size (row-token budget) calibration driver.
  chunkcal.py plan [B...]               print the (process, sessions, rowtok list) plan
  chunkcal.py launch <B> <G1|G2> <host> start one process on a host (through pfrun.sh + hangwatch)
  chunkcal.py report [B...]             parse the logs -> table per B (TTFT, replay, min free DRAM, failures, output equality vs the default budget)
"""
import glob
import json
import os
import re
import subprocess
import sys

LOG = "/mnt/tt-data/ssinghal/dsv4-logs"
WT = "/mnt/tt-data/ssinghal/wt/pf_chunkcal"
PK = "models/demos/blackhole/deepseek_v41_flash"
ISL = {"4k": 3720, "8k": 7443, "32k": 30059, "64k": 60453}
GROUPS = {"G1": ["4k", "8k"], "G2": ["32k", "64k"]}
BUDGETS = [512, 1024, 2048, 4096, 8192, 16384]
PAD = 128


def default_budget(isl_name, U):  # the current auto rule (tt/generator.py auto_chunk): max_len is the prompt length
    return 4096 if ISL[isl_name] <= 16384 and U <= 4 else 2048


def chunk_of(budget, U):
    return max(128, (budget // U) // 128 * 128)


def padded(n):
    return -(-n // PAD) * PAD


def sweep(B, isl_name):
    """budgets to run for (B, ISL): distinct effective chunks C (budget -> U*C), the default budget always included, one single-chunk entry at most."""
    U = B // 4
    seen, out = set(), []
    for b in sorted(set(BUDGETS + [default_budget(isl_name, U)])):
        c = chunk_of(b, U)
        b = U * c  # effective row-token budget (a budget below 128 tokens per user is raised)
        single = c >= padded(ISL[isl_name])
        key = "single" if single else c
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def plan(B, g):
    ids, rt = [], []
    for k in GROUPS[g]:
        bl = sweep(B, k)
        for b in bl:
            ids.append(f"isl{k}_b{B}")
            rt.append(str(b))
        d = (B // 4) * chunk_of(default_budget(k, B // 4), B // 4)
        if ISL[k] * B <= 8 * 7443 * 4 and d in bl:  # cheap cells: repeat the default at the end (noise / reproducibility)
            ids.append(f"isl{k}_b{B}")
            rt.append(str(d))
    return ids, rt


def flags(B, g):
    f = ""
    if B == 64 and g == "G2":
        f += "DSV41_POOL_DTYPE=fp8 "  # bf16 pool cannot hold 64 users at >= 32k next to the chunk buffers (as in the grid)
    return f.strip()


def launch(B, g, host):
    ids, rt = plan(B, g)
    tag = f"b{B}_{g}"
    log = f"{LOG}/pf_chunkcal_{tag}.log"
    inner = f"{PK}/tools/chunkcal_exec.sh {tag} \"{flags(B, g)}\" {','.join(ids)} {','.join(rt)}"
    script = f"""cd {WT} && : > {log}
setsid nohup {PK}/tools/pfrun.sh '{inner}' {WT} > {log} 2>&1 < /dev/null &
"""
    subprocess.run(["ssh", "-o", "BatchMode=yes", f"10.82.97.{host}", "bash -s"], input=script, text=True)
    print("launched", tag, "on", host, "->", log)


def parse(path):
    """list of dict per scenario run in the log"""
    txt = open(path, errors="replace").read()
    parts = re.split(r"(?==== session scenario \S+ \(DSV41_PF_ASYNC)|(?==== session scenario \S+ ROW_TOKENS=\d+ SKIPPED)", txt)
    runs = []
    for p in parts:
        if not p:
            continue
        m = re.match(r"=== session scenario (\S+) \(DSV41_PF_ASYNC=\S+, ROW_TOKENS=(\d+)\)", p)
        if not m:
            m2 = re.match(r"=== session scenario (\S+) ROW_TOKENS=(\d+) SKIPPED", p)
            if m2:
                runs.append(dict(id=m2.group(1), budget=int(m2.group(2)), status="skipped"))
            continue
        r = dict(id=m.group(1), budget=int(m.group(2)), status="ok")
        f = re.search(r"SCENARIO FAILED \S+ ROW_TOKENS=\d+: (.*?) ===", p, re.S)
        if f:
            r["status"] = "FAIL"
            r["error"] = f.group(1)[:160].replace("\n", " ")
        t = re.search(r"prefill timing \{(.*?)\} chunk=(\S+)", p)
        if t:
            d = dict((a.strip(), float(b)) for a, b in (x.split(":") for x in t.group(1).split(",")))
            r["replay"] = d.get("total_replay_loop")
            r["capture"] = d.get("compile_and_capture")
            r["chunk"] = t.group(2)
        tt = re.search(r"TTFT \(whole batch of (\d+) users, ISL max (\d+)\): (\d+) ms", p)
        if tt:
            r["ttft"] = int(tt.group(3)) / 1000
            r["isl"] = int(tt.group(2))
        fr = [float(x) for x in re.findall(r"MEMLOG .*? free\s+([\d.]+)", p)]
        if fr:
            r["min_free"] = min(fr)
        e = re.search(r"MEMLOG PF_END.*? free\s+([\d.]+)", p)
        if e:
            r["end_free"] = float(e.group(1))
        ft = re.search(r"FIRSTTOK_IDS (\[.*?\])", p)
        if ft:
            r["first"] = json.loads(ft.group(1))
        r["texts"] = {
            int(a): b.strip()
            for a, b in re.findall(r"==USER (\d+) - OUTPUT\n(.*?)(?=\n==USER|\n==REPEAT|\n\d{4}-\d\d-\d\d |\Z)", p, re.S)
        }
        runs.append(r)
    return runs


def compare(r, ref):
    if not ref or "first" not in r or "first" not in ref:
        return "-"
    n = len(ref["first"])
    f = sum(a == b for a, b in zip(r["first"], ref["first"]))
    t = sum(r["texts"].get(u) == ref["texts"].get(u) for u in range(n))
    return f"first {f}/{n} text {t}/{n}"


def report(Bs):
    for B in Bs:
        for g in GROUPS:
            path = f"{LOG}/pf_chunkcal_b{B}_{g}.log"
            if not os.path.exists(path):
                continue
            runs = parse(path)
            print(f"\n### B={B} (U={B // 4}) group {g}: {path}")
            print("| ISL | budget | C | status | TTFT s | replay s | capture s | min free MiB | end free MiB | vs default |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for r in runs:
                k = r["id"].split("_")[0][3:]
                ref = next((x for x in runs if x["id"] == r["id"] and x["budget"] == (B // 4) * chunk_of(default_budget(k, B // 4), B // 4) and x.get("first")), None)
                print(
                    f"| {r['id']} | {r['budget']} | {r.get('chunk','-')} | {r['status']}{(' ' + r['error']) if 'error' in r else ''} | "
                    f"{r.get('ttft', '-')} | {r.get('replay', '-')} | {r.get('capture', '-')} | {r.get('min_free', '-')} | {r.get('end_free', '-')} | {compare(r, ref)} |"
                )


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "plan":
        for B in map(int, sys.argv[2:] or [4, 8, 16, 32, 64]):
            for g in GROUPS:
                ids, rt = plan(B, g)
                print(B, g, flags(B, g), ",".join(ids), ",".join(rt))
    elif c == "launch":
        launch(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif c == "report":
        report(list(map(int, sys.argv[2:] or [4, 8, 16, 32, 64])))
