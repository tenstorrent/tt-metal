#!/usr/bin/env python3
"""Chunk-size (row-token budget) calibration driver.
  chunkcal.py plan [B...]               print the (process, sessions, rowtok list) plan
  chunkcal.py launch <B> <base|uni> <host> start one process on a host (through pfrun.sh + hangwatch)
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
ISLS = ["4k", "8k", "32k", "64k"]
BUDGETS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
SPAD_MAX = 65536  # chunk trace sized once for a 64k context (the deployable configuration); 0 = per-prompt sizing
PAD = 128


def default_budget(isl_name, U):  # the current auto rule (tt/generator.py auto_chunk)
    return 4096 if ISL[isl_name] <= 16384 and U <= 4 else 2048


def chunk_of(budget, U):
    return max(128, (budget // U) // 128 * 128)


def budgets(B):
    if os.environ.get("CC_BUDGETS"):
        return [int(x) for x in os.environ["CC_BUDGETS"].split(",")]
    """effective row-token budgets U*C (a budget below 128 tokens per user is raised to U*128), ascending, distinct; the CURRENT default budgets are included"""
    U = B // 4
    out = sorted({U * chunk_of(b, U) for b in BUDGETS + [4096, 2048]})
    return out


def plan(B, mode="base"):
    ids, rt = [], []
    for k in ISLS:
        for b in budgets(B):
            ids.append(f"isl{k}_b{B}")
            rt.append(str(b))
    return ids, rt


def flags(B, mode="base"):
    f = "DSV41_PREFILL_ONLY=1 DSV41_CHUNK_FIXED=1"
    spad = int(os.environ.get("CC_SPAD", SPAD_MAX))
    if spad:
        f += f" DSV41_PREFILL_SPAD_MAX={spad}"
    if mode == "uni":
        f += " DSV41_PREFILL_MOE=unified DSV41_UNI_NODECODE=1"
    if B == 64:
        f += " DSV41_POOL_DTYPE=fp8"  # bf16 pool cannot hold 64 users at 64k (as in the grid)
    return f


def launch(B, mode, host):
    ids, rt = plan(B, mode)
    tag = f"{mode}_b{B}{os.environ.get('CC_TAG', '')}"
    log = f"{LOG}/pf_chunkcal_{tag}.log"
    inner = f"{PK}/tools/chunkcal_exec.sh {tag} \"{flags(B, mode)}\" {','.join(ids)} {','.join(rt)}"
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
        if "SKIPPED (budget" in p:
            continue
        ft_ = re.search(r"FAILTRACE (.*)", p)
        f = re.search(r"SCENARIO FAILED \S+ ROW_TOKENS=\d+: (.*?) ===", p, re.S)
        if f:
            r["status"] = "FAIL"
            r["error"] = f.group(1)[:330].replace("\n", " ").replace("TT_FATAL @ /mnt/tt-data/ssinghal/tests/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:495: false info: ", "")
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
        if r["status"] == "ok" and "replay" not in r and "first" not in r:
            continue  # header of a skipped / still running scenario
        runs.append(r)
    return runs


def compare(r, ref):
    if not ref or "first" not in r or "first" not in ref:
        return "-"
    n = len(ref["first"])
    f = sum(a == b for a, b in zip(r["first"], ref["first"]))
    return f"first {f}/{n}"


def report(Bs, modes=("base", "uni")):
    for mode in modes:
        for B in Bs:
            path = f"{LOG}/pf_chunkcal_{mode}_b{B}{os.environ.get('CC_TAG', '')}.log"
            if not os.path.exists(path):
                continue
            U = B // 4
            runs = parse(path)
            print(f"\n### {mode} B={B} (U={U}): {path}")
            print("| ISL | budget | C | status | TTFT s | replay s | ms/row-token | capture s | min free MiB | end free MiB | first tok vs 4096/2048 default |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in runs:
                k = r["id"].split("_")[0][3:]
                dflt = U * chunk_of(default_budget(k, U), U)
                ref = next((x for x in runs if x["id"] == r["id"] and x["budget"] == dflt and x.get("first")), None)
                msr = "-"
                if r.get("replay") and r.get("isl"):
                    c = int(r["chunk"]) if str(r["chunk"]).isdigit() else r["budget"] // U
                    ntok = -(-r["isl"] // c) * c * U
                    msr = f"{1e3 * r['replay'] / ntok:.3f}"
                print(
                    f"| {r['id']} | {r['budget']} | {r.get('chunk','-')} | {r['status']}{(' ' + r['error']) if 'error' in r else ''} | "
                    f"{r.get('ttft', '-')} | {r.get('replay', '-')} | {msr} | {r.get('capture', '-')} | {r.get('min_free', '-')} | {r.get('end_free', '-')} | {compare(r, ref)} |"
                )


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "plan":
        for B in map(int, sys.argv[2:] or [4, 8, 16, 32, 64]):
            ids, rt = plan(B)
            print(B, flags(B), ",".join(ids), ",".join(rt))
    elif c == "launch":
        launch(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif c == "report":
        report(list(map(int, sys.argv[2:] or [4, 8, 16, 32, 64])))
