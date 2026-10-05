#!/usr/bin/env python3
"""FULL GRID driver: ISL {128 (gsm8k), 4k, 8k, 32k, 64k, 128k, 256k} x batch {4,8,16,32,64,128} x spec {off,on}.
One process per (batch, group): G1={gsm8k,4k,8k}, G2={32k,64k}, G3a={128k}, G3b={256k}; spec runs in the SAME process (DSV41_SPEC=k: the demo prints plain, then spec).
  grid.py plan                 cell list, groups, wall-time estimate
  grid.py launch <B> <G> <host>   start one process from MAIN via run_main.sh, marker-based hangwatch (45 min stall)
  grid.py sched [hosts...]     scheduler loop: assigns pending processes (priority order) to idle hosts, retries a HANG once
  grid.py report               parses the logs -> GRID.md
"""
import json
import os
import re
import subprocess
import sys
import time

LOG = "/mnt/tt-data/ssinghal/dsv4-logs"
MAIN = "/mnt/tt-data/ssinghal/tests/tt-metal"
WT = f"{MAIN}/models/demos/blackhole/deepseek_v41_flash/tools"  # run_main.sh lives here; GRID.md is written to OUT
OUT = "/mnt/tt-data/ssinghal/wt/h47i"
DEMO = "models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py"
STATE = f"{OUT}/grid_state.json"
BATCHES = [16, 4, 8, 32, 64, 128]  # priority order
GROUPS = {
    "G1": ["gsm8k", "isl4k", "isl8k"],
    "G2a": ["isl32k"],
    "G2b": ["isl64k"],
}  # user limited ISL to 64k (128k and 256k groups dropped)
EXTRA = [(128, "S128", ["gsm8k"])]  # B=128 spec k=1 attempt, separate process
ISL = {
    "gsm8k": 128,
    "isl4k": 3720,
    "isl8k": 7443,
    "isl32k": 30059,
    "isl64k": 60453,
    "isl128k": 126000,
    "isl256k": 250000,
}
SPEC_K = {
    4: 0,  # spec k=3 at B=4 failed (AssertionError mhc_mixes2 T=5), see grid_b4_G1_specfail.log; plain re-run
    8: 3,
    16: 3,
    32: 3,
    64: 1,
    128: 0,
}  # B=128 spec is ATTEMPTED in its own process (group "S128", k=1; expected to hit the 32-row limit: T=U*(1+k)=64 rows/mesh row) so a failure cannot abort the plain sessions
EXTRA_ENV = "DSV41_MEMLOG=1 DSV41_ENGRAM_RAM=1"
DEFAULT_HOSTS = [
    "30",
    "31",
    "34",
    "35",
    "41",
    "44",
    "45",
    "46",
    "47",
    "48",
    "32",
    "33",
    "42",
]  # .40/.43: batch-4 agent; .32/.33 free once the gate runs end; .42 last (first-op hangs)
SPEC_ENV = "DSV41_TRACE_REGION=1900000000"


def head_hash():
    return subprocess.run(
        ["git", "-C", MAIN, "rev-parse", "--short=11", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def tree_clean():
    """The main tree must be clean (tools/ and untracked files ignored) and at the expected hash."""
    r = subprocess.run(
        ["git", "-C", MAIN, "status", "--porcelain", "-uno", "--", "models", "ttnn", "tt_metal"],
        capture_output=True,
        text=True,
    ).stdout
    dirty = [l for l in r.splitlines() if "/tools/" not in l]
    return not dirty, dirty


def check_hash():
    want = os.environ.get("GRID_HASH")
    if not want:
        sys.exit("set GRID_HASH=<convergence commit> (refusing to start without it)")
    ok, dirty = tree_clean()
    if head_hash()[: len(want)] != want[: len(head_hash())] and not head_hash().startswith(want[:11]):
        sys.exit(f"HEAD {head_hash()} != GRID_HASH {want}: refusing to start")
    if not ok:
        sys.exit(f"main tree dirty: {dirty[:5]}: refusing to start")


def scenarios(B, g):
    if g == "S128":
        return ["gsm8k_b128"]
    return [f"{k}_b{B}" for k in GROUPS[g]]


def env_for(B, g, spec=True):
    e = EXTRA_ENV
    if (B == 128 and g != "G1") or (B == 64 and g.startswith("G2")):
        e += " DSV41_POOL_DTYPE=fp8"  # bf16 pool cannot hold 128 users at >= 32k (recorded in GRID.md)
    k = (1 if g == "S128" else SPEC_K[B]) if spec else 0
    if g.startswith("G2"):
        e += " DSV41_BUILD_SLOTS=10"  # user: run the long-ISL cells fully in parallel (cap raised from 5)
        k = 0  # spec at >= 32k asserts 'spec verify needs the matmul indexer backend' (B=8/16/32 G2 first pass); plain re-run
    if k:
        e += f" DSV41_SPEC={k} {SPEC_ENV}"
    return e


def procs():
    out = []
    order = {"G1": 0, "G2a": 1, "G2b": 2, "G3a": 3, "G3b": 4}
    for g in sorted(GROUPS, key=lambda x: order[x]):
        for B in BATCHES:
            out.append((B, g))
    out.append((128, "S128"))
    return out  # G1 before G2 before G3; B=16 first inside each; the B=128 spec attempt last


def est_minutes(B, g):
    if g == "S128":
        return 100
    tok_s = 3000 if B <= 32 else 2800
    pre = sum(B * ISL[k] / tok_s / 60 * (1 + (ISL[k] / 100000) * 0.5) for k in GROUPS[g])
    dec = 3 * len(GROUPS[g])  # decode + spec phases (64 tokens each)
    return int(
        75 + pre * 2 + dec + (8 if SPEC_K[B] else 0)
    )  # build ~75 min under the slot cap; x2: compile run + measured run


def tag(B, g):
    return f"grid_b{B}_{g}"


def cell_env(B, g):
    """documented per-cell variables (everything else DSV41_* is unset by grid_exec.sh); DSV41_LAYERS is forced to 0-39 there."""
    return env_for(B, g) + f" DSV41_SESSION={','.join(scenarios(B, g))}"


def resolved_env(B, g):
    d = dict(kv.split("=", 1) for kv in cell_env(B, g).split())
    d["DSV41_LAYERS"] = "0-39"
    return d


def cmd(B, g):
    r = resolved_env(B, g)
    assert r["DSV41_LAYERS"] == "0-39", "every grid run must use all 40 layers"
    kv = "+".join(cell_env(B, g).split())  # '+'-joined: no spaces/quotes survive the nested shells
    inner = f"GRID_KV={kv} {WT}/grid_exec.sh timeout 43200 pytest -x -s -q -o junit_suite_name={tag(B, g)} {DEMO} -k session"
    return f"{LOG}/{tag(B, g)}.log", inner


def launch(B, g, host):
    if not os.environ.get("DRY"):
        check_hash()
    log, inner = cmd(B, g)
    script = f"""cd {WT} && : > {log}
setsid nohup ./run_main.sh '{inner}' > {log} 2>&1 < /dev/null &
setsid nohup bash -c 'until PID=$(pgrep -n -f "[p]ython_env/bin/python3 .*junit_suite_name={tag(B, g)}"); [ -n "$PID" ]; do sleep 10; done; sleep 20; exec /mnt/tt-data/ssinghal/hangwatch.sh $PID {log} 45' > {log}.hw 2>&1 < /dev/null &
"""
    if os.environ.get("DRY"):
        print(script)
        return
    subprocess.run(["ssh", "-o", "BatchMode=yes", f"10.82.97.{host}", "bash -s"], input=script, text=True)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    st.setdefault(tag(B, g), {"launches": []})["launches"].append({"host": host, "t": time.time()})
    json.dump(st, open(STATE, "w"), indent=1)
    print("launched", tag(B, g), "on", host)


def host_idle(h):
    r = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            f"10.82.97.{h}",
            "pgrep -fc '[p]ython_env/bin/python3 .*(pytest|tt-triage)' ; pgrep -fc '[f]lock .*dsv4_dev.lock'; uptime | sed 's/.*average: //'",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode not in (0, 1) and not r.stdout:
        return False
    lines = r.stdout.split()
    try:
        return int(lines[0]) == 0 and int(lines[1]) == 0 and float(lines[2].strip(",")) < 3
    except Exception:
        return False


def proc_status(B, g):
    f = f"{LOG}/{tag(B, g)}.log"
    if not os.path.exists(f):
        return "pending"
    t = open(f, errors="ignore").read()
    if re.search(r"\d+ (passed|failed)", t):
        return "done"
    if "hangwatch] killing" in t:
        return "hang"
    return "running"


def sched(hosts=None):
    check_hash()
    hosts = hosts or DEFAULT_HOSTS
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    while True:
        pend = []
        for B, g in procs():
            s = proc_status(B, g)
            n = len(st.get(tag(B, g), {}).get("launches", []))
            if s == "pending" or (s == "hang" and n < 2):
                pend.append((B, g))
        if not pend:
            print("nothing pending")
            return
        for h in hosts:
            if not pend:
                break
            if host_idle(h):
                B, g = pend.pop(0)
                launch(B, g, h)
                st = json.load(open(STATE))
        time.sleep(300)


def parse_log(f, B):
    t = open(f, errors="ignore").read()
    cells = {}
    parts = re.split(r"=== session scenario (\S+) ===", t)
    for i in range(1, len(parts), 2):
        sc, seg = parts[i], parts[i + 1]
        c = {"scenario": sc}
        m = re.search(r"TTFT \(whole batch of (\d+) users, ISL max (\d+)\): (\d+) ms -> prefill (\d+) tok/s", seg)
        if m:
            c.update(B=m.group(1), ISL=m.group(2), ttft=int(m.group(3)) / 1000, ptps=m.group(4))
        m = re.search(r"Decode: ([\d.]+) ms/token @ ([\d.]+) tok/s/user \(([\d.]+) tok/s throughput\)", seg)
        if m:
            c.update(ms=m.group(1), tpsu=m.group(2), tpst=m.group(3))
        m = re.search(
            r"=== SPEC k=(\d+).*?([\d.]+) accepted drafts/round.*?round ([\d.]+) ms -> ([\d.]+) tok/s/user vs plain",
            seg,
        )
        if m:
            c.update(spec_k=m.group(1), acc=m.group(2), rnd=m.group(3), stpsu=m.group(4))
        m = re.search(r"SPEC exactness.*?: (\d+)/(\d+) users identical", seg)
        if m:
            c["exact"] = f"{m.group(1)}/{m.group(2)}"
        m = re.search(r"MEMLOG prefill end\s+allocated\s+[\d.]+ MiB/bank\s+free\s+([\d.]+)", seg)
        if m:
            c["free"] = f"{float(m.group(1)) * 8 / 1024:.1f}"
        if sc.startswith("gsm8k"):
            gold = [
                x["answer"]
                for x in json.load(
                    open(
                        f"{MAIN}/models/demos/blackhole/deepseek_v41_flash/demo/sample_prompts/input_data_gsm8k_128_with_answers.json"
                    )
                )
            ]
            ok = fin = n = 0
            plain = seg.split("SPEC OUTPUT")[0]
            for u, body in re.findall(r"==USER (\d+) - OUTPUT\n(.*?)(?=\n==|\Z)", plain, re.S):
                n += 1
                x = re.findall(r"boxed\{([^}]*)\}", body)
                p = x[-1].replace(",", "").replace("$", "").replace("\\", "").strip() if x else None
                fin += p is not None
                ok += p == gold[int(u)]
            if n:
                c["gsm"] = f"{ok}/{fin}/{n}"
        cells[sc] = c
    return cells, t


def errtext(txt):
    m = re.findall(
        r"(?:^|\n)E\s+(\w*(?:Error|Exception|assert)[^\n]{0,160})|(AssertionError[^\n]{0,160})|(TT_FATAL[^\n]{0,160})|(TT_THROW[^\n]{0,160})",
        txt,
    )
    for tup in m[-1:]:
        return " ".join(x for x in tup if x).replace("|", "/")
    return ""


def report():
    rows, fails = [], []
    for B, g in procs():
        f = f"{LOG}/{tag(B, g)}.log"
        cells = {}
        txt = ""
        if os.path.exists(f):
            cells, txt = parse_log(f, B)
        layers = (max([int(x) for x in re.findall(r"built layer (\d+) \(", txt)] or [-1]) + 1) if txt else 0
        hang = "hangwatch] killing" in txt
        oom = "Out of Memory" in txt
        for sc in scenarios(B, g):
            c = cells.get(sc, {"scenario": sc})
            if "ttft" in c and layers != 40:
                status = f"INVALID(layers={layers})"
            elif "ttft" in c:
                status = "OK"
            elif not os.path.exists(f):
                status = "not run"
            elif hang:
                status = "HANG"
            elif oom:
                status = "OOM"
            elif proc_status(B, g) == "running":
                status = "running"
            else:
                status = "FAIL"
            if "spec_k" in c:
                spec = None
            elif B == 4:
                spec = "FAIL: AssertionError mhc_mixes2.py:32 (T=5 drafter rows at U=1, k=3); grid_b4_G1_specfail.log"
            elif B == 128 and g == "S128":
                spec = "FAIL: k=1 attempt asserts (T=U*(1+k)=64 rows/mesh row > 32)"
            elif B == 128:
                spec = "n/a (kernel limit: T=U*(1+k) rows)"
            elif g.startswith("G2"):
                spec = "FAIL at >=32k: 'spec verify needs the matmul indexer backend' (B=8/16/32 first pass, *_firstpass_fail.log); plain re-run"
            else:
                spec = None
            if spec is None:
                spec = (
                    "n/a (kernel limit: T=U*(1+k) rows)"
                    if not SPEC_K[B]
                    else (
                        "spec: "
                        + (
                            f"k={c['spec_k']} acc {c['acc']}/round, round {c['rnd']} ms, {c['stpsu']} tok/s/user, ratio {float(c['stpsu']) / float(c['tpsu']):.2f}x, exact {c.get('exact', '?')}"
                            if "spec_k" in c
                            else "no result"
                        )
                    )
                )
            rows.append(
                (
                    B,
                    sc,
                    status,
                    c,
                    spec,
                    os.path.basename(f),
                    "fp8" if B == 128 and g in ("G2a", "G2b", "G3a", "G3b") else "bf16",
                    layers,
                    env_for(B, g),
                )
            )
    hdr = "| B | scenario | status | ISL | TTFT s | prefill tok/s | decode ms/tok | tok/s/user | tok/s total | GSM ok/fin/n | free GiB/chip | pool | layers | spec | non-default env | log |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {B} | {sc} | {st} | {c.get('ISL','')} | {c.get('ttft','')} | {c.get('ptps','')} | {c.get('ms','')} | {c.get('tpsu','')} | {c.get('tpst','')} | {c.get('gsm','')} | {c.get('free','')} | {pool} | {ly} | {sp} | {ev} | {lg} |\n"
        for B, sc, st, c, sp, lg, pool, ly, ev in rows
    )
    open(f"{OUT}/GRID.md", "w").write(f"# FULL GRID (grid.py report), main HEAD {head_hash()}\n\n" + hdr + body)
    print(hdr + body)


def plan():
    tot = 0
    for B, g in procs():
        m = est_minutes(B, g)
        tot += m
        print(f"B={B:3d} {g:4s} {scenarios(B, g)} spec k={SPEC_K[B]} est {m:4d} min  env: {env_for(B, g)}")
    print(
        "processes:",
        len(procs()),
        " sum of process minutes:",
        tot,
        f" -> /15 hosts, build cap 5: wall ~{tot / 5 / 60:.1f}-{tot / 15 / 60 + 3:.1f} h",
    )


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "plan":
        plan()
    elif c == "report":
        report()
    elif c == "launch":
        launch(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif c == "hosts":
        print(" ".join(DEFAULT_HOSTS))
    elif c == "sched":
        sched(sys.argv[2:])
