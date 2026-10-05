#!/usr/bin/env python3
"""FINAL matrix driver. `final_matrix.py plan` prints the groups (one process = one model build = one batch size, several ISLs back to back: flat DRAM verified),
`final_matrix.py launch <group> <host>` starts it from the MAIN tree (run_main.sh) with MEMLOG=1 + hangwatch on the real pytest pid,
`final_matrix.py report` parses the logs into FINAL_MATRIX.md."""
import glob
import os
import re
import subprocess
import sys

LOG = "/mnt/tt-data/ssinghal/dsv4-logs"
DEMO = "models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py"
E2E = "models/demos/blackhole/deepseek_v41_flash/tests/test_e2e_prefill_decode.py"
# group -> (env, scenario ids, est. minutes incl. ~35 min build)
GROUPS = {
    "b4_small": ("", ["prefill_128_b4", "isl4k_b4", "isl8k_b4", "isl16k_b4"], 70),
    "b4_32k": ("", ["isl32k_b4"], 55),
    "b4_64k": ("", ["isl64k_b4"], 70),
    "b16_small": ("", ["gsm8k_b16", "prefill_128_b16", "isl4k_b16", "isl8k_b16", "isl16k_b16"], 85),
    "b16_32k": ("", ["isl32k_b16"], 55),
    "b16_64k": ("", ["isl64k_b16"], 85),
    "b32_small": ("", ["prefill_128_b32", "isl4k_b32", "isl8k_b32", "isl16k_b32"], 100),
    "b32_small_g1": ("DSV41_MOE_G=1", ["prefill_128_b32", "isl4k_b32", "isl8k_b32", "isl16k_b32"], 120),
    "b64_small": ("", ["gsm8k_b64", "isl4k_b64", "isl8k_b64", "isl16k_b64"], 110),
    "b128_small": ("DSV41_POOL_DTYPE=fp8", ["prefill_128_b128", "isl4k_b128"], 80),
    "b32_64k": ("DSV41_POOL_DTYPE=fp8", ["isl64k_b32"], 120),  # only if it fits after the U=8 fix
    "spec_b16_gsm": ("DSV41_SPEC=3 DSV41_TRACE_REGION=1900000000 DSV41_PREFILL_ROW_TOKENS=2048", ["gsm8k_b16"], 50),
    "plain_b16_gsm_ref": ("DSV41_TRACE_REGION=1900000000 DSV41_PREFILL_ROW_TOKENS=2048", ["gsm8k_b16"], 50),
    "spec_b16_2k": ("DSV41_SPEC=3 DSV41_TRACE_REGION=1900000000 DSV41_PREFILL_ROW_TOKENS=2048", ["isl2k_b16"], 50),
    "spec_b4_2k4k": (
        "DSV41_SPEC=3 DSV41_TRACE_REGION=1900000000 DSV41_PREFILL_ROW_TOKENS=2048",
        ["isl2k_b4", "isl4k_b4"],
        60,
    ),
}


def cmd(group, host):
    global TAG
    TAG = f"final_{group}"
    env, ids, _ = GROUPS[group]
    tag = f"final_{group}"
    log = f"{LOG}/{tag}.log"
    inner = f"DSV41_MEMLOG=1 DSV41_ENGRAM_RAM=1 {env} DSV41_SESSION={','.join(ids)} timeout 20000 pytest -x -s -q -o junit_suite_name={tag} {DEMO} -k session"
    return log, inner


def plan():
    tot = 0
    for g, (env, ids, m) in GROUPS.items():
        print(f"{g:20s} {m:4d} min  env='{env}'  {ids}")
        tot += m
    print("sum of group minutes:", tot)


def launch(group, host):
    log, inner = cmd(group, host)
    tag = TAG
    script = f"""cd /mnt/tt-data/ssinghal/wt/h47i && : > {log}
setsid nohup ./run_main.sh '{inner}' > {log} 2>&1 < /dev/null &
setsid nohup bash -c 'until PID=$(pgrep -n -f "[p]ython_env/bin/python3 .*junit_suite_name={TAG}"); [ -n "$PID" ] && true; do sleep 10; done; sleep 20; exec /mnt/tt-data/ssinghal/hangwatch.sh $PID {log} 30' > {log}.hw 2>&1 < /dev/null &
"""
    if os.environ.get("DRY"):
        print(script)
        return
    subprocess.run(["ssh", "-o", "BatchMode=yes", f"10.82.97.{host}", "bash -s"], input=script, text=True)
    print("launched", group, "on", host, "->", log)


def report():
    rows = []
    for f in sorted(
        glob.glob(f"{LOG}/final_*.log")
        + glob.glob(f"{LOG}/h41d_b4_*.log")
        + glob.glob(f"{LOG}/h41d_v_prefill_128_b4.log")
    ):
        t = open(f, errors="ignore").read()
        sc = None
        for m in re.finditer(
            r"=== session scenario (\S+) ===|TTFT \(whole batch of (\d+) users, ISL max (\d+)\): (\d+) ms -> prefill (\d+) tok/s|Decode: ([\d.]+) ms/token @ ([\d.]+) tok/s/user \(([\d.]+) tok/s throughput\)|MEMLOG prefill end\s+allocated\s+([\d.]+) MiB/bank\s+free\s+([\d.]+)",
            t,
        ):
            if m.group(1):
                sc = {"scenario": m.group(1), "log": os.path.basename(f)}
                rows.append(sc)
            elif m.group(2) and sc is not None:
                sc.update(B=m.group(2), ISL=m.group(3), ttft_s=int(m.group(4)) / 1000, prefill_tps=m.group(5))
            elif m.group(6) and sc is not None:
                sc.update(ms_tok=m.group(6), tps_user=m.group(7), tps_total=m.group(8))
            elif m.group(9) and sc is not None:
                sc["free_gib_chip"] = f"{float(m.group(10)) * 8 / 1024:.2f}"
        if "TT_FATAL" in t or "TT_THROW" in t:
            for r in rows:
                if r["log"] == os.path.basename(f) and "ms_tok" not in r:
                    r["status"] = "FAIL: " + ("OOM" if "Out of Memory" in t else "TT_THROW/FATAL")
    hdr = "| scenario | B | ISL | TTFT s | prefill tok/s | decode ms/tok | tok/s/user | tok/s total | free GiB/chip | status | log |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r['scenario']} | {r.get('B','')} | {r.get('ISL','')} | {r.get('ttft_s',''):} | {r.get('prefill_tps','')} | {r.get('ms_tok','')} | {r.get('tps_user','')} | {r.get('tps_total','')} | {r.get('free_gib_chip','')} | {r.get('status','ok' if 'ms_tok' in r else 'no result')} | {r['log']} |\n"
        for r in rows
    )
    open("/mnt/tt-data/ssinghal/wt/h47i/FINAL_MATRIX.md", "w").write(
        "# FINAL matrix (generated by final_matrix.py report)\n\n" + hdr + body
    )
    print(hdr + body)


if __name__ == "__main__":
    {"plan": plan, "report": report}.get(sys.argv[1], lambda: launch(sys.argv[2], sys.argv[3]))()
