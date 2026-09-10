import json
import subprocess
import sys
import time

sys.path.insert(0, "/tmp/timer_bench")


# ---------- A) CURRENT arithmetic (what ships today) ----------
def decide_current(s):
    # replicates run.py: stall_sec=600 FROZEN check, max_no_progress=_adaptive_cap(max(600*4,2400))
    base = s["base"]
    stall = 600
    cap = min(10800, max(max(stall * 4, 2400), int(3 * base)))
    alive = (s["device_cpu_delta"] > 0) or (s["transcript_growth_bytes"] > 0)
    if not alive and s["elapsed"] > stall:
        return "kill", f"FROZEN>{stall}s"
    if s["since_commit"] > cap:
        return "kill", f"UNPRODUCTIVE>{cap}s"
    return "wait", f"within cap {cap}s"


# ---------- B) IMPROVED arithmetic (proportional floors, per-op base) ----------
_OP_COST = {"check_pcc": 12.0, "measure_candidate": 3.0, "profile_model": 3.0, "kernel_compile(host)": 6.0}


def decide_improved(s):
    base = s["base"] or 60.0  # cold-start prior
    op_mult = _OP_COST.get(s["tool_in_flight"] or "", 4.0)
    budget = max(30.0, min(10800.0, op_mult * base * 3))  # proportional, no absolute floor
    alive = (s["device_cpu_delta"] > 100) or (s["transcript_growth_bytes"] > 1000)
    if not alive and s["elapsed"] > max(60.0, 20 * base):
        return "kill", f"no progress > {max(60.0, 20*base):.0f}s"
    if s["since_commit"] > budget:
        return "kill", f"exceeded op budget {budget:.0f}s"
    return "wait", f"within op budget {budget:.0f}s"


# ---------- C) CLAUDE CODE AGENT ----------
_PROMPT = """You are an optimization-run watchdog. Decide whether to KEEP WAITING or KILL this round.

Evidence:
- model/pipeline: {model}
- baseline profile duration (base): {base} s  (0 = no history yet, cold start)
- elapsed in this round: {elapsed} s
- time since last commit/kernel attempt: {since_commit} s
- device CPU jiffies consumed in the last sampling window: {device_cpu_delta}
- agent transcript growth in the last window: {transcript_growth_bytes} bytes
- tool currently in flight: {tool_in_flight}

KILL only if the round is genuinely stuck (nothing is progressing anywhere).
WAIT if it is doing legitimately slow work, even if it has been a long time.
Note: a host-side kernel compile consumes NO device CPU and may emit no transcript, yet is healthy.
A tiny trickle of device CPU with zero transcript growth can be a zombie, not progress.

Reply with ONLY one JSON object: {{"decision":"wait"|"kill","reason":"<12 words max>"}}"""


def decide_agent(s, model="claude-sonnet-4-6"):
    p = _PROMPT.format(**s)
    t0 = time.time()
    try:
        r = subprocess.run(
            ["claude", "-p", p, "--output-format", "text", "--model", model],
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = (r.stdout or "").strip()
        i, j = out.find("{"), out.rfind("}")
        d = json.loads(out[i : j + 1])
        return d.get("decision", "?"), d.get("reason", "")[:60], time.time() - t0
    except Exception as e:
        return "ERR", f"{type(e).__name__}", time.time() - t0
