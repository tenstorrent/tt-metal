"""Per-stage wall clock and token cost for the Qwen3.6-27B bring-up.

Method matches run-cost-analysis/extract_run_costs.py so the numbers are comparable to the
fleet table: start = first `ts` in the stage jsonl, end = last `ts`, tokens = the LAST
`thread/tokenUsage/updated` event, which carries the cumulative total for the stage thread.
"""

import glob
import json
import os
import re
from datetime import datetime

ROOT = os.path.expanduser("~/_fmf-qwen-logs")
STAGE_NAMES = {
    "4": "04 multichip-decoder", "5": "05 optimized-multichip-decoder",
    "6": "06 full-model", "7": "07 optimized-full-model",
    "8": "08 datatype-sweep", "9": "09 vllm",
    "10": "10 optimized-vllm", "11": "11 tti-release",
}


def pt(ts):
    return datetime.strptime(ts, "%Y%m%dT%H%M%SZ")


def scan(path):
    first = last = None
    usage = None
    with open(path, "rb") as f:
        for line in f:
            m = re.match(rb'\{"ts": "([0-9TZ]+)"', line)
            if m:
                ts = m.group(1).decode()
                if first is None:
                    first = ts
                last = ts
            if b"thread/tokenUsage/updated" in line:
                try:
                    usage = json.loads(line)["message"]["params"]["tokenUsage"]["total"]
                except Exception:
                    pass
    return first, last, usage


rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "stage*")),
                key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
    num = re.sub(r"\D", "", os.path.basename(d))
    mains = [p for p in glob.glob(os.path.join(d, "*.jsonl")) if "initialize" not in p]
    inits = [p for p in glob.glob(os.path.join(d, "*.jsonl")) if "initialize" in p]
    if not mains:
        continue
    path = mains[0]
    first, last, usage = scan(path)
    # include the initialize thread in wall clock if it starts earlier
    for ip in inits:
        f2, l2, _ = scan(ip)
        if f2 and (first is None or f2 < first):
            first = f2
    dur = (pt(last) - pt(first)).total_seconds() if (first and last) else None
    rows.append({
        "stage": STAGE_NAMES.get(num, num),
        "file": os.path.basename(path),
        "start": first, "end": last,
        "hours": round(dur / 3600, 2) if dur else None,
        "tok_total": (usage or {}).get("totalTokens"),
        "tok_in": (usage or {}).get("inputTokens"),
        "tok_cached": (usage or {}).get("cachedInputTokens"),
        "tok_out": (usage or {}).get("outputTokens"),
        "tok_reason": (usage or {}).get("reasoningOutputTokens"),
        "log_mb": round(os.path.getsize(path) / 1e6, 1),
    })

print("%-34s %-15s %-15s %7s %12s %12s %10s" %
      ("stage", "start (UTC)", "end (UTC)", "hours", "total tok", "out tok", "log MB"))
tot_h = 0.0
tot_t = 0
for r in rows:
    tot_h += r["hours"] or 0
    tot_t += r["tok_total"] or 0
    print("%-34s %-15s %-15s %7s %12s %12s %10s" % (
        r["stage"], (r["start"] or "?")[:15], (r["end"] or "?")[:15],
        r["hours"], f'{r["tok_total"]:,}' if r["tok_total"] else "-",
        f'{r["tok_out"]:,}' if r["tok_out"] else "-", r["log_mb"]))
print("%-34s %-15s %-15s %7.2f %12s" % ("TOTAL (stages 4-11)", "", "", tot_h, f"{tot_t:,}"))

# blocked / restarted threads, recorded separately
for extra in ("_stage6-blocked-thread", "_stage9-blocked-thread"):
    p = os.path.join(ROOT, extra)
    for f in glob.glob(os.path.join(p, "*.jsonl")):
        if "initialize" in f:
            continue
        first, last, usage = scan(f)
        if first and last:
            dur = (pt(last) - pt(first)).total_seconds()
            print("  [abandoned] %-28s %-15s %5.2f h  tok=%s" % (
                extra, first[:15], dur / 3600,
                f'{(usage or {}).get("totalTokens"):,}' if usage and usage.get("totalTokens") else "-"))

with open(os.path.expanduser("~/_fmf-qwen-logs/stage_times.json"), "w") as f:
    json.dump(rows, f, indent=2)
print("\nwrote ~/_fmf-qwen-logs/stage_times.json")
