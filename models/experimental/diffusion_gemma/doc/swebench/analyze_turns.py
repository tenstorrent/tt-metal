"""Test whether DiffusionGemma's agentic failures are context-length-driven degeneracy.

For every assistant turn in a run (successful action turns and format-error turns alike),
record the turn index, the prompt token count, and a repetition metric. Then compare the
distributions for turns that produced a valid action vs turns that failed.

Repetition metric: fraction of 8-grams that are duplicates (0 = no repetition).
"""

import collections
import glob
import json
import re
import sys

ACTION_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)


def rep8(text: str) -> float:
    w = text.split()
    if len(w) < 16:
        return 0.0
    grams = [" ".join(w[i:i + 8]) for i in range(len(w) - 7)]
    return 1.0 - len(set(grams)) / len(grams)


def bucket(n, edges):
    for e in edges:
        if n < e:
            return "<%d" % e
    return ">=%d" % edges[-1]


run = sys.argv[1]
rows = []
for f in sorted(glob.glob(run + "/*/*.traj.json")):
    d = json.load(open(f))
    inst = d["instance_id"]
    status = d["info"].get("exit_status", "?")
    turn = 0
    for m in d["messages"]:
        raw = None
        if m["role"] == "assistant":
            raw = (m.get("extra") or {}).get("response")
            ok = True
        elif m["role"] == "user" and ("Format error" in str(m.get("content")) or "output token limit" in str(m.get("content"))):
            raw = (m.get("extra") or {}).get("response")
            ok = False
        else:
            continue
        if not isinstance(raw, dict):
            continue
        turn += 1
        msg = raw["choices"][0]["message"]
        text = (msg.get("content") or "") or (msg.get("reasoning") or msg.get("reasoning_content") or "")
        usage = raw.get("usage") or {}
        rows.append({
            "inst": inst, "status": status, "turn": turn, "ok": ok,
            "prompt_tokens": usage.get("prompt_tokens") or 0,
            "rep8": rep8(text), "chars": len(text),
        })

print("assistant turns analysed:", len(rows))
okr = [r for r in rows if r["ok"]]
bad = [r for r in rows if not r["ok"]]


def stats(rs, key):
    v = sorted(r[key] for r in rs)
    if not v:
        return "n/a"
    return "median %.3f  p90 %.3f  max %.3f" % (v[len(v) // 2], v[int(len(v) * 0.9)], v[-1])


print("\n                      valid-action turns (n=%d)      failed turns (n=%d)" % (len(okr), len(bad)))
for key in ("turn", "prompt_tokens", "rep8", "chars"):
    print("  %-14s %-32s %s" % (key, stats(okr, key), stats(bad, key)))

print("\nfailure rate by turn index:")
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    b = bucket(r["turn"], [5, 10, 20, 40, 80])
    by[b][0] += 1
    by[b][1] += 0 if r["ok"] else 1
for b in ["<5", "<10", "<20", "<40", "<80", ">=80"]:
    if b in by:
        tot, f = by[b]
        print("  turn %-6s n=%-5d failed %5.1f%%" % (b, tot, 100.0 * f / tot))

print("\nfailure rate by prompt tokens:")
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    b = bucket(r["prompt_tokens"], [4000, 8000, 16000, 32000, 64000])
    by[b][0] += 1
    by[b][1] += 0 if r["ok"] else 1
for b in ["<4000", "<8000", "<16000", "<32000", "<64000", ">=64000"]:
    if b in by:
        tot, f = by[b]
        print("  prompt %-9s n=%-5d failed %5.1f%%" % (b, tot, 100.0 * f / tot))

print("\nhigh-repetition turns (rep8 > 0.3): %d of %d (%.1f%%)  -- of which failed: %.1f%%" % (
    sum(1 for r in rows if r["rep8"] > 0.3), len(rows),
    100.0 * sum(1 for r in rows if r["rep8"] > 0.3) / len(rows),
    100.0 * sum(1 for r in rows if r["rep8"] > 0.3 and not r["ok"]) / max(sum(1 for r in rows if r["rep8"] > 0.3), 1)))

print("\nper-instance: turn of first failure vs total turns (Submitted first)")
per = collections.defaultdict(list)
for r in rows:
    per[(r["status"], r["inst"])].append(r)
for (status, inst), rs in sorted(per.items())[:16]:
    first_bad = next((r["turn"] for r in rs if not r["ok"]), None)
    print("  %-22s %-34s turns=%-4d first_fail=%s" % (status, inst, len(rs), first_bad))
