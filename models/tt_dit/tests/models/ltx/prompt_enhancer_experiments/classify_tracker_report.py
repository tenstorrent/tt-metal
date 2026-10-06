"""Classify every buffer the report-only tracker flagged by the code that allocated it."""

import json
import re
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "tracker_report.json"
rep = json.load(open(path))
print(
    f"replays checked={rep['replays']['checked']} flagged={rep['replays']['flagged']} unique unsafe={rep['unique_unsafe_buffers']}"
)

ENHANCER = ("prompt_enhancer.py", "models/demos/gemma4", "models/tt_transformers")
LTX = (
    "models/tt_dit/pipelines/ltx",
    "models/tt_dit/utils/tracing.py",
    "models/tt_dit/models",
    "models/tt_dit/encoders",
    "models/tt_dit/utils",
)

rows, first_seen = {}, {}
for rec in rep["records"]:
    text = rec["report"]
    for m in re.finditer(
        r"Buffer (\d+) \[op: ([^\]\n]*)\](.*?)(?=\nBuffer \d+ \[op:|\n--- Python referrer|\nUse ttnn\.tools|\Z)",
        text,
        re.S,
    ):
        bid, op, body = m.group(1), m.group(2), m.group(3)
        if bid in rows:
            continue
        frames = re.findall(r'File "([^"]+)", line (\d+), in (\w+)\n\s*(.*)', body)
        ours = [(f.split("/tt-metal/")[-1], l, fn, c.strip()) for f, l, fn, c in frames if "/models/" in f]
        site = " <- ".join(f"{f}:{l}" for f, l, _, _ in ours[-3:]) or "(no models/ frame)"
        code = ours[-1][3][:90] if ours else ""
        owner = (
            "enhancer"
            if any(k in f for f, *_ in ours for k in ENHANCER)
            else "ltx"
            if any(k in f for f, *_ in ours for k in LTX)
            else "other"
        )
        rows[bid] = (owner, op[:60], site, code)
        first_seen[bid] = rec["trace_id"]

by_owner = Counter(o for o, *_ in rows.values())
print("owners:", dict(by_owner))
for bid, (owner, op, site, code) in sorted(rows.items(), key=lambda kv: (kv[1][0], kv[1][2])):
    print(f"[{owner:8}] {bid:>9} first@trace {first_seen[bid]:>3} | {op}\n           {site}\n           {code}")
