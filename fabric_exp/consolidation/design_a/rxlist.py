#!/usr/bin/env python3
"""rxlist.py ELF [--fn=KEY] [--chain]: the receiver loop function's instructions with source lines (innermost frame),
marking sp accesses ("S") and instructions inside the loop region ("L"; rxcmp.loop_region). KEY picks the function by a
symbol substring (default: receiver3run, else receiver_pass). --chain prints the inline chain's function names."""
import os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import rxcmp
import pcs_sections as ps

elf = sys.argv[1]
opts = dict((x[2:].split("=", 1) + [""])[:2] for x in sys.argv[2:] if x.startswith("--"))
fns = rxcmp.functions(elf)
key = opts.get("fn")
body = None
for name, b in fns.items():
    if (key and key in name) or (not key and any(k in name for k in rxcmp.RX_KEYS)):
        body = b
        break
loop = rxcmp.loop_region(body)
fs = rxcmp.frame_saves(body)
loc = ps.a2l(elf, [a for a, *_ in body])
short = lambda f: re.sub(r"\(.*", "", re.sub(r"<[^<>]*(<[^<>]*>[^<>]*)*>", "", f)).split("::")[-1]
for a, op, arg, raw in body:
    fr = loc.get(a, [])
    src = f"{fr[0][1]}:{fr[0][2]}" if fr else "?"
    mark = ("L" if a in loop else " ") + ("S" if "(sp)" in raw and a not in fs else " ")
    chain = " < ".join(short(f) for f, _, _ in fr) if "chain" in opts else ""
    print(f"{a:06x} {mark} {op:7s} {raw:34s} {src:32s} {chain}")
