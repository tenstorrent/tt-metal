#!/usr/bin/env python3
"""fsd_subset.py FSD OUT HOST [HOST ...]: the FSD restricted to HOSTs, in the given order (host_id 0, 1, ...): their
hosts entries, board locations and every Ethernet connection with both ends among them (intra-host and between them).
Text-level: the aggregated FSD's blocks are regular (hosts {...}, board_types { board_locations {...} },
eth_connections { connection { endpoint_a {...} endpoint_b {...} } })."""
import re
import sys

src, out, keep = sys.argv[1], sys.argv[2], sys.argv[3:]
text = open(src).read()
host_blocks = re.findall(r"^hosts \{\n.*?^\}\n", text, re.S | re.M)
names = [re.search(r'hostname: "([^"]+)"', b).group(1) for b in host_blocks]
idx = {n: i for i, n in enumerate(names)}
missing = [h for h in keep if h not in idx]
assert not missing, missing
new_id = {idx[h]: i for i, h in enumerate(keep)}


def hid(block):
    m = re.search(r"host_id: (\d+)", block)
    return int(m.group(1)) if m else 0


def rehost(block, old):
    n = new_id[old]
    block = re.sub(r"\n(\s*)host_id: \d+\n", "\n", block)  # drop, then add when nonzero
    if n:
        block = re.sub(r"\{\n(\s*)", lambda m: "{\n" + m.group(1) + f"host_id: {n}\n" + m.group(1), block, count=1)
    return block


parts = [host_blocks[idx[h]] for h in keep]
bt = text[text.index("board_types {"):text.index("eth_connections {")]
locs = re.findall(r"  board_locations \{\n.*?\n  \}\n", bt, re.S)
parts.append("board_types {\n" + "".join(rehost(b, hid(b)) for b in locs if hid(b) in new_id) + "}\n")
ec = text[text.index("eth_connections {"):]
conns = re.findall(r"  connection \{\n    endpoint_a \{\n.*?\n    \}\n    endpoint_b \{\n.*?\n    \}\n  \}\n", ec, re.S)
kept = []
for c in conns:
    a, b = re.findall(r"endpoint_[ab] \{\n(.*?)\n    \}", c, re.S)
    ha, hb = hid(a), hid(b)
    if ha in new_id and hb in new_id:
        na = rehost("endpoint_a {\n" + a + "\n    }", ha)
        nb = rehost("endpoint_b {\n" + b + "\n    }", hb)
        kept.append("  connection {\n    " + na + "\n    " + nb + "\n  }\n")
parts.append("eth_connections {\n" + "".join(kept) + "}\n")
open(out, "w").write("".join(parts))
print(f"{out}: hosts {keep}, board locations {sum(1 for b in locs if hid(b) in new_id)}, connections {len(kept)}")
