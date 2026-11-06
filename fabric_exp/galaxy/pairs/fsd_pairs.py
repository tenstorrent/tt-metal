#!/usr/bin/env python3
"""fsd_pairs.py FSD OUT [PREFIX]: host pairs (both hosts starting with PREFIX, default bh-glx-110-a) with Ethernet connections between them, from the aggregated
factory system descriptor. Per pair: connection count (one per endpoint pair: a QSFP-DD cable carries 2 such
connections... counted as listed), and which (tray, asic, chan) on each side. Hosts are indexed in listing order
(host_id, proto3 default 0 omitted)."""
import collections, re, sys, json
F = sys.argv[1]
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "bh-glx-110-a"
text = open(F).read()
hosts = re.findall(r'hosts \{\s*hostname: "([^"]+)"', text)
start = text.index("eth_connections {")
conns = []
for m in re.finditer(r"connection \{\s*endpoint_a \{(.*?)\}\s*endpoint_b \{(.*?)\}\s*\}", text[start:], re.S):
    def ep(s):
        g = lambda k: int(re.search(k + r": (\d+)", s).group(1)) if re.search(k + r": (\d+)", s) else 0
        return (g("host_id"), g("tray_id"), g("asic_location"), g("chan_id"))
    conns.append((ep(m.group(1)), ep(m.group(2))))
print(f"hosts {len(hosts)}, connections {len(conns)}", file=sys.stderr)
pairs = collections.defaultdict(list)
for a, b in conns:
    ha, hb = hosts[a[0]], hosts[b[0]]
    if ha == hb:
        continue
    if not (ha.startswith(PREFIX) and hb.startswith(PREFIX)):
        continue
    if hb < ha:
        a, b, ha, hb = b, a, hb, ha
    pairs[(ha, hb)].append((a[1:], b[1:]))
json.dump({f"{k[0]}|{k[1]}": v for k, v in sorted(pairs.items())}, open(sys.argv[2], "w"))
for (ha, hb), v in sorted(pairs.items()):
    ta = collections.Counter((x[0][0], x[0][1]) for x in v)
    tb = collections.Counter((x[1][0], x[1][1]) for x in v)
    print(f"{ha} {hb} {len(v)} A trays/asics {dict(sorted(ta.items()))} B {dict(sorted(tb.items()))}")
