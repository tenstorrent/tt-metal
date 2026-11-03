#!/usr/bin/env python3
"""glxq_links.py: the Ethernet-link preflight of a multi-Galaxy job (MULTI_GALAXY_HARNESS.md).

  glxq_links.py expect FSD HOST,HOST,...            per host: the FSD's Ethernet endpoints whose connection has both
                                                    ends inside the allocation (cables to hosts outside it don't count)
  glxq_links.py check FSD HOST,HOST,... ETH_DIR     the same, against each host's read_eth_speed.py dump
                                                    (ETH_DIR/eth_<host>.txt): channels UP (port_status 1) and trained
                                                    (train_status 2). Exit 0 when every host has at least the expected
                                                    count, else 1.

A count, not a per-channel match: it catches a host whose links did not come up after the reset. The per-cable
comparison is the control plane's own, when the run gets --factory-system-descriptor (fsd_host_filter / ingest).
"""
import collections
import os
import re
import sys


def expected(fsd, hosts):
    t = open(fsd).read()
    names = re.findall(r'hosts \{\s*hostname: "([^"]+)"', t)
    idx = {h: i for i, h in enumerate(names)}
    missing = [h for h in hosts if h not in idx]
    if missing:
        sys.exit(f"hosts not in the FSD {fsd}: {missing}")
    want = {idx[h] for h in hosts}
    body = t[t.find("eth_connections"):]
    hid = lambda e: int((re.search(r"host_id: (\d+)", e) or [0, 0])[1])
    cnt = collections.Counter()
    for a, b in re.findall(r"connection \{\s*endpoint_a \{([^}]*)\}\s*endpoint_b \{([^}]*)\}", body):
        ha, hb = hid(a), hid(b)
        if ha in want and hb in want:
            cnt[ha] += 1
            cnt[hb] += 1
    return {h: cnt[idx[h]] for h in hosts}


def live(path):
    up = 0
    for line in open(path, errors="replace"):
        if re.search(r"port_status=1\b", line) and re.search(r"train_status=2\b", line):
            up += 1
    return up


def main():
    mode, fsd, hosts = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    exp = expected(fsd, hosts)
    if mode == "expect":
        for h in hosts:
            print(f"{h}\t{exp[h]}")
        return 0
    eth_dir, bad = sys.argv[4], []
    for h in hosts:
        p = os.path.join(eth_dir, f"eth_{h}.txt")
        got = live(p) if os.path.exists(p) else -1
        ok = got >= exp[h]
        bad += [] if ok else [h]
        print(f"   {h}: {got} channels UP/PASS, FSD expects {exp[h]} inside the allocation: {'ok' if ok else 'SHORT'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
