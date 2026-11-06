#!/usr/bin/env python3
"""cap_ring_packets.py [--apply] YAML...: cap a data-checked test's packets per flow at its receiver ring's slots.

Proposed fix (2026-11-04, not applied while jobs read these yamls) for a latent test-harness race: test_tt_fabric's
receiver validates packets from a ring of default_payload_chunk_size bytes (1 MiB / max_configs_per_core when the yaml
sets allocation_policies; 256 KiB at max_configs_per_core 4), one slot per packet. Without enable_flow_control the
sender wraps that ring without waiting, so a flow of more packets than slots can overwrite packets its receiver has not
validated yet; across MPI ranks the start sync is per mesh, so one rank's senders can run ahead of the other rank's
receivers (the port's VC1+VC2 "hang", broker 660: 100 x 4096 B into 64 slots). Capping num_packets at the slots
removes the overwrite whatever the start skew.

For each test with benchmark_mode off and enable_flow_control off, slots = chunk // its largest size; num_packets
above that is lowered to it (in place with --apply, keeping the text; dry run by default). Tests the yaml leaves to
the dynamic policy (no allocation_policies) are reported, not changed.
"""
import re
import sys

import yaml

L1 = 0x100000


def as_list(v):
    return v if isinstance(v, list) else [v]


def main():
    args = sys.argv[1:]
    apply = "--apply" in args
    paths = [a for a in args if a != "--apply"]
    if not paths:
        sys.exit(__doc__)
    for path in paths:
        text = open(path).read()
        d = yaml.safe_load(text)
        pol = d.get("allocation_policies") or {}
        if "default_payload_chunk_size" in pol:
            chunk = pol["default_payload_chunk_size"]
        elif pol:
            chunk = L1 // (pol.get("receiver") or {}).get("max_configs_per_core", 1)
        else:
            print(f"{path}: dynamic allocation policy, not changed")
            continue
        caps = {}
        for t in d.get("Tests", []):
            if t.get("benchmark_mode", False) or t.get("enable_flow_control", False):
                continue
            defaults = t.get("defaults") or {}
            sizes = [s for s in as_list((t.get("parametrization_params") or {}).get("size", defaults.get("size", 0)))
                     if isinstance(s, int) and s > 0]
            n = defaults.get("num_packets")
            if sizes and isinstance(n, int) and n > chunk // max(sizes):
                caps[t["name"]] = (n, chunk // max(sizes), max(sizes))
        if not caps:
            print(f"{path}: nothing to cap (chunk {chunk} B)")
            continue
        out, cur = [], None
        for line in text.splitlines(keepends=True):
            m = re.match(r'^\s*- name:\s*"?([^"\n]+)"?\s*$', line)
            if m:
                cur = m.group(1).strip()
            m = re.match(r"^(\s*num_packets:\s*)(\d+)(.*)$", line, re.S)
            if m and cur in caps and int(m.group(2)) == caps[cur][0]:
                line = f"{m.group(1)}{caps[cur][1]}{m.group(3)}"
            out.append(line)
        for name, (n, slots, size) in caps.items():
            print(f"{path}: {name}: num_packets {n} -> {slots} ({chunk} B ring / {size} B = {slots} slots)")
        if apply:
            open(path, "w").write("".join(out))


if __name__ == "__main__":
    main()
