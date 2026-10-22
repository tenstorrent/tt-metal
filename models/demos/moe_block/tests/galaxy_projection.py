# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Galaxy-projected whole-head time from a LoudBox (2x4, TP line of 4) fused-head profile.

usage: python galaxy_projection.py <profile_log_device.csv> glm|k3

The LoudBox's TP axis is a line: the x all-gather and the SE reduce-scatter each bind on an end link carrying 3 units
(u: one shard / one owner block - 640 rows a mesh row; GLM 1.31 MB, K3 2.29 MB at 2 links x 23.6 GB/s). The Galaxy's
TP axis is a ring of 4: both balance to 1.5u a link direction. Per device (zone times from the head's last run):
  SE path  = max(gate/up end - 1.5u x E_GATHER, GU_ALONE) + (down + RS end - gate/up end) - 1.5u
             (the gather shrinks by the ring's 1.5u at the gather's measured efficiency, but gate/up never finishes
             before its compute alone; the RS is link-bound on the end link at link rate: it shrinks by 1.5u)
  chain    = planner end - (K3, x-first: its logits leave after the gather) the gather's shrink + CHAIN_GROWTH
             (the routing exchange + planner at N = 32 on the 8x4: see below); GLM's logits go first: no shift
  head     = max over devices of max(SE path, chain)
CHAIN_GROWTH (modelled, not measured): planner pre-exchange +4 us (3 N NoC writes a source), post-arrival +7 (N
header / entry reads, N x M source loops; fewer experts an instance), the slowest of 8 TP groups' chains instead of 2
(+9, ~10 us per-group spread, the same launch skew per chip as the LoudBox's); paths: Y in transit (<= 4 hops x 650 ns)
+ <= 2 X store-and-forward hops (29 forwards a chip, <= 15 a port) ~ the LoudBox's. Not credited: the logits' shorter
transit on the ring (conservative)."""

import collections
import csv
import sys

U = {"glm": 28.0, "k3": 49.0}  # (us: one unit at 47 GB/s)
E_GATHER = 1.3  # (the fused gather's time over its link floor, measured on the line)
GU_ALONE = {"glm": 60.0, "k3": 250.0}  # (gate/up (+ K3's latent) compute alone: engine-alone / fused minimum)
CHAIN_GROWTH = {"glm": 20.0, "k3": 20.0}
LOGITS_AFTER_GATHER = {"glm": False, "k3": True}
FREQ = 1350.0


def zones(path):
    rows = list(csv.reader(open(path)))[2:]
    out = {}
    for dev in sorted({r[0].strip() for r in rows}, key=int):
        rr = [r for r in rows if r[0].strip() == dev]
        last = max(int(r[7]) for r in rr if r[10].strip().startswith("SE-"))
        rr = [r for r in rr if int(r[7]) == last]
        t0 = min(int(r[5]) for r in rr if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START")
        end = collections.defaultdict(float)
        for r in rr:
            if r[11].strip() == "ZONE_END":
                n = r[10].strip()
                end[n] = max(end[n], (int(r[5]) - t0) / FREQ)
        out[dev] = end
    return out


def project(path, model):
    u = U[model]
    res = []
    for dev, e in zones(path).items():
        gu, ds = e["SE-GU"], e["SE-DOWNRS"]
        front = max(gu - 1.5 * u * E_GATHER, GU_ALONE[model])
        se = front + (ds - gu) - 1.5 * u
        shift = (gu - front) if LOGITS_AFTER_GATHER[model] else 0.0
        chain = e.get("SE-RPLAN", 0.0) - shift + CHAIN_GROWTH[model]
        res.append((dev, ds, e.get("SE-RPLAN", 0.0), se, chain))
    return res


if __name__ == "__main__":
    res = project(sys.argv[1], sys.argv[2])
    for dev, ds, rp, se, ch in res:
        print(f"dev {dev}: LoudBox SE end {ds:6.1f} planner end {rp:6.1f} | Galaxy SE {se:6.1f} chain {ch:6.1f}")
    print(
        f"Galaxy-projected head: {max(max(se, ch) for *_, se, ch in res):.0f} us "
        f"(SE path {max(se for *_, se, _ in res):.0f}, routing chain {max(ch for *_, ch in res):.0f})"
    )
