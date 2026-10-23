# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Galaxy-projected whole-head time from a LoudBox (2x4, TP line of 4) fused-head profile.

usage: python galaxy_projection.py <profile_log_device.csv> glm|k3 [logits_first 0|1 - default: the model's current]
       python galaxy_projection.py <profile_log_device.csv> skew 7,6,4,5/3,2,0,1  (each mesh row's devices by TP
       position, test_head_se3's SKEW_MAP log): the launch skew from the ports' ready handshake (SE-RDYWAIT)

The LoudBox's TP axis is a line: the x all-gather and the SE reduce-scatter each bind on an end link carrying 3 units
(u: one shard / one owner block - 640 rows a mesh row; GLM 1.31 MB, K3 2.29 MB at 2 links x 23.6 GB/s). The Galaxy's
TP axis is a ring of 4: both balance to 1.5u a link direction. Per device (zone times from the head's last run):
  SE path  = max(gate/up end - 1.5u x E_GATHER, GU_ALONE) + (down + RS end - gate/up end) - 1.5u
             (the gather shrinks by the ring's 1.5u at the gather's measured efficiency, but gate/up never finishes
             before its compute alone; the RS is link-bound on the end link at link rate: it shrinks by 1.5u)
  chain    = planner end - (K3, x-first: its logits leave after the gather) the gather's shrink + CHAIN_GROWTH
             (the routing exchange + planner at N = 32 on the 8x4: see below); GLM's logits go first: no shift
  head     = max over devices of max(SE path, chain)
LATENT (K3, when the head runs it - SE-LHAND): the ports send the latent all-to-all before the RS, so it is in the
SE path's (down + RS end - gate/up end). The line's chip at TP position p sends max(p, G - 1 - p) latent blocks one way
(3 at the ends), the ring of 4 at most 2: the SE path is also credited (that - 2) blocks at link rate.
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
# (gate / up's end with no gather wait, from a chip's start - the router phase on the engine included when it runs
# there: the head's own shapes, SE_PROBE=NOXWAIT, the slowest chip; keyed by the router's place)
GU_ALONE = {("glm", True): 82.0, ("glm", False): 73.0, ("k3", True): 307.0, ("k3", False): 272.0}
CHAIN_GROWTH = {"glm": 20.0, "k3": 20.0}
LOGITS_AFTER_GATHER = {"glm": False, "k3": False}
FREQ = 1350.0


# ---- Galaxy floors from first principles (a chip of the 8x4: TP ring of 4, 640 rows a mesh row, 160 owned) ----
LINK = 2 * 23.6e3  # (B/us a direction of a TP neighbour: 2 links)
PEAK_HIFI2 = 110 * 1024 * 2 * 1350  # (FLOP/us: 110 engine cores, 2048 MAC/cycle LoFi halved at HiFi2)
DIMS = {  # rows a mesh row, H, the chip's I slice x 2 (gate / up), latent (K3), experts, K
    "glm": dict(rows=640, H=4096, N=1024, I=512, lat=0, E=288, K=8),
    "k3": dict(rows=640, H=7168, N=3072, I=1536, lat=3584, E=448, K=16),
}
# the routing chain's per-step minimums (us): router matmul on the 10 mover-column cores at 75% of their HiFi3 peak;
# top-K (SFPU, measured after the K-run folds); joint rows / route words (4 RISCs); planner before / after the
# exchange (measured phases with the N = 32 growth, 10 RISCs); exchange path (watcher + Y transit + 2 X hops)
CHAIN_STEPS = {
    "glm": dict(topk=14, rows=5, plan_pre=5, path=10, plan_post=15),
    "k3": dict(topk=20, rows=8, plan_pre=6, path=10, plan_post=20),
}


def floors(model):
    d, G = DIMS[model], 4
    shard = d["rows"] * d["H"] // G * 2  # (an x shard / an RS block, bf16)
    owned = d["rows"] // G
    logit = owned * d["E"] * 4  # (one owner's partial logits, fp32)
    latb = owned * d["lat"] // G * 2  # (one latent block, bf16)
    # ring of 4 a direction: gather 1.5 shards, RS 1.5 blocks, each all-to-all (logits, latent) 2 blocks
    links = (1.5 * shard + 1.5 * shard + 2 * logit + 2 * latb) / LINK
    flops = 2 * d["rows"] * d["H"] * d["N"] + 2 * d["rows"] * d["I"] * d["H"] + 2 * d["rows"] * d["H"] * d["lat"] // G
    compute = flops / PEAK_HIFI2
    router = 2 * d["rows"] * d["H"] // G * d["E"] / (10 * 1024 * 2 * 1350 * 2 / 3 * 0.75)  # (HiFi3, 10 cores)
    st = CHAIN_STEPS[model]
    chain = router + 2 * logit / LINK + 1.3 + st["topk"] + st["rows"] + st["plan_pre"] + st["path"] + st["plan_post"]
    return dict(links=links, compute=compute, chain=chain, head=max(links, compute, chain))


def latent_block_us(model):
    """One latent block (an owner's rows of the chip's latent slice, bf16) at link rate."""
    d, G = DIMS[model], 4
    return d["rows"] // G * d["lat"] // G * 2 / LINK


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


def project(path, model, logits_first=None, offsets=None, rows=None):
    """Per device (dev, LoudBox end, planner end, Galaxy SE path, Galaxy chain). offsets {dev: us from its row's last
    launch, <= 0}: every time is then from the row's last launch (skew-corrected), else from the device's own start
    (eager); a chip's compute still never finishes before its own start + its compute alone. rows ([[dev by TP
    position] a mesh row]): the latent's line-vs-ring credit by position (module docstring)."""
    u = U[model]
    pos = {d: i for row in rows or [] for i, d in enumerate(row)}
    after = not logits_first if logits_first is not None else LOGITS_AFTER_GATHER[model]
    zs = zones(path)
    on_engine = any("SE-RTRLD" in e for e in zs.values())  # (the router's matmul ran on the engine)
    latent = any("SE-LHAND" in e for e in zs.values())
    gu_alone = GU_ALONE[(model, on_engine)]
    res = []
    for dev, e in zs.items():
        sh = offsets.get(int(dev), 0.0) if offsets else 0.0
        gu, ds, rp = e["SE-GU"] + sh, e["SE-DOWNRS"] + sh, e.get("SE-RPLAN", 0.0) + sh
        end = max(v for k, v in e.items() if k.endswith("-KERNEL")) + sh
        front = max(gu - 1.5 * u * E_GATHER, gu_alone + sh)
        se = front + (ds - gu) - 1.5 * u
        if latent and int(dev) in pos:
            p = pos[int(dev)]
            se -= max(0, max(p, 3 - p) - 2) * latent_block_us(model)
        shift = (gu - front) if after else 0.0
        chain = rp - shift + CHAIN_GROWTH[model]
        res.append((dev, end, rp, se, chain))
    return res


def yhello(path):
    """{dev: when the other row's watcher's "started" landed here, us from this chip's start (SE-YHELLO)}."""
    rows_ = list(csv.reader(open(path)))[2:]
    out = {}
    for dev in sorted({r[0].strip() for r in rows_}, key=int):
        rr = [r for r in rows_ if r[0].strip() == dev]
        last = max(int(r[7]) for r in rr if r[10].strip().startswith("SE-"))
        rr = [r for r in rr if int(r[7]) == last]
        t0 = min(int(r[5]) for r in rr if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START")
        for r in rr:
            if r[10].strip() == "SE-YHELLO":
                out[int(dev)] = (int(r[5]) - t0) / FREQ
    return out


def global_offsets(path, rows):
    """{dev: its launch offset from the mesh's last launch (<= 0)}: rows aligned to each other by the watchers' Y
    "started" (per column: the chip that sees the other row's later waited for it; the earliest sighting ~ the
    send + a hop when the other row launched first), averaged over the columns; None without the stamps."""
    hy = yhello(path)
    if len(rows) < 2 or any(d not in hy for row in rows for d in row):
        return None
    within = launch_skew(path, rows)
    c = min(hy.values())
    rel = [0.0]  # (row k's start - row 0's start, its earliest chip's)
    for k in range(1, len(rows)):
        ds = []
        for a, b, oa, ob in zip(rows[0], rows[k], within[0], within[k]):
            ta, tb = hy[a] - c, hy[b] - c
            dba = ta if ta > tb else -tb  # (b's start - a's start)
            ds.append(dba - ob + oa)
        rel.append(sum(ds) / len(ds))
    st = {d: rel[k] + o for k, row in enumerate(rows) for d, o in zip(row, within[k])}
    m = max(st.values())
    return {d: v - m for d, v in st.items()}


def offsets_of(path, rows):
    """{dev: its launch offset from its row's last launch (<= 0)}."""
    out = {}
    for devs, st in zip(rows, launch_skew(path, rows)):
        for d, o in zip(devs, st):
            out[d] = o - max(st)
    return out


def launch_skew(path, rows):
    """Each mesh row's chips' start offsets (us, from the row's earliest): the ports' ready handshake - a port facing
    chip q waits for q's ready (SE-RDYWAIT); its wait ends ~C (reaching the wait) after its own start if q started
    first, else (q's start - its start) + C. Ports 2 / 3 face the next position (fwd), 4 / 5 the previous (bwd)."""
    rows_ = list(csv.reader(open(path)))[2:]
    ends = collections.defaultdict(dict)
    for dev in sorted({r[0].strip() for r in rows_}, key=int):
        rr = [r for r in rows_ if r[0].strip() == dev]
        last = max(int(r[7]) for r in rr if r[10].strip().startswith("SE-"))
        rr = [r for r in rr if int(r[7]) == last]
        t0 = min(int(r[5]) for r in rr if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START")
        for r in rr:
            if r[10].strip() == "SE-RDYWAIT" and r[11].strip() == "ZONE_END":
                ends[int(dev)][int(r[2])] = (int(r[5]) - t0) / FREQ
    out = []
    for devs in rows:
        c = min(min(e.values()) for d, e in ends.items() if d in devs)  # (the shortest wait: no skew, a hop)
        st = [0.0]
        for a, b in zip(devs, devs[1:]):
            fa = min(ends[a].get(2, 1e9), ends[a].get(3, 1e9)) - c  # (a waits on b: b's start - a's start, if > 0)
            fb = min(ends[b].get(4, 1e9), ends[b].get(5, 1e9)) - c  # (b waits on a)
            st.append(st[-1] + (fa if fa > fb else -fb))
        m = min(st)
        out.append([round(x - m, 1) for x in st])
    return out


if __name__ == "__main__":
    if sys.argv[2] == "skew":
        for devs, st in zip(
            [list(map(int, g.split(","))) for g in sys.argv[3].split("/")],
            launch_skew(sys.argv[1], [list(map(int, g.split(","))) for g in sys.argv[3].split("/")]),
        ):
            print(f"row {devs}: start offsets {st} us (spread {max(st):.1f})")
        sys.exit(0)
    import json
    import os

    lf = bool(int(sys.argv[3])) if len(sys.argv) > 3 else None
    mp = os.path.join(os.path.dirname(sys.argv[1]), "head_map.json")  # (test_head_se3: [[dev by TP position] a row])
    rows = json.load(open(mp)) if os.path.exists(mp) else None
    f = floors(sys.argv[2])
    cases = [
        ("from the mesh's last launch", global_offsets(sys.argv[1], rows) if rows else None),
        ("from each row's last launch", offsets_of(sys.argv[1], rows) if rows else None),
        ("eager", None),
    ]
    for name, off in cases:
        if name != "eager" and off is None:
            continue
        res = project(sys.argv[1], sys.argv[2], lf, off, rows)
        lb = max(end for _, end, *_ in res)
        head = max(max(se, ch) for *_, se, ch in res)
        print(
            f"{name}: LoudBox {lb:.0f} us | Galaxy-projected {head:.0f} us (SE path {max(se for *_, se, _ in res):.0f}, "
            f"routing chain {max(ch for *_, ch in res):.0f}) = {head / f['head']:.2f}x the Galaxy floor {f['head']:.0f}"
        )
    if rows and any("SE-LHAND" in e for e in zones(sys.argv[1]).values()):
        print(
            f"(SE path credited the latent's line-vs-ring: {latent_block_us(sys.argv[2]):.1f} us on the line's end chips)"
        )
    print(f"Galaxy floors: links {f['links']:.0f} us, compute {f['compute']:.0f}, routing chain {f['chain']:.0f}")
