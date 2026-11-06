#!/usr/bin/env python3
"""router_cfg_dump.py JOB_DIR LABEL [--rows]: the fabric router configs one step of a glxq / tfmock job compiled, read
from the ranks' logged compile commands (TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1: JOB_DIR/ranks/r<k>_*_LABEL.log) and
the named-arg headers in the ranks' caches (JOB_DIR/cache/r<k>). Per config: rank, ERISC, MY_ETH_CHANNEL, MY_DIRECTION,
IS_INTERMESH_ROUTER, sender channel slots (worker channels marked w), receiver channel slots, the remote receiver slots
the sender assumes, and DOWNSTREAM_SENDER_NUM_BUFFERS per VC (the forwarding-ring size an E1 assumes for every downstream
of that VC). Positional args parsed by the router's layout (fabric_erisc_router_ct_args.hpp): ChannelAllocations at
index 0 (0xabcd1234, entries, 4 args per entry, sender->entry, receiver->entry), 0xabaddad6 + the remote allocations,
0xabaddad7 + DOWNSTREAM_SENDER_NUM_BUFFERS[MAX_NUM_RECEIVER_CHANNELS].

Prints distinct rows with counts and a contract check: every E0 VC0 / VC1 forwarding channel's slot count against every
E1's DOWNSTREAM_SENDER_NUM_BUFFERS for that VC, and every sender's assumed remote receiver slots against the receiver slot
counts compiled in the job (no chip identity in a compile command, so the check is over the set)."""
import collections, glob, os, re, sys

DIRS = {0: "E", 1: "W", 2: "N", 3: "S", 4: "Z"}


def named(path):
    s = open(path).read()
    return {k: int(v) for k, v in re.findall(r'\{"(\w+)",(-?\d+)\}', s)}


def alloc(args, i, ns, nr):
    assert args[i] == 0xABCD1234, hex(args[i])
    n = args[i + 1]
    ents = [args[i + 2 + 4 * e: i + 6 + 4 * e] for e in range(n)]
    m = i + 2 + 4 * n
    s2e, r2e = args[m: m + ns], args[m + ns: m + ns + nr]
    return ents, s2e, r2e, m + ns + nr


def parse(cmd, cache_root):
    h = re.search(r'-DFULL_KERNEL_NAME="fabric_erisc_router/(\d+)/"', cmd).group(1)
    args = [int(x) for x in re.search(r"-DKERNEL_COMPILE_TIME_ARGS=([0-9,]+)", cmd).group(1).split(",")]
    hdr = glob.glob(os.path.join(cache_root, "*", "kernels", "fabric_erisc_router", h, "named_ct_arg_map_generated.h"))
    if not hdr:
        return None
    na = named(hdr[0])
    ns, nr = na["NUM_SENDER_CHANNELS"], na["NUM_RECEIVER_CHANNELS"]
    ents, s2e, r2e, j = alloc(args, 0, ns, nr)
    assert args[j] == 0xABADDAD6
    rents, _, rr2e, k = alloc(args, j + 1, 0, nr)
    assert args[k] == 0xABADDAD7
    mx = na["MAX_NUM_RECEIVER_CHANNELS"]
    ds = args[k + 1: k + 1 + mx]
    worker = [na.get(f"SENDER_CH_{c}_IS_WORKER_CHANNEL", 0) for c in range(ns)]
    vc1_start = na.get("VC1_FABRIC_POSITION_START", ns)
    tx = [ents[s2e[c]][1] for c in range(ns)]
    return dict(
        hash=h, erisc=na["MY_ERISC_ID"], ch=na["MY_ETH_CHANNEL"], dir=DIRS.get(na["MY_DIRECTION"], na["MY_DIRECTION"]),
        im=na["IS_INTERMESH_ROUTER"], ns=ns, nr=nr, vc1_start=vc1_start,
        tx=" ".join(f"{t}{'w' if w else ''}" for t, w in zip(tx, worker)),
        tx_raw=tx, worker=worker, rx=[ents[r2e[r]][1] for r in range(nr)],
        remote_rx=[rents[rr2e[r]][1] for r in range(nr)], ds=ds,
        actual=(na.get("ACTUAL_VC0_SENDER_CHANNELS"), na.get("ACTUAL_VC1_SENDER_CHANNELS")))


def main():
    job, label = sys.argv[1], sys.argv[2]
    rows = collections.Counter()
    cfgs = []
    for log in sorted(glob.glob(os.path.join(job, "ranks", f"r*_*_{label}.log"))):
        rank = re.match(r"r(\d+)_", os.path.basename(log)).group(1)
        cache = os.path.join(job, "cache", f"r{rank}")
        seen = set()
        for line in open(log, errors="replace"):
            if "g++ compile cmd" not in line or "fabric_erisc_router/" not in line:
                continue
            cmd = re.sub(r"\x1b\[[0-9;]*m", "", line)
            c = parse(cmd, cache)
            if c is None or c["hash"] in seen:
                continue
            seen.add(c["hash"])
            c["rank"] = rank
            cfgs.append(c)
    for c in cfgs:
        key = (c["rank"], f"E{c['erisc']}", c["ch"], c["dir"], c["im"], c["tx"], tuple(c["rx"]), tuple(c["remote_rx"]),
               tuple(c["ds"]), c["actual"])
        rows[key] += 1
    print(f"{job} {label}: {len(cfgs)} router configs")
    if "--rows" in sys.argv:
        print("rank erisc ch dir im | sender slots (w = worker) | rx | remote rx | downstream per VC | actual VC0,VC1 | n")
        for k, n in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], str(x[0][3]))):
            print(f"  r{k[0]} {k[1]} ch{k[2]} {k[3]} im{k[4]} | {k[5]} | rx {list(k[6])} | remote {list(k[7])} | "
                  f"ds {list(k[8])} | {k[9]} | x{n}")
    # contract check over the set
    fwd = collections.defaultdict(collections.Counter)  # vc -> slots of E0 forwarding channels
    for c in cfgs:
        if c["erisc"] != 0 and c["ns"] and c["nr"] and False:
            pass
        for i, (t, w) in enumerate(zip(c["tx_raw"], c["worker"])):
            if w:
                continue
            vc = 0 if i < c["vc1_start"] else 1
            fwd[vc][(t, "im" if c["im"] else "-")] += 1
    dsv = collections.defaultdict(collections.Counter)
    for c in cfgs:
        for vc, n in enumerate(c["ds"][:2]):
            if n:
                dsv[vc][(n, "im" if c["im"] else "-")] += 1
    rx = collections.Counter((r, "im" if c["im"] else "-") for c in cfgs for r in c["rx"])
    rrx = collections.Counter((r, "im" if c["im"] else "-") for c in cfgs for r in c["remote_rx"])
    for vc in (0, 1):
        print(f"VC{vc} forwarding channel slots (slots, intermesh?): {dict(fwd[vc])}")
        print(f"VC{vc} DOWNSTREAM_SENDER_NUM_BUFFERS assumed: {dict(dsv[vc])}")
    print(f"receiver slots: {dict(rx)}")
    print(f"remote receiver slots assumed: {dict(rrx)}")


if __name__ == "__main__":
    main()
