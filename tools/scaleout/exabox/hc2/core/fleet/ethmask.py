# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# ethmask.py: hide one ETH link from UMD topology discovery for one process run, by setting the link's ETH firmware
# port_status word (eth_status_t.port_status, L1 0x7CC04; UMD treats a channel as active only if it reads 1 = PORT_UP /
# EthTrainingStatus::SUCCESS) to 3 (PORT_UNUSED / NOT_CONNECTED) on both ends. Volatile chip L1 only: a warm reset
# re-runs ETH training and restores it. Channels are UMD logical ETH channels (cluster descriptor 'chan', triage e0,N).
#   ethmask.py show <devnode>...                      read-only: per channel core, port/train status, local/remote ids
#   ethmask.py drop <devA> <chA> <devB> <chB> [val]   verify A:chA <-> B:chB is one link, then write val (default 3) on both
#   ethmask.py disc <devnode>...                      UMD topology discovery over those devnodes (TT_VISIBLE_DEVICES): links
#   ethmask.py stats <devnode>:<ch>[,<ch>...]...      read-only ETH FW link stats (addresses from triage
#                                                     check_eth_status.py, blackhole eth_fw_api.h): port/link, retrain, bad FCS, FEC
#   ethmask.py up <devnode>:<ch>[,<ch>...]...         read-only: exit 0 when every named channel reads port_status 1 (PORT_UP)
#   ethmask.py kept <disc.json> <chA> <chB>           no device: exit 0 when that discovery saw exactly one link, chA <-> chB
# In tray-test for run.sh --repro link-a (lib/fleet.sh fl_linka); tt_umd is imported only by the device commands.
import json, sys, time

PORT, TRAIN = 0x7CC04, 0x7CC08
ETH_NOC0 = [(1, 1), (16, 1), (2, 1), (15, 1), (3, 1), (14, 1), (4, 1), (13, 1), (5, 1), (12, 1), (6, 1), (11, 1), (7, 1), (10, 1)]


def r32(d, c, a):
    return int.from_bytes(d.noc_read(c[0], c[1], a, 4), "little")


def r8(d, c, a):
    return d.noc_read(c[0], c[1], a, 1)[0]


def chip(n):
    import tt_umd as u
    d = u.TTDevice.create(int(n)); d.init_tt_device()
    sd = u.SocDescriptor(d)
    cores = {}
    for ch, n0 in enumerate(sd.get_cores(u.CoreType.ETH, u.CoordSystem.NOC0)):
        tr = sd.translate_coord_to(n0, u.CoordSystem.TRANSLATED)
        cores[ch] = {"noc0": (n0.x, n0.y), "tr": (tr.x, tr.y)}
    return d, cores


def kept(path, ka, kb):
    """True when the discovery JSON (the last '{' line of path) holds exactly two chips joined by one link, ka <-> kb."""
    d = json.loads([l for l in open(path) if l.startswith("{")][-1])
    loc = sorted(int(ch) for v in d["links"].values() for ch in v)
    return len(d["links"]) == 2 and loc == sorted([int(ka), int(kb)])


def info(d, c):
    t = c["tr"]
    return {"noc0": list(c["noc0"]), "translated": list(t), "phys_idx": ETH_NOC0.index(tuple(c["noc0"])) if tuple(c["noc0"]) in ETH_NOC0 else None,
            "port_status": r32(d, t, PORT), "train_status": r32(d, t, TRAIN),
            "local_asic": (r32(d, t, 0x7CFD4) << 32) | r32(d, t, 0x7CFD8),
            "remote_asic": (r32(d, t, 0x7CFF4) << 32) | r32(d, t, 0x7CFF8),
            "remote_chan_phys": r8(d, t, 0x7CFE2), "remote_chan_logical": r8(d, t, 0x7CFE3)}


def show(nodes):
    out = {}
    for n in nodes:
        d, cores = chip(n)
        out[n] = {ch: info(d, c) for ch, c in cores.items()}
        del d
    return out


def r64(d, c, a):  # blackhole eth FW counters: u64, low word first
    return r32(d, c, a) | (r32(d, c, a + 4) << 32)


def stats(specs):
    out = {}
    for sp in specs:
        n, chs = sp.split(":")
        d, cores = chip(n)
        out[n] = {}
        for ch in (int(x) for x in chs.split(",")):
            t = cores[ch]["tr"]
            out[n][ch] = {"port_status": r32(d, t, PORT), "pcs_link": r32(d, t, 0xFFB9800C), "rx_link_up": r32(d, t, 0x7CE04),
                          "retrain": r32(d, t, 0x7CE00), "rx_bad_fcs": r64(d, t, 0x7CE60), "corr_cw": r64(d, t, 0x7CE90),
                          "uncorr_cw": r64(d, t, 0x7CE98), "heartbeat": hex(r32(d, t, 0x7CC70)), "err_stat": hex(r32(d, t, 0xFFB980D8))}
        del d
    return out


cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "stats":
    print(json.dumps({"t": time.time(), "stats": stats(sys.argv[2:])}), flush=True)
elif cmd == "show":
    print(json.dumps(show(sys.argv[2:])), flush=True)
elif cmd == "up":
    s = stats(sys.argv[2:])
    print(json.dumps({"t": time.time(), "stats": s}), flush=True)
    sys.exit(0 if all(p["port_status"] == 1 for c in s.values() for p in c.values()) else 1)
elif cmd == "kept":
    ok = kept(sys.argv[2], sys.argv[3], sys.argv[4])
    print("DISC", "OK one link" if ok else "NOT the single kept link", sys.argv[3], sys.argv[4], flush=True)
    sys.exit(0 if ok else 1)
elif cmd == "disc":
    import tt_umd as u
    cd = u.TopologyDiscovery.create_cluster_descriptor()
    conn = cd.get_ethernet_connections()
    print(json.dumps({"chips": sorted(cd.get_all_chips()), "bdfs": {int(k): v for k, v in cd.get_chip_pci_bdfs().items()},
                      "active": {int(c): sorted(cd.get_active_eth_channels(c)) for c in cd.get_all_chips()},
                      "links": {int(c): {int(ch): [int(rc), int(rch)] for ch, (rc, rch) in sorted(v.items())} for c, v in conn.items()}}), flush=True)
elif cmd == "drop":
    A, ca, B, cb = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    val = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    da, coa = chip(A); db, cob = chip(B)
    ia, ib = info(da, coa[ca]), info(db, cob[cb])
    ok = (ia["remote_asic"] == ib["local_asic"] and ib["remote_asic"] == ia["local_asic"]
          and ia["port_status"] == 1 and ib["port_status"] == 1
          and ib["phys_idx"] in (ia["remote_chan_phys"], ia["remote_chan_logical"])
          and ia["phys_idx"] in (ib["remote_chan_phys"], ib["remote_chan_logical"]))
    res = {"A": [A, ca, ia], "B": [B, cb, ib], "pair_verified": ok, "value": val}
    if not ok:
        print(json.dumps({**res, "error": "not one verified PORT_UP link; nothing written"}), flush=True); sys.exit(6)
    for d, c in ((da, coa[ca]), (db, cob[cb])):
        d.noc_write(c["tr"][0], c["tr"][1], PORT, val.to_bytes(4, "little"))
    res["readback_0s"] = [r32(da, coa[ca]["tr"], PORT), r32(db, cob[cb]["tr"], PORT)]
    time.sleep(5)
    res["readback_5s"] = [r32(da, coa[ca]["tr"], PORT), r32(db, cob[cb]["tr"], PORT)]
    print(json.dumps(res), flush=True)
    sys.exit(0 if res["readback_5s"] == [val, val] else 7)
else:
    print("ethmask.py: show, drop, disc, stats, up or kept", file=sys.stderr); sys.exit(2)
