#!/usr/bin/env python3
"""Synthetic hang_capture.py dump built from a real router cache (test fixture for hang_decode.py).

  hang_synth.py OUT_DIR CACHE_DIR LOG CLUSTER_DESC

Places a real E0/E1 config pair (from the cache's ELFs and logged compile commands) on every connected Ethernet core
of the cluster descriptor, writes the mailbox / launch message, both XIP images, runtime args in kernel_main's layout,
connection info, credit counters, routing table, slot headers and stream registers of a quiet fabric, then injects:
  A  chip 0: an E0 VC1 sender channel register wrapped to 0x1FFFF (over-fill), its two slots holding packets
  B  one link: the sender's completion counter one behind the receiver's (lost credit frame)
  C  chip 2: one E1 VC1 downstream copy at 0 while the downstream channel shows 2 free (lost NoC credit)
  D  chip 3: one receiver with a pending VC1 packet
  E  chip 5 eth 4: heartbeat and one stream register change between the passes (still moving)
"""
import json
import os
import struct
import sys

sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import hang_decode as hd  # noqa: E402
import yaml  # noqa: E402

ETH_NOC0 = [(1, 1), (16, 1), (2, 1), (15, 1), (3, 1), (14, 1), (4, 1), (13, 1), (5, 1), (12, 1), (6, 1), (11, 1),
            (7, 1), (10, 1)]


def put32(b, a, v):
    struct.pack_into("<I", b, a, v & 0xFFFFFFFF)


def put16(b, a, v):
    struct.pack_into("<H", b, a, v & 0xFFFF)


def put_hdr(b, a, hb):
    b[a:a + len(hb)] = hb


def header(cfg, typ, size, src, dst_mesh, dst_chip, noc_xy, addr, route):
    hs, rbs = cfg.header_size
    b = bytearray(hs)
    na = (noc_xy[1] << 42) | (noc_xy[0] << 36) | addr
    struct.pack_into("<Q", b, 0, na)
    if typ == 2:
        put32(b, 8, 1)
    put16(b, 40, size)
    b[42] = typ
    b[43] = src
    for i, a in enumerate(route):
        b[48 + i] = a
    put16(b, 48 + rbs, dst_chip)
    put16(b, 50 + rbs, dst_mesh)
    return b


def main():
    out, cache, log, cdesc = sys.argv[1:5]
    cfgs = hd.load_cfgs(cache)
    hd.attach_compile_cmds(cfgs, "/nonexistent", [log])
    cd = yaml.safe_load(open(cdesc))
    eth_harv = cd["harvesting"][0]["eth_harvesting_mask"]
    phys = [p for p in range(14) if not eth_harv >> p & 1]  # logical channel -> physical channel
    conns = {}
    for a, b in cd["ethernet_connections"]:
        conns[(a["chip"], a["chan"])] = (b["chip"], b["chan"])
        conns[(b["chip"], b["chan"])] = (a["chip"], a["chan"])
    chips = sorted({c for c, _ in conns})
    # pair E0 and E1 configs of one router
    keyf = lambda c: (tuple(c.ct), c.g("MY_ETH_CHANNEL"), c.g("MY_DIRECTION"), c.g("EDM_CHANNELS_MASK"),  # noqa: E731
                      c.g("IS_INTERMESH_ROUTER"), c.g("LOCAL_HANDSHAKE_MASTER_ETH_CHAN"))
    e1s = {}
    for c in cfgs.values():
        if c.erisc == 1 and c.ct:
            e1s.setdefault(keyf(c), []).append(c)
    pairs = [(c, e1s[keyf(c)][0]) for c in sorted(cfgs.values(), key=lambda c: c.hash)
             if c.erisc == 0 and c.ct and keyf(c) in e1s]
    print(f"{len(pairs)} E0/E1 pairs")
    used = set()
    routers = {}  # (chip, chan) -> (e0, e1)
    for chip in chips:
        chans = sorted(ch for c, ch in conns if c == chip)
        mask = sum(1 << ch for ch in chans)
        for ch in chans:
            cand = [p for p in pairs if p[0].g("MY_ETH_CHANNEL") == ch and p[0].g("EDM_CHANNELS_MASK") == mask
                    and id(p[0]) not in used] or [p for p in pairs if p[0].g("MY_ETH_CHANNEL") == ch]
            routers[(chip, ch)] = cand[0]
            used.add(id(cand[0][0]))
    xlat = lambda ch: (20 + ch, 25)  # noqa: E731
    KCB = 0xF330
    l1 = {k: bytearray(hd.L1_SIZE) for k in routers}
    streams = {k: {s: [0, 0, 0] for s in range(32)} for k in routers}
    hb = {k: 0xDCBA0000 | (0x100 + i) for i, k in enumerate(sorted(routers))}

    def st(k, sid, val, size=None):
        if sid is not None and sid < 32:
            streams[k][sid][2] = val
            streams[k][sid][1] = val if size is None else size

    # images, launch message, routing table, status, stream registers of a quiet fabric
    for (chip, ch), (e0, e1) in routers.items():
        b = l1[(chip, ch)]
        t0 = 0x40
        t1 = (t0 + len(e0.image) + 15) // 16 * 16
        rta = (t1 + len(e1.image) + 15) // 16 * 16
        b[KCB + t0:KCB + t0 + len(e0.image)] = e0.image
        b[KCB + t1:KCB + t1 + len(e1.image)] = e1.image
        put32(b, hd.MAILBOX + 12, 0)
        lb = hd.MAILBOX + 16
        put32(b, lb + 4, KCB)
        put16(b, lb + 28, rta)
        put16(b, lb + 32, rta + 0x400)
        put32(b, lb + 52, t0)
        put32(b, lb + 56, t1)
        put32(b, lb + 72, len(e0.image))
        put32(b, lb + 76, len(e1.image))
        put32(b, lb + 112, 3)
        b[hd.GO_MSGS + 3] = 0x80
        rt = hd.ROUTING_TABLE
        put16(b, rt + 32, 0 if chip < 4 else 1)
        put16(b, rt + 34, chip % 4)
        b[rt + 2700], b[rt + 2701], b[rt + 2702], b[rt + 2703] = (chip % 4) // 2, chip % 2, 2, 2
        put32(b, e0.g("EDM_STATUS_PTR_ADDR"), 0xA3B3C3D3)
        put32(b, hd.HEARTBEAT, hb[(chip, ch)])
        k = (chip, ch)
        for c in range(e0.g("NUM_SENDER_CHANNELS")):
            if e0.sender_serviced(c):
                st(k, e0.sender_stream(c), e0.sender_slots[c])
        for rx in range(e1.g("NUM_RECEIVER_CHANNELS")):
            if e1.receiver_serviced(rx):
                st(k, e1.g(f"TO_RECEIVER_{rx}_PKTS_SENT_ID"), 0, e1.recv_slots[rx])
        for s in e1.ds_sets():
            for c in range(s["size"]):
                st(k, s["local_stream"][c], s["num_slots"])
        # worker connection on channel 0
        ci = e0.g("LOCAL_SENDER_CH_0_CONN_INFO_ADDR")
        put32(b, ci, 0x19F00)
        put32(b, ci + 16, 0x19F10)
        put32(b, ci + 32, (3 << 16) | 2)
        put32(b, ci + 96, 1)
    # runtime args and downstream connections (same link index among routers of one direction)
    by_dir = {}
    ds_used = {}
    for (chip, ch), (e0, e1) in sorted(routers.items()):
        by_dir.setdefault((chip, e0.g("MY_DIRECTION")), []).append(ch)
    for (chip, ch), (e0, e1) in routers.items():
        b = l1[(chip, ch)]
        my = e0.g("MY_DIRECTION")
        rank = by_dir[(chip, my)].index(ch)
        ms = e0.g("MAX_NUM_SENDER_CHANNELS")
        words = [0] * 160
        for c in range(e0.g("NUM_SENDER_CHANNELS")):
            ci = e0.g(f"LOCAL_SENDER_CH_{c}_CONN_INFO_ADDR")
            words[c], words[ms + c] = ci + 96, ci - 16
        i = 2 * ms
        for s in e1.ds_sets():
            nds = e1.g(f"NUM_DOWNSTREAM_SENDERS_VC{s['vc']}")
            mask, dense = 0, []
            for c in range(s["size"]):
                lst = by_dir.get((chip, s["dirs"][c]), [])
                if lst and len(dense) < nds:
                    dch = lst[min(rank, len(lst) - 1)]
                    mask |= 1 << c
                    dense.append((c, dch))
            words[i] = mask
            ds_used.setdefault((chip, ch), {})[s["vc"]] = dense[:nds]
            i += 1
            nx = ny = 0
            for j, (c, dch) in enumerate(dense[:nds]):
                d0 = routers[(chip, dch)][0]
                pos = s["remote_pos"][c]
                dk = d0.compact_of_pos(pos)
                words[i + j] = d0.sender_base[dk]
                x, y = xlat(dch)
                nx |= x << (8 * j)
                ny |= y << (8 * j)
                words[i + nds + 2 + j] = d0.g(f"LOCAL_SENDER_CH_{dk}_CONN_INFO_ADDR") + 96
                words[i + 2 * nds + 2 + j] = d0.g(f"LOCAL_SENDER_CH_{dk}_CONN_INFO_ADDR")
                # the downstream's connection info names this router as its upstream
                db = l1[(chip, dch)]
                dci = d0.g(f"LOCAL_SENDER_CH_{dk}_CONN_INFO_ADDR")
                put32(db, dci, hd.STREAM_BASE + s["local_stream"][c] * hd.STREAM_STRIDE + hd.REG_UPDATE_IDX * 4)
                put32(db, dci + 32, (25 << 16) | (20 + ch))
                put32(db, dci + 96, 1)
            words[i + nds], words[i + nds + 1] = nx, ny
            i += 2 + 4 * nds
        rta0 = KCB + struct.unpack_from("<H", b, hd.MAILBOX + 16 + 28)[0]
        rta1 = KCB + struct.unpack_from("<H", b, hd.MAILBOX + 16 + 32)[0]
        for base in (rta0, rta1):
            for j, w in enumerate(words):
                put32(b, base + 4 * j, w)
    # credit counters and old packets in every ring (quiet: counters agree)
    for (chip, ch), (e0, e1) in routers.items():
        a = l1[(chip, ch)]
        pk = conns[(chip, ch)]
        bb = l1[pk]
        pe1 = routers[pk][1]
        for c in range(e0.g("NUM_SENDER_CHANNELS")):
            if not e0.sender_serviced(c):
                continue
            n = 40 + 7 * c + chip
            put32(a, e0.g("TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR") + 4 * c, n)
            put32(bb, pe1.g("LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR") + 4 * c, n)
            for s in range(min(e0.sender_slots[c], 4)):
                put_hdr(a, e0.sender_base[c] + s * e0.g("CHANNEL_BUFFER_SIZE"),
                        header(e0, 0, 4096, c, 1, 2, (1, 2), 0x20000 + 0x1000 * s, [0, 0x20, 0x20, 0]))
    # A: chip 0, an E0 VC1 sender channel wrapped
    k = next(k for k in sorted(routers) if k[0] == 0 and routers[k][0].g("ACTUAL_VC1_SENDER_CHANNELS") > 0)
    e0 = routers[k][0]
    ch = e0.g("ACTUAL_VC0_SENDER_CHANNELS")
    st(k, e0.sender_stream(ch), 0x1FFFF, e0.sender_slots[ch])
    for s in range(e0.sender_slots[ch]):
        put_hdr(l1[k], e0.sender_base[ch] + s * e0.g("CHANNEL_BUFFER_SIZE"),
                header(e0, 2, 0, 0, 1, 3, (5, 6), 0x1A000, [0x08, 0x20, 0, 0x02]))
    print(f"A: {k} E0 ch{ch} stream {e0.sender_stream(ch)} -> 0x1FFFF")
    # B: a link with the sender one completion behind
    kb = next(k for k in sorted(routers) if k[0] == 1)
    e0b = routers[kb][0]
    cb = e0b.g("ACTUAL_VC0_SENDER_CHANNELS") + 1
    a = e0b.g("TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR") + 4 * cb
    put32(l1[kb], a, struct.unpack_from("<I", l1[kb], a)[0] - 1)
    print(f"B: {kb} ch{cb} sender completions one behind")
    # C: chip 2, an E1 VC1 downstream copy at 0 while the downstream channel is free
    kc = next(k for k in sorted(routers) if k[0] == 2 and ds_used.get(k, {}).get(1))
    e1c = routers[kc][1]
    s1 = e1c.ds_sets()[1]
    c0, dch = ds_used[kc][1][0]
    st(kc, s1["local_stream"][c0], 0)
    print(f"C: {kc} VC1 copy stream {s1['local_stream'][c0]} (slot {c0} -> eth{dch}) -> 0")
    # D: chip 3, one pending VC1 packet at a receiver
    kd = next(k for k in sorted(routers) if k[0] == 3)
    e1d = routers[kd][1]
    st(kd, e1d.g("TO_RECEIVER_1_PKTS_SENT_ID"), 1)
    up = conns[kd]
    ue0 = routers[up][0]
    tot = sum(struct.unpack_from("<I", l1[kd], e1d.g("LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR") + 4 * c)[0]
              for c in range(ue0.g("NUM_SENDER_CHANNELS")) if ue0.vc_of_sender(c) == 1)
    slot = tot % e1d.recv_slots[1]
    put_hdr(l1[kd], e1d.recv_base[1] + slot * e1d.g("CHANNEL_BUFFER_SIZE"),
            header(e1d, 3, 4096, 5, 0, 1, (7, 8), 0x30000, [0x20, 0, 0x01, 0]))
    print(f"D: {kd} rx1 pending 1 in slot {slot}")
    # write the capture
    os.makedirs(out, exist_ok=True)
    index = {"tool": "hang_synth.py", "format": 1, "host": "synthetic", "start_time": "synthetic",
             "meta": {"synthetic": "1"}, "constants": {"stream_regs": [4, 10, 297]}, "chips": {},
             "eth_connections": [], "errors": [], "complete": True, "timings": {}}
    for (chip, ch), b in sorted(l1.items()):
        os.makedirs(f"{out}/chip{chip}", exist_ok=True)
        fn = f"chip{chip}/eth{ch:02d}_l1.bin"
        open(f"{out}/{fn}", "wb").write(b)
        p1 = {"streams": {str(s): list(v) for s, v in streams[(chip, ch)].items()},
              "txq": {str(q): {hex(o): 0 for o in (0, 4, 8, 0x30, 0x34, 0x3c, 0x40, 0x4c, 0x68, 0x6c)} for q in range(3)},
              "soft_reset": 0, "l1_words": {hex(hd.HEARTBEAT): hb[(chip, ch)]}, "errors": []}
        p2 = json.loads(json.dumps(p1))
        if (chip, ch) == (5, 4):  # E: still moving
            p2["l1_words"][hex(hd.HEARTBEAT)] += 1
            p2["streams"]["0"][2] -= 1
        crec = index["chips"].setdefault(str(chip), {"unique_id": cd["chip_unique_ids"][chip], "cores": {}})
        crec["cores"][str(ch)] = {"eth_channel": ch, "noc0": list(ETH_NOC0[phys[ch]]), "translated": list(xlat(ch)),
                                  "logical": [0, ch], "connected_to": list(conns[(chip, ch)]),
                                  "l1": {"file": fn, "addr": 0, "size": hd.L1_SIZE, "missing": []},
                                  "regs": {"pass1": p1, "pass2": p2}}
    for (c, ch), (rc, rch) in sorted(conns.items()):
        if (c, ch) < (rc, rch):
            index["eth_connections"].append([c, ch, rc, rch])
    json.dump(index, open(f"{out}/index.json", "w"), indent=1)
    print(f"wrote {len(l1)} cores to {out}")


if __name__ == "__main__":
    main()
