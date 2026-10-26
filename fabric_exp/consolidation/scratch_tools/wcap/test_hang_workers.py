#!/usr/bin/env python3
"""Offline tests of the hang capture's worker part (no device).

  python_env/bin/python fabric_exp/consolidation/test_hang_workers.py [--keep DIR] [--kernel-cache DIR]

1. Router-only captures (results/p15cvc1_auto_hangcap, results/v3l3_o3_capselftest_hangcap) decode byte-identically
   with this hang_decode.py and with the committed one (git HEAD), full and --quiet-routers.
2. End to end on a fake device: hang_capture.main() runs against a fake tt-exalens context whose Ethernet cores serve
   the fourth hang's router L1 and registers (p15cvc1_auto_hangcap) and whose Tensix cores serve synthetic L1 images of
   the VC1 all-to-all test (per chip one sender with 7 configs over 3 router connections, 7 receivers, one sync core,
   one dispatch core, one stale test core), built from the real kernel ELFs' layouts (gdb) of that run's cache
   (--kernel-cache, default /home/aagarwal/.tt-metal-cache-p1vc1-v3vc-auto). Two scenarios:
     clean: every worker finished; the decode must report no worker anomaly;
     hang:  injected faults (receiver waiting with its slot overwritten by a later packet, sync core short of sync
            packets, sender parked in local sync, a sender connection that lost credits with the router showing them
            delivered, NoC writes not acked); the decode must name each one.
   The capture's reads, selection (scan, window vs whole L1), files and index are what the decoder then reads.
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hang_capture as hc  # noqa: E402
import hang_workers as hw  # noqa: E402

ROOT = hw.ROOT
PY = os.path.join(ROOT, "python_env/bin/python")
ROUTER_CAP = os.path.join(ROOT, "fabric_exp/results/p15cvc1_auto_hangcap")
WORKER_SELFTEST_CAP = os.path.join(ROOT, "fabric_exp/results/wcapv3v2_o3_capselftest_hangcap")  # job 308
SELFTEST_CAP = os.path.join(ROOT, "fabric_exp/results/v3l3_o3_capselftest_hangcap")
FAILS = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


# ====================================================================== 1. router-only regression
def router_regression(tmp):
    """Router-only captures: this decoder against (a) each capture's stored decode.txt (full or --quiet-routers, as it
    was written; lines 1-2 name the paths and are skipped) and (b) the decoder committed at git HEAD, both modes."""
    print("== router-only captures decode unchanged ==")
    new = os.path.join(HERE, "hang_decode.py")

    def run(script, cap, extra):
        o = subprocess.run(
            [PY, script, cap, os.path.join(cap, "cache")] + extra, capture_output=True, text=True, cwd=ROOT
        )
        return o.stdout + o.stderr

    for cap in (ROUTER_CAP, SELFTEST_CAP):
        if not os.path.isdir(cap):
            print(f"  (missing {cap}: skipped)")
            continue
        stored = open(os.path.join(cap, "decode.txt")).read().splitlines()[2:]
        extra = [] if any(line.startswith("=== chip") for line in stored) else ["--quiet-routers"]
        got = run(new, cap, extra).splitlines()[2:]
        check(
            got == stored and len(stored) > 100,
            f"{os.path.basename(cap)}: same as its stored decode.txt ({' '.join(extra) or 'full'}, "
            f"{len(stored)} lines)",
        )
    old = os.path.join(tmp, "hang_decode_head.py")
    r = subprocess.run(
        ["git", "-C", ROOT, "show", "HEAD:fabric_exp/consolidation/hang_decode.py"], capture_output=True, text=True
    )
    if r.returncode:
        print("  (git HEAD version unavailable: skipped)")
        return
    open(old, "w").write(r.stdout)
    for cap in (ROUTER_CAP, SELFTEST_CAP):
        if not os.path.isdir(cap):
            continue
        for extra in ([], ["--quiet-routers"]):
            a, b = run(old, cap, extra), run(new, cap, extra)
            check(
                a == b and len(a) > 1000,
                f"{os.path.basename(cap)} {' '.join(extra) or 'full'}: identical to git HEAD's decoder "
                f"({len(a.splitlines())} lines)",
            )


def worker_selftest_regression():
    """The hardware self-test with workers (job 308: idle device after a clean VC1 sanity run): the decode must be
    clean, every flow matched, earlier tests' cores left out, closed connections' credit lag classed as expected."""
    print("== hardware worker self-test capture decodes clean ==")
    if not os.path.isdir(WORKER_SELFTEST_CAP):
        print(f"  (missing {WORKER_SELFTEST_CAP}: skipped)")
        return
    o = subprocess.run(
        [PY, os.path.join(HERE, "hang_decode.py"), WORKER_SELFTEST_CAP, os.path.join(WORKER_SELFTEST_CAP, "cache")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    out = o.stdout + o.stderr
    anoms = out[out.index("== Anomalies ==") :].splitlines()[1:] if "== Anomalies ==" in out else ["(none section)"]
    worker_anoms = [a for a in anoms if "tensix" in a]
    check(not worker_anoms and anoms[-1].strip() == "none", f"no worker anomaly ({worker_anoms[:2]})")
    for e in (
        "flow(s) complete on both ends",
        "connection(s) quiet and consistent",
        "reported DONE",
        "last ran an earlier program and are left out",
        "rank 0 mesh 0:",
        "rank 1 mesh 1:",
    ):
        check(e in out, f"self-test decode names: {e}")
    for e in ("(no captured sender)", "credit(s) not delivered", "decoder error", "Traceback"):
        check(e not in out, f"self-test decode does not say: {e}")


# ====================================================================== synthetic Tensix images
class Img:
    def __init__(self, size=hw.MAILBOX + 0x180000):
        self.b = bytearray(1536 * 1024)

    def w8(self, a, v):
        self.b[a] = v & 0xFF

    def w16(self, a, v):
        struct.pack_into("<H", self.b, a, v & 0xFFFF)

    def w32(self, a, v):
        struct.pack_into("<I", self.b, a, v & 0xFFFFFFFF)

    def w64(self, a, v):
        struct.pack_into("<Q", self.b, a, v)

    def put(self, a, data):
        self.b[a : a + len(data)] = data


class SW:
    """Struct writer at an address with a gdb layout (mirror of hang_workers.View)."""

    def __init__(self, img, addr, lay):
        self.img, self.addr, self.lay = img, addr, lay

    def a(self, path, i=0, elem=4):
        return self.addr + self.lay.off(path) + i * elem

    def w32(self, path, v, i=0):
        self.img.w32(self.a(path, i, 4), v)

    def w16(self, path, v, i=0):
        self.img.w16(self.a(path, i, 2), v)

    def w8(self, path, v, i=0):
        self.img.w8(self.a(path, i, 1), v)


BASE = 0x1B480
LOCAL_ARGS, RESULTS, KCONFIG, MUX, MUXSYNC = BASE, BASE + 0x8000, BASE + 0x9000, BASE + 0x19000, BASE + 0x19400
SEND_HDRS, GSYNC, LSYNC, SEND_PAYLOAD = BASE + 0x1A440, BASE + 0x22440, BASE + 0x22450, BASE + 0x22460
RECV_PAYLOAD = BASE + 0x1A440 + 128
KCB, RTA_OFF, TEXT_OFF, SEM_OFF = 0xA080, 0x40, 0x800, 0x400
PSIZE, VBUF, NPK = 4096, 4 * 4096, 100
SENDER_XY, SYNC_XY, DISPATCH_XY, STALE_XY = (14, 11), (14, 10), (2, 11), (3, 11)
KID_BASE, KID_STALE_OLD, KID_STALE, KID_DISPATCH = 1000, 200, 500, 49  # watcher kernel ids
KCB_STALE_OLD, KCB_STALE = 0x13890, 0xD330


def seq(s0, k):
    for _ in range(k):
        s0 = hw.prng_next(s0)
    return s0


def write_slot(img, slot, kcb, text_off, text_size, kid, enables=0, mode=0):
    """One launch-message slot. The host-assigned program id stays 0, as on the device (hardware self-test, job 308);
    programs are told apart by the watcher kernel ids, a process-wide counter."""
    lb = hw.MAILBOX + hw.LAUNCH_OFF + hw.LAUNCH_SIZE * slot
    img.w32(lb, kcb)
    img.w16(lb + 28, RTA_OFF)
    img.w8(lb + 48, mode)  # DISPATCH_MODE_DEV 0, HOST 1
    img.w32(lb + 52, text_off)
    img.w32(lb + 72, text_size)
    img.w32(lb + 108, 0)
    img.w32(lb + 112, enables)
    img.w16(lb + 120, kid)


def write_head(img, slot, kcb, text_off, text_size, kid, running, mode=0):
    """A core whose last executed launch slot is `slot`: GO (running, rd_ptr == slot) or DONE (rd_ptr advanced, enables
    cleared, as brisc.cc does; host-mode launches do not advance rd_ptr)."""
    m = hw.MAILBOX
    rd = slot if running or mode == 1 else (slot + 1) % 8
    img.w32(m + hw.RD_PTR_OFF, rd)
    write_slot(img, slot, kcb, text_off, text_size, kid, 1 if running else 0, mode)
    img.w32(m + hw.GO_OFF, (0x80 if running else 0x00) << 24)
    img.w32(m + hw.GO_IDX_OFF, 0)


def write_common(img, kb, extra, kcb=None):
    kcb = KCB if kcb is None else kcb
    words = [LOCAL_ARGS, 0x8000, RESULTS, 0x1000, KCONFIG, 0x10000, MUX, 0x400, MUXSYNC] + extra
    for i, v in enumerate(words):
        img.w32(kcb + RTA_OFF + 4 * i, v)
    img.put(kcb + TEXT_OFF, kb.image)


def write_result(img, status, packets):
    img.w32(RESULTS, status)
    img.w32(RESULTS + 8, packets)


class Topo:
    """Routers of the fourth hang: per chip (mesh, dev) and E0 channel-0 worker connection words."""

    def __init__(self):
        idx = json.load(open(os.path.join(ROUTER_CAP, "index.json")))
        self.idx = idx
        self.chips = sorted(int(c) for c in idx["chips"])
        self.mesh = {}
        self.routers = {}  # chip -> list of dicts
        for c, crec in idx["chips"].items():
            for ch, rec in crec["cores"].items():
                b = open(os.path.join(ROUTER_CAP, rec["l1"]["file"]), "rb").read()
                self.mesh[int(c)] = (hw.u16(b, 0x6F340 + 32), hw.u16(b, 0x6F340 + 34))
                ci = 89280  # LOCAL_SENDER_CH_0_CONN_INFO_ADDR of every config in this run
                w = lambda a: hw.u32(b, a)  # noqa: E731
                self.routers.setdefault(int(c), []).append(
                    {
                        "chan": int(ch),
                        "xlat": tuple(rec["translated"]),
                        "ci": ci,
                        "sem": w(ci),
                        "td": w(ci + 16),
                        "wxy": (w(ci + 32) & 0xFFFF, w(ci + 32) >> 16),
                        "rdc": w(ci + 48),
                        "cursor": w(ci + 112),
                    }
                )


def write_adapter(img, addr, lay, r, sent, credit, N=51):
    s = SW(img, addr, lay)
    s.w32("edm_buffer_addr", 0x20000 + (sent % N) * 4352)
    s.w32("worker_credits_stream_id", 0xFFFFFFFF)
    s.w32("edm_buffer_local_free_slots_read_ptr", r["sem"])
    s.w32("edm_buffer_local_free_slots_update_ptr", r["sem"])
    s.w32("edm_buffer_remote_free_slots_update_addr", 0xFFB40000 + 22 * 0x1000 + 270 * 4)
    s.w32("edm_connection_handshake_l1_addr", r["ci"] + 96)
    s.w32("edm_worker_location_info_addr", r["ci"])
    s.w32("edm_copy_of_wr_counter_addr", r["ci"] + 112)
    s.w32("local_producer_cursor_addr", KCB + SEM_OFF + 0x40)
    s.w32("worker_teardown_addr", r["td"])
    s.w32("edm_buffer_base_addr", 0x20000)
    s.w8("buffer_slot_index", sent % N)
    s.w32("buffer_slot_write_counter.counter", sent)
    s.w8("buffer_slot_write_counter.index", sent % N)
    s.w16("buffer_size_bytes", 4352)
    s.w8("num_buffers_per_channel", N)
    s.w8("edm_noc_x", r["xlat"][0])
    s.w8("edm_noc_y", r["xlat"][1])
    img.w32(r["sem"], credit)  # the router's credit word in worker L1
    img.w32(r["td"], 0)
    # fabric connection table entry for this Ethernet channel
    ch = (r["sem"] - hw.FABRIC_CONN_RW) // 16
    e = hw.FABRIC_CONN + 24 * ch
    img.w32(e, 0x20000)
    img.w32(e + 4, r["ci"] + 96)
    img.w32(e + 8, r["ci"])
    img.w32(e + 12, r["ci"] + 112)
    img.w16(e + 16, 4352)
    img.w8(e + 19, r["xlat"][0])
    img.w8(e + 20, r["xlat"][1])
    img.w8(e + 21, N)
    img.w32(hw.FABRIC_CONN + 384, hw.u32(img.b, hw.FABRIC_CONN + 384) | 1 << ch)


def build(topo, bins, faults, lags=None):
    """{chip: {(x, y): (Img, soft_reset, niu)}} for the synthetic all-to-all run."""
    kb = {k.kind: k for k in bins["brisc"] if k.kind in ("sender", "receiver", "sync")}
    disp = next(k for k in bins["brisc"] if k.name == "cq_dispatch")
    for k in kb.values():
        k.load_layouts(hw.DEFAULT_GDB)
        assert "main" in k.layouts, (k.label(), k.gdb_error)
    S, R, Y = kb["sender"], kb["receiver"], kb["sync"]
    out = {}
    others = {c: [d for d in topo.chips if d != c] for c in topo.chips}
    for c in topo.chips:
        cores = out.setdefault(c, {})
        mesh, dev = topo.mesh[c]
        rts = topo.routers[c]
        snd_r = sorted([r for r in rts if r["wxy"] == SENDER_XY], key=lambda r: r["chan"])
        syn_r = sorted([r for r in rts if r["wxy"] == SYNC_XY], key=lambda r: r["chan"])
        f_hang = faults.get(c, {})
        kid = KID_BASE + 20 * c  # this chip's newest program: sync kid, sender kid + 1, receivers kid + 2 ...
        # ---------------- sender
        img = Img()
        running = bool(f_hang.get("sender"))
        write_head(img, 3, KCB, TEXT_OFF, len(S.image), kid + 1, running)
        write_common(img, S, [SEND_HDRS, SEND_PAYLOAD, 0x180000])
        img.w16(hw.ROUTING_TABLE + 32, mesh)
        img.w16(hw.ROUTING_TABLE + 34, dev)
        L = S.layouts
        m = SW(img, KCONFIG, L["main"])
        for i, v in enumerate([LOCAL_ARGS, 0x8000, RESULTS, 0x1000, KCONFIG, 0x10000, MUX, 0x400, MUXSYNC]):
            img.w32(KCONFIG + 4 * i, v)
        m.w8("connections.num_connections", len(snd_r))
        stride = L["main"].fsize("connections.storage") // L["main"].fsize("connections.is_mux")
        for i, r in enumerate(snd_r):
            sent, credit = r["rdc"], r["rdc"] - (lags or {}).get((c, i), 0)
            if f_hang.get("lost_credit") == i:
                credit = r["rdc"] - 51
            write_adapter(img, m.a("connections.storage") + i * stride, L["adapter"], r, sent, credit)
        ls = SW(img, m.a("local_sync_config_storage"), L["local_sync"])
        ls.w32("sync_core_xy_encoding_", SYNC_XY[1] << 6 | SYNC_XY[0], 0)
        ls.w32("sync_core_xy_encoding_", SENDER_XY[1] << 6 | SENDER_XY[0], 1)
        ls.w32("sync_address", LSYNC)
        ls.w32("sync_val", 2)
        img.w32(LSYNC, f_hang.get("sender_lsync", 3))
        tsz = L["traffic"].size
        for i, d in enumerate(others[c]):
            p = m.a("traffic_configs_storage") + i * tsz
            m.w32("traffic_config_ptrs", p, i)
            t = SW(img, p, L["traffic"])
            ci = i % max(len(snd_r), 1)
            t.w8("connection_idx_", ci)
            t.w8("connection_indices_", ci, 0)
            t.w8("num_connections_", 1)
            s0 = 0x10000 * c + d + 1
            sent = NPK - (1 if (f_hang.get("lost_credit") == ci and i == ci) else 0)
            t.w32("metadata.num_packets", NPK)
            t.w32("metadata.seed", seq(s0, sent))
            t.w32("metadata.payload_buffer_size", VBUF)
            t.w32("num_packets_processed", sent)
            t.w32("payload_size_bytes", PSIZE)
            t.w8("noc_send_type_", 0)
            rx = (1 + others[d].index(c), 2)
            enc = rx[1] << 6 | rx[0]
            t.w32("noc_fields_.write_fields.payload_size_bytes", PSIZE)
            t.w32("noc_fields_.write_fields.dst_address", RECV_PAYLOAD)
            t.w32("noc_fields_.write_fields.dst_noc_encoding", enc)
            ha = SEND_HDRS + 112 * i
            t.w32("packet_header", ha)
            img.w64(ha, enc << 36 | RECV_PAYLOAD)
            img.w16(ha + 40, PSIZE)
            img.w8(ha + 42, 0)
            dm, dd = topo.mesh[d]
            img.w16(ha + 48 + 36, dd)
            img.w16(ha + 50 + 36, dm)
        write_result(img, 0xABC00000 if running else 0xABC00001, 0 if running else 7 * NPK)
        niu = {}
        if f_hang.get("niu"):
            niu = {(0, 0x28): 1000, (0, 0x4): 999}
        cores[SENDER_XY] = (img, 0, niu)
        # ---------------- receivers
        for j, s in enumerate(others[c]):
            img = Img()
            key = ("recv", s)
            fr = f_hang.get(key)
            running = fr is not None
            write_head(img, 3, KCB, TEXT_OFF, len(R.image), kid + 2 + j, running)
            write_common(img, R, [RECV_PAYLOAD - 128, RECV_PAYLOAD])
            img.w16(hw.ROUTING_TABLE + 32, mesh)
            img.w16(hw.ROUTING_TABLE + 34, dev)
            L = R.layouts
            for i, v in enumerate([LOCAL_ARGS, 0x8000, RESULTS, 0x1000, KCONFIG, 0x10000, MUX, 0x400, MUXSYNC]):
                img.w32(KCONFIG + 4 * i, v)
            m = SW(img, KCONFIG, L["main"])
            m.w8("credit_connections.num_connections", 0)
            p = m.a("validation_configs_storage")
            m.w32("traffic_configs_", p, 0)
            s0 = 0x10000 * s + c + 1
            for i, v in enumerate([NPK, s0, VBUF, 0, PSIZE, RECV_PAYLOAD, 0]):
                img.w32(LOCAL_ARGS + 4 * i, v)
            got = fr["got"] if running else NPK
            v = SW(img, p, L["vbase"])
            v.w32("metadata.num_packets", NPK)
            v.w32("metadata.seed", seq(s0, got))
            v.w32("metadata.payload_buffer_size", VBUF)
            v.w32("num_packets_processed", got)
            pb = p + L["v_write"].off("payload_buffer_storage")
            SW(img, p, L["v_write"]).w32("payload_buffer_", pb)
            off, wrapped = 0, 0
            for _ in range(got):
                off += PSIZE
                if off + PSIZE > VBUF:
                    off, wrapped = 0, 1
            sb = SW(img, pb, L["streambuf"])
            sb.w32("base_address_", RECV_PAYLOAD)
            sb.w32("total_size_", VBUF)
            sb.w32("payload_size_", PSIZE)
            sb.w32("current_offset_", off)
            sb.w8("has_wrapped_", wrapped)
            if running:
                holds = fr["holds"]  # 0-based packet whose data sits in the polled slot
                st = seq(s0, holds)
                for k in range(PSIZE // 4):
                    img.w32(RECV_PAYLOAD + off + 4 * k, st + k)
            write_result(img, 0xABC00000 if running else 0xABC00001, 0 if running else NPK)
            cores[(1 + j, 2)] = (img, 0, {})
        # ---------------- sync core
        img = Img()
        fs = f_hang.get("sync")
        running = fs is not None
        write_head(img, 3, KCB, TEXT_OFF, len(Y.image), kid, running)
        write_common(img, Y, [SEND_HDRS, SEND_PAYLOAD, 0x180000])
        img.w16(hw.ROUTING_TABLE + 32, mesh)
        img.w16(hw.ROUTING_TABLE + 34, dev)
        L = Y.layouts
        for i, v in enumerate([LOCAL_ARGS, 0x8000, RESULTS, 0x1000, KCONFIG, 0x10000, MUX, 0x400, MUXSYNC]):
            img.w32(KCONFIG + 4 * i, v)
        m = SW(img, KCONFIG, L["main"])
        conns = syn_r[:1] + snd_r[:1]
        m.w8("sync_connections.num_connections", len(conns))
        stride = L["main"].fsize("sync_connections.storage") // L["main"].fsize("sync_connections.is_mux")
        for i, r in enumerate(conns):
            write_adapter(img, m.a("sync_connections.storage") + i * stride, L["adapter"], r, r["rdc"], r["rdc"])
        Lval = 7
        for i in range(2):
            lv = SW(img, m.a("line_sync_configs_storage") + i * L["line_sync"].size, L["line_sync"])
            lv.w8("num_connections_", 1)
            lv.w8("connection_indices_", i, 0)
            lv.w32("packet_header", SEND_HDRS + 112 * i)
            lv.w32("line_sync_ptr", GSYNC)
            lv.w32("line_sync_val", Lval)
        ls = SW(img, m.a("local_sync_config_storage"), L["local_sync"])
        ls.w32("sync_core_xy_encoding_", SYNC_XY[1] << 6 | SYNC_XY[0], 0)
        ls.w32("sync_core_xy_encoding_", SENDER_XY[1] << 6 | SENDER_XY[0], 1)
        ls.w32("sync_address", LSYNC)
        ls.w32("sync_val", 2)
        img.w32(GSYNC, fs["global"] if running else 11 * Lval)
        img.w32(LSYNC, fs["local"] if running else 6)
        write_result(img, 0xABC00000 if running else 0xABC00001, 0)
        cores[SYNC_XY] = (img, 0, {})
        # ---------------- dispatch core (GO, old program) and a stale test core
        img = Img()  # dispatch: launched by the host (mode host, rd_ptr not advanced), runs for good
        write_head(img, 0, KCB, TEXT_OFF, len(disp.image), KID_DISPATCH, True, mode=1)
        img.put(KCB + TEXT_OFF, disp.image)
        cores[DISPATCH_XY] = (img, 0, {})
        # A receiver of an earlier program, left out of the later ones: in lockstep with the others (rd_ptr 4, slot 3
        # executed last, with nothing enabled), slot 3 holds a launch from 8 programs back, its own latest is slot 1.
        img = Img()
        write_head(img, 3, KCB_STALE_OLD, TEXT_OFF, len(R.image), KID_STALE_OLD, False)
        write_slot(img, 1, KCB_STALE, TEXT_OFF, len(R.image), KID_STALE)
        write_common(img, R, [RECV_PAYLOAD - 128, RECV_PAYLOAD], kcb=KCB_STALE)
        write_result(img, 0xABC00001, NPK)
        cores[STALE_XY] = (img, 0x47800, {})
    return out


# ====================================================================== fake tt-exalens
class FakeLoc:
    def __init__(self, dev, kind, noc0, logical, chan=None):
        self.dev, self.kind, self.noc0, self.logical, self.chan = dev, kind, noc0, logical, chan

    def to(self, cs):
        if cs in ("noc0", "translated"):
            return self.noc0
        if cs == "noc1":
            return (16 - self.noc0[0], 11 - self.noc0[1])
        if cs == "logical":
            return (self.logical, self.kind)
        raise ValueError(cs)

    def noc_read(self, addr, buf):
        data = self.dev.read(self, addr, len(buf))
        buf[:] = data


class FakeDev:
    def __init__(self, chip, idx, tensix, eth_patch=None, dead=()):
        self.chip, self.idx, self.tensix = chip, idx, tensix
        self.eth_patch = eth_patch or {}  # {chan: {addr: u32}} applied over the router L1
        self.dead = set(dead)  # Tensix noc0 xy whose reads fail (an unreachable core)
        self.eth_l1 = {}
        crec = idx["chips"][str(chip)]
        self.eth_recs = {int(ch): rec for ch, rec in crec["cores"].items()}
        xs = [x for x in range(1, 17) if x not in (8, 9)]
        self.tlocs = [FakeLoc(self, "tensix", (x, y), (xs.index(x), y - 2)) for y in range(2, 12) for x in xs]
        self.elocs = [FakeLoc(self, "eth", (20 + ch, 25), (0, ch), ch) for ch in range(14)]
        self.reads = 0

    def get_block_locations(self, kind):
        return {"functional_workers": self.tlocs, "eth": self.elocs}[kind]

    def read(self, loc, addr, n):
        self.reads += 1
        if loc.kind == "eth":
            rec = self.eth_recs.get(loc.chan)
            if addr < 0x80000:
                if rec is None:
                    return bytes(n)
                if loc.chan not in self.eth_l1:
                    b = bytearray(open(os.path.join(ROUTER_CAP, rec["l1"]["file"]), "rb").read())
                    for a, v in self.eth_patch.get(loc.chan, {}).items():
                        struct.pack_into("<I", b, a, v)
                    self.eth_l1[loc.chan] = bytes(b)
                return self.eth_l1[loc.chan][addr : addr + n]
            p1 = (rec or {}).get("regs", {}).get("pass1", {})
            if hc.STREAM_BASE <= addr < hc.STREAM_BASE + 32 * hc.STREAM_STRIDE:
                sid, reg = (addr - hc.STREAM_BASE) // hc.STREAM_STRIDE, (addr % hc.STREAM_STRIDE) // 4
                vals = p1.get("streams", {}).get(str(sid))
                v = vals[list(hc.STREAM_REGS).index(reg)] if vals and reg in hc.STREAM_REGS else 0
                return struct.pack("<I", v or 0)
            if addr == hc.SOFT_RESET_REG:
                return struct.pack("<I", p1.get("soft_reset") or 0)
            if hc.TXQ_BASE <= addr < hc.TXQ_BASE + 3 * hc.TXQ_STRIDE:
                q, off = (addr - hc.TXQ_BASE) // hc.TXQ_STRIDE, (addr - hc.TXQ_BASE) % hc.TXQ_STRIDE
                t = (rec or {}).get("regs", {}).get("txq1", {}).get("txq", {}).get(str(q), {})
                return struct.pack("<I", t.get(hex(off)) or 0)
            raise RuntimeError(f"unexpected eth read {addr:#x}")
        if loc.noc0 in self.dead:
            raise RuntimeError("fake: NoC read timed out")
        img, sr, niu = self.tensix.get(loc.noc0, (None, 0x47800, {}))
        if addr + n <= 0x180000:
            return bytes(img.b[addr : addr + n]) if img else bytes(n)
        if addr == hc.SOFT_RESET_REG:
            return struct.pack("<I", sr)
        for noc, base in enumerate(hc.W_NIU_BASES):
            if base + 0x200 <= addr < base + 0x300:
                return struct.pack("<I", niu.get((noc, addr - base - 0x200), 0))
        raise RuntimeError(f"unexpected tensix read {addr:#x}")


class FakeCD:
    def __init__(self, idx):
        self.idx = idx

    def get_ethernet_connections(self):
        out = {}
        for c, ch, rc, rch in self.idx["eth_connections"]:
            out.setdefault(c, {})[ch] = (rc, rch)
            out.setdefault(rc, {})[rch] = (c, ch)
        return out

    def get_chip_unique_ids(self):
        return {int(c): r["unique_id"] for c, r in self.idx["chips"].items()}

    def get_chip_pci_bdfs(self):
        return {}

    def get_board_type(self, chip):
        return "P150"


class FakeCtx:
    def __init__(self, idx, synth, eth_patch=None, dead=None):
        self.cluster_descriptor = FakeCD(idx)
        self.devices = {
            c: FakeDev(c, idx, synth.get(c, {}), (eth_patch or {}).get(c), (dead or {}).get(c, ()))
            for c in sorted(int(x) for x in idx["chips"])
        }


def run_capture(out_dir, ctx, kcache, extra=()):
    def fake_open(self):
        self.ctx = ctx
        self.index["noc_id"] = 1
        self.index["timings"]["open_s"] = 0.0

    saved = hc.Capture.open_device, sys.argv
    hc.Capture.open_device = fake_open
    sys.argv = ["hang_capture.py", out_dir, "--meta", f"cache={kcache}", "--meta", "label=synthetic"] + list(extra)
    try:
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = hc.main()
    finally:
        hc.Capture.open_device, sys.argv = saved
    return rc, buf.getvalue()


def decode(cap, extra=()):
    o = subprocess.run(
        [PY, os.path.join(HERE, "hang_decode.py"), cap, os.path.join(cap, "cache")] + list(extra),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return o.stdout + o.stderr


def scenario(tmp, name, topo, bins, kcache, faults, expect, forbid=(), eth_patch=None, dead=None, lags=None):
    print(f"== scenario {name} ==")
    synth = build(topo, bins, faults, lags)
    ctx = FakeCtx(topo.idx, synth, eth_patch, dead)
    cap = os.path.join(tmp, f"{name}_hangcap")
    rc, log = run_capture(cap, ctx, kcache)
    open(os.path.join(tmp, f"{name}_capture.log"), "w").write(log)
    shutil.copytree(os.path.join(ROUTER_CAP, "cache"), os.path.join(cap, "cache"), dirs_exist_ok=True)
    shutil.copy(os.path.join(ROUTER_CAP, "compile_cmds.txt"), cap)
    idx = json.load(open(os.path.join(cap, "index.json")))
    nd = sum(len(v) for v in (dead or {}).values())
    errs = [e for e in idx["errors"] if "tensix" not in e.lower() or not dead]
    check(rc == 0 and idx.get("complete") and not errs, f"capture rc {rc}, complete, errors {errs[:3]}")
    if dead:
        check(
            any("consecutive read failures" in e or "tensix" in e for e in idx["errors"]),
            f"unreachable Tensix core recorded as read errors ({len(idx['errors'])} errors), capture went on",
        )
    wch = idx.get("workers", {}).get("chips", {})
    nsel = sum(len(c["cores"]) for c in wch.values())
    nfull = sum(1 for c in wch.values() for r in c["cores"].values() if r.get("full"))
    alive = 8 - sum(1 for v in (dead or {}).values() if len(v) > 100)
    check(
        nsel == alive * 11,
        f"selected {nsel} Tensix cores ({alive} chips x (sender, 7 receivers, sync, dispatch, stale))",
    )
    nrun = sum(
        1 for c in faults.values() for k, v in c.items() if (k in ("sender", "sync") or (isinstance(k, tuple))) and v
    )
    check(nfull == nrun, f"whole L1 for the {nfull} still-running test cores (expected {nrun})")
    check(len(idx.get("workers", {}).get("kernel_elfs", {}).get("files", [])) >= 3, "kernel ELFs copied")
    out = decode(cap)
    open(os.path.join(tmp, f"{name}_decode.txt"), "w").write(out)
    check("decoder error" not in out and "Traceback" not in out, "decode has no decoder error")
    for e in expect:
        check(e in out, f"decode names: {e}")
    for e in forbid:
        check(e not in out, f"decode does not say: {e}")
    q = decode(cap, ["--quiet-routers", "--quiet-workers"])
    check("== Workers not finished" in q and "== Workers: Tensix" not in q, "--quiet-workers keeps only the summaries")
    g = decode(cap, ["--gdb", "/nonexistent/gdb"])
    check("Traceback" not in g and "gdb not found" in g, "without gdb: status-only decode, no traceback")
    n = decode(cap, ["--no-workers"])
    check("== Workers" not in n and "== Anomalies" in n, "--no-workers: router decode only")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default="", help="keep the synthetic captures and decodes in this directory")
    ap.add_argument("--kernel-cache", default="/home/aagarwal/.tt-metal-cache-p1vc1-v3vc-auto")
    args = ap.parse_args()
    tmp = args.keep or tempfile.mkdtemp(prefix="hangw_")
    os.makedirs(tmp, exist_ok=True)
    router_regression(tmp)
    worker_selftest_regression()
    bins = hw.find_kernel_bins([args.kernel_cache])
    if not bins.get("brisc"):
        print(f"no kernel ELFs under {args.kernel_cache}: synthetic scenarios skipped")
    else:
        topo = Topo()
        scenario(
            tmp,
            "clean",
            topo,
            bins,
            args.kernel_cache,
            {},
            [
                "none: every captured test kernel wrote status PASS",
                "the wait is outside the test kernels",
                "== Worker flows: sender vs receiver",
                "all 56 flow(s) complete on both ends",
                "== Worker-router connections",
                "all 32 connection(s) quiet and consistent",
                "in 1 of them the worker's credit word trails by up to the slot count",
                "trails the router read counter",
                "rank 0 mesh 0:",
                "rank 1 mesh 1:",
                "= tt_fabric_test_sender/",
                "dispatch: expected to run",
                "belong to an earlier program",
                "launch slot 1 (its own last kernels), kernel ids [500]",
                "running non-test kernels [cq_dispatch] on 8 core(s)",
            ],
            ["<<<", "credit(s) not delivered", "(no captured sender)"],
            lags={(3, 0): 2},
        )
        c0, c1, c2 = 0, 1, 2
        src1 = 5
        faults = {
            c0: {"sync": {"global": 11 * 7 - 1, "local": 4}, "sender": True, "sender_lsync": 2},
            c1: {("recv", src1): {"got": 57, "holds": 59}},
            c2: {"sender": True, "lost_credit": 1, "niu": True, "sender_lsync": 1},
        }
        r3 = next(r for r in topo.routers[3] if r["wxy"] == SENDER_XY)
        scenario(
            tmp,
            "hang",
            topo,
            bins,
            args.kernel_cache,
            faults,
            [
                "end global sync: global semaphore = 76, waits >= 77 (1 sync packets missing)",
                "local sync 2: waits semaphore == 3",
                "validated 57/100, waits for packet 58",
                "slot for packet 58 holds packet 60's data (overwritten by a later packet",
                "sent all 100 packets, receiver validated 57: 43 not arrived",
                "credit(s) not delivered to the worker",
                "no free router slot",
                "1 non-posted write(s), 0 read(s), 0 atomic(s) not acknowledged",
                "config(s) still to send but the router connection is unused",
                "sent 99/100",
                "validated 100 packets but its sender chip2/tensix(14,11) cfg1 counted only 99 sent",
                f"chip3/eth{r3['chan']} worker channel open to (14,11) = chip3/tensix(14,11) sender go done",
                "still has its worker connection open to this core, which finished",
                "worker finished but the router connection is still open",
                "51 credit(s) not delivered to the worker (worker still running and out of free slots)",
                "60 credit(s) not delivered to the worker (more than the 51 slots in flight at close)",
            ],
            eth_patch={3: {r3["chan"]: {r3["ci"] + 96: 1}}},
            dead={7: [(9 + 1, 5)]},
            lags={(4, 1): 60},
        )
        allxy = [(x, y) for y in range(2, 12) for x in range(1, 17) if x not in (8, 9)]
        out = scenario(
            tmp,
            "deadchip",
            topo,
            bins,
            args.kernel_cache,
            {},
            [
                "Tensix cores in a row unreadable",
                "none: every captured test kernel wrote status PASS",
                "worker scan incomplete: chip(s) [6] partly unreadable",
            ],
            ["Traceback", "the wait is outside the test kernels"],
            dead={6: allxy},
        )
        idx = json.load(open(os.path.join(tmp, "deadchip_hangcap", "index.json")))
        n6 = len(idx["workers"]["chips"]["6"]["scan"]["cores"])
        check(n6 == 4, f"chip 6 scan cut after 4 unreadable cores ({n6} scanned), the other chips scanned in full")
    print(f"\n{len(FAILS)} failure(s)" + (": " + "; ".join(FAILS) if FAILS else "") + f"  (files in {tmp})")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
