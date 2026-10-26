#!/usr/bin/env python3
"""Offline decoder for hang_capture.py dumps of the Blackhole fabric routers (2D, v3 two-ERISC routers).

  hang_decode.py CAPTURE_DIR CACHE_DIR [--log LOG ...] [--all-slots] [--chip C] [--quiet-routers]

CACHE_DIR is the run's TT_METAL_CACHE (or any directory above kernels/fabric_erisc_router_v2/<hash>/, or a
consolidation builds/<build> dir). Per router config hash the decoder reads named_ct_arg_map_generated.h (named
compile-time args) and the ERISC ELF (.elf.xip.elf preferred: the loader's XIP image). The positional compile-time args
(channel buffer bases and slot counts) and the -D defines are not in the cache: they come from logged compile commands
(TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1) in, by priority: --log files, CAPTURE_DIR/*.log and
CAPTURE_DIR/compile_cmds*.txt, fabric_exp/consolidation/offline/cmds.json, fabric_exp/results/*vc1*.log.

Per core: the launch message in the mailbox (0x100) gives the kernel config base and each ERISC's text offset and
runtime-arg offset; the bytes there are matched against every cached ELF image of that ERISC to identify the E0 and
E1 configs. Then, per router:
  - sender channels (E0): free-slots register (stream SENDER_CHANNEL_<pos>_FREE_SLOTS_STREAM_ID, index 297, 17-bit
    field) = slots without an unsent packet: unsent = N - value; range [0, N]; above N (or wrapped near 2^17) is
    over-fill / over-credit. Completion and ack counters (L1, written over Ethernet by the far receiver), connection
    info (the upstream's credit register address and NoC xy, or the worker's), slot headers;
  - receiver channels (E1): packets-received register (TO_RECEIVER_<rx>_PKTS_SENT_ID) = packets arrived and not yet
    retired, range [0, N]; local completion/ack counter block (sent to the far sender); slot headers;
  - E1's downstream connections (runtime args): local copy of each downstream sender channel's free slots
    (VC<v>_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_<c+1>_STREAM_ID) against that channel's own register on the downstream.
Then a cross-router summary: per Ethernet link the sender's view (unsent, completion counters received) against the
receiver's (packets pending, completion counters sent); per router-to-router NoC connection the upstream copy against
the downstream register (downstream - copy = packets the downstream sent and has not credited back; 0 when quiet).
"Inferred" slot positions assume a quiet ring (every retired packet completed); they are marked as such.
"""
import argparse
import glob
import json
import os
import re
import struct
import sys
import traceback

try:
    import numpy as np
except ImportError:  # similarity scoring falls back to pure Python
    np = None

ROOT = "/home/aagarwal/tt-metal"
KDIR = "kernels/fabric_erisc_router_v2/"
L1_SIZE = 0x80000
STREAM_BASE = 0xFFB40000
STREAM_STRIDE = 0x1000
REG_UPDATE_IDX = 270  # STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX
FREE_MASK = (1 << 17) - 1  # REMOTE_DEST_WORDS_FREE_WIDTH = MEM_WORD_ADDR_WIDTH = 17
UNUSED_STREAM = 32  # k_unused_stream_id
MAILBOX = 0x100  # MEM_AERISC_MAILBOX_BASE
LAUNCH_SIZE = 144  # sizeof(launch_msg_t)
GO_MSGS = MAILBOX + 1168
ACTIVE_ETH = 1  # ProgrammableCoreType::ACTIVE_ETH
KCFG_LO, KCFG_HI = 0xF330, 0xF330 + 26 * 1024  # MEM_AERISC_MAP_END, + MEM_ERISC_KERNEL_CONFIG_SIZE
SEARCH_LO, SEARCH_HI = 0x3000, 0x20000  # fallback image search: firmware + kernel config + low buffers
ROUTING_TABLE = 0x6F340  # MEM_AERISC_ROUTING_TABLE_BASE (routing_l1_info_t, 2704 B)
POSTCODE = 0x6FE50
HEARTBEAT = 0x7CC70
DIRS = ["E", "W", "N", "S", "Z"]
EDM_INDEX_TO_DIR = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]
NOC_SEND_TYPES = ["UNICAST_WRITE", "INLINE_WRITE", "ATOMIC_INC", "FUSED_ATOMIC_INC", "SCATTER_WRITE",
                  "MCAST_WRITE", "MCAST_ATOMIC_INC", "UNICAST_READ", "SPARSE_MCAST_WRITE"]
EDM_STATUS = {0xA0B0C0D0: "STARTED", 0xA1B1C1D1: "REMOTE_HANDSHAKE_COMPLETE", 0xA2B2C2D2: "LOCAL_HANDSHAKE_COMPLETE",
              0xA3B3C3D3: "READY_FOR_TRAFFIC", 0xA4B4C4D4: "TERMINATED", 0xB0C0D0E0: "INITIALIZATION_STARTED",
              0xB1C1D1E1: "TXQ_INITIALIZED", 0xB2C2D2E2: "STREAM_REG_INITIALIZED",
              0xB3C3D3E3: "DOWNSTREAM_EDM_SETUP_STARTED", 0xB4C4D4E4: "EDM_VCS_SETUP_COMPLETE",
              0xB6C6D6E6: "WORKER_INTERFACES_INITIALIZED", 0xB7C7D7E7: "ETHERNET_HANDSHAKE_COMPLETE",
              0xB8C8D8E8: "VCS_OPENED", 0xB9C9D9E9: "ROUTING_TABLE_INITIALIZED", 0xBACADAEA: "INITIALIZATION_COMPLETE"}
TERM = {0: "KEEP_RUNNING", 1: "GRACEFULLY_TERMINATE", 2: "IMMEDIATELY_TERMINATE"}
GO_SIGNAL = {0x80: "GO", 0x00: "DONE", 0x40: "INIT", 0xC0: "RESET_READ_PTR", 0xE0: "RESET_READ_PTR_FROM_HOST"}
CONN = {0: "unused", 1: "open", 2: "close-requested"}


def u32(b, a):
    return struct.unpack_from("<I", b, a)[0]


def u16(b, a):
    return struct.unpack_from("<H", b, a)[0]


def h(v):
    return "None" if v is None else f"{v:#x}"


# ====================================================================== cache: configs, ELFs, compile commands
def parse_named(path):
    txt = open(path).read()
    return {k: int(v) for k, v in re.findall(r'\{"([A-Z0-9_]+)",(-?\d+)\}', txt)}


def elf_image(path):
    """The loader's contiguous image: every PT_LOAD's file contents in program-header order (XIP packing)."""
    b = open(path, "rb").read()
    if b[:4] != b"\x7fELF" or b[4] != 1:
        raise ValueError(f"{path}: not an ELF32")
    phoff, phentsize, phnum = u32(b, 0x1C), u16(b, 0x2A), u16(b, 0x2C)
    img = bytearray()
    segs = []
    for i in range(phnum):
        p = phoff + i * phentsize
        ptype, off, vaddr, paddr, filesz, memsz = struct.unpack_from("<6I", b, p)
        if ptype == 1 and filesz:
            seg = b[off:off + filesz]
            seg += b"\0" * (-len(seg) % 4)
            segs.append((vaddr, paddr, len(seg)))
            img += seg
    return bytes(img), segs


class Cfg:
    """One router config hash (one ERISC): named args, defines, positional args and what follows from them."""

    def __init__(self, hsh, d):
        self.hash = hsh
        self.dir = d
        self.n = parse_named(os.path.join(d, "named_ct_arg_map_generated.h"))
        self.erisc = self.n.get("MY_ERISC_ID")
        self.elf = None
        for sub in ("active_erisc", "subordinate_active_erisc"):
            for cand in (f"{d}/{sub}/{sub}.elf.xip.elf", f"{d}/{sub}/{sub}.elf", f"{d}/{sub}.elf"):
                if os.path.exists(cand):
                    self.elf = cand
                    break
            if self.elf:
                break
        self.image = None
        if self.elf:
            try:
                self.image, self.segs = elf_image(self.elf)
            except Exception as e:  # noqa: BLE001
                print(f"warning: {self.elf}: {e}", file=sys.stderr)
        self.defines = None
        self.ct = None
        self.ct_clean = False
        self.cmd_source = None

    def g(self, k, default=None):
        return self.n.get(k, default)

    # ---- positional args
    def set_ct(self, ct, defines, source, clean=True):
        """Positional args and defines from a logged compile command. A clean line replaces one taken from a line
        that another rank's output had cut into; args are kept only if every marker of the layout is in place."""
        if self.ct is not None and (self.ct_clean or not clean):
            return True
        try:
            self._parse_ct(ct)
        except Exception:  # noqa: BLE001 - a cut or foreign line; try the next source
            return False
        self.ct, self.defines, self.cmd_source, self.ct_clean = ct, defines, source, clean
        return True

    def _parse_ct(self, ct):
        ns, nr = self.n["NUM_SENDER_CHANNELS"], self.n["NUM_RECEIVER_CHANNELS"]

        def allocs(i, ns_, nr_):
            assert ct[i] == 0xABCD1234
            ne = ct[i + 1]
            ents = [tuple(ct[i + 2 + 4 * k:i + 6 + 4 * k]) for k in range(ne)]
            m = i + 2 + 4 * ne
            s2e, r2e = ct[m:m + ns_], ct[m + ns_:m + ns_ + nr_]
            assert all(x < ne for x in s2e + r2e)
            return ents, s2e, r2e, 2 + 4 * ne + ns_ + nr_

        ents, s2e, r2e, used = allocs(0, ns, nr)
        assert ct[used] == 0xABADDAD6
        rents, _, rr2e, rused = allocs(used + 1, 0, nr)
        i = used + 1 + rused
        assert ct[i] == 0xABADDAD7
        mr = self.n["MAX_NUM_RECEIVER_CHANNELS"]
        dsn = ct[i + 1:i + 1 + mr]
        assert ct[i + 1 + mr] == 0xABADDAD9
        self.entries, self.s2e, self.r2e = ents, s2e, r2e
        self.sender_base = [ents[e][0] for e in s2e]
        self.sender_slots = [ents[e][1] for e in s2e]
        self.recv_base = [ents[e][0] for e in r2e]
        self.recv_slots = [ents[e][1] for e in r2e]
        self.remote_recv_base = [rents[e][0] for e in rr2e]
        self.remote_recv_slots = [rents[e][1] for e in rr2e]
        self.ds_slots = list(dsn)

    # ---- derived facts (mirror fabric_erisc_router_ct_args.hpp / the router)
    def has(self, define):
        return self.defines is not None and define in self.defines

    @property
    def is_2d(self):
        return self.g("IS_2D_FABRIC", 0) == 1

    @property
    def vc1_active(self):
        if self.defines is not None:
            return "FABRIC_2D_VC1_ACTIVE" in self.defines
        return self.g("ACTUAL_VC1_SENDER_CHANNELS", 0) > 0

    @property
    def crossover(self):
        return self.has("FABRIC_2D_VC0_CROSSOVER_TO_VC1")

    @property
    def header_size(self):
        r = 36
        if self.defines and "FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE" in self.defines:
            try:
                r = int(self.defines["FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE"])
            except ValueError:
                pass
        return (60 + r + 15) // 16 * 16, r

    def vc_of_sender(self, ch):
        v0, v1 = self.g("ACTUAL_VC0_SENDER_CHANNELS"), self.g("ACTUAL_VC1_SENDER_CHANNELS")
        return 0 if ch < v0 else (1 if ch < v0 + v1 else 2)

    def pos_of_compact(self, ch):
        v0, v1 = self.g("ACTUAL_VC0_SENDER_CHANNELS"), self.g("ACTUAL_VC1_SENDER_CHANNELS")
        if ch < v0:
            return ch
        if ch < v0 + v1:
            return self.g("VC1_FABRIC_POSITION_START") + ch - v0
        return self.g("VC2_FABRIC_POSITION_START") + ch - v0 - v1

    def compact_of_pos(self, pos):
        v0, v1 = self.g("ACTUAL_VC0_SENDER_CHANNELS"), self.g("ACTUAL_VC1_SENDER_CHANNELS")
        p1, p2 = self.g("VC1_FABRIC_POSITION_START"), self.g("VC2_FABRIC_POSITION_START")
        if pos < p1:
            return pos if pos < v0 else None
        if pos < p2:
            return v0 + pos - p1 if pos - p1 < v1 else None
        return v0 + v1 + pos - p2 if pos - p2 < self.g("ACTUAL_VC2_SENDER_CHANNELS", 0) else None

    def sender_stream(self, ch):
        return self.g(f"SENDER_CHANNEL_{self.pos_of_compact(ch)}_FREE_SLOTS_STREAM_ID")

    def sender_serviced(self, ch):
        return self.g(f"IS_SENDER_CHANNEL_{ch}_SERVICED", 0) == 1

    def receiver_serviced(self, rx):
        return self.g(f"IS_RECEIVER_CHANNEL_{rx}_SERVICED", 0) == 1

    def ds_pos(self, vc, compact):
        my = self.g("MY_DIRECTION")
        ds = EDM_INDEX_TO_DIR[my][compact]
        if vc == 0:
            return my if my > ds else my + 1
        p1 = self.g("VC1_FABRIC_POSITION_START")
        return p1 - 1 + my if my > ds else p1 + my

    def ds_sets(self):
        """E1's downstream connection sets (E1Downstreams<VC>): VC0's, and VC1's when VC1 is active."""
        sets = []
        for v in [0, 1] if self.vc1_active else [0]:
            pos_vc = 1 if (v == 1 or self.crossover) else 0
            size = self.g(f"VC{v}_DOWNSTREAM_EDM_SIZE", 0)
            # Receiver channels that forward through this set: rx0 through VC1's under crossover (kernel_main's
            # run_receiver_channel_step<0, ..., DownstreamSenderVC1T> and E1's ds_ch0 = ds_vc1), rx1 through VC1's.
            # The VC0 set of a crossover router is still built and opened (its copies initialised) but never used.
            users = [rx for rx in range(self.g("NUM_RECEIVER_CHANNELS", 0))
                     if self.receiver_serviced(rx) and (1 if (rx == 0 and self.crossover) else rx) == v]
            sets.append({
                "vc": v, "size": size, "num_slots": self.ds_slots[v] if self.ct else None, "users": users,
                "remote_pos": [self.ds_pos(pos_vc, i) for i in range(size)],
                "local_stream": [self.g(f"VC{v}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{i + 1}_STREAM_ID") for i in range(size)],
                "dirs": [EDM_INDEX_TO_DIR[self.g("MY_DIRECTION")][i] for i in range(size)],
                "receiver_serviced": self.receiver_serviced(v),
            })
        return sets


def find_cfg_dirs(cache):
    dirs = set()
    for pat in (f"{cache}/*/{KDIR}*", f"{cache}/{KDIR}*", f"{cache}/*/*/{KDIR}*", f"{cache}/fabric_erisc_router_v2/*",
                f"{cache}/*"):
        for d in glob.glob(pat):
            if os.path.isfile(os.path.join(d, "named_ct_arg_map_generated.h")):
                dirs.add(os.path.normpath(d))
    return sorted(dirs)


def load_cfgs(cache):
    cfgs = {}
    for d in find_cfg_dirs(cache):
        hsh = os.path.basename(d)
        if hsh in cfgs:
            continue
        cfgs[hsh] = Cfg(hsh, d)
    return cfgs


CMD_RE = re.compile(r"g\+\+ compile cmd: (.*?)\s*\(build\.cpp:\d+\)\s*$")


def parse_cmd(cmd):
    m = re.search(r'-DFULL_KERNEL_NAME="?fabric_erisc_router_v2/(\d+)/', cmd)
    c = re.search(r"-DKERNEL_COMPILE_TIME_ARGS=([0-9,]+)", cmd)
    if not m or not c:
        return None
    ct = [int(x) for x in c.group(1).strip(",").split(",") if x]
    defines = {}
    for name, val in re.findall(r"(?:^|\s)-D([A-Za-z_][A-Za-z0-9_]*)(?:=(\S*))?", cmd):
        defines.setdefault(name, val)
    return m.group(1), ct, defines


def attach_compile_cmds(cfgs, capture_dir, logs):
    need = lambda: [c for c in cfgs.values() if c.ct is None or not c.ct_clean]  # noqa: E731

    def feed_lines(lines, src):
        for line in lines:
            if "fabric_erisc_router_v2/" not in line or "KERNEL_COMPILE_TIME_ARGS" not in line:
                continue
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = CMD_RE.search(line)
            cmd = m.group(1) if m else line
            r = parse_cmd(cmd)
            if r and r[0] in cfgs:
                clean = m is not None and "<stdout>" not in cmd and "<stderr>" not in cmd
                cfgs[r[0]].set_ct(r[1], r[2], src if clean else src + " (cut line)", clean)

    sources = list(logs)
    sources += sorted(glob.glob(os.path.join(capture_dir, "*.log")))
    sources += sorted(glob.glob(os.path.join(capture_dir, "compile_cmds*.txt")))
    for s in sources:
        if not need():
            break
        try:
            with open(s, errors="replace") as f:
                feed_lines(f, os.path.basename(s))
        except OSError as e:
            print(f"warning: {s}: {e}", file=sys.stderr)
    db = os.path.join(ROOT, "fabric_exp/consolidation/offline/cmds.json")
    if need() and os.path.exists(db):
        d = json.load(open(db))
        for opt, hs in d.items():
            for hsh, per in hs.items():
                if hsh in cfgs and (cfgs[hsh].ct is None or not cfgs[hsh].ct_clean):
                    for e in per.values():
                        if "compile" in e:
                            r = parse_cmd(e["compile"])
                            if r:
                                cfgs[hsh].set_ct(r[1], r[2], f"offline/cmds.json[{opt}]", True)
    if need():
        for s in sorted(glob.glob(os.path.join(ROOT, "fabric_exp/results/*vc1*.log")), key=os.path.getmtime,
                        reverse=True):
            if not need():
                break
            wanted = {c.hash for c in need()}
            with open(s, errors="replace") as f:
                feed_lines((l for l in f if any(w in l for w in wanted)), os.path.basename(s))


# ====================================================================== capture
class Core:
    def __init__(self, cap_dir, chip, chan, rec, stream_regs=(4, 10, 297)):
        self.chip, self.chan, self.rec = chip, chan, rec
        self.reg_pos = {r: i for i, r in enumerate(stream_regs)}
        self.l1 = None
        l1 = rec.get("l1")
        if l1 and os.path.exists(os.path.join(cap_dir, l1["file"])):
            self.l1 = open(os.path.join(cap_dir, l1["file"]), "rb").read()
            self.missing = l1.get("missing", [])
        self.regs = rec.get("regs", {})
        self.e0 = self.e1 = None
        self.match = {}
        self.notes = []

    @property
    def name(self):
        return f"chip{self.chip}/eth{self.chan}"

    def captured(self, a, n):
        """[a, a + n) lies in the L1 image and outside every range the capture could not read."""
        if self.l1 is None or a < 0 or a + n > len(self.l1):
            return False
        return all(a + n <= m[0] or a >= m[0] + m[1] for m in getattr(self, "missing", []))

    def coord(self, cs="translated"):
        v = self.rec.get(cs)
        return tuple(v) if v else None

    def stream(self, sid, reg=297, pas="pass1"):
        if sid is None or sid >= 32:
            return None
        p = self.regs.get(pas)
        if not p:
            return None
        vals = p["streams"].get(str(sid))
        if not vals:
            return None
        return vals[self.reg_pos[reg]]

    def w(self, a):
        if a is None or not self.captured(a, 4):
            return None
        return u32(self.l1, a)

    def cfg(self):
        return self.e0 or self.e1


def fmt_free(v, n):
    """A free-slots / count register against its range [0, n]: (text, anomaly or None)."""
    if v is None:
        return "?", None
    f = v & FREE_MASK
    hi = v >> 17
    s = f"{f}" + (f" (raw {v:#x})" if hi else "")
    if n is None:
        return s, None
    if f <= n:
        return s, None
    if f > FREE_MASK - 256:
        return s, f"WRAPPED below 0 by {FREE_MASK + 1 - f} (more decrements than increments: over-fill)"
    return s, f"OUT OF RANGE: {f} > {n} slots (more increments than decrements: over-credit)"


def decode_noc(a):
    return f"({(a >> 36) & 0x3F},{(a >> 42) & 0x3F})@{a & ((1 << 36) - 1):#x}"


def decode_header(b, off, cfg):
    """One packet header (HybridMeshPacketHeaderT) at L1 offset off: a short text, and a plausibility flag."""
    hs, rbs = cfg.header_size if cfg else (96, 36)
    if b is None or off + hs > len(b):
        return "<outside capture>", False
    raw = b[off:off + hs]
    if not any(raw):
        return "<zero>", False
    size, typ, src = u16(raw, 40), raw[42], raw[43]
    tname = NOC_SEND_TYPES[typ] if typ < len(NOC_SEND_TYPES) else f"type{typ}"
    q = lambda i: struct.unpack_from("<Q", raw, i)[0]  # noqa: E731
    if typ == 0:
        cmd = f"dst {decode_noc(q(0))}"
    elif typ == 1:
        cmd = f"dst {decode_noc(q(0))} val {u32(raw, 8):#x}"
    elif typ == 2:
        cmd = f"sem {decode_noc(q(0))} inc {u32(raw, 8)} flush {raw[12]}"
    elif typ == 3:
        cmd = f"dst {decode_noc(q(0))} sem {decode_noc(q(8))} inc {u32(raw, 16)} flush {raw[20]}"
    elif typ == 4:
        cnt = raw[38]
        cmd = "chunks " + ", ".join(decode_noc(q(8 * i)) for i in range(min(cnt, 4))) + \
              f" sizes {[u16(raw, 32 + 2 * i) for i in range(3)]} enc {raw[39]:#x}"
    elif typ == 5:
        cmd = f"mcast addr {u32(raw, 0):#x} start ({raw[4]},{raw[5]}) size ({raw[6]},{raw[7]})"
    else:
        cmd = "cmd " + raw[:40].hex()
    my, mx = (cfg.g("MESH_Y_SIZE", 0), cfg.g("MESH_X_SIZE", 0)) if cfg else (0, 0)
    rb = raw[48:48 + rbs]
    act = lambda a: "".join(DIRS[i] for i in range(5) if a >> i & 1) + ("L" if a & 0x20 else "") or "."  # noqa: E731
    route = "Y[" + " ".join(act(x) for x in rb[:my]) + "] X[" + " ".join(act(x) for x in rb[my:my + mx]) + "]"
    dchip, dmesh = u16(raw, 48 + rbs), u16(raw, 50 + rbs)
    mc = [u16(raw, 52 + rbs + 2 * i) for i in range(4)]
    plaus = typ <= 8 and src < 16 and size <= (cfg.g("CHANNEL_BUFFER_SIZE", 1 << 16) if cfg else 1 << 16)
    txt = (f"{tname} size {size} src_ch {src} {cmd} | dst mesh {dmesh} chip {dchip} route {route}"
           + (f" mcast {mc}" if any(mc) else "") + (f" rf {u32(raw, 44):#x}" if u32(raw, 44) else ""))
    return txt, plaus


# ====================================================================== matching and per-core decode
def mailbox(core):
    b = core.l1
    if b is None:
        return None
    rd = u32(b, MAILBOX + 12)
    out = {"rd_ptr": rd, "go_signal": b[GO_MSGS + 3], "launch": []}
    for i in range(8):
        lb = MAILBOX + 16 + LAUNCH_SIZE * i
        out["launch"].append({
            "kcb": u32(b, lb + 4 * ACTIVE_ETH),
            "rta": [u16(b, lb + 28 + 4 * p) for p in range(5)],
            "crta": [u16(b, lb + 30 + 4 * p) for p in range(5)],
            "text_off": [u32(b, lb + 52 + 4 * p) for p in range(5)],
            "text_size": [u32(b, lb + 72 + 4 * p) for p in range(5)],
            "host_id": u32(b, lb + 108), "enables": u32(b, lb + 112),
        })
    return out


def similarity(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    if np is not None:
        return float(np.mean(np.frombuffer(a, np.uint8) == np.frombuffer(b, np.uint8)))
    return sum(x == y for x, y in zip(a, b)) / len(a)


def match_core(core, cfgs_by_erisc):
    mb = core.mb = mailbox(core)
    if mb is None:
        core.notes.append("no L1 captured")
        return
    rd = mb["rd_ptr"] if mb["rd_ptr"] < 8 else 0
    launch = mb["launch"][rd]
    core.launch = launch
    for p, attr in ((0, "e0"), (1, "e1")):
        cands = cfgs_by_erisc.get(p, [])
        addr = launch["kcb"] + launch["text_off"][p]
        best = []
        if KCFG_LO <= addr < L1_SIZE:
            for c in cands:
                img = c.image
                if img and addr + len(img) <= len(core.l1) and core.l1[addr:addr + len(img)] == img:
                    best.append((1.0, c, addr))
        how = "exact at launch-message address"
        if not best:
            # Not where the launch message says: search the kernel config window for each image's opening bytes.
            for c in cands:
                img = c.image
                if not img:
                    continue
                anchor = img[:64]
                a = core.l1.find(anchor, SEARCH_LO, SEARCH_HI)
                while a >= 0:
                    s = similarity(core.l1[a:a + len(img)], img)
                    if s > 0.9:
                        best.append((s, c, a))
                    a = core.l1.find(anchor, a + 4, SEARCH_HI)
            how = "anchor search"
        if not best and KCFG_LO <= addr < L1_SIZE:
            scored = [(similarity(core.l1[addr:addr + len(c.image)], c.image), c, addr) for c in cands if c.image]
            scored.sort(key=lambda t: -t[0])
            best = scored[:1] if scored and scored[0][0] > 0.5 else []
            how = "best similarity at launch-message address"
        if not best:
            core.match[attr] = {"how": "no match", "addr": addr, "ties": []}
            continue
        top = max(s for s, _, _ in best)
        ties = [(c, a) for s, c, a in best if s == top]
        # Configs with byte-identical images differ only in args the code does not use (MY_ETH_CHANNEL,
        # PEER_ERISC_NOC_X/Y, EDM_NOC_VC on E0; EDM_CHANNELS_MASK, LOCAL_HANDSHAKE_MASTER_ETH_CHAN on E1): name the one
        # that fits this core. Decoding is the same for all of them.
        xy = core.coord()

        def fit(t):
            c = t[0]
            return (4 * (c.g("MY_ETH_CHANNEL") == core.chan)
                    + 2 * (xy is not None and (c.g("PEER_ERISC_NOC_X"), c.g("PEER_ERISC_NOC_Y")) == xy)
                    + 1 * (c.g("EDM_CHANNELS_MASK") == core.chip_mask))

        ties.sort(key=lambda t: (-fit(t), t[0].hash))
        c, a = ties[0]
        setattr(core, attr, c)
        core.match[attr] = {"how": how, "score": top, "addr": a, "hash": c.hash,
                            "ties": [t[0].hash for t in ties if t[0] is not c],
                            "size_ok": launch["text_size"][p] in (0, len(c.image))}


def parse_rta(core, cfg, proc):
    """Runtime args of one ERISC in kernel_main's layout (2D)."""
    if core.l1 is None or not hasattr(core, "launch"):
        return None
    base = core.launch["kcb"] + core.launch["rta"][proc]
    if not (0 < base < L1_SIZE - 640):
        return None
    words = [u32(core.l1, base + 4 * i) for i in range(160)]
    ms = cfg.g("MAX_NUM_SENDER_CHANNELS")
    i = 0
    r = {"base": base, "conn_sem": words[0:ms], "buf_idx": words[ms:2 * ms]}
    i = 2 * ms
    if not cfg.is_2d:
        r["raw"] = words[i:i + 24]
        return r
    sets = []
    for v in [0, 1] if cfg.vc1_active else [0]:
        nds = cfg.g(f"NUM_DOWNSTREAM_SENDERS_VC{v}", 0)
        s = {"mask": words[i]}
        i += 1
        s["bases"] = words[i:i + nds]
        i += nds
        s["noc_x"], s["noc_y"] = words[i], words[i + 1]
        i += 2
        s["reg"] = words[i:i + nds]
        i += nds
        s["loc"] = words[i:i + nds]
        i += nds
        s["bufidx"] = words[i:i + nds]
        i += nds
        sets.append(s)
    r["sets"] = sets
    r["teardown"] = words[i:i + ms]
    i += ms
    r["tensix_relay"] = words[i]
    return r


# ====================================================================== report
class Report:
    def __init__(self, args):
        self.args = args
        self.anoms = []

    def flag(self, where, what):
        self.anoms.append(f"{where}: {what}")
        return "  <<< " + what

    @staticmethod
    def p(*a):
        print(*a)


def router_report(rep, core, by_coord, cores):
    P = rep.p
    e0, e1 = core.e0, core.e1
    cfg = core.cfg()
    peer = core.rec.get("connected_to")
    hdr = f"=== {core.name} noc0 {core.coord('noc0')} xlat {core.coord('translated')}"
    if peer:
        hdr += f" <-> chip{peer[0]}/eth{peer[1]}"
    if cfg:
        hdr += f" | facing {DIRS[cfg.g('MY_DIRECTION')]}" + (" intermesh" if cfg.g("IS_INTERMESH_ROUTER") else "")
    P(hdr)
    for attr in ("e0", "e1"):
        m = core.match.get(attr)
        if m:
            extra = f" ties {m['ties']}" if m.get("ties") else ""
            src = getattr(getattr(core, attr), "cmd_source", None) if getattr(core, attr) else None
            P(f"  {attr.upper()}: {m.get('hash', '-')} ({m['how']}"
              + (f", {m['score']:.4f}" if "score" in m else "") + f", text @{m['addr']:#x}){extra}"
              + ("" if src else ("  [no positional args: buffer layout unknown]" if getattr(core, attr) else "")))
    for n in core.notes:
        P(f"  note: {n}")
    if core.l1 is None:
        return
    if cfg is None:
        P("  (no router config identified; mailbox: " + json.dumps(getattr(core, "launch", None)) + ")")
        return
    # identity cross-checks
    if cfg.g("MY_ETH_CHANNEL") != core.chan:
        P(rep.flag(core.name, f"config MY_ETH_CHANNEL {cfg.g('MY_ETH_CHANNEL')} != core channel {core.chan}"))
    pe = (cfg.g("PEER_ERISC_NOC_X"), cfg.g("PEER_ERISC_NOC_Y"))
    if core.coord() and pe != core.coord():
        P(f"  note: PEER_ERISC_NOC_X/Y {pe} != translated {core.coord()}")
    if e0 and e1 and (e0.ct and e1.ct and e0.ct != e1.ct):
        P(rep.flag(core.name, "E0 and E1 configs carry different positional args (not one router?)"))
    # status words
    st = core.w(cfg.g("EDM_STATUS_PTR_ADDR"))
    term = core.w(cfg.g("TERMINATION_SIGNAL_ADDR"))
    hb1 = core.regs.get("pass1", {}).get("l1_words", {}).get(hex(HEARTBEAT))
    hb2 = core.regs.get("pass2", {}).get("l1_words", {}).get(hex(HEARTBEAT))
    sr = core.regs.get("pass1", {}).get("soft_reset")
    rt = core.l1[ROUTING_TABLE:ROUTING_TABLE + 2704]
    mesh = f"mesh {u16(rt, 32)} dev {u16(rt, 34)} (y,x)=({rt[2700]},{rt[2701]}) of {rt[2702]}x{rt[2703]}"
    P(f"  status {EDM_STATUS.get(st, h(st))}, termination {TERM.get(term, h(term))}, "
      f"go {GO_SIGNAL.get(core.mb['go_signal'], hex(core.mb['go_signal']))}, postcode {h(core.w(POSTCODE))}, {mesh}")
    live = "?"
    if hb1 is not None and hb2 is not None:
        live = "MOVING" if hb1 != hb2 else "static"
    P(f"  heartbeat {h(hb1)} -> {h(hb2)} ({live}); soft reset {h(sr)}"
      + ("" if sr is None else f" (ERISC0 {'in reset' if sr >> 11 & 1 else 'running'},"
                               f" ERISC1 {'in reset' if sr >> 12 & 1 else 'running'})"))
    for q in range(3):
        t1 = (core.regs.get("txq1") or core.regs.get("pass1", {})).get("txq", {}).get(str(q))
        t2 = (core.regs.get("txq2") or core.regs.get("pass2", {})).get("txq", {}).get(str(q))
        if not t1:
            continue
        busy = t1.get("0x8")
        ongoing = None if busy is None else (busy >> 16) & 1
        cnts = " ".join(f"{k} {t1.get(o)}" + (f"->{t2.get(o)}" if t2 and t2.get(o) != t1.get(o) else "")
                        for k, o in (("xfer", "0x30"), ("pkt_start", "0x34"), ("pkt_end", "0x3c"),
                                     ("words", "0x40"), ("resend", "0x68")))
        P(f"  TXQ{q}: cmd {h(t1.get('0x4'))} status {h(busy)} (ongoing {ongoing}) {cnts} "
          f"seq_to {t1.get('0x4c')} accept_ahead {t1.get('0x6c')}")
    sender_report(rep, core, by_coord)
    receiver_report(rep, core, cores)
    downstream_report(rep, core, by_coord)


def slot_lines(rep, core, cfg, base, n, pending, label):
    hs = cfg.header_size[0]
    stride = cfg.g("CHANNEL_BUFFER_SIZE")
    for i in range(n):
        off = base + i * stride
        txt, pl = decode_header(core.l1, off, cfg) if core.captured(off, hs) else ("<not captured>", False)
        mark = "*" if i in pending else " "
        rep.p(f"      {mark}[{i}] @{off:#x}: {txt}")
    if pending:
        rep.p(f"      (* = {label})")


def sender_report(rep, core, by_coord):
    e0 = core.e0
    if not e0:
        return
    P = rep.p
    P("  sender channels (E0)  [free = slots without an unsent packet; unsent = N - free]")
    rcc = e0.g("TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR")
    rac = e0.g("TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR")
    for ch in range(e0.g("NUM_SENDER_CHANNELS")):
        if not e0.sender_serviced(ch):
            continue
        sid = e0.sender_stream(ch)
        n = e0.sender_slots[ch] if e0.ct else None
        v1, v2 = core.stream(sid), core.stream(sid, pas="pass2")
        txt, bad = fmt_free(v1, n)
        moved = "" if v2 is None or v2 == v1 else f" -> {v2 & FREE_MASK} (moving)"
        unsent = (n - (v1 & FREE_MASK)) if (n is not None and v1 is not None and not bad) else None
        comp = core.w(rcc + 4 * ch) if rcc else None
        ack = core.w(rac + 4 * ch) if rac else None
        ci = e0.g(f"LOCAL_SENDER_CH_{ch}_CONN_INFO_ADDR")
        wsem, wtd, wxy, rdc = core.w(ci), core.w(ci + 16), core.w(ci + 32), core.w(ci + 48)
        conn = core.w(ci + 96)
        up = ""
        if wsem is not None and STREAM_BASE <= wsem < STREAM_BASE + 32 * STREAM_STRIDE:
            us = (wsem - STREAM_BASE) // STREAM_STRIDE
            ux, uy = wxy & 0xFFFF, wxy >> 16
            uc = by_coord.get((core.chip, ux, uy))
            up = f"upstream {uc.name if uc else f'({ux},{uy})?'} copy stream {us}"
            core.upstream = getattr(core, "upstream", {})
            core.upstream[ch] = (uc, us)
        elif wxy:
            up = f"worker ({wxy & 0xFFFF},{wxy >> 16}) sem {h(wsem)} edm_read_counter {rdc}"
        line = (f"   ch{ch} vc{e0.vc_of_sender(ch)} pos{e0.pos_of_compact(ch)} stream {sid} N {n} free {txt}{moved}"
                f" unsent {unsent} | completions rcvd {comp}" + (f" acks {ack}" if ack else "")
                + f" | conn {CONN.get(conn, h(conn))} {up}")
        if bad:
            line += rep.flag(f"{core.name} E0 ch{ch} (stream {sid})", bad)
        P(line)
        if e0.ct and (rep.args.all_slots or bad or (unsent or 0) > 0):
            first = (comp % n) if comp is not None and n else None
            pend = set() if first is None or not unsent else {(first + k) % n for k in range(unsent)}
            slot_lines(rep, core, e0, e0.sender_base[ch], n, pend,
                       "inferred unsent: starting at completions mod N, assumes nothing in flight")


def far_e0(core, cores):
    peer = core.rec.get("connected_to")
    if not peer:
        return None
    pc = cores.get((peer[0], peer[1]))
    return pc.e0 if pc else None


def receiver_report(rep, core, cores):
    e1 = core.e1
    if not e1:
        return
    P = rep.p
    P("  receiver channels (E1)  [pending = packets arrived and not retired]")
    lrc = e1.g("LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR")
    lac = e1.g("LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR")
    ns = e1.g("NUM_SENDER_CHANNELS")
    comps = [core.w(lrc + 4 * i) for i in range(ns)] if lrc else []
    acks = [core.w(lac + 4 * i) for i in range(ns)] if lac else []
    fe0 = far_e0(core, cores) or e1
    for rx in range(e1.g("NUM_RECEIVER_CHANNELS")):
        if not e1.receiver_serviced(rx):
            continue
        sid = e1.g(f"TO_RECEIVER_{rx}_PKTS_SENT_ID")
        n = e1.recv_slots[rx] if e1.ct else None
        v1, v2 = core.stream(sid), core.stream(sid, pas="pass2")
        txt, bad = fmt_free(v1, n)
        moved = "" if v2 is None or v2 == v1 else f" -> {v2 & FREE_MASK} (moving)"
        pend_n = (v1 & FREE_MASK) if v1 is not None and not bad else None
        chs = [c for c in range(ns) if fe0.vc_of_sender(c) == rx] if fe0 else []
        tot = sum(comps[c] for c in chs if c < len(comps) and comps[c] is not None)
        line = (f"   rx{rx} stream {sid} N {n} pending {txt}{moved} | completions credited (far sender ch "
                f"{chs}) {[comps[c] for c in chs if c < len(comps)]} total {tot}")
        if bad:
            line += rep.flag(f"{core.name} E1 rx{rx} (stream {sid})", bad)
        P(line)
        if e1.ct and (rep.args.all_slots or bad or (pend_n or 0) > 0):
            first = tot % n if n else None
            pend = set() if first is None or not pend_n else {(first + k) % n for k in range(pend_n)}
            slot_lines(rep, core, e1, e1.recv_base[rx], n, pend,
                       "inferred pending: from completions credited mod N, assumes every retired packet completed")
    if any(acks):
        P(f"   local ack counters {acks}")


def downstream_report(rep, core, by_coord):
    e1 = core.e1
    if not e1 or not e1.is_2d:
        return
    P = rep.p
    rta = core.rta1 = parse_rta(core, e1, 1)
    if rta is None:
        P("  (no runtime args for E1)")
        return
    P("  downstream connections (E1 local copy vs the downstream sender channel's own free-slots register)")
    core.ds = []
    for s, rs in zip(e1.ds_sets(), rta["sets"]):
        active = bool(s["users"])
        if not active and rs["mask"]:
            P(f"   vc{s['vc']} set: not used for forwarding (no receiver channel forwards through it"
              + (", VC0-to-VC1 crossover" if e1.crossover else "") + "); listed without checks")
        dense = 0
        for c in range(s["size"]):
            if not (rs["mask"] >> c) & 1:
                continue
            nx, ny = (rs["noc_x"] >> (8 * dense)) & 0xFF, (rs["noc_y"] >> (8 * dense)) & 0xFF
            d = by_coord.get((core.chip, nx, ny))
            ring, loc = rs["bases"][dense], rs["loc"][dense]
            dense += 1
            ls = s["local_stream"][c]
            pos = s["remote_pos"][c]
            rstream = e1.g(f"SENDER_CHANNEL_{pos}_FREE_SLOTS_STREAM_ID")
            n = s["num_slots"]
            cv1, cv2 = core.stream(ls), core.stream(ls, pas="pass2")
            ctxt, cbad = fmt_free(cv1, n)
            dch, dv1, dtxt, dbad, extra = None, None, "?", None, ""
            if d is not None:
                dv1 = d.stream(rstream)
                dtxt, dbad = fmt_free(dv1, n)
                if d.e0:
                    dch = d.e0.compact_of_pos(pos)
                    if dch is not None:
                        if d.e0.ct and d.e0.sender_base[dch] != ring:
                            extra += f" [ring base {ring:#x} != downstream ch{dch} base {d.e0.sender_base[dch]:#x}]"
                        if d.e0.g(f"LOCAL_SENDER_CH_{dch}_CONN_INFO_ADDR") != loc:
                            extra += f" [location info {loc:#x} != downstream conn info]"
                        wsem = d.w(d.e0.g(f"LOCAL_SENDER_CH_{dch}_CONN_INFO_ADDR"))
                        want = STREAM_BASE + ls * STREAM_STRIDE + REG_UPDATE_IDX * 4
                        if wsem != want:
                            extra += f" [downstream conn info credit addr {h(wsem)} != {want:#x}]"
            diff = None
            if cv1 is not None and dv1 is not None and not cbad and not dbad:
                diff = (dv1 & FREE_MASK) - (cv1 & FREE_MASK)
            if not active:
                cbad = dbad = None
                extra = ""
            core.ds.append({"set": s["vc"], "c": c, "dir": s["dirs"][c], "d": d, "dch": dch, "n": n, "copy": cv1,
                            "ds": dv1, "diff": diff, "stream": ls, "rstream": rstream, "active": active})
            moved = "" if cv2 is None or cv2 == cv1 else f"->{cv2 & FREE_MASK}"
            line = (f"   vc{s['vc']} slot{c} {DIRS[s['dirs'][c]]}: copy stream {ls} = {ctxt}{moved} | "
                    f"{d.name if d else f'({nx},{ny})?'} ch{dch} stream {rstream} = {dtxt} | N {n} "
                    f"| ds - copy = {diff}{extra}")
            if cbad:
                line += rep.flag(f"{core.name} E1 copy stream {ls} (vc{s['vc']} {DIRS[s['dirs'][c]]})", cbad)
            if dbad:
                line += rep.flag(f"{d.name if d else '?'} stream {rstream}", dbad)
            if not active:
                line += "  (inactive set)"
            elif diff is not None and diff < 0:
                line += rep.flag(f"{core.name} -> {d.name if d else '?'} vc{s['vc']}",
                                 f"copy {cv1 & FREE_MASK} > downstream free {dv1 & FREE_MASK}: a forward's "
                                 f"doorbell (-1 on the downstream register) lost, or a credit applied twice")
            P(line)


def link_summary(rep, cores, conns):
    P = rep.p
    P("\n== Ethernet links: sender E0 view vs receiver E1 view ==")
    P("   (completions: counter the sender has received / counter the receiver last sent; equal when quiet)")
    for a in sorted(cores.values(), key=lambda c: (c.chip, c.chan)):
        peer = a.rec.get("connected_to")
        if not peer or not a.e0:
            continue
        b = cores.get((peer[0], peer[1]))
        if b is None or not b.e1:
            P(f"  {a.name} -> chip{peer[0]}/eth{peer[1]}: receiver not decoded")
            continue
        e0, e1 = a.e0, b.e1
        rcc = e0.g("TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR")
        lrc = e1.g("LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR")
        parts, bad = [], []
        for ch in range(e0.g("NUM_SENDER_CHANNELS")):
            if not e0.sender_serviced(ch):
                continue
            s = a.w(rcc + 4 * ch) if rcc else None
            r = b.w(lrc + 4 * ch) if lrc else None
            n = e0.sender_slots[ch] if e0.ct else None
            v = a.stream(e0.sender_stream(ch))
            uns = n - (v & FREE_MASK) if (n is not None and v is not None and (v & FREE_MASK) <= n) else "?"
            parts.append(f"ch{ch}:{s}/{r} unsent {uns}")
            if s != r:
                bad.append(f"ch{ch} sender has {s}, receiver sent {r}")
        pend = []
        for rx in range(e1.g("NUM_RECEIVER_CHANNELS")):
            if e1.receiver_serviced(rx):
                v = b.stream(e1.g(f"TO_RECEIVER_{rx}_PKTS_SENT_ID"))
                pend.append(f"rx{rx} {None if v is None else v & FREE_MASK}")
        line = f"  {a.name} -> {b.name}: pending at receiver [{', '.join(pend)}] | " + "  ".join(parts)
        if bad:
            line += rep.flag(f"link {a.name}->{b.name}", "completion counters differ: " + "; ".join(bad))
        P(line)


def noc_summary(rep, cores):
    P = rep.p
    P("\n== Router-to-router NoC connections: upstream E1 copy vs downstream E0 channel register ==")
    P("   (ds - copy = packets the downstream sent over Ethernet and has not credited back; 0 when quiet)")
    for u in sorted(cores.values(), key=lambda c: (c.chip, c.chan)):
        for e in getattr(u, "ds", []):
            d = e["d"]
            tag = "" if e["active"] else "  (inactive set)"
            if e["active"] and e["diff"] is not None and e["diff"] > 0 and d is not None and d.e0 and e["dch"] is not None:
                # Has the downstream's far receiver completed everything this channel sent?
                peer = d.rec.get("connected_to")
                b = cores.get((peer[0], peer[1])) if peer else None
                if b is not None and b.e1:
                    r = b.w(b.e1.g("LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR") + 4 * e["dch"])
                    s = d.w(d.e0.g("TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR") + 4 * e["dch"])
                    pend = [b.stream(b.e1.g(f"TO_RECEIVER_{rx}_PKTS_SENT_ID"))
                            for rx in range(b.e1.g("NUM_RECEIVER_CHANNELS")) if b.e1.receiver_serviced(rx)]
                    tag = f" | far receiver {b.name}: completions sent {r}, received by {d.name} {s}, pending {pend}"
                    if r == s and all((p or 0) & FREE_MASK == 0 for p in pend):
                        tag += rep.flag(f"{u.name} -> {d.name} ch{e['dch']}",
                                        f"{e['diff']} credits not returned to the upstream copy although the far "
                                        f"receiver is drained and its completions reached {d.name}: its E0 has not "
                                        f"processed them (channel not visited) or its credit write was lost")
            P(f"  {u.name} vc{e['set']} {DIRS[e['dir']]} -> {d.name if d else '?'} ch{e['dch']}: copy "
              f"{None if e['copy'] is None else e['copy'] & FREE_MASK} ds {None if e['ds'] is None else e['ds'] & FREE_MASK}"
              f" N {e['n']} diff {e['diff']}{tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir")
    ap.add_argument("cache_dir")
    ap.add_argument("--log", action="append", default=[], help="run log(s) with logged compile commands")
    ap.add_argument("--all-slots", action="store_true", help="print every slot header, not only busy channels")
    ap.add_argument("--chip", type=int, action="append", help="report only these chips (summaries stay global)")
    ap.add_argument("--quiet-routers", action="store_true", help="only the summaries and anomalies")
    args = ap.parse_args()

    idxp = os.path.join(args.capture_dir, "index.json")
    if not os.path.exists(idxp):
        idxp = os.path.join(args.capture_dir, "index.partial.json")
    index = json.load(open(idxp))
    cfgs = load_cfgs(args.cache_dir)
    if not cfgs:
        sys.exit(f"no router configs (named_ct_arg_map_generated.h) under {args.cache_dir}")
    attach_compile_cmds(cfgs, args.capture_dir, args.log)
    by_erisc = {}
    for c in cfgs.values():
        if c.image is not None:
            by_erisc.setdefault(c.erisc, []).append(c)
    missing = [c.hash for c in cfgs.values() if c.ct is None]
    print(f"capture {args.capture_dir}: {index.get('start_time')} on {index.get('host')}, "
          f"complete={index.get('complete')}, {len(index.get('errors', []))} capture errors, meta {index.get('meta')}")
    print(f"cache {args.cache_dir}: {len(cfgs)} router configs ({len(by_erisc.get(0, []))} E0, "
          f"{len(by_erisc.get(1, []))} E1); positional args for {len(cfgs) - len(missing)}"
          + (f"; missing for {len(missing)}: {missing[:6]}{'...' if len(missing) > 6 else ''}" if missing else ""))
    for e in index.get("errors", [])[:10]:
        print(f"  capture error: {e}")

    cores = {}
    for chip, crec in index["chips"].items():
        for chan, rec in crec["cores"].items():
            cores[(int(chip), int(chan))] = Core(args.capture_dir, int(chip), int(chan), rec,
                                                 tuple(index.get("constants", {}).get("stream_regs", (4, 10, 297))))
    chip_masks = {}
    for chip, crec in index["chips"].items():
        m = 0
        for chan, rec in crec["cores"].items():
            if rec.get("connected_to"):
                m |= 1 << int(chan)
        chip_masks[int(chip)] = m
    by_coord = {}
    for c in cores.values():
        c.chip_mask = chip_masks.get(c.chip, 0)
        for cs in ("translated", "noc0"):
            v = c.coord(cs)
            if v:
                by_coord.setdefault((c.chip, v[0], v[1]), c)
    for c in cores.values():
        if c.l1 is not None:
            match_core(c, by_erisc)

    rep = Report(args)
    import io
    import contextlib
    for c in sorted(cores.values(), key=lambda c: (c.chip, c.chan)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                router_report(rep, c, by_coord, cores)
            except Exception:  # noqa: BLE001 - one undecodable core must not hide the others
                print(f"  !!! decode error on {c.name}:\n" + traceback.format_exc())
                rep.anoms.append(f"{c.name}: decoder error (see its router section)")
        if not args.quiet_routers and (not args.chip or c.chip in args.chip):
            print()
            sys.stdout.write(buf.getvalue())
    for fn in (lambda: link_summary(rep, cores, index.get("eth_connections", [])), lambda: noc_summary(rep, cores)):
        try:
            fn()
        except Exception:  # noqa: BLE001
            print("  !!! decoder error in summary:\n" + traceback.format_exc())
    print("\n== Traffic waiting (non-zero counts; in a hang this is where packets and credits stopped) ==")
    waiting = 0
    for c in sorted(cores.values(), key=lambda c: (c.chip, c.chan)):
        if c.e0 and c.e0.ct:
            for ch in range(c.e0.g("NUM_SENDER_CHANNELS")):
                v = c.stream(c.e0.sender_stream(ch)) if c.e0.sender_serviced(ch) else None
                if v is not None and (v & FREE_MASK) != c.e0.sender_slots[ch]:
                    waiting += 1
                    n, f = c.e0.sender_slots[ch], v & FREE_MASK
                    what = f"{n - f} unsent" if f <= n else fmt_free(v, n)[1]
                    print(f"  {c.name} E0 ch{ch} (vc{c.e0.vc_of_sender(ch)}): free {f:#x} of {n} -> {what}")
        if c.e1:
            for rx in range(c.e1.g("NUM_RECEIVER_CHANNELS")):
                v = c.stream(c.e1.g(f"TO_RECEIVER_{rx}_PKTS_SENT_ID")) if c.e1.receiver_serviced(rx) else None
                if v is not None and (v & FREE_MASK) != 0:
                    waiting += 1
                    print(f"  {c.name} E1 rx{rx}: {v & FREE_MASK} packets pending")
        for e in getattr(c, "ds", []):
            if e["active"] and e["diff"]:
                waiting += 1
                print(f"  {c.name} E1 vc{e['set']} {DIRS[e['dir']]} -> {e['d'].name if e['d'] else '?'} ch{e['dch']}: "
                      f"downstream free {e['ds'] & FREE_MASK} vs copy {e['copy'] & FREE_MASK} (diff {e['diff']})")
    if not waiting:
        print("  none: every channel empty, every copy equal to its downstream register")
    print("\n== Anomalies ==")
    unmatched = [c.name + ":" + a for c in cores.values() for a in ("e0", "e1")
                 if c.l1 is not None and c.match.get(a, {}).get("how") == "no match"]
    if unmatched:
        print(f"  no config matched: {unmatched}")
    moving = [c.name for c in cores.values() if c.regs.get("pass1") and c.regs.get("pass2") and any(
        c.regs["pass1"]["streams"].get(s) != c.regs["pass2"]["streams"].get(s) for s in c.regs["pass1"]["streams"])]
    if moving:
        print(f"  stream registers changed between passes (traffic still moving) on: {moving}")
    hb = [(c.name, c.regs.get(p, {}).get("l1_words", {}).get(hex(HEARTBEAT)))
          for c in sorted(cores.values(), key=lambda c: (c.chip, c.chan)) for p in ("pass1", "pass2")]
    beats = {}
    for name, v in hb:
        beats.setdefault(name, []).append(v)
    alive = [n for n, v in beats.items() if len(v) == 2 and None not in v and v[0] != v[1]]
    print(f"  liveness: fabric heartbeat (0x7CC70) changed between passes on {len(alive)} of {len(beats)} cores"
          + (f": {alive}" if 0 < len(alive) < len(beats) else ""))
    for a in rep.anoms:
        print("  " + a)
    if not rep.anoms and not unmatched:
        print("  none")


if __name__ == "__main__":
    main()
