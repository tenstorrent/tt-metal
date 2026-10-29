#!/usr/bin/env python3
"""Worker (Tensix) side of the hang capture: offline decode of the fabric test kernels' state.

Used by hang_decode.py when a capture has a "workers" part (hang_capture.py, on by default). Per Tensix core the
capture keeps its mailbox head (launch-message ring, go messages), and for the cores that ran (or still run) a
kernel of the last program, its L1 from 0 up to a window (default 256 KiB: firmware mailbox, fabric connection table,
kernel-config ring with the kernel binaries and runtime args, the test's local args, result buffer, kernel config
buffer, sync words), the whole L1 for cores still running, the soft-reset register and the NIU (NoC) transaction
counters. This module:
  - names each core's kernel per RISC by matching the binary at the launch message's text address against the cached
    kernel ELFs (kernels/<name>/<hash>/<risc>/<risc>.elf.xip.elf under the capture's cache/ or the run's cache);
  - takes struct layouts and compile-time args from those ELFs' DWARF with the toolchain's gdb (`ptype /o`, `print`),
    so the decode follows the kernel source without hand-kept offsets;
  - decodes tt_fabric_test_sender / _receiver / _sync (tests/tt_metal/tt_fabric/test_infra/kernels): result buffer,
    per traffic config packets sent / validated / expected, the word each receiver polls and what is there, credit
    and sync semaphores, and each worker-to-router connection (WorkerToFabricEdmSender in the kernel config buffer,
    the credit word the router writes into worker L1, the teardown word);
  - cross-checks every worker connection against the captured router (E0 channel 0 connection info, read counter,
    producer cursor, free-slots register), every sender flow against its receiver, and every sync core across ranks.
Limits: RISC PCs and BRISC-private memory (kernel globals, stack) are not readable in safe mode, so a core's exact
wait point is inferred from its L1 state; see HANG_CAPTURE.md.
"""

import glob
import os
import re
import struct
import subprocess

try:
    import numpy as np
except ImportError:
    np = None

ROOT = "/home/aagarwal/tt-metal"
DEFAULT_GDB = os.path.join(ROOT, "runtime/sfpi/compiler/bin/riscv-tt-elf-gdb")

# ---------------------------------------------------------------- Blackhole Tensix L1 (dev_mem_map.h, dev_msgs.h)
# Since the merge of main 6d8a103 (2026-10-28): go messages are rings, one per launch slot per sub-device
# (go_messages[9].slot[8]), go_message_index at +1472, and every base after the mailbox moved 0x100.
MAILBOX = 0x60  # MEM_MAILBOX_BASE
HEAD_SIZE = 1488  # mailboxes_t from its start through go_message_index (+1472), rounded to 16
RD_PTR_OFF, LAUNCH_OFF, LAUNCH_SIZE, NUM_LAUNCH = 12, 16, 144, 8
GO_OFF, NUM_GO, GO_RING, GO_IDX_OFF = 1168, 9, 8, 1472
TENSIX = 0  # ProgrammableCoreType::TENSIX (kernel_config_base index)
ROUTING_TABLE = 0x86E0  # MEM_TENSIX_ROUTING_TABLE_BASE (routing_l1_info_t; mesh id +32, device id +34)
FABRIC_CONN = 0x9170  # MEM_TENSIX_FABRIC_CONNECTIONS_BASE (tensix_fabric_connections_l1_info_t)
FABRIC_CONN_RW = FABRIC_CONN + 400  # read_write[16]: worker_flow_control_semaphore, 16 B apart
MAX_EP = 16
DEFAULT_UNRESERVED = 0x1B580  # HAL DEFAULT_UNRESERVED for Tensix (test CommonMemoryMap base)
RISCS = ["BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"]  # TensixProcessorTypes DM0, DM1, MATH0..2
RISC_DIRS = ["brisc", "ncrisc", "trisc0", "trisc1", "trisc2"]
RESET_BIT = {"BRISC": 11, "TRISC0": 12, "TRISC1": 13, "TRISC2": 14, "NCRISC": 18}
GO_SIGNAL = {
    0x80: "GO",
    0x00: "DONE",
    0x40: "INIT",
    0xC0: "RESET_READ_PTR",
    0xE0: "RESET_READ_PTR_FROM_HOST",
    0xF0: "REPLAY_TRACE",
}
DISPATCH_MODE = {0: "dev", 1: "host"}
STATUS = {
    0xABC00000: "STARTED",
    0xABC00001: "PASS",
    0xABCDEAD0: "TIMEOUT",
    0xABCDEAD1: "BAD_HEADER",
    0xABC00003: "DATA_MISMATCH",
}
STATUS_IDX, WORD_CNT_IDX, CYCLES_IDX, PER_CONFIG_IDX = 0, 2, 4, 32
NOC_SEND = {
    0: "unicast_write",
    1: "inline_write",
    2: "atomic_inc",
    3: "fused_atomic_inc",
    4: "scatter_write",
    5: "mcast_write",
    6: "mcast_atomic_inc",
    7: "unicast_read",
    8: "sparse_mcast_write",
}
CONN_STATE = {0: "unused", 1: "open", 2: "close-requested"}
NIU_MST = [
    "ATOMIC_RESP_RECEIVED",
    "WR_ACK_RECEIVED",
    "RD_RESP_RECEIVED",
    "RD_DATA_WORD_RECEIVED",
    "CMD_ACCEPTED",
    "RD_REQ_SENT",
    "NONPOSTED_ATOMIC_SENT",
    "POSTED_ATOMIC_SENT",
    "NONPOSTED_WR_DATA_WORD_SENT",
    "POSTED_WR_DATA_WORD_SENT",
    "NONPOSTED_WR_REQ_SENT",
    "POSTED_WR_REQ_SENT",
    "NONPOSTED_WR_REQ_STARTED",
    "POSTED_WR_REQ_STARTED",
    "RD_REQ_STARTED",
    "NONPOSTED_ATOMIC_STARTED",
]  # NOC_STATUS(0..15)
TEST_KINDS = {"tt_fabric_test_sender": "sender", "tt_fabric_test_receiver": "receiver", "tt_fabric_test_sync": "sync"}


def u16(b, a):
    return struct.unpack_from("<H", b, a)[0]


def u32(b, a):
    return struct.unpack_from("<I", b, a)[0]


def h(v):
    return "None" if v is None else f"{v:#x}"


def prng_next(x):  # tt_fabric_test_kernels_utils.hpp prng_next (xorshift32)
    x &= 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x


def noc_xy(enc):  # NOC_XY_ENCODING(x, y) = y << 6 | x (Blackhole NOC_ADDR_NODE_ID_BITS = 6)
    return (enc & 0x3F, (enc >> 6) & 0x3F)


class Mem:
    """A core's captured L1 bytes; every read returns None outside what was captured."""

    def __init__(self, data, missing=()):
        self.b = data or b""
        self.missing = list(missing)

    def ok(self, a, n):
        if a is None or a < 0 or a + n > len(self.b):
            return False
        return all(a + n <= m[0] or a >= m[0] + m[1] for m in self.missing)

    def u8(self, a):
        return self.b[a] if self.ok(a, 1) else None

    def u16(self, a):
        return u16(self.b, a) if self.ok(a, 2) else None

    def u32(self, a):
        return u32(self.b, a) if self.ok(a, 4) else None

    def u64(self, a):
        return struct.unpack_from("<Q", self.b, a)[0] if self.ok(a, 8) else None

    def raw(self, a, n):
        return bytes(self.b[a : a + n]) if self.ok(a, n) else None


# ====================================================================== gdb: struct layouts and constants
HDR_RE = re.compile(r"^/\*\s*(\d+)(?::\s*\d+)?\s*\|\s*(\d+)\s*\*/\s?(.*)$")  # /* off | size */ decl
UHDR_RE = re.compile(r"^/\*\s+(\d+)\s*\*/\s?(.*)$")  # union member: /* size */ decl
CLOSE_RE = re.compile(r"^\s*\}\s*([A-Za-z_]\w*)?\s*((?:\[\d+\])*)\s*;?\s*$")
TOTAL_RE = re.compile(r"/\*\s*total size \(bytes\):\s*(\d+)\s*\*/")


def _leaf_name(decl):
    d = decl.strip().rstrip(";").strip()
    m = re.search(r"\(\s*\*\s*(\w+)\s*\)\s*\(", d)  # function pointer declarator
    if m:
        return m.group(1), d, 1
    m = re.search(r"([A-Za-z_]\w*)\s*((?:\[\d+\])*)\s*(?::\s*\d+)?\s*$", d)
    if not m:
        return None, d, 1
    count = 1
    for n in re.findall(r"\[(\d+)\]", m.group(2) or ""):
        count *= int(n)
    return m.group(1), d[: m.start()].strip(), count


class Layout:
    """A struct's members flattened to dotted paths ('memory_map.common.result_buffer_base') with their offset,
    size, declared type text and array element count (C arrays only; std::array elements are '<path>._M_elems')."""

    def __init__(self, name, size, fields):
        self.name, self.size, self.fields = name, size, fields

    def has(self, path):
        return path in self.fields or (path + "._M_elems") in self.fields

    def f(self, path):
        if path in self.fields:
            return self.fields[path]
        if path + "._M_elems" in self.fields:
            return self.fields[path + "._M_elems"]
        raise KeyError(f"{self.name}: no member {path}")

    def off(self, path):
        return self.f(path)["off"]

    def fsize(self, path):
        return self.f(path)["size"]


def parse_ptype(text):
    """`ptype /o` output of one struct -> Layout. Members with their absolute offsets (union members carry their own
    members' offsets); nested aggregates are expanded by gdb and kept under their member name."""
    lines = text.splitlines()
    name, start = None, None
    for i, line in enumerate(lines):
        m = re.search(
            r"type = (?:struct|class|union) (.*?)(?: \[with .*\])?(?: : (?:public|private|protected) .*)? \{\s*$", line
        )
        if m:
            name, start = m.group(1).strip(), i + 1
            break
    if name is None:
        return None
    stack = [{"off": 0, "fields": {}, "size": None, "decl": name}]
    pending_total = None
    for line in lines[start:]:
        t = TOTAL_RE.search(line)
        if t and not HDR_RE.match(line) and not UHDR_RE.match(line):
            pending_total = int(t.group(1))
            continue
        if "XXX" in line and ("hole" in line or "padding" in line):
            continue
        m = HDR_RE.match(line)
        um = None if m else UHDR_RE.match(line)
        if m or um:
            off = int(m.group(1)) if m else None
            size = int(m.group(2)) if m else int(um.group(1))
            decl = (m.group(3) if m else um.group(2)).strip()
            if decl.endswith("{"):
                stack.append({"off": off, "fields": {}, "size": size, "decl": decl[:-1].strip()})
            else:
                n, typ, count = _leaf_name(decl)
                if n:
                    if off is None:
                        off = stack[-1]["off"]
                    stack[-1]["fields"][n] = {"off": off, "size": size, "type": typ, "count": count}
            continue
        c = CLOSE_RE.match(line)
        if c and "{" not in line:
            fr = stack.pop()
            if fr["size"] is None:
                fr["size"] = pending_total
            pending_total = None
            if not stack:  # the outer struct
                return Layout(name, fr["size"], fr["fields"])
            off = fr["off"]
            if off is None:  # union member struct: take its first member's offset, else the union's
                offs = [v["off"] for v in fr["fields"].values() if v["off"] is not None]
                off = min(offs) if offs else stack[-1]["off"]
            mname = c.group(1)
            if mname:
                parent = stack[-1]["fields"]
                parent[mname] = {"off": off, "size": fr["size"], "type": fr["decl"], "count": 1}
                for k, v in fr["fields"].items():
                    parent[f"{mname}.{k}"] = v
            continue
    fr = stack[0]
    return Layout(name, fr["size"] or pending_total, fr["fields"])


def run_gdb(gdb, elf, cmds, timeout=60):
    """One batch gdb run over ELF; returns the output of each command (markers echoed between them)."""
    args = [gdb, "-nx", "-batch", "-ex", "set width 0", "-ex", "set height 0", "-ex", "set print pretty off"]
    for i, c in enumerate(cmds):
        args += ["-ex", f"echo @@@{i}@@@\\n", "-ex", c]
    args += ["-ex", f"echo @@@{len(cmds)}@@@\\n", elf]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    out = r.stdout
    res = []
    for i in range(len(cmds)):
        a = out.find(f"@@@{i}@@@\n")
        b = out.find(f"@@@{i + 1}@@@\n")
        res.append(out[a + len(f"@@@{i}@@@\n") : b] if a >= 0 and b >= 0 else "")
    return res, r.stderr


KIND_TYPES = {  # (regex over `info types` names, short key)
    "sender": [
        (r"^tt::tt_fabric::fabric_tests::SenderKernelConfig<.*>$", "main"),
        (r"^tt::tt_fabric::fabric_tests::SenderKernelTrafficConfig<.*>$", "traffic"),
        (r"^tt::tt_fabric::fabric_tests::LocalSyncConfig<.*>$", "local_sync"),
        (r"^tt::tt_fabric::WorkerToFabricEdmSenderBase<false, 0, \d>$", "adapter"),
        (r"^tt::tt_fabric::fabric_tests::StreamingBuffer$", "streambuf"),
    ],
    "receiver": [
        (r"^tt::tt_fabric::fabric_tests::ReceiverKernelConfig<.*>$", "main"),
        (r"^tt::tt_fabric::fabric_tests::TrafficValidationConfigBase$", "vbase"),
        (r"^tt::tt_fabric::fabric_tests::WriteValidationConfig$", "v_write"),
        (r"^tt::tt_fabric::fabric_tests::AtomicIncValidationConfig$", "v_atomic"),
        (r"^tt::tt_fabric::fabric_tests::WriteAtomicIncValidationConfig$", "v_fused"),
        (r"^tt::tt_fabric::fabric_tests::ScatterWriteValidationConfig$", "v_scatter"),
        (r"^tt::tt_fabric::fabric_tests::ReceiverCreditManager$", "credit"),
        (r"^tt::tt_fabric::fabric_tests::StreamingBuffer$", "streambuf"),
        (r"^tt::tt_fabric::WorkerToFabricEdmSenderBase<false, 0, \d>$", "adapter"),
    ],
    "sync": [
        (r"^tt::tt_fabric::fabric_tests::SyncKernelConfig<.*>$", "main"),
        (r"^tt::tt_fabric::fabric_tests::LineSyncConfig<.*>$", "line_sync"),
        (r"^tt::tt_fabric::fabric_tests::LocalSyncConfig<.*>$", "local_sync"),
        (r"^tt::tt_fabric::WorkerToFabricEdmSenderBase<false, 0, \d>$", "adapter"),
    ],
}
KIND_CONSTS = {
    "sender": [
        "IS_2D_FABRIC",
        "NUM_FABRIC_CONNECTIONS",
        "NUM_TRAFFIC_CONFIGS",
        "BENCHMARK_MODE",
        "LINE_SYNC",
        "NUM_LOCAL_SYNC_CORES",
        "KERNEL_CONFIG_BUFFER_SIZE",
        "HAS_MUX_CONNECTIONS",
        "NUM_MUXES_TO_TERMINATE",
        "VC_ID",
        "STREAM_MODE",
    ],
    "receiver": [
        "IS_2D_FABRIC",
        "NUM_TRAFFIC_CONFIGS",
        "BENCHMARK_MODE",
        "KERNEL_CONFIG_BUFFER_SIZE",
        "NUM_CREDIT_CONNECTIONS",
        "HAS_MUX_CONNECTIONS",
        "NUM_MUXES_TO_TERMINATE",
    ],
    "sync": [
        "IS_2D_FABRIC",
        "NUM_SYNC_FABRIC_CONNECTIONS",
        "NUM_LOCAL_SYNC_CORES",
        "KERNEL_CONFIG_BUFFER_SIZE",
        "HAS_MUX_CONNECTIONS",
        "NUM_MUXES_TO_TERMINATE",
        "USE_UNICAST_SYNC_PACKETS",
        "NUM_SYNC_CONFIGS",
        "NUM_START_GLOBAL_SYNC_ITERS",
    ],
}


def elf_segments(path):
    """PT_LOAD contents in program-header order, each padded to 4 (the JIT loader's packed image)."""
    b = open(path, "rb").read()
    if b[:4] != b"\x7fELF" or b[4] != 1:
        raise ValueError(f"{path}: not an ELF32")
    phoff, phentsize, phnum = u32(b, 0x1C), u16(b, 0x2A), u16(b, 0x2C)
    segs = []
    for i in range(phnum):
        ptype, off, vaddr, paddr, filesz, memsz, flags = struct.unpack_from("<7I", b, phoff + i * phentsize)
        if ptype == 1 and filesz:
            seg = b[off : off + filesz]
            segs.append((vaddr, flags, seg + b"\0" * (-len(seg) % 4)))
    return segs


class KernelBin:
    """One cached kernel binary for one RISC: its load image, and (lazily) its layouts and constants via gdb."""

    def __init__(self, name, hsh, risc, path):
        self.name, self.hash, self.risc, self.path = name, hsh, risc, path
        self.kind = TEST_KINDS.get(name, "dispatch" if name.startswith("cq_") else "other")
        self.image = self.text = None
        try:
            segs = elf_segments(path)
            self.image = b"".join(s[2] for s in segs)
            ex = [s for s in segs if s[1] & 1]
            self.text = (ex[0] if ex else segs[0])[2] if segs else None
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
        self.layouts = None
        self.consts = {}
        self.gdb_error = None

    def label(self):
        return f"{self.name}/{self.hash}"

    def load_layouts(self, gdb):
        if self.layouts is not None or self.kind not in KIND_TYPES:
            return
        self.layouts = {}
        if not gdb or not os.path.exists(gdb):
            self.gdb_error = f"gdb not found ({gdb}): no struct layouts, worker state limited to fixed L1 words"
            return
        try:
            pats = KIND_TYPES[self.kind]
            words = sorted({re.sub(r"[\^$\\]|<.*|\(.*", "", p).split("::")[-1] for p, _ in pats})
            outs, err = run_gdb(gdb, self.path, [f"info types {w}" for w in words])
            names = set()
            for o in outs:
                for line in o.splitlines():
                    m = re.match(r"^\s*\d*:?\s*(tt::.*?);\s*$", line)
                    if m and not line.lstrip().split(":", 1)[-1].strip().startswith("typedef"):
                        names.add(m.group(1).strip())
            chosen = {}
            for pat, key in pats:
                cands = sorted(n for n in names if re.match(pat, n))
                if key == "adapter":  # the EDM sender the kernel was built for (VC2 adapters end in ", 2>")
                    pref = [n for n in cands if n.endswith(", 0>")]
                    cands = pref or cands
                if cands:
                    chosen[key] = cands[0]
            consts = KIND_CONSTS.get(self.kind, [])
            cmds = [f"ptype /o '{n}'" for n in chosen.values()] + [f"print (int){c}" for c in consts]
            outs, err = run_gdb(gdb, self.path, cmds)
            for (key, n), o in zip(chosen.items(), outs):
                lay = parse_ptype(o)
                if lay is not None:
                    self.layouts[key] = lay
            for c, o in zip(consts, outs[len(chosen) :]):
                m = re.search(r"\$\d+ = (-?\d+)", o)
                if m:
                    self.consts[c] = int(m.group(1))
            missing = [k for _, k in KIND_TYPES[self.kind] if k not in self.layouts]
            if "main" in missing:
                self.gdb_error = f"gdb gave no layout for the kernel config struct ({err.strip()[:200]})"
        except Exception as e:  # noqa: BLE001
            self.gdb_error = f"gdb failed: {type(e).__name__}: {e}"


def find_kernel_bins(dirs):
    """Every cached kernel binary (not the router's) under the given cache dirs: {risc: [KernelBin]}."""
    out, seen = {}, set()
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        pats = [
            f"{d}/kernels/*/*/*/*.elf.xip.elf",
            f"{d}/*/kernels/*/*/*/*.elf.xip.elf",
            f"{d}/kernels/*/*/*/*.elf",
            f"{d}/*/kernels/*/*/*/*.elf",
        ]
        for pat in pats:
            for p in sorted(glob.glob(pat)):
                parts = p.split("/")
                risc, hsh, name = parts[-2], parts[-3], parts[-4]
                if name == "fabric_erisc_router_v2" or risc not in RISC_DIRS:
                    continue
                if not os.path.basename(p).startswith(risc + ".elf"):
                    continue
                key = (name, hsh, risc)
                if key in seen:
                    continue
                seen.add(key)
                kb = KernelBin(name, hsh, risc, p)
                if kb.image:
                    out.setdefault(risc, []).append(kb)
    return out


def similarity(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    if np is not None:
        return float(np.mean(np.frombuffer(a, np.uint8) == np.frombuffer(b, np.uint8)))
    return sum(x == y for x, y in zip(a, b)) / len(a)


def match_kernel(mem, addr, cands):
    """The cached binary at L1 address addr: (KernelBin, how) or (None, why)."""
    if addr is None or not cands:
        return None, "no cached binaries" if not cands else "no address"
    for kb in cands:
        if kb.image and mem.raw(addr, len(kb.image)) == kb.image:
            return kb, "exact"
    for kb in cands:
        if kb.text and mem.raw(addr, len(kb.text)) == kb.text:
            return kb, "exact text"
    best = None
    for kb in cands:
        if kb.text:
            got = mem.raw(addr, len(kb.text))
            if got:
                s = similarity(got, kb.text)
                if s > 0.9 and (best is None or s > best[1]):
                    best = (kb, s)
    if best:
        return best[0], f"similarity {best[1]:.3f}"
    return None, "no cached binary matches"


# ====================================================================== mailbox
def parse_head(b, base=0):
    """mailboxes_t head (launch ring, go messages) from bytes b, the mailbox starting at offset base."""
    if b is None or len(b) < base + GO_IDX_OFF + 4:
        return None
    rd = u32(b, base + RD_PTR_OFF)
    slots = []
    for i in range(NUM_LAUNCH):
        lb = base + LAUNCH_OFF + LAUNCH_SIZE * i
        slots.append(
            {
                "kcb": u32(b, lb + 4 * TENSIX),
                "kcb_all": [u32(b, lb + 4 * k) for k in range(4)],
                "rta": [u16(b, lb + 28 + 4 * p) for p in range(5)],
                "crta": [u16(b, lb + 30 + 4 * p) for p in range(5)],
                "mode": b[lb + 48],
                "text_off": [u32(b, lb + 52 + 4 * p) for p in range(5)],
                "text_size": [u32(b, lb + 72 + 4 * p) for p in range(5)],
                "noc": b[lb + 104],
                "noc_mode": b[lb + 105],
                "host_id": u32(b, lb + 108),
                "enables": u32(b, lb + 112),
                "wids": [u16(b, lb + 120 + 2 * p) for p in range(5)],  # watcher_kernel_ids (host kernel ids)
                "preload": b[lb + 143],
            }
        )
    for sl in slots:  # the slot's newest kernel id; -1 when it loaded no kernel
        sl["wid"] = max((w for w, n in zip(sl["wids"], sl["text_size"]) if n), default=-1)
    gi = u32(b, base + GO_IDX_OFF)
    # This sub-device's go ring: one go message per launch slot.
    go = [u32(b, base + GO_OFF + 4 * (GO_RING * (gi if gi < NUM_GO else 0) + i)) for i in range(GO_RING)]
    cur = rd & (NUM_LAUNCH - 1)
    sig = go[cur] >> 24
    if sig == 0x80:
        slot, why = cur, "current (running)"
    else:
        prev = (cur - 1) & (NUM_LAUNCH - 1)
        slot, why = (prev, "last run (rd_ptr - 1)") if slots[prev]["mode"] == 0 else (cur, "rd_ptr (host mode)")
    # The slot this core was last given a kernel in: the highest kernel id. Kernel ids (watcher_kernel_ids, one per
    # CreateKernel, process-wide counter) grow with program creation; the host-assigned program id is 0 in these runs.
    latest = max(range(NUM_LAUNCH), key=lambda i: slots[i]["wid"])
    return {
        "rd_ptr": rd,
        "slots": slots,
        "go": go,
        "go_idx": gi,
        "signal": sig,
        "slot": slot,
        "slot_why": why,
        "latest": latest if slots[latest]["wid"] >= 0 else None,
        "launch": slots[slot],
    }


def newest_program(heads):
    """A chip's newest program as (kernel id, kernel config base): the highest kernel id among the device-dispatched
    launch slots its cores last executed, where that slot is also the core's most recent one. All cores of one program
    share its kernel config base (one ring-buffer allocation per program)."""
    best = None
    for hd in heads:
        if hd is None:
            continue
        sl = hd["launch"]
        if sl["wid"] < 0 or sl["mode"] != 0 or (hd["latest"] != hd["slot"] and hd["signal"] != 0x80):
            continue
        if best is None or sl["wid"] > best[0]:
            best = (sl["wid"], sl["kcb"])
    return best


def in_program(hd, newest):
    """The core runs the chip's newest program (go GO with kernels enabled), or ran it: the launch slot it last
    executed is its most recent one and carries that program's kernel config base. A core left out of later programs
    keeps executing its stale slots with nothing enabled, so its last-executed slot is older than its latest."""
    if hd is None:
        return False
    if hd["signal"] == 0x80 and hd["launch"]["enables"]:
        return True
    sl = hd["launch"]
    return (
        newest is not None
        and sl["wid"] >= 0
        and hd["latest"] == hd["slot"]
        and sl["kcb"] == newest[1]
        and sl["mode"] == 0
    )


# ====================================================================== per-core model
class Worker:
    def __init__(self, cap_dir, chip, key, rec, consts):
        self.chip, self.key, self.rec = chip, key, rec
        self.noc0 = tuple(rec.get("noc0") or ())
        self.xlat = tuple(rec["translated"]) if rec.get("translated") else None
        self.logical = tuple(rec["logical"]) if rec.get("logical") else None
        self.mem = Mem(b"")
        l1 = rec.get("l1")
        if l1 and os.path.exists(os.path.join(cap_dir, l1["file"])):
            self.mem = Mem(open(os.path.join(cap_dir, l1["file"]), "rb").read(), l1.get("missing", []))
        self.head = parse_head(self.mem.raw(MAILBOX, HEAD_SIZE)) if self.mem.ok(MAILBOX, HEAD_SIZE) else None
        if self.head is None and rec.get("head_hex"):
            self.head = parse_head(bytes.fromhex(rec["head_hex"]))
        self.regs = rec.get("regs", {})
        self.niu = rec.get("niu", {})
        self.kernels = {}  # risc -> (KernelBin or None, how)
        self.kb = None  # the BRISC test kernel
        self.kind = None
        self.common = None
        self.state = None  # "running" | "done" | ...
        self.waits, self.anoms, self.notes = [], [], []
        self.conns, self.flows_out, self.flows_in = [], [], []
        self.sync = None
        self.mesh_dev = None
        self.status = None
        self.bench = None
        self.stale = False
        self.niu_out = {}
        self.local_sync = None

    @property
    def name(self):
        x = self.xlat or self.noc0
        return f"chip{self.chip}/tensix({x[0]},{x[1]})" if x else f"chip{self.chip}/{self.key}"

    def flag(self, what):
        self.anoms.append(what)


def soft_reset_text(v):
    if v is None:
        return "?"
    run = [r for r in RISCS if not (v >> RESET_BIT[r]) & 1]
    return f"{v:#x} (running: {' '.join(run) or 'none'})"


def niu_outstanding(niu):
    """Per NoC: non-posted writes not acked, reads not answered, atomics not answered (hardware counters)."""
    out = {}
    for noc, regs in sorted((niu or {}).items()):
        g = {n: regs.get(hex(4 * i)) for i, n in enumerate(NIU_MST)}
        if None in (g["NONPOSTED_WR_REQ_SENT"], g["WR_ACK_RECEIVED"]):
            continue

        def diff(a, b):  # requests not yet answered; negative (more answers, e.g. multicast acks) counts as 0
            if a is None or b is None:
                return None
            d = (a - b) & 0xFFFFFFFF
            return d if d < 1 << 31 else 0

        out[noc] = {
            "wr": diff(g["NONPOSTED_WR_REQ_SENT"], g["WR_ACK_RECEIVED"]),
            "rd": diff(g["RD_REQ_SENT"], g["RD_RESP_RECEIVED"]),
            "at": diff(g["NONPOSTED_ATOMIC_SENT"], g["ATOMIC_RESP_RECEIVED"]),
            "raw": g,
        }
    return out


class View:
    """A struct instance at L1 address addr with a gdb layout."""

    def __init__(self, mem, addr, lay):
        self.mem, self.addr, self.lay = mem, addr, lay

    def a(self, path, i=0, elem=None):
        f = self.lay.f(path)
        e = elem if elem is not None else (f["size"] // max(f["count"], 1) if f["count"] > 1 else f["size"])
        return self.addr + f["off"] + i * e

    def u8(self, path, i=0):
        return self.mem.u8(self.a(path, i, 1))

    def u16(self, path, i=0):
        return self.mem.u16(self.a(path, i, 2))

    def u32(self, path, i=0):
        return self.mem.u32(self.a(path, i, 4))

    def u64(self, path):
        return self.mem.u64(self.a(path))


def decode_adapter(w, addr, lay, idx):
    """A WorkerToFabricEdmSender in worker L1 (connection idx of the core's FabricConnectionArray)."""
    v = View(w.mem, addr, lay)
    c = {"idx": idx, "addr": addr}
    for k in (
        "edm_buffer_addr",
        "edm_buffer_local_free_slots_read_ptr",
        "edm_buffer_remote_free_slots_update_addr",
        "edm_connection_handshake_l1_addr",
        "edm_worker_location_info_addr",
        "edm_copy_of_wr_counter_addr",
        "local_producer_cursor_addr",
        "worker_teardown_addr",
        "edm_buffer_base_addr",
    ):
        c[k] = v.u32(k) if lay.has(k) else None
    c["slot_index"] = v.u8("buffer_slot_index") if lay.has("buffer_slot_index") else None
    c["sent"] = v.u32("buffer_slot_write_counter.counter") if lay.has("buffer_slot_write_counter.counter") else None
    ci_path = "buffer_slot_write_counter.index"
    c["counter_index"] = v.u8(ci_path) if lay.has(ci_path) else None
    c["buffer_size"] = v.u16("buffer_size_bytes") if lay.has("buffer_size_bytes") else None
    c["N"] = v.u8("num_buffers_per_channel") if lay.has("num_buffers_per_channel") else None
    c["edm_xy"] = (v.u8("edm_noc_x"), v.u8("edm_noc_y")) if lay.has("edm_noc_x") else None
    rp = c["edm_buffer_local_free_slots_read_ptr"]
    c["credit"] = w.mem.u32(rp)  # the router's read counter as last written into worker L1
    c["teardown"] = w.mem.u32(c["worker_teardown_addr"])
    lc = c["local_producer_cursor_addr"]
    c["cursor_in"] = (w.mem.u32(lc), w.mem.u32(lc + 4)) if lc else None  # cursor read back at the last open
    in_table = (
        rp is not None and FABRIC_CONN_RW <= rp < FABRIC_CONN_RW + 16 * MAX_EP and (rp - FABRIC_CONN_RW) % 16 == 0
    )
    c["eth_chan"] = (rp - FABRIC_CONN_RW) // 16 if in_table else None
    c["initialized"] = bool(c["N"]) and c["edm_xy"] not in (None, (0, 0)) and rp is not None
    if c["initialized"] and c["sent"] is not None and c["credit"] is not None:
        used = (c["sent"] - c["credit"]) & 0xFFFFFFFF
        c["used"] = used
        c["free"] = 0 if used >= c["N"] else c["N"] - used
    else:
        c["used"] = c["free"] = None
    return c


def fabric_table(w):
    """tensix_fabric_connections_l1_info_t: per Ethernet channel the router the worker connects to."""
    m = w.mem
    if not m.ok(FABRIC_CONN, 656):
        return None
    mask = m.u32(FABRIC_CONN + 384)
    ents = {}
    for ch in range(MAX_EP):
        if not (mask >> ch) & 1:
            continue
        e = FABRIC_CONN + 24 * ch
        ents[ch] = {
            "buffer_base": m.u32(e),
            "handshake": m.u32(e + 4),
            "location_info": m.u32(e + 8),
            "cursor_addr": m.u32(e + 12),
            "buffer_size": m.u16(e + 16),
            "dir": m.u8(e + 18),
            "edm_xy": (m.u8(e + 19), m.u8(e + 20)),
            "N": m.u8(e + 21),
            "stream": m.u16(e + 22),
            "credit_word": m.u32(FABRIC_CONN_RW + 16 * ch),
        }
    return {"mask": mask, "ents": ents}


def decode_common(w):
    """Kernel config base (launch message), runtime args and the CommonMemoryMap the test passes first."""
    hd = w.head
    if hd is None:
        return
    ln = hd["launch"]
    w.kcb = ln["kcb"]
    w.rta_base = w.kcb + ln["rta"][0]
    words = [w.mem.u32(w.rta_base + 4 * i) for i in range(12)]
    if None in words[:9]:
        w.notes.append(f"runtime args at {w.rta_base:#x} not captured")
        return
    keys = [
        "local_args_base",
        "local_args_size",
        "result_buffer_base",
        "result_buffer_size",
        "kernel_config_base",
        "kernel_config_size",
        "mux_local_addresses_base",
        "mux_local_addresses_size",
        "mux_termination_sync_address",
    ]
    w.common = dict(zip(keys, words[:9]))
    w.rta_words = words
    c = w.common
    if not (0 < c["result_buffer_base"] < 0x180000 and 0 < c["kernel_config_base"] < 0x180000):
        w.notes.append(f"runtime args do not look like the test's CommonMemoryMap: {[h(x) for x in words[:9]]}")
        w.common = None
        return
    rb = c["result_buffer_base"]
    w.status = w.mem.u32(rb + 4 * STATUS_IDX)
    lo, hi = w.mem.u32(rb + 4 * WORD_CNT_IDX), w.mem.u32(rb + 4 * WORD_CNT_IDX + 4)
    w.result_packets = None if lo is None or hi is None else lo | hi << 32
    lo, hi = w.mem.u32(rb + 4 * CYCLES_IDX), w.mem.u32(rb + 4 * CYCLES_IDX + 4)
    w.result_cycles = None if lo is None or hi is None else lo | hi << 32


def status_text(s):
    return STATUS.get(s, h(s))


# ---------------------------------------------------------------- sender
def decode_sender(w, kb, decode_header, rbs=36):
    L, K = kb.layouts, kb.consts
    main, tc, ad = L.get("main"), L.get("traffic"), L.get("adapter")
    if tc is None or ad is None:
        what = "the traffic config" if tc is None else "the EDM adapter"
        w.notes.append(f"no layout for {what}: sender not decoded")
        return
    base = w.common["kernel_config_base"]
    v = View(w.mem, base, main)
    ncfg = K.get("NUM_TRAFFIC_CONFIGS")
    w.bench = K.get("BENCHMARK_MODE")
    nconn = v.u8("connections.num_connections")
    nmax = main.fsize("connections.is_mux")
    stride = main.fsize("connections.storage") // max(nmax, 1)
    for i in range(nconn or 0):
        if i >= nmax:
            break
        is_mux = v.u8("connections.is_mux", i)
        if is_mux:
            w.conns.append({"idx": i, "mux": True, "initialized": False})
            continue
        w.conns.append(decode_adapter(w, v.a("connections.storage") + i * stride, ad, i))
    if K.get("LINE_SYNC") and "local_sync" in L:
        ls = View(w.mem, v.a("local_sync_config_storage"), L["local_sync"])
        addr = ls.u32("sync_address")
        n = L["local_sync"].fsize("sync_core_xy_encoding_") // 4
        w.local_sync = {
            "addr": addr,
            "value": w.mem.u32(addr) if addr else None,
            "val": ls.u32("sync_val"),
            "cores": [noc_xy(ls.u32("sync_core_xy_encoding_", i) or 0) for i in range(n)],
        }
    for i in range(ncfg or 0):
        p = v.u32("traffic_config_ptrs", i)
        expect = v.a("traffic_configs_storage") + i * tc.size
        f = {"i": i, "ptr": p}
        if p != expect:
            w.flag(f"traffic config {i} pointer {h(p)} != storage slot {expect:#x} (config not built)")
            p = expect
        t = View(w.mem, p, tc)
        f["n"] = t.u32("metadata.num_packets")
        f["seed"] = t.u32("metadata.seed")
        f["buf"] = t.u32("metadata.payload_buffer_size")
        f["sent"] = t.u32("num_packets_processed")
        f["payload"] = t.u32("payload_size_bytes")
        f["conn_idx"] = t.u8("connection_idx_")
        nc = t.u8("num_connections_") or 0
        f["conn_list"] = [t.u8("connection_indices_", j) for j in range(min(nc, 5))]
        f["type"] = t.u8("noc_send_type_")
        hdr = t.u32("packet_header")
        f["hdr_addr"] = hdr
        f["hdr"] = decode_header(w.mem.b, hdr, None)[0] if hdr and w.mem.ok(hdr, 96) else None
        if hdr and w.mem.ok(hdr, 96):  # HybridMeshPacketHeader: dst chip, dst mesh after the route buffer
            f["dst_dev"], f["dst_mesh"] = w.mem.u16(hdr + 48 + rbs), w.mem.u16(hdr + 50 + rbs)
        else:
            f["dst_dev"] = f["dst_mesh"] = None
        ty = f["type"]
        sel = {
            0: "noc_fields_.write_fields",
            2: "noc_fields_.atomic_inc_fields",
            3: "noc_fields_.write_atomic_inc_fields.write_fields",
            4: "noc_fields_.scatter_write_fields",
        }.get(ty)
        if sel:
            dst = t.u32(f"{sel}.dst_addresses", 0) if ty == 4 else t.u32(f"{sel}.dst_address")
            enc = t.u32(f"{sel}.dst_noc_encoding")
            f["dst_addr"], f["dst_xy"] = dst, noc_xy(enc) if enc is not None else None
        else:
            f["dst_addr"] = f["dst_xy"] = None
        if ty == 3:
            f["atomic_addr"] = t.u32("noc_fields_.write_atomic_inc_fields.atomic_inc_fields.dst_address")
        cm = "credit_manager_"
        f["credit_on"] = t.u8(f"{cm}.enabled_")
        if f["credit_on"]:
            base_c = t.u32(f"{cm}.sender_credit_info_.credit_reception_address_base")
            nr = t.u32(f"{cm}.num_receivers_") or 0
            f["credit"] = {
                "receivers": nr,
                "total": t.u32(f"{cm}.total_credits_"),
                "initial": t.u32(f"{cm}.initial_credits_"),
                "estimated": t.u32(f"{cm}.estimated_available_credits_"),
                "prev": t.u32(f"{cm}.prev_processed_credits_"),
                "got_all": t.u8(f"{cm}.got_all_credits_back_"),
                "words": [w.mem.u32(base_c + 16 * r) for r in range(min(nr, 16))] if base_c else [],
            }
        w.flows_out.append(f)


# ---------------------------------------------------------------- receiver
def receiver_local_args(w, ncfg, is_2d):
    """The host's per-config receiver args (initial seed, type, destination), parsed in the kernel's order."""
    base = w.common["local_args_base"]
    words = [w.mem.u32(base + 4 * i) for i in range(64 * max(ncfg, 1))]
    out, k = [], 0
    try:
        for _ in range(ncfg):
            c = {"n": words[k], "seed0": words[k + 1], "buf": words[k + 2], "type": words[k + 3]}
            k += 4
            t = c["type"]
            if t == 0:
                c["payload"], c["dst"] = words[k], words[k + 1]
                k += 2
            elif t == 2:
                c["inc"], c["dst"] = words[k] & 0x3FFFFFFF, words[k + 1]
                k += 2
            elif t == 3:
                c["payload"], c["dst"], c["inc"], c["atomic_dst"] = (
                    words[k],
                    words[k + 1],
                    words[k + 2] & 0x3FFFFFFF,
                    words[k + 3],
                )
                k += 4
            elif t == 4:
                c["payload"], cnt = words[k], words[k + 1]
                c["dsts"] = words[k + 2 : k + 2 + cnt]
                c["dst"] = c["dsts"][0] if c["dsts"] else None
                c["chunks"] = words[k + 2 + cnt : k + 2 + cnt + cnt - 1]
                k += 2 + cnt + cnt - 1
            else:
                raise ValueError(f"type {t}")
            has_credit = words[k]
            k += 1
            if has_credit:
                k += 5 if is_2d else 4
            out.append(c)
    except (IndexError, TypeError, ValueError) as e:
        w.notes.append(f"receiver local args not parsed past config {len(out)}: {e}")
    return out


def seed_index(seed0, value, limit):
    """k with prng^k(seed0) == value, for k <= limit (None if absent)."""
    s = seed0
    for k in range(limit + 1):
        if s == value:
            return k
        s = prng_next(s)
    return None


def decode_receiver(w, kb):
    L, K = kb.layouts, kb.consts
    main, vb, sb = L.get("main"), L.get("vbase"), L.get("streambuf")
    if vb is None or "adapter" not in L:
        w.notes.append("no layout for the validation config base or the EDM adapter: receiver not decoded")
        return
    base = w.common["kernel_config_base"]
    v = View(w.mem, base, main)
    ncfg = K.get("NUM_TRAFFIC_CONFIGS") or 0
    w.bench = K.get("BENCHMARK_MODE")
    nconn = v.u8("credit_connections.num_connections")
    nmax = main.fsize("credit_connections.is_mux")
    stride = main.fsize("credit_connections.storage") // max(nmax, 1)
    for i in range(min(nconn or 0, nmax)):
        if v.u8("credit_connections.is_mux", i):
            w.conns.append({"idx": i, "mux": True, "initialized": False})
        else:
            w.conns.append(decode_adapter(w, v.a("credit_connections.storage") + i * stride, L["adapter"], i))
    args = receiver_local_args(w, ncfg, K.get("IS_2D_FABRIC", 1))
    vsize = main.fsize("validation_configs_storage") // max(ncfg, 1)
    for i in range(ncfg):
        p = v.u32("traffic_configs_", i)
        expect = v.a("validation_configs_storage") + i * vsize
        f = {"i": i, "ptr": p}
        if p != expect:
            w.flag(f"validation config {i} pointer {h(p)} != storage slot {expect:#x} (config not built)")
            p = expect
        t = View(w.mem, p, vb)
        f["n"] = t.u32("metadata.num_packets")
        f["seed"] = t.u32("metadata.seed")
        f["buf"] = t.u32("metadata.payload_buffer_size")
        f["got"] = t.u32("num_packets_processed")
        f["credit_mgr"] = t.u32("credit_manager_")
        a = args[i] if i < len(args) else {}
        f["args"] = a
        f["type"] = a.get("type")
        f["dst_addr"] = a.get("dst")
        f["poll"] = poll_state(w, p, f, a, L)
        w.flows_in.append(f)


def poll_state(w, p, f, a, L):
    """What the receiver polls for its next packet and what L1 holds there."""
    t, mem, out = f["type"], w.mem, {}
    seed, k = f["seed"], f["got"]
    seed0 = a.get("seed0")
    if seed0 is not None and k is not None and seed is not None and k <= 1 << 20:
        s = seed0
        for _ in range(k):
            s = prng_next(s)
        out["seed_ok"] = s == seed
    if t in (0, 3) and "v_write" in L and "streambuf" in L:
        lay = L["v_fused"] if t == 3 and "v_fused" in L else L["v_write"]
        pb = View(mem, p, lay).u32("payload_buffer_")
        sbv = View(mem, pb, L["streambuf"]) if pb else None
        if sbv:
            b0, cur, ps = sbv.u32("base_address_"), sbv.u32("current_offset_"), sbv.u32("payload_size_")
            out.update(
                {
                    "buf_base": b0,
                    "offset": cur,
                    "payload": ps,
                    "wrapped": sbv.u8("has_wrapped_"),
                    "total": sbv.u32("total_size_"),
                }
            )
            if None not in (b0, cur, ps, seed) and ps >= 4:
                addr = b0 + cur + ps - 4
                out["addr"], out["want"] = addr, (seed + ps // 4 - 1) & 0xFFFFFFFF
                out["have"] = mem.u32(addr)
                first = mem.u32(b0 + cur)
                out["first"] = first
                if first is not None and seed0 is not None and f["n"]:
                    out["slot_holds"] = seed_index(seed0, first, f["n"] + 64)
        if t == 3:
            vf = View(mem, p, L["v_fused"])
            aa = vf.u32("atomic_inc_address")
            out.update(
                {
                    "atomic_addr": aa,
                    "atomic_want": vf.u32("expected_atomic_value"),
                    "atomic_have": mem.u32(aa) if aa else None,
                    "atomic_step": vf.u32("atomic_inc_val"),
                }
            )
    elif t == 2 and "v_atomic" in L:
        va = View(mem, p, L["v_atomic"])
        aa = va.u32("poll_address")
        out.update(
            {
                "atomic_addr": aa,
                "atomic_want": va.u32("expected_value"),
                "atomic_step": va.u32("value_step_size"),
                "atomic_have": mem.u32(aa) if aa else None,
            }
        )
    elif t == 4 and "v_scatter" in L:
        vs = View(mem, p, L["v_scatter"])
        d0, d1 = vs.u32("dst_addresses", 0), vs.u32("dst_addresses", 1)
        c0, last = vs.u16("chunk_sizes", 0), vs.u32("last_chunk_size")
        out.update({"chunks": [d0, d1], "chunk0": c0, "last": last, "offset": vs.u32("current_offset")})
        if None not in (d0, d1, c0, last, seed) and c0 >= 4 and last >= 4:
            out["addr"], out["want"] = d0 + c0 - 4, (seed + c0 // 4 - 1) & 0xFFFFFFFF
            out["have"] = mem.u32(d0 + c0 - 4)
            out["addr2"], out["want2"] = d1 + last - 4, (seed + c0 // 4 + last // 4 - 1) & 0xFFFFFFFF
            out["have2"] = mem.u32(d1 + last - 4)
            first = mem.u32(d0)
            out["first"] = first
            if first is not None and seed0 is not None and f["n"]:
                out["slot_holds"] = seed_index(seed0, first, f["n"] + 64)
    return out


# ---------------------------------------------------------------- sync
def decode_sync(w, kb, decode_header):
    L, K = kb.layouts, kb.consts
    main = L.get("main")
    if "adapter" not in L:
        w.notes.append("no layout for the EDM adapter: sync core not decoded")
        return
    base = w.common["kernel_config_base"]
    v = View(w.mem, base, main)
    nconn = v.u8("sync_connections.num_connections")
    nmax = main.fsize("sync_connections.is_mux")
    stride = main.fsize("sync_connections.storage") // max(nmax, 1)
    for i in range(min(nconn or 0, nmax)):
        if v.u8("sync_connections.is_mux", i):
            w.conns.append({"idx": i, "mux": True, "initialized": False})
        else:
            w.conns.append(decode_adapter(w, v.a("sync_connections.storage") + i * stride, L["adapter"], i))
    s = {"configs": []}
    nsc = K.get("NUM_SYNC_CONFIGS") or 0
    lsz = L["line_sync"].size if "line_sync" in L else 0
    for i in range(nsc):
        if not lsz:
            break
        lv = View(w.mem, v.a("line_sync_configs_storage") + i * lsz, L["line_sync"])
        hdr = lv.u32("packet_header")
        nc = lv.u8("num_connections_") or 0
        s["configs"].append(
            {
                "ptr": lv.u32("line_sync_ptr"),
                "val": lv.u32("line_sync_val"),
                "conns": [lv.u8("connection_indices_", j) for j in range(min(nc, 5))],
                "hdr": decode_header(w.mem.b, hdr, None)[0] if hdr and w.mem.ok(hdr, 96) else None,
            }
        )
    if s["configs"]:
        c0 = s["configs"][0]
        s["global_addr"], s["L"] = c0["ptr"], c0["val"]
        s["global"] = w.mem.u32(c0["ptr"]) if c0["ptr"] else None
    if "local_sync" in L:
        ls = View(w.mem, v.a("local_sync_config_storage"), L["local_sync"])
        addr = ls.u32("sync_address")
        n = L["local_sync"].fsize("sync_core_xy_encoding_") // 4
        s["local_addr"], s["local"] = addr, w.mem.u32(addr) if addr else None
        s["local_cores"] = [noc_xy(ls.u32("sync_core_xy_encoding_", i) or 0) for i in range(n)]
    s["iters"] = K.get("NUM_START_GLOBAL_SYNC_ITERS", 10)
    s["N"] = K.get("NUM_LOCAL_SYNC_CORES")
    w.sync = s


# ====================================================================== top level
class WorkerSet:
    def __init__(
        self, cap_dir, index, cache_dirs, gdb, routers, by_coord, decode_header, rank_of_mesh, route_buffer=36
    ):
        self.cap_dir, self.index = cap_dir, index
        self.rbs = route_buffer  # FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE of the run (from a router config)
        self.all_rows = False
        self.wi = index.get("workers") or {}
        self.gdb = gdb
        self.routers, self.by_coord = routers, by_coord
        self.decode_header = decode_header
        self.rank_of_mesh = rank_of_mesh
        self.bins = find_kernel_bins(cache_dirs)
        self.workers = []
        self.scans = {}  # chip -> list of (xy, head, status, sr) for every scanned core
        self.anoms = []
        consts = self.wi.get("constants", {})
        for chip, crec in sorted(self.wi.get("chips", {}).items(), key=lambda kv: int(kv[0])):
            for key, rec in sorted(crec.get("cores", {}).items()):
                self.workers.append(Worker(cap_dir, int(chip), key, rec, consts))
            self.scans[int(chip)] = self.load_scan(crec)
        self.chip_mesh = self.mesh_map()
        self.newest = {}  # chip -> (kernel id, kernel config base) of its newest program
        chips = set(self.scans) | {w.chip for w in self.workers}
        for c in chips:
            heads = [e["head"] for e in self.scans.get(c, [])] or [w.head for w in self.workers if w.chip == c]
            self.newest[c] = newest_program(heads)
        self.lag_notes = 0

    def load_scan(self, crec):
        out = []
        sc = crec.get("scan") or {}
        fn = sc.get("heads_file")
        data = None
        if fn and os.path.exists(os.path.join(self.cap_dir, fn)):
            data = open(os.path.join(self.cap_dir, fn), "rb").read()
        for i, ent in enumerate(sc.get("cores", [])):
            head = None
            if data is not None and ent.get("head_ok", True) and (i + 1) * HEAD_SIZE <= len(data):
                head = parse_head(data[i * HEAD_SIZE : (i + 1) * HEAD_SIZE])
            out.append(
                {
                    "noc0": tuple(ent.get("noc0") or ()),
                    "xlat": tuple(ent["translated"]) if ent.get("translated") else None,
                    "head": head,
                    "status": ent.get("status"),
                    "soft_reset": ent.get("soft_reset"),
                    "selected": ent.get("selected"),
                }
            )
        return out

    def mesh_map(self):
        """chip -> (mesh, dev) from the routers' routing tables, else the workers' (same struct in Tensix L1)."""
        m = {}
        for r in self.routers.values():
            if getattr(r, "l1", None) is not None and r.chip not in m:
                rt = r.l1[0x6F340 : 0x6F340 + 40]
                if len(rt) == 40:
                    m[r.chip] = (u16(rt, 32), u16(rt, 34))
        for w in self.workers:
            if w.chip not in m and w.mem.ok(ROUTING_TABLE + 32, 4):
                m[w.chip] = (w.mem.u16(ROUTING_TABLE + 32), w.mem.u16(ROUTING_TABLE + 34))
        return m

    def where(self, chip):
        md = self.chip_mesh.get(chip)
        if md is None:
            return f"chip{chip}"
        r = self.rank_of_mesh.get(md[0])
        return f"chip{chip}=M{md[0]}D{md[1]}" + (f"/rank{r}" if r is not None else "")

    def decode(self):
        import traceback

        for w in self.workers:
            try:
                self.decode_one(w)
            except Exception as e:  # noqa: BLE001 - one undecodable core must not hide the others
                tb = traceback.format_exc().strip().splitlines()
                w.notes.append(f"decoder error: {type(e).__name__}: {e} ({tb[-3].strip() if len(tb) > 2 else ''})")
                w.flag("decoder error (see its section)")
        self.open_rows = []
        for step in (self.cross_router, self.match_flows, self.router_open_conns_step):
            try:
                step()
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc().strip().splitlines()
                self.anoms.append(
                    f"workers: decoder error in {step.__name__}: {type(e).__name__}: {e} "
                    f"({tb[-3].strip() if len(tb) > 2 else ''})"
                )
        for w in self.workers:
            for a in w.anoms:
                self.anoms.append(f"{w.name} {w.kind or ''}: {a}")

    def decode_one(self, w):
        w.mesh_dev = self.chip_mesh.get(w.chip)
        hd = w.head
        if hd is None:
            w.notes.append("no mailbox captured")
            return
        sig = hd["signal"]
        ln = hd["launch"]
        w.state = {0x80: "running", 0x00: "done"}.get(sig, GO_SIGNAL.get(sig, h(sig)))
        newest = self.newest.get(w.chip)
        w.stale = not in_program(hd, newest)
        if w.stale and hd["launch"]["mode"] == 1:
            w.notes.append("host-launched kernels (dispatch), not part of the test's program")
        elif w.stale:
            lat = hd["latest"]
            w.notes.append(
                f"not in the chip's newest program (kernel ids up to {newest[0] if newest else None}, kernel config "
                f"{h(newest[1]) if newest else None}): its last kernels"
                + (
                    f" (ids {sorted(set(x for x, n in zip(hd['slots'][lat]['wids'], hd['slots'][lat]['text_size']) if n))})"
                    if lat is not None
                    else ""
                )
                + " belong to an earlier program; state shown for reference only"
            )
            if lat is not None and lat != hd["slot"] and sig != 0x80:  # describe what it last ran, not a stale slot
                hd = w.head = dict(hd, slot=lat, slot_why="its own last kernels", launch=hd["slots"][lat])
                ln = hd["launch"]
        w.niu_out = niu_outstanding(w.niu)
        for p, risc in enumerate(RISC_DIRS):
            ran = (ln["enables"] >> p) & 1 if sig == 0x80 else ln["text_size"][p] > 0
            if not ran:
                continue
            kb, how = match_kernel(w.mem, ln["kcb"] + ln["text_off"][p], self.bins.get(risc, []))
            w.kernels[RISCS[p]] = (kb, how)
        kb = w.kernels.get("BRISC", (None, ""))[0]
        w.kb = kb
        w.kind = kb.kind if kb else None
        if kb is None or kb.kind not in KIND_TYPES:
            return
        for noc, o in w.niu_out.items():  # counters span every RISC and kernel since reset: a running test core only
            if (o["wr"] or o["rd"] or o["at"]) and w.state == "running" and not w.stale:
                w.flag(
                    f"NoC{noc}: {o['wr']} non-posted write(s), {o['rd']} read(s), {o['at']} atomic(s) not "
                    f"acknowledged (NIU counters)"
                )
        decode_common(w)
        if w.common is None or w.stale:  # a core outside the newest program: status only, no checks
            return
        kb.load_layouts(self.gdb)
        if kb.gdb_error:
            w.notes.append(kb.gdb_error)
        if "main" not in (kb.layouts or {}):
            return
        kcs = kb.consts.get("KERNEL_CONFIG_BUFFER_SIZE")
        if kcs and w.common["kernel_config_size"] != kcs:
            w.notes.append(f"kernel config size {w.common['kernel_config_size']} != compile-time {kcs}")
        if not w.mem.ok(w.common["kernel_config_base"], kb.layouts["main"].size or 4):
            w.notes.append(f"kernel config buffer {w.common['kernel_config_base']:#x} not captured")
            return
        if kb.kind == "sender":
            decode_sender(w, kb, self.decode_header, self.rbs)
        elif kb.kind == "receiver":
            decode_receiver(w, kb)
        elif kb.kind == "sync":
            decode_sync(w, kb, self.decode_header)
        self.phase(w)

    # ------------------------------------------------------------ what each core waits for
    def phase(self, w):
        done = w.status == 0xABC00001
        if w.status == 0xABC00003:
            w.flag("result status DATA_MISMATCH")
        if w.state == "running" and done:
            w.waits.append("status PASS written, kernel not yet returned (final NoC barrier, or exiting)")
        if w.state == "done" and w.status == 0xABC00000:
            w.flag("go message DONE but the result status is still STARTED")
        if w.kind == "sender":
            self.phase_sender(w, done)
        elif w.kind == "receiver":
            self.phase_receiver(w, done)
        elif w.kind == "sync":
            self.phase_sync(w, done)
        for c in w.conns:
            if c.get("initialized") and c.get("free") == 0 and not done:
                rt = self.router_side(w, c)
                rtxt = f"; router {rt['name']} read counter {rt['rdc']}" if rt and rt.get("name") else ""
                w.waits.append(
                    f"conn{c['idx']}: no free router slot: sent {c['sent']}, credit word {c['credit']} "
                    f"(N {c['N']}){rtxt}"
                )

    def phase_sender(self, w, done):
        ls = getattr(w, "local_sync", None)
        left = [f for f in w.flows_out if f["n"] is not None and f["sent"] is not None and f["sent"] < f["n"]]
        if done:
            return
        if w.bench:
            w.notes.append(
                "benchmark mode: the connection and counters in L1 are written back only after the send loop"
            )
        if ls and ls["value"] is not None:
            v = ls["value"]
            if v == 0:
                w.waits.append(
                    f"local sync 0: waits for its sync semaphore @{ls['addr']:#x} == 1 (start signal from "
                    f"the sync core), value {v}"
                )
                return
            if v > 3:
                w.flag(f"local sync semaphore {v} > 3: an equality wait can never complete (extra increment)")
        if left:
            for f in left:
                why = []
                if f.get("credit_on") and f.get("credit"):
                    cr = f["credit"]
                    mn = min([x for x in cr["words"] if x is not None], default=None)
                    if cr["estimated"] == 0 and mn is not None and mn - (cr["prev"] or 0) <= 0:
                        why.append(f"no receiver credits (min credit word {mn}, already used {cr['prev']})")
                for ci in f["conn_list"] or [f["conn_idx"]]:
                    c = next((c for c in w.conns if c["idx"] == ci), None)
                    if c and c.get("free") == 0:
                        why.append(f"conn{ci} full")
                w.waits.append(
                    f"cfg{f['i']}: sent {f['sent']}/{f['n']}"
                    + (
                        f" ({'; '.join(why)})"
                        if why
                        else " (not blocked on credits or slots in L1: still sending, or stuck elsewhere)"
                    )
                )
            return
        if ls and ls["value"] == 1:
            w.waits.append(
                "all packets sent; local sync 1 waits semaphore == 2 (the sync core is still in local "
                "sync 0, waiting for the other senders' start acks), or closing connections"
            )
            for c in w.conns:
                if c.get("initialized") and c.get("teardown") not in (None, 0, 1):
                    w.flag(f"conn{c['idx']} teardown word {c['teardown']} (expected 0 or 1)")
        elif ls and ls["value"] == 2:
            w.waits.append(
                "all packets sent; local sync 2: waits semaphore == 3 (the sync core is in its end "
                "global sync), or closing connections (teardown ack)"
            )
        elif ls and ls["value"] == 3:
            w.waits.append("all syncs done; terminating muxes / writing results")
        else:
            w.waits.append("all packets sent (no line sync): closing connections or writing results")

    def phase_receiver(self, w, done):
        if done or w.bench:
            return
        for f in w.flows_in:
            if f["n"] is None or f["got"] is None or f["got"] >= f["n"]:
                continue
            p = f["poll"]
            t = NOC_SEND.get(f["type"], f["type"])
            s = f"cfg{f['i']} ({t}): validated {f['got']}/{f['n']}, waits for packet {f['got'] + 1}"
            if "addr" in p:
                s += f": polls @{p['addr']:#x} for {h(p['want'])}, holds {h(p['have'])}"
                if p.get("have") == p.get("want"):
                    s += " (match: data is there, not yet validated)"
                hold = p.get("slot_holds")
                if hold is not None:
                    s += f"; slot data is packet {hold + 1}'s"
                    if hold > f["got"]:
                        w.flag(
                            f"cfg{f['i']}: slot for packet {f['got'] + 1} holds packet {hold + 1}'s data "
                            f"(overwritten by a later packet: receiver buffer overrun)"
                        )
                elif p.get("first") is not None:
                    s += f"; slot first word {h(p['first'])} matches no packet of this flow (stale or foreign data)"
            if "addr2" in p:
                s += f"; chunk 2 @{p['addr2']:#x} wants {h(p['want2'])} holds {h(p['have2'])}"
            if "atomic_addr" in p:
                s += (
                    f"; atomic @{h(p['atomic_addr'])} = {p['atomic_have']} wants >= {p['atomic_want']} "
                    f"(step {p['atomic_step']})"
                )
                if None not in (p["atomic_have"], p["atomic_want"], p["atomic_step"]) and p["atomic_step"]:
                    if p["atomic_have"] < p["atomic_want"]:
                        miss = -(-(p["atomic_want"] - p["atomic_have"]) // p["atomic_step"])
                        s += f": {miss} increment(s) short"
                    else:
                        s += ": counter reached (not yet validated)"
            if p.get("seed_ok") is False:
                w.flag(f"cfg{f['i']}: seed {h(f['seed'])} is not the initial seed advanced {f['got']} times")
            w.waits.append(s)

    def phase_sync(self, w, done):
        s = w.sync
        if not s or done:
            return
        L, g, it, N = s.get("L"), s.get("global"), s.get("iters", 10), s.get("N")
        lv = s.get("local")
        if not L or g is None or lv is None or not N:
            w.waits.append(f"sync state not readable (per-round value {L}, global {g}, local {lv}, cores {N})")
            return
        rounds = g // L if L else None
        if lv == 0:
            need = L * (min(rounds, it - 1) + 1) if rounds is not None else None
            w.waits.append(
                f"start global sync round {min(rounds, it - 1) + 1}/{it}: global semaphore "
                f"@{s['global_addr']:#x} = {g}, waits >= {need} ({L} sync packets per round)"
            )
            if g % L:
                w.notes.append(f"global semaphore {g} is not a multiple of {L}: a round is part way in")
        elif lv < N:
            w.waits.append(
                f"local sync 0 (start): local semaphore @{s['local_addr']:#x} = {lv}, waits == {N} "
                f"(acks from {N - 1} sender core(s))"
            )
        elif lv < 2 * N:
            w.waits.append(
                f"local sync 1 (end of traffic): local semaphore = {lv}, waits == {2 * N}: "
                f"{2 * N - lv} sender core(s) not finished sending"
            )
        elif lv == 2 * N:
            tgt = L * (it + 1)
            w.waits.append(
                f"end global sync: global semaphore = {g}, waits >= {tgt} "
                + (f"({tgt - g} sync packets missing)" if g < tgt else "(already reached: stuck after it)")
            )
        elif lv < 3 * N:
            w.waits.append(f"local sync 2: local semaphore = {lv}, waits == {3 * N}")
        else:
            w.waits.append(f"syncs done (local {lv}); terminating muxes / writing results")
        if lv > 3 * N:
            w.flag(f"local semaphore {lv} > {3 * N}: an equality wait can never complete (extra increment)")

    # ------------------------------------------------------------ worker <-> router
    def router_side(self, w, c):
        if not c.get("initialized") or not c.get("edm_xy"):
            return None
        r = self.by_coord.get((w.chip, c["edm_xy"][0], c["edm_xy"][1]))
        out = {"core": r, "name": r.name if r else None}
        if r is None or r.l1 is None or not getattr(r, "e0", None):
            return out
        e0 = r.e0
        ci = e0.g("LOCAL_SENDER_CH_0_CONN_INFO_ADDR")
        out.update(
            {
                "ci": ci,
                "sem": r.w(ci),
                "teardown_addr": r.w(ci + 16),
                "wxy": r.w(ci + 32),
                "rdc": r.w(ci + 48),
                "conn": r.w(ci + 96),
            }
        )
        cur = c.get("edm_copy_of_wr_counter_addr")
        out["cursor"] = (r.w(cur), r.w(cur + 4)) if cur else None
        # Router channel layout (erisc_datamover_builder.cpp): conn info (64 B), local flow-control, producer
        # terminate, connection semaphore (+96), buffer-index semaphore = SenderChannelProducerCursor (+112).
        out["cursor_addr_ok"] = cur == ci + 112 if cur and ci else None
        sid = e0.sender_stream(0)
        n = e0.sender_slots[0] if e0.ct else None
        v = r.stream(sid)
        out["N"] = n
        out["unsent"] = n - (v & 0x1FFFF) if (n is not None and v is not None and (v & 0x1FFFF) <= n) else None
        rcc = e0.g("TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR")
        out["completions"] = r.w(rcc) if rcc else None
        out["location_ok"] = c.get("edm_worker_location_info_addr") == ci
        return out

    @staticmethod
    def conn_has_left(w, c):
        """A sender config that still has packets to send through this connection."""
        return any(
            f["sent"] is not None
            and f["n"] is not None
            and f["sent"] < f["n"]
            and c["idx"] in (f["conn_list"] or [f["conn_idx"]])
            for f in w.flows_out
        )

    def cross_router(self):
        self.conn_rows = []
        for w in self.workers:
            ft = fabric_table(w)
            for c in w.conns:
                if c.get("mux") or not c.get("initialized"):
                    if c.get("mux"):
                        self.conn_rows.append((w, c, None, ["mux connection: not decoded"]))
                    continue
                rt = self.router_side(w, c)
                issues = []
                me = w.xlat
                if ft and c["eth_chan"] in ft["ents"]:
                    e = ft["ents"][c["eth_chan"]]
                    if e["edm_xy"] != c["edm_xy"] or e["location_info"] != c["edm_worker_location_info_addr"]:
                        issues.append(f"adapter != fabric table entry for eth{c['eth_chan']}")
                if rt is None or rt.get("core") is None:
                    issues.append(f"router at {c['edm_xy']} not in the capture")
                    self.conn_rows.append((w, c, rt, issues))
                    continue
                if rt.get("ci") is None:
                    issues.append("router config not decoded")
                    self.conn_rows.append((w, c, rt, issues))
                    continue
                if not rt["location_ok"]:
                    issues.append(
                        f"adapter location info {h(c['edm_worker_location_info_addr'])} != router ch0 "
                        f"conn info {h(rt['ci'])}"
                    )
                if rt.get("cursor_addr_ok") is False:
                    issues.append(
                        f"adapter producer-cursor address {h(c['edm_copy_of_wr_counter_addr'])} != router "
                        f"ch0 cursor {rt['ci'] + 112:#x}"
                    )
                wxy = rt["wxy"]
                last = me is not None and wxy is not None and (wxy & 0xFFFF, wxy >> 16) == me
                rt["last_producer"] = last
                done = w.status == 0xABC00001
                if last:
                    if rt["sem"] != c["edm_buffer_local_free_slots_read_ptr"]:
                        issues.append(
                            f"router writes credits to {h(rt['sem'])}, adapter reads "
                            f"{h(c['edm_buffer_local_free_slots_read_ptr'])}"
                        )
                    rd, cr, sent = rt["rdc"], c["credit"], c["sent"]
                    if None not in (rd, cr):
                        if rd != cr:
                            diff = (rd - cr) & 0xFFFFFFFF
                            n = c.get("N") or rt.get("N")
                            waiting = not done and (c.get("free") == 0 or self.conn_has_left(w, c))
                            # The router credits the worker's L1 only while it is connected (credit_worker); from the
                            # teardown on, completions of packets still in flight go to its connection info
                            # (edm_read_counter), so a closed worker's word may trail by up to N.
                            if diff < 1 << 31 and rt["conn"] == 0 and not waiting and n and diff <= n:
                                c["lag"] = diff
                                self.lag_notes += 1
                                w.notes.append(
                                    f"conn{c['idx']} -> {rt['name']}: credit word {cr} trails the router read counter "
                                    f"{rd} by {diff} (<= {n} slots) after close: packets in flight at close are "
                                    f"credited to the router's connection info, not to the worker (expected)"
                                )
                            elif diff < 1 << 31:
                                why = (
                                    (
                                        "worker still running and "
                                        + ("out of free slots" if c.get("free") == 0 else "has packets to send on it")
                                    )
                                    if waiting
                                    else (
                                        f"more than the {n} slots in flight at close"
                                        if n and diff > n
                                        else "connection open: the router writes every credit to the worker"
                                    )
                                )
                                w.flag(
                                    f"conn{c['idx']} -> {rt['name']}: router read counter {rd} but worker credit "
                                    f"word {cr}: {diff} credit(s) not delivered to the worker ({why})"
                                )
                            else:
                                w.flag(
                                    f"conn{c['idx']} -> {rt['name']}: worker credit word {cr} ahead of router read "
                                    f"counter {rd} (extra credit)"
                                )
                    if None not in (rd, sent) and rt.get("unsent") == 0:
                        d = (sent - rd) & 0xFFFFFFFF
                        if d and d < 1 << 31:
                            w.flag(
                                f"conn{c['idx']} -> {rt['name']}: worker counted {sent} packets sent, router freed "
                                f"{rd} and holds none: {d} packet(s) the router never received"
                            )
                        elif d:
                            w.flag(f"conn{c['idx']} -> {rt['name']}: router freed {rd} slots, worker sent only {sent}")
                    cs = CONN_STATE.get(rt["conn"], h(rt["conn"]))
                    if done and rt["conn"] == 1:
                        w.flag(
                            f"conn{c['idx']} -> {rt['name']}: worker finished but the router connection is still open"
                        )
                    if rt["conn"] == 2:
                        issues.append("router: close requested, not yet processed")
                    if not done and rt["conn"] == 0 and w.kind == "sender":
                        left = [
                            f
                            for f in w.flows_out
                            if f["sent"] is not None
                            and f["n"] is not None
                            and f["sent"] < f["n"]
                            and c["idx"] in (f["conn_list"] or [f["conn_idx"]])
                        ]
                        if left:
                            w.flag(
                                f"conn{c['idx']} -> {rt['name']}: {len(left)} config(s) still to send but the "
                                f"router connection is {cs}"
                            )
                    cur = rt.get("cursor")
                    if cur and rt["conn"] == 0 and sent is not None and cur[0] is not None and cur[0] != sent:
                        issues.append(f"router producer cursor {cur[0]} != worker counter {sent} after close")
                else:
                    issues.append(
                        "router's last producer is "
                        + (f"({wxy & 0xFFFF},{wxy >> 16})" if wxy else "none")
                        + ": counters not comparable"
                    )
                self.conn_rows.append((w, c, rt, issues))

    def router_open_conns_step(self):
        self.open_rows = self.router_open_conns()

    def router_open_conns(self):
        """Routers whose worker channel (E0 ch0) is open or closing: is that worker captured, and still running?"""
        rows = []
        by_xy = {(w.chip, w.xlat): w for w in self.workers if w.xlat}
        for r in sorted(self.routers.values(), key=lambda c: (c.chip, c.chan)):
            if r.l1 is None or not getattr(r, "e0", None):
                continue
            ci = r.e0.g("LOCAL_SENDER_CH_0_CONN_INFO_ADDR")
            st, wxy = r.w(ci + 96), r.w(ci + 32)
            if st in (None, 0) or not wxy:
                continue
            xy = (wxy & 0xFFFF, wxy >> 16)
            w = by_xy.get((r.chip, xy))
            what = f"{r.name} worker channel {CONN_STATE.get(st, h(st))} to ({xy[0]},{xy[1]})"
            if w is None:
                rows.append(what + ": that core is not in the worker capture")
            else:
                rows.append(what + f" = {w.name} {w.kind} go {w.state} status {status_text(w.status)}")
                if w.state == "done" and st == 1:
                    w.flag(f"{r.name} still has its worker connection open to this core, which finished")
        return rows

    # ------------------------------------------------------------ sender <-> receiver
    def match_flows(self):
        self.flow_rows = []
        recv = {}
        for w in self.workers:
            if w.kind != "receiver" or w.mesh_dev is None or w.xlat is None:
                continue
            for f in w.flows_in:
                for a in {f.get("dst_addr"), (f.get("args") or {}).get("atomic_dst")} - {None}:
                    recv[(w.mesh_dev[0], w.mesh_dev[1], w.xlat[0], w.xlat[1], a)] = (w, f)
        used = set()
        for w in self.workers:
            if w.kind != "sender":
                continue
            for f in w.flows_out:
                key = None
                if None not in (f.get("dst_mesh"), f.get("dst_dev"), f.get("dst_xy"), f.get("dst_addr")):
                    key = (f["dst_mesh"], f["dst_dev"], f["dst_xy"][0], f["dst_xy"][1], f["dst_addr"])
                r = recv.get(key) if key else None
                if r:
                    used.add((id(r[0]), r[1]["i"]))
                self.flow_rows.append((w, f, r))
        self.unmatched_recv = [
            (w, f) for w in self.workers if w.kind == "receiver" for f in w.flows_in if (id(w), f["i"]) not in used
        ]
        for w, f, r in self.flow_rows:
            if not r or f["sent"] is None or r[1]["got"] is None:
                continue
            rw, rf = r
            if rw.bench or w.bench:
                continue
            if f["sent"] == f["n"] and rf["got"] < rf["n"] and rw.status != 0xABC00001:
                rw.flag(
                    f"cfg{rf['i']}: its sender {w.name} cfg{f['i']} sent all {f['sent']} packets, receiver "
                    f"validated {rf['got']}: {f['sent'] - rf['got']} not arrived (or not as expected)"
                )
            if rf["got"] > f["sent"]:
                rw.flag(
                    f"cfg{rf['i']}: validated {rf['got']} packets but its sender {w.name} cfg{f['i']} counted "
                    f"only {f['sent']} sent"
                )

    # ------------------------------------------------------------ report
    def report_detail(self, P):
        P("\n== Workers: Tensix cores of the last program (per core) ==")
        cst = self.wi.get("constants", {})
        P(
            f"   (scan of {sum(len(v) for v in self.scans.values())} Tensix cores; {len(self.workers)} captured: "
            f"window [0, {h(cst.get('window'))}), whole L1 for cores still running a test kernel; "
            f"kernel binaries from {sum(len(v) for v in self.bins.values())} cached ELFs)"
        )
        for w in sorted(self.workers, key=lambda w: (w.chip, w.xlat or w.noc0)):
            self.core_lines(w, P)

    def core_lines(self, w, P):
        hd = w.head
        sel = ",".join(w.rec.get("selected_because", []))
        full = " (whole L1)" if w.rec.get("full") else ""
        P(f"=== {w.name} noc0 {w.noc0} logical {w.logical} {self.where(w.chip)} [{sel}]{full}")
        if hd is None:
            P("  no mailbox")
            for n in w.notes:
                P(f"  note: {n}")
            return
        ln = hd["launch"]
        sr = (w.regs.get("pass1") or {}).get("soft_reset")
        P(
            f"  go {GO_SIGNAL.get(hd['signal'], h(hd['signal']))} (go_msg[{hd['go_idx']}].slot[{hd['rd_ptr'] & (NUM_LAUNCH - 1)}]), launch slot {hd['slot']} "
            f"({hd['slot_why']}), kernel ids {sorted(set(x for x, n in zip(ln['wids'], ln['text_size']) if n))}"
            + (f" program id {ln['host_id']}" if ln["host_id"] else "")
            + f", kernel config {ln['kcb']:#x}, mode "
            f"{DISPATCH_MODE.get(ln['mode'], ln['mode'])}, enables {ln['enables']:#x}; soft reset {soft_reset_text(sr)}"
        )
        for p, risc in enumerate(RISCS):
            if risc in w.kernels:
                kb, how = w.kernels[risc]
                P(
                    f"  {risc}: text @{ln['kcb'] + ln['text_off'][p]:#x} ({ln['text_size'][p]} B) "
                    + (f"= {kb.label()} ({how})" if kb else f"unidentified ({how})")
                )
        p2 = w.regs.get("pass2") or {}
        p1 = w.regs.get("pass1") or {}
        moved = [
            k
            for k in ("go", "status", "fabric_rw")
            if p1.get(k) is not None and p2.get(k) is not None and p1.get(k) != p2.get(k)
        ]
        if moved:
            P(f"  changed between the two samples: {moved}")
        niu = getattr(w, "niu_out", None) or {}
        if niu:
            parts = []
            for noc, o in niu.items():
                g = o["raw"]
                parts.append(
                    f"NoC{noc} wr {g['NONPOSTED_WR_REQ_SENT']}/{g['WR_ACK_RECEIVED']} rd "
                    f"{g['RD_REQ_SENT']}/{g['RD_RESP_RECEIVED']} at {g['NONPOSTED_ATOMIC_SENT']}/"
                    f"{g['ATOMIC_RESP_RECEIVED']} posted {g['POSTED_WR_REQ_SENT']}"
                    + (
                        f" (outstanding wr {o['wr']} rd {o['rd']} at {o['at']})"
                        if o["wr"] or o["rd"] or o["at"]
                        else ""
                    )
                )
            P("  NIU sent/acked: " + "; ".join(parts))
        if w.common:
            c = w.common
            P(
                f"  test memory map: local args {c['local_args_base']:#x}, results {c['result_buffer_base']:#x}, "
                f"kernel config {c['kernel_config_base']:#x} | status {status_text(w.status)}, packets "
                f"{w.result_packets}, cycles {w.result_cycles}"
            )
        if w.kb and w.kb.consts:
            P(f"  compile-time: " + " ".join(f"{k}={v}" for k, v in w.kb.consts.items()))
        for f in w.flows_out:
            P(
                f"   send cfg{f['i']}: {NOC_SEND.get(f['type'], f['type'])} {f['payload']} B, sent {f['sent']}/{f['n']} "
                f"via conn {f['conn_list'] or f['conn_idx']} -> M{f['dst_mesh']}D{f['dst_dev']} {f['dst_xy']} "
                f"@{h(f['dst_addr'])}" + (f" | credits {f['credit']}" if f.get("credit_on") else "")
            )
            if f.get("hdr"):
                P(f"      header @{h(f['hdr_addr'])}: {f['hdr']}")
        for f in w.flows_in:
            p = f["poll"]
            busy = f["got"] is not None and f["n"] is not None and f["got"] < f["n"]
            P(
                f"   recv cfg{f['i']}: {NOC_SEND.get(f['type'], f['type'])} validated {f['got']}/{f['n']} @{h(f['dst_addr'])}"
                f" seed {h(f['seed'])} (initial {h((f.get('args') or {}).get('seed0'))})"
                + (
                    f" | buffer {h(p.get('buf_base'))} offset {p.get('offset')} of {p.get('total')} payload {p.get('payload')}"
                    if "buf_base" in p
                    else ""
                )
                + (
                    f" | poll @{h(p.get('addr'))} want {h(p.get('want'))} have {h(p.get('have'))}"
                    if "addr" in p and busy
                    else ""
                )
                + (
                    f" | atomic @{h(p.get('atomic_addr'))} = {p.get('atomic_have')} want {p.get('atomic_want')}"
                    if "atomic_addr" in p
                    else ""
                )
            )
        if w.sync:
            s = w.sync
            P(
                f"   sync: global semaphore @{h(s.get('global_addr'))} = {s.get('global')} (per round {s.get('L')}, "
                f"{s.get('iters')} start rounds + 1 end), local semaphore @{h(s.get('local_addr'))} = {s.get('local')} "
                f"(local cores {s.get('local_cores')})"
            )
            for i, c in enumerate(s["configs"]):
                P(f"      sync cfg{i}: conns {c['conns']} header {c['hdr']}")
        ls = getattr(w, "local_sync", None)
        if ls:
            P(
                f"   local sync semaphore @{h(ls['addr'])} = {ls['value']} (sync core {ls['cores'][0] if ls['cores'] else None})"
            )
        for c in w.conns:
            if c.get("mux"):
                P(f"   conn{c['idx']}: mux connection (not decoded)")
                continue
            if not c.get("initialized"):
                P(f"   conn{c['idx']}: adapter not initialized")
                continue
            P(
                f"   conn{c['idx']}: eth{c['eth_chan']} router {c['edm_xy']} N {c['N']} | sent {c['sent']} slot "
                f"{c['slot_index']} | credit word @{h(c['edm_buffer_local_free_slots_read_ptr'])} = {c['credit']} "
                f"(free {c['free']}) | teardown word @{h(c['worker_teardown_addr'])} = {c['teardown']} | cursor at open "
                f"{c['cursor_in']}"
            )
        for wt in w.waits:
            P(f"  WAITS: {wt}")
        for n in w.notes:
            P(f"  note: {n}")
        for a in w.anoms:
            P(f"  <<< {a}")

    def report_summary(self, P):
        P("\n== Workers not finished (who waits for what) ==")
        running = [
            w
            for w in self.workers
            if w.kind in KIND_TYPES
            and not getattr(w, "stale", False)
            and (w.status != 0xABC00001 or w.state == "running")
        ]
        stale_t = [w for w in self.workers if w.kind in KIND_TYPES and getattr(w, "stale", False)]
        if stale_t:
            P(
                f"  ({len(stale_t)} test core(s) last ran an earlier program and are left out: "
                + ", ".join(f"{w.name} {w.kind} {status_text(w.status)}" for w in stale_t[:8])
                + (" ..." if len(stale_t) > 8 else "")
                + ")"
            )
        if not self.workers:
            P("  no worker cores captured")
        for w in sorted(running, key=lambda w: (w.chip, w.xlat or w.noc0)):
            P(f"  {w.name} {self.where(w.chip)} {w.kind}: go {w.state}, status {status_text(w.status)}")
            for wt in w.waits or ["(no wait point inferred)"]:
                P(f"      {wt}")
        others = [w for w in self.workers if w.kind not in KIND_TYPES and w.state == "running"]
        groups = {}
        for w in sorted(others, key=lambda w: (w.chip, w.xlat or w.noc0)):
            names = " ".join(sorted({kb.name if kb else "?" for kb, _ in w.kernels.values()})) or "?"
            groups.setdefault(names, []).append(f"chip{w.chip}{w.xlat or w.noc0}")
        for names, where in groups.items():
            disp = all(n.startswith("cq_") for n in names.split())
            P(
                f"  running non-test kernels [{names}] on {len(where)} core(s)"
                + (" (dispatch: expected to run)" if disp else "")
                + ": "
                + " ".join(where[:12])
                + (" ..." if len(where) > 12 else "")
            )
        counts = {c: len(sc) for c, sc in self.scans.items()}
        full_n = max(counts.values()) if counts else 0
        partial = sorted(c for c, sc in self.scans.items() if len(sc) < full_n or any(e["head"] is None for e in sc))
        router_chips = {r.chip for r in self.routers.values()}
        missing = sorted(router_chips - set(self.scans)) if self.scans else sorted(router_chips)
        if partial or missing:
            P(
                f"  worker scan incomplete: chip(s) {partial} partly unreadable, chip(s) {missing} not scanned; what "
                f"follows covers the captured cores only"
            )
        if not running:
            tests = [w for w in self.workers if w.kind in KIND_TYPES and not getattr(w, "stale", False)]
            all_done = tests and all(w.state == "done" for w in tests)
            P(
                "  none: every captured test kernel wrote status PASS"
                + (f" and reported DONE ({len(tests)} cores)" if all_done else "")
                + (
                    ": if the run hung, the wait is outside the test kernels (dispatch completion, the host or the "
                    "other rank)"
                    if all_done and not partial and not missing
                    else ""
                )
            )
        stale = []
        for chip, sc in sorted(self.scans.items()):
            for e in sc:
                hd = e["head"]
                if hd and hd["signal"] == 0x80 and not e.get("selected"):
                    stale.append(f"chip{chip}{e['xlat'] or e['noc0']}")
        if stale:
            P(f"  cores in GO state not captured: {stale[:20]}")
        P("\n== Workers by rank (mesh) ==")
        by = {}
        for w in self.workers:
            if w.kind in KIND_TYPES and not getattr(w, "stale", False):
                md = w.mesh_dev or (None, None)
                by.setdefault((self.rank_of_mesh.get(md[0]), md[0]), {}).setdefault(w.chip, []).append(w)
        for (rank, mesh), chips in sorted(by.items(), key=lambda kv: (kv[0][1] is None, kv[0][1] or 0)):
            P(f"  rank {rank} mesh {mesh}:")
            for chip, ws in sorted(chips.items()):
                cnt = {}
                for w in ws:
                    k = f"{w.kind} {'done' if w.status == 0xABC00001 else status_text(w.status)}"
                    cnt[k] = cnt.get(k, 0) + 1
                sy = next((w for w in ws if w.kind == "sync" and w.sync), None)
                stxt = ""
                if sy:
                    s = sy.sync
                    gw = (s.get("L") or 0) * (s.get("iters", 10) + 1)
                    stxt = f" | sync core global {s.get('global')}/{gw} local {s.get('local')}/{3 * (s.get('N') or 0)}"
                P(f"    {self.where(chip)}: " + ", ".join(f"{v} {k}" for k, v in sorted(cnt.items())) + stxt)
        P("\n== Worker flows: sender vs receiver (packets) ==")
        ok = listed = 0
        for w, f, r in self.flow_rows:
            rtxt = "receiver not captured/matched"
            same = False
            if r:
                rw, rf = r
                rtxt = f"{rw.name} cfg{rf['i']} validated {rf['got']}/{rf['n']}"
                same = f["sent"] is not None and f["sent"] == rf["got"] == f["n"] == rf["n"]
            if same and not self.all_rows:
                ok += 1
                continue
            bad = r and f["sent"] is not None and r[1]["got"] is not None and f["sent"] != r[1]["got"]
            listed += 1
            P(
                f"  {w.name} cfg{f['i']} sent {f['sent']}/{f['n']} -> M{f.get('dst_mesh')}D{f.get('dst_dev')} "
                f"{f.get('dst_xy')} @{h(f.get('dst_addr'))}: {rtxt}" + ("  <<< differ" if bad else "")
            )
        for w, f in self.unmatched_recv:
            if getattr(w, "stale", False):
                continue
            listed += 1
            P(f"  (no captured sender) -> {w.name} cfg{f['i']} validated {f['got']}/{f['n']} @{h(f.get('dst_addr'))}")
        if ok:
            P(
                f"  {f'{ok} more' if listed else f'all {ok}'} flow(s) complete on both ends (sent == validated == "
                f"expected; --all-worker-rows lists them)"
            )
        if not self.flow_rows:
            P("  no sender traffic configs decoded")
        P("\n== Worker-router connections: worker adapter vs router channel 0 ==")
        P("   (sent = worker's write counter; credit = router read counter as written into worker L1; quiet: equal)")
        for row in self.open_rows:
            P(f"  open: {row}")
        ok = lagged = listed = 0
        for w, c, rt, issues in self.conn_rows:
            listed += 1
            if rt is None or rt.get("ci") is None:
                P(f"  {w.name} conn{c['idx']}: " + "; ".join(issues))
                continue
            trail = None if None in (c.get("sent"), c.get("credit")) else (c["sent"] - c["credit"]) & 0xFFFFFFFF
            quiet = (
                rt.get("conn") == 0
                and trail is not None
                and trail <= (c.get("N") or 0)
                and rt.get("unsent") == 0
                and (not rt.get("last_producer") or rt.get("rdc") == c.get("sent"))
                and not any(a.startswith(f"conn{c['idx']} ") for a in w.anoms)
            )
            real = [i for i in issues if not i.startswith("router's last producer is")]
            if quiet and not real and not self.all_rows:
                ok += 1
                listed -= 1
                lagged += bool(trail)
                continue
            rtxt = (
                f"{rt.get('name')} conn {CONN_STATE.get(rt.get('conn'), h(rt.get('conn')))} rd {rt.get('rdc')} "
                f"unsent {rt.get('unsent')} completions {rt.get('completions')} cursor {rt.get('cursor')} "
                f"last producer {'this worker' if rt.get('last_producer') else 'other'}"
            )
            P(
                f"  {w.name} {w.kind} conn{c['idx']} eth{c['eth_chan']}: sent {c['sent']} credit {c['credit']} free "
                f"{c['free']}/{c['N']} teardown {c['teardown']} | {rtxt}" + ("".join(f"  [{i}]" for i in issues))
            )
        if ok:
            P(
                f"  {f'{ok} more' if listed else f'all {ok}'} connection(s) quiet and consistent (closed, nothing "
                f"unsent, sent == router read counter "
                f"when this worker was the router's last producer; --all-worker-rows lists them)"
                + (
                    f"; in {lagged} of them the worker's credit word trails by up to the slot count: credits for "
                    f"packets in flight at close go to the router's connection info, not to the worker (expected)"
                    if lagged
                    else ""
                )
            )

    def anomaly_lines(self):
        return self.anoms


def rank_bindings(path):
    """mesh id -> rank from a ttrun rank-bindings yaml (only the rank / mesh_id pairs are read)."""
    out = {}
    try:
        txt = open(path).read()
    except OSError:
        return out
    for m in re.finditer(r"rank:\s*(\d+)\s*\n\s*mesh_id:\s*(\d+)", txt):
        out[int(m.group(2))] = int(m.group(1))
    return out
