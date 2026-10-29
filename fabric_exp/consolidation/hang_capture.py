#!/usr/bin/env python3
"""Read-only hang-state capture of the fabric routers on every chip (Blackhole, tt-exalens).

  hang_capture.py OUT_DIR [--all-eth] [--chips 0,1,..] [--noc 0|1] [--time-budget S] [--meta KEY=VALUE ...]

Run with /home/aagarwal/tt-metal/python_env/bin/python after every process using the device has exited. For each
chip and each connected Ethernet core (--all-eth: every non-harvested one) it reads, in this order:
  pass1  registers: NoC overlay stream registers 0..31 (indices 4 REMOTE_SRC/scratch, 10 REMOTE_DEST_BUF_SIZE,
         297 REMOTE_DEST_BUF_SPACE_AVAILABLE = the free-slots / packets / credit counts), the RISC soft-reset register
         (0xFFB121B0) and a few L1 words that move while a router runs (fabric heartbeat 0x7CC70, fabric postcode
         0x6FE50, go message 0x590);
  l1     the whole 512 KiB L1 (0x0..0x80000): mailbox + launch message, both kernel binaries and their runtime args,
         channel buffers (packet headers in every slot), connection info, credit counters, routing table, base FW area;
  pass2  the pass1 registers again, so the decoder can tell static (hung) state from state that still moves;
  txq1, txq2  last, twice: the three Ethernet TX queues' documented registers (0xFFB90000 + q*0x1000; the block
         the router itself drives, so it is read only once everything else is on disk; --no-txq skips it).
Output: OUT_DIR/index.json (topology, coordinates, register values, file list, timings, errors) and
OUT_DIR/chip<C>/eth<NN>_l1.bin. hang_decode.py turns that into per-router channel/credit tables.

Workers (Tensix cores; on by default, --no-workers skips them), after the routers, in this order:
  wscan  every Tensix core of every chip: the mailbox head (0x60, 1488 B: launch-message ring, go messages, read
         pointer), the test result-status word (--worker-status-addr, default 0x23480 = the test's result buffer at
         the HAL's DEFAULT_UNRESERVED base + 32 KiB of local args) and the soft-reset register. A core is captured if
         its go signal is GO with kernels enabled, or the launch slot it last executed holds the chip's newest program
         (highest kernel id; same kernel config base) and is its own most recent slot, or its status word carries the
         test's 0xabc0000x status;
  wl1    each captured core's L1 from 0 to --worker-window (default 256 KiB: mailbox, fabric connection table,
         kernel-config ring with binaries and runtime args, the test's local args, result buffer, kernel config
         buffer, sender credit / header / sync regions); the whole L1 (1.5 MiB) for cores that still run a test
         kernel (go GO in the newest program, or status STARTED), at most --worker-full-max of them;
  wpass2 per captured core again: mailbox head, status word, soft reset, fabric connection credit words;
  wniu   per captured core and NoC, the NIU transaction counters (NOC_STATUS 0..15, 48..61: requests sent / acks
         and responses received), --no-worker-niu skips them;
and then, without the device, the kernel ELFs of every non-router kernel of the run's cache (--kernel-cache, default
the --meta cache=DIR value) are copied to OUT_DIR/cache/kernels/<name>/<hash>/<risc>/ (test kernels keep their
DWARF types: the decoder takes struct layouts and compile-time args from them). A chip whose Tensix cores stop
answering (4 in a row) has the rest of its scan skipped. Output: OUT_DIR/chip<C>/tensix_heads.bin (the scan) and
OUT_DIR/chip<C>/tensix_<x>-<y>_l1.bin (noc0 x, y), described under index.json "workers".

Read-only by construction: the only device accesses are tt-exalens NoC reads (Coordinate.noc_read -> UMD
TTDevice.noc_read through a PCIe TLB window), with tt-exalens safe mode on (every address checked against its memory
map; eth_mac_pcs_regs is marked unsafe and never touched). No writes, no resets, no ARC messages, no RISC halt /
debug-interface access. RISC PCs are NOT read: on Blackhole that needs either the debug bus (a write to
RISCV_DEBUG_REG_DBG_BUS_CNTL_REG to select the signal) or halting the core through the debug interface.

Device open (tt-exalens 0.4.1 init_ttexalens -> UmdApi -> tt_umd.TopologyDiscovery.discover, UMD 0.9.12): opens
/dev/tenstorrent/N, maps BARs and allocates TLB windows (host-bridge address windows, KMD ioctls), asks KMD for the
busy power state (TENSTORRENT_IOCTL_SET_POWER_STATE, as every tt-metal process does on open), then only reads:
the NOC_NODE_ID hang check, ARC boot status / message-queue descriptor / telemetry, and per Ethernet core the L1 words
for training status, FW version, board/ASIC ids and the remote chip/channel, plus the soft-reset register. Blackhole
retrain_eth_cores is a no-op and no remote (Ethernet-tunnelled) devices are created on Blackhole. No reset, no firmware
load, no core init. Device-init failures are downgraded to warnings (options set to IGNORE).
"""
import argparse
import json
import os
import socket
import struct
import sys
import time
import traceback

# ---------------------------------------------------------------- what is read (Blackhole Ethernet tile)
L1_SIZE = 0x80000  # 512 KiB Ethernet L1
STREAM_BASE = 0xFFB40000  # NOC_OVERLAY_START_ADDR
STREAM_STRIDE = 0x1000  # NOC_STREAM_REG_SPACE_SIZE
NUM_ETH_STREAMS = 32  # ETH_NOC_NUM_STREAMS (the overlay of an Ethernet tile has 32 streams)
# Only registers with no read side effects, the ones the router and fabric_stream_regs.hpp read. Index 270 (the
# write-only increment port) and the auto-advancing read registers (e.g. 290) are deliberately not read.
STREAM_REGS = (4, 10, 297)  # REMOTE_SRC (scratch), REMOTE_DEST_BUF_SIZE, REMOTE_DEST_BUF_SPACE_AVAILABLE
TXQ_BASE = 0xFFB90000  # ETH_TXQ0_REGS_START
TXQ_STRIDE = 0x1000  # ETH_TXQ_REGS_SIZE
NUM_TXQ = 3  # NUM_ETH_QUEUES on Blackhole
TXQ_OFFSETS = {  # tt_eth_ss_regs.h, documented registers only
    0x00: "CTRL", 0x04: "CMD", 0x08: "STATUS", 0x0C: "MAX_PKT_SIZE_BYTES", 0x10: "BURST_LEN",
    0x14: "TRANSFER_START_ADDR", 0x18: "TRANSFER_SIZE_BYTES", 0x1C: "DEST_ADDR", 0x20: "CTRL_WORD",
    0x30: "TRANSFER_CNT", 0x34: "PKT_START_CNT", 0x3C: "PKT_END_CNT", 0x40: "WORD_CNT", 0x44: "REMOTE_REG_DATA",
    0x48: "REMOTE_SEQ_TIMEOUT", 0x4C: "LOCAL_SEQ_UPDATE_TIMEOUT", 0x50: "DEST_MAC_ADDR_HI", 0x54: "DEST_MAC_ADDR_LO",
    0x58: "SRC_MAC_ADDR_HI", 0x5C: "SRC_MAC_ADDR_LO", 0x60: "ETH_TYPE", 0x64: "MIN_PACKET_SIZE_WORDS",
    0x68: "RESEND_CNT", 0x6C: "DATA_PACKET_ACCEPT_AHEAD",
}
SOFT_RESET_REG = 0xFFB121B0  # RISCV_DEBUG_REG_SOFT_RESET_0 (ERISC0 reset bit 11, ERISC1 bit 12)
L1_WORDS = {  # L1 words sampled in both register passes (liveness)
    0x7CC70: "fabric_heartbeat",  # 0xDCBA0000 | counter, written by both router ERISCs every 64 outer iterations
    0x6FE50: "fabric_postcode",  # MEM_AERISC_FABRIC_POSTCODES_BASE
    0x100 + 1168: "go_msg0",  # mailboxes_t.go_messages[0] (signal in the top byte: 0x80 GO, 0 DONE)
    0x100 + 12: "launch_msg_rd_ptr",
}
# Tensix workers (Blackhole dev_mem_map.h / dev_msgs.h; the same mailboxes_t layout as the ERISC one above)
W_L1_SIZE = 1536 * 1024
W_MAILBOX = 0x60  # MEM_MAILBOX_BASE
# Since the merge of main 6d8a103 (2026-10-28) a Tensix core's go messages are rings, one go message per launch slot
# per sub-device (go_messages[9].slot[8]), so go_message_index sits at +1472 and every base after the mailbox moved
# 0x100 (MEM_MAILBOX_SIZE 0x3480 -> 0x3580). The ERISC mailbox keeps one go message per sub-device (unchanged).
W_HEAD_SIZE = 1488  # mailboxes_t through go_message_index (+1472), rounded to 16
W_LAUNCH_OFF, W_LAUNCH_SIZE, W_NUM_LAUNCH = 16, 144, 8
W_GO_OFF, W_NUM_GO, W_GO_RING, W_GO_IDX_OFF = 1168, 9, 8, 1472
W_STATUS_ADDR = 0x1B580 + 0x8000  # test result buffer status word: DEFAULT_UNRESERVED + LOCAL_ARGS_BUFFER_SIZE
W_FABRIC_RW = 0x9170 + 400  # tensix_fabric_connections_l1_info_t.read_write[16]: the routers' credit words
W_WINDOW = 0x40000
W_NIU_BASES = (0xFFB20000, 0xFFB30000)  # NOC0 / NOC1 register blocks; status counters at +0x200
W_NIU_STATUS = tuple(range(0x0, 0x40, 4)) + tuple(range(0xC0, 0xF8, 4))  # NIU_MST_* 0..15, NIU_SLV_* 48..61
W_STATUS_MAGIC = 0xABC00000  # TT_FABRIC_STATUS_MASK
W_OBJCOPY = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy"  # as vc1_capture_lib.sh


def now():
    return time.monotonic()


def device_holders():
    """PIDs (readable /proc entries only) holding /dev/tenstorrent open, for the record."""
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    if os.readlink(f"/proc/{pid}/fd/{fd}").startswith("/dev/tenstorrent"):
                        out.append(int(pid))
                        break
                except OSError:
                    pass
        except OSError:
            pass
    return sorted(set(out))


class Capture:
    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out_dir)
        self.t0 = now()
        self.index = {
            "tool": "hang_capture.py",
            "format": 1 if args.no_workers else 2,  # 2: has a "workers" part
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "argv": sys.argv,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "meta": dict(kv.split("=", 1) for kv in args.meta),
            "constants": {
                "l1_size": args.l1_size, "stream_base": STREAM_BASE, "stream_stride": STREAM_STRIDE,
                "num_streams": NUM_ETH_STREAMS, "stream_regs": list(STREAM_REGS), "txq_base": TXQ_BASE,
                "txq_stride": TXQ_STRIDE, "num_txq": NUM_TXQ, "soft_reset_reg": SOFT_RESET_REG,
                "l1_words": {hex(a): n for a, n in L1_WORDS.items()},
            },
            "device_holders_at_start": device_holders(),
            "chips": {},
            "eth_connections": [],
            "timings": {},
            "errors": [],
            "complete": False,
        }
        self.cores = []  # (chip_id, chan, OnChipCoordinate, core_record)
        self.n_reads = 0
        self.n_read_errors = 0
        # A read over a hung NoC path takes ~0.7 s before UMD gives up; after MAX_FAILS consecutive failures in one
        # region of one core (streams, misc, l1, txq) the rest of that region is skipped, so that one unreachable
        # core cannot use up the time budget and one unreadable register block cannot cost the L1 dump.
        self.fails = {}
        self.dead = set()

    # ------------------------------------------------------------ bookkeeping
    def err(self, where, e):
        msg = f"{where}: {type(e).__name__}: {e}"
        print("  !! " + msg, flush=True)
        self.index["errors"].append(msg)

    def save_index(self, final=False):
        self.index["timings"]["elapsed_s"] = round(now() - self.t0, 3)
        self.index["n_reads"] = self.n_reads
        self.index["n_read_errors"] = self.n_read_errors
        name = "index.json" if final else "index.partial.json"
        tmp = os.path.join(self.out, name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.index, f, indent=1, sort_keys=True)
        os.replace(tmp, os.path.join(self.out, name))
        if final and os.path.exists(os.path.join(self.out, "index.partial.json")):
            os.remove(os.path.join(self.out, "index.partial.json"))

    def over_budget(self):
        return now() - self.t0 > self.args.time_budget

    # ------------------------------------------------------------ device access (reads only)
    MAX_FAILS = 4

    def _failed(self, key):
        self.n_read_errors += 1
        self.fails[key] = self.fails.get(key, 0) + 1
        if self.fails[key] >= self.MAX_FAILS and key not in self.dead:
            self.dead.add(key)
            where = key[1] if isinstance(key[1], str) else f"eth {key[1]}"  # workers: "tensix <x>-<y>"
            msg = (f"chip {key[0]} {where} {key[2]}: {self.MAX_FAILS} consecutive read failures, "
                   f"its remaining {key[2]} reads skipped")
            print("  !! " + msg, flush=True)
            self.index["errors"].append(msg)

    def rd32(self, loc, addr, key):
        if key in self.dead:
            return None, "skipped (unreachable after repeated failures)"
        buf = bytearray(4)
        self.n_reads += 1
        try:
            loc.noc_read(addr, buf)
        except Exception as e:  # noqa: BLE001 - a failed read is recorded, the capture goes on
            self._failed(key)
            return None, f"{type(e).__name__}: {e}"
        self.fails[key] = 0
        return struct.unpack("<I", buf)[0], None

    def read_block(self, loc, addr, size, key):
        if key in self.dead:
            raise RuntimeError("skipped (unreachable after repeated failures)")
        buf = bytearray(size)
        self.n_reads += 1
        try:
            loc.noc_read(addr, buf)
        except Exception:
            self._failed(key)
            raise
        self.fails[key] = 0
        return buf

    # ------------------------------------------------------------ phases
    def open_device(self):
        os.environ.pop("TT_VISIBLE_DEVICES", None)  # open all chips, whatever the calling job's env says
        t = now()
        import ttexalens  # noqa: F401
        from ttexalens.tt_exalens_init import init_ttexalens
        from ttexalens.context import NocId

        try:
            import importlib.metadata as md
            self.index["versions"] = {"tt-exalens": md.version("tt-exalens"), "tt-umd": md.version("tt-umd")}
        except Exception:  # noqa: BLE001
            pass
        noc = NocId.NOC0 if self.args.noc == "0" else NocId.NOC1
        self.ctx = init_ttexalens(noc_id=noc, safe_mode=True)
        self.index["noc_id"] = int(self.args.noc)
        self.index["timings"]["open_s"] = round(now() - t, 3)
        print(f"device open: {self.index['timings']['open_s']} s", flush=True)

    def discover(self):
        cd = self.ctx.cluster_descriptor
        conns = {}
        try:
            conns = cd.get_ethernet_connections()
        except Exception as e:  # noqa: BLE001
            self.err("get_ethernet_connections", e)
        seen = set()
        for c, chans in sorted(conns.items()):
            for ch, (rc, rch) in sorted(chans.items()):
                key = tuple(sorted([(c, ch), (rc, rch)]))
                if key not in seen:
                    seen.add(key)
                    self.index["eth_connections"].append([c, ch, rc, rch])
        uids, bdfs = {}, {}
        try:
            uids = cd.get_chip_unique_ids()
        except Exception as e:  # noqa: BLE001
            self.err("get_chip_unique_ids", e)
        try:
            bdfs = cd.get_chip_pci_bdfs()
        except Exception:  # noqa: BLE001
            pass
        want_chips = None if not self.args.chips else {int(x) for x in self.args.chips.split(",")}
        for chip_id, dev in sorted(self.ctx.devices.items()):
            if want_chips is not None and chip_id not in want_chips:
                continue
            crec = {"unique_id": uids.get(chip_id, getattr(dev, "unique_id", None)), "pci_bdf": bdfs.get(chip_id),
                    "cores": {}}
            try:
                crec["board_type"] = str(cd.get_board_type(chip_id))
            except Exception:  # noqa: BLE001
                pass
            self.index["chips"][str(chip_id)] = crec
            active = set(conns.get(chip_id, {}).keys())
            try:
                locs = dev.get_block_locations("eth")
            except Exception as e:  # noqa: BLE001
                self.err(f"chip {chip_id} eth block locations", e)
                continue
            for i, loc in enumerate(locs):
                chan = i
                rec = {"noc0": list(loc.to("noc0"))}
                for cs in ("noc1", "translated"):
                    try:
                        rec[cs] = list(loc.to(cs))
                    except Exception:  # noqa: BLE001
                        rec[cs] = None
                try:
                    lg = loc.to("logical")  # ((0, channel), "eth"): UMD's logical Ethernet coordinate is (0, channel)
                    rec["logical"] = list(lg[0])
                    chan = int(lg[0][1])
                except Exception:  # noqa: BLE001
                    rec["logical"] = None
                    if rec.get("translated") and rec["translated"][1] == 25:  # Blackhole: translated (20 + ch, 25)
                        chan = rec["translated"][0] - 20
                rec["eth_channel"] = chan
                rec["connected_to"] = list(conns[chip_id][chan]) if chan in active else None
                if chan in active or self.args.all_eth:
                    crec["cores"][str(chan)] = rec
                    self.cores.append((chip_id, chan, loc, rec))
            print(f"chip {chip_id}: uid {crec['unique_id']}, eth cores {sorted(int(c) for c in crec['cores'])}",
                  flush=True)
        self.save_index()

    def reg_pass(self, name, txq_only=False):
        t = now()
        for chip_id, chan, loc, rec in self.cores:
            if self.over_budget():
                self.index["errors"].append(f"{name}: time budget reached before chip {chip_id} eth {chan}")
                break
            p = {"t": round(now() - self.t0, 4), "errors": []}
            if txq_only:
                p["txq"] = {}
                for q in range(NUM_TXQ):
                    qd = {}
                    for off in TXQ_OFFSETS:
                        v, e = self.rd32(loc, TXQ_BASE + q * TXQ_STRIDE + off, (chip_id, chan, "txq"))
                        qd[hex(off)] = v
                        if e:
                            p["errors"].append(f"txq {q} {hex(off)}: {e}")
                    p["txq"][str(q)] = qd
                p["t_end"] = round(now() - self.t0, 4)
                rec.setdefault("regs", {})[name] = p
                if p["errors"]:
                    print(f"  chip {chip_id} eth {chan} {name}: {len(p['errors'])} read errors (first: {p['errors'][0]})",
                          flush=True)
                continue
            p["streams"], p["l1_words"] = {}, {}
            for sid in range(NUM_ETH_STREAMS):
                vals = []
                for reg in STREAM_REGS:
                    v, e = self.rd32(loc, STREAM_BASE + sid * STREAM_STRIDE + reg * 4, (chip_id, chan, "streams"))
                    vals.append(v)
                    if e:
                        p["errors"].append(f"stream {sid} reg {reg}: {e}")
                p["streams"][str(sid)] = vals
            v, e = self.rd32(loc, SOFT_RESET_REG, (chip_id, chan, "misc"))
            p["soft_reset"] = v
            if e:
                p["errors"].append(f"soft_reset: {e}")
            for a in L1_WORDS:
                v, e = self.rd32(loc, a, (chip_id, chan, "misc"))
                p["l1_words"][hex(a)] = v
                if e:
                    p["errors"].append(f"l1 {hex(a)}: {e}")
            p["t_end"] = round(now() - self.t0, 4)
            rec.setdefault("regs", {})[name] = p
            if p["errors"]:
                print(f"  chip {chip_id} eth {chan} {name}: {len(p['errors'])} read errors (first: {p['errors'][0]})",
                      flush=True)
        self.index["timings"][name + "_s"] = round(now() - t, 3)
        print(f"{name}: {len(self.cores)} cores, {self.index['timings'][name + '_s']} s", flush=True)
        self.save_index()

    def l1_pass(self):
        t = now()
        total = 0
        chunk = self.args.chunk
        for chip_id, chan, loc, rec in self.cores:
            if self.over_budget():
                self.index["errors"].append(f"l1: time budget reached before chip {chip_id} eth {chan}")
                print(f"  time budget reached, L1 of chip {chip_id} eth {chan} and later cores skipped", flush=True)
                break
            d = os.path.join(self.out, f"chip{chip_id}")
            os.makedirs(d, exist_ok=True)
            fn = f"chip{chip_id}/eth{chan:02d}_l1.bin"
            img = bytearray(self.args.l1_size)
            missing = []
            tc = now()
            for off in range(0, self.args.l1_size, chunk):
                n = min(chunk, self.args.l1_size - off)
                for attempt in (1, 2):
                    try:
                        img[off:off + n] = self.read_block(loc, off, n, (chip_id, chan, "l1"))
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 2 or (chip_id, chan, "l1") in self.dead:
                            missing.append([off, n, f"{type(e).__name__}: {e}"])
                            self.err(f"chip {chip_id} eth {chan} L1 [{off:#x}, {off + n:#x})", e)
            with open(os.path.join(self.out, fn), "wb") as f:
                f.write(img)
            dt = now() - tc
            rec["l1"] = {"file": fn, "addr": 0, "size": self.args.l1_size, "missing": missing,
                         "t": round(tc - self.t0, 4), "read_s": round(dt, 4)}
            total += self.args.l1_size - sum(m[1] for m in missing)
            if self.args.verbose:
                print(f"  chip {chip_id} eth {chan}: {self.args.l1_size} B in {dt:.3f} s", flush=True)
            self.save_index()
        dt = now() - t
        self.index["timings"]["l1_s"] = round(dt, 3)
        self.index["timings"]["l1_bytes"] = total
        self.index["timings"]["l1_MBps"] = round(total / dt / 1e6, 3) if dt > 0 else None
        print(f"l1: {total} B in {dt:.2f} s ({self.index['timings']['l1_MBps']} MB/s)", flush=True)

    # ------------------------------------------------------------ workers (Tensix cores), reads only
    @staticmethod
    def head_info(head):
        """From a mailbox head: the go signal, the launch slot last executed (the current one if GO), that slot's
        kernel id (highest watcher_kernel_id of a RISC with a kernel; -1 if none), kernel config base, enables and
        dispatch mode, and whether it is the core's most recent slot (highest kernel id of the 8). Kernel ids come from
        a process-wide counter, so they order programs; the host-assigned program id is 0 in these runs."""
        rd = struct.unpack_from("<I", head, 12)[0] & (W_NUM_LAUNCH - 1)
        gi = struct.unpack_from("<I", head, W_GO_IDX_OFF)[0]
        ring = W_GO_OFF + 4 * W_GO_RING * (gi if gi < W_NUM_GO else 0)  # this sub-device's go ring
        sig = struct.unpack_from("<I", head, ring + 4 * rd)[0] >> 24  # the go message of the current launch slot
        prev = (rd - 1) & (W_NUM_LAUNCH - 1)
        slot = rd if sig == 0x80 else (prev if head[W_LAUNCH_OFF + W_LAUNCH_SIZE * prev + 48] == 0 else rd)

        def kernel_id(i):
            lb = W_LAUNCH_OFF + W_LAUNCH_SIZE * i
            sizes = struct.unpack_from("<5I", head, lb + 72)
            ids = struct.unpack_from("<5H", head, lb + 120)
            return max((k for k, n in zip(ids, sizes) if n), default=-1)

        ids = [kernel_id(i) for i in range(W_NUM_LAUNCH)]
        lb = W_LAUNCH_OFF + W_LAUNCH_SIZE * slot
        return {"sig": sig, "slot": slot, "kid": ids[slot], "kcb": struct.unpack_from("<I", head, lb)[0],
                "enables": struct.unpack_from("<I", head, lb + 112)[0], "mode": head[lb + 48],
                "latest": ids[slot] >= 0 and ids[slot] == max(ids)}

    def worker_scan(self):
        t = now()
        a = self.args
        wi = self.index.setdefault("workers", {})
        wi["constants"] = {"mailbox": W_MAILBOX, "head_size": W_HEAD_SIZE, "status_addr": a.worker_status_addr,
                           "window": a.worker_window, "l1_size": W_L1_SIZE, "fabric_rw": W_FABRIC_RW,
                           "niu_bases": list(W_NIU_BASES), "niu_status": list(W_NIU_STATUS),
                           "soft_reset_reg": SOFT_RESET_REG, "full_max": a.worker_full_max}
        wi["chips"] = {}
        self.wcores = []  # (chip_id, key, loc, rec, full)
        want_chips = None if not a.chips else {int(x) for x in a.chips.split(",")}
        nscan = 0
        for chip_id, dev in sorted(self.ctx.devices.items()):
            if want_chips is not None and chip_id not in want_chips:
                continue
            if self.over_budget():
                self.index["errors"].append(f"wscan: time budget reached before chip {chip_id}")
                break
            try:
                locs = dev.get_block_locations("functional_workers")
            except Exception as e:  # noqa: BLE001
                self.err(f"chip {chip_id} functional_workers block locations", e)
                continue
            fn = f"chip{chip_id}/tensix_heads.bin"
            os.makedirs(os.path.join(self.out, f"chip{chip_id}"), exist_ok=True)
            heads = bytearray()
            ents = []
            scanned = []
            bad_run = 0  # consecutive cores whose mailbox read failed: a chip that stopped answering is cut short
            for loc in locs:
                if bad_run >= self.MAX_FAILS or self.over_budget():
                    msg = (f"wscan: chip {chip_id}: " + (f"{bad_run} Tensix cores in a row unreadable"
                                                         if bad_run >= self.MAX_FAILS else "time budget reached")
                           + f", {len(locs) - len(scanned)} cores not scanned")
                    print("  !! " + msg, flush=True)
                    self.index["errors"].append(msg)
                    break
                rec = {"noc0": list(loc.to("noc0"))}
                try:
                    rec["translated"] = list(loc.to("translated"))
                except Exception:  # noqa: BLE001
                    rec["translated"] = None
                try:
                    rec["logical"] = list(loc.to("logical")[0])
                except Exception:  # noqa: BLE001
                    rec["logical"] = None
                key = (chip_id, f"tensix {rec['noc0'][0]}-{rec['noc0'][1]}", "wscan")
                head = None
                try:
                    head = self.read_block(loc, W_MAILBOX, W_HEAD_SIZE, key)
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"
                    self.err(f"chip {chip_id} tensix {rec['noc0'][0]}-{rec['noc0'][1]} mailbox", e)
                bad_run = 0 if head is not None else bad_run + 1
                status, _ = self.rd32(loc, a.worker_status_addr, key)
                sr, _ = self.rd32(loc, SOFT_RESET_REG, key)
                nscan += 1
                heads += head if head is not None else bytes(W_HEAD_SIZE)
                ent = {"noc0": rec["noc0"], "translated": rec["translated"], "status": status, "soft_reset": sr,
                       "head_ok": head is not None}
                ents.append(ent)
                scanned.append((loc, rec, head, status, sr, ent))
            with open(os.path.join(self.out, fn), "wb") as f:
                f.write(heads)
            infos = []
            for loc, rec, head, status, sr, ent in scanned:
                info = None
                if head is not None:
                    try:
                        info = self.head_info(head)
                    except Exception:  # noqa: BLE001
                        info = None
                infos.append(info)
            # The chip's newest program: the highest kernel id among device-dispatched slots that their cores last
            # executed and that are those cores' most recent ones; its cores share that slot's kernel config base. A
            # core left out of later programs keeps executing stale slots with nothing enabled, so its last-executed
            # slot is not its most recent one.
            cands = [i for i in infos if i and i["kid"] >= 0 and i["mode"] == 0 and (i["latest"] or i["sig"] == 0x80)]
            top = max(cands, key=lambda i: i["kid"]) if cands else None
            newest = {"kernel_id": top["kid"], "kernel_config": top["kcb"]} if top else None
            crec = {"scan": {"heads_file": fn, "cores": ents, "newest_program": newest}, "cores": {}}
            for (loc, rec, head, status, sr, ent), info in zip(scanned, infos):
                why = []
                if info is not None:
                    if info["sig"] == 0x80 and info["enables"]:
                        why.append("go")
                    if (top and info["kid"] >= 0 and info["latest"] and info["mode"] == 0
                            and info["kcb"] == top["kcb"]):
                        why.append("newest-program")
                if status is not None and (status & 0xFFF00000) == W_STATUS_MAGIC:
                    why.append("status")
                if a.workers_all:
                    why.append("all")
                if not why:
                    continue
                running = ("go" in why and "newest-program" in why) or status == W_STATUS_MAGIC
                key = f"{rec['noc0'][0]}-{rec['noc0'][1]}"
                rec["selected_because"] = why
                rec["regs"] = {"pass1": {"status": status, "soft_reset": sr,
                                         "go": None if info is None else info["sig"], "t": round(now() - self.t0, 4)}}
                ent["selected"] = True
                crec["cores"][key] = rec
                self.wcores.append((chip_id, key, loc, rec, running))
            wi["chips"][str(chip_id)] = crec
        self.index["timings"]["wscan_s"] = round(now() - t, 3)
        print(f"wscan: {nscan} Tensix cores, {len(self.wcores)} selected "
              f"({sum(1 for c in self.wcores if c[4])} still running a test kernel), "
              f"{self.index['timings']['wscan_s']} s", flush=True)
        self.save_index()

    def worker_l1(self):
        t = now()
        a = self.args
        total, nfull = 0, 0
        for chip_id, key, loc, rec, running in self.wcores:
            if self.over_budget():
                self.index["errors"].append(f"wl1: time budget reached before chip {chip_id} tensix {key}")
                print(f"  time budget reached, L1 of chip {chip_id} tensix {key} and later cores skipped", flush=True)
                break
            full = running and nfull < a.worker_full_max
            if running and not full:
                rec.setdefault("notes", []).append(f"whole-L1 cap ({a.worker_full_max}) reached: window only")
            nfull += full
            size = W_L1_SIZE if full else a.worker_window
            fkey = (chip_id, f"tensix {key}", "l1")
            img = bytearray(size)
            missing = []
            tc = now()
            for off in range(0, size, a.chunk):
                n = min(a.chunk, size - off)
                for attempt in (1, 2):
                    try:
                        img[off:off + n] = self.read_block(loc, off, n, fkey)
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 2 or fkey in self.dead:
                            missing.append([off, n, f"{type(e).__name__}: {e}"])
                            self.err(f"chip {chip_id} tensix {key} L1 [{off:#x}, {off + n:#x})", e)
            fn = f"chip{chip_id}/tensix_{key}_l1.bin"
            with open(os.path.join(self.out, fn), "wb") as f:
                f.write(img)
            rec["l1"] = {"file": fn, "addr": 0, "size": size, "missing": missing, "t": round(tc - self.t0, 4),
                         "read_s": round(now() - tc, 4)}
            rec["full"] = full
            if W_FABRIC_RW + 256 <= size and not any(m[0] <= W_FABRIC_RW < m[0] + m[1] for m in missing):
                rec["regs"]["pass1"]["fabric_rw"] = bytes(img[W_FABRIC_RW:W_FABRIC_RW + 256]).hex()
            total += size - sum(m[1] for m in missing)
        dt = now() - t
        self.index["timings"]["wl1_s"] = round(dt, 3)
        self.index["timings"]["wl1_bytes"] = total
        print(f"wl1: {len(self.wcores)} cores, {total} B in {dt:.2f} s", flush=True)
        self.save_index()

    def worker_pass2(self):
        t = now()
        a = self.args
        for chip_id, key, loc, rec, running in self.wcores:
            if self.over_budget():
                self.index["errors"].append(f"wpass2: time budget reached before chip {chip_id} tensix {key}")
                break
            fkey = (chip_id, f"tensix {key}", "wpass2")
            p = {"t": round(now() - self.t0, 4), "errors": []}
            try:
                head = self.read_block(loc, W_MAILBOX, W_HEAD_SIZE, fkey)
                p["go"] = self.head_info(head)["sig"]
                p["head_hex"] = bytes(head).hex()
            except Exception as e:  # noqa: BLE001
                p["errors"].append(f"head: {type(e).__name__}: {e}")
            p["status"], e = self.rd32(loc, a.worker_status_addr, fkey)
            if e:
                p["errors"].append(f"status: {e}")
            p["soft_reset"], e = self.rd32(loc, SOFT_RESET_REG, fkey)
            if e:
                p["errors"].append(f"soft_reset: {e}")
            try:
                p["fabric_rw"] = bytes(self.read_block(loc, W_FABRIC_RW, 256, fkey)).hex()
            except Exception as e:  # noqa: BLE001
                p["errors"].append(f"fabric_rw: {type(e).__name__}: {e}")
            rec["regs"]["pass2"] = p
        self.index["timings"]["wpass2_s"] = round(now() - t, 3)
        print(f"wpass2: {len(self.wcores)} cores, {self.index['timings']['wpass2_s']} s", flush=True)
        self.save_index()

    def worker_niu(self):
        t = now()
        for chip_id, key, loc, rec, running in self.wcores:
            if self.over_budget():
                self.index["errors"].append(f"wniu: time budget reached before chip {chip_id} tensix {key}")
                break
            fkey = (chip_id, f"tensix {key}", "niu")
            niu, errs = {}, []
            for noc, base in enumerate(W_NIU_BASES):
                regs = {}
                for off in W_NIU_STATUS:
                    v, e = self.rd32(loc, base + 0x200 + off, fkey)
                    regs[hex(off)] = v
                    if e:
                        errs.append(f"noc{noc} {hex(off)}: {e}")
                niu[str(noc)] = regs
            rec["niu"] = niu
            if errs:
                rec["niu_errors"] = errs[:8]
        self.index["timings"]["wniu_s"] = round(now() - t, 3)
        print(f"wniu: {len(self.wcores)} cores, {self.index['timings']['wniu_s']} s", flush=True)
        self.save_index()

    def copy_kernel_elfs(self):
        """No device access: the run's non-router kernel ELFs into OUT/cache. Test kernels (tt_fabric_*) keep the
        DWARF type information the decoder reads with gdb (.debug_info/.debug_abbrev/.debug_str; line, location and
        frame tables, relocations and symbols dropped: 0.5 MB instead of 1.5 MB, same layouts); the others (dispatch)
        are stripped to their load image, which is all the decoder needs to name them."""
        t = now()
        src = self.args.kernel_cache or self.index["meta"].get("cache")
        out = []
        if not src or not os.path.isdir(src):
            self.index.setdefault("workers", {})["kernel_elfs"] = {"source": src, "files": [], "note": "no cache dir"}
            return
        import glob
        import shutil
        import subprocess
        objcopy = W_OBJCOPY if os.path.exists(W_OBJCOPY) else None

        def put(p, dst, name):
            if objcopy:
                opts = (["--remove-section=.debug_line", "--remove-section=.debug_loclists",
                         "--remove-section=.debug_rnglists", "--remove-section=.debug_frame",
                         "--remove-section=.debug_aranges", "--remove-section=.rela.*", "--remove-section=.symtab",
                         "--remove-section=.strtab"] if name.startswith("tt_fabric_") else ["--strip-all"])
                try:
                    r = subprocess.run([objcopy] + opts + [p, dst], capture_output=True, text=True, timeout=30)
                    if r.returncode == 0:
                        return
                except (OSError, subprocess.SubprocessError):
                    pass
            shutil.copy2(p, dst)

        seen = set()
        for d in sorted(glob.glob(f"{src}/*/kernels/*/*/") + glob.glob(f"{src}/kernels/*/*/")):
            parts = d.rstrip("/").split("/")
            name, hsh = parts[-2], parts[-1]
            if name == "fabric_erisc_router_v2" or (name, hsh) in seen:
                continue
            seen.add((name, hsh))
            for risc in ("brisc", "ncrisc", "trisc0", "trisc1", "trisc2"):
                for fname in (f"{risc}.elf.xip.elf", f"{risc}.elf"):
                    p = os.path.join(d, risc, fname)
                    if os.path.isfile(p):
                        dst = os.path.join(self.out, "cache", "kernels", name, hsh, risc)
                        os.makedirs(dst, exist_ok=True)
                        try:
                            put(p, os.path.join(dst, fname), name)
                            out.append(f"cache/kernels/{name}/{hsh}/{risc}/{fname}")
                        except OSError as e:
                            self.err(f"copy {p}", e)
                        break
        self.index.setdefault("workers", {})["kernel_elfs"] = {"source": src, "files": out}
        self.index["timings"]["kernel_elfs_s"] = round(now() - t, 3)
        print(f"kernel ELFs: {len(out)} copied from {src} in {self.index['timings']['kernel_elfs_s']} s", flush=True)

    def summary(self):
        print("== capture summary ==")
        for chip_id, crec in sorted(self.index["chips"].items(), key=lambda kv: int(kv[0])):
            parts = []
            nbytes = 0
            for chan, rec in sorted(crec["cores"].items(), key=lambda kv: int(kv[0])):
                l1 = rec.get("l1")
                if l1:
                    got = l1["size"] - sum(m[1] for m in l1["missing"])
                    nbytes += got
                    parts.append(f"{chan}" + ("" if got == l1["size"] else f"(partial {got})"))
                else:
                    parts.append(f"{chan}(no L1)")
                if "pass2" not in rec.get("regs", {}):
                    parts[-1] += "(no pass2)"
            print(f"chip {chip_id}: eth {' '.join(parts) or '-'}; {nbytes} B L1")
        for chip_id, crec in sorted(self.index.get("workers", {}).get("chips", {}).items(), key=lambda kv: int(kv[0])):
            parts = []
            for key, rec in sorted(crec["cores"].items()):
                l1 = rec.get("l1")
                tag = "full" if rec.get("full") else ("win" if l1 else "no L1")
                if l1 and l1["missing"]:
                    tag += f" partial {l1['size'] - sum(m[1] for m in l1['missing'])}"
                parts.append(f"{key}({tag})")
            print(f"chip {chip_id} tensix: {len(crec['scan']['cores'])} scanned, {len(parts)} captured: "
                  f"{' '.join(parts) or '-'}")
        t = self.index["timings"]
        print(f"reads {self.n_reads}, read errors {self.n_read_errors}, total {t.get('elapsed_s')} s "
              f"(open {t.get('open_s')}, pass1 {t.get('pass1_s')}, l1 {t.get('l1_s')}, pass2 {t.get('pass2_s')}, "
              f"txq {t.get('txq1_s')}+{t.get('txq2_s')}"
              + (f", workers scan {t.get('wscan_s')} l1 {t.get('wl1_s')} pass2 {t.get('wpass2_s')} "
                 f"niu {t.get('wniu_s')}" if "wscan_s" in t else "") + ")")
        print(f"output: {self.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--all-eth", action="store_true", help="every non-harvested Ethernet core, not only connected ones")
    ap.add_argument("--chips", default="", help="comma-separated chip ids (default: all)")
    ap.add_argument("--noc", choices=["0", "1"], default="1", help="NoC for the reads (tt-exalens fails over)")
    ap.add_argument("--l1-size", type=lambda x: int(x, 0), default=L1_SIZE)
    ap.add_argument("--chunk", type=lambda x: int(x, 0), default=0x10000)
    ap.add_argument("--time-budget", type=float, default=210.0, help="seconds; later phases are skipped past it")
    ap.add_argument("--meta", action="append", default=[], help="KEY=VALUE recorded in index.json")
    ap.add_argument("--no-txq", action="store_true", help="skip the Ethernet TX queue registers")
    ap.add_argument("--no-workers", action="store_true", help="skip the Tensix (test worker) capture")
    ap.add_argument("--worker-window", type=lambda x: int(x, 0), default=W_WINDOW,
                    help="L1 bytes from 0 read per captured worker core (default 0x40000)")
    ap.add_argument("--worker-full-max", type=int, default=32,
                    help="at most this many still-running worker cores get their whole L1 read")
    ap.add_argument("--worker-status-addr", type=lambda x: int(x, 0), default=W_STATUS_ADDR,
                    help="the test result buffer's status word, read on every Tensix core (default 0x23480)")
    ap.add_argument("--workers-all", action="store_true", help="capture every Tensix core's window (debug)")
    ap.add_argument("--no-worker-niu", action="store_true", help="skip the workers' NIU transaction counters")
    ap.add_argument("--kernel-cache", default="",
                    help="run cache to copy the kernel ELFs from (default: the --meta cache=DIR value)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cap = Capture(args)
    rc = 0
    try:
        if cap.index["device_holders_at_start"]:
            print(f"note: processes holding /dev/tenstorrent: {cap.index['device_holders_at_start']}", flush=True)
        cap.open_device()
        cap.discover()
        cap.reg_pass("pass1")
        cap.l1_pass()
        cap.reg_pass("pass2")
        if not args.no_txq:  # last: the TX queue register block is the one this tool reads that the router owns
            cap.reg_pass("txq1", txq_only=True)
            cap.reg_pass("txq2", txq_only=True)
        if not args.no_workers:  # after the routers, so their capture is as before whatever happens here
            try:
                cap.worker_scan()
                cap.worker_l1()
                cap.worker_pass2()
                if not args.no_worker_niu:
                    cap.worker_niu()
            except Exception as e:  # noqa: BLE001 - the router capture above is kept
                cap.err("workers", e)
                cap.index["workers_traceback"] = traceback.format_exc()
                traceback.print_exc()
        cap.index["complete"] = not cap.over_budget()
    except Exception as e:  # noqa: BLE001
        cap.err("capture", e)
        cap.index["traceback"] = traceback.format_exc()
        traceback.print_exc()
        rc = 2
    cap.index["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if not args.no_workers:  # files only, no device access
        try:
            cap.copy_kernel_elfs()
        except Exception as e:  # noqa: BLE001
            cap.err("kernel ELF copy", e)
    cap.save_index(final=True)
    cap.summary()
    return rc


if __name__ == "__main__":
    sys.exit(main())
