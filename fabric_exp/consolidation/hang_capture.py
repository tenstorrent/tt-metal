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
            "format": 1,
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
            msg = (f"chip {key[0]} eth {key[1]} {key[2]}: {self.MAX_FAILS} consecutive read failures, "
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
        t = self.index["timings"]
        print(f"reads {self.n_reads}, read errors {self.n_read_errors}, total {t.get('elapsed_s')} s "
              f"(open {t.get('open_s')}, pass1 {t.get('pass1_s')}, l1 {t.get('l1_s')}, pass2 {t.get('pass2_s')}, "
              f"txq {t.get('txq1_s')}+{t.get('txq2_s')})")
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
        cap.index["complete"] = not cap.over_budget()
    except Exception as e:  # noqa: BLE001
        cap.err("capture", e)
        cap.index["traceback"] = traceback.format_exc()
        traceback.print_exc()
        rc = 2
    cap.index["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    cap.save_index(final=True)
    cap.summary()
    return rc


if __name__ == "__main__":
    sys.exit(main())
