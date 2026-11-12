#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# N3 code readback: read every RISC's instruction memory back through
# exalens and judge it without a reference copy. Tensix: the BRISC/NCRISC/TRISC0-2 firmware regions in L1
# and the kernel window at MEM_MAP_END; ETH: the base-FW text only (never its data, stack or mailboxes). All chips of one box run the same firmware
# and, after the heavy stage, the same kernels on the same logical cores. The firmware regions vote by NOC0 x,y. The kernel
# window votes by logical core: each chip's own Tensix grid after harvesting, columns and rows ranked
# left to right and top to bottom as tt-metal places programs. Chips harvest different columns, so one NOC0 x,y is
# logical core 0,0 on one chip and 1,0 on another, with another kernel and its own runtime args (seen on one box:
# 120 false CODE FAIL lines at 0xa180 and 0xb9f4 on the 7 chips with column x=1 or x=2 harvested). A kernel-window
# word that two or more chips read with the same other value is another program's word or config (INFO 'kernel
# variant'), never a FAIL: corruption is one chip's own. The cores at one key vote word by word. A strong majority (>= max(3, 75 %) of the chips read) sets the expected word; a chip whose
# word differs is re-read, and only a word that reads back the same wrong value again is a CODE FAIL.
# A word with no strong majority is INFO; a word that changes on re-read is volatile (INFO); a read that
# times out skips that chip (WARN), never a FAIL. With --elf-dir, executable PT_LOAD segments of
# <risc>.elf are compared too. Reads only: nothing is written to any chip.
#
# Core-major and budgeted: one core (its regions in turn, all chips per region) at a time; the budget is checked
# before every region, so one slow core overruns it by one region at most. The cursor file carries the next core
# and region into the next run, so plain runs rotate over the whole box.
# Every chip line names the BDF and tray 1-4 (goals 5 and 47); a chip with no known BDF reads 'chip<id> tray ?'.
# Output lines (fleet.sh greps them):
#   CODE FAIL <bdf> tray <n> core x,y <RISC> <region> addr 0x.. words k (got 0x.., majority 0x..|elf 0x..)
#   CODE WARN <bdf> tray <n> ...      CODE INFO ...      CODE SUMMARY ...

import glob
import json
import os
import re
import signal
import struct
import sys
import time
from collections import Counter

# Blackhole Tensix L1 (tt-metal a869f55 dev_mem_map.h): firmware base and size per RISC, then the kernel
# window at MEM_MAP_END (41344 with MaxDM=2; 46528 with MaxDM=5: pass --region kernel:kernel:46528:8192).
TENSIX_REGIONS = [
    ("BRISC", "fw", 15328, 8704),
    ("NCRISC", "fw", 24032, 2560),
    ("TRISC0", "fw", 26592, 2560),
    ("TRISC1", "fw", 29152, 2560),
    ("TRISC2", "fw", 31712, 2560),
    ("kernel", "kernel", 41344, 8192),
]
# Blackhole ERISC base FW: tzp bh_arc/eth.c loads the SPI boot-fs 'ethfw' image at ETH_FW_BASE_ADDR 0x70000, params
# at ETH_PARAM_ADDR 0x7c000. In that image the code comes first, then a zero pad up to a 4 KB boundary, then read-only
# tables (jump tables), initialized data and bss (live counters at 0x7b480 on 19.16); 0x7c000 up holds params, debug
# buffer, heartbeat, link counters and mailboxes (dev_mem_map.h). Only the code is voted: the other words differ between
# chips by design (0x7b480, 0x7c088 and 0x7c180 read as CODE FAIL on a healthy box).
# Where the code ends depends on the firmware (read from the boot-fs 'ethfw' entry of the public
# tt-zephyr-platforms galaxy.fwbundle of every release 18.6.0-19.17.1, GALAXY-1 and every other galaxy board alike):
#   18.6.0-18.9.1  read-only data from 0x78000 (code to 0x75cf0-0x761c8)
#   18.10.0-19.6.0 read-only data from 0x7a000 (18.12.2: code to 0x7996c, image 0xa6ec)
#   19.7.0-19.17.1 read-only data from 0x7b000 (19.7.0 code to 0x7a0a8, 19.16.0 0x7a37c image 0xb554, 19.17.1 0x7adf0 0xb430)
# Each chip's tt_fw_bundle_ver (sysfs, no device open) picks its end; the box votes up to the lowest end of its chips.
# A chip whose version is unread or older than 18.6 counts as 0x78000 (the lowest end known): coverage is lost, never a
# false CODE FAIL on data words.
ETH_TEXT_END = [((18, 6), None), ((18, 10), 0x78000), ((19, 7), 0x7A000), (None, 0x7B000)]
ETH_TEXT_END_LOW = 0x78000
ETH_REGIONS = [("ERISC", "basefw", 0x70000, 0xB000)]   # run() without fw_of: 19.7 and later
ETH_SKIP = {0x7CC70, 0x7CC74}    # base-FW and metal heartbeats count by design (only an explicit --region reaches them)
SMOKE_CORES = {"tensix": 2, "eth": 1}


class ReadTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise ReadTimeout("read timed out")


def timed(sec, fn):
    """fn() under a SIGALRM timer; ReadTimeout when it takes longer than sec."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def parse_region(s):
    """'RISC:name:start:size' (start and size decimal or 0x..) -> tuple; kind tensix unless RISC is ERISC*."""
    p = s.split(":")
    if len(p) != 4:
        raise ValueError(f"--region wants RISC:name:start:size, got {s!r}")
    return (p[0], p[1], int(p[2], 0), int(p[3], 0))


def core_xy(loc):
    for f in (lambda: loc.to("noc0"), lambda: loc._noc0_coord):
        try:
            v = f()
            if isinstance(v, tuple) and len(v) == 2:
                return f"{int(v[0])},{int(v[1])}"
        except Exception:
            pass
    if isinstance(loc, tuple) and len(loc) == 2:
        return f"{loc[0]},{loc[1]}"
    n = re.findall(r"\d+", str(loc))
    return f"{n[0]},{n[1]}" if len(n) >= 2 else str(loc)


def to_words(data):
    if isinstance(data, (bytes, bytearray, memoryview)):
        b = bytes(data)
    else:
        b = bytes(bytearray(int(x) & 0xFF for x in data))
    n = len(b) // 4
    return list(struct.unpack(f"<{n}I", b[:4 * n]))


def elf_words(path):
    """{address: word} for every executable PT_LOAD segment (p_filesz bytes at p_vaddr) of a 32-bit LE ELF."""
    b = open(path, "rb").read()
    if b[:4] != b"\x7fELF" or b[4] != 1 or b[5] != 1:
        raise ValueError(f"{path}: not a 32-bit little-endian ELF")
    phoff, = struct.unpack_from("<I", b, 28)
    phentsize, phnum = struct.unpack_from("<HH", b, 42)
    out = {}
    for i in range(phnum):
        p_type, p_off, p_vaddr, _pa, p_filesz, _ms, p_flags, _al = struct.unpack_from("<8I", b, phoff + i * phentsize)
        if p_type != 1 or not (p_flags & 1):
            continue
        seg = b[p_off:p_off + p_filesz]
        for j in range(0, len(seg) - 3, 4):
            out[p_vaddr + j], = struct.unpack_from("<I", seg, j)
    return out


def load_elfs(elf_dir, riscs):
    got = {}
    if not elf_dir:
        return got
    for r in riscs:
        hits = sorted(glob.glob(os.path.join(elf_dir, r.lower() + "*.elf")) +
                      glob.glob(os.path.join(elf_dir, r.lower(), "*.elf")))
        if hits:
            got[r] = elf_words(hits[0])
    return got


BUS_TO_TRAY = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}    # as run.sh rk_tray and fleet/chunk_sched.py


def tray_of(bdf):
    """'0000:41:00.0' or '41:00.0' -> '2'; '?' when the BDF is unknown or not on a galaxy tray bus."""
    try:
        return str(BUS_TO_TRAY.get(int(str(bdf).replace("0000:", "", 1).split(":")[0], 16) & 0xF0, "?"))
    except (ValueError, IndexError):
        return "?"


def _umd():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import numeric_check
    return numeric_check


def bdf_from_umd(text, sys_root="/sys"):
    """{logical chip id: bdf} from UMD's open line (numeric_check.umd_map_from_text) and sysfs."""
    try:
        m = _umd().umd_map_from_text(text)
    except Exception:
        return {}
    return bdf_from_map(m, sys_root)


def bdf_from_stdout(sys_root="/sys"):
    """The same from this process's own stdout file (run.sh: > run.log), where exalens's UMD prints its open line."""
    try:
        m = _umd().umd_map_from_stdout()
    except Exception:
        return {}
    return bdf_from_map(m, sys_root)


def bdf_from_map(m, sys_root="/sys"):
    """{logical id: PCIe id} -> {logical id: bdf} through sysfs /class/tenstorrent/tenstorrent!<pcie id>/device."""
    out = {}
    for lg, pc in (m or {}).items():
        try:
            out[lg] = os.path.basename(os.readlink(f"{sys_root}/class/tenstorrent/tenstorrent!{pc}/device"))
        except OSError:
            pass
    return out


def fw_tuple(ver):
    """'19.17.1.0' -> (19, 17, 1); None when unread, unparsable or 0.0.0 (firmware not ready)."""
    m = re.match(r"\s*(\d+)\.(\d+)\.(\d+)", str(ver or ""))
    t = tuple(int(x) for x in m.groups()) if m else None
    return t if t and any(t) else None


def eth_text_end(ver):
    """The ERISC base-FW code end for one firmware bundle version (ETH_TEXT_END); None when the layout is not known."""
    t = fw_tuple(ver)
    if t is None:
        return None
    for below, end in ETH_TEXT_END:
        if below is None or t < below:
            return end
    return None


def eth_regions(fw_of):
    """{chip: tt_fw_bundle_ver} -> (ERISC regions, CODE INFO line): the lowest code end of the box's chips."""
    ends = {c: eth_text_end(v) for c, v in fw_of.items()}
    end = min([e or ETH_TEXT_END_LOW for e in ends.values()] or [ETH_TEXT_END_LOW])
    vers = Counter(str(v) if eth_text_end(v) else f"{v or 'unread'} (layout unknown)" for v in fw_of.values())
    why = ", ".join(f"{v} on {n} chip{'s' if n > 1 else ''}" for v, n in sorted(vers.items())) or "no chip"
    return [("ERISC", "basefw", 0x70000, end - 0x70000)], f"CODE INFO ERISC basefw voted 0x70000-{end:#x}: tt_fw_bundle_ver {why}"


def fw_from_sysfs(bdf_of, sys_root="/sys", timeout_s=2.0):
    """{chip: tt_fw_bundle_ver or None} from sysfs (tt-kmd's attribute, no device open), by each chip's BDF."""
    out = {}
    for c, b in bdf_of.items():
        v = None
        if b:
            b = b if b.count(":") == 2 else "0000:" + b
            for f in sorted(glob.glob(f"{sys_root}/bus/pci/devices/{b}/tenstorrent/tenstorrent!*/tt_fw_bundle_ver")):
                try:
                    v = timed(timeout_s, lambda f=f: open(f).read().strip()) or None
                except (OSError, ReadTimeout):
                    v = None
                break
        out[c] = v
    return out


class Reader:
    def __init__(self, ctx, read_word, read_block, timeout_s):
        self.ctx, self.rw, self.rb, self.t = ctx, read_word, read_block, timeout_s

    def block(self, dev, loc, addr, size):
        """size bytes as words: one block read, word reads when the block read fails (not on a timeout)."""
        if self.rb is not None:
            try:
                return to_words(timed(self.t, lambda: self.rb(loc, addr, device_id=dev.id, num_bytes=size, context=self.ctx)))
            except ReadTimeout:
                raise
            except Exception:
                pass
        return [timed(self.t, lambda a=a: int(self.rw(loc, a, device_id=dev.id, context=self.ctx)))
                for a in range(addr, addr + size, 4)]


def vote(vals):
    """vals {chip: word} -> (majority word or None, threshold)."""
    n = len(vals)
    need = max(3, -(-3 * n // 4))
    if n < need:
        return None, need
    w, c = Counter(vals.values()).most_common(1)[0]
    return (w if c >= need else None), need


def run(ctx, devices, bdf_of, budget_s=5.0, cursor_file=None, scope="plain", elf_dir=None, regions=None,
        read_timeout_s=2.0, read_word=None, read_block=None, out=print, now=time.monotonic, fw_of=None):
    t0 = now()
    rd = Reader(ctx, read_word, read_block, read_timeout_s)
    t_regions = [r for r in (regions or []) if not r[0].upper().startswith("ERISC")] if regions else TENSIX_REGIONS
    e_regions = [r for r in (regions or []) if r[0].upper().startswith("ERISC")] if regions else ETH_REGIONS
    if not regions and fw_of is not None:     # the ERISC code end of the firmware each chip runs (ETH_TEXT_END)
        e_regions, line = eth_regions({int(d.id): fw_of.get(int(d.id)) for d in devices})
        out(line)
    elfs = load_elfs(elf_dir, sorted({r[0] for r in t_regions + e_regions}))
    bdf = {int(d.id): (bdf_of.get(int(d.id)) or "").replace("0000:", "", 1) for d in devices}
    name = {c: f"{b or f'chip{c}'} tray {tray_of(b)}" for c, b in bdf.items()}
    for c in sorted(c for c, b in bdf.items() if not b):
        out(f"CODE INFO chip{c} tray ?: BDF unknown (no cluster descriptor BDF, no UMD open line); lines name it chip{c}")

    # every core of the box once, by kind and x,y (the kernel window by logical core); each chip's own location object
    # and NOC0 x,y per core
    cores, loc_of, phys = [], {}, {}
    k_regions = [r for r in t_regions if r[1] == "kernel"]
    t_regions = [r for r in t_regions if r[1] != "kernel"]
    num = lambda k: tuple(int(v) if v.isdigit() else 0 for v in k.split(","))
    for kind, blocks, regs in (("tensix", ("functional_workers",), t_regions or k_regions),
                               ("eth", ("active_eth", "eth"), e_regions)):
        if not regs:
            continue
        seen, lseen = [], []
        for d in devices:
            locs = None
            for b in blocks:
                try:
                    locs = list(d.get_block_locations(b))
                    break
                except Exception:
                    continue
            xys = [(core_xy(loc), loc) for loc in locs or []]
            xs = sorted({num(xy)[0] for xy, _ in xys if "," in xy})
            ys = sorted({num(xy)[1] for xy, _ in xys if "," in xy})
            for xy, loc in xys:
                loc_of[(int(d.id), kind, xy)] = loc
                if (kind, xy) not in seen:
                    seen.append((kind, xy))
                if kind == "tensix" and k_regions and "," in xy:     # this chip's logical core: rank in its own grid
                    lk = f"L{xs.index(num(xy)[0])},{ys.index(num(xy)[1])}"
                    loc_of[(int(d.id), "kernel", lk)], phys[(int(d.id), "kernel", lk)] = loc, xy
                    if lk not in lseen:
                        lseen.append(lk)
        seen.sort(key=lambda k: num(k[1]))
        lseen.sort(key=lambda k: num(k[1:]))
        if scope == "smoke":
            seen, lseen = seen[:SMOKE_CORES[kind]], lseen[:SMOKE_CORES[kind]]
        ph = [(kind, xy, regs) for kind, xy in seen] if (kind != "tensix" or t_regions) else []
        lg = [("kernel", lk, k_regions) for lk in lseen]
        for i in range(max(len(ph), len(lg))):        # core-major: NOC0 core i's firmware, then logical core i's kernel
            cores += ph[i:i + 1] + lg[i:i + 1]

    T = len(cores)
    start = rstart = 0
    if cursor_file and scope != "smoke":
        try:
            cur = json.load(open(cursor_file))
            start = int(cur.get("next", 0)) % max(T, 1)
            if int(cur.get("total", T)) == T:          # a region index is only good for the same core list
                rstart = max(int(cur.get("region", 0)), 0)
        except Exception:
            start = rstart = 0
    res = {"cores_total": T, "start": start, "start_region": rstart, "cores_done": 0, "regions_done": 0,
           "fail": [], "warn": [], "info": Counter(), "words": 0, "chips": len(devices)}
    dead = set()
    done = rdone = 0
    stop = None            # region of the core the budget stopped in (resumed there next run)
    for i in range(T):
        if now() - t0 >= budget_s and (done or rdone):
            break
        kind, xy, regs = cores[(start + i) % T]
        at = lambda c: phys.get((c, kind, xy), xy)        # the chip's own NOC0 x,y (the kernel window's key is logical)
        r0 = rstart if i == 0 and rstart < len(regs) else 0
        for ri in range(r0, len(regs)):
            if now() - t0 >= budget_s and rdone:      # budget between regions: a slow core stops here, resumed next run
                stop = ri
                break
            risc, rname, base, size = regs[ri]
            rdone += 1
            reads = {}
            for d in devices:
                cid = int(d.id)
                loc = loc_of.get((cid, kind, xy))
                if loc is None or cid in dead:
                    continue
                try:
                    reads[cid] = rd.block(d, loc, base, size)
                except ReadTimeout:
                    dead.add(cid)
                    msg = f"CODE WARN {name[cid]} core {at(cid)} {risc} {rname} read timeout ({read_timeout_s:g} s); chip skipped for this pass"
                    res["warn"].append(msg)
                    out(msg)
                except Exception as e:
                    res["info"]["read error"] += 1
                    out(f"CODE INFO {name[cid]} core {at(cid)} {risc} {rname} read error {type(e).__name__}: {str(e)[:120]}")
            if not reads:
                continue
            nw = min(len(w) for w in reads.values())
            dev_words = {}    # cid -> [(addr, got, expected, how)]
            elf = elfs.get(risc, {})
            for j in range(nw):
                addr = base + 4 * j
                if kind == "eth" and addr in ETH_SKIP:
                    continue
                res["words"] += 1
                vals = {c: w[j] for c, w in reads.items()}
                maj, _need = vote(vals)
                want = elf.get(addr)
                for c, v in vals.items():
                    if want is not None and v != want:
                        dev_words.setdefault(c, []).append((addr, v, want, "elf"))
                    elif maj is not None and v != maj:
                        dev_words.setdefault(c, []).append((addr, v, maj, "majority"))
                if maj is None and want is None:
                    res["info"]["no consensus"] += 1
            if kind == "kernel":       # the same other value on 2+ chips: another program's word or config, not corruption
                shared = Counter((a, g) for bad in dev_words.values() for a, g, _e, how in bad if how == "majority")
                var = {ag for ag, k in shared.items() if k >= 2}
                if var:
                    vc = sorted({c for c, bad in dev_words.items() for b in bad if (b[0], b[1]) in var})
                    res["info"]["kernel variant"] += len(var)
                    fa, fg = min(var)
                    out(f"CODE INFO kernel core {xy} {rname} {len(var)} words read the same other value on 2+ chips "
                        f"(another program or its config, not a fault; first addr 0x{fa:x} got 0x{fg:08x}): "
                        f"{', '.join(f'{name[c]} core {at(c)}' for c in vc)}")
                    dev_words = {c: kept for c, kept in ((c, [b for b in bad if (b[0], b[1]) not in var])
                                                         for c, bad in dev_words.items()) if kept}
            byid = {int(d.id): d for d in devices}
            for c, bad in dev_words.items():
                try:
                    again = rd.block(byid[c], loc_of[(c, kind, xy)], base, size)
                except ReadTimeout:
                    dead.add(c)
                    msg = f"CODE WARN {name[c]} core {at(c)} {risc} {rname} re-read timeout; {len(bad)} deviating words not confirmed"
                    res["warn"].append(msg)
                    out(msg)
                    continue
                except Exception:
                    res["info"]["re-read error"] += 1
                    continue
                conf = [b for b in bad if (b[0] - base) // 4 < len(again) and again[(b[0] - base) // 4] == b[1]]
                res["info"]["volatile"] += len(bad) - len(conf)
                if conf:
                    a, g, e, how = conf[0]
                    msg = (f"CODE FAIL {name[c]} core {at(c)} {risc} {rname} addr {a:#x} words {len(conf)} "
                           f"(got {g:#010x}, {how} {e:#010x})")
                    res["fail"].append({"bdf": bdf[c] or f"chip{c}", "tray": tray_of(bdf[c]), "core": at(c), "risc": risc,
                                        "region": rname, "addr": a, "words": len(conf), "got": g, "expected": e, "by": how})
                    out(msg)
        if stop is not None:
            break
        done += 1
    res["cores_done"], res["regions_done"] = done, rdone
    stop_region = stop or 0
    nxt = (start + done) % max(T, 1)
    if cursor_file and scope != "smoke":
        try:
            tmp = cursor_file + ".tmp"
            json.dump({"next": nxt, "region": stop_region, "total": T, "t": time.time()}, open(tmp, "w"))
            os.replace(tmp, cursor_file)
        except OSError as e:
            out(f"CODE INFO cursor file not written: {e}")
    res["next"], res["next_region"] = nxt, stop_region
    res["seconds"] = round(now() - t0, 2)
    info = ", ".join(f"{k} {v}" for k, v in sorted(res["info"].items())) or "none"
    part = (f" from region {rstart}" if rstart and T else "") + (f" next core {nxt} region {stop_region}" if stop_region else "")
    out(f"CODE SUMMARY scope {scope} cores {done} of {T} (from {start}){part} regions {rdone} chips {len(devices)} "
        f"words {res['words']} fail {len(res['fail'])} warn {len(res['warn'])} info: {info} elf {','.join(sorted(elfs)) or 'none'} "
        f"in {res['seconds']} s")
    res["info"] = dict(res["info"])
    return res
