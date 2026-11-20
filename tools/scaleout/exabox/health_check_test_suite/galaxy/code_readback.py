#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# N3 code readback: read every RISC's instruction memory back through
# exalens and judge it without a reference copy. Only code is judged, never data, bss, stack, counters or mailboxes
# (a NOC transaction counter at 0x8514 and kernel runtime args read as CODE FAIL on healthy boxes).
# Tensix firmware: the executable sections (SHF_EXECINSTR) of the BRISC/NCRISC/TRISC0-2 firmware ELFs, from their
# section headers: --elf-dir, else the installed ttnn's pre-compiled firmware and TT_METAL_CACHE's firmware. A layout
# counts only when its first code words match what the chips hold (another arch's or ttnn version's ELF never sets a
# window). Kernels: tt-metal relocates kernel code into the kernel config buffer at load time, so a kernel ELF's
# addresses are not its L1 addresses; the executable sections of TT_METAL_CACHE's Tensix kernel ELFs are found by content
# in each logical core's majority read of the L1 after the firmware, and only words inside a found section are judged.
# No readable or matching ELF: 'CODE INFO code readback not judged: <reason>', never a FAIL. ETH: the base-FW text only
# (never its data, stack or mailboxes), its end from each chip's firmware bundle version (the base FW has no ELF on the
# host). All chips of one box run the same firmware
# and, after the heavy stage, the same kernels on the same logical cores. The firmware regions vote by NOC0 x,y. The kernel
# window votes by logical core: each chip's own Tensix grid after harvesting, columns and rows ranked
# left to right and top to bottom as tt-metal places programs. Chips harvest different columns, so one NOC0 x,y is
# logical core 0,0 on one chip and 1,0 on another, with another kernel and its own runtime args (seen on one box:
# 120 false CODE FAIL lines at 0xa180 and 0xb9f4 on the 7 chips with column x=1 or x=2 harvested). A kernel-window
# word that two or more chips read with the same other value is another program's word or config (INFO 'kernel
# variant'), never a FAIL: corruption is one chip's own; so is a word that equals the code of a kernel ELF found at that
# place in the chip's own read (another program variant, e.g. li a4,9 for li a3,0). The cores at one key vote word by
# word. A strong majority (>= max(3, 75 %) of the chips read) sets the expected word; a chip whose
# word differs is re-read, and only a word that reads back the same wrong value again is a CODE FAIL.
# A word with no strong majority is INFO; a word that changes on re-read is volatile (INFO); a read that
# times out skips that chip (WARN), never a FAIL. With --elf-dir, the executable sections of <risc>.elf are compared
# word by word too. --region replaces the derived windows (raw, for tests and manual reads). Reads only: nothing is
# written to any chip.
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
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))  # the sibling programs, also when run or imported through a link
import re
import signal
import struct
import time
from collections import Counter

# Tensix RISCs whose firmware and kernel ELFs are read; L1 addresses only (local memory 0xffb00000 up is not L1).
TENSIX_RISCS = ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")
L1_TOP = 0x200000
KERNEL_SCAN = (
    16384  # bytes of L1 after the last firmware code searched for kernel code (MEM_MAP_END and its config buffer)
)
KERNEL_ELF_CAP = 5000  # kernel ELFs read at most (newest first)
PROBE_WORDS = 4  # first code words of BRISC firmware read to tell which ELF layout the chips run
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
ETH_REGIONS = [("ERISC", "basefw", 0x70000, 0xB000)]  # run() without fw_of: 19.7 and later
ETH_SKIP = {0x7CC70, 0x7CC74}  # base-FW and metal heartbeats count by design (only an explicit --region reaches them)
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
    return list(struct.unpack(f"<{n}I", b[: 4 * n]))


def elf_exec_sections(path):
    """[(address, bytes)] of every allocated executable PROGBITS section (SHF_ALLOC|SHF_EXECINSTR, from the section
    headers) of a 32-bit little-endian ELF; ValueError when the file is not one."""
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != b"\x7fELF" or len(b) < 52 or b[4] != 1 or b[5] != 1:
        raise ValueError(f"{path}: not a 32-bit little-endian ELF")
    (shoff,) = struct.unpack_from("<I", b, 32)
    shentsize, shnum = struct.unpack_from("<HH", b, 46)
    if not shoff or shentsize < 40 or shoff + shnum * shentsize > len(b):
        raise ValueError(f"{path}: no readable section headers")
    out = []
    for i in range(shnum):
        _n, sh_type, flags, addr, off, size = struct.unpack_from("<6I", b, shoff + i * shentsize)
        if sh_type != 1 or (flags & 0x6) != 0x6 or not size:  # SHT_PROGBITS, SHF_ALLOC|SHF_EXECINSTR
            continue
        if off + size > len(b):
            raise ValueError(f"{path}: section {i} runs past the end of the file")
        out.append((addr, b[off : off + size]))
    return out


def elf_words(path):
    """{address: word} for every executable section (elf_exec_sections) of a 32-bit LE ELF."""
    out = {}
    for addr, data in elf_exec_sections(path):
        for j in range(0, len(data) - 3, 4):
            (out[addr + j],) = struct.unpack_from("<I", data, j)
    return out


def risc_elf(d, risc):
    """<d>/<risc>*.elf or <d>/<risc>/*.elf (first by name), or None."""
    r = risc.lower()
    hits = sorted(glob.glob(os.path.join(d, r + "*.elf")) + glob.glob(os.path.join(d, r, "*.elf")))
    return hits[0] if hits else None


def load_elfs(elf_dir, riscs):
    got = {}
    if not elf_dir:
        return got
    for r in riscs:
        h = risc_elf(elf_dir, r)
        if h:
            got[r] = elf_words(h)
    return got


def elf_roots():
    """(firmware roots, kernel roots) searched by default: TT_METAL_CACHE (tt-metal's default ~/.cache/tt-metal-cache when
    unset) for JIT firmware and kernels, and the pre-compiled firmware of the installed ttnn and of TT_METAL_HOME."""
    cache = os.environ.get("TT_METAL_CACHE") or os.path.expanduser("~/.cache/tt-metal-cache")
    fw = [cache]
    try:
        import importlib.util

        spec = importlib.util.find_spec("ttnn")  # finds the package without importing it
        for loc in (spec.submodule_search_locations or []) if spec else []:
            fw.append(os.path.join(loc, "tt_metal", "pre-compiled"))
    except Exception:
        pass
    if os.environ.get("TT_METAL_HOME"):
        fw.append(os.path.join(os.environ["TT_METAL_HOME"], "tt_metal", "pre-compiled"))
    return fw, [cache]


def fw_sets(roots):
    """Firmware set dirs under the roots: a dir holding brisc*.elf or brisc/*.elf itself (--elf-dir), pre-compiled/<hash>/,
    and TT_METAL_CACHE's [<build>/]firmware/<hash>/."""
    out = []
    for root in roots:
        cand = [root] if risc_elf(root, "BRISC") else []
        for pat in ("*/brisc/brisc.elf", "firmware/*/brisc/brisc.elf", "*/firmware/*/brisc/brisc.elf"):
            cand += [os.path.dirname(os.path.dirname(h)) for h in sorted(glob.glob(os.path.join(root, pat)))]
        out += [c for c in cand if c not in out]
    return out[:200]


def fw_layouts(roots):
    """{layout: [(set dir, {risc: words of its first code})]}, the errors seen and the sets found. A layout is
    ((risc, start, end), ...) of the executable sections in L1 of the five Tensix firmware ELFs of one set."""
    lay, errs, sets = {}, [], fw_sets(roots)
    for d in sets:
        rng, first = [], {}
        try:
            for r in TENSIX_RISCS:
                h = risc_elf(d, r)
                if not h:
                    raise ValueError(f"{d}: no {r.lower()} ELF")
                secs = sorted((a, len(b), b) for a, b in elf_exec_sections(h) if a + len(b) <= L1_TOP)
                if not secs:
                    raise ValueError(f"{h}: no executable section in L1")
                for a, n, b in secs:
                    if rng and rng[-1][0] == r and rng[-1][2] >= a:  # adjacent sections of one RISC: one region
                        rng[-1] = (r, rng[-1][1], max(rng[-1][2], a + n))
                    else:
                        rng.append((r, a, a + n))
                first[r] = (secs[0][0], to_words(secs[0][2][: 4 * PROBE_WORDS]))
        except (OSError, ValueError, struct.error) as e:
            errs.append(str(e)[:200])
            continue
        lay.setdefault(tuple(rng), []).append((d, first))
    return lay, errs, sets


def kernel_sections(roots, cap=KERNEL_ELF_CAP):
    """(sections, ELFs read, errors): the distinct executable sections (as word tuples, at least 8 words with 4+ distinct
    words) of the Tensix kernel ELFs under the roots ([<build>/]kernels/<name>/<hash>/<risc>/<risc>.elf), newest first.
    """
    paths = []
    for root in roots:
        for r in TENSIX_RISCS:
            r = r.lower()
            for pat in (f"kernels/*/*/{r}/{r}.elf", f"*/kernels/*/*/{r}/{r}.elf"):
                paths += glob.glob(os.path.join(root, pat))
    try:
        paths.sort(key=lambda p: -os.path.getmtime(p))
    except OSError:
        pass
    secs, errs, n = set(), [], 0
    for p in paths[:cap]:
        try:
            for _a, b in elf_exec_sections(p):
                w = tuple(to_words(b))
                if len(w) >= 8 and len(set(w)) >= 4:
                    secs.add(w)
            n += 1
        except (OSError, ValueError, struct.error) as e:
            errs.append(str(e)[:200])
    return sorted(secs), n, errs


def locate(words, sections, ratio=0.9):
    """{word index: {ELF words}} for every section found by content in words (a read, None where unknown): at an offset
    where 4 words of the section match and >= ratio of its overlap with the read matches (words tt-metal relocates at
    load time may differ). A section must overlap the read by 16 words or all of its own."""
    m, idx, out = len(words), {}, {}
    for p in range(m - 3):
        t = tuple(words[p : p + 4])
        if None not in t and len(set(t)) > 1:
            idx.setdefault(t, []).append(p)
    for sec in sections:
        n = len(sec)
        step = max(1, (n - 3) // 8)
        offs = set()
        for i in range(0, n - 3, step):
            for p in idx.get(tuple(sec[i : i + 4]), ())[:8]:
                offs.add(p - i)
        for off in sorted(offs):
            lo, hi = max(0, off), min(m, off + n)
            if hi - lo < min(n, 16):
                continue
            if sum(words[k] == sec[k - off] for k in range(lo, hi)) < ratio * (hi - lo):
                continue
            for k in range(lo, hi):
                out.setdefault(k, set()).add(sec[k - off])
    return out


BUS_TO_TRAY = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}  # as run.sh rk_tray and fleet/chunk_sched.py


def tray_of(bdf):
    """'0000:41:00.0' or '41:00.0' -> '2'; '?' when the BDF is unknown or not on a galaxy tray bus."""
    try:
        return str(BUS_TO_TRAY.get(int(str(bdf).replace("0000:", "", 1).split(":")[0], 16) & 0xF0, "?"))
    except (ValueError, IndexError):
        return "?"


def _umd():
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
    return [
        ("ERISC", "basefw", 0x70000, end - 0x70000)
    ], f"CODE INFO ERISC basefw voted 0x70000-{end:#x}: tt_fw_bundle_ver {why}"


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
                return to_words(
                    timed(self.t, lambda: self.rb(loc, addr, device_id=dev.id, num_bytes=size, context=self.ctx))
                )
            except ReadTimeout:
                raise
            except Exception:
                pass
        return [
            timed(self.t, lambda a=a: int(self.rw(loc, a, device_id=dev.id, context=self.ctx)))
            for a in range(addr, addr + size, 4)
        ]


def vote(vals):
    """vals {chip: word} -> (majority word or None, threshold)."""
    n = len(vals)
    need = max(3, -(-3 * n // 4))
    if n < need:
        return None, need
    w, c = Counter(vals.values()).most_common(1)[0]
    return (w if c >= need else None), need


def _inject(reads, bdf, at, kind, xy, cores, start, risc, rname, base):
    """TRAY_INJECT (test-only, fleet/tray_inject.py): flips one code word in the target chip's read of the
    target core (spec core= by NOC0 x,y, else the pass's first firmware core) in place, before the vote; returns
    {chip: (word index, spec)} so the confirming re-read gets the same flip. {} and reads untouched when unset."""
    import tray_inject as ti

    hit = {}
    for c in list(reads):
        s = ti.want("N3", bdf[c]) if bdf.get(c) else None
        if s is None or kind == "kernel":  # the kernel window judges only words found in a kernel section: never here
            continue
        first = next((k, x) for k, x, _r in cores[start:] + cores[:start] if k != "kernel")
        if (at(c) != f"{s['core'][0]},{s['core'][1]}") if s["core"] else ((kind, xy) != first):
            continue
        j = next((j for j in range(len(reads[c])) if not (kind == "eth" and base + 4 * j in ETH_SKIP)), None)
        if j is None:
            continue
        reads[c] = list(reads[c])
        reads[c][j] = ti.flip_int(s, reads[c][j], 32)
        ti.applied(s, core=tuple(int(v) for v in at(c).split(",")), where=f"{risc} {rname} addr {base + 4 * j:#x}")
        hit[c] = (j, s)
    return hit


def not_judged(out, what, why):
    out(f"CODE INFO code readback not judged: {what}: {why}")


def tensix_regions(devices, rd, elf_dir, fw_roots, kernel_roots, out):
    """(Tensix regions, kernel code sections or None) from the ELFs (header comment); 'not judged' INFO lines for what
    has no readable or matching ELF."""
    roots = [elf_dir] if elf_dir else list(fw_roots if fw_roots is not None else elf_roots()[0])
    lay, errs, sets = fw_layouts(roots)
    for e in errs[:3]:
        out(f"CODE INFO firmware ELF unreadable: {e}")
    where = ", ".join(roots) or "no root"
    if not lay:
        why = f"no readable firmware ELF set ({len(sets)} found under {where}" + (f"; {errs[0]}" if errs else "") + ")"
        not_judged(out, "Tensix firmware and kernel code", why)
        return [], None
    # which layout the chips run: the first BRISC code words of each layout, read on the first Tensix core of up to 3 chips
    probe = []
    for d in devices:
        try:
            xy = lambda loc: tuple(int(v) if v.isdigit() else 0 for v in core_xy(loc).split(","))
            locs = sorted(d.get_block_locations("functional_workers"), key=xy)
        except Exception:
            continue
        if locs:
            probe.append((d, locs[0]))
        if len(probe) == 3:
            break
    ok, sets_ok = [], []
    for rng, members in lay.items():
        addr = members[0][1]["BRISC"][0]
        got = []
        for d, loc in probe:
            try:
                got.append(tuple(rd.block(d, loc, addr, 4 * PROBE_WORDS)))
            except Exception:
                pass
        if not got:
            continue
        w, c = Counter(got).most_common(1)[0]
        hit = [d for d, f in members if tuple(f["BRISC"][1]) == w]
        if 2 * c > len(got) and hit:
            ok.append(rng)
            sets_ok += hit
    if not ok:
        why = f"none of {len(lay)} firmware ELF layout{'s' if len(lay) > 1 else ''} ({len(sets)} sets under {where}) matches the firmware the chips run"
        not_judged(out, "Tensix firmware and kernel code", why)
        return [], None
    if len(ok) > 1:  # several layouts match: judge only the words every one of them calls code
        keep = []
        for r, a, e in ok[0]:
            lo, hi = a, e
            for other in ok[1:]:
                span = [(a2, e2) for r2, a2, e2 in other if r2 == r and a2 < hi and e2 > lo]
                lo, hi = (max(lo, span[0][0]), min(hi, span[0][1])) if len(span) == 1 else (0, 0)
            if hi > lo:
                keep.append((r, lo, hi))
        rng = tuple(keep)
    else:
        rng = ok[0]
    regs = [(r, "fw", a, e - a) for r, a, e in rng]
    out(
        "CODE INFO Tensix firmware code from ELF section headers (executable sections only): "
        + ", ".join(f"{r} {a:#x}-{a + n:#x}" for r, _k, a, n in regs)
        + f" ({len(sets_ok)} matching set{'s' if len(sets_ok) > 1 else ''}, e.g. {sets_ok[0]})"
    )
    kroots = list(kernel_roots if kernel_roots is not None else elf_roots()[1])
    secs, nk, kerrs = kernel_sections(kroots)
    for e in kerrs[:3]:
        out(f"CODE INFO kernel ELF unreadable: {e}")
    base = -(-max(a + n for _r, _k, a, n in regs) // 16) * 16
    if not secs:
        not_judged(
            out, "kernel code", f"no readable Tensix kernel ELF under {', '.join(kroots) or 'no root'} ({nk} read)"
        )
        return regs, None
    out(
        f"CODE INFO kernel code: {len(secs)} executable sections of {nk} kernel ELFs, found by content in "
        f"{base:#x}-{base + KERNEL_SCAN:#x}; only words inside a found section are judged"
    )
    return regs + [("kernel", "kernel", base, KERNEL_SCAN)], secs


def run(
    ctx,
    devices,
    bdf_of,
    budget_s=5.0,
    cursor_file=None,
    scope="plain",
    elf_dir=None,
    regions=None,
    read_timeout_s=2.0,
    read_word=None,
    read_block=None,
    out=print,
    now=time.monotonic,
    fw_of=None,
    fw_roots=None,
    kernel_roots=None,
):
    t0 = now()
    rd = Reader(ctx, read_word, read_block, read_timeout_s)
    kcode = None  # kernel code sections (default regions): only words inside one found in the read are judged
    if regions:
        t_regions = [r for r in regions if not r[0].upper().startswith("ERISC")]
        e_regions = [r for r in regions if r[0].upper().startswith("ERISC")]
    else:
        e_regions = ETH_REGIONS
        if fw_of is not None:  # the ERISC code end of the firmware each chip runs (ETH_TEXT_END)
            e_regions, line = eth_regions({int(d.id): fw_of.get(int(d.id)) for d in devices})
            out(line)
        t_regions, kcode = tensix_regions(devices, rd, elf_dir, fw_roots, kernel_roots, out)
    elfs = load_elfs(elf_dir, sorted({r[0] for r in t_regions + e_regions}))
    bdf = {int(d.id): (bdf_of.get(int(d.id)) or "").replace("0000:", "", 1) for d in devices}
    name = {c: f"{b or f'chip{c}'} tray {tray_of(b)}" for c, b in bdf.items()}
    for c in sorted(c for c, b in bdf.items() if not b):
        out(
            f"CODE INFO chip{c} tray ?: BDF unknown (no cluster descriptor BDF, no UMD open line); lines name it chip{c}"
        )

    # every core of the box once, by kind and x,y (the kernel window by logical core); each chip's own location object
    # and NOC0 x,y per core
    cores, loc_of, phys = [], {}, {}
    k_regions = [r for r in t_regions if r[1] == "kernel"]
    t_regions = [r for r in t_regions if r[1] != "kernel"]
    num = lambda k: tuple(int(v) if v.isdigit() else 0 for v in k.split(","))
    for kind, blocks, regs in (
        ("tensix", ("functional_workers",), t_regions or k_regions),
        ("eth", ("active_eth", "eth"), e_regions),
    ):
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
                if kind == "tensix" and k_regions and "," in xy:  # this chip's logical core: rank in its own grid
                    lk = f"L{xs.index(num(xy)[0])},{ys.index(num(xy)[1])}"
                    loc_of[(int(d.id), "kernel", lk)], phys[(int(d.id), "kernel", lk)] = loc, xy
                    if lk not in lseen:
                        lseen.append(lk)
        seen.sort(key=lambda k: num(k[1]))
        lseen.sort(key=lambda k: num(k[1:]))
        if scope == "smoke":
            seen, lseen = seen[: SMOKE_CORES[kind]], lseen[: SMOKE_CORES[kind]]
        ph = [(kind, xy, regs) for kind, xy in seen] if (kind != "tensix" or t_regions) else []
        lg = [("kernel", lk, k_regions) for lk in lseen]
        for i in range(max(len(ph), len(lg))):  # core-major: NOC0 core i's firmware, then logical core i's kernel
            cores += ph[i : i + 1] + lg[i : i + 1]

    T = len(cores)
    start = rstart = cycle = 0
    # the cursor is kept per host outside the run dir (fleet.sh), so successive plain runs cover every core; it starts
    # over at core 0 when the chips' firmware changes (another layout, and the old cycle no longer says what was read)
    fw = ",".join(sorted({str(v) for v in fw_of.values() if v})) if fw_of else ""
    if cursor_file and scope != "smoke":
        try:
            cur = json.load(open(cursor_file))
            if fw and cur.get("fw") and cur.get("fw") != fw:
                out(f"CODE INFO code cursor reset to core 0: firmware {fw} (was {cur.get('fw')})")
            else:
                start = int(cur.get("next", 0)) % max(T, 1)
                cycle = max(int(cur.get("cycle", 0)), 0)
                if int(cur.get("total", T)) == T:  # a region index is only good for the same core list
                    rstart = max(int(cur.get("region", 0)), 0)
        except Exception:
            start = rstart = cycle = 0
    res = {
        "cores_total": T,
        "start": start,
        "start_region": rstart,
        "cores_done": 0,
        "regions_done": 0,
        "fail": [],
        "warn": [],
        "info": Counter(),
        "words": 0,
        "chips": len(devices),
    }
    dead = set()
    done = rdone = 0
    stop = None  # region of the core the budget stopped in (resumed there next run)
    for i in range(T):
        if now() - t0 >= budget_s and (done or rdone):
            break
        kind, xy, regs = cores[(start + i) % T]
        at = lambda c: phys.get((c, kind, xy), xy)  # the chip's own NOC0 x,y (the kernel window's key is logical)
        r0 = rstart if i == 0 and rstart < len(regs) else 0
        for ri in range(r0, len(regs)):
            if now() - t0 >= budget_s and rdone:  # budget between regions: a slow core stops here, resumed next run
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
                    out(
                        f"CODE INFO {name[cid]} core {at(cid)} {risc} {rname} read error {type(e).__name__}: {str(e)[:120]}"
                    )
            if not reads:
                continue
            # TRAY_INJECT (test-only): one flipped code word in the target chip's read; a no-op when unset
            flip = _inject(reads, bdf, at, kind, xy, cores, start, risc, rname, base)
            nw = min(len(w) for w in reads.values())
            dev_words = {}  # cid -> [(addr, got, expected, how)]
            elf = elfs.get(risc, {})
            code = None  # the kernel window's word indexes inside a kernel code section found in the majority read
            if kind == "kernel" and kcode is not None:
                code = locate([vote({c: w[j] for c, w in reads.items()})[0] for j in range(nw)], kcode)
                if not code:
                    res["info"]["kernel window without known kernel code"] += 1
            for j in range(nw):
                addr = base + 4 * j
                if kind == "eth" and addr in ETH_SKIP:
                    continue
                if code is not None and j not in code:
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
            if kind == "kernel":  # the same other value on 2+ chips: another program's word or config, not corruption
                shared = Counter((a, g) for bad in dev_words.values() for a, g, _e, how in bad if how == "majority")
                var = {ag for ag, k in shared.items() if k >= 2}
                if var:
                    vc = sorted({c for c, bad in dev_words.items() for b in bad if (b[0], b[1]) in var})
                    res["info"]["kernel variant"] += len(var)
                    fa, fg = min(var)
                    out(
                        f"CODE INFO kernel core {xy} {rname} {len(var)} words read the same other value on 2+ chips "
                        f"(another program or its config, not a fault; first addr 0x{fa:x} got 0x{fg:08x}): "
                        f"{', '.join(f'{name[c]} core {at(c)}' for c in vc)}"
                    )
                    dev_words = {
                        c: kept
                        for c, kept in (
                            (c, [b for b in bad if (b[0], b[1]) not in var]) for c, bad in dev_words.items()
                        )
                        if kept
                    }
            if code is not None and dev_words:  # the chip's own read holds another kernel's code there: a variant
                for c in list(dev_words):
                    own = locate(reads[c][:nw], kcode)
                    kept = [b for b in dev_words[c] if b[1] not in own.get((b[0] - base) // 4, ())]
                    if len(kept) < len(dev_words[c]):
                        res["info"]["kernel variant"] += len(dev_words[c]) - len(kept)
                        b = next(b for b in dev_words[c] if b not in kept)
                        out(
                            f"CODE INFO kernel core {xy} {rname} {len(dev_words[c]) - len(kept)} words of {name[c]} core {at(c)} "
                            f"are another kernel's code (first addr 0x{b[0]:x} got 0x{b[1]:08x}, majority 0x{b[2]:08x}), not a fault"
                        )
                    if kept:
                        dev_words[c] = kept
                    else:
                        del dev_words[c]
            byid = {int(d.id): d for d in devices}
            for c, bad in dev_words.items():
                try:
                    again = rd.block(byid[c], loc_of[(c, kind, xy)], base, size)
                    if c in flip:  # TRAY_INJECT: the re-read holds the same flipped word (a stuck fault, not volatile)
                        again = list(again)
                        again[flip[c][0]] = reads[c][flip[c][0]]
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
                    msg = (
                        f"CODE FAIL {name[c]} core {at(c)} {risc} {rname} addr {a:#x} words {len(conf)} "
                        f"(got {g:#010x}, {how} {e:#010x})"
                    )
                    res["fail"].append(
                        {
                            "bdf": bdf[c] or f"chip{c}",
                            "tray": tray_of(bdf[c]),
                            "core": at(c),
                            "risc": risc,
                            "region": rname,
                            "addr": a,
                            "words": len(conf),
                            "got": g,
                            "expected": e,
                            "by": how,
                        }
                    )
                    out(msg)
        if stop is not None:
            break
        done += 1
    res["cores_done"], res["regions_done"] = done, rdone
    stop_region = stop or 0
    nxt = (start + done) % max(T, 1)
    ncycle = cycle + (start + done) // max(T, 1)  # a cycle ends each time the cursor wraps past the last core
    if cursor_file and scope != "smoke":
        try:
            tmp = cursor_file + ".tmp"
            json.dump(
                {"next": nxt, "region": stop_region, "total": T, "cycle": ncycle, "fw": fw, "t": time.time()},
                open(tmp, "w"),
            )
            os.replace(tmp, cursor_file)
        except OSError as e:
            out(f"CODE INFO cursor file not written: {e}")
    res["next"], res["next_region"], res["cycle"] = nxt, stop_region, cycle
    if cursor_file and scope != "smoke" and T:
        span = f"cores {start}-{(start + done - 1) % T}" if done else f"core {start} part (regions {rdone})"
        out(f"CODE INFO code coverage: {span} of {T}, cycle {cycle}")
    res["seconds"] = round(now() - t0, 2)
    info = ", ".join(f"{k} {v}" for k, v in sorted(res["info"].items())) or "none"
    part = (f" from region {rstart}" if rstart and T else "") + (
        f" next core {nxt} region {stop_region}" if stop_region else ""
    )
    out(
        f"CODE SUMMARY scope {scope} cores {done} of {T} (from {start}){part} regions {rdone} chips {len(devices)} "
        f"words {res['words']} fail {len(res['fail'])} warn {len(res['warn'])} info: {info} elf {','.join(sorted(elfs)) or 'none'} "
        f"in {res['seconds']} s"
    )
    res["info"] = dict(res["info"])
    return res
