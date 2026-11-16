# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# fw_reads.py: C1 pre firmware reads, read-only (no SPI write, no reset, no register write; the only device-side setup
# is the TLB window the library opens for its own reads). Three reads per chip:
#   SPI boot-fs: every descriptor's fd_crc, and the data checksum of each image up to FL_C1_FW_SPI_MAX bytes (8 KiB:
#     the 4 KiB config images; reading the 64-512 KiB firmware images is too slow for C1's budget (FL_C1_FW_T, 20 s;
#     the 8 KiB reads alone can take up to ~60 s on a whole box, so the timeout can stop them), their stored
#     data_crc is voted across the box instead). Runtime-written images (RUNTIME) are not recomputed.
#   ETH L1 mailbox: per ETH core the firmware version word, the port status and two heartbeat reads (stuck heartbeat
#     = WARN, judged only on cores whose port reads PORT_UP: an unused, uncabled or down port's firmware may not tick).
#   TLB readback: the 2M TLB config registers, one must hold the window of the reader's last NOC read.
#   fw_reads.py spi <chips.txt>                       pyluwen python: one JSON line per chip (descriptor table, sums)
#   fw_reads.py eth <chips.txt>                       tt_umd python ($PY): one JSON line per chip (ETH words, TLB regs)
#   fw_reads.py judge <trays.txt> <spi.jsonl> <eth.jsonl>   no device: FAIL/WARN/INFO lines, then the summary line
# chips.txt: "<bdf> <devnum>" per chip to read; trays.txt: "<bdf> tray N". A device step that fails writes
# {"err": why}; the judge turns that into one 'C1 INFO ... not checked: <why>' line, never a FAIL. When FL_C1_FW_T's
# timeout stops a step (SIGTERM), each chip not yet read gets {"err": "not checked (timeout)"}, one INFO line per chip,
# and the chip being read is released first (pyluwen holds the ARC's SPI lock while its spi_read runs: an exception
# unwinds it and dropping the chip closes its device, so no lock is left held for the next reader).
import gc
import json
import os
import signal
import sys
import time

# boot-fs (tt-zephyr-platforms include/tenstorrent/tt_boot_fs.h, lib/tenstorrent/boot_fs/tt_boot_fs.c): table at SPI 0,
# 32-byte descriptors {spi_addr, copy_dest, flags, data_crc, security_flags, image_tag[8], fd_crc}, little-endian;
# flags bits 0-23 image_size, bit 24 invalid (end of table); the "crc" is tt_boot_fs_cksum, a 32-bit sum of LE dwords;
# fd_crc sums the descriptor's first 28 bytes; CONFIG_TT_BOOT_FS_IMAGE_COUNT_MAX 32 (Kconfig default)
FD_SIZE, FD_MAX, TABLE_ADDR = 32, 32, 0x0
# Blackhole tags (boards/tenstorrent/tt_blackhole/tt_blackhole_fixed_partitions.dtsi). Not voted across chips: per-board
# data (boardcfg/origcfg: board ids; flshinfo: flash info; cmfwcfg/ethfwcfg/memfwcfg/ethsdreg: per-product and
# per-position tables) and the runtime-written ones below
RUNTIME = {"dmfw", "dmfwimg", "dmfwtail", "safetail", "imgtail", "ccfgovra", "ccfgovrb"}  # MCUboot/tools write these
NOVOTE = {"boardcfg", "origcfg", "flshinfo", "cmfwcfg", "ethfwcfg", "memfwcfg", "ethsdreg"} | RUNTIME
# ETH L1 (tt-umd device/api/tt-umd/types/blackhole_eth.hpp, arch/blackhole_implementation.hpp ETH_FW_*_ADDR,
# src/firmware/erisc_firmware.hpp): heartbeat[0] 0x7CC70, upper 16 bits a signature; version dword 0x7CFBC =
# patch | minor << 8 | major << 16. Unconfirmed on silicon: tt-umd topology_discovery skips the heartbeat on Blackhole
ETH_HB, ETH_VER = 0x7CC70, 0x7CFBC
# port_status (eth_status_t.port_status, the word hc/fleet/ethmask.py reads): 0 unknown, 1 PORT_UP, 2 down, 3 unused
ETH_PORT, PORT_UP = 0x7CC04, 1
HB_SIG = {0xABCD, 0xAABB, 0xDCBA}  # base firmware, metal, fabric
# TLB (tt-umd blackhole_implementation.hpp, pci_device.cpp configure_tlb): 2M config registers at BAR0 0x1fc00000,
# 12 bytes each, 202 of them; bits: local_offset 0 (addr >> 21, 43 bits), x_end 43, y_end 49, x_start 55, y_start 61.
# Unconfirmed: tt-umd's bar_read32 refuses addresses below 0x1FD00000, so this read may report 'not checked'
TLB_CFG, TLB_N, TLB_SZ = 0x1FC00000, 202, 12


def cksum(b):
    """tt_boot_fs_cksum: 32-bit sum of the little-endian dwords of b (a short tail is zero-padded)."""
    b = bytes(b) + b"\0" * (-len(b) % 4)
    return sum(int.from_bytes(b[i : i + 4], "little") for i in range(0, len(b), 4)) & 0xFFFFFFFF


def parse_table(raw):
    """The boot-fs descriptors of raw (the table's bytes), up to the first invalid one: list of dicts."""
    out = []
    for i in range(min(FD_MAX, len(raw) // FD_SIZE)):
        fd = raw[i * FD_SIZE : (i + 1) * FD_SIZE]
        w = [int.from_bytes(fd[j : j + 4], "little") for j in range(0, 20, 4)]
        if w[2] >> 24 & 1:
            break
        tag = fd[20:28].split(b"\0")[0].decode("ascii", "replace")
        fdc = int.from_bytes(fd[28:32], "little")
        out.append(dict(tag=tag, addr=w[0], size=w[2] & 0xFFFFFF, data_crc=w[3], fd_crc=fdc, fd_sum=cksum(fd[:28])))
    return out


def lbl(bdf):
    return str(bdf).replace("0000:", "")


def _err(e):
    return f"{type(e).__name__}: {e}".strip()[:200]


# device steps
def chips_of(path):
    return [(l.split()[0], int(l.split()[1])) for l in open(path) if len(l.split()) >= 2 and l.split()[1].isdigit()]


TIMEOUT = "not checked (timeout)"


class Timeout(BaseException):
    """FL_C1_FW_T's SIGTERM, raised inside the read so it unwinds (a BaseException: the per-chip except leaves it)."""


def _on_term(*_):
    raise Timeout()


def each_chip(path, read):
    """read(bdf, n, r, hold) fills r for one chip; hold is a list the read appends its library objects to. One JSON line
    per chip. On SIGTERM the chip being read is released (its objects dropped and collected) and each chip not yet read
    gets a TIMEOUT line."""
    chips = chips_of(path)
    old = signal.signal(signal.SIGTERM, _on_term)
    done = 0
    try:
        for bdf, n in chips:
            r, hold = {"bdf": lbl(bdf)}, []
            try:
                read(bdf, n, r, hold)
            except Exception as e:
                r["err"] = _err(e)
            finally:
                del hold[:]  # the chip's handle: dropping it closes the device and frees a lock it held
            line = json.dumps(r)
            # SIGTERM held over the print and the count: the chip's line prints once and is counted, never lost (count
            # first) nor followed by a TIMEOUT line for it as well (print first); a held SIGTERM raises right after
            prev = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            try:
                print(line, flush=True)
                done += 1
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, prev)
    except Timeout:
        pass
    gc.collect()  # the timed-out read's frame went with the exception: its handle is closed before the lines below
    try:
        for bdf, _ in chips[done:]:
            print(json.dumps({"bdf": lbl(bdf), "err": TIMEOUT}), flush=True)
    finally:
        signal.signal(signal.SIGTERM, old)


def spi(path):
    try:
        import pyluwen
    except Exception as e:
        print(json.dumps({"err": f"pyluwen not importable ({_err(e)})"}))
        return
    cap = int(os.environ.get("FL_C1_FW_SPI_MAX", 8192))
    end = time.time() + float(os.environ.get("FL_C1_FW_DEADLINE", 15))

    def read(bdf, n, r, hold):
        r.update(sums={}, late=[])
        hold.append(pyluwen.PciChip(n).as_bh())
        bh = hold[0]
        raw = bytearray(FD_SIZE * FD_MAX)
        bh.spi_read(TABLE_ADDR, raw)  # read-only
        r["table"] = raw.hex()
        for fd in parse_table(raw):
            if fd["tag"] in RUNTIME or not 0 < fd["size"] <= cap:
                continue
            if time.time() > end:
                r["late"].append(fd["tag"])
                continue
            buf = bytearray(fd["size"])
            bh.spi_read(fd["addr"], buf)
            r["sums"][fd["tag"]] = cksum(buf)

    each_chip(path, read)


def eth(path):
    try:
        import tt_umd as u
    except Exception as e:
        print(json.dumps({"err": f"tt_umd not importable ({_err(e)})"}))
        return
    rd = lambda d, c, a: int.from_bytes(d.noc_read(c[0], c[1], a, 4), "little")

    def read(bdf, n, r, hold):
        r["eth"] = []
        hold.append(u.TTDevice.create(n))
        d = hold[0]
        d.init_tt_device()
        sd = u.SocDescriptor(d)
        cs = [
            sd.translate_coord_to(c, u.CoordSystem.TRANSLATED) for c in sd.get_cores(u.CoreType.ETH, u.CoordSystem.NOC0)
        ]
        cs = [(c.x, c.y) for c in cs]
        hb1 = [rd(d, c, ETH_HB) for c in cs]
        ver = [rd(d, c, ETH_VER) for c in cs]
        port = [rd(d, c, ETH_PORT) for c in cs]
        time.sleep(0.01)
        hb2 = [rd(d, c, ETH_HB) for c in cs]
        r["eth"] = [[c[0], c[1], v, a, b, p] for c, v, a, b, p in zip(cs, ver, hb1, hb2, port)]
        if cs:  # the window of the last read above: (x, y, address)
            r["tlb_req"] = [cs[-1][0], cs[-1][1], ETH_HB]
            try:
                r["tlb"] = [[d.bar_read32(TLB_CFG + i * TLB_SZ + 4 * w) for w in range(3)] for i in range(TLB_N)]
            except Exception as e:
                r["tlb_err"] = _err(e)

    each_chip(path, read)


# judge (pure)
def load(path):
    """JSON lines of path; ([], why) when there are none or a step-wide error line."""
    recs, why = [], None
    try:
        for l in open(path):
            if l.startswith("{"):
                try:
                    j = json.loads(l)
                except ValueError:
                    continue
                if "bdf" in j:
                    recs.append(j)
                elif j.get("err"):
                    why = why or j["err"]
    except OSError as e:
        why = _err(e)
    if not recs:
        why = why or "no output"
    return recs, why


def tray_map(path):
    try:
        return {lbl(l.split()[0]): " ".join(l.split()[1:]) for l in open(path) if l.split()}
    except OSError:
        return {}


def majority(vals):
    """(value, count) held by more than half of vals, else (None, 0)."""
    best = max(set(vals), key=vals.count) if vals else None
    return (best, vals.count(best)) if vals and vals.count(best) * 2 > len(vals) else (None, 0)


def errlines(what, errs, n, at, dflt):
    """One 'C1 INFO <what> not checked' line per error reason, naming the chips unless it is all of them."""
    by = {}
    for r in errs:
        if r.get("err") != TIMEOUT:
            by.setdefault(r.get("err") or dflt, []).append(r["bdf"])
    return [f"C1 INFO {what} on {at(r['bdf'])}: {TIMEOUT}" for r in errs if r.get("err") == TIMEOUT] + [
        f"C1 INFO {what} not checked"
        + ("" if len(bs) == n else " on " + ", ".join(at(b) for b in sorted(bs)))
        + f": {why}"
        for why, bs in by.items()
    ]


def judge_spi(recs, at, inject=None):
    """recs: the spi step's chip records; at(bdf) -> 'bdf tray N'. Returns (lines, summary text)."""
    out, rows, nd, nd_ok, late = [], {}, 0, 0, 0
    errs = [r for r in recs if r.get("err") or not r.get("table")]
    out += errlines("SPI images", errs, len(recs), at, "no table read")
    for r in recs:
        if r in errs:
            continue
        fds = parse_table(bytes.fromhex(r["table"]))
        sums = dict(r.get("sums", {}))
        if inject and sums:
            s = inject(r["bdf"])
            if s:
                t = sorted(sums)[0]
                sums[t] = inject.flip(s, sums[t])
                inject.applied(s, where=f"image {t}")
        it = {}
        for fd in fds:
            it[(fd["tag"], "descriptor")] = (fd["fd_sum"], fd["fd_crc"])
            if fd["tag"] in sums:
                it[(fd["tag"], "")] = (sums[fd["tag"]], fd["data_crc"])
        rows[r["bdf"]] = dict(fds=fds, it=it)
        late += len(r.get("late", []))
    if not rows:
        return out, "SPI images not checked"
    allit = [v for c in rows.values() for v in c["it"].values()]
    if not allit or all(a != b for a, b in allit):
        out.append(
            "C1 INFO not checked: layout not recognised (SPI boot-fs: every descriptor and image on every chip "
            "mismatched)"
        )
        return out, "SPI images not checked (layout not recognised)"
    keys = sorted({k for c in rows.values() for k in c["it"]})
    for k in keys:
        has = {b: c["it"][k] for b, c in rows.items() if k in c["it"]}
        bad = sorted(b for b, (a, s) in has.items() if a != s)
        name = k[0] + (" " + k[1] if k[1] else "")
        if bad and len(bad) == len(has):
            out.append(
                f"C1 INFO SPI image {name}: checksum differs from the stored one on all {len(has)} chips, not judged"
            )
            continue
        for b in bad:
            a, s = has[b]
            out.append(f"C1 SPI FAIL {at(b)} image {name}: crc read 0x{a:08x} stored 0x{s:08x}")
    img = [v for c in rows.values() for k, v in c["it"].items() if not k[1]]
    nd = sum(1 for c in rows.values() for k in c["it"] if k[1])
    nd_ok = sum(1 for c in rows.values() for k, (a, s) in c["it"].items() if k[1] and a == s)
    # cross-chip vote of the stored image checksums (all sizes), board-specific tags left out
    voted, differ = set(), 0
    tags = sorted({fd["tag"] for c in rows.values() for fd in c["fds"]} - NOVOTE)
    for t in tags:
        v = {b: fd["data_crc"] for b, c in rows.items() for fd in c["fds"] if fd["tag"] == t}
        m, cnt = majority(list(v.values()))
        if m is None or len(v) < 2:
            continue
        voted.add(t)
        for b in sorted(v):
            if v[b] != m:
                differ += 1
                out.append(f"C1 WARN {at(b)} image {t}: stored crc 0x{v[b]:08x}, {cnt}/{len(v)} chips have 0x{m:08x}")
    ts, n = [tuple(fd["tag"] for fd in c["fds"]) for c in rows.values()], len(rows)
    m, cnt = majority(ts)
    for b, c in sorted(rows.items()):
        mine = tuple(fd["tag"] for fd in c["fds"])
        if m is not None and mine != m:
            out.append(
                f"C1 WARN {at(b)}: boot-fs image list differs from {cnt}/{n} chips (missing "
                f"{','.join(t for t in m if t not in mine) or '-'}, extra {','.join(t for t in mine if t not in m) or '-'})"
            )
    ok = sum(1 for a, s in img if a == s)
    sm = (
        f"SPI images {ok}/{len(img)} CRC ok ({nd_ok}/{nd} descriptors, {len(voted)} tags voted, {differ} differ"
        + (f", {late} not read in time" if late else "")
        + ")"
    )
    return out, sm


def ver_s(v):
    return f"{v >> 16 & 0xFF}.{v >> 8 & 0xFF}.{v & 0xFF}"


def judge_eth(recs, at):
    """recs: the eth step's chip records. Returns (lines, ETH summary text, TLB summary text)."""
    out = []
    ok = [r for r in recs if not r.get("err") and r.get("eth")]
    out += errlines("ETH mailboxes", [r for r in recs if r not in ok], len(recs), at, "no ETH cores")
    cores = [(r["bdf"], e) for r in ok for e in r["eth"]]
    sane = lambda e: 0 < (e[2] >> 16 & 0xFF) < 0xFF and e[3] >> 16 in HB_SIG
    if not cores:
        es = "ETH mailboxes not checked"
    elif not any(sane(e) for _, e in cores):
        out.append(
            "C1 INFO not checked: layout not recognised (ETH mailbox: no core reads a known version and "
            "heartbeat signature)"
        )
        es = "ETH mailboxes not checked (layout not recognised)"
    else:
        for r in ok:
            b = [f"{e[0]}-{e[1]}" for e in r["eth"] if not sane(e)]
            if b:
                out.append(
                    f"C1 WARN {at(r['bdf'])}: ETH mailbox not sane on cores {','.join(b)} (version or heartbeat "
                    f"signature)"
                )
        vs = {
            r["bdf"]: max(set(e[2] & 0xFFFFFF for e in r["eth"]), key=[e[2] & 0xFFFFFF for e in r["eth"]].count)
            for r in ok
        }
        m, cnt = majority(list(vs.values()))
        for b in sorted(vs):
            if m is not None and vs[b] != m:
                out.append(f"C1 WARN {at(b)}: ETH firmware {ver_s(vs[b])}, {cnt}/{len(vs)} chips have {ver_s(m)}")
        for r in ok:  # cores of one chip on a version other than the chip's own
            b = [f"{e[0]}-{e[1]}" for e in r["eth"] if sane(e) and e[2] & 0xFFFFFF != vs[r["bdf"]]]
            if b:
                out.append(
                    f"C1 WARN {at(r['bdf'])}: ETH firmware differs between cores ({','.join(b)} not "
                    f"{ver_s(vs[r['bdf']])})"
                )
        # heartbeat: only cores whose port is up (a record with no port word, an older reader, judges every core)
        live = lambda e: len(e) < 6 or e[5] == PORT_UP
        hb = [(bd, e) for bd, e in cores if live(e)]
        stuck = [(bd, e) for bd, e in hb if e[3] == e[4]]
        if not hb:
            out.append(f"C1 INFO ETH heartbeat not judged: no port up on any of {len(cores)} cores")
        elif len(stuck) == len(hb):
            out.append(
                f"C1 INFO ETH heartbeat did not advance on any of {len(hb)} cores between two reads, not "
                f"judged (the firmware may not update it)"
            )
        else:
            for r in ok:
                b = [f"{e[0]}-{e[1]}" for e in r["eth"] if live(e) and e[3] == e[4]]
                if b:
                    out.append(f"C1 WARN {at(r['bdf'])}: ETH heartbeat not advancing on cores {','.join(b)}")
        nok = sum(1 for _, e in cores if sane(e))
        idle = len(cores) - len(hb)
        es = f"ETH mailboxes {nok}/{len(cores)} ok (heartbeat {len(hb) - len(stuck)} advancing" + (
            f", {idle} cores with no port up not judged)" if idle else ")"
        )
    # TLB readback
    tl = [r for r in ok if r.get("tlb")]
    if not tl:
        why = next((r["tlb_err"] for r in ok if r.get("tlb_err")), "no register read")
        out.append(f"C1 INFO TLB not checked: {why}")
        return out, es, "TLB not checked"
    hit = {r["bdf"]: any(tlb_match(w, r["tlb_req"]) for w in r["tlb"]) for r in tl}
    if not any(hit.values()):
        out.append("C1 INFO not checked: layout not recognised (TLB: no config register holds the reader's window)")
        return out, es, "TLB not checked (layout not recognised)"
    for b in sorted(hit):
        if not hit[b]:
            x, y, a = next(r["tlb_req"] for r in tl if r["bdf"] == b)
            out.append(f"C1 WARN {at(b)}: no TLB config register reads back the window requested ({x},{y} 0x{a:x})")
    return out, es, f"TLB {sum(hit.values())}/{len(hit)} match"


def tlb_decode(w):
    v = w[0] | w[1] << 32 | w[2] << 64
    f = lambda o, n: v >> o & ((1 << n) - 1)
    return dict(local_offset=f(0, 43), x_end=f(43, 6), y_end=f(49, 6), x_start=f(55, 6), y_start=f(61, 6))


def tlb_match(w, req):
    t = tlb_decode(w)
    return (t["x_end"], t["y_end"], t["local_offset"]) == (req[0], req[1], req[2] >> 21)


class Inject:
    """TRAY_INJECT check=C1: the chip's computed SPI checksum is flipped before the compare (fleet/tray_inject.py)."""

    def __init__(self, ti):
        self.ti = ti

    def __call__(self, bdf):
        return self.ti.want("C1", bdf, n=0)

    def flip(self, s, v):
        return self.ti.flip_int(s, v)

    def applied(self, s, where=""):
        self.ti.applied(s, where=where)


def judge(trays, spi_p, eth_p):
    tm = tray_map(trays)
    at = lambda b: f"{lbl(b)} {tm.get(lbl(b), 'tray ?')}"
    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path:  # python3 -I leaves the script's directory off sys.path
        sys.path.insert(1, d)
    try:
        import tray_inject

        inj = Inject(tray_inject) if tray_inject.active() else None
    except Exception:
        inj = None
    lines, sums = [], []
    recs, why = load(spi_p)
    if recs:
        try:
            o, s = judge_spi(recs, at, inj)
        except Exception as e:
            o, s = [f"C1 INFO SPI images not checked: {_err(e)}"], "SPI images not checked"
    else:
        o, s = [f"C1 INFO SPI images not checked: {why}"], "SPI images not checked"
    lines += o
    sums.append(s)
    recs, why = load(eth_p)
    if recs:
        try:
            o, e, t = judge_eth(recs, at)
        except Exception as x:
            o, e, t = [f"C1 INFO ETH mailboxes not checked: {_err(x)}"], "ETH mailboxes not checked", "TLB not checked"
    else:
        o, e, t = [f"C1 INFO ETH mailboxes not checked: {why}"], "ETH mailboxes not checked", "TLB not checked"
    lines += o
    sums += [e, t]
    for l in lines:
        print(l)
    print("C1 firmware reads: " + ", ".join(sums))


if __name__ == "__main__":
    a = sys.argv[1:]
    try:
        if a[:1] == ["spi"]:
            spi(a[1])
        elif a[:1] == ["eth"]:
            eth(a[1])
        elif a[:1] == ["judge"]:
            judge(*a[1:4])
        else:
            sys.exit("usage: fw_reads.py spi|eth <chips.txt> | judge <trays.txt> <spi.jsonl> <eth.jsonl>")
    except Exception as e:  # never a crash: a device step's error line, the judge's INFO line
        print(json.dumps({"err": _err(e)}) if a[:1] != ["judge"] else f"C1 INFO firmware reads not checked: {_err(e)}")
