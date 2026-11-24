# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# fw_reads.py: firmware reads before the load, read-only (no SPI or register write, no reset). Device steps, one JSON
# line per chip of chips.txt ("<bdf> <devnum>" lines), an {"err": why} line when a step cannot run:
#   spi <chips.txt>  pyluwen: the SPI boot-fs table and the checksum of each image up to SPI_MAX bytes
#   eth <chips.txt>  tt_umd: each ETH core's firmware version, port status and two heartbeat reads; the 2M TLB registers
#   judge <trays.txt> <spi.jsonl> <eth.jsonl>  no device: FAIL/WARN/INFO lines, then the summary line
import gc
import importlib
import json
import os
import signal
import sys
import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)))  # tray_inject, also when run through a link

# boot-fs (tt-zephyr-platforms tt_boot_fs.h): at SPI 0, 32-byte LE descriptors {spi_addr, copy_dest, flags (bits
# 0-23 size, bit 24 invalid: end of table), data_crc, security_flags, image_tag[8], fd_crc}; a crc: 32-bit sum of dwords
FD_SIZE, FD_MAX, TABLE_ADDR = 32, 32, 0x0
SPI_MAX, SPI_DEADLINE_S = 8192, 15
# images written at runtime are not summed; those and per-board data are not voted across the chips
RUNTIME = {"dmfw", "dmfwimg", "dmfwtail", "safetail", "imgtail", "ccfgovra", "ccfgovrb"}
NOVOTE = {"boardcfg", "origcfg", "flshinfo", "cmfwcfg", "ethfwcfg", "memfwcfg", "ethsdreg"} | RUNTIME
# ETH L1 (tt-umd blackhole_eth.hpp): heartbeat (upper 16 bits a signature), version (major.minor.patch), port status
ETH_HB, ETH_VER, ETH_PORT, PORT_UP = 0x7CC70, 0x7CFBC, 0x7CC04, 1
HB_SIG = {0xABCD, 0xAABB, 0xDCBA}  # base firmware, metal, fabric
# 2M TLB config registers (tt-umd blackhole_implementation.hpp), 12 bytes each: local_offset bits 0-42, x_end 43, y 49
TLB_CFG, TLB_N, TLB_SZ = 0x1FC00000, 202, 12
TIMEOUT = "not checked (timeout)"


def cksum(b):
    b = bytes(b) + b"\0" * (-len(b) % 4)
    return sum(int.from_bytes(b[i : i + 4], "little") for i in range(0, len(b), 4)) & 0xFFFFFFFF


def parse_table(raw):
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


def chips_of(path):
    return [(l.split()[0], int(l.split()[1])) for l in open(path) if len(l.split()) >= 2 and l.split()[1].isdigit()]


class Timeout(BaseException):  # SIGTERM, raised inside the read so it unwinds past the per-chip except
    @staticmethod
    def on(*_):
        raise Timeout()


def each_chip(path, mod, read):
    """read(lib, n, r, hold) fills r for one chip, its library objects in hold: one JSON line per chip."""
    try:
        lib = importlib.import_module(mod)
    except Exception as e:
        return print(json.dumps({"err": f"{mod} not importable ({_err(e)})"}))
    chips, done = chips_of(path), 0
    old = signal.signal(signal.SIGTERM, Timeout.on)
    try:
        for bdf, n in chips:
            r, hold = {"bdf": lbl(bdf)}, []
            try:
                read(lib, n, r, hold)
            except Exception as e:
                r["err"] = _err(e)
            finally:
                del hold[:]  # dropping the chip's handle closes the device and frees a lock it held
            line = json.dumps(r)
            prev = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})  # the line printed and counted once
            try:
                print(line, flush=True)
                done += 1
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, prev)
    except Timeout:
        pass
    gc.collect()  # the timed-out read's handle is closed before the lines below
    try:
        for bdf, _ in chips[done:]:
            print(json.dumps({"bdf": lbl(bdf), "err": TIMEOUT}), flush=True)
    finally:
        signal.signal(signal.SIGTERM, old)


def spi(path):
    end = time.time() + SPI_DEADLINE_S

    def read(pyluwen, n, r, hold):
        r.update(sums={}, late=[])
        hold.append(bh := pyluwen.PciChip(n).as_bh())
        bh.spi_read(TABLE_ADDR, raw := bytearray(FD_SIZE * FD_MAX))
        r["table"] = raw.hex()
        for fd in parse_table(raw):
            if fd["tag"] in RUNTIME or not 0 < fd["size"] <= SPI_MAX:
                continue
            elif time.time() > end:
                r["late"].append(fd["tag"])
            else:
                bh.spi_read(fd["addr"], buf := bytearray(fd["size"]))
                r["sums"][fd["tag"]] = cksum(buf)

    each_chip(path, "pyluwen", read)


def eth(path):
    def read(u, n, r, hold):
        r["eth"] = []
        hold.append(d := u.TTDevice.create(n))
        d.init_tt_device()
        sd = u.SocDescriptor(d)
        cores = sd.get_cores(u.CoreType.ETH, u.CoordSystem.NOC0)
        cs = [(t.x, t.y) for t in (sd.translate_coord_to(c, u.CoordSystem.TRANSLATED) for c in cores)]
        rd = lambda a: [int.from_bytes(d.noc_read(x, y, a, 4), "little") for x, y in cs]  # noqa: E731
        hb1, ver, port = rd(ETH_HB), rd(ETH_VER), rd(ETH_PORT)
        time.sleep(0.01)
        r["eth"] = [[c[0], c[1], v, a, b, p] for c, v, a, b, p in zip(cs, ver, hb1, rd(ETH_HB), port)]
        if cs:  # the window of the last read: (x, y, address)
            r["tlb_req"] = [cs[-1][0], cs[-1][1], ETH_HB]
            try:
                r["tlb"] = [[d.bar_read32(TLB_CFG + i * TLB_SZ + 4 * w) for w in range(3)] for i in range(TLB_N)]
            except Exception as e:
                r["tlb_err"] = _err(e)

    each_chip(path, "tt_umd", read)


def load(path):
    recs, why = [], None
    try:
        for l in open(path):
            try:
                j = json.loads(l) if l.startswith("{") else {}
            except ValueError:
                continue
            if "bdf" in j:
                recs.append(j)
            elif j.get("err"):
                why = why or j["err"]
    except OSError as e:
        why = _err(e)
    return recs, why or (None if recs else "no output")


def majority(vals):
    best = max(set(vals), key=vals.count) if vals else None
    return (best, vals.count(best)) if vals and vals.count(best) * 2 > len(vals) else (None, 0)


def errlines(what, errs, n, at, dflt):
    by = {}
    for r in errs:
        by.setdefault(r.get("err") or dflt, []).append(r["bdf"])
    return [f"C1 INFO {what} on {at(b)}: {TIMEOUT}" for b in by.pop(TIMEOUT, [])] + [
        f"C1 INFO {what} not checked"
        + ("" if len(bs) == n else " on " + ", ".join(at(b) for b in sorted(bs)))
        + f": {why}"
        for why, bs in by.items()
    ]


def judge_spi(recs, at, ti=None):
    errs = [r for r in recs if r.get("err") or not r.get("table")]
    out, rows, late = errlines("SPI images", errs, len(recs), at, "no table read"), {}, 0
    for r in [r for r in recs if r not in errs]:
        fds, sums = parse_table(bytes.fromhex(r["table"])), dict(r.get("sums", {}))
        if ti and sums and (s := ti.want("C1", r["bdf"], n=0)):
            t = sorted(sums)[0]
            sums[t] = ti.flip_int(s, sums[t])
            ti.applied(s, where=f"image {t}")
        it = {(fd["tag"], "descriptor"): (fd["fd_sum"], fd["fd_crc"]) for fd in fds}
        it.update({(fd["tag"], ""): (sums[fd["tag"]], fd["data_crc"]) for fd in fds if fd["tag"] in sums})
        rows[r["bdf"]] = dict(fds=fds, it=it)
        late += len(r.get("late", []))
    if not rows:
        return out, "SPI images not checked"
    if all(a != b for c in rows.values() for a, b in c["it"].values()):
        out.append(
            "C1 INFO not checked: layout not recognised (SPI boot-fs: every descriptor and image on every chip "
            "mismatched)"
        )
        return out, "SPI images not checked (layout not recognised)"
    for k in sorted({k for c in rows.values() for k in c["it"]}):
        has = {b: c["it"][k] for b, c in rows.items() if k in c["it"]}
        bad = sorted(b for b, (a, s) in has.items() if a != s)
        name = k[0] + (" " + k[1] if k[1] else "")
        if bad and len(bad) == len(has):
            out.append(
                f"C1 INFO SPI image {name}: checksum differs from the stored one on all {len(has)} chips, not judged"
            )
            continue
        out += [f"C1 SPI FAIL {at(b)} image {name}: crc read 0x{has[b][0]:08x} stored 0x{has[b][1]:08x}" for b in bad]
    img = [v for c in rows.values() for k, v in c["it"].items() if not k[1]]
    nd = [v for c in rows.values() for k, v in c["it"].items() if k[1]]
    voted = differ = 0  # the stored image checksums voted across the chips, board-specific tags left out
    for t in sorted({fd["tag"] for c in rows.values() for fd in c["fds"]} - NOVOTE):
        v = {b: fd["data_crc"] for b, c in rows.items() for fd in c["fds"] if fd["tag"] == t}
        m, cnt = majority(list(v.values()))
        if m is None or len(v) < 2:
            continue
        voted, off = voted + 1, [b for b in sorted(v) if v[b] != m]
        differ += len(off)
        out += [f"C1 WARN {at(b)} image {t}: stored crc 0x{v[b]:08x}, {cnt}/{len(v)} chips have 0x{m:08x}" for b in off]
    ts = {b: tuple(fd["tag"] for fd in c["fds"]) for b, c in rows.items()}
    m, cnt = majority(list(ts.values()))
    for b, mine in sorted(ts.items()):
        if m is not None and mine != m:
            out.append(
                f"C1 WARN {at(b)}: boot-fs image list differs from {cnt}/{len(ts)} chips (missing "
                f"{','.join(t for t in m if t not in mine) or '-'}, extra {','.join(t for t in mine if t not in m) or '-'})"
            )
    late = f", {late} not read in time" if late else ""
    sm = f"({sum(a == s for a, s in nd)}/{len(nd)} descriptors, {voted} tags voted, {differ} differ{late})"
    return out, f"SPI images {sum(a == s for a, s in img)}/{len(img)} CRC ok {sm}"


def judge_eth(recs, at):
    ok = [r for r in recs if not r.get("err") and r.get("eth")]
    out = errlines("ETH mailboxes", [r for r in recs if r not in ok], len(recs), at, "no ETH cores")
    cores = [e for r in ok for e in r["eth"]]
    sane = lambda e: 0 < (e[2] >> 16 & 0xFF) < 0xFF and e[3] >> 16 in HB_SIG  # noqa: E731
    names = lambda r, cond: ",".join(f"{e[0]}-{e[1]}" for e in r["eth"] if cond(e))  # noqa: E731
    ver_s = lambda v: f"{v >> 16 & 0xFF}.{v >> 8 & 0xFF}.{v & 0xFF}"  # noqa: E731
    if not cores:
        es = "ETH mailboxes not checked"
    elif not any(sane(e) for e in cores):
        out.append(
            "C1 INFO not checked: layout not recognised (ETH mailbox: no core reads a known version and "
            "heartbeat signature)"
        )
        es = "ETH mailboxes not checked (layout not recognised)"
    else:
        out += [
            f"C1 WARN {at(r['bdf'])}: ETH mailbox not sane on cores {b} (version or heartbeat signature)"
            for r in ok
            if (b := names(r, lambda e: not sane(e)))
        ]
        vs = {r["bdf"]: max(set(vl := [e[2] & 0xFFFFFF for e in r["eth"]]), key=vl.count) for r in ok}
        m, cnt = majority(list(vs.values()))
        out += [
            f"C1 WARN {at(b)}: ETH firmware {ver_s(vs[b])}, {cnt}/{len(vs)} chips have {ver_s(m)}"
            for b in sorted(vs)
            if m is not None and vs[b] != m
        ]
        out += [
            f"C1 WARN {at(r['bdf'])}: ETH firmware differs between cores ({b} not {ver_s(vs[r['bdf']])})"
            for r in ok
            if (b := names(r, lambda e: sane(e) and e[2] & 0xFFFFFF != vs[r["bdf"]]))
        ]
        hb = [e for e in cores if e[5] == PORT_UP]  # heartbeat: only cores whose port is up
        stuck = sum(e[3] == e[4] for e in hb)
        if not hb:
            out.append(f"C1 INFO ETH heartbeat not judged: no port up on any of {len(cores)} cores")
        elif stuck == len(hb):
            out.append(
                f"C1 INFO ETH heartbeat did not advance on any of {len(hb)} cores between two reads, not "
                f"judged (the firmware may not update it)"
            )
        else:
            out += [
                f"C1 WARN {at(r['bdf'])}: ETH heartbeat not advancing on cores {b}"
                for r in ok
                if (b := names(r, lambda e: e[5] == PORT_UP and e[3] == e[4]))
            ]
        idle = len(cores) - len(hb)
        es = f"ETH mailboxes {sum(map(sane, cores))}/{len(cores)} ok (heartbeat {len(hb) - stuck} advancing" + (
            f", {idle} cores with no port up not judged)" if idle else ")"
        )
    tl = [r for r in ok if r.get("tlb")]
    if not tl:
        why = next((r["tlb_err"] for r in ok if r.get("tlb_err")), "no register read")
        return out + [f"C1 INFO TLB not checked: {why}"], es, "TLB not checked"
    win = lambda w: ((v := w[0] | w[1] << 32 | w[2] << 64) >> 43 & 63, v >> 49 & 63, v & (1 << 43) - 1)  # noqa: E731
    hit = {r["bdf"]: (*r["tlb_req"][:2], r["tlb_req"][2] >> 21) in map(win, r["tlb"]) for r in tl}
    if not any(hit.values()):
        out.append("C1 INFO not checked: layout not recognised (TLB: no config register holds the reader's window)")
        return out, es, "TLB not checked (layout not recognised)"
    for r in sorted(tl, key=lambda r: r["bdf"]):
        if not hit[r["bdf"]]:
            x, y, a = r["tlb_req"]
            out.append(
                f"C1 WARN {at(r['bdf'])}: no TLB config register reads back the window requested ({x},{y} 0x{a:x})"
            )
    return out, es, f"TLB {sum(hit.values())}/{len(hit)} match"


def judge(trays, spi_p, eth_p):
    try:
        tm = {lbl(l.split()[0]): " ".join(l.split()[1:]) for l in open(trays) if l.split()}
    except OSError:
        tm = {}
    at = lambda b: f"{lbl(b)} {tm.get(lbl(b), 'tray ?')}"  # noqa: E731
    try:
        ti = importlib.import_module("tray_inject")
        ti = ti if ti.active() else None
    except Exception:
        ti = None
    lines, sums = [], []
    for p, what, fn, dflt in (
        (spi_p, "SPI images", lambda r: judge_spi(r, at, ti), ("SPI images not checked",)),
        (eth_p, "ETH mailboxes", lambda r: judge_eth(r, at), ("ETH mailboxes not checked", "TLB not checked")),
    ):
        recs, why = load(p)
        try:
            o, *s = fn(recs) if recs else ([f"C1 INFO {what} not checked: {why}"], *dflt)
        except Exception as e:
            o, *s = [f"C1 INFO {what} not checked: {_err(e)}"], *dflt
        lines += o
        sums += s
    print("".join(f"{l}\n" for l in lines) + "C1 firmware reads: " + ", ".join(sums))


if __name__ == "__main__":
    a = sys.argv[1:]
    try:
        if a[:1] in (["spi"], ["eth"]):
            (spi if a[0] == "spi" else eth)(a[1])
        elif a[:1] == ["judge"]:
            judge(*a[1:4])
        else:
            sys.exit("usage: fw_reads.py spi|eth <chips.txt> | judge <trays.txt> <spi.jsonl> <eth.jsonl>")
    except Exception as e:  # never a crash: a device step's error line, the judge's INFO line
        print(json.dumps({"err": _err(e)}) if a[:1] != ["judge"] else f"C1 INFO firmware reads not checked: {_err(e)}")
