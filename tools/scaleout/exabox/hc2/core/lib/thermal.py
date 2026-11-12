#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/thermal.py: per-chip and per-GDDR-channel temperatures through every fleet load chunk,
# the chassis hot spots and why AICLK fell below its target. Read-only: it reads each chip's ARC telemetry table (pyluwen
# get_telemetry and plain axi_read32 of the same table), never writes a chip, adds no load and changes no clock or limit.
#   thermal.py sample <out|-> <period s> <stop file> <skip file> <iso file> <hung dir> <hang dir> [<bdf> ...]
#       one line per chip per period into <out> until <stop file> or <hang dir> (the load's op-timeout snapshot) exists:
#       <epoch> <bdf> asic=<C> aiclk=<MHz> fmax=<MHz> pw=<W> tdp_lim=<W> tdc=<A> tdc_lim=<A> thm_lim=<C> vcore=<mV>
#         max_temp=<C> temp=<c0>,..,<c7> corr_rd=.. corr_wr=.. uncorr=0x.. train=<t0>,<t1> en=0x<mask> arb=<MHz>/<n> ppm=<a>/<r>
#       The <bdf>s, the lines of <skip file> and <iso file> (the MCE guard's cut chips) and every <hung dir>/*/hung_chips.txt
#       (heavy-log snapshots), re-read each round, are never opened; a chip whose config space reads ffffffff stops the
#       whole sampler (a loss: no chip is read again), as does a read error on any chip (that chip is not retried).
#   thermal.py chunk <thermal.txt> <label>    the chunk's AICLK throttle lines (plain python, no pyluwen)
#   thermal.py summary <thermal.txt> ...      the run's per-chip and per-channel peaks and the hot-spot ranking, and the
#       sample interval with what it cannot see (a ~5 s read misses a 50 ms burst)
# Fields (firmware 19.x telemetry, tt-system-firmware lib/tenstorrent/bh_arc/telemetry.h):
#   asic: ASIC_TEMPERATURE (16.16 fixed point); pw/tdc: TDP (W) and TDC (A) now; fmax: AICLK_LIMIT_MAX (tag 63);
#   temp/max_temp/corr/uncorr: as lib/gddr_telem.py (GDDR_x_y_TEMP: [7:0] x bottom, [15:8] x top, [23:16] y bottom,
#   [31:24] y top die, tt-smi's decode_gddr_pair_temps); train: GDDR_TRAIN_TEMP0/1; en: ENABLED_GDDR;
#   arb: AICLK_ARB_MAX (tag 66): [15:0] the lowest max-arbiter clock, [31:16] which arbiter set it (enum aiclk_arb_max);
#   ppm: AICLK_PPM_INFO (tag 69): [15:0] the arbiter that decided the clock, [31:16] the reason (min_arb, max_arb, ...).
#   A field the firmware or pyluwen does not give is '-'; then the throttle cause is inferred from the limits.
import os, sys, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gddr_telem import tray, decode as gd_decode, devices, config_dead, short_bdf, N   # noqa: E402

ARB = ["fmax", "tdp", "fast_tdc", "tdc", "thm", "board_power", "voltage", "gddr_thm", "doppler_slow", "doppler_critical",
       "host_fmax"]   # enum aiclk_arb_max, the same in firmware v19.12.0 and v19.17.1
REASON = ["min_arb", "max_arb", "fmin", "sweep", "forced"]   # aiclk_ppm.c's reason for the clock it picked
TAGS = {63: "fmax", 66: "arb", 69: "ppm"}
SCRATCH_TABLE, SCRATCH_DATA = "arc_ss.reset_unit.SCRATCH_RAM[13]", "arc_ss.reset_unit.SCRATCH_RAM[12]"   # as luwen's BH
BELOW = 0.97   # AICLK under 97% of its target counts as below it (fleet/load_line.py's THROTTLE)
TOP = 5        # hot spots listed


def tag_addrs(c):
    """The data address of each TAGS tag in the chip's telemetry table ({u16 tag, u16 offset} entries after version, count)."""
    tab = c.axi_read32(c.axi_translate(SCRATCH_TABLE).addr)
    dat = c.axi_read32(c.axi_translate(SCRATCH_DATA).addr)
    n = c.axi_read32(tab + 4)
    out = {}
    for i in range(min(n, 256)):
        e = c.axi_read32(tab + 8 + 4 * i)
        if e & 0xFFFF in TAGS:
            out[e & 0xFFFF] = dat + 4 * (e >> 16)
    return out


def lo(v):   # tdp and tdc carry the value in [15:0]
    return v & 0xFFFF if isinstance(v, int) else v


def fields(t, raw):
    """A pyluwen Telemetry plus the raw tags {63, 66, 69: value} -> the sample's fields; '-' where missing."""
    def get(name):
        v = getattr(t, name, None)
        return v if isinstance(v, int) else None
    def s(v):
        return "-" if v is None else str(v)
    a = get("asic_temperature")
    f = {"asic": "-" if a is None else f"{a / 65536 if a > 1000 else a:.1f}", "aiclk": s(get("aiclk")),
         "fmax": s(raw[63] & 0xFFFF if 63 in raw else get("aiclk_limit_max")), "pw": s(lo(get("tdp"))),
         "tdp_lim": s(get("tdp_limit_max")), "tdc": s(lo(get("tdc"))), "tdc_lim": s(get("tdc_limit_max")),
         "thm_lim": s(get("thm_limit_throttle")), "vcore": s(get("vcore"))}
    f.update(gd_decode(t))
    t0, t1, en = get("gddr_train_temp0"), get("gddr_train_temp1"), get("enabled_gddr")
    f["train"] = f"{s(t0)},{s(t1)}"
    f["en"] = "-" if en is None else f"0x{en:02x}"
    f["arb"] = f"{raw[66] & 0xFFFF}/{raw[66] >> 16}" if 66 in raw else "-"
    f["ppm"] = f"{raw[69] & 0xFFFF}/{raw[69] >> 16}" if 69 in raw else "-"
    return f


ORDER = ["asic", "aiclk", "fmax", "pw", "tdp_lim", "tdc", "tdc_lim", "thm_lim", "vcore", "max_temp", "temp", "corr_rd",
         "corr_wr", "uncorr", "train", "en", "arb", "ppm"]


def lines_of(path):
    try:
        with open(path) as fh:
            return {short_bdf(w if w.count(":") == 2 else "0000:" + w) for w in fh.read().split() if ":" in w}
    except OSError:
        return set()


def gone(bdf):   # lost: config reads ffffffff, or the device left sysfs (PCI removed)
    sysfs = os.environ.get("TRAY_GDDR_SYS", "/sys")
    return config_dead(bdf) or not os.path.exists(os.path.join(sysfs, "bus", "pci", "devices", "0000:" + bdf, "config"))


def sample(out, period, stop, skipf, isof, hung, hang, extra=()):
    extra = {short_bdf(b if b.count(":") == 2 else "0000:" + b) for b in extra if b}
    fo = open(1, "w", buffering=1, closefd=False) if out == "-" else open(out, "a", buffering=1)   # '-': stdout (th_start's redirect, root-owned file)
    parent = int(os.environ.get("TH_PARENT") or 0)
    def done():   # the stop file, the fleet hang dir, or run.sh gone (killed: no th_stop ever comes)
        if os.path.exists(stop) or (hang and os.path.exists(hang)):
            return True
        try:
            parent and os.kill(parent, 0)
        except ProcessLookupError:
            return True
        except PermissionError:   # alive, owned by root (sudo -u)
            pass
        return False
    try:
        import pyluwen
    except Exception as e:
        fo.write(f"# unavailable no pyluwen: {e}\n")
        return
    devs = devices()
    if devs is None:   # which index is which chip is unknown: a skip list cannot be honoured, open none
        fo.write("# unavailable no sysfs map of /dev/tenstorrent to PCI addresses: no chip opened\n")
        return
    chips, bad = {}, set()
    while not done():
        skip = extra | lines_of(skipf) | lines_of(isof) | bad
        for h in glob.glob(os.path.join(hung, "*", "hung_chips.txt")) if hung else []:
            skip |= lines_of(h)
        t0 = time.time()
        for i, bdf in devs:
            if done():
                break
            if bdf in skip:
                chips.pop(bdf, None)
                continue
            if gone(bdf):
                fo.write(f"# stopped {int(time.time())}: {bdf} config space reads ffffffff or is gone (lost): no chip read again\n")
                return
            try:
                if bdf not in chips:
                    c = pyluwen.PciChip(i)
                    try:
                        tags = tag_addrs(c)
                    except Exception:   # an older pyluwen without axi_translate: limits-inferred cause only
                        tags = {}
                    chips[bdf] = (c, tags)
                c, tags = chips[bdf]
                t = c.get_telemetry()
                raw = {k: c.axi_read32(a) for k, a in tags.items()}
                f = fields(t, raw)
                fo.write(f"{int(time.time())} {bdf} " + " ".join(f"{k}={f[k]}" for k in ORDER) + "\n")
                fo.flush()
            except Exception as e:
                bad.add(bdf)
                fo.write(f"# {int(time.time())} {bdf} unavailable {str(e).splitlines()[0][:120] if str(e) else type(e).__name__}: not read again\n")
        while not done() and time.time() < t0 + period:
            time.sleep(0.2)


def parse(path):
    """thermal.txt -> [(epoch, bdf, {field: str})]; comment and bad lines dropped."""
    rows = []
    try:
        fh = open(path)
    except OSError:
        return rows
    with fh:
        for l in fh:
            w = l.split()
            if len(w) < 3 or l.startswith("#") or not w[0].isdigit():
                continue
            rows.append((int(w[0]), w[1], dict(x.split("=", 1) for x in w[2:] if "=" in x)))
    return rows


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def nums(s):
    return [num(x) for x in (s or "").split(",")]


def classify(f):
    """One sample -> None (AICLK at its target), 'idle' (below it with no busy request: the min arbiter, no throttle), or
    (cause, source): source 'firmware' from AICLK_ARB_MAX / AICLK_PPM_INFO, 'inferred' from the limits."""
    ck, fm = num(f.get("aiclk")), num(f.get("fmax"))
    if ck is None or not fm or ck >= BELOW * fm:
        return None
    arb, ppm = f.get("arb", "-"), f.get("ppm", "-")
    if arb != "-" and ppm != "-":
        amhz, aid = (int(x) for x in arb.split("/"))
        pa, pr = (int(x) for x in ppm.split("/"))
        if amhz < BELOW * fm:   # a max arbiter holds the clock below fmax: that arbiter is the cause
            return (ARB[aid] if aid < len(ARB) else f"arbiter {aid}", "firmware")
        if pr == 0:
            return "idle"
        return (REASON[pr] if pr < len(REASON) else f"reason {pr}", "firmware")
    a, tl, pw, pl, tc, cl = (num(f.get(k)) for k in ("asic", "thm_lim", "pw", "tdp_lim", "tdc", "tdc_lim"))
    if a is not None and tl and a >= tl - 1:
        return ("thm", "inferred")
    if pw is not None and pl and pw >= BELOW * pl:
        return ("tdp", "inferred")
    if tc is not None and cl and tc >= BELOW * cl:
        return ("tdc", "inferred")
    return "idle"   # no limit reached: a gap between loads or a host-side wait, not a throttle


def chunk(path, label):
    by = {}
    for _, b, f in parse(path):
        by.setdefault(b, []).append(f)
    if not by:
        print(f"THERMAL THROTTLE {label}: no samples")
        return
    held, idle, src = 0, 0, set()
    for b in sorted(by):
        fs = by[b]
        cl = [classify(f) for f in fs]
        thr = [c for c in cl if isinstance(c, tuple)]
        if not thr:
            held += 1
            idle += any(c == "idle" for c in cl)
            continue
        causes = {}
        for c, s in thr:
            causes[c] = causes.get(c, 0) + 1
            src.add(s)
        fm = max((num(f.get("fmax")) or 0) for f in fs)
        mn = min(num(f.get("aiclk")) for f, c in zip(fs, cl) if isinstance(c, tuple))
        why = ", ".join(f"{c} {n}" for c, n in sorted(causes.items(), key=lambda x: -x[1]))
        print(f"THERMAL THROTTLE {b} (tray {tray(b)}) {label}: AICLK below its {fm:.0f} MHz target in {len(thr)}/{len(fs)} "
              f"samples, min {mn:.0f} MHz; cause {why} ({'/'.join(sorted({s for _, s in thr}))})")
    srcs = {"firmware" if all(f.get("arb", "-") != "-" for f in fs) else "inferred" for fs in by.values()}
    print(f"THERMAL THROTTLE {label}: {held}/{len(by)} chips never throttled"
          + (f" ({idle} below target only while not busy, the min arbiter)" if idle else "")
          + f"; cause from {'/'.join(sorted(srcs))}")


def summary(paths):
    pk, gk, pw, ck, n, tr, ep = {}, {}, {}, {}, {}, {}, []
    dur, gaps = 0, []
    for p in paths:   # each file is one load chunk: the load time is the sum of their spans
        pe, last = [], {}
        for e, b, f in parse(p):
            ep.append(e)
            pe.append(e)
            if b in last:
                gaps.append(e - last[b])
            last[b] = e
            n[b] = n.get(b, 0) + 1
            a = num(f.get("asic"))
            if a is not None and a > pk.get(b, -1e9):
                pk[b] = a
            g = gk.setdefault(b, [None] * N)
            for i, v in enumerate(nums(f.get("temp"))[:N]):
                if v is not None and (g[i] is None or v > g[i]):
                    g[i] = v
            w = num(f.get("pw"))
            if w is not None and w > pw.get(b, -1):
                pw[b] = w
            k = num(f.get("aiclk"))
            if k is not None and (b not in ck or k < ck[b]):
                ck[b] = k
            t = nums(f.get("train"))
            if any(x for x in t):   # 0/0: firmware without training temps
                tr[b] = t
        dur += max(pe) - min(pe) if pe else 0
    if not n:
        print("THERMAL: no samples (no load chunk ran, no pyluwen, or every chip skipped)")
        return
    def c(v):
        return "-" if v is None else f"{v:.0f}"
    for b in sorted(n):
        g = gk.get(b, [None] * N)
        gm = max((v for v in g if v is not None), default=None)
        gi = g.index(gm) if gm is not None else None
        print(f"THERMAL CHIP {b} (tray {tray(b)}): ASIC peak {pk[b]:.1f} C, GDDR peak {c(gm)} C"
              + (f" (channel {gi})" if gi is not None else "") + f", power peak {c(pw.get(b))} W, AICLK min "
              f"{c(ck.get(b))} MHz, {n[b]} samples" if b in pk else
              f"THERMAL CHIP {b} (tray {tray(b)}): no ASIC temperature read, {n[b]} samples")
        t = tr.get(b, [])
        print(f"THERMAL GDDR {b} (tray {tray(b)}): channel peaks " + " ".join(f"c{i}={c(v)}" for i, v in enumerate(g))
              + " C" + (f"; training temps {'/'.join(c(x) for x in t)} C" if t else ""))
    hot = sorted(pk.items(), key=lambda x: -x[1])[:TOP]
    print("THERMAL HOT ASIC: " + "; ".join(f"tray {tray(b)} {b} {v:.1f} C" for b, v in hot))
    gl = sorted(((v, b, i) for b, g in gk.items() for i, v in enumerate(g) if v is not None), key=lambda x: -x[0])[:TOP]
    print("THERMAL HOT GDDR: " + ("; ".join(f"tray {tray(b)} {b} channel {i} {v:.0f} C" for v, b, i in gl) or "no GDDR temperature read"))
    if gaps:
        g = sorted(gaps)[len(gaps) // 2]
        print(f"THERMAL RATE: each chip read every {g} s (median): a burst shorter than that (e.g. 50 ms of power, "
              "current or heat) falls between reads and is not seen; peaks here are of the reads, not of the load")
    trays = {}
    for b, v in pk.items():
        trays[tray(b)] = max(v, trays.get(tray(b), v))
    print(f"THERMAL: {len(n)} chips sampled {len(ep)} times over {f'{dur // 60} min' if dur >= 120 else f'{dur} s'} of load; hottest ASIC per tray "
          + ", ".join(f"tray {t} {v:.1f} C" for t, v in sorted(trays.items())))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 8 and a[0] == "sample":
        sample(a[1], float(a[2]), a[3], a[4], a[5], a[6], a[7], a[8:])
    elif len(a) == 3 and a[0] == "chunk":
        chunk(a[1], a[2])
    elif len(a) >= 2 and a[0] == "summary":
        summary(a[1:])
    else:
        sys.exit(__doc__ or "usage: thermal.py sample|chunk|summary ...")
