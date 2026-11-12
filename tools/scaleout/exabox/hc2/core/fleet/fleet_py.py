#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""fleet_py.py: helpers of run.sh --mode (lib/fleet.sh). Only 'disc' opens a device (run them with the tray-test venv's
python); the rest read files or sysfs and run with any python3.
  disc                                  UMD topology discovery (read-only): JSON chips, bdfs, active, links
  count <disc.json>                     C3: "chips=N eth_local=N qsfp_up=N" (active ETH channels to a local chip /
                                        active channels that lead off-host)
  pairs <disc.json> <map.txt> [skip]    C5: local chip pairs in edge-coloured rounds (no chip twice in a round):
                                        idx round chipA chipB bdfA bdfB devA devB chansA chansB; skip = comma list of
                                        UMD chip ids whose pairs are left out. map.txt: "<bdf> <devnode>" (run.sh's rk_map)
  aer <sysfs> <bdf>...                  C1: per endpoint and its root port: bdf role cor nonfatal fatal rxerr badtlp
  aerdelta <pre> <post> <rate_warn> <hours> [min] [host] [when] [fail]
                                        C1 run window: "AER WARN" for a correctable rise >= rate_warn per hour of the run (and
                                        >= min events, default 2); "AER FAIL" for any uncorrectable (nonfatal/fatal) rise,
                                        and for one root port's correctable rise >= fail with every other port +2 or less.
  aerhang <rows> <stall epoch> [host]   a stall naming no single chip: AER FAIL for the one port whose AER poll rows rose.
                                        journal (C1 history) also prints "AER FAIL" for a chip whose correctable rate is an
                                        outlier, >= 10x the box median and >= 20/h; not WARN-only.
                                        Each line names the chip, tray, root port and host
  hostlink <snap> <design> <host> [pre snap]
                                        host link (lib/link_check.sh): "PCIE FAIL" per chip off PCIe (not on the bus, config
                                        space ffff or unread), with LnkSta unreadable (not trained), or slower/narrower than
                                        the design at its slot (chip or root port side), or (with a pre snap) whose link went
                                        down since; "PCIE WARN" above the design. Lines name chip, tray, root port and host.
                                        design: "<GT/s>x<width> [<slot>:<GT/s>x<width>...]", slot = bus & 0x3f; or "auto":
                                        per slot the most common link over the host's own answering chips (pre snap if given)
  journal <boot> <7d> <aer.tsv> <boot_warn> <7d_warn> [<repair epoch> <repair label>]
                                        C1 history: kernel AER events per chip/root port in `journalctl _TRANSPORT=kernel`
                                        of the current boot and of 7 days (native "PCIe Bus Error" lines and GHES records):
                                        correctable -> WARN at the levels, uncorrectable -> WARN; MCE/GHES processor error
                                        lines -> WARN. History is from before the run, so it never fails it;
                                        each line ends with the last event's time (PDT)
  aerrun <journal> <aer.tsv> <t0> <t1> <rate_warn> [min] [findings]
                                        C1 run window: kernel AER events per chip and root port stamped in
                                        [t0, t1] (the run's own start and end): "AER FAIL" for any uncorrectable one or a
                                        correctable rate outlier (>= 10x the box median and >= 20/h), "AER WARN"
                                        for correctable >= rate_warn/h and >= min (default 2). A line that a findings file
                                        already holds for the same chip, role and grade from aerdelta is not repeated
  sel <sel elist> <now> <buscor_warn> <frb2_warn> [<repair epoch> <repair label>]
                                        C1 SEL, 7 days: Bus Correctable per host, FRB3 per tray, FRB2 per ASIC (at frb2_warn),
                                        critical threshold crossings: all WARN (before the run). No Device Absent rule (every
                                        cycle logs it). SEL times are the BMC's, read in the host's zone (UTC on the quads)
                                        With a repair time (run.sh --since, the host's last repair), journal and sel judge only
                                        events at or after it; each earlier event they would judge is one
                                        "HIST <tray N|host>\t<count>\t<what>" line instead
  hist1 <history file>                  C1 pre: journal, sel and chain lines on stdin -> at most one WARN
                                        line per check (AER, SEL, FRB, DMESG), "<K> WARN host: <items>; before the run:
                                        pre-existing, not from this run", never a FAIL; every input line goes to the file.
                                        HIST lines and other checks (SDR: state now, not history) pass through unchanged
  histinfo <repair label> <file>...     C1: the HIST lines -> one "C1 INFO tray N: M events before repair <label>, ignored
                                        (...)" line per tray (then host), BDF named for chip events; report only
  selnew <pre sel> <post sel> <out>     C1: new SEL rows during the run: FRB3 -> one "SEL WARN tray N (<BDF range>)"
                                        (an SEL row alone is no power loss; a lost chip, a newly set HSC latch
                                        or the CPLD fails the tray on its own line), other FRB/presence/power/voltage rows
                                        -> one "SEL WARN", naming tray N
  sdr <sdr elist>                       C1: "SDR WARN" per sensor whose status is not ok/ns
  ethdown <tt-smi snapshot> <label>     C3: "ETH WARN" per chip with ENABLED_ETH ports not up in ETH_LIVE_STATUS's low 16 bits
  ethlinks <disc.json> <label> <eth_map.txt> [peer state] [--host H] [--since EPOCH]
                                        C3: "ETH FAIL" per chip with a local ETH port (eth_map.txt L) not
                                        linked; QSFP ports (Q) not up: FAIL only when every cable peer in [peer state]
                                        ("<HOST> up|down|busy|unknown [why]" lines) is up and idle, else a "ETH WARN" that
                                        names the peer and why (partner down: "partner host <H> down: ... not judged").
                                        --host H: each port is judged on its own peer from fleet/eth_peers.txt
                                        (next to eth_map.txt), named with its tray and BDF; a Slurm cluster host reads
                                        eth_map_exabox.txt instead. --since EPOCH (this host's own
                                        C2/C9 reset): a peer whose "reset=<epoch>" (last chip reset or boot) is older than
                                        EPOCH - FLEET_PEER_RESET_WINDOW (300 s), or missing, makes it a WARN "peer not
                                        reset" (a QSFP link stays down until both ends are reset; pre: EPOCH
                                        is this host's last chip reset before the run). A "# why"
                                        line in [peer state] (a rejected state file) is quoted when no peer is read.
                                        --smi SNAPSHOT --named FINDINGS (after a reset only): a local port is
                                        also down when tt-smi's ETH_LIVE_STATUS says so (host in eth_peers.txt only, else
                                        ethboth judges ports 0-7); the FAIL line ends "; still down
                                        after the <label> reset", and ports an earlier label already named are left out.
                                        --pre FILE (C2 only): pre's in-box "local ETH ports" lines, held back at
                                        pre; a port up again is "ETH WARN .. down at pre (to ..); recovered by C2 reset",
                                        one still down is this label's FAIL, one on a chip off PCIe a FAIL "not judged".
  ethback <findings> <stage tsv> <disc.json> <tt-smi snapshot> <label> [eth_map.txt [host]]
                                        C3 after a reset: each earlier "local ETH ports" FAIL line gets
                                        "; <label> reset: back up | still down | ports [..] back up, [..] still down" in
                                        findings and the stage tsv; the changed lines on stdout
  ethdiff <before disc.json> <after disc.json>
                                        C3 after C2: the local ETH links up before and not after, one line per
                                        chip pair: "<bdf> (tray N) - <bdf> (tray M): K link(s) down (channels a,b - c,d)";
                                        nothing when none went down (e.g. C2 left eth_local 248 of 256)
  ethboth <pre snapshot> <post snapshot> <post label> [--host H] [--held C2 snapshot]
                                        C3 after the end reset: "ETH FAIL" per chip whose enabled ETH ports are down in both
                                        (the reset did not bring them up; e.g. chip 47:00.0 ports 10-13); a
                                        host not in eth_peers.txt (standalone): ports 10-13 leave the box, WARN
  ethboth ... --host H [--map <disc.json>...]
                                        a host not in eth_peers.txt whose link map (offhost) shows inter-host
                                        links judges ports 10-13 from its own link map: FAIL, far host not inspected
                                        (no neighbour check); --map given but no readable link map:
                                        WARN "not judged: no link map"
  attr <map.txt> <log|json>...          a failed step's chips: "<kmd> <bdf> <why>" per chip its error names
                                        ("Device N" / "chip_id N" are tt-metal/UMD chip ids, mapped to the kmd index by UMD's
                                        "Opening local chip ids/PCIe ids" line, identity without it; "/dev/tenstorrent/N" and
                                        PCI addresses are direct; "device <asic id>", UMD's unique id, through map.txt's optional
                                        3rd column, the chip's tt_asic_id). Exit 1 when no chip is named. map.txt: run.sh's rk_map
  ethcore <map.txt> <tt-smi snapshot> <log>...
                                        C4: "ETH FAIL" per chip and ETH core named by tt-metal's "Device N: Timed out while waiting
                                        for active ethernet core X-Y" (llrt.cpp, at mesh open): BDF, tray, tt-smi index, core,
                                        channel (BH translated X - 20) and tt-smi port (the channel-th ENABLED_ETH bit, as UMD
                                        harvests), the link status tt-metal printed for that core and the port in the snapshot
                                        (else "unknown"). Exit 1 when the logs name no such core
  ethfec <map.txt> <pre dir> <post dir> [stage]
                                        off-box ETH FEC over a run: uncorrected FEC or retrains on 3/4 or more
                                        of the up off-box ch8/9 ends (ASICs 1,2,5,6; QSFP-DD ports 7,8,11,12) is one
                                        "ETH INFO far end reset during run" line; one or two ends over 10x their siblings'
                                        median (and >= 100) are "ETH FEC WARN tray T / QSFP-DD port P / ASIC A / BDF". Never FAIL.
                                        A snapshot with no counters: "ETH FEC: not judged (why)"
  ethsnap <eth_counters.jsonl>          heavy_logs index check: "FLAG ETH bad FCS ..." or "ok (...)"
  hangpin <map.txt> <pre dir> <post dir> [telem csv]
                                        C4 try hang: "C4 HANG <bdf> (tray N) tt-smi K" per chip whose ETH links sent
                                        under half the median frames between two heavy-logs reads (<dir>/eth_counters.jsonl,
                                        when.txt); none, or over a quarter of the chips: one "C4 HANG not pinpointed" line with
                                        the chips at load (>= 80 W) in the telemetry and those never seen at load (exit 1)
  telem <c8_telemetry> <temp_c> <power_w> <curr_a> <vcore_low_v>
                                        C8: "TELEMETRY WARN" per chip from its valid samples; 0xFFFF.. sentinels and values out
                                        of range (unread telemetry on some chips) are counted on stderr only
  thermal <c8_telemetry> <tlimits> <phases> <stages.tsv> <fault_times.tsv> <fallback_c> <fallback source>
                                        C8 overheating: "THERMAL FAIL <bdf> (tray N)" per chip whose
                                        peak ASIC temp in a stage reached its thermal limit (hwmon temp1_max, else the fallback),
                                        from samples before that chip's first fault only; summary with ignored counts on stderr
  offhost <want> <disc.json>...         link map of a host outside quads 1-4: its own ETH link map from the
                                        run's discoveries, read-only. "OFFHOST <bdf> (tray N) e<ch>: ..." per ETH channel
                                        whose far end is not a chip of this host (an active channel with no local link, a
                                        link to a chip with no PCI address here, or such a chip itself), then a summary.
                                        Exit 0 none, 1 any, 2 a map that cannot be read (a file missing or not JSON, no
                                        chips, or fewer than <want> chips across all files; want 0 = no count check)
  linkdrop <pre links> <post links>     C1: "PCIE FAIL" per chip whose link speed or width went down during the run
"""
import calendar, contextlib, csv, io, json, os, re, sys, time


def last_json(path):
    return json.loads([l for l in open(path) if l.startswith("{")][-1])


def tray_name(sensor):  # BMC sensor name -> tray wording: "UBB2_ASIC3" -> "tray 3 ASIC3", "UBB1_TEMP" -> "tray 2 TEMP"
    return re.sub(r"UBB", "tray", re.sub(r"UBB(\d)_?", lambda m: f"tray {int(m.group(1)) + 1} ", sensor)).strip()


def ubb(bdf):
    bus = int(bdf.split(":")[-2], 16) & 0xF0
    return {0x00: "tray 1", 0x40: "tray 2", 0xC0: "tray 3", 0x80: "tray 4"}.get(bus, "BDF not in map")


TRAY_BDFS = {"tray 1": "01:00.0-08:00.0", "tray 2": "41:00.0-48:00.0", "tray 3": "c1:00.0-c8:00.0", "tray 4": "81:00.0-88:00.0"}   # as ubb()


def tray_of(sensor):  # a SEL sensor name -> "tray N" (UBBn is tray n+1), else "host"
    m = re.search(r"UBB(\d)", sensor)
    return f"tray {int(m.group(1)) + 1}" if m else "host"


def pdt(t):  # epoch -> "10-22 20:32 PDT" (the project's zone; the hosts run UTC)
    try:
        from zoneinfo import ZoneInfo
        import datetime
        return datetime.datetime.fromtimestamp(t, ZoneInfo("America/Los_Angeles")).strftime("%m-%d %H:%M %Z")
    except Exception:
        return time.strftime("%m-%d %H:%M UTC", time.gmtime(t))


def count(d):
    loc = sum(len(v) for v in d["links"].values())
    off = sum(len(set(ch) - {int(c) for c in d["links"].get(str(k), {})}) for k, ch in d["active"].items())
    return f"chips={len(d['chips'])} eth_local={loc} qsfp_up={off}"


def ethdiff(a, b):
    def edges(d):
        return {tuple(sorted(((int(c), int(ch)), (int(rc), int(rch))))) for c, v in d["links"].items()
                for ch, (rc, rch) in v.items()}
    p = {}
    for (x, y) in sorted(edges(a) - edges(b)):
        e = p.setdefault((x[0], y[0]), [[], []]); e[0].append(x[1]); e[1].append(y[1])
    bdf = lambda c: (a["bdfs"].get(str(c)) or b["bdfs"].get(str(c)) or f"chip {c}").replace("0000:", "")
    for (x, y), (cx, cy) in sorted(p.items(), key=lambda i: (bdf(i[0][0]), bdf(i[0][1]))):
        if bdf(y) < bdf(x):
            x, y, cx, cy = y, x, cy, cx
        print(f"{bdf(x)} ({ubb(bdf(x))}) - {bdf(y)} ({ubb(bdf(y))}): {len(cx)} link{'s' if len(cx) > 1 else ''} down "
              f"(channels {','.join(map(str, cx))} - {','.join(map(str, cy))})")


def pairs(d, bdf2dev, skip):
    p = {}
    for c, v in d["links"].items():
        for ch, (rc, rch) in v.items():
            c, ch, rc, rch = int(c), int(ch), int(rc), int(rch)
            a, b = sorted((c, rc))
            e = p.setdefault((a, b), [set(), set()])
            e[0].add(ch if c == a else rch); e[1].add(rch if c == a else ch)
    # greedy edge colouring, most-connected chips first: a grid (degree <= 4) needs 4-5 rounds
    deg = {}
    for a, b in p:
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    rounds = []
    for a, b in sorted(p, key=lambda e: (-(deg[e[0]] + deg[e[1]]), e)):
        if a in skip or b in skip:
            continue
        for ri, r in enumerate(rounds):
            if a not in r and b not in r:
                r.update((a, b)); p[(a, b)].append(ri); break
        else:
            rounds.append({a, b}); p[(a, b)].append(len(rounds) - 1)
    print("idx\tround\tchipA\tchipB\tbdfA\tbdfB\tdevA\tdevB\tchansA\tchansB")
    for i, ((a, b), e) in enumerate(sorted(p.items())):
        if len(e) < 3:
            continue
        ba, bb = d["bdfs"][str(a)], d["bdfs"][str(b)]
        print(f"{i}\t{e[2]}\t{a}\t{b}\t{ba}\t{bb}\t{bdf2dev.get(ba, '?')}\t{bdf2dev.get(bb, '?')}\t"
              f"{','.join(map(str, sorted(e[0])))}\t{','.join(map(str, sorted(e[1])))}")


def aer_file(path):
    v = {}
    try:
        for l in open(path):
            k, _, n = l.strip().rpartition(" ")
            v[k] = int(n)
    except (OSError, ValueError):
        return None
    return v


def aer(sysfs, bdfs):
    print("bdf\trole\tof\tcor\tnonfatal\tfatal\trxerr\tbadtlp")
    for b in bdfs:
        ep = os.path.join(sysfs, "bus/pci/devices", b)
        rp = os.path.basename(os.path.dirname(os.path.realpath(ep)))
        for role, dev, name in (("ep", ep, b), ("rp", os.path.dirname(os.path.realpath(ep)), rp)):
            c = aer_file(os.path.join(dev, "aer_dev_correctable"))
            nf = aer_file(os.path.join(dev, "aer_dev_nonfatal")) or {}
            f = aer_file(os.path.join(dev, "aer_dev_fatal")) or {}
            if c is None:
                print(f"{name}\t{role}\t{b}\t-\t-\t-\t-\t-"); continue
            print(f"{name}\t{role}\t{b}\t{c.get('TOTAL_ERR_COR', 0)}\t{nf.get('TOTAL_ERR_NONFATAL', 0)}\t"
                  f"{f.get('TOTAL_ERR_FATAL', 0)}\t{c.get('RxErr', 0)}\t{c.get('BadTLP', 0)}")


def one_port(rise, fail, other):
    """{port: corrected rise} -> the one port that rose >= fail while every other port rose <= other, else None (seen: one
    port >= 20 in a stage in 11 of 49 runs, healthy boxes' ports never over 10 in 169)."""
    hi = [p for p, n in rise.items() if n >= fail]
    if fail <= 0 or len(hi) != 1 or any(n > other for p, n in rise.items() if p != hi[0]):
        return None
    return hi[0]


def _fail(v):
    """FLEET_AER_STAGE_FAIL as an int; a bad value ('20x', '2.5') is the default 20, never a crash that loses every
    stage AER finding."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 20


def aerdelta(pre, post, rate, hours, least=2, host="?", when="during the run", fail=0):
    """fail (FLEET_AER_STAGE_FAIL, 0 = off): one root port's corrected rise (its own count or its chip's, the larger) >=
    fail while every other port rose <= 2 is an AER FAIL naming the chip, in place of that port's WARN."""
    def rows(p):
        out = {}
        for l in open(p).read().splitlines()[1:]:
            f = l.split("\t")
            if len(f) == 8 and f[3] != "-":
                out[(f[0], f[1])] = (f[2], [int(x) for x in f[3:]])
        return out
    a, b = rows(pre), rows(post)
    rps = {of: k[0] for k, (of, _) in b.items() if k[1] == "rp"}
    rise = {}
    for k, (of, v) in b.items():
        if k in a:
            p = rps.get(of, k[0])
            rise[p] = max(rise.get(p, 0), v[0] - a[k][1][0])
    hot = one_port(rise, fail, 2)
    hk = [k for k, (of, _) in b.items() if k in a and rps.get(of, k[0]) == hot] if hot else []
    hk = min(hk, key=lambda k: (k[1] != "rp", k)) if hk else None   # the FAIL goes on the rp row, else the chip's row
    for k, (of, v) in sorted(b.items()):
        if k not in a:
            continue
        d = [y - x for x, y in zip(a[k][1], v)]
        role = f"endpoint, root port {rps.get(of, '?')}" if k[1] == "ep" else f"root port {k[0]}"
        who = f"{of[5:]} ({ubb(of)}) {role}, host {host}"
        if d[1] > 0 or d[2] > 0:
            print(f"AER FAIL {who}: uncorrectable nonfatal +{d[1]} fatal +{d[2]} {when}")
        if hot and rps.get(of, k[0]) == hot:
            if k == hk:   # an unreadable rp row ('-') still gets its FAIL, on the chip's row
                print(f"AER FAIL {of[5:]} ({ubb(of)}) root port {rps.get(of, '?')}, host {host}: correctable +{rise[hot]} (RxErr +{d[3]}, BadTLP +{d[4]}) {when} "
                      f"(every other root port +2 or less)")
            continue
        per_h = d[0] / max(hours, 1 / 60)
        if d[0] >= least and per_h >= rate:
            print(f"AER WARN {who}: correctable +{d[0]} (RxErr +{d[3]}, BadTLP +{d[4]}) {when}, "
                  f"{per_h:.1f}/h (warn at {rate:g}/h)")


def aerpre(pre, host="?"):
    """At most one WARN for uncorrectable AER counts already set before the run: names each chip by BDF and
    tray 1-4; worded so fl_find keeps it out of the verdict (it never fails a run: the counts are not from this run)."""
    seen = []
    for l in open(pre).read().splitlines()[1:]:
        f = l.split("\t")
        if len(f) != 8 or f[3] == "-" or (int(f[4]) == 0 and int(f[5]) == 0):
            continue
        role = "endpoint" if f[1] == "ep" else f"root port {f[0]}"
        seen.append(f"{f[2][5:]} ({ubb(f[2])}) {role} nonfatal {f[4]} fatal {f[5]}")
    if seen:
        print(f"AER WARN host {host}: sysfs uncorrectable AER counts at the first load stage: {'; '.join(seen)}; "
              "before the run: pre-existing, not from this run")


BDF = re.compile(r"\b([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\b")


def aer_port_cor(path, chips):
    """A root port's corrected AER count from sysfs: aer_rootport_total_err_cor (every ERR_COR the port took, its own and
    those from below) or TOTAL_ERR_COR of the port's and its chips' aer_dev_correctable, the larger, so a port whose one
    counter stays put while the other moves still counts; None when no file reads."""
    try:
        r = int(open(os.path.join(path, "aer_rootport_total_err_cor")).read().split()[0])
    except (OSError, ValueError, IndexError):
        r = None
    d = None
    for p in [path] + [c for c in chips if c != path]:
        v = aer_file(os.path.join(p, "aer_dev_correctable"))
        if v is not None:
            d = (d or 0) + v.get("TOTAL_ERR_COR", 0)
    return d if r is None else r if d is None else max(r, d)


def aerrate(sysfs, state, dmesg, now, warn, chips, fail=0, host="?"):
    """One sample of the bus-bar/sustained AER poll: per chip's upstream PCIe root port, the corrected AER
    count from sysfs and the corrected AER/GHES events in dmesg (a firmware-first host's GHES records leave sysfs at 0) newer
    than the last sample. chips: "<bdf>=<tray>". One row per port: the rise since the last sample (the larger of the
    two sources) and per_min = rise * 60 / max(s since the last sample, 60); the first sample is the baseline (rise -).
    Read-only: sysfs reads and a dmesg file the caller wrote."""
    try:
        st = json.load(open(state))
    except (OSError, ValueError):
        st = {}
    ports = st.get("ports") or {}
    if not ports:   # the port map from the first sample, kept: a chip that left PCIe still has its root port read
        for a in chips:
            b, _, t = a.partition("=")
            b = b if b.count(":") == 2 else "0000:" + b
            real = os.path.realpath(os.path.join(sysfs, "bus/pci/devices", b))
            if not os.path.isdir(real):
                continue
            up = os.path.dirname(real)
            port, path = (os.path.basename(up), up) if BDF.fullmatch(os.path.basename(up)) else (b, real)
            e = ports.setdefault(port, {"path": path, "chips": [], "paths": []})
            e["chips"].append([b, t]); e["paths"].append(real)
    lines = []
    try:
        for l in open(dmesg, errors="replace").read().splitlines():
            m = re.match(r"\[\s*(\d+\.\d+)\]\s?(.*)$", l)
            lines.append(f"{m.group(1)} {m.group(2)}" if m else l)
    except OSError:
        lines = None
    t_prev = st.get("dmesg_t")
    ev, last_t = {}, t_prev
    if lines is not None:
        ev = kernel_events(None, t_prev + 1e-7 if t_prev is not None else None, lines)[0]
        ts = [float(l.split(" ", 1)[0]) for l in lines if re.match(r"\d+\.\d+ ", l)]
        last_t = max(ts + ([t_prev] if t_prev is not None else []), default=t_prev)
    first, dt = "t" not in st, now - st.get("t", now)
    out, rise = {}, {}
    for port, e in sorted(ports.items()):
        c = aer_port_cor(e["path"], e["paths"])
        p = (st.get("cor") or {}).get(port)
        sr = c - p if c is not None and p is not None and c >= p else None
        dr = sum(ev.get(b, [0, 0])[0] for b in {port} | {b for b, _ in e["chips"]}) if lines is not None and t_prev is not None else None
        r = None if first else max(x for x in (sr, dr, 0) if x is not None)
        pm = None if r is None else r * 60 / max(dt, 60)
        out[port] = c
        rise[port] = r or 0
        nm = ",".join(f"{b[5:]}:{t}" for b, t in e["chips"])
        f = lambda x: "-" if x is None else (f"{x:.0f}" if isinstance(x, float) else str(x))
        print(f"t={now} port={port[5:]} chips={nm} sysfs={f(c)} sysfs_rise={f(sr)} dmesg_rise={f(dr)} rise={f(r)} "
              f"per_min={f(pm)}{' OVER' if pm is not None and pm > warn else ''}")
    if fail > 0 and not first:   # live, the minute one port's running count this stage crosses fail (others <= 2)
        tot = st.setdefault("sum", {})
        for port in rise:
            tot[port] = tot.get(port, 0) + rise[port]
        hot = None if st.get("live") else one_port(tot, fail, 2)
        if hot:
            st["live"] = hot
            nm = ", ".join(f"{b[5:]} (tray {t})" for b, t in ports[hot]["chips"])
            print(f"AER FAIL {nm} root port {hot}, host {host}: correctable +{tot[hot]} live at minute "
                  f"{round((now - st.get('t0', now)) / 60)} of the stage (every other root port +2 or less)")
    st.setdefault("t0", now)
    st.update(ports=ports, cor=out, t=now, dmesg_t=last_t)
    with open(state + ".tmp", "w") as fh:
        json.dump(st, fh)
    os.replace(state + ".tmp", state)


def aerrate_judge(rows, warn, stage):
    """The AER poll's rows -> one AER WARN per root port whose corrected count rose over warn per minute in any sample,
    naming each chip below it by BDF and tray (one chip logged
    5,230 GHES corrected errors in a run that reported nothing, then dropped)."""
    seen = {}
    try:
        rs = open(rows, errors="replace").read().splitlines()
    except OSError:
        return
    for l in rs:
        kv = dict(x.split("=", 1) for x in l.split() if "=" in x)
        try:
            pm = float(kv["per_min"])
        except (KeyError, ValueError):
            continue
        s = seen.setdefault(kv["port"], {"chips": kv.get("chips", ""), "n": 0, "over": 0, "peak": 0.0, "sum": 0,
                                          "src": set()})
        s["n"] += 1; s["sum"] += int(float(kv.get("rise", 0)))
        if pm > warn:
            s["over"] += 1; s["peak"] = max(s["peak"], pm)
            s["src"].update(k for k in ("sysfs", "dmesg") if kv.get(k + "_rise", "-").isdigit() and int(kv[k + "_rise"]) > 0)
    for port, s in sorted(seen.items()):
        if not s["over"]:
            continue
        cs = ", ".join(f"{b} (tray {t})" for b, _, t in (c.rpartition(":") for c in s["chips"].split(",") if c))
        print(f"AER WARN weak PCIe link {cs}: root port {port} corrected AER peak {s['peak']:.0f}/min (over {warn:g}/min in "
              f"{s['over']} of {s['n']} min, +{s['sum']} in all; {'/'.join(sorted(s['src'])) or 'sysfs'}) during the {stage}")


def aerhang(rows, stall, host="?"):
    """A stall that names no single chip (seen: 9 whole-box stalls where only one root port
    logged corrected AER in each): the AER poll's rows of the stage; when one port rose >= 3 while every other rose <= 1,
    one AER FAIL naming its chips, the stage's count and the count before the stall (rows stamped at or before stall)."""
    tot, pre, chips = {}, {}, {}
    try:
        rs = open(rows, errors="replace").read().splitlines()
    except OSError:
        return
    for l in rs:
        kv = dict(x.split("=", 1) for x in l.split() if "=" in x)
        if not kv.get("rise", "-").isdigit() or "port" not in kv:
            continue
        p, n = kv["port"], int(kv["rise"])
        tot[p] = tot.get(p, 0) + n; chips[p] = kv.get("chips", "")
        if kv.get("t", "").isdigit() and int(kv["t"]) <= stall:
            pre[p] = pre.get(p, 0) + n
    hot = one_port(tot, 3, 1)
    if not hot:
        return
    nm = [(b, t) for b, _, t in (c.rpartition(":") for c in chips[hot].split(",") if c)]
    cs = ", ".join(f"{b} (tray {t})" for b, t in nm)
    o = max((n for p, n in tot.items() if p != hot), default=0)
    print(f"AER FAIL {cs} root port {hot}, host {host}: correctable +{tot[hot]} this stage (+{pre.get(hot, 0)} before the "
          f"stall), other ports {'0' if not o else '+1 or less'}, at a stall that names no single chip: likely "
          f"{' and '.join(b for b, _ in nm)}'s PCIe link")


def kernel_events(path, since=None, lines=None):
    """kernel journal -> ({bdf: [correctable, uncorrectable]}, MCE/processor error lines, the same two for lines stamped
    (-o short-unix) before since, {bdf or "mce": last stamp at or after since}). One event per native "PCIe Bus Error"
    line or per GHES record's "device_id: <bdf>" line; a GHES record's severity comes from its "event severity:" line
    ({N} is the record tag). An unstamped line counts as after since. lines: already read ones instead of path."""
    ev, mce, sev, old, mce_old, last = {}, 0, {}, {}, 0, {}
    if lines is None:
        try:
            lines = open(path, errors="replace").read().splitlines()
        except OSError:
            return ev, mce, old, mce_old, last
    for l in lines:
        try:
            t = float(l.split(" ", 1)[0])
        except ValueError:
            t = None
        pre = since is not None and t is not None and t < since
        e = old if pre else ev
        seen = lambda k: pre or t is None or last.__setitem__(k, max(last.get(k, 0), t))   # newest counted event
        if "[Hardware Error]" in l:
            tag = re.search(r"\{(\d+)\}\[Hardware Error\]", l)
            tag = tag.group(1) if tag else ""
            m = re.search(r"event severity: (\w+)", l)
            if m:
                sev[tag] = m.group(1).lower(); continue
            m = re.search(r"device_id: ([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])", l)
            if m:
                e.setdefault(m.group(1), [0, 0])[sev.get(tag, "corrected") not in ("corrected", "info")] += 1; seen(m.group(1))
            elif re.search(r"general processor|[Pp]rocessor [Ee]rror|memory error|mce", l):
                mce_old += pre; mce += not pre; seen("mce")
        elif "PCIe Bus Error" in l:
            m = BDF.search(l)
            if m:
                e.setdefault(m.group(1), [0, 0])["Uncorrect" in l] += 1; seen(m.group(1))
        elif "mce:" in l or "Machine check" in l or "machine check" in l:
            mce_old += pre; mce += not pre; seen("mce")
    return ev, mce, old, mce_old, last


AER_OUT_X, AER_OUT_RATE = 10, 20   # C1 outlier: correctable rate >= 10x the box median and >= 20 per hour


def aer_outliers(ew, of, hours):
    """Chips whose correctable kernel AER count over the history window is an outlier: bdf -> (count, per hour, median)."""
    eps = sorted(b for b in of if of[b][1] == "ep")
    cs = sorted(ew.get(b, [0, 0])[0] for b in eps)
    med = cs[len(cs) // 2] if cs else 0
    return {b: (ew[b][0], ew[b][0] / hours, med) for b in eps if b in ew
            and ew[b][0] / hours >= AER_OUT_RATE and ew[b][0] >= AER_OUT_X * max(med, 1)}


def journal(boot, week, aertsv, bw, ww, since=None, label=""):
    of = {}   # bdf -> (chip, role) for the run's chips and their root ports
    for l in open(aertsv).read().splitlines()[1:]:
        f = l.split("\t")
        if len(f) >= 3:
            of.setdefault(f[0], (f[2], f[1]))
    name = lambda b: (f"{b[5:]} ({ubb(b)})" if of[b][1] == "ep" else f"{of[b][0][5:]} ({ubb(of[b][0])}) root port {b}")
    (eb, _, ob, _, lb), (ew, mce, ow, mo, lw) = kernel_events(boot, since), kernel_events(week, since)
    win = "in 7 days" if since is None else f"since repair {label}"
    lt = lambda k: f", last {pdt(max(lb.get(k, 0), lw.get(k, 0)))}" if max(lb.get(k, 0), lw.get(k, 0)) else ""
    for b in sorted(set(eb) | set(ew)):
        if b not in of:
            continue
        cb, ub = eb.get(b, [0, 0]); cw, uw = ew.get(b, [0, 0])
        if ub:
            print(f"AER WARN {name(b)}: {ub} uncorrectable kernel AER event(s) this boot, before the run{lt(b)}")
        elif uw:
            print(f"AER WARN {name(b)}: {uw} uncorrectable kernel AER event(s) {win}, none this boot{lt(b)}")
        if cb >= bw:
            print(f"AER WARN {name(b)}: {cb} correctable kernel AER events this boot (warn at {bw}){lt(b)}")
        elif cw >= ww:
            print(f"AER WARN {name(b)}: {cw} correctable kernel AER events {win} (warn at {ww}; {cb} this boot){lt(b)}")
    hours = 168.0 if since is None else max((time.time() - since) / 3600, 1.0)
    ow7 = "over 7 days" if since is None else win   # window and last event up front: tray_why cuts evidence at 140 chars
    for b, (c, r, med) in sorted(aer_outliers(ew, of, hours).items()):
        print(f"AER FAIL {name(b)}: {c} correctable kernel AER events {ow7}{lt(b)} = {r:.0f}/h, >= {AER_OUT_X}x the box "
              f"median ({med}) and >= {AER_OUT_RATE}/h: PCIe path fault (chip, slot, retimer or link), from the journal, "
              f"not only this run")
    if mce:
        print(f"MCE WARN host: {mce} kernel MCE/GHES processor error line(s) {win}{lt('mce')}")
    if since is None:
        return
    for b in sorted(set(ob) | set(ow)):   # before the repair: the boot and 7-day journals overlap, so the larger count
        if b not in of:
            continue
        c, u = (max(x) for x in zip(ob.get(b, [0, 0]), ow.get(b, [0, 0])))
        ep = of[b][0] if of[b][1] != "ep" else b
        what = f"AER {ep[5:]}" + ("" if of[b][1] == "ep" else f" root port {b}")
        for n, k in ((u, "uncorrectable"), (c, "correctable")):
            if n:
                print(f"HIST {ubb(ep)}\t{n}\t{what} {n} {k}")
    if mo:
        print(f"HIST host\t{mo}\tkernel MCE/GHES processor error lines {mo}")


def aerrun(path, aertsv, t0, t1, rate, least=2, findings=None):
    """The run window's kernel AER events per chip and root port: only lines stamped in [t0, t1] count."""
    of = {}
    for l in open(aertsv).read().splitlines()[1:]:
        f = l.split("\t")
        if len(f) >= 3:
            of.setdefault(f[0], (f[2], f[1]))
    name = lambda b: (f"{b[5:]} ({ubb(b)})" if of[b][1] == "ep" else f"{of[b][0][5:]} ({ubb(of[b][0])}) root port {b}")
    keep = []
    try:
        for l in open(path, errors="replace").read().splitlines():
            try:
                t = float(l.split(" ", 1)[0])
            except ValueError:
                continue   # unstamped: not provably in the run
            if t0 <= t <= t1:
                keep.append(l)
    except OSError:
        return
    ev, _, _, _, last = kernel_events(None, lines=keep)
    have = set()   # aerdelta's lines for the same chip, role and grade: not repeated
    for l in (open(findings, errors="replace").read().splitlines() if findings and os.path.exists(findings) else []):
        m = re.match(r"AER (FAIL|WARN) (\S+) \(tray [1-4]\) (root port (\S+?),|endpoint)", l)
        if m and "during the run" in l:
            have.add((m.group(1), m.group(2), m.group(4)))
    hours = max((t1 - t0) / 3600, 1 / 60)
    win = f"during the run ({pdt(t0)} to {pdt(t1)})"
    outl = aer_outliers(ev, of, hours)
    for b in sorted(ev):
        if b not in of:
            continue
        c, u = ev[b]
        key = lambda g: (g, of[b][0][5:], None if of[b][1] == "ep" else b)
        lt = f", last {pdt(last[b])}" if last.get(b) else ""
        if u and key("FAIL") not in have:
            print(f"AER FAIL {name(b)}: {u} uncorrectable kernel AER event(s) {win}{lt}")
        if b in outl:
            if key("FAIL") not in have:
                print(f"AER FAIL {name(b)}: {c} correctable kernel AER events {win}{lt} = {outl[b][1]:.0f}/h, >= "
                      f"{AER_OUT_X}x the box median ({outl[b][2]}) and >= {AER_OUT_RATE}/h: PCIe path fault (chip, slot, "
                      "retimer or link)")
        elif c >= least and c / hours >= rate and key("WARN") not in have:
            print(f"AER WARN {name(b)}: {c} correctable kernel AER events {win}{lt} = {c / hours:.1f}/h (warn at {rate:g}/h)")

def sel(path, now, bw, fw, since=None, label=""):
    n_bc, frb3, frb2, crit, old, last = 0, {}, {}, {}, {}, {}   # old: (scope, kind) -> events before since
    for l in open(path, errors="replace").read().splitlines():
        p = [x.strip() for x in l.split("|")]
        if len(p) < 6:
            continue
        try:
            t = time.mktime(time.strptime(p[1] + " " + p[2], "%m/%d/%Y %H:%M:%S"))
        except ValueError:
            continue
        if now - t > 7 * 86400:
            continue
        sn, e = p[3].split()[-1] if p[3] else "?", p[4]
        if since is not None and t < since:
            k = ("SEL Bus Correctable" if "Bus Correctable" in e else "SEL FRB3" if "FRB3" in e else "SEL FRB2" if "FRB2" in e
                 else "SEL critical threshold crossings" if "Critical going high" in e or "Critical going low" in e else "")
            if k:
                k = (tray_of(sn) if k != "SEL Bus Correctable" else "host", k); old[k] = old.get(k, 0) + 1
            continue
        if "Bus Correctable" in e:
            n_bc += 1; k = "bc"
        elif "FRB3" in e:
            k = u = sn.split("_")[0]; frb3[u] = frb3.get(u, 0) + 1
        elif "FRB2" in e:
            k = a = sn.replace("_Stat", ""); frb2[a] = frb2.get(a, 0) + 1
        elif "Critical going high" in e or "Critical going low" in e:
            k = (sn, "upper" if "high" in e else "lower"); crit[k] = crit.get(k, 0) + 1
        else:
            continue
        last[k] = max(last.get(k, 0), t)
    win = "in 7 days" if since is None else f"since repair {label}"
    lt = lambda k: f", last {pdt(last[k])}"
    # all before the run: WARN, never FAIL; a new FRB3 during the run is a WARN too (selnew)
    if n_bc >= bw:
        print(f"SEL WARN host: {n_bc} SEL Bus Correctable errors {win} (warn at {bw}){lt('bc')}")
    for u, n in sorted(frb3.items()):
        print(f"SEL WARN {tray_name(u)}: {n} SEL FRB3 processor init failure(s) {win}, before the run (busbar signature){lt(u)}")
    for a, n in sorted(frb2.items()):
        if n >= fw:
            print(f"SEL WARN {tray_name(a)}: {n} SEL FRB2 hang-in-POST {win} (warn at {fw}){lt(a)}")
    for (s, d), n in sorted(crit.items()):
        print(f"SEL WARN {tray_name(s)}: {n} {d} critical threshold crossing(s) {win}{lt((s, d))}")
    for (sc, k), n in sorted(old.items()):
        print(f"HIST {sc}\t{n}\t{k} {n}")


HIST_TAIL = "before the run: pre-existing, not from this run"
HIST_CHECKS = (("AER", "kernel PCIe AER history"), ("SEL", "SEL history"), ("FRB", "SEL FRB2/FRB3 history"),
               ("DMESG", "kernel log history (MCE, F18 chain)"))


def hist1(lines, keep, cap=700):
    """History read before the run: at most one WARN line per check, never a FAIL, never a
    per-chip or per-tray line that a tray verdict would count. The full lines go to keep (the evidence file)."""
    got = {}
    for l in lines:
        l = l.rstrip("\n")
        m = re.match(r"(AER|MCE|SEL|CHAIN) (?:FAIL|WARN|INFO) (.*)$", l)
        if not m:
            if l:
                print(l)   # HIST lines and SDR (sensor state now) pass through
            continue
        keep.write(l + "\n")
        k = {"MCE": "DMESG", "CHAIN": "DMESG"}.get(m.group(1), m.group(1))
        if k == "SEL" and "FRB" in l:
            k = "FRB"
        got.setdefault(k, []).append(m.group(2).replace("history before the run, not graded: ", "")
                                     .replace(", before the run (busbar signature)", " (busbar signature)"))
    for k, what in HIST_CHECKS:
        if k in got:
            s = "; ".join(got[k])
            if len(s) > cap:
                s = s[:cap].rsplit(";", 1)[0] + f"; ... ({len(got[k])} items, all in {os.path.basename(keep.name)})"
            print(f"{k} WARN host: {what}: {s}; {HIST_TAIL}")


def f18chain(aertsv, runstart, files, since=None):
    """F18 (e.g. 46:00.0, tray 2): ARC timeout on a chip -> AER on its root port -> MCE, in -o short-unix journals.
    Stamped before runstart: WARN; at or after: FAIL; ending before since (the last repair): a HIST line (C1 INFO).
    Each line names the chip's BDF and tray."""
    of = {}
    for l in open(aertsv).read().splitlines()[1:]:
        f = l.split("\t")
        if len(f) >= 3 and f[1] == "ep":
            of[f[0]] = f[2]
    rp = {l.split("\t")[0]: l.split("\t")[2] for l in open(aertsv).read().splitlines()[1:] if l.split("\t")[1] == "rp"}
    arc, aer, mce, seen = {}, {}, [], set()   # the boot and 7-day journals overlap: count each line once
    for fn in files:
        for l in open(fn, errors="replace").read().splitlines():
            if l in seen:
                continue
            seen.add(l)
            try:
                t = float(l.split(" ", 1)[0])
            except ValueError:
                continue
            m = re.search(r"tenstorrent ([0-9a-f:.]{12}): Timeout waiting for ARC response", l)
            if m and m.group(1) in of:
                arc.setdefault(m.group(1), []).append(t)
            m = re.search(r"device_id: ([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])$", l)
            if m and m.group(1) in rp:
                aer.setdefault(rp[m.group(1)], []).append(t)
            if "Machine check" in l or "mce:" in l:
                mce.append(t)
    for b in sorted(arc):
        t0 = min(arc[b]); a = [t for t in aer.get(b, []) if t >= t0]; m = [t for t in mce if t >= t0]
        if not a or not m:
            continue
        last = max(max(arc[b]), max(a), max(m))
        if since is not None and last < since:
            print(f"HIST {ubb(b)}\t1\tF18 chain {b[5:]} 1"); continue
        sev = "FAIL" if last >= runstart else "WARN"
        when = "during the run" if sev == "FAIL" else f"before the run, last {pdt(last)}"
        print(f"CHAIN {sev} {b[5:]} ({ubb(b)}): ARC timeout x{len(arc[b])} -> AER on its root port x{len(a)} -> MCE x{len(m)} {when}")


def histinfo(label, files):
    """HIST lines (journal, sel with a repair time) -> one INFO line per tray, then host; report only."""
    by = {}
    for f in files:
        for l in open(f, errors="replace").read().splitlines():
            p = l.split("\t")
            if len(p) == 3 and p[0].startswith("HIST "):
                sc = p[0][5:]; n, w = by.get(sc, (0, [])); by[sc] = (n + int(p[1]), w + [p[2]])
    for sc in sorted(by, key=lambda x: (x == "host", x)):
        n, w = by[sc]
        print(f"C1 INFO {sc}: {n} event{'s' if n != 1 else ''} before repair {label}, ignored ({'; '.join(w)})")


def selnew(pre, post, out):
    """SEL rows in post, not in pre, about FRB, presence, power, voltage or temperature -> raw rows to out, one WARN naming
    tray N; a thermal trip (upper critical / non-recoverable going high, thermal trip) FAILs its tray as thermal"""
    try:
        old = set(open(pre, errors="replace").read().splitlines())
        rows = open(post, errors="replace").read().splitlines()
    except OSError:
        return
    new = [l for l in rows if l not in old and re.search(r"FRB|Presence|PresDet|Power Unit|Power Supply|PSU|Current|Voltage|Critical|Temp|Thermal", l, re.I)]
    with open(out, "w") as f:
        f.write("".join(l + "\n" for l in new))
    # an FRB3 during the run (often at the C9 end reset) is one WARN naming its tray, BDF range and sensors, never a FAIL
    # (POWER needs 12 V-lost evidence, the CPLD or a newly set HSC latch; a lost chip, a newly set
    # HSC OFF/VIN_UV latch or a CPLD 12 V loss fails the tray on its own line).
    f3 = {}
    for l in new:
        p = [x.strip() for x in l.split("|")]
        if len(p) >= 6 and "FRB3" in p[4] and "Deasserted" not in p[5]:
            f3.setdefault(tray_of(p[3]), []).append(p[3].split()[-1])
    for tr, sn in sorted(f3.items()):
        rg = TRAY_BDFS.get(tr)
        print(f"SEL WARN {tr}{f' ({rg})' if rg else ''}: {len(sn)} SEL FRB3 processor init failure(s) during the run "
              f"({tray_name(', '.join(sorted(set(sn))))[:120]}); an SEL row alone, no 12 V evidence: not a power fault")
    th = {}   # quads 1-2's thermal trips: a temperature sensor's upper critical or trip assertion
    for l in new:
        p = [x.strip() for x in l.split("|")]
        if (len(p) >= 6 and re.search(r"Temp|Thermal|TMAX", p[3], re.I) and "Deasserted" not in p[5]
                and re.search(r"Upper (Critical|Non-recoverable)|Thermal Trip|Overtemp", p[4] + " " + p[5], re.I)):
            th.setdefault(tray_of(p[3]), []).append(l)
    for tr, sn in sorted(th.items()):
        print(f"THERMAL FAIL {tr}: {len(sn)} SEL thermal trip event(s) during the run ({tray_name(sn[0])[:120]})")
    tl = {l for v in th.values() for l in v}
    rest = [l for l in new if "FRB3" not in l and l not in tl]
    if rest:
        print(f"SEL WARN {len(rest)} new event(s): {tray_name(rest[0])[:140]}")


def sdr(path):
    for l in open(path, errors="replace").read().splitlines():
        p = [x.strip() for x in l.split("|")]
        if len(p) >= 3 and p[2] and p[2] not in ("ok", "ns"):
            print(f"SDR WARN {tray_name(p[0])}: status {p[2]}")


def ethdown(path, label):
    try:
        j = json.load(open(path))
    except (OSError, ValueError):
        return
    for x in j.get("device_info", []):
        s, b = x.get("smbus_telem", {}), x.get("board_info", {}).get("bus_id", "?")
        try:
            live, en = int(s["ETH_LIVE_STATUS"], 16), int(s["ENABLED_ETH"], 16)
        except (KeyError, TypeError, ValueError):
            continue
        # low 16 bits: per-port link up; the high 16 bits vary by host and position with no port meaning (T5)
        down = en & ~(live & 0xFFFF)
        if down:
            ports = [i for i in range(16) if down >> i & 1]; n = bin(en).count("1")
            print(f"ETH WARN {b[5:] if b.count(':') == 2 else b} ({ubb(b)}) {label}: ETH ports {ports} enabled but not up "
                  f"({n - len(ports)}/{n} up; live 0x{live:08x}, enabled 0x{en:04x})")


def ethdown2(path, repath, label, settle_s=None):
    # C2/C9's ETH WARN single-snapshot artifact. The post-reset snapshot can catch link training in
    # flight: the same down-set (e.g. live 0x02df32df, ports 10, 11) flits across random chips each leg. A WARN is
    # printed only for a port still down in the recheck snapshot taken FL_ETH_RECHECK_S later; a port that came up
    # between the two snapshots is a settle transient, not a fault. A chip missing from the recheck keeps its WARN.
    try:
        re = ethports(repath)
    except Exception:
        re = None  # no readable recheck: keep the pre-WARN as it was
    try:
        j = json.load(open(path))
    except (OSError, ValueError):
        return
    for x in j.get("device_info", []):
        s, b = x.get("smbus_telem", {}), x.get("board_info", {}).get("bus_id", "?")
        try:
            live, en = int(s["ETH_LIVE_STATUS"], 16), int(s["ENABLED_ETH"], 16)
        except (KeyError, TypeError, ValueError):
            continue
        # low 16 bits: per-port link up; the high 16 bits vary by host and position with no port meaning (T5)
        down = en & ~(live & 0xFFFF)
        if not down:
            continue
        bdf = b[5:] if b.count(":") == 2 else b
        still = {i for i in range(16) if down >> i & 1}
        # recheck sets are keyed by full bus_id; fall back to the pre-WARN's set when the chip is
        # missing from the recheck (a chip dropping off PCIe is itself a finding, not an ETH-port one)
        redown = re.get(b) if re is not None else None
        keep = sorted(still & redown) if redown is not None else sorted(still)
        if not keep:
            continue
        n = bin(en).count("1")
        extra = f", still down after {settle_s} s" if settle_s else ""
        print(f"ETH WARN {bdf} ({ubb(b)}) {label}: ETH ports {keep} enabled but not up{extra} "
              f"({n - len(keep)}/{n} up; live 0x{live:08x}, enabled 0x{en:04x})")


def ethports(path):  # tt-smi snapshot -> {bdf: set of enabled ETH ports not up}
    try:
        j = json.load(open(path))
    except (OSError, ValueError):
        return {}
    r = {}
    for x in j.get("device_info", []):
        s, b = x.get("smbus_telem", {}), x.get("board_info", {}).get("bus_id", "?")
        try:
            down = int(s["ENABLED_ETH"], 16) & ~(int(s["ETH_LIVE_STATUS"], 16) & 0xFFFF)
        except (KeyError, TypeError, ValueError):
            continue
        r[b] = {i for i in range(16) if down >> i & 1}
    return r


def ethboth(pre, post, label, host=None, disc=None, held=None):
    """tt-smi enabled ports down before the run and after the reset -> ETH FAIL. host: the verdict
    on cabled channels is UMD's (ethlinks), so ports 0-7 (= UMD ch 0-7) are left to ethlinks, and ports 10, 11 (= ch 8, 9)
    are dropped where fleet/eth_peers.txt has no cable on that chip and WARNed (cross-quad, not judged) where
    it has one. Ports 8, 9, 12, 13 keep the FAIL. No host: every port, as before. A host not in eth_peers.txt is a standalone
    galaxy (single-galaxy and Slurm cluster nodes): ports 10-13 (ch 8-11, no in-box link in
    eth_map.txt) leave the box, so one WARN per chip names them and the rest are judged as before. A host not in
    eth_peers.txt whose own link map (disc, the run's discoveries; offhost) shows inter-host links (a cluster node) is no
    standalone: there ports 10-13 are judged from that link map alone, a FAIL "inter-host QSFP ETH ports"; the far host is
    not inspected (no neighbour check gates it). With no link map asked for (disc None, no --map), the
    standalone WARN stays; a link map asked for (--map) that the run could not read gets a WARN
    "not judged: no link map", never the standalone one. With a link map read, the same port down before the
    run on chips of two or more trays is one WARN per port (far end down or resetting), not a FAIL. held:
    the C2 snapshot for a later recheck (c3g, c9); a grouped chip whose port was up there came back and dropped again
    during the run, so it keeps its per-chip FAIL (one not read there keeps the WARN). Chips whose inter-host port was down before the run, up there and down again after this later reset are our own reset
    dropping the off-box link (the far host is never reset by us): one test-side WARN per port, not a FAIL. This holds for any number of chips, one chip included; a port never up
    at C2 on one tray keeps the FAIL. In-box links are judged as before."""
    a, b = ethports(pre), ethports(post)
    h = ethports(held) if held else b
    pm = ethpeers(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eth_peers.txt"), host) if host else {}
    far = None   # far: the link map shows inter-host links (True), shows none (False), none read (None)
    mp0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eth_map.txt")
    lay = None   # the closest layout to the run's first readable discovery (hostlayout), as ethlinks
    for f in (disc or ()) if host else ():   # quad hosts too, to tell a sub-torus recabled one (subquad)
        try:
            lay = hostlayout(last_json(f), mp0, host); break
        except (OSError, ValueError, IndexError, KeyError, TypeError):
            continue
    sub = bool(pm) and subquad(lay and lay[4])
    ext = (lay[2] if lay else ethext(mp0, host)) if host and not pm else set()
    if host and not pm:
        if disc:
            with contextlib.redirect_stdout(io.StringIO()):
                r = offhost(0, disc)
            far = {0: False, 1: True}.get(r)   # no far-host gate; the link map judges
    # a platform axis whose wrap ports (W lines) are all down before and after is a box cabled without
    # them (partial torus): ethlinks says ETH NOT-CABLED, so no FAIL here; a group only partly down keeps it
    wr = (lay[3] if lay else ethwraps(mp0, host)) if host and not pm else {}
    on = {ta(x) for x in set(a) & set(b)}
    dd = {(*ta(x), p) for x in set(a) & set(b) for p in a[x] & b[x] if p < 8}
    wg = {p for g in wr.values() if g <= dd and all(x[:2] in on for x in g) for p in g}
    # an inter-host port (10-13) down before the run on chips of two or more trays at once is the far end down
    # or resetting (e.g. port 10 on two chips each of trays 1 and 3), judged from this host's own chips only:
    # one WARN per port group, never a FAIL. A port down on one tray only stays the FAIL below. Accepted gap: a
    # real cable fault that coincides with such a far-end event at the run start is WARNed too.
    mg = {}
    if host and not pm and far:
        for p in range(10, 14):
            g = sorted(x for x in a if p in a[x])
            if len({ta(x)[0] for x in g}) >= 2:
                mg[p] = g
    # ports 12, 13 of all 32 chips down at pre, up after C2 and
    # down again after C9: our own reset drops the off-box links while the far host, never reset by us, does not retrain
    # them. The same holds for any number of chips, one included: a chip whose port 10-13 was down at pre, up at held (C2) and down after this reset is one test-side WARN per
    # port, judged from this host's own chips only. tray-test stays a
    # single-box test, so the far end is never read. A port never up at C2 keeps the rules above.
    rg = {}
    for p in range(10, 14) if host and not pm and far and held else ():
        r = sorted(x for x in a if p in a[x] and x in b and p in b[x] and x in h and p not in h[x])
        if r:
            rg[p] = r
    for p in sorted(set(mg) | set(rg)):
        g = mg.get(p, [])
        st = [x for x in g if x in b and p in b[x] and p in h.get(x, {p})]
        nm = lambda x: (f"{ubb(x)} / QSFP-DD port {OFFBOX89.get((ta(x)[1], p - 2), 'not in map')} / ASIC {ta(x)[1]} / "
                        f"{x[5:] if x.count(':') == 2 else x} [ch {p - 2}]")
        if p in rg:
            r = rg[p]; tr = sorted({ta(x)[0] for x in r})
            print(f"ETH WARN inter-host QSFP port {p} ({'trays' if len(tr) > 1 else 'tray'} {','.join(str(t) for t in tr)}): "
                  f"dropped by our own reset on {len(r)} {'chips at once' if len(r) > 1 else 'chip'}: test-side, the far host "
                  f"is not reset by us; not judged as this host's fault (tt-smi port {p} [ch {p - 2}]: "
                  f"{'; '.join(nm(x) for x in r)}; down before the run; up after the C2 reset, down after the {label} reset; "
                  f"far host not inspected)")
        if st:
            # ' WARN ' with spaces: fl_report and tray_why count a line as a warning only so
            print(f"ETH WARN inter-host QSFP port {p} (trays {','.join(str(t) for t in sorted({ta(x)[0] for x in g}))}): "
                  f"down on {len(g)} chips at once: far end down or resetting; not judged as this host's fault (tt-smi port {p} [ch {p - 2}]: {'; '.join(nm(x) for x in g)}; down before the run; "
                  f"still down after the {label} reset: {'; '.join(nm(x) for x in st)}; far host not inspected)")
    for bdf in sorted(set(a) & set(b)):
        ports = sorted(a[bdf] & b[bdf]); nb = bdf[5:] if bdf.count(":") == 2 else bdf; xq = []
        if pm:
            t, s_ = ta(bdf)
            xq = [p for p in ports if p in (10, 11) and (t, s_, p - 2) in pm]
            ports = [p for p in ports if p >= 8 and p not in (10, 11)]
        elif host:
            xb = [p for p in ports if 10 <= p <= 13 and not ((bdf in mg.get(p, ()) and p in h.get(bdf, {p})) or bdf in rg.get(p, ()))]
            # ports 0-7 (= UMD ch 0-7) that leave the box on this platform (Slurm cluster nodes: ports 0, 1
            # of asic 4) are out of scope; ethlinks already prints them as ETH EXT, so no line here
            t, s_ = ta(bdf)
            ports = [p for p in ports if not 10 <= p <= 13 and (p >= 8 or (t, s_, p) not in ext) and (t, s_, p) not in wg]
            if xb and far:
                print(f"ETH FAIL {nb} ({ubb(bdf)}): inter-host QSFP ETH ports {xb} (ch {[p - 2 for p in xb]}) down before the "
                      f"run and still down after the {label} reset; far host not inspected (neighbours never gate), "
                      f"judged from {host}'s own link map")
            elif xb and far is None and disc is not None:   # --map given, no readable link map
                print(f"ETH WARN {nb} ({ubb(bdf)}): inter-host QSFP ETH ports {xb} (ch {[p - 2 for p in xb]}) down before the "
                      f"run and still down after the {label} reset; not judged: no link map ({host})")
            elif xb:
                print(f"ETH WARN {nb} ({ubb(bdf)}): ETH ports {xb} (ch {[p - 2 for p in xb]}) down before the run and still "
                      f"down after the {label} reset; ports 10-13 leave the box, not judged on a standalone galaxy ({host})")
        if xq and sub:
            print(f"ETH EXT {nb} ({ubb(bdf)}): external ETH ports {xq} (ch {[p - 2 for p in xq]}) down before the run and still "
                  f"down after the {label} reset; cross-quad, not judged: {SUBEXT}")
        elif xq:
            print(f"ETH WARN {nb} ({ubb(bdf)}): cross-quad ETH ports {xq} (ch {[p - 2 for p in xq]}) down before the run and "
                  f"still down after the {label} reset; not judged (peer {', '.join(sorted({pm[(t, s_, p - 2)][0] for p in xq}))})")
        if ports:
            print(f"ETH FAIL {nb} ({ubb(bdf)}): enabled ETH ports {ports} down before the "
                  f"run and still down after the {label} reset")


def ethmap(path):  # fleet/eth_map.txt -> ({(tray, asic, ch): (tray, asic, ch)} intra-host, {(tray, asic, ch)} QSFP to a quad mate)
    loc, q = {}, set()
    for l in open(path):
        w = l.split()
        if w and w[0] == "L":
            a, b = tuple(map(int, w[1:4])), tuple(map(int, w[4:7])); loc[a] = b; loc[b] = a
        elif w and w[0] == "Q":
            q.add(tuple(map(int, w[1:4])))
    return loc, q


def ethplat(mapf, host):
    """This host's platform for the ETH maps: FLEET_PLATFORM, else the cluster platform for a bh-glx-* hostname. "" for a host
    in eth_peers.txt (quad hosts: as before) or with no host."""
    if not host or ethpeers(os.path.join(os.path.dirname(mapf), "eth_peers.txt"), host):
        return ""
    pl = os.environ.get("FLEET_PLATFORM") or ("exabox" if re.match(r"bh-glx-", host, re.I) else "")
    return pl if re.fullmatch(r"[a-z0-9_]+", pl) else ""


def platmap(mapf, host):
    """eth_map_<platform>.txt next to mapf (cluster platform: UMD's in-box link map) where ethplat names a
    platform that has one, else mapf. Quad hosts keep eth_map.txt."""
    pl = ethplat(mapf, host)
    p = os.path.join(os.path.dirname(mapf), f"eth_map_{pl}.txt")
    return p if pl and os.path.isfile(p) else mapf


# The valid in-box cabling layouts of a Blackhole galaxy. Each is the fixed
# in-box links of its board revision (traces and linking boards: the L lines of eth_map.txt, Rev C as on the X16 quad hosts,
# or of eth_map_exabox.txt, Rev A/B, without their QSFP ports) plus the QSFP cables of a tt-CableGen node layout
# (experiment_topologies BH_GALAXY_REV_{AB,C}_{SC16,ST-SC16}). QSFP_DD ports 1-6 of a tray (UBB_BLACKHOLE, tt-metal
# tools/scaleout/board/board.cpp) -> (asic, channels); ports 7-14 use channels 8, 9, not judged by C3.
QSFP = {1: (5, (2, 3)), 2: (1, (2, 3)), 3: (1, (0, 1)), 4: (2, (0, 1)), 5: (3, (0, 1)), 6: (4, (0, 1))}
P12, P13 = ((1, 2), (3, 4)), ((1, 3), (2, 4))   # a cable per tray pair: tray 1<->2 and 3<->4, or 1<->3 and 2<->4
WRAPS = {"8": (1, 2), "4": (3, 4)}   # the QSFP ports that close the 4x8 mesh's 8-chip and 4-chip rings on a sub-torus
# Rev A/B: both (the W lines of eth_map_exabox.txt). Rev C: ports 1, 2 only, the 8-chip wraps
# (ports 2,3 on 01/05/41/45/81/85/c1/c5:00.0); ports 3-6 are cabled the same on the Rev C SC16 quad host, which has no rings
WRAPAX = {"eth_map.txt": ("8",), "eth_map_exabox.txt": ("8", "4")}
# (name, fixed links from, {port: tray pairs, or "Q" for a cable to a quad mate}, what it is); a port not named has no
# in-box cable (it leaves the box or is not cabled). A tie in the pick goes to the layout with more links, then this order.
LAYOUTS = (
    ("revc-sc16", "eth_map.txt", {1: "Q", 2: "Q", 3: P12, 4: P12, 5: P12, 6: P12}, "Rev C SC16, X16 quad host"),
    ("revc-torus", "eth_map.txt", {1: P13, 2: P13, 3: P12, 4: P12, 5: P12, 6: P12}, "Rev C single-galaxy torus"),
    ("revc-torus-p6x", "eth_map.txt", {1: P13, 2: P13, 3: P12, 4: P12, 5: P12, 6: P13},
     "Rev C torus, port 6 cabled tray 1<->3, 2<->4 (alternate pair layout)"),
    ("revc-sc16-p6x", "eth_map.txt", {1: "Q", 2: "Q", 3: P12, 4: P12, 5: P12, 6: P13},
     "Rev C SC16, port 6 cabled tray 1<->3, 2<->4 (alternate pair layout)"),
    ("revc-subtorus", "eth_map.txt", {1: P13, 2: P13, 3: P12, 4: P12, 5: P13}, "Rev C ST-SC16 sub-torus, A67/A89 DiT"),
    ("revab-subtorus", "eth_map_exabox.txt", {1: P12, 2: P12, 3: P13, 4: P13, 5: P12}, "Rev A/B sub-torus, Slurm cluster"),
    ("revab-sc16", "eth_map_exabox.txt", {1: "Q", 2: "Q", 3: P13, 4: P13, 5: P13, 6: P13}, "Rev A/B SC16"),
)


def layouts(mapf):
    """[(name, what, loc {(tray, asic, ch): (tray, asic, ch)}, q {(tray, asic, ch)}, ext {(tray, asic, ch)} ports 0-7 with no
    in-box cable, wr {axis: wrap ports})]: LAYOUTS. A ring layout with a whole wrap axis uncabled (A67/A89
    boxes whose 8-chip-axis wraps are not cabled) is the same layout: closest scores it so, notcabled says it."""
    here, out, fixed = os.path.dirname(os.path.abspath(mapf)), [], {}
    qe = {(a, c) for a, cs in QSFP.values() for c in cs}
    for name, base, ports, what in LAYOUTS:
        if base not in fixed:
            fixed[base] = {k: v for k, v in ethmap(os.path.join(here, base))[0].items() if k[1:] not in qe and v[1:] not in qe}
        axs = WRAPAX[base] if all(p in ports and ports[p] != "Q" for ax in WRAPAX[base] for p in WRAPS[ax]) else ()
        loc, q = dict(fixed[base]), set()
        for p, pr in ports.items():
            a, cs = QSFP[p]
            for c in cs:
                if pr == "Q":
                    q |= {(t, a, c) for t in (1, 2, 3, 4)}
                else:
                    for x, y in pr:
                        loc[(x, a, c)] = (y, a, c); loc[(y, a, c)] = (x, a, c)
        ext = {(t, a, c) for t in (1, 2, 3, 4) for a, c in qe if (t, a, c) not in loc and (t, a, c) not in q}
        wr = {ax: {(t, QSFP[p][0], c) for t in (1, 2, 3, 4) for p in WRAPS[ax] for c in QSFP[p][1]} for ax in axs}
        out.append((name, what, loc, q, ext, wr))
    return out


def dlinks(d):  # UMD discovery -> ({(tray, asic)} on PCIe, {frozenset of 2 (tray, asic, ch)} in-box links on channels 0-7)
    bdf = {int(k): v for k, v in d["bdfs"].items()}
    up = set()
    for c, ls in d["links"].items():
        for ch, pv in (ls or {}).items():
            if int(c) in bdf and int(pv[0]) in bdf and int(ch) < 8 and int(pv[1]) < 8:
                up.add(frozenset(((*ta(bdf[int(c)]), int(ch)), (*ta(bdf[int(pv[0])]), int(pv[1])))))
    return {ta(b) for b in bdf.values()}, up


def closest(d, mapf, quad=False, base="revc-torus"):
    """the layout closest to the discovered in-box links: fewest links it expects that are not up
    (both chips on PCIe) plus discovered in-box links it does not have; a tie goes to the layout with more links. A wrap axis
    whose every link is not up counts as uncabled, not missing (notcabled still judges it before and after a
    reset). quad (a host in eth_peers.txt): the Rev C SC16 layouts, its ports 1, 2 go to the quad mates, and revc-subtorus
    (a quad host recabled sub-torus, picked by fit; subquad). base is the
    host's own layout (its platform map, as on main). Another layout wins only on evidence, never by expecting
    fewer links: it must have some discovered link base does not (a cut cable never makes one), and an SC16 layout on a host
    not in eth_peers.txt must explain every link (0 missing, 0 not its own), so a cable fault is never absorbed into
    another layout. -> (name, what, loc, q, ext, wr, uncabled wrap axes, missing links, links it does not explain, number of
    layouts)."""
    on, up = dlinks(d)
    ls, cs = [l for l in layouts(mapf) if not quad or l[0].startswith("revc-sc16") or l[0] == "revc-subtorus"], []
    for i, (name, what, loc, q, ext, wr) in enumerate(ls):
        e = {frozenset((k, v)) for k, v in loc.items() if k[:2] in on and v[:2] in on}
        m, u, nw, k = e - up, up - e, [], len(e)
        for ax, g in sorted(wr.items()):
            gl = {l for l in e if any(x in g for x in l)}
            if gl and gl <= m:   # uncabled: its links count neither as missing nor as links for the tie (SC16 first)
                m, nw, k = m - gl, nw + [ax], k - len(gl)
        key = (len(m) + len(u), -k, len(nw), i)   # same links: the layout that has them without a wrap axis down
        cs.append((key, name, e, u, (name, what, loc, q, ext, wr, tuple(nw), m, u, len(ls))))
    b = next((c for c in cs if c[1] == base), cs[0])
    sc = lambda c: not quad and c[1].startswith("revc-sc16")   # an SC16 layout on a host not in eth_peers.txt
    ok = [c for c in cs if c is b or (sc(c) and not c[0][0]) or (not sc(c) and c[2] & b[3])]
    return min(ok, key=lambda c: c[0])[4]


def hostlayout(d, mapf, host):
    """(loc, q, ext, wr, closest's tuple or None) for C3: the closest layout to the discovery d (closest), on
    every host; a host in eth_peers.txt picks among the Rev C SC16 layouts and revc-subtorus. Its base: revc-sc16 on a quad host or with no
    host (eth_map.txt, as on main), revab-subtorus
    on the cluster platform (platmap), else revc-torus. No discovery: as before (platmap, ethext, ethwraps)."""
    if not d or not d.get("bdfs"):
        loc, q = ethmap(platmap(mapf, host))
        return loc, q, ethext(mapf, host), ethwraps(mapf, host), None
    quad = bool(host and ethpeers(os.path.join(os.path.dirname(os.path.abspath(mapf)), "eth_peers.txt"), host))
    base = "revc-sc16" if quad or not host else ("revab-subtorus" if os.path.basename(platmap(mapf, host)) == "eth_map_exabox.txt"
                                     else "revc-torus")
    c = closest(d, mapf, quad, base)
    return c[2], c[3], c[4], c[5], c


def subquad(c):
    """closest's tuple (or None) of a host in eth_peers.txt -> True when that host is judged sub-torus,
    not SC16. eth_peers.txt is the X16-era SC16 cabling, so its host-to-host ports (QSFP channels 0-7, cross-quad 8, 9)
    are ETH EXT, not judged, until a sub-torus peer map exists (DC's hookup, CSOPS-503/504). Judged SC16: full peer checks."""
    return bool(c) and not c[0].startswith("revc-sc16")


SUBEXT = "host judged sub-torus: fleet/eth_peers.txt is the X16-era SC16 map, no sub-torus peer map yet (CSOPS-503/504)"


def cables(miss, wr):
    """missing links -> one phrase per QSFP cable they lie on ("05:00.0<->45:00.0 (tray 1<->2)"), wrap cables first, and how
    many are wrap cables; a link not on a QSFP port (trace or linking board) is no cable."""
    qe = {(a, c): p for p, (a, cs) in QSFP.items() for c in cs}
    wp = {p for ax in wr for p in WRAPS[ax]}
    cs = {}
    for l in miss:
        x, y = sorted(l)
        if x[1:] == y[1:] and x[1:] in qe:
            cs.setdefault((qe[x[1:]] not in wp, x[0], y[0], x[1]), set())
    ph = [f"{tbdf(t, a)}<->{tbdf(u, a)} (tray {t}<->{u})" for _, t, u, a in sorted(cs)]
    return ph, sum(1 for k in cs if not k[0])

def ethext(mapf, host):
    """{(tray, asic, ch)} that are not in-box links on this host's platform (X lines of its map, platmap).
    Empty for a host in eth_peers.txt (quad hosts: as before) or with no host."""
    pl = ethplat(mapf, host)
    r = set()
    for l in open(platmap(mapf, host)) if pl else ():
        w = l.split()
        if len(w) == 5 and w[0] == "X" and w[1] == pl:
            r.add(tuple(map(int, w[2:5])))
    return r


def ethwraps(mapf, host):
    """{axis: {(tray, asic, ch)}} wrap ports of this host's platform (W lines of its
    map, platmap): the in-box links that close the mesh's rings. Empty for quad hosts and where the map has no W lines."""
    pl = ethplat(mapf, host)
    r = {}
    for l in open(platmap(mapf, host)) if pl else ():
        w = l.split()
        if len(w) == 6 and w[0] == "W" and w[1] == pl:
            r.setdefault(w[2], set()).add(tuple(map(int, w[3:6])))
    return r


NCLINE = re.compile(r"^ETH NOT-CABLED ([0-9a-f]{2}:[0-9a-f]{2}\.[0-7]) \(tray [1-4]\) \w+: local ETH ports \[([0-9, ]*)\]")


def notcabled(wr, down, on, files):
    """{(tray, asic, ch): axis} of the wrap ports of every axis whose wraps are ALL down now (down:
    {(tray, asic, ch)}, chips on PCIe on) and all were down earlier: in pre's held in-box FAIL lines or an earlier
    "ETH NOT-CABLED" line (files). That box has no wrap cable on that axis (partial torus, e.g. a 4x8 mesh cabled without
    its 8-chip rings): not a fault of the box. A group only partly down, or up at pre, is a lost cable and stays FAIL."""
    was = set()
    for f in files:
        try:
            ls = open(f, errors="replace").read().splitlines() if f else []
        except OSError:
            continue
        for l in ls:
            m = LOCALFAIL.match(l) or NCLINE.match(l)
            if m and (m.re is NCLINE or m.group(2) == "pre"):
                t, a = ta(m.group(1)); ps = m.group(3) if m.re is LOCALFAIL else m.group(2)
                was |= {(t, a, int(x)) for x in ps.replace(" ", "").split(",") if x}
    r = {}
    for ax, g in wr.items():
        if g <= down and g <= was and all(p[:2] in on for p in g):
            r.update(dict.fromkeys(g, ax))
    return r


def topo(d, mapf, host):
    """the fabric topology the present links support for the bus-bar CCL (ub0_repro.py --topo):
    ring when every wrap port of the platform's map (W lines) links in the discovery d, else linear. No W lines (quad
    single-galaxy hosts): ring, as before."""
    wr = ethwraps(mapf, host)
    if not wr:
        return print("ring: no wrap map for this host (as before)")
    bdf = {int(k): v for k, v in d["bdfs"].items()}
    up = {(*ta(b), int(ch)) for c, b in bdf.items() for ch in d["links"].get(str(c), d["links"].get(c, {}))}
    nm = lambda b: b[5:] if b.count(":") == 2 else b
    dn = sorted({p for g in wr.values() for p in g} - up)
    if not dn:
        return print(f"ring: all {sum(map(len, wr.values()))} wrap ports up")
    axes = sorted(ax for ax, g in wr.items() if g - up)
    print(f"linear: {len(dn)} wrap port(s) of the {'- and '.join(axes)}-chip axis down: "
          + ", ".join(f"{tbdf(t, a)} (tray {t}) ch {c}" for t, a, c in dn[:8]) + (" ..." if len(dn) > 8 else ""))


def ta(bdf):  # "0000:45:00.0" -> (tray 1-4, asic 1-8)
    bus = int(bdf.split(":")[-2], 16)
    return {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}.get(bus & 0xF0, 0), bus & 0x0F


def ethpeers(path, host):  # fleet/eth_peers.txt -> {(tray, asic, ch): (peer host, tray, asic, ch)} for host; {} if unknown
    r = {}
    try:
        for l in open(path):
            w = l.split()
            if len(w) == 9 and w[0] == "P" and w[1] == host:
                r[tuple(map(int, w[2:5]))] = (w[5], *map(int, w[6:9]))
    except OSError:
        pass
    return r


def tbdf(t, a):  # (tray 1-4, asic 1-8) -> "c5:00.0"
    return f"{({1: 0x00, 2: 0x40, 3: 0xC0, 4: 0x80}[t] | a):02x}:00.0"


LOCALFAIL = re.compile(r"^ETH FAIL ([0-9a-f]{2}:[0-9a-f]{2}\.[0-7]) \(tray [1-4]\) (\w+): local ETH ports \[([0-9, ]*)\]")


def localdown(d, smi, loc, label):
    """{short bdf: set of local channels UMD does not link} (as ethlinks). Partner chip off PCIe: not judged. When smi (a
    tt-smi snapshot) is given and label is not pre, also {short bdf: in-box ports 0-7 (= UMD ch) down in tt-smi only}: UMD
    judges ports 0-7 on eth_peers hosts, so those stay fl_c3eth's WARN and never join the FAIL set. Before the reset tt-smi reads 0 live ports on every chip, so pre is UMD only. Returns (down, smi-only)."""
    bdf = {int(k): v for k, v in d["bdfs"].items()}
    at = {ta(b) for b in bdf.values()}
    nm = lambda b: b[5:] if b.count(":") == 2 else b
    sm = {nm(b): v for b, v in ethports(smi).items()} if smi and label != "pre" else {}
    down, only = {}, {}
    for c in sorted(int(x) for x in d["chips"]):
        b = bdf.get(c)
        if not b:
            continue
        t, a = ta(b)
        up = {int(x) for x in d["links"].get(str(c), d["links"].get(c, {}))}
        mine = {ch for ch in range(8) if (t, a, ch) in loc and loc[(t, a, ch)][:2] in at}
        u = {ch for ch in mine if ch not in up}
        s_ = {ch for ch in sm.get(nm(b), ()) if ch in mine}
        if u:
            down[nm(b)] = u
        if s_ - u:
            only[nm(b)] = s_ - u
    return down, only


def named_local(path, label):  # findings.txt -> {short bdf: channels an earlier label's "local ETH ports" FAIL named}
    r = {}
    try:
        for l in open(path, errors="replace"):
            m = LOCALFAIL.match(l)
            if m and m.group(2) != label:
                r.setdefault(m.group(1), set()).update(int(x) for x in m.group(3).replace(" ", "").split(",") if x)
    except OSError:
        pass
    return r


def ethback(findings, stages, d, smi, label, mapf, host=None):
    """after a reset, say on each earlier local-port FAIL line whether this reset brought its ports back. A port
    whose chip or partner is off PCIe after the reset is not judged, named per missing chip.
    host: its platform's map (platmap; cluster platform: eth_map_exabox.txt)."""
    loc = hostlayout(d, mapf, host)[0]   # the closest layout, as ethlinks
    down, _ = localdown(d, smi, loc, label)
    nm = lambda b: b[5:] if b.count(":") == 2 else b
    on = {nm(b) for b in d["bdfs"].values()}
    try:
        lines = open(findings, errors="replace").read().splitlines()
    except OSError:
        return
    ch = {}
    for l in lines:
        m = LOCALFAIL.match(l)
        if not m or m.group(2) == label or f"; {label} reset: " in l:
            continue
        b = m.group(1); t, a = ta(b)
        ps = [int(x) for x in m.group(3).replace(" ", "").split(",") if x]
        off = {}   # missing chip -> ports not judged
        for p in ps:
            o = b if b not in on else tbdf(*loc[(t, a, p)][:2]) if (t, a, p) in loc else None
            if o is not None and o not in on:
                off.setdefault(o, []).append(p)
        nj = {p for v in off.values() for p in v}; jp = [p for p in ps if p not in nj]
        st = sorted(set(jp) & down.get(b, set())); bk = [p for p in jp if p not in st]
        nt = {o: f"not judged ({o} (tray {ta(o)[0]}) not on PCIe after the {label} reset)" for o in off}
        if not off:
            w = "still down" if not bk else "back up" if not st else f"ports {bk} back up, {st} still down"
        elif not jp and len(off) == 1:
            w = nt[next(iter(off))]
        else:
            w = ", ".join([f"ports {bk} back up"] * bool(bk) + [f"ports {st} still down"] * bool(st) +
                          [f"ports {v} {nt[o]}" for o, v in sorted(off.items())])
        ch[l] = f"{l}; {label} reset: {w}"
    if not ch:
        return
    for f in (findings, stages):
        try:
            t = open(f, errors="replace").read().split("\n")
        except OSError:
            continue
        t = [ch.get(x, x) if f == findings else (x.split("\t", 1)[0] + "\t" + ch[x.split("\t", 1)[1]]
             if "\t" in x and x.split("\t", 1)[1] in ch else x) for x in t]
        open(f + ".tmp", "w").write("\n".join(t)); os.replace(f + ".tmp", f)
    for v in ch.values():
        print(v)


def prerec(pre, label, loc, bdf, at, ldown):
    """ports the pre label found down (pre's "local ETH ports" FAIL lines, held back in pre) judged
    after the <label> reset: up again -> WARN "recovered by <LABEL> reset"; chip or partner off PCIe -> FAIL (not judged).
    Still down: ethlinks' own FAIL for this label names them ("; still down after the <label> reset")."""
    try:
        ls = open(pre, errors="replace").read().splitlines()
    except OSError:
        return
    nm = lambda b: b[5:] if b.count(":") == 2 else b
    on = {nm(b) for b in bdf.values()}
    far = lambda t, a, ps: ", ".join(f"{tbdf(*loc[(t, a, p)][:2])} (tray {loc[(t, a, p)][0]}) port {loc[(t, a, p)][2]}" for p in ps)
    for l in ls:
        m = LOCALFAIL.match(l)
        if not m or m.group(2) != "pre":
            continue
        b = m.group(1); t, a = ta(b)
        ps = [int(x) for x in m.group(3).replace(" ", "").split(",") if x if (t, a, int(x)) in loc]
        ok = [p for p in ps if b in on and loc[(t, a, p)][:2] in at]
        rec = [p for p in ok if p not in ldown.get(b, set())]
        un = [p for p in ps if p not in ok]
        if rec:
            print(f"ETH WARN {b} (tray {t}) {label}: local ETH ports {rec} down at pre (to {far(t, a, rec)}); "
                  f"recovered by {label.upper()} reset")
        if un:
            print(f"ETH FAIL {b} (tray {t}) {label}: local ETH ports {un} not up (to {far(t, a, un)}); down at pre, "
                  f"not judged after the {label} reset (chip or partner off PCIe)")


def peerpres(path):  # '<HOST> <bdf> <rev>' lspci lines of a cable peer -> {HOST: {bdf: rev}} (a host listed = read ok)
    r = {}
    try:
        for l in open(path):
            w = l.split()
            if len(w) == 1:
                r.setdefault(w[0].upper(), {})
            elif len(w) >= 3:
                r.setdefault(w[0].upper(), {})[w[1].lower().removeprefix('0000:')] = w[2].lower()
    except OSError:
        pass
    return r


def peermiss(fe, pres):  # far ends {(host, tray, asic): ports} -> 'HOST bdf (tray N)' texts of peer chips missing, rev ff or 0xffffffff, or tray short
    out = []
    for (h, pt, pa), _ in sorted(fe.items()):
        pr = pres.get(h)
        if pr is None:
            continue
        bad = lambda b: pr.get(b) in (None, 'ff', '0xffffffff')
        n = sum(not bad(tbdf(pt, a)) for a in range(1, 9))
        if bad(tbdf(pt, pa)) or n < 8:
            out.append(f"{h} {tbdf(pt, pa)} (tray {pt}) is missing from PCIe (peer tray down, {n}/8 chips present)")
    return out


def ethlinks(d, label, mapf, peers, host=None, since=None, smi=None, named=None, pre=None, pres=None, nc=None):
    """UMD discovery -> per-chip ETH FAIL / WARN lines. Local ports (both ends on this host, eth_map.txt L):
    an endpoint whose partner chip is on PCIe but which UMD does not link is a FAIL. QSFP ports (eth_map.txt Q): a port
    not active is judged only on the cable peers' state read at the same moment (peers: {HOST: (state, why)}): FAIL when
    every peer is up and idle, else a report-only WARN naming the peer and why. Channels 8, 9 (cross-quad) not judged.
    host: the peers of a chip are its own ports' peers in eth_peers.txt (all peers when host is not listed).
    since: this host's own reset (C2/C9; pre: its last chip reset before this run); a peer with no reset or
    boot since then (window FLEET_PEER_RESET_WINDOW s) = WARN.
    nc: earlier "ETH NOT-CABLED" lines; with pre, a platform axis whose wrap ports are all down before and after
    this reset is an "ETH NOT-CABLED" INFO line, not a FAIL (notcabled)."""
    # the valid in-box cabling layout closest to this discovery (hostlayout); only its links are
    # judged, and the ports it leaves without an in-box cable (Slurm cluster nodes: ports 0,1 of asic 4) are ETH EXT
    loc, q, ext, wr0, lay = hostlayout(d, mapf, host)
    pm = ethpeers(os.path.join(os.path.dirname(mapf), "eth_peers.txt"), host) if host else {}
    loc = {k: v for k, v in loc.items() if k not in ext and v not in ext}
    sub = bool(pm) and subquad(lay)   # its QSFP ports to other hosts are ETH EXT (q is empty off SC16)
    win = int(os.environ.get("FLEET_PEER_RESET_WINDOW", "300"))
    note = peers.pop("#", ("", ""))[1] if peers else ""
    bdf = {int(k): v for k, v in d["bdfs"].items()}
    at = {ta(b): c for c, b in bdf.items()}
    nm = lambda b: b[5:] if b.count(":") == 2 else b
    # tt-smi's ports 0-7 only where ethboth leaves them to this check (host in eth_peers.txt); elsewhere ethboth already
    # FAILs a port down before and after the reset, and a second line would count one port twice
    ldown, _ = localdown(d, smi if pm else None, loc, label)
    nl = named_local(named, label) if named and label != "pre" else {}
    if pre and label != "pre":   # the pre label's in-box ports held back for this reset's verdict
        prerec(pre, label, loc, bdf, at, ldown)
    wr = wr0 if label != "pre" else {}
    ncp = notcabled(wr, {(*ta(b), ch) for b, s_ in ldown.items() for ch in s_}, set(at), (pre, nc)) if wr else {}
    for (t, a, ch), ax in sorted(ncp.items()):
        ldown[nm(tbdf(t, a))].discard(ch)
    for t, a in sorted({p[:2] for p in ncp}):
        cs = sorted(ch for t_, a_, ch in ncp if (t_, a_) == (t, a)); ax = ncp[(t, a, cs[0])]
        to = ", ".join(f"{tbdf(*loc[(t, a, ch)][:2])} (tray {loc[(t, a, ch)][0]}) port {loc[(t, a, ch)][2]}" for ch in cs)
        print(f"ETH NOT-CABLED {tbdf(t, a)} (tray {t}) {label}: local ETH ports {cs} (to {to}) down at pre and after the "
              f"{label} reset, as every wrap port of the {ax}-chip axis: no wrap cable on this box (partial torus): INFO, not judged")
    if lay:
        name, what, _, _, _, _, nw0, m, u, n = lay
        ux = ", ".join(f"{tbdf(*x[:2])} (tray {x[0]}) port {x[2]}<->{tbdf(*y[:2])} (tray {y[0]}) port {y[2]}"
                       for x, y in sorted(tuple(sorted(l)) for l in u)[:4]) + (" ..." if len(u) > 4 else "")
        what += f", no {' or '.join(nw0)}-chip wrap cables" if nw0 else ""
        print(f"ETH LAYOUT {label}: {name} ({what}), the closest of {n} valid in-box cablings: {len(m)} of its links not up, "
              f"{len(u)} discovered in-box links it does not have{': ' + ux if u else ''}")
        dn = {(*ta(b), ch) for b, s_ in ldown.items() for ch in s_}
        ph, nw = cables([l for l in m if l <= dn], wr0) if label != "pre" else ([], 0)
        if ph:   # the cable line: a cabling item for the DC; the per-chip lines below still name each port
            t, a = ta(ph[0][:7]); k = len(ph)
            kind = "wrap cable" if nw == k else "QSFP cable"
            print(f"ETH FAIL {ph[0][:7]} (tray {t}) {label}: {k} {kind}{'s' if k > 1 else ''} missing or bad: "
                  f"{', '.join(ph[:-1]) + ' and ' + ph[-1] if k > 1 else ph[0]} (layout {name}); cabling, for the DC")
    for c in sorted(int(x) for x in d["chips"]):
        b = bdf.get(c)
        if not b:
            continue
        t, a = ta(b)
        up = {int(x) for x in d["links"].get(str(c), d["links"].get(c, {}))}
        act = {int(x) for x in d["active"].get(str(c), d["active"].get(c, []))}
        # a port an earlier label named is not named again (ethback says on that line whether this reset
        # brought it back)
        dn = sorted(ldown.get(nm(b), set()) - nl.get(nm(b), set()))
        miss = [(ch, loc[(t, a, ch)]) for ch in dn]
        if miss:
            to = ", ".join(f"{nm(bdf[at[p[:2]]])} (tray {p[0]}) port {p[2]}" for _, p in miss)
            x = f"; still down after the {label} reset" if label != "pre" and smi else ""
            print(f"ETH FAIL {nm(b)} (tray {t}) {label}: local ETH ports {[ch for ch, _ in miss]} not up (to {to}){x}")
        xd = sorted(ch for ch in range(8) if (t, a, ch) in ext and ch not in up)
        if xd:   # fl_c3links holds these (INCOMPLETE when no reset judged them), never a finding of its own
            print(f"ETH EXT {nm(b)} (tray {t}) {label}: external ETH ports {xd} not up (not an in-box link on this platform"
                  f"{'; ' + SUBEXT if sub else ''})")
        qd = [ch for ch in range(8) if (t, a, ch) in q and ch not in act]
        if not qd:
            continue
        if not peers:
            print(f"ETH WARN {nm(b)} (tray {t}) {label}: QSFP ETH ports {qd} not up; cable peer state not read"
                  f"{' (' + note + ')' if note else ''}: not judged")
            continue
        own = sorted({pm[(t, a, ch)][0] for ch in qd if (t, a, ch) in pm})
        ps = {h: peers.get(h, ("unknown", "no state line for it")) for h in own} if own else peers
        fe = {}
        for ch in qd:
            if (t, a, ch) in pm:
                fe.setdefault(pm[(t, a, ch)][:3], []).append(pm[(t, a, ch)][3])
        far = "; ".join(f"{h} tray {pt} {tbdf(pt, pa)} ports {pc}" for (h, pt, pa), pc in sorted(fe.items()))
        far = f" (to {far})" if far else ""
        down = [h for h, (s, _) in sorted(ps.items()) if s == "down"]
        other = [f"{h} {s}: {w}" if w else f"{h} {s}" for h, (s, w) in sorted(ps.items()) if s not in ("up", "down")]
        rs = {h: re.search(r"\breset=(\d+)", w) for h, (_, w) in ps.items()}
        stale = [h for h in sorted(ps) if since is not None and (not rs[h] or int(rs[h].group(1)) < since - win)]
        if down:
            print(f"ETH WARN {nm(b)} (tray {t}) {label}: partner host {' '.join(down)} down: QSFP ETH ports {qd}{far} not judged")
        elif other:
            print(f"ETH WARN {nm(b)} (tray {t}) {label}: QSFP ETH ports {qd} not up{far}; cable peer {'; '.join(other)}: not judged")
        elif stale:
            print(f"ETH WARN {nm(b)} (tray {t}) {label}: QSFP ETH ports {qd} not up{far}; peer not reset: {' '.join(stale)} had "
                  f"no reset or boot since this host's {'last chip reset before this run' if label == 'pre' else label + ' reset'} "
                  f"(a QSFP link needs both ends reset): not judged")
        elif pres and (pm_ := peermiss(fe, peerpres(pres))):
            print(f"WARN {nm(b)} (tray {t}) {label}: QSFP ports {qd} down because peer {'; '.join(pm_)}; not this box")
        else:
            print(f"ETH FAIL {nm(b)} (tray {t}) {label}: QSFP ETH ports {qd} not up{far}; cable peers {' '.join(sorted(ps))} up and idle")


def peerstate(path):  # "<HOST> up|down|busy|unknown [why]" lines -> {HOST: (state, why)}; a missing or empty file -> {}
    r = {}
    try:
        for l in open(path):
            if l.startswith("#") and "#" not in r:   # a rejected state file (lib/fleet.sh): why, quoted when no peer is read
                r["#"] = ("note", l[1:].strip())
                continue
            w = l.split(None, 2)
            if len(w) >= 2 and w[1] in ("up", "down", "busy", "unknown"):
                r[w[0].upper()] = (w[1], w[2].strip() if len(w) > 2 else "")
    except OSError:
        pass
    return r


ERRLINE = re.compile(r"critical|\berror\b|TT_THROW|TT_FATAL|exception|Timed out|timed out|RESULT ERROR|Traceback|hang|Read 0xffffffff over PCIe", re.I)


def attr(mapf, files):
    kmd, asic = {}, {}  # kmd index -> bdf; asic id -> kmd index (map.txt's optional 3rd column, from lib/fw_check.sh)
    for l in open(mapf):
        f = l.split()
        if len(f) >= 2 and f[1].isdigit():
            kmd[int(f[1])] = f[0]
            if len(f) >= 3 and re.fullmatch(r"(0x)?[0-9a-fA-F]{1,16}", f[2]):
                asic[int(f[2], 16)] = int(f[1])
    texts, umd = [], None
    for p in files:
        try:
            t = open(p, errors="replace").read()
        except OSError:
            continue
        if p.endswith(".json"):
            try:
                j = json.loads(t)
            except ValueError:
                continue
            for k in ("error", "hang", "fault"):
                if isinstance(j.get(k), str):
                    texts.append(j[k])
            for c, v in (j.get("chips") or {}).items():  # the try's own kmd -> pci table, where the map lacks one
                if str(c).isdigit() and isinstance(v, dict) and v.get("pci"):
                    kmd.setdefault(int(c), v["pci"])
            continue
        m = re.search(r"Opening local chip ids/PCIe ids: \{([0-9, ]*)\}/\[([0-9, ]*)\]", t)
        if m:  # positional: the k-th local chip id opens the k-th PCIe (kmd) id (tt-umd cluster.cpp)
            ids = [int(x) for x in m.group(1).split(",") if x.strip()]
            pci = [int(x) for x in m.group(2).split(",") if x.strip()]
            if len(ids) == len(pci):
                umd = dict(zip(ids, pci))
        texts += [l for l in t.splitlines() if ERRLINE.search(l)]
    out = {}
    for t in texts:
        for l in t.splitlines():
            why = l.strip()[:160]
            for m in re.finditer(r"\b(?:Device|device|chip_id|chip id)[ :=]+(\d+)\b", l):
                n = int(m.group(1))
                if n in asic:  # UMD's unique id, e.g. "Firmware bundle version mismatch for device <asic id>"
                    out.setdefault(asic[n], why)
                elif n < 1 << 32:  # an unmatched 64-bit id names no chip: leave it to the probe
                    out.setdefault((umd or {}).get(n, n), why)
            for m in re.finditer(r"/dev/tenstorrent/(\d+)", l):
                out.setdefault(int(m.group(1)), why)
            for m in re.finditer(r"\bPCIe ID (\d+)\b", l):  # UMD's "Read 0xffffffff over PCIe ID <n>": the kmd index
                out.setdefault(int(m.group(1)), why)
            for m in re.finditer(r"\b(?:([0-9a-f]{4}):)?([0-9a-f]{2}):00\.0\b", l):
                b = f"{m.group(1) or '0000'}:{m.group(2)}:00.0"
                k = next((k for k, v in kmd.items() if v == b), None)
                if k is not None:
                    out.setdefault(k, why)
    for k in sorted(out):
        print(k, kmd.get(k, "?"), out[k])
    return 0 if out else 1




ETHTO = re.compile(r"Device (\d+): Timed out while waiting for active ethernet core (\d+)-(\d+) to become active")
ETHST = re.compile(r"Device (\d+): Virtual core (\d+)-(\d+), (Port status: [^(]*)")
LINKF = ("Port status", "Rx link up", "Train status", "Retrain count", "Heartbeat changed")


# QSFP-DD front-panel port per (ASIC, ETH channel) (tray-test fleet QSFP map, tt-metal board.cpp
# UBB_BLACKHOLE): ports 1-6 on channels 0-3, the off-box channel 8/9 ends on ports 7, 8, 11 and 12 (OFFBOX89)
QSFPDD = {(5, 2): 1, (5, 3): 1, (1, 2): 2, (1, 3): 2, (1, 0): 3, (1, 1): 3, (2, 0): 4, (2, 1): 4, (3, 0): 5, (3, 1): 5,
          (4, 0): 6, (4, 1): 6, (1, 8): 7, (2, 8): 7, (5, 8): 8, (6, 8): 8, (1, 9): 11, (2, 9): 11, (5, 9): 12, (6, 9): 12}


def ethcore(mapf, smi, files):
    host = None
    if "--host" in files:  # off-box core: WARN naming port and peer host, not a FAIL
        i = files.index("--host"); host = files[i + 1].upper(); files = files[:i] + files[i + 2:]
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        qs = ethmap(os.path.join(here, "eth_map.txt"))[1]
    except OSError:
        qs = set()
    pm = ethpeers(os.path.join(here, "eth_peers.txt"), host) if host else {}
    kmd = {int(f[1]): f[0] for f in (l.split() for l in open(mapf)) if len(f) >= 2 and f[1].isdigit()}
    try:
        snap = {x.get("board_info", {}).get("bus_id"): x.get("smbus_telem", {}) for x in json.load(open(smi)).get("device_info", [])}
    except (OSError, ValueError, AttributeError):
        snap = {}
    umd, cores, link = {}, [], {}
    for p in files:
        try:
            t = open(p, errors="replace").read()
        except OSError:
            continue
        m = re.search(r"Opening local chip ids/PCIe ids: \{([0-9, ]*)\}/\[([0-9, ]*)\]", t)
        if m:  # positional, as in attr
            ids, pci = ([int(x) for x in g.split(",") if x.strip()] for g in m.groups())
            if len(ids) == len(pci):
                umd = dict(zip(ids, pci))
        for m in ETHTO.finditer(t):
            if m.groups() not in cores:
                cores.append(m.groups())
        for m in ETHST.finditer(t):  # tt-metal's link status line for the core (llrt.cpp:413)
            kv = dict(re.findall(r"([A-Za-z][A-Za-z ]*?): (\w+)", m.group(4)))
            link[m.groups()[:3]] = ", ".join(f"{k.lower()} {kv[k]}" for k in LINKF if k in kv)
    for n, x, y in cores:
        k = umd.get(int(n), int(n)); bdf = kmd.get(k, "?"); s = snap.get(bdf, {})
        ch = int(x) - 20 if y == "25" and int(x) >= 20 else None  # BH translated ETH: (20 + channel, 25), tt-umd
        port, pre = "?", "unknown"
        try:
            en = [i for i in range(16) if int(s["ENABLED_ETH"], 16) >> i & 1]
            port = en[ch]
            pre = "up" if int(s["ETH_LIVE_STATUS"], 16) >> port & 1 else "down"
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        name = f"{bdf[5:]} ({ubb(bdf)}) tt-smi {k}" if bdf != "?" else f"tt-smi {k} (BDF not in map)"
        tr, asic = ta(bdf) if bdf != "?" else (0, 0)
        lk = link.get((n, x, y)) or "unknown"
        if ch is not None and tr and (ch >= 8 or (tr, asic, ch) in qs):  # QSFP / cross-quad: leaves the box
            # judged from this host's own evidence only: the
            # core's link status line says rx link up 0, or (no link line) its port was down in tt-smi before the try
            rx = re.search(r"rx link up (0x[0-9a-fA-F]+|\d+)", lk)
            if (int(rx.group(1), 0) == 0) if rx else pre == "down":
                pe = pm.get((tr, asic, ch))
                peer = f" to {pe[0]} [{tbdf(pe[1], pe[2])}]" if pe else ""
                why = f"rx link up {rx.group(1)}" if rx else "no link line"
                print(f"WARN tray {tr} / QSFP-DD port {QSFPDD.get((asic, ch), '?')} / ASIC {asic} / {bdf[5:]} (ETH ch{ch}, "
                      f"core {x}-{y}, tt-smi {k}): device open timed out on this off-box link{peer}; {why}, port in tt-smi "
                      f"before the try: {pre}; peer host down or resetting: off-box link not judged")
                continue
            lk += "; off-box, peer not shown down"
        print(f"ETH FAIL {name}: ETH core {x}-{y} (channel {'?' if ch is None else ch}, port {port}) not active "
              f"again at mesh open (Device {n}); link: {lk}; port in tt-smi before the try: {pre}")
    return 0 if cores else 1


def _umd(t):  # a log's own "Opening local chip ids/PCIe ids" line: tt-metal Device N -> PCIe (kmd) index, positional
    m = re.search(r"Opening local chip ids/PCIe ids: \{([0-9, ]*)\}/\[([0-9, ]*)\]", t)
    if m:
        ids, pci = ([int(x) for x in g.split(",") if x.strip()] for g in m.groups())
        if len(ids) == len(pci):
            return dict(zip(ids, pci))
    return {}


def _kmd(mapf):
    try:
        return {int(f[1]): f[0] for f in (l.split() for l in open(mapf)) if len(f) >= 2 and f[1].isdigit()}
    except OSError:
        return {}


def c5one(pairsf, d, failf, mapf, stage="C5"):
    """every failing C5 pair shares one chip (2+ failing pairs) -> one line naming it, its links (both ends,
    eth channels) and what tt-metal said about it in those pairs' logs (Device N mapped by each log's UMD line). Exit 1
    when the failures do not share exactly one chip."""
    rows = {}
    for l in open(pairsf):
        f = l.rstrip("\n").split("\t")
        if len(f) >= 10 and f[0].isdigit():
            rows["p" + f[0]] = f
    fails = []
    for l in open(failf):
        lab, _, why = l.rstrip("\n").partition("\t")
        if lab in rows and lab not in [x[0] for x in fails]:
            fails.append((lab, why))
    if len(fails) < 2:
        return 1
    common = set(rows[fails[0][0]][4:6])
    for lab, _ in fails[1:]:
        common &= set(rows[lab][4:6])
    if len(common) != 1:
        return 1
    chip = common.pop()
    kmd = _kmd(mapf)
    idx = {v: k for k, v in kmd.items()}.get(chip)
    try:
        res = dict(l.split()[:2] for l in open(os.path.join(d, "result.txt")) if len(l.split()) >= 2)
    except OSError:
        res = {}
    mine = [k for k, f in rows.items() if chip in f[4:6]]
    others = sum(1 for k, f in rows.items() if chip not in f[4:6] and res.get(k) == "PASS")
    links, said = [], {}
    for lab, why in fails:
        f = rows[lab]
        a, ca, b, cb = (f[4], f[8], f[5], f[9]) if f[4] == chip else (f[5], f[9], f[4], f[8])
        links.append(f"{a[5:]} e{ca} <-> {b[5:]} e{cb} {why} ({lab})")
        try:
            t = open(os.path.join(d, lab + ".log"), errors="replace").read()
        except OSError:
            continue
        umd = _umd(t)
        for m in re.finditer(r"Device (\d+): (Timed out while waiting for active ethernet core \d+-\d+|Timeout \(\d+ ms\) waiting for [^.:(]*)", t):
            k = umd.get(int(m.group(1)), int(m.group(1)))
            if kmd.get(k) == chip:
                said.setdefault(m.group(2), []).append(lab)
    nm = f"{chip[5:]} ({ubb(chip)})" + (f" tt-smi {idx}" if idx is not None else "")
    tm = "; ".join(f"tt-metal errors on this chip in {len(set(v))} of them: {k}" for k, v in said.items())
    print(f"LINK FAIL one chip {nm}: in all {len(fails)} failing {stage} pairs ({len(fails)} of its {len(mine)} pairs failed, "
          f"{others} pairs without it passed): {'; '.join(links)}{'; ' + tm if tm else ''} ({d})")
    return 0


DEVLINE = re.compile(r"Device (\d+): ([^\n]*)")


def devmap(mapf, root, files):
    """tt-metal 'Device N: ...' error lines in a run's logs -> the chip's BDF and tray 1-4. Device N is the
    log's own logical id: its UMD 'Opening local chip ids/PCIe ids' line maps it to the kmd index (identity without
    one), map.txt maps that to the BDF. One line per chip and message: '<bdf> (tray N) tt-smi k|Device N in <files>, ...|
    <message>'. Link-status dumps (Virtual core ...) are left out; a message cut short in one log folds into the full one."""
    kmd = _kmd(mapf)
    got = {}   # (bdf, msg) -> {device: [files]}
    for p in files:
        try:
            t = open(p, errors="replace").read()
        except OSError:
            continue
        if "Device " not in t:
            continue
        umd, rel = _umd(t), os.path.relpath(p, root)
        for m in DEVLINE.finditer(t):
            msg, lvl = m.group(2), 0
            for i, c in enumerate(msg):   # cut at a ")" of the line around it ("...etherne); one queue")
                lvl += (c == "(") - (c == ")")
                if lvl < 0:
                    msg = msg[:i]
                    break
            msg = re.split(r"\. ", re.sub(r" \([\w./]+:\d+\)\s*$", "", msg))[0].strip().rstrip(".")
            if not msg or msg.startswith("Virtual core") or len(msg) < 12:
                continue
            k = umd.get(int(m.group(1)), int(m.group(1)))
            fs = got.setdefault((kmd.get(k, f"tt-smi {k}"), msg[:110]), {}).setdefault(int(m.group(1)), [])
            if rel not in fs:
                fs.append(rel)
    for (b, msg) in list(got):   # a message cut short in a summary line ("...active etherne") folds into the full one
        if any(b == b2 and msg != m2 and m2.startswith(msg) for b2, m2 in got):
            for n, fs in got.pop((b, msg)).items():
                full = next(m2 for b2, m2 in got if b2 == b and m2.startswith(msg))
                for x in fs:
                    if x not in got[(b, full)].setdefault(n, []):
                        got[(b, full)][n].append(x)
    for (b, msg), devs in sorted(got.items()):
        k = {v: i for i, v in kmd.items()}.get(b)
        nm = f"{b[5:]} ({ubb(b)}) tt-smi {k}" if k is not None else f"{b} (BDF not in map)"
        ds = "; ".join(f"Device {n} in {', '.join(sorted(fs)[:4])}{f' +{len(fs) - 4} more' if len(fs) > 4 else ''}" for n, fs in sorted(devs.items()))
        print(f"{nm}|{ds}|{msg}")
    return 0 if got else 1


# C4 try hang: which chip stopped. ETH frames sent per link between two heavy-logs reads (before the run,
# before the reset); a chip whose links sent under half the median stopped during the try. Else the try's telemetry
ETHX = range(20, 32)  # BH translated ETH cores (20 + channel, 25), channels 0-11


def eth_rows(p):
    d = {}
    try:
        for l in open(p, errors="replace"):
            try:
                j = json.loads(l)
                n, xy = j["core"].split(":")
                x, y = (int(v) for v in xy.split(","))
            except (ValueError, KeyError, AttributeError):
                continue
            if y == 25 and x in ETHX:
                d[(int(n), x)] = j
    except OSError:
        pass
    return d


def when(d):  # heavy_logs' when.txt: "YYYY-MM-DD HH:MM:SS UTC" -> epoch
    try:
        return calendar.timegm(time.strptime(open(os.path.join(d, "when.txt")).readline().strip(), "%Y-%m-%d %H:%M:%S UTC"))
    except (OSError, ValueError):
        return None


def runs(xs):  # sorted ints -> (first, last) of each run of consecutive values
    o = []
    for x in xs:
        if o and x == o[-1][1] + 1:
            o[-1][1] = x
        else:
            o.append([x, x])
    return o


# Off-box ETH FEC during a run. Channels 8/9 of ASICs 1,2,5,6 leave the box
# (QSFP-DD ports 7/8/11/12, tt-metal board.cpp UBB_BLACKHOLE); ASICs 3,4,7,8 link in-box on 8/9 and are not judged here.
# A burst on (nearly) every off-box end at once is the far end (another host) resetting: one info line, never a finding.
# Only one or two ends that stand out from their quiet siblings get a WARN, named tray / QSFP-DD port / ASIC / BDF.
OFFBOX89 = {(1, 8): 7, (2, 8): 7, (5, 8): 8, (6, 8): 8, (1, 9): 11, (2, 9): 11, (5, 9): 12, (6, 9): 12}
FEC_KEYS = ("uncorr_cw", "retrain_count", "frames_rxd_badfcs")


def _sane(r):  # an up port with real counters; a disabled port reads values like 17913946297206021938
    return r.get("port_status") == 1 and all(isinstance(r.get(k, 0), int) and 0 <= r.get(k, 0) < 1e12 for k in FEC_KEYS)


def ethfec_ends(mapf, pre, post):
    """[(tray, port, asic, bdf, ch, uncorr, retrains, badfcs)] per up off-box ch8/9 end, counted over the run."""
    kmd = {int(f[1]): f[0] for f in (l.split() for l in open(mapf)) if len(f) >= 2 and f[1].isdigit()}
    a, b = eth_rows(os.path.join(pre, "eth_counters.jsonl")), eth_rows(os.path.join(post, "eth_counters.jsonl"))
    o = []
    for (n, x), r in sorted(b.items()):
        if n not in kmd or x - 20 not in (8, 9) or not _sane(r):
            continue
        bdf = kmd[n][5:] if kmd[n].count(":") == 2 else kmd[n]
        asic = int(bdf.split(":")[0], 16) & 0xF
        port = OFFBOX89.get((asic, x - 20))
        if port is None or ubb(bdf) == "BDF not in map":
            continue
        p = a.get((n, x), {})
        d = []
        for k in FEC_KEYS:
            v = r.get(k, 0)
            w = p.get(k, 0) if _sane(p) else 0
            d.append(v - w if v >= w else v)   # a counter that went back (a reset in between): the post value
        o.append((int(ubb(bdf)[5:]), port, asic, bdf, x - 20, *d))
    return o


def ethfec_why_not(d, what):
    """Why one snapshot gives no counters (never-tested is not a pass), or "" when it has some."""
    f = os.path.join(d, "eth_counters.jsonl")
    if not os.path.isfile(f):
        try:
            r = open(os.path.join(d, "device_reads_skipped.txt"), errors="replace").readline().strip()
        except OSError:
            r = ""
        return f"no ETH counters read {what}" + (f": {r[9:] if r.startswith('skipped: ') else r}" if r else "")
    if eth_rows(f):
        return ""
    try:   # a file that exists but cannot be opened (permissions) is unreadable, never a traceback
        r = open(f, errors="replace").readline().strip()
    except OSError:
        r = ""
    return f"ETH counters read {what} " + (r if r.startswith("skipped:") else "unreadable")


def ethfec(mapf, pre, post, stage="the load", floor=100):
    why = [w for w in (ethfec_why_not(pre, "before the run"), ethfec_why_not(post, "before the reset")) if w]
    if why:   # one plain line through say, never a finding, verdict or rc
        print(f"ETH FEC: not judged ({'; '.join(why)})")
        return 0
    ends = ethfec_ends(mapf, pre, post)
    hit = [e for e in ends if e[5] or e[6]]
    if not hit:
        return 0
    rng = lambda i: (f"{min(e[i] for e in hit)}-{max(e[i] for e in hit)}" if len({e[i] for e in hit}) > 1 else f"{hit[0][i]}")
    info = (f"ETH INFO far end reset during run: {len(hit)} of {len(ends)} off-box ch8/9 ends at once, uncorrected FEC "
            f"{rng(5)} (total {sum(e[5] for e in hit)}), retrains {sum(e[6] for e in hit)}, bad FCS {sum(e[7] for e in hit)}, "
            f"during {stage}; not a finding")
    if len(ends) >= 4 and 4 * len(hit) >= 3 * len(ends):
        print(info)   # common mode: one line, whatever single ends stand out inside it
        return 0
    out = []
    for e in hit:
        sib = sorted(s[5] for s in ends if s is not e)
        med = sib[len(sib) // 2] if sib else 0
        if e[5] >= max(floor, 10 * med):
            out.append(f"ETH FEC WARN tray {e[0]} / QSFP-DD port {e[1]} / ASIC {e[2]} / {e[3]} (ETH ch{e[4]}): "
                       f"uncorrected FEC {e[5]}, retrains {e[6]}, bad FCS {e[7]} during {stage}, siblings' median {med} "
                       f"over {len(sib)} off-box ch8/9 end(s); far end off-box, not mapped; WARN only, no fail")
    if 1 <= len(out) <= 2:
        print("\n".join(out))
    elif out:   # several ends stand out, not the whole box: one line, no per-port list
        print(info.replace("far end reset during run", "off-box FEC during run").replace(" at once", ""))
    return 0


def ethsnap(f):  # heavy_logs' index check of one eth_counters.jsonl: bad FCS on an up port is a FLAG; FEC is judged per run
    rs = [r for r in eth_rows(f).values() if _sane(r)]
    bad = [r for r in rs if r.get("frames_rxd_badfcs", 0)]
    if bad:
        print(f"FLAG ETH bad FCS on {len(bad)} up port(s)")
    else:
        u = sum(1 for r in rs if r.get("uncorr_cw", 0))
        print(f"ok ({len(rs)} up ports" + (f"; uncorrected FEC on {u}, judged by run.sh's ETH FEC line)" if u else ")"))
    return 0


def hangpin(mapf, pre, post, tel=None, watts=80.0):
    kmd = {int(f[1]): f[0] for f in (l.split() for l in open(mapf)) if len(f) >= 2 and f[1].isdigit()}
    name = lambda k: f"{kmd[k][5:]} ({ubb(kmd[k])}) tt-smi {k}" if k in kmd else f"tt-smi {k} (BDF not in map)"
    a, b = eth_rows(os.path.join(pre, "eth_counters.jsonl")), eth_rows(os.path.join(post, "eth_counters.jsonl"))
    tx = {}
    for c, r in b.items():
        if r.get("port_status") == 1 and a.get(c, {}).get("port_status") == 1:
            d = r.get("frames_txd", -1) - a[c].get("frames_txd", 0)
            if 0 <= d < 1e12:  # a counter that went back or reads garbage says nothing
                tx[c] = d
    med = sorted(tx.values())[len(tx) // 2] if len(tx) >= 8 else 0
    low = {}
    for (n, x), d in sorted(tx.items()):
        if med and d < med / 2:
            low.setdefault(n, []).append((x, d))
    t0, t1 = when(pre), when(post)
    if low and len(low) <= max(1, len({n for n, _ in tx}) // 4):  # a few chips stand out, not the whole box
        for n, ls in sorted(low.items()):
            r = sum(d for _, d in ls) / len(ls) / med
            at = (f"; it stopped ~{time.strftime('%H:%M:%S', time.gmtime(t0 + r * (t1 - t0)))} UTC if the rate was steady"
                  if t0 and t1 and t1 > t0 else "")
            print(f"C4 HANG {name(n)}: ETH links {','.join(str(x) for x, _ in ls)} sent {r:.0%} of the median frames "
                  f"of {len(tx)} up links between the reads before the run and before the reset{at}")
        return 0
    busy, seen = set(), set()
    try:
        for row in csv.DictReader(open(tel, errors="replace")) if tel else ():
            try:
                k = int(row["dev"]); seen.add(k)
                if float(row["power_w"]) >= watts:
                    busy.add(k)
            except (KeyError, TypeError, ValueError):
                pass
    except OSError:
        pass
    seen |= set(kmd)
    why = (f"ETH links: {sum(len(v) for v in low.values())} of {len(tx)} under half the median frames, on {len(low)} chips"
           if med else "ETH counters not read before and after")
    idle = sorted(seen - busy)
    ks = lambda ks: ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs(sorted(ks))) or "none"
    print(f"C4 HANG not pinpointed ({why}); at load (>= {watts:.0f} W) in the try's telemetry: tt-smi {ks(busy)} "
          f"({len(busy)} of {len(seen)}); never seen at load: {', '.join(name(k) for k in idle) or 'none'}")
    return 1


# C8: hwmon units -> (scale, valid range); outside it a value is unread telemetry (0xFFFF.. sentinel, or 0 V)
TEL = dict(temp=(1e-3, 1, 150), power=(1e-6, 1, 1000), curr=(1e-3, 1, 2000), vcore=(1e-3, 0.3, 2))


def telem(path, tw, pw, cw, vlow, faults=None):
    hi, lo, bad = {}, {}, {}
    first = c8_faults(path, faults) if faults else {}   # a chip's temps from its first fault on do not count
    cols = ("temp", "power", "aiclk", "curr", "vcore")
    for l in open(path, errors="replace").read().splitlines():
        f = l.split()
        if len(f) < 3:
            continue
        b = f[1]
        for k, v in zip(cols, f[2:]):
            if k not in TEL:
                continue
            sc, a, z = TEL[k]
            try:
                x = float(v) * sc
            except ValueError:
                x = None
            if x is None or not a <= x <= z:
                if v != "-" or k in ("temp", "power"):
                    bad[(b, k)] = bad.get((b, k), 0) + 1
                continue
            if k == "temp" and b in first and f[0].isdigit() and int(f[0]) >= first[b]:
                continue
            hi[(b, k)] = max(hi.get((b, k), x), x); lo[(b, k)] = min(lo.get((b, k), x), x)
    for b in sorted({b for b, _ in list(hi) + list(bad)}):
        w = []
        if hi.get((b, "temp"), 0) > tw: w.append(f"max temp {hi[(b, 'temp')]:.0f} C (warn above {tw:g})")
        if hi.get((b, "power"), 0) > pw: w.append(f"max power {hi[(b, 'power')]:.0f} W (warn above {pw:g})")
        if hi.get((b, "curr"), 0) > cw: w.append(f"max current {hi[(b, 'curr')]:.0f} A (warn above {cw:g})")
        if lo.get((b, "vcore"), 9) < vlow: w.append(f"min vcore {lo[(b, 'vcore')]:.3f} V (warn below {vlow:g})")
        if w:
            print(f"TELEMETRY WARN {b[5:]} ({ubb(b)}): {'; '.join(w)}")
    if bad:
        n = len({b for b, _ in bad})
        print(f"C8: {sum(bad.values())} unread or sentinel sample(s) on {n} chip(s) ignored "
              f"({', '.join(sorted({k for _, k in bad}))})", file=sys.stderr)


# thermal: overheating report from C8's own samples. Only a chip's samples
# from before its first fault count: a fault line run.sh logged (fl_find -> fault_times.tsv, by BDF, else its tray, else
# every chip), a C8 sample reading all ones or out of range (temp over TEMP_MAX C, power over 1000 W: e.g. 4293 W),
# or a C8 poll round the chip is missing from (chip count short). Implausible temps are dropped and only counted.
TEMP_MAX = 150.0
BDF_RE = re.compile(r"\b(?:[0-9a-f]{4}:)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\b")


def c8_faults(tel, faults):
    """{bdf: first fault epoch} from fault_times.tsv ("<epoch>\t<line>") and the C8 samples themselves"""
    first, rnd, seen, allb = {}, [], set(), []

    def hit(b, t):
        first[b] = min(first.get(b, t), t)

    def close(r):   # one poll round: chips missing from it (box count short) fault at its first sample
        if r:
            for b in set(allb) - {x for x, _ in r}:
                hit(b, r[0][1])
    for l in open(tel, errors="replace").read().splitlines():
        f = l.split()
        try:
            t = float(f[0])
        except (IndexError, ValueError):
            continue
        b = f[1] if len(f) > 1 else ""
        if b in seen:
            close(rnd); rnd, seen = [], set()
        seen.add(b); rnd.append((b, t))
        if b not in allb:
            allb.append(b)
        for v, sc, z in ((f[2] if len(f) > 2 else "-", 1e-3, TEMP_MAX), (f[3] if len(f) > 3 else "-", 1e-6, 1000.0)):
            try:
                x = float(v) * sc
            except ValueError:
                continue   # unread ("-"): not a fault by itself
            if x < 0 or x > z:
                hit(b, t)
    close(rnd)
    try:
        rows = open(faults, errors="replace").read().splitlines()
    except OSError:
        rows = []
    for r in rows:
        ts, _, line = r.partition("\t")
        try:
            t = float(ts)
        except ValueError:
            continue
        named = {b for b in allb if b[5:] in set(BDF_RE.findall(line))}
        if not named:
            trs = set(re.findall(r"tray ([1-4])\b", line))
            named = {b for b in allb if trs and ubb(b)[5:] in trs} if trs else set(allb)
        for b in named:
            hit(b, t)
    return first


def thermal(tel, limits, phases, stages, faults, fallback, src):
    first = c8_faults(tel, faults)
    lim = {}
    for l in (open(limits, errors="replace").read().splitlines() if os.path.exists(limits) else []):
        f = l.split()
        try:
            c = float(f[1]) / 1000
        except (IndexError, ValueError):
            continue
        if 40 <= c <= TEMP_MAX:
            lim[f[0]] = c
    names = {}
    for l in (open(stages, errors="replace").read().splitlines() if os.path.exists(stages) else []):
        f = l.split("\t")
        if len(f) >= 3 and not l.startswith("#"):
            names[f[1]] = f"stage {f[0]} {f[2]}"
    ph = []
    for l in (open(phases, errors="replace").read().splitlines() if os.path.exists(phases) else []):
        f = l.split()
        try:
            ph.append((float(f[-1]), names.get(f[0], f[0])))
        except (IndexError, ValueError):
            continue
    ph.sort()

    def stage(t):
        s = "before stage 1"
        for t0, n in ph:
            if t0 <= t:
                s = n
        return s
    peak, post, junk = {}, 0, 0   # peak[(bdf, stage)] = (C, epoch)
    for l in open(tel, errors="replace").read().splitlines():
        f = l.split()
        if len(f) < 3:
            continue
        try:
            t, c = float(f[0]), float(f[2]) / 1000
        except ValueError:
            junk += f[2] != "-" if len(f) > 2 else 0
            continue
        b = f[1]
        if b in first and t >= first[b]:
            post += 1; continue
        if not 0 < c <= TEMP_MAX:
            junk += 1; continue
        k = (b, stage(t))
        if c > peak.get(k, (-1, 0))[0]:
            peak[k] = (c, t)
    over = {}
    for (b, s), (c, t) in peak.items():
        lm = lim.get(b, fallback)
        if c >= lm and (b not in over or c > over[b][0]):
            over[b] = (c, t, s, lm)
    for b, (c, t, s, lm) in sorted(over.items()):
        print(f"THERMAL FAIL {b[5:]} ({ubb(b)}): peak {c:.1f} C at or above its thermal limit {lm:g} C "
              f"({'hwmon temp1_max' if b in lim else src}) in {s}, {pdt(t)}")
    top = {}
    for (b, s), (c, t) in peak.items():
        if c > top.get(s, (-1, ""))[0]:
            top[s] = (c, b)
    st = ", ".join(f"{s} {c:.1f} C ({b[5:]}, {ubb(b)})" for s, (c, b) in sorted(top.items(), key=lambda x: min(
        t for (bb, ss), (_, t) in peak.items() if ss == x[0])))
    print(f"thermal: peak ASIC temp per stage: {st or 'no valid samples'}; limit per chip hwmon temp1_max "
          f"({len(lim)} chips), else {fallback:g} C ({src}); post-fault temps ignored: {post}; "
          f"implausible temps ignored: {junk}", file=sys.stderr)


def linkdrop(pre, post):
    def rows(p):
        out = {}
        for l in open(p).read().splitlines():
            f = l.split()
            if len(f) >= 2:
                m = re.match(r"([0-9.]*)\D*x(\d*)$", f[1])
                out[f[0]] = (f[1], float(m.group(1) or 0) if m else 0.0, int(m.group(2) or 0) if m else 0)
        return out
    a, b = rows(pre), rows(post)
    for k in sorted(a):
        if k in b and (b[k][1] < a[k][1] or b[k][2] < a[k][2]):
            print(f"PCIE FAIL {k[5:]} ({ubb(k)}): link {a[k][0]} -> {b[k][0]} during the run (speed or width down)")


def offhost(want, files):  # inter-host ETH links of this host from its discoveries; exit 0 none, 1 any, 2 unreadable
    seen, off = set(), []
    for f in files:
        try:
            d = last_json(f)
            bdfs = {int(k): v for k, v in d["bdfs"].items()}
            chips, active = [int(c) for c in d["chips"]], {int(k): set(v) for k, v in d["active"].items()}
            links = {int(c): {int(ch): r for ch, r in v.items()} for c, v in d["links"].items()}
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            print(f"link map {f} cannot be read ({type(e).__name__}: {str(e)[:80]})"); return 2
        if not chips:
            print(f"link map {f} lists no chips"); return 2
        name = lambda c: f"{bdfs[c][-7:]} ({ubb(bdfs[c])})" if c in bdfs else f"chip {c} (no PCI address here)"
        for c in chips:
            if c not in bdfs:
                off.append(f"OFFHOST {name(c)}: a chip of another host, reached over ETH ({f})"); continue
            seen.add(bdfs[c])
            for ch in sorted(active.get(c, set()) | set(links.get(c, {}))):
                r = links.get(c, {}).get(ch)
                if r is None:
                    off.append(f"OFFHOST {name(c)} e{ch}: active, its far end is not a chip of this host ({f})")
                elif int(r[0]) not in bdfs:
                    off.append(f"OFFHOST {name(c)} e{ch}: linked to chip {r[0]} e{r[1]}, not a chip of this host ({f})")
    for l in dict.fromkeys(off):
        print(l)
    if off:
        print(f"inter-host ETH links: {len(dict.fromkeys(off))} channel(s) lead off this host"); return 1
    if want and len(seen) < want:
        print(f"link map incomplete: {len(seen)} of {want} chips read in {len(files)} discovery file(s)"); return 2
    print(f"no inter-host links: {len(seen)} chips read in {len(files)} discovery file(s), every active ETH channel ends on a chip of this host")
    return 0


def link_design(spec):  # "32.0x1 6:32.0x8" -> ((32.0, 1), {6: (32.0, 8)})
    base, slots = None, {}
    for t in spec.split():
        k, _, v = t.rpartition(":")
        g, _, w = v.partition("x")
        if k:
            slots[int(k)] = (float(g), int(w))
        else:
            base = (float(g), int(w))
    return base, slots


def hostlink(snap, design, host, pre=None):
    """snap rows (lib/link_check.sh lk_snap): bdf rp bus vendor speed width rp_speed rp_width; '-' = no file"""
    auto = design.strip() == "auto"  # the reference per slot is the host's own majority, not a fixed design
    base, slots = (None, {}) if auto else link_design(design)
    lab = "slot majority" if auto else "design"

    def rows(p):
        out = {}
        for l in open(p).read().splitlines():
            f = l.split("\t")
            if len(f) == 8:
                out[f[0]] = f
        return out

    def link(g, w):  # (GT/s, width) or None when LnkSta did not read back (ffff: "Unknown", width 63; no file)
        try:
            g, w = float(g), int(w)
        except ValueError:
            return None
        return (g, w) if 0 < w < 32 else None

    def txt(l):
        return f"{l[0]:g}GT/s x{l[1]}" if l else "unread"

    a = rows(pre) if pre else {}
    if auto:  # per slot position (bus & 0x3f), the most common answering link over the host's chips (pre snap when given);
        # a tie goes to the faster/wider link. Slurm cluster Blackhole galaxies train every chip at 16 GT/s, quads 1-4 at 32
        n = {}
        for b, f in (a or rows(snap)).items():
            l = link(f[4], f[5]) if f[2] == "on" and f[3] == "1e52" else None
            if l:
                k = n.setdefault(int(b.split(":")[-2], 16) & 0x3F, {})
                k[l] = k.get(l, 0) + 1
        slots = {sl: max(c, key=lambda l: (c[l], l)) for sl, c in n.items()}
    for b, f in sorted(rows(snap).items(), key=lambda x: (ubb(x[0]), x[0])):  # tray order
        rp = f[1] if f[1] != "-" else "not found"
        want = slots.get(int(b.split(":")[-2], 16) & 0x3F, base)
        ep, rpl = link(f[4], f[5]), link(f[6], f[7])
        rpt = "down" if f[7] == "0" else txt(rpl)
        who = f"PCIE %s {b[5:]} ({ubb(b)}) endpoint, root port {rp}, host {host}"
        if f[2] != "on":
            print(who % "FAIL" + f": off PCIe (not on the bus); root port link {rpt}, {lab} {txt(want)}")
            continue
        if f[3] != "1e52":
            print(who % "FAIL" + f": off PCIe (config space {'unread' if f[3] in ('-', 'unread') else 'reads ' + f[3]});"
                  f" root port link {rpt}, {lab} {txt(want)}")
            continue
        if ep is None:
            print(who % "FAIL" + f": link not trained (LnkSta unreadable: {f[4]} x{f[5]}); root port link {rpt},"
                  f" {lab} {txt(want)}")
            continue
        for side, l in (("link", ep), ("root port link", rpl)):
            if l is None or want is None or l == want:
                continue
            slow, narrow = l[0] < want[0], l[1] < want[1]
            if slow or narrow:
                print(who % "FAIL" + f": {side} {txt(l)}, {lab} {txt(want)}"
                      f" ({' and '.join(x for x, y in (('slower', slow), ('narrower', narrow)) if y)})")
            else:
                print(who % "WARN" + f": {side} {txt(l)}, {lab} {txt(want)} (above the {lab})")
        if b in a:
            was = link(a[b][4], a[b][5])
            if was and (ep[0] < was[0] or ep[1] < was[1]) and not (want and (ep[0] < want[0] or ep[1] < want[1])):
                print(who % "FAIL" + f": link {txt(was)} -> {txt(ep)} during the run (speed or width down)")


def main(a):
    if a[0] == "disc":
        import tt_umd as u
        cd = u.TopologyDiscovery.create_cluster_descriptor()
        conn = cd.get_ethernet_connections()
        print(json.dumps({"chips": sorted(cd.get_all_chips()), "bdfs": {int(k): v for k, v in cd.get_chip_pci_bdfs().items()},
                          "active": {int(c): sorted(cd.get_active_eth_channels(c)) for c in cd.get_all_chips()},
                          "links": {int(c): {int(ch): [int(rc), int(rch)] for ch, (rc, rch) in sorted(v.items())}
                                    for c, v in conn.items()}}), flush=True)
    elif a[0] == "count":
        print(count(last_json(a[1])))
    elif a[0] == "pairs":
        m = {b: n for b, n in (l.split()[:2] for l in open(a[2]) if l.strip())}
        skip = {int(x) for x in a[3].split(",") if x} if len(a) > 3 else set()
        pairs(last_json(a[1]), m, skip)
    elif a[0] == "aer":
        aer(a[1], a[2:])
    elif a[0] == "aerrate":   # <sysfs> <state.json> <dmesg file> <epoch> <warn/min> <bdf=tray>...
        aerrate(a[1], a[2], a[3], int(a[4]), float(a[5]), a[6:], _fail(os.environ.get("FLEET_AER_STAGE_FAIL") or "0"),
                os.environ.get("FLEET_AER_HOST") or "?")
    elif a[0] == "aerrate-judge":   # <rows file> <warn/min> <stage name>
        aerrate_judge(a[1], float(a[2]), a[3])
    elif a[0] == "aerhang":   # <rows file> <stall epoch> [host]
        aerhang(a[1], int(a[2]), a[3] if len(a) > 3 else "?")
    elif a[0] == "aerdelta":
        aerdelta(a[1], a[2], float(a[3]), float(a[4]), int(a[5]) if len(a) > 5 else 2, a[6] if len(a) > 6 else "?",
                 a[7] if len(a) > 7 else "during the run", _fail(a[8]) if len(a) > 8 else 0)
    elif a[0] == "aerpre":
        aerpre(a[1], a[2] if len(a) > 2 else "?")
    elif a[0] == "hostlink":
        hostlink(a[1], a[2], a[3], a[4] if len(a) > 4 else None)
    elif a[0] == "journal":
        journal(a[1], a[2], a[3], int(a[4]), int(a[5]), *((float(a[6]), a[7]) if len(a) > 7 else ()))
    elif a[0] == "aerrun":   # <journal> <aer.tsv> <t0> <t1> <rate_warn> [min] [findings]
        aerrun(a[1], a[2], float(a[3]), float(a[4]), float(a[5]), int(a[6]) if len(a) > 6 else 2, a[7] if len(a) > 7 else None)
    elif a[0] == "sel":
        sel(a[1], float(a[2]), int(a[3]), int(a[4]), *((float(a[5]), a[6]) if len(a) > 6 else ()))
    elif a[0] == "hist1":   # <history file>, lines on stdin
        with open(a[1], "a") as k:
            hist1(sys.stdin, k)
    elif a[0] == "chain":
        sn = (float(a[4]),) if a[3] == "--since" else ()   # chain aer.tsv runstart [--since epoch] journal...
        f18chain(a[1], float(a[2]), a[5:] if sn else a[3:], *sn)
    elif a[0] == "histinfo":
        histinfo(a[1], a[2:])
    elif a[0] == "selnew":
        selnew(a[1], a[2], a[3])
    elif a[0] == "sdr":
        sdr(a[1])
    elif a[0] == "ethdown":
        ethdown(a[1], a[2])
    elif a[0] == "ethdown2":
        ethdown2(a[1], a[2], a[3], a[4] if len(a) > 4 else None)
    elif a[0] == "ethlinks":
        kw = {"host": None, "since": None}
        while len(a) > 5 and a[-2] in ("--host", "--since", "--smi", "--named", "--pre", "--pres", "--nc"):
            k = a[-2][2:]; kw[k] = a[-1].upper() if k == "host" else int(a[-1]) if k == "since" else a[-1]; a = a[:-2]
        ethlinks(last_json(a[1]), a[2], a[3], peerstate(a[4]) if len(a) > 4 else {}, **kw)
    elif a[0] == "topo":   # topo <disc.json> <eth_map.txt> [--host H] -> "ring: ..." or "linear: ..."
        topo(last_json(a[1]), a[2], a[4].upper() if a[3:4] == ["--host"] else None)
    elif a[0] == "ethback":
        ethback(a[1], a[2], last_json(a[3]), a[4], a[5], a[6] if len(a) > 6 else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "eth_map.txt"), a[7].upper() if len(a) > 7 else None)
    elif a[0] == "ethdiff":
        ethdiff(last_json(a[1]), last_json(a[2]))
    elif a[0] == "ethboth":
        o = {"--host": None, "--held": None}; i = 4
        while i + 1 < len(a) and a[i] in o:
            o[a[i]] = a[i + 1]; i += 2
        ethboth(a[1], a[2], a[3], o["--host"] and o["--host"].upper(), a[i + 1:] if a[i:i + 1] == ["--map"] else None,
                o["--held"])
    elif a[0] == "attr":
        return attr(a[1], a[2:])
    elif a[0] == "ethcore":
        return ethcore(a[1], a[2], a[3:])
    elif a[0] == "c5one":
        return c5one(*a[1:6])
    elif a[0] == "devmap":
        return devmap(a[1], a[2], a[3:])
    elif a[0] == "hangpin":
        return hangpin(*a[1:5])
    elif a[0] == "ethfec":
        return ethfec(*a[1:5])
    elif a[0] == "ethsnap":
        return ethsnap(a[1])
    elif a[0] == "telem":
        telem(a[1], *(float(x) for x in a[2:6]), *a[6:7])
    elif a[0] == "thermal":
        thermal(a[1], a[2], a[3], a[4], a[5], float(a[6]), a[7])
    elif a[0] == "linkdrop":
        linkdrop(a[1], a[2])
    elif a[0] == "offhost":
        return offhost(int(a[1]), a[2:])
    else:
        print(__doc__, file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
