#!/usr/bin/env bash
# ethchk-upstream-tests.sh - Ethernet link check for a Tenstorrent host, in one run. Run it on the host itself:
#
#   bash ethchk-upstream-tests.sh
#
# 1. PCIe first: if a chip has dropped off PCIe it stops there, without touching the chips (don't reset: report it).
# 2. Reads every eth link twice, 30 s apart, with tt-metal triage in Docker. Raw output: ~/ethchk-runs/ethchk.<host>.<utc>/
#    (kept across reboots, so runs before and after a fix can be compared; older runs left in /tmp are read too)
# 3. One report with a verdict, and what changed since the previous run on this host.
#      PROBLEM  wrong on any host: a chip missing, a link down, errors rising, a link retraining, a built-in link
#               (board trace, linking board) not linked, a cable with only part of its lanes linked
#      WARNING  probably wrong, but it cannot be proven from here: a port linked on some trays and not on the others,
#               links that were up in the previous run and are gone
#      HEALTHY  none of the above
#    The cabling (which QSFP ports are linked inside the host) is described, never judged. It differs from host to host
#    and changes over time, so there is no setting for it and the script never expects a cable. A port that is linked on
#    no tray has no cable, a cable to another host, or dead cables: those cannot be told apart from here.
#    Right under the verdict, the LINKS line gives the link count and the cabling found, by name when this script has a
#    name for it. A cabling without a name here is said so; that is a note for the reader, never a verdict.
#    Link maps: Blackhole Galaxy, Wormhole Galaxy, QuietBox 2 (host names qb2-* or *qbge*). Any other host gets the
#    general checks only: link state, errors, changes since the last run. A Galaxy is recognised by the card type the
#    driver gives for its chips (galaxy-blackhole, galaxy-wormhole), or by 16 or more chips when the driver gives none.
#    Tests and the tables they are checked against: tests/run_tests.py (see the comment at its top).
#
# Settings (environment): IMG docker image, TRIAGE_PYTHON / TRIAGE_SCRIPT (python and triage.py inside that image),
# SAMPLE_SECS (30), ETHCHK_DIR (~/ethchk-runs), ETHCHK_FORCE=1 to run the link check
# even when a chip dropped off PCIe, UNCORR_PROBLEM_PER_S / UNCORR_WARN_PER_S / CRC_WARN_PER_S / UNCORR_WARN_TOTAL /
# RETRAIN_WARN thresholds. None of them is about the cabling.
# The default image is tt-metal's public upstream-tests image: no docker login or GitHub token needed. The first run on a
# host downloads it once (about 9 GB). Only its tt-metal triage is used, so the same image should do for every host type.
# Same checks and report as ethchk.sh, which uses the private Shield CI image instead.
# Exit code: 0 healthy, 2 problem, 3 warning, 4 stopped because a chip dropped off PCIe, 5 Docker or the image unusable.

ethchk_main() {
  local img="${IMG:-ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.83.0-dev20261108}"
  local py="${TRIAGE_PYTHON:-/opt/venv/bin/python}" tri="${TRIAGE_SCRIPT:-/home/user/tt-metal/tools/triage/triage.py}"
  local d
  d="${ETHCHK_DIR:-${HOME:-/tmp}/ethchk-runs}/ethchk.$(hostname -s).$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$d" || return 1
  hostname > "$d/host"; echo "$img" > "$d/image"
  cat > "$d/ethchk.py" <<'PY'
import csv, datetime, glob, os, re, sys, time

MODE, d = sys.argv[1], sys.argv[2]
E = os.environ.get
SYS = E("ETHCHK_SYSFS", "/sys")  # tests point this at a fake tree
UP, UW, CW = float(E("UNCORR_PROBLEM_PER_S", 100)), float(E("UNCORR_WARN_PER_S", 10)), float(E("CRC_WARN_PER_S", 1))
UT, RW = int(E("UNCORR_WARN_TOTAL", 1000000)), int(E("RETRAIN_WARN", 3))
LOC = re.compile(r"(\d+-\d+)(?:\s*\(e\d+,(\d+)\))?")
P, W = "PROBLEM", "WARNING"
IDLE = ("unused", "disabled", "not connected")
VERSION = "version 3.1, 2026-11-15"
# Every table below comes from tt-metal or tt-umd source, and tests/run_tests.py checks each one against that source
# (last checked against tt-metal 2c00273ebb38 of 2026-11-14 and tt-umd v0.9.12). Nothing that is judged was measured on
# one host. The one exception is a name: SEEN_CABLINGS below, used only to name the cabling that is found.
# Galaxy layout. Tray = PCI bus high nibble, ASIC slot = low nibble (tt-umd tools/system_health.cpp, ubb_bus_ids).
TRAY_OF_BUS = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}
WH_TRAY_OF_BUS = {0xC0: 1, 0x80: 2, 0x00: 3, 0x40: 4}
# Blackhole Galaxy: port -> [(ASIC slot, eth channel)] on one tray (tt-metal tools/scaleout/board/board.cpp, UBB_BLACKHOLE).
PORTS = {
    ("QSFP_DD", 1): [(5, 2), (5, 3)], ("QSFP_DD", 2): [(1, 2), (1, 3)], ("QSFP_DD", 3): [(1, 0), (1, 1)],
    ("QSFP_DD", 4): [(2, 0), (2, 1)], ("QSFP_DD", 5): [(3, 0), (3, 1)], ("QSFP_DD", 6): [(4, 0), (4, 1)],
    ("QSFP_DD", 7): [(1, 8), (2, 8)], ("QSFP_DD", 8): [(5, 8), (6, 8)], ("QSFP_DD", 9): [(3, 8), (4, 8)],
    ("QSFP_DD", 10): [(7, 8), (8, 8)], ("QSFP_DD", 11): [(1, 9), (2, 9)], ("QSFP_DD", 12): [(5, 9), (6, 9)],
    ("QSFP_DD", 13): [(3, 9), (4, 9)], ("QSFP_DD", 14): [(7, 9), (8, 9)],
    ("LINKING_BOARD_1", 1): [(5, 6), (5, 7)], ("LINKING_BOARD_1", 2): [(6, 6), (6, 7)],
    ("LINKING_BOARD_2", 1): [(7, 6), (7, 7)], ("LINKING_BOARD_2", 2): [(8, 6), (8, 7)],
    ("LINKING_BOARD_3", 1): [(8, 4), (8, 5)], ("LINKING_BOARD_3", 2): [(4, 4), (4, 5)],
    ("TRACE", 1): [(5, 0), (5, 1)], ("TRACE", 2): [(1, 6), (1, 7)], ("TRACE", 3): [(5, 4), (5, 5)],
    ("TRACE", 4): [(6, 2), (6, 3)], ("TRACE", 5): [(1, 4), (1, 5)], ("TRACE", 6): [(2, 2), (2, 3)],
    ("TRACE", 7): [(6, 0), (6, 1)], ("TRACE", 8): [(2, 6), (2, 7)], ("TRACE", 9): [(6, 4), (6, 5)],
    ("TRACE", 10): [(7, 2), (7, 3)], ("TRACE", 11): [(2, 4), (2, 5)], ("TRACE", 12): [(3, 2), (3, 3)],
    ("TRACE", 13): [(7, 0), (7, 1)], ("TRACE", 14): [(3, 6), (3, 7)], ("TRACE", 15): [(7, 4), (7, 5)],
    ("TRACE", 16): [(8, 2), (8, 3)], ("TRACE", 17): [(3, 4), (3, 5)], ("TRACE", 18): [(4, 2), (4, 3)],
    ("TRACE", 19): [(8, 0), (8, 1)], ("TRACE", 20): [(4, 6), (4, 7)],
}
PORT_OF = {ac: p for p, acs in PORTS.items() for ac in acs}
KIND = {"QSFP_DD": "cable", "TRACE": "on-board trace, nothing to reseat"}  # anything else is a linking board
# Two kinds of links. Built-in links (board traces, linking boards) are on every Galaxy: no cable choice is involved, so
# one that is not linked is a fault on any host. Cable links (QSFP ports) depend on how the host was cabled. That differs
# from host to host and changes over time, so this script has no setting for the cabling and never expects a cable.
QSFP = sorted(pid for t, pid in PORTS if t == "QSFP_DD")
# Names for the cabling that is found, from tt-metal's own cablings of one Galaxy (tools/scaleout/node/node.cpp:
# Y torus = QSFP 1-2, X torus = QSFP 3-6 on every tray; the Z ports 7-14 are in
# tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus_z_ports.textproto). Used to describe, never to judge.
CABLINGS = {(): "mesh", (1, 2): "y-torus", (3, 4, 5, 6): "x-torus", (1, 2, 3, 4, 5, 6): "xy-torus",
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14): "xy-torus+z"}
GROUPS = (((1, 2), "Y torus"), ((3, 4, 5, 6), "X torus"), ((7, 8, 9, 10, 11, 12, 13, 14), "Z ports"))
CABLING_TEXT = {"mesh": "looks like a mesh: no torus cables linked", "y-torus": "looks like tt-metal's Y torus", "x-torus": "looks like tt-metal's X torus",
                "xy-torus": "looks like tt-metal's XY torus", "xy-torus+z": "looks like tt-metal's XY torus plus the Z ports"}
# One more name, not from tt-metal: a Galaxy inside a sub-torus cluster, as seen on real cluster hosts (QSFP 1-5, 9, 10,
# 13, 14 linked inside the host; 6, 7, 8, 11, 12 go to other hosts). Like the names above: to describe, never to judge.
SEEN_CABLINGS = {(1, 2, 3, 4, 5, 9, 10, 13, 14): "subtorus-cluster"}
CABLING_TEXT["subtorus-cluster"] = "looks like a Galaxy inside a sub-torus cluster (QSFP 6-8, 11, 12 go to other hosts there)"
CABLING_AS = {"mesh": "as in a mesh", "y-torus": "as in a Y torus", "x-torus": "as in an X torus", "xy-torus": "as in an XY torus",
              "xy-torus+z": "as in an XY torus plus the Z ports", "subtorus-cluster": "as on a Galaxy inside a sub-torus cluster"}
CABLING_SHORT = {"mesh": "mesh", "y-torus": "Y torus", "x-torus": "X torus", "xy-torus": "XY torus", "xy-torus+z": "all 14 ports",
                 "subtorus-cluster": "sub-torus cluster"}
_BUILTIN = 4 * sum(len(v) for (t, _), v in PORTS.items() if t != "QSFP_DD")  # on the 4 trays of a Blackhole Galaxy
# "mesh 208, Y torus 224 ...": the on-host links of a healthy Blackhole Galaxy in each cabling this script has a name for.
KNOWN = ", ".join(f"{CABLING_SHORT[n]} {_BUILTIN + 4 * sum(len(PORTS['QSFP_DD', p]) for p in k)}"
                  for k, n in sorted({**CABLINGS, **SEEN_CABLINGS}.items(), key=lambda kv: len(kv[0])))
NOT_NAMED = ("If this host is meant to be one of those, cables are missing or dead. Only the people who cabled it know.")
# Wormhole Galaxy (UBB_WORMHOLE in board.cpp): every connector carries 4 links. Built-in links per ASIC slot 1 to 8, then
# the slots that get 4 more from a Y-torus cable (QSFP 1-2) and from an X-torus cable (QSFP 3-6).
WH_BUILTIN = (8, 12, 12, 12, 12, 16, 16, 16)
WH_Y = (5, 1)
WH_X = (1, 2, 3, 4)
# QuietBox 2 (tt-metal node P300_QB_GE: two P300 boards joined by one WARP400 cable inside the box): 4 links per chip, all built in.
QB2_LINKS = 4
CANON = [hi | lo for hi in TRAY_OF_BUS for lo in range(1, 9)]  # bus of dev N if sysfs can't be read (tray order)
COUNTERS = ("retrain", "crc", "corr", "uncorr")  # only ever go up until the chips are reset


def rd(n, base=None):
    try:
        return open(os.path.join(base or d, n)).read().strip()
    except OSError:
        return None


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def tables(path):
    t, cur = {}, None
    try:
        text = open(path).read()
    except OSError:
        return t
    for line in text.splitlines():
        m = re.match(r"([a-z_]+)\.py\b.*:$", line)
        if m: cur = t[m.group(1)] = [None, []]
        elif cur is None or not line.strip() or line[0].isspace(): continue
        elif cur[0] is None: cur[0] = [v.strip() for v in next(csv.reader([line]))]
        else: cur[1].append([v.strip() for v in next(csv.reader([line]))])
    return t


def num(v):
    try: return int(v, 16) if v.lower().startswith("0x") else int(v)
    except (AttributeError, ValueError): return None


def links(t):
    hdr, rows = t.get("check_eth_status", [None, []]); out = {}
    col = {h.lower(): i for i, h in enumerate(hdr or [])}
    for r in rows:
        dev = re.match(r"\d+", r[0]) if r else None
        loc = next((m for m in map(LOC.fullmatch, r[1:]) if m), None)
        if not dev or not loc: continue
        g = lambda k: r[col[k]] if k in col and col[k] < len(r) else None
        dv = int(dev.group())
        out[dv, loc.group(1)] = dict(name=f"dev {dv} core {loc.group(1)}" + (f" ch{loc.group(2)}" if loc.group(2) else ""),
            dev=dv, ch=int(loc.group(2)) if loc.group(2) else None, port=g("port status"), link=g("link"),
            hb=g("heartbeat"), rx=g("rx link up"), retrain=num(g("retrain count")), crc=num(g("crc errors")),
            corr=num(g("corr cw")), uncorr=num(g("uncorr cw")), errstat=num(g("err_stat")))
    return out


def dev_bdfs():
    m = {}
    for p in glob.glob(os.path.join(SYS, "class/tenstorrent/tenstorrent!*")):
        try: m[int(p.rsplit("!", 1)[1])] = os.path.basename(os.path.realpath(os.path.join(p, "device")))
        except (ValueError, OSError): pass
    return m


def pcie_state():
    """[(bdf, vendor id read live, PCI Command register)] for every Tenstorrent PCI function."""
    rows = []
    for p in sorted(glob.glob(os.path.join(SYS, "bus/pci/devices/*"))):
        try:
            if open(os.path.join(p, "vendor")).read().strip() != "0x1e52": continue
            c = open(os.path.join(p, "config"), "rb").read(8)
        except OSError: continue
        rows.append((os.path.basename(p), int.from_bytes(c[0:2], "little"), int.from_bytes(c[4:6], "little")))
    return rows


def card_types(skip=()):
    """{card type: number of chips}, as the driver gives it for each chip (tt-kmd sysfs attribute tt_card_type)."""
    out, t0 = {}, time.time()
    for p in sorted(glob.glob(os.path.join(SYS, "class/tenstorrent/tenstorrent!*"))):
        if time.time() - t0 > 5: break  # a slow read must not hold up the report
        try:
            if os.path.basename(os.path.realpath(os.path.join(p, "device"))) in skip: continue
            v = open(os.path.join(p, "tt_card_type")).read().strip().lower()
        except (OSError, ValueError): continue
        if v: out[v] = out.get(v, 0) + 1
    return out


BDFS = dev_bdfs()  # /dev/tenstorrent/N -> PCI address; N has gaps when chips are gone (e.g. 0-15, 24-31)
def pos(bdf):
    try:
        bus = int(bdf.split(":")[1], 16)
        return TRAY_OF_BUS.get(bus & 0xF0), bus & 0x0F
    except (IndexError, ValueError):
        return None, None
def place(dv): tray, asic = pos(bdf_of(dv)); return f"T{tray} ASIC{asic}"
def port_of(dv, ch): return PORT_OF.get((pos(bdf_of(dv))[1], ch), ("?", "?"))
def wh_pos(bdf):
    try:
        bus = int(bdf.split(":")[1], 16)
        return WH_TRAY_OF_BUS.get(bus & 0xF0), bus & 0x0F
    except (IndexError, ValueError):
        return None, None


host = rd("host")
pc = pcie_state()
# Triage and tt-smi ("UMD Chip ID") number the chips 0, 1, 2 ... in /dev/tenstorrent order, skipping the gaps:
# e.g. with tray 3 unpowered, chip 16 = /dev/tenstorrent/24 = 0000:81:00.0. Without sysfs, assume tray order.
def _rank(b):
    try: bus = int(b.split(":")[1], 16)
    except (IndexError, ValueError): return 999
    return CANON.index(bus) if bus in CANON else 500 + bus
SEEN = bool(BDFS or pc)  # chip addresses were read from this host
ORDER = [b for _, b in sorted(BDFS.items())] or sorted((b for b, _, _ in pc), key=_rank) or [f"0000:{x:02x}:00.0" for x in CANON]
def bdf_of(dv): return ORDER[dv] if 0 <= dv < len(ORDER) else "?"
# Blackhole Galaxy = 16+ Tenstorrent PCI functions, all Blackhole (device id 0xb140). Tray / ASIC names apply only there.
GX = len(pc) >= 16 and all(rd("device", os.path.join(SYS, "bus/pci/devices", b)) == "0xb140" for b, _, _ in pc)
pbad = [(b, v, c) for b, v, c in pc if v == 0xFFFF or not c & 2]
devof = {b: n for n, b in enumerate(ORDER)}
pmsg = [f"dev {devof.get(b, '?')} {b}" + (f" (T{pos(b)[0]} ASIC{pos(b)[1]})" if GX else "") + ": " + ("not answering on PCIe"
        if v == 0xFFFF else f"dropped off PCIe (Command=0x{c:04x}, memory decoding off)") for b, v, c in pbad]
if MODE == "pcie":
    if pbad and E("ETHCHK_FORCE"):
        print("\n  PCIe: a chip dropped off PCIe - running the link check anyway (ETHCHK_FORCE is set)")
    elif pbad:
        print(f"\n================ TT ETHERNET LINK REPORT ================\n  host {host}   utc "
              f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n\n"
              "  VERDICT: PROBLEM - a chip dropped off PCIe. Link check NOT run, so the chips are left as they are.")
        for m in pmsg: print(f"    - {m}")
        print("\n  Do NOT reset yet: report the device above first (a reset wipes the evidence)."
              + "\n  To run the link check anyway: ETHCHK_FORCE=1 bash ethchk-upstream-tests.sh\n")
        sys.exit(4)
    sys.exit(0)

s1, s2 = tables(os.path.join(d, "sample1.csv")), tables(os.path.join(d, "sample2.csv"))
l1, l2 = links(s1), links(s2)
if not l1:
    log = (rd("sample1.csv.log") or "").splitlines()
    low = "\n".join(log).lower()
    hint = ("Docker needs root on this host: run it again as  sudo bash ethchk-upstream-tests.sh" if "permission denied" in low and "docker" in low
            else "The image is not on this host and could not be downloaded. It is public, so if the log says denied or\n"
                 "  unauthorized, an old ghcr.io login is saved here: run  docker logout ghcr.io  and try again. Images on this host:\n"
                 "    docker images | grep upstream-tests\n  then pick one:  IMG=<REPOSITORY>:<TAG> bash ethchk-upstream-tests.sh"
            if any(s in low for s in ("unable to find image", "unauthorized", "denied", "manifest unknown", "pull access", "not found: manifest"))
            else "Triage itself failed - the lines above say why (a hung chip shows up as a timeout).")
    sys.exit(f"\n  ERROR: the link check printed no link table. Last lines of {d}/sample1.csv.log:\n"
             + ("\n".join("    " + x[:200] for x in log[-15:]) or "    (empty)") + f"\n\n  {hint}\n")
now, t1, t2 = l2 or l1, fnum(rd("t1")), fnum(rd("t2"))
dt = t2 - t1 if l2 and t1 and t2 else None
F = [(P, f"{m} - do NOT reset, report it") for m in pmsg]
if rd("sample2.csv") is not None and not l2: F.append((W, f"second read listed no links, judged one read only - see {d}/sample2.csv.log"))
ys = [k[1].split("-")[1] for k in now]
arch = "blackhole" if ys.count("1") * 2 > len(ys) else "wormhole"
per = {}
def live(lks): return [l for l in lks.values() if (l["port"] or "").lower() not in IDLE]  # an idle port is not a link
for l in live(now): per[l["dev"]] = per.get(l["dev"], 0) + 1
# What kind of host is this? Only these three have a link map in this script. Any other host gets the general checks only.
# A Galaxy is recognised from the hardware: the driver gives the card type of each chip (read on real hosts as
# galaxy-blackhole, galaxy-wormhole and, on a QuietBox 2, p300c). So a Galaxy that lost most of its chips is still a Galaxy
# with chips missing. The card type can only add: when the driver gives none, or a name this script does not know, 16 or
# more chips still mean a Galaxy, so it never turns a check off. Read here, after triage: the PCIe gate reads PCI config only.
cards = card_types({b for b, _, _ in pbad})  # a chip that dropped off PCIe is not read
galaxy = "galaxy-blackhole" in cards or (arch == "blackhole" and len(per) >= 16)  # Blackhole Galaxy, 32 chips
whgx = not galaxy and ("galaxy-wormhole" in cards or (arch == "wormhole" and len(per) >= 16))  # Wormhole Galaxy, 32 chips
qb2 = arch == "blackhole" and not galaxy and re.match(r"qb2-|.*qbge", host or "") is not None  # QuietBox 2 (host names qb2-* / *qbge*)
kind = "bh-galaxy" if galaxy else "wh-galaxy" if whgx else "qb2" if qb2 else None
want = {"bh-galaxy": 32, "wh-galaxy": 32, "qb2": 4}.get(kind)  # chips a healthy host of this kind has
pmiss = bool(pc) and want is not None and len(pc) < want
if pmiss:
    there = {b for b, _, _ in pc}
    lost = [f"0000:{x:02x}:00.0" for x in CANON if f"0000:{x:02x}:00.0" not in there] if galaxy else []
    trays = {}
    for b in lost: trays.setdefault(pos(b)[0], []).append(b)
    gone_txt = "; ".join(f"all of T{t} ({bs[0][5:7]}-{bs[-1][5:7]}:00.0)" if len(bs) == 8
                         else ", ".join(f"{b[5:]} (T{t} ASIC{pos(b)[1]})" for b in bs) for t, bs in trays.items())
    holes = sorted(set(range(max(BDFS) + 1)) - set(BDFS)) if BDFS else []  # gaps in /dev/tenstorrent numbering
    F.append((P, f"only {len(pc)} of {want} chips on PCIe - " + (f"missing: {gone_txt}" if lost
              else "missing: /dev/tenstorrent/" + ", ".join(map(str, holes)) if holes
              else "a chip is missing (present: " + ", ".join(b[5:] for b in sorted(there)) + ")" if len(there) <= 8
              else "a chip is missing - tt-smi -ls shows which ones are there")))
def tr(ts): return ", ".join(f"T{t}" for t in sorted(ts))
def ranges(ns):  # (1, 2, 4, 5, 6) -> "1,2,4-6"
    ns, out, i = sorted(ns), [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        out.append(str(ns[i]) if j == i else f"{ns[i]},{ns[j]}" if j == i + 1 else f"{ns[i]}-{ns[j]}")
        i = j + 1
    return ",".join(out)
def lab(dv): return (f" ({place(dv)})" if galaxy else f" ({bdf_of(dv)})") if SEEN and dv < len(ORDER) else ""

# Which links are there is judged only when every chip is there. With a chip missing or off PCIe, the links to it are not
# linked either, and that says nothing new: the chip is reported on its own and the links are checked again after the fix.
whole = not pmiss and not pbad and (bool(pc) or (want is not None and len(per) == want))
linked = {(l["dev"], l["ch"]) for l in live(now)}
# Blackhole Galaxy: tray / ASIC slot of every chip comes from its PCI address, read from this host. Never assumed.
mapped = galaxy and SEEN and all(l["ch"] is not None for l in now.values())
slot = {}  # (tray, ASIC slot) -> chip number
if mapped:
    for dv in range(len(ORDER)):
        t, a = pos(bdf_of(dv))
        if t and a: slot[t, a] = dv
judged = mapped and whole and len(slot) == 32
why_not = ("chips are missing from PCIe" if pmiss else "a chip dropped off PCIe" if pbad
           else "/sys could not be read, so the place of each chip is not known" if not SEEN
           else "triage printed no channel numbers" if galaxy and not mapped
           else "the PCI addresses of the chips are not the usual ones" if galaxy and len(slot) != 32 else "not every chip listed links")
bi_have = bi_total = 0
half, uneven, spots, pstate, cabling = [], [], [], {}, None
if judged:
    for (ptype, pid), acs in sorted(PORTS.items()):
        for t in (1, 2, 3, 4):
            off = [(slot[t, a], ch) for a, ch in acs if (slot[t, a], ch) not in linked]
            if ptype == "QSFP_DD":
                pstate.setdefault(pid, {})[t] = len(acs) - len(off)
                if off and len(off) < len(acs):  # proven: the lanes of one port go through one cable
                    half.append((t, pid)); spots += off
                    F.append((P, f"T{t} QSFP_DD P{pid}: only {len(acs) - len(off)} of {len(acs)} lanes linked - the cable is there, part of it is dead"))
            else:
                bi_total += len(acs); bi_have += len(acs) - len(off)
                if off:  # proven: every Galaxy has this link, whatever its cabling
                    spots += off
                    F.append((P, f"T{t} {ptype} P{pid}: {len(off)} of {len(acs)} built-in links not linked ({KIND.get(ptype, 'linking board')})"))
    for pid in QSFP:
        n = len(PORTS["QSFP_DD", pid])
        full, none = [t for t, k in pstate[pid].items() if k == n], [t for t, k in pstate[pid].items() if k == 0]
        if full and none:  # probable, not proven: nothing here says how this host is meant to be cabled
            uneven.append(pid); spots += [(slot[t, a], ch) for t in none for a, ch in PORTS["QSFP_DD", pid]]
            F.append((W, f"QSFP_DD port {pid}: linked on {tr(full)}, not on {tr(none)} - trays are normally cabled alike, so a cable is probably dead or missing"))
    even = tuple(p for p in QSFP if all(k == len(PORTS["QSFP_DD", p]) for k in pstate[p].values()))
    cabling = "irregular" if half or uneven else CABLINGS.get(even) or SEEN_CABLINGS.get(even) or "ports:" + ranges(even)

# Wormhole Galaxy: no port map here yet, so the links are judged per chip. Each connector carries 4 links, every chip has at
# least 8 built-in ones, and the four trays show the same set of per-chip link counts when they are cabled alike. Only the
# tray of each chip is used for that (PCI bus high nibble), not its place on the tray.
wh_bad, wh_diff, wh_rows = [], [], {}
if whgx and whole:
    counts = [per.get(dv, 0) for dv in range(want)]
    for dv, n in enumerate(counts):
        if n % 4:
            wh_bad.append(dv); F.append((P, f"dev {dv}{lab(dv)}: {n} links - a Wormhole Galaxy connector carries 4, so one of them is linked only in part"))
        elif n < min(WH_BUILTIN):
            wh_bad.append(dv); F.append((P, f"dev {dv}{lab(dv)}: {n} links - fewer than the {min(WH_BUILTIN)} built-in links every chip has"))
    if SEEN and len(ORDER) == want:
        for dv in range(want):
            t = wh_pos(bdf_of(dv))[0]
            if t: wh_rows.setdefault(t, []).append((counts[dv], dv))
        if sorted(len(v) for v in wh_rows.values()) != [8, 8, 8, 8]: wh_rows = {}
        # The fullest tray is the yardstick: links can go missing, they cannot appear.
        best = [max(ns) for ns in zip(*(sorted(n for n, _ in v) for v in wh_rows.values()))]
        low = [(t, dv, n, b) for t, v in sorted(wh_rows.items()) for (n, dv), b in zip(sorted(v), best) if n < b and not n % 4]
        wh_diff = sorted({t for t, _, _, _ in low})
        if low:  # probable, not proven: nothing here says how this host is meant to be cabled
            F.append((W, f"{tr(wh_diff)}: fewer links than on {tr(set(wh_rows) - set(wh_diff)) or 'the fullest tray'} ("
                         + ", ".join(f"dev {dv}: {n}, not {b}" for _, dv, n, b in low)
                         + ") - trays are normally cabled alike, so a cable is probably dead or missing"))
    def wh_counts(y, x): return sorted(4 * [b + 4 * y * (s in WH_Y) + 4 * x * (s in WH_X) for s, b in enumerate(WH_BUILTIN, 1)])
    cabling = next((nm for nm, y, x in (("mesh", 0, 0), ("y-torus", 1, 0), ("x-torus", 0, 1), ("xy-torus", 1, 1))
                    if sorted(counts) == wh_counts(y, x)), "unusual")

# Every port is linked alike on all trays, but in a way this script has no name for. It can be a valid cabling, so this
# is only said in the report. It is never a finding: a name missing from the list above must not cause a false alarm.
unnamed = bool(cabling) and (cabling.startswith("ports:") or (cabling == "unusual" and not wh_bad and not wh_diff))

# QuietBox 2: all of its links are built in.
qb_short = []
if qb2 and whole:
    bi_total, bi_have = want * QB2_LINKS, sum(min(per.get(dv, 0), QB2_LINKS) for dv in range(want))
    for dv in range(want):
        if per.get(dv, 0) < QB2_LINKS:
            qb_short.append(dv); F.append((P, f"dev {dv}{lab(dv)}: {per.get(dv, 0)} of {QB2_LINKS} built-in links linked"))
down, retrained, errstat, stats = 0, 0, 0, []
for k in sorted(l1.keys() - now.keys()): down += 1; F.append((P, f"{l1[k]['name']}: link disappeared between the two reads"))
for k in sorted(now):
    l, a = now[k], (l1.get(k) if l2 else None); n, port = l["name"], (l["port"] or "").lower()
    if "unreadable" in (port, (l["link"] or "").lower()): bad = ["eth core unreadable"]
    # A port that isn't meant to be connected: "Unused" on Blackhole, "Disabled" / "Not connected" on Wormhole (same rule as triage).
    elif l["link"] is not None: bad = ["link down"] if l["link"].lower() != "up" and port not in IDLE else []  # live PCS status, newer triage
    else: bad = (["port down"] if port == "down" else []) + (["RX link down"] if (l["rx"] or "up").lower() != "up" and port not in IDLE else [])
    if (l["hb"] or "up").lower() in ("down", "false"): bad.append("eth core stopped (heartbeat not moving)")
    for b in bad: down += 1; F.append((P, f"{n}: {b}"))
    if l["port"] in ("Unknown", "Invalid", "Unsupported FW"): F.append((W, f"{n}: port status {l['port']}"))
    dl = {f: (l[f] - a[f] if a and l[f] is not None and a[f] is not None else None) for f in ("uncorr", "crc", "retrain")}
    if any(v is not None and v < 0 for v in dl.values()): dl = dict.fromkeys(dl)  # device reset between the reads
    ur = dl["uncorr"] / dt if dl["uncorr"] is not None and dt else None
    cr = dl["crc"] / dt if dl["crc"] is not None and dt else None
    stats.append((l, ur, cr))
    if ur is not None and ur >= UP: F.append((P, f"{n}: uncorrected FEC errors rising {ur:,.0f}/s - this link corrupts data"))
    elif ur is not None and ur >= UW: F.append((W, f"{n}: uncorrected FEC errors rising {ur:,.1f}/s"))
    elif ur is None and (l["uncorr"] or 0) >= UT: F.append((W, f"{n}: {l['uncorr']:,} uncorrected FEC errors since last reset - read twice to see if still rising"))
    if cr is not None and cr >= CW: F.append((W, f"{n}: CRC errors (bad frames) rising {cr:,.1f}/s"))
    if dl["retrain"]: retrained += 1; F.append((P, f"{n}: retrained {dl['retrain']}x during the check - link is flapping"))
    elif (l["retrain"] or 0) >= RW: F.append((W, f"{n}: {l['retrain']} retrains since last reset"))
    if l["errstat"]: errstat += 1; F.append((W, f"{n}: ERR_STAT {l['errstat']:#x} (latched link error)"))

# What changed since the previous run on this host?
host_s = os.path.basename(d).split(".")[1]
prev, prev_dir = {}, None
def earlier_runs():  # this host's runs in the results folder, plus older ones left in /tmp, sorted by time
    dirs = {os.path.dirname(d), E("ETHCHK_OLD_DIR", "/tmp")}  # ETHCHK_OLD_DIR: tests only
    found = {os.path.basename(p): p for q in dirs for p in glob.glob(os.path.join(q, f"ethchk.{host_s}.*"))}
    return [found[n] for n in sorted(found) if n < os.path.basename(d)]
for p in reversed(earlier_runs()):
    for nm in ("sample2.csv", "sample1.csv"):
        prev = links(tables(os.path.join(p, nm)))
        if prev: break
    if prev: prev_dir = p; break
try: open(os.path.join(d, "chips"), "w").write(str(len(pc)))  # lets the next run see whether the same chips were there
except OSError: pass
pk = {k for k, l in prev.items() if (l["port"] or "").lower() not in IDLE}
nk = {k for k, l in now.items() if (l["port"] or "").lower() not in IDLE}
pchips = num(rd("chips", prev_dir)) if prev_dir else None
# Chip numbers only line up between two runs when the same chips were there both times.
same_chips = bool(prev) and (pchips == len(pc) if pchips is not None and pc else {k[0] for k in pk} == {k[0] for k in nk})
gone = sorted(pk - nk) if same_chips and not pmiss and not pbad else []
if gone:  # probable, not proven: the host may have been recabled on purpose
    F.append((W, f"{len(gone)} link(s) that were up in the last run are not linked now: " + ", ".join(prev[k]["name"] for k in gone[:4])
              + (" ..." if len(gone) > 4 else "") + " - a fault, unless the host was recabled"))

fw, si, fw_line, mixed = s1.get("firmware_versions"), s1.get("system_info"), None, False
if fw and fw[0]:
    parts = []
    for i, h in enumerate(fw[0][1:], 1):
        vals = {}
        for r in fw[1]:
            if i < len(r): vals[r[i]] = vals.get(r[i], 0) + 1
        if len(vals) > 1: mixed = True; F.append((W, f"mixed {h} across chips: " + ", ".join(f"{v} on {c}" for v, c in sorted(vals.items(), key=lambda x: -x[1]))))
        parts.append(f"{h} {'/'.join(sorted(vals))}")
    fw_line = ", ".join(parts)
probs, warns = [m for v, m in F if v == P], [m for v, m in F if v == W]
has_ctr = any(l["uncorr"] is not None for l in now.values())
if probs: verdict, code, head = P, 2, ("PROBLEM - a chip dropped off PCIe" + (", and more is wrong" if len(probs) > len(pbad) else "")
                                     if pbad else f"PROBLEM - {want - len(pc)} chip(s) missing from PCIe" if pmiss
                                     else "PROBLEM - links broken or corrupting data") + f" ({len(probs)} found)"
elif warns: verdict, code, head = W, 3, f"WARNING - nothing proven broken, worth a look ({len(warns)} found)"
else: verdict, code, head = "HEALTHY", 0, ("HEALTHY - no fault found" + (", no errors rising" if has_ctr and dt else "")
                                          + (". The cabling is described below, not judged." if cabling
                                             else " (this kind of host has no link map in this script: only link state, errors and changes were judged)" if kind is None
                                             else f" (which links are there was not judged: {why_not})" if not whole or (galaxy and not judged) else ""))
def bullets(ms):
    for m in ms[:8]: print(f"    - {m}")
    if len(ms) > 8: print(f"    ... and {len(ms) - 8} more")
def row(a, b, c, s): print(f"  {a:<28} {b:<22} {c:<20} {s}")
def lvl(bad, warn=False): return "** PROBLEM **" if bad else "** WARNING **" if warn else "OK"
total = sum(per.values())
mur = max((s[1] for s in stats if s[1] is not None), default=None)
mcr = max((s[2] for s in stats if s[2] is not None), default=None)
munc = max((l["uncorr"] or 0 for l in now.values()), default=0)
print("\n================ TT ETHERNET LINK REPORT ================")
print(f"  host {host}   utc {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}   arch {arch}   " + (f"2 reads {dt:.0f} s apart" if dt else "1 read"))
card_s = f"card type from the driver: {', '.join(f'{v} x{n}' for v, n in sorted(cards.items()))}" if cards else "no card type from the driver"
by_card = (galaxy and "galaxy-blackhole" in cards) or (whgx and "galaxy-wormhole" in cards)
mach = ({"bh-galaxy": "Blackhole Galaxy", "wh-galaxy": "Wormhole Galaxy", "qb2": "QuietBox 2"}[kind]
        + f" ({'' if by_card else 'host name; ' if qb2 else '16 or more chips; '}{card_s})" if kind
        else f"not one this script has a link map for ({card_s})")
print(f"  machine {mach}\n  image {rd('image')}\n  ethchk {VERSION}\n\n  VERDICT: {head}")
bullets(probs or warns)
if probs and warns: print("    also worth a look:"); bullets(warns)
if pmiss: print("    => get the missing chips back first: links to them are not linked either, so check the links again after that")
if not has_ctr: bullets(["this image has no error counters: a link that is up but corrupting data cannot be seen"])
# The LINKS line: how many links, and the cabling found. It describes; it is never part of the verdict.
pad = "\n           "
if galaxy and judged:
    on_cables = sum(sum(v.values()) for v in pstate.values())
    rest = total - bi_have - on_cables
    where = f"QSFP {ranges(even)} linked on all 4 trays" if even else "No QSFP port linked inside this host"
    links_txt = (f"{total} on-host links: {bi_have if bi_have == bi_total else f'{bi_have} of {bi_total}'} built-in + {on_cables} on cables"
                 + (f" + {rest} on a channel with no port name here" if rest else "") + ". "
                 + ("The cable ports are not linked alike on every tray: see the findings above." if cabling == "irregular"
                    else f"{where}, {CABLING_AS[cabling]}." if not unnamed
                    else f"{where}.{pad}That is not a cabling this script has a name for. It has names for:{pad}{KNOWN}.{pad}{NOT_NAMED}"))
elif whgx and whole:
    wh_known = ", ".join(f"{CABLING_SHORT[n]} {sum(wh_counts(y, x))}" for n, y, x in (("mesh", 0, 0), ("y-torus", 1, 0), ("x-torus", 0, 1), ("xy-torus", 1, 1)))
    links_txt = f"{total} on-host links. " + (f"Links per chip {CABLING_AS[cabling]}." if cabling in CABLING_AS
                 else f"The links per chip fit no cabling this script has a name for ({wh_known}).{pad}{NOT_NAMED}" if unnamed
                 else "The links per chip are not alike on every tray, or not whole connectors: see the findings above.")
elif qb2 and whole: links_txt = f"{total} on-host links: {bi_have} of {bi_total} built-in. A QuietBox 2 has no cabling to choose."
else:
    links_txt = f"{total} on-host links on {len(per)} chips. " + (f"Which links are there was not judged: {why_not}." if kind
                 else "No link map for this kind of host: how many chips and links it should have was not judged.")
print("  LINKS:   " + links_txt)
print("\n  CHECK                        RESULT                 EXPECTED             STATUS\n  " + "-" * 82)
npc = max(len(pc), want or 0)
if pc: row("chips on PCIe", f"{len(pc) - len(pbad)} of {npc} ok", str(npc), lvl(pbad or pmiss))
else: row("chips on PCIe", "can't read sysfs", "-", "NOT CHECKED")
if kind: row("chips with links", str(len(per)), str(want), lvl(len(per) < want))
if galaxy and judged:
    row("built-in links (boards)", f"{bi_have} of {bi_total} linked", "all", lvl(bi_have < bi_total))
    row("cable ports linked in part", f"{len(half)} port(s)", "0", lvl(half))
    row("cable ports, trays alike", "yes" if not uneven else f"no: port {ranges(uneven)}", "yes", lvl(False, uneven))
elif whgx and whole:
    row("links per chip", "ok" if not wh_bad else f"{len(wh_bad)} chip(s) wrong", "whole connectors", lvl(wh_bad))
    if wh_rows: row("trays alike", "yes" if not wh_diff else f"no: {tr(wh_diff)}", "yes", lvl(False, wh_diff))
elif qb2 and whole: row("built-in links", f"{bi_have} of {bi_total} linked", "all", lvl(qb_short))
elif kind: row("which links are there", "not judged", "-", "NOT CHECKED")
row("on-host links in total", str(total), "-", "not judged")
if prev: row("links gone since last run", str(len(gone)) if same_chips else "can't compare", "0", lvl(False, gone))
row("links down / cores stopped", str(down), "0", lvl(down))
if not has_ctr: row("uncorrected FEC errors", "not in this image", "-", "NOT CHECKED")
elif dt:
    row("uncorrected FEC errors", f"max {mur or 0:,.1f}/s", f"< {UW:g}/s per link", lvl((mur or 0) >= UP, (mur or 0) >= UW))
    row("CRC errors (bad frames)", f"max {mcr or 0:,.1f}/s", f"< {CW:g}/s per link", lvl(False, (mcr or 0) >= CW))
else: row("uncorrected FEC errors", f"max {munc:,} total", f"< {UT:,}", lvl(False, munc >= UT))
if has_ctr: row("ERR_STAT latched", f"{errstat} link(s)", "0", lvl(False, errstat))
if dt: row("retrains during the check", f"{retrained} link(s)", "0", lvl(retrained))
if fw_line: row("firmware across chips", "mixed" if mixed else "same", "same", lvl(False, mixed))
if kind and not (judged or (kind != "bh-galaxy" and whole)):
    print(f"\n  Which links are there was NOT judged: {why_not}.")
if mapped:  # what is cabled, as seen from the chips that are there. Described, never judged against a number.
    trays_here = sorted({t for t, _ in slot})
    def seen(pid): return {t: sum((slot[t, a], ch) in linked for a, ch in PORTS["QSFP_DD", pid] if (t, a) in slot) for t in trays_here}
    def state(s, n):
        full, part, none = ([t for t, k in s.items() if c(k)] for c in (lambda k: k == n, lambda k: 0 < k < n, lambda k: k == 0))
        if len(none) == len(s): return "not linked inside this host"
        if len(full) == len(s): return "linked on all 4 trays" if len(s) == 4 else f"linked on {tr(full)}"
        return ", ".join(x for x in (f"linked on {tr(full)}" if full else "", f"in part on {tr(part)}" if part else "", f"not on {tr(none)}" if none else "") if x)
    print("\n  CABLING FOUND (cables linked inside this host: described, not judged)")
    if not judged: print(f"    ({why_not}: ports that lead to them are not linked either)")
    else: print(f"    built-in links (board traces, linking boards)   {bi_have} of {bi_total} linked")
    for ports, role in GROUPS:
        i = 0
        while i < len(ports):
            j = i
            while j + 1 < len(ports) and seen(ports[j + 1]) == seen(ports[i]): j += 1
            print(f"    QSFP {'port ' if i == j else 'ports'} {str(ports[i]) if i == j else f'{ports[i]}-{ports[j]}':<7} ({role})   {state(seen(ports[i]), 2)}")
            i = j + 1
    if judged:
        print("    => " + (CABLING_TEXT.get(cabling) or ("not the same on every tray: see the findings above" if cabling == "irregular" else
              f"QSFP {cabling[6:]} linked on every tray: not a cabling this script has a name for (see LINKS under the verdict)")))
    extra = sorted((l["dev"], l["ch"]) for l in live(now) if (pos(bdf_of(l["dev"]))[1], l["ch"]) not in PORT_OF)
    if extra: print(f"    {len(extra)} link(s) on channels this script has no port name for: " + ", ".join(f"dev {dv} ch{ch}" for dv, ch in extra[:8]))
    print("    A port that is not linked inside this host has no cable, a cable to another host, or dead cables:\n"
          "    this check cannot tell which.")
if spots:
    print("\n  LINKS NOT LINKED ON THE PORTS NAMED ABOVE (tray / ASIC / port)")
    for dv, ch in sorted(set(spots)):
        t, pid = port_of(dv, ch)
        print(f"    dev {dv:>2}  {bdf_of(dv)}  {place(dv):<10} ch{ch:<3} {t} P{pid}")
if whgx and whole:
    print("\n  LINKS PER CHIP (Wormhole Galaxy: 4 links per connector. Described, not judged)")
    for t, v in sorted(wh_rows.items()):
        print(f"    T{t}: " + " ".join(f"{n:>2}" for n, _ in sorted(v, key=lambda x: x[1])) + "   (chips in PCI order)")
    print("    => " + (CABLING_TEXT.get(cabling) or ("the link counts fit no cabling this script has a name for (see LINKS under the verdict)" if unnamed
          else "the link counts are not alike on every tray, or not whole connectors: see the findings above")))
if prev:
    age = (t1 - (fnum(rd("t2", prev_dir)) or fnum(rd("t1", prev_dir)) or t1)) / 3600 if t1 else None
    ago = "" if age is None else f", {age * 60:.0f} min ago" if age < 1 else f", {age:.1f} h ago"
    print(f"\n  SINCE THE LAST RUN ({os.path.basename(prev_dir)}{ago})")
    print(f"    links found: was {len(pk)}, now {len(nk)}" + (" (same links)" if pk == nk else " (changed)"))
    if same_chips:  # name the links that changed
        for label, ks, src in (("up last time, not linked now", sorted(pk - nk), prev), ("not linked last time, up now", sorted(nk - pk), now)):
            if ks: print(f"    {label}: " + ", ".join(src[k]["name"] for k in ks[:8]) + (f" (+{len(ks) - 8} more)" if len(ks) > 8 else ""))
    else: print("    the chips on PCIe are not the same as last time, so single links cannot be compared")
    pairs = [(prev[k][f], l1[k][f]) for k in l1 if k in prev for f in COUNTERS if prev[k][f] is not None and l1[k][f] is not None]
    went_down, nonzero = sum(b < a for a, b in pairs), sum(a > 0 for a, _ in pairs)
    if went_down: print(f"    chips were reset in between ({went_down} counters went down)")
    elif not nonzero: print("    can't tell if the chips were reset in between: every counter was 0 last time")
    else:
        print(f"    no reset in between (none of {nonzero} non-zero counters went down)")
        up = {}
        for k in sorted(l1):
            a, b = (prev.get(k) or {}).get("retrain"), l1[k]["retrain"]
            if a is not None and b is not None and b > a:
                l = l1[k]
                if galaxy and l["ch"] is not None:  # Galaxy: group by cable / trace
                    t, pid = port_of(l["dev"], l["ch"])
                    up.setdefault(f"T{pos(bdf_of(l['dev']))[0]} {t} P{pid}", []).append(f"dev {l['dev']} ch{l['ch']} +{b - a}")
                else: up.setdefault(f"dev {l['dev']}", []).append(f"core {k[1]}" + (f" ch{l['ch']}" if l["ch"] is not None else "") + f" +{b - a}")
        print("    links that dropped and came back in between:" + ("" if up else " none"))
        for where, items in sorted(up.items()): print(f"      {where}: {', '.join(items)}")
try:  # Every earlier run on this host, oldest first (what ethmap.py showed across runs). Report only, never the verdict.
    _runs = []
    for p in earlier_runs():
        a1, a2 = links(tables(os.path.join(p, "sample1.csv"))), links(tables(os.path.join(p, "sample2.csv")))
        if a1 or a2: _runs.append((os.path.basename(p).split(".")[-1], a1 or a2, a2 or a1))
    _runs = _runs[-6:] + [("now", l1, now)]
    if len(_runs) >= 3:  # with one earlier run, SINCE THE LAST RUN already says it all
        print("\n  HISTORY ON THIS HOST (oldest first)")
        _last = None
        for nm, first_s, last_s in _runs:
            lv = [l for l in last_s.values() if (l["port"] or "").lower() not in IDLE]
            note = ""
            if _last is not None:
                pr = [(_last[k][f], first_s[k][f]) for k in first_s if k in _last for f in COUNTERS
                      if _last[k][f] is not None and first_s[k][f] is not None]
                note = ("chips reset before this run" if any(b < a for a, b in pr)
                        else "no reset before this run" if any(a > 0 for a, _ in pr) else "")
            when = nm if nm == "now" else f"{nm[0:4]}-{nm[4:6]}-{nm[6:8]} {nm[9:11]}:{nm[11:13]}Z"
            print(f"    {when:<18} {len({l['dev'] for l in lv}):>2} chips {len(lv):>5} links"
                  + (f"   {note}" if note else ""))
            _last = last_s
except Exception: pass
if has_ctr:
    print(f"\n  WORST LINKS (by uncorrected FEC errors)\n    {'link':<24} {'uncorr/s':>10} {'crc/s':>8} {'uncorr total':>14} {'retrains':>9}  ERR_STAT")
    for l, ur, cr in sorted(stats, key=lambda s: (s[1] or 0, s[0]["uncorr"] or 0), reverse=True)[:5]:
        print(f"    {l['name']:<24} {'-' if ur is None else f'{ur:,.1f}':>10} {'-' if cr is None else f'{cr:,.1f}':>8} {l['uncorr'] or 0:>14,} {l['retrain'] or 0:>9}  {l['errstat'] or 0:#x}")
if fw_line: print(f"\n  firmware  {fw_line}")
if si and si[0] and si[1]: print("  system    " + ", ".join(f"{h} {v}" for h, v in zip(si[0], si[1][0])))
if kind is None and per:
    print("\n  NO LINK MAP for this kind of host in this script. Link state, errors and changes since the last run were judged;\n"
          "  how many links there should be was not. Links per chip now: " + " ".join(str(per.get(i, 0)) for i in range(max(per) + 1)))
print("\n  NOT CHECKED HERE\n"
      "    - whether the cabling is the one this host is meant to have: only the people who cabled it know\n"
      "    - a port that is linked on no tray: no cable, a cable to another host and dead cables look the same from here\n"
      "    - links to other hosts: check_eth_status lists on-host links only\n"
      "    - where each cable goes: the far end of a link is not read here, so cables plugged into the wrong ports can look healthy\n"
      "    - traffic: a fabric router can wedge on a link that still reads healthy")
print(f"\n  raw triage output: {d}\n\nSUMMARY host={host} verdict={verdict} pcie_ok={f'{len(pc) - len(pbad)}/{npc}' if pc else '?'} chips={len(per)}"
      f" kind={kind or 'unknown'} card={'+'.join(sorted(cards)).replace(' ', '_') or '-'} onhost_links={total} builtin={f'{bi_have}/{bi_total}' if bi_total else '-'} cabling={cabling or '-'}"
      f" half_ports={len(half) if judged else '-'} uneven_ports={len(uneven) if judged else '-'}"
      f" gone_since_last={len(gone) if same_chips else '-'} down={down}"
      f" uncorr_max_per_s={'-' if mur is None else f'{mur:.1f}'} retrained_during={retrained if dt else '-'}"
      f" errstat_links={errstat}\n=========================================================\n")
sys.exit(code)
PY
  python3 -I "$d/ethchk.py" pcie "$d" || return 4
  if ! docker info >/dev/null 2>"$d/docker.log"; then
    echo "  ERROR: Docker isn't usable as $(id -un): $(tail -n 1 "$d/docker.log")"
    echo "  If that says permission denied, run it as:  sudo bash ethchk-upstream-tests.sh"; return 5
  fi
  if ! docker image inspect "$img" >/dev/null 2>&1; then  # first run on this host: download the image once
    echo "  Downloading the triage image once (about 9 GB, a few minutes; public, no login needed): $img"
    docker pull "$img" || { echo "  ERROR: download failed. The default image is public: if it says denied or unauthorized, an old"
      echo "  ghcr.io login is saved on this host - run  docker logout ghcr.io  and then this again."
      echo "  (A private image set with IMG needs  docker login ghcr.io  first.)"; return 5; }
  fi
  local hp=()  # mount the 1G hugepages only where the host has them
  [ -d /dev/hugepages-1G ] && hp=(--mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G)
  _ethchk_tri() { docker run --rm \
      --device /dev/tenstorrent:/dev/tenstorrent "${hp[@]}" \
      --entrypoint "$py" "$img" \
      "$tri" \
      --run=check_eth_status "$@" --dev=all --llm-output; }
  date +%s.%N > "$d/t1"; _ethchk_tri --run=firmware_versions --run=system_info > "$d/sample1.csv" 2> "$d/sample1.csv.log"
  if grep -q '^check_eth_status' "$d/sample1.csv"; then  # no link table means docker or triage failed: the report says why
    sleep "${SAMPLE_SECS:-30}"; date +%s.%N > "$d/t2"; _ethchk_tri > "$d/sample2.csv" 2> "$d/sample2.csv.log"
  fi
  python3 -I "$d/ethchk.py" report "$d"
}
ethchk_main "$@"
