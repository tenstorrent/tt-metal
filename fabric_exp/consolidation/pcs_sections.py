#!/usr/bin/env python3
"""pcs_sections.py twins COMPILE_LOG VARIANT GDIR
   pcs_sections.py rows CACHE GDIR VARIANT ROW... [--out PREFIX]

Phase/cost analysis, stage 3 (RESULTS §98): turn pc_sampler.py samples into time per code section per router ERISC.

twins: rebuild, with -g, every router kernel the run compiled (its logged compile and link commands, as
  offline_build.py replays them, from VARIANT's sources) into GDIR/<hash>/<proc>/<proc>.elf. The -g build has the same
  loadable code (checked on every config at -O3 and -Os), so its line tables describe what ran.
rows: per row (fabric_exp/results/pcs3_<row>_samples.{bin,json}, rate from pcs3_<row>_13129088_trial.csv):
  - each target's build: the run cache's relocated image (.xip.elf) equal to the image head read from L1 at the launch
    message's load address (the end-of-run read; it falls back to the start read); its -g twin by kernel hash;
  - each sampled PC in the kernel maps to the ELF address pc - load + text start, then through addr2line -i (inline
    chain) to a section of the main loop: by the helper it is inlined from, else by the loop-body line it sits on;
  - samples outside the kernel are firmware; the first and last 5% (by time) of each target's kernel samples are
    dropped (outside the traffic window);
  - a target carries traffic when more than 15% of its kernel samples are in work sections (sends, completions and
    credits, forwards, local delivery); every such target is reported, with each section's share of its time and
    that share times the row's cycles/packet.
Sections follow the loop bodies of the v3 sender_erisc / receiver_erisc programs (line ranges below, v3q sources)."""
import argparse
import bisect
import collections
import glob
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys

ROOT = "/home/aagarwal/tt-metal"
A2L = ROOT + "/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
RES = ROOT + "/fabric_exp/results"
SND = "fabric_erisc_router_v3_sender_erisc.hpp"
RCV = "fabric_erisc_router_v3_receiver_erisc.hpp"


def load_ob():
    spec = importlib.util.spec_from_file_location("ob", ROOT + "/fabric_exp/consolidation/offline_build.py")
    ob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ob)
    return ob


def twins(log, variant, gdir):
    ob = load_ob()
    db = {}
    for line in open(log, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"g\+\+ (compile|link) cmd: (.*?)\s*\(build\.cpp:\d+\)\s*$", line)
        if not m or ob.KDIR not in m.group(2):
            continue
        kind, cmd = m.group(1), m.group(2)
        if not ob.wellformed(kind, cmd):
            continue
        h = re.search(ob.KDIR + r"(\d+)/(\w+)/", cmd)
        opt = re.search(r"\s-(O\w+)\s", cmd).group(1)
        db.setdefault(opt, {}).setdefault(h.group(1), {}).setdefault(h.group(2), {}).setdefault(kind, cmd)
    os.makedirs(gdir, exist_ok=True)
    path = os.path.join(gdir, "cmds.json")
    json.dump(db, open(path, "w"))
    ob.DB = path
    for opt, hs in db.items():
        print(f"{opt}: {len(hs)} router configs logged")
        ob.build(argparse.Namespace(variant=variant, out=gdir, opt=opt, set_opt=None, hashes="all", flags="-g",
                                    jobs=32))


def text_segment(path):
    d = open(path, "rb").read()
    phoff, = struct.unpack_from("<I", d, 28)
    phentsize, phnum = struct.unpack_from("<HH", d, 42)
    for i in range(phnum):
        p_type, p_off, p_vaddr, _, p_filesz, _, p_flags, _ = struct.unpack_from("<8I", d, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:
            return p_vaddr, d[p_off:p_off + p_filesz]
    return None, b""


# ------------------------------------------------------------------ sections
# E0 (sender_erisc::run_main_loop): blocks of the pass by the loop-body line (v3q sender_erisc.hpp), refined by helper.
E0_BLOCKS = [  # (first line, last line, section)
    (683, 695, "E0 loop and round control"),
    (696, 697, "E0 VC0 free-slots group read"),
    (698, 716, "E0 worker: send check"),
    (717, 740, "E0 worker: completion check"),
    (741, 751, "E0 worker: connection poll"),
    (752, 758, "E0 VC0 forwarding"),
    (759, 771, "E0 VC1 forwarding"),
    (772, 810, "E0 round end (telemetry, end_round)"),
]
E0_HELPERS = [  # (helper name substring, section): an inline frame with this name overrides the block
    ("read_grouped_free_slots", "E0 VC0 free-slots group read"),
    ("worker_connection_event", "E0 worker: connection event (out of line)"),
    ("next_round", "E0 per-round (out of line)"),
    ("end_round", "E0 per-round (out of line)"),
    ("termination_requested", "E0 per-round (out of line)"),
    ("run_router_command", "E0 per-round (out of line)"),
]
E0_FWD_SUB = [("send_packet", "send"), ("credit_upstream_router", "upstream credit")]
VISIT_ONE = re.compile(r"visit_forwarding_channel<")  # one channel's visit (not visit_forwarding_channels<)
# E1 (receiver_erisc::rx_channel_step): by helper first, then by the step-body line.
E1_HELPERS = [
    ("send_completions", "E1 completions"),
    ("local_delivery", "E1 local delivery"),
    ("local_write_with_state", "E1 local delivery"),
    ("execute_local_semaphore_inc", "E1 local delivery"),
    ("forward_2d", "E1 forward"), ("forward_to", "E1 forward"), ("forward_payload", "E1 forward"),
    ("forward_slots_shared", "E1 forward"),
    ("admit_2d", "E1 admission"), ("downstream_has_space", "E1 admission"), ("admit_slots_shared", "E1 admission"),
    ("decode_action", "E1 decode"), ("PackedHeaderWord40", "E1 decode"),
    ("retire_slot", "E1 retire"),
    ("drain_pending_atomic_flush", "E1 deferred atomic"), ("complete_pending_atomic_flush", "E1 deferred atomic"),
    ("read_stream_count", "E1 poll"),
    ("termination_requested", "E1 per-round (out of line)"), ("end_round", "E1 per-round (out of line)"),
]
SEND_COMPLETIONS_RETURN = 822  # v3q receiver_erisc.hpp: `if (!batch_ready) { return; }` ends at this line
E1_BLOCKS = [
    (982, 1004, "E1 poll"),
    (1005, 1030, "E1 decode"),
    (1031, 1067, "E1 admission"),
    (1068, 1076, "E1 packet bookkeeping"),
    (1077, 1136, "E1 deferred atomic"),
    (1137, 1160, "E1 forward"),
    (1161, 1171, "E1 local delivery"),
    (1172, 1212, "E1 1D route decision"),
    (1213, 1216, "E1 retire"),
    (1217, 1224, "E1 deferred atomic"),
    (1225, 1227, "E1 completions"),
]
WORK = ("worker: send", "worker: credit", "visit:", "completions: send", "E1 forward", "local delivery", "retire",
        "decode", "admission", "deferred atomic", "packet bookkeeping", "1D route decision")


def classify(erisc, frames):
    """frames: [(function, file, line)], innermost first."""
    names = [f for f, _, _ in frames]
    if erisc == 0:
        for sub, sec in E0_HELPERS:
            if any(sub in n for n in names):
                return sec
        loop = [(f, fl, ln) for f, fl, ln in frames if "run_main_loop" in f and fl.endswith(SND)]
        if not loop:
            return "E0 other (" + (names[-1].split("(")[0].split("<")[0][-40:] if names else "?") + ")"
        ln = loop[0][2]  # the innermost loop-body frame: the line the instruction or its inlined helper is on
        sec = next((s for a, b, s in E0_BLOCKS if a <= ln <= b), "E0 loop and round control")
        # Checks every pass makes (is there a packet and room? new completions?) are pass overhead; the send and the
        # credit write are work.
        if sec == "E0 worker: send check" and any("send_packet" in n for n in names):
            sec = "E0 worker: send"
        elif sec == "E0 worker: completion check" and any("credit_worker" in n for n in names):
            sec = "E0 worker: credit"
        elif sec in ("E0 VC0 forwarding", "E0 VC1 forwarding"):
            if any("read_free_slots_group" in n for n in names):
                sub = "group read"
            elif any(VISIT_ONE.search(n) for n in names):
                sub = "visit: " + next((label for k, label in E0_FWD_SUB if any(k in n for n in names)),
                                       "completions and bookkeeping")
            else:
                sub = "idle test"
            sec = f"{sec}: {sub}"
        return sec
    comp = [(f, fl, ln) for f, fl, ln in frames if "send_completions" in f and fl.endswith(RCV)]
    if comp:
        # Lines up to the early return decide whether a batch goes out (every pass); the rest proves and sends it.
        return "E1 completions: batch check" if comp[0][2] <= SEND_COMPLETIONS_RETURN else "E1 completions: send"
    for sub, sec in E1_HELPERS:
        if any(sub in n for n in names):
            return sec
    step = [(f, fl, ln) for f, fl, ln in frames if "rx_channel_step" in f and fl.endswith(RCV)]
    if step:
        ln = step[0][2]
        return next((s for a, b, s in E1_BLOCKS if a <= ln <= b), "E1 step other")
    return "E1 loop and round control"


def is_work(sec):
    return any(w in sec for w in WORK) and not sec.endswith("check")


def a2l(elf, addrs):
    """{addr: [(function, file, line)]} via addr2line -a -f -i -C."""
    out = subprocess.run([A2L, "-a", "-f", "-i", "-C", "-e", elf], input="\n".join(hex(a) for a in addrs),
                         capture_output=True, text=True).stdout.splitlines()
    res, cur, i = {}, None, 0
    while i < len(out):
        line = out[i]
        if re.fullmatch(r"0x[0-9a-f]+", line):
            cur = int(line, 16)
            res[cur] = []
            i += 1
            continue
        fn = line
        loc = out[i + 1] if i + 1 < len(out) else "??:0"
        m = re.match(r"(.*):(\d+)", loc)
        res[cur].append((fn, os.path.basename(m.group(1)) if m else loc, int(m.group(2)) if m else 0))
        i += 2
    return res


def row_report(cache, gdir, row, prefix="pcs3"):
    meta = json.load(open(f"{RES}/{prefix}_{row}_samples.json"))
    samples = collections.defaultdict(list)
    for t, i, pc in struct.Struct("<dHI").iter_unpack(open(f"{RES}/{prefix}_{row}_samples.bin", "rb").read()):
        samples[i].append((t, pc))
    rate = None
    csvp = f"{RES}/{prefix}_{row}_13129088_trial.csv"
    if os.path.exists(csvp):
        import csv
        vals = [float(r["avg_packets_per_s"]) for r in csv.DictReader(open(csvp))]
        rate = 1.35e9 / (sum(vals) / len(vals)) if vals else None
    xips = {}
    for proc in ("active_erisc", "subordinate_active_erisc"):
        for e in glob.glob(f"{cache}/**/fabric_erisc_router_v2/*/{proc}/{proc}.elf.xip.elf", recursive=True):
            base, text = text_segment(e)
            xips.setdefault(proc, []).append((e.split("fabric_erisc_router_v2/")[1].split("/")[0], base, text))
    a2l_cache = {}
    targets = []
    for i, tgt in enumerate(meta["targets"]):
        key = f"{tgt['chip']}/{tgt['chan']}"
        lay = meta.get("layouts_end", {}).get(key) or meta.get("layouts", {}).get(key, {})
        e = lay.get("erisc", {}).get(str(tgt["erisc"]), {}) if isinstance(lay, dict) else {}
        proc = "active_erisc" if tgt["erisc"] == 0 else "subordinate_active_erisc"
        head = bytes.fromhex(e.get("head", "")) if e else b""
        match = [(h, b) for h, b, t in xips.get(proc, []) if head and t[:len(head)] == head]
        smp = samples.get(i, [])
        if not match:
            targets.append({"target": tgt, "samples": len(smp), "kernel": 0, "note": "no build matched"})
            continue
        h, base = match[0]
        twin = f"{gdir}/{h}/{proc}/{proc}.elf"
        load, size = e["load"], e["text_size"]
        kern = sorted((t, pc - load + base) for t, pc in smp if load <= pc < load + size)
        if len(kern) > 20:
            k = len(kern) // 20
            kern = kern[k:len(kern) - k]
        if not os.path.exists(twin):
            targets.append({"target": tgt, "samples": len(smp), "kernel": len(kern), "note": f"no -g twin {h}"})
            continue
        if twin not in a2l_cache:
            a2l_cache[twin] = {}
        need = sorted({a for _, a in kern} - set(a2l_cache[twin]))
        if need:
            a2l_cache[twin].update(a2l(twin, need))
        secs = collections.Counter(classify(tgt["erisc"], a2l_cache[twin].get(a, [])) for _, a in kern)
        n = sum(secs.values())
        work = sum(c for s, c in secs.items() if is_work(s)) / n if n else 0.0
        targets.append({"target": tgt, "samples": len(smp), "kernel": len(kern), "build": h, "work_share": work,
                        "sections": {s: c / n for s, c in secs.most_common()}})
    return {"row": row, "cycles_per_packet": rate, "duration_s": meta["duration_s"], "rate_per_s": meta["rate_per_s"],
            "targets": targets}


def print_row(r, busy_cut=0.15):
    cpp = r["cycles_per_packet"]
    print(f"\n### {r['row']}: {cpp:.1f} cycles/packet" if cpp else f"\n### {r['row']}")
    busy = [t for t in r["targets"] if t.get("work_share", 0) > busy_cut]
    idle = [t for t in r["targets"] if "sections" in t and t.get("work_share", 0) <= busy_cut]
    print(f"{len(busy)} router ERISCs carry traffic, {len(idle)} idle, "
          f"{sum(1 for t in r['targets'] if 'sections' not in t)} without a kernel match (unused links)")
    for t in sorted(busy, key=lambda t: (t["target"]["erisc"], -t["work_share"])):
        tg = t["target"]
        print(f"  chip {tg['chip']} eth {tg['chan']} E{tg['erisc']} ({t['kernel']} kernel samples, "
              f"{100 * t['work_share']:.0f}% in work sections):")
        for s, f in t["sections"].items():
            if f < 0.01:
                continue
            print(f"     {100 * f:5.1f}%  {(f * cpp if cpp else 0):6.1f} cyc/pkt  {s}")


def main():
    if sys.argv[1] == "twins":
        twins(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("cache")
    ap.add_argument("gdir")
    ap.add_argument("variant")
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="pcs3", help="sample/label prefix (pcsv_runs.sh writes pcsv_<tag>)")
    args = ap.parse_args()
    reports = [row_report(args.cache, args.gdir, row, args.prefix) for row in args.rows]
    for r in reports:
        print_row(r)
    if args.out:
        json.dump(reports, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
