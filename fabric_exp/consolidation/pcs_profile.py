#!/usr/bin/env python3
"""pcs_profile.py CACHE GDIR PREFIX ROW [ROW ...] [--top N]: where the busiest router ERISCs spend their time, for any
router variant whose functions name their work (v4: send, visit_worker, receive, deliver, release_completions, ...).

PREFIX names the sampled runs: results/<PREFIX>_<row>_samples.{bin,json} and results/<PREFIX>_<row>_13129088_trial.csv
(pcsv_runs.sh writes PREFIX = pcsv_<tag>). GDIR holds the -g twins (pcs_sections.py twins). Each sample maps through its
inline chain to the innermost frame whose function is one of SECTIONS (else the loop function's own lines). The busiest
ERISC of each kind is the one with the most samples in its work sections (E0: send; E1: deliver, forward, retire).
Prints each section's share of the ERISC's kernel samples and that share times the row's cycles/packet."""
import argparse, collections, csv, glob, json, os, re, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcs_sections as ps  # noqa: E402

# (function-name pattern, section), innermost first wins.
SECTIONS = [
    (r"sender::send<", "E0 send"),
    (r"sender::credit_worker", "E0 worker credit"),
    (r"sender::credit_upstream", "E0 upstream credit"),
    (r"sender::worker_connection_event", "E0 connection event"),
    (r"sender::visit_worker<", "E0 worker visit (checks)"),
    (r"sender::visit_forwarding<", "E0 forwarding visit (checks)"),
    (r"sender::send_forwarded<", "E0 forwarded sends (loop)"),
    (r"sender::complete_forwarded<|sender::read_completion_counters<", "E0 forwarding completions"),
    (r"sender::visit_forwarding_channels<", "E0 forwarding dispatch"),
    (r"sender::channels_with_packets<|sender::read_free_slots<|sender::filled<", "E0 free-slots group read"),
    (r"FreeSlots<.*>::read", "E0 free-slots group read"),
    (r"FreeSlots<.*>::none_has_packet", "E0 forwarding idle test"),
    (r"FreeSlots<.*>::has_packet", "E0 has-packet test"),
    (r"sender::total_sent", "E0 per-round"),
    (r"next_round|end_round|start_running|terminating|run_router_command", "per-round"),
    (r"receiver::acknowledge_arrival<", "E1 ack"),
    (r"release_completions", "E1 completions"),
    (r"slot_retired", "E1 retire"),
    (r"receiver::retire<", "E1 retire"),
    (r"local_write<|local_atomic_inc|deliver_other|fabric_router::deliver<", "E1 local delivery"),
    (r"Downstreams<.*>::forward", "E1 forward"),
    (r"Downstreams<.*>::admit", "E1 admission"),
    (r"Routing2D<.*>::|Routing1D::|action_to_slots", "E1 decode"),
    (r"HeaderWord::load", "E1 header"),
    (r"read_stream_count", "E1 poll"),
    (r"update_buffer_slot_to_next_trid", "E1 trid"),
    (r"receiver::receive<", "E1 receive (other)"),
]
WORK = {0: ("E0 send",), 1: ("E1 local delivery", "E1 forward", "E1 retire")}


def classify(frames):
    for fn, fl, ln in frames:
        for pat, sec in SECTIONS:
            if re.search(pat, fn):
                return sec
    if frames:
        fn, fl, ln = frames[-1]
        return f"loop {fl}:{ln}"
    return "?"


def target_samples(cache, prefix, row):
    """{target index: (target, build hash, [elf addrs])} as pcs_sections.row_report maps them."""
    meta = json.load(open(f"{ps.RES}/{prefix}_{row}_samples.json"))
    samples = collections.defaultdict(list)
    for t, i, pc in struct.Struct("<dHI").iter_unpack(open(f"{ps.RES}/{prefix}_{row}_samples.bin", "rb").read()):
        samples[i].append((t, pc))
    xips = {}
    for proc in ("active_erisc", "subordinate_active_erisc"):
        for e in glob.glob(f"{cache}/**/fabric_erisc_router_v2/*/{proc}/{proc}.elf.xip.elf", recursive=True):
            base, text = ps.text_segment(e)
            xips.setdefault(proc, []).append((e.split("fabric_erisc_router_v2/")[1].split("/")[0], base, text))
    out = {}
    for i, tgt in enumerate(meta["targets"]):
        key = f"{tgt['chip']}/{tgt['chan']}"
        lay = meta.get("layouts_end", {}).get(key) or meta.get("layouts", {}).get(key, {})
        e = lay.get("erisc", {}).get(str(tgt["erisc"]), {}) if isinstance(lay, dict) else {}
        proc = "active_erisc" if tgt["erisc"] == 0 else "subordinate_active_erisc"
        head = bytes.fromhex(e.get("head", "")) if e else b""
        match = [(h, b) for h, b, t in xips.get(proc, []) if head and t[:len(head)] == head]
        if not match:
            continue
        h, base = match[0]
        load, size = e["load"], e["text_size"]
        kern = sorted((t, pc - load + base) for t, pc in samples.get(i, []) if load <= pc < load + size)
        if len(kern) > 20:
            k = len(kern) // 20
            kern = kern[k:len(kern) - k]
        out[i] = (tgt, h, [a for _, a in kern])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("gdir"); ap.add_argument("prefix"); ap.add_argument("rows", nargs="+")
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()
    for row in a.rows:
        vals = [float(r["avg_packets_per_s"]) for r in csv.DictReader(open(f"{ps.RES}/{a.prefix}_{row}_13129088_trial.csv"))]
        cpp = 1.35e9 / (sum(vals) / len(vals))
        print(f"\n### {row}: {cpp:.1f} cycles/packet")
        a2l = {}
        best = {}
        for i, (tgt, h, addrs) in target_samples(a.cache, a.prefix, row).items():
            proc = "active_erisc" if tgt["erisc"] == 0 else "subordinate_active_erisc"
            twin = f"{a.gdir}/{h}/{proc}/{proc}.elf"
            if not os.path.exists(twin) or not addrs:
                continue
            cache = a2l.setdefault(twin, {})
            need = sorted(set(addrs) - set(cache))
            if need:
                cache.update(ps.a2l(twin, need))
            secs = collections.Counter(classify(cache.get(x, [])) for x in addrs)
            work = sum(secs[s] for s in WORK[tgt["erisc"]]) / len(addrs)
            if work > best.get(tgt["erisc"], (0, None))[0]:
                best[tgt["erisc"]] = (work, tgt, secs, len(addrs))
        for erisc in (0, 1):
            if erisc not in best:
                continue
            work, tgt, secs, n = best[erisc]
            print(f"  E{erisc} chip {tgt['chip']} chan {tgt['chan']} ({n} samples)")
            for s, c in secs.most_common(a.top):
                print(f"    {c / n * cpp:6.1f}  {c / n * 100:5.1f}%  {s}")


if __name__ == "__main__":
    main()
