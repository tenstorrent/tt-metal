#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router bring-up tools (temporary: drop before merge; see README.md). BLACKHOLE ONLY (reads pc_sampler.py runs).

pcs_profile.py CACHE_DIR SAMPLES_PREFIX [--cycles-per-packet X] [--top N]: where the busiest router ERISCs of a sampled
row spend their time, by named piece of work.

Each kernel sample maps through its inline chain (addr2line -i on the kernel cache's own ELF) to the innermost frame
whose function matches one of SECTIONS (else the loop function's own source line). The busiest ERISC of each kind is
the one with the largest share of samples in its work sections (E0: send; E1: deliver, forward, retire). Prints each
section's share of that ERISC's kernel samples and, with --cycles-per-packet (the row's rate, e.g. from ab_report.py),
that share times the rate: a split of cycles within one ERISC. It cannot say which ERISC bounds the row (both poll
without stopping, so their samples always add up to the row's rate): that takes an A/B that changes one side's cost.
"""
import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcs_common as pc  # noqa: E402

# (function-name pattern, section), innermost frame first wins.
SECTIONS = [
    (r"sender::send<", "E0 send"),
    (r"sender::credit_worker", "E0 worker credit"),
    (r"sender::credit_upstream", "E0 upstream credit"),
    (r"sender::worker_connection_event", "E0 connection event"),
    (r"sender::visit_worker<", "E0 worker visit (checks)"),
    (r"sender::visit_forwarding<|sender::visit_channel_with_work<", "E0 forwarding visit (checks)"),
    (r"sender::visit_forwarding_channels<", "E0 forwarding dispatch"),
    (r"sender::channels_with_packets<|sender::read_free_slots", "E0 free-slots group read"),
    (r"sender::total_sent", "E0 per-round"),
    (r"next_round|end_round|start_running|terminating|run_router_command", "per-round"),
    (r"receiver::acknowledge_arrival", "E1 ack"),
    (r"release_completions", "E1 completions"),
    (r"slot_retired|receiver::retire<", "E1 retire"),
    (r"local_write<|local_atomic_inc|deliver_other|fabric_router::deliver<", "E1 local delivery"),
    (r"Downstreams<.*>::forward|::forward<", "E1 forward"),
    (r"Downstreams<.*>::admit|::admit<", "E1 admission"),
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache")
    ap.add_argument("prefix")
    ap.add_argument("--cycles-per-packet", type=float, default=None)
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()
    meta, samples = pc.load_run(a.prefix)
    a2l, best = {}, {}
    for i, tgt in enumerate(meta["targets"]):
        lay = pc.layout(meta, tgt)
        build = pc.match_build(a.cache, tgt["erisc"], lay)
        if build is None:
            continue
        elf, base, h = build
        addrs = [x for _, x in pc.kernel_addrs(samples.get(i, []), lay, base)]
        if not addrs:
            continue
        cache = a2l.setdefault(elf, {})
        need = sorted(set(addrs) - set(cache))
        if need:
            cache.update(pc.a2l(elf, need))
        secs = collections.Counter(classify(cache.get(x, [])) for x in addrs)
        work = sum(secs[s] for s in WORK[tgt["erisc"]]) / len(addrs)
        if work >= best.get(tgt["erisc"], (-1, None))[0]:
            best[tgt["erisc"]] = (work, tgt, secs, len(addrs), h)
    if not best:
        sys.exit("no target matched a build in the cache (wrong cache, or the routers were not running)")
    cpp = a.cycles_per_packet
    for erisc in sorted(best):
        work, tgt, secs, n, h = best[erisc]
        print(f"E{erisc} chip {tgt['chip']} chan {tgt['chan']} ({n} kernel samples; build {h})")
        for s, c in secs.most_common(a.top):
            print(f"  {(f'{c / n * cpp:6.1f} ' if cpp else '')}{c / n * 100:5.1f}%  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
