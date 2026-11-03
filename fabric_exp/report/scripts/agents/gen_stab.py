#!/usr/bin/env python3
"""Builds fabric_exp/report/data/stability.json: headroom guard results (parsed from loopvars_maps/*/headroom_*.txt),
register maps, measured fragility, opt-level and jitter-band results (transcribed from the cited sections)."""
import json, re, sys

OUT = sys.argv[1]
ROOT = "/home/aagarwal/tt-metal/"
LM = "fabric_exp/consolidation/loopvars_maps/"
RO = "fabric_exp/consolidation/V4_RECEIVER_OWNERSHIP.md"
SO = "fabric_exp/consolidation/V4_SENDER_OWNERSHIP.md"
SIZE = "fabric_exp/consolidation/V4_SIZE.md"
RES = "fabric_exp/RESULTS.md"
RS = "fabric_exp/consolidation/V4_RECEIVER_SHAPE.md"
MEM = "/home/aagarwal/.claude/projects/-home-aagarwal-tt-metal/memory/project-receiver-ownership-pass.md"
PH1 = "fabric_exp/consolidation/PHASE1.md"
VC2L = "/home/aagarwal/tt-metal-vc2line/fabric_exp/consolidation/VC2_LINE_BRINGUP.md"
CP = "cycles/packet"

recs = []
ids = set()


def rec(id, system, topology, config, test, send_type, size, router, metric, value, unit, source_file,
        source_section, job, date, notes, bound_side=None, bound_status=None, same_job=False, pair_key=None, **opt):
    assert id not in ids, id
    ids.add(id)
    r = dict(id=id, system=system, topology=topology, config=config, test=test, send_type=send_type,
             packet_size_B=size, router=router, metric=metric, value=value, unit=unit, bound_side=bound_side,
             bound_status=bound_status, source_file=source_file, source_section=source_section, job=job, date=date,
             same_job_with_main=same_job, pair_key=pair_key, notes=notes)
    r.update(opt)
    recs.append(r)


TOPO = {"bh_1d": "1D line", "bh_ring": "ring", "bh_2d": "2D mesh", "bh_torus": "torus", "bh_2d_vc2": "2D mesh",
        "bh_torus_vc2": "torus", "bh_vc1": "VC1 split (two 2x2 meshes)", "bh_2d_vc2_trim_w0": "2D mesh",
        "bh_vc1_trim_w0": "VC1 split (two 2x2 meshes)", "glx_express": "Galaxy express mesh"}
FAMNOTE = {"bh_2d_vc2": "2D + VC2 (two receiver channels / two worker channels)", "bh_torus_vc2": "torus + VC2",
           "bh_vc1": "VC1 (VC0 + VC1)", "bh_2d_vc2_trim_w0": "T1: 2D + VC2 with VC0's worker trimmed",
           "bh_vc1_trim_w0": "T5: VC1 with VC0's worker trimmed", "glx_express": "Galaxy express (5 VC0 channels)"}
OVL = {"x1": "+1 carried value (opaque, per pass)", "x2": "+2 carried values", "fd": "flush-defer record per channel (in memory, settled out of line)",
       "tx1": "+1 carried value in sender::pass", "tx2": "+2 carried values", "twk": "worker-sized state (5 values, a line VC's worker)",
       "trr": "remote-receiver progress (2 values per send)"}

line_re = re.compile(r"^\s+(\S+)\s+(\d+)/(\d+) pass(.*)$")
hdr_re = re.compile(r"^== (\S+) \+ (\S+)")


def parse(path):
    out = []
    cur = None
    for ln in open(ROOT + path):
        m = hdr_re.match(ln)
        if m:
            cur = (m.group(1), m.group(2))
            continue
        m = line_re.match(ln)
        if m and cur:
            fam, p, n, rest = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            hot = re.search(r"HOT MOVED (\{[^}]*\})", rest)
            out.append((cur[0], cur[1], fam, p, n, hot.group(1) if hot else ""))
    return out


HR = [  # (file, section, role, router override for date/notes)
    (LM + "packed/headroom_all.txt", "§14 Headroom test (packed cursors; jobs 415-417 era)", "receiver (E1)"),
    (LM + "polish/headroom_e1.txt", "§15 Headroom (v4rxe1)", "receiver (E1)"),
    (LM + "structure/headroom_f1.txt", "§17 Headroom (v4rxf1)", "receiver (E1)"),
    (LM + "structure/headroom_f2.txt", "§17 Headroom (v4rxf2 = S4 alone: reproduces every changed cell)", "receiver (E1)"),
    (LM + "structure/headroom_f4.txt", "§17 Headroom (v4rxf4)", "receiver (E1)"),
    (LM + "structure/headroom_g3.txt", "§18 Headroom (v4rxg3, the receiver lead)", "receiver (E1)"),
    (LM + "sender_pass/headroom_v4txo0.txt", "§12 Sender headroom on X0 (recorded known answers)", "sender (E0)"),
    (LM + "sender_pass/headroom_v4txo1.txt", "§12 X1 headroom", "sender (E0)"),
]
for path, sec, role in HR:
    srcdoc = SO if "sender" in role else RO
    for router, ovl, fam, p, n, hot in parse(path):
        pf = "pass" if p == n else ("fail" if p == 0 else "partial")
        note = (f"{role} register-map guard (loopvars.py --headroom, -O3): {p}/{n} images pass (no hot value newly on the stack) with {OVL.get(ovl, ovl)}; "
                f"family {FAMNOTE.get(fam, fam)}; {pf}. ")
        note += ("Hot values moved (images): " + hot + ". " if hot else "No hot value moved. ")
        note += f"Raw file {path}; doc {srcdoc} {sec.split()[0]}."
        rec(f"hr-{router}-{ovl}-{fam}", "offline", TOPO.get(fam, fam), f"{role}, synthetic overlay {ovl}, two ERISCs", fam, None, None, router,
            "headroom_cells_pass", p, "count", path, sec, None, "2026-11-02", note, overlay=ovl, images_total=n, opt_level="-O3")

# §14 summary: families (of 7) passing per variant and overlay
for router, vals in (("v4rxs2x", (3, 0, 6)), ("v4rxb2p", (5, 3, 7)), ("v4rxp2p", (3, 3, 2))):
    for ovl, v in zip(("x1", "x2", "fd"), vals):
        rec(f"hr-fam-{router}-{ovl}", "offline", "all (7 families)", f"receiver (E1) guard, overlay {ovl}", "families passing of 7", None, None, router,
            "headroom_families_pass", v, "count", RO, "§14 Headroom test table", None, "2026-11-02",
            "Families (1D, ring, 2D, torus, 2D+VC2, torus+VC2, VC1) whose every image passes. S2-explicit (v4rxs2x) was adopted; its +2 cell is 0/7 (doc: 'the guard is conservative on ring; right on the two-channel rows').",
            overlay=ovl, opt_level="-O3")

# ---------------------------------------------------------------- measured fragility (job 417)
for test, d, topo in (("ew4352", -0.1, "2D mesh"), ("ring1", -0.1, "ring"), ("ewscatter", 2.5, "2D mesh"), ("VC1Inter 1_0", 0.6, "VC1 inter-mesh"),
                      ("torusns (VC2 on)", 5.1, "Y-torus"), ("torusns_scatter (VC2 on)", 6.6, "Y-torus"),
                      ("VC1Mesh2HopMerge 16 B", 5.4, "VC1 split (two 2x2 meshes)"), ("VC1Mesh2HopMerge 4352 B", 4.4, "VC1 split (two 2x2 meshes)")):
    two = "VC2" in test or "VC1Mesh" in test
    size = 16 if "16 B" in test else (4352 if ("4352" in test) else None)
    rec(f"frag-s2xh2-{test.replace(' ','').replace('(','').replace(')','')}", "P150x8", topo, "two ERISCs; v4rxs2x + 2 synthetic carried loop values (v4rxs2xh2) vs v4rxs2x",
        test, None, size, "v4rxs2xh2", "fragility_delta", d, CP, RO, "§14 Stability cost on the device", "417", "2026-11-02",
        ("Two-channel row: " if two else "Single-channel row: ") + "same job, two passes in reversed order. Doc: 0 on single-channel rows, +4 to +7 on two-channel rows (torus + VC2, VC1). "
        "The receiver lead v4rxg3 keeps S2-explicit's placement (v4rxs2x -> v4rxe1 polish -> v4rxf* -> v4rxg3); no fragility probe was rerun on the lead.",
        same_job=False, opt_level="-O3")

# ---------------------------------------------------------------- placement rule costs (A/Bs that change only the placement)
for test, topo, d, sec, job, date, n in (
        ("torusns, VC2 on", "Y-torus", 4.2, "§8 (job 403): S2 166.9-167.0, S3 171.1", "403", "2026-11-01", ""),
        ("torusns_scatter, VC2 on", "Y-torus", 19.1, "§8 (job 403): S2 209.1-209.2, S3 228.3", "403", "2026-11-01", " Spill-store cycles 5.2 -> 24.0 cycles/packet (+18.8) by PC samples.")):
    k = test.replace(" ", "").replace(",", "")
    rec(f"place-S3-{k}", "P150x8", topo, "two ERISCs; S3/B2 (every receiver cursor a loop local) vs S2 (single change)", test, None, None, "v4rxo5", "placement_delta",
        d, CP, RO, sec, job, date,
        "Isolating A/B (E1 only), two passes reversed. Mechanism (PC samples, hypothesis): GCC spills VC0's long-latency loads right after the load (TX-queue status 7.4, chunk count 7.9, encoding 6.5 cycles/packet) when every cursor is a loop local." + n,
        None, None, opt_level="-O3")
for test, topo, d in (("ewscatter", "2D mesh", 3.8), ("torusns, VC2 on", "Y-torus", 6.2), ("torusns_scatter, VC2 on", "Y-torus", 10.9),
                      ("VC1Inter 1_0 16 B", "VC1 inter-mesh", 3.4), ("VC1Inter 1_0 4352 B", "VC1 inter-mesh", 3.6), ("ew4352", "2D mesh", 0.3)):
    k = test.replace(" ", "").replace(",", "")
    rec(f"place-third-{k}", "P150x8", topo, "two ERISCs; third shape (placement by access frequency) vs lead v4rxv2i3", test, None, None, "v4rxo8", "placement_delta",
        d, CP, RO, "§9 table, column 'third shape - lead' (jobs 404-405)", "404-405", "2026-11-02",
        "E0 images byte-identical to S2's, so the loss is the receiver's (A/B on E1). Acceptance (within 1-2 of the lead) failed. Mechanism on torusns_scatter: the TX-queue status spill 8.3 + 3.0 (hypothesis, not isolated).",
        None, None, opt_level="-O3")
for test, topo, b2, p2 in (("ew4352", "2D mesh", 1.0, 3.6), ("ewscatter", "2D mesh", 7.2, 22.8), ("ring1", "ring", 11.7, 21.7), ("torusns, VC2 on", "Y-torus", 7.0, 18.4),
                           ("torusns_scatter, VC2 on", "Y-torus", 10.3, 17.6), ("ew4352, one ERISC", "2D mesh", 1.1, 17.0),
                           ("VC1Mesh2HopMerge 16", "VC1 split (two 2x2 meshes)", 8.8, 16.9), ("VC1Mesh2HopMerge 4352", "VC1 split (two 2x2 meshes)", 7.4, 15.3),
                           ("VC1Inter 1_0 16", "VC1 inter-mesh", 6.8, 25.3), ("VC1Inter 1_0 4352", "VC1 inter-mesh", 7.2, 25.2)):
    k = test.replace(" ", "").replace(",", "")
    for router, v, n in (("v4rxb2p", b2, "B2-packed (every cursor a loop local, counts packed in one word)"), ("v4rxp2p", p2, "P2-packed (every channel's state in memory, packed)")):
        rec(f"place-{router}-{k}", "P150x8", topo, f"{'one' if 'one ERISC' in test else 'two'} ERISC(s); {n} vs lead", test, None, None, router, "placement_delta", v, CP, RO,
            "§14 Device table (jobs 415-417)", "415-417", "2026-11-02",
            "Same job as lead / S2 / S2-explicit; two passes reversed, every row within 0.2. S2-explicit (adopted) is 0 on every row. B2-packed passes the most headroom cells (5/3/7 families) but costs this; hypothesis: packed counts serialize through one register (RV32IM+Zba/Zbb has no bitfield insert/extract).",
            opt_level="-O3")

# ---------------------------------------------------------------- sender register maps (2D + VC2 E0 image, kernel 12364934172383773789)
maps = {"v4rxf1": (596, 665, 15, 31, 24, None), "v4txpb": (605, 669, 12, 34, 20, 37), "v4txpc": (583, 641, 13, 31, 22, None),
        "v4txpbk": (589, 650, 10, 32, 20, 32)}
for router, (loop, rnd, creg, cslot, preg, pslot) in maps.items():
    base = dict(system="offline", topology="2D mesh", config="sender E0, 2D + VC2 image (vc02 rows' build), -O3", test="loopvars.py --live --role sender",
                send_type=None, size=None, router=router)
    sec = "§8.3 Register maps / §8.4 Register map"
    src = SO
    note0 = ("lead (P-A); sender identical v4rxf1..v4rxg3. " if router == "v4rxf1" else "") + "loopvars_maps/sender_budget/txp_live_<router>.txt / txpk_live_<router>.txt. "
    for metric, v, unit, n in (("loop_instructions", loop, "count", f"pass loop (round loop {rnd})"), ("carried_registers", creg, "count", "values carried at the pass head in registers"),
                               ("carried_stack_slots", cslot, "count", "values carried at the pass head on the stack (includes objects in memory by design)"),
                               ("peak_live_registers", preg, "count", "registers at the peak")):
        rec(f"map-tx-{router}-{metric}", base["system"], base["topology"], base["config"], base["test"], None, None, router, metric, v, unit, src, sec, None, "2026-11-02",
            note0 + n + ".", opt_level="-O3")
    hot = {"v4rxf1": 4, "v4txpb": 0, "v4txpc": 1, "v4txpbk": 0}[router]
    hotn = {"v4rxf1": "VC0 and VC2: worker slot and far next slot on the stack (§7.1: 28 of 28 2D+VC2 images; §8.3 map)",
            "v4txpb": "no hot value on the stack in any family (13 key cases); VC2's in-visit far-ring copy's free_slots spilled",
            "v4txpc": "VC0 all in registers; VC2's far next slot on the stack (as the lead) on 2D + VC2 and torus + VC2",
            "v4txpbk": "no new hot value on the stack; VC2's in-visit copy in a register"}[router]
    rec(f"map-tx-{router}-hotspills", "offline", "2D mesh", base["config"], base["test"], None, None, router, "spills", hot, "count", src, sec, None, "2026-11-02",
        "Hot send-path values on the stack: " + hotn + ". Count transcribed from the doc's description (lead: VC0 slot + far next slot, VC2 slot + far next slot = 4, as §7.2 'four hot values move to the stack'; P-C: VC2 far next slot = 1).", opt_level="-O3")
rec("map-tx-sp-v4txpb", "offline", "2D mesh", "sender E0, 2D + VC2 image, -O3", "sp accesses in the pass loop", None, None, "v4txpb", "stack_accesses", 99, "count", SO,
    "§8.4 Static check", None, "2026-11-02", "70 loads, 29 stores (P-B).", opt_level="-O3")
rec("map-tx-sp-v4txpbk", "offline", "2D mesh", "sender E0, 2D + VC2 image, -O3", "sp accesses in the pass loop", None, None, "v4txpbk", "stack_accesses", 84, "count", SO,
    "§8.4 Static check", None, "2026-11-02", "60 loads, 24 stores (packed P-B).", opt_level="-O3")
for router, loop, creg, peak in (("v4txpb", 828, 14, 66), ("v4txpbk", 819, 13, 62)):
    rec(f"map-tx-vc1-{router}-carried", "offline", "VC1 split (two 2x2 meshes)", "sender E0, VC1 image bh_vc1__10354273653968310346, -O3", "loopvars.py --live", None, None,
        router, "carried_registers", creg, "count", SO, "§8.4 Register map (VC1 image)", None, "2026-11-02",
        f"Pass loop {loop} instructions; peak {peak}; VC1's far ring {'2 slots' if router == 'v4txpb' else '1 slot'} in memory.", opt_level="-O3")
# sender inventory by family (§7.1, lead v4rxf1 build)
for fam, topo, loop, creg, peak, hot in (("1D", "1D line", 213, 14, 20, 0), ("ring", "ring", 279, 13, 20, 0), ("2D", "2D mesh", 455, 13, 22, 0),
                                         ("torus", "torus", 597, 13, 22, 0), ("Galaxy express", "Galaxy express mesh", 753, 15, 24, 0),
                                         ("VC1", "VC1 split (two 2x2 meshes)", 832, 15, 24, 1), ("2D + VC2", "2D mesh", 596, 15, 24, 4)):
    k = fam.replace(" ", "").replace("+", "p")
    rec(f"inv-tx-{k}-peak", "offline", topo, "sender E0 inventory over every image of the v4rxf1 coverage build, -O3", fam, None, None, "v4rxf1", "peak_live_registers", peak, "count",
        SO, "§7.1 Demand by family", None, "2026-11-02",
        f"Budget 27 allocatable registers. Carried registers {creg}; pass loop {loop} instructions (upper end of the family's range where a range is given). Hot values carried on the stack: {hot} "
        + ("(VC1's remote receiver next slot, 22 of 22 images)." if fam == "VC1" else ("(both workers' next slot, both remote receivers' next slot, 28 of 28)." if fam == "2D + VC2" else "(none).")),
        opt_level="-O3")
    rec(f"inv-tx-{k}-hot", "offline", topo, "sender E0 inventory, v4rxf1 coverage build, -O3", fam, None, None, "v4rxf1", "spills", hot, "count", SO,
        "§7.1 Demand by family", None, "2026-11-02", "Hot send-path values carried on the stack (the lead's placement, P-A). Single-worker routers have 3-7 registers of headroom at the peak; every multi-VC router is at or over budget (doc).",
        opt_level="-O3")
# receiver demand (§12, S2 peak, budget 27)
for fam, topo, lo, hi in (("1D, ring", "1D line / ring", 26, 28), ("2D, Galaxy 2D", "2D mesh", 27, 28), ("torus", "torus", 29, 30), ("2D + VC2", "2D mesh", 34, 35),
                          ("torus + VC2", "torus", 34, 35), ("VC1", "VC1 split (two 2x2 meshes)", 28, 36), ("one ERISC", "all", 15, 21)):
    k = fam.replace(" ", "").replace(",", "").replace("+", "p")
    rec(f"inv-rx-{k}-peakS2", "offline", topo, "receiver E1 loop (S2 placement = the lead's), -O3, loopvars.py --live", fam, None, None, "v4rxo4", "peak_live_max", hi, "count", RO,
        "§12 Demand by family (peak, S2)", None, "2026-11-02",
        f"Peak live values (registers + stack) over the family's images: {lo}-{hi} (max recorded). Budget 27 allocatable registers; the peak is always inside deliver's NoC programming. Two-channel routers are 1-10 over budget and GCC picks what goes to the stack.",
        opt_level="-O3")
# golden / g3 map facts
rec("map-rx-g3-vc2only", "offline", "2D mesh", "receiver E1, VC0 receiver trimmed (loop_local_vc 2), two ERISCs", "v4rxg3 vs v4rxg2 (28 images)", None, None, "v4rxg3",
    "loop_instructions_delta", -36, "count", RO, "§18 g3's 84 changed images", None, "2026-11-02",
    "-O3 loop -36 instructions, sp stores -7, loads -9; VC2's in-memory state spills gone. -Os: -12 / -6 / -9. Only mock-captured trimmed shapes change; every production image identical.",
    opt_level="-O3")
rec("map-rx-f4-golden-header", "offline", "2D mesh", "receiver E1 golden maps on census caches, two ERISCs", "header word spill", None, None, "v4rxf4", "spills", 36, "count", RO,
    "§17 Guard / golden maps", None, "2026-11-02",
    "v4rxf4's receivers spill the header word on 36 configs (lead v4rxv2i3: the route word on 72), plus the first VC's idle count and credit_frame_due; VC1: header word and volatile header on 20 configs (lead 4).",
    opt_level="-O3")

# ---------------------------------------------------------------- identity of readability stages (codegen held under edits)
ident = [
    ("v4txo0", "v4rxg3", "-O3", 2863, 2863, "X0: names and one-block regroup; byte-identical (coverage_txo.py, swap guard)", SO, "§12 X0", "2026-11-02"),
    ("v4txo0", "v4rxg3", "-Os", 2863, 2863, "X0 at -Os (3,124 configs, ok-sets equal)", SO, "§12 X0", "2026-11-02"),
    ("v4txo2", "v4txo1", "-O3", 2863, 2863, "X2: host facts; byte-identical to X1 at -O3 and -Os", SO, "§12 X2", "2026-11-02"),
    ("v4rxg1", "v4rxf4", "-O3", 2713, 2713, "g1 renames, asserts out", RO, "§18 Identity", "2026-11-02"),
    ("v4rxg1", "v4rxf4", "-Os", 2713, 2713, "g1 at -Os", RO, "§18 Identity", "2026-11-02"),
    ("v4rxg3", "v4rxg2", "-O3", 2779, 2863, "g3: 84 differ, all VC0-receiver-trimmed mock captures; all 2,713 original configs identical", RO, "§18 Identity", "2026-11-02"),
    ("v4rxf1", "v4rxe1", "-O3", 2697, 2713, "flat RingProgress: 16 UDM images change single::receiver_pass (one load moved; no path longer; accepted)", RO, "§16 Identity", "2026-11-02"),
    ("v4rxf1", "v4rxe1", "-Os", 2713, 2713, "", RO, "§16 Identity", "2026-11-02"),
    ("v4rxf4", "v4rxf3", "-O3", 2713, 2713, "C1 comment trim", RO, "§17 Identity", "2026-11-02"),
]
for router, base, lvl, same, total, n, src, sec, date in ident:
    rec(f"ident-{router}-{base}-{lvl}", "offline", "all coverage cases", f"readability stage {router} vs {base}, {lvl}", "full coverage .text compare", None, None, router,
        "images_identical", same, "count", src, sec, None, date, f"{same} of {total} built images byte-identical. {n}".strip(), images_total=total, opt_level=lvl)
rec("ident-v4rxe1-O3-changed", "offline", "all coverage cases", "readability stage v4rxe1 vs v4rxs2x, -O3", "full coverage .text compare", None, None, "v4rxe1",
    "images_changed", 1024, "count", RO, "§15 Identity", None, "2026-11-02",
    "1,024 of 2,713 images differ, all two-ERISC E1: 880 within the layout-only classes (frame offsets, init stores, frame size); 74 outside, all instrumentation builds (register choice/order of a few loop instructions).",
    images_total=2713, opt_level="-O3")
rec("ident-v4rxe1-Os-loopchanged", "offline", "all coverage cases", "readability stage v4rxe1 vs v4rxs2x, -Os", "full coverage .text compare", None, None, "v4rxe1",
    "images_changed_in_loop", 326, "count", RO, "§15 Identity (-Os: the polish is not layout-only)", None, "2026-11-02",
    "326 images change register allocation or scheduling inside the loop across production families (bh_trim_gtest 38, bh_2d_vc2 28, ...), plus 446 with a memset call in setup. Cause: a dead cursor field grew the frame 16-32 B and -Os re-laid the loop; a by-value helper also changed -O3 instrumentation loops (fixed in v4rxf1 by copy-init from a reference).",
    images_total=2713, opt_level="-Os")
rec("ident-v4txo1-guard", "offline", "all coverage cases", "sender X1 structure vs X0, -O3", "guard_x0_x1_O3.txt", None, None, "v4txo1", "images_changed_in_loop", 1695, "count", SO,
    "§12 X1 Static", None, "2026-11-02",
    "1,695 E0 / one-ERISC images differ (every E1 identical); loop deltas -10..+2 instructions by family; one new spill on Galaxy express (return_slots<0>:returned). -Os: only Galaxy express changes. Device A/B of X1 not yet in the doc.",
    opt_level="-O3")

# ---------------------------------------------------------------- opt-level sensitivity
# job 367 (lead v4rxv2i3), VC1 split and inter-mesh at -Os (builder default) vs -O3 vs -Os + note<> always_inline
J367 = "fabric_exp/results/vo{a,b}vc1[i]_{os,osni,o3}_default_p1_*_13129088_vc1_tests.csv"
for suite, topo, os_, osni, o3, pct in (("VC1 split (vc1_alloc, 13 rows)", "VC1 split (two 2x2 meshes)", 178.5, 168.4, 159.3, 10.7),
                                        ("VC1 inter-mesh (vc1_inter, 8 rows)", "VC1 inter-mesh", 385.5, 372.3, 336.6, 12.7)):
    k = "vc1" if "split" in suite else "vc1i"
    rec(f"opt-367-{k}-pct", "P150x8", topo, "two ERISCs, VC1 routers, -O3 vs -Os (builder default)", suite, None, None, "v4rxv2i3", "opt_level_delta", pct, "%", MEM,
        "2026-11-01 host changes paragraph ('Job 367: -O3 beats -Os by 10.7% on the VC1 split and 12.7% inter-mesh')", "367", "2026-11-01",
        "Suite-mean gain of -O3 over -Os, same job (vc1opt_runs.sh, passes a/b). The decision followed: production builds v4's VC1 routers at -O3 (host since job 399); main's v1 router stays -Os (it does not fit at -O3). tt_buf_rw::note<> always_inline recovers about half at -Os (skipped: a dataflow-API change for every kernel).",
        opt_level="-O3 vs -Os")
    for arm, v, lvl in (("os", os_, "-Os"), ("osni", osni, "-Os + note<> always_inline (v4rxv2i3ni)"), ("o3", o3, "-O3")):
        rec(f"opt-367-{k}-{arm}", "P150x8", topo, f"two ERISCs, VC1 routers, {lvl}", suite, None, None, "v4rxv2i3ni" if arm == "osni" else "v4rxv2i3", "rate_suite_mean", v, CP,
            J367, "computed with census_report.py's method (slowest sender per test, MixCmd excluded, mean of passes a and b)", "367", "2026-11-01",
            "No doc table holds these means; computed read-only from the job's CSVs with census_report.py's rows() logic. They reproduce the memory note's 10.7% / 12.7% (1 - O3/Os) exactly.",
            same_job=False, pair_key=f"opt367-{k}", opt_level=lvl)
# main's router: -Os vs -O2 on VC1 routers (jobs 089/090), -O3 does not fit
for test, os_, o2 in (("1-hop E/W 16 B", 307.2, 305.4), ("1-hop E/W 4352 B", 310.8, 306.6), ("1-hop N/S 16 B", 309.6, 304.9), ("1-hop N/S 4352 B", 303.5, 302.8), ("multicast 4352 B", 553.9, 548.7)):
    k = test.replace(" ", "").replace("/", "")
    size = 16 if "16 B" in test else 4352
    for lvl, v in (("-Os", os_), ("-O2", o2)):
        rec(f"opt-main-vc1-{k}-{lvl}", "P150x8", "VC1 split (two 2x2 meshes)", f"main's router (kernel v1, flags 0) on VC1 routers, {lvl}", test, "write", size, "main", "rate", v, CP,
            RES, "Main's router on VC1 routers: the reference for v4's VC1 work (2026-10-27)", "089/090", "2026-10-27",
            "VC0 rows on VC1 routers, slowest sender; each level in a fresh cache; main -O3 does not fit the kernel config buffer. Doc: 'The opt level is worth ~1%'; main core-bound at ~305 on every 1-hop row.",
            pair_key=f"main-opt-{k}", opt_level=lvl)
for lvl, v, note in (("-O3", 29940, "over the 26,624 B buffer by 3,316: does not fit"), ("-Os", 25252, "1,372 B left")):
    rec(f"size-main-vc1-{lvl}", "offline", "VC1 split (two 2x2 meshes)", f"main's router, VC1 E0 + E1 (largest of each), {lvl}", "bh_vc1", None, None, "main", "code_size", v, "B", SIZE,
        "Baseline (2026-10-30; 32 cases both build)", None, "2026-10-30", f"{note}. Main builds VC1 at -Os, everything else at -O3 (get_kernel_opt_level). Least free stack E0 at -O3 424 B (VC1), at -Os 320 B.",
        pair_key=f"size-vc1-{lvl}", opt_level=lvl)
for router, lvl, v, date, sec in (("v4pl6", "-O3", 23264, "2026-10-30", "Baseline"), ("v4pl6", "-Os", 22596, "2026-10-30", "Baseline"),
                                  ("v4rxf4", "-O3", 22996, "2026-11-02", "Log (v4rxf4)"), ("v4rxv2i3", "-Os", 22564, "2026-11-01", "Log (job 367 row)")):
    rec(f"size-{router}-vc1-{lvl}", "offline", "VC1 split (two 2x2 meshes)", f"VC1 E0 + E1 (largest of each), {lvl}", "bh_vc1", None, None, router, "code_size", v, "B", SIZE, sec, None, date,
        "Limit 26,624 B (kernel config buffer, also holds runtime args). v4rxg3 = v4rxf4's size. Cross-design comparison with main's offline build of the same captures.",
        pair_key=f"size-vc1-{lvl}" if router == "v4pl6" else None, opt_level=lvl)
rec("stack-main-vc1-E0-O3", "offline", "VC1 split (two 2x2 meshes)", "main's router, least free stack E0, -O3", "bh_vc1", None, None, "main", "free_stack", 424, "B", SIZE, "Baseline (2026-10-30)",
    None, "2026-10-30", "Main E0 keeps its sender state in kernel_main's frame (1,792 B on VC1 at -O3). v4pl6: 1,604 B. -Os: main 320 B, v4pl6 1,556 B.", pair_key="stack-vc1-E0", opt_level="-O3")
rec("stack-v4pl6-vc1-E0-O3", "offline", "VC1 split (two 2x2 meshes)", "least free stack E0, -O3", "bh_vc1", None, None, "v4pl6", "free_stack", 1604, "B", SIZE, "Baseline (2026-10-30)",
    None, "2026-10-30", "Unchanged through v4rxf4 (1,604 B at -O3).", pair_key="stack-vc1-E0", opt_level="-O3")
# study predecessors: s79mc (§82), v3t (§95), v3tk (§96)
for suite, o2, os_ in (("4352", 3.83, 25.33), ("mix", 3.94, 27.85), ("flushmix", 2.59, 21.98), ("1D", 2.55, 19.14)):
    for lvl, v in (("-O2", o2), ("-Os", os_)):
        rec(f"opt-s79mc-{suite}-{lvl}", "P150x8", "2D mesh" if suite != "1D" else "1D line", f"two ERISCs, {lvl} vs -O3", f"{suite} suite mean", None, None, "s79mc", "opt_level_delta", v, CP, RES,
            "§82 Consolidation Phase 0", "206-208", "2026-10-24",
            "Study router before v3/v4 (s79m, kernel v2 slot; not main, not v4). Doc: -Os costs +19 to +28 per suite (E/W 16 B 121.7 -> 146.9, +21%); the steepest edge found.", opt_level=lvl)
for lvl, vals in (("-O3", (-8.4, -4.8, -3.8, -6.8)), ("-O2", (-6.6, -4.5, -3.9, -6.3)), ("-Os", (-5.4, -2.1, -1.3, -1.4))):
    for suite, v in zip(("4352", "mix", "flushmix", "1D"), vals):
        rec(f"opt-v3t-{suite}-{lvl}", "P150x8", "2D mesh" if suite != "1D" else "1D line", f"two ERISCs, {lvl}", f"{suite} suite mean vs v3a -O3", None, None, "v3t", "suite_mean_delta", v, CP, RES,
            "§95 suite means vs v3a -O3", "277-281", "2026-10-25", "v3 predecessor (not v4). v3t at -Os beats v3a at -O3 on every suite but flushmix4; VC1 34/34 at -Os/-O2/-O3.", opt_level=lvl)
rec("opt-v3tk-Os-cost", "P150x8", "2D mesh", "two ERISCs, -Os vs -O3", "suite means (4352/mix/flushmix/1D)", None, None, "v3tk", "opt_level_delta", 5.3, CP, RES, "§96 Conclusions", "285/286", "2026-10-25",
    "Upper end of '-Os costs +2.8 to +5.3 on the means (v3tk -Os vs -O3)'; '-Os is not more stable than -O3'. v3 predecessor.", opt_level="-Os")
rec("opt-s79mc-vc1-O3-load", "P150x8", "VC1 split (two 2x2 meshes)", "VC1 routers at -O3", "program load", None, None, "s79mc", "code_size", 30048, "B", RES, "§84", "216", "2026-10-24",
    "'Program size (30048) too large for kernel config buffer (26624) on ACTIVE_ETH'. Study router s79mc (pre-v3), not main.", opt_level="-O3")

# ---------------------------------------------------------------- jitter band (ordinary edits; v3 predecessors; no v4 run)
JB = [("v3tk", "-O3", "4352", -0.10, 0.08), ("v3tk", "-O3", "mix", -0.01, 0.19), ("v3tk", "-O3", "flushmix", -0.03, 0.04), ("v3tk", "-O3", "1D", -0.07, 0.36),
      ("v3tk", "-Os", "4352", -0.28, 0.00), ("v3tk", "-Os", "mix", -0.01, 0.02), ("v3tk", "-Os", "flushmix", -0.36, 0.03), ("v3tk", "-Os", "1D", -0.31, 0.07)]
for router, lvl, suite, lo, hi in JB:
    for end, v in (("min", lo), ("max", hi)):
        rec(f"jit-{router}-{lvl}-{suite}-{end}", "P150x8", "2D mesh" if suite != "1D" else "1D line", f"two ERISCs, 8 ordinary edits, {lvl}", f"{suite} suite mean change", None, None, router,
            "jitter_band", v, CP, RES, "§96 Runs (suite means)", "285/286", "2026-10-25",
            f"{end} of the 8 edits' mean change vs v3tk (run-to-run -O3 / -Os: 4352 +0.02 / -0.01, mix +0.01 / -0.01, flushmix +0.01 / -0.06, 1D +0.01 / -0.09). v3 predecessor; no jitter-band run exists for any v4 variant.",
            opt_level=lvl)
for lvl, med, p90, big in (("-O3", 0.06, 0.25, 6), ("-Os", 0.06, 0.31, 3)):
    for m, v in (("median", med), ("p90", p90), ("max", big)):
        rec(f"jit-v3tk-rows-{lvl}-{m}", "P150x8", "mixed", f"two ERISCs, 8 ordinary edits, {lvl}", "single rows (608 row results)", None, None, "v3tk", f"jitter_band_rows_{m}", v, CP, RES,
            "§96 Single rows", "285/286", "2026-10-25", "|change| per row. Largest -O3: 1D scatter / half-ring rows 3-6 under alignment-only edits; -Os largest outside two bimodal rows. Doc yardstick: ±0.5 on means, ±6 (or ±4%) on rows.", opt_level=lvl)
for router, lvl, v, jobs, date, src, sec, n in (
        ("v3v6", "-O3", 0.06, "314/315", "2026-10-26", RES, "§98 v3v6 jitter band", "2D, mix, flushmix means within ±0.06 at -O3 (±0.03 at -Os); 1D +0.27 at most at -O3."),
        ("v3v6", "-Os", 0.03, "314/315", "2026-10-26", RES, "§98 v3v6 jitter band", "2D, mix, flushmix within ±0.03 at -Os."),
        ("v3x", "-O3", 0.03, "332/333", "2026-10-26", RES, "§98 Jitter band (v3x)", "2D, mix, flushmix means ±0.03 at -O3 and -Os; 1D -0.38..+0.19."),
        ("v3z", "-O3", 0.18, "350/351", "2026-10-26", RES, "§98 Jitter band (v3z)", "2D, mix, flushmix means within -0.18..+0.06 at both levels (magnitude recorded); edit row p90 0.28 / 0.25 vs run-to-run 0.22 / 0.33.")):
    rec(f"jit-{router}-{lvl}-absmax", "P150x8", "2D mesh", f"two ERISCs, 8 ordinary edits, {lvl}", "2D/mix/flushmix suite means", None, None, router, "jitter_band_absmax", v, CP, src, sec, jobs, date,
        n + " v3 predecessor.", opt_level=lvl)
rec("jit-s79mc-edits", "P150x8", "2D mesh", "two ERISCs, -O3; 2 struct edits + 4 compiler flags (10-302 hot instructions changed)", "suite means", None, None, "s79mc", "jitter_band_absmax", 0.14, CP, RES,
    "§82 table", "206-208", "2026-10-24", "Every suite within noise (max |change| 0.14, a 1D mean pulled by noisy half-ring rows); structural edits, not random ones, moved this router before (§77-§81).", opt_level="-O3")
# hook probe: a noinline debug call per pass in each loop
for router, vals, jobs, date, sec in (("s79mc", (0.99, 2.16, 1.27, 0.79), "209", "2026-10-24", "§83"),
                                      ("v3tk", (2.1, 2.5, 1.2, 1.4), "283/284", "2026-10-25", "§95 keepalive / hook reproduces")):
    for suite, v in zip(("4352", "mix", "flushmix", "1D"), vals):
        rec(f"hook-{router}-{suite}", "P150x8", "2D mesh" if suite != "1D" else "1D line", "two ERISCs, -O3, + a noinline debug call per pass (perturb.py hook)", f"{suite} suite mean change",
            None, None, router, "fragility_delta", v, CP, RES, sec, jobs, date, "Cost of a contributor-style hook; the probe's own compare is about 1 per pass. Not run on any v4 variant.", opt_level="-O3")
# 2x2 offline test (§97)
for cell, v, n in (("P0 (per-round work out of line)", 0, "never: loop code identical or moved under all 4 cold edits, every level, with or without t*17"),
                   ("P1 (per-round work inlined)", 3, "3 of 4 cold edits change the loop at -O3 and -O2 (none at -Os), whole-function re-layouts")):
    rec(f"x2x2-{cell[:2]}", "offline", "2D mesh / 1D line (W, N/S, 1D configs)", f"v3tk 2x2 cell {cell}", "cold edits changing run_main_loop", None, None, "v3tk", "edits_changing_loop", v, "count",
        RES, "§97 Result", None, "2026-10-25", n + ". v4 keeps per-round work out of line (end_round runs once per round; V4_RECEIVER_OWNERSHIP §12).")

# ---------------------------------------------------------------- main's router: codegen facts
rec("main-vc1vc2-stack", "offline", "VC1 split (two 2x2 meshes)", "main's router, VC1 + VC2, two ERISCs", "job 320 capture build", None, None, "main", "stack_usage", 2176, "B", RS, "§28", "320", "2026-11-01",
    "Build fails: 'stack usage is 2176 bytes [-Werror=stack-usage=]'. One ERISC fails on 'VC2 receiver channel not serviced'. v4rxc7v builds and runs it (26,076 B, least free stack E0 1,476 / E1 3,408).")

# VC2 line multi-mesh code size by level (device-loaded)
for lvl, v in (("-O3", 28420), ("-Os", 27436)):
    rec(f"size-vc2line-split-{lvl}", "BH Galaxy", "VC1 split (two 2x2 meshes)", f"VC2 line mode + VC1, two ERISCs, {lvl}", "loaded router size", None, None, "vc2l_p3b", "code_size", v, "B", VC2L,
        "Phase 3 / Galaxy runs (job 167327)", "167327", "2026-11-01", "Both exceed 26,624 B: -Os does not rescue the fit; the buffer grows to 29 KiB (dev_mem_map.h) for this config.", opt_level=lvl)

json.dump(recs, open(OUT, "w"), indent=1)
print(len(recs), "records")
