#!/usr/bin/env python3
"""Builds fabric_exp/report/data/fanin.json from the study's notes (values transcribed from the cited sections)."""
import json, sys

OUT = sys.argv[1]
VC2L = "/home/aagarwal/tt-metal-vc2line/fabric_exp/consolidation/VC2_LINE_BRINGUP.md"
VC2L_1D = "/home/aagarwal/tt-metal-vc2line/fabric_exp/consolidation/VC2_LINE_1D_VCS.md"
VC2L_HOST = "/home/aagarwal/tt-metal-vc2line/fabric_exp/consolidation/LINE_VCS_HOST_DESIGN.md"
RS = "fabric_exp/consolidation/V4_RECEIVER_SHAPE.md"
SO = "fabric_exp/consolidation/V4_SENDER_OWNERSHIP.md"
SB = "fabric_exp/consolidation/V4_SEND_BATCH.md"
SS = "fabric_exp/consolidation/SLOT_STUDY.md"
BND = "fabric_exp/consolidation/V4_BOUNDS.md"
PROF = "fabric_exp/consolidation/V4_PROFILE.md"
RES = "fabric_exp/RESULTS.md"
RO = "fabric_exp/consolidation/V4_RECEIVER_OWNERSHIP.md"

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


CP = "cycles/packet"
XRUN = "cross-run: main's router has no VC2 traffic rate in any source (main runs VC2 neighbour exchange on two ERISCs only, job 141, pass/fail)."

# ---------------------------------------------------------------- P150: VC0 + VC2 (neighbour VC2) on one link, lead v4rxv2i3
for slot, sl in (("d", "17,14,16 + VC2 2,4 (study row)"), ("v8", "17,10,16 + VC2 8,10")):
    rows = {
        "d": [("ew4352 (VC2 on, VC2 idle)", "vc0_alone", 134.6, "333"), ("vc2ew4352", "vc2_alone", 858.1, "333"),
              ("vc02ew4352", "vc0_contended", 163.6, "340"), ("vc02bew4352", "vc2_contended", 989.5, "340"),
              ("vc02ew4352 + vc02bew4352", "combined_derived", 140.4, "340")],
        "v8": [("ew4352 (VC2 on, VC2 idle)", "vc0_alone", 134.8, "333"), ("vc2ew4352", "vc2_alone", 240.6, "333"),
               ("vc02ew4352", "vc0_contended", 190.7, "340"), ("vc02bew4352", "vc2_contended", 368.7, "340"),
               ("vc02ew4352 + vc02bew4352", "combined_derived", 125.7, "340")],
    }[slot]
    for test, what, v, job in rows:
        metric = {"vc0_alone": "rate", "vc2_alone": "rate", "vc0_contended": "rate_contended",
                  "vc2_contended": "rate_contended", "combined_derived": "rate_combined_derived"}[what]
        note = {
            "vc0_alone": "VC0 sender alone with VC2 compiled in and idle.",
            "vc2_alone": "VC2 sender alone; window-bound by VC2's worker slots (2 slots ~860-880, 8 slots ~240; §35).",
            "vc0_contended": "VC0 sender while VC2 sends on the same link (VC0 finishes first, so VC0's rate is the contended one).",
            "vc2_contended": "VC2 sender while VC0 sends (vc02bew: VC0 40M, VC2 2M packets, so VC2's rate is the contended one).",
            "combined_derived": "The doc's estimate from the two contended rows, 1/(1/VC0 + 1/VC2); not a direct measurement.",
        }[what]
        bs, bst = None, None
        if slot == "v8" and what == "vc2_alone":
            bs, bst = "window", "hypothesis"
            note += " At 8 slots still about twice VC0's rate; the doc's window model says it needs ~16-18 worker slots (model, not an A/B at v8)."
        if slot != "d":
            pass
        elif what == "vc0_alone":
            bs, bst = "E0", "A/B"
            note += " Bound: VC2-on writes E0-bound (V4_BOUNDS §1, job 349: tx c32 +49.1, slack 0); idle VC2's cost is E0's (§29 one-side probe A/B, job 322, on v4rxc7)."
        elif what == "vc2_alone":
            bs, bst = "window", "A/B"
            note += " Window bound settled by slot A/B d vs v8 (same router, same job)."
        elif what == "vc0_contended":
            bs, bst = "E0", "A/B"
            note += " V4_BOUNDS §1 (job 349): VC0 sender in vc02ew4352 TX-bound by one-side delay A/B."
        elif what == "vc2_contended":
            bs, bst = "E1", "A/B"
            note += " V4_BOUNDS §1 (job 349) lists the VC2 sender in vc02bew4352 as RX-bound (one-side delay A/B; the slot row there is the study row d)."
        rec(f"p150-vc02-{slot}-{what}-v4rxv2i3", "P150x8", "2D mesh", f"two ERISCs, VC2 on (neighbour), slots {sl}",
            test, None, 4352, "v4rxv2i3", metric, v, CP, RS, "§35 (job 333) / §36 (job 340)", job, "2026-11-01",
            note + " " + XRUN, bs, bst, slots=sl, opt_level="-O3")

# idle VC2 on/off at the lead (job 326)
for test, on, off, bs in (("2D E/W write 4352", 134.6, 126.8, "E0"), ("TorusY1HopNS write", 166.9, 149.2, "both")):
    t = "2D mesh" if test.startswith("2D") else "Y-torus"
    rec(f"p150-idlevc2-{t[:2]}-on-v4rxv2i3", "P150x8", t, "two ERISCs, VC2 on and idle, slots 17,14,16 + 2,4", test,
        "write", 4352, "v4rxv2i3", "rate", on, CP, RS, "§32 table (job 326)", "326", "2026-11-01",
        f"VC0 traffic with VC2 compiled in and idle; same job VC2 off {off}: idle VC2 costs {round(on-off,1)} (doc: 7.7 / 17.7). " + XRUN,
        bs, "A/B", slots="17,14,16 + 2,4", opt_level="-O3")
    rec(f"p150-idlevc2-{t[:2]}-off-v4rxv2i3", "P150x8", t, "two ERISCs, VC2 off, slots 17,14,16", test, "write", 4352,
        "v4rxv2i3", "rate", off, CP, RS, "§32 table (job 326)", "326", "2026-11-01",
        "Control for the idle-VC2 row (same job).", None, None, opt_level="-O3")
# idle VC2 cost: one-side probe A/B (job 322, v4rxc7)
for test, off, on, p1, p2, p3 in (("2D E/W write 4352", 126.9, 152.7, 129.3, 150.7, 155.7),
                                  ("TorusY1HopNS write", 149.2, 181.5, 167.2, 175.9, 182.9)):
    t = "2D mesh" if test.startswith("2D") else "Y-torus"
    for arm, v, n in (("vc2on", on, "VC2 on, v4rxc7 (pre-lead)"), ("p1", p1, "probe v4rxv2p1: E0 neither reads nor visits VC2's worker (one-side lean, E0 only)"),
                      ("p3", p3, "probe v4rxv2p3: E1 neither reads nor steps VC2's receiver (one-side lean, E1 only)")):
        rec(f"p150-idlevc2-ab-{t[:2]}-{arm}", "P150x8", t, "two ERISCs, VC2 on and idle, slots 17,14,16 + 2,4", test, "write",
            4352, "v4rxv2p1" if arm == "p1" else ("v4rxv2p3" if arm == "p3" else "v4rxc7"), "rate", v, CP, RS, "§29 (job 322)", "322",
            "2026-11-01", f"{n}. VC2 off {off}. Doc: idle VC2's cost is E0's (A/B); E0's idle VC2 visit is 91% of the 2D gap, 44% of the torus gap.",
            "E0", "A/B", opt_level="-O3")

# ---------------------------------------------------------------- P150: sender placement probes on VC0+VC2 fan-in (v8)
# §8.4 table (lead and P-C from §8.3 jobs 422/425/432; P-B re-run with packed in 445/446, within 0.3-0.5 of §8.3)
cols = [("v4rxf1", "lead (P-A); sender identical to v4rxf4/v4rxg3", "422/425/432", SO + "", "§8.3 / §8.4 table"),
        ("v4txpc", "probe P-C (later worker state in memory)", "422/425/432", SO, "§8.3 / §8.4 table"),
        ("v4txpb", "probe P-B (later VCs' state and far rings in memory)", "422/425/432; 445/446", SO, "§8.4 table"),
        ("v4txpbk", "probe packed P-B (far ring one word; the user's adopted sender placement)", "445/446", SO, "§8.4 table")]
fan = {
    256: {"vc0": [201.9, 200.2, 216.8, 204.8], "vc2": [281.7, 281.8, 279.1, 281.2], "comb": [117.6, 117.0, 122.0, 118.5],
          "share": [58, 58, 56, 58]},
    1024: {"vc0": [201.3, 199.8, 215.7, 204.2], "vc2": [283.7, 283.5, 282.6, 283.8], "comb": [117.8, 117.2, 122.3, 118.8],
           "share": [None, None, None, None]},
    4352: {"vc0": [190.4, 190.1, 205.2, 191.3], "vc2": [369.9, 371.1, 324.5, 366.5], "comb": [125.7, 125.7, 125.7, 125.7],
           "share": [66, 66, 61, 66]},
}
alone = {256: [223.7, 224.6, 228.2, 224.2], 1024: [226.5, 226.4, 229.5, 226.0], 4352: [240.8, 242.3, 245.5, 241.9]}
for size, d in fan.items():
    for i, (router, desc, job, sf, sec) in enumerate(cols):
        base = f"two ERISCs, VC2 on (neighbour), v8 slots 17,10,16 + VC2 8,10; {desc}"
        rec(f"p150-fanin-v8-{size}-vc0-{router}", "P150x8", "2D mesh", base, "vc02ew" + str(size), None, size, router,
            "rate_contended", d["vc0"][i], CP, sf, sec, job, "2026-11-02",
            "VC0 sender while VC2 busy on the same link (from vc02ew; VC0 finishes first). Two passes in reversed order, within 0.3-0.4. " + XRUN,
            "E0" if router == "v4rxf1" else None, "hypothesis" if router == "v4rxf1" else None, slots="17,10,16 + 8,10", opt_level="-O3")
        rec(f"p150-fanin-v8-{size}-vc2-{router}", "P150x8", "2D mesh", base, "vc02bew" + str(size), None, size, router,
            "rate_contended", d["vc2"][i], CP, sf, sec, job, "2026-11-02",
            "VC2 sender while VC0 busy (from vc02bew; VC2 finishes first). " + ("P-B's 4352 B VC2 gain is a share shift (VC0's share 66% -> 61%), timing hypothesis (TXQ), isolating delay A/B not run. " if router == "v4txpb" and size == 4352 else "") + XRUN,
            None, None, slots="17,10,16 + 8,10", opt_level="-O3")
        rec(f"p150-fanin-v8-{size}-comb-{router}", "P150x8", "2D mesh", base, "vc02ew/vc02bew combined", None, size, router,
            "rate_combined_derived", d["comb"][i], CP, sf, sec, job, "2026-11-02",
            "Link total while both VCs send, the doc's 1/(1/VC0 + 1/VC2) from the two contended rows (an estimate made as in V4_RECEIVER_SHAPE §36). " + ("At 4352 B pinned at 125.7 for every router: the link or the far side bounds it (doc; not an A/B). " if size == 4352 else "At 256/1024 B the doc calls these rows E0-bound. ") + XRUN,
            None, None, slots="17,10,16 + 8,10", opt_level="-O3")
        if d["share"][i] is not None:
            rec(f"p150-fanin-v8-{size}-share-{router}", "P150x8", "2D mesh", base, "vc02 share", None, size, router,
                "vc0_share", d["share"][i], "%", sf, sec, job, "2026-11-02", "VC0's share of the packets while both send.",
                None, None, slots="17,10,16 + 8,10", opt_level="-O3")
        rec(f"p150-vc2alone-v8-{size}-{router}", "P150x8", "2D mesh", base, "vc2ew" + str(size), None, size, router,
            "rate", alone[size][i], CP, sf, sec, job, "2026-11-02",
            "VC2 alone at 8 worker slots: window-bound (doc). " + XRUN, "window", "hypothesis", slots="17,10,16 + 8,10", opt_level="-O3")
idle = [134.6, 131.4, 131.8, 131.2]
for i, (router, desc, job, sf, sec) in enumerate(cols):
    rec(f"p150-idlevc2-ew4352-{router}", "P150x8", "2D mesh", f"two ERISCs, VC2 on and idle, study row 17,14,16 + 2,4; {desc}",
        "ew4352", "write", 4352, router, "rate", idle[i], CP, sf, sec, job, "2026-11-02",
        "VC0 traffic with VC2 compiled in and idle (E0-bound, slack 0, V4_BOUNDS §1). VC2 off on the same routers: 126.9 (packed 127.0).",
        "E0", "A/B", slots="17,14,16 + 2,4", opt_level="-O3")

# multi-VC total vs one VC (jobs 389/390)
rec("p150-twovc-total-sb", "P150x8", "2D mesh", "two ERISCs, two busy VCs on one link", None, None, None, "v4rxv2i3",
    "rate_combined", 115, CP, SB, "User decisions (2026-11-02)", "389/390", "2026-11-01",
    "Doc text: 'Two busy VCs on one P150 link total ~115 cycles/packet at 256 B-1 KiB, the same as one VC'; value approximate as written; no table; router not named in the sentence (the lead at the time, v4rxv2i3, by context; jobs 389/390 fall between job 388 (2026-11-01 22:10) and job 400). The doc concludes the sender's per-packet work is the ceiling for multi-VC fan-in (inference, not an A/B).")

# ---------------------------------------------------------------- Main's VC2 alternatives
rec("main-vc2-1erisc-build", "P150x8", "2D mesh", "one ERISC, VC2 on", "test_fabric_vc2_at_least_2x2_mesh.yaml", None, None,
    "main", "builds", 0, "bool", RES, "VC2 on v4: v4vc2, first device runs (§98 area, line ~8289)", "141", "2026-10-29",
    "Main does not build VC2 on one ERISC: static_assert 'VC2 receiver channel not serviced'. Main two ERISCs: 10/10 pass. v4vc2 (job 152): 10/10 on two ERISCs and on one.",
    same_job=False, pair_key="vc2-1erisc-build")
rec("v4-vc2-1erisc-build", "P150x8", "2D mesh", "one ERISC, VC2 on", "test_fabric_vc2_at_least_2x2_mesh.yaml", None, None,
    "v4vc2", "tests_passed", 10, "count", RES, "VC2 on v4: v4vc2 (job 152)", "152", "2026-10-29",
    "10/10 on one ERISC (and 10/10 on two). Cross-run with main's build failure (job 141).", pair_key="vc2-1erisc-build")
rec("main-vc1vc2-build", "offline", "VC1 split (two 2x2 meshes)", "VC1 + VC2 (multi-mesh with VC2 on)", None, None, None,
    "main", "builds", 0, "bool", RS, "§28", "320", "2026-11-01",
    "Main's router cannot build VC1 + VC2: two ERISCs fail its stack limit ('stack usage is 2176 bytes [-Werror=stack-usage=]'), one ERISC on 'VC2 receiver channel not serviced' (job 320's capture).",
    pair_key="vc1vc2-build")
rec("v4-vc1vc2-device", "P150x8", "VC1 split (two 2x2 meshes)", "VC1 + VC2, two ERISCs, slots 19,8,16,4,4 + 2,4", "vc1_alloc", None, None,
    "v4rxc7v", "tests_passed", 25, "count", RS, "§28-§29 (device job 323)", "323", "2026-11-01",
    "25/25 per rank data-checked on two ERISCs (and one), vc1_inter 8/8, VC2 yaml 10/10 per rank. Size E0 13,792 + E1 12,284 = 26,076 B (548 B under 26,624). Cross-run with main's build failure.",
    pair_key="vc1vc2-build")
# main's VC2 tier slot row on the v4 router (job 318)
for test, study, tier in (("2D E/W write 4352", 152.6, 163.9), ("2D E/W scatter", 153.7, 183.6), ("2D N/S write 4352", 152.5, 164.0),
                          ("TorusY1HopNS write", 181.5, 182.2), ("TorusY1HopNS scatter", 208.8, 218.0)):
    t = "Y-torus" if "Torus" in test else "2D mesh"
    st = "scatter_write" if "scatter" in test else "write"
    key = test.replace(" ", "")
    rec(f"p150-vc2tier-{key}-tier", "P150x8", t, "two ERISCs, VC2 on and idle, main's tier slot row 43,7,11 + 2,4", test, st, 4352,
        "v4rxc7", "rate", tier, CP, RS, "§26 (job 318)", "318", "2026-11-01",
        "Main's VC0+VC2 tier slot row (Blackhole {7,11,0,0,2,4} + spare to VC0's worker, §21) applied to the study router; not main's router. Same job as the study-row arm.",
        slots="43,7,11 + 2,4", opt_level="-O3")
    rec(f"p150-vc2tier-{key}-study", "P150x8", t, "two ERISCs, VC2 on and idle, study row 17,14,16 + 2,4", test, st, 4352,
        "v4rxc7", "rate", study, CP, RS, "§26 (job 318)", "318", "2026-11-01",
        "Study slot row; same job as main's tier row arm.", slots="17,14,16 + 2,4", opt_level="-O3")

# ---------------------------------------------------------------- Tensix mux mode census rows (main vs v4trm, same job)
mux = [("MuxFullRingMulticast", "ring", 1024, 419.6, 435.9), ("MuxFullRingMulticast", "ring", 4352, 419.8, 436.6),
       ("MuxLinear1HopEW", "1D line", 1024, 312.6, 312.6), ("MuxLinear1HopEW", "1D line", 4352, 330.4, 330.7),
       ("MuxLinear1HopEWBidir", "1D line", 1024, 313.4, 313.4), ("MuxLinear1HopEWBidir", "1D line", 4352, 331.5, 331.3),
       ("MuxLinear3HopEW", "1D line", 1024, 312.6, 312.6), ("MuxLinear3HopEW", "1D line", 4352, 330.7, 330.6),
       ("MuxLinearMulticast", "1D line", 1024, 299.6, 281.4), ("MuxLinearMulticast", "1D line", 4352, 319.5, 300.6),
       ("MuxMesh1HopEW", "2D mesh", 1024, 445.4, 445.3), ("MuxMesh1HopEW", "2D mesh", 4352, 467.5, 467.4),
       ("MuxMesh1HopEWBidir", "2D mesh", 1024, 445.8, 445.8), ("MuxMesh1HopEWBidir", "2D mesh", 4352, 468.1, 467.8),
       ("MuxMesh3HopEW", "2D mesh", 1024, 445.4, 445.3), ("MuxMesh3HopEW", "2D mesh", 4352, 467.3, 467.4),
       ("MuxMeshMulticast", "2D mesh", 1024, 343.6, 343.6), ("MuxMeshMulticast", "2D mesh", 4352, 368.1, 367.5),
       ("MuxRing1HopEW", "ring", 1024, 340.6, 340.7), ("MuxRing1HopEW", "ring", 4352, 358.8, 358.7),
       ("MuxRingMulticast", "ring", 1024, 326.2, 331.2), ("MuxRingMulticast", "ring", 4352, 345.4, 351.7)]
for test, topo, size, m, v in mux:
    key = f"mux-{test}-{size}"
    both = test == "MuxFullRingMulticast"
    for router, val in (("main", m), ("v4trm", v)):
        rec(f"p150-census-{key}-{router}", "P150x8", topo, "mux mode (Tensix mux extension; one ERISC forced)", test, "write", size,
            router, "rate", val, CP, "fabric_exp/consolidation/census_report.py",
            "stdout of `census_report.py --rows --routers=main_fw,v4trm mux` over fabric_exp/results/cen{a,b}_{main_fw,v4trm}_mux.csv; RESULTS.md 'Perf census after the features' (2026-10-29)", "164-171", "2026-10-29",
            ("main = main_fw (main as shipped, firmware keepalive); mean of passes a and b. " if router == "main" else
             "Mean of passes a and b. Bound from the one-side delay A/B on v4trmd (V4_PROFILE §2, jobs 172-174; again on v4rxv2i3, V4_BOUNDS §1b job 357): "
             + ("both ERISCs (full-ring multicast, below main)." if both else "neither ERISC moves; the docs name the Tensix mux as the bound (by elimination; no mux-side A/B).")),
            (None if router == "main" else ("both" if both else "neither ERISC")), (None if router == "main" else "A/B"),
            True, key, opt_level="-O3")
rec("p150-census-mux-mean-main", "P150x8", "mixed (1D line, ring, 2D mesh)", "mux mode (Tensix mux extension; one ERISC forced)",
    "mux suite mean (22 rows)", "write", None, "main", "rate_suite_mean", 373.3, CP, "fabric_exp/consolidation/V4_PERF_GAPS.md",
    "census table (mux mode (22))", "164-171", "2026-10-29", "main as shipped (main_fw); main at keepalive 1000 also 373.3; v3qm 368.7.",
    same_job=True, pair_key="mux-suite-mean")
rec("p150-census-mux-mean-v4trm", "P150x8", "mixed (1D line, ring, 2D mesh)", "mux mode (Tensix mux extension; one ERISC forced)",
    "mux suite mean (22 rows)", "write", None, "v4trm", "rate_suite_mean", 373.6, CP, "fabric_exp/consolidation/V4_PERF_GAPS.md",
    "census table (mux mode (22))", "164-171", "2026-10-29", "v4trm / main 1.001 (equal): the mux, not the router, sets these rows.",
    same_job=True, pair_key="mux-suite-mean")
for router, v, job in (("v4rxv2i3", 371.2, "420-421, 423-424, 427-428"), ("v4rxf4", 371.1, "420-421, 423-424, 427-428")):
    rec(f"p150-census-mux-mean-{router}", "P150x8", "mixed (1D line, ring, 2D mesh)", "mux mode (Tensix mux extension; one ERISC forced)",
        "mux suite mean", "write", None, router, "rate_suite_mean", v, CP, RO, "§17 Census (passes f4a/f4b)", job, "2026-11-02",
        "Cross-run against main's 373.3 (job 164-171); v4rxg3 has v4rxf4's per-packet code. Row count of this census not restated in §17 (22 in the 2026-10-29 census).")

# ---------------------------------------------------------------- Galaxy: VC2 line mode (worktree tt-metal-vc2line, router vc2l_p3b)
GL = "vc2l_p3b"
GLN = "VC2 line mode on v4rxv2i3 (+ v4rxc7v's VC1+VC2 gate); trees vc2l_p4/p4b/p4c/p4d. Not yet re-measured on vc2l_g3a (= line mode on the lead v4rxg3; Galaxy jobs 168748/168774 queued). "
GXR = "No main counterpart: main has no VC2 line mode and no native multi-worker fan-in; main's alternative (mux core) not run here."
one = [("VC0 inside, mesh 0 chip 2 -> chip 1", "2 VC0 hops (north, then east)", 353.5, 402.7),
       ("VC2 inside, mesh 0 chip 0 -> chip 1", "1 VC2 hop", 465.0, 507.1),
       ("VC0 across, mesh 0 chip 2 -> mesh 1 chip 1", "2 VC0 hops, landing, 2 VC1 hops", 525.7, 578.6),
       ("VC2 across, mesh 0 chip 3 -> mesh 1 chip 1", "1 VC2 hop, landing, 2 VC1 hops", 554.2, 601.9)]
for i, (flow, path, k1, k4) in enumerate(one):
    for size, v in ((1024, k1), (4096, k4)):
        rec(f"glx-p4-alone-{i}-{size}", "BH Galaxy", "VC1 split (two 2x2 meshes)", "two ERISCs, VC2 line mode, slots VC0 16,7,16 + VC1 4,4 + VC2 4,4,4, -O3, sync",
            flow, None, size, GL, "rate", v, CP, VC2L, "Phase 4 / Performance: one flow alone", "167441/167449/167468", "2026-11-01",
            GLN + f"Path: {path}. Jobs agree within 0.6%. 4 KiB = 4096 B (13.7 GB/s at 402.7). " + GXR,
            ("window" if flow.startswith("VC2") else None), ("A/B" if flow.startswith("VC2 inside") else ("hypothesis" if flow.startswith("VC2") else None)),
            slots="16,7,16 + 4,4 + 4,4,4", opt_level="-O3")
for key, label, a0, w0, a2, w2, both in (
        ("exit", "mesh 0 chip 3's exit + the landing router's VC1", 578.9, 1063.8, 601.4, 1102.8, 551.4),
        ("link", "mesh 0 chip 0 -> chip 1 link (VC0 forwarded, VC2 injected)", 402.7, 453.3, 507.1, 586.6, 293.3)):
    for what, v, n in (("vc0_alone", a0, "VC0 alone"), ("vc0_with_vc2", w0, "VC0 with VC2 on the same path"),
                       ("vc2_alone", a2, "VC2 alone"), ("vc2_with_vc0", w2, "VC2 with VC0 on the same path"),
                       ("both", both, "both packets over the slower flow's cycles")):
        rec(f"glx-p4-twoflow-{key}-{what}", "BH Galaxy", "VC1 split (two 2x2 meshes)", "two ERISCs, VC2 line mode, default row, no sync",
            label, None, 4096, GL, "rate" if "alone" in what else ("rate_combined" if what == "both" else "rate_contended"), v, CP,
            VC2L, "Phase 4 / Two flows on one path (no sync, per flow)", "167468", "2026-11-01",
            GLN + n + ". " + ("Doc: neither VC starves; across the meshes VC0 51%, VC2 49%. " if key == "exit" else "Doc: VC0 finishes first; VC2 gets ~42% while both run (estimate). ") + GXR,
            slots="16,7,16 + 4,4 + 4,4,4", opt_level="-O3")
for flow, k, dflt, v1, v2 in (("VC2 inside, 1 hop", 4096, 506.6, 508.1, 293.9), ("VC2 inside, 1 hop", 1024, 465.1, 462.8, 264.4),
                              ("VC2 across (VC2 hop, VC1 hops)", 4096, 603.4, 598.7, 551.7), ("VC2 across (VC2 hop, VC1 hops)", 1024, 554.4, 547.9, 501.8)):
    for arm, v, sl in (("default", dflt, "16,7,16 + 4,4 + 4,4,4"), ("vc1x2", v1, "6,5,16,8,8,4,4,4 (VC1 8/8)"), ("vc2x2", v2, "4,7,16,4,4,8,8,8 (VC2 8/8/8)")):
        rec(f"glx-p4-winab-{flow[:10].replace(' ','')}-{k}-{arm}", "BH Galaxy", "VC1 split (two 2x2 meshes)", f"two ERISCs, VC2 line mode, window A/B arm {arm}",
            flow, None, k, GL, "rate", v, CP, VC2L, "Phase 4 / Window A/B", "167468", "2026-11-01",
            GLN + "Doc (A/B): VC2's 4-slot windows bound VC2 inside a mesh (-42%/-43% at 8/8/8; VC1 8/8 control unchanged); VC1 4/4 does not bound the VC2 flow across meshes (-0.8%); VC2 8/8/8 moves the across flow only -8.6%, and phase 4 calls its bound 'not established' while the later perf section (job 167482) says VC2 flows across are bound by VC2's windows first.",
            "window", "A/B" if flow.startswith("VC2 inside") else "hypothesis", slots=sl, opt_level="-O3")
# idle VC cost, one mesh (job 167481), two ERISCs
idle1 = [("1-hop E/W", [(256, 124.9, 10.5, 20.0), (1024, 127.8, 9.5, 17.3), (4352, 252.0, 0.0, 0.0)]),
         ("1-hop N/S", [(256, 125.3, 10.6, 19.6), (1024, 128.1, 10.0, 17.0), (4352, 252.0, 0.0, 0.0)]),
         ("3-hop E/W", [(256, 279.1, 22.9, 30.9), (1024, 287.5, 22.6, 30.0), (4352, 298.8, 34.6, 46.4)]),
         ("4-hop N/S", [(256, 390.6, 18.8, 23.3), (1024, 397.6, 18.3, 24.1), (4352, 403.8, 53.6, 60.0)]),
         ("multicast E:3", [(256, 291.7, 17.7, 26.4), (1024, 299.8, 18.9, 25.8), (4352, 312.8, 55.3, 64.5)])]
for row, cells in idle1:
    for size, off, nbr, line in cells:
        k = row.replace(" ", "").replace("/", "").replace(":", "")
        rec(f"glx-idle-mesh-{k}-{size}-off", "BH Galaxy", "2D mesh", "two ERISCs, VC2 off, VC0 pinned 16,7,16", row, None, size, GL,
            "rate", off, CP, VC2L, "Perf: idle and full VCs / 1. Idle-VC cost (one 8x4 mesh)", "167481", "2026-11-01",
            GLN + "VC0's rate with VC2 off.", slots="16,7,16", opt_level="-O3")
        rec(f"glx-idle-mesh-{k}-{size}-nbr", "BH Galaxy", "2D mesh", "two ERISCs, VC2 neighbour (idle), VC0 pinned 16,7,16", row, None, size, GL,
            "idle_vc_cost", nbr, CP, VC2L, "Perf: idle and full VCs / 1. Idle-VC cost (one 8x4 mesh)", "167481", "2026-11-01",
            GLN + "Delta of VC0's rate against VC2 off, same job. Which ERISC bears it not tested here.", slots="16,7,16", opt_level="-O3")
        rec(f"glx-idle-mesh-{k}-{size}-line", "BH Galaxy", "2D mesh", "two ERISCs, VC2 line mode (idle), VC0 pinned 16,7,16", row, None, size, GL,
            "idle_vc_cost", line, CP, VC2L, "Perf: idle and full VCs / 1. Idle-VC cost (one 8x4 mesh)", "167481", "2026-11-01",
            GLN + "Delta against VC2 off, same job; repeats within 0.9. The wire (~248 at 4352 B) hides it only on 1-hop at 4352 B.", slots="16,7,16", opt_level="-O3")
for row, size, off, nbr, line in (("1-hop E/W", 256, 199.2, 37.8, 53.4), ("1-hop E/W", 4352, 251.1, 0.0, 2.4),
                                  ("1-hop N/S", 256, 202.8, 38.6, 54.1), ("1-hop N/S", 4352, 251.0, 0.0, 6.5)):
    k = row.replace(" ", "").replace("/", "")
    for arm, v, m in (("off", off, "rate"), ("nbr", nbr, "idle_vc_cost"), ("line", line, "idle_vc_cost")):
        rec(f"glx-idle-mesh1e-{k}-{size}-{arm}", "BH Galaxy", "2D mesh", f"one ERISC, VC2 {arm}", row, None, size, GL, m, v, CP, VC2L,
            "Perf: idle and full VCs / 1. Idle-VC cost (one mesh, one ERISC)", "167481", "2026-11-01",
            GLN + ("Rate with VC2 off." if arm == "off" else "Delta against VC2 off, same job."), opt_level="-O3")
for row, size, off, nbr, line in (("1-hop E/W", 256, 141.8, 25.3, 35.5), ("1-hop E/W", 1024, 143.5, 23.7, 33.9), ("1-hop E/W", 4352, 252.0, 0.0, 0.0),
                                  ("2-hop", 256, 310.4, 21.7, 24.6), ("2-hop", 1024, 315.4, 21.8, 27.9), ("2-hop", 4352, 334.2, 47.8, 55.2)):
    k = row.replace(" ", "").replace("/", "")
    for arm, v, m in (("off", off, "rate"), ("nbr", nbr, "idle_vc_cost"), ("line", line, "idle_vc_cost")):
        rec(f"glx-idle-split-{k}-{size}-{arm}", "BH Galaxy", "VC1 split (two 2x2 meshes)", f"two ERISCs, VC1 compiled in and idle, VC2 {arm}, VC0 16,7,16 + VC1 4,4",
            row, None, size, GL, m, v, CP, VC2L, "Perf: idle and full VCs / 1. Idle-VC cost (2x2 split)", "167482", "2026-11-01",
            GLN + ("Rate with VC2 off (VC1 idle)." if arm == "off" else "Delta against VC2 off, same job; repeats within 0.5%."), slots="16,7,16 + 4,4", opt_level="-O3")
rec("glx-idle-split1e-1hopEW-line", "BH Galaxy", "VC1 split (two 2x2 meshes)", "one ERISC, VC1 idle, VC2 line (idle)", "1-hop E/W", None, None, GL,
    "idle_vc_cost", 64.0, CP, VC2L, "Perf: idle and full VCs / 1 (2x2 split, one ERISC)", "167482", "2026-11-01",
    GLN + "Base 266.7-267.1 at every size (router-bound even at 4352 B); neighbour +38.6, line +64.0.", opt_level="-O3")
# full VCs on link L (job 167482)
L = "Link L, mesh 1 chip 2 -> chip 3 (VC1 = mesh 0 chip 3's VC0 flow after the crossover)"
full = [(256, 177.6, 448.0, 467.2, (239.7, 478.2, 163.3), (327.0, 558.4, 543.3, 161.8), 20),
        (1024, 177.5, 459.1, 483.9, (238.6, 487.9, 163.8), (325.8, 566.1, 553.0, 162.3), 62),
        (4352, 252.1, 507.2, 556.5, (320.2, 690.1, 252.0), (397.4, 807.3, 810.7, None), 248)]
for size, a0, a2, a1, two, three, wire in full:
    for m, v, n in (("vc0_alone", a0, "VC0 alone"), ("vc2_alone", a2, "VC2 alone (4/4/4: window-bound, A/B below)"), ("vc1_alone", a1, "VC1 alone (4/4: window-bound, A/B below)"),
                    ("vc0_of_vc0vc2", two[0], "VC0 while VC0+VC2 both send"), ("vc2_of_vc0vc2", two[1], "VC2 while VC0+VC2 both send"),
                    ("total_vc0vc2", two[2], "link total VC0+VC2 (packets moved while both overlap; exact for two flows)"),
                    ("vc0_of_all3", three[0], "VC0 with all three VCs"), ("vc2_of_all3", three[1], "VC2 with all three"), ("vc1_of_all3", three[2], "VC1 with all three"),
                    ("total_all3", three[3], "link total, all three VCs (a lower bound for three flows)"), ("wire_est", wire, "wire estimate (payload + 96 B scaled; optimistic)")):
        if v is None:
            continue
        metric = "rate_combined" if m.startswith("total") else ("wire_estimate" if m == "wire_est" else ("rate" if m.endswith("alone") else "rate_contended"))
        rec(f"glx-full-L-{size}-{m}", "BH Galaxy", "VC1 split (two 2x2 meshes)", "two ERISCs, VC2 line mode, default row VC0 16,7,16 + VC1 4,4 + VC2 4,4,4, no sync",
            L, None, size, GL, metric, v, CP, VC2L, "Perf: idle and full VCs / 2. Full VCs and total throughput", "167482", "2026-11-01",
            GLN + n + (". At 4352 B all three: total <= 274.2 cycles/packet, a lower bound on throughput (the three-flow method undercounts packets; the earlier '>= 274' was WRONG, VC2_LINE_BRINGUP.md)." if m == "vc1_of_all3" and size == 4352 else ".") + " " + GXR,
            slots="16,7,16 + 4,4 + 4,4,4", opt_level="-O3")
for size, vc0, vc2, both0, both2, tot in ((256, 222.6, 494.8, 353.0, 581.1, 229.4), (1024, 222.8, 505.9, 349.4, 590.5, 229.4), (4352, 252.1, 575.3, 322.1, 737.2, 251.9)):
    for m, v in (("vc0_alone", vc0), ("vc2_alone", vc2), ("vc0_with_vc2", both0), ("vc2_with_vc0", both2), ("total", tot)):
        rec(f"glx-full-I-{size}-{m}", "BH Galaxy", "VC1 split (two 2x2 meshes)", "two ERISCs, VC2 line mode, default row, no sync",
            "Intermesh link I, mesh 0 chip 3 -> mesh 1 chip 2", None, size, GL, "rate_combined" if m == "total" else ("rate" if "alone" in m else "rate_contended"), v, CP, VC2L,
            "Perf: idle and full VCs / 2 (intermesh link I)", "167482", "2026-11-01", GLN + "Both VCs' packets delivered at the landing chip. Doc: no gain from the second VC on the intermesh router. " + GXR,
            slots="16,7,16 + 4,4 + 4,4,4", opt_level="-O3")
rec("glx-full-RAll-4352-payload", "BH Galaxy", "VC1 split (two 2x2 meshes)", "two ERISCs, VC2 line mode, one chip sends VC0 and VC2 on three links",
    "RAll: mesh 1 chip 2 east, north, south", None, 4352, GL, "payload_bandwidth", 50.7, "GB/s", VC2L, "Perf: idle and full VCs / 2 (RAll)", "167482", "2026-11-01",
    GLN + "Lower bound ('>= 50.7 GB/s', seven flows over the slowest flow's time); one chip's three links do not slow each other.")
# window A/B (job 167482)
wab = [("VC2 on L", [(256, 448.0, 448.0, 253.9), (4352, 507.2, 508.9, 294.3)]),
       ("VC2 on I", [(256, 494.8, 494.5, 289.5), (4352, 575.3, 574.2, 323.4)]),
       ("VC1 on L (VC0 -> VC1)", [(256, 467.2, 332.2, 467.7), (4352, 556.5, 338.2, 554.8)]),
       ("VC0 on L", [(256, 177.6, 186.6, 177.4), (4352, 252.1, 252.0, 252.1)]),
       ("VC0 on I", [(256, 222.6, 251.7, 223.1), (4352, 252.1, 252.5, 252.1)])]
for flow, cells in wab:
    for size, d, a1, a2 in cells:
        k = flow.split(" (")[0].replace(" ", "")
        for arm, v, sl in (("default", d, "16,7,16 + 4,4 + 4,4,4"), ("vc1x2", a1, "16,2,15,8,8,4,4,4"), ("vc2x2", a2, "16,3,16,4,4,8,8,8")):
            bs = "window" if flow.startswith("VC2") or flow.startswith("VC1") else None
            rec(f"glx-winab2-{k}-{size}-{arm}", "BH Galaxy", "VC1 split (two 2x2 meshes)", f"two ERISCs, VC2 line mode, window A/B arm {arm}",
                flow, None, size, GL, "rate", v, CP, VC2L, "Perf: idle and full VCs / 3. What binds (window A/Bs)", "167482", "2026-11-01",
                GLN + "Doc: VC2 flows bound by VC2's 4-slot windows (-42% to -44% doubled); VC0 flows into VC1 bound by VC1's 4/4 (-29% at 256 B, -39% at 4352 B); VC0 1-hop small-size cap and the intermesh router's ~223 not isolated.",
                bs, "A/B" if bs else None, slots=sl, opt_level="-O3")

# ---------------------------------------------------------------- Galaxy single mesh: VC2 neighbour rows in the slot study (lead v4rxv2i3)
ss = [("4352 B slots: 17,14,16 + 2,4 (study row)", 4352, 869.3, 135.2, 163.0, 916.0),
      ("4352 B slots: 17,10,16 + 8,10", 4352, 246.3, 135.3, 189.6, 358.2),
      ("2 KiB slots: 105,14,16 + 2,4 (default)", 2048, 866.8, 131.3, 159.3, 911.1),
      ("2 KiB slots: 34,28,32 + 4,8", 2048, 473.2, 131.0, 164.1, 536.2),
      ("2 KiB slots: 34,24,32 + 8,8", 2048, 245.6, 131.0, 172.6, 398.1),
      ("2 KiB slots: 34,20,32 + 16,16 (two passes)", 2048, 143.8, 130.2, 201.4, 322.5)]
for i, (sl, slotsize, a2, idle0, c0, c2) in enumerate(ss):
    for m, v, metric, n in (("vc2_alone", a2, "rate", "VC2 alone"), ("vc0_vc2idle", idle0, "rate", "VC0 alone with VC2 on and idle (VC2 off: 127.9 at 4352 B slots, 124.0 at 2 KiB)"),
                            ("vc0_contended", c0, "rate_contended", "VC0 while VC2 sends"), ("vc2_contended", c2, "rate_contended", "VC2 while VC0 sends")):
        rec(f"glx-sls2k-vc2-{i}-{m}", "BH Galaxy", "2D mesh", f"two ERISCs, VC2 neighbour on, {sl}", "VC0 and VC2 from [0,0] to [0,1]", None, 2048,
            "v4rxv2i3", metric, v, CP, SS, "2 KiB slots on a 200G Galaxy / VC2-on rows", "167648/167673", "2026-11-02",
            n + f". Payload 2048 B; slot size {slotsize} B. Doc: VC2's 2-slot window binds VC2 at every slot size; on a shared link the two VCs' rates add to the wire (16/16: 1/(1/201.4+1/322.5) = 124.0). " + XRUN,
            "window" if m == "vc2_alone" and a2 > 400 else None, "A/B" if m == "vc2_alone" and a2 > 400 else None,
            slots=sl.split(": ", 1)[1], opt_level="-O3")
for size, a2, both in ((3072, 179.6, "348 / 371"), (3584, 207.3, "400 / 431")):
    rec(f"glx-sls3k-vc2alone-{size}", "BH Galaxy", "2D mesh", "two ERISCs, VC2 neighbour on, VC2 16/16 slots", "VC2 alone", None, size, "v4rxv2i3",
        "rate", a2, CP, SS, "3-3.5 KiB slots / one table per path", "167723", "2026-11-02",
        f"At the wire ({size} B slots). VC0 + VC2 on one link (16/16): {both} cycles/packet (~half the wire each). With builder defaults (VC2 2/4) VC2 alone stays at 833-944 at every size. " + XRUN,
        "wire", "hypothesis", slots="VC2 16,16", opt_level="-O3")
rec("glx-sls3k-idlevc2-1k", "BH Galaxy", "2D mesh", "two ERISCs, VC2 neighbour on and idle, 3 KiB-class allocations", "VC0 1-hop with VC2 idle", None, 1024, "v4rxv2i3",
    "rate", 131.3, CP, SS, "3-3.5 KiB slots / small payloads", "167723", "2026-11-02",
    "Range 131.3-131.8 (lower end recorded); VC2 off 109.5-110: idle VC2 costs E0 ~21 at 1 KiB (doc's attribution to E0 is not an A/B on this row: hypothesis); hidden by the wire at 3072/3584 B.",
    "E0", "hypothesis", opt_level="-O3")

# ---------------------------------------------------------------- Code size of fan-in configs (VC2 line mode, -O3)
sizes = [("P150 2D, VC2 line", "2D mesh", "P150x8", 12364, 8488, 20852, 21116), ("Galaxy 2D, VC2 line", "2D mesh", "BH Galaxy", 12500, 9824, 22324, 22496),
         ("Galaxy torus, VC2 line", "torus", "BH Galaxy", 13360, 9824, 23184, 23408),
         ("Galaxy 2x2 split, VC2 line", "VC1 split (two 2x2 meshes)", "BH Galaxy", 14656, 13236, 27892, 28404),
         ("Galaxy 4x4 split, VC2 line", "VC1 split (two 4x4 meshes)", "BH Galaxy", 14776, 13548, 28324, None)]
for case, topo, sysn, e0, e1, tot, p3a in sizes:
    k = case.replace(" ", "").replace(",", "")
    rec(f"off-size-{k}-g3a", "offline", topo, f"two ERISCs, {case}, -O3, buffer 29,696 B", case, None, None, "vc2l_g3a", "code_size", tot, "B", VC2L,
        "Rebase onto v4rxg3 / Code size", None, "2026-11-02",
        f"E0 {e0} + E1 {e1} (largest per ERISC, size_report.py). " + (f"p3a (VC2 line on v4rxv2i3): {p3a}. " if p3a else "") +
        "Fits the 29 KiB kernel config buffer (largest leaves 1,372 B); main-line buffer is 26,624 B: the multi-mesh line configs need the 29 KiB buffer.",
        opt_level="-O3")
rec("off-size-p150split-1e-g3a", "offline", "VC1 split (two 2x2 meshes)", "one ERISC, P150 split, VC2 line, -O3", "P150 split, VC2 line, one ERISC", None, None,
    "vc2l_g3a", "code_size", 23020, "B", VC2L, "Rebase onto v4rxg3 / Code size", None, "2026-11-02", "p3a: 23,348.", opt_level="-O3")
for lvl, e0, e1, tot in (("-O3", 14656, 13764, 28420), ("-Os", 13600, 13836, 27436)):
    rec(f"glx-size-loaded-split-{lvl}", "BH Galaxy", "VC1 split (two 2x2 meshes)", f"two ERISCs, VC2 line, {lvl}", "mm_fanin loaded router", None, None,
        "vc2l_p3b", "code_size", tot, "B", VC2L, "Phase 3 / Galaxy runs (job 167327)", "167327", "2026-11-01",
        f"Loaded sizes E0 {e0} + E1 {e1}; both exceed the old 26,624 B buffer and run in the 29 KiB one ('so the default -Os would not have fit 26 KiB either'). mm_fanin 10/10 per rank at -O3, one ERISC and -Os.",
        opt_level=lvl)
for case, v, one in (("1D line (VC0 only)", 14956, 9552), ("ring (VC0 only)", 15016, 9656)):
    rec(f"off-size-1d-{case.split()[0]}", "offline", "1D line" if "line" in case else "ring", "two ERISCs, -O3, build p3a", case, None, None, "vc2l_p3a",
        "code_size", v, "B", VC2L_1D, "§1.5 Code size", None, "2026-11-01",
        f"Measured base for the N line-VC census (one ERISC {one} B). Per-VC increments for N >= 2 are estimates (est ~4.5 KB line, ~5.0 KB ring) because N >= 2 does not build yet.",
        opt_level="-O3")

# ---------------------------------------------------------------- 1D line VCs (host side only; no device data)
rec("off-linevcs-unit", "offline", "1D line VCs", "host only, TT_METAL_FABRIC_LINE_VCS=1..3", "fabric_unit_tests router-builder suites", None, None, "vc2l_p3b",
    "tests_passed", 73, "count", VC2L_HOST, "§9 As implemented / Checks", None, "2026-11-02",
    "73 of 73 (8 new). The router does not compile N >= 2 yet (its brought_up gate and its transaction-ID rule): no device or perf data for 1D line VCs exist.")
rec("off-linevcs-identity", "offline", "1D line VCs", "host before/after, N = 1 and every 2D case (38 coverage cases)", "host_identity.py --pairs", None, None, "vc2l_p3b",
    "images_identical", 871, "count", VC2L_HOST, "§9 Checks / Identity", None, "2026-11-02",
    "871 of 871 configs built on both sides with equal inputs: identical .text and loadable bytes; inputs identical in 36 of 38 cases (other two: garbled interleaved two-rank logs).")
for n, per, sl in ((2, 41, "13/12/16"), (3, 27, "10/9/8")):
    rec(f"off-linevcs-slots-N{n}", "offline", "1D line VCs", f"N = {n} line VCs, 4352 B payload, 29 KiB buffer (82 slots)", "mock capture slot row", None, 4352, "vc2l_p3b",
        "slots_per_vc", per, "count", VC2L_HOST, "§9 Checks (mock captures, line_vcs_dump_check.py)", None, "2026-11-02",
        f"Slot row per VC {sl} (worker/forwarding/receiver). The census (VC2_LINE_1D_VCS §1.1) estimates a lone VC at N = 3 window-bound short of the wire for multi-hop (est).",
        slots=sl)

json.dump(recs, open(OUT, "w"), indent=1)
print(len(recs), "records")
