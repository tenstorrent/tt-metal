#!/usr/bin/env python3
"""Build fabric_exp/report/data/bandwidth_galaxy.json and galaxy_v4_only.json (BH Galaxy part of the report bundle).

Values come from the study's own report tools run on the stored result files:
  glx_report.py rows_for()  (cycles/packet = recovered device clock / summary packets per second; prints .1f)
  sls_report.py / sls2k_report.py rows (cycles / packets of each sender row, 1.35 GHz)
  sls3k_table.py vc0()/vc1() (mean of every pass found)
and literal values transcribed from SLOT_STUDY.md / V4_NS_FLOOR.md / RESULTS.md tables where the doc is the source.
"""
import json, os, statistics, sys

REPO = "/home/aagarwal/tt-metal"
FE = REPO + "/fabric_exp"
JOBS = "/data/aagarwal/frs-glx/jobs"
OUT = FE + "/report/data"
sys.path.insert(0, FE + "/galaxy")
import glx_report as gr  # noqa: E402

sys.path.insert(0, FE + "/consolidation")
_argv = sys.argv
sys.argv = ["x", FE + "/results/glx167723"]
import sls3k_table as s3  # noqa: E402  (imports sls2k_report as r2, sls_report as r1)
sys.argv = _argv
r2 = s3.r2
r1 = s3.r1

SYSTEM = "BH Galaxy"
UNIT = "cycles/packet"


def rnd(v, d=1):
    return float(f"{v:.{d}f}")


def rec(**kw):
    base = dict(id=None, system=SYSTEM, topology=None, config=None, test=None, send_type=None, packet_size_B=None,
                router=None, metric="rate", value=None, unit=UNIT, bound_side=None, bound_status=None, source_file=None,
                source_section=None, job=None, date=None, same_job_with_main=False, pair_key=None, notes="")
    for k, v in kw.items():
        base[k] = v
    if "keepalive" not in kw and base["router"] != "wire (model)":
        base["keepalive"] = "firmware (8000 cycles)" if base["router"] == "main" else "1000 cycles (study setting)"
    missing = [k for k in ("id", "topology", "config", "router", "value", "source_file", "source_section", "date")
               if base[k] is None]
    assert not missing, (missing, kw)
    return base


SEND = {"noc_unicast_write": "write", "noc_unicast_scatter_write": "scatter_write",
        "noc_fused_unicast_atomic_inc": "fused_atomic_inc", "noc_unicast_atomic_inc": "atomic_inc"}

# ----------------------------------------------------------------------------------------------------------------------
# File 1: main against v4, Galaxy, same job (and flagged cross-run pairs)
# ----------------------------------------------------------------------------------------------------------------------
VARIANT = {
    "v4l": "v4l: early v4 lead (2026-10-27; generic ring loop of v4, before the per-packet credit work)",
    "v4u": "v4u: v4l + cheaper per-packet credits (worker credit preset, grouped counter reads, single-slot completion "
           "with a bounded wait); lead on 2026-10-27 morning (RESULTS §98 'Per-packet credit cost: v4u')",
    "v4p": "v4p: v4u + completions held up to 16 passes under first-level acks; dropped (bubble collapse on saturated "
           "rings, RESULTS §98 'Credit frames per packet: v4p, v4p2')",
    "v4p2": "v4p2: v4u + at most one held completion; dropped (same collapse as v4p)",
    "v4uf": "v4uf: v4u(a) + paired forward command-buffer checks, single slot checked once (lead late 2026-10-27; "
            "RESULTS §98 'Galaxy check of the lead v4uf')",
    "v4wt": "v4wt: v4uf with the single-slot bounded wait back (v4u's rule); not adopted",
    "v4eg": "v4eg: v4uf with single slots due as soon as one waits (main's rule), checked once; not adopted",
    "v4trm": "v4trm: lead 2026-10-29 (v4tel + channel trimming; production images equal v4rs); census lead",
}
PRE = ("pre-merge trees (main's router at the Oct 9 merge base; the tree later kept as trees/base_pre); the "
       "pre-merge Galaxy trees' host predates the 1D slot split, so v4 ring routers keep 16/16/32 (RESULTS §98 "
       "'Galaxy check of the lead v4uf')")
POST = "post-merge trees (trees/base rebuilt 2026-10-29 from the merge of main 6d8a103; v4 tree hard-linked from it)"

SAME = [
    dict(job="155833", date="2026-10-27", node="bh-glx-110-a02u02", suites={"ring": [""]}, routers=["main", "v4l"],
         tree=PRE, doc="RESULTS.md §98 'Galaxy rings and torus, v4l and main' (job 155833)"),
    dict(job="156067", date="2026-10-27", node="bh-glx-120-c02u02", suites={"mcs": [""]},
         routers=["main", "v4l", "v4u"], tree=PRE,
         doc="RESULTS.md §98 'Per-packet credit cost: v4u', Galaxy multicast size sweep table (job 156067)"),
    dict(job="157929", date="2026-10-27", node="bh-glx-120-c06u08", suites={"mcs": ["", "_r2"]},
         routers=["main", "v4u", "v4p", "v4p2"], tree=PRE,
         doc="RESULTS.md §98 'Credit frames per packet: v4p, v4p2' (Galaxy job 157929)"),
    dict(job="158477", date="2026-10-27", node="bh-glx-110-a04u14", suites={"mcs": ["", "_r2"]},
         routers=["main", "v4u", "v4uf"], tree=PRE,
         doc="RESULTS.md §98 'Galaxy check of the lead v4uf' (job 158477)"),
    dict(job="158562", date="2026-10-27", node="bh-glx-110-a04u14", suites={"mcs": ["", "_r2"]},
         routers=["main", "v4uf", "v4wt", "v4eg"], tree=PRE,
         doc="RESULTS.md §98 'Single-slot completion timing on Galaxy' (job 158562)"),
    dict(job="163369", date="2026-10-29", node="bh-glx-120-c01u14",
         suites={"mesh": ["", "_r2"], "linear": ["", "_r2"], "ring": ["", "_r2"], "mcs": ["", "_r2"]},
         routers=["main", "v4trm"], tree=POST,
         doc="RESULTS.md §98 'Perf census after the features' (Galaxy line); consolidation/V4_PERF_GAPS.md census "
             "section (Galaxy: Slurm job 163369)"),
]
SUITE = {
    "mesh": ("2D mesh", "fabric_exp/galaxy/glx_mesh.yaml (8x4 single-Galaxy Mesh, 1 link)"),
    "linear": ("1D line", "fabric_exp/galaxy/glx_linear.yaml (Linear, 1 link)"),
    "ring": ("ring", "glx_ring.yaml (/data/aagarwal/frs-glx/yamls; Ring, 1 link)"),
    "mcs": ("ring", "glx_ring_mcast_sweep.yaml (/data/aagarwal/frs-glx/yamls; Ring, full-ring multicast, 1 link)"),
}
UNICAST_WIRE_TESTS = {"GlxMesh1HopEW", "GlxMesh1HopNS", "GlxMesh1HopEWScatter", "GlxMesh1HopEWFused", "GlxMesh3HopEW",
                      "GlxMesh7HopNS", "GlxMeshFwd", "GlxLinear1HopEW", "GlxLinear3HopEW", "GlxRing1HopEW"}
MCAST = {"GlxFullRingMulticast", "GlxFullRingMcastSweep"}


def keepalive_note(r, job):
    if r == "main":
        return ("main's router: kernel v1, TT_FABRIC_EXP_FLAGS=0, firmware keepalive (v1 reads only the study's "
                "override, unset here)")
    s = ("v4: kernel v2 slot, TT_FABRIC_EXP_FLAGS=13129088, keepalive from the host setting (frs_glx_torus.sbatch "
         "header: 'v4* ... flags 13129088, host keepalive'; 1000 on Blackhole per RESULTS §98 job 155240)")
    if job == "163369":
        s += ("; CONFLICT: V4_PERF_GAPS.md census header says 'firmware keepalive on both' for this job; the logs do "
              "not record it. The +1.0 on every wire-bound 4352 B row here (v4trm 252.1 vs main 251.1) matches the "
              "keepalive-1000 stretch measured on main's router in job 155240 (251.1 -> 252.1), consistent with the "
              "sbatch (keepalive field set to 1000 on that basis)")
    return s


def file1():
    out = []
    for j in SAME:
        res = f"{JOBS}/{j['job']}/results"
        for suite, passes in j["suites"].items():
            data = {}  # router -> [ {key: cyc} per pass ]
            for r in j["routers"]:
                per = []
                for p in passes:
                    rows, clock = gr.rows_for(res, f"{r}_{suite}{p}")
                    per.append({k: v[1] for k, v in rows.items()} if rows else None)
                data[r] = per
            main1 = data["main"][0]
            assert main1, (j["job"], suite)
            topo, yaml = SUITE[suite]
            for key in main1:
                test, size, ntype = key
                v4_present = [r for r in j["routers"][1:] if data[r][0] and key in data[r][0]]
                if not v4_present:
                    continue
                pk = f"glx-{j['job']}-{suite}-{test}-{size}"
                for r in ["main"] + v4_present:
                    vals = [d[key] for d in data[r] if d and key in d]
                    p1 = data[r][0][key]
                    pass_labels = [f"{r}_{suite}{p}" for p, d in zip(passes, data[r]) if d and key in d]
                    bs, bst, bnote = None, None, ""
                    if size == 4352 and test in UNICAST_WIRE_TESTS:
                        bs, bst = "wire", "hypothesis"
                        bnote = ("Bound: 'wire' from the docs' reading (V4_PERF_GAPS census: '4352 B unicast rows are "
                                 "wire-bound on both (~250, equal)'; RESULTS §98 job 155240: every 4352 B row "
                                 "wire-bound at 200G); no one-side A/B on this job. ")
                    if test in MCAST and size == 4352 and r != "main":
                        bnote = ("Full-ring multicast at 4352 B: the one Galaxy production row below main. RESULTS §98 "
                                 "(job 156067): 4352 B is the one size whose 200G wire time (~248) exceeds v4's router "
                                 "time; v4 keeps the links ~82% busy, main ~94%. Established by single-change A/B: a "
                                 "bounded completion wait on idle passes is what the saturated ring needs (v4wt vs "
                                 "v4uf, ~350 -> ~310); crediting the newest slot early does not help (v4eg). Remaining "
                                 "causes are hypotheses (how the sender shares a saturated link between injection and "
                                 "forwarded traffic). ")
                    if test in MCAST and r == "main":
                        bnote = ("RESULTS §98 reads main's full-ring multicast as router-bound at ~261 at every size "
                                 "(no A/B; hypothesis). ")
                    if r in ("v4p", "v4p2"):
                        bnote += ("Collapse (~17,000-20,500 cycles/packet): bubble flow control, a held completion is "
                                  "the saturated ring's free-slot credit (RESULTS §98 mechanism; v4p/v4p2 are the "
                                  "single change against v4u). ")
                    tail = ""
                    if ntype == "noc_unicast_atomic_inc":
                        tail += ("Atomic-increment row: 4352 is the yaml's size parameter; the packet carries no "
                                 "payload, so its GB/s column is not payload bandwidth. ")
                    if j["job"] == "155833" and test in MCAST and size == 4352:
                        tail += "RESULTS §98: v4l 19% slower here (43.9 M vs 36.9 M cycles, slowest sender). "
                    if j["job"] == "156067" and r == "main" and size in (2048, 3072):
                        tail += ("CONFLICT: RESULTS §98's job-156067 table gives main %s B as %s; the job's summary CSV "
                                "gives %.1f (glx_report.py). This value is the CSV's. " %
                                (size, {2048: "265.7", 3072: "261.2"}[size], p1))
                    if j["job"] == "163369" and test == "GlxFullRingMcastSweep" and size == 1024 and r == "main":
                        tail += ("Pass 2 (main_mcs_r2) 261.7 gives 1.30x for v4trm, below V4_PERF_GAPS's quoted "
                                "1.32-1.37x for 1-3 KB. ")
                    notes = (f"Same Slurm job {j['job']} on {j['node']}, routers back to back; {j['tree']}. "
                             f"{VARIANT.get(r, '')}{'. ' if r != 'main' else ''}"
                             f"{keepalive_note(r, j['job'])}. Yaml {yaml}; telemetry off (TT_FABRIC_NO_BW_TELEMETRY=1). "
                             f"Slots: builder defaults of the tree, not logged in this job (main: tier tables). "
                             f"value = pass 1 ({pass_labels[0]}); passes = {pass_labels}. "
                             f"Cycles/packet = recovered device clock (1350 MHz) / summary packets per second. "
                             f"{bnote}{tail}")
                    out.append(rec(
                        id=f"glx-{j['job']}-{suite}-{test}-{SEND.get(ntype, ntype)}-{size}-{r}",
                        topology=topo, config=f"two ERISCs, 1 link, max payload 4352 B, {suite} suite",
                        test=test, send_type=SEND.get(ntype, ntype), packet_size_B=size, router=r,
                        value=rnd(p1), bound_side=bs, bound_status=bst,
                        source_file=f"{res}/{r}_{suite}.csv",
                        source_section=(f"glx_report.py {res} --configs {','.join(j['routers'])} --suites {suite}"
                                        f"{',' + suite + '_r2' if '_r2' in passes else ''}: {test} {size} B; doc: {j['doc']}"),
                        job=f"Slurm {j['job']} ({j['node']})", date=j["date"], same_job_with_main=True,
                        pair_key=pk, notes=notes.strip(), passes=[rnd(v) for v in vals]))
    return out


def cross_run_lr():
    """Line-rate sizes with no same-job main-vs-v4 pair: main from job 155281 (pre-merge) vs the lead's LR sweep."""
    out = []
    main_rows, _ = gr.rows_for(FE + "/results/glx155281", "main_sweep_mesh")
    a_rows, _ = gr.rows_for(FE + "/results/glx167648", "lr_a_base")
    b_rows, _ = gr.rows_for(FE + "/results/glx167673", "lr_b_base")
    for d, mtest, ltest in (("E/W", "GlxSweep1HopEW", "LrEW_s4352"), ("N/S", "GlxSweep1HopNS", "LrNS_s4352")):
        for size in (1536, 2304, 2560, 3072, 3584, 4096):
            mk, lk = (mtest, size, "noc_unicast_write"), (ltest, size, "noc_unicast_write")
            if mk not in main_rows or lk not in a_rows:
                continue
            pk = f"glx-xrun-155281-167648-1hop{d.replace('/', '')}-{size}"
            common = ("CROSS-RUN pair (not the same job): no same-job main-vs-v4 Galaxy number exists at this payload "
                      "(job 163369's glx_mesh.yaml has 16/256/1024/2048/4352 B only). Same traffic (1-hop single flow "
                      f"[0,0] -> {'[0,1]' if d == 'E/W' else '[0,4]'}, Mesh, 1 link, 200,000 packets, 4352 B max "
                      "payload) but different jobs, nodes, dates and trees: main on the PRE-merge base (job 155281, "
                      "2026-10-26, bh-glx-110-a01u08, firmware keepalive); the lead v4rxv2i3 on the post-merge "
                      "sls_bdd tree (job 167648, 2026-11-02, bh-glx-120-c05u02). Main's router on the post-merge "
                      "base agrees with the pre-merge rows within ~1.4 where both exist (1-hop E/W 1024 B 224.1 vs "
                      "223.0, 2048 B 223.4 vs 222.7, 4352 B 251.1 both; jobs 163369 / 155281). ")
            mv = main_rows[mk][1]
            wire = 12.80 + size / 18.50 + (3.0 if size + 96 > 4096 else 0.0)
            out.append(rec(
                id=f"glx-xrun-main155281-1hop{d.replace('/', '')}-{size}-main", topology="2D mesh",
                config="two ERISCs, 1 link, max payload 4352 B (main: tier-table slots)", test=mtest,
                send_type="write", packet_size_B=size, router="main", value=rnd(mv),
                bound_side="wire" if mv <= wire * 1.01 else None,
                bound_status="hypothesis" if mv <= wire * 1.01 else None,
                source_file="fabric_exp/results/glx155281/main_sweep_mesh.csv",
                source_section=(f"glx_report.py fabric_exp/results/glx155281 --configs main,v3q,v3qfw,v3qr --suites "
                                f"sweep_mesh: {mtest} {size} B; RESULTS.md §98 'Galaxy size sweep: where each router "
                                "meets the 200G wire'"),
                job="Slurm 155281 (bh-glx-110-a01u08)", date="2026-10-26", same_job_with_main=False, pair_key=pk,
                notes=common + ("'wire' = within 1% of 12.80 + P/18.50 (model match), no A/B; RESULTS §98: main "
                                "meets the 2D wire line only at ~3.9 KiB. " if mv <= wire * 1.01 else
                                "RESULTS §98: main's 2D floor ~223 (router-bound; side not measured), meets the wire "
                                "only at ~3.9 KiB. ")))
            av, bv = a_rows[lk][1], b_rows[lk][1]
            out.append(rec(
                id=f"glx-xrun-lead167648-1hop{d.replace('/', '')}-{size}-v4rxv2i3", topology="2D mesh",
                config="two ERISCs, 1 link, 4352 B slots, default 2D split 17,16,16", test=ltest, send_type="write",
                packet_size_B=size, router="v4rxv2i3", value=rnd(av),
                bound_side="wire" if av <= wire * 1.01 else None,
                bound_status="hypothesis" if av <= wire * 1.01 else None,
                source_file="fabric_exp/results/glx167648/lr_a_base.csv",
                source_section=(f"glx_report.py fabric_exp/results/glx167648 --configs lr_a --suites base: {ltest} "
                                f"{size} B; SLOT_STUDY.md '2 KiB slots on a 200G Galaxy', line-rate table"),
                job="Slurm 167648 (bh-glx-120-c05u02); pass 2 Slurm 167673 (bh-glx-110-a08u20)", date="2026-11-02",
                same_job_with_main=False, pair_key=pk,
                notes=common + ("Router: tree sls_bdd = the lead v4rxv2i3 + v4bdd delay hooks, byte-identical at "
                                "delay 0 (464/464 Galaxy images). Keepalive host 1000. value = pass a (lr_a_base, job "
                                "167648); passes = [lr_a_base, lr_b_base (job 167673)]. "
                                + ("'wire' = within 1% of 12.80 + P/18.50 (model match; SLOT_STUDY), no A/B. "
                                   if av <= wire * 1.01 else "SLOT_STUDY: below ~2.3 KiB with 4352 B slots the 17-slot "
                                                        "worker window stops the row (hypothesis at this size). ")),
                passes=[rnd(av), rnd(bv)], slots="17,16,16"))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# File 2: v4-only Galaxy rows
# ----------------------------------------------------------------------------------------------------------------------
LEAD = "v4rxv2i3"
BDD = ("tree sls_bdd = the lead v4rxv2i3 (temporary lead 2026-11-01) + variants/v4bdd delay hooks, byte-identical at "
       "delay 0 (464/464 Galaxy coverage images, bnd_loadcmp.py); -O3, flags 13129088, keepalive 1000, BW telemetry "
       "off, slots logged and checked")
LEADTREE = ("tree sls_lead = live post-merge study tree + the lead v4rxv2i3's 18 files; -O3, flags 13129088, "
            "keepalive 1000, BW telemetry off, slots logged and checked")
NODES = {"167464": "bh-glx-110-a07u08", "167648": "bh-glx-120-c05u02", "167673": "bh-glx-110-a08u20",
         "167723": "bh-glx-110-a08u20"}


def wire_records():
    out = []
    sec = "SLOT_STUDY.md '2 KiB slots on a 200G Galaxy', line-rate table, column 'wire'"
    for p, v in ((1024, 68.2), (1536, 95.8), (2048, 123.5), (2304, 137.3), (2560, 151.2), (3072, 178.9),
                 (3584, 206.5), (4096, 237.2), (4352, 251.0)):
        out.append(rec(
            id=f"glx-wire-model-2d-{p}", topology="2D mesh", config="200G wire model, 2D header (237 B per packet)",
            test=None, send_type=None, packet_size_B=p, router="wire (model)", metric="wire_time", value=v,
            bound_side=None, bound_status=None, source_file="fabric_exp/consolidation/SLOT_STUDY.md",
            source_section=sec, job="fit on Slurm 155281 (v3qfw wire-bound rows 2560/3072/3584 B)", date="2026-10-26",
            same_job_with_main=False, pair_key=None,
            notes=("Model line, not a router: cycles/packet = 12.80 + P/18.50, +3.0 when P + 96 > 4096 (a second "
                   "Ethernet frame); fitted on v3qfw's wire-bound rows of job 155281 (residuals <= 0.01; RESULTS §98 "
                   "'Galaxy size sweep'); no keepalive stretch. Measured lead rows on the line: 4352 B 252.1, 3072 B "
                   "179.6 (+0.4%: keepalive 1000 stretches wire-bound rows by 0.40%, RESULTS §98). router='wire (model)' (schema exception for model rows).")))
    sec2 = "RESULTS.md §98 'Galaxy size sweep: where each router meets the 200G wire', share-of-the-wire table"
    for topo, vals, fit in (("2D mesh", ((1024, 68.4), (1536, 96.2), (1792, 110.1), (2048, 124.0), (2304, 137.9),
                                         (2560, 151.8)), "12.80 + P/18.50"),
                            ("1D line", ((1024, 65.9), (1536, 93.7), (1792, 107.5), (2048, 121.4), (2304, 135.3),
                                         (2560, 149.2)), "10.31 + P/18.51")):
        for p, v in vals:
            out.append(rec(
                id=f"glx-wire-model-ka1000-{'2d' if topo == '2D mesh' else '1d'}-{p}", topology=topo,
                config="200G wire model with the keepalive-1000 stretch (+0.40%)", test=None, send_type=None,
                packet_size_B=p, router="wire (model)", metric="wire_time", value=v, bound_side=None, bound_status=None,
                source_file="fabric_exp/RESULTS.md", source_section=sec2,
                job="fit on Slurm 155281", date="2026-10-26", same_job_with_main=False, pair_key=None,
                notes=(f"Model line ({fit} cycles/packet, fitted on v3qfw, job 155281) as RESULTS §98 tabulates it "
                       "'with keepalive 1000', i.e. 0.4% above the bare line SLOT_STUDY uses (2048 B: 124.0 here vs "
                       "123.5). 1D carries a 46 B smaller route header. router='wire (model)' (schema exception for model rows).")))
    return out


def lr_records():
    out = []
    a_rows, _ = gr.rows_for(FE + "/results/glx167648", "lr_a_base")
    b_rows, _ = gr.rows_for(FE + "/results/glx167673", "lr_b_base")
    slots = {"s4352": ("4352 B slots (81 per ERISC), default 2D split 17,16,16", "17,16,16"),
             "s2560": ("2560 B slots (137 per ERISC), default 57,16,32", "57,16,32"),
             "s2048": ("2048 B slots (169 per ERISC), default 41,32,32", "41,32,32")}
    for key, (cyc_gb) in sorted(a_rows.items()):
        test, size, ntype = key
        dirn, cfg = test[2:4], test.split("_")[1]
        conf, sl = slots[cfg]
        av = cyc_gb[1]
        bv = b_rows[key][1] if key in b_rows else None
        wire = 12.80 + size / 18.50 + (3.0 if size + 96 > 4096 else 0.0)
        bs = bst = None
        bn = ""
        if av <= wire * 1.01:
            bs, bst = "wire", "hypothesis"
            bn = ("'wire': within 1% of the 12.80 + P/18.50 line (model match, SLOT_STUDY), no one-side A/B on this "
                  "row. ")
            if cfg == "s4352" and size == 4352:
                bn += ("Job 167648's arms (same 4352 B slots, Sls1HopEW/NS 4352 B, sls2k_report.py arms) show tx/rx "
                       "spins of 16/32 cost 0.0: neither ERISC binds (one-side A/B data; the docs state the wire bound "
                       "only from the model). ")
        elif cfg == "s4352" and size in (1024, 2048):
            bs, bst = "E0", "A/B"
            bn = ("V4_NS_FLOOR.md §1 (job 167648 one-side spins, 4352 B slots 17,16,16, 1-hop E/W and N/S at 1 KiB, "
                  "'2 KiB alike'): E0 (tx32 +43.8, rx32 +21.1 on E/W 1 KiB). SLOT_STUDY calls it the 17-slot worker "
                  "window (41 worker slots: 124.0 vs 128.2 at 2048 B). ")
        elif cfg == "s2048" and size == 1024:
            bs, bst = "E0", "A/B"
            bn = ("V4_NS_FLOOR.md §1 (2 KiB slots 41,32,32, 1-hop E/W 1 KiB): E0 by one-side spins (tx32 +54.1, rx32 "
                  "+29.4); SLOT_STUDY line-rate table: '(E0)'. ")
        else:
            bs, bst = "E0", "hypothesis"
            bn = ("SLOT_STUDY: below ~1.9 KiB the router (E0, ~110) binds 1-hop rows (4352 B slots: the worker "
                  "window); no one-side A/B at this payload/configuration. ")
        out.append(rec(
            id=f"glx-lr-{test}-{size}", topology="2D mesh", config=f"two ERISCs, 1 link, {conf}", test=test,
            send_type="write", packet_size_B=size, router=LEAD, value=rnd(av), bound_side=bs, bound_status=bst,
            source_file="fabric_exp/results/glx167648/lr_a_base.csv",
            source_section=("SLOT_STUDY.md '2 KiB slots on a 200G Galaxy', 'Line rate (G1 ...)' table; "
                            "sls2k_report.py fabric_exp/results/glx167648 lr; glx_report.py ... --configs lr_a "
                            f"--suites base: {test} {size} B"),
            job="Slurm 167648 (bh-glx-120-c05u02); pass 2 Slurm 167673 (bh-glx-110-a08u20)", date="2026-11-02",
            same_job_with_main=False, pair_key=None,
            notes=(f"Line-rate sweep, 1-hop single flow {dirn} ([0,0] -> {'[0,1]' if dirn == 'EW' else '[0,4]'}), "
                   f"200,000 packets, fabric max_packet_size {cfg[1:]} B. {BDD}. Payload GB/s {cyc_gb[0]:.2f}. "
                   f"value = pass a (lr_a_base, job 167648); passes = [a, b (lr_b_base, job 167673)]. {bn}"
                   "No main run of this yaml (main's 155281 sweep: bandwidth_galaxy.json cross-run pairs)."),
            passes=[rnd(av)] + ([rnd(bv)] if bv is not None else []), slots=sl))
    return out


NSROWS = [("1-hop E/W", "Sls1HopEW", "E0"), ("3-hop E/W", "Sls3HopEW", "E0"), ("3-hop N/S", "Sls3HopNS", "E1"),
          ("7-hop N/S", "Sls7HopNS", "E1"), ("10-hop corner", "SlsCorner10", "E1"),
          ("turn (1 N/S + 3 E/W)", "SlsTurn4", "E1"), ("column merge (7 flows)", "SlsColMerge", "E1")]
# V4_NS_FLOOR.md §1 table: base / c_tx32 / c_rx32 (1 KiB)
NS1 = {
    "4352": {"Sls1HopEW": (122.9, 43.8, 21.1), "Sls3HopEW": (143.4, 61.0, 20.4), "Sls3HopNS": (182.3, 14.8, 43.4),
             "Sls7HopNS": (186.0, 11.0, 40.8), "SlsCorner10": (186.7, 17.7, 40.3), "SlsTurn4": (178.1, 25.9, 42.8),
             "SlsColMerge": (166.2, 21.4, 42.2)},
    "2048": {"Sls1HopEW": (109.6, 54.1, 29.4), "Sls3HopEW": (140.7, 61.3, 14.2), "Sls3HopNS": (172.5, 21.8, 44.1),
             "Sls7HopNS": (173.1, 21.1, 44.0), "SlsCorner10": (173.2, 27.9, 43.8), "SlsTurn4": (170.5, 30.7, 44.3),
             "SlsColMerge": (158.4, 25.9, 44.4)},
}


def ns_records():
    out = []
    # §1: bound side by one-side spins
    for slot, kind, alloc, conf in (("4352", "2dk", "17,16,16", "4352 B slots, 2D split 17,16,16 (study row)"),
                                    ("2048", "2d2k", "41,32,32", "2 KiB slots (2048 B payload max), builder default "
                                                                 "41,32,32")):
        pa = r2.rows(f"{FE}/results/glx167648/{kind}_a_base_tests.csv")
        pb = r2.rows(f"{FE}/results/glx167648/{kind}_b_base_tests.csv")
        for label, test, side in NSROWS:
            base, tx, rx = NS1[slot][test]
            win = (" + window remnant at 16 forwarding slots (V4_NS_FLOOR §1: ~13 above the E1 floor)"
                   if slot == "4352" and test in ("Sls7HopNS", "SlsCorner10") else "")
            out.append(rec(
                id=f"glx-nsfloor-s{slot}-{test}-1024", topology="2D mesh", config=f"two ERISCs, 1 link, {conf}",
                test=test, send_type="write", packet_size_B=1024, router=LEAD, value=base, bound_side=side,
                bound_status="A/B", source_file="fabric_exp/consolidation/V4_NS_FLOOR.md",
                source_section=("§1 'Bound side (Galaxy job 167648, one-side spins of 16 / 32 cycles per packet, "
                                f"v4bdd)', column '{'4352 B slots 17,16,16' if slot == '4352' else '2 KiB slots 41,32,32'}'"
                                f"; data: sls2k_report.py fabric_exp/results/glx167648 arms ({kind})"),
                job="Slurm 167648 (bh-glx-120-c05u02)", date="2026-11-02", same_job_with_main=False, pair_key=None,
                notes=(f"N/S floor study, row '{label}'. {BDD}. Base = mean of 2 runs ({kind}_a_base, {kind}_b_base); "
                       f"a 32-cycle spin costs +{tx} on the sender (E0, c_tx32) and +{rx} on the receiver (E1, "
                       f"c_rx32); a side with no slack pays ~N + 12. Bound {side}{win}: settled by the one-side A/B "
                       "(V4_NS_FLOOR §1). The floor is per packet (1 KiB = 2 KiB within 0.4 on E1-bound rows)."),
                passes=[rnd(pa[(test, 1024)][0], 2), rnd(pb[(test, 1024)][0], 2)], slots=alloc))
    # §2: diagnostic single-change variants on the Galaxy (job 167673), 2 KiB slots, 1 KiB
    diag = {"base": (LEAD, "lead (pass c of job 167673: 2d2k_c_base)"),
            "base_sl": ("v4bddsl", "diagnostic v4bddsl = v4bdd with forwards_are_stateful() false: every forward "
                                   "takes the N/S routers' stateless path (not for adoption)"),
            "base_ml": ("v4bddml", "diagnostic v4bddml = v4bdd with every 2D router's downstreams live (E/W routers "
                                   "get N/S routers' multi-downstream bookkeeping and stateless forwards; not for "
                                   "adoption)")}
    vals = {}
    for tag in ("base", "base_sl", "base_ml"):
        acc = {}
        for p in ("a", "b", "c", "d"):
            f = f"{FE}/results/glx167673/2d2k_{p}_{tag}_tests.csv"
            if os.path.exists(f):
                for k, v in r2.rows(f).items():
                    acc.setdefault(k, []).append((p, statistics.mean(v)))
        vals[tag] = acc
    doc3 = {"base": 140.2, "base_sl": 147.5, "base_ml": 164.6}
    for tag, (router, desc) in diag.items():
        for label, test, side in NSROWS[1:]:
            k = (test, 1024)
            pv = vals[tag].get(k)
            if not pv:
                continue
            if test == "Sls3HopEW":
                v = doc3[tag]
                bs, bst = ("E0", "A/B") if tag == "base" else ("E1", "A/B")
                bn = {"base": "E0 (V4_NS_FLOOR §1, 2 KiB slots 41,32,32: tx32 +61.3, rx32 +14.2)",
                      "base_sl": "E1 by one-side spin (rx32 +44.3; V4_NS_FLOOR §2 table)",
                      "base_ml": "E1 by one-side spin (rx32 +44.0; V4_NS_FLOOR §2 table)"}[tag]
                srcsec = "§2 'Why: two single-change A/Bs', table row '3-hop E/W, 1 KiB, 2 KiB slots | Galaxy (job 167673)'"
                vnote = ("Value as in V4_NS_FLOOR §2 (" + ("pass c, 2d2k_c_base" if tag == "base" else
                                                          "mean of passes a, b") + "). ")
            else:
                if tag == "base":
                    continue  # the lead's N/S rows are in the §1 records
                v = rnd(statistics.mean(x for _, x in pv))
                bs, bst = "E1", "hypothesis"
                bn = ("E1: the doc says only that N/S rows do not move in either variant (control: their routers "
                      "already work that way); the variant's own rx32 / tx32 runs in this job (2d2k_a_rx32_*, "
                      "2d2k_a_tx32_*) show rx +44 / tx +22 on these rows (one-side A/B data, not stated in the docs)")
                srcsec = ("§2 table row 'N/S rows, 1 KiB, 2 KiB slots | Galaxy | 172.5-173.3 (unchanged)'; per-row "
                          "values: sls_report.py glx fabric_exp/results/glx167673 2d2k base_sl base_ml --sizes=1024")
                vnote = "Value = mean of passes a, b (sls_report.py). "
            out.append(rec(
                id=f"glx-nsfloor-diag-{router}-{test}-1024", topology="2D mesh",
                config="two ERISCs, 1 link, 2 KiB slots (2048 B payload max), default 41,32,32", test=test,
                send_type="write", packet_size_B=1024, router=router, value=v, bound_side=bs, bound_status=bst,
                source_file="fabric_exp/consolidation/V4_NS_FLOOR.md", source_section=srcsec,
                job="Slurm 167673 (bh-glx-110-a08u20)", date="2026-11-02", same_job_with_main=False, pair_key=None,
                notes=(f"N/S floor mechanism, row '{label}'. {desc}. Trees sls_bdd / sls_bddsl / sls_bddml (hard-linked "
                       f"from sls_lead, each differing in its variant's files only). Bound: {bn}. {vnote}"
                       "Settled (V4_NS_FLOOR §2): the stateless forward adds ~37 E1 cycles, multi-downstream "
                       "bookkeeping ~17-19 more; together the Galaxy floor to within ~8 (the rest: hypothesis, the "
                       "N/S decode)."),
                passes=[rnd(x, 2) for _, x in pv], slots="41,32,32"))
    # 4352 B slots, the diagnostics' 3-hop E/W (not tabulated in the doc for the Galaxy)
    for tag, router in (("base_sl", "v4bddsl"), ("base_ml", "v4bddml")):
        f = f"{FE}/results/glx167673/2dk_a_{tag}_tests.csv"
        d = r2.rows(f)
        v = statistics.mean(d[("Sls3HopEW", 1024)])
        out.append(rec(
            id=f"glx-nsfloor-diag-s4352-{router}-Sls3HopEW-1024", topology="2D mesh",
            config="two ERISCs, 1 link, 4352 B slots, 2D split 17,16,16", test="Sls3HopEW", send_type="write",
            packet_size_B=1024, router=router, value=rnd(v), bound_side=None, bound_status=None,
            source_file="fabric_exp/results/glx167673/2dk_a_" + tag + "_tests.csv",
            source_section=("sls_report.py glx fabric_exp/results/glx167673 2dk base base_sl base_ml: Sls3HopEW 1024 "
                            "(not tabulated for the Galaxy in V4_NS_FLOOR §2, which shows P150 at 4352 B slots)"),
            job="Slurm 167673 (bh-glx-110-a08u20)", date="2026-11-02", same_job_with_main=False, pair_key=None,
            notes=(f"Diagnostic {router} (not for adoption) at 4352 B slots, one pass (2dk_a_{tag}); lead's same row "
                   "143.4 (E0, V4_NS_FLOOR §1). No one-side spins were run on this configuration: bound not "
                   "measured.")))
    # P3 (job 167464): N/S-involving rows across 2D splits, 1 KiB, 4352 B slots
    splits = [("s17", "17,16,16"), ("s20", "20,15,16"), ("s26", "26,13,16"), ("s11", "11,18,16"), ("s8", "8,19,16"),
              ("s25r8", "25,16,8"), ("s19r32", "19,10,32")]
    p3rows = [("1-hop E/W", "Sls1HopEW"), ("3-hop E/W", "Sls3HopEW"), ("turn (1 N/S + 3 E/W)", "SlsTurn4"),
              ("3-hop N/S", "Sls3HopNS"), ("7-hop N/S", "Sls7HopNS"), ("10-hop corner", "SlsCorner10"),
              ("column merge (7 flows)", "SlsColMerge")]
    for tag, alloc in splits:
        per = {}
        for p in ("a", "b"):
            for k, v in r1.rows_of(f"{FE}/results/glx167464/2d_{p}_{tag}_tests.csv", r1.VC0_ROWS).items():
                per.setdefault(k, []).append(statistics.mean(v))
        for label, test in p3rows:
            k = (test, "1024")
            if k not in per:
                continue
            v = statistics.mean(per[k])
            out.append(rec(
                id=f"glx-p3-{tag}-{test}-1024", topology="2D mesh",
                config=f"two ERISCs, 1 link, 4352 B slots, 2D split {alloc} (worker, forwarding, receiver; 81 slots)",
                test=test, send_type="write", packet_size_B=1024, router=LEAD, value=rnd(v), bound_side=None,
                bound_status=None, source_file="fabric_exp/consolidation/SLOT_STUDY.md",
                source_section=("'P3: one mesh, VC0 only: Galaxy 8x4 against the P150 2x4', 1 KiB table (Galaxy "
                                "column); values: sls_report.py glx fabric_exp/results/glx167464 2d "
                                f"{tag} --sizes=1024,4352"),
                job="Slurm 167464 (bh-glx-110-a07u08)", date="2026-11-01", same_job_with_main=False, pair_key=None,
                notes=(f"N/S floor across splits, row '{label}'. {LEADTREE}. Mean of passes a, b (doc table rounds "
                       "to integers). SLOT_STUDY: the N/S-involving rows have a floor no split moves (3-hop N/S "
                       "182-184 at 15-16 forwarding slots, turn 178-182, 7-hop/corner ~186 at 16); 7-hop N/S and "
                       "corner are forwarding-window-bound below 16 slots (the doc calls the split sweep an 'A/B "
                       "across splits'; not a one-side A/B, so no bound is recorded here). At 17,16,16 the one-side "
                       "spins of job 167648 give E0 for E/W and 1-hop rows and E1 for the N/S rows (see the "
                       "glx-nsfloor-s4352 records)."),
                passes=[rnd(x, 2) for x in per[k]], slots=alloc))
    return out


DOC_CELLS = {  # SLOT_STUDY.md '3-3.5 KiB slots', 'One table per path' (cycles/packet at the slot's largest payload)
    "2D 1-hop E/W": (124.0, 179.6, 207.4, 252.1), "2D 3-hop E/W": (139.7, 179.6, 207.4, 252.1),
    "2D 3-hop N/S": (172.6, 179.6, 207.4, 252.1), "2D 7-hop N/S": (173.3, 180.0, 207.4, 252.1),
    "2D 10-hop corner": (173.3, 180.0, 207.4, 252.1), "2D turn": (170.6, 179.6, 207.4, 252.1),
    "2D row merge (link)": (129.3, 179.6, 207.4, 252.1), "2D column merge (link)": (163.1, 180.8, 207.3, 252.1),
    "2D multicast": (553.5, 705.8, 817.5, 993.0),
    "VC1 split: VC0 1-hop": (138.6, 179.6, 207.4, 252.1), "VC1 split: VC0 2-hop": (168.1, 179.7, 207.6, 252.9),
    "VC1 split: multicast": (131.6, 180.9, 209.5, 255.1), "inter direct": (206.7, 208.3, 208.6, 252.1),
    "VC0 hop + inter direct": (236.2, 238.7, 242.2, 278.6), "inter + 1 VC1 hop E/W": (279.0, 286.5, 292.4, 298.6),
    "inter + 2 VC1 hops": (276.7, 285.0, 302.8, 295.9),
}
DOC_BOUND = {  # 'bound below the wire' column of the same table
    "2D 1-hop E/W": "E0 ~110", "2D 3-hop E/W": "E0 ~140", "2D 3-hop N/S": "E1 ~172 + window at 16 forwarding",
    "2D 7-hop N/S": "E1 ~173; window ~3,000 / forwarding", "2D 10-hop corner": "E1 ~173; window ~3,000 / forwarding",
    "2D turn": "E1 ~170", "2D row merge (link)": "E0", "2D column merge (link)": "E1 ~158-163",
    "2D multicast": "the links' sharing", "VC1 split: VC0 1-hop": "VC1 routers' E0 ~139",
    "VC1 split: VC0 2-hop": "the 2-hop path ~168", "VC1 split: multicast": "-",
    "inter direct": "the crossover ~207", "VC0 hop + inter direct": "~236-242; window at 4352 B",
    "inter + 1 VC1 hop E/W": "the VC1 path ~276-299", "inter + 1 VC1 hop N/S": "the VC1 path (not in the doc table)",
    "inter + 2 VC1 hops": "the VC1 path",
}
# rows with a one-side A/B at that slot size and allocation (V4_NS_FLOOR §1, 2 KiB slots 41,32,32, '2 KiB alike')
AB_2K = {"2D 3-hop E/W": "E0", "2D 3-hop N/S": "E1", "2D 7-hop N/S": "E1", "2D 10-hop corner": "E1", "2D turn": "E1",
         "2D column merge (link)": "E1"}
DEFAULTS = {(2048, "41,32,32"), (3072, "35,16,32"), (3584, "19,16,32"), (4352, "17,16,16"),
            (2048, "107,9,16,5,4"), (3072, "53,9,16,5,4"), (3584, "37,9,16,5,4"), (4352, "19,9,16,5,4")}


def slot_records():
    out = []
    jobs_of = {2048: ("Slurm 167648 (bh-glx-120-c05u02) + 167673 (bh-glx-110-a08u20)", "2026-11-02"),
               3072: ("Slurm 167723 (bh-glx-110-a08u20)", "2026-11-02"),
               3584: ("Slurm 167723 (bh-glx-110-a08u20)", "2026-11-02")}
    for p, a2d, av in s3.SIZES:
        cells = [("2d", name, test, dirs, kind, tag, lab) for name, test in s3.ROWS2D for dirs, kind, tag, lab in a2d]
        cells += [("vc1", name, row, dirs, kind, tag, lab) for name, row in s3.ROWSV for dirs, kind, tag, lab in av]
        for fam, name, test, dirs, kind, tag, lab in cells:
            if fam == "2d":
                d = s3.vc0(dirs, kind, tag)
                big = d.get((test, p), (None,))[0]
                smalls = [(1024, d.get((test, 1024), (None,))[0]), (256, d.get((test, 256), (None,))[0])]
                topo = "2D mesh"
                tname = test
            else:
                d = s3.vc1(dirs, kind, tag)
                big = d.get((test, p))
                s256 = d.get((test, 256))
                smalls = [(256, s256)] if s256 else [(16, d.get((test, 16)))]
                topo = "VC1 split (two 2x2 meshes)" if not name.startswith(("inter", "VC0 hop")) else "VC1 inter-mesh"
                tname = {lab_: pre_ for lab_, pre_, _ in r1.GLX_VC1_ROWS}[test]
            jobdirs = sorted({os.path.basename(x) for x in dirs})
            if p == 4352:
                j = "167648" if kind == "2dk" else "167464"
                job, date = f"Slurm {j} ({NODES[j]})", ("2026-11-02" if j == "167648" else "2026-11-01")
                tree = BDD if j == "167648" else LEADTREE
            else:
                job, date = jobs_of[p]
                tree = BDD
            is_def = (p, lab) in DEFAULTS
            for size, v in [(p, big)] + smalls:
                if v is None:
                    continue
                v = rnd(v)
                wire = 12.80 + size / 18.50 + (3.0 if size + 96 > 4096 else 0.0)
                bs = bst = None
                bn = ""
                if v < wire * 1.01:
                    bs, bst = "wire", "hypothesis"
                    bn = "'wire' = within 1% of 12.80 + P/18.50 (SLOT_STUDY's model match), no one-side A/B. "
                    if p == 4352 and kind == "2dk" and size == 4352 and name != "2D multicast":
                        bn += ("Job 167648's arms (sls2k_report.py arms, 2dk) show tx/rx spins of 16/32 cost 0.0 on "
                               "this row: neither ERISC (one-side A/B data; the docs state the wire bound from the "
                               "model only). ")
                elif fam == "2d" and p == 2048 and lab == "41,32,32" and name in AB_2K:
                    bs, bst = AB_2K[name], "A/B"
                    bn = ("V4_NS_FLOOR §1: one-side spins on 2 KiB slots 41,32,32 (job 167648; rows at 1 KiB, "
                          "'2 KiB alike'). ")
                elif fam == "2d" and p == 4352 and lab == "17,16,16" and size == 1024 and name in AB_2K:
                    bs, bst = AB_2K[name], "A/B"
                    bn = "V4_NS_FLOOR §1: one-side spins on 4352 B slots 17,16,16 (job 167648), 1 KiB. "
                elif fam == "2d" and p == 4352 and lab == "17,16,16" and size == 1024 and name == "2D 1-hop E/W":
                    bs, bst = "E0", "A/B"
                    bn = "V4_NS_FLOOR §1: one-side spins on 4352 B slots 17,16,16 (job 167648), 1 KiB. "
                elif fam == "2d" and p == 2048 and lab == "41,32,32" and size == 1024 and name == "2D 1-hop E/W":
                    bs, bst = "E0", "A/B"
                    bn = "V4_NS_FLOOR §1: one-side spins on 2 KiB slots 41,32,32 (job 167648), 1 KiB. "
                if fam == "2d" and p == 2048 and name == "2D 1-hop E/W" and size == 2048:
                    bn += ("CONFLICT: V4_NS_FLOOR §1 lists 1-hop E/W as E0 (A/B) at 1 KiB, '2 KiB alike', while "
                           "SLOT_STUDY reads this cell as the wire; the 2048 B arms of job 167648 (sls2k_report.py arms, "
                           "2d2k 41,32,32) show tx16 +16.3 / tx32 +39.1, rx16 +0.0 / rx32 +15.0: E0 has ~5 cycles of "
                           "slack, so E0 and the wire are both near-binding. ")
                docv = DOC_CELLS.get(name)
                docflag = ""
                if size == p and docv and abs(docv[(2048, 3072, 3584, 4352).index(p)] - v) < 0.06:
                    docflag = "= the SLOT_STUDY per-path table cell for this path and slot size (best measured). "
                gb = size / v * 1.35
                out.append(rec(
                    id=f"glx-slot{p}-{fam}-{name.replace(' ', '_').replace('/', '').replace(':', '')}-{lab}-{size}",
                    topology=topo,
                    config=(f"two ERISCs, 1 link, {p} B slots (max payload {p} B), allocation {lab}"
                            + (" (builder default at this slot size)" if is_def else "")),
                    test=tname, send_type="write", packet_size_B=size, router=LEAD, value=v, bound_side=bs,
                    bound_status=bst, source_file="fabric_exp/consolidation/SLOT_STUDY.md",
                    source_section=("'3-3.5 KiB slots on the 200G Galaxy', 'One table per path', path "
                                    f"'{name}'; values: sls3k_table.py fabric_exp/results/glx167723 (results "
                                    f"{', '.join(jobdirs)}, labels {kind}_*_{tag})"),
                    job=job, date=date, same_job_with_main=False, pair_key=None,
                    notes=(f"Slot-size study, path '{name}'{' (VC1 split row ' + test + ')' if fam == 'vc1' else ''}, "
                           f"payload {size} B ({'the slot' + chr(39) + 's largest' if size == p else 'small payload'}"
                           f"), payload {gb:.1f} GB/s at 1.35 GHz. {tree}. Mean of every pass found "
                           f"(sls3k_table.py). {docflag}{bn}Doc 'bound below the wire': {DOC_BOUND.get(name, '-')}."
                           + (" Allocation order: worker, VC0 forwarding, VC0 receiver, VC1 forwarding, VC1 receiver."
                              if fam == "vc1" else " Allocation order: worker, forwarding, receiver.")),
                    slots=lab))
    # VC2 (doc values)
    vc2 = [
        ("VC2 alone", 2048, "34,20,32 + 16,16", 143.8, "VC2 sender alone, 16/16 VC2 slots; router-bound (~141-144)"),
        ("VC2 alone", 3072, "20,14,16 + 16,16", 179.6, "VC2 alone at the wire (16/16)"),
        ("VC2 alone", 3584, "18,11,16 + 16,16", 207.3, "VC2 alone at the wire (16/16)"),
        ("VC0 + VC2 one link: VC0", 2048, "34,20,32 + 16,16", 201.0, "shared link; the two VCs' rates add up to the "
                                                                     "wire (1/(1/201.4 + 1/322.5) = 124.0)"),
        ("VC0 + VC2 one link: VC2", 2048, "34,20,32 + 16,16", 323.0, "shared link"),
        ("VC0 + VC2 one link: VC0", 3072, "20,14,16 + 16,16", 348.0, "~half the wire each"),
        ("VC0 + VC2 one link: VC2", 3072, "20,14,16 + 16,16", 371.0, "~half the wire each"),
        ("VC0 + VC2 one link: VC0", 3584, "18,11,16 + 16,16", 400.0, "~half the wire each"),
        ("VC0 + VC2 one link: VC2", 3584, "18,11,16 + 16,16", 431.0, "~half the wire each"),
    ]
    for name, p, alloc, v, txt in vc2:
        out.append(rec(
            id=f"glx-slot{p}-vc2-{name.replace(' ', '_').replace(':', '').replace('+', 'p')}", topology="2D mesh, VC2 on",
            config=f"two ERISCs, 1 link, {p} B slots, VC0 w,f,r + VC2 w,r = {alloc}", test=None, send_type="write",
            packet_size_B=p, router=LEAD, value=v, bound_side="wire" if "wire" in txt and "alone" in name else None,
            bound_status="hypothesis" if "wire" in txt and "alone" in name else None,
            source_file="fabric_exp/consolidation/SLOT_STUDY.md",
            source_section=("'3-3.5 KiB slots on the 200G Galaxy', 'One table per path', rows 'VC2 alone (16 / 16 "
                            "slots)' / 'VC0 + VC2, one link (16 / 16)'"),
            job=("Slurm 167648 + 167673" if p == 2048 else "Slurm 167723 (bh-glx-110-a08u20)"), date="2026-11-02",
            same_job_with_main=False, pair_key=None,
            notes=(f"{name}: {txt}. {BDD}. Per-sender rates from the profiler's log lines (sls2k_report.py vc2). "
                   "Doc values; the VC0 + VC2 cells are integers in the doc (2048 B: 201.4 / 322.5 in the 2 KiB "
                   "section's VC2 table, 34,20,32 + 16,16, two passes). Allocation as run: vc22k v16 "
                   "34,20,32,16,16; vc23k deep 20,14,16,16,16; vc235k deep 18,11,16,16,16."),
            slots=alloc))
    # 2 KiB-section VC2 table (2048 B payload)
    t = [("4352 B: 17,14,16 + 2,4 (study row)", 4352, 869.3, 135.2, 163.0, 916.0),
         ("4352 B: 17,10,16 + 8,10", 4352, 246.3, 135.3, 189.6, 358.2),
         ("2 KiB: 105,14,16 + 2,4 (default)", 2048, 866.8, 131.3, 159.3, 911.1),
         ("2 KiB: 34,28,32 + 4,8", 2048, 473.2, 131.0, 164.1, 536.2),
         ("2 KiB: 34,24,32 + 8,8", 2048, 245.6, 131.0, 172.6, 398.1),
         ("2 KiB: 34,20,32 + 16,16 (two passes)", 2048, 143.8, 130.2, 201.4, 322.5)]
    for lab, slot, a, idle, c0, c2 in t:
        alloc = lab.split(": ")[1].split(" (")[0]
        for what, v in (("VC2 alone", a), ("VC0 alone, VC2 on (idle)", idle), ("contention: VC0", c0),
                        ("contention: VC2", c2)):
            out.append(rec(
                id=f"glx-vc2tbl-{slot}-{alloc.replace(' ', '').replace('+', 'p')}-{what.split(':')[-1].strip().replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')}",
                topology="2D mesh, VC2 on", config=f"two ERISCs, 1 link, {slot} B slots, VC0 w,f,r + VC2 w,r = {alloc}",
                test=None, send_type="write", packet_size_B=2048, router=LEAD, value=v, bound_side=None,
                bound_status=None, source_file="fabric_exp/consolidation/SLOT_STUDY.md",
                source_section=("'2 KiB slots on a 200G Galaxy', 'VC2-on rows (Galaxy single mesh, 2048 B payload ...)' "
                                f"table, row '{lab}', column '{what}'"),
                job="Slurm 167648 (bh-glx-120-c05u02) / 167673 (bh-glx-110-a08u20)", date="2026-11-02",
                same_job_with_main=False, pair_key=None,
                notes=(f"VC0 and VC2 from [0,0] to [0,1], 2048 B payload; {what}. {BDD}. VC0 alone with VC2 off: "
                       "127.9 at 4352 B slots, 124.0 at 2 KiB. Doc: VC2's 2-slot worker window binds VC2 at every "
                       "slot size (window, from the slot sweep; no one-side A/B); an idle VC2 costs VC0's 1-hop row "
                       "~7 at 2048 B."),
                slots=alloc))
    return out


def main():
    f1 = file1() + cross_run_lr()
    f2 = wire_records() + lr_records() + ns_records() + slot_records()
    for name, recs in (("bandwidth_galaxy.json", f1), ("galaxy_v4_only.json", f2)):
        ids = [r["id"] for r in recs]
        dup = {i for i in ids if ids.count(i) > 1}
        assert not dup, dup
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(recs, f, indent=1)
            f.write("\n")
        print(name, len(recs))


if __name__ == "__main__":
    main()
