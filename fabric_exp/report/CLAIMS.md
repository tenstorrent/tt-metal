# Claims the data supports: v4 against main's router (2026-11-02)

Every number below is in `data/*.json` with its source, job and date. Rates are cycles per packet at 1350 MHz; lower is
better. Speedup is main's cycles divided by v4's.

How each claim is marked:
- **same job**: main and v4 ran in the same job;
- **cross-run**: they ran in different jobs, sometimes on different hosts or trees;
- **A/B**: a bound or cause settled by an isolating A/B (a one-side delay spin, a one-side lean variant, or a variant
  that changes only that piece);
- **hypothesis**: everything else about causes and bounds (repo CLAUDE.md, memory feedback-attribution-needs-ab).

The lead is v4rxg3. It has v4rxf4's per-packet code on every production config, so v4rxf4 numbers stand for it.
**No main column exists for v4rxf4.** The sender pass is changing E0 now (jobs 451-456), so every v4 number predates the
final lead.

## A. Headline claims

### A1. P150, two ERISCs: v4 runs 1.3-2.5x main's packet rate on every suite except bursts

The same-job column is v4trm (a09u32 jobs 164-171, 2026-10-29). The v4rxf4 column is cross-run (jobs 420-421,
2026-11-02). Source: bandwidth.json `rate_suite_mean`; V4_PERF_GAPS census; V4_RECEIVER_OWNERSHIP §17.

| suite (rows) | main | main, keepalive 1000 | v4trm (same job) | speedup | v4rxf4 (cross-run) | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D 4352 B (10) | 217.2 | 217.2 | 144.4 | 1.50 | 143.3 | 1.52 |
| 1D (16) | 200.5 | 196.8 | 152.2 | 1.32 | 150.6 | 1.33 |
| stream mixes (30) | 222.2 | 219.9 | 128.7 | 1.73 | 127.2 | 1.75 |
| ring (14) | 239.9 | 238.6 | 188.6 | 1.27 | 179.9 | 1.33 |
| Y-torus (12) | 233.7 | 233.8 | 157.4 | 1.48 | 153.5 | 1.52 |
| trimmed 2D (20) | 215.9 | 211.9 | 154.4 | 1.40 | 150.2 | 1.44 |
| VC1 split (13) | 332.6 | - | 171.5 | 1.94 | 159.0 | 2.09 |
| VC1 inter-mesh (8) | 841.0 | - | 342.5 | 2.46 | 336.9 | 2.50 |
| bursts (44) | 555.0 | 555.0 | 523.3 | 1.06 | 522.9 | 1.06 |

**Per row (v4trm, same job):**
- 2D: 1.25-1.74x.
- Mixes: 1.22-2.87x. The largest is pure atomics: 374.2 against 130.4.
- Ring: 1.18-1.41x.
- VC1 split: 1.71-2.04x.
- Inter-mesh: 1.57-2.74x.
- 1D: 1.00-1.51x. NeighborExchange write 124.9 against 124.8: both at the wire.
- Bursts: the rows with long gaps equal main; the gaps set them, not the router.

**Caveats to state with the claim.** "As shipped" differs in three settings besides the router:
1. **Keepalive.** Main's firmware keepalive (8000) floors one-way rows above 4114 B at ~187.5 (RESULTS §15). Main's 1D
   1-hop write runs 187.6 as shipped and 168.4 at keepalive 1000 (same job). v4 runs 124.5. So part of the 1D gain is
   the host keepalive setting; the "main, keepalive 1000" column is the router-only comparison.
2. **VC1 slots.** Main uses its tier table 51,4,8,2,4, v4 uses 19,9,16,5,4. At equal slots (25,8,16,4,4, job 112, same
   job, v4gi):
   - 1-hop: main 303.3 / 306.5 (16 / 4352 B) against v4gi 158.1 / 157.9, still ~1.9x;
   - 2-hop: main 305.8 / 303.4 against v4gi 290.6 / 310.6, about even.
3. **VC1 opt level.** Main at -Os (it does not fit at -O3), v4 at -O3. -O3 is worth 10.7% to v4 on the VC1 split (job
   367).

### A2. One ERISC and mux mode: v4 ahead or level

| suite | main | v4trm (same job) | v4rxf4 (cross-run) |
| --- | ---: | ---: | ---: |
| 2D | 331.3 | 276.6 | 272.6 |
| 1D | 318.3 | 305.7 | 299.6 |
| ring | 375.7 | 375.1 | 356.9 |
| mux | 373.3 | 373.6 | 371.1 |

- Mux rows are bound by the mux: neither ERISC binds them (A/B, V4_BOUNDS §1b).
- Rows that still trail are in section C.

### A3. v4 is faster than the 400G wire where the wire binds; main is not

- **1D 1-hop write, 4352 B:** v4 124.5 (v4trm and v4rxf4). The 1D wire is 124.1 by the §29 frame model (rate / wire =
  1.003). On v4rxv2i3 neither ERISC binds: E0 has 24.3 cycles of slack, E1 29.9 (A/B, one-side spins, job 349;
  V4_BOUNDS §1c / §2).
- **2D 1-hop write, 4352 B:** v4 127.0-127.6 against a 125.4 wire, E0-bound (A/B). Main 222.5, 1.77x the wire.
- **Wire model.** `(P + H + 56 x frames + 84) B / link bytes per cycle`, with frames of at most 4114 B and H = 96 B
  (2D) or 48 B (1D). It reproduces every quoted wire figure within 0.3 cycles (wire.json):
  - 400G, 4096 B: 118.5;
  - 400G, 4352 B: 125.4;
  - 200G, 4352 B: 250.8 (quoted 251.0, measured 252.1);
  - 200G, 3072 B: 178.6 (quoted 178.9, measured 179.6).

### A4. Galaxy, 200G: v4 1.07-1.85x main below 4352 B; both at the wire at 4352 B except full-ring multicast

Slurm 163369 (2026-10-29), same job, v4trm, two passes:
- **Below 4352 B** (32 row-size pairs): 1.07-1.85x.
  - Mesh 1-hop 1.73-1.85x;
  - mesh 3-hop ~1.47x;
  - mesh 7-hop N/S 1.18-1.20x;
  - line 1.29-1.62x;
  - ring 1-hop 1.27-1.29x;
  - full-ring multicast 1-3 KB 1.32-1.37x on pass means (pass 2 at 1024 B alone: 1.30x), 3584 B 1.07x.
- **At 4352 B unicast:** both at the wire (main 251.1, v4trm 252.1; line and ring 248.5-248.6 against 249.5).
- **Full-ring multicast 4352 B:** 0.76x (section C).
- **Latest v4 on the Galaxy** (v4rxv2i3, slot study, no main in the job):
  - it reaches the 200G wire from ~2.3 KiB with 4352 B slots, and at 2048 B with 2 KiB slots;
  - every VC0 path is on the wire at 3072 B slots (SLOT_STUDY G1 and 3-3.5 KiB sections; galaxy_v4_only.json).

### A5. DeepSeek decode CCLs: v4's op time is 13.3% lower to 0.6% higher than main's; mostly a cross-job comparison

Ratios are v4rxf4 / main, mean of passes a and b (ccl.json; CCL_MATRIX.md).

| CCL | v4rxf4 / main | job relation |
| --- | --- | --- |
| all_gather | 0.954-0.963 | cross-job |
| all_reduce | 0.973-1.006 | cross-job |
| broadcast | 0.985-0.994 | cross-job |
| SDPA, 4096-8192 B | 0.962-0.977 | |
| SDPA, 15232 B | 0.867 | |

- **Same-job SDPA only:**
  - job 440 (pass a): 4096 B 0.974, 4352 B 0.984, 8192 B 0.961, 15232 B 0.867, 2048 B 1.000;
  - job 444 (SDPA 2048 B, passes c / d / e): 0.949, 0.860, 0.872.
- **SDPA 2048 B is noisy on every router.** Five-pass medians: main 5.69, v4rxf4 5.40.
- The docs say decode CCLs are latency-bound and SDPA is compute-bound (TRISC_2). No isolating A/B is in the bundle,
  so treat both as **hypotheses**.

### A6. Code size and stack: v4 is smaller than main on every two-ERISC config and has 2-5x the free E0 stack

Coverage builds of 2026-11-02, main against v4rxg3 (size.json; V4_SIZE.md):

| config | main | v4rxg3 -O3 |
| --- | --- | ---: |
| VC1 E0 + E1 | 29,940 B at -O3 (over the 26,624 B buffer by 3,316); 25,252 B at its production -Os | 22,996 B |
| 2D E0 + E1 | 19,884 B (-O3) | 16,576 B |
| 2D + VC2 E0 + E1 | 23,836 B | 19,460 B |
| Galaxy express E0 + E1 | 21,068 B | 19,072 B |
| least free E0 stack | 424 B (VC1, -O3); 320 B (VC1, -Os); 868-1,860 B elsewhere | 1,604 B (VC1); 1,732-2,132 B elsewhere |

- v4's VC1 has 3,628 B left in the buffer before runtime args.
- **Exception: one-ERISC binaries.** v4 is larger on four of them:
  - glx_express_1erisc and its line variant: 13,848 B against 12,604 B (+1,244);
  - bh_2d_mux: +332;
  - bh_torus_1erisc: +184;
  - bh_neighbor_1erisc: +120.
- Before Wormhole bring-up, check the Wormhole IRAM limit for one-ERISC `.text` (V4_SIZE).

### A7. Feature reach: v4 builds and runs configurations main cannot

- **Main cannot build VC1 + VC2.** Its two-ERISC router fails -Werror=stack-usage at 2,176 B (V4_RECEIVER_SHAPE §25,
  job 320's capture).
- **Main cannot build VC2 on one ERISC.** It hits a static_assert (RESULTS §98, job 141; size.json: main_O3 and main_Os
  fail all 28 bh_2d_vc2_1erisc configs).
- **v4 runs VC0, VC1 and VC2 traffic together**, data-checked on the Galaxy 2x2 split: 24/24 runs per rank, one and two
  ERISCs, -O3 and -Os (job 167441). This was the VC2 line worktree's vc2l_p3b, v4rxv2i3 plus line mode (fanin.json).
- **Caveat:** the lead itself refuses VC1 + VC2 at compile time today (fabric_erisc_router_v2.cpp:113). v4rxc7v and
  vc2l_p3b admitted it as variants.
- **Mux mode:** v4trm equals main (373.6 against 373.3, same job).

### A8. Correctness parity on the validation the study runs

- **Validation:** v4rxg3 passes 13/13 (job 447), including the channel-trimming gtests on two and one ERISC.
- **Inline write followed by a flushed atomic** (fabric_unit_tests), 1D and 2D: main 2/2, 2/2; v4rxf4 2/2, 2/2 (job 423).
- **History:** v4rxv2i3 hung on this sequence while main passed in the same job (job 351). The fix (ii') is in the
  lead.

### A9. Latency: v4 is lower on all 12 cases, but only across hosts

v4rxv2i3 (job 373, a09u32, 2026-11-01) against main as shipped (`dqs_stock_kafw`, a04u24, pre-merge, 2026-10-19).
Medians of 1,000 samples, single runs:
- 1-hop: -28 to -85 ns (-1.6 to -5.9%);
- 3-hop: -170 to -207 ns (-3.8 to -5.9%).

**cross-run, cross-host.** The host-to-host offset reached 72 ns (RESULTS §66):
- the 3-hop gap is 2-3x that offset;
- the 1-hop gap is within it, so do not state it as a result.

## B. Claims to check before stating them

### B1. "Main requires power-of-2 slot counts": DOES NOT HOLD as stated

Main's router (git 6d8a103, `edm_fabric_flow_control_helpers.hpp:28-125`) has a mask path for power-of-2 depths and a
compare-and-wrap path otherwise in:
- `wrap_increment` and `wrap_increment_n`;
- `normalize_ptr` and `distance_behind`;
- `ChannelBufferPointer`;
- the transaction-ID tracker.

Both paths compile. Main's own Blackhole defaults are not powers of 2:
- 2D worker 17 (tier 16 plus the spare slot, allocator.cpp:707-728);
- VC1 worker 51;
- VC0+VC2 tier row {7, 11, 0, 0, 2, 4}, logged as 43,7,7,7 rx 11.

Main also ran non-power-of-2 depths cleanly: 26/25/32, 19/32, 32/19 and 35/16 on 1D (jobs 076-077), and 25,8,16,4,4 on
VC1 (job 112).

What main does have:
- **A hard limit:** at most 127 slots per channel (`MAX_CHANNEL_BUFFER_SLOTS = UINT8_MAX / 2`, fabric_common.h:32). The
  2N pointer is a uint8_t, enforced by static_assert.
- **A measured preference:** a non-power-of-2 sender channel costs main's flows through it 2-6 cycles/packet (RESULTS
  §98 "1D slot split", jobs 076-077). For example, ring 3-hop goes 209.6 → 215.8 at forwarding 25.
- **A tabulation threshold of 32 slots** (`MAX_TABULATED_SLOTS`). Deeper channels compute slot addresses instead of
  tabulating them.

v4's 19,9,16,5,4 is therefore legal for main's router, and main has run comparable depths. v4 keeps its *receivers* at
powers of 2 by choice: a non-power-of-2 receiver ring costs v4's E1 ~8 cycles/packet (v4us sweep, jobs 070-075).

What to say: "main's router runs any depth up to 127; powers of 2 are 2-6 cycles/packet cheaper for it on the flows
through that channel".

### B2. "Every config at -O3 with headroom": DOES NOT HOLD for VC1 + VC2

- **The lead refuses VC1 + VC2 at compile time**: static_assert "this configuration is not brought up yet",
  fabric_erisc_router_v2.cpp:113. That is 131 configs in the cov_m set: bh_vc1_vc2 44, its 1erisc 22, trim_w0 43,
  trim_w0_1erisc 22.
- **Production builds the study router's VC1 + VC2 at -Os** (erisc_datamover_builder.cpp `get_kernel_opt_level`). Only
  VC1 alone moved to -O3.
- **The 26,156 B at -O3 with "468 B left" is a probe**: v4rxv2i3 plus v4rxc7v's gate line (V4_SIZE log, 2026-11-01), not
  the lead.
  - The 468 B is before runtime args, which share the 26,624 B buffer. The real margin is smaller and unmeasured.
  - v4rxc7v measured 26,076 B (548 B left).
- **What does hold:** every config the lead builds runs at -O3 in production.
  - The largest two-ERISC E0+E1 is VC1's 22,996 B, leaving 3,628 B before runtime args.
  - The largest one-ERISC binary is 16,860 B (bh_vc1_trim_w0, one ERISC; v4-only case).
- **The max-feature config** (inter-mesh + express + bubble flow control + VC1 + VC2) has never been built for either
  router, and no capture combines them (DATA_INVENTORY gap 4).

### B3. "3-hop latency -124 to -203 ns against main" (RESULTS §98, V4_PERF_GAPS): WRONG BASELINE

- The baseline `a8v1` is kernel v1 at the study flags 13129088 with keepalive 1000 (dq66b_runs.sh:21), on a08u24. It is
  not main as shipped, and it is cross-host.
- Use A9 instead, with its cross-host caveat.

### B4. "Main's router: flushed fused ~495 against ~154 unflushed" (FLUSH_DEFER_DESIGN.md): NOT MAIN AS SHIPPED

- Those are RESULTS §62's columns: kernel v1 with the study flags and §62's atomic staging.
- §58 says the flush was "Not measured on main".
- No flushed-atomic number exists for main as shipped or for the lead. State only that production kernels flush and that
  the cost is large on the study builds (522.8 against 152.0 on E/W fused 4352 B, §58).

### B5. "v4 is faster than main on every CCL / 0.5-4.6%": NEEDS QUALIFIERS

- It is a cross-job comparison except SDPA.
- all_reduce at 4096 B is slower on v4rxf4: 1.259 against 1.251 us at 1 link (+0.6%), 1.250 against 1.247 at 2 links
  (+0.2%).
- SDPA 15232 B is -13.3%, outside the quoted range.
- CCL_MATRIX's job list omits job 426.

### B6. "v4 is at least as fast as main everywhere": DOES NOT HOLD

See section C.

### B7. "The 1D / VC1 gains are the router's": PARTLY

- The 1D one-way rows include the keepalive (A1, caveat 1).
- VC1 includes the slot row and the opt level (A1, caveats 2-3).
- The router-only figures:
  - main at keepalive 1000: 1D 196.8 against 152.2;
  - job 112's equal-slot VC1 1-hop: ~1.9x.

### B8. "Native line VCs add router capacity": NOT SUPPORTED

Fan-in does not multiply small-packet capacity (fanin.json, VC2_LINE_BRINGUP; V4_SEND_BATCH):
- Galaxy link L at 256 B: one VC 177.6, two or three VCs 162-164 (+9%), ~139-140 with doubled windows.
- P150: two busy VCs total ~115, the same as one VC (jobs 389 / 390).

The headline fan-in row (N workers on N line VCs against N workers through main's mux) has never been measured. The
router does not yet compile two or more line VCs.

### B9. Other notes to correct before reuse

- **The 400G wire at 4 KiB.**
  - "117" counts one Ethernet frame. A 4096 B payload plus a 96 B header needs two frames: 118.5.
  - Measured 118.1 for 4176 B on the bare link.
- **"`note<>` always_inline recovers about half the -Os gap".** It recovers 53% on the VC1 split but ~27% inter-mesh
  (job 367 CSVs, stability.json).
- **The memory note feedback-no-compiler-fragile-tricks** says VC1 builds at -Os in production. That is stale since job
  399 for v4.
- **V4_SENDER_OWNERSHIP §7.3**'s "no packed send progress or counters" is superseded by §8.4 and not marked WRONG there.
- **"main" in slot-study labels** is the lead at main's slot table, not main's router.

## C. Rows where v4 trails main in the latest data

**P150, latest lead code** (v4rxf4 against main, **cross-run**; gaps.json):

| row | main | v4rxf4 | delta |
| --- | ---: | ---: | --- |
| one ERISC, NeighborExchange write 4352 B | 234.1 | 247.2 | +5.6% |
| one ERISC, half-ring multicast write 4352 B (1D suite) | 442.3 | 447.1 | +1.1% |
| one ERISC, half-ring multicast scatter 4352 B (1D suite) | 488.0 | 493.3 | +1.1% |
| one ERISC, half-ring multicast write 4352 B (ring suite) | 445.1 | 447.1 | +0.4% |
| one ERISC, NeighborExchange scatter 4352 B | 288.2 | 289.2 | +0.3% |
| ties: BurstEW / BurstNS N2_G500 16 B, MuxLinear1HopEW 4352 B | | | +0.0-0.1 |

**P150, same job** (v4trm, jobs 164-171). Thirteen one-ERISC and mux rows trailed by +0.7% to +7.3%:
- NeighborExchange write +7.3%;
- half-ring multicast +3.6% to +4.3%;
- ring 1-hop bidirectional +4.0% / +4.1%;
- mux full-ring multicast +3.9% / +4.0%;
- full-ring multicast +0.7% / +2.5%;
- mux ring multicast +1.5% / +1.8%.

v4rxf4's cross-run numbers no longer trail on most of these. A same-job census is needed to say they closed.

**Galaxy, same job** (v4trm, Slurm 163369):
- **Full-ring multicast 4352 B:** main 263.7, v4trm 347.7 (passes 349.2 / 346.2), **0.76x**. The sweep row is 263.8
  against 348.2. It is the only production-topology row far behind.
  - Earlier variants: v4l 0.84x, v4u 0.87x, v4wt 0.85x, v4uf 0.75x, v4eg 0.74x. No lead after v4trm has run it.
- **Wire-bound 4352 B unicast rows:** +1.0 cycle (252.1 against 251.1; line / ring +0.9-1.0), 0.4%. **Hypothesis:** the
  keepalive-1000 stretch. The sbatch gives v4 the host keepalive; V4_PERF_GAPS says firmware on both (conflict).

**Galaxy, cross-run** (v4rxv2i3, job 167648, against main, job 155281): 1-hop 4096 B 238.2 against 237.2 (+0.4%), both at
the wire.

**Older variant at equal slots** (job 112, same job, VC1 25,8,16,4,4): v4gi 2-hop 4352 B 310.6 against main 303.4
(+2.4%). The lead's own row, 19,9,16,5,4, was not run against main at those slots.

**CCL** (cross-job): all_reduce 4096 B, +0.6% (1 link) and +0.2% (2 links).

**Size:** the four one-ERISC binaries in A6, up to +1,244 B.

## D. Bounds and causes: what is settled

**Settled by A/B:**
- **Bound side per row on P150.** One-side delay spins on v4trm (jobs 172-176) and v4rxv2i3 (jobs 349, 357), in
  slack.json:
  - 2D 1-hop / 3-hop / multicast writes: E0;
  - scatter, ring and torus N/S: E1;
  - 1D 1-hop writes: neither (the wire);
  - mux rows: neither (the mux).
  These are for those variants, not the lead: the codec stage added +2 to +4 on E/W and forwarded small-packet rows
  since.
- **VC1 inter-mesh rows 1_1..1_3 are bound by the VC1 windows**, not by either ERISC. Deeper windows (13,8,16,7,7) cost
  -51 to -98 (job 360). Of the two windows it is the forwarding window, not the ring (SLOT_STUDY P1, one window deepened
  at a time).
- **Galaxy N/S-involving rows are E1-bound, ~172 cycles/packet at 1-2 KiB** (V4_NS_FLOOR §1, job 167648). Two
  single-change variants account for the floor within ~8 cycles (V4_NS_FLOOR §2, jobs 167673, 412-413):
  - an E/W forwarder that takes the N/S routers' stateless forward rises ~37 on E1;
  - with the multi-downstream bookkeeping, another ~17-19.
- **The keepalive's share of main's 1D one-way rows:** 187.6 → 168.4 at keepalive 1000 (same job, a single setting
  changed).
- **Main's VC1 tier table costs main too:** 2-hop 454.1 / 518.4 → 305.8 / 303.4 at 25,8,16,4,4 (job 112, slots
  changed only).
- **Full-ring multicast on the Galaxy:** a bounded completion wait on idle passes is what the saturated ring needs
  (v4wt against v4uf, 350 → 310). Holding completions collapses the ring (v4p / v4p2).
- **An idle compiled-in VC2 costs VC0's 2D E/W 1-hop write +7.7 at 4352 B** (126.9 → 134.6, v4rxv2i3,
  V4_RECEIVER_SHAPE tables; fanin.json). The row is E0-bound with VC2 on (one-side A/B, V4_BOUNDS §1). How the 7.7
  splits across source lines comes from PC samples (V4_BOUNDS §3), so that split is a **hypothesis**.

**Hypotheses** (no isolating A/B):
- **Why v4 beats main overall.** No single-change attribution against main exists. Statements such as "fewer MMIO
  register operations" or "lean channel visits" are hypotheses.
- **Why v4trm trails main on Galaxy full-ring multicast 4352 B.** Candidates: how the sender shares a saturated link
  between injection and forwarded traffic; single-slot completion timing (worth ~40 of the gap by the v4wt A/B, not
  all of it). The lead has not been measured there.
- **Why one-ERISC rings and NeighborExchange trail:** v4's single-ERISC pass structure (profile only, job 131).
- **The codec stage's +2 to +4:** a volatile route-byte load plus zext.b on E/W-facing routers (V4_RECEIVER_OWNERSHIP
  §17).
- **The remaining ~8 of the Galaxy N/S floor:** the two-byte volatile route decode.
- **"Main's ~305 per-packet pass caps its VC1 rows"** (RESULTS §98, job 112): no A/B on main.
- **The +1.0 on Galaxy wire-bound rows:** the keepalive.
- **CCL decode ops being latency-bound and SDPA compute-bound:** stated in the docs, no A/B in the bundle. Main's SDPA
  15232 B outlier is "Cause not examined".
- **What binds VC2 across meshes:** the windows first is a hypothesis; VC2_LINE_BRINGUP phase 4 says "not established".
