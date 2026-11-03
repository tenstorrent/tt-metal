# Galaxy full-ring multicast at 4352 B: why v4rxg3 trails main (offline analysis, 2026-11-03)

Scope: existing data only (no device, Slurm or build jobs). Method rules from the repo's `.claude/CLAUDE.md`:
- a bound (E0 or E1) is named only by a one-side A/B;
- a cause stays a hypothesis until a variant that changes only that piece reproduces it;
- PC samples are used only to split cycles within one ERISC, and to compare routers of different ring classes in the
  same run (section 3.4 says what that comparison assumes).

Tables: `consolidation/fr_gap_samples.py` (`timeline`, `classes`, `split`). It reads the phase-2 sample sets through
`report/scripts/bd_map.py` and does not touch the shared builder.

Short version:
- The row's number is the time of the slowest sender, normalized to the 8-chip rings' links.
- The samples point to those 8-chip rings, which v4 runs at about 0.74 of the rate its saturated 4-chip-ring links
  reach (main: 0.97). v4's 8-chip-ring routers are neither busy nor back-pressured by the TX queue.
- The leading hypothesis is a per-hop credit-loop bound that grows with packet size. One piece of that loop already
  has an A/B behind it and was never ported to the lead: the bounded completion wait (about -40 at 4352 B).
- Six of the eight A/Bs below need no build, and the remaining two are small variants.

## 1. Data

### 1.1 The sweep (Slurm 169482 pass a on bh-glx-110-a10u20, 169483 pass b on a10u02; `report/data/bandwidth_glx_sweep.json`)

| cycles/packet | 256 | 1024 | 2048 | 3072 | 4352 |
| --- | ---: | ---: | ---: | ---: | ---: |
| main (passes) | 274.5 (270.1 / 278.8) | 270.3 | 271.1 | 270.0 (272.6 / 267.4) | 263.8 (263.8 / 263.7) |
| main, keepalive 1000 | 275.4 | 272.7 | 275.8 | 270.0 | 263.7 (263.7 / 263.7) |
| v4rxg3 (passes) | 191.5 | 191.7 | 192.0 | 192.1 | **372.2 (373.2 / 371.2)** |
| 1D wire (48 B header, `build_page.wire_cycles`) | 24 | 65 | 121 | 176 | 248 (2 Ethernet frames) |

- **Main's spread.** Main's passes differ by up to 9 cycles at 256 B and 5 at 3072 B, but only 0.1 at 4352 B.
- **v4's spread.** v4's two passes differ by 2.1 at 4352 B and by 0.0-0.1 below that.
- **Keepalive.** Main's keepalive (firmware 8000 vs 1000) changes nothing at 4352 B.
- **Other 4352 B rows sit at the wire for both routers.** Ring 1-hop, linear multicast, mesh and torus all run at
  248-252, except the bidirectional rows (259-268, both routers) and mesh multicast (331, both). Only the saturated
  full ring loses at 4352 B.

### 1.2 The sampled run (Slurm 169481, bh-glx-110-a10u08, point 5, 380 M packets per sender, ~720 s)

- **Rates.** v4rxg3 365.03 / 365.04 and main 262.83 / 262.83 (passes a / b).
- **Node effect.** The sampled rates are within 2% of the sweep: v4 is 7 below on this node, main 1 below.
- **Pass order.** Reversing the order on a10u08 changed nothing.

### 1.3 History (all Galaxy; slots = VC0 worker / forwarding / receiver on v4's ring routers)

| build | job, node | v4 ring slots | 3072 | 3584 | 4352 |
| --- | --- | --- | ---: | ---: | ---: |
| main | 156067 c02u02; 163369 c01u14; 169482/3 | tier 16 / 16 / 32 | 261-272 | 261.1 | 263.4-263.8 |
| v4u | 156067, 158477 | 16 / 16 / 32 (pre-merge host) | 217.2 | 218.6-218.9 | 302.7-305.2 |
| v4uf (single slot checked once) | 158477, 158562 a04u14 | 16 / 16 / 32 | 208.7 | 235.0-240.1 | 345.5-351.9 |
| v4wt (v4uf + bounded wait) | 158562 | 16 / 16 / 32 | 210-212 | 215.9-216.0 | 309.7-310.3 |
| v4eg (v4uf, newest slot credited early) | 158562 | 16 / 16 / 32 | 205-207 | 234.9-235.0 | 356.0-356.9 |
| v4trm | 163369 c01u14 | 26 / 25 / 32 (builder split, likely; not logged) | 201.7 | 244.2 | 347.7 (349.2 / 346.2) |
| v4rxg3 | 169482/3 a10u20 / a10u02 | 26 / 25 / 32 (builder split) | 192.1 | - | 372.2 |

- **The v4trm number fits the history.** It sits with v4uf, which had the same completion rule but different slots.
- **v4trm to v4rxg3 is +24.5, across nodes.** The node spread seen above is about ±1-2%. Whether the receiver
  redesign cost ~15-25 here is a hypothesis; it needs a same-node A/B.
- **v4 already loses at 3584 B, with one Ethernet frame.** v4trm runs 244 there, against a wire of 204, a router-bound
  201 and main's 261. So two frames per packet are not needed for the loss (section 5, H6).
- **Slots.** The receiver depth is 32, a power of 2, in both routers and at every packet size. The slot size is
  4400 B (max payload 4352 B), so the slot counts do not change between 3072 and 4352 B. Main (16 / 16 / 32) and v4
  (26 / 25 / 32) differ only in the sender channels.

### 1.4 The P150 analogue (400G, `report/data/bandwidth_sweep.json`, broker jobs 465/467, two passes within 0.1)

| cycles/packet, v4rxg3 (main) | 3072 | 4352 | step |
| --- | ---: | ---: | ---: |
| UnidirRingMulticast write | 198.9 (272.8) | 219.4 (273.0) | **+20.4** (main +0.2) |
| UnidirRingMulticast scatter | 239.5 (323.9) | 265.2 (324.1) | **+25.7** (main +0.2) |
| HalfRingMulticast scatter | 212.5 (299.7) | 225.4 (299.2) | **+12.9** (main -0.5) |
| RingMulticast write | 141.1 (212.2) | 147.5 (212.4) | **+6.4** (main +0.2) |
| HalfRingMulticast write | 171.9 (248.0) | 173.1 (247.2) | +1.1 |
| ring 1-hop / 3-hop / bidir, unicast | flat | flat | 0 (main 1-hop write +5.6) |

- **Only v4, only multicast.** v4 steps at 4352 B on multicast rows; main is flat on every ring row except its 1-hop
  write.
- **The wire is not involved.** At 400G the wire is 124 cycles at 4352 B, far below every one of these rows.
- **What this gives.** A size-dependent per-packet cost in v4's multicast hop, independent of the wire, that a P150 can
  A/B (H3).
- **What it does not give.** The P150 has no saturated, wire-bound full ring, so it cannot reproduce the Galaxy's
  amplification.

## 2. What the row measures

- **The pattern.** `full_ring` expands per axis (`tt_fabric_test_config.cpp:2731`, `get_full_or_half_ring_mcast_hops`).
  Every chip's four sender cores (11-5 W, 11-6 E, 11-7 S, 11-8 N) multicast around both rings through that chip, in
  both directions:
  - the 8-chip rings: 7 hops each way, 7 flows on every directed link;
  - the 4-chip rings: 3 hops each way, 3 flows per link.
- **The metric** (`tt_fabric_test_bandwidth_profiler.cpp:552-560`) is the slowest sender's cycles divided by the
  busiest link's packets (7 x num_packets). Every sender's cycles include the final global sync
  (`tt_fabric_test_sender.cpp:416-423`), so all 128 senders report the same cycles to 0.13%.
- **So cycles/packet** is the time at which the last flow finishes, per 8-chip-ring link packet. Link utilization and
  fairness between flows both enter it.

## 3. Facts from the point-5 PC samples (job 169481)

### 3.1 Data caveat for the breakdown

The `glx_169481_b_new_p5` window (v4, pass b) is idle for its first ~17.5 s of 25: `fr_gap_samples.py timeline` shows
0% work until the 8th 2.5-s bin. Its category shares are therefore diluted about 3.4x. Restricted to t ≥ 17.5 s, it
reproduces pass a to within 1 cycle. The other three sets carry traffic for the whole window.

### 3.2 Two ring classes

- **Which channels belong to which ring.** Channels 2 and 4 are the 8-chip rings (R8); channels 0, 6 and 8 are the
  4-chip rings (R4). Three things support this:
  - On R8, about 1/7 of the send-path samples belong to the worker (v4 0.147, main 0.158), which matches 7 flows per
    link.
  - The forward:local sample ratio, divided by each class's forwarded fraction, gives the same per-copy cost ratio in
    all three classes: k = 1.37 for v4 and 1.85 for main. The forwarded fractions are 6/7 on R8, and on R4 they follow
    from the unfair split in 3.3.
  - In the example `bh_x32_torus_xy_cluster_desc.yaml`, channels 0/6 form the 4-chip rings. a10u08 numbers its channels
    up to 9, so this supports the reading but does not prove it.
- **The routers the breakdown picks are not the ones that likely set the row.** The breakdown picks the busiest E0 by
  work share, which is an R4 router. By 3.4 the row is likely set by R8 (hypothesis until A/B 1).

### 3.3 The 4-chip rings settle into an alternating unfair state (both routers)

| E0 per class (cycles/packet at the row rate) | R4sat send w / f | R4sat TX-queue spin w / f | worker share R4sat / R4low |
| --- | --- | --- | --- |
| v4rxg3 (pass a; pass b from 17.5 s) | 38.1 / 52.8 | 81.3 / 100.1 | 0.44 / 0.12 (b: 0.45 / 0.13) |
| main (pass a) | 22.4 / 22.1 | 60.1 / 57.5 | 0.51 / 0.05 |

- **Which routers saturate.** On 32 routers, half the 4-chip-ring chips, E0 spends half its time in the post-send
  TX-queue spin:
  - v4: `router_sender.hpp:314`, ~181 cycles/packet;
  - main: `fabric_erisc_router.cpp:603`, ~118 cycles/packet.

  V4_BOUNDS §6 never saw this spin taken on uncongested rows (0 of 1.28 billion sends). These are the same 32 routers
  in v4 pass a, v4 pass b and main pass a; main pass b picked another set, still on channels 0/6/8 only.
- **Arbitration and starvation.** E0 alternates worker and forwarding sends 1:1 there (`pass()`: one send per channel
  visit), against a fair share of 1/3. The other chips' workers starve.
- **Implied injection ratio** between favoured and starved chips (flow conservation in a 3-flow ring, from worker
  shares and local-delivery ratios): about 4-6 for v4, 15 or more for main.
- **Both routers have this fairness defect.** By 3.4's rates the starved 4-chip-ring flows still finish before the
  8-chip-ring flows (estimate), so it is probably not this row's bound (H5).

### 3.4 The 8-chip rings: v4 runs them below the wire with neither ERISC busy

| local-delivery samples (rx.local / m.pt.local, cycles/packet) | R8 | R4sat core | R4low core | R8 / R4low |
| --- | ---: | ---: | ---: | ---: |
| v4rxg3 a (b from 17.5 s) | 34.9 (34.5) | 28.6 (28.8) | 47.0 (46.4) | **0.74** |
| main a | 36.1 | 19.4 | 37.4 | **0.97** |

- **Which links these measure.** An E1 on an R4low core receives an R4sat (TX-queue-saturated) link. One local write
  per packet at every hop makes these sample shares proportional to link packet rates, assuming equal per-write cost.
  The R4 forward:local consistency in 3.2 supports that assumption.
- **Link rates.** If the saturated R4 links run near the wire plus framing (~250-270), the 8-chip-ring links run at
  about 340-365 (v4) and 258-278 (main). These match the measured 365.0 and 262.8. This is an estimate, and it is why
  R8 is taken to set the row.
- **v4's R8 E0 is never back-pressured by the TX queue** (0 spin samples; main 10.5). Its send path is 64.5
  cycles/packet; the rest is:
  - pass polling: checks 92, of which the free-slots group read is 41 (`holds_packet`, `router_sender.hpp:198`);
  - counter reads and credits: credits 166, including the completion-counter re-reads at `visit_forwarding:515` and
    `visit_worker:469-476`;
  - loop: 42.
- **v4's R8 E1 polls far more than main's**, and spends more on admission:

| R8 E1, cycles/packet | checks | loop | admission | credits | decode | noc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v4rxg3 (365.0) | 70.6 | 26.9 | 45.8 (rx.down.admit) | 120.0 | 72.6 | 74.9 |
| main (262.8) | 16.3 | 3.5 | 22.7 (can_forward space + route) | 100.7 | 47.9 | 94.5 |

- **v4's admission lines.** The admission time sits at `router_downstream.hpp:348` (first use of the downstream credit
  register, re-read when the cached count runs out) and `:251`. The header word's first use
  (`fabric_edm_packet_header.hpp:459`) is 29.6.
- **What the split suggests.** On v4's 8-chip rings both ERISCs poll and the links are not back-pressured. The E1 often
  finds the next router's forwarding channel full, while that router's E0 waits for far completions. That is the
  signature of a credit-loop (bubble) limit rather than a CPU or wire limit. Samples cannot name the bound
  (CLAUDE.md), so this is a hypothesis for A/B 7.
- **Main's R8 E1 looks close to busy** (19.8 of 263 polling). It is consistent with main being E1-limited at ~264,
  near its router-bound ~270 at ≤3072 B. That too is unmeasured, since main has no one-side A/B.

### 3.5 Against points 2/4 and ≤3072 B

- **Points 2 and 4 are not comparable line by line.** They are 2D mesh rows at 1024 B without first-level acks, and run
  different router configurations.
- **No same-configuration sample exists at ≤3072 B**, so which source lines grow from 3072 to 4352 cannot be read from
  existing data. Probe P1 (a 3072 B full-ring sample run, no build) closes this.

## 4. Earlier answers: consistency and conflicts

- **RESULTS §98, job 156067: "loss appears exactly when the wire binds".** Consistent: v4 is router-bound at ≤3072 B,
  v4trm loses at 3584 B, and loses more at 4352 B.
  - Its suspect "E0's turnaround after the TX queue frees" is not supported on the 8-chip rings, where v4's TX queue
    is never busy. This is from samples, not an A/B; it stands as a refinement, not a conflict.
- **RESULTS §98, v4p / v4p2** (held completions collapse the ring) and the frame-count argument: consistent. No new
  frame evidence.
- **RESULTS §98, v4wt vs v4uf** (A/B fact: a bounded wait on passes without a packet gives 350 to 310 at 4352 B and
  238 to 216 at 3584 B) and v4eg (no gain): never applied to the lead.
  - v4rxg3 still checks a single slot once on every pass: `router_receiver.hpp:581`,
    `proof_checks = completion_group_size == 1 ? 1 : COMPLETION_PROOF_MAX_CHECKS`.
  - V4_PERF_GAPS Gap 1 lists this A/B as the first step when the gap is resumed. It was not run.
- **V4_PERF_GAPS Gap 1, "how v4's sender shares a saturated link between injection and forwarded traffic".** The 1:1
  sharing is real but appears on the 4-chip rings in both routers. The 8-chip rings, the likely bound, are fair.
  Refinement; not an A/B conflict.
- **Memory: rings are E1-bound** (P150 A/Bs). The Galaxy 8-chip ring at 4352 B looks different (section 3.4) but is
  unmeasured. A/B 7 decides.
- **Memory fabric-slot-defaults: check windows before router code.** A/B 5 is that check.
- No earlier A/B result is contradicted by the new data.

## 5. Hypotheses, ranked

**H1. The 8-chip rings are credit-loop bound at 4352 B; v4's per-hop loop is longer than main's.**
- **Status (section 7.2):**
  - Its first claim is confirmed by A/B 1: the 8-chip rings set the row. v4 8-ring-only 374.1, against 368.1 for
    the full ring; main 263.7 against 263.7.
  - The latency (credit-loop) mechanism is still a hypothesis. A/B 5 rules out a far-ring window bound, and P1 shows
    v4's per-packet send and NoC work unchanged from 3072 B, with the extra time all waiting on both ERISCs.
    A/B 7 names the bound.
  - The torus row (7.2) has the same first-level-ack completion rule and shows no cliff, so the longer loop is
    specific to the 1D ring routers (hypothesis). **WRONG as stated (user, 2026-11-03):** main is flat on the torus
    (290.5 -> 291.5) while v4 grows 246.4 -> 270.2 from 3072 to 4352 B (+23.8), a size-dependent step in the same
    direction, smaller and capped near the 2D wire (250). The torus does not rule out a shared (ring / first-level-ack)
    mechanism; it shows a smaller one.
  - The "one-NoC two-copy (0.038/B)" estimate below is **WRONG in size**: the P150 slope is 0.026 cycles/B.
- Claim:
  - The row is the 8-chip rings' link rate: ~365 for v4, ~264 for main.
  - v4's 8-chip rings sit at a latency (bubble) limit set by the per-hop credit loop, not by either ERISC's per-packet
    work and not by the wire.
  - The loop's size-dependent parts:
    - the packet's wire time before the first-level ack (248 vs 176);
    - the forward and local writes on NoC1 before the completion proof (2 x 4400 B, about 166 cycles at ~53 B/cycle per
      NoC, RESULTS §14, vs about 118 at 3072 B);
    - the completion check (H2) and the credit frames.
  - At ≤3072 B the loop is short enough that v4's router (192) binds.
- Evidence: the facts in sections 3.3-3.4, and v4trm's growth with size (3072 201.7, 3584 244.2, 4352 347.7). Above
  the wire crossing the loss grows ~0.08-0.13 cycles/B, more than the wire's 0.054/B; the extra is about one-NoC
  two-copy write time (0.038/B). That last link is an estimate.
- What contradicts it:
  - an 8-chip-ring-only row at 4352 B near the wire (A/B 1);
  - a one-side delay that moves the row by about its full size on one side and ~0 on the other (a CPU bound, A/B 7);
  - a deeper receiver ring speeding the row up in proportion (a window bound, A/B 5).

**H2. Completion-credit timing on that loop: the missing bounded wait.**
- **Status:** open; needs variant A/B 2. The torus row uses the same single-check rule without a cliff, which
  lowers the prior that this rule alone carries the 1D gap. **Corrected (user, 2026-11-03):** the torus does step
  with size on v4 only (+23.8, main flat), so the prior is not lowered by it. H2 stays open.
- Claim: of the per-hop latency, the single-slot proof "checked once" defers a completion credit to a later pass. On a
  saturated ring that credit is the bubble's.
- Evidence: the A/B on v4uf (section 4), worth ~40 of the then-gap. v4rxg3 has the same rule. Main checks once per
  step too, so this does not explain the whole gap to main (v4wt still ran 310).
- What contradicts it: A/B 2 moving 4352 B by less than 10.

**H3. A size-dependent per-packet cost in v4's multicast hop, seen on the P150.**
- **Status:** the size dependence is confirmed by A/B 3. There is no step between 4000 and 4096 B (+1.1 to +2.1); v4
  grows with size from 3584-3840 B and main stays flat. The one-NoC mechanism is still a hypothesis (variant A/B 4),
  and its predicted slope was too steep (section 7.2).
- Claim:
  - The P150 multicast rows step up at 4352 B for v4 only, at 400G where the wire is 124.
  - Candidate mechanism: the forward and local writes serialized on one NoC. Both routers do this
    (`edm_to_local_chip_noc = 1`, forwarding NoC id 1; v4 asserts it, `router_receiver.hpp:497`). Main's slower E1
    hides it.
  - On Galaxy this time sits inside H1's loop.
- Evidence: the section 1.4 table, and RESULTS §14 (two 4 KiB copies on one NoC take 149 cycles, on two NoCs 102).
- Caveat: RingMulticast write runs at 147.5 at 4352 B, below a 166-cycle one-NoC floor if every link packet were both
  forwarded and written. So the floor model is not exact for that row (its normalization or forward fraction differs);
  hypothesis.
- What contradicts it: the P150 step staying with the local write skipped or moved to NoC0 (A/B 4), or a jump between
  4000 and 4096 B instead of growth with size (A/B 3).

**H4. Slot allocation (v4 26 / 25 / 32 vs main 16 / 16 / 32).**
- **Status: contradicted by A/B 5.** Tier depths 16 / 16 / 32 give 370.3 against the lead's 370.9; 18 / 17 / 32
  gives 369.6. A 48-slot receiver gives 361.0, about -9 against 18 / 17 / 32 despite its known ~+7 non-power-of-2
  cost at 3072 B, so it is not a window bound.
- Claim: deeper sender channels change the saturated ring's equilibrium (queueing per hop, injection pressure).
- Evidence: weak. The older v4 builds at 16 / 16 / 32 ran 303-357, and v4trm at the split 347.7, with different code.
  Every 4352 B v4 number since the split is at 26 / 25 / 32.
- What contradicts it: A/B 5 within ±10.

**H5. The 4-chip rings' starvation sets the row instead.**
- **Status: contradicted by A/B 1.** The 4-chip-ring-only row takes ~1020 cycles per sender packet in both routers
  (v4 340.0 x 3, main 342.7 x 3). The 8-chip-ring-only row takes 2619 (v4) and 1846 (main).
- Claim: the starved 4-chip-ring flows finish last.
- Evidence against: the section 3.4 rates put the starved R4 flows' total time well under the R8 flows' (estimate).
- What contradicts it (that is, confirms H5): the 4-chip-ring-only row slower per sender than the 8-chip-ring-only
  row (A/B 1).

**H6. The Ethernet link layer: two frames per packet, sequence acks, keepalive.**
- **Status: contradicted.**
  - A/B 6: keepalive at firmware 8000 / 1000 / 250 gives 370.9 / 370.9 / 370.6.
  - A/B 8: 4000 to 4096 B adds +7.4 on Galaxy, after 192 to 358 between 3072 and 4000 B.
  - A/B 3: +1.1 to +2.1 on the P150.
- Claim: the TX queue's window or acks lose wire time at 4352 B.
- Evidence:
  - TX-queue back-pressure appears only on the 4-chip rings' saturated links, in both routers; v4's 8-chip-ring links
    show none.
  - v4trm loses at 3584 B with one frame.
  - Main's keepalive changes nothing.
- What contradicts it (that is, supports H6): A/B 6 moving the row by more than 10, or A/B 8 jumping at 4000 to 4096 B.

**H7. Node variance.** About 2% (365 on a10u08 against 372 on a10u20 / a10u02). It does not explain 41%. Keep the A/Bs
same-node and same-job.

## 6. Isolating A/Bs, cheapest first (predictions written before any run)

No-build Galaxy batch (one allocation per pass, both routers, trees `base_1102` / `v4rxg3_1102`, yamls only plus env):

| # | change (only this) | tests | prediction |
| --- | --- | --- | --- |
| 1 | full-ring multicast on one axis: senders on all 32 chips with explicit hops N 7 / S 7 (8-chip rings only), then E 3 / W 3 (4-chip rings only); 3072 and 4352 B. Check the axis against the log's busiest-link count (7 x num_packets); swap if the mesh is 4x8 | H1 vs H5 | v4 8-ring-only 4352 B = 355-375 (3072 B ~192); main 8-ring-only 258-268; per-sender time of the 4-ring-only row below the 8-ring-only row for both routers. **Contradicts H1:** v4 8-ring-only ≤ 300 |
| 8 | full ring at 3584, 4000, 4096 B | H1 / H3 vs H6 | (**magnitudes WRONG**, 7.2: measured 295 / 358 / 365 / 368) v4 3584 225-250, 4000 300-335, 4096 310-345, 4352 ~372: growth with size, at most ~10 extra between 4000 and 4096; main 258-272 throughout. **H6** predicts flat-ish to 4000, then a jump at 4096 |
| 5 | v4rxg3 4352 B with `TT_FABRIC_EXP_TIER_SLOTS=1` (16 / 16 / 32), and with `TT_FABRIC_EXP_SLOTS_1D=18,17,48` (receiver 48 of 83; 48 is not a power of 2, and `index_add` handles that at a cycle or two) | H4, window bound | both within ±10 of the lead if the loop is bubble-bound (H1). A window bound on the far ring would show receiver 48 well below the lead (toward ~260); H4 would show the tier slots below the lead |
| 6 | v4rxg3 4352 B with keepalive unset (firmware 8000), 1000 (lead), 250 | H6 | all within ±5 |

P150 (broker, yaml only, ~15 min, both passes):

| # | change | tests | prediction |
| --- | --- | --- | --- |
| 3 | ring suite multicast rows (UnidirRingMulticast write/scatter, HalfRingMulticast scatter, RingMulticast write) at 3072, 3584, 3840, 4000, 4096, 4352 B | H3 vs frames | (frames prediction held; **floor constants WRONG**, 7.2: onset earlier, slope 0.026 not 0.038/B) v4 UnidirRingMulticast write ~199 to 3584 B, then rising smoothly (3840 ~200, 4000 ~206, 4096 ~209, 4352 ~219; a one-NoC two-copy floor ≈ 2(s+48)/53 + ~53); no extra jump at 4000 to 4096 (< 4); main flat. Frames predict ~199 to 4000, then ~219 at 4096 |

Variants (each one change; per memory, design agreed with the user before building):

| # | variant | tests | prediction |
| --- | --- | --- | --- |
| 2 | v4rxg3 + bounded completion wait on passes without a packet: `proof_checks = (group 1 && handled_packet) ? 1 : COMPLETION_PROOF_MAX_CHECKS` (v4wt's rule) | H2 | Galaxy 4352 B 372 to 327-342; 3584 B -15 to -25; 3072 B 0 to +3; P150 ring rows 0 to +3 (v4wt paid +2.8). **Contradicts H2:** 4352 B moves < 10 |
| 4 | v4rxg3 + skip the local write of data-only packets (main's `EXP_RX_SKIP_LOCAL_DELIVERY` idea; test-only, P150 first). Cleaner but costlier: local writes on NoC0 with the proof read on both NoCs | H3 (and its share of H1) | P150 4352 B multicast steps ≤ +3; Galaxy 4352 B drops by ≥ 40. **Contradicts H3:** P150 step unchanged |
| 7 | one-side delay on Galaxy: v4bdd's hooks (now on v4rxv2i3) rebased on v4rxg3; `EXP_TX_DELAY` 0 / 32 and `EXP_RX_DELAY` 0 / 32 on A/B 1's 8-ring-only row at 4352 B | names the bound (CLAUDE.md) | latency-bound (H1): each side's 32 moves the row by a third to all of 32, both sides (V4_BOUNDS §1b signature). A CPU bound shows ≥ 32 on one side, ~0 on the other; a wire bound ~0 on both |

Probes (not A/Bs; no build): P1, the point-5 sampler at 3072 B (`galaxy/pcs/glx_fullring4352.yaml` with the size
changed), for line-by-line growth on R8. P2, the same row with debug logging, for the per-sender stream counters
(busy cycles, slot waits): which senders lag.

Order:
1. The Galaxy batch (1, 8, 5, 6, plus P1/P2) and P150 A/B 3, in parallel, needing no build.
2. A/B 2, built on the batch's outcome.
3. A/B 4 or 7, whichever the batch leaves open.

## 7. Runs (2026-11-03)

Jobs:
- Galaxy Slurm 169776 (bh-glx-110-a08u14): A/B 1 and A/B 8, two passes in reversed order.
- Galaxy Slurm 169777 (a08u20): A/B 5, A/B 6 and probe P1.
- P150 lb2 169780 (bh-lb-120-a07u28): A/B 3.

Files:
- `fabric_exp/galaxy/frs_glx_frgap.sbatch`: a copy of frs_glx_rsz.sbatch with the gap's suites and configs.
- `fabric_exp/galaxy/frgap/*.yaml`, `consolidation/frgap/p150_ring_mcast_sizes.yaml`, `consolidation/frg_p150_runs.sh`.
- Tables: `consolidation/frg_tables.py`.

Every new yaml and slot/keepalive setting reached program launch on the mock Galaxy / P150 descriptor before it ran.

### 7.1 Torus row: prediction (written 04:27 UTC, before the run)

The coordinator added a full-ring multicast on the 2D XY torus: `galaxy/frgap/glx_frg_torus.yaml`, `full_ring` with
Torus XY at 3072 and 4352 B, main against v4rxg3, two passes in one job.

- **Why it tests something new.** v4's single-check completion rule (`router_receiver.hpp:581`) applies to every
  receiver channel with first-level acks, which means tori as well as rings. The mock gives both routers 17/16/16/16
  sender slots and a 16-slot receiver there (1D: 26/25/32 vs main 16/16/32).
- **Prediction if the cliff comes from the shared first-level-ack loop (H1/H2):**
  - v4 at 4352 B is 330-410 and below main (v4/main 0.65-0.85);
  - v4 at 3072 B is 190-235, ahead of main;
  - main is 260-290 at both sizes (its torus 1-hop rows run 265-269).
- **What contradicts it.** v4 at 4352 B within 10% of the wire-bound value (~250-280, 80 B header, wire 250) points to
  something specific to the 1D ring routers instead: their slot split, or 1D routing.
- **Outcome (7.2): the shared-loop prediction's numbers missed.** v4 ran 270.2 at 4352 B, ahead of main's 291.5. A/B 5
  rules out the slot split.
- **But the torus still steps on v4.** v4 goes 246.4 -> 270.2 (+23.8) while main stays flat, so the torus does not
  rule 2D out (user correction, 7.2).

### 7.2 Results (cycles/packet, pass a / pass b; tables by `frg_tables.py`)

**A/B 1, one ring axis (Galaxy 169776, a08u14).**

| row | main 3072 | main 4352 | v4rxg3 3072 | v4rxg3 4352 |
| --- | ---: | ---: | ---: | ---: |
| N/S only = 8-chip rings (7 flows/link, from the log) | 261.3 / 266.7 | 263.7 / 263.7 | 191.9 / 191.9 | 371.3 / 376.9 |
| E/W only = 4-chip rings (3 flows/link) | 285.0 / 286.5 | 342.8 / 342.6 | 181.5 / 181.5 | 340.0 / 340.1 |
| full ring, same job (A/B 8) | 267.2 / 267.3 | 263.7 / 263.8 | 192.1 / 192.1 | 367.2 / 368.9 |

- **Every prediction held.** v4 8-ring-only is 355-375 at 4352 B (pass b 376.9, just above) and 192 at 3072 B. Main is
  258-268.
- **Per-sender times.** The 4-ring-only row takes 1020 / 1028 cycles per sender packet (v4 / main) at 4352 B. That is
  below the 8-ring-only row's 2619 / 1846.
- **The 4-chip rings are as slow on main as on v4 at 4352 B** (342.7 vs 340.0, per 3-flow link packet). The unfair
  4-ring state of section 3.3 does not depend on the router; it just never sets the row.

**A/B 8, full ring by size (169776).**

| | 3072 | 3584 | 4000 | 4096 | 4352 |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 267.2 / 267.3 | 261.4 / 261.1 | 261.5 / 261.5 | 261.8 / 261.7 | 263.7 / 263.8 |
| v4rxg3 | 192.1 / 192.1 | 295.3 / 294.7 | 357.4 / 358.1 | 363.4 / 366.7 | 367.2 / 368.9 |
| wire (1D, 48 B header) | 176 | 204 | 226 | 234 (2 frames) | 248 |

- **No frame step.** 4000 to 4096 B adds only +7.4 (predicted ≤ 10).
- **The predicted magnitudes were WRONG.** The note said 3584 225-250, 4000 300-335 and 4096 310-345. Most of the loss
  arrives already between 3072 and 4000 B, and v4 then sits at ~360-370 whatever the size.
- **v4rxg3 loses more at 3584 B than v4trm did** (295 vs 244, different nodes and jobs).

**A/B 5 and A/B 6, slots and keepalive (Galaxy 169777, a08u20); full ring, v4rxg3.**

| config | 3072 | 4352 |
| --- | ---: | ---: |
| lead (26 / 25 / 32, keepalive 1000) | 192.1 / 192.1 | 371.1 / 370.7 |
| tier 16 / 16 / 32 | 196.1 / 196.0 | 368.5 / 372.1 |
| 18 / 17 / 32 | 195.0 / 195.0 | 369.9 / 369.4 |
| 18 / 17 / 48 | 202.4 / 202.4 | 362.1 / 360.0 |
| keepalive firmware (8000) | 192.1 / 192.1 | 370.5 / 371.3 |
| keepalive 250 | 192.1 / 192.1 | 369.8 / 371.3 |

- **Every arm is within ±10 at 4352 B, as predicted for a non-window, non-Ethernet-ack limit.** The slots were logged
  per run and match each arm.

**A/B 3, P150 (lb2 169780, a07u28), v4rxg3 (main flat throughout, 212-324).**

| row | 3072 | 3584 | 3840 | 4000 | 4096 | 4352 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UnidirRingMulticast write | 198.9 | 199.0 | 205.7 | 211.7 | 213.5 | 219.3 |
| UnidirRingMulticast scatter | 239.5 | 254.3 | 255.6 | 259.9 | 262.0 | 265.1 |
| HalfRingMulticast scatter | 212.5 | 219.5 | 220.4 | 222.4 | 223.6 | 225.4 |
| RingMulticast write | 141.1 | 141.0 | 140.9 | 143.0 | 144.1 | 147.6 |

- **Frames: no step.** 4000 to 4096 B adds +1.1 to +2.1.
- **Unidirectional write against the prediction:** 3584 199 (predicted ~199), 3840 205.7 (~200), 4000 211.7 (~206),
  4096 213.5 (~209), 4352 219.3 (~219). Growth starts earlier, and the slope is shallower than the one-NoC two-copy
  floor (0.026 vs 0.038 cycles/B). The floor model's shape fits; its constants do not.

**Torus row (Galaxy 169808, a08u14).** The run logs show "axis topologies derived as Y=RING X=RING", with no fallback
line.

| | 3072 | 4352 |
| --- | ---: | ---: |
| main | 290.5 / 290.5 | 291.4 / 291.6 |
| v4rxg3 | 246.4 / 246.4 | 270.3 / 270.1 |
| 2D wire (80 B header) | 178 | 250 |

- **The prediction for the shared-loop hypothesis was WRONG.** It said v4 at 330-410 and below main. Instead v4 is
  ahead of main at both sizes (1.08x at 4352 B) and within 8% of the wire. That was the note's "contradicts" branch.
- **So the 1D cliff is specific to the 1D ring routers (hypothesis).** The torus has the same single-check completion
  rule, 16-slot receivers, four sender channels and 2D routing. **WRONG as stated (user, 2026-11-03):** v4 grows
  246.4 -> 270.2 (+23.8) from 3072 to 4352 B while main stays flat (290.5 -> 291.5). That is the same size-dependent
  step, smaller, and at 4352 B v4 sits near the 2D wire (250), which may cap it. 2D is not ruled out. The 1D ring's
  larger step may come from its longer per-link load (7 flows) on the same mechanism. Open.
- **Caveat.** v4's torus router-bound rate (246 at 3072 B) sits close to its 4352 B wire time (250). But the 1D ring
  lost 103 cycles at 3584 B with its router (192) just under the wire (204), so the difference is real.

**Probe P1, samples at 3072 B (169777, a08u20, both passes; window checked: traffic in every bin on R8).** v4rxg3's
8-chip-ring routers, cycles/packet at 192.0 vs 365.0 (point 5):

| | E0 send | E0 credits / checks / loop | E1 forward / local | E1 admission | E1 checks + loop | E1 credits | E1 decode |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3072 B | 61.8 | 78.5 / 34.8 / 16.8 | 39.2 / 32.1 | 4.2 | 23.9 | 71.4 | 25.4 |
| 4352 B | 64.5 | 166.3 / 92.3 / 41.9 | 40.0 / 34.9 | 45.8 | 97.5 | 120.0 | 72.6 |

- **Per-packet work does not grow.** The work done per packet barely changes from 3072 to 4352 B: the sender's send
  path and the receiver's forward and local NoC commands.
- **Where the added ~173 cycles go:**
  - waiting for downstream room on E1 (admission, +41.6);
  - polling on both ERISCs;
  - E1's header first use (decode, +47).
- **Main's 8-ring profile is the same at both sizes** (E1 polling ~20, admission ~14).
- **What this means.** The added time is waiting, not work, as a latency limit predicts (H1). Samples cannot name the
  bound, so this is a hypothesis until A/B 7.

### 7.3 Hypothesis table after the no-build A/Bs

| hypothesis | status | next |
| --- | --- | --- |
| H1 the 8-chip rings set the row | **confirmed** (A/B 1) | - |
| H1 they are credit-loop (latency) bound; v4's loop longer | open; supported by P1 (waiting, not work) and A/B 5 (no window bound) | A/B 7 (delay hooks) |
| H1' the cliff is specific to v4's 1D ring routers (new) | open; **weakened (user, 2026-11-03): the torus also steps +23.8 on v4 only** | a saturated Linear multicast row; run the delay and local-write A/Bs on the torus row too |
| H2 single-slot completion check | open (the torus step keeps it in play) | A/B 2 (variant) |
| H3 size-dependent multicast-hop cost | size dependence **confirmed** (P150, no frame step); mechanism open; slope prediction WRONG (0.026 vs 0.038/B) | A/B 4 (variant) |
| H4 slot allocation | **contradicted** (A/B 5) | - |
| H5 4-chip rings set the row | **contradicted** (A/B 1) | separate fairness item (both routers) |
| H6 Ethernet layer (frames, keepalive) | **contradicted** (A/Bs 6, 8, 3) | - |
| H7 node variance | about 1-2% (368 on a08u14) | - |

## 8. Second round (user-approved, 2026-11-03)

### 8.1 Linear saturated multicast: design and prediction (written 04:53 UTC, before the run)

**Existing data.** The sweep already has a saturated line multicast: `GlxLinearMulticast`, all_to_all on Linear (Slurm
169343/169403, a02u14).
- Every chip multicasts to both ends of its column and its row.
- The busiest links (into rows 0 and 7) carry 7 flows, as on the 8-chip ring: 34.99 M cycles per 20000 sender packets
  is 1749 per packet, 7.0 x 249.9.
- v4rxg3 runs it at the wire: 177.3 at 3072 B, 249.9 at 4352 B. Main runs 197.1 / 250.6. So a line with the ring's
  busiest-link load shows no cliff.

**The row (`galaxy/frgap/glx_frg_line.yaml`).**
- `GlxLineMcastNS` is the line analogue of A/B 1's 8-ring-only row: 32 explicit senders, each multicasting to every
  other chip of its 8-chip column (N: row hops, S: 7 - row).
- `GlxLinearMulticast` (all_to_all) is re-run as the same-job reference.
- 3072 and 4352 B, main against v4rxg3, two passes reversed.
- Linear means no deadlock avoidance: no first-level acks, no bubble rule, and completion groups of 4 instead of single
  slots. Slots are the same as on the ring (v4 26 / 25 / 32 from the mock, main 16 / 16 / 32).
- Mock-vetted on both routers.

**Prediction (no cliff on the line):**
- v4rxg3 `GlxLineMcastNS` at 3072 B is 176-185 and at 4352 B is 248-260, at the wire.
- Main is 185-200 at 3072 B and 249-255 at 4352 B.
- `GlxLinearMulticast` reproduces 177.3 / 249.9 (v4) and 197.1 / 250.6 (main) within 2%.
- If this holds: the cliff needs the ring's deadlock-avoidance mode (first-level acks, the bubble rule, single-slot
  completions) or the ring's uniform saturation. It is not the 1D multicast hop as such.
- **What contradicts it:** v4 more than 10% above the wire at 4352 B on `GlxLineMcastNS`. Then the cliff needs
  neither the ring mode nor wraparound.

### 8.2 A/B 7 and A/B 4: variants, identity checks, predictions (written ~05:10 UTC, before any run)

**Variants.**
- `variants/v4rxg3bd` = v4rxg3 + v4bdd's two hooks:
  - `fabric_exp_tx_delay` / `fabric_exp_rx_delay` (NAMED_CT_ARG EXP_TX_DELAY / EXP_RX_DELAY);
  - a spin after the packets-received update in `send()`;
  - a spin after `retire` in `handle()`;
  - the phase-profiler include.
- `variants/v4rxg3sl` = v4rxg3 + `exp_skip_local_write = EXP_FLAGS bit 0`. With the bit set, `handle()` skips
  `deliver()` for NOC_UNICAST_WRITE packets only. The host does not use bit 0 (checked: it only sizes the
  phase-profile buffer, from other bits).

**Galaxy trees.** `trees/v4rxg3bd_1102` and `trees/v4rxg3sl_1102` (`BASE=base_1102 mk_glx_tree.sh`).

**Identity check.**
- coverage.py's offline build could not be used: it reads the live tree's router headers, which another job's swap
  rewrote at 04:56. Even an unmodified v4rxg3 failed there.
- Instead, the identity check mock-compiles the same rows on `v4rxg3_1102` and on each variant tree, then compares
  every router ELF's loadable sections pairwise (`consolidation/frg_elfcmp.py`): Galaxy 8-ring-only and torus rows,
  and the P150 multicast rows for v4rxg3sl.
- Results are in 8.3.

**A/B 7 predictions** (v4rxg3bd; base / tx32 / rx32; 32-cycle spin per packet sent / received).

| row | base (measured before) | if latency (credit-loop) bound | if one ERISC bounds | if wire bound |
| --- | ---: | --- | --- | --- |
| 8-ring only, 4352 B | ~372 | tx32 and rx32 each +11 to +32 | ≥ +32 on that side, ~0 on the other | ~0 both |
| 8-ring only, 3072 B (router-bound) | ~192 | - | expected E1 (rings were E1-bound on P150): rx32 +28 to +40, tx32 0 to +10 | - |
| torus full ring, 3072 B | ~246 | - | rx32 +28 to +40, tx32 0 to +10 (same expectation, 2D E1) | - |
| torus full ring, 4352 B | ~270 | both +11 to +32 | one side ≥ +32 | ~0 both (it is within 8% of the 250 wire) |

- (Outcome, 8.5: **WRONG** for the 8-ring at both sizes; right for torus 3072 B; half right for torus 4352 B.)
- My call for each row: 8-ring at 4352 latency-bound; both 3072 B rows E1-bound; torus at 4352 latency-bound, since it
  also steps on v4 only.
- A latency-bound 4352 B row is the H1 signature (V4_BOUNDS §1b: "a spin on either side moves them by about a third to
  a half of its cost").

**A/B 4 predictions** (v4rxg3sl, flag off / on; the local write of data-only packets dropped).
- P150, flag on:
  - the 3072-to-4352 B step on the multicast rows falls to ≤ +3;
  - every size gets faster where E1 binds: unidirectional write ~199 -> 180-195 at 3072 B.
- Galaxy 8-ring only: 4352 B drops by ≥ 40 (372 -> ≤ 332); 3072 B gets faster by the write's issue cost (192 -> 180-190).
  **WRONG (8.6):** +1.9 at 4352 B and +86.6 at 3072 B.
- Torus: 4352 B 270 -> ≤ 258 (step mostly gone); 3072 B 246 -> 230-240. **WRONG at 4352 B (342.2);** 3072 B went
  the predicted way (216.1).
- What contradicts H3: the P150 step unchanged with the flag on.

### 8.3 Identity checks (05:07 UTC)

Method: mock compiles on the Galaxy descriptor (8-ring-only and torus rows) and the P150 descriptor (A/B 3 rows), then
`frg_elfcmp.py`, which compares every router ELF's loadable sections by config.

| comparison | result |
| --- | --- |
| v4rxg3bd at delays 0 vs v4rxg3 | 240 / 240 Galaxy ELFs byte-identical |
| v4rxg3sl, flag off, vs v4rxg3 | Galaxy 240 / 240 and P150 216 / 216 byte-identical |
| tx32 / rx32 / flag on (paired by config, EXP_* named args ignored) | each changes exactly its core: tx32 differs on all 120 E0 images (E1 identical), rx32 on all 120 E1 images (E0 identical), the flag on every E1 image (Galaxy 120, P150 108; E0 identical) |

### 8.4 Linear row: results (Galaxy 169854, bh-glx-110-a09u14, two passes reversed, identical to 0.1)

| row | main 3072 | main 4352 | v4rxg3 3072 | v4rxg3 4352 |
| --- | ---: | ---: | ---: | ---: |
| GlxLineMcastNS (8-chip lines; busiest link 7 flows: 34.99 M / 20000 = 1749 = 7 x 249.9) | 197.1 | 250.5 | 177.3 | 249.9 |
| GlxLinearMulticast (all_to_all, both axes; same-job reference) | 197.1 | 250.5 | 177.3 | 249.9 |
| 1D wire | 176 | 248 | 176 | 248 |

- **The prediction held.**
  - v4 sits at the wire at both sizes (176-185 / 248-260 predicted);
  - main runs 197.1 / 250.5 (185-200 / 249-255 predicted);
  - the reference reproduces the sweep exactly (177.3 / 249.9 and 197.1 / 250.5).
- **No cliff on a saturated line.** The line has the ring's busiest-link load (7 flows) and the same v4 1D routers:
  - the same slots (26 / 25 / 32);
  - 1D routing;
  - forward + local write on NoC1;
  - the same multicast hop.
- **Ruled out:** the 1D multicast hop by itself is not enough for the cliff. H3's size-dependent hop cost alone is not
  enough on Galaxy 1D; at most it contributes inside the ring's loop.
- **Still standing** (the line differs from the ring in two ways that this row cannot separate):
  - the ring's deadlock-avoidance mode: first-level acks, the bubble rule (injection needs two free far slots) and
    single-slot completions. The line has none of these and credits in groups of 4.
  - uniform saturation: on the ring every link carries 7 flows; on the line only the links next to the ends do, and
    the middle links have slack.
- A/B 7 (delays) and A/B 4 (local write) run next on the 8-ring and torus rows (8.2 predictions).

### 8.5 A/B 7, one-side delays (Galaxy 169870, bh-glx-110-a10u20, v4rxg3bd, two passes reversed)

| row | base | tx32 (E0 +32/packet) | rx32 (E1 +32/packet) |
| --- | ---: | ---: | ---: |
| 8-ring only, 3072 B | 191.9 / 191.9 | 248.7 / 249.6 (**+57.3**) | 237.5 / 237.5 (**+45.6**) |
| 8-ring only, 4352 B | 372.9 / 376.4 | 385.1 / 384.6 (+10.2) | 330.5 / 330.4 (**-44.2**) |
| torus full ring, 3072 B | 246.4 / 246.4 | 246.4 / 246.4 (0.0) | 290.0 / 290.0 (+43.6) |
| torus full ring, 4352 B | 270.0 / 269.9 | 274.8 / 274.7 (+4.8) | 290.8 / 290.8 (+20.8) |

- **8-ring at 4352 B: prediction WRONG.** I predicted a latency bound, both arms +11 to +32. In fact a slower receiver
  makes the row *faster* by 44, and the TX arm adds +10. That is none of the CPU, latency or wire signatures. The row
  depends on the receiver's speed, the wrong way round.
- **8-ring at 3072 B: prediction WRONG.** I predicted E1-bound (rx +28 to +40, tx 0 to +10). Both sides slow the row by
  more than 32, E0 more so: a coupled credit loop with slopes above 1 (V4_PROFILE §6's compounding).
- **Torus at 3072 B: E1-bound, as predicted** (rx +43.6, tx 0).
- **Torus at 4352 B: mixed, mostly E1** (rx +20.8, tx +4.8). The prediction (latency, both +11 to +32) half held.

### 8.6 A/B 4, local-write skip (Galaxy 169871 a08u20; P150 lb2 169891 a08u28; v4rxg3sl, flag off / on)

| Galaxy row | flag off | flag on (local write of data-only packets dropped) |
| --- | ---: | ---: |
| 8-ring only, 3072 B | 191.9 / 191.9 | 278.7 / 278.4 (**+86.6, slower**) |
| 8-ring only, 4352 B | 371.5 / 373.2 | 373.9 / 374.4 (+1.9) |
| torus full ring, 3072 B | 246.4 / 246.4 | 216.1 / 216.1 (-30.3) |
| torus full ring, 4352 B | 270.1 / 270.0 | 342.4 / 342.0 (**+72.2, slower**) |

| P150 row (passes c / d within 0.1; flag off = 169780's v4rxg3 within 0.4) | off 3072 -> 4352 | on 3072 -> 4352 |
| --- | --- | --- |
| UnidirRingMulticast write | 198.9 -> 219.4 (+20.5) | 167.8 -> 168.0 (**+0.2**) |
| HalfRingMulticast write | 171.8 -> 173.0 | 144.5 -> 145.2 |
| RingMulticast write | 141.1 -> 147.6 (+6.5) | 135.1 -> 139.9 (+4.8) |
| scatter rows (the flag skips only NOC_UNICAST_WRITE: control) | step kept (e.g. unidirectional 239.5 -> 265.1) | the same +1 |

**P150: the H3 prediction held.**
- The write rows' size step goes away with the local write skipped: unidirectional +20.5 -> +0.2.
- Every size gets faster: unidirectional 198.9 -> 167.8, beyond the predicted 180-195.
- On the P150, the local write is the size-dependent piece of the multicast hop. That is an A/B fact; whether NoC time
  or issue cost inside the write matters is not yet separated.

**Galaxy: the predictions were WRONG.**
- I predicted the 8-ring at 4352 B drops by ≥ 40: it moved +1.9.
- I predicted the 8-ring at 3072 B gets faster: it got +86.6 slower.
- I predicted the torus at 4352 B ≤ 258: it ran 342.
- Only the torus at 3072 B moved in the predicted direction (216.1).
- So the local write is not the Galaxy cliff, and removing it pushes both rows into a slow mode.

### 8.7 What the round shows (hypothesis H9, new; marked as such until its own A/B)

**H9: the ring and torus routers fall into a slow mode, about 1.4-1.6x the wire, whenever v4's receiver serves a packet
faster than the link delivers one.** Five rows agree:

| case | E1 service vs link time | result |
| --- | --- | --- |
| 1D full ring by size (A/B 8) | E1 ~192; wire 176 at 3072 B, 204 at 3584 B and up | router-bound 192 at 3072 B; slow mode from 3584 B (295 = 1.45x, 358 = 1.58x, 365, 368 = 1.48x wire) |
| 1D 8-ring, 3072 B, local write skipped | E1 drops below the 176 wire | slow mode: 278.5 = 1.58x wire |
| 1D 8-ring, 4352 B, rx32 | E1 slower, closer to the 248 wire | partly out of it: 374.6 -> 330.4 |
| torus, 4352 B | E1 ~246 against wire 250 (borderline) | the +23.8 step; with the local write skipped (E1 well under the wire) 342 = 1.37x |
| torus, 3072 B, local write skipped | E1 216, still above wire 178 | router-bound and faster (216.1) |

- **Main** serves at ~264, slower than the 248 wire at 4352 B, so it never enters this mode.
- **The Linear row (8.4)** has a v4 receiver faster than the wire at 4352 B but no slow mode. So the mode needs the ring
  or torus setting: deadlock-avoidance mode (first-level acks, the bubble rule, single-slot completions) or uniform
  saturation.
- **The mechanism is open.** "Something v4's receiver does when it has slack between packets costs link throughput."
  Candidates:
  - (a) more credit frames per packet when acks and completions do not batch;
  - (b) the single-slot completion credited a pass later on idle passes (H2; v4wt's bounded wait, -40, fits);
  - (c) first-level acks returning faster, which changes the bubble rule's injection/forward balance.

**Cheapest next A/B (no build; the variant exists): an RX-delay sweep on v4rxg3bd.**
- Run EXP_RX_DELAY 0 / 16 / 32 / 48 / 64 / 96 on the 8-ring-only row at 3584 and 4352 B, and the torus row at 4352 B.
- **H9 predicts** at 4352 B a minimum near the delay that brings E1 to the link time (~48-64 cycles; around 250-280),
  then a linear rise once E1 is slower than the wire.
- **What contradicts H9:** a monotonic fall, or a flat curve, beyond 32.

### 8.8 Hypothesis table after round two

| hypothesis | status |
| --- | --- |
| H1a the 8-chip rings set the row | confirmed (A/B 1) |
| H1b the 8-ring is credit-loop (latency) bound | **WRONG as stated** (A/B 7: the RX delay speeds it up; not a latency bound) |
| H1' specific to 1D ring routers | WRONG as stated (user; the torus steps too); refined by H9 |
| H2 single-slot completion timing | open; fits H9 as candidate (b); design back to the coordinator first (exit on a new packet) |
| H3 the local write is the size-dependent hop cost | confirmed on P150 (A/B 4); not the Galaxy cliff (A/B 4 Galaxy) |
| H4 slots / H5 4-chip rings / H6 Ethernet layer | contradicted (round one) |
| H9 slow mode when E1 is faster than the link (ring/torus mode only) | open; supported by 8.7's five rows; next: RX-delay sweep (no build) |

### 8.9 RX-delay sweep: setup and additions to 8.7's predictions (written 05:43 UTC, before the run)

**Setup.**
- v4rxg3bd (byte-identical to v4rxg3 at delay 0, §8.3), EXP_RX_DELAY 0 / 16 / 32 / 48 / 64 / 96.
- Rows: the 8-ring-only row at 3072 (control), 3584 and 4352 B (`galaxy/frgap/glx_frg_nsrx.yaml`), and the torus full
  ring at 3072 and 4352 B.
- One job, both passes reversed (a: 0 -> 96, b: 96 -> 0). Mock-vetted at delay 96.

8.7's prediction stands unchanged. These additions are written before the run:

- **At delay 96, the TX arm plays no part.** E0 is untouched in every arm, so each row should be E1-bound at 96 and
  run at about E1 service + 96 if H9 holds:
  - the 8-ring at about 192 + 96 ≈ 290 at every size;
  - the torus at about 246 + 96 ≈ 340.
  - At 3072 B the 32-cycle arm already showed a slope above 1 (+45.6), so allow +96 to +140 there.
- **3072 B control:** a plain rise from 191.9, no dip.
- **3584 B (wire 204, E1 ~192):** under H9 the slow mode (295) clears at the first delay that brings E1 past the
  wire, around 16. Expect the minimum at 16-32, near 210-240, then a rise.
- **4352 B:** the minimum near 48-64, around 250-280, as 8.7 says.
- **Torus at 4352 B (borderline, E1 ~246 vs wire 250):** no dip expected beyond a few cycles. Already at 32 it rose
  +20.8.

### 8.10 RX-delay sweep: results (Galaxy 169903, bh-glx-110-a01u14, v4rxg3bd, both passes reversed in one job)

| row | rx 0 | 16 | 32 | 48 | 64 | 96 | wire |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8-ring only, 3072 B (control) | 192.0 | 218.9 | 237.5 | 255.4 | 267.8 | 303.3 | 176 |
| 8-ring only, 3584 B | **300.7** | 219.0 | 237.5 | 255.4 | 267.8 | 303.3 | 204 |
| 8-ring only, 4352 B | **373.5** | **333.7** | **331.6** | 256.3 | 268.8 | 303.6 | 248 |
| torus, 3072 B | 246.4 | 272.1 | 290.0 | 307.7 | 320.4 | 355.4 | 178 |
| torus, 4352 B | **269.9** | 273.5 | 290.8 | 308.3 | 320.8 | 355.6 | 250 |

Passes a / b agree within 0.1 everywhere except the 8-ring at rx 0 (3584 B: 299.3 / 302.1; 4352 B: 373.8 / 373.1) and
4352 B rx 32 (332.3 / 330.9).

**H9 holds** (the predictions in 8.7 and 8.9 held):
- **3072 B control:** a plain rise, no dip; +111 at 96 (allowed: +96 to +140).
- **3584 B:** the slow mode (300.7) clears at the first step, 16 (219.0, predicted 210-240 at 16-32).
- **4352 B:** the minimum is at 48 (256.3, predicted 250-280 at 48-64), 117 below the base and 7.5 below main's 263.8.
- **Torus at 4352 B:** no dip beyond a few cycles. Its base sits 23.5 above its E1-bound line (246.4), and from 16 on
  it rides that line.
- **The switch point is where E1 service crosses the link time, on every row.** Once E1's service is slower than the
  link's per-packet time, every row collapses onto the same E1-bound line whatever the packet size. At 48 and above,
  the 8-ring's 3072, 3584 and 4352 B rows are equal within 1, and the torus's within 0.8.
  - 3584 B: E1 192 < 204 (slow) and 219 > 204 (out).
  - 4352 B: 237.5 at 32 < 248 (still slow: 331.6) and 255.4 at 48 > 248 (out).
  - Torus at 4352 B: 246.4 < 250 (mildly slow) and 272 > 250 (out).
- **At 96 the rows run at E1 service + 96 with the same slope above 1 as at 3072 B** (303 = 192 + 111). That matches
  the TX-arm-free note's "about E1 service + 96" within 5%.
- **H9 is now an A/B fact about the regime:** the slow mode exists exactly when v4's receiver serves faster than the
  link, in the ring and torus settings. Its mechanism is still a hypothesis.

### 8.11 Mechanism: candidates after the coordinator's code read, and the fairness check in existing data

**Code read (coordinator, 2026-11-03), summarized; this is not an A/B:**
- **(b), completion one pass late, is weak in the idle regime.** `credit_frame_due` is sticky. On an idle pass E1
  polls the proof and its own TX queue every few tens of cycles, so a starved E1 returns credits *sooner*. A bounded
  wait (A/B 2) would not change idle-mode behaviour.
- **(a), extra credit frames, is weak against main.** Main always sends 2 frames per packet; v4 sends 1 merged frame
  when busy and up to 2 when idle, never more than main.
- **What remains:** (c) faster first-level acks shifting the bubble balance, or another sender-side reaction to fast
  credits, such as E0's 1:1 worker/forward alternation (§3.3).

**The per-sender spread the coordinator asked for is not in the existing data.**
- The run logs' per-sender cycles all equal the global end time (end sync, section 2).
- The per-test CSV keeps one row (the slowest sender).
- The per-sender stream counters (slot waits, busy cycles) are logged at debug level only, and none of these runs had
  "worker-bound" warnings.
- No runs were added for this.

**The only existing proxy points against a worker-share shift on the 8-chip rings (samples, not an A/B).** The worker's
share of v4's E0 send-path samples on the 8-ring routers is the same in the slow mode and the router-bound mode: 0.147
at 4352 B (point 5, slow) vs 0.142 at 3072 B (P1, router-bound); main 0.158 vs 0.160. The 1:1 alternation of §3.3 shows
on the 4-chip rings only, in both routers and in both modes. So "the local worker takes more of the 8-ring link when
credits return fast" is not supported by the samples. It stays a hypothesis.

**Proposed separation tests (none run or built; designs to the coordinator first).**

1. **No-build counter probe** (a read-only script, no router change). Snapshot each 8-ring Ethernet core's TX-queue
   hardware counters twice, ~10 s apart, during the 8-ring row at 4352 B with rx 0 (slow) and rx 48 (fast mode).
   - Counters: TXQ0 and TXQ1 TRANSFER_CNT (exact per RESULTS §16; PKT_*_CNT undercount at ≥ 4000 B) and RESEND_CNT,
     at the offsets hang_capture.py uses.
   - It gives credit frames per data packet (TXQ1 transfers / (TXQ0 transfers / 2)) and link resends per packet in
     each mode. It decides (a) directly and checks an unlisted candidate: link-level resends in the slow mode.
   - Safety first: these registers have only been read from hung routers. A P150 ring row with and without the
     snapshot must show the same rate before any Galaxy use.
2. **For (c), an ack-timing switch** (variant; design for agreement). Accumulate a packet's first-level ack when it is
   handled (at `retire`), not when it arrives. E1's service time is unchanged; only the ack is later.
   - Prediction if (c): the 4352 B 8-ring slow mode clears, as with rx 48, while the 3072 B row stays within a few
     cycles.
3. **For E0 fairness, a forwarding-priority switch** (variant; design for agreement). The worker channel sends only
   when no forwarding channel holds a packet with room.
   - Prediction if the mechanism is the worker taking more of the link: the slow mode clears.
   - Under (c) alone it does not.

Order: 1 (its safety trial on the P150, then Galaxy), then 2 or 3 depending on what 1 shows.

### 8.12 Hypothesis table after the sweep

| hypothesis | status |
| --- | --- |
| H1a the 8-chip rings set the row | confirmed (A/B 1) |
| H9 slow mode exactly when E1 serves faster than the link (ring/torus only; not on the line) | **confirmed as a regime** (8.10, A/B; with A/B 4 and the Linear row) |
| H9 mechanism (a) more credit frames | weak by code read (never more than main's 2 per packet); counter probe would decide |
| H9 mechanism (b) completion a pass late | weak by code read (an idle E1 credits sooner); A/B 2 parked |
| H9 mechanism (c) faster first-level acks shift the bubble balance | open; ack-at-retire switch (design) |
| H9 mechanism, E0 worker/forward alternation | open; not supported on the 8-ring by samples (8.11); forwarding-priority switch (design) |
| H3 local write = size-dependent hop cost | confirmed on P150; not the Galaxy cliff |
| H1b latency bound, H1' 1D-only, H4, H5, H6 | wrong or contradicted (sections 5, 7.3, 8.8) |

### 8.13 Prior art: the throughput drop is a known bubble-flow-control behavior (main session, 2026-11-03; user pointer)

Chen, Wang, Pinkston, "Critical Bubble Scheme: An Efficient Implementation of Globally Aware Network Flow Control",
IPDPS 2011 (IEEE 6012872, doi 10.1109/IPDPS.2011.63):

- §II.C: "It is possible for some networks under certain traffic patterns to suffer from a throughput drop beyond the
  saturation point." The remedy is buffer-occupancy-based injection throttling, local (inject only below a channel
  occupancy threshold) or global (aggregate ring occupancy).
- §V.C (Fig. 13): with one bubble per ring, "when all channel buffers are full in the unidirectional ring except for
  the bubble, only one packet can make movement per cycle, resulting in a decreased throughput." "Best Local FC"
  (the best local free-buffer threshold) avoids it by keeping the network below saturation.
- Localized BFC, the scheme both our routers implement (two free slots in the receiving channel), checks only the
  receiving buffer.

**How this maps here (hypothesis until an A/B).** One ring hop holds two buffers in series: the receiver ring on E1 and
the next E0's forwarding channel. The bubble check reads only the first. When E1 is faster than the link, the queue
sits in the forwarding channel, so the check sees free slots while the ring fills. The ring then reaches the paper's
full-except-the-bubble state. Main's E1 is slower than a 200G link, so its queue stays in the receiver ring, where the
check sees it.
- Also pointed to by the user: Dai, Lu, Ma, Su, Li, "Bubble-Swap Flow Control", ACM TACO 22(1), 2025
  (doi 10.1145/3705316). Read from the abstract only; the full text is behind ACM's access check. It treats deadlock
  as a credit shortage and swaps buffers atomically between adjacent routers using local credit state, so no
  bubble is reserved. It reports +64% average throughput over bubble-based schemes, and -29% area / -26% power
  against CBS with credit reservation. Its target is buffer efficiency under adaptive routing. Whether it addresses
  the post-saturation drop is not known from the abstract.

### 8.14 Injection-throttle probe (user go, 2026-11-03): design, implementation, predictions (written ~06:55 UTC, before any run)

**Hypothesis under test (§8.13; a hypothesis until this A/B).** A bubble-flow-control jam:
- the bubble check reads only the far receiver ring;
- when E1 outruns the link, the queue moves into the next E0's forwarding channel;
- the ring fills and advances one bubble at a time.

**Design (agreed with the user; unchanged here).**
- **Scope:** only deadlock-avoidance (bubble) VCs, per sending direction.
  - The in-ring channel is the straight forwarding channel (not marked injection).
  - The injection group is the worker plus, on a torus, the turn channels (`sender_channel_is_traffic_injection_channel_all_`).
- **State:** one count per in-ring channel, "forwarded since the last injection". Each in-ring send adds one (saturating
  at K); any injection-group send resets it.
- **Gate:** an injection channel may send with ≥ 2 far free slots (unchanged) AND either:
  - the in-ring backlog < T (num_slots - (free-slots register & stream_count_mask), from the pass's grouped read), or
  - count ≥ K.
- It is a count, not a second send; the per-pass connection check is kept.

**Implementation (`variants/v4rxg3it` = v4rxg3 + the throttle).**
- **The switch:** TT_FABRIC_EXP_DRISC_DELAY (named arg EXP_DRISC_DELAY, reused as v4bdd did: v4 has no DRISC offload) =
  (1 << 16) | (T << 8) | K; unset or 0 = off.
  - The host only forwards this variable as a named arg, so there is no host change.
  - Arms: A = 65794 (T 1, K 2), B = 65798 (T 1, K 6), C = 66566 (T 4, K 6).
- **In `router_sender.hpp`:**
  - `channels_with_packets` can hand back the in-ring channel's register from the group read it already does.
  - `pass()` builds an `InjectionGate` (the backlog and a pointer to the count).
  - `visit_worker` and `visit_forwarding` add the gate for injection channels and update the count after a send.
  - The count lives in `Workers` as a `Present<>` member that is empty when off, so `pass()`'s signature and the
    single-ERISC caller are unchanged.
- **Scope note for the coordinator:** VC0 only. VC1 (multi-mesh) and VC2 (neighbour exchange) are untouched; no row
  here uses a bubble VC1. Every ring/torus router in the mock builds has exactly one VC0 in-ring channel:
  - 1D ring: ch1;
  - torus E/W routers: ch1, with turns ch2 and ch3;
  - torus N/S routers: ch3, with turns ch1 and ch2.
- **Galaxy tree:** `trees/v4rxg3it_1102`. Identity and size: 8.15.

**Arms.** base (off) / A / B / C, two passes in reversed order, each pass set on one host.

**Rows.**
- Galaxy:
  - the 8-ring only at 3072 / 3584 / 4352 B (`glx_frg_nsrx`);
  - the 4-ring only at 4352 B (`glx_frg_ew`);
  - the torus full ring at 3072 / 4352 B (`glx_frg_torus`);
  - short trips: ring 1-hop unicast and half-ring multicast at 3072 / 4352 B (`glx_frg_short`; half-ring is new on
    Galaxy).
- P150 (lb2):
  - ring 1-hop unicast and Unidir/Half/Ring multicast, write and scatter, at 3072 / 4352 B (`p150_it_ring`);
  - the XY torus rows on 2x4 and 4x2 (`torus/torusxy_{2x4,4x2}_sizes.yaml`).
- Validation first, per on-arm, data-checked with flow control on:
  - Galaxy `glx_frg_val`: full-ring multicast and all-to-all on Ring and on the XY torus;
  - P150 `p150_frg_val_ring` and `p150_frg_val_torus` (both torus shapes).

**Predictions** (the user's, recorded verbatim in substance):
- **A (T 1, K 2):** the 8-ring slow mode mostly stays (4352 B well above 300); the 4-ring rows improve or hold.
- **B and C:**
  - the 8-ring at 3584 / 4352 B drops to about 250-260, at or below main's 263.8; **3584 B: WRONG on magnitude** (job
    170016: B 225.7, C 228.7, below the range; 4352 B held: 254.4 / 254.4);
  - 3072 B moves only a few cycles;
  - the torus at 4352 B moves toward its E1 line (~246-255); **WRONG** (B moved only to 259.6, above the range; C went
    the other way, 269.9 -> 271.5).
- **Short-trip rows:** C equals base within noise; B may lose a little. **WRONG for the half-ring multicast at 4352 B:**
  it was in the slow mode too (base 332.2) and dropped to 253.7 on B and C. The 1-hop unicast and the 3072 B half
  ring held, as predicted.
- **What contradicts the hypothesis:** B or C leaving the 8-ring at 4352 B above ~320.
- **Also watched** (no prediction): starvation or hangs on the XY torus rows, where turns are throttled.

### 8.15 Injection throttle: identity with the switch off, code size with it on (mock compiles, 2026-11-03)

**First build: not identical.** The first version passed a gate argument through the visit functions and rewrote the
send test as `bool may_send = ...; if (may_send)`. With the switch off:
- the ring, torus and line images matched v4rxg3;
- every 2D-mesh E0 image differed (Galaxy GlxCheck2D 84 of 168, P150 exp_2d_4352 200 of 400).

The difference was only the operand order of 4 compare branches (`beq a1,a0` vs `beq a0,a1`, in visit_forwarding's
`busy` tests at router_sender.hpp:538). That is semantically equal and the same size, but not byte-identical. The cause is
GIMPLE shape: the new locals and statements shift GCC's SSA numbering, which orders commutative operands. The ring
images matched by chance.

**Rebuilt** so that the off path has v4rxg3's own statements:
- every addition sits in an `if constexpr` that is dependent on a template parameter, so it is not instantiated
  when the switch is off;
- the state lives in VC0's `FarRing` as an empty-when-off `Present<>` member (`InjectionThrottle`: the backlog and
  the count). The visit functions already receive the far ring, so no signature changes;
- the gate narrows `has_packet` at the top of the visit, so the send test line is unchanged;
- `pass()` keeps the original group-read call on the off path.

Semantics are unchanged from the first build.

**Identity, switch off (`frg_elfcmp.py`, v4rxg3_1102 vs v4rxg3it_1102 trees): byte-identical on every image.**
- Galaxy (frg_ns, frg_torus, frg_line, glx_check_2d): 648 / 648 same.
- P150 (p150_ring_mcast_sizes, torusxy_2x4_sizes, exp_2d_4352): 1384 / 1384 same.
- P150 4x2 torus (torusxy_4x2_sizes, 4x2 descriptor): 768 / 768 same.

**Code size, switch on** (E0 = active_erisc `.text`, vs off, per arm A / B / C; Galaxy frg_ns + frg_torus configs):
- 1D ring E0: +32 / +32 / +36 B;
- 2D torus E0: +88 / +96 / +108 B;
- E1 (subordinate): identical in every arm (the throttle is sender-only).

**Channel roles, from the named args of the on builds** (`SENDER_CH_n_IS_INJECTION`):
- 1D ring: worker ch0 gated; in-ring ch1.
- Torus E/W routers: ch0, ch2 and ch3 gated; in-ring ch1.
- Torus N/S routers: ch0, ch1 and ch2 gated; in-ring ch3.

### 8.16 Separate item (outside the probe): a flow-control full-ring row killed twice on the P150 LoudBoxes

**Do not run this row, or any flow-control full-ring row, on a LoudBox** (coordinator, 2026-11-03). The main_fw vs
v4rxg3 A/B on it is on hold; the coordinator is taking the risk question to the user.

**The row.** `consolidation/frgap/p150_frg_val_ring.yaml`, test FrgValFullRingMcast:
- Ring, num_links 1, `enable_flow_control: true`, data-checked (benchmark_mode off);
- ftype mcast, full_ring pattern, 4096 B, 2000 packets;
- ntype unicast_write and fused_atomic_inc;
- receiver allocation RoundRobin, max 4 configs per core.

Both kills came in its first case (size 4096, unicast_write, mcast). The row had never run on any router on P150
before.

**Kill 1:** lb2 170018, bh-lb-120-a08u28, label frgitv_A_ring (arm A, T 1 K 2).
- Process start 07:28:09.8, "Launching programs" 07:28:20.4, "Waiting for programs" 07:28:21.7.
- SIGKILL (exit 137) at 36 s (about 07:28:45, ~23 s into "Waiting for programs"). Nothing of ours sent it: run_cap's
  timeout is 600 s, lb2's 4200 s.

**a08u28's device loss**, in the same job after kill 1 (no reset in between):
- The next run (07:28:46) waited ~50 s on "ETH training" for device 2 cores (25,25) and (31,25), then aborted with
  `std::out_of_range` (`unordered_map::at`) right after UMD's "Starting devices in cluster completed".
- Five more runs aborted the same way within 3-5 s (exit 134/135), arms A, B and C.
- At 07:30:57 the switch-off pass-a run (frgita_base_ring) aborted in topology discovery: "Query mappings failed on
  device 2: No such device" (pci_device.cpp:450). PCI device 2 had vanished.
- At 07:32 (read-only srun): /dev/tenstorrent/0-7 were present, and no other user ran device processes. The node runs a
  root tt-device broker (tt_device_mcp) and tt_device_usage.py outside Slurm. dmesg and the broker logs are not
  readable (permission denied).
- /home had 41 GB free at job start and end (99% of 3.5 TB). No "No space" error in any log.

**Kill 2:** lb2 170023, bh-lb-120-a07u28, label frgitv_base_ring: **switch off**, v4rxg3's images byte-identical
(§8.15).
- The job started with `tt-smi -r` and a link check: 56 of 56 baseline links UP/PASS/400G.
- Process start 07:35:43.1, "Launching programs" 07:35:54.9, "Waiting for programs" 07:35:56.1.
- SIGKILL (exit 137) at 37 s (about 07:36:20, ~24 s into "Waiting for programs").
- The runner then stopped (no further run), as frg_it_p150_runs.sh now does on any run without a result line.

**The same yaml content passes on Galaxy.** Job 170016 (bh-glx-110-a06u02): glx_frg_val.yaml, which holds this test
plus ring all-to-all and XY-torus rows, passed 10 / 10 on each of arms A, B and C.

**WRONG: my first reading of kill 1.** I reported to the coordinator: "node or device fault on a08u28, not arm A". Kill 2
reproduced it on another node, after a fresh reset, with the switch off. The row (with v4rxg3 on the P150 ring), or
something it triggers on the LoudBoxes, is implicated, not a08u28.

**Main hangs on it too** (coordinator, 2026-11-03): a09u32 broker job 537, main's router as shipped, the same test.
- It sat in "Waiting for programs" for over 150 s; run_cap then killed it and captured the routers into
  `results/fcval_main_fw_hangcap`.
- v4rxg3 was not run there, and a09u32 was reset afterwards (health OK).
- So this is **not a v4 regression**: it is either the test on the P150 ring or something both routers share.

**Cause: open. Hypothesis, not tested:** the row hangs on the P150 cube ring, under both routers (main's hang above; v4's
SIGKILLs fit a hang).
- On the LoudBoxes something then kills the process and resets the chip after ~20 s of a stuck run. a09u32 did not:
  main sat there for 150 s. a08u28's ETH retraining and PCI loss would follow from that reset.
- Nothing here separates the test infrastructure from a shared router behavior, or names the killer.

**Probe impact.** The P150 validation now uses the rows v4rxg3 passes on P150:
- v4/val_ring_p150.yaml: 19 / 19 for v4rxg3 (val_v4rxg3_ring);
- torus/val_torus_xy_p150.yaml on 2x4 and 4x2: 12 / 12 each (txyvc_v4rxg3_*).

Galaxy keeps the flow-control rows, so the throttle's flow-control coverage is there. The two flow-control P150 yamls
carry a do-not-run header.

### 8.17 Injection throttle: Galaxy results (job 170016, bh-glx-110-a06u02, 2026-11-03 07:24-07:45)

**Run.** One job, tree v4rxg3it_1102. Order:
1. check;
2. validation: glx_frg_val per on-arm (data-checked, flow control on);
3. pass a: base, A, B, C;
4. pass b: C, B, A, base.

36 steps, all exit 0, no timeouts. frg_val: 10 / 10 on each of A, B, C.

Cycles per packet (pass a / pass b, mean); base = the switch off (v4rxg3's images):

| row | size | base | A (T1 K2) | B (T1 K6) | C (T4 K6) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8-ring full-ring mcast (NS) | 3072 | 191.9 | 191.9 | 191.8 | 191.8 |
| 8-ring full-ring mcast (NS) | 3584 | 301.3 | 247.7 | 225.7 | 228.7 |
| 8-ring full-ring mcast (NS) | 4352 | 371.4 | 334.7 | **254.4** | **254.4** |
| 4-ring full-ring mcast (EW) | 4352 | 339.3 | 255.0 | 255.0 | 255.0 |
| XY torus full-ring mcast | 3072 | 246.4 | 246.4 | 246.3 | 246.5 |
| XY torus full-ring mcast | 4352 | 269.9 | 262.8 | 259.6 | 271.5 |
| ring half-ring mcast | 3072 | 186.2 | 186.6 | 186.5 | 186.3 |
| ring half-ring mcast | 4352 | 332.2 | 274.9 | 253.7 | 253.7 |
| ring 1-hop unicast | 3072 | 177.0 | 177.0 | 177.0 | 177.0 |
| ring 1-hop unicast | 4352 | 249.5 | 249.5 | 249.5 | 249.5 |

Passes agree within 2 cycles, except A at 3584 B (245.0 / 250.5). For reference: main's 8-ring 4352 B row is 263.8, and
the 1D wire at 4352 B is ~248.

**Against the predictions (§8.14; WRONG marks are there):**
- A: the 8-ring slow mode mostly stays (334.7), and the 4-ring improves (255.0). Held.
- B and C: the 8-ring 4352 B is 254.4, at the wire plus 6 and 9.4 below main. Held.
- B and C at 8-ring 3584 B: 225.7 / 228.7, below the predicted 250-260. WRONG on magnitude.
- 3072 B: unchanged. Held.
- Torus 4352 B: B moved 10 cycles but stays above the predicted E1 line (259.6); C lost 1.6. WRONG.
- Short trips: 1-hop unicast is identical in every arm, and the 3072 B half ring is within 0.4. The 4352 B half-ring
  multicast was itself in the slow mode (332.2) and drops to 253.7 on B and C. WRONG for that row.
- **The contradiction test** (B or C leaving the 8-ring 4352 B above ~320) **did not occur.**

**What the A/B establishes (CLAUDE.md attribution rules).** The arms change one thing: E0's admission of VC0 injection
sends while the in-ring channel has a backlog. That change alone removes the slow mode:
- every deadlock-avoidance row at 3584-4352 B: the 8-ring, the 4-ring, the half ring, and part of the torus;
- with no change where there was no slow mode (3072 B, 1-hop unicast).

So the slow mode is caused by how E0 admits injection sends against a backlogged in-ring channel.

The finer mechanism stays a hypothesis. The bubble-flow-control jam (§8.13) and plain E0 worker/forward unfairness both
fit this result, and this A/B does not separate them.

The K dependence matches the design's sizing, a floor of N-2 in-ring sends between injections:
- K 2 fixes the 4-ring (N-2 = 2) but only part of the 8-ring (334.7);
- K 6 (N-2 = 6) fixes the 8-ring.

T (1 vs 4) is neutral on the 1D rings at 4352 B. On the torus T 4 gives back the gain (271.5 vs B's 259.6); T 4 admits
turns more often there.

**Starvation and hang signs: none seen.**
- No step timed out, and validation passed on all three arms.
- The metric is the slowest sender's time, so a starved injector would make a row slower. No row got slower except
  torus C, by 1.6 cycles. There is no per-sender data to look further (the `_tests.csv` holds one row per test).

### 8.18 Injection throttle: P150 results (lb2 170034, bh-lb-120-a07u28, 2026-11-03 07:50-07:59)

**Run.**
- Host check first (lb2 170032): 8 chips, wiring identical to a09u32, 48 / 48 mesh link ends UP/PASS/400G.
- Then one job: reset and link check, validation per arm (base, A, B, C), pass a (base, A, B, C), pass b (C, B, A,
  base).
- Validation used the proven rows (§8.16): v4/val_ring_p150.yaml 19 / 19, val_torus_xy_p150.yaml 12 / 12 on 2x4 and
  12 / 12 on 4x2, in **every arm**.
- 24 benchmark runs, all exit 0, no hang. Results: `results/frgit{a,b}_{base,A,B,C}_{ring,t24,t42}`; table tool
  `frg_tables.py p150it results`.

**Ring rows** (cycles/packet, mean of the two passes; passes agree within 0.3 except where shown; 400G):

| row | size | ntype | base | A | B | C |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| RingMulticast (all-to-all mcast) | 3072 | write | 141.1 | 143.5 | **162.6** | **156.2** |
| RingMulticast | 4352 | write | 147.6 | 154.0 | **167.7** | 155.4 |
| RingMulticast | 3072 / 4352 | scatter | 180.2 / 180.5 | 180.2 / 180.3 | 181.3 / 184.1 | 180.2 / 180.7 |
| HalfRingMulticast | 3072 | write | 171.9 | 175.2 | 177.3 | 171.9 |
| HalfRingMulticast | 4352 | write | 173.0 | 176.3 | 177.5 | 172.7 |
| HalfRingMulticast | 3072 / 4352 | scatter | 212.4 / 225.4 | 213.6 / 226.1 | 213.3 / 225.8 | 212.4 / 225.4 |
| UnidirRingMulticast | 3072 / 4352 | write | 198.9 / 219.4 | 198.7 / 219.3 | 198.7 / 219.3 | 198.7 / 219.3 |
| UnidirRingMulticast | 3072 / 4352 | scatter | 239.5 / 265.2 | 239.9 / 264.8 | 239.9 / 264.8 | 240.0 / 264.9 |
| Ring1HopEW (unicast) | 3072 / 4352 | write | 141.2 / 140.9 | 140.7 / 140.9 | 140.8 / 140.9 | 140.7 / 140.9 |
| Ring1HopEW | 3072 / 4352 | scatter | 179.6 / 179.8 | 179.6 / 180.3 | 179.7 / 180.2 | 179.8 / 180.5 |

**XY torus rows** (5 sizes, 256-4352 B), delta range against base (cycles/packet):

| shape | rows | base | A | B | C |
| --- | --- | ---: | ---: | ---: | ---: |
| 2x4 | unicast write (1-hop, bidir, 3-hop, wrap; 30) | 142.6-148.4 | +2.9..+10.5 | +2.7..+10.3 | +3.9..+11.6 |
| 2x4 | unicast scatter (30) | 181.8-187.9 | ±0.6 | ±0.6 | ±0.7 |
| 2x4 | TorusXYMulticast write (5) | 126.1-126.3 | +15.0..+18.0 | +21.2..+22.1 | +0.4..+0.8 |
| 2x4 | other mcast (RowMcast, scatter; 15) | | ±0.5 | ±0.5 | ±0.6 |
| 4x2 | unicast write (30) | 146.4-150.3 | +1.0..+6.3 | +1.0..+6.4 | +1.9..+7.2 |
| 4x2 | unicast scatter (30) | 185.7-189.9 | ±0.5 | ±0.4 | ±0.5 |
| 4x2 | Multicast write / scatter (10) | 187.5-298.3 | -5.3..+4.9 | -4.8..+2.9 | -4.0..+4.1 |

**Reading.**
- P150 is not expected to have a slow mode to remove: at 400G v4's E1 is slower than the wire (H9, §8.7). And no row
  improved beyond 5 cycles (4x2 multicast write at 1024 B: -5.3 on A).
- **It does cost.**
  - Multicast rows with a busy ring lose with a low T: RingMulticast write +15 to +21 on B, +8 to +15 on C;
    TorusXYMulticast 2x4 +15 to +22 on A and B.
  - Hypothesis, not tested: on P150 the forwarding channel normally holds a backlog (E1-bound), so T 1 holds the
    injectors back on almost every pass. The metric is the slowest sender, which here is a throttled injector: the
    mild form of the starvation the user asked to watch for. No row hung or stalled.
- **The torus unicast write rows lose 1-7 cycles in every arm** (3-hop EW up to +11.6), including single-flow rows
  where the in-ring channel is empty and the gate is open.
  - The scatter rows on the same routers do not move, and E1's images are byte-identical across arms (§8.15).
  - So the E0-only change moved these rows: by this A/B, E0 is on their bound.
  - Why the added E0 work costs ~5 cycles per packet on the torus but nothing on the 1D ring 1-hop row (140.9 in
    every arm) is a hypothesis: the torus gates three channels and keeps more throttle state live.
- T 4 (C) is the gentlest arm on P150 and on the 2x4 multicast, but it still loses on the ring all-to-all and on every
  torus unicast write row.

### 8.19 Injection throttle: conclusion (2026-11-03)

**The hypothesis holds where it was posed.** On Galaxy (200G, v4's E1 faster than the wire) the slow mode is caused by
how E0 admits injection sends against a backlogged in-ring channel. Changing only that admission (B, C) takes:
- the 8-ring full-ring multicast at 4352 B from 371.4 to 254.4 (main: 263.8);
- the 4-ring from 339.3 to 255.0, and the half ring from 332.2 to 253.7.

The contradiction test did not occur. The finer mechanism (bubble jam vs plain worker/forward unfairness) stays a
hypothesis.

**The throttle as built is not a generic fix.** On P150 it costs:
- up to +22 cycles (15-17%) on multicast rows with a busy ring;
- 1-11 cycles on every torus unicast write row, in every arm.

On Galaxy, torus C loses 1.6. Per the generic-over-narrow rule this does not pass. A shipping form would need:
- to act only when the slow mode can form (injection actually displacing an in-ring packet that is waiting), not
  whenever the forwarding channel holds a backlog;
- to cost nothing per pass where the gate is open.

That is a design question for the user (feedback-discuss-design-before-variant). Nothing is built or queued.

### 8.20 Ring-jam fix: design (2026-11-03)

The design is in consolidation/RING_JAM_FIX.md, on v4txa11, for the user's review. Nothing is built or queued. In
short:
- **The probe's open-gate cost:** read from its P150 run images, it is per-pass gate work plus one more loop-carried
  register. That register makes the worker's send spill three values on the torus E0 (headroom check fails).
- **Mechanism (hypothesis, from v4txa11's code):** per-pass channel round robin shares the link 1:1 by channel when
  the far ring is not the constraint. The bubble rule arbitrates only when it is (H9's regime split).
- **Candidates:** C1, a contest gate with its state in memory; C2, an in-ring burst (stateless weighted round robin,
  recommended). Both use K = ring length minus 2 from the host and a remote-room test. A far-credit refusal trigger
  is analysed and not recommended: it fires in the regime without a jam.
- **First A/B:** a telemetry-only trigger check.
