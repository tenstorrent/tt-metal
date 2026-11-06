# Transaction-ID allocation that follows the slot counts (2026-11-04)

Status:
- Design user-approved and relayed by the coordinator on 2026-11-04; built as specified, with no deviation.
- Built and checked offline: the host's rule, the router variant v4txa11id, and an offline compile and code-size
  check on captured router configs.
- Not in the host library yet: `host_rebuild_tridalloc.sh`, a broker job through the coordinator, on top of the dyn
  slots (broker 648).
- No device data yet: the plan is in section 6.

## 1. The rule

It is computed per router. All of a router's receiver channels run on one ERISC (E1 with two ERISCs, E0 with one) and
share that NoC's 16 IDs (NOC_MAX_TRANSACTION_ID 0xF).

1. Per receiver channel c:
   - group size G_c = 1 under first-level acks, else 1 / 2 / 4 for fewer than 8 / 8-15 / 16 or more slots. This is
     today's router rule (completion_group_size_for and first_level_ack).
   - Groups g_c = ceil(slots_c / G_c), so a partial last group counts.
2. If the g_c sum to at most 16: ids_c = g_c, and a slot's ID is start_c + group. The mapping is the identity, with no
   mask, and the count needn't be a power of 2.
3. Otherwise the largest count shrinks until the total fits, keeping the largest count that fits:
   - a count that is not a power of 2 is first rounded down to one (20 -> 16);
   - a power of 2 is halved (32 -> 16, 16 -> 8);
   - so a lone ring receiver of 20 groups gets 16 IDs, not 8, and 20 groups next to a VC1 receiver of 4 go 20 -> 16 -> 8.
   - A channel with ids_c < g_c shares its IDs: ID = start_c + (group & (ids_c - 1)).
   - Every channel keeps at least 1 ID; leftover IDs stay unused.
   - Correction 2026-11-04 (coordinator): the first version took floor_pow2(count / 2), which sent 20 to 8. WRONG for
     non-power-of-2 counts; fixed in fabric_trid_allocator.cpp and the trid_offline_build.py mirror. No configuration
     in §4 or in the offline compile changes: their shrinking counts are powers of 2 (32a), the rest fit as is.
4. Starts are consecutive from 0, in receiver-channel order.

Correctness: a group's proof waits until every write with its ID has left the NIU, so sharing an ID only makes credits
later, never early.

## 2. Ownership: the host

`tt_metal/fabric/builder/fabric_trid_allocator.{hpp,cpp}` is pure and standard-library only, next to the slot
allocator.
- `assign_transaction_ids({slots, first_level_ack} per receiver channel) -> {G, groups, ids, start, reason}`.
- `uniform_transaction_ids(...)` is today's rule, as a comparison: 16 for a lone channel without deadlock avoidance,
  else 8 with deadlock avoidance, else 4, each from c x ids.

`erisc_datamover_builder.cpp` (`variants/host_dynslots/2_trid_alloc.patch`) runs it per router once the slot counts are
final, from the tier table or the dynamic allocator:
- it reads the slot counts from the router's allocator and first-level acks from its ENABLE_FIRST_LEVEL_ACK_VC<n>
  args;
- it emits RX_CH_<c>_COMPLETION_GROUP_SIZE / _NUM_TRIDS / _TRID_START for c = 0..2 (unused channels: 1 / 0 / 0);
- with `TT_FABRIC_EXP_TRID_ALLOC=1` it uses the rule, otherwise today's uniform rule, so every run keeps today's
  behaviour; main's router and v4txa11 don't read the args;
- with `TT_FABRIC_EXP_LOG_SLOTS=1` it logs every channel's values with their reason.

## 3. The router (variants/v4txa11id = v4txa11 plus per-channel IDs; no study refs in its code)

- `fabric_erisc_router_ct_args.hpp`:
  - RX_CH_COMPLETION_GROUP_SIZE, RX_CH_NUM_TRIDS and RX_CH_TRID_STARTS come from the named args, in place of
    NUM_TRANSACTION_IDS and initialize_receiver_channel_trid_starts.
  - static_assert: every used channel has G in {1,2,4}, at least 1 ID, consecutive ranges, and a total ≤ 16.
- `router_receiver.hpp`:
  - completion_group_size comes from the host; static_assert G == 1 under first-level acks.
  - CompletionGroupTrids<G, FIRST, IDS, GROUPS> maps by identity when IDS ≥ GROUPS, else by mask; static_assert a
    power of 2 where it shares.
  - completion_group_size_for is removed.
- `router_init.hpp`: teardown waits on each channel's own range (WriteTransactionIdTracker<slots, ids, start>).
- `fabric_edm_packet_transmission.hpp`: flush_write_to_noc_pipeline flushes the channel's own range on every per-ID
  path (deadlock avoidance, non-Blackhole, two NoCs).
  - This fixes main's flush-range bug: the non-deadlock-avoidance loop ran over IDs 0..N-1 whatever the channel
    (MAINLINE_PLAN §2.1).
  - Blackhole's single-NoC fast path (one global flush) is unchanged.
- Audit, by grep: no NUM_TRANSACTION_IDS, completion_group_size_for or initialize_receiver_channel_trid_starts is left
  in the variant. Every ID a receiver issues comes from CompletionTrids::of_slot: local writes, downstream forwards, the
  UDM relay forward, and the proof.

## 4. Offline table (`slotalloc/slot_alloc_table.sh --trids --sizes 4352,3072`)

Notation: G/ids@start per receiver channel (VC0 | VC1 | VC2); "!" marks groups sharing IDs; "a" marks first-level
acks. The full table covers 1D, ring, 2D, both tori, the VC1 split, VC1 first, VC2 on, three VCs and a multi-mesh
torus, for main's tier and the dynamic slots, at 4352 and 3072 B.

| config (router) | slots | today | new |
| --- | --- | --- | --- |
| 2D mesh, 16 | tier = dyn 4352 | 4/16@0 | 4/4@0 |
| 1D line, 32 | 4352 | 4/16@0 | 4/8@0 |
| 1D ring, 32a (deadlock avoidance) | any | 1/8@0! | 1/16@0! |
| torus, wrapped axis, 16a | 4352 | 1/8@0! | 1/16@0 |
| torus, unwrapped axis, 16 | 4352 | 4/16@0 | 4/4@0 |
| VC1 split | dyn 4352: 16 / 4 | 4/4@0 / 1/4@4 | 4/4@0 / 1/4@4 |
| VC1 split | dyn 3072: 16 / 6 (before the VC0 receiver floor) | 4/4@0 / 1/4@4! | 4/4@0 / 1/6@4 |
| VC1 split | dyn 3072 with the floor: 32 / 6 | 4/4@0! / 1/4@4! | 4/8@0 / 1/6@8 |
| VC2 on | tier 4352: 11 / 4 | 2/4@0! / 1/4@4 | 2/6@0 / 1/4@6 |
| VC2 on | dyn 4352: 16 / 2 | 4/4@0 / 1/4@4 | 4/4@0 / 1/2@4 |
| multi-mesh torus | tier: 8a / 4 | 1/8@0 / 1/8@8 | 1/8@0 / 1/4@8 |
| multi-mesh torus | dyn 3072: 16a / 6 (32a / 6 with the floor: same IDs) | 1/8@0! / 1/8@8 | 1/8@0! / 1/6@8 |

The dyn rows follow the slot allocator, so the 2026-11-04 VC0 receiver floor (SLOT_ALLOCATOR.md section 10) moved
the 3072 B dyn VC0 receivers from 16 to 32: 2D 4/8@0 (8 IDs), three VCs 4/8 + 1/5 + 1/3 (all 16). The offline
compile below captured the pre-floor 16 | 6 VC1 split; the post-rebuild vet recaptures it.

The table first runs unit checks of the rule and stops on a failure (2026-11-04, all pass):

| case | new |
| --- | --- |
| 2D lone 16 | 4/4@0 |
| 1D 32 | 4/8@0 |
| ring 32a: 32 -> 16 (halve a power of 2) | 1/16@0! |
| ring 20a: 20 -> 16 (round down, not 8) | 1/16@0! |
| ring 24a: 24 -> 16 | 1/16@0! |
| torus 16a + VC1 4: 16 -> 8 | 1/8@0! / 1/4@8 |
| 20a + 4: 20 -> 16 -> 8 | 1/8@0! / 1/4@8 |
| 12a + 8a: 12 -> 8 | 1/8@0! / 1/8@8 |
| three VCs 16 / 6 / 2: fits as is | 4/4@0 / 1/6@4 / 1/2@10 |

The coordinator's expected values all hold:
- 2D lone 16 -> 4;
- 1D 32 -> 8;
- lone ring with first-level acks 32 -> 16 (today 8);
- torus 16 + VC1 4 -> 8 + 4;
- three VCs at the dynamic slots: 4 + 4 + 2 at 4352 B. That row's dynamic VC1 receiver is 4, not 6; the 16 / 6 / 2
  example gives 4 + 6 + 2.

## 5. Checks before any device run

1. **Offline compile (`slotalloc/trid_offline_build.py`).** It reruns the router configs a mock captured (compile
   commands logged) with v4txa11id (tree /data/.../v4txa11id_pre) and the new args added to each generated header.
   - Configs: the P150 1D ring at 3072 B (32a: shared 16 IDs) and the Galaxy VC1 split at 3072 B dyn (16 / 6: 4 + 6).
   - Result: all 152 ELFs build (static_asserts hold) under both rules.

   Code size, largest per ERISC:

   | config | arm | E0 | E1 |
   | --- | --- | ---: | ---: |
   | ring | v4txa11 | 8,812 | 6,056 |
   | ring | v4txa11id, either rule | 8,812 | 6,056 |
   | VC1 split | v4txa11 | 12,072 | 10,072 |
   | VC1 split | v4txa11id, either rule | 12,072 | 10,104 (+32) |

   Least free stack is unchanged (E0 1,684 / 2,052, E1 4,252 / 4,496).
2. **VC1 + VC2 at -O3.** v4txa11's router does not compile VC2 with VC1 at any slot or ID count
   (router_common.hpp: "VC2 is supported as neighbour exchange on 2D single-mesh routers"), so there is no three-VC
   router to size.
3. **After the host rebuild:**
   - a new Galaxy base and a v4txa11id tree;
   - the mock vet, which must log the table's values (`receiver transaction IDs (...)` lines);
   - the code-size check over every configuration, with the switch on and off.

Mock vet on the rebuilt host (2026-11-04, `slotalloc/trid_vet.sh`, tree trees/v4txa11id_1104 on base_1104trid;
out/trid_vet_20261104.txt): 8 configurations x switch on / off, 728 router configs, every one reaching launch.
`trid_vet_check.py` reads each config's emitted RX_CH_* args against the rule for its own receivers (from its CT
args), on both ERISCs: 0 mismatches. On: 2D 16 -> 4/4, 1D 32 -> 4/8, ring 32a -> 1/16, Y torus 16 -> 4/4 and
16a -> 1/16, XY torus 16a -> 1/16, VC1 split and VC2 on 16 | 4 -> 4/4 | 1/4, the VC1 split with the floor at 3072 B
32 | 6 -> 4/8 | 1/6 (the recapture). Off: today's uniform 16 / 8 / 4 for the same configs. Largest code E0 12,108,
E1 10,160 (E0+E1 22,268 of 26,624), least free stack E0 1,684, E1 4,236; on and off the same (the counts are template
parameters, the code paths the same).

## 6. Device plan (commands through the coordinator)

1. **P150 validation**, v4txa11id with `TT_FABRIC_EXP_TRID_ALLOC=1` (`slotalloc/trid_val_p150_runs.sh a|b|all`;
   two-ERISC runs must log the assigned IDs and no uniform-rule line; the host logs them for E1 only, so one-ERISC
   and UDM runs are not checked for them):
   - mesh sanity, the 1D sanity, val_ring_p150 and the XY-torus validation;
   - the VC1 and VC1 + VC2 split (VC2 with VC1 is unsupported on v4: noted above);
   - the inline-write and flushed-atomic gtests;
   - UDM unit and nightly.
   **Done 2026-11-04, all passed** (broker 661 part a, 662 part b; logs fabric_exp/results/tridv_*):

   | run | result | IDs logged (E1, per router) |
   | --- | --- | --- |
   | mesh sanity | 96/96 | 80 assigned, 0 uniform: 4/4 (2D) and 4/8 (its 1D tests) |
   | val1d_m1 | 143/143 | 80 assigned, 0 uniform: 4/4 and 4/8 |
   | val_ring_p150 | 19/19 | 40 assigned, 0 uniform: 1/16 |
   | val_torus_y_p150 | 12/12 | 40 assigned, 0 uniform: 1/16 (wrapped) and 4/4 |
   | val_torus_xy_p150 | 12/12 | 48 assigned, 0 uniform: 1/16 |
   | mesh sanity, one ERISC | 96/96 | not logged on one ERISC |
   | val_ring_p150, one ERISC | 19/19 | not logged on one ERISC |
   | VC1 split (vc1_alloc, two ranks) | 50/50 finishes, 0 errors | 96 assigned, 0 uniform: 4/4 + 1/4@4 |
   | inline write + fused atomic, 1D | 2/2 | 40 assigned, 0 uniform: 4/8 |
   | inline write + fused atomic, 2D | 2/2 | 40 assigned, 0 uniform: 4/4 |
   | UDM unit / nightly (one ERISC) | 8/8, 68/68 | not logged on one ERISC |

2. **A/B: v4txa11 against v4txa11id with the switch on**, two passes reversed, one host
   (`slotalloc/trid_ab_p150_runs.sh a|b`, one broker job per pass; a third router, v4txa11id with the switch off,
   splits the code change from the ID counts):
   - ring and torus multicast and jam-family rows (lone ring 8 -> 16 IDs);
   - VC1 rows at the dynamic slots (VC1 4 -> 6-7);
   - census rows where counts drop (2D 16 -> 4, 1D 16 -> 8), expected neutral.

   **Results** (broker 671 pass a, 672 pass b, a09u32, 2026-11-04; every run passed, every run's logged IDs its
   router's; out/trid_ab_p150_671_672.md). Change in cycles/packet against a11, median over each row set; both passes
   agree within 0.1 point:

   | rows (IDs a11 -> id) | idoff (code change only) | id (code + IDs) | read |
   | --- | --- | --- | --- |
   | ring, p150_it_ring (32a: 8 -> 16) | 0.0% | 0.0% (all 16 within 0.1%) | no effect from 16 IDs |
   | XY torus 2x4 (16a: 8 -> 16) | 0.0% | -0.5% (all 80 rows -0.8..+0.2%) | small gain from the IDs |
   | XY torus 4x2 (16a: 8 -> 16) | 0.0% | +1.0% (N/S rows +1.0..+1.5%, ~+2 cycles; worst +2.1%) | loss from the IDs |
   | VC1 split 3072 B, floor off (VC1 4 -> 6) | +0.1% | +0.1% (max +0.75%, = idoff) | no effect from the IDs |
   | VC1 split with the floor (VC0 4 -> 8, VC1 4 -> 6) | 0.0% | +0.1%; 2-hop merge +1.0%, multicast +1.5% at 3072 B | small loss from the IDs |
   | census 2D (16 -> 4) | 0.0% | 0.0% (max +0.3%) | neutral, as expected |
   | census 1D (16 -> 8) | +0.3% (max +1.0%) | +0.3% (= idoff) | the code change, not the IDs |

   - Attribution: idoff differs from a11 only by the code (same IDs), id from idoff only by the ID counts, so the
     columns split the two (one host, two passes reversed).
   - The rule does not pay: no row gains more than 0.8%, the 4x2 torus's N/S rows lose ~2 cycles/packet, and the
     lone ring (the case it was for) is unchanged. Why the 4x2 N/S rows lose while the 2x4 X-ring rows gain is open
     (hypothesis only, not tested: the identity mapping with 16 IDs against the 8-ID mask on those routers' NoC).
3. **Galaxy ring and torus rows** once the P150 is clean, within the cap: `slotalloc/trid_glx.sbatch`, a copy of
   galaxy/frs_glx_frgap.sbatch with configs a11 (v4txa11_1104, uniform IDs) and a11id (v4txa11id_1104 with the
   switch); each run's ID lines are checked. frg_val for both, then ring sizes, full-ring multicast, torus full-ring,
   short trips and XY-torus sizes; two passes with the arms reversed, in one job. Submitted as Slurm 172183
   (2026-11-04).

   **Results** (Slurm 172183, bh-glx-110-a08u20, 03:20-03:48; results /data/aagarwal/frs-glx/jobs/172183/results):
   frg_val passed on both arms (10/10 each); every run's IDs its arm's (ring routers 1/8 uniform, 1/16 with the rule).
   a11id against a11, cycles/packet, both passes:

   | suite | rows | median | range | read |
   | --- | ---: | ---: | --- | --- |
   | ring sizes | 10 | 0.0% | -0.3..+0.04% | no effect |
   | full-ring multicast | 2 | -0.4% / +0.3% (pass a / b) | -0.9..+0.7% | within pass spread |
   | torus full-ring multicast | 2 | +0.5% | 0.0..+1.1% (3072 B row, both passes) | small loss |
   | short trips | 4 | 0.0% | 0.0..+0.25% | no effect |
   | XY-torus sizes | 30 | 0.0% | -0.8..+1.5% | the N/S rows lose: 1-hop N/S, bidirectional, wrap N/S +1.3..+1.5% (147.7 -> 149.8) |

   The Galaxy N/S torus loss matches the P150 4x2 torus's N/S rows (147.6 -> 149.7), where the idoff arm puts it on
   the ID counts, not the code. On the Galaxy there is no idoff arm, so there it is a hypothesis by analogy. No suite
   gains.

4. **Credit-cadence A/B** (user go, 2026-11-04; `slotalloc/cc/`, no router build): why 16 IDs lose ~2
   cycles/packet on the torus N/S rows.
   - Hypothesis A (credit cadence): with 8 IDs over 16 G1 slots a slot's proof also waits for the
     aliased slot 8 later, so credits go back later and batched (fewer credit frames); with 16 IDs each proof passes
     as soon as its own writes leave (more credit frames), which compete with data on the bidirectional N/S links.
     **WRONG (broker 705, below): the credit frames per data packet are the same with 8 and 16 IDs.**
   - Hypothesis B (occupancy), not tested yet: earlier credits let upstream inject more, raising ring occupancy.
   - Measure: `txq_snap.py` reads every connected Ethernet core's TXQ0 / TXQ1 TRANSFER_CNT and RESEND_CNT once a
     second while the row runs (tt-exalens safe mode, opened after "Running Test:", as pc_sampler.py; NoC reads only).
     `cc_report.py` takes each data link's steady window and gives credit transfers (the peer's TXQ1) per data packet,
     and k = data transfers per packet from the test's rate (expected 2: payload + pointer update, RESULTS §16).
   - Safety first: the same row and arm (idoff, 1-hop N/S 4352 B) without and with the snapshot loop (plain, snap,
     snap, plain); the counts are used only if the rate is unchanged within noise.
   - Rows: 1-hop N/S write, its bidirectional pair, wrap 1-hop N/S on the 4x2 XY torus at 4352 and 3072 B,
     120M packets per sender (~13 s each). Arms idoff and id on v4txa11id, two passes reversed, one broker job
     (`credit_cadence_runs.sh all`, 28 runs, ~14 min estimated).
   - Dry runs: the six yamls mock-vetted (three reached launch on the mock P150 4x2 torus, 16 IDs logged with the
     switch); the runner's control flow with stubs (28 runs, 12 with the switch); cc_report.py on synthetic data.
   - Reading: credits per packet higher with 16 IDs -> A holds (follow-up: a credit-pacing rule that keeps the ID
     rule; design to the user, no variant). Equal -> A is WRONG; then B with round-sampled far-ring room (the v4txa11k
     method). The VC1-with-floor row is left out (two ranks; it would not fit the same job).

   **Results** (2026-11-04, a09u32 broker: safety broker 704, A/B broker 705, `credit_cadence_runs.sh safety` / `ab`;
   results fabric_exp/results/ccs_*, cca_* (pass a), ccb_* (pass b); `cc_report.py`, reported 2026-11-04 evening):
   - Safety: idoff 1-hop N/S 4352 B ran 147.80 / 147.80 cycles/packet without the snapshot loop and 147.83 / 147.80
     with it (+0.02). The rate is unchanged within noise, so the counts are usable.
   - Every run: one steady window of 12-13 s per data link (14-16 snapshots, no read errors). Every row's links
     appear (the bidirectional rows' both directions). TXQ0 and TXQ1 resends 0. Each run's routers logged their arm's
     IDs (RX_CH_0 G 1 with 8 IDs on idoff, 16 on id).
   - Per row, mean over the two passes and the row's data links:

     | row (4x2 XY torus) | cycles/packet idoff / id | credits per data packet idoff / id | k idoff / id |
     | --- | --- | --- | --- |
     | 1-hop N/S, 4352 B | 147.78 / 149.97 | 1.000 / 1.000 | 2.00 / 2.00 |
     | 1-hop N/S, 3072 B | 147.93 / 149.90 | 1.000 / 1.000 | 2.00 / 2.00 |
     | bidirectional N/S, 4352 B | 148.12 / 150.17 | 1.000 / 0.998 | 2.00 / 2.00 |
     | bidirectional N/S, 3072 B | 147.73 / 149.79 | 1.000 / 1.000 | 2.00 / 2.00 |
     | wrap 1-hop N/S, 4352 B | 147.76 / 149.99 | 1.000 / 1.000 | 2.00 / 2.00 |
     | wrap 1-hop N/S, 3072 B | 147.94 / 149.93 | 1.000 / 1.000 | 2.00 / 2.00 |

   - The loss reproduces: 16 IDs run 2.0-2.2 cycles/packet slower than 8 on every row, in both passes.
   - k = 2.00 data transfers per packet on both arms (payload + pointer update, as expected from RESULTS §16).
   - Verdict by the reading rule: credits per data packet are equal (1.000 against 1.000; the one 0.998 is pass a's
     bidirectional 4352 B run, whose k also reads 1.993, so its credits per data transfer stay at 0.5). **Hypothesis
     A is WRONG:** 16 IDs do not send more credit frames, so extra credit traffic on the N/S links does not explain
     the ~2 cycles.
   - What the counts do not show: when each credit leaves (they count frames, not timing). Earlier credits are
     hypothesis B's premise, which is next (round-sampled far-ring room, the v4txa11k method). Not run; it needs a
     go and a design. The cause of the ~2-cycle loss stays open (hypothesis B untested).

5. **Offline codegen diff of the N/S torus loss** (user go, 2026-11-04; no device runs). Finding to explain: on the
   P150 4x2 XY torus every N/S row loses a near-constant +2.0 to +2.2 cycles/packet with 16 per-slot IDs (id) against
   8 shared IDs (idoff, same code), at every size, write and scatter (broker 671/672, item 2); the 2x4 torus's rows,
   whose ring routers change the same way (8 -> 16), do not lose.
   - Images: mock captures of v4txa11id with `TT_FABRIC_EXP_TRID_ALLOC` unset (idoff) and set (id), on
     torus/torusxy_4x2_sizes.yaml (4x2 XY torus descriptor) and torus/torusxy_2x4_sizes.yaml (tree v4txa11id_1104cg,
     whose host builds the same images as the 649 host with the new switch unset, SLOT_ALLOCATOR §12.10's gate).
     96 router configs per torus, paired across the arms by every named arg except the three RX_CH_0 ID args (none
     unpaired). Every receiver logged G 1 with 8 IDs (idoff) or 16 (id). Tools: router_bringup line_compare.py
     (addr2line inline chains), plus slotalloc/trid_codegen/ (pair.py: the pairing; sites.py: instructions per call
     site; hunks.py / sdiff.py: register-normalized instruction diffs of receiver::run).
   - E0 (sender::run): byte-identical .text in both arms on every router of both tori. sender::run has no transaction
     ID or status-read code. Nothing on the sending ERISC changes.
   - E1 (receiver::run) changes on every router. Common to all receivers (both tori, all directions):
     - The slot's ID: idoff `andi rX, index, 7` (the shared mask), id `zext.b rX, index` (identity): one instruction
       each.
     - Each NoC write and each forward tags its ID with one more instruction in id (`zext.b` before the shift into
       the command buffer's transaction-ID field): +1 per local write, +1 per forward.
     - The completion proof: idoff computes the oldest uncredited slot (`addi, sub, andi 15`) and its ID (`andi 7`)
       separately; id uses the index as the ID: -2 instructions per check. One status read per check in both arms:
       the NoC status register NIU_MST_WRITE_REQS_OUTGOING_ID(ID) at base + 4 x ID, IDs 0-7 (idoff) against 0-15
       (id); same register family, same count. (id's static count shows two reads: the idle-pass path got its own
       copy of the check; it runs only on a pass with no packet.)
   - Per packet on the fast path (static, from the call-site counts; a unicast write delivered, or a 2-chunk scatter;
     G 1, so one proof and one credit per packet):

     | receivers | route decode | slot ID | local write(s) | proof | credit accumulate | per packet (id - idoff) |
     | --- | --- | --- | --- | --- | --- | --- |
     | 4x2 N/S (lose ~2 cycles) | +1, a stack load | 0 | +1 | -2 | +1 | **+1 instruction, +1 stack load** |
     | 4x2 E/W | 0 | 0 | +1 | -2 | 0 | -1 instruction |
     | 2x4 E/W (ring of 4, no loss) | 0 | 0 | +1 | -2 | 0 | -1 instruction |
     | 2x4 N/S | 0 | 0 | +1 | -2 | 0 | -1 instruction |

     The scatter path's other local-write sites (chunks 2 and 3 of 3-4 chunk packets) lose a stack store and load in
     id on the 4x2 N/S receivers; the rows' 2-chunk packets do not run them.
   - What only the losing receivers get: in id, one of the chip coordinates the route decode reads
     (`routing_table.my_mesh_coord_*`, loaded once before the loop) no longer stays in a register. It is stored to the
     stack before the loop and reloaded on every packet (`lw a4,12(sp)` at fabric_common.h:281,
     decode_action_y_first; one instruction sits between the load and its use). idoff keeps both coordinates in registers and reloads
     one per round. The credit accumulate also gains one instruction (`addi a5,sp,96`, the counters' base). These
     receivers have the largest loop of the set (run() 658 / 646 instructions, three forwarding downstreams). The
     4x2 N/S receivers' run() is 12 instructions shorter in id overall, and 3 fewer stack stores. Most of the diff is
     register allocation and scheduling that does not run per packet.
   - Reading: the code is not equivalent apart from constants on the losing routers. The 4x2 N/S receivers, and only
     they, gain a per-packet stack load and a net +1 instruction per packet in id, where every other receiver nets -1
     and no stack traffic. Hypothesis C (codegen, not tested): that spill on the receiving ERISC causes the ~2 cycles.
     Two things it would need, neither tested: that these rows are bound by the receiving ERISC (only a one-side A/B
     can say so), and that a stack load costs the per-packet cycles (static counts do not give stall time; the load's
     result is used two instructions later). Hypothesis B (occupancy) stays untested too. The cheapest isolating
     A/B would change only that spill (the coordinate kept out of the loop's register pressure, or the same spill
     forced in idoff) and nothing else; its design goes to the user before any variant.

The rebuild changes the LB stage again (`lb2_stage.sh full`). Pending LB jobs must not straddle a re-stage: that is
the dyn-slot A/B, lb2 171995.

## 7. On the port: per-channel IDs, offline gate (user go 2026-11-05; offline only, no device run)

**The tree.** Worktree /home/aagarwal/tt-metal-port-trid, branch aagarwal/fabric-router-mainline-trid. All local and unpushed.
- 6dad5fec638: the cleanup batch plus the both-byte decode, pushed.
- 3678a62eb2d: the staged allocator, the same change as c53b5a5cbbd (cherry-picked).
- f9c258d280b: "fabric router: receiver transaction IDs follow each channel's slot count".
  - Host: fabric_trid_allocator.{hpp,cpp}, a pure function next to the slot allocator. The builder runs it per router from the final receiver counts and first-level acks, and emits RX_CH_<c>_COMPLETION_GROUP_SIZE / _NUM_TRIDS / _TRID_START (unused channels 1 / 0 / 0) in place of NUM_TRANSACTION_IDS.
  - Router: reads them and checks them at compile time (G in {1,2,4}, at least one ID per channel, consecutive ranges, 16 at most, G = 1 under first-level acks). The completion group size moved from the router to the host. A slot's group maps to its ID by identity when the channel has an ID per group, by mask otherwise.
  - Teardown and flush_write_to_noc_pipeline wait on the channel's own range. The old non-deadlock-avoidance flush walked IDs 0..N-1 whatever the channel; that bug is fixed.
  - initialize_receiver_channel_trid_starts and completion_group_size_for are removed.
  - The VC1 + VC2 ring/torus TT_FATAL stays, with the same condition: no longer an ID-budget limit, still not brought up.
  - No study references.
- The uniform-rule switch for the A/B arm is local only (fabric_exp/port_trid/trid_uniform_switch.patch, with the slot log and the tier switch). TT_FABRIC_EXP_TRID_UNIFORM=1 gives every channel the earlier 16 / 8 / 4 IDs with the same group sizes.
- The first build sat on the pre-fix allocator. After the allocator fix the branch was rebased onto it and recaptured: the ID tables and the pattern of changed images are identical (receiver counts don't move).

**Gate.** Tool: fabric_exp/port_trid/trid_gate.py. Outputs: trid_table_rc.md, trid_images_rc.txt, trid_loads_*.txt, trid_size_rc.md, and trid_table_cov_partial.md (coverage: uniform 51 cases, rule 34, stopped to free CPU for the allocator gate; pre-rebase, same receiver counts).

1. **ID table** (rule against uniform; G / groups / ids@start, "!" shared, "a" first-level acks):

   | config | receivers | rule | uniform |
   | --- | --- | --- | --- |
   | 2D mesh, 2D 3072, mesh 4x2, Galaxy mesh | 16 | G4, 4 groups, 4@0 | 16@0 |
   | 2D 8704 | 8 | G2, 4 groups, 4@0 | 16@0 |
   | 1D line 4352 / 3072, NeighborExchange | 32 | G4, 8 groups, 8@0 | 16@0 |
   | 1D line 8704 | 16 | 4@0 | 16@0 |
   | ring | 32a | G1, 32 groups, 16@0! | 8@0! |
   | torus wrapped axis (P150 tori, Galaxy torus, Y torus N/S) | 16a | G1, 16 groups, 16@0 | 8@0! |
   | VC1 split 4352 / 3072, 4x4 | 8 \| 4, 16 \| 8 | 4@0 \| 4@4 | the same |
   | VC1 split 8704 | 4 \| 2 | 4@0 \| 2@4 | 4@0 \| 4@4 |
   | VC2 on 4352 / 3072 | 16 \| 4 | 4@0 \| 4@4 | the same |
   | VC2 on 8704 | 8 \| 2 | 4@0 \| 2@4 | 4@0 \| 4@4 |

2. **Images, rule against uniform:**
   - Every E0 image is identical.
   - E1 images differ exactly where the ID rows differ: 2D, 1D, ring, tori, NeighborExchange, and VC1 / VC2 at 8704 B.
   - E1 images are identical where the rows are equal: VC1 split 4352 / 3072 / 4x4 and VC2 on 4352 / 3072.

3. **Hypothesis C check: a new per-packet stack reload, the §6 kind?** Yes, at the decode site, on the 4x2 torus N/S receivers (32 builds) and the Galaxy torus N/S receivers (20), rule against uniform. Static counts only.
   - **Decode site:**
     - The rule arm spills local_x (the routing table's my_mesh_coord_x) at receiver::run's entry (`sw a4,12(sp)`).
     - It reloads it at the head of the per-packet loop (`lw a4,12(sp)`, fabric_common.h:360, decode_action_y_first) for the X byte's address. That runs once per packet.
     - The uniform arm keeps both coordinates in registers in the per-packet loop and reloads them from the stack once per outer iteration (32 passes).
   - **Proof site:** both arms reload the NoC status base from the stack per proof (rule `lw a4,16(sp)` on two paths, uniform `lw a2,8(sp)` on one). This is not new.
   - **Elsewhere, uniform only:**
     - a store and reload of t1 around the local write's command-buffer-ready wait (ncrisc_noc_set_transaction_id, 2 static copies);
     - one more reload at advance() (router_receiver.hpp:213);
     - one more around the deliver_other call.
   - **Totals:** rule 19, uniform 21 stack loads in receiver::run. E/W torus receivers, mesh N/S receivers and ring receivers have the same stack loads in both arms.
   - Stack is core-local (1-2 cycles), L1 is 7-10. Whether the rule's decode reload or the uniform's local-write traffic costs more per packet is not settled by these counts: a device A/B would settle it.
4. **Size and stack at -O3**, reconcile configs, rebased build:
   - Largest E0+E1 is 21,744 B in both arms (4,880 B left).
   - E1 per config, rule minus uniform: -48 to +96 B.
   - Least free stack is the same in both arms (E0 1,620 B, E1 4,232 B).

**Options for the decode-site reload (none built; the user decides):**
- **(i) Cap first-level-ack receivers at 8 IDs.**
  - Ring and torus wrapped-axis receivers would then get G1 with 8 IDs shared, exactly the uniform arm's arguments. Their images would be the uniform arm's, without the decode spill. This is derived from equal arguments, not compiled.
  - Receivers without first-level acks keep the rule: mesh 4, line 8.
- **(ii) Keep the rule as is, and let the device A/B decide.** One P150 job, rule against uniform, on the 4x2 torus N/S rows plus the census rows the rule changes. This is the cheapest isolating A/B.
- **(iii) A documented placement primitive for the coordinate.** Not proposed: it is narrow.

**Device A/B: rule vs uniform, P150, lb2 176756 (bh-lb-120-a07u28, 2026-11-05; the user chose option (ii)).**
- **Build:** one build, aagarwal/fabric-router-mainline-trid at f9c258d280b plus trid_uniform_switch.patch, staged as port_trid_f9c2.tar.zst. No test-infra patch.
- **Run:** two passes reversed. Every run passed and logged the expected IDs for its arm:
  - rule: 16 on the ring and the torus, 4 on the 2D mesh, 8 on the line and NeighborExchange;
  - uniform: 8 on the ring and the torus, 16 elsewhere.
- **Data-checked rows passed in both arms:** 2D, ring and the 4x2 XY torus.
- **Skipped tests:** the port's test binary skips the ring validation's 3- and 4-link tests on this host (2 routing planes): 19 of 37 run.
- An earlier attempt, 176742 on a04u24, stopped on that count. That was the runner's check; it is fixed.
- **Report:** fabric_exp/port_trid/report_trid_176756.md. Both passes agree within 0.2 cycles on every row.

| rows | what differs | rule vs uniform, mean of the passes |
| --- | --- | --- |
| 4x2 torus N/S: 1-hop, bidir, 3-hop, wrap; writes, 256-4352 B | behaviour (16 IDs, one per slot, vs 8 shared), and the decode-site spill (§7 check 3) | +1.2% to +1.6% (+2.0 to +2.3 cycles/packet), all 20 rows |
| same, scatter writes | same | +1.0% to +1.1% (all 20 rows) |
| 4x2 torus E/W 1-hop: writes and scatter, 256-4352 B | behaviour; no spill (same stack loads in both arms) | -0.5% to -0.8% (-0.8 to -1.1 cycles/packet), all 10 rows |
| ring (exp_ring, 14 rows) | behaviour: 16 IDs shared by 2 slots vs 8 shared by 4 | +0.0% (every row within 0.05%) |
| 2D mesh census (10), 1D Linear (12), NeighborExchange (2) | code generation only: the same slot-to-ID map; a flushed atomic takes the counter path | -0.1% to +0.1% |

**Readings:**
- **Mesh and 1D rows are within ±0.1%.** The rule's 4 / 8 IDs give the same slot-to-ID map as the uniform 16 there, as confirmed from the code before the run, so the arms differ only in generated code.
- **Torus E/W and ring rows don't lose under the rule.** These rows get the ID-policy change without the decode spill: E/W is 0.5-0.8% faster and the ring is unchanged.
- **Only the 4x2 torus N/S rows lose, by 1.0-1.6%.** They are the rows whose receivers also carry the per-packet stack reload at the decode site (§7 check 3).
- **Hypothesis, not isolated:** the N/S loss comes from that spill, not from the ID policy. The E/W receivers get the same policy change and gain. An isolating A/B would keep 16 IDs and remove only the reload. No such variant is built; its design goes to the user first.
