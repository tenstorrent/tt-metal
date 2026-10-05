# Fused MSDA optimization candidates

Op: `ttnn/cpp/ttnn/operations/experimental/fused_msda/`. Parent issue
[#55198](https://github.com/tenstorrent/tt-metal/issues/55198). State as of
`44c5fc4e3e7` (branch `ctr-mmicic/55231-msda-sfpu-geometry`, SFPU geometry
landed), updated for E15 (#58253), E16 (#58602, split gather) and E17 (the two
data-movement RISCs rebalanced). Measured on
**Blackhole, 110 cores**. The launch configs set
`MESH_DEVICE=N150`, but that does not change the hardware.

## 1. Candidates

Base shape, op = **3472 us** with `value` in DRAM. E15 removed the reader's
scatter (tilize on the unpacker): with `value` in L1 interleaved the op is
**1055 us (3.3x)**, with `value` in DRAM still 3172 us. The DRAM endpoints are
now the only big limit, so the lever is getting `value` into L1 in the model
(P0), §3. E16 then split the gather across both data-movement RISCs: **736 us**
with `value` in L1, 1952 us in DRAM (#58602). E17 rebalanced the two RISCs
(post-before-geometry, split row 15, heads-innermost units with reference-point
reuse): **658 us** in L1, 1660-1740 us in DRAM.

Direction (mentor, 2026-09-24): stop treating each corner as an independent
random 64 B DRAM read. Rearrange data and work so the reader gathers from
wider, already-local pieces of the feature map. The model for this is conv's
shard + halo ([ttcnn.md](../../../../../../tech_reports/CNNs/ttcnn.md), "Halo").
It is not a 1:1 mapping, because MSDA offsets are learned while a conv window
is fixed. Prototype on a small pyramid first.

Legend: **P** = prerequisite, **E** = experiment. What can limit the gather,
and what the experiments say (details in §3):
* *DRAM locality* (row misses): **ruled out** by E0.
* *NoC link bandwidth*: **ruled out** by E11 / E11b, busiest link ≤ 21%.
* *Reader software*: address work is 7% (E10), but the **scatter is 60%** of
  the reader's time (E13). It is the first limit.
* *DRAM endpoint request rate*: E1 hid it (−9.5%) because the scatter sets the
  pace; with the scatter skipped, `value` in L1 is **2.7x faster** (E11c). It is
  the second limit.

The reader of the op does the same thing for every sampling point: it
computes where the 4 bilinear corners are, reads 4 small pieces (64 B) of the
feature map, and copies them into a tile for compute. Almost every candidate
below makes those reads fewer, cheaper or local.

| # | Candidate | In plain words | Targets | Ceiling | Cost | Gate / status |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | Working-set sweep: base read count, pyramid shrunk 15.4 → 3.0 → 0.8 MB | Same number of reads, a 20x smaller feature map. If a small map were much faster, DRAM locality would be the problem. It was not faster, so locality is not the problem. | diagnosis | — | done | **Done, 2026-09-24.** Cost per issued read is flat: 309 / 308 / 309 ns. Not DRAM-locality-bound (§3) |
| E10 | Split the gather: keep corner decode, validity and address computation, skip only `noc.async_read` + barrier | Remove `noc.async_read` completely, so nothing is read but everything around it still runs (corner positions, bounds checks, address computation). The time that disappears is the cost of reading. The time that remains is preparation. Result: reading is 93%, preparation 7%. Making the address math cheaper cannot help much. | diagnosis | — | done | **Done, 2026-09-25.** NoC reads 2313 us (93%), address work 171 us (7%). Skipping the scatter saves only 408 us (§3) |
| E1 | `value` in L1 interleaved (remote L1 reads, no DRAM) | Put the whole feature map in L1 spread over the cores instead of DRAM, with no kernel change. For one core, a 64 B read costs the same from L1 and DRAM (NoC table). But with 110 cores reading at once, all requests land on a few DRAM endpoints; in L1 they spread over 110. If it gets much faster, the DRAM endpoints are the jam. Result: only ~9.5% faster, because the scatter sets the pace; without the scatter L1 is 2.7x faster (E11c). | gather | ~10% | zero kernel change | **Done, 2026-09-25.** 3485 → 3154 us (15.4 MB), 3371 → 3097 us (0.77 MB); output matches DRAM (PCC ≥ 0.9999). With the scatter in place remote L1 buys ~10%; without it, 2.7x (E11c) |
| E11 | NoC trace + `tt-npe` on the base run: link and endpoint utilization | Record every NoC transaction of one op run and replay it in the NoC simulator. It shows which links or endpoints are full, so we know whether the jam is a few hot spots or the whole mesh. Result: nothing is full. The links are mostly idle, and each core spends most of its time waiting between groups of reads. | diagnosis | — | done | **Done, 2026-09-28.** Busiest link 21%, average 5.8%, congestion impact 0%, DRAM 5.7% on each of 7 controllers. All value reads go on NoC0. Not link-bound (§3)
| E12 | Split value reads across NoC0 and NoC1 | Each core has two NoCs, and the reader sends all value reads on one. Send half on the other. If one NoC's links are the jam, this could nearly double gather throughput. | gather | ~0 while links are at ≤21% | small reader change; must not clash with the writer's NoC use | **Deprioritized.** E11 (before E16): NoC0 carried all value reads, no link near full. Since E16 the writer issues rows `[SPLIT_ROW, 32)` of them (17 of 32 since E17) on its own NoC |
| P1 | Realistic sampling locations in harness: projected BEV reference points (real `lidar2img`), or a dump from a real run | The harness picks reference points uniformly at random, so neighbour queries look at unrelated places. Real BEVFormer projects neighbour BEV cells to neighbour image pixels. E9 needs the real pattern. | E9 | — | harness | Harness uses `torch.rand` reference points ([test_fused_msda_perf.py:103](../test_fused_msda_perf.py#L103)). E0 shows locality does not set the gather cost, so P1 now matters only for E9 / E3b sizing |
| E8 | **L1 smoke test on a small pyramid**: tiny, each core bulk-loads its head's `value` slice (230 KB) into L1 once, then gathers locally | On the small (tiny) pyramid each core first copies its head's whole feature map (230 KB) into its own L1, then reads locally. This is the cheapest check of the mentor's L1 idea. If even this is not faster, the big L1 designs (E3) are not worth building. | gather | local vs remote L1 vs DRAM | prototype reader hack, ~1-2 days | Mentor's "prototype on smaller pyramid". Compare with tiny DRAM (323 us) and E1. No gain here means stop E3 |
| E9 | **Footprint analysis** (host, no device): per 32-query block, bbox of sampled corners per level; % of reads inside region + halo margin M | On the host, for each group of 32 queries, measure how big the area they sample is. If queries that sit next to each other also sample next to each other, a core can preload "its" area (E3b). If they sample all over the map, it cannot. | E3b feasibility | — | script, needs P1 | Picks the halo size M and decides E3a vs E3b |
| P0 | Production-shaped SCA workload (6 cams, `rebatch_len`), confirm model dims | The harness runs 1 camera (15.4 MB). Production runs 6 cameras (92.5 MB). Any L1 sizing must use 92.5 MB. | sizing | — | harness | Harness is `B=1`, 15.4 MB. Production is 6 x 15.4 = **92.5 MB**. Needed before E3 sizing |
| E2 | Corner-pair reads, [#58110](https://github.com/tenstorrent/tt-metal/issues/58110): head-major `value` so `x0`, `x0+1` of one head are contiguous → 2x128 B (NW+NE, SW+SE) per point. `(B, H, S*D)` (one page per head) or `(B, H, ceil(S/k), k*D)`; a pair that straddles a page is split into two reads | Store the feature map head by head, so the left and right corner of a point sit next to each other in memory. One 128 B read then replaces two 64 B reads: 2 reads per point instead of 4. | gather 72% | measured: −14.8% op | transpose S↔H after `value_proj`, per layer (not measured yet) | **Phase 1 done, 2026-09-28, below the 25% gate** (§3). Base 3472 → 2958 us with k=32, bit-exact. Halving requests saved ~22% of NoC time, not ~50%, so gather cost is not per transaction. One page per head is worse (a head in one of 7 banks). Next decision: transpose cost in the module |
| E3a | L1 residency by head: levels 1-3 replicated, level 0 sharded per head group | Keep the small pyramid levels (1-3) copied in every core's L1, and split the big level 0 across the cores of one head. 75% of reads become local. Fits 1 camera, not 6. | gather (+ scatter) | removes NoC for 75% of reads | large | Fits 1 camera, not 6 cams. Fallback if E9 shows wide footprints |
| E3b | **L1 residency by space (halo analogy)**: core owns a normalized image region at **all 4 levels**, all 256 ch (512 B sticks, bulk contiguous DRAM reads), plus halo margin M. Units assigned by reference point location. Reads outside the halo fall back to DRAM | Each core owns one area of the image, at all 4 levels and for all heads, plus a margin around it. It loads that area from DRAM once, in large reads. Queries go to the core that owns the area they look at. Reads that fall outside the margin still go to DRAM. This is the largest change, and it depends on E8 and E9. | gather (+ scatter) | most reads local | largest: new work split, per-frame region config | The win is removing NoC transactions (local L1 reads), not DRAM locality (E0). Gated on E8 + E9. Remote-L1 reads are 2.7x faster than DRAM once the scatter is cheap (E11c), so remote halo reads are acceptable. 92.5 MB / 110 cores ≈ 840 KB per core before halo, so it is tight |
| E4 | Access grouping: strided vs contiguous unit order; sort reads by `y0` | Reorder which core handles which queries, or sort the reads, so reads close in memory go out together. Deprioritized, because E0 shows closeness does not matter. | gather | low | factory A/B | **Deprioritized.** It is a locality lever, and E0 shows locality does not matter. E17 does change the unit order (heads innermost), but for reference-point reuse, not value locality |
| E5 | Second dataflow RISC splits gather/scatter | Each Tensix core has two data-movement RISCs, and the second is almost idle. Split the reading work across both. It was deprioritized on the belief that all cores wait on one shared jam; E11 shows the wait is per core, with the NoC mostly idle. | reader | open | semaphores, CB ownership | **Superseded by E16 (done).** Split by rows of each row-major block rather than moving the scatter, which E15 had already removed |
| E13 | Reader cycle counters: time the barrier wait, the scatter, `reserve_back`, the `x0`/`y0` wait and the rest of the loop separately, without NoC tracing | Read the core's clock around each step of the reader loop and print the totals once per core. It shows where each reader actually spends its time. | diagnosis | — | done | **Done, 2026-09-28.** Scatter 1935 us (60%), read issue 697 us (22%, 84 cycles per read), barrier 209 us (6%); everything else ≤ 5% each (§3). Removing the scatter cannot give more than E10's 408 us |
| E14 | Cheap scatter (plain, unrolled L1 copies instead of volatile) + `value` in L1 interleaved | Make the tile copy cheaper, and move the feature map to L1 without moving any work. Tests both limits at once, with no work redistribution. | scatter + DRAM | measured: −28% | small reader change + `to_memory_config` | **Done, 2026-09-28.** Base 3472 → 2498 us (L1), 3250 us (DRAM); tiny 323 → 254 / 283 us. Scatter only 30% cheaper (~10 cycles per word), still 55% of the reader (§3). Kernel change passes all fused-MSDA suites |
| E15 | Tilize on the unpacker instead of a RISC scatter | For D = 32, 32 staged sticks of 64 B are exactly one row-major 32x32 tile. The reader lands a point's four corners side by side as one row-major block and compute tilizes it on the unpacker; the reader copies nothing and only zeroes skipped slots. | scatter 55% | measured: 3.3x with `value` in L1 | reader + compute change | **Done, 2026-09-28.** Base 3472 → **1055 us** (L1) / 3172 us (DRAM); tiny 323 → **132** / 212 us. All fused-MSDA suites pass (677). D % 32 != 0 keeps the scatter path |
| E16 | Cheaper read issue: split the reads across both data-movement RISCs (E5), or a hand-rolled NoC issue path | After E15 the reader spends 57% of its time issuing 64 B reads at 68 cycles each; compute never makes it wait. BRISC is nearly idle and has its own command buffer, so splitting the reads can halve the issue time. The stateful API (`noc_async_read_one_packet_set_state` / `_with_state`) does not fit as is: it fixes the target core's coordinates, and nearly every read here goes to a different bank. A hand-rolled path that sets length and MID once still writes 4 of the 6 registers per read. | read issue 57% | E5: up to ~−30% of the op; hand-rolled path: a few %, unless the address math (`page % 110` banks, bank table lookup) is also cheaper | E5: split corners or rows between two kernels, sync barriers; hand-rolled: reader only | **Done, 2026-09-30 (#58602).** Rows 0-15 on the reader, 16-31 on the writer, mailbox CB + two semaphores. Base L1 1055 → **736 us (−30%)**, DRAM 3202 → 1952 us, tiny L1 132 → 104 us; module 1906 → 1555 us. Blackhole only. The hand-rolled issue path was not pursued: bank decode is already a constant divisor (§3) |
| E17 | Rebalance the two data-movement RISCs after E16: post each point to the writer before building the next point's geometry tiles, move the split row, order units heads innermost so the V2 reader stages the head-invariant `reference_points` once per query block | Pipeline spans after E16 showed the reader busy ~97% of the op and the writer idle ~230 us: it waited while the reader built geometry tiles and staged each block. Waiting on compute was ~10 us, so a deeper lookahead was not the lever. The block prologue is issue-bound (192 small reads per block, 128 of them reference points), so prefetching it would only move the issue. | reader | measured: −10.6% (L1) | reader + factory order | **Done, 2026-10-02.** Base L1 736 → **658 us**, DRAM 1956 → 1660-1740 us, canonical DRAM 2708 → 2456-2486 us, tiny L1 −0.7%, tiny DRAM −3.0%; layer SCA 4063 → 3630 us, TSA 698 → 683 us. Split row 15 (§3) |
| E6 | Reduction on FPU: DEST accumulate / diagonal-weight matmul / batch inits | The weighted sum of corners runs as many small tile operations. Accumulate them in the destination register instead of packing to L1 each time. At most 17% of the op. | ≤17% slice | ≤17% | medium | After gather. Split the 17% first |
| E7 | Pack several sampling points across tile columns in SFPU geometry | The corner-position math uses only 32 of 1024 lanes in a tile. Pack more points into one tile. At most 17% of the op. | ≤17% slice | ≤17% | medium; weight must return to col 0 for `mul_tiles_bcast<COL>` | After gather |
| P2 | Full-model A/B: old composition vs fused op | The old (unfused) path was deleted when the fused op landed, so nobody has measured the whole-model gain. | reporting | — | restore old path from git | The old path was replaced outright, so this A/B has never run. Do it or drop it explicitly |
| — | Fix perf-counter chip-lock deadlock (§2) | The profiler option that would show how long compute waits for the reader hangs on its own device lock. We measure by compiling parts of the reader out instead. | tooling | — | — | `CB-COMPUTE-WAIT-FRONT` is still uncapturable. The compile-out split (§3) is the workaround |
| — | One writer barrier per tile, not per row | The writer waits after every output row instead of once per tile. Harmless today because the writer is not the bottleneck. | writer | ~0 while reader-bound | trivial | Fold into next writer change |
| — | Close [#56768](https://github.com/tenstorrent/tt-metal/issues/56768) | The ticket proposes reading straight into tile layout with 32 B reads. Blackhole DRAM needs 64 B alignment, so the reads are illegal, and they would double the request count anyway. | — | — | — | Not viable on BH (§4). Re-file only as part of E3 |

Scatter (12%) has no separate candidate. An L1→L1 NoC copy adds transactions,
which is the wrong direction. With E3 the source is local L1 at 16 B alignment,
so the scatter can copy directly into tile faces.

Status 2026-09-29: `value` is in L1 in the model (TSA MSDA 3395 → 1015 us,
SCA 16162 → 6035 us). The fused op is now 28.6% of a layer; data-movement ops
(untilize, slice, permute, reshape, tilize, concat) are ~49%. After E16 the
larger win is outside this op.

Order (toward E3: `value` in L1 first, then decide whether work must move to
the data):
1. P0: does `value` fit in L1 interleaved inside the model? 1 camera is
   15.4 MB; 6 cameras are 92.5 MB, ~840 KB per core, next to the op's CBs and
   the rest of the model. Then put it there in `TTMSDeformableAttention`
   (`value_proj` output to L1) and measure the module, not just the kernel.
2. If 6 cameras do not fit: shard `value` spatially (E3b) or per head (E3a),
   gated on P1 + E9. E8 checks the local-L1 case on the tiny pyramid.
3. E2 (#58110) only matters while `value` stays in DRAM. E6 / E7 once the
   reader is no longer the limit: re-profile first. Re-capture
   `PM FPU UTIL` (the profiler reports 0.0%, counters are not captured).

## 2. How to measure

Harness: `models/experimental/bevformer/tests/perf/test_fused_msda_perf.py`.
Launch configs in `.vscode/launch.json`:

| Config | Captures |
| --- | --- |
| `[bevformer][profile] fused MSDA performance report` | all tests |
| `[bevformer][profile] fused MSDA module performance report` | full `TTMSDeformableAttention`, tiny + base |
| `[bevformer][profile] fused MSDA kernel packed multi-level` | production path |
| `[bevformer][profile] fused MSDA kernel A/B layouts and levels` | all kernel variants |
| `[bevformer][profile] fused MSDA NoC/DRAM report` | production path + NoC traces |
| `[bevformer][profile] fused MSDA L1 counters report` | perf counters |

The PCC gate doubles as warmup. Read only the signposted rows. Run-to-run
variance is ~0.3%.

Known tooling gaps:
* `--profiler-capture-perf-counters` self-deadlocks on `CHIP_IN_USE_*_PCIe`,
  so `CB-COMPUTE-WAIT-FRONT` cannot be captured. Workaround: compile out parts
  of the reader (§3).
* NoC trace analysis needs `tt-npe`. It is built in `../tt-npe`; add
  `../tt-npe/install/lib` and `../tt-npe/install/bin` to `PYTHONPATH` and
  tracy runs it after `--collect-noc-traces`. Tracing slows the op by ~18%
  (3472 → 4105 us), so read timings from an untraced run.
* `-k 'a and b'` breaks under tracy re-invocation. Use full node ids.
* The PCC suite's abs/rel/ratio gates print but do not assert. **Read the
  high-error ratio, not the PCC** (§4).

## 3. Current numbers

| | tiny | base |
| --- | --- | --- |
| Q / heads / levels / points | 900 / 4 / 1 / 4 | 2500 / 8 / 4 / 4 |
| spatial shapes | `(80,45)` | `(200,113) (100,57) (50,29) (25,15)` |
| work units / per core | 116 / 2 | 632 / 6 |
| value reads / call | 59 k x 64 B | 1.29 M x 64 B |
| `value` size (1 camera) | 0.92 MB | 15.4 MB |

Kernel-only device time, `packed_multi` (production): tiny **323 us**, base
**3472-3482 us**, 2.1-2.3x faster than before SFPU geometry. `canonical_multi`
is +26.8%, 4 per-level calls +25.7%. Module base: 8609 → **4083 us**, the
fused op is 85.5% of it. Run-to-run variance ~0.3%.

### What limits the op

1. **The reader is paced by its own scatter.** Per core (E13): scatter 60%,
   read issue 22% (84 cycles per read), barrier wait 6%, everything else
   ≤ 5% each. The reader never waits on compute (0.5%). Plain unrolled copies
   (E14) make it only 30% cheaper: ~10 cycles per word, still 55%.
2. **Behind the scatter sit the DRAM endpoints.** Skip the scatter and the op
   only drops to 3070 us (E10): reads get 4x slower per group, because all
   cores' 64 B requests converge on 7 DRAM endpoints. With `value` in L1 the
   same build runs 1152 us (E11c).
3. **Not the NoC links.** `tt-npe` finds the busiest link at 18-21% and
   congestion impact 0%, with and without the scatter (E11, E11b).
4. **Not DRAM locality.** A 20x smaller pyramid costs the same per read
   (E0).
5. **Placement matters under load.** With the scatter skipped, cores far
   along the NoC route are up to 2x slower, and switching the reader to NoC1
   moves the slow cores from rows to columns (E11c). The full build shows only
   a 9% spread.

So the two limits stack, and only a change that attacks both pays: a cheap or
offloaded scatter *and* fewer DRAM requests (E2) or local `value` (E8, E3).
The ceiling of that pair, measured with scatter skipped and `value` in L1, is
**~1150 us against 3472 us (~3x)**. Either change alone is worth 10-15%.
E14 takes both halfway: **2498 us (−28%)** with a cheaper scatter and `value`
in L1 interleaved, no work moved. **E15 removes the scatter: 1055 us (3.3x)**
with `value` in L1 interleaved, still no work moved. With `value` in DRAM the
same kernel runs 3172 us: the DRAM endpoints are now the whole story, and
getting `value` into L1 is the lever.

### Experiments

All base `packed_multi` on Blackhole (P100, 7 DRAM banks, 110 cores) unless
noted. "Scatter skipped" builds give wrong results and right timings.

| # | Setup | Result | Conclusion |
| --- | --- | --- | --- |
| E0 | Pyramid shrunk 15.4 → 3.0 → 0.8 MB, same read count | 309 / 308 / 309 ns per read | Cost follows reads, not footprint |
| E10 | Reader parts compiled out | full 3472, scatter skipped 3070, NoC reads skipped 757, whole gather skipped 586 us | Scatter alone is worth only 408 us (12%) |
| E1 | `value` in L1 interleaved, full build | 3485 → 3154 us (−9.5%) | Small only because the scatter sets the pace (see E11c) |
| E2 | Head-major `value`, NW+NE / SW+SE as 128 B pairs (#58110) | k = 32 (2 KB pages) 3472 → 2958 us (−14.8%), bit-exact; one page per head is worse (a head in one bank) | Halves read issue; capped by the scatter |
| E11 | NoC trace + `tt-npe`, full build | busiest link 21%, congestion 0%, DRAM 5.7% on every controller; all value reads on NoC0 (before E16) | Links are not the limit |
| E13 | Cycle counters per reader step | scatter 1935, issue 697, barrier 209 us (of 3242 mean) | The scatter paces the reader |
| E11b | Scatter skipped, counters + `tt-npe` | barrier 4x per group; busiest link 18%, congestion 0%; row 0 2x slower than row 9 | Dense reads slow down outside the links |
| E11c | Scatter skipped: reader on NoC1; `value` in L1 | NoC1: gradient moves to columns; L1: busiest core 3093 → **1152 us**, no gradient | The DRAM endpoints are the second limit |
| E14 | Plain unrolled scatter; `value` in DRAM and in L1 interleaved | DRAM 3250 us, **L1 2498 us** (base); tiny 283 / 254 us. Per core with L1: scatter 1354 (55%), issue 639, barrier 76 us | L1 pays once the scatter is cheaper; the scatter is still the limit |
| E15 | Row-major staging + tilize on the unpacker; `value` in DRAM and in L1 interleaved | DRAM 3172 us, **L1 1055 us** (base); tiny 212 / **132 us**; 677 tests pass | Scatter gone; DRAM is the only big limit left |
| E13b | E13 counters on the E15 kernel, `value` in L1 | read issue 568 us (57%, 68 cycles per read), decode 145, geometry push 122, zeroing 68, staging 56 us; barrier, `reserve_back`, `x0`/`y0` wait ≤ 1% each (of 1004 mean, 1074 busiest) | Reader-bound on read issue; compute is not the limit, so E6/E7 would buy nothing now |
| E16 | Split the row-major gather: reader rows 0-15, writer rows 16-31, each on its own NoC | base L1 1055.1 → **736.2 us**, DRAM 3201.8 → 1951.8 us; tiny L1 131.9 → 104.4, DRAM 211.5 → 144.3 us; module 1905.9 → 1555.4 us; same-session A/B, 674 tests pass. Split point 16/16 best in L1 (12: 855, 14: 799, 18: 748 us), 14/18 in DRAM (1783 us); E17 moved the optimum once the reader also builds geometry tiles during the writer's gather. Compile-out before the split: read issue 204 us, address math 227 us (index 112, bank decode 115). After it: reader-only floor 598 us, compute ≲ 630 us | Reader and compute are now both ~600 us; the rest looked like overlap lost to the one-point lookahead, which E17's spans disproved (the reader waits on compute ~10 us). The DRAM baselines here (3202 / 212 us) differ from E15's (3172 / 212) by DRAM run-to-run spread (±2.5%) |
| E17 | Post-before-geometry, split row 15, heads-innermost units + reference-point reuse (V2) | same-session A/B vs E16 (2026-10-05): packed L1 736.4 → **658.5 us**, packed DRAM 1956.2 → 1659.7 / 1739.8 us (two runs), canonical DRAM 2708.1 → 2485.9 / 2456.4 us (two runs), tiny L1 104.4 → 103.7, tiny DRAM 144.4 → 140.1 us; layer (summed FW) SCA 4063.1 → 3630.0, TSA 698.0 → 683.2 us; PCC 1.000000, layer 0.999667. Split row with post-before-geometry, L1: 12: 749.2, 13: 721.5, 14: 693.6, **15: 676.4**, 16: 697.6, 17: 726.2 us (before the unit reorder); DRAM packed / canonical (after the unit reorder): 13: 1783 / 2412, 14: 1715 / 2451, 15: 1740 / 2456, 16: 1775 / 2546 us | Per step: post-before-geometry −38.5, split row 15 −22.2, reference-point reuse −18.0 us. Not linear in the split row; Two DRAM runs of split row 15 differ by 4.8%, so on DRAM 15 is within run-to-run spread of the best value for both layouts (packed 13-16 all within it; canonical 16 is ~4% slower). Reuse gave half the −35..40 us an issue-only model predicted |

The two scatter-skipped builds that settle the stacking (core totals, us):

| Build | mean | busiest | barrier |
| --- | --- | --- | --- |
| full build (E13) | 3242 | 3541 | 209 |
| scatter skipped, `value` DRAM | 2114 | 3093 | 843 |
| scatter skipped, `value` L1 | 1076 | 1152 | 78 |
| cheap scatter, `value` DRAM (E14) | 2805 | 3223 | 342 |
| cheap scatter, `value` L1 (E14) | 2455 | 2607 | 76 |

E15 op times (kernel, not per-core counters): `value` DRAM 3172 us,
`value` L1 **1055 us**.

Method notes: NoC tracing slows the op ~18%, so timings come from untraced
runs. The E13 counters are temporary `get_timestamp_32b()` + `DPRINT` patches,
not part of the branch; their busiest-core total (3541 us) matches the op.
`tt-npe` models link contention only, not DRAM controller queues, endpoint
request rate or arbitration fairness.

Blackhole facts that bound the design: L1 is 1.5 MB per Tensix (~165 MB over
110 cores). DRAM alignment is 64 B and L1 alignment is 16 B.

## 4. Rejected: measured or proven

Kept so these are not re-proposed.

| Idea | Why not |
| --- | --- |
| Batch 4 corners behind one NoC barrier (128 reads in flight vs 32) | **3574 us, 2.7% worse.** The gather is throughput-bound, not latency-bound. Also kills "prefetch next corner" |
| Direct-to-face reads (2x32 B per row), i.e. [#56768](https://github.com/tenstorrent/tt-metal/issues/56768) | The `+32` B offset is illegal with DRAM on BH (64 B alignment). It also doubles transactions. Legal only from L1 (E3) |
| Coalesce corners in current layout | Horizontally adjacent corners are 512 B apart in packed layout and 8 pages apart in canonical. Adjacent pages sit on different banks. This is fixable only by a layout change (E2, #58110) |
| Reuse a value page across heads (incl. reading the full 512 B stick per corner and computing all 8 heads on one core) | Offsets are per-head (`sampling_offsets` is `(B, Q, H, L*P*2)`), so each head samples a different pixel. A 512 B stick serves only one head's 64 B: same transaction count, 8x the bytes. Only `reference_points` is head-invariant: <1% of the bytes, but 128 of the 192 small reads in each block's prologue; E17 reuses it across heads. E3b sidesteps it: a spatial shard holds all 8 heads, so every head's read is local |
| Fuse `output_proj` + residual into the writer | `output_proj` mixes all 8 heads, but a writer block holds 1 head. Doing it would need a cross-core reduction |
| `fp32_dest_acc_en` off | Saves 0.4% (noise). High-error ratio goes 0.099 → **0.628** and PCC still passes. Keep it on |
| Canonical operand forms | +26.8% vs packed now (+3.9% before SFPU). Keep #55232-#55236 packing |
| 4 per-level calls ([#55201](https://github.com/tenstorrent/tt-metal/issues/55201)) | +5.1% before SFPU, +25.7% after (4375 vs 3481 us). Keep one multi-level call |
| Absorb remaining layout prep ([#55200](https://github.com/tenstorrent/tt-metal/issues/55200)) | All non-op work was capped at 6.9% of the module before SFPU. Low priority |
| Deeper CBs / compute tweaks (pre-SFPU) | FPU util was 0.0%. Obsolete now that geometry runs on SFPU. Re-measure before reusing |

Precision floor, not a bug: `reference_points` are bf16, which gives up to
~1.5 px of error on the 200-wide level. PCC 0.999 is fine.

Architecture: all numbers are Blackhole only. For Quasar/Trinity, prefer L1
residency and DMA-friendly staging (E3, E8) over BH-specific DRAM page tuning.
