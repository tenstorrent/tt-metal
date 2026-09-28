# Fused MSDA optimization candidates

Op: `ttnn/cpp/ttnn/operations/experimental/fused_msda/`. Parent issue
[#55198](https://github.com/tenstorrent/tt-metal/issues/55198). State as of
`44c5fc4e3e7` (branch `ctr-mmicic/55231-msda-sfpu-geometry`, SFPU geometry
landed). Measured on **Blackhole, 110 cores**. The launch configs set
`MESH_DEVICE=N150`, but that does not change the hardware.

## 1. Candidates

Base shape, op = **3482 us**. Split: NoC gather **72%**, tile scatter **12%**,
geometry + reduction + bookkeeping **17%** (§3).

Direction (mentor, 2026-09-24): stop treating each corner as an independent
random 64 B DRAM read. Rearrange data and work so the reader gathers from
wider, already-local pieces of the feature map. The model for this is conv's
shard + halo ([ttcnn.md](../../../../../../tech_reports/CNNs/ttcnn.md), "Halo").
It is not a 1:1 mapping, because MSDA offsets are learned while a conv window
is fixed. Prototype on a small pyramid first.

Legend: **P** = prerequisite, **E** = experiment. What can limit the gather:
* *DRAM-locality-bound*: DRAM row misses. **Ruled out by E0** (§3).
* *Software-bound*: reader RISC work per read (corner decode, validity,
  page and bank address). **Ruled out by E10**: 7% of the gather.
* *DRAM-endpoint-bound*: DRAM controllers cannot serve the request rate.
  **Mostly ruled out by E1**: moving `value` to L1 saves only 14% of the NoC
  time.
* *NoC-bound*: the NoC reads themselves, 93% of the gather (E10), and still
  ~86% of that with no DRAM involved (E1). The NoC itself saturates when 110
  cores read at once. Where exactly (links, endpoints, one NoC) is open: E11.

**The limit is shared, not per core.** The
[BH NoC bandwidth table](https://docs.google.com/spreadsheets/d/1nN1ZPUNc2f0d6YEenBOHS7BdkpGlBysZ6SCArinGOrE/edit?gid=2113527795#gid=2113527795)
(bandwidth vs transfer size, one core) gives 2.67 GB/s for a 64 B read from
DRAM or L1 alike, ~24 ns per read. Under the op, the NoC part of the gather
costs **188 ns per read** on the busiest core (E10), 8x more. All 110 cores
together reach only 34 GB/s (536 M reads/s), against 110 x 2.67 ≈ 290 GB/s
if cores did not interfere. The single-core table cannot show that. E1 shows
the same jam with `value` in L1 (~160 ns per read), so it is the NoC, not
DRAM.

The reader of the op does the same thing for every sampling point: it
computes where the 4 bilinear corners are, reads 4 small pieces (64 B) of the
feature map, and copies them into a tile for compute. Almost every candidate
below makes those reads fewer, cheaper or local.

| # | Candidate | In plain words | Targets | Ceiling | Cost | Gate / status |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | Working-set sweep: base read count, pyramid shrunk 15.4 → 3.0 → 0.8 MB | Same number of reads, a 20x smaller feature map. If a small map were much faster, DRAM locality would be the problem. It was not faster, so locality is not the problem. | diagnosis | — | done | **Done, 2026-09-24.** Cost per issued read is flat: 309 / 308 / 309 ns. Not DRAM-locality-bound (§3) |
| E10 | Split the gather: keep corner decode, validity and address computation, skip only `noc.async_read` + barrier | Remove `noc.async_read` completely, so nothing is read but everything around it still runs (corner positions, bounds checks, address computation). The time that disappears is the cost of reading. The time that remains is preparation. Result: reading is 93%, preparation 7%. Making the address math cheaper cannot help much. | diagnosis | — | done | **Done, 2026-09-25.** NoC reads 2313 us (93%), address work 171 us (7%). Not software-bound (§3) |
| E1 | `value` in L1 interleaved (remote L1 reads, no DRAM) | Put the whole feature map in L1 spread over the cores instead of DRAM, with no kernel change. For one core, a 64 B read costs the same from L1 and DRAM (NoC table). But with 110 cores reading at once, all requests land on a few DRAM endpoints; in L1 they spread over 110. If it gets much faster, the DRAM endpoints are the jam. Result: only ~9.5% faster, so DRAM is a small part of the jam and the NoC is the rest. | gather | ~10% | zero kernel change | **Done, 2026-09-25.** 3485 → 3154 us (15.4 MB), 3371 → 3097 us (0.77 MB); output matches DRAM (PCC ≥ 0.9999). Remote L1 buys ~10%; only reads that skip the NoC can buy more (§3) |
| E11 | NoC trace + `tt-npe` on the base run: link and endpoint utilization | Record every NoC transaction of one op run and replay it in the NoC simulator. It shows which links or endpoints are full, so we know whether the jam is a few hot spots or the whole mesh. | diagnosis | — | profiler flag + `tt-npe` (now installed), ~1-2 h | **Next.** Decides whether E12 can work and how E3b must place data |
| E12 | Split value reads across NoC0 and NoC1 | Each core has two NoCs, and the reader sends all value reads on one. Send half on the other. If one NoC's links are the jam, this could nearly double gather throughput. | gather | up to ~2x if links saturate | small reader change; must not clash with the writer's NoC use | After E11, only if it shows link saturation. Feasibility not checked yet |
| P1 | Realistic sampling locations in harness: projected BEV reference points (real `lidar2img`), or a dump from a real run | The harness picks reference points uniformly at random, so neighbour queries look at unrelated places. Real BEVFormer projects neighbour BEV cells to neighbour image pixels. E9 needs the real pattern. | E9 | — | harness | Harness uses `torch.rand` reference points ([test_fused_msda_perf.py:103](../test_fused_msda_perf.py#L103)). E0 shows locality does not set the gather cost, so P1 now matters only for E9 / E3b sizing |
| E8 | **L1 smoke test on a small pyramid**: tiny, each core bulk-loads its head's `value` slice (230 KB) into L1 once, then gathers locally | On the small (tiny) pyramid each core first copies its head's whole feature map (230 KB) into its own L1, then reads locally. This is the cheapest check of the mentor's L1 idea. If even this is not faster, the big L1 designs (E3) are not worth building. | gather | local vs remote L1 vs DRAM | prototype reader hack, ~1-2 days | Mentor's "prototype on smaller pyramid". Compare with tiny DRAM (323 us) and E1. No gain here means stop E3 |
| E9 | **Footprint analysis** (host, no device): per 32-query block, bbox of sampled corners per level; % of reads inside region + halo margin M | On the host, for each group of 32 queries, measure how big the area they sample is. If queries that sit next to each other also sample next to each other, a core can preload "its" area (E3b). If they sample all over the map, it cannot. | E3b feasibility | — | script, needs P1 | Picks the halo size M and decides E3a vs E3b |
| P0 | Production-shaped SCA workload (6 cams, `rebatch_len`), confirm model dims | The harness runs 1 camera (15.4 MB). Production runs 6 cameras (92.5 MB). Any L1 sizing must use 92.5 MB. | sizing | — | harness | Harness is `B=1`, 15.4 MB. Production is 6 x 15.4 = **92.5 MB**. Needed before E3 sizing |
| E2 | Corner-pair reads, [#58110](https://github.com/tenstorrent/tt-metal/issues/58110): head-major `value` so `x0`, `x0+1` of one head are contiguous → 2x128 B (NW+NE, SW+SE) per point. `(B, H, S*D)` (one page per head) or `(B, H, ceil(S/k), k*D)`; a pair that straddles a page is split into two reads | Store the feature map head by head, so the left and right corner of a point sit next to each other in memory. One 128 B read then replaces two 64 B reads: 2 reads per point instead of 4. | gather 72% | measured: −14.8% op | transpose S↔H after `value_proj`, per layer (not measured yet) | **Phase 1 done, 2026-09-28, below the 25% gate** (§3). Base 3472 → 2958 us with k=32, bit-exact. Halving requests saved ~22% of NoC time, not ~50%, so gather cost is not per transaction. One page per head is worse (a head in one of 7 banks). Next decision: transpose cost in the module |
| E3a | L1 residency by head: levels 1-3 replicated, level 0 sharded per head group | Keep the small pyramid levels (1-3) copied in every core's L1, and split the big level 0 across the cores of one head. 75% of reads become local. Fits 1 camera, not 6. | gather (+ scatter) | removes NoC for 75% of reads | large | Fits 1 camera, not 6 cams. Fallback if E9 shows wide footprints |
| E3b | **L1 residency by space (halo analogy)**: core owns a normalized image region at **all 4 levels**, all 256 ch (512 B sticks, bulk contiguous DRAM reads), plus halo margin M. Units assigned by reference point location. Reads outside the halo fall back to DRAM | Each core owns one area of the image, at all 4 levels and for all heads, plus a margin around it. It loads that area from DRAM once, in large reads. Queries go to the core that owns the area they look at. Reads that fall outside the margin still go to DRAM. This is the largest change, and it depends on E8 and E9. | gather (+ scatter) | most reads local | largest: new work split, per-frame region config | The win is removing NoC transactions (local L1 reads), not DRAM locality (E0). Gated on E8 + E9. Remote-L1 reads buy only ~10% (E1), so the design must keep most reads core-local. 92.5 MB / 110 cores ≈ 840 KB per core before halo, so it is tight |
| E4 | Access grouping: strided vs contiguous unit order; sort reads by `y0` | Reorder which core handles which queries, or sort the reads, so reads close in memory go out together. Deprioritized, because E0 shows closeness does not matter. | gather | low | factory A/B | **Deprioritized.** It is a locality lever, and E0 shows locality does not matter |
| E5 | Second dataflow RISC splits gather/scatter | Each Tensix core has two data-movement RISCs, and the second is almost idle. Split the reading work across both. Deprioritized: E10 shows the jam is shared by all cores, so two readers per core wait on the same jam. | reader | low | semaphores, CB ownership | **Deprioritized.** E10: the limit is shared NoC/DRAM throughput, and a second issuer hits the same wall |
| E6 | Reduction on FPU: DEST accumulate / diagonal-weight matmul / batch inits | The weighted sum of corners runs as many small tile operations. Accumulate them in the destination register instead of packing to L1 each time. At most 17% of the op. | ≤17% slice | ≤17% | medium | After gather. Split the 17% first |
| E7 | Pack several sampling points across tile columns in SFPU geometry | The corner-position math uses only 32 of 1024 lanes in a tile. Pack more points into one tile. At most 17% of the op. | ≤17% slice | ≤17% | medium; weight must return to col 0 for `mul_tiles_bcast<COL>` | After gather |
| P2 | Full-model A/B: old composition vs fused op | The old (unfused) path was deleted when the fused op landed, so nobody has measured the whole-model gain. | reporting | — | restore old path from git | The old path was replaced outright, so this A/B has never run. Do it or drop it explicitly |
| — | Fix perf-counter chip-lock deadlock (§2) | The profiler option that would show how long compute waits for the reader hangs on its own device lock. We measure by compiling parts of the reader out instead. | tooling | — | — | `CB-COMPUTE-WAIT-FRONT` is still uncapturable. The compile-out split (§3) is the workaround |
| — | One writer barrier per tile, not per row | The writer waits after every output row instead of once per tile. Harmless today because the writer is not the bottleneck. | writer | ~0 while reader-bound | trivial | Fold into next writer change |
| — | Close [#56768](https://github.com/tenstorrent/tt-metal/issues/56768) | The ticket proposes reading straight into tile layout with 32 B reads. Blackhole DRAM needs 64 B alignment, so the reads are illegal, and they would double the request count anyway. | — | — | — | Not viable on BH (§4). Re-file only as part of E3 |

Scatter (12%) has no separate candidate. An L1→L1 NoC copy adds transactions,
which is the wrong direction. With E3 the source is local L1 at 16 B alignment,
so the scatter can copy directly into tile faces.

Order:
1. E11. Re-capture `PM FPU UTIL` in the same run.
2. E2 ([#58110](https://github.com/tenstorrent/tt-metal/issues/58110)):
   phase 1 measured −14.8%. Continue only if the device-side transpose costs
   well under the ~510 us it saves.
   E12 if E11 shows link saturation.
3. E8, then P1 + E9 + P0, then E3b (E3a if E9 shows wide footprints). E1
   shows remote L1 buys only ~10%, so E3 must make reads core-local (no NoC)
   to pay.
4. E6 / E7 last.

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
* NoC trace analysis needs `tt-npe`, which is not built in this workspace.
* `-k 'a and b'` breaks under tracy re-invocation. Use full node ids.
* The PCC suite's abs/rel/ratio gates print but do not assert. **Read the
  high-error ratio, not the PCC** (§4).

## 3. Current numbers

| | tiny | base |
| --- | --- | --- |
| Q / heads / levels / points | 900 / 4 / 1 / 4 | 2500 / 8 / 4 / 4 |
| spatial shapes | `(80,45)` | `(200,113) (100,57) (50,29) (25,15)` |
| work units / per core | 116 / 2 | 632 / 6 |
| value reads / call | 59 k x 64 B = 3.8 MB | 1.29 M x 64 B = 82.8 MB |
| `value` size (1 camera) | 0.92 MB | 15.4 MB (1.93 MB/head) |

Kernel-only device time:

| Variant | Shape | Before SFPU (`329670f2e52`) | Now | Speedup |
| --- | --- | --- | --- | --- |
| `packed_multi` (production) | tiny | 690.4 us | **323.3 us** | 2.14x |
| `packed_multi` (production) | base | 8015.1 us | **3481.5 us** | 2.30x |
| `canonical_multi` | base | 8322.6 us | 4412.9 us (+26.8% vs packed) | 1.89x |
| `packed_per_level` (4 calls + adds) | base | 8416.3 us | 4375.4 us | 1.92x |

Module base: 8609 → **4083 us**, fused op is 85.5% of it (tiny: 447 us, 72%).

Where the 3482 us goes, measured by compiling parts of the reader out (results
wrong, timings correct):

| Build | base | Term |
| --- | --- | --- |
| full | 3481.5 us | |
| scatter skipped | 3070 us | scatter ≈ **408 us (12%)** |
| gather + scatter skipped | 583 us | gather ≈ **2487 us (72%)** |
| — | | geometry + reduction + staging + decode ≈ **583 us (17%)** |

Gather rate is ~33 GB/s, far under the DRAM roof.

**E0, working-set sweep** (2026-09-24, `test_fused_msda_working_set_perf`).
Base dims (Q=2500, H=8, L=4, P=4, 632 units) with only the pyramid shrunk.
Issued reads are counted on the host, because the reader skips out-of-bounds
corners and a smaller map drops some. Each row is the mean of 3 signposted
iterations, spread ≤0.33%.

| Pyramid | `value` | Issued reads | Op | ns / read / core |
| --- | --- | --- | --- | --- |
| base shapes | 15.42 MB | 1 239 375 | 3485.2 us | 309.3 |
| 4 x `(50,29)` | 2.97 MB | 1 235 507 | 3455.3 us | 307.6 |
| 4 x `(25,15)` | 0.77 MB | 1 197 360 | 3357.7 us | 308.5 |

A 20x smaller footprint gives a 3.7% faster op, and 3.4% fewer issued reads
explain all of it. At 0.77 MB each DRAM bank holds ~100 KB, comparable to its
open-row capacity, so row hits go from ~0 to a sizeable fraction. The op does
not notice. **Cost follows transaction count, not footprint.**

This replaces the earlier "111 ns tiny vs 212 ns base → locality" reading.
Counted per read on the busiest core, the two shapes agree: tiny is 323 us /
1024 reads = 316 ns, base is 3482 us / 12 288 reads = 283 ns.

Limit of the test: 0.77 MB still spans many DRAM rows. A pyramid of a few
KB per bank would close the last gap. E1 is the more useful test.

**E10, gather split** (2026-09-25, base, `ws15mb`). Temporary compile-time
switch in `fused_msda_reader_common.hpp`, reverted after the run. Mean of 3
signposted iterations.

| Reader build | Op | Term |
| --- | --- | --- |
| full | 3472.2 us | reproduces 3482 |
| scatter skipped | 3069.8 us | reproduces 3070 |
| scatter + NoC read + barrier skipped, address computed | 757.2 us | NoC reads = **2313 us (93% of gather)** |
| scatter + whole gather loop skipped | 586.0 us | address work = **171 us (7%)**; reproduces 583 |

NoC cost per read on the busiest core: 2313 us / 12 288 = 188 ns, i.e. ~6 us
per group of 32 reads behind one barrier. One-core NoC figures predict ~1.3 us
for that group (32 x 24 ns issue + ~0.5 us DRAM latency). The gap is the
110-core contention described in §1.

**E1, `value` in L1 vs DRAM** (2026-09-25, same test with `value_memory`).
`value` is moved to L1 interleaved after the DRAM golden run, and the L1
output is gated against it (PCC ≥ 0.9999). Mean of 3 signposted iterations.

| Pyramid | DRAM | L1 | Saved |
| --- | --- | --- | --- |
| 15.42 MB | 3484.7 us | 3153.6 us | 331 us (9.5%) |
| 2.97 MB | 3457.9 us | 3139.4 us | 319 us (9.2%) |
| 0.77 MB | 3370.5 us | 3097.3 us | 273 us (8.1%) |

Against E10's 2313 us of NoC time, DRAM accounts for ~14%. The remaining
~2000 us is spent on NoC reads that never touch DRAM: ~160 ns per read on the
busiest core, still ~7x the one-core figure.

**E2, corner-pair reads** (2026-09-28, #58110, `test_fused_msda_kernel_perf`).
`value` rearranged head-major on the host, outside the measured region. One
signposted iteration each; every variant is bit-exact against `packed_multi`.

| Variant | tiny | base |
| --- | --- | --- |
| `packed_multi` | 322.2 us | 3472.0 us |
| head-major, 1 page per head | 369.4 us | 3207.5 us (3094.1 in an earlier run) |
| k = 8 (512 B pages) | 300.4 us | 3022.3 us |
| k = 32 (2 KB pages) | 298.1 us | **2957.7 us (−14.8%)** |
| k = 128 (8 KB pages) | 297.0 us | 2964.2 us |

Page `i` of an interleaved buffer lives in bank `i % num_banks`, and this
device has **7** DRAM banks. With one page per head, base puts heads 0 and 7
in bank 0, and tiny (H = 4) leaves 3 banks idle, which is why it regresses.
With k ≥ 32 each head spreads over all banks. The best case saves 514 us,
~22% of the 2313 us NoC time: halving the requests does not halve the cost,
so bytes on the NoC matter too. E11 has to show where.

Blackhole facts that bound the design: L1 is 1.5 MB per Tensix (~165 MB over
110 cores). DRAM alignment is 64 B and L1 alignment is 16 B.

## 4. Rejected: measured or proven

Kept so these are not re-proposed.

| Idea | Why not |
| --- | --- |
| Batch 4 corners behind one NoC barrier (128 reads in flight vs 32) | **3574 us, 2.7% worse.** The gather is throughput-bound, not latency-bound. Also kills "prefetch next corner" |
| Direct-to-face reads (2x32 B per row), i.e. [#56768](https://github.com/tenstorrent/tt-metal/issues/56768) | The `+32` B offset is illegal with DRAM on BH (64 B alignment). It also doubles transactions. Legal only from L1 (E3) |
| Coalesce corners in current layout | Horizontally adjacent corners are 512 B apart in packed layout and 8 pages apart in canonical. Adjacent pages sit on different banks. This is fixable only by a layout change (E2, #58110) |
| Reuse a value page across heads (incl. reading the full 512 B stick per corner and computing all 8 heads on one core) | Offsets are per-head (`sampling_offsets` is `(B, Q, H, L*P*2)`), so each head samples a different pixel. A 512 B stick serves only one head's 64 B: same transaction count, 8x the bytes. Only `reference_points` is head-invariant (<1%). E3b sidesteps it: a spatial shard holds all 8 heads, so every head's read is local |
| Fuse `output_proj` + residual into the writer | `output_proj` mixes all 8 heads, but a writer block holds 1 head. Doing it would need a cross-core reduction |
| `fp32_dest_acc_en` off | Saves 0.4% (noise). High-error ratio goes 0.099 → **0.628** and PCC still passes. Keep it on |
| Canonical operand forms | +26.8% vs packed now (+3.9% before SFPU). Keep #55232-#55236 packing |
| 4 per-level calls ([#55201](https://github.com/tenstorrent/tt-metal/issues/55201)) | +5.1% before SFPU, same ratio after. Keep one multi-level call |
| Absorb remaining layout prep ([#55200](https://github.com/tenstorrent/tt-metal/issues/55200)) | All non-op work was capped at 6.9% of the module before SFPU. Low priority |
| Deeper CBs / compute tweaks (pre-SFPU) | FPU util was 0.0%. Obsolete now that geometry runs on SFPU. Re-measure before reusing |

Precision floor, not a bug: `reference_points` are bf16, which gives up to
~1.5 px of error on the 200-wide level. PCC 0.999 is fine.

Architecture: all numbers are Blackhole only. For Quasar/Trinity, prefer L1
residency and DMA-friendly staging (E3, E8) over BH-specific DRAM page tuning.
