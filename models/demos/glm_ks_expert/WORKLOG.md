# GLM K-split expert worklog

Goal: GLM-5.3's routed expert (H 6144, I 2048, bfp4 weights, LoFi, fp32 DEST, x / h bfp8, 16 experts a chip at the
4x4 stage, M 320 a mean expert) with gate/up K split in pairs, built ground-up as generic_op programs that copy the flat
expert's mechanics, then integrated into flat_routed_expert (and through it the MoE block's experts + combine).

Why (flat expert today, bh-lb-09 one chip, GLM e16 M320 1184 us): every 128-row gate/up sub-block takes 26.6 us against
18.2 us at LoFi peak; each gate/up core makes 1 pair (2 tiles) from all of K, receiving 835 KB of x a sub-block along a
40-core (west) / 24-core (east) chain. K split in pairs: core (c, r) of a pair holds K half r of both pairs (1x2 blocks
of each), sends the partner its bf16 partials, adds the partner's into DEST before its own half, activation as today.

All runs: bh-lb-09 (12 x 10 Galaxy-like grid), 1.35 GHz, `scripts/run_safe_pytest.sh`.

## Stage 1: one core, operands resident (tests/test_kb_bench.py, kernels/kb_*.cpp)

128-row sub-block = 4 row tiles in passes of 2 (fp32 DEST half-sync holds 4 tiles), 100 sub-blocks a launch:

| form | cyc / tile product | % LoFi | us / sub-block | h err vs unsplit |
|---|---|---|---|---|
| today: 1 pair, K 192 | 17.89 | 89.5% | 20.35 | 1.04e-2 |
| K split, partials into DEST by copy (fp32, UnpackToDestFp32) | 18.91 | 84.6% | 21.52 | 1.04e-2 |
| K split, copy, bf16 partials | 18.41 | 86.9% | 20.94 | 1.05e-2 |
| **K split, partials into DEST by an identity matmul (bf16)** | **18.15** | **88.1%** | **20.66** | 1.12e-2 |

- The matmul is not GLM's problem: today's sub-block runs at 89.5% resident (the flat expert gets 68%).
- Identity matmul: in0 = the partner's partials [2 rows][g, u] (srcB, 7 mantissa bits = bf16), in1 = [[I, 0], [0, I]]
  (CB 3), kt 2: 8 tile products a pass (2%), no copy init / matmul re-init. Then this half of K accumulates on top and
  the activation runs on the PACK thread as today (overlapped with the next pass's MATH).
- Order a sub-block: P passes (partner's pair over this half of K -> partials out), then O passes (own pair, the
  partner's partials preloaded). The partner computes my partials in its P passes, so they arrive a half sub-block
  before my O passes need them.

## Stage 2: x chains, weights resident (ks_chain.py, kernels/kc_*.cpp, tests/test_kc_chain.py)

The flat expert's x path: an injector (relay, x resident: never short) unicasts x blocks ([4 row tiles][8 K tiles] bfp8,
34.8 KB) in 4 pieces into the chain head's x ring; each core's NCRISC forwards pieces (NOC0, 6 in flight, transaction
ids) once the successor freed the slot, the BRISC (NOC1) publishes blocks to compute and posts the consumed count, the
chain runs down a column from the relay's row (wrapping), then the next. K split: columns paired (2, 3), (4, 5); chain A
the even columns (K half 0), chain B the odd (half 1); partials (8 bf16 tiles a sub-block) written by the BRISC into
the partner's 2-slot partial ring, credited back. West rectangle 4 x 10 = 40 cores, 48 sub-blocks (16 experts x 3).

| case | x blocks a sub-block | x ring | steady us / sub-block |
|---|---|---|---|
| today | 24 | 24 (one sub-block) | 21.56 |
| K split | 12 | 24 (two sub-blocks) | 22.04 |
| K split | 12 | 16 | 22.60 |
| K split | 12 | 12 | 25.37 |

The chain alone keeps up in both forms (the flat expert's 26.6 is not the x path on its own).

## Stage 3: + DRAM weight streams (w_stream)

Each core's NCRISC also reads its weights (the flat expert's gate_up_weight_read.hpp form): 2 reads in flight of 9.2 KB
(today 1 a block; K split 2 a block of 18.4 KB), transaction ids 8.., from its DRAM bank (round robin, HEIGHT_SHARDED
over the 8 banks; one expert's region re-read by every expert) into a 2-expert weight ring (CB 1, the NCRISC its
producer), freed after the expert's last sub-block.

| case | total (48 sub-blocks) | steady us / sub-block | weight GB/s |
|---|---|---|---|
| today | 1470.9 us | 27.89 | 96 |
| **K split, x ring 24** | **1147.4 us (-22%)** | **22.63 (-19%)** | 123 |
| K split, x ring 16 | 1384.9 | 27.77 | 102 |

- Today's form degrades to the flat expert's rate once the weights stream (21.6 -> 27.9 us): the x ring holds exactly one
  sub-block (row passes free it only in the last pass), so the next sub-block's x can only arrive during that pass, and
  the NCRISC that forwards x also issues the weight reads.
- The K split's x ring of the same 835 KB holds two sub-blocks: x for the next one arrives during the whole current
  one. With a 16-block ring the gain is gone - the slack, not the halved x bytes, is what matters.
- (Accuracy: a DRAM-sharded bfp4 upload rounds differently from the host conversion (3% of the values' ulp-level
  differences); the check uses the weights read back from the device. PCC 0.99993 every case.)

## Stage 4: the chip (plan_geometry: the flat plan's GLM e16 cores, banks, relays) and the down side

Reference, the real flat expert (tests/test_flat_ref.py, GLM e16, one chip): M 320 1186.1 us, **M 384 (48 full
sub-blocks, the model's case) 1359.1 us**, M 512 1763.2 us.

Geometry from the plan: 64 gate/up cores (west 4 x 10, east 3 x 8), gate/up index ci in bank ci // 8 (the plan's
min-max order), relays (1, 4) (8, 3) primaries, (1, 5) (8, 4) helpers; today one snake a rectangle from the primary's
row; K split pairs consecutive snake positions (chain A the even positions, fed by the primary; B the odd, the helper).
Down side (down_load, kernels kd_*.cpp): every gate/up core writes its h slice into the 7 down chain heads' h_all
(NOC1, 3 buffers) once GO allows (a coordinator down core bumps GO on all gate/up cores when all 32 down cores are done
with v); chains forward h in 16 pieces (4 in flight); down compute 1 x 6 blocks over K 64 (bf16 DEST), weights
streamed (2-expert ring, 27.6 KB blocks in 4 reads, 2 in flight) from each down core's plan bank; y written to DRAM.

| case (48 sub-blocks, weights streamed) | total | steady us / sub-block |
|---|---|---|
| today, gate/up only | 1208.9 us | 22.50 |
| K split, gate/up only, x ring 24 | 1194.9 | 22.77 |
| K split, x ring 16 | 1532.6 | 28.35 |
| K split, x ring 12 | 1724.9 | 32.99 |
| today + down side | 1508.5 | 29.51 |
| K split + down side | 1494.3 | 29.30 |

All bit-checked (h pcc >= 0.99992, y pcc >= 0.99972).

- With the plan's banks (nearest, min-max) today's gate/up runs at 22.5 us a sub-block: stage 3's 27.9 was the round
  robin bank assignment (far banks), not the x path. The K split is even with today (+0.3 us) and still needs the
  two-sub-block x ring (no L1 freed).
- So the real flat expert's ~25.5 us a steady sub-block (1359 us at M 384) is not gate/up's x feed. This model's
  down side is slower than the real one (29.5 us a sub-block, 1508 us): it paces both forms equally.
- Conclusion for GLM: the K split is not a lever; the gap to the floor sits in the coupling with the down side / h path
  (or in what the real kernels do beyond this model). Next: find the real flat's extra ~3 us a sub-block.

## Where the real flat expert's time goes (GLM e16 M 384; temporary local probes, not committed)

Probes (ZZ_FLAT_PROBE bit mask compiled into every flat kernel: the stream's bytes are not moved, its counters and
semaphores move as before; perf only):

| stream off | us | vs 1358 |
|---|---|---|
| none | 1358.2 | |
| h to the down chain heads / along the chains / both | 1359.5 / 1352.8 / 1352.0 | -0.5% |
| down weight reads | 1339.9 | -1.3% |
| gate/up weight reads | 1298.9 | -4.4% |
| both weight streams | 1292.5 | -4.8% |
| y writes | 1359.6 | 0 |
| x forwarding along the snakes | 1317.8 | -3.0% |
| relays (primary -> head / helper -> primary / helper DRAM reads / all three) | 1348 / 1355 / 1356 / 1339 | -0.3..-1.4% |
| all x / x + both weights / everything | 1307.7 / 1257.5 / 1245.6 | -3.7 / -7.4 / -8.3% |

- With no data at all the expert still takes 1246 us: data costs ~110 us (gate/up weights ~60, x forwarding ~40,
  down weights ~18, h ~6), roughly additive.
- Inside the gate/up compute (no data): the activation off -0.2%, the per-pass pack reconfig off 0, the matmul off ->
  962 us (the down side then paces). The ARC firmware's matmul throttle (matmul_block_math_dynamic_throttle) forced off:
  no change (it never engages here).
- The profiler's GU-SB zone (26.6 us) is inflated by the profiling itself: the unprofiled totals are what compare.
  Perf counters (FPU|PACK|UNPACK|INSTRN), gate/up cores: FPU busy 71.5% of the window in the real expert without data,
  75.1% in this model with data (today_chip, 1209 us); no srcA / srcB valid waits in either.
- One core, chain-structured (today_1core): 21.15 us a sub-block resident (bench 20.35: +0.8 for the [4 x 8] x blocks,
  their CB syncs and ring offsets), 21.83 with its weights streamed (+0.7).

Budget of the 1358 us (874 us = the matmuls at LoFi peak): matmul efficiency at 89.5% ~100 us, sub-block structure
(block syncs, weight ring) ~70 us, multi-core coupling ~70 us, fill / drain ~100 us, data movement ~110 us.

## Why one core gets 90% and the expert with every stream off does not: a runtime rt_dim

- Same GU-SB zone in the one-core bench: 20.29 us (the profiler does not inflate it); the expert with all streams off
  26.6 us a sub-block - a real 6.3 us. Sum zones in the real kernel: unpacker CB waits ~2.1 us a sub-block (weights
  1.7, mostly the zones' own cost), the MATH thread's DEST waits 1 us a call, the packer's 1 us a call.
- The real gate_up_compute.cpp alone on one core (test_flat_compute_alone.py: the plan's named args, x / weight blocks
  handed over as fast as the rings free, kf_feed / kf_drain): 28.2 us a sub-block - the kernel, not its surroundings.
- The difference: its matmul_block gets the pass's row tiles (rt_dim) as a runtime value (computed from the
  schedule's rows); a compile-time rt_dim lets the LLK unroll its per-tile loops. Bench, GLM 2 x 2 blocks, K 192:
  compile-time 17.89 cycles a tile product (89.5%), runtime 20.82 (76.8%). (Perf counters: the MATH thread has no
  instruction queued 25% of the call in the expert vs 7% in the bench.)
- Fix (gate_up_compute.cpp matmul_x_block<FULL>): full sub-blocks / passes use the compile-time row count, a short
  last sub-block the runtime one. The kernel alone: 22.9 us a sub-block (bench 20.3; the rest is its [4 x 8] x
  blocks' syncs and ring offsets). Flat checks: 67 passed (accuracy, determinism). Perf, LoudBox, % vs before:
  GLM-5.3 8x4 5120 hot -6.1, 10240 uniform -4.2, zipf -2.6, 5120 zipf -1.1, 4x4 7680 -1.0; GLM e16 (test_flat_ref)
  M 320 1186 -> 1141 (-3.8%), M 384 1358 -> 1288 (-5.2%), M 512 1763 -> 1661 (-5.8%); K3, DSv4.1-Flash, GLM-Flash,
  K2.7, M3 within +-0.6% (their gate/up compute is not what paces them).
- Other one-core matmul facts (test_kb_mm): the activation on the PACK thread costs ~2.3 points (FPU busy 88.3 ->
  90.1% without it); matmul_block_k (all K steps one call) changes nothing; the row MOP with 1-row passes 17.51 cycles
  (91.4%); unpacker A busy 98.5% at 2 x 2 (it delivers 2 weight + 2 x tiles every 4 tile products).

## Gate/up kernel structure (the unpack thread's own work a block)

Kernel alone (test_flat_compute_alone.py, slope over 8 / 16 experts, us a sub-block; bench 20.29): after the rt_dim
fix 22.13 (22.90 average). Counters a sub-block vs the bench: identical FPU / MATH / unpacker-busy cycles; the extra
~2.1 K cycles are all on the unpack thread - fewer cycles with an unpacker instruction queued (7.59 vs 8.90 K) and
waits for the unpacker to go idle (0.81 vs 0 K: the per-block x pops).

| change (cumulative) | alone us / sub-block |
|---|---|
| x waited for only in the first pass, ring slot stepped (no modulo), a weight block waited for once | 22.13 |
| + x waits / pops in groups of 4 / 8 / 2 / 12 / 24 | 21.81 / 21.69 / 21.97 / 21.85 / 22.17 |
| + the load's weight offsets once a pass (a load is region-aligned and freed whole) | 21.62 (4) / 21.51 (8) |
| + matmul_block_k for a block's 8 K steps | 21.51 (no change) |

In the expert (GLM e16, us at M 320 / 384 / 512; before 1141 / 1288 / 1661): groups of 4 1141 / 1287 / 1633, of 8
1223 / 1418 / 1812; waits grouped 4 / 8 with per-block frees 1164 / 1307 / 1673 and 1205 / 1370 / 1759; waits 4 +
frees 2 1146 / 1291 / 1643; **block by block 1133 / 1275 / 1632**. Grouping costs the x chain its slack (the ring holds
exactly one sub-block). Kept: block-by-block x waits (first pass only) and frees, stepped slot offsets, weight offsets
once a pass, each weight block waited for once. Flat checks 67 passed; LoudBox perf vs the rt_dim fix: GLM-5.3 8x4
10240 uniform -1.7%, 4x4 7680 -0.8%, 5120 hot / zipf -0.6%, the rest within +-0.4%; GLM e16 1136 / 1276 / 1634 us
(-0.4 / -0.9 / -1.6%); K3 e28 / e56 unchanged.

The kernel alone gains 3-6% but the expert only 0.4-1.7%: at GLM's loads the expert is paced by x delivery along the
chain, not by the gate/up kernel.

## The combine's traffic floor: a bare fabric all-to-all (a2a.py, kernels/a2a_*.cpp, tests/test_a2a.py)

GLM-5.3 16 experts a chip, M 384 (the combine's case): every chip sends 768 rows of 12320 B (6144 bf16 + the 32 B
trailer) to each of the other 7 chips, 66.2 MB out a chip, as fabric unicasts: one sender core a (direction, link)
with its own connection (the rows leave by their first hop, Y then X), rows landing round-robin on 20 cores a chip,
the last row of each flow bumping the destination's counter. LoudBox 2 x 4, FABRIC_2D, 200G, device time after a
mesh barrier:

| case | us | GB/s out a chip |
|---|---|---|
| 12320 B rows, router payload 12320 | 1863.7 | 35.5 |
| 12320 B rows, router payload 15232 | 1869.6 | 35.4 |
| 6560 B rows (bfp8-size), payload 6560 | 994.6 | 35.5 |
| 12320 B rows, 1 link a direction | 3683.5 | 18.0 |

- Link-bound: time follows the bytes and halves with the links. The 2 x 4's middle X links carry ~75.7 MB a direction
  (8 flows x 768 rows): 1.86 ms is ~20 GB/s a link, 86% of 200G. The fabric has 2 channels between neighbours here
  (link_idx 2: out of bounds), so 2 links is the ceiling.
- The combine at this load: 2252 us decoupled (+21% over the bare traffic), 2556 us with the experts (+37%).
