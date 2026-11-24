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
