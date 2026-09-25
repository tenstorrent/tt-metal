# MiniMax-H3 on Wormhole Galaxy: the to_out AGMM — baseline, zones, experiments, results

The attention output projection, `ttnn.experimental.all_gather_minimal_matmul_async` with the fused addcmul epilogue
(`out = residual + gate * (x @ w)`). Measured 2026-09-21 on the 4x8 Wormhole Galaxy `UF-EV-B12-GWH02`, branch
`minimax_h3_wh_optimizations`, with the generic op tools written that day. Block-level context: [README.md](README.md);
the ff1 investigation this follows: [ff1.md](ff1.md). The one AGMM of the block whose K loop waits on operand
delivery.

## 1. The op and the baseline

| | |
|---|---|
| model call | `ColParallelLinear(7168, 5376, bias=False)` with `addcmul_a=residual`, `addcmul_b=gate`, `addcmul_scalar=1.0`, `attention_minimax_h3.py:155-163, 619-626`; AGMM branch `models/tt_dit/layers/linear.py` |
| shape per device | M = 13664 rows, K = 7168 (K_local 1792, gathered over the TP = 4 ring), N = 1344 |
| config | fused addcmul (`scalar`, `addcmul_input_tensor1` = a [M, N], `addcmul_input_tensor2` = b [M, N], the per-token gate), bias = False, bf16, HiFi2, `fp32_dest_acc_en = True`, `packer_l1_acc = True`, `math_approx_mode = True` |
| grid / blocking | 8x8 worker grid, `AGMM_BLOCK_SIZES[(7168, 1344)] = (8, 8, 6)`, subblock 2x2, 4 links, 2 workers per link; N per core is 6 tiles (5.25 useful), one N block |
| epilogue | `add_bias_and_addcmul_block` (`compute.cpp:136`): two full passes over the fp32 intermediate through L1 -- multiply by `b` and the scalar back into `intermediate_cb`, then add `a` into `out_cb` -- because `unary_bcast_tile` did not work under fp32 dest |
| roofline | compute **2.01 ms** (0.26 TFLOP), DRAM 0.75 ms, fabric 1.47 ms; compute-bound on Wormhole, fabric-bound on Blackhole at 2 links |
| blocking sweep (2026-09-17, 352 combos) | shipped (8,8,6) 4,332.8 us; best (14,8,6) 4,312.8 (0.5%); default kept. **Caveat found 2026-09-21: swept with the harness's `plain` use case, i.e. without the addcmul epilogue and with `math_approx_mode=False`.** The harness now has a `to_out` row (`13664_7168_1344_8x8_agmm_to_out`) that runs the real epilogue |
| baseline | **4.3 ms** in the 2026-09-17 roofline cross-check (the plain number, 46%); **5.29 / 5.31 ms** device kernel with the real epilogue (two harness runs, 2026-09-21; includes the harness's ~50 us bias add), 38% of peak |
| mesh bench (`transformer_op_mesh_bench.py --op to_out`, host-timed, 10 calls) | fused 5.43 ms / plain 4.56 ms: **the epilogue costs 0.87 ms** |
| single device (`transformer_op_single_device_bench.py --op to_out`) | plain `minimal_matmul` 4.06 ms; `dit_minimal_matmul_addcmul_fused` (a one-pass epilogue kernel) 4.34 ms |

## 2. Baseline breakdown by zone

Tracy device zones on the AGMM compute kernel through the sweep harness (`agmm_compute_zones.py apply
block|opwait|sampled`, shape id `13664_7168_1344_8x8_agmm_to_out`, combo (8,8,6) 2x2), per core.

| zone | per core, us | share | how |
|---|---|---|---|
| K loop (`KLOOP`; 196 iterations = 7 M blocks x 1 N block x 28 K blocks, 23.3 us each) | **4,558** | 87% | block zone |
| epilogue (`EPILOGUE_ADDCMUL`, 7 blocks) | **705** | 13% | block zone |
| kernel (`TRISC-KERNEL`) | 5,266 | | |
| operand wait per iteration (`OPWAIT`, first ~100 iterations) | **5.75 us (25% of the iteration, ~1.1 ms per call)** | | opwait zone on the unpack thread |

Sampled 2x2 subblock (every 30th K iteration, 8 K tiles = 32 tile-MACs), cycles: MATH acquire 21 / MAC **1508** /
UNPACK busy 1570 / PACK wait 25 / PACK 1432. That is **47 cycles per tile-MAC** on the math thread and 358 per fp32
L1-accumulate pack, the same pipeline pace as ff1 and to_qkv; the MAC time explains 18 of the 23.3 us per iteration
and the operand wait the rest.

### Decomposition

| component | ms | how |
|---|---|---|
| FPU work at HiFi2 peak | 2.0 | roofline |
| grid padding (N per core 5.25 -> 6 tiles, M 54 -> 56) | 0.4 | issued tile-MACs / useful |
| 2x2 fp32 pipeline issue efficiency (47 vs 32 cycles per tile-MAC) | 1.1 | sampled zones |
| **operand waits on the store-and-forward relay** | **1.1** | `OPWAIT` |
| **epilogue (addcmul, two passes)** | **0.7** | block zone; 0.87 by mesh A/B |
| **total** | **~5.3** | device 5.29 |

Why this op waits and the other two do not: its N per core is 6 tiles in one N block, so every delivered in0 byte
feeds fewer tile-MACs. At the pipeline's pace it would consume operands at ~12.8 GB/s per core and the relay chains
(in0 down a column of 8 cores, in1 across a row, store-and-forward with a semaphore round trip per hop) deliver ~10,
so the unpack thread waits a quarter of the loop.

## 3. Experiments, by zone

Mesh numbers are `transformer_op_mesh_bench.py`, host ms over 10 calls (about 0.2 ms above device time for this op);
PCC / rel-RMSE vs fp32 torch, bar 0.9995 / 0.02.

### 3.1 Operand delivery (1.1 ms)

| lever | result | status |
|---|---|---|
| relay prefetch (`tools/agmm_relay_prefetch.patch`: receivers request block k+1 right after handing block k to compute, before waiting for the downstream hop) | fused 5.52 vs 5.43 ms, plain 4.60 vs 4.56, fp32-off 4x2 5.32 vs 5.23: **no gain** (slightly worse, within noise) | rejected: the patch removes request latency, not bandwidth; the relay is bandwidth-bound at ~10 GB/s per core |
| fewer in1 re-reads: M_block 16 (4 M blocks instead of 7) | fp32 off (16,8,6) 4x2: mesh 5.22 vs 5.23; (16,8,6) with fp32 on overflows L1 | no gain: in0, not in1, is the delivered volume that matters |
| K_block 14 (fewer, larger hops) | fp32 on (8,14,6) 2x2: 5.43 (0); single device 4.00 vs 4.10; the 09-17 sweep had (8,14,6) at 4,568 us, slower | rejected |
| in0 multicast to the chain instead of store-and-forward (each hop today re-writes 128 KB per iteration), or the fabric-bound strided AGMM path (`strided_all_gather_minimal_matmul_async`, no Wormhole config exists) | not run | **proposed: the only lever of size**, dataflow work |

### 3.2 The K loop (1.1 ms of pipeline issue cost)

| lever | single device (plain) | mesh (fused) | status |
|---|---|---|---|
| fp32 dest off, same (8,8,6) 2x2 | 4.10 -> 3.88 | **5.43 -> 5.24 ms (-3.5%)**, plain 4.56 -> 4.37; rel-RMSE 0.0056 -> 0.0086 | measured; precision decision |
| fp32 dest off + 2x3 subblock | 3.70 | -- | |
| fp32 dest off + 4x2 subblock | 3.66 | 5.23 | the single-device gain does not survive the relay: the loop waits more instead |
| fp32 dest off + M_block 16, 4x2 | 3.65 | 5.22 | same |
| LoFi (diagnostic; the 2.0 ms of math halves) | 4.10 -> 3.61: -0.5 ms for -1.0 ms of math | -- | delivery/unpack-paced |

Compute-side levers are absorbed by the waits on the mesh: single-device to_out drops to 3.65 ms while the mesh
stays at 5.22. fp32 dest off is worth 3.5% here, against 8-9% on ff1 and to_qkv.

### 3.3 The epilogue (0.7 ms)

| lever | result | status |
|---|---|---|
| plain vs fused on the mesh | 4.56 vs 5.43 ms: 0.87 ms | measured (the zone says 705 us per core) |
| fp32 dest off does not shrink it | fused 5.24 vs plain 4.37: still 0.87 | measured |
| single-device one-pass reference (`dit_minimal_matmul_addcmul_fused`) | 4.34 vs plain 4.06: 0.28 ms for the same math | the target cost |
| **one DST pass**: `mul_tiles(interm, b)` -> DST0, `mul_unary` by the scalar only when it is not 1.0 (the model passes 1.0), `copy_tile(a)` -> DST1, `add_binary_tile(DST0, DST1)`, one pack to `out_cb`; halves the unpack/pack traffic over the fp32 intermediate | ~-0.35 ms expected | **proposed** |

## 4. Results

| step | per call | status |
|---|---|---|
| baseline as the model runs it, (8,8,6) 2x2 fp32 dest, fused addcmul | 5.3 ms (38%) | shipped; the 4.33 ms in the 09-17 sweep table was the plain epilogue |
| fp32 dest off | 5.1 ms (mesh host 5.24 vs 5.43) | measured; precision decision |
| + one-pass epilogue | ~4.7 | proposed |
| + in0 multicast (waits removed) | ~3.6 | proposed, dataflow work |
| roofline | 2.0 | |

Per forward: to_out 5.3 -> ~4.2 ms (epilogue + fp32 off) would be ~55 ms of the 12.4 s; with the delivery fix ~85 ms.

## 5. What is left

1. **The one-pass addcmul epilogue** (§3.3): ~0.35 ms, no numerical change, kernel work in `compute.cpp`.
2. **A higher-bandwidth in0 path** (§3.1): in0 multicast or the strided AGMM; the only lever that reaches the 1.1 ms
   of waits, and the reason the relay-prefetch patch kept for ff1 can be retired for good.
3. **fp32 dest off**: -3.5%, a precision decision; here it should wait for the delivery fix, which is what makes the
   larger subblocks pay.

## 6. Tooling and recipes

Everything below is shared with [ff1.md](ff1.md) §6 (mesh reproducer, sweep-harness timing, device zones, recovery);
only the op-specific invocations are listed here.

```bash
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out [--no-fusion] [--fp32-dest 0] [--gate-broadcast]
python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_single_device_bench.py --op to_out --no-fusion --cases "8,8,6,2,2,1;8,8,6,4,2,0" [--fidelity LoFi]
# zones through the harness (apply, run, parse, revert):
python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py apply block   # or opwait / sampled
MM_SWEEP_EXPLICIT_COMBOS='[[8,8,6,2,2]]' MM_SWEEP_PROFILER_DUMP_EVERY=100000 python -m pytest \
  models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep -k "13664_7168_1344_8x8_agmm_to_out and wh_4x8_ring" -s --timeout 7200
python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py parse generated/profiler/mm_sweep_wh_4x8_ring_13664_7168_1344_8x8_agmm_to_out/reports/<ts>/profile_log_device.csv
python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py revert
# the relay prefetch A/B: git apply models/tt_dit/tests/models/minimax_h3/tools/agmm_relay_prefetch.patch, run the mesh bench, git checkout -- the two dm_*.cpp kernels
```
