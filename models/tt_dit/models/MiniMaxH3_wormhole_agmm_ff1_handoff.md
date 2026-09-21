# MiniMax-H3 on Wormhole Galaxy: taking the ff1 AGMM to roofline — handoff (2026-09-18, updated 2026-09-21)

Branch `jameslee/exp_ring_sdpa_wh`. Written for whoever picks up the AGMM work next; everything below was
measured on this 4x8 Wormhole Galaxy (UF-EV-B12-GWH02) on 2026-09-18/19. **Status 2026-09-19:** lever §4.1 A
(bf16-grade silu) is landed in both kernels, ff1 is 15.85 → 15.29 ms on the mesh; the "mesh hang" it was
blamed for was a semaphore-reuse race in the test tooling (exp 8/9), now fixed. Per-thread zones then showed the
K-loop gap is **pipeline issue efficiency of the 2x2 fp32 subblock** (math, unpack and pack all ~95% busy, no
waits), not operand delivery: §4.2 is retired as a lever (exp 10-12), and what is left is in §4.3 and §5. The investigation write-up that this
hands off from is the section *ff1 AGMM: where the other 49% goes* in `MiniMaxH3_wormhole_perf.md`; this
file adds the SwiGLU findings made after it, the exact recipes for each lever, and the tooling.

## 1. The op and the target

`ttnn.experimental.all_gather_minimal_matmul_async` (AGMM) for ff1, per device at 15 s / 768P / 16:9:

| | |
|---|---|
| shape | M = 13664 rows, K = 5376 (K_local 1344, gathered over the TP = 4 ring), N = 7168 packed gate\|up, out N = 3584 |
| config | fused SwiGLU, bias = False, bf16 in/out, HiFi2, `fp32_dest_acc_en = True`, `packer_l1_acc = True`, `math_approx_mode = True` |
| grid / blocking | 8x8 worker grid (bottom row of the 8x9 chip holds the in0 muxes), `(M_block, K_block, N_block) = (8, 7, 10)`, subblock 2x2, 4 links, `num_workers_per_link = 2`, `num_buffers_per_channel = 48` |
| call sites | `models/tt_dit/layers/linear.py:395-467` (`ColParallelLinear.forward`, AGMM branch); blocking from `models/tt_dit/utils/matmul.py:140` (`grid_88_configs`); compute config `transformer_block_minimax_h3.py:145-151` |
| kernels | `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/` — `all_gather_minimal_matmul_async_program_factory.cpp`, `kernels/compute.cpp`, `kernels/dm_in0_sender.cpp`, `kernels/dm_in1_sender_out.cpp`, `kernels/matmul_dataflow_common.hpp` |
| roofline | compute **8.03 ms** (2·M·K·N = 1.053 TFLOP at 64 cores x 2048 FLOP/cycle x 1.0 GHz = 131.1 TFLOP/s), DRAM 0.78 ms (224 MB at 288 GB/s), fabric 1.10 ms (13.8 MB per link at 12.5 GB/s) |
| measured | **15.7 ms** (15,632-15,925 us across the day's runs; 15,709.9 us in the perf doc) = 51% of HiFi2 peak; 50 calls per forward = 0.79 s of the 12.4 s forward |
| roofline script | `models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py` (`--dump`, `--figs all`, `--selftest`; `--ops agmm` is the default, `--ops all` adds ff2). Renamed from `agmm_roofline.py` on 2026-09-21; the op table lives in `minimax_h3_ops.py` |

`fp32_dest_acc_en` does **not** move the roofline: math fidelity fixes the FPU rate; the flag only sets the
DST accumulator format (halving DST to 4 tiles) and the pack format. LoFi halves the bar and is a quality
decision, used below only as a diagnostic.

## 2. Where the 15.7 ms goes (Tracy device zones, one core = every core)

Zones were added temporarily to `kernels/compute.cpp` and read back from `profile_log_device.csv` of a
sweep-harness run (recipe in §6). All 64 cores and all devices agree to within 1%.

| component | per core, us | share | how measured |
|---|---|---|---|
| K loop (`for k_block` in `compute.cpp:483-521`, 504 iterations = 7 M-blocks x 3 N-blocks x 24 K-blocks) | **12,934** | 82% | zone `AGMM_KLOOP` around the loop, math thread |
| SwiGLU epilogue (`swiglu_block`, `compute.cpp:28-70`, 21 calls) | **2,809** | 18% | zone `AGMM_SWIGLU` |
| output-CB reserve + intermediate wait | 0-26 | 0% | zone `AGMM_OUTWAIT` |
| outside all zones | 10-76 | 0% | kernel minus zones |
| **kernel** | **15,750** | | `TRISC-KERNEL` zone |

Inside the K loop (per-iteration zones, first 60 of 504 iterations fit the profiler buffer):

| | per iteration, us |
|---|---|
| pure math at HiFi2 (8x10 tiles x 7 K-tiles x 32 cycles) | 17.9 |
| `matmul_blocks` on the math thread (`AGMM_MM`) | **27.3** |
| operand wait on the unpack thread (`AGMM_OPWAIT`, `in0/in1 wait_front`) | **1.0** |

So at HiFi2 the operands arrive in time and the loss is inside `matmul_blocks`. Per-thread zones on one 2x2
subblock of every 30th K iteration (exp 12, `agmm_compute_zones.py apply sampled`; the compute kernel runs on all
three TRISCs, so each zone is timed on each thread) pin where:

| thread | zone | cycles per 2x2 subblock (7 K tiles = 28 tile-MACs) | nominal |
|---|---|---|---|
| MATH | `tile_regs_acquire` (wait for a free DST half) | **21** | |
| MATH | MAC issue (`matmul_block` x 7) | **1321** | 896 (28 x 32 at HiFi2) |
| UNPACK | unpack busy over the same 7 calls (28 tiles) | **1357** | |
| PACK | `tile_regs_wait` (wait for math to commit) | **26** | |
| PACK | pack 4 fp32 tiles with L1 accumulate | **1236** | |

Nobody waits: math, unpack and pack are each 90-100% busy at ~1350 cycles per subblock = 20 subblocks x 1.35 us =
27 us per iteration, the measured pace. Math issues at **47 cycles per tile-MAC instead of 32**, unpack at ~42
per tile, pack at **309 per fp32 tile**. The K-loop gap is the issue efficiency of the 2x2 fp32 subblock pipeline
(per-`matmul_block`-call setup on MATH and UNPACK, fp32 read-modify-write packs), and all three would have to
speed up together. This replaces the first draft's "operand delivery co-limits" reading: the LoFi result (only
-1.65 ms when FPU work halves, exp 11) is the unpacker pacing at LoFi, and a relay-protocol prefetch that removes
the request/forward serialization changes nothing at either fidelity (exp 10).

Inside the SwiGLU epilogue (single-device A/B on the identical `swiglu_block`, HiFi2, blocks (8,7,10)):

| variant | ms | delta vs full SwiGLU |
|---|---|---|
| plain matmul, no SwiGLU (copy epilogue) | 14.49 | |
| full SwiGLU | 16.45 | |
| SwiGLU without `silu_tile` | 14.29 | **-2.16 ms: silu is the whole cost** |
| SwiGLU without `mul_binary_tile` | 16.35 | -0.10 |
| no SFPU at all (2 copies + pack) | 14.23 | |
| two pairs per DST acquire, inits hoisted (see §4.1) | 16.45 | -0.10 (not worth its diff) |
| **`silu_tile<false>(...)` — bf16-grade silu under fp32 DST** | **15.52** | **-0.93 ms**, PCC 0.999928 vs 0.999930 (single device); **-0.56 ms on the mesh AGMM**, exp 9 |

`silu_tile` costs ~2,570 cycles per tile because under fp32 dest `_sfpu_sigmoid_<true>` uses the fp32-accurate
exp (Juffa) plus a two-iteration Newton reciprocal (`ckernel_sfpu_sigmoid.h`, `ckernel_sfpu_recip.h`); the
`<false>` instantiation uses the fast `_sfpu_exp_21f_bf16_` and one iteration. The SwiGLU output is packed to
bf16 anyway, so bf16-grade silu loses nothing measurable.

### Decomposition

| component | ms | lever |
|---|---|---|
| FPU work at HiFi2 peak | 8.0 | none (the roofline) |
| K-loop pipeline issue efficiency at 2x2/fp32 (math 47 vs 32 cyc per tile-MAC, unpack 42, pack 309 per fp32 tile; no waits) | 5.0 | §4.3 only; delivery is not a factor (§4.2) |
| SwiGLU epilogue (2.19 ms after §4.1 A; ~1.8 of the 2.6 kcycles per pair is the bf16-grade `silu`) | 2.2 | §4.1 B/C, both precision or design trade-offs |
| waits, fabric, dispatch | <0.1 | none |

## 3. Experiments already run (do not repeat)

| # | experiment | result |
|---|---|---|
| 1 | K_block 14 / 21 (never in the original sweep: its L1 pre-filter over-counts this op's CBs) | slower: (8,14,8) 16,227 us, (8,14,6) 16,281, (6,14,10) 16,898, (8,21,4) 18,131 vs 15,725; (8,14,10), (8,21,6) OOM at warm-up |
| 2 | fp32 dest off, mesh (`MM_SWEEP_FP32_DEST_ACC=0`) | (8,7,10) 2x2 15,162 (-3.6%); best (8,7,16) 2x4 **14,475 (-8.0%)**; (12,7,8) 4x2 14,677; (8,7,14) 2x2 14,756 |
| 3 | single device, same per-device shape (`transformer_op_single_device_bench.py --op ff1`, then `agmm_ff1_single_device_bench.py`) | `minimal_matmul`+SwiGLU 16.56 ms ≈ the mesh AGMM: the ring gather is hidden. Plain 14.52 → SwiGLU costs 2.0 ms. fp32 off (12,7,8) 4x2 plain 12.67 ms (63%). `ttnn.matmul` auto-config 8x8: 23.4 ms fp32 on, 15.2 off (no better) |
| 4 | fidelity scaling, single device | plain (8,7,10) fp32: LoFi 13.17 / HiFi2 14.48 / HiFi4 20.72 ms → ~12-13 ms non-math floor at this blocking; (12,7,8) fp32-off LoFi 10.73 |
| 5 | per-iteration zones | operand wait 1 us/iter, `matmul_blocks` 27.3 us/iter vs 17.9 math |
| 6 | per-block zones | K loop 12.9 ms, SwiGLU 2.8 ms, waits 0 |
| 7 | SwiGLU attribution and variants, single device | table in §2: silu 2.16 ms; batching 0.1 ms; `silu_tile<false>` -0.93 ms with PCC intact |
| 8 | `silu_tile<false>` in the AGMM kernel on the mesh, through the sweep harness | hung 3 of 3 times. **Root-caused 2026-09-19: not the kernel.** The harness reused one semaphore pair and one gathered-in0 buffer for every call, and its warm-up enqueues calls back to back (`sync=False`): a device that finishes call i early starts call i+1 and signals ring semaphores a neighbour is still consuming in call i. Reproduced with the **unmodified** kernel: 5 back-to-back calls with one shared set hang after the first completes. The model never sees this: `CCLManager.get_ag_ping_pong_semaphore/buffer` alternates two sets. Harness fixed to ping-pong (§6) |
| 10 | relay prefetch: in0/in1 receivers request block k+1 right after pushing block k to compute, before waiting for the downstream hop's request (60-line reorder of `dm_in0_sender.cpp` / `dm_in1_sender_out.cpp`, `tools/agmm_relay_prefetch.patch`) | correct (PCC identical, no hang) and **no gain**: HiFi2 16.08 → 16.06 ms, LoFi 14.52 → 14.68. Delivery does not pace the loop. Not landed |
| 11 | blocking sweep with the bench, HiFi2 host ms: (8,7,10) 2x2 **16.06**; (8,7,10) **4x1 18.28** (same 4 DST tiles, 5 unpacks per K step instead of 4); (4,7,10) 23.68; (4,14,10) 23.37 (doubling K_block = halving fp32 L1-acc pack passes: -1.3%); plain (8,7,10) 14.30; plain LoFi 12.65 | non-math time scales with tile-MACs and with the unpack count, not with iterations or pack passes → issue-bound pipeline, see §2 |
| 12 | per-thread sampled zones (`agmm_compute_zones.py apply sampled`) | table in §2: ACQ 21 / MAC 1321 / UNPACK 1357 / PWAIT 26 / PACK 1236 cycles per subblock |
| 13 | block zones after §4.1 A (`apply block`) | KLOOP **13,034** us, SWIGLU **2,191** us per core (was 12,934 / 2,809): the silu change removed 618 us on the mesh; 2,608 cycles per gate/up pair remain |
| 9 | `silu_tile<false>` on the mesh with ping-pong semaphores (`transformer_op_mesh_bench.py --op ff1`, then `agmm_ff1_mesh_bench.py`, 10 back-to-back calls) and through the fixed harness | **runs.** Host 16.58 → 16.08 ms per call; device kernel (harness, Tracy) **15,852 → 15,289 us (-563 us, -3.6%)**; PCC 0.9999838 → 0.9999834, rel-RMSE 0.00806 → 0.00828 (bar 0.9995 / 0.02); `test_linear_swiglu` 4/4 at 0.99998. **Landed** in both `compute.cpp` |

Numerics baseline (perf doc, ff1 real ring op at M=13664): pcc 0.9999843, rel-RMSE 0.00837; bar pcc > 0.9995,
rel-RMSE < 0.02. Single-device SwiGLU vs fp32 torch: fp32 dest 0.99993 / 0.0087, fp32 dest off 0.99985 / 0.0169.

## 4. Levers, with the change to make and how to test it

### 4.1 SwiGLU epilogue (2.8 ms → ~1.0 ms possible, no numerical change of consequence)

**Change A — bf16-grade silu (one token).** In `all_gather_minimal_matmul_async/device/kernels/compute.cpp`
`swiglu_block`, and the identical function in `minimal_matmul/device/kernels/compute.cpp`:

```cpp
-            silu_tile(GATE_DST);
+            silu_tile<false>(GATE_DST);  // bf16-grade exp + 1 NR step: the output is packed to bf16 anyway
```

**Landed 2026-09-19** (exp 9): -0.93 ms single device, **-0.56 ms on the mesh AGMM** (15,852 → 15,289 us), PCC
unchanged to the fourth decimal. The mesh gain is smaller than the single-device one; the remaining epilogue
(~2.3 ms) is the target of change C.

Why the earlier "mesh hang" was not this change: on Wormhole `SFPU_UNARY_CALL` **ignores** its `DST_ACCUM`
argument (`llk_math_eltwise_unary_sfpu_macros.h:43-46` only forwards the functor, DST index and vector mode to
`_llk_math_eltwise_unary_sfpu_params_`), so `silu_tile<false>` changes nothing but the sigmoid's exp
(`_sfpu_exp_21f_bf16_`, inline constants, no programmable-register dependency) and its reciprocal iteration
count; DST addressing is untouched. The hang was the harness's shared semaphore set (exp 8). The DST-addressing
triage list in the first draft of this file is therefore moot and has been removed.

**Change B — cheaper sigmoid.** `calculate_sigmoid_appx` (`ckernel_sfpu_sigmoid_appx.h`) is a 3-segment LUT
(~5 SFPU instructions per vector vs ~30). Accuracy is the question: slope 0.2266 near zero against the true
0.25 puts ~1e-2 absolute error on the sigmoid, likely too much for the rel-RMSE < 0.02 bar. Try only if A
lands and more is needed; validate at the model level (CLIP / VBench), not just op PCC.

**Change C — overlap the epilogue instead of shrinking it.** `silu_tile_pack` / `silu_tile_init_pack`
(`compute_kernel_api.h:680-689`) run the SFPU from the **pack** thread. Moving the epilogue's SFPU work to the
pack thread lets the math thread start the next block's K loop while the previous block's SwiGLU drains,
using the two DST halves naturally (math on one half, pack+SFPU on the other). Needs a `mul_binary_tile`
pack-thread variant or reordering so the multiply is done on the math thread before commit. Largest gain
(the whole 2.8 ms hides behind the next K loop) and the most invasive change; measure on the single-device
kernel first.

Not worth it: two pairs per DST acquire with hoisted inits (-0.1 ms, measured).

### 4.2 Operand delivery — measured, not a lever (exp 5, 10, 12)

The store-and-forward relay (in0 down a column chain of 8 cores, in1 across a row chain, one request/response
semaphore round-trip per hop per K block; head cores read DRAM; `dm_in0_sender.cpp:446-476`,
`dm_in1_sender_out.cpp:511-541`) keeps up: `in0/in1 wait_front` is 1 us per iteration (exp 5), and a prefetching
receiver that decouples the upstream request from the downstream forward is correct but changes nothing at HiFi2 or
LoFi (exp 10; the patch is kept at `tools/agmm_relay_prefetch.patch` in case a faster compute pipeline ever exposes
delivery). Two ideas in the first draft do not survive the numbers: **double-buffering the out CB** (the out-CB
reserve wait is 0-26 us total, exp 6) and **keeping in1 resident across M blocks** (that needs the whole
K x N_block panel, 168 x 10 tiles = 3.4 MB per N block at K = 5376, not one 140 KB block; L1 is 1.4 MB). Multicast
would cut relay traffic but the loop is not waiting on it. Revisit only if §4.3 lowers the compute pace below ~20 us
per iteration.

### 4.3 Compute pipeline (the 27 us per iteration vs 18 us of math): what is left

The pace is set by three ~equally loaded threads (§2), so any single fix moves the pace only as far as the next
thread. In order of evidence:

- **fp32 dest off for ff1 only.** DST then holds 8 bf16 tiles per half, so the subblock can be 2x4 (fewer
  `matmul_block` calls per tile-MAC, better in0 reuse on the unpacker) and the partial packs are 2 KB bf16 instead
  of 4 KB fp32 with L1 accumulate. Measured on the mesh: (8,7,16) 2x4 **14,475 us (-8%)** before the silu change.
  `ParallelFeedForward.forward` (`models/tt_dit/layers/feedforward.py:125-131`) hands one compute config to ff1 and
  ff2; give ff1 its own or set `self.ff.ff1.compute_config` (`linear.py:232`, used at `:442`), and pick the
  (8,7,16) 2x4 entry in `matmul.py:140`. Roofline unchanged; accumulator error doubles (rel-RMSE 0.0087 → 0.0169
  on the single-device SwiGLU output, bar 0.02) — a precision decision that needs a model-level check (CLIP /
  VBench), not only op PCC.
- **`dst_full_sync_en`** (8 fp32 tiles, one bank) is not a lever: it removes the math/pack overlap, so a subblock
  costs MAC issue + pack serially (~2240 + 2470 cycles per 8 tiles = 590 per tile vs 340 today). Not worth a run.
- **Larger K_block** does not help while the pace is per tile-MAC (exp 1, exp 11); only relevant if the pack
  became the pacer.
- **LLK-level:** the per-call setup of `matmul_block` for ct=rt=2 on MATH (47 vs 32 cycles per tile-MAC) and on
  UNPACK (~42 per tile) is where the 5 ms sits. Options are a `kt_dim`-looping variant so one call covers the 7 K
  tiles of a subblock (today `compute.cpp:matmul_blocks` calls `matmul_block` once per K tile with `kt_dim` used
  only as an index stride), or MOP/REPLAY programming across the K loop. This is tt-llk work on
  `llk_math_matmul` / `llk_unpack_AB_matmul` for Wormhole, not a kernel edit; the sampled zones above are the
  measurement to hold it against.
- **The epilogue (2.19 ms).** Change B (LUT sigmoid, `calculate_sigmoid_appx`) is the only remaining large cut
  (~5 SFPU instructions per vector vs ~25); its ~1e-2 error on the sigmoid is a precision decision like fp32 off.
  Change C (pack-thread SFPU overlap) does not fit under fp32 dest: the 2x2 matmul subblock uses all 4 fp32 tiles
  of a DST half, so there is no DST room to hold a gate/up pair alongside the MACs; it becomes possible only
  together with fp32 off (8 bf16 tiles per half).

### 4.4 Not levers (measured or bounded)

K_block (exp 1, 11); operand delivery and the relay protocol (exp 5, 10, 12); out-CB depth (exp 6); the ring
gather (hidden, exp 3/5); fabric bandwidth (1.1 ms bound, 7% used); aggregate DRAM (0.78 ms bound); LoFi (halves
the bar); `dst_full_sync_en` (serializes math and pack); the grid (mux row fixed by the op).

## 5. Path to roofline

| step | K loop | epilogue | total | util | status |
|---|---|---|---|---|---|
| 2026-09-18 | 12.9 | 2.8 | 15.7 | 51% | |
| **§4.1 A landed (2026-09-19)** | 13.0 | 2.2 | **15.3** (device 15,289 us) | 53% | done, numerics unchanged |
| + fp32 dest off, (8,7,16) 2x4 | ~11 | ~1.5 | **~13** (14,475 measured before A) | ~60% | precision decision, model-level check needed |
| + LUT sigmoid (§4.1 B) | ~11 | ~0.5 | ~12 | ~65% | precision decision |
| + LLK `matmul_block` issue efficiency to ~35 cyc/tile-MAC | ~9.5 | | ~10-11 | 75% | tt-llk work |
| roofline | 8.0 | 0 | 8.0 | 100% | |

Without a precision decision the kernel is at its practical floor for this blocking: the remaining 5 ms of K loop is
the per-tile issue cost of the 2x2 fp32 subblock on all three TRISCs, and the 2.2 ms epilogue is SFPU-bound with the
accurate-enough sigmoid.

Per forward (50 blocks) each ms per call is 50 ms; 15.7 → 10 ms is ~0.29 s of the 12.4 s forward, 2.3%.
to_qkv and to_out share the kernel without SwiGLU; measured 2026-09-21 (§7): §4.3 carries over to to_qkv (fp32
dest off + 4x2: -8.7% on the mesh), but to_out is co-limited by operand delivery, which §4.2 had ruled out for ff1.

## 6. Tooling and recipes

**Mesh reproducer, 15 s per run (use this first).** `models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff1`
(2026-09-21: the generic successor of `agmm_ff1_mesh_bench.py`; `--op to_qkv|to_out|ff2` runs the block's other
matmuls with their own fusion, shape and blocking from `minimax_h3_ops.py`) runs the op on one TP ring (the harness's
4x1 cluster submesh) exactly as the model does: fused SwiGLU, `bias=None`, HiFi2, fp32 dest, (8,7,10) 2x2,
`math_approx_mode=True`, **two semaphore pairs and two gathered-in0 buffers alternated per call** like `CCLManager`.
Prints host ms per call over N back-to-back calls and PCC / rel-RMSE against fp32 torch on the first 2048 rows of
every device. `--blocks`, `--fp32-dest 0`, `--no-fusion` (was `--no-swiglu`), `--sync-each`, `--no-pingpong`
(reproduces the exp 8 hang). Wrap in `timeout 600`; a hang never prints "ms per call".

Two things that cost hours before they were found: (1) `ttnn.from_torch` of an **fp32** torch tensor of
13664x5376 to bf16 tiles takes ~160 s on the host (0.1 s from a bf16 torch tensor) — the first draft of this
bench sat at 100% CPU for 15 minutes before its first op call and looked like a hang; convert to bf16 in torch
first. (2) A ring op hung by the semaphore race (exp 8) also wedges the ETH heartbeat, so the *next* open fails
with "Timed out waiting for ETH heartbeat" until `tt-smi -r all` + 75 s; the two failures look alike from the outside.

**Mesh timing of one blocking (the validated runner; now ping-pongs semaphores and hands ttnn bf16 host tensors).** Appends rows to `sweep_results_mm.csv` (gitignored,
holds the 1419-row 15 s sweep; back it up first, fp32-off rows are indistinguishable from fp32-on rows):

```bash
MM_SWEEP_EXPLICIT_COMBOS='[[8,7,10,2,2]]' MM_SWEEP_PROFILER_DUMP_EVERY=100000 [MM_SWEEP_FP32_DEST_ACC=0] \
  python -m pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
  -k "13664_5376_7168_8x8_agmm_ff1_swiglu and wh_4x8_ring" -s --timeout 7200
```

About 20 s per combo including setup. Warm-up enqueues with `sync=False`, so a "1/N" progress line does
not mean a combo completed — a hang shows up as the run never finishing (with the ping-pong fix this no longer
happens for the AGMM; the non-AGMM branches were already safe, they have no ring semaphores). Wrap in `timeout 400` when testing
kernel changes. Note the harness passes a **bias** for `ff1_swiglu` (the model does not) and runs
`math_approx_mode=False` (the model runs True; silu ignores it).

**Device zones.** `models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py apply block|sampled` edits the
AGMM `compute.cpp` in place (exact-match, asserts if the kernel moved on), `revert` restores it, `parse <csv>` prints
per-thread mean cycles and per-core sums. `block` zones the K loop and whichever epilogue the op compiles (`SWIGLU`
for ff1, `EPILOGUE_ADDCMUL` for to_out, `EPILOGUE_COPY` / `EPILOGUE_BIAS` for to_qkv without / with a bias -- the
harness always passes one); `help` prints each AGMM's sweep `-k` id. By hand: include `"tools/profiler/kernel_profiler.hpp"` in `compute.cpp`
and wrap regions in `{ DeviceZoneScopedN("NAME"); ... }`; the sweep harness already runs under `tracy -p`, so zones land in
`generated/profiler/mm_sweep_wh_4x8_ring_13664_5376_7168_8x8_agmm_ff1_swiglu/reports/<ts>/profile_log_device.csv`.
The per-core buffer holds ~120 zone events per RISC, so use per-output-block zones (21 blocks x 3 zones) for
the full op and per-iteration zones only to sample the first ~60 iterations. Parse: skip the first header line,
`csv.DictReader`, filter `RISC processor type` in TRISC_0/1/2 (unpack/math/pack), keep the largest
`run host ID` per `PCIe slot`, pair `ZONE_START`/`ZONE_END` by `zone name` per core, sum. Zone overhead was
not measurable (15.74 vs 15.72 ms). The zones are not in the tree.

**Single device, fast loop.** `transformer_op_single_device_bench.py --op ff1` (was `agmm_ff1_single_device_bench.py`;
25 s per case; `--fidelity LoFi,HiFi2` for the delivery-floor test; `--cases` for blockings and fp32 on/off;
`--op to_qkv|to_out|ff2` for the other matmuls through `minimal_matmul_split` / `dit_minimal_matmul_addcmul_fused` /
`minimal_matmul`). Host tensors are bf16 since 2026-09-21, so its PCC reads ~0.99992 where the fp32-input version read
0.99993; the ms are unchanged. PCC for SwiGLU kernel changes:
`pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py::test_linear_swiglu`
(4 variants, 5 s).

**Recovery.** A hang wedges the Ethernet heartbeat. Kill the run (by pid; a `pkill -f` pattern that appears in
your own shell command kills your shell), then:

```bash
for p in $(ls /proc | grep -E '^[0-9]+$'); do ls -l /proc/$p/fd 2>/dev/null | grep -q tenstorrent && echo "pid $p"; done
tt-smi -r all      # ~1 min; ignore the CPLD < 1.16 warning
sleep 75           # links retrain; opening the mesh earlier fails with "Graph specified in MGD could not fit"
python -c "import sys; sys.path.insert(0,'.'); from models.tt_dit.utils.sweep_mm_block_sizes import *; p,m=open_mesh(resolve_config('wh_4x8_ring'),trace_region_size=0); print('OK'); close_mesh(p)"
```

One reset today came back with a degraded topology and needed a second reset plus the wait.

**Known harness gap.** `estimate_l1_kb` in `sweep_mm_block_sizes.py:683-717` models the out CB as
double-buffered full width plus a bias CB; the AGMM factory single-buffers it, halves it for SwiGLU and
allocates bias only with `use_bias`. Combos near the budget (K_block >= 14 at M_block 8) are filtered although
they build; they measured slower anyway (exp 1).

## 7. to_qkv and to_out (2026-09-21)

Measured with the generic tools (`transformer_op_mesh_bench.py --op ...`, `transformer_op_single_device_bench.py`,
`agmm_compute_zones.py apply block|opwait|sampled`, op registry `minimax_h3_ops.py`); full tables in the perf doc
section *to_qkv / to_out AGMM: attribution* and rows 17-18.

- **to_qkv** (10.3 ms, 58%) is ff1 without the SwiGLU: K loop 98% of the kernel at the same 47 cycles per tile-MAC,
  operand waits 3%, the `chunks=3` writer 0.08 ms, the copy epilogue 0.17 ms. **fp32 dest off with (8,7,12) 4x2:
  11.30 -> 10.32 ms on the mesh (-8.7%)**, rel-RMSE 0.0044 -> 0.0107 -- the same precision decision as §4.3's first
  bullet; take it for ff1 and to_qkv together.
- **to_out** is **5.3 ms** as the model runs it (fused addcmul, approx on), not the 4.33 ms the sweep recorded with the
  `plain` use case. 0.7 ms is the two-pass addcmul epilogue (`add_bias_and_addcmul_block`; a one-DST-pass version is
  ~-0.35 ms), and **~1.1 ms is the unpack thread waiting on the in0/in1 relay** (`OPWAIT` 5.75 us of every 23 us
  iteration): with 6 N tiles per core the op needs ~12.8 GB/s per core of operands at the pipeline's pace and the
  store-and-forward relay delivers ~10. The relay prefetch patch does not help (latency, not bandwidth), and every
  compute-side lever that works single-device (fp32 off 4x2: 3.88 -> 3.66 ms; M_block 16) is absorbed by the waits on
  the mesh (5.23 / 5.22 vs 5.24). fp32 dest off alone: -3.5%. The lever of size is a higher-bandwidth in0 path (multicast).
- Tool caveats: the harness always passes a bias, so its to_qkv zone is `EPILOGUE_BIAS` (the model runs `copy_block`,
  which the mesh bench does); `(16,8,6)` and `(8,14,12) 2x4` overflow L1 (`program.cpp:2097`) -- the bench prints a
  traceback and the next run is unaffected.
