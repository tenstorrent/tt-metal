# MiniMax-H3 on Wormhole Galaxy: the ff1 AGMM — baseline, zones, experiments, results

The first feed-forward linear of the transformer block, `ttnn.experimental.all_gather_minimal_matmul_async` (AGMM)
with the fused SwiGLU epilogue, and the work done on it 2026-09-18 to 2026-09-21 on the 4x8 Wormhole Galaxy
`UF-EV-B12-GWH02`, branch `minimax_h3_wh_optimizations`. This merges the ff1 handoff (2026-09-18, updated 09-19 and
09-21) and the block write-up's ff1 section (2026-09-18) into one document ordered as baseline -> zone breakdown ->
experiments per zone -> results, and adds two results measured on 2026-09-21 that had not been written down (the
6-segment LUT silu and the fp32-dest-off epilogue regression). Block-level context: [README.md](README.md); the two
sibling AGMMs: [to_qkv.md](to_qkv.md), [to_out.md](to_out.md).

## 1. The op and the baseline

`ttnn.experimental.all_gather_minimal_matmul_async` (AGMM) for ff1, per device at 15 s / 768P / 16:9:

| | |
|---|---|
| shape | M = 13664 rows, K = 5376 (K_local 1344, gathered over the TP = 4 ring), N = 7168 packed gate\|up, out N = 3584 |
| config | fused SwiGLU, bias = False, bf16 in/out, HiFi2, `fp32_dest_acc_en = True`, `packer_l1_acc = True`, `math_approx_mode = True` |
| grid / blocking | 8x8 worker grid (bottom row of the 8x9 chip holds the in0 muxes), `(M_block, K_block, N_block) = (8, 7, 10)`, subblock 2x2, 4 links, `num_workers_per_link = 2`, `num_buffers_per_channel = 48` |
| call sites | `models/tt_dit/layers/linear.py:395-467` (`ColParallelLinear.forward`, AGMM branch); blocking from `models/tt_dit/utils/matmul.py:140` (`grid_88_configs`); compute config `transformer_block_minimax_h3.py:145-151` |
| kernels | `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/` — `all_gather_minimal_matmul_async_program_factory.cpp`, `kernels/compute.cpp`, `kernels/dm_in0_sender.cpp`, `kernels/dm_in1_sender_out.cpp`, `kernels/matmul_dataflow_common.hpp` |
| roofline | compute **8.03 ms** (2·M·K·N = 1.053 TFLOP at 64 cores x 2048 FLOP/cycle x 1.0 GHz = 131.1 TFLOP/s), DRAM 0.78 ms (224 MB at 288 GB/s), fabric 1.10 ms (13.8 MB per link at 12.5 GB/s) |
| measured | **15.7 ms** (15,632-15,925 us across the day's runs; 15,709.9 us in the 2026-09-17 blocking sweep) = 51% of HiFi2 peak; 50 calls per forward = 0.79 s of the 12.4 s forward |
| roofline script | `models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py` (`--dump`, `--figs all`, `--selftest`; `--ops agmm` is the default, `--ops all` adds ff2); the op table lives in `minimax_h3_ops.py` |
| in the block | 3 AGMM calls = 32.25 ms in the 246.92 ms 2026-09-17 block (13%); 31.85 ms on 2026-09-21 after the silu landing |

`fp32_dest_acc_en` does **not** move the roofline: math fidelity fixes the FPU rate; the flag only sets the
DST accumulator format (halving DST to 4 tiles) and the pack format. LoFi halves the bar and is a quality
decision, used below only as a diagnostic.

Method: mine the 320-combo ff1 block sweep already on disk, read the op's kernels, then six on-device
experiments (all on this Galaxy, baseline re-measured in every run: 15,632-15,743 us). The block sweep and
the AGMM runs use `sweep_mm_block_sizes.py`; the single-device runs use a host-timed script (8 iterations
after warm-up, dispatch included, so absolute numbers are ~1-2% pessimistic).

### What the kernel does (source read, `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/`)

- No multicast. in0 (the gathered activation, a DRAM buffer) and in1 (weights, DRAM) each travel down a
  serial store-and-forward chain of 8 cores with a request/response semaphore round-trip per hop per
  K-block (`dm_in0_sender.cpp:446-476`, `dm_in1_sender_out.cpp:511-541`). Only the chain head reads DRAM,
  tile by tile with one barrier per block.
- Loop order m -> n -> k, so in0 is re-read from DRAM once per N block (3x) and in1 once per M block (7x):
  ~1.0 GB of DRAM reads per device per op, all issued by the 16 chain-head cores, and ~127 MB of relay
  writes received and re-sent by every core.
- Compute: DST holds 4 tiles under fp32 dest (`get_dest_reg_count`), so the subblock is 2x2 and every
  K_block (7 tiles) the partial sums are packed as fp32 into an L1 intermediate CB with L1 accumulate:
  36,288 accumulating fp32 packs per core against 254,016 tile-MACs. `packer_l1_acc` is not plumbed
  (always on). The intermediate CB is 320 KB, the out CB is single-buffered.
- SwiGLU is a separate epilogue pass per output block (`swiglu_block`, `compute.cpp:28-70`): copy gate and
  up tiles from the intermediate into DST, `silu`, `mul`, pack, with the two SFPU inits re-armed per tile
  pair. It runs on the same math/pack threads after the K loop, so nothing overlaps it.

## 2. Baseline breakdown by zone

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
| two pairs per DST acquire, inits hoisted (see §3.2) | 16.45 | -0.10 (not worth its diff) |
| **`silu_tile<false>(...)` — bf16-grade silu under fp32 DST** | **15.52** | **-0.93 ms**, PCC 0.999928 vs 0.999930 (single device); **-0.56 ms on the mesh AGMM**, exp 9 |

`silu_tile` costs ~2,570 cycles per tile because under fp32 dest `_sfpu_sigmoid_<true>` uses the fp32-accurate
exp (Juffa) plus a two-iteration Newton reciprocal (`ckernel_sfpu_sigmoid.h`, `ckernel_sfpu_recip.h`); the
`<false>` instantiation uses the fast `_sfpu_exp_21f_bf16_` and one iteration. The SwiGLU output is packed to
bf16 anyway, so bf16-grade silu loses nothing measurable.

### Decomposition

(The 2026-09-18 first read of these zones split the 15.7 ms as 8.03 FPU / 4.9 K-loop overhead "co-limited by operand
delivery" / 2.8 SwiGLU. The per-thread zones of 2026-09-19 replaced the middle term: no thread waits, the loop runs at
the pipeline's issue pace. The table below is the corrected decomposition.)

| component | ms | lever |
|---|---|---|
| FPU work at HiFi2 peak | 8.0 | none (the roofline) |
| K-loop pipeline issue efficiency at 2x2/fp32 (math 47 vs 32 cyc per tile-MAC, unpack 42, pack 309 per fp32 tile; no waits) | 5.0 | §3.1 only; delivery is not a factor (§3.1) |
| SwiGLU epilogue (2.19 ms after change A of §3.2; ~1.8 of the 2.6 kcycles per pair is the bf16-grade `silu`) | 2.2 | §3.2 changes B/C, both precision or design trade-offs |
| waits, fabric, dispatch | <0.1 | none |

## 3. Experiments, by zone

### 3.0 Index (do not repeat)

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
| 14 | 2026-09-21: silu variants A/B on the single device and the mesh (§3.2) | fitted 6-segment fp16 LUT silu fused with the up multiply: single device 15.68 -> **14.42 ms**, PCC 0.999922, rel-RMSE 0.00914; mesh **16.04 -> 14.57 ms (-9.2%)**, PCC 0.9999858. Measured, not adopted; opt-in through `TT_MM_SWIGLU_LUT_SILU=1` since 2026-09-24, block and video numbers in the README (*Roofline with every optimization on*) |
| 16 | 2026-09-23 (galaxy GWH01): engine isolation of the K loop, 11 variants of `matmul_blocks` with one engine mocked out (§3.1) | bare unpack stream 185 of 208 cycles per step, HiFi4 256 + 24, rawunpack / fixedaddr / packmulti no change: **unpacker-paced at 46 cycles per 2 KB tile**; K2-K4 closed, only bytes per tile-MAC move it |
| 17 | 2026-09-23 (this galaxy): the eleven exp-16 variants re-run with a KLOOP zone, plus Tensix hardware perf counters on the production loop (§3.1) | every row within 1-2% of GWH01 (full 209, LoFi 200, HiFi4 281, nopack 196-200, unpackmock 193, mathmock 193, bare unpack 186-188); counters: FPU 60%, math thread 0.5% stalled with instructions always pending, src data ready exactly 60%, unpacker write requests 52-58% blocked by overwrite protection and 0% by the L1 port. **Confirmed unpacker-side; mechanism is the src-register handshake (4 per step) plus the exposed srcB refill of the 2x2 reuse scheme, not raw L1 bandwidth.** 17b: the two mocks under counters confirm the prediction (unpack mocked: refusals 0%, FPU 66% = 128/193, unpack thread waiting on srcB clear 70%; math mocked: unpacker writes 32 cycles per tile with no refusals) |
| 18 | 2026-09-23 (this galaxy, same evening): the K-loop MVMUL reorder of [kloop_refill_reorder_handoff.md](kloop_refill_reorder_handoff.md) built and measured (§3.1) | `matmul_block_kloop` alternates the 2x2 reuse-A order with its A/B mirror between K tiles: HiFi2 **209.7 -> 202.3** cycles per step (LoFi 200.3 -> 193.8, HiFi4 281.5 -> 279.6), bit-exact (128 LLK harness cases, 0 mismatches); mesh ff1 16.04 -> **15.69 ms**, to_qkv 11.25 -> 11.03, ff2 fused 8.98 -> 8.68, PCC identical. The plan's -15..20 did not survive the hand trace: the exposed refill was 5 of the 17 cycles above the 193 mock floor (nopack 197.9 -> **192.8**, i.e. the alternating loop without the packer sits on the floor; both mock rows unchanged at 193), the other ~9-12 are the packer's interaction with the loop (L1-port refusals stay 0%; likely the per-subblock DST handoff, exp 12). Unpack issue order: no effect; no MVMUL order can go below the four-handshake floor |
| 15 | 2026-09-21: block zones under fp32 dest off, (8,7,16) 2x4 (§3.2) | KLOOP 12,930 -> 11,692 us but SWIGLU 2,191 -> 2,744: the bf16-dest `mul_binary_tile` does a software round-to-nearest-even per vector, and `swiglu_block` processes the padded N. Net 15,169 -> 14,506 us (-4.4%); the -8% of exp 2 included the silu gain now banked separately |
| 10 | relay prefetch: in0/in1 receivers request block k+1 right after pushing block k to compute, before waiting for the downstream hop's request (60-line reorder of `dm_in0_sender.cpp` / `dm_in1_sender_out.cpp`, `tools/agmm_relay_prefetch.patch`) | correct (PCC identical, no hang) and **no gain**: HiFi2 16.08 → 16.06 ms, LoFi 14.52 → 14.68. Delivery does not pace the loop. Not landed |
| 11 | blocking sweep with the bench, HiFi2 host ms: (8,7,10) 2x2 **16.06**; (8,7,10) **4x1 18.28** (same 4 DST tiles, 5 unpacks per K step instead of 4); (4,7,10) 23.68; (4,14,10) 23.37 (doubling K_block = halving fp32 L1-acc pack passes: -1.3%); plain (8,7,10) 14.30; plain LoFi 12.65 | non-math time scales with tile-MACs and with the unpack count, not with iterations or pack passes → issue-bound pipeline, see §2 |
| 12 | per-thread sampled zones (`agmm_compute_zones.py apply sampled`) | table in §2: ACQ 21 / MAC 1321 / UNPACK 1357 / PWAIT 26 / PACK 1236 cycles per subblock |
| 13 | block zones after change A of §3.2 (`apply block`) | KLOOP **13,034** us, SWIGLU **2,191** us per core (was 12,934 / 2,809): the silu change removed 618 us on the mesh; 2,608 cycles per gate/up pair remain |
| 9 | `silu_tile<false>` on the mesh with ping-pong semaphores (`transformer_op_mesh_bench.py --op ff1`, then `agmm_ff1_mesh_bench.py`, 10 back-to-back calls) and through the fixed harness | **runs.** Host 16.58 → 16.08 ms per call; device kernel (harness, Tracy) **15,852 → 15,289 us (-563 us, -3.6%)**; PCC 0.9999838 → 0.9999834, rel-RMSE 0.00806 → 0.00828 (bar 0.9995 / 0.02); `test_linear_swiglu` 4/4 at 0.99998. **Landed** in both `compute.cpp` |

Numerics baseline (ff1 real ring op at M=13664, README Part 4 numerics table): pcc 0.9999843, rel-RMSE 0.00837; bar pcc > 0.9995,
rel-RMSE < 0.02. Single-device SwiGLU vs fp32 torch: fp32 dest 0.99993 / 0.0087, fp32 dest off 0.99985 / 0.0169.

### 3.1 The K loop (12.9 ms; 8.0 ms of it is the roofline)

**Operand delivery — measured, not a lever (exp 5, 10, 12).** The store-and-forward relay (in0 down a column chain of 8 cores, in1 across a row chain, one request/response semaphore round-trip per hop per K block; head cores read DRAM; `dm_in0_sender.cpp:446-476`, `dm_in1_sender_out.cpp:511-541`) keeps up: `in0/in1 wait_front` is 1 us per iteration (exp 5), and a prefetching receiver that decouples the upstream request from the downstream forward is correct but changes nothing at HiFi2 or LoFi (exp 10; the patch is kept at `tools/agmm_relay_prefetch.patch` in case a faster compute pipeline ever exposes delivery). Two ideas in the first draft do not survive the numbers: **double-buffering the out CB** (the out-CB reserve wait is 0-26 us total, exp 6) and **keeping in1 resident across M blocks** (that needs the whole K x N_block panel, 168 x 10 tiles = 3.4 MB per N block at K = 5376, not one 140 KB block; L1 is 1.4 MB). Multicast would cut relay traffic but the loop is not waiting on it. Revisit only if the compute pipeline below is brought under ~20 us per iteration. (to_out does wait on this relay: [to_out.md](to_out.md).)

**Compute pipeline — the 27 us per iteration vs 18 us of math.** The pace is set by three ~equally loaded threads (§2), so any single fix moves the pace only as far as the next thread. In order of evidence:

- **fp32 dest off for ff1 only.** DST then holds 8 bf16 tiles per half, so the subblock can be 2x4 (fewer
  `matmul_block` calls per tile-MAC, better in0 reuse on the unpacker) and the partial packs are 2 KB bf16 instead
  of 4 KB fp32 with L1 accumulate. Measured on the mesh: (8,7,16) 2x4 **14,475 us (-8%)** before the silu change; 14,506 us against 15,169 (-4.4%) after it, exp 15 -- part of the original -8% was the silu, now banked separately.
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

**What the 47 cycles are made of (source read, 2026-09-23).** Per `matmul_block` call on a 2x2 subblock the math
thread runs `_llk_math_matmul_` (`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h:852-1002`): for each of
the 4 output tiles a `set_dst_write_addr` (`common/inc/cmath_common.h:257-277`, a `TT_SETC16` config write), one
`ckernel_template::run()` (`common/inc/ckernel_template.h:320-323`, a `TTI_MOP` replaying the 16-MVMUL image twice at
HiFi2, programmed by `matmul_configure_mop`, `llk_math_matmul.h:~330-496`), then a `SETRWC` / `CLEARDVALID` bank switch. The
unpack thread runs `_llk_unpack_AB_matmul_` (`llk_unpack_AB_matmul.h:287-447`): per K tile one context acquire,
`_llk_unpack_configure_addresses_`, a semaphore post, a `STALLWAIT`, two srcB `UNPACR`s with a `SETDMAREG`/`REG2FLOP` address
rewrite between them, a `TT_MOP` for the two srcA tiles, a semaphore get and a context switch -- four tile unpacks per four
tile-MACs. The kernel passes the subblock dims at run time (`current_subblock_h/w`, `compute.cpp:475-486`, needed for the
ragged edge), so none of those LLK loops unroll; the `state_configure` call inside `matmul_block`
(`tt_metal/hw/inc/api/compute/matmul.h:259`) compiles to nothing without `TT_METAL_COMPUTE_KERNEL_SENTINEL_ENABLED`. The
Blackhole LLK has the same per-tile loop (`tt_llk_blackhole/llk_lib/llk_math_matmul.h:742-780`), so there is no ready
variant to port.

**Why the per-thread zones cannot say which thread paces (and what would).** The MATH zone of exp 12 is RISC-V issue
time, back-pressured by the Tensix instruction FIFO: an MVMUL stalled on srcA/srcB dvalid looks identical to a slowly
issued one. The fidelity scaling of exp 4 (plain, single device: LoFi 13.17 / HiFi2 14.48 / HiFi4 20.72 ms against
4.0 / 8.0 / 16.1 ms of nominal MAC time) fits `pace = max(~43, FPU + ~16..19)` cycles per tile-MAC: a ~16-cycle term
serialized with every tile-MAC (the size of a per-tile config write plus bank switch that cannot overlap the MOP), and a
~43-cycle floor (four 2 KB unpacks per four tile-MACs at 2x2). Both would have to fall for the loop to reach 32. The
disambiguator is the Tensix hardware counters, which the profiler exposes (`tech_reports/PerfCounters/perf-counters.md`):
`FPU Util` (FPU active cycles / elapsed), `SrcA/SrcB Valid Wait` (math waiting for the unpacker), `Thread 1 Stall Rate`
and the `INSTRN` availability rates, `Unpacker0/1 Write Efficiency`. Three signatures:

| counters read | pacer | lever |
|---|---|---|
| FPU Util ~68%, SrcA/B Valid Wait ~0, T1 stall low, THCON/CFG avail high | math-side issue: cfg write + bank switch per tile serialize with the MOP | LLK: dest increment through the addr_mod inside the replay so one MOP covers the 2x2 (no `SETC16` per tile), or a MOP outer loop over the K tiles; kernel-only prelude: compile-time subblock dims (exp K2 below) |
| SrcA/SrcB Valid Wait >= 25%, unpacker busy ~100% | unpacker: 4 unpacks per 4 tile-MACs at ~43 cycles each | only fewer unpacks per MAC: 2x4 / 4x2 subblocks (fp32 dest off, exp 2 / 15) or a narrower in1 format (bfp8, a precision decision); no code lever at 2x2 bf16 |
| both ~50% | the two limits coincide (the exp 4 fit) | both changes are needed for any gain; take the subblock change first, it lifts the floor to ~32 |

The capture itself is `TT_METAL_DEVICE_ARCH=wormhole_b0 python -m tracy -r -p --profiler-capture-perf-counters=fpu,pack,unpack,instrn --perf-counter-multipass <bench> ...`
(two passes of the workload, merged into the ops CSV as per-op `FPU Util`, `SrcA Valid Wait`, ... columns). Two things
found 2026-09-23 on the way: without `TT_METAL_DEVICE_ARCH`, `tools/tracy/perf_counter_multipass.py:76-88` opens device 0
in the *parent* tracy process to learn the architecture, which on this Galaxy leaves the `CHIP_IN_USE_16_PCIe` mutex held
and the child workload waits on it forever; and with it set, the mesh bench (`--op ff1 --no-fusion`) compiled with
`-DPROFILE_PERF_COUNTERS=3` and hung in its first AGMM call (25 min after `tensors on device`), wedging the ETH heartbeat.
The readout runs on BRISC after the TRISCs finish (`tt_metal/hw/firmware/src/tt-1xx/brisc.cc:542-545`,
`tt_metal/tools/profiler/perf_counters.hpp:471`) and pushes each group to DRAM; the interaction with the ring op's BRISC
dataflow kernels and the mux row is not root-caused. **Run the capture on the single-device bench** (`minimal_matmul`,
`transformer_op_single_device_bench.py --op ff1 --no-plain` for the SwiGLU op or `--no-fusion` for the plain K loop; the
`matmul_blocks` it runs is the AGMM's, `minimal_matmul/device/kernels/compute.cpp:356-410`), which has no ring, no fabric mux
and one BRISC dataflow kernel per core. Not run here: the board needed `tt-smi -r all` after the mesh hang, and the question
was answered the same day by the engine-isolation study below.

**Engine isolation (exp 16, 2026-09-23, galaxy GWH01, plain unfused ff1, shipped blocking).** Instead of counters, each
variant edits `matmul_blocks` so one engine does nothing (or only its handshake) and the step time is read from device
zones: 63,504 K-tile steps per core, one step = one `matmul_block` call = 4 unpacks + 4 tile-MACs of a 2x2 subblock.
Output is garbage in the mock variants; only the time counts.

| variant | what is removed | cycles per step | per tile-MAC |
|---|---|---|---|
| full loop, HiFi2 | nothing | **208** | 52 |
| rawunpack | the unpack thread's per-call handshake (semaphore, address config, STALLWAIT, context switch) for 6 of 7 steps | 209 | |
| fixedaddr | the L1 address pattern (same tiles every step) | 209 | |
| LoFi | half the MVMUL cycles | 198 | |
| nopack | every PACR | 197 | |
| unpackmock | every UNPACR (math + pack left) | 192 | |
| mathmock | every MVMUL (unpack + pack left) | 191 | |
| rawstream_a | 4 tiles through one unpacker, nothing else running | 193 | 48 |
| **rawstream_nopack** | **the bare UNPACR stream, nothing else running** | **185** | **46 per tile** |
| packmulti | half the pack instructions | 209 | |
| HiFi4 | (double the MVMUL cycles) | 280 = 256 + 24 | 70 |

Reading: the bare unpack stream is 185 of the 208 cycles, so the 2x2 step is paced by moving 4 x 2 KB of operands through the
unpacker at ~44 bytes per cycle, and the remaining 23 cycles are handshake bubbles between unpacker and FPU. The HiFi4 row
settles the math-side story: 4 x 64 = 256 cycles of MVMUL plus only 24, so the per-tile config write and bank switch on the
math thread cost ~6 cycles per tile-MAC, not the ~16 the fidelity fit above allowed; at HiFi2 the math side would run at
~4 x 38 = 152 if fed, under the 185 unpack floor. `rawunpack` and `fixedaddr` at 209 say the unpack thread's RISC-V
bookkeeping is already hidden under its data movement; `packmulti` says the pack instruction count is not on the path and
`nopack`'s 11 cycles are the packer's read-modify-write L1 traffic contending with the unpacker's reads. One unpacker moving
all four tiles (193) costs the same as the two sharing them (185-192): the limit is one shared port, not two engines.

This is the second row of the three-signature table: **unpacker-paced**. Consequences for the ladder in §5: K2 (compile-time
subblock dims), K3 (no-MOP replay) and K4 (one MOP per subblock) all attack math-thread issue cost that is not on the critical
path and are closed without being run. The only quantity that moves the step is bytes of operand per tile-MAC:

| lever | mechanism | projected | cost |
|---|---|---|---|
| fp32 dest off, 2x4 subblock | 6 unpacks per 8 tile-MACs: 35 unpack cycles per tile-MAC against 32 of math | KLOOP 12,930 -> 11,692 us measured (exp 15), ~42 per tile-MAC | bf16 partial sums; rel-RMSE 0.0083 -> 0.0166, under the 0.02 bar; model-level check |
| bfp8 in1 (weights) | the two srcA unpacks per step move ~1 KB tiles instead of 2 KB: ~140 unpack cycles against 128 of math | ~150-160 per step, about -25% | weight precision; not measured |
| bfp8 in0 as well | all four unpacks halve: ~93 unpack cycles, math-bound at 128 + bubbles | ~-35% | activation precision; not measured |
| `dst_full_sync_en` with fp32, 2x4 | 8 fp32 tiles to math at identical numerics, but pack no longer overlaps: ~2,400 pack cycles per subblock on the path | ~79 per tile-MAC at K_block 7, break-even ~50 at K_block 21 | no gain |

**Verified on this galaxy (GWH02, 2026-09-23 evening, exp 17).** The same eleven variants re-run on one device of `UF-EV-B12-GWH02`
through the single-device bench with a per-output-block `KLOOP` zone (`tools/mm_kloop_variants.py apply <v>` / `parse`;
the mocks follow the tt-llk perf harness's dvalid scheme, `tt_metal/tt-llk/tests/helpers/include/perf.h:51-120`), 3 calls per
variant, mean over the 64 cores:

| variant | GWH01 (artifact) | GWH02 (this run) | cycles per step |
|---|---|---|---|
| full, HiFi2 | 208 | **209** (208.8-210.4 over 4 calls) | |
| full, LoFi | 198 | 200 | |
| full, HiFi4 | 280 | 281 | = 256 + 25 |
| nopack | 197 | 196-200 | |
| unpackmock (no UNPACR; math + pack, mock unpacker sets dvalid) | 192 | 193 | = 128 + 65 |
| mathmock (no MVMUL; unpack + pack, mock math clears dvalid) | 191 | 193 | |
| mathmock + nopack (bare unpack stream) | 185 | 186-188 | |

Every row agrees to 1-2%. Two of them sharpen the reading. `unpackmock` at 193 = 128 + 65 says the math side with instantly
valid sources still pays ~16 cycles per tile-MAC of dvalid round trip (STALLWAIT on clear, SETDVALID, MVMUL, CLEARDVALID),
and `mathmock` at 193 says the unpack side with instant consumption pays the same: **the ~185-193 floor is the source-register
handshake protocol at four handshakes per step, on whichever engine is real**, and the full loop's 208 adds ~15-20 cycles for
the refill of the srcB banks that the 2x2 reuse scheme holds through the whole step (`_llk_math_matmul_init_` disables the
srcB valid clear for `t_dim > 1`, `llk_math_matmul.h:789-799`, so B0 and B1 are released only by the final `CLEARDVALID` and
the next step's B0 write is exposed). HiFi4 hides all of it under the longer MVMULs (256 + 25).

The hardware counters (`tools/tensix_perf_counters.py`, from a `--profiler-capture-perf-counters=fpu,pack,unpack,instrn`
run of the same bench; tracy's own merge of the two passes asserts on run-host-ids, so the parser reads the raw marker rows)
say the same thing from the inside, production 2x2 fp32 case, per core over the whole kernel (13.52 M cycles, K loop + copy
epilogue), 4x2 fp32-off case alongside:

| counter | 2x2 fp32 | 4x2 fp32-off |
|---|---|---|
| FPU active cycles | 8,141,936 = 63,504 steps x 128.2 | 8,444,912 |
| FPU util (active / elapsed) | 60.2% (61.6% of the K loop = 128/208) | 71.6% |
| math instructions started / available | 99.9% | 99.9% |
| math src data ready / elapsed | 60.2% (= FPU active: the FPU idles exactly when operands are not ready) | 71.4% |
| MATH instruction pending on T1 | 97% | 97% |
| T1 (math thread) stall | 0.5% | 0.9% |
| wait srcA / srcB valid (T1) | 0 / 0 | 0 / 0 |
| unpacker 0 / 1 busy | 93% / 93% | 94% / 94% |
| srcA / srcB write requests (cycles / elapsed) | 73% / 63% | 77% / 70% |
| srcA / srcB write requests blocked by **overwrite protection** | 58% / 52% of requests | 76% / 49% |
| srcA / srcB write requests blocked by the **L1 port** | 0% / 0% | 0% / 0% |
| packer busy | 32% | 20% |
| T2 (pack thread) stall | 13% | 5% |

The math thread never stalls and always has work queued; the FPU is idle 40% of cycles and those are exactly the cycles with
no source data ready. The unpackers are "busy" 93% but never port-blocked: half of their write requests are refused by
overwrite protection, i.e. they hold a tile and wait for the FPU to release the bank. The unblocked write cycles come to
~32 per 2 KB tile (73% x 42% x 208 / 2 = 64 cycles for two srcA tiles), so the engine moves ~64 B/cycle when it can write;
the artifact's 46 cycles per tile is 32 of data plus ~14 of handshake round trip, not a 44 B/cycle port. The consequence
for the levers is the same as the artifact's, with one refinement: hiding the exposed srcB refill (a MOP order or bank
schedule in which the next step's first operands land during the current step's last two MVMULs) is worth at most
208 -> ~185, ~11% of the K loop (~1.4 ms on ff1, ~0.6 ms on to_qkv per call), and is tt-llk work on `_llk_math_matmul_` /
`_llk_unpack_AB_matmul_`; past that floor only fewer handshakes and bytes per tile-MAC move it, which is the subblock shape
(fp32 dest off) and the operand format (bfp8).

**The two mocks under counters (exp 17b, same evening), which test that reading directly.** Prediction: with the unpacker
mocked, overwrite refusals should vanish and FPU util should be 128/193; with math mocked, the real unpacker should write with
no refusals and its write cycles should show the per-tile data time.

| counter | unpack mocked (real math) | math mocked (real unpack) |
|---|---|---|
| K-loop pace | 12.36 M cycles per core, ~193 per step | 12.43 M, ~193 per step |
| FPU util | **65.9%** (= 128 / 193 over the K loop) | 0.1% |
| src data ready / elapsed | 65.9% (= FPU active) | 0.1% |
| unpacker 0 / 1 busy | 1.2% / 1.2% (mock) | 85% / 85% |
| srcA / srcB write requests, cycles / elapsed | 0.9% / 0% | **33.6% / 33.0%** = 65 + 64 cycles per step for 2 + 2 tiles: **~32 cycles per 2 KB tile** |
| requests refused by overwrite protection | **0% / 0%** | 0.1% / 0.9% |
| requests refused by the L1 port | 0 / 0 | 0 / 0 |
| unpack thread waiting for **srcB clear** / srcA clear | **70.5%** / 1.0% (the mock's STALLWAIT before each SETDVALID) | 0 / 0 |
| math thread stalled | 0.5% | 94% (the mock's STALLWAIT: 74% on srcA valid, 20% on srcB valid) |

Read together: with no bytes moved at all the loop still runs at 193 and the FPU still idles 34% for "source not ready"; the
mock unpacker spends 70% of its time waiting for a **srcB** bank to clear and 1% for srcA, which is the reuse scheme holding
both B banks until MVMUL ④. With no math at all the unpacker writes 32 cycles per tile, is never refused, and is busy but not
writing for another ~16 cycles per tile (address config, context switch, dvalid); 4 x (32 + 16) = 193. So the artifact's 46
cycles per tile is 32 of data plus ~14-16 of per-tile unpack-side overhead and handshake, the 208 of the real loop is that
floor plus the exposed srcB refill, and the L1 port is idle in every configuration. `tools/tensix_perf_counters.py` on the
device CSVs of `mm_kloop_variants.py apply 2` and `apply 1` runs.

**The reorder, built and measured (exp 18, same evening).** `matmul_block_kloop` (`tt_metal/hw/inc/api/compute/matmul.h:335-368`)
runs a subblock's whole K loop in one call and, for 2x2 on Wormhole, `_llk_math_matmul_kloop_`
(`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h:1060-1115`) alternates the reuse-A order (A0B0, A0B1, A1B0,
A1B1) with its mirror (A0B0, A1B0, A0B1, A1B1) between K tiles, with both DVALID auto-clears off and explicit `CLEARDVALID`s.
Same-session A/B on the plain single-device loop: **209.7 -> 202.3 cycles per step at HiFi2** (LoFi 200.3 -> 193.8, HiFi4
281.5 -> 279.6); counters, same session before -> after: FPU util 60.2% -> 62.9% on identical FPU active cycles (8.14 M), unpacker write requests refused by overwrite protection srcA 57.6% -> 49.8% and srcB 52.0% -> 45.5% of requests (absolute refused cycles down ~25% on both), L1-port refusals 0% both.
Mesh (`transformer_op_mesh_bench.py`): ff1 fused 16.04 -> 15.69 ms, to_qkv 11.25 -> 11.03, ff2 fused 8.98 -> 8.68, numerics
identical. The engine-isolation ladder of exp 16/17, re-run before and after in one session (`mm_kloop_variants.py apply <v>
[legacy]`, HiFi2): full 210.2 -> 202.2; **nopack 197.9 -> 192.8**; unpack mocked 192.6 -> 192.9; math mocked 193.3 -> 193.2;
bare unpack stream 188.2 -> 187.7. The two mock rows do not move (the reorder does not change the four valid/clear round
trips per step) and the loop without the packer now sits exactly on their floor, so the exposed refill was worth 5 cycles,
not the 15-20 projected above: the rest of the 208 -> 193 gap is the packer's interaction with the loop (the `nopack`
gain exp 16 already measured on the old loop; the counters keep L1-port refusals at 0%, so the likely account is the
per-subblock DST handoff of exp 12, `ACQ` 21 + `PWAIT` 26 cycles per 7 steps), and that is what the remaining 202 vs 193 is. The hand trace before coding also showed the plan's mechanism was not the point: in any 4-MVMUL order over a 2x2 the
two tiles of MVMUL ④ are released at 128 and the other two at 64 and 96, and the next step's first MVMUL always needs one
of the 64/96 pair, so the production order already released first what was needed first; what the alternation removes is
the second late refill queueing behind the first on the same (srcB) unpacker. The unpacker's issue order (srcA MOP before
or after the srcB pair) makes no difference. No MVMUL order can go below the four-handshake floor, so this lever is
exhausted at 2x2; it should be carried into the 2x4 / 4x2 orders if fp32 dest off lands.
**Before / after, all three configurations under hardware counters (exp 18b, 2026-09-23 23:31-23:33, one session).** The
production loop and the two mocks of exp 17b, each built on the per-tile `matmul_block` loop (`mm_kloop_variants.py apply <v>
legacy`, standard reuse-A order every K tile) and on `matmul_block_kloop` (alternating orders), HiFi2, `--iters 2`, two
counter passes each; per-core means over 64 cores, K-loop pace from the `KLOOP` zone of the same run, "per step" = per core /
63,504. `tools/tensix_perf_counter_matrix.py name=csv ...` prints this table from the six device CSVs.

| metric | full, old order | full, alternating | unpack mocked, old | unpack mocked, alt | math mocked, old | math mocked, alt |
|---|---|---|---|---|---|---|
| K-loop pace, cycles per step (KLOOP zone) | 210.2 | 202.9 | 192.2 | 193.4 | 192.7 | 193.2 |
| elapsed cycles per core (kernel, M) | 13.5 | 13.0 | 12.4 | 12.4 | 12.4 | 12.4 |
| FPU util % of kernel | 60.3 | 62.4 | 65.8 | 65.4 | 0.1 | 0.1 |
| FPU util % of K loop (FPU active / KLOOP) | 61.0 | 63.2 | 66.7 | 66.3 | 0.1 | 0.1 |
| src data ready % of kernel | 60.3 | 62.4 | 65.8 | 65.4 | 0.1 | 0.1 |
| math thread stalled % | 0.5 | 0.5 | 0.5 | 0.5 | 94.7 | 94.5 |
| T1 wait srcA valid / srcB valid % | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 76.2 / 17.9 | 72.3 / 21.5 |
| unpacker0 / unpacker1 busy % | 93.4 / 93.4 | 90.0 / 90.0 | 1.2 / 1.2 | 1.2 / 1.2 | 85.6 / 85.8 | 84.7 / 84.9 |
| srcA / srcB write-request cycles per step | 155.0 / 133.3 | 130.9 / 117.6 | 1.7 / 0.0 | 1.7 / 0.0 | 65.7 / 64.6 | 65.8 / 64.6 |
| srcA / srcB unblocked write cycles per step | 65.7 / 64.0 | 65.7 / 64.0 | 1.7 / 0.0 | 1.7 / 0.0 | 65.7 / 64.0 | 65.7 / 64.0 |
| srcA / srcB refused by overwrite, % of requests | 57.6 / 52.0 | 49.8 / 45.6 | 0.0 / n/a | 0.0 / n/a | 0.0 / 1.0 | 0.1 / 0.9 |
| srcA / srcB refused by overwrite, cycles per step | 89.3 / 69.3 | 65.2 / 53.6 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.6 | 0.1 / 0.6 |
| srcA / srcB refused by L1 port, % of requests | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / n/a | 0.0 / n/a | 0.0 / 0.0 | 0.0 / 0.0 |
| T0 wait srcA clear / srcB clear % | 0.0 / 0.0 | 0.0 / 0.0 | 1.0 / 70.4 | 1.0 / 70.5 | 0.0 / 0.0 | 0.0 / 0.0 |
| unpack thread stalled % | 1.9 | 2.1 | 0.0 | 0.0 | 1.8 | 1.8 |
| packer busy % | 31.8 | 33.3 | 30.7 | 30.5 | 36.8 | 36.7 |
| pack thread stalled % | 13.1 | 13.8 | 12.3 | 12.2 | 16.2 | 16.2 |

Reading it column pair by column pair:

- **Full loop, old -> alternating.** 210.2 -> 202.9 cycles per step; FPU util of the K loop 61.0 -> 63.2%. The unpackers'
  *unblocked* write cycles are identical (65.7 + 64.0 per step, i.e. ~32 cycles per 2 KB tile, the port width) and the L1 port
  refuses nothing in either order. What changes is the waiting: write-request cycles per step drop from 155 + 133 to 131 + 118,
  the cycles refused by overwrite protection from 89 + 69 to 65 + 54 (the srcA side loses 24, the srcB side 16), and the
  unpackers' busy fraction from 93% to 90%. Every recovered cycle is one in which a bank was released earlier.
- **Unpack mocked, old -> alternating.** 192.2 -> 193.4, within noise; FPU util of the K loop 66.7 vs 66.3%. With no bytes
  moved the order does not matter, which is the expected result: the reorder buys nothing on the handshake floor. The 70%
  "waiting for srcB clear" is the same in both because the mock's own STALLWAIT order (B first, then A) is fixed; it measures
  where the mock waits, not how long the real loop loses.
- **Math mocked, old -> alternating.** 192.7 -> 193.2, unchanged; the real unpacker writes 65.7 + 64.0 unblocked cycles per
  step with ~1% refusals in both, and the mock math thread waits for srcA valid 72-76% / srcB valid 18-22% of the time in both.
  The unpack side is order-independent, which is why the change lives entirely on the math side.

So the eight cycles the reorder recovers appear only where real data meets real MVMULs (the first column pair), and the two
mock pairs put the floor at ~193 regardless of order. The remaining ~10 cycles between 193 and 203 do not move with the MVMUL
order and are not L1 refusals; the handoff's §0.1 attributes them to the packer's DST-half handoff between subblocks.

**The rest of the order space (exp 19, 2026-09-24).** A compile-time schedule interpreter (`kloop_sched`, `MM_KLOOP_SCHED`,
`llk_math_matmul.h`) replaced the two hand-written orders and was swept over nine 2x2 schedules: both period-1 orders 210 / 198
(full / no packer), both period-2 mirrors 201-202 / 193, PCC 1.000000 throughout; any schedule that starts a step on tile 1 or
flips a source an odd number of times deadlocks against the unpacker's write order (one hung the device) and is now rejected at
compile time. tt-llk golden on the interpreter: 112 passed. fp32 dest off at 2x2 runs 196-198 (the pack residual halves with
the pack bytes); 4x2 fp32 off runs at 45.8 cycles per tile-MAC with ~14 cycles per step of boundary exposure that only an
unpacker-side change could hide. The 2x2 order question is closed. Tables and reasoning: [kloop_order_space_plan.md](kloop_order_space_plan.md) §2.5, §3.1, §4.

Details, code map and validation: [kloop_refill_reorder_handoff.md](kloop_refill_reorder_handoff.md).

### 3.2 The SwiGLU epilogue (2.8 ms at baseline, 2.2 ms after change A)

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

**Change B — a lookup-table sigmoid (measured 2026-09-21, exp 14; not landed).** Wormhole's SFPU evaluates a
sign-symmetric piecewise-linear function of |x| in one instruction: `SFPLUT` (3 segments switching at |x| = 1 and 2,
8-bit coefficients) and `SFPLUTFP32` (fp16 or fp32 coefficients, 3 segments, or 6 segments switching at |x| = 0.5, 1,
1.5, 2 and 3 or 4). The table shipped in `ckernel_sfpu_sigmoid_appx.h` is a poor fit (its middle segment is off by
0.10 at |x| = 2), which is why it fails the bar; a fitted 6-segment table does not. Single-device A/B on
`minimal_matmul` + SwiGLU, HiFi2, (8,7,10) 2x2, fp32 dest, bf16 inputs, PCC / rel-RMSE vs fp32 torch:

| silu variant | ms | vs baseline | PCC | rel-RMSE |
|---|---|---|---|---|
| baseline: `exp_21f` + 1 Newton reciprocal (`silu_tile<false>`) | 15.68 | | 0.999928 | 0.00893 |
| no silu at all (floor for the current epilogue structure) | 14.18 | -1.50 | | |
| shipped 3-segment 8-bit LUT (`sigmoid_appx`) | 14.64 | -1.04 | 0.999600 | **0.02669, fails the 0.02 bar** |
| shipped LUT, fused with the `up` multiply in one SFPU pass | 14.47 | -1.21 | 0.999600 | 0.02669, fails |
| exp + quadratic reciprocal estimate, no Newton step | 15.53 | -0.15 | 0.999929 | 0.00878 |
| fitted 3-segment fp16 LUT, fused | 14.49 | -1.19 | 0.999876 | 0.01266 |
| **fitted 6-segment fp16 LUT (cutoffs 0.5/1/1.5/2/4), fused** | **14.42** | **-1.26** | 0.999922 | 0.00914 |
| fitted 6-segment LUT, separate multiply | 14.61 | -1.07 | 0.999922 | 0.00914 |

On the mesh AGMM the fused 6-segment variant measures **16.04 -> 14.57 ms host-timed (-9.2%)**, worst-device PCC
0.9999858 (baseline 0.9999834), rel-RMSE 0.00852 (0.00828). The table (slope, intercept per segment of
sigmoid(x) - 0.5, sign retained): (0.24492, 0.00049), (0.21720, 0.01508), (0.17303, 0.05953), (0.12645, 0.12930),
(0.05061, 0.29043), (0.0, 0.5); max |error| 0.018 at |x| = 4, ~1e-3 below 2; programmed through
`sfpi::vLut16ss` / `vLut16ii` into LReg0-2 / 4-6 the way the Blackhole gelu kernel does, evaluated with
`sfpi::lut<LutMode::Fp16x6_HWM4>`. All LUT variants cost the same one instruction per vector; the speed comes from
replacing ~35-40 instructions of exp and reciprocal, and from fusing the `up` multiply into the same pass. The
bench's gate pre-activations have standard deviation 0.5, so almost nothing lands above |x| = 2; the error is bounded
everywhere but the model-level check (CLIP / VBench) still applies before landing. **Update 2026-09-21 evening:** the
fused variant is in the tree behind an inert compile define and was A/B/C-measured against the baseline on the whole
mesh op (host 16.56 / 15.97 / 14.62 ms; device 15,773 / 15,173 / 13,554 us); the model's path is unchanged. Paused
before adoption: [ff1_swiglu_lut_handoff.md](ff1_swiglu_lut_handoff.md).

**Change C — overlap the epilogue instead of shrinking it (proposed; needs fp32 dest off).** `silu_tile_pack` / `silu_tile_init_pack`
(`compute_kernel_api.h:680-689`) run the SFPU from the **pack** thread. Moving the epilogue's SFPU work to the
pack thread lets the math thread start the next block's K loop while the previous block's SwiGLU drains,
using the two DST halves naturally (math on one half, pack+SFPU on the other). Needs a `mul_binary_tile`
pack-thread variant or reordering so the multiply is done on the math thread before commit. Largest gain
(the whole epilogue hides behind the next K loop) and the most invasive change; measure on the single-device
kernel first. It does not fit under fp32 dest: the 2x2 matmul subblock uses all 4 fp32 tiles of a DST half, so
there is no room for a gate/up pair alongside the MACs; it becomes possible only with fp32 dest off (8 bf16 tiles
per half).

Not worth it: two pairs per DST acquire with hoisted inits (-0.1 ms, measured).

**The epilogue under fp32 dest off (measured 2026-09-21, exp 15; two fixes proposed).** With `fp32_dest_acc_en=False`
and the (8,7,16) 2x4 blocking, block zones read KLOOP 12,930 -> 11,692 us (-9.6%) but SWIGLU 2,191 -> 2,744 us
(+25%), per pair 2,608 -> 3,062 cycles. Two causes, both in the kernel: `mul_binary_tile` defaults to the bf16-dest
path, which runs a software round-to-nearest-even plus a zero-check branch on every SFPU vector where the fp32 path is
one multiply (fix: instantiate it as `mul_binary_tile<true>`, letting the store truncate to bf16, the same shape of
fix as change A); and `swiglu_block` is called with the full `N_block_tiles` while the K loop clips to the core's real
width, so at N_block 16 it processes 896 pairs per core instead of 840 (fix: pass the clipped width). The fused
LUT pass of change B removes the separate multiply and with it the first cause.

### 3.4 fp32 dest off, explored end to end (exp 20, 2026-09-24 00:20-01:35, this galaxy)

The precision decision of §3.2 / §5, taken through the op, the block and the pipeline in one session, with the two epilogue fixes
of §3.2 applied first and the alternating K loop (exp 18) as the baseline.

**The two epilogue fixes** (`all_gather_minimal_matmul_async/device/kernels/compute.cpp` and `minimal_matmul/device/kernels/compute.cpp`,
`swiglu_block`): the epilogue now takes the block's live row and column counts and computes only the gate/up pairs that hold
real output (the writer already skips the padded tiles, `matmul_dataflow_common.hpp:559-590`, so the out CB keeps the full
block layout and padded rows are pushed uncomputed), and the multiply is `mul_binary_tile<true>`, the fp32-DST code path,
which under a bf16 DST truncates on the store instead of running the software round-to-nearest-even per vector
(`ckernel_sfpu_binary.h:147-170`). Under fp32 dest both are no-ops in effect. `test_linear_swiglu` 4/4.

| ff1 on the mesh bench (`transformer_op_mesh_bench.py --op ff1`, host ms per call) | before the fixes | after |
|---|---|---|
| fp32 on, (8,7,10) 2x2 (production) | 15.62 | 15.63 |
| fp32 off, (8,7,10) 2x2 | 15.55 | — |
| fp32 off, (8,7,16) 2x4 | 15.38 | **14.30 (-8.5%)**, rel-RMSE 0.0157 (was 0.0166) |
| fp32 off, (12,7,8) 4x2 | 15.34 | 14.54 |

**Model plumbing, behind an exploration switch.** `MINIMAX_H3_MM_FP32_DEST` (`transformer_block_minimax_h3.py`,
`attention_minimax_h3.py`; default `1`): `0` gives ff1 and to_qkv fp32 dest off with 8-tile subblocks, `ff1` or `qkv` one of
them; `MINIMAX_H3_QKV_BLOCKS` overrides to_qkv's blocking. ff1 takes its own compute config through a new
`ff1_compute_kernel_config` argument of `ParallelFeedForward.forward` / `forward_fused_addcmul` (`layers/feedforward.py`), so
ff2 keeps fp32 dest; and `get_matmul_config` (`utils/matmul.py`) now accepts a 4-tuple `default_block_size` carrying the
subblock, used as given ahead of the M-keyed tables (the 3-tuple keeps its fallback meaning). Verified in the profiles below
that the switched ops ran the intended blockings.

**Block (`test_minimax_h3_transformer_block_perf`, 15 s / 768P, fsdp1, device ms per call, mean over 32 devices; the three
AGMMs in call order):**

| config | to_qkv | to_out | ff1 | net |
|---|---|---|---|---|
| production (fp32 on) | 9.96 | 6.10 | 14.46 | |
| **ff1 only off, (8,7,16) 2x4** | 9.97 | 6.06 | **13.62** | **-0.84 ms per layer** |
| to_qkv only off, (12,7,8) 4x2 | 9.74 | 6.35 | 14.49 | ~0 |
| to_qkv only off, (8,7,12) 2x2 | 9.57 | 6.28 | 14.54 | -0.2 |
| both off | 9.77 | 6.63 | 13.66 | -0.3 |
| both off, two earlier pairs | 9.81 / 9.72 | 6.41 / 6.50 | 13.62 / 13.63 | (-0.06 device-only vs 240.79) |

ff1 alone is a clean -0.84 ms (-5.8% of the op) with nothing else moving. Switching to_qkv makes to_out's *kernel duration*
longer by 0.2-0.5 ms in every run (five of five), which at first read cancelled to_qkv's gain. It does not: **the per-op
durations are not additive under FSDP.** On every device an FSDP `AllGatherAsync` (to_out's own weight gather, on the CCL
sub-device) launches 0.12 ms before to_out and runs inside its window; to_out's kernel waits for those weights, so its
duration is compute plus however much of the gather lands inside it. In production that already makes to_out ring-uniform
but row-dependent, 5.15 ms on rows 3 and 5 of the mesh and 7.2 on rows 0 and 6, anticorrelated with the gather's own recorded
duration (2.0 vs 0.5-0.7 ms); the fsdp0 profile in Part 2 of the README has to_out at 5.3 with no gather. A faster to_qkv shifts
the phase of that overlap and moves time from the gather's row into to_out's. The metric that cannot double count is the
**device-busy wall time per layer, the union of every op's device interval** (`tools/block_device_busy.py`), repeatable to
0.1 ms across the night's repeats:

| config | device-busy per layer | vs production |
|---|---|---|
| production (fp32 on), two runs | 238.12 / 238.13 ms | |
| **ff1 only off** | 237.22 | **-0.91** |
| to_qkv only off, 4x2 | 237.80 | -0.33 |
| to_qkv only off, 2x2 | 237.74 | -0.39 |
| both off, three runs | 236.80 / 236.96 / 236.91 | **-1.22**, additive |

So to_qkv off is worth ~-0.35 ms per layer of wall time (its op-level -8.7% shrinks because the alternating K loop already
took part of it), not zero; it stays a secondary candidate because it doubles the output shift (below) for a third of ff1's
gain. `block_profile_stats.py compare`'s "device only" row (240.79 -> 240.73 for both off) is a per-op sum and hides this;
use the union for any change that shifts timing under FSDP. The fsdp0 profile, which has no gathers, aborts inside Tracy on a
mismatched zone pair from `dit_fused_norm_forwarder.cpp:159` and could not be used.

**Pipeline, 10 steps (`test_parallel_sweep_minimax_h3.py`, 4x8 TP4/SP8, 16:9 15 s, FSDP on), per forward and output:**

| config | ms per forward | frames vs production | audio |
|---|---|---|---|
| production, run 1 / run 2 | 12117 / 12147 | run 1 vs run 2: PCC 0.99925, mean abs diff 1.2 of 255 | |
| ff1 and to_qkv off | 12132 | PCC 0.629, mean abs diff 33.5, per-frame PCC 0.47-0.78 | PCC 0.962 |
| ff1 only off | 12062 | PCC 0.891, mean abs diff 16.4 (the TP8 experiment's range, 12-19) | PCC 0.969 |

The end-to-end time cannot resolve the change (-0.84 ms of a 241 ms layer is 0.35% of a forward, inside the 10-step
run-to-run spread). The output does: two production runs agree to 1.2 of 255, and the fp32-off output differs by 33.5, so the
difference is the precision change, not run-to-run variation. Rendered frames (first, middle, last) show the same fox, snow
field and lighting with a different pose and gait in every frame: a different sample of the same prompt, not a degraded one.
The TP8 experiment in the README, which also changed only bf16 reduction order, moved frames by 12-19; ff1 alone sits in
that range, both ops together are twice it.

**Pipeline, 50 steps with the CLIP gate (`test_t2va_end_to_end[wormhole_b0-4x8_WH-16x9_15s]`, FSDP on, `RUN_VBENCH=0`), ff1 only off:**

| | ff1 only off (02:00) | production, same session (02:22) | production, this host, 2026-09-17 (README Part 4) |
|---|---|---|---|
| CLIP prompt alignment, mean / min / max | **35.91 / 34.96 / 36.88** (bar 33.0) | 35.75 / 34.83 / 36.88 | 35.88 / 34.69 / — |
| per forward | **12086.9 ms** | 12143.1 ms | 12058-12230 across runs (best 11988) |
| denoise / realtime | 592.3 s / 40.8x | 595.0 s / 41.0x | 590.9-599.2 s / 40.6-41.2x |
| audio / A-V sync | 2 ch 15.075 s at 32 kHz, sync delta -0.008 s, sanity OK | same, sanity OK | |

Same host, same weights, same seed, back to back: **-56 ms per forward (-0.46%)**, against the -45 ms the block profile
predicts (0.91 ms x 50 layers) and inside the 0.5% the 10-step pairs could not resolve; CLIP 35.91 vs 35.75, the two runs'
spread (production alone has ranged 35.75-35.88 on this host). Videos: `~/h3_fp32_videos/` on this host: the two 50-step
generations with audio (`50step_ff1_only_fp32off/`, `50step_production_fp32on/`), 10-step clips of every configuration, and a
side-by-side.

**Verdict.** ff1 with fp32 dest off and the (8,7,16) 2x4 blocking is a **-0.9 ms per layer of device wall time (-5.8% of
the op, ~-45 ms per forward)** change with production-level CLIP and an output that differs from production by the amount a
reduction-order change does. to_qkv off adds another -0.35 ms per layer but doubles the output shift (33 vs 16 of 255 at 10
steps) and has no CLIP run of its own; it is the secondary candidate, not a rejected one. Adopting it means turning the switch's
ff1 branch into the default (`transformer_block_minimax_h3.py`, `ff1_compute_kernel_config` / `ff1_block_size`), keeping
`MINIMAX_H3_MM_FP32_DEST=1` as the way back, and re-baselining the numerics rows in this file (op rel-RMSE 0.0157 against the
0.02 bar). Not landed by default in this session; the user decides.

### 3.3 Not levers (measured or bounded)

K_block (exp 1, 11); operand delivery and the relay protocol (exp 5, 10, 12); out-CB depth (exp 6); the ring
gather (hidden, exp 3/5); fabric bandwidth (1.1 ms bound, 7% used); aggregate DRAM (0.78 ms bound); LoFi (halves
the bar); `dst_full_sync_en` (serializes math and pack); the grid (mux row fixed by the op).

## 4. Results

| step | K loop | epilogue | total | util | status |
|---|---|---|---|---|---|
| 2026-09-18 baseline | 12.9 | 2.8 | 15.7 | 51% | |
| **change A landed (2026-09-19): bf16-grade silu** | 13.0 | 2.2 | **15.3** (device 15,289 us) | 53% | done, numerics unchanged; the block's 3-AGMM row 32.25 -> 31.85 ms |
| + fp32 dest off, (8,7,16) 2x4 (exp 2, 15) | 11.7 | 2.7 (regression, fixable to ~2.0) | 14.5 (device 14,506 us) | 55% | measured; precision decision (rel-RMSE 0.0083 -> 0.0166), model-level check needed |
| + fitted 6-segment LUT silu, fused (change B, exp 14) | 13.0 | ~1.0 | **~13.8** (mesh host 14.57 vs 16.04) | ~58% | measured; precision decision (error bounded at 0.018 on the sigmoid), model-level check needed |
| both precision decisions | ~11.7 | ~0.8 | ~12.5 | ~64% | projected |
| + bfp8 in1 (unpack floor ~140 of 128 math cycles per step) | ~9.5 | | ~10-11 | 75% | projected from exp 16; weight-precision decision, not measured |
| ~~LLK `matmul_block` issue efficiency~~ | | | | | closed by exp 16: the loop is unpacker-paced, math-thread issue is not on the path |
| roofline | 8.0 | 0 | 8.0 | 100% | |

Numerics: the real ring op at M=13664 reads pcc 0.9999843, rel-RMSE 0.00837 at baseline (bar pcc > 0.9995,
rel-RMSE < 0.02); single-device SwiGLU vs fp32 torch 0.99993 / 0.0087 with fp32 dest, 0.99985 / 0.0169 without.

Without a precision decision the kernel is at its practical floor for this blocking: the remaining 5 ms of K loop
is the per-tile issue cost of the 2x2 fp32 subblock on all three TRISCs, and the 2.2 ms epilogue is SFPU-bound with
the accurate-enough sigmoid. Per forward (50 blocks) each ms per call is 50 ms; 15.7 -> 10 ms would be ~0.29 s of
the 12.4 s forward, 2.3%.

## 5. What is left

1. **The two precision decisions.** fp32 dest off was taken through the block and the pipeline on 2026-09-24 (§3.4): the
   two epilogue fixes are in, the switch is in the tree (`MINIMAX_H3_MM_FP32_DEST`, default on), **ff1 alone is worth -0.84 ms
   per layer** and to_qkv is not (it slows to_out by as much as it gains). The 50-step CLIP run of the ff1-only configuration
   decides adoption (§3.4). The fitted 6-segment LUT silu (-9%) is still pending its own model-level check.
2. **Change C** (pack-thread epilogue overlap) once fp32 dest is off.
3. **tt-llk `matmul_block` issue efficiency** (§3.1): the only lever left on the K loop after that, and it moves
   all three AGMMs. Order of work as planned on 2026-09-23, and how it resolved the same day (§3.1, exp 16: the loop is
   unpacker-paced, so K2-K4 are closed and K5 is the path):
   - **K1 — which engine paces.** Answered by the engine-isolation study (§3.1, exp 16): the bare unpack stream is 185 of
     the 208 cycles per step; HiFi4 puts the math-side per-tile overhead at ~6 cycles. Unpacker-paced. The counter capture
     stays only as a check on whether 46 cycles per tile is the unpacker's intrinsic rate or L1 port contention.
   - **K2 — compile-time subblock dims in the kernel** (`matmul_blocks<sb_h, sb_w>` with a run-time dispatch for the
     ragged last subblock, `compute.cpp:352-402` / `:475-486`), so `_llk_math_matmul_` and `_llk_unpack_AB_matmul_`
     unroll and constant-fold. **Closed without running:** `rawunpack` / `fixedaddr` (209 vs 208) show the thread
     bookkeeping is hidden under the data movement.
   - **K3 — `matmul_block_no_mop` A/B** (the SDPA's replay path,
     `tt_metal/hw/inc/api/compute/experimental/matmul_custom.h:57-93`, `llk_math_matmul_custom_no_mop.h`): same per-tile
     structure with `lltt::replay` in place of the MOP. **Closed:** math-side issue is not on the critical path.
   - **K4 — LLK: one MOP per 2x2 subblock.** Program the dest increment into the replay image's addr_mods (the
     `.dest = {.incr}` fields `matmul_configure_addrmod` already sets, `llk_math_matmul.h:~40-320`) and the srcA/srcB
     bank switches into the MOP end-ops, so the four `SETC16` + `SETRWC` pairs per K tile disappear. Only if K1 reads
     the math-side signature. **Closed:** the HiFi4 row (256 + 24) leaves at most ~6 cycles per tile-MAC to win here, and
     the loop would still sit on the 185-cycle unpack floor.
   - **K4b — hide the srcB refill. Landed 2026-09-23 (exp 18), smaller than projected.** `matmul_block_kloop` alternates the
     2x2 MVMUL order between K tiles: -7 cycles per step at HiFi2 (209.7 -> 202.3), -0.35 ms on ff1, -0.22 on to_qkv, -0.30 on
     ff2 fused per call, bit-exact. The -15..20 projection assumed both first operands of a step could be released early; one
     of them is always the tile released third, so the gain is only the split of the two late refills over both unpackers;
     the isolation ladder puts the exposed refill at 5 cycles (nopack 198 -> 193, on the mock floor) and the rest of the
     production gap at the packer's interaction with the loop (DST handoff, not L1 bandwidth). Closed at 2x2; carry the alternation into 2x4 / 4x2 with fp32 dest off.
     [kloop_refill_reorder_handoff.md](kloop_refill_reorder_handoff.md).
     The rest of the order space (2x2 confirmation sweep, the packer-interaction cycles, the 2x4 / 4x2 orders that open with fp32
     dest off) is enumerated in [kloop_order_space_plan.md](kloop_order_space_plan.md).
   - **K5 — the path.** Below the ~185 handshake floor there is no code lever at 2x2 bf16; the levers are bytes of operand per tile-MAC: the 8-tile
     subblock (fp32 dest off, item 1, measured) and bfp8 in1 / in0 (not measured; a weight- and activation-precision
     decision). Both are precision decisions for the model owner, to be validated at the model level (CLIP / VBench).
4. Housekeeping: `sweep_mm_block_sizes.py`'s L1 pre-filter over-estimates the AGMM footprint (it filters K_block >= 14
   combos that build; they measured slower anyway).

## 6. Tooling and recipes

**Mesh reproducer, 15 s per run (use this first).** `models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff1`
(`--op to_qkv|to_out|ff2` runs the block's other matmuls with their own fusion, shape and blocking from
`minimax_h3_ops.py`) runs the op on one TP ring (the harness's
4x1 cluster submesh) exactly as the model does: fused SwiGLU, `bias=None`, HiFi2, fp32 dest, (8,7,10) 2x2,
`math_approx_mode=True`, **two semaphore pairs and two gathered-in0 buffers alternated per call** like `CCLManager`.
Prints host ms per call over N back-to-back calls and PCC / rel-RMSE against fp32 torch on the first 2048 rows of
every device. `--blocks`, `--fp32-dest 0`, `--no-fusion`, `--sync-each`, `--no-pingpong`
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

**K-loop engine isolation and hardware counters (single device).** `tools/mm_kloop_variants.py apply 0..4` edits
`minimal_matmul/device/kernels/compute.cpp` in place (KLOOP zone + one engine mocked), `parse <profile_log_device.csv>` prints
cycles per K-tile step per call, `revert` restores the kernel; run the bench between them under `python -m tracy -r -p`. About
90 s per variant. `tools/tensix_perf_counters.py <device csv> ['{"name": ["run ids"]}']` prints the derived counter metrics per
group of run host IDs from a `--profiler-capture-perf-counters=fpu,pack,unpack,instrn --perf-counter-multipass` run
(`TT_METAL_DEVICE_ARCH=wormhole_b0` is mandatory on the Galaxy, see §3.1; do not run the counter capture on the ring op).

**Device zones.** `models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py apply block|opwait|sampled` edits the
AGMM `compute.cpp` in place (exact-match, asserts if the kernel moved on), `revert` restores it, `parse <csv>` prints
per-thread mean cycles and per-core sums. `block` zones the K loop and whichever epilogue the op compiles (`SWIGLU`
for ff1, `EPILOGUE_ADDCMUL` for to_out, `EPILOGUE_COPY` / `EPILOGUE_BIAS` for to_qkv without / with a bias -- the
harness always passes one); `opwait` adds the per-iteration `in0/in1 wait_front` zone (first ~100 iterations);
`help` prints each AGMM's sweep `-k` id. By hand: include `"tools/profiler/kernel_profiler.hpp"` in `compute.cpp`
and wrap regions in `{ DeviceZoneScopedN("NAME"); ... }`; the sweep harness already runs under `tracy -p`, so zones land in
`generated/profiler/mm_sweep_wh_4x8_ring_13664_5376_7168_8x8_agmm_ff1_swiglu/reports/<ts>/profile_log_device.csv`.
The per-core buffer holds ~120 zone events per RISC, so use per-output-block zones (21 blocks x 3 zones) for
the full op and per-iteration zones only to sample the first ~60 iterations. Parse: skip the first header line,
`csv.DictReader`, filter `RISC processor type` in TRISC_0/1/2 (unpack/math/pack), keep the largest
`run host ID` per `PCIe slot`, pair `ZONE_START`/`ZONE_END` by `zone name` per core, sum. Zone overhead was
not measurable (15.74 vs 15.72 ms). The zones are not in the tree.

**Single device, fast loop.** `transformer_op_single_device_bench.py --op ff1` (25 s per case; `--fidelity LoFi,HiFi2` for the delivery-floor test; `--cases` for blockings and fp32 on/off;
`--op to_qkv|to_out|ff2` for the other matmuls through `minimal_matmul_split` / `dit_minimal_matmul_addcmul_fused` /
`minimal_matmul`). Host tensors are bf16, so its PCC reads ~0.99992 where an fp32-input golden reads 0.99993; the ms are unaffected. PCC for SwiGLU kernel changes:
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
