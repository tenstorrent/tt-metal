# MiniMax-H3 on Wormhole Galaxy: taking the ff1 AGMM to roofline — handoff (2026-09-18, updated 2026-09-19)

Branch `jameslee/exp_ring_sdpa_wh`. Written for whoever picks up the AGMM work next; everything below was
measured on this 4x8 Wormhole Galaxy (UF-EV-B12-GWH02) on 2026-09-18/19. **Status 2026-09-19:** lever §4.1 A
(bf16-grade silu) is landed in both kernels, ff1 is 15.85 → 15.29 ms on the mesh; the "mesh hang" it was
blamed for was a semaphore-reuse race in the test tooling (exp 8/9), now fixed. Next levers: §4.2 and §4.1 C. The investigation write-up that this
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
| roofline script | `models/tt_dit/tests/models/minimax_h3/tools/agmm_roofline.py` (`--dump`, `--figs all`, `--selftest`) |

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

So at HiFi2 the operands arrive just in time and the loss is inside `matmul_blocks`: the 2x2 subblock
(DST holds 4 tiles under fp32 dest), the fp32 L1-accumulating pack every K_block = 7 tile-MACs, and the
DST acquire/commit/wait/release handshake per subblock. But the data movement is only just keeping up:
running the same kernel at LoFi (math halved, ideal 4.0 ms) only saves 1.3 ms (14.48 → 13.17 ms plain
single-device), because the relay chain delivers ~250 KB per core per iteration at ~10 GB/s ≈ 24 us. The two
sides are balanced co-limiters at ~24-27 us per iteration; moving one alone yields a few percent.

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
| K-loop overhead: compute-thread structure and operand delivery, balanced | 4.9 | §4.2 + §4.3 together |
| SwiGLU epilogue, of which ~2.2 is `silu` | 2.8 | §4.1 |
| waits, fabric, dispatch | <0.1 | none |

## 3. Experiments already run (do not repeat)

| # | experiment | result |
|---|---|---|
| 1 | K_block 14 / 21 (never in the original sweep: its L1 pre-filter over-counts this op's CBs) | slower: (8,14,8) 16,227 us, (8,14,6) 16,281, (6,14,10) 16,898, (8,21,4) 18,131 vs 15,725; (8,14,10), (8,21,6) OOM at warm-up |
| 2 | fp32 dest off, mesh (`MM_SWEEP_FP32_DEST_ACC=0`) | (8,7,10) 2x2 15,162 (-3.6%); best (8,7,16) 2x4 **14,475 (-8.0%)**; (12,7,8) 4x2 14,677; (8,7,14) 2x2 14,756 |
| 3 | single device, same per-device shape (`agmm_ff1_single_device_bench.py`) | `minimal_matmul`+SwiGLU 16.56 ms ≈ the mesh AGMM: the ring gather is hidden. Plain 14.52 → SwiGLU costs 2.0 ms. fp32 off (12,7,8) 4x2 plain 12.67 ms (63%). `ttnn.matmul` auto-config 8x8: 23.4 ms fp32 on, 15.2 off (no better) |
| 4 | fidelity scaling, single device | plain (8,7,10) fp32: LoFi 13.17 / HiFi2 14.48 / HiFi4 20.72 ms → ~12-13 ms non-math floor at this blocking; (12,7,8) fp32-off LoFi 10.73 |
| 5 | per-iteration zones | operand wait 1 us/iter, `matmul_blocks` 27.3 us/iter vs 17.9 math |
| 6 | per-block zones | K loop 12.9 ms, SwiGLU 2.8 ms, waits 0 |
| 7 | SwiGLU attribution and variants, single device | table in §2: silu 2.16 ms; batching 0.1 ms; `silu_tile<false>` -0.93 ms with PCC intact |
| 8 | `silu_tile<false>` in the AGMM kernel on the mesh, through the sweep harness | hung 3 of 3 times. **Root-caused 2026-09-19: not the kernel.** The harness reused one semaphore pair and one gathered-in0 buffer for every call, and its warm-up enqueues calls back to back (`sync=False`): a device that finishes call i early starts call i+1 and signals ring semaphores a neighbour is still consuming in call i. Reproduced with the **unmodified** kernel: 5 back-to-back calls with one shared set hang after the first completes. The model never sees this: `CCLManager.get_ag_ping_pong_semaphore/buffer` alternates two sets. Harness fixed to ping-pong (§6) |
| 9 | `silu_tile<false>` on the mesh with ping-pong semaphores (`agmm_ff1_mesh_bench.py`, 10 back-to-back calls) and through the fixed harness | **runs.** Host 16.58 → 16.08 ms per call; device kernel (harness, Tracy) **15,852 → 15,289 us (-563 us, -3.6%)**; PCC 0.9999838 → 0.9999834, rel-RMSE 0.00806 → 0.00828 (bar 0.9995 / 0.02); `test_linear_swiglu` 4/4 at 0.99998. **Landed** in both `compute.cpp` |

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

### 4.2 Operand delivery (the ~24 us per K-iteration floor; 2.4 GB of relay + 1 GB of DRAM re-reads per op)

Facts from the kernels (all in `all_gather_minimal_matmul_async/device/`):

- No multicast. in0 goes down a column chain of 8 cores, in1 across a row chain of 8, each hop a
  request/response semaphore round-trip per K-block (`dm_in0_sender.cpp:446-476`,
  `dm_in1_sender_out.cpp:511-541`); only the chain head reads DRAM, tile by tile with one barrier per block
  (`matmul_dataflow_common.hpp:436-489`).
- Loop order m → n → k (`dm_in0_sender.cpp:304-317`, `compute.cpp:460-483`): in0 re-read from DRAM once per
  N block (3x), in1 once per M block (7x); ~1.0 GB DRAM per device per op on the 16 head cores.
- out CB single-buffered (`program_factory.cpp:392`); the plain and fabric-bound factories double-buffer it.

Changes, smallest first:

1. **Resident in1 across M blocks.** The fabric-bound factory already does this with a `c_7` in1 scratch CB
   (`minimal_matmul/device/minimal_matmul_fabric_bound_program_factory.cpp:347-357`, "so the in1 injector can
   read it once and re-present it to compute across M blocks"). Port that CB and the reader logic to the AGMM
   factory (`program_factory.cpp:371-394` CB sizing; `dm_in1_sender_out.cpp:497-510` reader). L1: one
   K_block x N_block bf16 block = 140 KB at (8,7,10); the current footprint is ~904 KB of 1464 KB.
   Expected: in1 DRAM/relay traffic 7x → 1x per N block. The (12,7,8) result (fewer M blocks, -13% plain)
   bounds what traffic reduction is worth.
2. **Double-buffer the out CB** (`program_factory.cpp:392`, mirror `minimal_matmul_program_factory.cpp:321`).
   +80 KB L1. Removes the compute→writer sync point at every output block.
3. **Spread head-core DRAM reads**: each of the 8 chain cores reads 1/8 of the block and forwards, instead
   of one head reading all of it (`read_in0_block_sync` / `read_in1_block_sync` callers at
   `dm_in0_sender.cpp:425-445`, `dm_in1_sender_out.cpp:497-510`).
4. **Multicast** in place of store-and-forward (the `matmul_2d` pattern, `noc_async_write_multicast` +
   semaphore). Removes 7 serialized hops and the per-hop handshakes. Largest change.

How to test: `minimal_matmul` shares the data-movement design, so iterate on
`minimal_matmul/device/` with `agmm_ff1_single_device_bench.py --fidelity LoFi,HiFi2` — the **LoFi time is
the data-movement floor** (13.17 ms today at (8,7,10)); a delivery fix shows up there first, then at HiFi2
once the compute side (§4.3) is also lowered. Port to the AGMM factory afterwards and confirm on the mesh
with the sweep harness (§6). Check operand waits with the `AGMM_OPWAIT` zone (§6) — they should stay near zero.

### 4.3 Compute-thread structure (the 27 us per iteration vs 18 us of math)

- **fp32 dest off** for ff1 only. `ParallelFeedForward.forward` (`models/tt_dit/layers/feedforward.py:125-131`)
  hands one compute config to ff1 and ff2; give ff1 its own (`fp32_dest_acc_en=False`) or set
  `self.ff.ff1.compute_config` (`linear.py:232`, used at `:442`). Then the blocking can move to an 8-tile
  subblock: swept best (8,7,16) sb 2x4 at 14,475 us (`matmul.py:140` entry). Roofline unchanged; accumulator
  error doubles (rel-RMSE 0.0087 → 0.0169 on the single-device SwiGLU output, bar 0.02) — needs a
  model-level check (CLIP/VBench), not only op PCC. Pays off fully only together with §4.2.
- **Larger K_block** does not help while delivery co-limits (exp 1); revisit after §4.2.
- `packer_l1_acc` is hard-coded on in the factory (`program_factory.cpp:1085-1090` passes only fidelity,
  fp32 and approx); `dst_full_sync_en` is never plumbed, so DST is always half-banked. Plumbing
  `dst_full_sync_en=True` would give 8 fp32 tiles per acquire without giving up fp32 — untested, worth one
  single-device run.

### 4.4 Not levers (measured or bounded)

K_block (exp 1); the ring gather (hidden, exp 3/5); fabric bandwidth (1.1 ms bound, 7% used); aggregate DRAM
(0.78 ms bound); LoFi (halves the bar); the grid (mux row fixed by the op).

## 5. Path to roofline

| step | K loop | epilogue | total | util |
|---|---|---|---|---|
| 2026-09-18 | 12.9 | 2.8 | 15.7 | 51% |
| **§4.1 A landed (2026-09-19)** | 12.9 | ~2.3 | **15.3** (device 15,289 us) | 53% |
| §4.1 C (epilogue hidden behind the next K loop) | 12.9 | ~0 | ~12.9 | 62% |
| §4.2 + §4.3 (delivery and compute side both lowered) | ~9-10 | ~0 | **~9-10** | 80-90% |
| roofline | 8.0 | 0 | 8.0 | 100% |

Per forward (50 blocks) each ms per call is 50 ms; 15.7 → 10 ms is ~0.29 s of the 12.4 s forward, 2.3%.
The same three levers apply to to_qkv (10.4 ms, 58%) and to_out (4.3 ms, 46%), which share the kernel
without SwiGLU; §4.2 and §4.3 carry over unchanged.

## 6. Tooling and recipes

**Mesh reproducer, 15 s per run (use this first).** `models/tt_dit/tests/models/minimax_h3/tools/agmm_ff1_mesh_bench.py`
runs the ff1 AGMM on one TP ring (the harness's 4x1 cluster submesh) exactly as the model does: fused SwiGLU,
`bias=None`, HiFi2, fp32 dest, (8,7,10) 2x2, `math_approx_mode=True`, **two semaphore pairs and two gathered-in0
buffers alternated per call** like `CCLManager`. Prints host ms per call over N back-to-back calls and PCC /
rel-RMSE against fp32 torch on the first 2048 rows of every device. `--blocks`, `--fp32-dest 0`, `--no-swiglu`,
`--sync-each`, `--no-pingpong` (reproduces the exp 8 hang). Wrap in `timeout 600`; a hang never prints "ms per call".

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

**Device zones.** Include `"tools/profiler/kernel_profiler.hpp"` in `compute.cpp` and wrap regions in
`{ DeviceZoneScopedN("NAME"); ... }`; the sweep harness already runs under `tracy -p`, so zones land in
`generated/profiler/mm_sweep_wh_4x8_ring_13664_5376_7168_8x8_agmm_ff1_swiglu/reports/<ts>/profile_log_device.csv`.
The per-core buffer holds ~120 zone events per RISC, so use per-output-block zones (21 blocks x 3 zones) for
the full op and per-iteration zones only to sample the first ~60 iterations. Parse: skip the first header line,
`csv.DictReader`, filter `RISC processor type` in TRISC_0/1/2 (unpack/math/pack), keep the largest
`run host ID` per `PCIe slot`, pair `ZONE_START`/`ZONE_END` by `zone name` per core, sum. Zone overhead was
not measurable (15.74 vs 15.72 ms). The zones are not in the tree.

**Single device, fast loop.** `agmm_ff1_single_device_bench.py` (25 s per case; `--fidelity LoFi,HiFi2` for
the delivery-floor test; `--cases` for blockings and fp32 on/off). PCC for SwiGLU kernel changes:
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
