# SDPA Optimization Proposals

Comprehensive optimization list for the tt-train SDPA forward and backward kernels.
Sources: FlashAttention-4 paper, TTNN SDPA implementation (`ttnn/cpp/ttnn/operations/transformer/sdpa/`),
and analysis of the current tt-train SDPA code.

## Current Architecture

The current forward kernel (`sdpa_fw_compute_kernel.cpp`) processes **one Q row at a time**,
iterating over K/V tiles **one tile per inner-loop iteration**. Each iteration:
1. Q[row] @ K[tile]^T → 1 score tile (matmul engine)
2. Scale by `1/sqrt(d)` via `mul_unary_tile` (SFPU)
3. Online softmax: update max, compute exp, update sum (SFPU)
4. P[tile] @ V[tile] → qWt output tiles (matmul engine)
5. Rescale previous output by exp(prev_max - cur_max) (SFPU + matmul)

The backward runs two separate kernels: Q-backward and KV-backward, each re-reading
Q, K, V, dO, and intermediates from DRAM.

## Implementation Status

Tracking of completed, in-progress, and blocked work.

### References

| Type | Link | Description |
|------|------|-------------|
| **PR** | [#41683](https://github.com/tenstorrent/tt-metal/pull/41683) | FP32 logsumexp intermediates, precision improvements, ring attention update |
| **PR** | [#42820](https://github.com/tenstorrent/tt-metal/pull/42820) | Phase A: SFPU optimizations (F13, F1, F3, F5) |
| **PR** | [#39812](https://github.com/tenstorrent/tt-metal/pull/39812) | B5: Precomputed u_scalar |
| **Ticket** | [#41686](https://github.com/tenstorrent/tt-metal/issues/41686) | SFPU FP32 softmax path (blocked on LLK primitives) |
| **Blocked on** | [#41593](https://github.com/tenstorrent/tt-metal/issues/41593) | LLK: SFPU row-reduce-max |
| **Blocked on** | [#41594](https://github.com/tenstorrent/tt-metal/issues/41594) | LLK: SFPU sub_bcast_col |
| **Reference** | `models/demos/deepseek_v3_b1/kernel_includes/.../sdpa.h` | DeepSeek V3 SDPA: FPU/SFPU overlap, MOVD2B, bcast_col_reuse, semaphore signalling |

### Completed / In PR

| Optimization | Status | PR / Notes |
|---|---|---|
| **B11** — FP32 logsumexp intermediates | **Done** (in [#41683](https://github.com/tenstorrent/tt-metal/pull/41683)) | Forward emits single FP32 LSE tile (32 wide, down from 2× 64 BF16). Backward uses fused `apply_softmax_statistics_on_dst`. Ring attention rewritten to logaddexp merge. |
| **FP32 forward buffers** | **Done** (in [#41683](https://github.com/tenstorrent/tt-metal/pull/41683)) | `cb_attention_weights` promoted to FP32. `cb_prev_mm_out`/`cb_cur_mm_out` promoted to FP32 with `UnpackToDestFp32`. Output update rewritten to SFPU for full FP32 DST precision. |
| **FP32 backward buffers** | **Done** (in [#41683](https://github.com/tenstorrent/tt-metal/pull/41683)) | Intermediate CB switched to FP32. Q-kernel: `UnpackToDestFp32` for attention weights (full FP32, no transpose needed). KV-kernel: FP32 data format with Default unpack mode (TF32 through SrcA) — SFPU `transpose_wh_dest` is broken on Blackhole, so FPU path is required for cross-arch compatibility. |
| **Deferred scaling** (partial F1) | **Done** (in [#41683](https://github.com/tenstorrent/tt-metal/pull/41683)) | Scale factor deferred from masking phase to after max-subtraction in `apply_exp_inplace_and_find_exp_sum`. Not a full F1 (scale not fused into exp intrinsic), but removes one SFPU pass from the hot path. |
| **Balanced pairing** (related to B16) | **Done** (already in main) | Light/heavy row pairing for both `sdpa_fw` and `sdpa_bw` via `BALANCED_PARALLELISM` kernel mode. Each pair has constant work `Ht+1`, eliminating per-row work variance. Imbalance comes only from pair-to-core remainder (e.g. 128 pairs / 56 cores = 33% for B=1 GQA on N300, but drops to ~10% at B=4 and <1% for MHA). B16 proposes a further upgrade to host-side LPT scheduling for the small-batch GQA case. |
| **LLK reconfig fix** | **Done** (in [#41683](https://github.com/tenstorrent/tt-metal/pull/41683)) | Replaced `reconfig_data_format(srcA_cb, srcB_cb)` with `reconfig_data_format_srcb(srcB_cb)` for B2D broadcast operations in `init_unary_bcast_col` (FW) and `apply_softmax_statistics_on_dst` (BW). Fixes Blackhole CI hangs without causing Wormhole precision regression. `row_reduce_tile_inplace` uses `mm_init` (full matmul init) instead of `mm_init_short` to properly reset unpacker/math state after broadcast. |
| **F13** — Heavy Row First (LPT) | **Done** (in Phase A PR) | Swap heavy/light row processing order in balanced-parallelism mode so each core processes its longest row first. Applied to all 10 FW + BW kernel files (compute, reader, writer). Classic LPT scheduling reduces tail latency. |
| **F1** — Fuse Scale into Exp (full) | **Done** (in Phase A PR) | Completes the partial F1 from [#41683](https://github.com/tenstorrent/tt-metal/pull/41683). Scale factor is now a compile-time `uint16_t` BF16 constant passed directly to the fused `exp_tile<false, true>(idx, VectorMode::RC, scaler_bf16)` intrinsic. Eliminates the separate `mul_unary_tile` SFPU pass entirely. Applied to `apply_exp_inplace_and_find_exp_sum` and `update_exp_max_diff`. |
| **F3** — `recip_tile_first_column` | **Done** (in Phase A PR) | Custom SFPU intrinsic that processes only column 0 of a tile using `VectorMode::C` (2 faces) with 4 iterations per face and stride-2 access. 4x fewer SFPU iterations than full `recip_tile`. Used in `recip_tile_inplace` for softmax normalization. |
| **F5** — `exp_tile_first_column` | **Done** (in Phase A PR) | Custom SFPU intrinsic for first-column-only fused scale+exp. Uses `_ckernel_sfpu_exp_accurate_` (same 7th-order Taylor series as `exp_tile<false>`) for training-grade accuracy. 4x fewer SFPU iterations. Used in `update_exp_max_diff` for max-correction factor. |
| **B5** — Eliminate redundant u_scalar | **Done** (merged [#39812](https://github.com/tenstorrent/tt-metal/pull/39812)) | Precomputed `u_scalar = rowsum(dO * O)` in a pre-pass kernel. Result stored as `(B, H, S, 32)` tensor. Eliminates O(S²) redundant recomputation in KV-kernel and removes the need to read O tensor in KV backward. ~1.4% step-time improvement. |
| **Different head dim for QK and V** | **Done** (merged) | Support different last dimension for QK and V tensors in SDPA forward and backward. Infrastructure change enabling future head-dim optimizations. |

### In Progress

(No items currently in progress.)

### Blocked

| Optimization | Status | Blocker |
|---|---|---|
| **Full SFPU FP32 softmax** | **Blocked** ([#41686](https://github.com/tenstorrent/tt-metal/issues/41686)) | Requires SFPU row-reduce-max ([#41593](https://github.com/tenstorrent/tt-metal/issues/41593)) and SFPU sub_bcast_col ([#41594](https://github.com/tenstorrent/tt-metal/issues/41594)) from LLK team. Once available: entire softmax (max, sub, exp, sum, recip) stays in DST at FP32, `cb_attention_weights` can revert to BF16, recovering ~20 ms/step regression. |

### Known Issues / Open Problems

- **`mm_init` vs `mm_init_short` in `row_reduce_tile_inplace`**: After `reconfig_data_format_srcb` + B2D broadcast, `mm_init_short` is insufficient to restore matmul state — `mm_init` (full init) is required. Root cause: `mm_init_short` skips `llk_unpack_hw_configure` and `llk_math_hw_configure`, leaving unpacker B and MATH ALU format registers in broadcast-configured state. Possible fix: call `unary_bcast_uninit` after broadcast to clean up, which could re-enable `mm_init_short`. Needs investigation.
- **SFPU `transpose_wh_dest` broken on Blackhole**: Prevents `UnpackToDestFp32` for attention weights in `sdpa_bw_kv` kernel. KV-kernel uses FP32 data format with Default unpack (TF32 through SrcA) + FPU transpose as workaround.
- **`UnpackToDestFp32` with `unary_bcast`**: Confirmed by LLK team that `UnpackToDestFp32` only works with plain `copy_tile`, not with `unary_bcast`. B2D broadcast path with `UnpackToDestFp32` CBs produces NaN/Inf. This limits where `UnpackToDestFp32` mode can be applied.

### Performance Impact (N300 TinyLlama 1B)

| Configuration | Mean step time | vs Baseline |
|---|---|---|
| Baseline (BF16 intermediates, before PR) | 1932 ms | — |
| FP32 intermediates only (B11) | 1912 ms | -1.0% |
| Full PR (FP32 intermediates + FP32 buffers + reconfig fixes) | 1933 ms | +0.1% |
| + B5 (u_scaler) + diff head dim | 1934 ms | — (new baseline) |
| **Phase A (F13 + F1 + F3 + F5)** | **1820 ms** | **-5.9%** |

The FP32 intermediate format itself is slightly faster (fewer tiles: 1 FP32 vs 2 BF16 per row). The ~20 ms overhead comes from FP32 `cb_attention_weights` promotion, which will be recovered by [#41686](https://github.com/tenstorrent/tt-metal/issues/41686).

Phase A optimizations target the SFPU-bound compute pipeline. The ~5.9% step-time reduction
(1934 ms baseline → 1820 ms) comes from a **22% SDPA FW kernel speedup** confirmed via
device profiling:

| Zone | Main (µs) | Phase A (µs) | Improvement |
|---|---|---|---|
| `sdpa-fw-compute` | 11,330.6 | 8,837.9 | **-22.0%** |
| `sdpa-fw-reader` | 11,300.1 | 8,814.3 | **-22.0%** |
| `sdpa-fw-writer` | 11,335.9 | 8,840.8 | **-22.0%** |
| Tail ratio (max/min) | 1.06x | 1.06x | unchanged |

All three pipeline stages improved equally because the kernel is compute-bound: reader and
writer block on circular-buffer synchronization with compute. Training loss convergence is
unchanged (bit-exact deterministic across runs).

---

## Tensix Compute Pipeline Reference

Understanding the hardware pipeline is essential for F12, B7, and any optimization
that aims to overlap matmul with SFPU work. This section documents the Tensix compute
architecture as it applies to SDPA kernel development.

### Three TRISC Threads

Each Tensix compute core runs **3 hardware threads (TRISCs)** in parallel as a pipeline:

```
TRISC0 (UNPACK)  ──(CBs)──▶  TRISC1 (MATH)  ──(DST)──▶  TRISC2 (PACK)  ──(CBs)──▶  output
```

| TRISC | Stage  | Role |
|-------|--------|------|
| TRISC0 | UNPACK | Reads tiles from L1 circular buffers, untilizes, feeds operands to MATH |
| TRISC1 | MATH   | Runs FPU (matmul engine) and SFPU (scalar/vector engine) on DST registers |
| TRISC2 | PACK   | Reads results from DST registers, packs and writes to output circular buffers |

All three TRISCs run the same kernel binary. Per-TRISC `#define`s (`TRISC_UNPACK`,
`TRISC_MATH`, `TRISC_PACK`) enable different code paths. The macros `MATH(x)`,
`PACK(x)`, `UNPACK(x)` route code to the correct thread.

### FPU and SFPU on the MATH Thread

Both **FPU** (tensor/matrix engine — matmul, eltwise binary) and **SFPU** (scalar/SIMD
engine — exp, recip, reduce, etc.) live on TRISC1 (MATH) and share the DST register file.

**They are strictly serialized on TRISC1.** SFPU startup issues:
```cpp
// llk_math_eltwise_unary_sfpu_common.h
TTI_STALLWAIT(p_stall::STALL_SFPU, 0, 0, p_stall::MATH);
// "Stalls till all FPU operations are done"
```

This means: on the MATH thread, you **cannot** run matmul and exp at the same time.
Any kernel that calls `matmul_tiles` followed by `exp_tile` on the MATH thread will
execute them sequentially with the SFPU waiting for the FPU to finish.

### DST Register File and Double-Buffering

The DST register is an array of **16 tiles of 32×32 elements** (source: `reg_api.h`).
It is shared between FPU, SFPU, and PACK.

Access is coordinated by semaphore `MATH_PACK`:

| Function | Thread | Action |
|----------|--------|--------|
| `tile_regs_acquire()` | MATH | Wait for DST availability (`llk_math_wait_for_dest_available`) |
| `tile_regs_commit()` | MATH | Signal "my half is ready for PACK" (`llk_math_dest_section_done`) |
| `tile_regs_wait()` | PACK | Wait for MATH to commit (`llk_packer_wait_for_math_done`) |
| `tile_regs_release()` | PACK | Signal "half is free for MATH" (`llk_pack_dest_section_done`) |

With `DstSync::SyncHalf`, DST is split into two halves (8 tiles each). While MATH
fills one half, PACK can read/process the other. This enables **MATH-PACK overlap**.

### Pack-Thread SFPU — The Key to FPU ‖ SFPU Overlap

SFPU instructions can be issued from **both TRISC1 (MATH) and TRISC2 (PACK)**. A mutex
(`semaphore::FPU_SFPU` / `semaphore::SFPU`) prevents simultaneous SFPU access from
both threads.

The compute API provides **pack-thread SFPU variants** designed for overlap:

| Standard (MATH thread) | Pack-thread variant (PACK thread) |
|-------------------------|-----------------------------------|
| `exp_tile_init()` | `exp_packthread_tile_init()` |
| `exp_tile()` | `exp_packthread_tile()` |
| `recip_tile_init()` | `recip_packthread_tile_init()` |
| `recip_tile()` | `recip_packthread_tile()` |

Source: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`:
```
"Pack-thread variant of exp_tile. Runs the exp computation on the pack thread
 to enable FPU/SFPU overlap with math-thread matmul operations."
```

**This is the mechanism for F12 and B7 overlap**: while MATH runs matmul (FPU) on
DST half A, PACK runs `exp_packthread_tile` (SFPU) on DST half B. They operate on
different DST halves, coordinated by the `SyncHalf` semaphore protocol.

### Overlap Pattern

```
MATH thread:  [─── matmul (FPU) on DST half A ───][─── matmul (FPU) on DST half B ───]
PACK thread:  [─── exp_packthread (SFPU) on DST half B ───][─── pack + exp on half A ───]
              ▲ different DST halves, different HW units ▲
```

Constraints:
- MATH must use only FPU ops (matmul) during overlap — no SFPU calls from MATH
- PACK must acquire the SFPU mutex before using `exp_packthread_tile`
- After PACK finishes SFPU, it must release the mutex before MATH can use SFPU
- `cb_push_back_hold_wr_ptr` makes tiles visible to UNPACK without advancing
  the write pointer, enabling subsequent subblocks to write at stable CB offsets

### Blocker: Current Softmax Uses FPU — Prevents True Overlap

The overlap pattern above assumes softmax is **purely SFPU** (exp, recip). In reality,
our current softmax implementation relies on several **FPU** operations:

| Operation | Current API | Unit | Precision loss |
|-----------|-------------|------|---------------|
| Row max of scores | `reduce_tile` | FPU | SrcA at TF32 |
| Score − max (broadcast) | `sub_tiles_bcast_cols` | FPU | SrcA at TF32 |
| Sum × correction (broadcast) | `mul_tiles_bcast_cols` | FPU | SrcA at TF32 |
| Output × correction (broadcast) | `unary_bcast<COL>` | FPU | SrcA at TF32 |

Since PACK can only issue **SFPU** instructions (not FPU), these operations **cannot**
be moved to the PACK thread. This means F12 and B7 cannot achieve true FPU‖SFPU overlap
for the full softmax — only the `exp` and `recip` portions can overlap with matmul.

### Required SFPU Primitives for True FPU/SFPU Overlap

To make the entire softmax pure-SFPU (enabling full overlap AND full FP32 precision),
three SFPU primitives are needed:

**1. SFPU row reduce (max and sum)**

Reduces a tile row-wise entirely in SFPU registers. Uses `SFPLOAD` to read DST rows
into SFPU local registers, `SFPSWAP` (MAX mode) or `SFPADD` (SUM mode) to reduce
across columns, and `SFPSHFT2` (sub-vector shuffle) for cross-lane reduction within
each SFPU vector. Result stored back via `SFPSTORE`.

Replaces FPU `reduce_tile` — eliminates TF32 truncation, keeps full FP32.

**2. SFPU sub_bcast_col**

Subtracts a column vector (broadcast across all 32 columns) from a full tile, operating
entirely on DST registers via SFPU. Replaces FPU `sub_tiles_bcast_cols`.

**3. SFPU mul_bcast_col**

Multiplies a full tile by a column vector (broadcast), entirely in SFPU/DST. Replaces
FPU `mul_tiles_bcast_cols` and `unary_bcast<COL>` + `mul_binary_tile`.

**What these three primitives enable together:**

- **Full FP32 softmax**: reduce_max → subtract_max → scale → exp → reduce_sum →
  correction_factor → rescale — all in SFPU at full FP32. Only the matmuls (Q@K^T
  and attn_weights@V) remain at TF32 per-element (inherent to FPU matmul).
- **No CB round-trips**: Data stays in DST registers between softmax steps, eliminating
  pack→CB→unpack truncation at every intermediate step.
- **True FPU‖SFPU overlap**: Entire softmax becomes SFPU-only, so PACK thread can run
  full softmax while MATH thread runs matmul — the F12/B7 pipeline becomes viable
  for the complete inner loop, not just the exp portion.

**DeepSeek V3 reference implementation (Blackhole-targeted, but instructions available on Wormhole):**

| Primitive | DeepSeek V3 file | Key functions |
|-----------|-----------------|---------------|
| SFPU row reduce | `models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sdpa_reduce_row.h` | `_calculate_sdpa_reduce_max_row_8x32_`, `_calculate_sdpa_reduce_sum_row_8x32_` |
| Fused max-sub-exp | `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h` | `calculate_fused_max_sub_exp_add_tile`, `non_approx_exp_mul_prev` |
| DST→SrcA/SrcB broadcast | `models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/llk_lib/llk_math_sdpa_bcast_col_srca_srcb_reuse.h` | `MOVD2A`/`MOVD2B` pattern |
| Compute API wrappers | `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h` | `sdpa_reduce_max_row`, `sdpa_reduce_sum_row`, `sdpa_sub_bcast_col_srca_srcb_reuse_tiles` |

**Wormhole B0 instruction availability (confirmed):**

| Instruction | Purpose | Confirmed in |
|-------------|---------|-------------|
| `SFPLOAD` / `SFPSTORE` | DST ↔ SFPU local regs | `ckernel_sfpu_binary_max_min.h` (WH) |
| `SFPSWAP` (ALL_ROWS_MAX) | Element-wise max in SFPU | `ckernel_sfpu_binary_max_min.h` (WH) |
| `SFPADD` | Element-wise add in SFPU | Standard SFPU instruction |
| `SFPSHFT2` (SUBVEC_SHFLSHR1/ROR1) | Cross-lane shuffle for reduction | `sfpi_constants.h` (common); used in DS-V3 WH path |
| `MOVD2A` / `MOVD2B` | DST → SrcA/SrcB direct transfer | `cmath_common.h`, `llk_math_reduce.h` (WH) |
| `_calculate_exponential_piecewise_` | SFPU exp with fused scale | `ckernel_sfpu_exp.h` (WH) |

**Action item**: Confirm with LLK team whether SFPU sub_bcast_col and mul_bcast_col
primitives can be provided (or if we need to build them from the low-level instructions
listed above). The SFPU row reduce already has a complete DeepSeek V3 implementation
that could be ported, though its `Float16_b` format assertion needs investigation for
FP32 DST mode.

### Quasar (Future)

Quasar adds TRISC3 (`UCK_CHLKC_ISOLATE_SFPU`) — a **dedicated SFPU thread**. This
would enable cleaner three-way overlap: UNPACK feeds data, MATH runs FPU (matmul),
TRISC3 runs SFPU (exp/recip) — all concurrently on non-overlapping DST sections.

### Key LLK Source Files

| Purpose | Path |
|---------|------|
| Compute API + TRISC routing | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| DST acquire/release | `tt_metal/hw/inc/api/compute/reg_api.h` |
| Pack-thread exp API | `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h` |
| CB wait/push/pop | `tt_metal/hw/inc/api/compute/cb_api.h` |
| MATH–PACK semaphore | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_common.h` |
| SFPU stall on FPU | `tt_metal/third_party/tt_llk/tt_llk_*/llk_lib/llk_math_eltwise_unary_sfpu_common.h` |
| SFPU mutex definition | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_structs.h` |
| TTNN SDPA streaming | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp` |

---

## Dependency Graph

Many optimizations are **independent** and can be applied to the current single-tile
architecture. Others **require multi-tile K chunking** (F9) as a prerequisite.

```
INDEPENDENT (apply to current architecture):
  F1  Fuse scale into exp
  F2  Approximate exp + ReLU clamping
  F3  recip_tile_first_column
  F5  exp_tile_first_column
  F7  Conditional rescaling (epsilon=0)
  F8  K/V sharing for balanced pairs
  F13 Heavy row first (LPT)
  F14 Uniform dataformat skip

REQUIRE MULTI-TILE K CHUNKING (F9):
  F4  Fused sub_exp + L1 acc sum ──────── needs chunk-level exp tiles to accumulate
  F6  Fused correction block ──────────── needs chunk-level statistics
  F9  Multi-tile K/V chunking ─────────── PREREQUISITE for F4, F6, F10, F12
  F10 Subblock matmul ─────────────────── needs multi-tile operands
  F12 Streaming pipeline (Phase 1+2) ──── needs F9 + F10 + F4

INDEPENDENT OF CHUNKING:
  F11 Software polynomial exp ─────────── alternative exp implementation, any architecture

Backward follows the same pattern:
  B1-B4, B14, B15 are independent.
  B5, B6, B7, B8, B9 are independent.
  B10, B12, B13 benefit from or require chunking.
  B11 is independent but affects both forward and backward.
```

---

# Profiling Results

Measured on Wormhole B0 (N300, 1 GHz clock, 56 active Tensix cores).

Configuration: TinyLlama — B=1, S=2048, D=64, qH=32, kvH=4, causal mask, balanced parallelism enabled.

Profiling tools: TT-Metal Tracy profiler (`python -m tracy`) + `DeviceZoneScopedN` kernel instrumentation.

L1 profiler buffer limit: **125 zones per RISC per kernel launch** (`PROFILER_L1_OPTIONAL_MARKER_COUNT = 250`,
2 markers per zone). This constrains how many inner-loop zones can be captured for long kernels.

## Forward Kernel (sdpa_fw) — TinyLlama

### Kernel Duration (pre-Phase A, before FP32 buffer promotions)

| Metric | Value |
|--------|-------|
| Device kernel duration | **10,257 us (10.26 ms)** |
| TRISC0 (UNPACK) | 10,256 us |
| TRISC1 (MATH) | 10,256 us |
| TRISC2 (PACK) | 10,256 us |
| NCRISC (reader) | 10,233 us |
| BRISC (writer) | 10,257 us |

All RISCs are near-identical, confirming the pipeline is well-balanced across threads.
See [Phase A Results](#phase-a-results--beforeafter-comparison) for current numbers after optimizations.

### Compute vs Memory Bound

**Verdict: STRONGLY COMPUTE BOUND (specifically SFPU-bound)**

Three independent lines of evidence:

**1. Bandwidth analysis — reader delivers data 2.3x faster than compute consumes it**

| Metric | Value |
|--------|-------|
| Data read per pair | 528 KB (Q + K + V tiles) |
| DRAM BW total (Wormhole B0, 6 channels) | 128 GB/s |
| Per-core BW share (÷56 cores) | 2.29 GB/s |
| Time for reader to deliver 1 pair | 237 us |
| Time for compute to process 1 pair | 538 us (measured) |
| **Reader / Compute speed ratio** | **2.3x** (reader is 2.3x faster) |
| DRAM BW utilization | 42% |

Even at 50% DRAM efficiency the reader is still 1.1x faster than compute.

**2. Direct measurement — cb_wait times are near zero (S=256, 100% zone capture)**

| Wait zone | Avg time | What it measures |
|-----------|----------|-----------------|
| WAIT-K | 18 ns | Compute waiting for K tiles from DRAM reader |
| WAIT-V | 19 ns | Compute waiting for V tiles from DRAM reader |
| WAIT-Q | 20 ns | Compute waiting for Q tiles from DRAM reader |
| Compute per KV iteration | ~9,740 ns | Actual compute work |
| **Wait as % of compute** | **0.2%** | Data is always ready in CBs |

18 ns means the data is already waiting in the circular buffer when compute asks for it.

**3. Compute time is dominated by SFPU, not FPU matmul**

78% of compute time is SOFTMAX + RESCALE (SFPU-heavy: exp, reduce-max, sub_bcast).
Only 22% is FPU matmul (QKT + PV). The SFPU is serialized with the FPU on TRISC1
and executes much slower per-element, making it the true bottleneck.

### Load Balance (Balanced Parallelism)

| Metric | Value |
|--------|-------|
| Total query rows | 2048 (B×qH×Ht = 1×32×64) |
| Total pairs | 1024 (rows/2) |
| Distribution | 40 cores × 18 pairs + 16 cores × 19 pairs |
| Per-pair work | 65 KV iterations (constant: light + heavy = Ht + 1) |
| Per-pair time | 538 us ± 6 us (stddev) |
| Fastest core | 9,711 us (18 pairs) |
| Slowest core | 10,253 us (19 pairs) |
| **Imbalance** | **1.056x (5.6%)** |

Balanced parallelism works extremely well — pair work is near-constant.
The 5.6% imbalance comes from 1024 not dividing evenly by 56 (18 vs 19 pairs).

For comparison, the Small config (S=256) **without** balanced parallelism had 1.43x imbalance (43%)
due to uneven row distribution (4 vs 6 rows per core).

### Per-KV-Iteration Time

From the ROW zone data (100% capture, 8192 events):

| Row position | KV iterations | Row duration |
|-------------|--------------|-------------|
| Lightest (seq pos 0) | 1 | 8 us |
| Median | ~32 | 273 us |
| Heaviest (seq pos 63) | 64 | 531 us |

Estimated time per KV iteration: **~8.3 us** (from heaviest row: 531 us / 64 iters).

### Compute Phase Breakdown

Measured on S=256 config with full zone capture (100%, no buffer overflow) and confirmed
on TinyLlama S=2048 partial data. Ratios are consistent across sequence lengths.

| Phase | % of Compute | Description |
|-------|-------------|-------------|
| **RESCALE** | **43.7%** | Rescale previous output by exp(prev_max - cur_max): exp, mul_bcast_cols, L1 accumulate |
| **SOFTMAX** | **34.2%** | Online softmax: reduce-max, sub, exp, pack exp_sum |
| **QKT** | **19.7%** | Q @ K^T matmul + scale + mask |
| **FINALIZE** | ~2% (per row) | reduce_sum → recip → mul_bcast_cols for final normalization |
| **PV** | **2.4%** | P @ V matmul (almost free on TRISC1 — well pipelined with UNPACK/PACK) |

**Key insight**: SOFTMAX + RESCALE = **77.9%** of compute. These are the primary optimization targets.
The RESCALE phase alone (43.7%) is dominated by `update_cur_mm_out` which applies `exp(prev_max - cur_max)`
correction to all qWt output tiles — this happens on every KV iteration except the first.

### Implications for Optimization Priorities

Based on the profiling data:

1. **Highest impact**: Reducing RESCALE (43.7%) and SOFTMAX (34.2%) operations.
   - F1 (Fuse scale into exp) — eliminates separate scale step in SOFTMAX
   - F2 (Approximate exp) — speeds up exp in both SOFTMAX and RESCALE
   - F7 (Conditional rescaling) — skips RESCALE when max doesn't change (common for later rows)
   - F12 (Streaming pipeline) — overlaps FPU matmul with SFPU softmax/rescale

2. **Medium impact**: Reducing QKT (19.7%).
   - F9/F10 (Multi-tile K chunking + subblock matmul) — improves matmul efficiency

3. **Low compute impact but useful**: PV is only 2.4% on TRISC1.
   The P@V matmul is almost entirely hidden by the UNPACK/PACK pipeline.

4. **Load balance is already good**: Balanced parallelism achieves 5.6% imbalance.
   F13 (LPT scheduling) could reduce this further but the gain is marginal.

### Phase A Results — Before/After Comparison

After implementing F13, F1, F3, F5 (Phase A PR), the forward kernel was re-profiled
with the same TinyLlama configuration and `DeviceZoneScopedN` instrumentation:

| Zone | Main (µs) | Phase A (µs) | Improvement |
|------|-----------|-------------|-------------|
| `sdpa-fw-compute` (avg) | 11,330.6 | 8,837.9 | **-22.0%** |
| `sdpa-fw-reader` (avg) | 11,300.1 | 8,814.3 | **-22.0%** |
| `sdpa-fw-writer` (avg) | 11,335.9 | 8,840.8 | **-22.0%** |
| Max (kernel latency) | 11,775.8 | 9,186.1 | **-22.0%** |
| Tail ratio (max/min) | 1.06x | 1.06x | unchanged |

All three pipeline stages improved by the same percentage — this confirms the kernel
remains compute-bound after Phase A. The reader/writer durations dropped because they
spend less time blocked on circular-buffer synchronization with compute.

**Note:** The "Main" baseline numbers (11,330 µs) are higher than the earlier profiling
run (10,257 µs) because the current main includes FP32 buffer promotions from
[#41683](https://github.com/tenstorrent/tt-metal/pull/41683). Phase A more than
compensates for that overhead.

**Training step time impact** (1200-step TinyLlama on Shakespeare):
- Main baseline: ~1955 ms/step
- Phase A: ~1842 ms/step → **~5.6% faster**
- Loss convergence unchanged (bit-exact deterministic across runs)

---

## Backward Kernels (sdpa_bw) — TinyLlama

The backward pass launches **two separate kernel programs** on all 56 cores:
1. **KV-kernel** (`sdpa_bw_kv_compute_kernel`) — computes dK and dV
2. **Q-kernel** (`sdpa_bw_q_compute_kernel`) — computes dQ

### Kernel Duration

| Kernel | Device Duration | BRISC | NCRISC | TRISC0 | TRISC1 | TRISC2 |
|--------|----------------|-------|--------|--------|--------|--------|
| KV-kernel | **14,619 us (14.62 ms)** | 14,618 us | 14,600 us | 14,617 us | 14,618 us | 14,618 us |
| Q-kernel | **7,537 us (7.54 ms)** | 7,536 us | 7,523 us | 7,536 us | 7,537 us | 7,536 us |
| **Combined backward** | **22,156 us (22.16 ms)** | — | — | — | — | — |
| Forward (for reference) | 10,231 us (10.23 ms) | — | — | — | — | — |

**Backward/forward ratio: 2.17x** — consistent with ~2x more compute per inner iteration
(4 matmuls + SFPU in backward vs 2 matmuls + SFPU in forward).

### Compute vs Memory Bound

**Verdict: STRONGLY COMPUTE BOUND (both kernels)**

**Evidence 1 — TRISC_1 finishes after NCRISC on every core:**

| Kernel | TRISC_1 − NCRISC gap (min) | TRISC_1 − NCRISC gap (max) | TRISC_1 − NCRISC gap (avg) |
|--------|---------------------------|---------------------------|---------------------------|
| KV-kernel | +14,025 ns | +18,543 ns | +16,258 ns |
| Q-kernel | +9,370 ns | +10,038 ns | +9,724 ns |

**Evidence 2 — Direct cb_wait_front measurement (zero data stalls):**

| Zone | Measures | Per-zone avg | Per-core total (max) | % of ROW |
|------|----------|-------------|---------------------|----------|
| `SDPA-BW-KV-WAIT-KV` | cb_wait(K) + cb_wait(V) at row start | **23 ns** | 160 ns | **0.001%** |
| `SDPA-BW-Q-WAIT` | cb_wait(O, dO, Q) at row start | **20 ns** | 760 ns | **0.01%** |

Data is **always already in the circular buffer** when compute asks for it. The reader
pre-loads everything before compute needs it.

**Evidence 3 — HEAD zone consistency (no inner-loop stalls):**

Within the same KV row, all 8 `heads_per_group` HEAD iterations have near-identical durations:

| Metric | Value |
|--------|-------|
| HEAD[0] (first, includes pipeline startup) | 394,137 ns |
| HEAD[1-7] avg | 373,027 ns |
| HEAD[1-7] std dev | **471 ns** (0.13%) |
| Pipeline startup overhead | 21,110 ns (5.4%) |

If the inner loop were memory-bound, HEAD durations would fluctuate due to DRAM contention.
The sub-0.2% variation proves the pipeline is compute-limited, not data-limited.

**Evidence 4 — Bandwidth utilization:**

| Metric | KV-kernel | Available |
|--------|-----------|-----------|
| Data per HEAD (row 0, 64 Q-tiles × 16 KB/iter) | 1,024 KB | — |
| HEAD duration | 374,063 ns | — |
| Effective BW needed per core | **2.80 GB/s** | ~5.36 GB/s |
| BW utilization | **52%** | — |
| Reader headroom | **1.9x** | — |

The reader uses only half the available bandwidth — the other half is idle time where
the reader blocks on `cb_reserve_back` waiting for compute to consume data.

**Evidence 5 — KV-kernel time breakdown (heaviest core):**

| Component | Time (ns) | % of ROW |
|-----------|-----------|----------|
| **HEAD (compute)** | **14,568,530** | **99.89%** |
| WRITE (output pack) | 12,938 | 0.09% |
| WAIT-KV (data load) | 160 | 0.001% |
| Overhead | 3,268 | 0.02% |
| **ROW total** | **14,584,896** | **100%** |

**Evidence 6 — Q-kernel time breakdown (heaviest core):**

| Component | Time (ns) | % of ROW |
|-----------|-----------|----------|
| **Inner loop (compute)** | **7,498,963** | **99.3%** |
| USCALAR | 50,917 | 0.7% |
| WAIT (data load) | 760 | 0.01% |
| **ROW total** | **7,550,640** | **100%** |

### Load Balance — **KV-kernel is the critical problem**

| Kernel | Per-core min | Per-core max | Imbalance | Rows/core |
|--------|-------------|-------------|-----------|-----------|
| **KV-kernel** | **9,745 us** | **14,617 us** | **33.3%** | 4–6 |
| Q-kernel | 7,132 us | 7,533 us | 5.3% | 36–38 |

**The KV-kernel's 33.3% load imbalance is the single largest backward bottleneck.**

Root cause: With balanced parallelism, the KV-kernel creates B × kvH × Ht/2 = 1 × 4 × 32 = **128 pairs**.
128 pairs ÷ 56 cores doesn't divide evenly:
- **40 cores** get 2 pairs (4 rows) → **~9,745 us**
- **16 cores** get 3 pairs (6 rows) → **~14,617 us**

Each pair is perfectly balanced internally (light row + heavy row = constant work), but
the extra pair on 16 cores adds **~4,872 us (33%)** of idle time on 40 cores.

### Per-Row Duration Distribution (KV-kernel)

| Percentile | Duration (ns) |
|------------|---------------|
| Min | 82,101 |
| P25 | 1,275,884 |
| P50 | 2,467,934 |
| P75 | 3,660,367 |
| Max | 4,803,481 |

**Max/min ratio: 58.5x** — the heaviest KV row (row 0, processes all 64 × 8 = 512 Q iterations)
vs lightest KV row (row 63, processes 1 × 8 = 8 Q iterations). Balanced pairing compensates,
but the pair-to-core distribution remains the limiting factor.

### Key Optimization Priorities for Backward (Informed by Profiling)

Since both kernels are **strongly compute-bound** (WAIT < 0.01%, BW utilization 52%),
optimizations that reduce compute cycles per inner iteration have the highest impact.
Memory/bandwidth optimizations (B8, B9, B11) have secondary impact — they reduce
reader work but the reader is already idle 48% of the time.

**Priority 1 — KV-kernel load imbalance (33.3%) [~3.6 ms saving]**

The KV-kernel takes 14.62 ms but ideal balanced time is ~10.97 ms. The problem is
structural: 128 pairs ÷ 56 cores → 40 cores idle for 33% of wall time.

Root cause for GQA: `heads_per_group = 8` moves all head accumulation INSIDE each KV
row (the `for head_idx` loop). This reduces total KV rows from 2048 (MHA) to 256 (GQA),
cutting distributable work units to 128 pairs. With MHA the same config would have
1024 pairs / 56 cores = 5.3% imbalance.

**Constraint:** Each core must process all `heads_per_group` heads for a given KV row
to avoid cross-core reduction for dK/dV. This means we cannot split the head loop
across cores without B10-style reduction infrastructure.

**Solution: LPT scheduling of individual rows (no pairing) — see B16 below.**

Simulation shows LPT reduces KV-kernel imbalance from **33.3% to 5.2%**, saving
**~3.07 ms (21%)** with zero kernel architecture changes (host-side scheduling only).

For larger batches, even balanced pairing improves: B=2 → 20%, B=4 → 10%, B=8 → 5.3%.

**Priority 2 — B7: Overlap softmax recomp with gradient matmuls [~30-40% saving on compute]**

This is the single highest-impact compute optimization. Currently FPU (matmul) and SFPU
(softmax recompute: exp, mul, sub, reduce) are serialized on TRISC1 — each unit is idle
~50% of the time. B7 runs softmax on the PACK thread via `exp_packthread_tile` while
gradient matmuls run on MATH, using `SyncHalf` double-buffering.

The backward inner loop has 4 matmuls + SFPU recompute. If SFPU recomp for iteration j
fully overlaps with matmuls for iteration j-1, effective compute time drops by up to
the SFPU fraction (~40-50% of each iteration based on forward profiling).

Applies to BOTH kernels. For KV-kernel heaviest core: could reduce HEAD time from
14.57 ms to ~9-10 ms. For Q-kernel: from 7.50 ms to ~5 ms.

**Priority 3 — B5: Eliminate redundant u_scalar [saves compute + eliminates O reads]**

In the KV-kernel, `compute_u_scalar_row(dO, O)` runs inside the inner loop for every
(head, Q-tile) pair. For row 0: computed 512 times, but u_scalar only depends on
(dO[q], O[q]) — same for all KV rows and heads. Pre-computing once eliminates:
- Redundant elementwise multiply + reduce per iteration
- The entire O tensor read from the KV-kernel reader

B10.1 (partial fusion) already implements this via `USE_PRECOMPUTED_U_SCALER`.

**Priority 4 — B6: Transposed recomputation [saves 2 transposes per iteration]**

Eliminates `transpose_wh_tile` (L1 round-trip: PACK → scratch CB → UNPACK) called
twice per inner iteration in KV-kernel. Each transpose is ~500-1000 ns of PACK/UNPACK
pipeline stall. For 520 iterations: ~0.5-1 ms saving on heaviest core.

**Priority 5 — B1/B3/B4 (SFPU quick wins) [saves ~5-10% SFPU cycles each]**

All three are LOSSLESS and low effort. Since both kernels are SFPU-bound:
- B1: Fuse scale into exp → saves one mul_unary_tile per softmax recomp
- B3: recip_first_column → saves 31/32 of recip SFPU cycles
- B4: exp_first_column → saves 31/32 of exp cycles in correction path

**Lower priority — Memory/bandwidth optimizations:**

- **B8/B9 (data sharing)**: Reduces DRAM reads per pair by ~25%. But the reader already
  has 1.9x headroom — it's idle waiting for compute. Benefit is marginal unless compute
  optimizations (B7) shift the bottleneck toward memory.
- **B11 (logsumexp)**: Halves intermediate reads. Same reasoning — marginal while compute-bound.
- **B14 (dataformat skip)**: Saves a few ns per reconfig. Negligible.
- **B15 (conditional rescaling)**: Skips rescaling when max is stable. Modest benefit.

---

# Part 1: SDPA Forward Optimizations

## Tier 1: Independent — Apply Now (no chunking needed)

### F1. Fuse Scale into Exp

> **Status: DONE** (Phase A PR) — Scale factor is now a compile-time `uint16_t` BF16
> constant passed to the fused `exp_tile<false, true>(idx, VectorMode::RC, scaler_bf16)`
> intrinsic. The separate `mul_unary_tile(scaler_bits)` + `exp_tile` sequence is replaced
> by a single SFPU pass in both `apply_exp_inplace_and_find_exp_sum` and
> `update_exp_max_diff`. Functions are now templated on `scaler_fp32`.

| | |
|---|---|
| **Source** | TTNN `sub_exp_block_bcast_cols_inplace` |
| **Impact** | Medium |
| **Effort** | Low |
| **Accuracy** | **LOSSLESS** — mathematically equivalent; intermediate stays in higher-precision DST instead of BF16 CB round-trip |

**Current code** (`sdpa_fw_compute_kernel.cpp:116-118`):
```cpp
binop_with_scalar_tile_init();
mul_unary_tile(matmul_accum_reg, scaler_bits);
```
Then later in `apply_exp_inplace_and_find_exp_sum` (`sdpa_compute_utils.hpp:110-111`):
```cpp
exp_tile_init</* approx */ false>();
exp_tile</* approx */ false>(exp_dst_idx);
```

**TTNN approach** — single call:
```cpp
exp_tile_init<true, true, scale_fp32, InputClamping::None>();
exp_tile<true, true, false, false, InputClamping::None, iterations>(j, vector_mode_exp);
```

**Change:** Remove the separate `mul_unary_tile(scaler_bits)` call. Pass the scale factor as a
compile-time constant into `exp_tile`. The exp computes `exp((x - max) * scale)` in one SFPU
pass. Also apply to `update_exp_max_diff` where `exp(prev_max - cur_max)` should carry the
same scale factor.

**Why it's safe:** The mathematical operation is identical: `exp(a * s) = exp(a * s)`. The only
difference is that intermediate `(score - max)` stays in DST (19-bit mantissa on Wormhole)
instead of being rounded to BF16 before the multiply. This actually **improves** precision.

---

### F2. Approximate Exp + ReLU Clamping

| | |
|---|---|
| **Source** | TTNN `compute_common.hpp:311-312` + FA4 |
| **Impact** | High |
| **Effort** | Very Low |
| **Accuracy** | **MODERATE RISK** — requires training convergence validation |

**Current code** (`sdpa_compute_utils.hpp:110-111`):
```cpp
exp_tile_init</* approx */ false>();
exp_tile</* approx */ false>(exp_dst_idx);
```

**TTNN approach:**
```cpp
exp_tile_init<true /* approx */, true /* fast+approx */, scale_fp32, InputClamping::None>();
PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
// ... exp operations ...
PACK((llk_pack_relu_config(ReluType::NO_RELU)));
```

**Change:** Switch to `exp_tile<true>()` (standard approximate). Enable packer ReLU before
the exp block, disable after. Since softmax inputs are always ≤ 0 (after max subtraction),
ReLU correctly zeros any garbage from the approximation.

**Accuracy analysis:**
- `exp_tile<true>()` (standard approximate): ~0.1-0.3% relative error. Within BF16's 0.78%
  precision. Likely safe for training.
- `exp_tile<true, true>()` (fast+approximate): ~0.5-1% relative error. Comparable to BF16
  precision. More risky for training.
- ReLU clamping: **accuracy-neutral or beneficial** — values below BF16 precision are zeroed
  instead of represented as garbage.

**Critical rule for training:** If you use approximate exp in forward, you **MUST** use the
same approximation in backward (B2). If `P_forward = approx_exp(x)` but
`P_backward = exact_exp(x)`, the mismatch directly corrupts gradients:
`dS = P_backward * (dP - u)` uses wrong P values.

**Recommendation:** Start with `exp_tile<true>()` only (not fast+approx). Validate with
1000+ training steps. Graduate to `exp_tile<true, true>()` only if convergence is confirmed.

---

### F3. recip_tile_first_column

> **Status: DONE** (Phase A PR) — Implemented as custom SFPU intrinsic in
> `sdpa_compute_utils.hpp`. Uses `_sfpu_reciprocal_` with `VectorMode::C` (2 faces)
> and 4 iterations per face with stride-2 access. Handles `DST_ACCUM_MODE` for FP32 DST.
> Replaces `recip_tile` in `recip_tile_inplace` via `MATH((recip_tile_first_column(dst_idx)))`.

| | |
|---|---|
| **Source** | TTNN `compute_common.hpp:220-264` |
| **Impact** | Medium |
| **Effort** | Low |
| **Accuracy** | **LOSSLESS** — only processes columns containing valid data; other columns are garbage from row reduction |

**Why it's safe:** After `reduce_max_row` or `reduce_sum_row`, only the first 8 elements per
face (columns 0:8) contain valid data. Standard `recip_tile` wastes 75% of SFPU cycles on
garbage in columns 8:32. This function is identical on the valid columns.

---

### F5. exp_tile_first_column for Max Correction

> **Status: DONE** (Phase A PR) — Implemented as custom SFPU intrinsic in
> `sdpa_compute_utils.hpp`. Uses `_ckernel_sfpu_exp_accurate_` (same 7th-order Taylor
> series as `exp_tile<false>`) for training-grade accuracy with fused scale. Combined
> with `VectorMode::C` and stride-2 access, gives 4x fewer SFPU iterations. Replaces
> the full-tile `exp_tile` in `update_exp_max_diff` via
> `MATH((exp_tile_first_column<scaler_bf16>(exp_max_diff_dst_idx)))`.

| | |
|---|---|
| **Source** | TTNN `compute_common.hpp:827-857` |
| **Impact** | Medium |
| **Effort** | Low |
| **Accuracy** | **LOSSLESS** — same reasoning as F3; only valid first-column data is processed. Uses `_ckernel_sfpu_exp_accurate_` (not the polynomial approximation from TTNN inference) for identical accuracy to full `exp_tile<false>`. |

**Implementation note:** TTNN's `exp_tile_first_column` uses `_calculate_exponential_piecewise_`
(degree-4 polynomial), which is insufficient for training. Our implementation uses
`_ckernel_sfpu_exp_accurate_<true, DST_ACCUM_MODE>` — the same function behind
`exp_tile<false, true>` — preserving full accuracy while getting the 4x iteration savings.

---

### F7. Conditional Online Softmax Rescaling

| | |
|---|---|
| **Source** | FlashAttention-4 §Forward |
| **Impact** | High (for causal attention) |
| **Effort** | Low |
| **Accuracy** | **LOSSLESS with epsilon=0**; **UNACCEPTABLE with epsilon>0** for training |

**Current code** (`sdpa_fw_compute_kernel.cpp:168-178`):
```cpp
if (h > 0) {
    update_exp_max_diff(alias_cb_prev_max, alias_cb_cur_max, cb_exp_max_diff);
    // ... always runs full rescaling
}
```

**Change:** After computing `cur_max`, compare with `prev_max`. If they are bit-identical
(same BF16 word), skip the entire correction block — `exp(0) = 1.0`, multiply by 1.0 is no-op.

```cpp
if (h > 0) {
    bool max_changed = !are_tiles_equal(alias_cb_prev_max, alias_cb_cur_max);
    if (max_changed) {
        update_exp_max_diff(...);
        update_cur_exp_sum_inplace(...);
        update_cur_mm_out(...);
    } else {
        // Just add cur_sum to prev_sum, add cur_mm_out to prev_mm_out
        add_tiles_inplace(alias_cb_cur_sum_exp, alias_cb_prev_sum_exp);
        add_tiles_inplace(alias_cb_cur_mm_out, alias_cb_prev_mm_out, qWt);
    }
}
```

For causal attention, the max typically stabilizes after the first ~2-4 K tiles in a row.
For a row of length `k`, the remaining `k-4` iterations skip the expensive
exp + broadcast-multiply correction.

**Accuracy with epsilon=0:** When `prev_max == cur_max` bit-exact, `exp(prev_max - cur_max)
= exp(0) = 1.0` exactly. Multiplying by 1.0 is a no-op. Skipping is identical.

**Why epsilon>0 is unacceptable for training:** With epsilon-based skipping, you accumulate a
multiplicative error of `exp(delta)` per skipped tile, where `0 < delta < eps`. Over `n`
skips this grows to `exp(n * eps)`. This is a systematic bias that doesn't cancel and causes
gradient drift over thousands of training steps.

---

### F8. K/V Sharing for Balanced Pairs

| | |
|---|---|
| **Source** | `BALANCED_PARALLELISM.md` Future Optimizations |
| **Impact** | High |
| **Effort** | Medium |
| **Accuracy** | **LOSSLESS** — pure data movement; same values fed to compute |

**Current code** (`sdpa_fw_compute_kernel.cpp:269-273`):
```cpp
process_single_row(light_global_row);
process_single_row(heavy_global_row);
```
Each row independently reads all its K/V tiles from DRAM.

**Change:** Light row `r` reads K/V tiles `[0..r]`, heavy row `Ht-1-r` reads `[0..Ht-1-r]`.
Since `r < Ht-1-r`, tiles `[0..r]` are read twice. Interleave the two rows' processing:
read each K/V tile once, feed both rows' softmax/matmul engines.

**Savings:** Eliminates ~25% of K/V DRAM traffic averaged across all pairs.

---

### F13. Heavy Row First (LPT Scheduling)

> **Status: DONE** (Phase A PR) — Heavy/light row order swapped in all 10 FW + BW
> kernel files (compute, reader, writer for both sdpa_fw, sdpa_bw_q, and sdpa_bw_kv).

| | |
|---|---|
| **Source** | FlashAttention-4 §Scheduling |
| **Impact** | Low |
| **Effort** | Very Low (swap 2 lines) |
| **Accuracy** | **LOSSLESS** — just execution order of independent computations |

The last task each core completes is the short one, reducing the gap between fastest and
slowest core. Classic longest-processing-time-first scheduling.

---

### F14. Uniform Dataformat Skip

| | |
|---|---|
| **Source** | TTNN `sdpa_program_factory.cpp:107-121` |
| **Impact** | Low |
| **Effort** | Very Low |
| **Accuracy** | **LOSSLESS** — skips no-op configuration calls |

**Current code** has many `pack_reconfig_data_format` and `reconfig_data_format` calls per
tile iteration (e.g., `sdpa_compute_utils.hpp:93`, `:118`, `:127`, `:142-143`, `:171`, `:186`).

**Change:** Add compile-time flag `uniform_dataformat`. When all CBs use the same format
(e.g., all Float16_b), guard out all reconfig calls:
```cpp
if constexpr (!uniform_dataformat) {
    pack_reconfig_data_format(cb_out);
    reconfig_data_format(cb_in0, cb_in1);
}
```

---

## Tier 2: Require Multi-Tile K Chunking (F9 first)

### F9. Multi-Tile K/V Chunking (PREREQUISITE)

| | |
|---|---|
| **Source** | TTNN `compute_common.hpp:1575-1990` + existing TODO at `sdpa_compute_utils.hpp:99-100` |
| **Impact** | Very High |
| **Effort** | High |
| **Accuracy** | **NEGLIGIBLE RISK** — online softmax is mathematically equivalent at any chunk size; DST accumulation may slightly improve matmul precision |

**Current code** (`sdpa_fw_compute_kernel.cpp:94`):
```cpp
for (uint32_t h = 0; h < num_kv_tiles_to_process; ++h) {
    // Process 1 K/V tile per iteration
}
```

**TTNN architecture:**
```cpp
// Process Sk_chunk_t K/V tiles per iteration
matmul_blocks(cb_q_in, cb_k_in, cb_qk_im,
    Sq_chunk_t, Sk_chunk_t, DHt, ...);         // Q chunk @ K chunk
sub_exp_block_bcast_cols_inplace<...>(...);     // Softmax on full chunk
matmul_blocks(cb_qk_im, cb_v_in, cb_out_im,
    Sq_chunk_t, vDHt, Sk_chunk_t, ...);        // P chunk @ V chunk
```

**Change:** Increase inner-loop granularity from 1 tile to `Sk_chunk_t` tiles (e.g., 2 or 4).
Each iteration:
1. `Q_row @ K_chunk^T` → `1 × Sk_chunk_t` score tiles (one `matmul_blocks` call)
2. Softmax over the chunk (max, sub, exp, sum — all on `Sk_chunk_t` tiles)
3. `P_chunk @ V_chunk` → `1 × qWt` output tiles (one `matmul_blocks` call)
4. Rescale once per chunk instead of per tile

**Why this matters:** This is the single biggest architectural difference between your SDPA
and TTNN's. Everything in Tier 2 depends on it:
- **F4** needs multiple exp tiles to L1-accumulate
- **F6** needs chunk-level statistics for fusion
- **F10** needs multi-tile operands for subblock utilization
- **F12** needs chunked processing for the Phase 1/Phase 2 pipeline

**Accuracy note:** The online softmax algorithm is mathematically identical regardless of
chunk size. The correction `exp(prev_max - cur_max)` happens less frequently (once per chunk
instead of per tile) but each correction is exact. Within a chunk, DST accumulation keeps
matmul results in 19-bit precision before BF16 pack — potentially **more accurate** than
single-tile processing.

---

### F4. Fused sub_exp + L1 Accumulation for Row Sum

| | |
|---|---|
| **Source** | TTNN `compute_common.hpp:300-386` |
| **Impact** | High |
| **Effort** | Medium |
| **Accuracy** | **NEGLIGIBLE RISK** — same sum, different accumulation order; L1 acc uses same BF16 precision as current sequential packing |
| **Requires** | F9 (multi-tile chunking) for full benefit |

**Current code** (`sdpa_compute_utils.hpp:121-128`):
```cpp
/* at the moment we pack one tile here
 * but we can use L1 accumulator to pack more tiles
 * in case we will be able to read more than one row of K and V */
cb_reserve_back(cb_cur_exp_sum, onetile);
pack_tile(exp_dst_idx, cb_cur_exp_sum);
```
Then later, `generate_matmul_row_reduce_tile` reduces the sum tile per iteration.

**TTNN approach** — L1 accumulation within `sub_exp_block_bcast_cols_inplace`:
```cpp
if constexpr (do_reduce) {
    if (u > 0) {
        PACK((llk_pack_reconfig_l1_acc(1)));  // Enable L1 accumulation
    }
    for (uint32_t j = 0; j < dst_tiles; ++j) {
        pack_tile<true>(j, reduce_cb, i);     // Accumulates onto existing tile
        if (u == 0 && j == 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
    }
}
// After all K chunks: one matmul_reduce to collapse partial sum
matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);
```

**Change:** Instead of packing full exp tiles to `cb_cur_exp_sum` and reducing per iteration,
L1-accumulate exp results into a partial-sum tile. After all K tiles are processed, a single
`matmul_reduce` (1×N matmul with column-identity vector) produces the final scalar sum.

**Why it requires F9:** With single-tile processing you only have 1 exp tile per iteration —
L1 accumulation of 1 tile is just a pack. The benefit comes when you have `Sk_chunk_t` exp
tiles to accumulate per iteration, eliminating per-tile reduce calls.

**Accuracy note:** The sum of exp values is computed via L1 accumulation (BF16 add) instead
of sequential pack + matmul-reduce. Both produce BF16 sums. The associativity difference
`(a+b)+c vs a+(b+c)` is at most 1 ULP per addition — standard numerical noise, not
systematic bias.

---

### F6. Fused Correction Block — Single SFPU Pass

| | |
|---|---|
| **Source** | TTNN `compute_common.hpp:901-952` |
| **Impact** | Medium |
| **Effort** | Medium |
| **Accuracy** | **SLIGHTLY BETTER** — eliminates 2 intermediate BF16 quantization steps by keeping values in 19-bit DST registers |
| **Requires** | F9 (multi-tile chunking) for meaningful benefit |

**Current code** — 3 separate DST acquire/release cycles:
```cpp
update_exp_max_diff(prev_max, cur_max, exp_max_diff);     // sub + exp → BF16 CB
update_cur_exp_sum_inplace(prev_sum, cur_sum, exp_diff);   // unpack → mul + add → BF16 CB
update_cur_mm_out(qWt, prev_out, cur_out, exp_diff);       // unpack → bcast mul + add → BF16 CB
```
Each function packs intermediate results to a BF16 CB (7-bit mantissa) and the next function
unpacks them. Two unnecessary BF16 round-trips.

**TTNN approach** — `fused_max_sub_exp_add_tile`:
```cpp
// Loads 5 tiles into DST. Single SFPU pass computes:
// cur_max = max(prev_max, worker_max)
// exp_prev = exp((prev_max - cur_max) * scale)
// exp_worker = exp((worker_max - cur_max) * scale)
// cur_sum = exp_prev * prev_sum + exp_worker * worker_sum
// All intermediates stay in 19-bit DST registers.
```

**Change:** Implement a fused SFPU function for the statistics correction. The output
rescaling (`update_cur_mm_out` over qWt tiles) remains separate since it operates on many
tiles, but the max + sum correction collapses from 3 DST cycles to 1.

---

### F10. Subblock Matmul for Better DST Utilization

| | |
|---|---|
| **Source** | TTNN `sdpa_subblock_utils.hpp` + `compute_common.hpp:1177-1259` |
| **Impact** | High |
| **Effort** | Medium |
| **Accuracy** | **LOSSLESS to SLIGHTLY BETTER** — same FMAs, more accumulation stays in high-precision DST |
| **Requires** | F9 (multi-tile chunking) |

**Current code** (`sdpa_compute_utils.hpp:144-152`):
```cpp
for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
    tile_regs_acquire();
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        matmul_tiles(cb_qk_result, cb_value, 0, tile_idx + block_idx, block_idx);
    }
    // ... pack block_size tiles
}
```
Processes `block_size` (up to 4) output tiles per DST cycle, but with a 1×1 input shape.

**TTNN approach** — `matmul_blocks`:
```cpp
mm_block_init_short(in0_cb, in1_cb, transpose,
    subblock_w, subblock_h, in0_block_w);
// Processes subblock_h × subblock_w output tiles per DST cycle.
// Candidates: {2,4}, {4,2}, {1,8}, {8,1}, ... up to 8 tiles per cycle.
matmul_block(in0_cb, in1_cb, in0_index, in1_index, dst_index,
    transpose, subblock_w, subblock_h, in0_block_w);
```

**Change:** Replace `matmul_tiles` with `matmul_block` using subblock dimensions.
For Q@K^T with Sk_chunk_t=4: use subblock {1, 4} = 4 tiles per cycle.
For P@V with qWt=4: use subblock {1, 4} or {2, 2} depending on chunk dimensions.

Use `determine_largest_subblock_size` from TTNN to auto-select optimal subblocks:
```cpp
auto [sbh, sbw] = determine_largest_subblock_size(block_height, block_width, 8);
```

---

### F11. Software Polynomial Exp (FA4 Horner)

| | |
|---|---|
| **Source** | TTNN `compute_common.hpp:663-822` + FA4 |
| **Impact** | Medium |
| **Effort** | Medium |
| **Accuracy** | **LOW RISK** for degree ≥ 2; degree 2 has ~0.05% error (15x below BF16 precision) |
| **Requires** | Nothing (independent), but most useful when SFPU is the bottleneck |

TTNN's `calculate_exponential_polynomial` implements Cody-Waite range reduction + Horner:
```
x → k = round(x / ln2), r = x - k * ln2
exp(r) ≈ c0 + c1*r + c2*r² + ... (Horner evaluation)
exp(x) = exp(r) * 2^k (via SFPSETEXP)
```

| Degree | Max Relative Error | vs BF16 (0.78%) | Training Safety |
|---|---|---|---|
| 1 | ~3% | Worse | **Unacceptable** |
| 2 | ~0.05% | 15x better | Safe |
| 3 | ~0.002% | 400x better | Safe |
| 4 | ~0.00008% | 10000x better | Safe |

**Change:** Port the polynomial exp from TTNN. Use degree 2 for BF16, degree 4 for
FP32 accumulator mode. Can either replace hardware exp entirely or split work between
SFPU (hardware exp) and FPU (polynomial exp) for parallel execution.

**Critical constraint:** Same polynomial must be used in forward and backward for
`P_recomputed == P_forward` consistency.

---

### F12. Streaming Pipeline — Phase 1 + Phase 2 Overlap

| | |
|---|---|
| **Source** | TTNN `compute_streaming.hpp` |
| **Impact** | Very High |
| **Effort** | Very High |
| **Accuracy** | **LOSSLESS** — pure scheduling; same operations in same order, just overlapped |
| **Requires** | F9 + F10 + F4 |

**Current problem:** In the inner loop, FPU (matmul) and SFPU (softmax) alternate on
the MATH thread, each idle ~50% of the time:
```
for each K/V chunk:
  1. Q @ K^T          [FPU on MATH, SFPU idle, PACK idle]
  2. scale + softmax   [SFPU on MATH, FPU idle, PACK idle]
  3. P @ V             [FPU on MATH, SFPU idle, PACK idle]
  4. rescale output    [SFPU on MATH, FPU idle, PACK idle]
```

**Hardware mechanism:** FPU and SFPU cannot run concurrently on the MATH thread (see
"Tensix Compute Pipeline Reference" above). The overlap exploits **MATH thread (FPU) vs
PACK thread (SFPU)** parallelism using `exp_packthread_tile` and DST double-buffering:

```
Iteration i:
  MATH (TRISC1):  matmul_block(Q, K[i]) using FPU    → writes DST half A
                  tile_regs_commit()                   → release half A
  PACK (TRISC2):  tile_regs_wait()                    → acquire DST half B
                  exp_packthread_tile(scores[i-1])     → SFPU on half B
                  pack_tile() + cb_push_back_hold_wr_ptr()
                  tile_regs_release()                  → free half B
```

MATH uses FPU on half A while PACK uses SFPU on half B — different hardware units,
different DST halves, running in parallel.

TTNN's `sdpa_inner_loop_step` implements this as a two-phase pipeline:

**Phase 1 — Q@K^T + in-place softmax pipeline:**
For each Q subblock, MATH computes Q@K^T via `matmul_block` into `cb_qkt_im` using
`pack_tile<true>` at absolute offsets. Key trick: `cb_push_back_hold_wr_ptr` makes tiles
visible to UNPACK without advancing the write pointer, so subsequent subblocks can keep
writing at stable offsets. While MATH computes Q@K^T for subblock `i`, PACK runs
`sub_exp_block_bcast_cols_inplace` (which uses `exp_packthread_tile`) on subblock `i-1`.

**Phase 2 — QK^T@V + SALAD corrections:**
For each Q subblock, MATH computes QK^T@V via `matmul_block`. While MATH computes V
matmul for subblock `i`, PACK runs SALAD corrections (exp_max_diff + mul_bcast_cols)
for subblock `i-1`. On the last K chunk, fuse per-row normalization (matmul_reduce +
recip + mul_bcast_cols).

**Implementation constraints:**

1. **Must use pack-thread SFPU API**: `exp_packthread_tile`, not `exp_tile`. These are
   distinct code paths that issue SFPU instructions from TRISC2 instead of TRISC1.
2. **DST half management**: MATH and PACK must never access the same DST half. The
   `SyncHalf` protocol with `tile_regs_acquire/commit/wait/release` enforces this.
3. **SFPU mutex**: Both TRISC1 and TRISC2 can issue SFPU instructions; a mutex
   (`semaphore::FPU_SFPU`) prevents simultaneous access. During overlap, MATH must use
   only FPU (matmul) — no SFPU calls from MATH while PACK holds the SFPU.
4. **Pipeline startup/drain**: First iteration has no previous scores for PACK to process;
   last iteration has no next matmul for MATH. Boundary handling is required.
5. **DST capacity**: With SyncHalf, each half has 8 tiles. Subblock dimensions must fit
   within this budget (e.g., `{1,4}` uses 4 tiles per half).
6. **Phase imbalance**: If matmul takes much longer than SFPU (or vice versa), the faster
   unit stalls waiting for the slower one. Chunk sizes should be tuned to balance phases.

**On Quasar** (4 TRISCs): TRISC3 (`ISOLATE_SFPU`) is a dedicated SFPU thread, which would
allow cleaner three-way overlap without the SFPU mutex constraint.

**Expected impact:** Theoretical 2× speedup by eliminating idle cycles. In practice,
pipeline startup/drain, phase imbalance, and DST pressure limit this to **1.3–1.5×**.
Applied to SDPA's 259 ms: savings of **60–86 ms** (259 → 173–199 ms).

This is the endgame optimization — maximum pipeline utilization.

---

## Forward Summary Table

| # | Optimization | Impact | Effort | Accuracy | Requires | Status |
|---|---|---|---|---|---|---|
| **F1** | Fuse scale into exp | Medium | Low | LOSSLESS | — | **Done** (Phase A PR) |
| **F2** | Approximate exp + ReLU | High | Very Low | MODERATE RISK | — | Not started |
| **F3** | recip_tile_first_column | Medium | Low | LOSSLESS | — | **Done** (Phase A PR) |
| **F5** | exp_tile_first_column | Medium | Low | LOSSLESS | — | **Done** (Phase A PR) |
| **F7** | Conditional rescaling (eps=0) | High | Low | LOSSLESS | — | Not started |
| **F8** | K/V sharing for balanced pairs | High | Medium | LOSSLESS | — | Not started |
| **F13** | Heavy row first (LPT) | Low | Very Low | LOSSLESS | — | **Done** (Phase A PR) |
| **F14** | Uniform dataformat skip | Low | Very Low | LOSSLESS | — | Not started |
| **F9** | Multi-tile K/V chunking | Very High | High | NEGLIGIBLE | — | Not started |
| **F4** | Fused sub_exp + L1 acc sum | High | Medium | NEGLIGIBLE | F9 | Not started |
| **F6** | Fused correction block | Medium | Medium | BETTER | F9 | Not started |
| **F10** | Subblock matmul | High | Medium | LOSSLESS | F9 | Not started |
| **F11** | Software polynomial exp | Medium | Medium | LOW (deg≥2) | — | Not started |
| **F12** | Streaming pipeline | Very High | Very High | LOSSLESS | F9+F10+F4 | Not started |

**Recommended implementation order:**
```
Phase A (quick wins, no architecture change):
  F13 ✓ → F1 ✓ → F3 ✓ → F5 ✓ → F14 → F2 → F7
  Done: F13, F1, F3, F5 — combined 22% FW kernel speedup, 5.6% step time reduction
  Remaining: F14 (dataformat skip), F2 (approx exp), F7 (conditional rescaling)

Phase B (major architecture change):
  F9 (multi-tile chunking — unlocks everything below)

Phase C (leverage chunking):
  F10 → F4 → F6 → F8

Phase D (advanced):
  F11 → F12
```

---

# Part 2: SDPA Backward Optimizations

## Tier 1: Independent — Apply Now

### B1. Fuse Scale into Exp (Backward)

> **Status: N/A** — Superseded by B11 (logsumexp intermediates). After B11, backward
> softmax recomputation is `P = exp(S - lse)` with no scale factor in the exp. The scale
> `1/sqrt(d)` is applied during the Q@K^T matmul, not in the exp call. There is no
> separate `mul_unary_tile` + `exp_tile` sequence to fuse.

| | |
|---|---|
| **Source** | TTNN (same as F1) |
| **Impact** | ~~Medium~~ N/A |
| **Effort** | Low |
| **Accuracy** | **LOSSLESS** |

**Current code** (`sdpa_bw_compute_utils.hpp:91-92`):
```cpp
exp_tile_init</* approx */ false>();
exp_tile</* approx */ false>(working_reg);
```

Same change as F1. Pass scale into exp. Applies to `apply_statistics_inplace` in both
Q and KV backward kernels.

---

### B2. Approximate Exp + ReLU Clamping (Backward)

| | |
|---|---|
| **Source** | TTNN + FA4 (same as F2) |
| **Impact** | High |
| **Effort** | Very Low |
| **Accuracy** | **MODERATE-HIGH RISK** — most sensitive optimization; backward P precision directly affects gradient quality |

The backward recomputes `P = exp(score - max) * recip_sum_exp` in `apply_statistics_inplace`.
Error in P directly corrupts gradients: `dS = P * (dP - u)`.

**Quantitative gradient error analysis:**
- With `exp_tile<true>()`: P error ~0.1-0.3% → gradient error ~0.1-0.3% → within BF16 noise
- With `exp_tile<true, true>()`: P error ~0.5-1% → gradient error ~0.5-1% → risky

**Critical constraint:** Forward and backward MUST use the same exp implementation.
If forward uses `exp_tile<true>()`, backward must also use `exp_tile<true>()`.
Mismatch between `P_forward` and `P_recomputed` is a direct gradient corruption that
doesn't cancel and accumulates over training.

**Recommendation:** Apply F2 and B2 together. Use `exp_tile<true>()` (not fast+approx).
Validate convergence.

---

### B3. recip_tile_first_column (Backward)

> **Status: N/A** — Superseded by B11 (logsumexp intermediates). After B11, backward
> computes `P = exp(S - lse)` directly — there is no `recip_sum_exp` and therefore no
> `recip_tile` call in the backward softmax recomputation path.

| | |
|---|---|
| **Source** | TTNN (same as F3) |
| **Impact** | ~~Medium~~ N/A |
| **Effort** | Low |
| **Accuracy** | **LOSSLESS** |

Same as F3. Applies wherever `recip_tile` is called on column-reduced data in backward.

---

### B4. exp_tile_first_column for Max Correction (Backward)

> **Status: N/A** — Superseded by B11 (logsumexp intermediates). After B11, backward
> does not perform online softmax with max-correction — it directly computes
> `P = exp(S - lse)` from the stored logsumexp. There is no `exp(prev_max - cur_max)`
> column-vector correction to optimize.

| | |
|---|---|
| **Source** | TTNN (same as F5) |
| **Impact** | ~~Medium~~ N/A |
| **Effort** | Low |
| **Accuracy** | **LOSSLESS** |

Same as F5. Applies to backward correction paths where `exp(prev_max - cur_max)` is
computed on column vectors.

---

### B5. Eliminate Redundant u_scalar Computation

> **Status: DONE** — Merged in [#39812](https://github.com/tenstorrent/tt-metal/pull/39812).
> ~1.4% step-time improvement (1927 ms vs 1955 ms baseline).

| | |
|---|---|
| **Source** | Your TODO at `sdpa_bw_kv_compute_kernel.cpp:203` |
| **Impact** | Very High |
| **Effort** | Medium |
| **Accuracy** | **LOSSLESS to SLIGHTLY BETTER** — pre-computing once eliminates non-determinism from repeated recomputation |

**Current code** (`sdpa_bw_compute_utils.hpp:135-163`):
```cpp
void compute_u_scalar_row(cb_grad_output, cb_attn_output, cb_u_scalar_row, ...) {
    // u = rowsum(dO * O) — computed for same Q row in EVERY K/V iteration
}
```
For KV backward, the same `u_scalar` for Q row `q` is recomputed in every one of the
`q+1` K/V iterations that reference that Q row. Total wasted: O(S²) compute.

**Change — Recommended approach:**
1. Run a small pre-pass kernel that computes `u_scalar` for all Q rows
2. Store result in tensor `(B, H, S, 32)` — 1 tile per Q row
3. KV reader streams `u_scalar` alongside intermediates
4. Also removes the need to read `O` tensor in KV kernel (O is only used for u_scalar)

**Accuracy note:** Computing `u` once vs recomputing it should give identical results, but
floating-point non-determinism in DST accumulation could cause bit-level variation between
recomputations. Pre-computing once actually **improves determinism**.

---

### B6. Avoid Explicit Transpose via Transposed Recomputation

| | |
|---|---|
| **Source** | FlashAttention-4 §Backward |
| **Impact** | High |
| **Effort** | Medium |
| **Accuracy** | **NEGLIGIBLE RISK** — same matmul, different operand order; at most 1 ULP difference from unpacker path |

**Current code** (`sdpa_bw_compute_utils.hpp:114-131`):
```cpp
inline void transpose_tile(const uint32_t cb_input, const uint32_t cb_transpose_wh) {
    // ... unpack → transpose_wh_tile → pack to scratch CB ...
}
```
Called twice per inner iteration: once for P^T (for `dV = P^T @ dO`), once for dS^T
(for `dK = dS^T @ Q`). Each call is a full L1 round-trip: unpack → transpose in DST →
pack to scratch CB → later unpack again for matmul. Two L1 round-trips per iteration.

#### The idea

Compute `S^T = K @ Q^T` instead of `S = Q @ K^T`. This produces attention scores in
transposed layout, so P^T and dS^T are naturally available without explicit transposes.

**Current (KV-backward inner loop):**
```
S    = Q @ K^T                            mm_init_short(cb_query, cb_key, transpose=1)
P    = softmax_recomp(S)                  apply_statistics_inplace (bcast_cols)
P^T  = transpose_tile(P)                  ← EXPLICIT L1 round-trip
dV  += P^T @ dO                           matmul_tiles(cb_transpose_wh, cb_grad_output)
dP   = dO @ V^T                           mm_init_short(cb_grad_output, cb_value, transpose=1)
dS   = P * (dP - u) * scale
dS^T = transpose_tile(dS)                 ← EXPLICIT L1 round-trip
dK  += dS^T @ Q                           matmul_tiles(cb_transpose_wh, cb_query)
```

**With B6 (KV-backward inner loop):**
```
S^T  = K @ Q^T                            mm_init_short(cb_key, cb_query, transpose=1)
P^T  = softmax_recomp_transposed(S^T)     apply_statistics_inplace (bcast_rows)
dV  += P^T @ dO                           matmul_tiles(cb_attn_weights, cb_grad_output)  ← NO transpose
dP^T = V @ dO^T                           mm_init_short(cb_value, cb_grad_output, transpose=1)
dS^T = P^T * (dP^T - u) * scale
dK  += dS^T @ Q                           matmul_tiles(cb_grad_scores, cb_query)  ← NO transpose
```

**DRAM access pattern does NOT change.** The same Q and K tiles are read from DRAM in
the same order. The "Q^T" is handled by the matmul hardware unpacker's transpose on
the right operand — same mechanism currently used for K^T. Swapping operand order in
`mm_init_short` and `matmul_tiles` is all that's needed for the score computation.

#### Impact on softmax recomputation (intermediates format)

The current `apply_statistics_inplace` (`sdpa_bw_compute_utils.hpp:76-112`) uses
**column broadcasts** to apply per-Q-row statistics:

```cpp
// Current: S[q,k] tile — rows are Q positions, columns are K positions
sub_tiles_bcast_cols(S, intermediates[0], ...);  // subtract per-row max
// exp, then:
unary_bcast<BroadcastType::COL>(intermediates[1], ...);  // multiply per-row recip_sum_exp
```

`BroadcastType::COL` broadcasts a column vector (one scalar per row) across all columns.
This works because rows of S correspond to Q positions, and the statistics are per-Q-position.

With S^T, the tile's rows are K positions and columns are Q positions. The per-Q-position
statistics must now be applied **per-column**, requiring `BroadcastType::ROW`:

```cpp
// B6: S^T[k,q] tile — rows are K positions, columns are Q positions
sub_tiles_bcast_rows(S_T, intermediates_transposed[0], ...);  // subtract per-col max
// exp, then:
unary_bcast<BroadcastType::ROW>(intermediates_transposed[1], ...);  // multiply per-col recip_sum_exp
```

The stored forward-pass intermediates (max, recip_sum_exp) are column-vector tiles
(scalar per row, broadcast across columns). For `bcast_rows`, they need to be row-vector
tiles (scalar per column, broadcast across rows).

**Options:**
1. **Transpose intermediate tiles once per outer iteration** — two `transpose_wh_tile`
   calls on the 2 small intermediate tiles (max, recip_sum_exp) before the inner loop.
   Cost: 2 transposes per outer iteration vs 2 transposes saved per inner iteration.
   Net saving: `2 × (St - 1)` transposes for causal, `2 × (St - 2)` for full attention.
2. **Change forward pass to store intermediates as row vectors** — eliminates the
   transpose entirely, but changes the forward-backward interface. Requires coordinated
   forward kernel change.

Option 1 is the practical first step: negligible cost (2 small tile transposes outside
the inner loop) for significant savings (2 large tile transposes removed from every
inner iteration).

#### Impact on causal masking

The causal mask is generated once by the writer kernel as a 32×32 tile
(`dataflow_utils.hpp:170-186`):

```cpp
// Current: lower-triangular — mask[row, col] = 1.0 if col <= row
*tile_ptr++ = (col <= row) ? one_value : zero_value;
```

In `S[q,k]`, rows = Q positions, cols = K positions. `col <= row` means K ≤ Q — correct
causal condition. In `S^T[k,q]`, rows = K positions, cols = Q positions. The causal
condition K ≤ Q becomes `row <= col` — **upper-triangular**.

**Change:** flip the comparison in `fill_causal_mask_tile`:
```cpp
*tile_ptr++ = (row <= col) ? one_value : zero_value;
```

The rest of the masking infrastructure is unchanged:
- **When to apply:** still `h == q_row_tile` (diagonal block), line 143 of q_compute_kernel
- **When to skip:** still `h > q_row_tile` (fully masked → loop bound `q_row_tile + 1`)
- **How to apply:** `apply_mask_on_reg` with `mask_tile` intrinsic works with any pattern —
  it zeros positions where the mask is 0, then fills -inf. Pattern-agnostic.

#### Which kernels benefit

| Kernel | Transposes eliminated | Intermediates change | Mask change | Net benefit |
|---|---|---|---|---|
| `sdpa_bw_kv` (outer K/V) | **2 per inner iter** (P^T, dS^T) | bcast_cols → bcast_rows + 2 transposes/outer | lower → upper tri | **High** — saves `2 × St` transposes, adds 2 |
| `sdpa_bw_q` (outer Q) | **0** (dQ = dS @ K needs no transposes) | No change needed | No change needed | **None** — Q-backward doesn't use transpose_tile |
| B10.2 fused (outer Q) | **2 per inner iter** (same as kv) | Same as kv kernel | Same as kv kernel | **High** — same saving in fused kernel |

B6 benefits the KV-backward kernel and the fused kernel (B10.2). The Q-backward kernel
already avoids transposes and needs no changes.

#### Accuracy note

`K[j,d] * Q[i,d]` vs `Q[i,d] * K[j,d]` — multiplication is commutative, so individual
products are bit-identical. The accumulation order across `d` tiles is the same.
The only possible difference is which operand comes from `in0` vs `in1` in the matmul
engine's unpacker, which may differ by at most 1 ULP due to different rounding paths.
`transpose_wh_tile` itself is a lossless permutation — eliminating it doesn't affect values.

---

### B7. Overlap Softmax Recomputation with Gradient Matmuls

| | |
|---|---|
| **Source** | FlashAttention-4 §Backward Pipeline |
| **Impact** | Very High |
| **Effort** | Medium-High |
| **Accuracy** | **LOSSLESS** — pure scheduling of independent operations |

**Current:** Everything in the inner loop is sequential. The MATH thread alternates
between softmax recomputation (SFPU: exp, mul, sub) and gradient matmuls (FPU), with
each unit idle ~50% of the time.

**Hardware mechanism:** Same MATH/PACK thread overlap as F12 (see "Tensix Compute
Pipeline Reference" above). The softmax recomputation uses SFPU (exp, mul, sub) and
can run on the PACK thread via `exp_packthread_tile`, while gradient accumulation
matmuls use FPU on the MATH thread. They operate on different DST halves with
`SyncHalf` double-buffering.

**Change:** Double-buffer `cb_attention_weights` and `cb_grad_scores`. Pipeline:

```
Q backward inner loop:
  tile j:
    MATH (FPU):  dQ += dS[j-1] @ K[j-1]               (matmul on DST half A)
    PACK (SFPU): recompute P[j], dP[j], dS[j]          (exp_packthread on DST half B)
                 using exp_packthread_tile for P recomputation

KV backward inner loop:
  tile j:
    MATH (FPU):  dV += P^T[j-1] @ dO[j-1]             (matmul on DST half A)
                 dK += dS^T[j-1] @ Q[j-1]              (matmul on DST half A)
    PACK (SFPU): recompute P[j], dP[j], u[j], dS[j]    (exp_packthread on DST half B)
```

**Why this is simpler than F12:** The backward already processes one tile per inner-loop
iteration (no multi-tile chunking needed). The recomputation for tile `j` is completely
independent of the gradient matmul for tile `j-1` — they read different inputs and write
different outputs. This means B7 does **not** require F9 (multi-tile chunking) as a
prerequisite. It can be implemented on the current single-tile architecture.

**Implementation notes:**
1. Use `exp_packthread_tile` (not `exp_tile`) for P recomputation on PACK thread
2. Double-buffer the intermediate CBs (`cb_attention_weights`, `cb_grad_scores`) so
   PACK can write tile `j`'s results while MATH reads tile `j-1`'s results
3. SFPU mutex: PACK holds SFPU for recomputation; MATH must use only FPU (matmul)
4. Pipeline startup: first iteration has no previous dS/P for MATH to accumulate;
   last iteration has no next tile for PACK to recompute

**Reference implementation — DeepSeek V3 SDPA:**
The DeepSeek V3 SDPA kernel (`models/demos/deepseek_v3_b1/kernel_includes/.../sdpa.h`)
implements exactly this FPU/SFPU overlap pattern with custom LLK extensions:

- **`MOVD2B` (Move DEST → SrcB):** Transfers column vector (max/lse) directly from DST
  to SrcB register without going through the unpacker. Avoids the TF32 truncation that
  the standard `unary_bcast` path introduces (though SrcB itself is TF32, this avoids
  L1 roundtrips).
- **`sdpa_bcast_col_srca_srcb_reuse`:** FPU `ELWSUB`/`ELWMUL` with `SRCB_BCAST_COL`.
  The preamble loads the column vector into SrcB via `MOVD2B` once, then reuses it
  across multiple tiles. Each tile's data is moved from DST → SrcA via `MOVD2A`
  (replay-buffered), then the FPU eltwise op runs with column broadcast from SrcB.
  `t6_semaphore_post(FPU_SFPU)` signals tile-by-tile completion.
- **SFPU exp on PACK thread:** After each FPU subtraction completes (signalled via
  semaphore), `fast_approx_exp` or `_ckernel_sfpu_exp_accurate_` runs on the PACK
  thread. For training accuracy we would use `_ckernel_sfpu_exp_accurate_` (7th-order
  Taylor, same as `exp_tile<false>`).
- **Tile-by-tile signalling:** `t6_semaphore_post(FPU_SFPU)` / `t6_semaphore_wait_on_zero`
  coordinate FPU and SFPU so exp starts as soon as each tile's subtraction finishes.

Key custom LLK files to adapt:
- `llk_math_sdpa_bcast_col_srca_srcb_reuse.h` — FPU bcast_col with SrcA+SrcB reuse
- `llk_math_sdpa_bcast_col_srcb_reuse.h` — FPU bcast_col with SrcB reuse only
- `llk_unpack_A_sdpa_api.h` — Custom unpack for SDPA
- `llk_math_sdpa_reduce_row.h` — SFPU row reduction (max/sum)

**Detailed KV backward inner-loop breakdown** (current sequential flow):
```
Step 1: S = Q @ K^T                    (FPU matmul)
Step 2: P = exp(S - lse)               (FPU bcast sub + SFPU exp, via apply_softmax_statistics_on_dst)
         → pack P to cb_attention_weights
Step 3: P^T = transpose_tile_fpu(P)    (FPU transpose via L1 roundtrip)
Step 4: dV += P^T @ dO                 (FPU matmul with L1 accum)
Step 5: dP = dO @ V^T                  (FPU matmul)
Step 6: dS = P * (dP - u) * scale      (FPU binary ops via compute_grad_scores)
Step 7: dS^T = transpose_tile_fpu(dS)  (FPU transpose via L1 roundtrip)
Step 8: dK += dS^T @ Q                 (FPU matmul with L1 accum)
```

**Practical overlap strategy** — the subtraction in step 2 uses FPU (`sub_tiles_bcast_cols`
or `ELWSUB` with `SRCB_BCAST_COL`), so only the exp portion can run on PACK thread.
The overlap window is:
```
MATH:  [── sub(S, lse) ──] commit → [── next FPU work (step 3/4/5) ──]
PACK:                       wait → [── exp_packthread_tile ── pack P ──] release
```
Even partial overlap of SFPU exp (~1000+ cycles for accurate 7th-order Taylor) with
the FPU transpose setup or dP matmul prep saves cycles on every inner iteration.

**DST register budget** (FP32 mode, 4 tiles per half):
- Half A: matmul result (1–2 tiles) + working space
- Half B: softmax recomputation (scores + lse = 2 tiles) + exp result
- Sufficient for the current single-tile-per-iteration architecture

**Open questions** (see `FPU_SFPU_OVERLAP_GUIDE.md` for details):
1. Pack-thread SFPU replay slots: Does `exp_tile<false>` (accurate, 7th-order Taylor)
   fit within BH's SFPU replay program capacity without clobbering matmul MOPs?
2. L1 accumulation + overlap: `pack_reconfig_l1_acc(true)` for dV/dK accumulation
   may conflict with pack-thread SFPU operations that also use the packer.
3. `mm_init` vs `mm_init_short` after SFPU: Known issue — full `mm_init` may be needed
   after pack-thread SFPU to restore matmul state.

**Accuracy note:** The softmax recomputation for tile `j` and the gradient matmul for
tile `j-1` operate on completely independent data. Running them concurrently vs
sequentially produces bit-identical results.

---

### B8. K/V Sharing for Balanced Pairs (Q Backward)

| | |
|---|---|
| **Source** | Same principle as F8 |
| **Impact** | Medium |
| **Effort** | Medium |
| **Accuracy** | **LOSSLESS** |

Same as F8 but for Q backward. Interleave light and heavy rows' processing, reading each
K/V tile once.

---

### B9. Q/dO Sharing for Balanced Pairs (KV Backward)

| | |
|---|---|
| **Source** | Same principle as F8/B8 |
| **Impact** | Medium |
| **Effort** | Medium |
| **Accuracy** | **LOSSLESS** |

KV backward in balanced mode pairs heavy and light KV rows that iterate over overlapping
Q rows. Interleave the two rows' inner loops; when processing a shared Q row, read
Q/dO/intermediates once for both.

> **Note (post-B5):** After B5 merged, KV backward no longer reads the O (attention output)
> tensor — O was only used for `u_scalar = rowsum(dO * O)`, which is now precomputed.
> The sharing benefit is reduced to Q, dO, and intermediates (lse).

---

### B11. Store logsumexp Instead of Separate max + recip_sum_exp

> **Status: DONE** — Implemented in [#41683](https://github.com/tenstorrent/tt-metal/pull/41683).
> Forward emits single FP32 LSE tile (32 wide, down from 2× 64 BF16). Backward uses
> fused `apply_softmax_statistics_on_dst` (subtraction + exp in DST, no CB round-trip for
> intermediates). Ring attention merge rewritten to logaddexp accumulation.
> Additionally, FP32 was chosen over BF16 for intermediate storage, eliminating the
> BF16 quantization error on stored LSE entirely.

| | |
|---|---|
| **Source** | FA4 + TTNN Ring SDPA |
| **Impact** | Medium (bandwidth) / **High (accuracy)** |
| **Effort** | Low |
| **Accuracy** | **BETTER** — eliminates `recip_tile` approximation error; one fewer BF16 rounding in backward |

#### Current intermediate storage

Forward stores 2 BF16 tiles per Q row (`sdpa_fw_program_factory.cpp:352-356`):
- `intermediates[0]` = `max_val` (column vector, value in col 0)
- `intermediates[1]` = `recip_sum_exp` = `1/sum(exp(x - max))` (column vector)

Backward reconstructs P (`sdpa_bw_compute_utils.hpp:76-112`):
```
P = exp(S - max) * recip_sum_exp       ← 2 operations: sub_bcast + exp + mul_bcast
```

#### Current accuracy chain (error sources)

1. `recip_tile(sum_exp)` — SFPU Newton-Raphson approximation (~0.5-1% relative error)
   (`sdpa_compute_utils.hpp:272-273`)
2. Result quantized to BF16 for DRAM storage (~0.4% additional quantization error)
3. Backward reads BF16 `recip_sum_exp` and multiplies into every tile in the inner loop
4. **The recip error is applied to every (q, k) tile in the entire backward pass**

This is likely the dominant source of observed `sdpa_bw` accuracy issues. The `recip`
approximation error is baked into the stored intermediate and propagates to every
gradient computation.

#### Change: logsumexp

Forward stores 1 tile per Q row: `lse = max + log(sum_exp)`.
Backward reconstructs: `P = exp(S - lse)`.

#### Forward kernel changes

Current finalization (`sdpa_fw_compute_kernel.cpp:172-207`):
```
Phase 5 (current):
  1. reduce_and_recip_tile_inplace(sum_exp)     ← row-reduce + recip in one function
  2. pack_intermediate_result(max, cb_intermediates)
  3. pack_intermediate_result(recip_sum_exp, cb_intermediates)
  4. O = mm_out * recip_sum_exp                  ← output normalization via mul_bcast_cols
```

With logsumexp:
```
Phase 5 (new):
  1. row_reduce(sum_exp)                         ← row-reduce only (split from recip)
  2. log_tile(sum_exp_reduced)                   ← compute log(sum_exp) in FP32 DST
  3. add_tiles(max, log_sum_exp) → lse           ← lse = max + log(sum_exp) in FP32 DST
  4. pack_intermediate_result(lse, cb_intermediates)  ← ONE tile instead of two
  5. recip_tile(sum_exp_reduced)                  ← recip still needed for O normalization
  6. O = mm_out * recip_sum_exp                   ← output normalization unchanged
```

`reduce_and_recip_tile_inplace` must be split into `row_reduce_tile` + separate
`recip_tile`, with the `log` + `add` inserted between them.

**`log_tile` availability:** No high-level `log_tile` wrapper exists in the compute API.
The LLK-level `calculate_log` is available (`ckernel_sfpu_log.h`) with both BF16
(5th-degree minimax polynomial) and FP32 (higher-order series with range reduction)
implementations. Options:
- Write a thin `log_tile` wrapper calling the LLK directly
- Use `log1p_tile(sum_exp - 1)` since `log(x) = log1p(x - 1)` (available in API)

The `log` computation happens **once per Q row** in the forward finalization (outside
the inner K/V loop). Cost is negligible.

#### Backward kernel changes

Current `apply_statistics_inplace` (`sdpa_bw_compute_utils.hpp:76-112`):
```cpp
// Read 2 intermediate tiles: intermediates[0] = max, intermediates[1] = recip_sum_exp
cb_wait_front(cb_intermediates, 2);
sub_tiles_bcast_cols(S, intermediates[0], ...);    // S - max
exp_tile(result);                                   // exp(S - max)
mul_tiles_bcast_cols(result, intermediates[1], ...); // * recip_sum_exp
```

With logsumexp:
```cpp
// Read 1 intermediate tile: intermediates[0] = lse
cb_wait_front(cb_intermediates, 1);
sub_tiles_bcast_cols(S, intermediates[0], ...);    // S - lse
exp_tile(result);                                   // exp(S - lse) = P  ← DONE
// No multiply step needed
```

Simpler, faster, and more accurate. The `mul_bcast_cols` step is completely eliminated.

#### Intermediate tensor changes

| | Current | logsumexp |
|---|---|---|
| Tiles per Q row | 2 (max + recip_sum_exp) | **1** (lse) |
| Tensor shape | (B, H, S, 64) | (B, H, S, 32) |
| DRAM per head (S=2048) | 64 × 2 × 2 KB = 256 KB | 64 × 1 × 2 KB = **128 KB** |
| BW reads per inner iter (bw) | 2 tiles | **1 tile** |
| Backward ops per inner iter | sub + exp + mul = 3 | sub + exp = **2** |

#### Accuracy analysis (corrected)

The original B11 analysis was **wrong** — it claimed "one additional BF16 rounding from
log." The correct analysis:

**Current path error sources:**
1. `recip_tile` SFPU approximation: ~0.5-1% relative error (Newton-Raphson)
2. BF16 quantization of `recip_sum_exp`: ~0.4% relative error
3. BF16 quantization of `max`: ~0.4% relative error
4. Backward `exp(S - max)`: BF16 rounding
5. Backward `* recip_sum_exp`: BF16 rounding (compounds recip error)
6. **Total: 5 error sources, 2 in backward per inner iteration**

**logsumexp path error sources:**
1. `log` SFPU approximation: computed in FP32 DST (minimax polynomial, <0.001% relative)
2. `max + log(sum_exp)`: computed in FP32 DST (exact for FP32 precision)
3. BF16 quantization of `lse`: ~0.4% relative error
4. Backward `exp(S - lse)`: BF16 rounding
5. **Total: 4 error sources, 1 in backward per inner iteration**
6. **`recip_tile` error completely eliminated**

logsumexp is **strictly more accurate** for the backward pass:
- Eliminates `recip_tile` approximation (the largest single error source)
- Eliminates one BF16 multiply in every inner iteration of backward
- The `log` is computed once per Q row at FP32 precision; its error is negligible
  compared to the `recip` it replaces

**Verdict:** B11 should be reclassified from "LOW RISK" to **"BETTER"** — it improves
accuracy while also halving intermediate bandwidth. TTNN Ring SDPA already uses
this approach.

#### Optional: FP32 intermediate storage

For maximum accuracy, intermediates can be stored in FP32 instead of BF16:
```cpp
create_circular_buffer(program, all_cores, kIntermediateCbIndex,
    DataFormat::Float32, fp32_single_tile_size_bytes, kIntermediateTiles);
```
This eliminates the BF16 quantization error on the stored `lse`. Combined with
logsumexp, the backward reconstruction `P = exp(S - lse)` would use the full
FP32-precision `lse` value, making the recomputed P nearly identical to the
forward-pass P. Cost: 2× DRAM for intermediates (4 KB per tile instead of 2 KB),
but with logsumexp it's only 1 tile per row, so total is same as current BF16 × 2 tiles.

#### Interaction with B6 (transposed recomputation)

With logsumexp, the intermediate is a single tile per Q row. For B6 (computing S^T),
the backward needs `sub_tiles_bcast_rows` instead of `sub_tiles_bcast_cols`. The `lse`
tile must be a row vector instead of a column vector.

Two options:
1. **Transpose in `pack_intermediate_result`**: After masking to column 0, transpose the
   tile so value moves to row 0. The function in `sdpa_compute_utils.hpp:289-316` already
   has the tile in DST register — adding a `transpose_wh_tile` before packing is one extra
   SFPU instruction. Zero cost in backward.
2. **Store as-is, transpose once in backward**: Transpose the single `lse` tile once per
   outer iteration. Negligible cost.

Option 1 is preferred — makes the intermediate directly usable by backward with no
extra work, and the transpose cost in forward is trivial (once per Q row, outside the
inner loop).

#### Implementation scope

| Component | Change |
|---|---|
| `sdpa_compute_utils.hpp` | Split `reduce_and_recip_tile_inplace` into `row_reduce` + `log` + `add` + `recip`. Add `log_tile` wrapper or use `log1p_tile`. |
| `sdpa_fw_compute_kernel.cpp` | Phase 5: insert lse computation between reduce and recip. Pack 1 intermediate instead of 2. |
| `sdpa_fw_program_factory.cpp` | Change intermediate CB size from 2 tiles to 1. Optionally change format to FP32. |
| `sdpa_fw_writer_kernel.cpp` | Write 1 intermediate tile per row instead of 2. |
| `sdpa_bw_compute_utils.hpp` | Simplify `apply_statistics_inplace`: remove `mul_bcast`, read 1 tile. |
| `sdpa_bw_*_reader_kernel.cpp` | Read 1 intermediate tile per row instead of 2. |
| `sdpa_bw_*_program_factory.cpp` | Update intermediate CB size. |
| **Total** | ~7 files. Backward kernel gets *simpler* (removes mul_bcast). |

---

### B14. Uniform Dataformat Skip (Backward)

| | |
|---|---|
| **Source** | TTNN (same as F14) |
| **Impact** | Low |
| **Effort** | Very Low |
| **Accuracy** | **LOSSLESS** |

Same as F14. Skip no-op reconfig calls in backward kernels.

---

### B15. Conditional Rescaling for Q Backward

> **Status: N/A** — Superseded by B11 (logsumexp intermediates). After B11, backward
> does not perform online softmax with max-correction rescaling. It directly computes
> `P = exp(S - lse)` from stored logsumexp. There is no conditional rescaling step
> to optimize.

| | |
|---|---|
| **Source** | FA4 (same as F7) |
| **Impact** | ~~Medium~~ N/A |
| **Effort** | Low |
| **Accuracy** | **LOSSLESS with epsilon=0; UNACCEPTABLE with epsilon>0** |

Same principle as F7 but in Q backward's online softmax recomputation.

---

### B16. Generic Host-Side Work Scheduling (Backward)

> **Status: BALANCED PAIRING DONE** — Light/heavy row pairing (`BALANCED_PARALLELISM`
> kernel mode) is implemented and merged for both `sdpa_fw` and `sdpa_bw`. Each pair
> has constant work `Ht+1`, so the only imbalance comes from the pair-to-core
> remainder. This is significant for small-batch GQA on N300 (128 pairs / 56 cores =
> 33%), but drops quickly with batch size (~10% at B=4) and is negligible for MHA
> or configs where pairs >> cores. The generic host-side LPT scheduler described
> below would further improve the small-batch GQA case but is not yet implemented.

| | |
|---|---|
| **Source** | Profiling analysis — KV-kernel 33.3% load imbalance |
| **Impact** | **Very High** — eliminates the single largest backward bottleneck |
| **Effort** | Medium (host-side scheduler + kernel interface change across 8 files) |
| **Accuracy** | **LOSSLESS** — same rows, same computation, different assignment order |

#### The problem with the current design

The current code has two scheduling policies baked into the kernels:

1. **Standard mode:** each core gets a contiguous range `[start_row, start_row + N)`.
2. **Balanced mode:** each core gets a contiguous range of *pairs*
   `[start_pair, start_pair + N)`, where pair→row conversion is computed inside
   the kernel via `light = pair * 2`, `heavy = pair * 2 + 1` within each sequence.

Both embed the scheduling policy in 8 kernel files (reader, writer, compute × 2 programs).
This is rigid — any new policy requires modifying all 8 kernels and adding a new `#ifdef`.
The balanced pairing policy specifically assumes causal mask, even `St`, and
`total_rows > num_cores`. It also produces poor balance when the number of pairs
doesn't divide evenly across the available cores:

| Config | Total pairs | Cores | Pairs/core | Imbalance |
|--------|-------------|-------|------------|-----------|
| TinyLlama N300 (56c) | 128 | 56 | 2–3 | **33.3%** |
| TinyLlama N150 (64c) | 128 | 64 | 2 | 0.0% |
| B=1 TP=2 (kvH=2) N300 | 64 | 56 | 1–2 | **50.0%** |
| B=2 N300 | 256 | 56 | 4–5 | 20.0% |
| B=4 N300 | 512 | 56 | 9–10 | 10.0% |
| MHA (kvH=32) N300 | 1024 | 56 | 18–19 | 5.3% |
| S=4096 N300 | 256 | 56 | 4–5 | 20.0% |

The 33.3% imbalance for TinyLlama on N300 wastes ~3 ms of wall time — the single
largest backward bottleneck.

#### Design principle: scheduling belongs on the host

The kernel should not encode a scheduling policy. It should receive a **row index list**
and process whatever the host decided. All scheduling intelligence belongs on the host,
which knows:
- The shape `(B, kvH, Ht, heads_per_group)`
- The mask type (causal → variable row work; non-causal → uniform)
- The core count `C` (56, 64, 128, ... — varies by hardware)
- TP configuration (affects effective kvH and hpg per device)

#### Kernel interface change

**Before (two code paths, 8 kernel files):**
```cpp
// Standard mode:
for (uint32_t r = 0; r < num_rows; ++r) {
    process_single_row(start_row + r);
}
// Balanced mode (#ifdef BALANCED_PARALLELISM):
for (uint32_t p = 0; p < num_pairs; ++p) {
    const auto global_pair_idx = start_pair_idx + p;
    // ... pair-to-row formula embedded in kernel ...
    process_single_row(light_global_row);
    process_single_row(heavy_global_row);
}
```

**After (single code path, simpler):**
```cpp
const uint32_t num_rows = get_arg_val<uint32_t>(row_list_arg_offset);
for (uint32_t r = 0; r < num_rows; ++r) {
    const uint32_t global_row_idx = get_arg_val<uint32_t>(row_list_arg_offset + 1 + r);
    process_single_row(global_row_idx);
}
```

This eliminates `#ifdef BALANCED_PARALLELISM` from all 8 kernel files. The reader,
writer, and compute kernels all receive the same row index list from the host.
`process_single_row(global_row_idx)` already works with arbitrary row indices —
no inner-loop changes needed.

**Runtime arg budget:** The hardware limit is **341 uint32s** per kernel per core
(`4096 / (3 × sizeof(uint32_t))`). Current kernels use 2–9 args depending on the
kernel type. The row list adds `1 + max_rows_per_core` uint32s. Buffer addresses
(same across all cores) can be moved to `common_runtime_args`, freeing per-core
budget for the row list.

| Config | Total rows | Cores | Max rows/core | Per-core args needed | Fits in 341? |
|--------|-----------|-------|---------------|---------------------|-------------|
| TinyLlama (S=2048, GQA) N300 | 256 | 56 | 5 | ~13 | YES |
| S=8192, GQA, N300 | 1,024 | 56 | 19 | ~27 | YES |
| S=65536, GQA, N300 | 8,192 | 56 | 147 | ~155 | YES |
| B=8, S=8192, GQA, N300 | 8,192 | 56 | 147 | ~155 | YES |
| MHA (kvH=32), S=4096, N300 | 4,096 | 56 | 74 | ~82 | YES |
| MHA (kvH=32), S=18K+, N300 | 18K+ | 56 | 333+ | 341+ | **NO** |

The row-list fits comfortably for all GQA configs up to S≈150K. It only overflows
for MHA with very long sequences (S>18K, kvH=32 on 56 cores). For those configs,
balanced pairing already achieves <5.3% imbalance, so the host scheduler can
**fall back to contiguous assignment** (2 args: `start_row, num_rows`) when the
row list would exceed the runtime arg limit.

**Dispatch overhead:** The dispatch payload scales with `num_cores × max_rt_args_per_core`
(not a fixed-size packet). Extra dispatch time for the row list:

| Config | Extra dispatch bytes | Extra time (@4 GB/s) | LPT kernel saving |
|--------|---------------------|---------------------|-------------------|
| TinyLlama N300 (5 rows/core) | 896 B | ~0.2 us | 3,070 us |
| B=4 N300 (19 rows/core) | 8.7 KB | ~2.2 us | 800 us |

Dispatch overhead is < 0.01% of the kernel improvement in all cases.
This is self-balancing: LPT helps most when rows/core is small (GQA, small batch, TP),
which is exactly when the row list is short and dispatch cost is lowest.
For large shapes where the row list would be long, balanced pairing already
achieves <1% imbalance and the fallback avoids extra dispatch cost entirely.

#### Host-side scheduler

```cpp
// Generic interface:
std::vector<std::vector<uint32_t>> schedule_kv_rows(
    uint32_t B, uint32_t kvH, uint32_t Ht, uint32_t hpg,
    AttentionMaskType mask_type,
    uint32_t num_cores);
// Returns: per-core list of global KV row indices
```

**Default implementation — LPT greedy (works for any config):**
```
1. N_rows = B × kvH × Ht
2. For each row i, compute work(i):
     causal:     hpg × (Ht − (i mod Ht))
     non-causal: hpg × Ht
3. max_rows_per_core_estimate = ceil(N_rows / C)
4. If (overhead + 1 + max_rows_per_core_estimate) > max_runtime_args:
     Fall back to contiguous assignment (balanced pairing already works well
     when rows >> cores, which is always the case at this threshold)
5. Sort rows by descending work
6. Initialize min-heap of (load, core_id, row_list) with C entries
7. For each row in sorted order:
     pop lightest core, append row, push back
8. Return per-core row lists
```

Complexity: O(N_rows × log C). For TinyLlama: 256 × log₂(56) ≈ 1500 operations —
sub-microsecond on the host.

For non-causal mask, all rows have equal work. LPT degenerates to round-robin,
which is optimal. No special case needed.

The fallback is self-consistent: the runtime arg limit is only exceeded when
rows/cores > ~333, which means balanced pairing imbalance is already <0.3%.
LPT adds value precisely when rows/cores is small (≤ ~20) — well within budget.

#### Simulated results across configs

| Config | Rows | Cores | Balanced pairing imbal. | LPT imbal. |
|--------|------|-------|------------------------|------------|
| TinyLlama N300 (56c) | 256 | 56 | 33.3% | **5.2%** |
| TinyLlama N150 (64c) | 256 | 64 | 0.0% | 0.0% |
| TinyLlama BH (128c) | 256 | 128 | 0.0% | 0.0% |
| B=2 N300 | 512 | 56 | 20.0% | **1.7%** |
| B=4 N300 | 1024 | 56 | 10.0% | **0.2%** |
| MHA (kvH=32) N300 | 2048 | 56 | 5.3% | **0.1%** |
| TP=2 (kvH=2) N300 | 128 | 56 | 50.0% | **9.9%** |
| S=4096 N300 | 512 | 56 | 20.0% | **1.8%** |

LPT is strictly better or equal to balanced pairing for every configuration.
The improvement is largest when rows/cores ratio is small (GQA, small batch, TP).

#### Degenerate case: rows < cores

When `B × kvH × Ht < C` (e.g., TP=4 with kvH=1, Ht=64, C=56), only 64 rows exist.
LPT assigns 1–2 rows per core but inherent work variance is extreme (heaviest row
has 64× the work of the lightest). No row-level scheduling can fix this — the problem
requires **row splitting** across multiple cores with cross-core dK/dV reduction
(B10.2). This is an orthogonal problem.

In practice this case is rare: TP on the head dimension with kvH=1 means only 1 KV
head per device, which is a fundamentally under-parallelized configuration.

#### Relationship to other optimizations

- **B8/B9 (data sharing):** Balanced pairing enables B9 (paired rows share Q/dO/O reads).
  Generic scheduling loses this guarantee, but the kernel is strongly compute-bound with
  1.9× reader bandwidth headroom. B9's benefit is negligible until the bottleneck
  shifts toward memory (after B7). If needed, the host scheduler can be extended to
  produce *ordered* row lists that group same-kv-head rows together — this preserves
  data reuse without baking it into the kernel.

- **B10.2 (full fusion):** Full fusion (outer-Q, reduce dK/dV) uses the same Q-row
  scheduling as the current architecture — B16 directly applies. The fused kernel
  has 2048 Q rows for scheduling (vs 256 KV rows in the current KV-kernel),
  making B16's LPT even more effective (0.1% imbalance).

- **Forward kernel:** The same generic scheduling approach applies to `sdpa_fw`
  (`F8` already uses balanced pairing for forward). A single `schedule_rows()`
  implementation can serve both forward and backward.

#### Implementation scope

| Component | Change |
|-----------|--------|
| Host factory (×2) | Replace `calculate_balanced_pair_distribution()` with `schedule_kv_rows()`. Pass row list as runtime args instead of `(start_pair_idx, num_pairs)`. |
| Compute kernels (×2) | Remove `#ifdef BALANCED_PARALLELISM`. Single loop over row index list. |
| Reader kernels (×2) | Same — remove ifdef, iterate row index list. |
| Writer kernels (×2) | Same — remove ifdef, iterate row index list. |
| **Total** | 10 files changed. Kernel code gets *simpler* (removes ifdef + pair formula). Host code gets slightly more complex (adds scheduler). Net: code reduction. |

---

### B17. SFPU Reduce SUM for u_scaler Row Reduction

| | |
|---|---|
| **Source** | LLK `sfpu_reduce` ([#41593](https://github.com/tenstorrent/tt-metal/issues/41593)) |
| **Impact** | Medium |
| **Effort** | Low |
| **Accuracy** | **LOSSLESS to SLIGHTLY BETTER** — SFPU reduce operates at full FP32 in DST, avoiding TF32 truncation through SrcA |

**Current code** (`sdpa_bw_compute_utils.hpp`, `compute_u_scalar_row`):
```cpp
// Row reduction via FPU matmul-with-ones vector
mm_init_short(cb_u_scalar_row, cb_mat_mul_reduction, 0);
matmul_tiles(cb_u_scalar_row, cb_mat_mul_reduction, 0, 0, accum_register);
```

The `u_scaler` computation uses `rowsum(dO * O)` which is implemented as a matmul
against a pre-filled ones-vector (`cb_mat_mul_reduction`). This requires:
1. A CB slot for the ones-vector (loaded from reader)
2. FPU matmul cycles for a simple reduction
3. Data passes through SrcA unpacker → TF32 truncation

**Change:** Replace with `sfpu_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>` which
is now available in LLK. This:
1. Eliminates the CB roundtrip for the ones-vector
2. Frees FPU for other work (relevant for B7 overlap)
3. Operates at full FP32 directly on DST registers — no TF32 truncation
4. Simplifies reader kernels (no need to load/manage ones-vector tile)

---

### B18. Inner-Loop CB Roundtrip Fusion

| | |
|---|---|
| **Source** | Analysis of pack→L1→unpack overhead in inner loop |
| **Impact** | High |
| **Effort** | Medium |
| **Accuracy** | **LOSSLESS** — same computation, fewer data movements |

**Current:** Several chained operations in the inner loop pack their result to a CB
and immediately unpack it for the next operation. Each pack→L1→unpack roundtrip costs:
- PACK: DST → CB (L1 write)
- UNPACK: CB → SrcA/SrcB (L1 read + TF32 truncation)
- Synchronization overhead (cb_push_back / cb_wait_front / cb_pop_front)

**Specific roundtrips to eliminate:**
1. Between `compute_grad_attn_weights` and `compute_grad_scores`: the grad_attn_weights
   result is packed to `cb_grad_attention`, then immediately unpacked for the next step.
   Can keep the result in DST.
2. Between softmax recomputation (`apply_softmax_statistics_on_dst`) and its consumers:
   attention weights are packed to `cb_attention_weights`, then re-read. With careful
   DST register management, these can stay in registers.

**Savings multiply:** With B=5, that's `1280 × avg_inner_iterations` redundant
roundtrips eliminated across all cores.

**Precision bonus:** Keeping data in DST avoids TF32 truncation from SrcA/SrcB
unpacker on the intermediate values.

---

### B19. Fused Softmax Recomputation (`copy_tile` + SFPU sub_bcast_col + exp)

| | |
|---|---|
| **Source** | Analysis of `apply_softmax_statistics_on_dst` precision and performance |
| **Impact** | Medium-High |
| **Effort** | Medium |
| **Accuracy** | **SLIGHTLY BETTER** — eliminates TF32 truncation of lse through SrcB, keeps full FP32 |

**Current code** (`sdpa_bw_compute_utils.hpp:105-126`, `apply_softmax_statistics_on_dst`):
```cpp
// Step 1: Load lse into DST via unary_bcast<COL> (B2D path)
//   lse goes: L1 → SrcB unpacker → TF32 truncation → FPU broadcast → DST[lse_reg]
unary_bcast<BroadcastType::COL>(cb_intermediates, 0, lse_reg);

// Step 2: SFPU subtract (scores - lse), both in DST
sub_binary_tile(scores_reg, lse_reg, scores_reg);  // SFPU traversal 1

// Step 3: SFPU exp
exp_tile<false>(scores_reg);                         // SFPU traversal 2
```

**Three problems:**
1. **TF32 truncation of lse**: `unary_bcast<COL>` uses `unpack_to_dest = false` with B2D
   path, routing lse through SrcB → TF32 (loses 4 mantissa bits from FP32 lse).
2. **Redundant SFPU traversal**: `sub_binary_tile` STORE→LOAD between sub and exp.
3. **FPU occupation**: `unary_bcast<COL>` uses FPU for broadcast, blocking matmul.

**Proposed approach — two options:**

**Option A (DeepSeek-style, best performance):**
1. `copy_tile(cb_intermediates, 0, lse_reg)` — uses Blackhole's **unpack-to-dest** path
   for FP32 data: L1 → Unpacker → **DST directly** (bypasses SrcA/SrcB entirely).
   Full FP32 precision, no TF32 truncation.
2. `MOVD2B` — move lse column from DST → SrcB (TF32 truncation, but avoids L1 roundtrip)
3. FPU `ELWSUB` with `SRCB_BCAST_COL` — fast FPU subtraction
4. `exp_packthread_tile` on PACK thread — SFPU exp overlapped with next FPU work

Requires custom LLK files from DeepSeek V3 SDPA (see B7 reference).

**Option B (pure SFPU, simpler, best precision):**
1. `copy_tile(cb_intermediates, 0, lse_reg)` — same unpack-to-dest path, full FP32
2. Custom SFPU function that reads column 0 from `DST[lse_reg]`, broadcasts across
   all 32 columns, subtracts from `DST[scores_reg]`, and applies exp — **one fused
   SFPU traversal** instead of two:
```cpp
// Pseudocode for fused bcast_col_sub_exp SFPU function
template <int ITERATIONS = 8>
inline void _calculate_bcast_col_sub_exp_(uint32_t dst_scores, uint32_t dst_lse) {
    constexpr uint32_t dst_tile_size = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat score = dst_reg[dst_scores * dst_tile_size];
        vFloat lse_val = dst_reg[dst_lse * dst_tile_size];
        // lse_val holds column 0 of the lse tile — already a scalar per row
        // (4 rows per SFPU iteration, col 0 value broadcast to all 8 lanes)
        vFloat diff = score - lse_val;
        vFloat result = _ckernel_sfpu_exp_accurate_<false, DST_ACCUM_MODE>(diff, 0);
        dst_reg[dst_scores * dst_tile_size] = result;
        dst_reg++;
    }
}
```

Full FP32 end-to-end: lse loaded at FP32 via unpack-to-dest, subtraction and exp
both operate on DST at native 32-bit precision. No FPU occupation at all.

**Note on `copy_tile` FP32 precision:** On Blackhole, `copy_tile` is compiled with
`UnpackToDestEn = true`. For 32-bit data (`is_32bit_input() == true`), the unpacker
writes directly to DST bypassing SrcA/SrcB. Confirmed in
`llk_unpack_A.h:54` (`unpack_to_dest && is_32bit_input`).

**Applies to:** `apply_softmax_statistics_on_dst` (Q and KV backward compute kernels)
and `apply_statistics_inplace` (KV backward).

---

## Tier 2: Require Multi-Tile K Chunking

### B10. Fuse Backward KV and Q into Single Kernel

| | |
|---|---|
| **Source** | FlashAttention-4 §Backward |
| **Impact** | Very High |
| **Effort** | High (partial) / Very High (full) |
| **Accuracy** | **Partial: LOSSLESS. Full: NEGLIGIBLE RISK** from accumulation order |

#### B10.1 Partial Fusion (recommended first step — already implemented)

Run Q backward first, have it write `u_scalar` as a side output. KV backward reads
`u_scalar` instead of O. Saves entire O tensor read from KV kernel. Complements B5.
This is what the current `USE_PRECOMPUTED_U_SCALER` codepath implements.

#### B10.2 Full Fusion — Design (Option 2: Outer Q, Reduce dK/dV)

**Goal:** Single kernel that computes all three gradients (dQ, dK, dV) by extending
the current `sdpa_bw_q_compute_kernel` structure (outer loop over Q rows, inner loop
over K/V tiles). Eliminates redundant attention-weight recomputation — P is computed
once instead of twice, saving ~29% of total backward compute (50% of SFPU softmax work).

**Why the current design uses two kernels:**

| | Q-backward (current) | KV-backward (current) |
|---|---|---|
| Outer loop | Q rows | K/V rows |
| Inner loop | K/V tiles | Q rows |
| L1 accumulates | dQ (across K/V) | dK, dV (across Q) |
| Each core owns | a set of Q rows | a set of K/V rows |
| Cross-core reduction | **none** (dQ complete per-core) | **none** (dK/dV complete per-core) |

Both kernels recompute P from stored `(max, recip_sum_exp)` intermediates. The Q-bw
kernel also computes `u_scalar = rowsum(dO ⊙ O)` which the KV-bw kernel reuses.

**Key insight for Option 2:** Keep the Q-outer structure from `sdpa_bw_q_compute_kernel`
and add dK/dV accumulation into the same inner loop. dQ remains local (accumulates
across K/V in the inner loop, always complete per core — no reduction needed). dK and
dV accumulate across Q rows processed by each worker and require cross-core reduction.

**Why Option 2 over Option 1 (outer K/V, reduce dQ):**

| Criterion | Option 1 (outer K/V, reduce dQ) | Option 2 (outer Q, reduce dK/dV) |
|---|---|---|
| Gradient needing reduction | dQ — one per Q row × qWt tiles | dK, dV — one per K/V row × qWt tiles |
| dQ handling | Large L1 accumulator or streaming | **Always local**, tiny accum (8–32 KB) |
| Work units for B16 scheduling | St_kv = 256 (KV rows) | St_q = 2048 (Q rows) |
| Load balance (TinyLlama) | 5.2% imbalance | **0.1% imbalance** |
| Sync rounds (D=64) | 8 (one per Q head in GQA group) | **2** (one dK pass + one dV pass) |
| Worker double-buffering | Required (compute next head while sending) | **Not required** |
| K/V re-reads | Workers re-read Q/dO for each K/V row | None (Q/dO streamed naturally) |
| Code base | Major loop restructuring | **Extends existing** `sdpa_bw_q` naturally |
| Scaling to large D | Works natively | K-chunking needed (§B10.5) |

Option 2 is recommended for superior load balancing, simpler synchronization, and
natural extension of the existing kernel.

#### B10.3 Fused Kernel — Worker Compute Loop

Each worker core owns a subset of Q rows (assigned by B16 host-side scheduler).
For each Q row, the worker iterates over all K/V tiles, computing all three gradients:

```
// Per-worker initialization
Init dK_local[0..St_kv-1] = 0  in L1 (FP32)    // accumulates across Q rows
Init dV_local[0..St_kv-1] = 0  in L1 (FP32)    // accumulates across Q rows

For each Q row q assigned to this core:          ← outer loop (B16 schedules)
  Load Q[q], dO[q], intermediates[q]
  Init dQ_accum = 0                              // tiny: qWt tiles FP32

  For each K/V tile k (0..St_kv-1):              ← inner loop
    Load K[k], V[k]
    Recompute P[q,k]                             ← ONCE, not twice
    Compute dS[q,k] from dO, P, u_scalar

    dQ_accum += dS[q,k] @ K[k]                  ← L1 accumulate across K/V  ✓  (LOCAL)
    dK_local[k] += dS^T @ Q[q]                  ← L1 accumulate across Q rows
    dV_local[k] += P^T @ dO[q]                  ← L1 accumulate across Q rows

  Write dQ[q] to DRAM                            ← complete, no reduction needed

// After all Q rows processed:
// dK_local and dV_local hold this worker's partial contributions
// → send to reducer for cross-core accumulation
```

**dQ is always local:** The dQ accumulator lives entirely within the inner K/V loop
and is complete when the inner loop ends. Size: `qWt` tiles FP32 = 8 KB for D=64,
32 KB for D=256. Never needs cross-core reduction regardless of configuration.

**dK/dV need reduction:** When W workers each process different Q rows for the same
(batch, kv_head) group, they each produce partial dK/dV contributions to the same
K/V rows. Final gradients require summing across workers:

```
Worker 0:  Q rows [q0, q1, ...)  → partial dK[k], dV[k] for all K/V rows
Worker 1:  Q rows [q2, q3, ...)  → partial dK[k], dV[k] for all K/V rows
...
Final:  dK[k] = Σ_{w} dK_partial_w[k]   for each K/V row k
        dV[k] = Σ_{w} dV_partial_w[k]   for each K/V row k
```

With GQA (heads_per_group > 1), multiple Q heads map to the same KV head. Workers
processing different Q heads within the same group all contribute to the same dK/dV.

#### B10.4 L1 Budget for dK/dV Local Accumulators

The worker must hold `dK_local` and `dV_local` for all K/V positions in the current
chunk while iterating over Q rows. In FP32:

| D | qWt | St_kv (S=2048) | dK_local | dV_local | Total accum | Fits ~1200 KB? |
|---|---|---|---|---|---|---|
| 64 | 2 | 64 | 512 KB | 512 KB | 1024 KB | ✓ (tight) |
| 96 | 3 | 64 | 768 KB | 768 KB | 1536 KB | ✗ |
| 128 | 4 | 64 | 1024 KB | 1024 KB | 2048 KB | ✗ |
| 256 | 8 | 64 | 2048 KB | 2048 KB | 4096 KB | ✗ |

For D=64, the full dK/dV accumulator fits in L1. For D≥96, we need **K-chunking**.

#### B10.5 K-Chunking for Large Head Dimensions

When the full dK/dV accumulator exceeds L1, workers process K/V positions in
**chunks** that fit. This is the same principle as B12 (multi-tile K/V chunking)
and aligns with standard FlashAttention patterns.

**Mechanism:**

```
chunk_size = floor(available_L1 / (2 × qWt × 4 KB))

For each K-chunk c = [k_start, k_start + chunk_size):
  Init dK_local[0..chunk_size-1] = 0
  Init dV_local[0..chunk_size-1] = 0

  For each Q row q assigned to this core:       ← re-read Q/dO per chunk
    Load Q[q], dO[q], intermediates[q]
    For each K/V tile k in chunk c:
      Load K[k], V[k]
      Recompute P[q,k], dS[q,k]
      dK_local[k - k_start] += dS^T @ Q[q]
      dV_local[k - k_start] += P^T @ dO[q]
      dQ_accum[q] += dS @ K[k]                 ← continues accumulating across chunks

  // Chunk c complete — send dK_local, dV_local to reducer
  // Reuse L1 for next chunk
```

**Chunk sizes and Q/dO re-read overhead:**

| D | qWt | chunk_size | Num chunks | Q/dO re-read per extra chunk | Total re-read |
|---|---|---|---|---|---|
| 64 | 2 | 64 (all) | 1 | 0 | 0 |
| 96 | 3 | 50 | 2 | 64 × 2 × 3 × 2 KB = 768 KB | 768 KB |
| 128 | 4 | 37 | 2 | 64 × 2 × 4 × 2 KB = 1024 KB | 1024 KB |
| 256 | 8 | 18 | 4 | 3 × 64 × 2 × 8 × 2 KB = 6 MB | 6 MB |

At ~200 GB/s DRAM bandwidth: 6 MB → ~30 μs. **Negligible** compared to compute time.

**dQ accumulation across chunks:** The dQ accumulator is tiny (qWt tiles FP32 = 8–32 KB)
and persists across K-chunks for each Q row. It accumulates contributions from all
K/V tiles regardless of chunking. After the last K-chunk's inner loop, dQ[q] is
complete and written to DRAM. No cross-chunk complication for dQ.

#### B10.6 dK/dV Cross-Core Reduction — MPSC Protocol

After processing all Q rows (or all Q rows for a given K-chunk), each worker holds
partial dK/dV contributions. A **Multi-Producer Single-Consumer (MPSC) pull pattern**
(similar to `conv3d` writer kernel) aggregates them.

**Core roles:**
- **W worker cores** per (batch, kv_head) group: process disjoint Q rows, produce partial dK/dV
- **1 reducer core** per group: accumulates dK/dV from all workers, writes final result to DRAM
- Reducer is also one of the W workers (processes its own Q rows AND reduces)

**Two-pass protocol (for each K-chunk):**

The reducer performs two sequential passes — first dK, then dV — reusing the same
L1 input buffer. This halves the reducer's L1 requirement compared to receiving both
simultaneously.

```
             ┌─────────────────────────────────────────────────────┐
             │  Pass 1: dK reduction                               │
             │                                                     │
             │  For each worker w (w ≠ reducer):                   │
             │    Reducer: noc_semaphore_inc(worker_w, 1)          │  "send dK"
             │    Worker w: waits for signal, NOC writes dK_local  │
             │              to reducer's L1 input buffer           │
             │    Worker w: noc_semaphore_inc(reducer, 1)          │  "dK ready"
             │    Reducer: waits for signal, accumulates dK        │
             │  Reducer writes final dK to DRAM                    │
             │                                                     │
             ├─────────────────────────────────────────────────────┤
             │  Pass 2: dV reduction (identical protocol)          │
             │                                                     │
             │  Same as above but for dV_local                     │
             │  Reducer writes final dV to DRAM                    │
             └─────────────────────────────────────────────────────┘
```

**Worker-side (dataflow writer, after all Q rows processed for this K-chunk):**

```cpp
for (uint32_t pass = 0; pass < 2; ++pass) {
    // pass 0 = dK, pass 1 = dV
    uint32_t* local_accum = (pass == 0) ? dK_local : dV_local;

    // Wait for reducer to request our data
    noc_semaphore_wait(send_credit_sem_ptr, pass + 1);

    // NOC write local accumulator to reducer's input buffer
    uint64_t dest = get_noc_addr(reducer_noc_x, reducer_noc_y,
                                  reducer_input_base + my_slot * chunk_size * qWt * tile_bytes);
    noc_async_write((uint32_t)local_accum, dest, chunk_size * qWt * tile_bytes);
    noc_async_write_barrier();

    // Signal reducer: "my data is ready"
    noc_semaphore_inc(reducer_data_sem_noc_addr, 1);
}
```

**Reducer-side (dataflow reader):**

```cpp
for (uint32_t pass = 0; pass < 2; ++pass) {
    // Initialize accumulator with own partial (no NOC transfer for self)
    // copy own dK_local or dV_local into accum buffer

    for (uint32_t w = 0; w < W; ++w) {
        if (w == my_worker_id) continue;

        // Request worker w to send
        noc_semaphore_inc(worker_send_credit_noc_addrs[w], 1);

        // Wait for worker's data to arrive
        noc_semaphore_wait_min(data_ready_sem_ptr, expected_count);

        // Accumulate: accum += input_buffer[w]
        // (push into CB for compute kernel to add)
        cb_push_back(cb_reduce_in, chunk_size * qWt);
    }

    // Write final result to DRAM
    // pass 0 → write dK[k_start:k_end], pass 1 → write dV[k_start:k_end]
}
```

**Sync rounds:** 2 per K-chunk (one for dK pass, one for dV pass). Within each pass,
workers are serialized but each transfer is fast (NOC L1→L1). For D=64 (1 chunk):
2 total sync rounds. For D=256 (4 chunks): 8 total sync rounds.

**Reducer L1 requirements:**
- Input buffer: `chunk_size × qWt` tiles FP32 (one worker's contribution). For D=64:
  64 × 2 × 4 KB = 512 KB. Only one worker at a time — no per-worker slots needed.
- Accumulator: `chunk_size × qWt` tiles FP32 = 512 KB (same size, can reuse between passes)
- Total reducer overhead: ~1024 KB. The reducer is a lightweight core — it doesn't need
  the worker's Q/dO/intermediates CBs, only the reduction buffers.

| Pros | Cons |
|---|---|
| dQ always local — no dQ reduction at all | dK/dV accum can exceed L1 for large D |
| 2 sync rounds (D=64) vs 8 for Option 1 | K-chunking adds Q/dO re-reads (negligible) |
| No worker-side double-buffering needed | Reducer core is dedicated (not computing) |
| Extends existing `sdpa_bw_q` kernel naturally | Two sequential passes (dK then dV) |
| 2048 Q rows for B16 → near-perfect load balance | — |

#### B10.7 Worker L1 Memory Layout

**Worker CB allocation (one core, D=64, 1 K-chunk):**

| CB Index | Name | Size (tiles) | Format | Purpose |
|---|---|---|---|---|
| c_0 | cb_grad_output | 2 × qWt | BF16 | dO[q] (double-buffered) |
| c_1 | cb_u_scalar | 2 | FP32 | u_scalar[q] (double-buffered) |
| c_2 | cb_query | 2 × qWt | BF16 | Q[q] (double-buffered) |
| c_3 | cb_key | qWt | BF16 | K[k] (streamed in inner loop) |
| c_4 | cb_value | qWt | BF16 | V[k] (streamed in inner loop) |
| c_5 | cb_attn_mask | 1 | BF16 | Mask tile (reused) |
| c_6 | cb_intermediates | 2 × 2 | BF16 | max, recip_sum_exp (double-buffered) |
| c_7 | cb_scratch | 1 | FP32 | Matmul reduction scratch |
| c_8 | cb_attn_weights | 1 | BF16 | Recomputed P tile |
| c_9 | cb_grad_attn_weights | 1 | BF16 | dP tile |
| c_10 | cb_grad_scores | 1 | BF16 | dS tile |
| c_11 | cb_transpose_wh | 1 | BF16 | P^T or dS^T scratch |
| c_12 | cb_dQ_accum | qWt | FP32 | dQ accumulator (per Q row, tiny) |
| c_13 | cb_dQ_out | qWt | BF16 | dQ output (packed, written to DRAM) |
| L1 raw | dK_local | chunk_size × qWt | FP32 | dK accumulator across Q rows |
| L1 raw | dV_local | chunk_size × qWt | FP32 | dV accumulator across Q rows |

**L1 budget estimate** (S=2048, D=64, qWt=2, chunk_size=64):
- CBs (c_0 through c_13): ~25 tiles BF16 + ~4 tiles FP32 ≈ 50 + 16 = 66 KB
- dQ_accum: 2 × 4 KB = 8 KB
- dK_local: 64 × 2 × 4 KB = 512 KB
- dV_local: 64 × 2 × 4 KB = 512 KB
- **Total: ~1098 KB** of ~1500 KB available ✓

For D≥96, K-chunking reduces dK_local + dV_local to fit within ~1200 KB.

#### B10.8 Load Balancing (Integration with B16)

The fused kernel's outer loop iterates over Q rows. B16 (generic host-side scheduling)
assigns Q rows to workers. The number of schedulable work units is:

| Config | Work units (Q rows) | Cores | Max rows/core | LPT imbalance |
|---|---|---|---|---|
| TinyLlama (S=2048, GQA 32/4) | 2048 | 56 | 37 | **0.1%** |
| S=4096, GQA | 4096 | 56 | 74 | **<0.1%** |
| MHA (qH=32, kvH=32) | 2048 | 56 | 37 | **0.1%** |
| TP=2 (qH=16, kvH=2) | 1024 | 56 | 19 | **0.2%** |

Compare with Option 1 (outer K/V): only 256 K/V rows for TinyLlama → 5.2% imbalance.
Option 2 has 8× more work units to schedule, yielding near-perfect balance.

**Causal mask work variance:** Q row q processes K/V tiles 0..q (with causal masking),
so early Q rows have less work. B16's LPT scheduler naturally handles this — it treats
each Q row's cost as proportional to `min(q+1, St_kv)` K/V iterations and balances
accordingly.

#### B10.9 Accuracy Analysis

**dQ accumulation:** Same as current `sdpa_bw_q` kernel — inner loop over K/V,
accumulating `dQ += dS @ K` in FP32. Loop order unchanged → **bit-identical**.

**dK/dV accumulation order change:** In the current `sdpa_bw_kv` kernel, dK/dV
accumulate as `Σ_q (dS^T @ Q[q])` with q iterating over all Q rows for a fixed K/V
row. In the fused kernel, each worker sees only its subset of Q rows. The cross-core
reduction sums across workers. Due to floating-point non-associativity, bit-level
differences may arise vs the current single-core result. For BF16 training with FP32
accumulators, this is standard and **does not affect convergence** (same class of error
as GPU's `atomicAdd` non-determinism in FlashAttention-4).

**Verdict:** Partial fusion (B10.1): LOSSLESS. Full fusion: NEGLIGIBLE RISK (dK/dV
accumulation order only, no algorithmic change).

#### B10.10 Performance Estimate

**Savings from fusion (eliminated work):**

| Eliminated work | Current cost | Notes |
|---|---|---|
| Redundant P recomputation | ~29% of combined bw compute | P computed in both Q-bw and KV-bw; fused computes once |
| Redundant SFPU chain (exp, recip, mul) | included in above | Softmax recomputation halved |
| Separate KV-bw kernel launch | dispatch overhead | Single kernel launch instead of two |
| O tensor DRAM reads in KV-bw | St × qWt tile reads/head | Already eliminated by B5/B10.1 |

**Added work (reduction overhead):**

| Added work | Cost estimate | Notes |
|---|---|---|
| NOC dK transfer (W workers → reducer) | ~0.02 ms | 512 KB × (W-1) at ~12 GB/s NOC BW |
| NOC dV transfer (W workers → reducer) | ~0.02 ms | Same as dK |
| Reduction compute (tile additions) | ~0.01 ms | St_kv × (W-1) tile additions, trivial |
| Q/dO re-reads (K-chunking, D≥96) | ~0.03 ms per extra chunk | Negligible at DRAM bandwidth |
| **Total reduction overhead** | **~0.07 ms** | |

**Detailed estimate for TinyLlama (B=1, S=2048, D=64, qH=32, kvH=4, N300 56 cores):**

| Component | Current (two kernels) | Fused (Option 2) |
|---|---|---|
| Q-bw compute | 7.54 ms | — |
| KV-bw compute | 14.62 ms | — |
| Fused compute (29% reduction) | — | ~15.7 ms |
| Reduction overhead | — | ~0.07 ms |
| Load balance impact | 33.3% KV imbalance → +4.9 ms | 0.1% → ~0 ms |
| **Total** | **~27.1 ms** (with imbalance) | **~15.8 ms** |
| **Speedup** | | **~42%** |

The bulk of the saving comes from eliminating redundant P recomputation (29% compute
reduction) and the dramatic load balance improvement (from 33.3% to 0.1%).

#### B10.11 Implementation Phases

**Phase 1 — Fused compute with DRAM scratch reduction (validate correctness):**
1. Extend `sdpa_bw_q_compute_kernel` to also accumulate dK_local, dV_local in L1
2. After all Q rows, each worker writes dK_local, dV_local to per-worker DRAM scratch
3. Separate lightweight reduction kernel sums across workers, writes final dK/dV
4. Validate all three gradients against current two-kernel implementation

**Phase 2 — NOC-based MPSC reduction (optimize performance):**
1. Replace DRAM scratch with NOC-based two-pass protocol (§B10.6)
2. Implement credit-based flow control with semaphores
3. Place reducer on core nearest to DRAM bank holding output dK/dV tiles
4. Validate performance improvement over Phase 1

**Phase 3 — K-chunking for D≥96 (generalize to all head dimensions):**
1. Implement K-chunk loop around the outer Q loop
2. Per-chunk MPSC reduction: send dK_chunk, dV_chunk to reducer after each chunk
3. Validate for D=96, 128, 256
4. dQ accumulation spans chunks naturally (tiny accumulator persists)

**Phase 4 — Integration and cleanup:**
1. Remove `sdpa_bw_kv_compute_kernel` and `sdpa_bw_kv_program_factory`
2. Integrate B16 scheduler for Q row assignment
3. Single program factory for the fused kernel

---

### B12. Multi-Tile K/V Chunking for Backward

| | |
|---|---|
| **Source** | TTNN (same principle as F9) |
| **Impact** | Very High |
| **Effort** | High |
| **Accuracy** | **NEGLIGIBLE RISK** |

Same as F9 for backward. Process multiple K/V tiles per inner-loop iteration to reduce
per-tile overhead and enable subblock matmul.

---

### B13. Fused Correction Block (Backward)

| | |
|---|---|
| **Source** | TTNN (same as F6) |
| **Impact** | Medium |
| **Effort** | Medium |
| **Accuracy** | **SLIGHTLY BETTER** — eliminates intermediate BF16 quantization |
| **Requires** | B12 (multi-tile chunking) for full benefit |

Same as F6 for backward correction paths.

---

## Backward Summary Table

| # | Optimization | Impact | Effort | Accuracy | Requires | Status |
|---|---|---|---|---|---|---|
| **B1** | Fuse scale into exp | ~~Medium~~ | Low | LOSSLESS | — | **N/A** (superseded by B11) |
| **B2** | Approximate exp + ReLU | High | Very Low | MODERATE-HIGH RISK | — (pair with F2) | Not started |
| **B3** | recip_tile_first_column | ~~Medium~~ | Low | LOSSLESS | — | **N/A** (superseded by B11) |
| **B4** | exp_tile_first_column | ~~Medium~~ | Low | LOSSLESS | — | **N/A** (superseded by B11) |
| **B5** | Eliminate redundant u_scalar | Very High | Medium | LOSSLESS | — | **Done** ([#39812](https://github.com/tenstorrent/tt-metal/pull/39812)) |
| **B6** | Transposed recomputation | High | Medium | NEGLIGIBLE | — | Not started (ROI unclear — see notes) |
| **B7** | Overlap softmax/MMA pipeline | Very High | Medium-High | LOSSLESS | — | Not started |
| **B8** | K/V sharing for pairs (Q bw) | Medium | Medium | LOSSLESS | — | Not started |
| **B9** | Q/dO sharing for pairs (KV bw) | Medium | Medium | LOSSLESS | — | Not started (O no longer read after B5) |
| **B11** | Store logsumexp (eliminates recip error) | Medium | Low | **BETTER** | — (changes FW too) | **Done** ([#41683](https://github.com/tenstorrent/tt-metal/pull/41683)) |
| **B14** | Uniform dataformat skip | Low | Very Low | LOSSLESS | — | Not started |
| **B15** | Conditional rescaling (eps=0) | ~~Medium~~ | Low | LOSSLESS | — | **N/A** (superseded by B11) |
| **B16** | Generic host-side work scheduling | **Very High** | Medium | LOSSLESS | — | Balanced pairing done; heavy-row-first (LPT within pairs) **Done** (Phase A PR); host-side LPT upgrade not started |
| **B17** | SFPU reduce SUM for u_scaler | Medium | Low | LOSSLESS-to-BETTER | — | Not started |
| **B18** | Inner-loop CB roundtrip fusion | High | Medium | LOSSLESS | — | Not started |
| **B19** | Fused softmax recomp (copy_tile + SFPU sub_bcast_col + exp) | Medium-High | Medium | **BETTER** (full FP32 lse) | — | Not started |
| **B10.1** | Fuse backward (partial: u_scalar) | Very High | Medium | LOSSLESS | B5 ✓ | Not started |
| **B10.2** | Fuse backward (full: outer Q, reduce dK/dV) | Very High | Very High | NEGLIGIBLE | B10.1 | Not started |
| **B10 dK/dV** | dK/dV cross-core MPSC reduction (2-pass) | — | High | NEGLIGIBLE | B10.2 | Not started |
| **B12** | Multi-tile K/V chunking | Very High | High | NEGLIGIBLE | — | Not started |
| **B13** | Fused correction block | Medium | Medium | BETTER | B12 | Not started |

**Recommended implementation order (informed by profiling, updated May 2026):**

Both kernels are **strongly compute-bound** (WAIT < 0.01%, BW utilization 52%).
The KV-kernel (14.62 ms) is 1.94x the Q-kernel (7.54 ms) and dominates the backward.

Two independent bottlenecks:
1. **33.3% load imbalance** on KV-kernel — caused by GQA reducing distributable work
   units to 128 pairs for 56 cores. **B16 (generic host-side scheduling with LPT)**
   fixes this: drops imbalance to 5.2%, saving ~3 ms (21%). Works for any shape/hardware.
2. **SFPU-bound compute** — FPU and SFPU serialized on TRISC1, each idle ~50%.
   B7 (pipeline overlap) addresses this directly.

```
COMPLETED:
  Phase 0: B11 (logsumexp intermediates) ✓ — merged in #41683
    Eliminated recip_tile approximation error. Backward simplified to P = exp(S - lse).
    Also halved intermediate DRAM bandwidth and enabled cleaner Ring Attention.

  Phase A: F13 + F1 + F3 + F5 ✓ — PR #42820
    Heavy-row-first + fused scale+exp + first-column SFPU ops.
    22% SDPA FW kernel speedup, 5.9% end-to-end step-time reduction.

  B5 (u_scalar precomputation) ✓ — merged in #39812
    Eliminated O(S²) redundant recomputation. ~1.4% step-time improvement.

NEXT — Phase B (backward per-iteration compute improvements):
  B17 (SFPU reduce SUM for u_scaler row reduction)
    Replace FPU matmul-with-ones row reduction in compute_u_scalar_row with
    sfpu_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>. Eliminates CB roundtrip
    for the ones-vector and frees FPU cycles. Low effort, immediate win.

  B19 (fused softmax recomputation — copy_tile + SFPU sub_bcast_col + exp)
    Replace unary_bcast<COL> (TF32 truncation of lse through SrcB) + separate
    sub_binary_tile + exp_tile (2 SFPU passes) with:
    - copy_tile using unpack-to-dest (full FP32 lse, bypasses SrcA/SrcB)
    - Fused SFPU function: bcast_col_sub + exp in single traversal
    Or DeepSeek-style: MOVD2B + FPU ELWSUB + SFPU exp (better for B7 overlap).
    Precision win (full FP32 lse) + performance win (1 SFPU pass instead of 2).

  B18 (inner-loop CB roundtrip fusion)
    Eliminate redundant pack→L1→unpack between chained compute steps:
    compute_grad_attn_weights → compute_grad_scores, and softmax recomputation
    → its consumers. Keep intermediate results in DST registers. With B=5,
    that's 1280 × avg_inner_iterations redundant roundtrips eliminated.

  B7 (FPU/SFPU overlap — highest single-optimization impact)
    Overlap softmax recomputation (SFPU on PACK thread) with gradient matmuls
    (FPU on MATH thread). Uses DeepSeek-style MOVD2B + ELWSUB(SRCB_BCAST_COL)
    + exp_packthread_tile with semaphore-based tile-by-tile signalling.
    Expected saving: ~30-40% of compute time on both kernels.
    Does NOT require multi-tile chunking. Works on current single-tile arch.
    Reference: models/demos/deepseek_v3_b1/kernel_includes/.../sdpa.h

Phase C (load balance):
  B16 (generic host-side work scheduling)
    Replaces hardcoded balanced-pairing kernel logic with host-side LPT scheduler.
    Expected saving: ~3 ms (21% of KV-kernel) for TinyLlama on N300.
    Lower priority for large batch sizes (B=5 → <1% imbalance already).

Phase D (architectural changes):
  B6 (transposed recomputation — ROI unclear, deferred)
    Net gain is ~1 transpose per inner iteration after accounting for u_scaler
    transpose overhead. May revisit if profiling shows transpose as bottleneck.

  B12 (multi-tile K/V chunking) → B13 (fused correction block)
    Reduces per-tile overhead, enables subblock matmul.

Phase E (full fusion):
  B10.1 (partial fusion — u_scalar pre-pass already done via B5)
  → B10.2 (full fusion: outer Q, reduce dK/dV)
  Expected saving: ~42% of combined backward (29% compute + load balance fix).

Defer (marginal while compute-bound):
  B8/B9 (data sharing) — reader has 1.9x headroom, benefit is negligible.
  Re-evaluate after B7 shifts the bottleneck toward memory.

Always pair:
  F2 + B2 (same exp approximation in forward and backward)
```

---

# Accuracy Risk Summary

| Risk Level | Optimizations | Guidance |
|---|---|---|
| **LOSSLESS** (zero risk) | F1, F3, F5, F7(eps=0), F8, F12, F13, F14, B5, B7, B8, B9, B10.1, B14, B16, B18 | Apply freely |
| **LOSSLESS to SLIGHTLY BETTER** | B17 (SFPU reduce avoids TF32 truncation) | Apply freely |
| **SLIGHTLY BETTER** (improves precision) | F6, F9, F10, B6, B11, B13, B19 (full FP32 lse via unpack-to-dest) | Apply freely |
| **NEGLIGIBLE** (BF16 rounding order) | F4, B10.2, B10-dK/dV-reduction, B12 | Apply freely |
| **LOW RISK** (within BF16 noise) | F11(degree≥2) | Quick validation recommended |
| **MODERATE RISK** (needs convergence test) | F2, B2 | Must validate with 1000+ training steps; always pair F2+B2 |
| **UNACCEPTABLE** for training | F7(eps>0), F11(degree=1) | Do NOT use |
| **N/A** (superseded by B11) | B1, B3, B4, B15 | No longer applicable |

**The golden rule:** Whatever exp approximation you choose for forward, use the exact same
one in backward. Consistency between `P_forward` and `P_recomputed_backward` matters more
than absolute accuracy of either.
