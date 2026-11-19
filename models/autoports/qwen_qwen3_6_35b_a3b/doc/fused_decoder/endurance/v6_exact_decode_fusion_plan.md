# Exact decode fusion hypothesis

Source/CPU review only; no existing source, snapshot, or hardware was modified. The proposed kernels have not been implemented or timed. Corrected-FMA full-graph integration is a separate correctness task. The most promising bounded first fusion is **terminal-only ordered cumsum**, followed by streaming product+scan after exactness is established. It addresses large intermediates while retaining the proven reduction schedule.

## Evidence and its limits

`v6_double_single_diagnostic_latency_l0_s256.json` measures the frozen `dcaa3608…` decoder at **2.452392597 ms** traced warmed decode, versus historical V5 **1.285348088 ms** in `../qualified_v5/final_0.json`. That is about 90.8% slower. The current measurement is a failing diagnostic elsewhere; the historical source is not correct for the layer34 long stream. Neither is a valid completed-stage winner.

The available detailed profile is historical V5, `../profile_v5_final/tracy/layer_0/decode_op_sequence.csv.gz` and `decode_perf_report_stacked.csv.gz`. It has 166 operations per replay and 1.17087575 ms summed device kernel time. It is **not** a current dcaa per-op profile, and its giant inter-replay host gaps are not traced latency. Historical per-replay sums include 240.618 us for 38 DRAM-input binary ops, 173.449 us for two sparse matmuls, 82.149 us for the large L1 cumsum, 35.644 us for its untilize, and 21.736 us for its L1 tilize. A representative ordered-projection sequence is:

| Step | Historical operation | Device time, us |
|---|---|---:|
| 45 | FP32 products | 83.761 |
| 46 | FP32 cumsum | 81.942 |
| 47 | Last-32-row slice | 5.841 |
| 48 | Untilize terminal band | 35.706 |
| 49 | Extract final logical row | 2.287 |
| 50 | Retilize final row | 21.634 |
| 51 | Merge block partials | 6.324 |
| 52 | Output cast | 2.319 |

Frozen v2 `FusedDecoder.linear` now invokes the ordered product/cumsum path for all four decode projections below. The historical profile above has only one such large cumsum. This source change makes scan/intermediate traffic a plausible regression contributor, but a current profile must quantify its share.

| Projection | K | Output columns | K512 blocks | One FP32 product or prefix tensor |
|---|---:|---:|---:|---:|
| State attention pack | 2048 | 6208 | 4 | 48.5 MiB |
| Q/Z output pack | 2048 | 6144 | 4 | 48 MiB |
| Attention output | 4096 | 2048 | 8 | 32 MiB |
| Router | 2048 | 256 | 4 | 2 MiB |

Total is **130.5 MiB per set**. Product write + scan read + prefix write alone account for 391.5 MiB of logical tensor traffic across these four operations; this is a shape-derived count, not DRAM traffic or measured bandwidth. Interleaved L1 and NoC behavior matter. BF16 weights total 65.25 MiB across these same ordered operands.

## Candidate A: fuse scan and terminal-row extraction

**Boundary:** take the existing FP32 product tensor `[blocks,1,512,N]` and produce FP32 `[blocks,1,1,N]` in interleaved TILE L1, with padded H32 and zeroed unused rows. Replace cumsum, terminal-band slice, untilize, row slice, and tilize with one model-local `generic_op`. Keep the existing `fast_reduce_nc(dims=[0], output=explicit_H1)` and BF16 cast unchanged. Start with the 256-column router for a small all-column proof, then test the 6208-column projection for useful timing, and extend to the other two widths.

**Reader/core partition:** assign each `(block, output-column tile)` scan run to one core; distribute runs across available cores. Read its sixteen 32x32 FP32 tiles in increasing K order using the existing TensorAccessor/NoC pattern. Allocate input double buffering and one final-output tile per active run. Match the original accumulation op's FP32 destination/unpack settings; do not let FP32 pass through narrowed Src registers.

**Compute:** embed the current tile-axis cumsum loop and LLK calls. `accumulation_tile_axis_compute.cpp:40–89` uses `cumsum_tile_init` plus `cumsum_tile(first=j==0)` and retains carried SFPU registers across tiles. `ckernel_sfpu_cumsum.h:68–104` explicitly programs sequential SFPADD replay; the uncompensated mode is the ordered projection's required mode. Process every input tile, but omit output packing/public writes for the first fifteen tiles. Retain acquire/commit/wait/release synchronization correctly even when no tile is emitted. Pack only the final tile into a private FP32 CB. No arithmetic changes and no compensation/FMA substitutions are permitted.

**Writer/layout:** copy only final logical row31, as integer payload, into row0 of the zero-initialized output tile. For standard 32x32 face layout, row31 occupies FP32 element offsets 752..767 and 1008..1023; output row0 occupies 0..15 and 256..271. These are two aligned 64-byte spans. Validate this mapping with the current tile descriptor and signed-zero payload fixtures before device use. The resulting padded tile directly matches the old logical-H1 partial consumed by fast_reduce. This avoids materializing full prefixes or a separately untilized terminal band. A terminal output still occupies padded H32: for K512 the full-prefix output footprint shrinks 16-fold, not 512-fold.

**Why a new attempt is justified:** `AUTOFIX_v5_terminal_gather.md` already rejects the transpose/gather/transpose adapter: full traced projection 0.280329→0.463937 ms and whole decode 1.285217→1.470923 ms. Do not repeat it. Candidate A performs extraction inside the scan program and emits the consumer layout directly; it introduces no transpose/gather adapter. Its timing remains unproven.

**Exactness gates:** compare every FP32 block partial before the unchanged merge, then every merged FP32/BF16 output. Use actual router fixtures at positions227/738/5799, existing projection fixtures, random BF16 rows, signed zeros, and cancellation cases. Check K512 and K1024 separately; decode integration initially remains K512 only. Require eager/trace identity and unchanged long-stream router IDs/contributions before any selection. End-to-end traced comparison includes all input/output placement and final casts.

## Candidate B: stream products into the same terminal scan

Once A is exact, move BF16 weight and coefficient reads into the same program, preserving setup's pair-swapped K order. Existing weights are already packed with `order = arange(K).reshape(-1,2).flip(1)`; reader activation address `k xor 1` supplies the matching coefficient directly from the logical input row, removing the standalone input gather/retile. Widen BF16 operands and multiply to exactly rounded FP32, then use A's sequential scan. BF16×BF16 products have enough precision for FP32 in the model's finite normal range, but compare codes rather than assuming exceptional cases away.

Do **not** replace `RN32(sum + RN32(a*b))` with addcmul/FMA, a tree sum, or corrected-FMA: the projection's proven schedule is separate product followed by increasing-K additions. Nor may multiplication clobber cumsum's carried LREG4–7. Either explicitly spill and restore these four registers in a reserved Dest/CB area around product generation, or implement a private product-and-scan loop with documented register ownership and the same SFPADD ordering. This register lifetime issue makes B a second experiment, not a trivial edit to A. Retain native block merge until its exact schedule has independently been copied and proven.

The larger extension can process both attention packs in one program with shared coefficient reads, but their output-column runs and block boundaries remain independent. No measured gain is inferred from fewer calls. Do not fuse corrected chunk-prefill GEMM into decode ordered projections: they intentionally implement different accumulation semantics.

## Candidate C: exact SiLU gather, multiply, and final packing

Use the already-normalized/weighted FP32 core `[1,32,128]` plus BF16 Z `[1,1,4096]` as the boundary. Retain the complete explicit RMS schedule initially, including BF16 core, normalized-value and gamma-product boundaries. A generic reader extracts each head's 128 BF16 Z codes directly from tiled rows and indexes a setup-owned copy of the existing 65,536-entry FP32 SiLU table. A 256-KiB table per active head core fits within the per-core L1 budget after CB accounting; alternatively share fewer table-owning cores and measure the NoC tradeoff. Replicating across 32 cores costs 8 MiB of setup L1, so verify placement rather than assuming it is free.

Feed exact table FP32 payloads and weighted FP32 core to compute with direct FP32 unpack. Apply the same separate FP32 multiplication and nearest-even BF16 boundary as current `multiply(... dtype=bfloat16, fast_and_approximate_mode=False)`. Emit logical `[1,1,4096]` TILE output directly, zeroing padded rows. This can remove the bitcast/layout/reshape/gather/tilize sequence and the later reshard/concat-heads adapters; bitcast itself may be metadata-only and should not be counted as saved kernel time. Keep the ordered output projection unchanged. Prove all 4096 BF16 output codes against the current device path, including all-code SiLU table address tests, before fusing RMS reduction or rsqrt. The explicit norm's input/post-attention arithmetic and its compensation corrections are independent future boundaries.

## Candidate D: remove selector transport without changing selection

`InlineSelector.__call__` reshapes 256 router logits into a face, moves them to one L1 shard, executes the exact BRISC nth-element code, moves values/IDs back, casts/softmaxes/casts. The smallest change is a generic dataflow reader that reads original `[1,1,1,256]` BF16 tile row payloads into contiguous 256-entry scratch on the selector core. Reuse the exact existing `v6_select_top8` source and index order. Write selected values directly in the current softmax input layout and IDs in the current sparse-matmul index layout, avoiding input reshape/reshard and output placement copies where the actual profile shows them. Keep the existing FP32 softmax and BF16 score cast; replacing that arithmetic is a separate precision experiment. No host scalar selector is introduced—the control work remains BRISC.

After ordered-router fusion, its output writer could hand off these 256 codes to the selector core inside one program, but that requires explicit completion semaphores across all column owners. First prove the standalone transport change; do not combine cross-core ordering, softmax, and projection arithmetic in the initial candidate.

## Selection and validation order

Obtain a complete profile of the now-correct integrated snapshot first; use the exact source hash for both traced latency and tt-perf-report. Prioritize A, then B if scan/product traffic dominates; C and D target smaller layout/dispatch costs and should not be assumed to erase the entire regression. Each candidate uses setup-owned descriptors/buffers, warms every shape before trace, includes all adapters in complete-path timing, and leaves prefill/contexts unchanged unless explicitly tested. Reject any candidate with slower full traced decode even if op count or bytes fall. Carry failures and untried items in the main work log under parent ownership. No roofline, Stop Rule, accuracy pass, or speed improvement is claimed by this plan.
