# AutoDebug: Stage08 BFP8 projection geometry exceeds L1

Investigation date: 2026-10-25. Checkout: `af283dd6d375f8f7ffe057b2c1e271f9e4aef3e9`, including the caller's uncommitted Stage08 precision plumbing. Inspection only: no target execution, hardware access, resets, implementation edits, or AutoDebug relaunch.

## Finding

**The model promotes projection weights from BFP4 to BFP8 but retains a BFP4-tuned explicit matmul geometry. The first linear-attention mix consequently requests exactly the static L1 endpoint in the failure: 1,405,568 bytes, crossing the live tensor frontier at 1,212,032 by 193,536 bytes.**

The earliest unsupported sizing assumption is in model policy resolution, before the explicit program configuration is constructed. It is not a demonstrated allocator error, numerical LoFi failure, or disagreement between the selected factory and its reader. The factory correctly accounts for the larger weight tiles and final validation rejects the geometry against its live compute-L1 frontier.

The smallest block-only repair candidate is to retain eight activation storage cores and one reader per bank, but use **mix K-block 10 instead of 20 for BFP8**. Its predicted static endpoint is **777,088** bytes. A second legal candidate is **two readers per bank with block 20**, predicting **801,664** bytes and preserving K-reduction grouping. Neither candidate has been tested here. A complete BFP8 sweep repair must also cover ordinary down, large prefill projections, and the copied batched/prefill policies described below.

## Direct observations and revision consistency

- [bfp8_lofi.log](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/bfp8_lofi.log) loads layers 0–63, then fails inside `prepare_decode()`'s first `_decode()`, at `MultichipDecoder._delta -> _linear -> OptimizedDecoder._linear -> ttnn.linear`.
- [results/bfp8_lofi.json](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/results/bfp8_lofi.json) records one attempted linear call: BF16 input `[1,1,1,5120]`, BFP8 weight `[5120,4608]`, LoFi, FP32 destination disabled. Instrumentation records arguments **before** calling the operation; this is not evidence that the operation completed (`capture_precision_calls` in [run_datatype_candidate.py](models/autoports/qwen_qwen3_8_27b/tests/run_datatype_candidate.py)).
- The error is `validate_dataflow_buffer_region` during `compile_and_allocate`, with static endpoint **1,405,568**, lowest occupied compute L1 address **1,212,032**, and core range `[0-0 - 7-9]`.
- [baseline_bfp4_lofi.json](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/results/baseline_bfp4_lofi.json) and [bfp4_hifi2.json](models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/results/bfp4_hifi2.json) both report `pass`, top1 `.99`, top5/top100 `1.0`, 100 compared positions and verified tracing. Both record the same mix geometry. These are full 64 B1 readiness results, not proofs of every long-context/batch combination.
- A cheap JSON/SHA256 check found that every recorded `tt/*.py` source hash in the failure matches the inspected files and both passing results. HEAD also matches the recorded commit. The runner gained instrumentation between runs, but the compared model sources match. No historical/current model path was spliced together.
- The MPI shared-memory warning precedes successful weight loading. It is not needed to explain the exact host-side L1 rejection. No evidence here establishes a hang or requires device recovery.

The causal explanation below is derived from source; the smaller configurations and sibling failure predictions are **not new runtime observations**.

## Policy-to-factory causal chain

1. [precision.py:65](models/autoports/qwen_qwen3_8_27b/tt/precision.py#L65) maps the precision candidate to `attention_dtype`, `mlp_dtype`, `down_dtype`, fidelity and activation choices. It performs no corresponding geometry adaptation. Layer exceptions are merged last.
2. [multichip_decoder.py:25](models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py#L25) installs Stage05 geometry, including `mix={cores:8, block:20, readers:1, weight_pad_readers:2}`. The linear-attention prefill block becomes 16 at line 73. The candidate is merged at line 85 without altering those dictionaries.
3. Real linear-attention weights are TP-split and packed at [multichip_decoder.py:157](models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py#L157). Per rank the widths are `512+512+1536+1536+32+32=4160`. The final two 32-wide groups each contain 12 values and 20 padding columns. Weight upload additionally rounds the full width to a multiple of `32 × 8 banks × 2 padding readers = 512`, yielding **4608** columns ([lines120–138](models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py#L120)). Thus 4160, not 4608, is the Python-visible packed projection width, but **4608 must be used for matmul storage and buffer sizing**.
4. `build_generator()` prepares automatically ([generator.py:689](models/autoports/qwen_qwen3_8_27b/tt/generator.py#L689)). `prepare_decode()` resets and invokes `_decode()` before prefill warmup ([generator.py:246](models/autoports/qwen_qwen3_8_27b/tt/generator.py#L246)). Embedding/gather and input RMSNorm feed layer 0's mix. No recurrence, SDPA, row reduction, or LM head matmul has executed on this failing decode path.
5. [optimized_decoder.py:224](models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py#L224) selects BF16 ordinary attention input, reshapes to `[1,1,B,5120]`, and width-shards it over eight cores. At B1 its padded shape is `[1,1,32,5120]`, shard `[32,640]`, i.e. **20 K tiles per storage core**. It explicitly supplies `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=20, per_core_M=1, per_core_N=18, num_workers_per_dram_bank=1)`, BF16 output, L1 width-sharded output, no bias/activation, `packer_l1_acc=True`, FP32 destination false.
6. This `linear` is an ordinary A×B operation, with neither transpose nor operand swapping. Rank normalization does not change K/N, dtype or sharding. The explicit configuration is returned unchanged by [matmul_program_config.cpp:1022](ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp#L1022); the automatic capacity planner is bypassed. [matmul_device_operation.cpp:2523](ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp#L2523) selects `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`.
7. Validation checks shape, one physical M tile, K divisibility and shard/block compatibility ([matmul_device_operation.cpp:1581](ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp#L1581)). All pass for block 20. These checks do not certify available L1 capacity.
8. The selected factory uses the actual input formats for buffer bytes and discovers actual DRAM readers. This yields the collision below. Final validation is the first exact live-allocation rejection on this path.

## Exact resource ledger

In this section, `F` refers to [matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp](ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp).

After lowering: batch product `B=1`, tile dimensions `M=1`, `K=160`, `N=144`; eight active DRAM banks, one worker per bank, 18 weight tiles per reader; K-block 20 gives eight K iterations. Python's `cores` controls activation storage. Reader compute width is separately derived from actual bank readers in `F:174–216`; changing `cores` alone does not automatically reduce the weight buffer.

The divisor-based output subblock is `1×6`. The factory's width-widening loop leaves 18 unchanged: widths7/8 do not reduce the three subblocks. `last_subblock_w_valid=6`. There is no output-padding discrepancy in the failing case (`F:218–255`).

For standard 32×32 tiles, BF16=2048 bytes, BFP4=576, BFP8=1088, including exponent storage ([tile.cpp:136](tt_metal/impl/data_format/tile.cpp#L136)). All already satisfy Blackhole's 64-byte DRAM alignment. The factory double-buffers A and triple-buffers B when more than one K iteration is present (`F:283–308`). BF16 partials and BF16 output alias one allocation (`F:479–536`). No bias, FP32 partials, or multi-shard gathering scratch applies.

| Per-core static allocation | BFP4 block 20 | BFP8 block 20 | BFP8 block 10 |
|---|---:|---:|---:|
| A buffer | `2×1×20×2048 = 81,920` | 81,920 | 40,960 |
| B buffer | `3×18×20×576 = 622,080` | `3×18×20×1088 = 1,175,040` | 587,520 |
| Output/partials, shared | `1×18×2048 = 36,864` | 36,864 | 36,864 |
| Static payload total | **740,864** | **1,293,824** | **665,344** |
| Static base | 111,744 | 111,744 | 111,744 |
| Static endpoint | **852,608** | **1,405,568** | **777,088** |
| Margin to recorded frontier 1,212,032 | 359,424 | **−193,536** | 434,944 |

The base is independently source-derived: Blackhole `MEM_MAP_END=41,088` ([dev_mem_map.h:111](tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h#L111), through line 182), plus default kernel-config reservation `69 KiB=70,656`; alignment leaves **111,744** ([bh_hal_tensix.cpp:43](tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp#L43), lines68–69). This is not a residual inferred solely by subtracting from the error.

Promotion increases the B allocation by **552,960** bytes, exceeding the former margin by exactly **193,536**. Changing LoFi to HiFi2 alone changes neither operand tile size nor this buffer geometry, explaining both passing BFP4 fidelities and predicting the same initial failure for an otherwise identical BFP8/HiFi2 candidate. The latter has not been established by this log.

The factory runs over the bounding rectangle of storage and reader cores (`F:375–380`, `F:1230–1232`), consistent with `[0-0 - 7-9]`. These are per-core sizes, not mesh-wide totals; do not multiply them by eight when comparing to per-core L1.

### Allocation timing and completion boundary

The normalized activation and persistent model/workspace allocations are live before `linear`. Output storage is allocated before the factory compiles its static buffers; its spec uses the caller's `per_core_N=18`, giving a BF16 `[32,576]` shard ([matmul_device_operation.cpp:2836](ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp#L2836)). The input's resident shard and output tensor are accessed separately from the static FIFO buffers (`F:496–500`). They must remain in the live-tensor budget, not be counted again as static buffers.

[dataflow_buffer.cpp:3070](tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp#L3070) compares the final static endpoint with the device's current lowest occupied compute-L1 address. The artifact does not identify which live tensor owns address1,212,032; it need not be the output, and the check is not a per-tensor forensic allocation map. That identity is unnecessary for the exact arithmetic above.

No failed mix kernel has been enqueued successfully: the exception is in compilation/allocation. Earlier work may have been enqueued, but this symptom is not a late synchronization report of bad recurrence/CCL execution. Future capacity admission must account for the output and guaranteed later workspaces if it runs before their allocation; a transient free-L1 snapshot is insufficient.

## Minimal adaptation and legality

Resolve geometry from the **final per-layer dtype policy**, after overrides and before weight packing and phase-view copying. Keep the existing BFP4 geometry and any explicitly supplied geometry that already satisfies the budget. Use the actual input/weight/output tile bytes, buffering depths and physical shapes for larger formats; do not key the adjustment only on fidelity or a config name. Record the resolved geometry. A small model-side policy helper is sufficient; this failure does not justify changing the allocator or shared kernel buffering depth.

For the immediate mix, block 20→10 preserves storage cores, weight layout/padding, readers, output storage and the Python slicing contract. Legal smaller blocks for a20-tile shard are 10,5,4,2,1. Block10 satisfies `160%10==0` and `20%10==0`; **block 16 is illegal despite dividing160**, because neither16 nor20 divides the other. The C++ validation explicitly requires both conditions.

The sender contract also permits the change: block 20 has 8 iterations and 1 block per activation shard; block 10 has 16 iterations and 2 blocks per shard. `per_core_M=1` allows this split (`F:310–399`). Each block produces/consumes20 A tiles and 360 B tiles in the old geometry, versus10 A tiles and 180 B tiles in the new geometry. Each worker still consumes all 160 K tiles and 18 N columns. The weight reader explicitly implements a three-block ring ([reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:120](ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp#L120)); its initial reservation of two blocks is not a two-block allocation contract. The active compute path is [bmm_large_block_zm_fused_bias_activation_metal2.cpp](ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp), selected at `F:1101`. The changed K grouping can change rounding and needs a real-weight comparison.

Alternatively set readers 1→2 while retaining block 20. Existing weight padding makes18 bank tiles divisible by 2; each reader consumes9 columns, while the factory widens compute width 9→10 using subblock1×5 and 4 valid lanes in the last subblock. Static bytes become81,920+587,520+20,480, plus111,744 base = **801,664**. Output storage still uses the Python `per_core_N=18`; reader geometry and storage geometry must not be conflated. This candidate preserves 8 K iterations but changes reader placement, so it also needs validation. Either is a plausible minimal intervention; performance does not follow from static byte counts.

When applying nested policy changes, copy the existing role dictionary and update its intended keys. The constructor uses a shallow top-level `policy.update(...)`: passing only `{"mix":{"block":10}}` discards role-specific prefill/padding settings and resets prefill geometry to unrelated global defaults. For this exact mix shape, the upload helper's missing-reader default of two happens to preserve the 4608-column packing; it is not evidence that a sparse replacement preserves the complete policy. Retain all untouched keys for a controlled comparison.

## Prefill and batched siblings

**These are source-predicted obligations for the repair, not failures observed after the reported mix.** The BFP8 run stops before reaching them. Their live-L1 frontier must be measured in their own phases;1,212,032 is not a universal budget.

### Shared mix and ordinary decode

`MultichipDecoder` deep-copies the resolved policy into `_batched_decode_impl` and `_prefill_impl` at [lines199–220](models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py#L199). B8–32 uses the batched view. Compact gather passes `decode_batch` back into the same ordinary `_linear` ([lines395–417](models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py#L395)). Packing B≤32 rows still gives one M tile, so the mix block 10 legality applies to B1/8/16/32. This is **not** the distinct `MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig` factory. A one-row prefill specialization also enters the shared DRAM-sharded branch, using the configured prefill input dtype. The generator deliberately preserves physical length 1; other lengths round to 32 ([generator.py:384](models/autoports/qwen_qwen3_8_27b/tt/generator.py#L384)). Thus its4097-token sequence really can exercise a4096-row chunk plus the one-row branch.

Ordinary down is another large one-reader projection: local K=4352→136 tiles, N=5120→160 tiles, storage shard17 tiles, block 17,20 weight columns per bank. A blanket block-halving rule is invalid: block 8 does not divide the 17-tile activation shard. A legal and less disruptive BFP8 candidate is readers 1→2, retaining block 17 and eight storage cores. The existing20-tile bank width is already divisible by 2 and physical N=5120 remains unchanged. Block1 is the smaller block-only fallback, with a potentially substantial performance cost. B8–32 native row fusion uses a different path, so changing ordinary down alone does not adapt it.

| Ordinary BFP8 down | A bytes | B bytes | Shared output/partials | Static endpoint including 111,744 |
|---|---:|---:|---:|---:|
| Block17/readers 1 | 69,632 | 1,109,760 | 43,008 | **1,334,144** |
| Block17/readers 2 | 69,632 | 554,880 | 20,480 | **756,736** |

The original down output/partials use **21 compute columns**, not20: subblock1×5 widens to 1×7 with three subblocks. With readers 2,10 reader columns remain10 compute columns (`F:208–255`). This is a real rounding effect but does not occur for the initial mix's18 columns. Historical Stage05 controls used down8/block 17/readers 2 successfully for both layer kinds ([matmul_search.csv:32](models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/matmul_search.csv#L32), line 51); those weights were BFP4, so they support the geometry contract, not a BFP8 numerical pass.

### Ordinary 2D prefill

For length>1, [optimized_decoder.py:262](models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py#L262) explicitly constructs a 2D multicast config. For BF16 output/partials, DRAM interleaved activations and no bias, its main static buffers total

`2×H×Kblock×A_tile_bytes + 2×W×Kblock×B_tile_bytes + H×W×2048`.

Here H/W are **output-block dimensions**, not necessarily `per_core_M/N`. Source: [2D factory:426](ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp#L426), aliasing at line 798. The input and weight buffering depths are two on this path, unlike the triple-buffered DRAM-sharded decode reader.

| Reachable B1 prefill case | Geometry at 4096 | BFP4-weight static bytes | BFP8-weight static bytes | Legal BFP8 candidate |
|---|---|---:|---:|---|
| Linear-attention mix | grid 11×10; H=13,W=14,Kblock 16; BF8 input | 1,083,392 | **1,312,768** | Kblock 8:842,752 |
| Separate gate/up | grid 11×8; per-core M=16, **H=8**,W=13,Kblock 20; BF8 input | 860,672 | **1,126,912** | Kblock 10:669,952 |

Add system reservation before comparing these payloads to live tensor allocations. For example, linear mix BFP8 gives 1,424,512 including 111,744 base. The gate/up BFP8 endpoint would be1,238,656. These have materially less space for live L1 than the passing BFP4 versions.

For linear mix, prefill_block 16→8 satisfies K=160 divisibility and brings the main static footprint below the BFP4 footprint. For gate/up,20→10 does likewise;20→16 is a smaller shrink and legal in the **2D prefill** factory, but should only be selected with measured headroom (payload 944,128, endpoint 1,055,872). The decode 20-tile-shard restriction must not be incorrectly applied to interleaved 2D prefill. Default full-attention mix uses a different grid/block and must be sized independently. The default packed gateup prefill path is bypassed for multirow prefill by `separate_prefill=True`; if that option changes, its wider output must receive its own budget.

### Fused row projections

Full 4096-row B1 prefill out/down route through [fused_prefill.py:21](models/autoports/qwen_qwen3_8_27b/tt/fused_prefill.py#L21), while B8–32 native out/down use [fused_decode.py:27](models/autoports/qwen_qwen3_8_27b/tt/fused_decode.py#L27). They read `prefill_rs_geometry` and `decode_rs_geometry`, respectively; ordinary role `block` changes do not reach them.

The active fused factory builds a descriptor via [matmul_reduce_scatter_async_program_factory.cpp:116](ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory.cpp#L116), then [ccl_fusion_mcast_2d.cpp:1973](ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp#L1973). Its matmul inputs are double-buffered. Fused prefill has BF8 output but BF16 packer partials and therefore allocates separate output/partials buffers ([line 2918](ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp#L2918)); native batched BF16 output can alias BF16 partials. Also budget the fused CCL buffers and persistent live allocations. Do not apply the decode three-buffer formula to this path.

| Fused BFP8-weight path | H,W,Kblock | Main matmul CB bytes, excluding base |
|---|---|---:|
| B1 prefill down, BF16 input/BF8 output | 4,15,17 | `278,528 + 554,880 + 65,280 + 122,880 = 1,021,568` |
| Native batched down, BF16 input/output | 1,16,17 | `69,632 + 591,872 + 32,768 = 694,272` |
| Native batched full-attention out | 1,16,24 | `98,304 + 835,584 + 32,768 = 966,656` |
| Native batched linear-attention out | 1,16,6 | `24,576 + 208,896 + 32,768 = 266,240` |

These main matmul demands do not by themselves require a fused geometry change. CCL is placed outside the selected matmul grids: prefill starts at(0,8) beyond 11×8; decode at(0,6) beyond 10×1. Nevertheless, the final validator uses the workload's live L1 frontier; actual phase-local acceptance still requires hardware evidence.

If a fused row budget needs reduction, choose a divisor of the actual local K: down K=136 permits 17→8 or 4; out K=48 permits 6→4/3/2, and full-attention batched out24→12/8/6. These interleaved-input paths do not have ordinary down's17-tile storage-shard constraint. Keep `out_block_w==per_core_N` in the native fused DRAM weight-reader contract ([fused_decode.py:4](models/autoports/qwen_qwen3_8_27b/tt/fused_decode.py#L4)); shrinking output width to save memory is not an interchangeable fix. No observed fused failure justifies changing these settings unconditionally.

A narrow initial BFP8 policy candidate is thus: adapt mix decode; linear mix prefill 16→8; gate/up prefill 20→10; ordinary down readers 1→2. Resolve these before creating both phase views. Verify fused paths with their own budgets and adapt only if needed. BF16 promotion, layer-specific exceptions, or changed activation dtype require another actual-byte calculation, not reuse of the BFP8 constants.

## Discriminating reduced real-weight checks for the hardware owner

These are proposed follow-ups, **not commands executed in this investigation**. Reuse serialized hardware ownership. Start with single projections, then two representative real layers, and only then full 64.

1. **Exact mix allocation contrast.** Load layer 0 real weights and recorded input through production packing (including 4160→4608 pad). Isolate the ordinary mix with BF16 activation, BF16 output and fixed LoFi. Compare BFP4 block 20/readers 1, BFP8 block 20/readers 1, BFP8 block 10/readers 1, and BFP8 block 20/readers 2. Record physical shapes, shard specs, actual config, static buffer sizes, static base, and lowest live L1 address after output allocation. Predict the four endpoints 852,608;1,405,568;777,088;801,664. A reduced model may leave a higher live frontier and let the original BFP8 geometry compile: that does not refute this diagnosis. Use matched persistent background allocations or a full-owner construction without full execution to reproduce the frontier; record actual allocation, not just requested background bytes.
2. **Separate geometry correctness from precision quality.** Compare the two legal BFP8 geometries on the same real normalized input and packed weights, then against an appropriate CPU/reference projection. Check every Q/K/V/Z/A/B slice and all TP ranks, not only the concatenated PCC. There are 4120 nonpadding channels:4096 Q/K/V/Z values, A at[4096:4108], and B at[4128:4140]. Mask the 40 group-padding columns within the Python-visible 4160 width and the 448 trailing physical-padding columns. Use LoFi/HiFi2 as a second axis while holding dtype/geometry fixed. K-block changes can change accumulation rounding, so bitwise equality is not an assumed requirement. Compilation success is not a numerical pass.
3. **Catch the next ordinary down failure directly.** Exercise real layer 0 and layer 3 down weights at B1 with BFP8, original block 17/readers 1, then readers 2. Compare actual static endpoints and output slices. If block 8 is offered as a generic fallback, a static check must reject it before hardware dispatch with eight activation storage cores. Also check the alternate mix geometry has not changed the physical down weight shape 5120.
4. **Prefill boundary matrix.** Use real layer 0 and layer 3 at logical lengths 1,31/32/33,203 (actual 224 prepared bucket),2049 (physical 2080),4095/4096/4097, plus a continuation with a one-token remainder. Exercise raw decoder and prepared-generator paths separately and record physical shapes. The generator preserves 1;31/32→32 and 33→64. At 4096 record H/W/Kblock and exact input dtypes for mix, separate gate/up and fused out/down. At 4097 ensure both the full 4096 chunk and one-row remainder run. A short 32-row smoke cannot establish4096-row capacity.
5. **Batched phase coverage.** Run reduced real layers at B1,8,16,32 (and 7 as the pre-native boundary), with production compact gather and the actual `decode_implementation`. Record `_batched_decode_impl`/`_prefill_impl` policies as well as the base policy. Check changed tokens/positions across several decode steps, recurrent state and full-attention cache updates. Include native fused out/down and prefill out/down; those configurations come from different dictionaries.
6. **Complete asynchronous work and cache coverage.** Synchronize or read back after each initial call. Warm the same configurations, capture/replay with changed input values, then synchronize and compare again. The existing [run_multichip_decoder.py](models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py) provides real layer 0/3 weights, recorded inputs, `--policy`, `--batch`, `--split-prefill`, background-L1 allocation, and blocking trace checks. Its single-chip comparison retains the default policy, so a BF8-versus-BF8 geometry check needs a controlled reference/probe; do not call the default BF4 baseline a same-dtype comparison.
7. **Re-enter the full Stage08 run only after those controls.** Use the same precision config and readiness reference, record resolved policies and actual calls, verify full 64 preparation, prefill/teacher forcing accuracy and warmed tracing. Keep BFP4 LoFi/HiFi2 controls unchanged. The failed BFP8 run has no accuracy/performance result and must not be ranked as an inferior datatype. Full-model long-context, serving-batch and performance gates remain separate from proving this allocation repair.

## Other Potential Issues and rejected explanations

| Candidate | Evidence and adjudication | Smallest discriminating check |
|---|---|---|
| Model dtype/geometry contract drift | Exact source-derived endpoint; matches same-source BFP4 pass/BFP8 compile failure; no extra dynamic precondition beyond recorded frontier. **Leading finding.** | Same real projection with only block/readers changed; inspect endpoints. |
| Generic TTNN CB planner underestimate | Generic helper uses a different accounting model, but explicit program_config bypasses it. It cannot be the executed cause here. | Verify supplied config is present at `get_program_config`; it is in source. |
| Weight padding wastes enough bytes to cause failure | Padding increases physical N and must be budgeted, but the reader-aware packing is deliberate and shared by passing cases. Removing it alters a downstream packing contract. | Retain padding and shrink a legal block; predicted overlap disappears. |
| Factory overallocates triple buffer by mistake | The weight reader explicitly uses three slots; no factory/reader disagreement found. Output/partials correctly alias. | Inspect ring-slot usage and allocation counts; already consistent. |
| LoFi numeric corruption, recurrence, CCL race, stale trace or teardown failure | The failure occurs in first-mix compilation before those consumers or trace capture. Does not explain exact byte match. | Compile same shapes at HiFi2; bytes should be unchanged. |
| Persistent L1 leak or unusually large full-model workspace | Live frontier matters, but no leak is needed. The artifact does not identify the lowest allocation; synthetic reduced-model success cannot rule out full-owner pressure. | Allocation map across owner construction/output creation; compare against recorded 1,212,032. |
| Prefill/down/fused failures after fixing mix | Ordinary down and large prefill have increased source-derived demands; the failed run never reaches them. Fused paths need separate accounting. These are follow-up capacity risks, not additional observed failures. | Reduced per-role real-weight matrix with actual phase-local frontier. |

## Report review

After an independent xhigh review, the draft was corrected to name the active Metal 2 compute kernel, derive the actual prepared token lengths, distinguish padding from numerical channels, and preserve the exact sparse-policy fallback behavior. The independent review returned a clean-pass inspection verdict. The report was checked against the recorded source hashes, explicit-config bypass, actual factory selection, physical padding, reader count, subblock rounding, buffering depth, alias allocation, Blackhole reserved base, and final overlap check. The precise initial failure is fully accounted for. Remaining uncertainty is hardware execution/numerical behavior/performance of proposed legal configurations and the phase-specific live capacity of later projections; none is claimed as verified.
