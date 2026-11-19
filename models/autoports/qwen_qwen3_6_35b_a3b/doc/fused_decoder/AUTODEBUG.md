# AutoDebug: DRAM-sharded packed projection reads an interleaved bias as contiguous bank shards

Inspection-only investigation of the prepared tree at `4c643884bf51a39fc1166e0b86b86bd373bedfee`. No device execution or source changes were performed. This report is the only investigator-created file. Model-relative paths below start at `models/autoports/qwen_qwen3_6_35b_a3b/`. Native `device/...`, `factory`, `reader`, and `compute` references abbreviate paths under `ttnn/cpp/ttnn/operations/matmul/`, with the full filenames identified below; `tt_metal/...` and `tests/ttnn/...` are repository-relative.

**Finding:** the candidate shards the packed weights but leaves `attention_bias` interleaved in DRAM. The selected native reader reads bias directly by DRAM bank, assuming the same contiguous column partition as the weights. It therefore adds the wrong bias columns. The wrapper and native validation silently admit this incompatible layout. This is a concrete preparation/reader contract violation, supported by a saved control in which changing only the bias to matching DRAM shards makes all four packed-output slices match the passing auto-config reference exactly.

The supported correction boundary is bias preparation for this explicit DRAM-sharded projection. No independent weight-sharding, FP32-output, accumulation, or column-padding defect was demonstrated. A complete decoder run after correcting bias preparation remains necessary; the saved successful control covers isolated projections.

## Direct observations and provenance

- `doc/fused_decoder/logs/grouped_prefill_adapted_0.log:22-24` records prefill PCC `0.9999250822088881`, eager/traced decode PCC `0.9998867224045151`, and deterministic replay.
- `doc/fused_decoder/logs/dram_reader1_block2_0.log:22-33` records the same prefill PCC, eager/traced decode PCC `0.9671179393579047`, deterministic replay, and failure of the unchanged `0.995` gate. The failure already exists before capture.
- The failing sidecar identifies Blackhole P300c, physical chip 0, `MeshShape(1,1)`, layer 0, length 256, context 512. Its candidate SHA256 is `f2846d119e1e44a0bd8bd977d45e4998e32ef6492b3efc9bdf1f3ae53e39f953`, matching the inspected `tt/fused_decoder.py` and both projection-probe sidecars.
- `doc/fused_decoder/grouped_prefill.py.snapshot` hashes `316df974c8e7041939848352f46ff051c8b5565335c5285884eb83b8bb5b7385`. Diffing it against the candidate finds only the `decode_linear` preparation and single-row dispatch branch. This snapshot exactly matches the passing `grouped_battery_0.json.provenance.json`; its log at lines 21-35 includes length-1 prefill/decode, awkward boundaries, continuation, and batch-32 checks. The adapted timing run above records a different source hash, `12debe3c…`; its precise PCC must not be attributed to the snapshot byte-for-byte.
- Existing `projection_probe.json` and `projection_probe_sharded_bias.json` are device results produced outside this investigation. Their sidecars identify the same candidate model hash. The latter, timestamp `2026-11-12T20:17:03Z`, records `--shard-bias` and matches current probe-source SHA256 `5bf57b6a49773dbb9f85ba226610efe2bb6dd3188d57bd0a610797df8cb665e5`. The original probe, timestamp `20:15:18Z`, records its earlier source hash `0ed6da24…`; the probe subsequently gained the optional bias-sharding control. Current probe lines 29-40 show that control and the separate-add comparison explicitly.

The probe uses seed 41 and one independent BF16 random input scaled by 0.05 per projection. Its reference is the unchanged interleaved-weight auto-config `ttnn.linear`, not HF. This localizes the implementation difference before convolution or recurrence; it does not replace the seed-42 full-decoder test with input normalization and recurrent state.

| Packed projection slice | Original native fused bias: PCC / max absolute error | Native no-bias matmul, then ordinary bias add: PCC / max error | Matching DRAM-sharded fused bias: PCC / max error |
|---|---|---|---|
| QKV `[0,8192)` | `0.096165493 / 15.5607681` | `0.999999977 / 0.000125706` | `1.0 / 0` |
| Z `[8192,12288)` | `0.999568292 / 0.0226440` | `0.999999976 / 0.000108808` | `1.0 / 0` |
| A + dt_bias `[12288,12320)` | `-0.033425473 / 15.5629272` | `0.999999979 / 0.00276566` | `1.0 / 0` |
| B `[12320,12352)` | `1.0 / 0` | `0.999999978 / 0.000048019` | `1.0 / 0` |

The bias-free output projection has PCC `1.0` and max error `0` in both probe artifacts. Zero max error establishes exact equality for the captured slices; this is stronger than PCC alone. The no-fused-bias branch **does add the original bias afterward** (`tests/projection_probe.py:39-40`).

## Causal chain through the selected native path

1. **Bias preparation does not follow the changed weights.** `tt/fused_decoder.py:62-65` uploads tensors into `DRAM_MEMORY_CONFIG`, which is interleaved. Lines 90-94 synthesize an FP32 bias `[1,1,12352]`, zero except for checkpoint `linear_attn.dt_bias` in columns `[12288,12320)`. Lines 128-145 create DRAM-width-sharded copies of the packed weight and output weight, but no sharded bias. Lines 177-179 pass the original interleaved bias unchanged into the explicit DRAM-sharded program.

2. **Canonicalization preserves the operands and fuses that bias.** The call has no transposes, BF16 A/B, full-width one-row FP32 bias, explicit program config, FP32 output, and L1-width-sharded output memory. `ttnn/cpp/ttnn/operations/matmul/matmul.cpp:139-222` checks bias shape and config when deciding postprocessing. For padded bias height 32 and this explicit config, lines 193-198 leave `post_process_bias=false`. Lines 341-350 pass the same bias to the primitive. There is no conversion to the weight's shard layout.

3. **This selects the single-reader bank-based implementation.** The candidate sets workers per bank to 1 and does not enable `use_multicore_pipeline`; its default is false (`device/config/matmul_program_config_types.hpp:82-89`). The alternative multi-core pipeline also requires workers >= 2 (`device/utilities/matmul_utilities.cpp:470-488`). `device/matmul_device_operation.cpp:2765-2766` selects `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`. Its implementation, `device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1843-1910`, takes the single-reader branch. It binds `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp`, `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`, and `bmm_large_block_zm_fused_bias_activation_metal2.cpp` (factory lines 916, 996, 1101).

4. **The reader uses the wrong coordinate system for this bias.** `device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:197-223` reads bias through `BankView`, using the weight reader's `dram_bank_id`, starting at bank-local byte offset zero and advancing contiguously. `tt_metal/hw/inc/api/tensor/noc_traits.h:296-325` implements `BankView` as allocator-bank base plus byte offset; it does not translate global tensor page IDs. Interleaved bias pages instead round-robin across banks (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:559-574`; `tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:20-39`). Matching dtype and logical shape cannot reconcile those mappings.

5. **Validation misses the contract.** Generic bias validation checks tiles and shapes (`device/matmul_device_operation.cpp:426-510`). The DRAM-specific validator receives only `has_bias`, not the bias tensor (`:1663-1670`, call at `:2895-2905`). The factory checks bias device/storage at lines 1757-1766, but not width-sharded DRAM layout, bank grid, or shard width. The mismatch therefore reaches the reader without rejection or fallback. This is the native API's enabling validation gap, rather than evidence that `BankView` should secretly implement interleaved reads.

6. **Both model inputs and recurrent controls are corrupted upstream.** QKV is cast to BF16 and fed into convolution (`tt/fused_decoder.py:214-230`), then saved in conv history (`:231`). A's intended dt_bias enters softplus and controls decay `g` (`:242-243`); q/k/v/g/beta feed the recurrent update and overwrite the FP32 state (`:245-269`). Corrupted QKV and absent/misplaced dt_bias can therefore produce a large, deterministic decoder error. Its exact final PCC is an observed consequence, not a number derivable from source alone.

The old auto path already supports native fused bias. `doc/fused_decoder/profile_initial/tracy/layer_0/ops.csv:183` records the packed decode projection as BF16 × BF16 → FP32 interleaved DRAM with a `MatmulMultiCoreReuseMultiCast1DProgramConfig`, grid 11×10, K block 2, per-core M/N 1/4, subblock 1×4, 99 cores. The ordinary branch of its bias reader uses tensor `page_id` (`device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:628-645`). Thus the meaningful intervention is the changed bias-read convention, not the mere presence of native bias fusion. The archived profile predates the final grouped snapshot and is evidence for that earlier auto path, not a fresh trace of the failing run.

Sibling preparation paths confirm the intended contract: `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py:88-92,131-144` prepares one logical bias row in WIDTH_SHARDED DRAM, with padded shard height 32 and the same bank-column width as the weight. The saved `--shard-bias` probe does exactly that while keeping the candidate model, input, weight layout, program config, compute config, and output dtype fixed.

## Concrete dimensions, padding, and buffer ledger

For the eight-bank Blackhole geometry, with standard 32×32 tiles:

| Quantity | Packed linear attention | Attention output projection |
|---|---|---|
| Input logical shape/dtype | `[1,1,2048]` BF16 | `[1,1,4096]` BF16 |
| Weight logical shape/dtype | `[2048,12352]` BF16 | `[4096,2048]` BF16 |
| Output logical shape/dtype | `[1,1,12352]` FP32 | `[1,1,2048]` BF16 |
| K / N tiles | 64 / 386 | 128 / 64 |
| Weight DRAM shard, elements | `[2048,1568]` | `[4096,256]` |
| Input L1 shard on eight storage cores | `[32,256]` | `[32,512]` |
| Output L1 shard on eight storage cores | `[32,1568]` | `[32,256]` |
| K block width / iterations | 2 / 32 | 2 / 64 |
| Blocks per input storage core | 4 | 8 |
| Reader / compute N tiles per worker | 49 / 52 | 8 / 8 |
| Output subblock / subblocks across N | 1×4 / 13 | 1×4 / 2 |
| Last subblock live columns, tiles | 1 | 4 |

Source calculations: model lines 129-144; factory lines 154-253 and 297-311. BF16 weights, HiFi4, approximation disabled, FP32 destination accumulation enabled, and packer L1 accumulation disabled are unchanged (`tt/fused_decoder.py:53-59`). **Only the packed projection requests FP32 output. The output projection requests BF16 in both versions** (`:174`, `:214`, `:275`).

There are two separate padding extents. Eight 49-tile weight shards provide 392 tiles for logical width 386: six right-padding tiles of tensor storage capacity. The factory additionally widens each worker's compute width from 49 to 52 to use 1×4 subblocks. It narrows the final matmul subblock to one live column (`bmm_large_block_zm_fused_bias_activation_metal2.cpp:425-472`), clamps the three padded bias indices to tile zero (`:627-655`), and writes only 49 columns per worker (`factory:811-832`; reader/writer `:233-295`). Public logical width remains 12352. These padded lanes are not architectural head columns.

Per-worker producer/consumer ledger for block 2:

| Buffer | Packed projection | Output projection | Movement contract |
|---|---|---|---|
| A | Capacity 4 BF16 tiles | Capacity 4 BF16 tiles | Produce/consume 2 each K block; total 64 / 128 |
| B | Capacity 294 BF16 tiles | Capacity 48 BF16 tiles | Produce/consume 98 / 16 each K block; total 3136 / 1024 |
| Partials | 52 FP32 tiles | 8 FP32 tiles | Subblocks of 4; reload across K blocks |
| Output | 52 FP32 tiles, aliases partial storage | 8 BF16 tiles, separate from FP32 partials | Compute publishes 52 / 8; writer waits and pops the same count |
| Bias | Capacity/published count 52 FP32 tiles; physical load 49 | Absent | Live accesses cover the loaded 49; three padded indices are clamped |
| Tensor output shard | 49 FP32 tiles | 8 BF16 tiles | Writer discards compute padding; interleaving respects logical width |

Allocations and bindings: factory lines 282-311, 468-562, 1023-1040, 1142-1163. B's reader initially reserves two blocks, publishes a completed block after its transaction barrier, reserves the next two-block window, and explicitly fences/publishes the final block (`reader:130-194`). Compute waits and consumes exactly the same per-block counts (`compute:418-419`, `:589-590`). With the concrete 32/64 block iterations there is no ragged final K block. No contradictory count/lifetime transition was found for these configurations.

**Predicted misplaced columns, not a measured coordinate capture:** desired dt_bias is bias tile 384. Interleaved storage places it at bank 0, local tile 48 (`384 % 8 = 0`, `384 // 8 = 48`). Bank-ordered readers emit 49 columns each (`matmul_utilities.cpp:413-417`, factory `:811-832`), so the bank-based reader treats that page as output tile 48: columns **`[1536,1568)` inside QKV**. Desired output tile 384 belongs to worker/bank 7 at local tile 41; that address instead reads interleaved bias page `7 + 8*41 = 335`, which is zero. This predicts both a large QKV bias injection and loss of A's bias. The probe's aggregate metrics support the scale/slices, but contain no coordinate samples proving the exact destination. The final bank-local page on banks 2-7 corresponds to nonexistent logical bias pages 386-391; its contents must not be inferred from the saved maxima.

## Why the other suspects are secondary

| Candidate | Source contradiction / controlled evidence | Assessment and smallest intervention |
|---|---|---|
| Bias left interleaved | Direct caller/reader contradiction; standalone add nearly restores every slice; matching sharded fused bias gives zero error | Leading diagnosis. Prepare a separate matching sharded decode bias, or keep bias outside this native matmul. |
| FP32 accumulation/reload/output | Factory uses Float32 partials and a separate `UnpackToDest` reload alias when fused bias also reads partials through SrcA (`factory:253-267`, `:1084-1094`; `compute:119-138`). Final bias phase reconfigures formats and packs the requested output (`compute:598-608`, `:689-695`). Correctly sharded-bias control is exact. | No independent defect demonstrated. Do not reduce precision or relax PCC to address this failure. |
| Weight/input/output sharding or live-column padding | Concrete counts above are coherent. Both packed no-bias-plus-add and matching-bias controls retain the same layouts and padding and pass. Output projection is exact for its independent input. | Not needed to explain the evidence. Actual attention-output inputs still merit end-to-end coverage. |
| Trace replay/state restore | Eager failure precedes capture and has the same PCC as blocking replay; three traced replay outputs pass `torch.equal`. Eager and traced tensors were not directly compared. | Trace-specific corruption does not explain the eager failure. |
| Existing recurrence/MoE changes | Snapshot-to-candidate diff adds only the projection layout branch; projection probe reproduces corruption before either subsystem. | Downstream amplification is expected; no independent module bug established. |

The earlier block-8 resource failure is real but separate. `logs/dram_reader1_0_adapted.log:22-49` fails compiling the packed projection, with static buffers growing to 2,979,776 bytes versus L1 maximum 1,572,864 bytes. For the stated block-8 geometry, the triple-buffered B allocation alone is `3 * 49 * 8 * 2048 = 2,408,448` bytes, already over that limit; block 2 reduces it to `602,112` bytes. A, shared FP32 output/partials, and bias add another 458,752 bytes at block 8 (raw DFB total 2,867,200 bytes, before placement base/alignment/other reservations). This explains why smaller blocking permits execution, without fixing the bias mapping. The older failed candidate's exact source snapshot was not supplied; the byte calculation is conditional on the reported block-8 setting, not a reconstructed complete allocator map.

`tests/run_decoder.py:103-145` also supplies meaningful completion boundaries: each warmed prefix starts from cleared state, synchronizes, completes before readback, and is snapshotted for decode. Eager decode is read back; state is restored and synchronized before capture; each replay restores the same prefix state and uses `blocking=True` plus readback. This conclusion does not rely on the Python test body merely reaching its end.

## Smallest corrective boundary and discriminating follow-ups

No correction was applied here. The most direct model intervention is to prepare a **separate** FP32 WIDTH_SHARDED DRAM bias for `attention_packed`, using the converted weight's bank grid, ROW_MAJOR orientation, shard width 1568, and shard height 32. Keep logical shape `[1,1,12352]` and the original interleaved bias for auto-config/prefill. Bind the prepared bias only in the explicit DRAM-sharded branch. An alternative is native bias-free matmul followed by ordinary add; the saved probe already shows its small numeric difference from native fusion.

The native API should also own an explicit contract: either validate compatible DRAM bias shards and reject unsupported layouts before dispatch, or deliberately support/convert/postprocess other bias layouts. At present it silently accepts input its reader cannot interpret. This is an additional API robustness correction, not a reason to rewrite matmul arithmetic.

Recommended discriminators, in priority order; these are proposed device follow-ups, not runs performed in AutoDebug:

1. **Full decoder with only prepared bias corrected.** Preserve both new projection configs, BF16 weights, HiFi4, FP32 accumulation, and output dtypes. Rerun the recorded layer-0 length-256/context-512 `run_decoder --compare-functional` case with `QWEN_DECODER=fused`. Compare eager/traced output, functional contribution, and conv/recurrent states from the same prefix. Require the existing 0.995 gate and deterministic replay. This closes the remaining gap between the exact isolated-projection control and the final decoder PCC; no saved full-decoder correction run was found.
2. **Coordinate-resolved zero-input bias test.** Use the packed shape, block 2, FP32 bias/output, same shards, and zero A. Put exactly representable markers in bias tile 384 and inspect all logical output columns. Compare interleaved fused bias, matching-sharded fused bias, and no-bias-plus-add. The bad branch predicts markers at `[1536,1568)` and none at the intended A slice; the two correct branches must place markers only at `[12288,12320)`. Log indices and values, not just PCC/maxima. This isolates addressing from reduction, convolution, and recurrence.
3. **Separate the two changed projections.** During setup, choose old/old, new-packed/old-output, old-packed/new-output, and new/new while cloning identical prefix state. Include the matching-bias packed variant. Capture projection outputs before their consumers. The diagnosis predicts corruption follows the original interleaved-bias packed branch; it does not predict failure from the bias-free output projection alone. This is useful if the first full-decoder correction does not recover.
4. **Target the native contract.** A regression test should reject the original incompatible interleaved bias or execute a documented fallback, and verify positive results with matching DRAM bias shards. Use sparse marked bias at N=12352 and an aligned nearby N=12288/12544, BF16 A/B, FP32 output/bias, and per-M=1/block-2/readers-1. Compare every live column; crop storage padding. Cross output BF16/FP32 only to check a proposed generic fix, without changing the primary reproducer's precision.
5. **Boundary scope after recovery.** The branch tests `x.shape[-2] == 1`, not `decode=True`, so also retest single-token prefill and continuation at 1/31/32/33/255/256/257. Layer 3/full attention has no packed bias and uses a different width (9216), making it a useful negative control. No claim here says it currently fails. Longer recurrent/context checks should follow once this local contract is corrected, using provenance for the corrected candidate.

## Other potential issues and limits

- `tests/ttnn/unit_tests/operations/matmul/test_matmul_fp32_partials_reload.py:175-186` uploads interleaved FP32 bias and passes it directly to the device primitive; its DRAM-sharded case at lines 213-250 therefore repeats the same incompatible preparation. Treat this as a test-contract issue requiring audit, not proof that interleaved fused bias is supported. No result for this test was executed or established here. Existing nightly bias tests use the matching DRAM layout.
- Final fused bias addition consumes partials through SrcA and can introduce existing TF32 rounding; the reload test documents this at lines 167-172. This can explain small differences between fused addition and a separate FP32 add, but not the misplaced roughly 15.56-sized contribution. The matching-shard probe retains the fused math and has zero error against its auto reference.
- Existing long-context artifacts predate this candidate. They do not validate the new projection preparation or its device-memory footprint. Setup also retains original and sharded weight copies, a separate memory-capacity consideration; no capacity failure from those copies was demonstrated for this one-layer run.
- The sidecars report package version `0.1.dev1+ga464944` while checkout HEAD is `4c643884…`; `a464944` is not a resolvable commit in this checkout. They record Python hashes but not native shared-library/kernel hashes. The native source contract and exact Python-controlled probe strongly agree, but byte-for-byte identity of every compiled native component with the inspected tree cannot be independently certified from these artifacts.
- Review pass: checked every headline link against the wrapper, validator, program factory, direct-bank accessor, reader, compute bias phase, writer, and recorded controls. Correctly sharded fused bias preserves FP32 math and the same padded geometry while removing the error, so those shared mechanisms remain secondary. Exact misplaced-column coordinates and restored full-decoder PCC remain clearly labeled predictions/follow-ups rather than measured facts.
