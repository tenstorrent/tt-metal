# Projection chunk row windows: source-only design and bounded proof

Prepared2026-11-13. No target imports, native/model/helper edits, build or device run. The live-row helper/probe remains frozen after its separately documented oracle repair. This audit addresses item1 of `remaining_bottleneck_audit25.md` and does not claim a measured optimization.

## Recommendation

First implement a **GDN combine destination-row offset**, retaining the existing joint matmul chunks and exact correction arithmetic. This removes the final BF16 concat without a native matmul API change. Then add a bounded **2D matmul row-window** capability that preserves the winning per-piece programs and reads their activation tiles from the full input allocation. Attention writes each piece directly into the final output; GDN continues to use one chunk-sized FP32 joint intermediate and combines directly into its final BF16 allocation. Do not allocate a full-prefill FP32 joint intermediate just to remove concat.

The real current selected factory is Metal2: `device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`, using `reader_bmm_tile_layout_in0_sender_padding_metal2.cpp`, the in1 sender/receiver **combined readers/writers**, and `bmm_large_block_zm_fused_bias_activation_metal2.cpp`. Editing the legacy standalone `writer_bmm_tile_layout.cpp` would miss this path. Existing named runtime origins already carry the required offsets; the first full-cap implementation can leave these device kernels unchanged. Host API/validation/scheduling changes are necessary.

## Existing paths and arithmetic to retain

`tt/optimized_decoder.py:_linear_raw` around499–513 flattens batched input, chooses cap512 by default, slices each piece, recursively projects, and concatenates. Pieces<=32 use the selected DRAM-sharded decode matmul, while larger pieces use the 2D program. The final attention32-row tail at B32/S33 must retain that path; changing it to2D would be a distinct arithmetic/program comparison.

`tt/packed_correction_candidate.py:project_prefill` around267–305 pads only internally to32 rows, caps4096 in the accepted explicit policy, slices, uses `prefill_plan`, runs the joint primary/correction matmul, combines each piece, and concatenates4160-wide BF16 results. The native matmul output width is7168 (joint logical6784, padded for the existing weights). Preserve its selected LoFi/FP32 accumulation/output pack, weight quantization and correction add/RNE. A short tail inside an otherwise large joint prefill currently remains2D; only an entire call<=32 takes `_project_compact`.

Current explicit policies in `fusions20_fused_l{0,3}_b32_s33.policy.json`, combined with `multichip_decoder.py` defaults, imply:

| Family | Full piece | K | Matmul N | Grid | Per-core M/N tiles | H block | K block | Subblock H/W |
|---|---:|---:|---:|---|---|---:|---:|---|
| Attention QKVG |512rows|5120|3584|11×10|2/11|2|8|2/1|
| Joint GDN |4096rows|5120|7168|8×8|16/28|4|8|1/4|

These derive from the existing CPU plan formulas, not a retuned configuration. Current normalizers/flat input materializations and later head/cache/MLP paths are outside this proposal. TP4 rank-local projection weights and inter-layer1280-wide residual contract remain unchanged.

## Step A: smallest safe combine-only change

`PackedCorrection._combine` around319–390 already partitions a local piece into `piece_tile_rows*130` disjoint tasks. Its reader and compute must keep their **local** task ids: primary/correction columns and tile-row modulo depend on that local index. Add an optional final BF16 output owner and a tile-aligned destination row offset, with strict shape/dtype/memory/mesh/capacity checks. The writer source `_NARROW_GDN_COMBINE_WRITER` in `optimized_decoder.py:4391` has exact anchor `{.page_id=tile}`. In a new isolated writer variant, change only the destination page to `tile + destination_row_tiles*130`, supplied in a new runtime argument after address/start/count. Reader start/count and compute start/count stay unchanged.

Preallocate final `[1,padded_rows,4160]` once, issue the same matmul and combine programs per piece, then apply the existing final logical slice/reshape. Preserve the native FP32 joint scratch as a chunk, not a full output. All chunks collectively overwrite every final physical tile once; no whole-output clearing or per-piece clearing of sibling regions is needed. Warm every tail shape and rebound output before capture. This can be implemented in isolated new Python generic-op adapter/probe files without rebuilding TTNN and is the smallest discriminating experiment.

## Step B: bounded native row-window API

No current supported public API specifies a matmul row window. `linear(..., optional_output_tensor=full_output)` with a physically smaller input piece is rejected at `matmul_device_operation.cpp:229–240`: output width, compatible rows and total volume must match the computed piece output. This is an observed source contract, not a hardware rejection. `in0_column_offset` is an existing **K-column** window and cannot substitute for an M-row offset. A tensor's interleaved buffer address cannot be shifted by `row_offset*K*element_size`: TensorAccessor bank mapping requires the original allocation base plus global page ids. A shape-spoofing view would bypass required capacity/admission checks and is not proposed.

A narrow extension analogous to `in0_column_offset` is feasible. Suggested explicit `in0_row_offset`, `row_count`, `out_row_offset` fields, with absence preserving every current path. Initial admission:

- Explicit `MatmulMultiCoreReuseMultiCastProgramConfig`; flat outer volume1; BF16 tiled interleaved DRAM A; existing BFP8B DRAM weight layout retained; no transpose, bias, untilize, column-window or batched/reuse modes in this first capability.
- Positive row_count and tile-aligned input/output offsets; row_count tile-aligned internally. Checked64-bit arithmetic proves input `[offset,offset+count)` fits allocated padded M and output interval fits its padded M. Require32×32 tiles and width matching the same computed N. Public arbitrary logical lengths remain supported through existing/internal padding and final slicing, never a new aligned-input requirement.
- Required preallocated interleaved DRAM output. It may be a full attention output with nonzero destination origin, or the existing chunk-sized GDN FP32 intermediate with output origin0. Strict dtype/config/tile/mesh checks; prohibit aliasing A/B and output. Native admission validates these bounds on cache hits as well as misses.
- For the minimal origin-only kernel implementation, require that selected per-core scheduling has no padding reads beyond its admitted row interval; full caps above satisfy this. A general short-window extension needs the additional reader bound described below. Tail handling remains explicit and measured rather than silently changing its matmul program.

Concrete host anchors:

1. `matmul.hpp`, `matmul.cpp:466–495`, `matmul_nanobind.cpp:926–950` and `device/matmul_device_operation_types.hpp:41`: thread optional window attributes alongside existing column-window support. `create_matmul_attributes` around3493 copies optional fields and must retain them.
2. `MatmulDeviceOperation::validate_on_program_cache_miss` around2803 already creates adjusted local shape values for the K window before shared validation. For row windows, keep the real tensor/buffer specs, but substitute the effective **chunk** M in the local logical/padded shape variables used for work-distribution validation. Updating only the factory is insufficient: the 2D validator around958–1005 would calculate full-input core counts and reject the unchanged chunk config.
3. `validate_matmul_optional_tensors` around229 and `compute_output_specs`: add a row-window-specific output interval check; preserve the ordinary exact-volume rule outside this mode. `create_output_tensors` already returns supplied output owners. Do not allocate output before trace or infer a full output spec from the effective chunk and then silently accept mismatched owners.
4. Factory `create_program_artifacts` around1806–1855: compute `Mt`, `Mt_per_batch`, padding and all compute/CB geometry from effective row_count while preserving the actual `in0_tensor` and `out_tensor` bindings. Preserve `in0_row_tiles=Ktiles` and all weights/compute flags.
5. Factory exact input origin around990:
   `(in0_tensor_start_tile_id_stride * in0_idx) + in0_column_offset_tiles`
   becomes that value plus `in0_row_offset/32 * Ktiles`.
6. Factory `build_in1_sender_writer_args` around144 and `build_in1_receiver_writer_args` around180 both form
   `in1_idx*per_core_N + in0_idx*per_core_M*Ntiles`.
   Add `out_row_offset/32*Ntiles` in **both** branches. Existing strides stay Ntiles; batch is1. All physical output tiles outside the interval stay untouched.
7. ProgramSpec tensor parameters around1650 and runtime `tensor_args` around1675 must bind actual allocations, not fabricated subviews. Initial safe caching may include offsets/count in the operation attribute hash (current `compute_descriptor_program_hash` hashes attributes); prewarm each offset/signature. Excluding origins from a cache key requires an explicit runtime-update mechanism and separate proof, not an assumption. Changed owners with identical metadata must still rebind via normal tensor parameters.
8. `create_op_performance_model` around3372 currently counts full output M. Count executed row_count in this mode and report written interval/actual transferred bytes. Otherwise a full-output preallocation would falsely inflate FLOPs and roofline.

## Short-window reader issue: precise source limitation, not family rejection

The selected input reader uses `num_blocks_h_dim=per_core_M/out_block_h` for every sender and reads when `bh < num_blocks_h_dim-1 || h < last_block_h` (reader around265–300). It has no equivalent to the output receiver writer's `last_num_blocks_h_dim`. At GDN1056 rows, the existing plan is Mt33/per_coreM5/Hblock1: seven row senders issue35 tile rows; final sender addresses rows33/34 beyond the logical33-row interval. The native writer suppresses these padded outputs using its last-block count. This audit identifies the source schedule, not a new observed hardware failure or proof of a physical out-of-allocation fault in past runs.

An origin-only row-window must not start reading another logical chunk's rows or assume accessible padding past the final allocation. Initial full-cap windows satisfy exact row ownership (attention16tile rows/pm2, GDN128/pm16). For arbitrary partial windows, adapt this reader with a per-core valid-row count, gate **every** load by `bh*out_block_h+h < valid_rows`, and zero the unread CB tiles before multicast while preserving all producer/consumer counts. The weight/compute/writer schedule can remain unchanged. Extend only the Metal2 branch actually selected, guard scope, and record shared-kernel consumers if its runtime schema changes. Another safe initial option is to retain existing isolated tail projection and copy only that small result into the final output; the tail copy is then a remaining measured cost, not a hidden fallback to full concat. Neither limitation rejects the complete family.

## Evidence and potential gain

Use existing S33 numbers only for S33:

- Full-attention fusions20 table: input piece slices3181=32us,3187=31us,3197=2us; final concat3211=41us. Their rounded kernel-time sum is106us. Keeping the final32-row tail input slice leaves104us in the two full-piece slices plus concat as the observed target before replacement overhead. Concat's60us op-gap is not automatically removable work. These numbers bound an opportunity, not a predicted saving.
- GDN fusions20 S33 has one1056-row joint matmul1621=978us and one combine1626=87us, with no projection-piece concat. This row-window family cannot claim savings from removing nonexistent chunk copies in that profile. The87us combine arithmetic remains; changing its output destination does not remove it.
- No available S513 operation profile is used to invent a scaling factor. B32/S512 first temporal chunk has16384 logical rows:32 attention pieces or4 GDN pieces. Parent needs a fresh long-prefill trace/profile to measure slice/concat contributions and verify whole-layer gains.

Pure byte arithmetic at16384 rows/rank, if every input slice copy and final output concat copy disappears: input read+write335,544,320bytes for both families; attention output read+write234,881,024bytes (total544MiB); GDN corrected BF16 output read+write272,629,760bytes (total580MiB). These are avoided copy payloads, **not latency estimates**, and exclude material flatten/pad/tail copies, native activation reads, weight reads, FP32 intermediate/combine traffic and AG/RS. Replacing concat with full-volume slice_write would not realize this reduction. A full GDN FP32 joint allocation would add up to448MiB/rank at16384×7168 and is deliberately avoided; existing4096-row scratch is112MiB.

## CPU proof and next measurements

`projection_row_window25_cpu.py` uses stdlib and the extracted pure existing `prefill_plan` AST. It passes30 family/row cases including1/3/31/33/99/511/513/1056/4096/4128/5152/16384/16416. It verifies identical reader global page ids for admitted windows, unchanged K iteration order/configs, disjoint complete output ownership, GDN130-tile-row destination offsets, internal alignment, and records exact source hashes. It also exposes the short-window overread schedule rather than clipping it away in the proof. Result: `projection_row_window25.cpu.json`. This is an address model, not C++ or device execution.

Smallest hardware sequence if parent selects implementation: combine-only real-weight rows4096/8192/16384 with unchanged matmul configs; then native input windows on full512/4096 caps and direct attention outputs; then awkward tails with explicit safe handling. Require bitwise raw matmul/combined output parity when arithmetic is unchanged, >=.995 whole-layer output/contribution/cache gates, per-rank/user/piece-edge checks, sibling-output sentinel preservation, input/output/weight rebinding, changed-input queued traces with strict tracking positive control, and watcher separately. Preallocate all final/scratch owners and warm every signature before capture. Benchmark complete projection boundary and then whole B32/S513 with the current explicit policy. No code has been implemented by this audit.
