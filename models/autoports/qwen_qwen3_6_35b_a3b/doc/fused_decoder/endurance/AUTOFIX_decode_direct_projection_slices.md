# Decode uses direct projection slices

Source/CPU-only candidate, no F edits or hardware execution. Patch `decode_direct_projection_slices.patch`; source base SHA256 `f098d3bfd185b9acda78712c22ab448eab385abc9a3232a7345650dfdd384354`. It factors `projection_a`, `projection_b`, and `projection_z` before the existing LUT index/lookup logic; those downstream internals are unchanged. If the root applies a LUT patch first, the only dependent hunk is replacing its original packed A slice with `projection_a`, and B with `projection_b` if still present.

## Index equivalence

The exact ordered state projection is `[K2048,V4096,A32,B32]` (6208 logical channels). The native output-only projection is `[Q2048,Z4096]` (6144 channels). The existing packed concat is `[Q,K,V,Z,A,B]` (12352 channels).

| Consumer | Old packed interval | Direct source interval |
|---|---:|---|
| convolution QKV | 0:8192 | concat(output0:2048,state0:6144) |
| A / decay-table index | 12288:12320 | state6144:6176 |
| B / beta | 12320:12352 | state6176:6208 |
| Z / output SiLU | 8192:12288 | output2048:6144 |

This is byte reordering/selection only. No multiplication, rounding, dtype, conv-tap, cache-boundary or attention equation changes. All offsets and widths are32-aligned. Both projection producers still execute exactly as before with the same ordered/native split.

Decode changes the full packed concat to a narrower QKV concat and reads A/B/Z directly. Prefill still builds the full packed concat used by native gated norm with `gate_column_offset=8192`; the same A/B/Z slices are named earlier and reused. Public prefill tails continue to take their existing decode path. No new length or alignment restriction is introduced, and rows are never mixed.

## Exactly changed structural calls

For decode, the old projection unpack sequence contains8 slice calls plus one12352-channel concat: initial Q/KV/Z/AB slices (4), then QKV/A/B/Z slices (4). The new sequence contains5 slices plus one8192-channel concat: Q/KV (2), direct A/B/Z (3). Net removal is3 slice calls, with the remaining concat narrowed by4160 channels. In particular the intermediate AB slice, full-packed QKV re-slice and duplicate Z selection disappear. Actual slice lowering may contain multiple kernels; use the current trace profile to quantify physical op savings rather than claiming3 kernels removed.

At single-token BF16 TILE geometry with physical height32, concat output footprint decreases from790528 bytes to524288 bytes, saving266240 bytes (260KiB) per materialized concat output. This is a buffer-size difference, not measured DRAM traffic or latency.

Direct A/B slices inherit the current state projection's L1 memory placement. They previously inherited the combined packed concat's placement. QKV still starts with the same output-Q operand; Z comes directly from the same native output projection. Slice/concat and subsequent bitcast/LUT index transformations preserve numerical codes, but changed A/B placement can change kernel dispatch and performance. The patch intentionally does not add copies solely to reproduce the old placement; root must validate legal lookup handling and complete traced execution.

## Validation and diagnostic compatibility

CPU index-mapping evidence is in `decode_direct_projection_slices_cpu.json`; synthetic uniquely labeled source channels compare all four consumers exactly at sequence lengths1,17,128. Candidate AST and patch applicability pass. This establishes mapping only, not TTNN layout correctness or device acceptance. Root should run short correctness, the three-stream gates (including sensitive routing), tail/nonaligned coverage as applicable, and paired traced timing before retention.

The frozen stress test does not name `projection_packed` or require a12352-channel concat, so its acceptance gates still exercise the fused runtime. Some optional intermediate diagnostics, notably `linear_frontend_stream_probe.py`, hook the12352-channel concat to recognize the reconstructed projection. They may need diagnostic-only reconstruction from captured `state_projected_bf16`/`output_projected_bf16`; this must never reintroduce the packed concat into measured runtime or provide an oracle fallback. Existing archived captures remain valid for their recorded sources. No tests are edited by this candidate.

## Native gather L1-index legality

The existing gate table and indices are both ROW_MAJOR at gather. Codegen gather rejects ROW_MAJOR (`gather/codegen/gather_codegen_supported.cpp:63`), so this path uses native RM gather. Its validation (`gather/device/gather_device_operation.cpp:98–148`) requires matching layouts, integer UINT16/UINT32 index dtype, allocated device buffers and valid shapes; it does not require matching buffer types or DRAM indices. The RM subalignment check concerns sharded rows, so interleaved L1 is legal.

The RM program factory (`gather/device/gather_program_factory.cpp:451`, and623 for the multi-core variant) constructs `TensorAccessorArgs` from the actual index buffer independently of the input/table buffer. `gather_reader_rm_single_row_single_core.cpp` builds that accessor and reads the index row through it; the address generator therefore carries L1 index placement while the separate table stream retains DRAM. The multi-core counterpart uses the same independent accessor pattern. No forced DRAM restoration is needed by this source contract.

Retain direct A/B L1 placement and measure it. This source finding is legality evidence, not a device validation result; current installed native behavior, LUT-compression composition and trace-memory lifetime remain subject to the root's short test and state-bit/PCC checks.
