# Exact gated-normalization fusion plan and first candidate

This is source/CPU analysis and an unmeasured model-local candidate. The parent owns hardware execution and profiling. The historical ~1.285ms and current ~2.45ms decode figures are context supplied by the parent, not measurements made here. No saved source or production file was edited.

The first candidate fuses only the arithmetic after the existing native FP32 sum. It retains the proven BF16-domain SiLU table lookup. This isolates arithmetic and rounding equivalence before attempting reduction or lookup/data-layout fusion.

## Required arithmetic contract

For each of 32 heads and its 128 coordinates, the current decoder performs:

```text
core = BF16(recurrent_output)
wide = FP32(core)
squares = RN32(wide * wide)
sum = current accurate FP32 TTNN reduction over128 coordinates
variance = RN32(sum * 2^-7)
shifted = RN32(variance + FP32(1e-6))
inverse = current nonapproximate FP32 TTNN rsqrt(shifted)
normalized = BF16(RN32(wide * inverse))
weighted = FP32(BF16(RN32(FP32(normalized) * FP32(gamma))))
gate = full_FP32_SiLU_table[raw_BF16_bits(projection_z)]
output = BF16(RN32(weighted * gate))
```

The core, normalized, weighted and output BF16 roundings must remain explicit. A fused weighted/gate product cannot move gamma through the normalized rounding or omit the weighted rounding. The table contains all65,536 BF16 codes, generated during setup using Torch FP32 SiLU; its values remain FP32. Native SiLU is not an interchangeable substitute: position3271 has one final BF16 difference despite matching normalized and weighted BF16 stages.

The frozen decoder source is [v6_inline_router_silu_chunk64_corrected_fma_explicit_norm.py.snapshot](v6_inline_router_silu_chunk64_corrected_fma_explicit_norm.py.snapshot), SHA256 `8391717476ae224ee9b82c203d7dec21ceab69632c5edd0abd89bf8cc4daeb03`. Its decode normalization occupies lines392–408, followed by head sharding, concat and the ordered output projection. The prefill path is distinct and is not changed by this candidate.

## Why native RMSNorm is not the candidate

The saved same-operand results are direct counterevidence:

| Position | Native BF16 normalized differences | Native FP32→BF16 normalized differences | Explicit normalized differences | Explicit gated differences with native SiLU |
|---:|---:|---:|---:|---:|
|128|405|401|0|1|
|129|405|398|0|0|
|130|382|383|0|0|
|137|418|414|0|0|
|181|370|366|0|0|
|3271|363|360|0|1|

Sources: [v5_gated_norm_same_operand_device.json](v5_gated_norm_same_operand_device.json), [v6_gated_norm_3271_device.json](v6_gated_norm_3271_device.json). At3271, explicit inverse differs from Torch in16 FP32 values and normalized FP32 in1,520 values, yet the normalized BF16 and weighted BF16 stages are exact. Preserve the already tested explicit TTNN arithmetic rather than silently changing the normalization target to a different CPU algorithm.

Local sources explain why `rms_norm(..., fp32_dest_acc_en=True)` alone does not preserve these boundaries:

- `ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp` delegates to layernorm RMS mode.
- `.../layernorm/device/layernorm_op_multi_core.cpp` allocates `EPS` as `Float16_b`, including for FP32 carriers. `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp` writes `scalar >> 16`; FP32 `1e-6` becomes `9.98377799987793e-7`.
- `.../layernorm/device/kernels/compute/layernorm.cpp` uses `BinaryFpu::Add` and `BinaryFpu::Mul` in its normalization chain. That is not the explicit SFPU sequence.

An additional CPU control using truncated epsilon still differs from the saved native-FP32 normalized BF16 in179/199/198/195/220/198 values across the six positions. Epsilon truncation is therefore a concrete source difference, not a complete numerical diagnosis of every native mismatch.

## Prepared post-sum candidate

[tests/v6_gated_norm_tail_fusion_probe.py](../../../tests/v6_gated_norm_tail_fusion_probe.py), SHA256 `47d50c370b8038bc1b10f24f86c5e58f61222e21b6cc7a1bb22d6318055961bd`, provides `GatedNormTail` and a hardware-opt-in component harness. [v6_gated_norm_tail_fusion_cpu_final.json](v6_gated_norm_tail_fusion_cpu_final.json) passes six saved positions plus eight seeded random BF16 operand cases. It records source/kernel/table/gamma hashes; source and compute snapshots accompany the report. No TTNN import occurs in CPU-only mode.

The adapter accepts FP32 `wide [1,32,128]`, native `sum [1,32,1]`, and FP32 lookup `gate [1,32,128]`. Gamma is expanded and uploaded once at setup. A single-core `SOURCE_CODE` compute kernel processes four width tiles; all FP32 inputs use direct-to-DST unpack. It retains native `rsqrt_tile<RsqrtMode::Default,true>` with approximation disabled and native `sfpu_mul_bcast_col<true>` for normalization. Separate DST store/load boundaries prevent the variance scale and epsilon addition from becoming FMA. Integer round-to-nearest-even BF16 masking follows the current typecast implementation. FP32 debug outputs expose inverse, normalized and weighted values before final packing.

The device harness computes the unchanged squares/sum and SiLU lookup, runs the existing explicit tail, then compares all fused intermediates and final BF16 output on the same operands. `GatedNormTail(debug=False)` omits debug buffers and stores; this is the version to profile after equivalence is established. The first adapter uses three small runtime placement copies to a single L1 core, plus a setup-only gamma allocation. Those costs and any output conversion required by the downstream consumer must be included in traces. It is possible this adapter is slower; fewer dispatches are not proof of benefit.

```bash
# CPU/source preparation only; use a new path for reruns.
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_gated_norm_tail_fusion_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_gated_norm_tail_fusion_cpu_final.json
# Parent-owned device component execution.
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_gated_norm_tail_fusion_probe --run-device --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_gated_norm_tail_fusion_device.json
```

## Further fusion candidates, contingent on evidence

1. **Remove placement-only copies.** Add a model-local dataflow reader for the existing interleaved L1/DRAM wide/sum/gate tensors and write the already-required consumer layout. Preserve tile ordering and direct FP32 unpack. Compare complete adapter latency before and after this change.
2. **Fuse table lookup.** A BRISC reader can read raw BF16 Z codes, index an immutable local32-bit table, and construct the gate tiles consumed by the arithmetic kernel. The table is256KiB; first try a single setup-resident copy, then only if measured worthwhile consider replication across a small number of cores. This is regular on-device table lookup, not a host or fixture fallback. A reader can interpret the existing BF16 tile face layout directly, avoiding bitcast→untilize→gather→tilize dispatches. For tiled Z `[1,1,4096]`, logical element `i` is in tile `i//32`, row0, column `i%32`; row0 byte offset is `2*(c if c<16 else256+c-16)`. Validate all65,536 codes including signed zeros/NaN classifications before replacing the existing lookup. Do not claim that256KiB replication is free: record per-core L1 occupancy and full-decoder capacity.
3. **Fuse the unchanged reduction.** Use `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` with `ReduceFp32Mode::Accurate`, retaining its four-tile width reduction and sum ordering; its post-reduce callback may apply the scale/epsilon/rsqrt tail. This avoids inventing a new reduction order. The native generic reduction defaults to accurate FP32 SFPU, unlike the lossy FPU route. Compare saved sums bitwise before relying on final BF16 agreement.
4. **Fuse head output layout.** A writer could emit the flattened4096-vector in the ordered output projection's required layout, avoiding concat/reshard conversions. This crosses a consumer boundary and must coordinate with its owner; do not optimize only this component while adding a conversion to the projection.

Relevant exact source hashes at preparation:

```text
619765014a46f1b23dc1fb73f230866fb0d218fd927623cc279894d9a10257ec  layernorm/device/layernorm_op_multi_core.cpp
69c55a172f3af54dd3efce3fd51aa3de8f724a33d545a90d6b754a517e153936  tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h
d0e11859398f958e0b0ca66a2028f2eb18759c79dc18acb4c8aad073e24cdabc  blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h
```

Keep any candidate only after same-operand checks, unchanged decoder stress/watcher gates, and traced warmed decode latency beat the best correct baseline. This memo does not claim a speedup, device compilation success, or stage completion.
