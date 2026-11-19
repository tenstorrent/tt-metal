# Exact decay LUT plateau compression

CPU/source-only; runtime F and device untouched. Reproduce with:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/gate_lut_plateau_cpu.py
```

`gate_lut_plateau_cpu.json` proves equality of **all2,097,152 FP32 bit patterns** (32heads×65536 BF16 codes), including infinities and every NaN payload, for the pinned layer0 checkpoint and the same CPU setup expressions as F. No approximate dtype or truncated domain is used.

|Table|Shared magnitude-code lo|hi|Finite interval length/sign|With127 NaN entries/sign, minimal bytes|Power-of-two sign stride|Padded bytes|
|---|---:|---:|---:|---:|---:|---:|
|decay|13056|17839|4784|1,257,216|8192|2,097,152|
|g|13056|32640|19585|5,046,272|32768|8,388,608|

Original table size is8,388,608bytes. Decay's magnitude endpoints represent2.9802322387695312e-8 and5600.0. Per-sign plateau upper bounds are17839 positive /16884 negative; common bounds simplify the runtime map. g continues changing through positive infinity and is not worth the same padded compression. Preserve prefill g/decay tables initially.

For raw unsigned16 code r, magnitude m=r&32767, shared finite interval length L=4784:

```
finite = clamp(m,13056,17839)-13056
nan_increment = max(m,32640)-32640
index = finite + nan_increment + ((r&32768)>>2)
```

For m<=32640, nan_increment is0 and the clamped finite endpoint reproduces the exact plateau. For m>=32641, finite=L-1 and the increment1..127 maps to the127 original NaN entries immediately after each finite window. Sign stride8192 keeps index<16384 and uses only a bit mask/shift. Both signed-zero inputs map to their own sign's original endpoint. Unused per-sign padding is never addressed.

## Candidate patch and native wiring

`compact_decay_plateau.py.patch` is the independently applicable setup/decode patch. The sibling snapshot/provenance capture its base; parent continues changing F, so use the patch rather than overwriting F with the snapshot. Setup computes lo/hi from bitwise transitions over both signs and32 heads, preserves the original full tables, and builds the compact table only if L+127<=8192. Otherwise it retains the old decode path. Setup includes exhaustive bitwise reconstruction assertion for the actual supplied layer weights.

Decode introduces three kernels before the existing index layout adapter:

1. UINT16 raw-code tensor→UINT32 typecast.
2. Integer add with two fused input chains: finite=[BITWISE_AND32767,MAXIMUMlo,MINIMUMhi,SUB_UNARY_SFPUlo], nan=[BITWISE_AND32767,MAXIMUM32640,SUB_UNARY_SFPU32640].
3. Integer add of that result and raw codes with RHS chain=[BITWISE_AND32768,RIGHT_SHIFT2].

Then the existing transpose/reshape→RM indices and native FP32 RM gather consume `[32,16384]` compact values. This keeps the existing32-core row ownership while reducing per-core table input CB/read from256KiB to64KiB (aggregate8MiB→2MiB). Index dtype is now UINT32, accepted by gather. Prefill uses the original UINT16 mapping and full tables. Finite/NaN branch selection is arithmetic on indices, not floating-point value arithmetic.

Source contracts:

- Binary Python keywords are `input_tensor_a_activations` / `input_tensor_b_activations`, `dtype`, and `fast_and_approximate_mode`; see `eltwise/binary/binary_nanobind.cpp` fast-approx wrappers.
- `binary_ng_program_factory.cpp:1039–1104` retains independent input activation lists and emits them with each operand dtype.
- `binary_ng_utils.cpp:439–446` chooses `add_int_tile<DataFormat::UInt32>` for UINT32 add.
- `unary/common/unary_op_utils.cpp:314–345,428–446,591–624` selects integer bitwise/shift, UINT32 min/max and integer subtraction kernels. All subtraction operands here are nonnegative with bounded results; no unsigned underflow or signed-wrap trick is required.
- `unary/device/unary_device_operation.cpp:37–86` lists these operations as UINT32-supported. CLAMP_TSS itself is not used because its integer support is INT32-only; explicit MAXIMUM/MINIMUM avoids that contract mismatch.

A device all-code index-map check should compare the generated UINT32 index tensor to the CPU expression before full decoder benchmarking. The CPU proof establishes the mapping and table, not device activation-chain behavior or performance. Three extra small kernels and widened index layout conversion can offset part or all of the6MiB table-read saving; measure the complete traced adapter and unchanged endurance gates.

## v2 API correction after parent device probe

`compact_decay_index.json` remains intact: UINT16 upload and UINT32 cast passed; the first ADD was rejected before kernel execution because accurate-mode=false is only accepted with BF16 ADD output (`binary_ng_device_operation.cpp:24–49`). The v2 probe, candidate patch and snapshot omit that flag on both UINT32 adds. No exactness threshold or mapping changed.

Omitting the flag retains integer SFPU arithmetic: `is_binary_sfpu_op` in the same source explicitly selects SFPU whenever matching operands are UINT32, irrespective of the flag. `binary_ng_program_factory.cpp:1103–1104` passes each UINT32 operand dtype into activation generation; `unary_op_utils.cpp:get_block_defines` forwards that dtype unchanged for every chain element. MAXIMUM/MINIMUM select `unary_max_uint32_tile` / `unary_min_uint32_tile` (lines591–624), not floating point. SUB_UNARY_SFPU on UINT32 intentionally uses `sub_unary_tile_int32` (lines438–446; native TODO for a separate uint32 API). Here both operands and all differences are nonnegative and below32768, so signed and unsigned subtraction agree exactly. BITWISE_AND/RIGHT_SHIFT use the UInt32-parameterized kernels; the enclosing add uses `add_int_tile<UInt32>`. All65536-code device checks remain required.

Setup now uses lo=hi=0 when the transition set is empty. An independent CPU case with64 distinct head/sign constants and all original127 NaN payloads/sign reconstructs all2,097,152 FP32 bit patterns exactly. This avoids min/max on an empty tensor without changing lookup behavior.
