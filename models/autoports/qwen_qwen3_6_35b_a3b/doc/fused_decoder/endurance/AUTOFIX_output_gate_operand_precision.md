# Why the output-gate Z widening is semantically necessary

Source-only analysis and device-ready microprobe; no hardware execution or F edits by this investigator. Root reports that the complete HF output-gated-norm boundary repair passes all three seeds, while jointly folding gamma's round/widen and removing Z's explicit FP32 cast reintroduces seed154 position5991 failure (contribution about0.993994). The combined experiment alone cannot assign the cause to either changed operation; the following source mechanism and isolated probe distinguish them.

## BinaryNG has an operand-activation pack boundary

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp` creates circular buffer c_4 for RHS preprocessing whenever `input_tensor_b_activations` is nonempty. At lines1229–1239, its SFPU-path data format is **b_data_format**, derived from the actual tensor B dtype, not the output dtype or the other operand.

`device/kernels/compute/eltwise_utils_sfpu.hpp:19–49` loads operand B into destination registers, runs its activation chain, then `pack_tile`s the activated value into this intermediate circular buffer. `eltwise_binary_sfpu_no_bcast.cpp` subsequently reloads that packed RHS before binary multiplication. This is a real numerical boundary despite being one public TTNN operation.

Consequently:

- With BF16 Z, fused `multiply(weighted, z, input_tensor_b_activations=[SILU])` materializes **BF16 SiLU(Z)** before multiplication.
- With FP32 Z, the same call materializes **FP32 SiLU(Z)** before multiplication, matching HF's explicit `F.silu(gate.to(torch.float32))`.
- FP32 destination accumulation, FP32 unpack, a FP32 LHS, or choosing a FP32 output cannot recover bits already lost by packing c_4 as BF16.
- Adding a widening TYPECAST inside the RHS activation list is not sufficient by this source contract: c_4 still derives its format from the BF16 input tensor. A real FP32 input tensor is required unless a different existing native op explicitly exposes a wider operand intermediate.

`binary_ng_utils.cpp:621–644` additionally specializes activation defines using the operand dtype, so changing Z dtype may also choose a different unary implementation. The intermediate pack alone already demonstrates why removing the cast is not an algebraically neutral conversion cleanup.

## Gamma round/widen is different

Gamma rounding is a **post-activation** on the multiply, applied before packing the final output. Keeping that output FP32 while applying explicit FP32→BF16→FP32 in the post-activation chain can preserve the intended BF16 gamma-product boundary without an additional standalone cast pair. It does not force the independent SiLU RHS intermediate to FP32. Therefore gamma-chain folding and Z-cast elimination must be assessed separately.

This source conclusion is subject to the fixed installed native binary's behavior; prior binding/source differences are already recorded in native_fingerprint.json. The focused microprobe below verifies actual numerical consequences without modifying production or relaxing thresholds.

## Focused diagnostic

`tests/output_gate_precision_probe.py` reads actual captured position5991 core/Z, loads only the checkpoint output-norm gamma, and creates serial mesh0. It tests:

1. Native unweighted RMSNorm + gamma multiply with separate BF16/FP32 casts versus gamma round/widen post-chain.
2. For each resulting weighted operand: fused SiLU with BF16 Z; fused SiLU with FP32 Z; standalone FP32 SiLU followed by multiplication.
3. A clearly labeled canonical-weighted upload control that isolates the SiLU preprocessing boundary from native RMSNorm/gamma discrepancies.
4. Two CPU expectations on the same weighted operand: HF FP32 SiLU, and deliberately extra-rounded BF16 SiLU.

The report records BF16 code differences, relative L2/PCC, provenance, and explicit diagnostic-only/non-acceptance labeling. Device is closed in finally. No latency claim is made. Syntax compilation passed; root runs hardware when free:

```
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.output_gate_precision_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/output_gate_precision_device.json
```

An explicit Z cast is necessary precision work if this actual-binary test confirms the predicted intermediate. It is not an unnecessary conversion merely because both versions compute the same symbolic SiLU expression. Retaining it does not waive the final runtime/path audit; the audit should record the demonstrated semantic purpose.
