# AutoFix: fused conv/gates selects the wrong softplus algorithm

**Status: concrete source mismatch; isolated repair prepared, no hardware execution.** This can explain the g discrepancy, but it does not yet prove the cause of the final per-head output failure, which also has a convolution/control mismatch under independent investigation. Existing thresholds remain unchanged.

## Evidence and earliest differing operation

`../conv_gates_precise_direct_watcher_wide_modes.json`, B13/wave104/step5/rank0, records beta bitwise equal, g maxabs error `3.814697265625e-05`, convolution maxabs error `0.001953125`, state min-head PCC `.9999939203262329`, and output min-head PCC `.988250195980072`. The last value fails .995. A high global g PCC does not establish recurrence equivalence, so the g discrepancy is repaired rather than dismissed.

Production `tt/multichip_decoder.py:982`–`992` computes:

```text
raw a BF16 -> exact FP32 widening
sum = ttnn.add(a, dt_bias)                 # FP32 output
 g  = ttnn.multiply(sum, decay,
        input_tensor_a_activations=[SOFTPLUS(beta=1, threshold=20)])
```

The wide-mode candidate genuinely has 64 unpack entries and overrides CB10,11,12,13 to `UnpackToDestFp32` (`tests/multichip_conv_gates_precise_wide_modes.py:80`–`89`). These preserve full FP32 bias, decay, sum, and softplus values. CB9 is BF16 a, which is exactly representable in FP32. FP32 add precedes softplus, then FP32 multiply; the candidate's extra FP32 pack/reload boundary does not intentionally quantize to BF16 or Tf32. The operand order is the same. No source-level threshold or beta-constant mismatch was found: both emit the exact bits `0x3f800000`, `0x3f800000`, `0x41a00000` (unary code generation at `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:547`–`558`).

**However, FP32 DEST and `math_approx_mode=False` do not select the FP32 softplus algorithm.** The candidate compute descriptor omits `INP_FLOAT32`. On Blackhole, `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h` selects its constants, polynomial and loop at compile time:

- Lines181–184: vector stride differs by `INP_FLOAT32`.
- Lines194–202: FP32 degree8 residual coefficients versus BF16 cubic coefficients.
- Lines220–262: FP32 Cody–Waite degree7 exp / degree8 residual path versus `softplus_bf16_eval`.
- Lines405–435: the missing macro chooses the BF16-oriented x2 loop, even with FP32 destination registers.

Production's multiply carries a FLOAT32 LHS activation. `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:1103` passes that dtype to activation code generation; `binary_ng_utils.cpp:638`–`642` invokes `add_input_dtype_defines`, and `unary/common/unary_op_utils.cpp:1196` sets `INP_FLOAT32=1`. The candidate does not run that host-side generator, so it must provide the same define explicitly.

The source mismatch is quantitatively material at ordinary inputs. At `sum=0`, both exp branches produce exactly `u=1`. The BF16 residual coefficients yield softplus `0.69317626953125`, versus `ln(2)=0.6931471805599453`, a difference `2.9088971304713773e-05` before multiplying by decay. The host FP32 Horner comparison is recorded in `host_validation.json`. This is an analytic discriminator, not a claimed device replay of the failing fixture.

At finite values above threshold20, both paths return identity. The BF16 path optimizes away its predicate for unit beta and threshold>=16 because its residual is below half an FP32 ULP there. Thus the identified error is the polynomial/exp path below threshold, not a different public softplus threshold. `APPROXIMATION_MODE` is passed as a template parameter but does not switch these two algorithm branches.

## Minimal repair

The only helper change from the wide-mode version is this argument on the **compute** `ttnn.KernelDescriptor`:

```python
defines=[("INP_FLOAT32", "1")]
```

This checkout binds `defines` as a sequence of `(str,str)` pairs, not a dict. Host API validation caught and corrected the dict form before handoff. Reader/writer descriptors and all original kernel files remain unchanged. Existing compute config remains HiFi4, exact mode, FP32 DEST, with64 unpack overrides.

`ckernel_sfpu_sigmoid.h`, `ckernel_sfpu_silu.h`, and `ckernel_sfpu_exp.h` contain no `INP_FLOAT32` selection branches in this checkout. The explicit beta sigmoid `<false>` numerical branch and convolution SiLU's FP32-DST selection therefore stay unchanged. Softplus still calls its init after beta/convolution have used programmable constants, so adding the macro also loads the matching FP32 softplus constants before evaluation. There remains exactly one hardware-startup call in the unchanged compute kernel.

## New immutable entry points

Under `tests/`:

- `multichip_conv_gates_fp32_softplus.py`: repaired conv/gate helper using original `kernels/conv_gates_precise`.
- `multichip_conv_gates_fp32_softplus_candidate.py`: same decoder candidate inheritance and gates as the prior wide-mode candidate.
- `multichip_conv_gates_fp32_softplus_probe.py`: unchanged legacy direct probe, new helper routing.
- `multichip_conv_gates_fp32_softplus_matched_probe.py`: unchanged production-convolution matched probe, new helper routing.
- `multichip_fast_zero_conv_gates_fp32_softplus.py`: identical define repair with original fast-zero conv reader/kernel directory.
- `multichip_fast_zero_conv_gates_fp32_softplus_candidate.py`: current-production cumulative candidate using this repaired fast-zero conv helper and unchanged fast-zero recurrent helper.
- `multichip_fast_zero_conv_gates_fp32_softplus_probe.py` and `multichip_fast_zero_conv_gates_fp32_softplus_matched_probe.py`: corresponding unchanged direct/matched controls.

All original helpers, probes, candidates and kernels remain immutable. No job list was created or launched; parent owns bounded serial scheduling. The direct/matched modules accept their usual `--output`, `--batches`, `--waves`, `--steps`; candidates accept the prior stack arguments including `--no-gate-padding` only where their original wrapper supported it.

## Discriminating device outcomes to inspect

1. Compare g bitwise flag/maxabs error before/after at B13/wave104/step5 and across the full direct fixture. The new macro should remove the identified algorithm difference. A remaining g difference then warrants an operand/rounding diagnostic; it is not evidence to revert to the lower-accuracy path.
2. Beta and conv tensors should reproduce the wide-mode candidate exactly. Their computational source is unchanged. Changing either would indicate an unexpected compilation/shared-state effect requiring investigation.
3. The matched-production-conv probe distinguishes the separate legacy convolution discrepancy from g/recurrence differences; retain its existing bitwise production-conv and same-gates-state checks.
4. Run model boundary and long-answer checks before evaluating speed. No claim that g repair alone fixes the failing per-head output is made here.

`host_verify.py` checks unchanged-source hashes, all eight exact file transformations, 64-entry unpack vectors, actual bound compute/kernel/program descriptors with the pair-sequence define, six CLI paths with device opening disabled, and the analytic polynomial discriminator. `host_validation.json` and `source_manifest.json` record the result. These are CPU/API/source checks only; no JIT/device/Watcher/PCC result is claimed.
