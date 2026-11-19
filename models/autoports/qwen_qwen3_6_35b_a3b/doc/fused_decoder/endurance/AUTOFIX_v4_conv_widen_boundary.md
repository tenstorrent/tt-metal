# v4 tilize/widen boundary follow-up

The [initial device report](v4_conv_adapter_probe.json) remains unchanged and correctly reports `device_exact=false`. Direct RM history passes all eight cases, including every captured FP32 tap operand and the exhaustive finite-BF16 case. Folded tilize/widen passes the seven seeded shapes, but each exhaustive-case FP32 tap operand has 1,408 differing codes; FP32 preactivation, BF16 final output and carry still match in that fixture. This does **not** establish an all-input convolution equivalence.

## Measured complete adapters

| Rows | Baseline ms | Folded tilize/widen ms | Direct RM history ms |
| --- | ---: | ---: | ---: |
| s1 | 0.169010088 | 0.153559586 | 0.155439135 |
| s32 | 0.215478893 | 0.203368813 | 0.202560192 |
| s33 | 0.292340899 | 0.267228903 | 0.277417386 |
| s127 | 0.614856416 | 0.568487914 | 0.568472827 |
| s128 | 0.610294798 | 0.563465199 | 0.569601916 |
| s129 | 0.727268262 | 0.661919359 | 0.681172311 |
| s256 | 0.964494841 | 0.876649888 | 0.917714415 |

These are root-collected complete-adapter timings. They do not waive the folded candidate's differing-intermediate result or qualify a complete decoder candidate. The initial exhaustive case was not timed.

## Source-backed distinction

The first reported difference is baseline `0x00000000` versus folded `0x80800000` (−1.1754943508222875e−38). Reconstructing the fixture's finite-code enumeration identifies the source BF16 code as `0x8080`: a **negative normal input**, not a BF16 subnormal. The largest reported operand difference is1.7540579766176321e−38. Do not summarize this as only signed-zero or only subnormal-input mismatch.

The source proves the two routes use different conversion configurations:

1. `copy/typecast/typecast.cpp:33–41` sets `fp32_dest_acc_en=true` when the **output** is FP32. For BF16 input, `preserve_fp32_precision` is false, so the input follows Src unpack rather than direct FP32 unpack (`copy/typecast/device/typecast_program_factory.cpp:34–44`). The BF16→FP32 typecast has no SFPU arithmetic body (`tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:158–159`); format conversion is performed by unpack/datacopy/pack.
2. `data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_default_program_factory.cpp:33–34` does **not** enable FP32 destination merely for FP32 output. BF16→FP32 tilize therefore uses its normal 16-bit destination/unpack configuration and an FP32 output pack. That configuration is applied at lines164–179. The ordinary aligned tilize factory follows the same output-dtype omission (`tilize/device/tilize_multi_core_default_program_factory.cpp:37–39`).
3. The tested differences are consistent with these distinct format paths near the smallest exponent range. This identifies the relevant source boundary, but does not prove which exact hardware zero-substitution/pack instruction causes every differing code. The original report saves only the first eight code differences per tensor; it cannot establish the complete affected input domain or that every baseline difference is zero.

The downstream operation does not provide a source-level erasure proof. `ternary/device/ternary_program_factory.cpp:1483–1520` enables FP32 destination and directly unpacks all three FP32 addcmul operands into destination. It does not repeat the standalone BF16→FP32 conversion. The Blackhole SFPU addcmul (`ckernel_sfpu_addcmul.h:46–59`) computes a value-scaled input and MAD with the accumulator. No explicit threshold clamp in this sequence guarantees replacement of these tiny operands by the baseline's zero.

Checkpoint tap maximum magnitudes are `[0.255859375, 0.375, 0.40625, 0.388671875]`. These make the changed products very small, explaining why ordinary accumulators can hide them, but small products do not prove equality around underflow, cancellation or BF16 rounding boundaries. In particular, one tiny tap embedded beside a carefully chosen near-underflow first-tap accumulator is a stronger discriminator than the original enumeration, whose neighboring temporal values come from widely separated exponent ranges.

## Follow-up diagnostic and minimum safe adaptation

New [tests/v4_conv_widen_boundary_probe.py](../../../tests/v4_conv_widen_boundary_probe.py) leaves the first probe and reports untouched. It compares the complete BF16→FP32 conversion routes over all finite BF16 codes, records **every** differing input/baseline/candidate code, and saves a compact `.conversion.pt` with the complete tensors for CPU diagnosis. CPU reference comparison is separate from baseline-path equivalence.

It then runs the actual extracted convolution blocks on144 crafted S1 histories: each of taps1/2/3 separately receives one of eight negative BF16 codes around zero/minimum-normal and nearby exponents; checkpoint first-tap weights anchor the accumulator around0/1/2/4/8/16 times FP32 minimum normal. All8192 actual channel weights participate. It compares FP32 preactivation, BF16 output and BF16 carry exactly. It contains no timing or tolerance override. [CPU setup](v4_conv_widen_boundary_probe_cpu.json) passes all144 finite-input cases; device execution is root-owned; completed results are recorded below.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_conv_widen_boundary_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_conv_widen_boundary_probe.json
```

The adversarial probe tests bitwise equivalence as a diagnostic; it does not replace the user’s functional PCC≥0.995 acceptance contract. A difference must remain visible in the diagnostic report, but does not automatically reject a faster model path. Selection depends on material model-visible effects, unchanged required PCC/regression checks and complete-path performance. Never relabel a differing diagnostic as exact or exclude tiny values from its counts.

The minimum change that retains the **baseline conversion codes** is to keep the three original explicit casts, as the independently exact direct-RM-history candidate does. That observation is not a recommendation to prefer a slower precision path when model-level acceptance and performance favor folded widening. The public `to_layout`/tilize interface exposes output dtype but no `fp32_dest_acc_en` compute override for reproducing the standalone conversion configuration. Reintroducing tilize(BF16)+typecast(FP32) restores the proven path and removes the proposed folding benefit. A guessed numerical clamp is not justified without the complete difference map and could alter legitimate FP32 inputs. No shared native edits are proposed.

## Completed adversarial and whole-model results

The [boundary diagnostic](v4_conv_widen_boundary_probe.json) completed with `conversion_exact=false` and `conv_exact=false`; these statuses and all original counts remain unchanged. Of144 actual-checkpoint-tap cases:

- 36 differ in FP32 preactivation, maximum absolute difference1.2745762415989952e−38.
- 24 differ in BF16 convolution output, maximum absolute difference1.6530389308438418e−38.
- All carry outputs are exact.

This establishes that conversion differences **can** propagate to output for crafted near-underflow accumulators. It rules out global all-input bitwise equivalence. It does not establish a material model error.

The complete conversion capture also clarifies the affected domain. The two paths differ for2,816 unique normal BF16 input codes: positive `0x0080..0x05ff` and negative `0x8080..0x85ff`. Their values are not all zero in the baseline; only64 of the differing entries have baseline positive zero. The folded path matches direct CPU BF16→FP32 widening on these normal codes. Across the full finite-code fixture, baseline differs from CPU on3,071 entries, versus255 for folded widening (the shared subnormal/signed-zero range). Thus the folded path is closer to exact numerical widening in this microprobe; the issue is baseline-path equivalence, not evidence that folded widening broadly reduces precision. The report and `.conversion.pt` preserve the full map.

Root's completed full decoder benchmarks use10 prefill/60 decode samples:

| Complete model candidate | Prefill ms | Decode ms |
| --- | ---: | ---: |
| [Raw folded widening on compact serial v4](v4_conv_tilize_widen_layer0_s256.json) |117.809963|1.337928|
| [Combined adapters baseline](v4_combined_adapters_layer0_s256.json), `82c8c629…` |106.958413|1.332007|
| [Combined adapters plus folded widening](v4_combined_tilize_widen_layer0_s256.json), `72d4f05d…` |108.104917|1.317329|

The combined pair improves measured decode by about1.10%; prefill increases about1.07%. Short PCC is unchanged: prefill0.999962395060449, contribution0.9999594863488576 and traced decode0.9999779064754709. Both reports show deterministic replay and runtime-fallback audit pass. The hashes are source-bound in the reports: baseline `82c8c62950b85fdea309f2eed91fbc0f7b9691cb5da78fa97b0728154e8be1cb`; folded combination `72d4f05d5e089c7f479e7d4bb35dcd2570aa4917dbd65c08961cc1ad0bcd9336`.

At this documentation update, all three candidate-selection 8192+33 stream artifacts are complete and pass (seed154 completed after the initial parent status message):

| Seed | Report | Minimum decode contribution PCC |
| --- | --- | ---: |
|913|[candidate stream](v4_combined_tilize_widen_stress_0_seed913.json)|0.996216457546374|
|42|[candidate stream](v4_combined_tilize_widen_stress_0_seed42.json)|0.9956661399915544|
|154|[candidate stream](v4_combined_tilize_widen_stress_0_seed154.json)|0.9960950212847441|

Root reports these contribution minima are identical to the base. These are cached-oracle, single-repeat candidate-selection runs, not final fresh-HF repeated acceptance. Required final model gates and root's selection remain open. The unchanged PCC≥0.995 requirement, not global bitwise intermediate equivalence, controls functional acceptance; the synthetic differences remain accurately recorded alongside the measured model evidence.
