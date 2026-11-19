# Final28 GDN HF contract failure: CPU-only diagnosis

The leading source-supported hypothesis is **inherited LoFi attention precision amplified by a sensitive token**, not a demonstrated BF16-HF/FP32-recurrence reference incompatibility. Packed primary/correction projection remains the principal new-path alternative to discriminate. No hardware, TTNN/Torch import, build, production or harness edit was performed for this report. This is a failure investigation after the user cap, not new tuning. Keep every .995 gate unchanged.

## Direct evidence

`final_validation28_contract_l0_tp4.json` fails B1/S31 contribution row12 at **0.9860137708810336**, although aggregate output PCC is0.9993891826645709 and all values are finite. S1 passes output0.999981286522186/contribution0.9999811426162558. The other30 S31 contribution rows pass (next-lowest≈0.997731). The failure occurs before continuation/mixed-row tests, and before any trace capture. Do not label those unexecuted tests failed or passed.

The unchanged frozen single-chip optimized implementation has the same HF BF16 sequence-of-lengths contract: `doc/optimized_decoder/final194_correctness_run_contract_l0.json` reports S31 aggregate0.9999649312265316 and worst contribution **the same row12**,0.9997872844895868. Its optimized source SHA315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22 is still the inherited implementation. That sensitivity at the same row supports investigating amplified numeric error; it contradicts an assumption that the HF BF16 oracle is intrinsically impossible for this workload. It is historical single-chip evidence, not a substitute TP4 pass or a raw-input hash match.

Stage4 TP4 did not contain `tests/run_multichip_contract.py` at commit8502f42bca20. Its principal gates compared TP4 against frozen optimized TTNN references, with per-user metrics. Existing stage4 evidence therefore does **not** prove this new per-token HF contribution gate passed. The stage4 anomaly is directly relevant:

- `doc/multichip_decoder/pcc31_control_summary.json`: default LoFi synthetic B31/S31 decode worst user25 PCC0.9946967451902956; matched HiFi4 worst-user decode0.9992108423356916.
- Disabling fused MMRS preserves those failing decode/state metrics. Matched recorded model activations pass LoFi; that does not waive today's explicit HF contract.
- `doc/multichip_decoder/debug_pcc31/AUTODEBUG.md` documents the effect through packed projection, convolution and output projection, with no lowered gate or claimed layout error.

## Source localization and exclusions

1. **Effective GDN attention is LoFi.** Current `MultichipDecoder.from_state_dict` sets policy `attention_fidelity="LoFi"`. Although its initial attention config selects HiFi4 for GDN, the subsequent role-override loop replaces it with LoFi. Frozen stage4 `stage4_control_multichip_decoder.py:61,97–121` does the same. The frozen single-chip GDN policy instead specifies HiFi4 (`tt/optimized_decoder.py` default policy). Thus this is a plausible inherited precision limitation, not necessarily introduced by stage5.

2. The same attention compute config reaches **joint primary/correction matmul**, native B1 multi-token causal convolution, and attention output projection. HiFi4 changes those arithmetic paths but preserves BF16/BFP8 types, FP32 destination accumulation, GDN packer-L1-acc=false, residual layout, weights and public interfaces. `PackedCorrection._phase_config` takes `owner.matmul_configs["attention"]`, so the existing override actually reaches the new packed projection; it is not dead policy metadata.

3. At **B1/S31**, `_gdn` uses the existing multi-token native convolution and `chunk_gated_delta_rule`, **not** column recurrence, cumulative-exp scalar recurrence, local fused decode convolution or beta/decay fusion. `tiled_prefill_conv` requires B>1 unless its separate B1 flag is set; that flag is absent. Local beta/decay requires `decode_batch is not None`; it is absent here. Distributed fused norm is off. Flat `_can_flatten` requires B>1. Changing these decode-only policies cannot directly repair this failed prefill row.

4. Native chunk recurrence consumes local Hk4/Hv12 per rank (frozen single-chip Hk16/Hv48); chunk size32, HiFi4 `self.ckc`, BF16 readout cast, weighted RMSNorm, head concatenation and SiLU-gate sequence are the same relevant stage4 multi-token path. State is zero at start; beta/g logical S31 are padded with zero. Row12 is far from padded row31. A tail or head partition defect is not ruled out, but is unsupported without matched intermediate discrepancies.

5. **New-path alternative: packed correction.** Current `_linear("packed_gdn")` sends `[1,31,5120]` to `PackedCorrection.project` (compact projection, phase="decode"), even though this is a31-token prefill. Current phase precision contracts are equal FP32 output/FP32 accumulation, so the phase label alone is not a demonstrated numerical defect. It does alter joint projection layout/config versus frozen separate primary/correction. The output combine rounds once to BF16. `project_prefill` uses the same compact kernel for rows<=32 with phase="prefill". A proposed “call prefill instead” fix therefore changes no arithmetic under current contracts; do not assume it repairs the gate.

6. Fused MMRS is disabled for31 rows (`mmrs_min_rows=256`), so the long-row native output path is not the direct suspect. The partial contribution subtracts the same input residual in FP32; retain it, as it exposes error hidden by aggregate output correlation.

7. The harness labels BF16 HF correctly and uses actual weights, fresh HF DynamicCache, reset device state and no guarded host conversions. However, the failing run did not save raw input/reference/intermediate tensors, and its seed label alone is insufficient for future matched controls. `load_layer()` calls `torch.manual_seed(17)` and HF construction consumes RNG; there is also a64-entry randperm and the preceding S1 random tensor. Recreate the exact full setup sequence or save input and HF output once in the diagnostic; simply calling `manual_seed(183); randn(1,31,5120)` will generate a different fixture.

## Minimal matched control experiment (parent-owned)

Use one saved, bit-identical S31 input and HF output from the failing harness's setup/S1/S31 sequence; hash the tensors. Reset state before every arm. Record all rows before evaluating assertions. Keep .995 output and per-row contribution gates, finite checks and watcher; do not rank timing from this diagnostic.

| Arm | Only intended difference | Discriminating meaning |
|---|---|---|
| A | Current final defaults | Must reproduce row12 failure on saved fixture; otherwise investigate input/provenance mismatch first. |
| B | Current plus `attention_fidelity="HiFi4"` | Smallest existing supported repair hypothesis. If all rows pass, precision dependence is established without changing the reference. |
| C | Frozen original TP4 SHA082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44 with its original defaults | Distinguishes inherited TP4 limitation from stage5 regression. Same source-loader approach as test-only original timing control; never runtime fallback. |

If A fails/B passes/C fails similarly, the safest supported repair is **GDN attention fidelity HiFi4**, followed by full unchanged contract/selected-path quality, long stress and current-harness timing/profiles. This is a single existing policy setting, not a kernel rewrite. Do not make it synthetic-input-, row12-, or S31-specific. Performance must be remeasured and final source pins updated; older LoFi numbers no longer describe the final default. If a phase-specific repair is required later, first isolate the actual sensitive producer; no new tuning is justified merely to reclaim latency under the cap.

If C passes or B fails, add **D: current with packed correction disabled at setup** (`packed_correction_options=None`, retaining original separate weights). This override prevents the installer, so the normal original primary/correction path exists; setting only a runtime helper toNone after release would be unsafe. Do not combine D with HiFi4 until its isolated result is saved. A frozen single-chip arm is optional diagnostic only, not completion.

For any unresolved arm, capture boundaries outside the guarded forward in a dedicated diagnostic: input norm; packed Q/K/V/Z/beta/a before convolution; native conv Q/K/V; chunk output before BF16 cast and after weighted norm/gate; attention output before/after RS; post-attention residual/norm; MLP contribution. Save per-rank/head **row12**, neighboring rows, norms/max absolute error and PCC, plus final state. Compare raw tensor differences A/B/C before assigning blame. HF FP32 or FP64 recurrence can be a secondary diagnostic, but cannot replace the accepted BF16 HF oracle or lower its gate.

## Safest decision now

No root cause or repair is claimed without the matched run. Existing stage4 source/evidence makes HiFi4 the highest-value first control; no source-supported reason currently warrants altering the HF precision, contribution definition, seed, public length support or threshold. The original failed receipt remains authoritative and must be retained.

## Matched frozen result received during diagnosis

Parent completed `final28_contract_stage4_l0.json`: frozen original TP4 reproduces **exactly the reported S31 aggregate0.9993891826645709 and row12 contribution0.9860137708810336**. This establishes an inherited discrepancy on the matched fixture rather than a stage5 optimization regression. It does not itself prove bit-identical raw outputs or locate the responsible operation. Current HiFi4 is running the complete unchanged contract in `final28_contract_hifi4_l0`; no result is assumed here.

### Narrow prefill fidelity repair, conditional on that control

A GDN **prefill-only attention-fidelity override** is a plausible narrower repair that preserves the measured actual decode policy, but must be explicit phase dispatch rather than row-count dispatch. B1S31 prefill and B31 compact decode both have `[1,31,H]`. Merely choosing HiFi4 for `rows>1` would alter decode; choosing it only for `rows>32` would miss this failure.

Source-supported minimum propagation map:

| Producer/consumer | Prefill admission | Required config propagation |
|---|---|---|
| Flat GDN packed input | `_flat_chunk` calls `_prefill_packed_gdn` before `_gdn(packed=...)` | Existing `project_prefill` and `_phase_config(...,"prefill")` can select a setup-created HiFi4 attention-prefill config. |
| Legacy/nonflat GDN packed input, including B1/S31 | `_gdn` has `decode_batch is None` | Must call the semantic prefill projection entrypoint even when rows<=32; current `_linear` otherwise calls `project(...phase="decode")`. |
| Native multi-token causal convolution | `_gdn`, `s>1` | Select the prefill attention config in `qkv_causal_conv1d_silu_with_carry`. B>1 tiled prefill conv is SFPU-based and must retain its existing precise arithmetic unless evidence requires changes. |
| GDN output projection | `_gdn` returns through `linear_attn.out_proj.weight` in both S1 and S>1 branches | Explicit prefill config must reach both separate DRAM-sharded matmul and the registered-row fused-MMRS descriptor. Leaving one branch on LoFi gives a length-dependent partial repair. |
| S1 prefill/resume tail | `decode_batch is None`, even when `s==1` | Preserve semantic prefill phase. The unchanged continuation gate checks consistency against full prefill; do not create an unmeasured S1-LoFi exception solely because standalone S1 passed. Actual `decode_forward` has non-None `decode_batch` with selected flat_decode and retains LoFi. |

Create both immutable compute configs during setup. Keep explicit phase/config arguments in call-local dispatch; **do not temporarily mutate shared `policy`/`matmul_configs`, install closures, or shallow-clone the decoder inside forward**. Existing packed `_phase_config` is the clean adapter point, but inherited `_linear_raw` currently reads `self.matmul_configs[group]` directly, so out-projection phase plumbing requires an explicit model-local implementation/adapter or a carefully scoped API change; it is not accomplished by one new dictionary key. Any alternative must preserve actual weight/projection ownership and runtime-audit coverage.

This smaller repair remains a hypothesis until it passes the complete unchanged HF contract (including31/32/33, full/resume545 and mixed rows), selected TP4 long stress and changed-input traced replay. A global HiFi4 pass changes projections and convolution together; it does not license a still-narrower per-operation fix without a discriminating measurement. Final B1 prefill timing and provenance must be rerun after any selected fidelity change; prefill state differences can also affect decode quality even if decode kernels are unchanged.
