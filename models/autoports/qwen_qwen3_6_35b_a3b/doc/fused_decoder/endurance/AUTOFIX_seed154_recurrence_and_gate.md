# Seed154 position5991: recurrence is locally exact; gated output boundaries differ

CPU/source-only analysis of parent-generated `routing_seed154.json.pos{128,5990,5991,5992}.pt`. Reproducer: `python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/routing_seed154_cpu.py`. The script imports Torch/safetensors, never TTNN; complete measurements and capture SHA256 are in `routing_seed154_cpu.json`. No F edits or device runs by this investigator.

## Supported localization

At all four captures, input norm, A/B/Z projections, Q/K/V, beta, and decay match HF numerically. Q/K code discrepancies are signed zeros; packed QKV discrepancies are subnormal-scale (max8.94e-37 at5991). Every meaningful same-input recurrence operation matches CPU exactly: memory contraction, subtract, beta multiply, separate state update, output products and output contraction. Complete CPU recurrence replay on the captured TT operands reproduces **the FP32 TT new state and core bit-for-bit**. Decay/product subnormals flush differently, but do not change the completed recurrence outputs at these captures.

HF-input CPU replay also reproduces HF BF16 core/cache bit-for-bit, validating the oracle reconstruction. Substituting HF incoming cache in the CPU TT-input replay yields exact HF BF16 core/cache at all four points. The BF16 cache boundary is present and correct. These sparse observations do not identify the origin of all historical state differences; they do rule out a local recurrence arithmetic mismatch at the failing token and its neighbors.

| Position | Incoming state relL2 | Outgoing TT.bfloat16 vs HF state: differing codes / relL2 | BF16 core differing values / max error |
|---|---:|---:|---:|
|128|4.9821e-6|85 /4.6470e-6|1 /2.9802e-8|
|5990|1.4816e-4|2186 /1.4836e-4|17 /1.1444e-5|
|5991|1.4836e-4|2182 /1.4791e-4|19 /1.5259e-5|
|5992|1.4791e-4|2184 /1.4563e-4|16 /1.5259e-5|

The earlier ~.0014–.0016 post-state relative errors mainly compare unrounded TT FP32 output against HF BF16 storage. The table compares like-for-like cache values.

At5991 HF experts233 and252 tie at BF16 logit5.21875, with233 selected in HF's top8. TT has233=5.21875 and252=5.25, selecting252 instead. Top8 match at5990. This is a real discontinuity on the fixed oracle, not grounds to relax its gate or arbitrarily modify tie-breaking.

## Concrete boundary mismatch

Installed HF `modeling_qwen3_5_moe.py:325–366` returns recurrent output in the original BF16 query dtype. `Qwen3_5MoeRMSNormGated.forward`, lines191–200, then computes:

1. BF16 core -> FP32; variance and inverse RMS in FP32.
2. Normalize without learned weight, then round normalized values to BF16.
3. Multiply by BF16 learned weight; result is BF16 (another rounding boundary).
4. Multiply that weighted result by `silu(z.float())`; this product is FP32.
5. Round gated output to BF16 before output projection.

The captured F returns FP32 core from `_recurrent_step` and feeds it directly to `self.norm(o,"linear_attn.norm.weight")`. That helper passes learned FP32 weight to native RMSNorm before its final BF16 cast. Its subsequent multiply applies SILU as an input activation to the BF16 z operand. Thus at least the BF16 core boundary and normalized-before-weight boundary are absent; the gate's FP32 semantic boundary also needs explicit preservation. These are source-level differences, independent of whether a particular fused kernel happens to produce the same rounded value.

## CPU downstream discriminator

The CPU control uses checkpoint BF16 Z/out-projection/normalization/router weights and saved tensors only. HF-core replay reproduces HF attention, postnorm, and router **bit-for-bit at every captured position**. Saved TT Z also equals canonical CPU Z bit-for-bit. The counterfactual uses the real saved TT core with canonical gated normalization and downstream CPU arithmetic; it does not replace recurrent state or pretend to be device acceptance.

| Position | Actual TT attention relL2 | Canonical downstream using TT core: attention relL2 | Counterfactual top8 |
|---|---:|---:|---|
|128|.00507456|.00010223|Matches HF|
|5990|.00328314|.00029644|Matches HF|
|5991|.00477833|.00131851|Matches HF;233=252=5.21875|
|5992|.00368707|.00096084|Matches HF|

Conversely, starting from **actual TT attention** and using canonical CPU postnorm/router still selects252 at5991, with233=5.21875,252=5.25. Fixing postnorm alone is therefore insufficient in this CPU discriminator. Canonical gated output plus CPU out-projection fixes the chosen expert despite retaining the small real cache/core error. This does not yet distinguish native output-projection rounding from the identified gated-normalization boundary errors; retain that projection while correcting explicit boundaries first.

## Minimal parent device experiment

At the decode core-to-output interface, preserve all current recurrence and cache logic. Cast core to BF16 before normalization; perform **unweighted** RMSNorm and BF16 output; multiply BF16 learned norm weight with BF16 result; widen z to FP32 for SiLU; multiply and cast gated output to BF16. Existing `output_norm_bf16` setup weight is available. Keep output projection/config unchanged for the first discriminator. Do not replace linear input norm or change expert tie rules. Validate the same seed1545991 case and existing stream gates; CPU replay is evidence for this experiment, not a hardware pass claim.
