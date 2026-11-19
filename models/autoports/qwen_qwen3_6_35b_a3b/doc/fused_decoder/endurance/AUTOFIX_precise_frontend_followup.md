# AutoFix precise frontend follow-up

CPU/source inspection only; no TTNN imports, devices, or runtime/test edits. Frozen source: `../endurance_precise_projection_silu.py.snapshot`. Captures: `../endurance_precise_probe.json.pos{128,129,130,1024,3979,5011,7551,7703,8157}.pt`.

## Result

The current convolution and Q/K normalization reproduce canonical CPU operations **exactly on their saved TT operands at all nine captured positions**. At late failing position7703, input norm, A/B projections, Q, K, V and beta are also bit-identical to their HF counterparts. The remaining g difference there is FP32 scale: relative L2 5.475e-8, max absolute7.629e-6. These results reject a material current-token frontend discrepancy as the explanation for the7703 failure. They do not rule out accumulated effects of earlier rare rounding differences.

There are two independently demonstrated earlier frontend discrepancies:

1. The first decode tokens consume inaccurate **prefill convolution history**. The current decode conv faithfully propagates those differing inputs; its arithmetic is not the source of that discrepancy.
2. At position130, input norm differs from HF by **one BF16 code at channel239**, at an exact FP32 BF16-rounding midpoint. The earlier six-case exact-normalization microprobe did not establish equivalence for every input. This is the first sampled decode operation to diverge at130, ahead of projection.

## Same-input convolution replay

For each capture, concatenate saved TT `conv_input_state[1,3,8192]` and rounded current `projection_packed[..., :8192]` to make the four-tap input. Apply CPU BF16 grouped conv1d with actual checkpoint weights, BF16 SiLU, split Q/K/V, and canonical BF16 L2 arithmetic. This gives **zero differing elements** versus the saved TT V, Q and K at every captured position.

The replay must retain canonical HF head duplication **before** CPU L2: reshape Q/K to[1,1,16,128], repeat each head twice to[1,1,32,128], then apply `(a * rsqrt((a*a).sum(-1,keepdim=True)+1e-6))`, and select every second head for comparison with TT. On this CPU, independently normalizing only16 heads gives different BF16 results from normalizing32 duplicated heads. For example at1024 it would falsely report668 Q mismatches; duplicating first gives zero. TT concatenates Q and K into32 heads for its elementwise sequence and matches the actual HF32-head result. Do not diagnose a TT L2 defect from the smaller CPU shape.

HF source contract: `modeling_qwen3_5_moe.py:220–235` concatenates cached/current projection, uses BF16 grouped conv1d, then SiLU; lines238–241 define BF16 L2. Forward duplicates Q/K at515–518 before invoking the recurrence.

## Early discrepancies and their sizes

| Position | Q relative L2 to HF | K relative L2 to HF | V relative L2 to HF | TT Q/K/V versus CPU replay on TT operands |
|---|---:|---:|---:|---|
|128|.004119|.005130|.002686|all exact|
|129|.002352|.004171|.002860|all exact|
|130|.002086|.002316|.001255|all exact|
|1024|0|0|0|all exact|
|3979|0|0|0|all exact|
|5011|0|0|0|all exact|
|7551|0|.00009156 (2 elements)|0|all exact|
|7703|0|0|0|all exact|
|8157|0|0|.00001779 (1 element)|all exact|

Reconstructed canonical prefix from routing_probe's seed913, validated regenerated first decode token exactly against the capture, normalized the128 prefix tokens and applied canonical BF16 prefix QKV linear. The three incoming TT conv-history slots at128 differ in13566/24576 BF16 elements, relative L2 .003513. Per-slot relative L2 is .003900/.003036/.003730. At129 the cache mismatch decreases to8824 elements/.002780, and at130 to4928/.002235 as inaccurate prefix slots roll out. This is consistent with the parent's HF-prefix-seeding discriminator; repairing decode conv alone cannot repair prefill history.

On exact same saved TT input norm, current SFPU QKV projection relative L2 against CPU FP32 linear is 1.814e-7 at128,1.933e-7 at129,1.964e-7 at130,1.973e-7 at1024,1.824e-7 at3979,1.912e-7 at5011. The previous~.00033 native projection discrepancy is absent. BF16 projection disagreements are now rare rounding cases plus CPU tiny subnormal artifacts: at3979 all100 differing values are at most3.292e-37 and produce no Q/K/V difference.

At130 the different input norm causes930 BF16 QKV differences (relative L2 .0008289),3 A differences and5 B differences. A/B are exactly HF at every other captured position. Beta is exactly the CPU sigmoid of rounded TT B at all nine positions; at130 these B differences still round to identical beta.

### Exact midpoint at input-norm channel239, position130

* Token: -0.044677734375; checkpoint norm weight: .1708984375 (effective gamma1.1708984375).
* CPU FP32 mean square: .0024054418317973614; reciprocal square root after epsilon1e-6:20.385074615478516.
* CPU FP32 normalized/scaled value: **-1.06640625**, exactly midway between adjacent BF16 codes.
* HF rounds to **-1.0625**, while TT returns **-1.0703125**.
* Float64 formula gives -1.0664062831077001, slightly below that midpoint. Thus merely demanding more real-number accuracy would not necessarily reproduce the canonical FP32 computation: HF's intermediate rounding puts the result precisely at an RNE tie.

The capture does not include TT mean square, rsqrt, normalized FP32, or pre-cast scaled value, so it cannot attribute this single-code difference to a specific primitive. Do not infer a typecast bug from the final BF16 code alone.

## Gate findings

Current g differs from canonical CPU gate evaluation on identical rounded A at relative L21.7e-12 to6.1e-8 across these captures. At7703 it differs in8/32 FP32 elements, relative L25.475e-8; beta, A and B are exact HF there. The gate formula and BF16 boundaries are correct. Whether these tiny g differences can flip later BF16 recurrent-cache ties remains a recurrence investigation, not evidence of a missing frontend boundary.

## Minimal next discriminators

* For the remaining frontend question, capture the four TT input-norm FP32 intermediates at130 and compare each to HF CPU. Test a same-input microcase at this exact token rather than further random cases. This determines whether a primitive's one-ULP error or reduction ordering crosses the documented midpoint.
* To test whether rare normalization/projection BF16 ties materially explain remaining endurance divergence, feed canonical saved HF input norms to the TT projection for a diagnostic stream, with canonical HF prefix state. This is a causal control, not a proposed model fallback. No production epsilon adjustment, biased rounding, or threshold relaxation is justified by this one tie.
* The late captured Q/K/V equality means a new blanket conv/L2 rewrite is not supported. Prioritize the parallel recurrence-intermediate investigation and prefix-state accuracy.

Parent-provided native matmul evidence further localizes the old projection error: same-face two-product sum has~.00013 error, cross-face sum is exact; residue-mask8 reduces full projection error to~5e-7 and mask16 to~3e-7. This supports within-MVMUL multi-product summation as the old error source and offers a measured native alternative to SFPU projection. It does not explain the newly identified input-norm midpoint or prove the remaining endurance failure is fixed.
