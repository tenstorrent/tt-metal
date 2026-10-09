# Next controlled accuracy experiment

The complete optimized 64K GPQA score is 163/198. Native recurrence at the same
64K budget, accurate attention, checkpoint and sampler reached 170 correct out
of 195 completed questions at 04:53 UTC on October 9, 2026, with no output
cutoffs yet. Even perfect answers on the remaining three cannot reach the
unchanged 177/198 gate. Keep the running test and all its results; do not change
its budget, precision or denominator.

The next prepared configuration is
[`precision_accurate_decode_bfp8_head.json`](../config/precision_accurate_decode_bfp8_head.json).
It restores the original model's LM-head precision: BFP8 weights and HiFi2
matmul, while keeping native recurrence and accurate decode attention. All 64
decoder-layer precision policies, BFP8 KV, BF16 activations and FP32 recurrence
remain identical to the running native control. The policy loader and explicit
decoder-policy comparison passed locally. **No hardware run or score is claimed
for this new configuration, and it is not yet queued.**

This tests whether the selected BFP4/LoFi head contributes to the remaining
accuracy loss. It does not establish that the head is responsible, and changing
weight precision and fidelity together measures the combined restoration only.
It increases LM-head storage and bandwidth; measure its throughput cost before
choosing a release configuration. Do not project an accuracy gain in advance.

Use a new immutable source snapshot and results directory after the current
owned hardware queue releases its lock. Set `QWEN_PRECISION_CONFIG` to the new
file before running eight-replica G0. The new precision fingerprint requires a
new G0 receipt; the existing native-control receipt cannot authorize it. Then
run the existing full-GPQA serving runner with 198 questions, 64K output,
concurrency 128, T1/p.95/k20 and seed 42, retaining private raw responses. The
same source/precision and worker-binding checks must pass before evaluation.

Compare full-set correctness, per-question changes, natural-stop failures,
output-budget cutoffs, mean decode rate and total wall time with the completed
native control. A separate higher-output-budget experiment is useful for cases
that actually truncate, but cannot repair already completed wrong answers by
itself. Preserve the full score and report any diagnostic subsets separately.

The existing experimental image build retains the original native-control
policy. This prepared ablation does not silently change its source or launch.
