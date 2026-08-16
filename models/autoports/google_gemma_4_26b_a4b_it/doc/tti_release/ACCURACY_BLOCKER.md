# Mandatory accuracy comparison blocker

The TTI workflow executed both mandatory tasks, but their report rows are `NA` because neither has a method-compatible reference:

- `meta_ifeval`: measured 82.62 on the fixed 5% CI-nightly subset; no official or same-harness reference is available.
- `meta_gpqa_cot`: measured 40.0 on the fixed 5% CI-nightly subset. Google's published GPQA Diamond 82.3% does not document equivalence to TTI's `gpqa_diamond_cot_zeroshot` recipe or this subset.

This is not waived. TTI's valid mechanisms require either a comparable full-set `gpu_reference_score` or an exact-subset `ModeReferenceScore` with recorded model snapshot, lm-eval task/version, chat-template/generation settings, seed, and document identities. Self-baselining the TT result would fabricate a control.

Unblock by running both tasks against a trusted HF/GPU implementation on the exact CI subset with identical harness settings, wiring the resulting references, and rerunning the TTI eval/report gate. Until then the classification is `release-workflow-pass/readiness-fail`, not Stage 11 readiness.
