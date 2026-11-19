# Stage-owner disposition: active-width numerical sensitivity

The frozen cross-active-width PCC diagnostic remains **FAILED**. It is a
controlled numerical limitation of comparing different active-width histories,
not a passing equality test. This decision does not change its thresholds,
rewrite any raw result, relax the HF accuracy bars, or change decoder policy.
[Independent stage review round1](stage_review_round1.md) accepted this scoped
known-limitation classification with a clean-pass verdict. No historical failed
comparison is relabeled.
Here "owner" means the stage agent's evidence classification, not a separate
human waiver of an accuracy gate.

The current production probe gives whole/min-row PCC 0.990041606/0.988163811
at forced token43, with identical top1 and 100% top5/top100 membership.
The matched original HF implementation also fails the same PCC limits:
0.992546833/0.987764712. HF additionally changes its unique greedy maximum
when a request survives a different batch-width history at44. These controls
use the pinned checkpoint's actual HF operators and cache, without emulating
TT quantization or changing precision to manufacture a failure. They establish
width/history sensitivity on the tested BF16 CPU/eager backend; they do not
establish identical error magnitude or a universal property of every backend.

Current TT greedy predictions meet the independent HF criterion: every one
of the 31 distinct history/position rows is in its matched HF top5 and top100,
and each per-history prefill/decode group meets the unchanged acceptance bar.
The repeated 31 rows establish reproducibility, not extra accuracy samples.
All actual sampled tokens are maxima of the same TT logits. In particular,
the survivor44 token81760 has strict HF rank4. Current forced43 has no
TT-self top1/top5 disagreement; the larger historical discrepancy and its
FAILED raw artifact remain retained separately.

Investigation traced the first historical TT divergence to the inherited
native routed-expert reduction with identical inputs/router/experts. Exact ISA
emulation and BF16/FP32 counterfactuals isolate its accumulation policy.
Using FP32 for every batched reduction or executing every row as a singleton
would abandon the required optimized policy. The current implementation keeps
the accepted singleton FP32 versus multi-row BF16 behavior. This is a source
and control explanation, not a claim that intended arithmetic guarantees
accuracy: the direct HF bars remain independently required.

The supported reproducibility guarantee is exact logits/tokens/state for
repeated identical execution histories and relocation across batch positions.
Changing active width can change rounding and subsequent recurrent history;
cross-width bitwise logits or identical top1 are not guaranteed, as the matched
HF control demonstrates. Mixed-length prefill equality, resumed-prefill,
explicit cache ownership, dormant-state preservation and true device token
feedback remain required. The corrected-seed reduced B32 diagnostic and the completed full48
`batch32_hf_factored_v2.json` both pass these state controls. Each full48 history
scores2645/2704 top1 and2704/2704 top5/top100; exact same-history logits, tokens
and final request state pass, with empty final/dormant mismatch dictionaries.
The original B32v1 failure remains preserved. These results support serving-row
closure; maximum-context and independent review remain separate stage gates.

This resolves the cross-width diagnostic by matched original-model control
under the user's instruction to record reproduced HF limitations and continue.
It does not resolve any independent failed HF accuracy, state, execution,
capacity or review gate. No context/batch capability is reduced and no new
fallback or sampler is selected.

Evidence: `batch_geometry_current_evidence.md`,
`current_geometry_sampler_v1.json`, `shared_pool_hf_control_v2.json`,
`shared_pool_hf_control_v3_rescore.json`,
`shared_pool_hf_control_interpretation.md`, `batch_reduction_AUTODEBUG.md`,
`AUTOFIX.md`, `batch_geometry_disposition_proposal.md`, and the unchanged
`shared_pool_generator_v5.json`. This is an owner disposition, not a
stage-review verdict or full-stage acceptance.
