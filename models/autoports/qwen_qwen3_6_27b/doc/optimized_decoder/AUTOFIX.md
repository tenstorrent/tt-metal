# AutoFix report

## Starting evidence

- Fresh source-only report: `AUTODEBUG.md`.
- Original failure:
  `python models/autoports/qwen_qwen3_6_27b/tests/full_attention_real_pcc.py --candidate default`
  produced official-weight PCC 0.078047.

## Hypothesis experiments

- Hypothesis: the packed projection is numerically wrong.
  Experiment: compare the real-weight TT packed output and split outputs with
  the corresponding host matmul.
  Result: BFP8 projection PCC 0.999841; QKV and gate split PCC 0.999841.
  Verdict: refuted.

- Hypothesis: `nlp_create_qkv_heads_decode` misreads the width-sharded packed
  result.
  Experiment: compare the extracted V heads before and after converting QKV to
  L1 interleaved.
  Result: V-head PCC -0.020282 before, 0.999837 after.
  Verdict: verified.
  Fix: add the narrow interleaved consumer boundary before head creation.
  Verification: official layer no longer has the near-zero attention branch.

- Hypothesis: low precision alone explains the original failure.
  Experiment: after the layout fix, compare BFP8 attention with BF16 attention,
  holding the BFP4 MLP policy.
  Result: BFP8 attention PCC 0.898606 versus the functional official-weight
  control; BF16 attention PCC 0.997086; BFP8 KV retains 0.997073.
  Verdict: verified as a second policy constraint, not the original cause.
  Fix: full-attention default uses BF16/HiFi4 attention, BFP4/LoFi MLP, BFP8 KV.

- Hypothesis: q_proj is flat `[all-q, all-gate]`.
  Experiment: inspect HF extraction and run the official dense layer.
  Result: HF stores `[q,gate]` per head; the completed functional decoder's
  flat split reaches only PCC 0.687928 against HF.
  Verdict: verified.
  Fix: optimized weight setup repacks per-head q/gate into
  `[all-q,k,v,all-gate]`; the completed functional file was not edited.
  Verification: optimized official HF layer PCC 0.997646.

## Final status

Fixed. Both durable real-weight tests pass, synthetic prefill/decode tests pass,
and the accuracy-safe final default still improves traced B1 and B32 decode.
The original and follow-on hypotheses are recorded in `AUTODEBUG.md`.

## Second-review optimization continuation

The later stage-review findings were performance-evidence blockers, not a
recurrence of the official-weight bug above. The authorized AutoFix
continuation tested them as follows:

- Hypothesis: larger final-policy gate/up K blocks improve the dominant BFP4
  MLP. Widths 10 and 20 were run at B1 and B32. Width 10 collided with
  persistent L1; width 20 required a 2,690,816-byte static CB on a
  1,572,864-byte worker. Refuted by exact device evidence.
- Hypothesis: same-dtype HiFi2 improves MLP quality enough to justify cost.
  Official PCC was unchanged, while traced latency regressed from
  1.268/1.454 to 1.674/1.859 ms. Refuted.
- Hypothesis: a coherent four-core residual/storage grid improves the full
  layer. The adapted end-to-end candidate produced exact L1/CB collisions at
  both batches. Refuted.
- Hypothesis: the remaining linear recurrent matmuls are best left to the
  framework-selected program. Explicit w1/w2/w4, 1x2-subblock, and HiFi4
  candidates were measured at both batches. Width-4 HiFi2 reduced whole-layer
  latency to 1.939/16.845 ms and was kept.
- Hypothesis: moving recurrent input A into L1 can address the remaining
  profiler advice. The 1-D contract requires fused batch for sharded A, while
  recurrent B has batch 48/1536 and fused batch requires B batch one. At B32,
  Tile32 A would also require 3 MiB/core over the four-core family. Refuted by
  exact validator and capacity evidence.

All structured results are in `artifacts/candidate_matrix.csv`,
`artifacts/candidates/`, `artifacts/tracy/`, and
`artifacts/program_contracts.json`.

## Recurrent-state dtype continuation

The fresh stage review identified the persistent FP32 gated-delta recurrent
matrix as an untested precision/movement boundary. Source inspection verified
that the functional loader allocates `[batch, 32, 128, 128]` in FP32, while
the chunked prefill scan already expands its initial state to BF16 and computes
the affine scan in BF16 before writing FP32.

- Hypothesis: BF16, BFP8, or BFP4 persistent state reduces recurrent-state
  movement without changing the selected packed projection, outer-product,
  width-4 recurrent program, MLP, or fidelity policy.
- Focused source experiment: independently selectable `linear_state_bf16`,
  `linear_state_bfp8`, and `linear_state_bfp4` policies. Setup converts the
  physical allocation, inherited prefill expands to its proven FP32
  destination and compresses each completed chunk, and decode explicitly
  expands to BF16 math then writes back the selected dtype.
- Focused runner:
  `tests/linear_recurrent_state_transition.py` crosses a non-aligned 129-token
  prefill and advances the same physical state for eight decode steps. It
  records every HF PCC, physical cache dtype/nonzero state, B1/B32 eager
  latency, repeated-run bit exactness, and watcher status.
- Source-only verification: `python -m py_compile ...` passed and
  `pytest -q tests/test_optimized_decoder.py` passed 94 tests.
- Verdict: verified after serialized hardware experiments. FP32, BF16, BFP8,
  and BFP4 were measured at B1/B32. BFP8 is the fastest candidate that passes
  the official-weight transition (minimum PCC 0.997965); faster BFP4 fails at
  0.993340. The selected BFP8 path passes non-aligned and S=513 transitions,
  repeated B32 bit-exactness, traced B1/B32 decode, watcher stress, and
  S=192511 capacity.

## Final-prefill profiler continuation

The next independent review found that the compact linear-prefill profiles
still represented the earlier FP32-state path. AutoFix extended the exact
provenance wrapper to prefill and regenerated all four final full/linear B1/B32
windows. The final linear reports record the BFP8 policy and contain both
BFP8-to-FP32 expansion and FP32-to-BFP8 writeback per iteration. Final linear
wall/device times are 10.983062/10.519938 ms at B1 and
275.380219/275.033794 ms at B32; both remain faster than the functional
baseline.

## Independent linear-projection precision continuation

The next fresh review found that the packed-input and output projections were
still fixed to the functional BF16/HiFi2 policy and had not been swept as
independent numerical boundaries.

- Hypothesis: independently reducing packed-input and output weight dtype and
  compute fidelity can reduce the remaining DRAM-sharded decode time while the
  BFP8 recurrent-state policy remains fixed.
- Focused source experiment: four independent policy fields now control input
  weight dtype/fidelity and output weight dtype/fidelity. Retained candidates
  isolate BF16/LoFi, BFP8/HiFi2, BFP8/LoFi, and legal BFP4/LoFi changes from a
  BF16/HiFi2 baseline, plus cumulative candidates.
- Synthetic verdict: all B1/B32 ten-step candidates pass. The BF16/HiFi2
  baseline is 1.925025/16.198853 ms. Input-only BFP4/LoFi reaches
  1.765712/16.027662 ms; output-only BFP4/LoFi reaches
  1.873437/16.138073 ms; cumulative BFP4/LoFi reaches
  1.710466/15.986414 ms.
- Official-weight verdict: input BFP8/LoFi, input BFP4/LoFi, and cumulative
  BFP4/LoFi prefill-to-decode transitions have minimum PCC
  0.997965/0.997432/0.997175, all above the 0.995 bar. The cumulative policy
  is selected.
- Profiler verdict: BF16/HiFi2, input-only BFP4/LoFi, output-only BFP4/LoFi,
  and cumulative BFP4/LoFi device times are respectively
  1.776752/16.146869, 1.615684/15.989006, 1.719812/16.089277, and
  1.562287/15.933102 ms at B1/B32.
- Final verification: official decode PCC 0.998677; ten-step traced decode
  1.709965/15.990722 ms; B32 watcher clean; fresh B32 decode and prefill are
  bit-exact. The final decode and prefill profiler windows were regenerated,
  and the S=192511 capability is unchanged because projection storage does not
  alter the public context or cache geometry.

## Precision-locked projection-geometry continuation

The next review found that the promoted BFP4/LoFi projection policy still used
the earlier input width 2/output width 3 geometry without a precision-locked
block sweep.

- Hypothesis: wider legal K blocks improve the two material DRAM-sharded
  projection rows without changing precision, recurrent state, recurrence,
  MLP, layout, or context semantics.
- Experiment: packed-input widths 1/4/5/10/20 and output widths
  1/2/4/6/8/12/24 were run at B1/B32 from the exact BFP4/LoFi, BFP8-state
  baseline. Leading output widths 8/12/24 were crossed with input width 5.
  Every passing contender has ten-step JSON and compact Tracy/perf evidence.
- Result: input width 5 is the isolated winner. Input width 10 overlaps an L1
  buffer at byte 1,357,824 with a static-CB end at 1,422,976; width 20 grows
  static CBs to 2,587,136 bytes versus 1,572,864 bytes of L1. Both exact
  blockers reproduce at B1/B32.
- Result: every legal output width passes. Cumulative input width 5/output
  width 12 is fastest traced B1 and best cumulative B32:
  1.670349/15.942844 ms versus the width-2/3 precision baseline
  1.710466/15.986414 ms. Focused device time is
  1.521726/15.890707 ms.
- Adapted storage-grid experiment: the precision-locked four-core candidate
  hits an exact L1/static-CB collision at B1 and B32 and is rejected.
- Output-subblock verdict: refuted as a user-tunable opportunity for this
  family. `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` exposes only
  `in0_block_w`, `per_core_M`, `per_core_N`, and fused activation. The factory
  internally selects 1x8 for packed input and 1x7 for output; the profiler's
  missing-subblock advice is an introspection limitation. Exact source/API
  contracts and static tests are retained in `program_contracts.json`.
- Fix: default now selects input width 5/output width 12. Final official
  decode PCC is 0.998717 and real transition minimum PCC is 0.997167. Final
  traced B1/B32 is 1.670179/15.949088 ms; final profiler device time is
  1.521271/15.893577 ms. Watcher is clean and B32 decode/prefill are bit-exact.
