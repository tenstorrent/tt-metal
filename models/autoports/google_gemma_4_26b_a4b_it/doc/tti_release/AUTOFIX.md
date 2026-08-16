# AutoFix Report

## Starting evidence

- The first no-Docker CI-nightly release wrote valid lm-eval JSON but reported
  `no eval results parsed (rc=0)` for `meta_ifeval` and `meta_gpqa_cot`.
- The same run skipped spec tests because Gemma 4 had no TTI server-test mapping.
- GPQA scored 0/10; eight responses ended mid-reasoning with `[invalid]` extraction.

## Hypothesis experiments

- **Release-row alias mismatch — verified.** TTI invoked public lm-eval tasks
  `ifeval` and `gpqa_diamond_cot_zeroshot`, while its parser searched only for
  stable report names `meta_ifeval` and `meta_gpqa_cot`. Alias-aware result
  matching now scores the public keys while preserving stable report-row names.
- **GPQA generation truncation — verified.** The aggregate config omitted
  `max_gen_toks`; all sampled responses were short and most stopped before the
  requested boxed answer. The task now uses a 32,768-token reasoning budget,
  still within the unchanged 262,144-token context contract.
- **Missing spec-test mapping — verified.** The filter had no Gemma 4 model or
  generic LLM matrix entry. The p300x2 mapping now selects
  `VLLMParamConformanceTest`.

## Verification

- Focused TTI tests: 40 passed.
- TTI fixes are committed on `codex/gemma4-stage11`; the live release rerun is
  the final integration proof.

## Final status

Harness defects are fixed locally. The subsequent exact HF control exposed a
separate model-correctness blocker: HF BF16 passes 10/10 GPQA while the TT
serving path remains below the mandatory 9/10 threshold.

## GPQA model-path investigation

- HF and TT use the same snapshot, tokenizer IDs, chat-template hash, seed,
  task subset, 262144-token context, and 32768-token task completion budget.
- Padding model decode to 32 rows made greedy trajectories depend on inactive
  lanes. Executing only scheduler-active rows improved the first-five finite
  probe from 2/5 to 4/5 and reduced wall time from 384 seconds to 86 seconds.
  The exact ten-document finite baseline remained 2/10, so this is a proven
  serving correctness/performance repair but not the GPQA accuracy solution.
- The serving precision override was previously ignored because the adapter
  hardcoded the selected config. `GEMMA4_PRECISION_CONFIG` now reaches the
  model constructor, enabling controlled policy isolation.
- On failing document 1, HF and TT prefill choose token ID 2021 with the same
  top-two ranking. Incremental HF decode and TT agree for generated IDs 0-14,
  then diverge at ID 15: HF selects 6608 and TT selects HF's second-ranked
  7395. Re-prefilling the shared 15-token prefix makes TT select 6608. The
  error is therefore iterative decode numerical drift, not prompt formatting.
- Traced and diagnostic eager TT decode produced the identical wrong 16-token
  trajectory. Concurrent-ten and standalone TT requests also matched exactly.
  Trace capture, scheduler batching, and token round-tripping are refuted.
- Exact ten-document, 1024-token diagnostic gates (flexible extract) were:
  selected policy 2/10; all-HiFi2 3/10; packed gate/up BFP8 plus HiFi2 3/10;
  packed gate/up BF16 4/10; packed BF16 plus BF16 attention 3/10; packed BF16
  plus BF16 dense-down 3/10; and unpacked dense gate/up 3/10.
- BF16 expert source weights cannot preserve the context contract: model load
  succeeds, but unchanged 262144-context KV allocation exhausts device DRAM.
  The context was not reduced. BF16 attention also requires a different
  decode sharding geometry because the shipped o-projection geometry clashes
  with L1 circular buffers; an interleaved diagnostic still scored only 3/10.
- ARC/active-Ethernet heartbeat failures encountered between repeated server
  launches were recovered with reservation-container `tt-smi -r`; they are
  infrastructure events and did not alter the accuracy conclusions.

## Current status

The model-path AutoFix hypotheses above are exhausted without reaching the
mandatory 9/10 GPQA gate. The best viable diagnostic policy is 4/10 and is not
retained as a release policy. Temporary precision and eager-decode overrides
were removed. Stage 11 remains blocked on TT decode numerical correctness.
