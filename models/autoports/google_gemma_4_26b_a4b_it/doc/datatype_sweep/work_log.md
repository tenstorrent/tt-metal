# Datatype sweep work log

## Contract and baseline — 2026-08-15

- Branch: `mvasiljevic/fmf/google-gemma-4-26b-a4b-it`; measured base/source
  commit: `bcb21b8d026464190a3ffceae72e420e2e026c56`. Measurements used the live
  datatype-sweep stage diff over that base. The exact diff is captured by the
  post-clean-review checkpoint commit below.
- Runtime environment: Python 3.12.13, repo-built TTNN, firmware 19.8.0, KMD
  2.8.0, `FABRIC_1D_RING`, fallback exceptions enabled, serialized hardware
  commands. Tensor/program caches may be warm; TTFT is retained but not ranked.
- Acceptance gates: top-1 >= 90%, top-5 >= 98%; top-100 retained as reported.
- Hardware health: `timeout 60 tt-smi -ls --local` found four P300C Blackhole
  devices. A 1x4 mesh open/close smoke passed.
- Main reference: `doc/full_model/readiness_aime24_chat.refpt`, checkpoint chat
  template, 100 tokens.
- Refreshed default prefill: 96% / 100% / 100% top-1/top-5/top-100.
- Refreshed default teacher forcing: 98% / 100% / 100%, 1269.64 ms cold TTFT,
  25.47 traced t/s/u. Policy-driven warmed reproduction: 382.99 ms TTFT and
  25.51 traced t/s/u with identical accuracy.
- Commands are recorded verbatim per row in `sweep_results.json`/CSV. Raw logs
  are retained below `artifacts/<config>/teacher_forcing.log`.

## Policy plumbing

- Added `tt/precision_policy.py`; the repo-local selected artifact is the
  default. `build_generator` forwards an optional explicit path and every full
  model resolves per-layer weight/fidelity policy, activation/residual, CCL,
  KV-cache, logits, and sampling assumptions.
- `Gemma4FullModel.precision_summary()` reports live tensor dtypes and compute
  configs. `artifacts/selected_token_out.log` contains the all-layer summary
  from the measured default construction path.
- The BFP8 KV smoke exposed a verified split API contract: `paged_fill_cache`
  requires cache-typed input, but tail `paged_update_cache` requires BF16/FP32.
  Tail slices now typecast to BF16. The exact original failure and passing rerun
  remain in `artifacts/kv_bfp8/teacher_forcing.log` and
  `artifacts/kv_bfp8/nonaligned_mixed_prompt.log`.
- A BFP8 residual candidate exposed embedding's row-major typecast restriction;
  requesting TILE output fixed construction. Its full-model accuracy still
  failed decisively, so it was rejected on evidence.

## Candidate results and decisions

| Config | Top-1 | Top-5 | Top-100 | Traced t/s/u | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline BFP8/HiFi2, BF16 KV/CCL | 98% | 100% | 100% | **25.51** | selected |
| BFP8 KV | 96% | 100% | 100% | 17.08 | pass, slower |
| BFP8 CCL | 98% | 100% | 100% | 18.23 | pass, slower |
| BFP8 attention LoFi | 94% | 100% | 100% | 25.47 | pass, slower/less accurate |
| BFP4 experts LoFi, inner layers | 95% | 100% | 100% | 23.53 | pass, slower |
| BFP4 experts HiFi2, inner layers | 94% | 100% | 100% | 18.92 | pass, slower |
| BFP4 dense down LoFi, inner layers | 95% | 100% | 100% | 22.10 | pass, slower |
| BFP8 activation/residual | 0% | 1% | 18% | 13.44 | accuracy fail |
| decode packed gate/up BFP4 HiFi2 | 96% | 100% | 100% | 18.16 | pass, slower than LoFi |

Every material BFP4 group considered has a real full-model LoFi run. Experts
also have a same-group HiFi2 comparison. The already-selected decode packed
gate/up BFP4 group has LoFi in the baseline and a HiFi2 control. Dense down was
considered/ran at BFP4+LoFi and rejected because it slowed by 13.4%; a higher
fidelity version was not needed to satisfy the BFP4+LoFi completeness rule.

## Post-selection and quality

- Normal default selected-config token-out command:
  `GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1
  GEMMA4_PROBE_PROMPT_LEN=128 GEMMA4_NO_HOST_TOKEN_OUT_BENCH=1
  GEMMA4_NO_HOST_WARMUPS=5 GEMMA4_NO_HOST_ITERATIONS=128
  GEMMA4_PRINT_PRECISION_SUMMARY=1
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' pytest -q -s
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe`.
- Result: 4.566510 s / 128 replays, 35.675863 ms/token, 28.030156 t/s/u,
  final position 262, zero token readbacks/refreshes and zero timed syncs.
- Six shared prompts ran through `run_autoregressive --chat-template` with the
  exact checkpoint, 64 greedy tokens, rendered IDs, and HF controls. Manual
  inspection: coherent haiku, supervised-learning explanation, story,
  thermodynamics explanation, French translation, and Python/Fibonacci answer.
  `check_degenerate_output` reports clean.

## Context and limitations

- BF16 selected KV: 2.548828125 GiB/device total KV; conservative total
  22.2853053 GiB/device; 9.7146947 GiB margin.
- BFP8 candidate KV: 1.2744140625 GiB/device total KV; conservative total
  21.0108912 GiB/device; 10.9891088 GiB margin. Mixed 33/47 prompts pass.
- Both preserve 262,144 tokens. The existing all-layer 262,111-token public
  non-aligned proof remains the selected BF16 evidence. No capability reduction.
- TTFT varies with cold/warm program and tensor-cache state and is not used for
  Pareto selection. Only trace-verified teacher-forcing decode t/s/u ranks rows.
- vLLM integration was not started. The selected loader is construction-path
  infrastructure intended for later reuse, not a vLLM adapter implementation.

## Review and checkpoint

- Initial independent stage review: `more-work-needed` because candidate rows
  omitted branch/base-SHA/environment provenance. Added those fields to every
  JSON/CSV row and this log; numerical and capability evidence required no
  change. Rereview verdict is recorded in `stage_review.md`.
- Local implementation/report checkpoint: `973be9f5451a9defaeb4ef5368fa1f239f70da34`
  (`Sweep Gemma4 full-model precision policy`). The follow-up evidence/metadata
  commit force-adds the repo-ignored CSV and raw logs and records this SHA.
  Neither commit was pushed.
