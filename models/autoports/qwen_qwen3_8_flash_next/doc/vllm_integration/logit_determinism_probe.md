# Selected-logprob reproducibility and canonical prefill-mask probes

The standalone full48 baseline passed on2026-11-16 (`readiness_vllm/logit_baseline.json`,535.33 s). The live-server comparison also passed with explicit served model ID and `--pause-batched-admission` (`readiness_vllm/logit_determinism.json`). They do not replace the optimized device-sampled serving checks. The standalone runner loads the current selected policy, all48 layers, B2 and diagnostic capacity1024 once. It emits full-vocabulary next-token logits for the same33/47-token histories used in `doc/full_model/prefill_factored_full48_v1.json`. Those historical snapshots contain full vocabulary rows (`mixed33[-1]`, `mixed47[-1]`) and prove solo/mixed/reversed equality for that older implementation, but do not record the subsequently selected precision policy; they are not a matching current baseline.

Run the baseline only after other device work has closed and before starting the final server, following `$tt-device-usage` ownership and bounded-execution rules:

```bash
timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  PYTHONPATH=/home/ttuser/qwen38-flash/tt-metal \
  /home/ttuser/qwen38-flash/tt-metal/python_env/bin/python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.probe_vllm_logit_baseline \
  --output models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/logit_baseline.json
```

The baseline saves `logit_baseline.pt` with raw logits and seed-state snapshots, plus JSON containing exact prompt IDs, source manifest, complete selected/runtime precision, physical slots, shape, top20 logprobs, comparisons and counters. Solo A, solo B, AB and BA run twice. Full-vocabulary repeat, reversed-position and solo-versus-batch comparisons require exact tensor equality. `argmax` and `log_softmax` here analyze diagnostic host logits; they do not implement serving sampling.

The same load also compares canonical on-device sampled unchunked prefill with a32-token all-intermediate chunk followed by its final remainder, plus mixed false/true rows in both submission orders. All sampling enters public `prefill_forward` and its existing `_sample`. The false rows deliberately carry top_k40/presence_penalty1 while final rows use top_k20/temperature0.8/top_p0.95 and fixed per-request seeds53/79. Gates require unchanged false-row seed state, zero sampling replays/readbacks for all-false chunks, expected final-row sampling/readback counts, exact emitted-token agreement, and exact final seed words on every rank. Reading seed words is explicit diagnostic instrumentation outside the generator's readback counters.

Then, on the final server with event logging and no unrelated client traffic:

```bash
/home/ttuser/qwen38-flash/vllm-env/bin/python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.check_vllm_logit_determinism \
  --server-log <final-server.log> --pause-batched-admission \
  --model <model-id-returned-by-v1-models> \
  --standalone-baseline models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/logit_baseline.json \
  --output models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/logit_determinism.json
```

The client uses `/v1/completions`, exact raw token histories, greedy generation, one output and `logprobs=20`. This intentionally requests the **optional host-logprob path** on the four-chip mesh. Every request body, raw response, parsed response, prompt ID, shape and server-event interval is retained. Token-ID keys avoid collisions between different IDs with identical text. It requires exact sampled-token agreement, no non-tied rank inversions and selected-logprob absolute error at most1e-5. That narrow tolerance only covers float32 log-softmax/serialization; it is not a BF16 model-error allowance. Exact-map equality and actual maximum deltas remain visible. Different cutoff members are accepted only when tied at the returned cutoff; no synthetic full-logit values are created.

Run-to-run and reversed-order results are separate from descriptive solo-versus-batch results. Server events must demonstrate both histories at both submitted batch indices and physical slots, with actual B2 final prefills starting at0 in every AB/BA request. API order alone is insufficient. If the scheduler serializes a nominal API batch, the probe retains results but fails the physical-position gate; inspect admission/batching before claiming cross-position coverage. Source manifests and selected policies must match the baseline. `--replay-artifact <prior.json>` reanalyzes responses without HTTP, allowing a later baseline comparison without repeating device work.

The OpenAI path exposes selected normalized logprobs, not full logits: it cannot prove unreturned vocabulary equality or distinguish a uniform additive shift of every logit. The paired standalone artifact proves full-vocabulary equality within standalone execution. If stage review requires full-vocabulary equality *through serving*, this client cannot establish that gate with the existing selected-logprob response; retain this limitation and resolve with explicit diagnostic evidence rather than call top20 evidence full logits. One-token requests also do not establish steady-decode determinism; the existing async/stale-token/page-table probes own that coverage.

Live result: all numerical comparisons passed even in the first executed run, but its physical B2 coverage failed because admission split an API batch. That artifact remains `logit_determinism_unbatched_admission.json`. The accepted run pauses only before paired admission with cache state retained, resumes before execution, and proves both actual physical row positions. Original model-ID404 failure is `logit_determinism_wrong_model.json`. No source-manifest or tolerance gate was weakened.
