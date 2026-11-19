# Host sampler control audit

The `--hf-sampler vllm_host` diagnostic is equivalent to the serving sampler **conditional on the same full-vocabulary logits and committed generated-token history** for this profile. I found no concrete sampler, masking, presence-penalty, or per-request RNG mismatch that explains the 2/10 HF versus 6/10 HTTP quality-failure result. In this installed CPU build, differing sampling APIs do **not** imply differing random draws: `torch.multinomial(..., 1)`, upstream vLLM exponential sampling, and the plugin's prefetched exponential sampling produced identical tokens and identical RNG states in the controlled test below.

The quality gate remains failed. This finding narrows the next comparison to actual sequence-matched logits and actual serving history/state; it does not prove model arithmetic is the cause or waive serving correctness.

## Scope and identity

Read-only source/CPU investigation under AutoFix. No model was loaded or executed by this audit; no TTNN or vLLM package was imported; no device operation, server interaction, reset, production-source edit, or collector edit occurred. The only new files are this report and its adjacent CPU diagnostic script/results.

Paths below use `P = ../vllm-tt-plugin/src/vllm_tt_plugin`, `V = python_env/lib/python3.12/site-packages/vllm`, and `M = models/autoports/qwen_qwen3_6_35b_a3b`, relative to tt-metal. Plugin HEAD is `5d1785b76d17a8989c9495f6939abea6cef62f09`, with clean plugin worktree. Installed vLLM is 0.31.0 and executed Torch is `2.11.0+cpu`.

The audited request is the pinned revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`, “Say hello to me,” thinking enabled, temperature 1, top-p 0.95, top-k 20, presence 1.5, frequency 0, repetition 1, 1024 generated-token cap, and seeds 0–9. All twelve forbidden token encodings in the saved artifact are singletons. The frozen `card_unblinded_summary.json` records a 40-point quality-failure excess against a 20-point allowed margin.

## What the diagnostic controls correctly

| Boundary | Source-backed conclusion |
|---|---|
| Logit dtype | HF generation converts its last-token logits to float32 before processors (`transformers/generation/utils.py:2810`). The diagnostic explicitly uses float32. Serving reads model logits into independent float32 storage (`M/tt/serving_state.py:526`) and upstream `Sampler.forward` also casts to float32 (`V/v1/sample/sampler.py:106`). Thus penalty/filter/sampling arithmetic uses float32 on both sides. The recorded TT model emits BF16 logits; converting to float32 does not undo model-side rounding. Matching sampler dtype is not evidence that raw model logits agree. |
| Presence penalty | Actual upstream penalty code subtracts `presence * (output_count > 0)`, once per generated token type (`V/model_executor/layers/utils.py:43`). Prompt tokens affect repetition, not presence. Diagnostic `unique(input_ids[prompt_len:])` is exactly the intended set. Frequency 0 and repetition 1 are identities. Empty history is valid on the first generated token. |
| Bad-word ordering | Serving masks before applying penalties; the diagnostic subtracts presence before the sampler masks. For this finite penalty and singleton bans these commute: masked logits remain negative infinity. Both call the same upstream bad-word processor. No prompt-history ambiguity exists for singleton bans. |
| Temperature/filtering | Both instantiate `create_host_sampler()` and use temperature 1, top-k 20, top-p 0.95. The plugin temperature-1 fast path is an identity. There are no requested logit biases, min-p, whitelist, grammar, custom processors, or positive min-tokens in this profile. |
| Seeded generator | Serving constructs a CPU generator seeded once per request (`P/input_batch.py:151`) and retains it while remapping batch rows (`P/model_runner.py:1417`). Diagnostic constructs the same generator once per HF seed. Both use default FP32 exponential noise. Prefetched noise advances clones; only consuming a sampled token commits the corresponding next state (`P/host_sampler.py:39,117`). |
| HF handoff | Diagnostic returns negative infinity everywhere except the selected token, whose score is zero. It sets HF `do_sample=False`; HF selects argmax (`transformers/generation/utils.py:2839`), so the driver consumes no additional sampling RNG. The checkpoint generation config has no other nontrivial processor that changes this one-hot result. |

## Executed CPU experiment

Command from tt-metal:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/host_sampler_control_cpu_audit.py
```

Result: **PASS**; machine-readable output is `host_sampler_control_cpu_audit.json` beside this report.

The script extracts the actual audited function/class bodies with AST, avoiding platform/plugin imports. It uses vocabulary 248320, seeds 0–9, 16 sequential draws per seed, temperature 1, top-p 0.95 and top-k 20. The processor test alternates FP32 synthetic logits with BF16-rounded logits converted to FP32, and includes explicit repeated output types. It compares the collector's extracted `shared_host_processor` against the upstream serving penalty/mask path and plugin sampler:

- 160/160 processed full logit vectors exactly equal, including banned negative-infinity positions.
- 160/160 selected tokens exactly equal.
- 160/160 complete per-request generator states exactly equal.
- A separate same-probability comparison gives 160/160 identical tokens for Torch multinomial, upstream exponential sampling, and plugin prefetched sampling, with identical RNG states after every draw.

Boundaries: platform-dispatch construction was replaced by the known native CPU path; CPU output-history padding was reproduced locally; actual upstream CPU repetition-penalty math was retained. Synthetic logits do not validate the TT model, actual serving state, scheduling, or arbitrary profiles. This experiment explicitly checks that neither `ttnn` nor `vllm` entered `sys.modules`.

The independently running model diagnostic adds real-output corroboration. At inspection its artifact `M/readiness_vllm/sampling_controls/hf_shared_sampler_card_1024.json` was still `RUNNING`, with three completed rows and resolved HF dtype `torch.bfloat16`. Seeds 0, 1, and 2 exactly matched their prior native HF Transformers token sequences, respectively 369, 424, and 250 tokens. I verified those arrays directly. This is a partial-run observation, not a claim about unfinished seeds. For seed 0, the HTTP sequence has 770 tokens and first differs at zero-based generated index 44. Therefore a generic “different HF and vLLM RNG algorithm” explanation is already contradicted for that completed comparison.

## Could serving history lag or omit repeated types?

No supporting source-level defect was found for these sequential requests:

1. `InputBatch.make_output_token_ids_tensor` reads every committed token from `token_ids_cpu[prompt_len:num_tokens]` (`P/input_batch.py:742`). No recent-window truncation or deduplication occurs there. `penalty_history` only removes `-1` padding (`P/model_input.py:283`).
2. Every sampled token updates the persistent token array, count, and request output list (`P/model_runner.py:4618`). Decode preparation builds history when either bad words or penalties are present (`:2680`), and the host-sampling metadata receives that history (`:4370`).
3. Both presence penalties and bad words prohibit the steady decode path (`P/async_decode.py:614`). A nonsteady next step drains outstanding work (`:947`; `P/model_runner.py:3213`). `build_model_input` applies completed sampled tokens after scheduler lifecycle updates and **before** building the next model input/history (`P/model_runner.py:2881`). Thus the intentional one-step stale overlap used elsewhere is not enabled for this request profile.
4. Seed ownership follows the request; intermediate-prefill discarded samples use cloned generators. The actual 14-token prompt is below the recorded prefill budget, so ordinary chunked-prefill suppression is not a plausible explanation for this particular first divergence.

These are implementation checks, not runtime assertions about the captured HTTP requests. A history capture would still be useful if logits fail to account for the divergence.

Presence 1.5 does not grow with repeated occurrences. Once a token type has appeared, its future subtraction remains 1.5 even after hundreds of repeats. A sustained loop therefore does not by itself demonstrate a missing or stale presence penalty. Even a one-token history lag would only change presence on the *first occurrence* of that type, not on repeated occurrences of already-seen types; that makes blanket attribution of long loops to a one-token lag weak without an actual first-divergence trace.

## Minimal next discriminating evidence

For seed 0, take the common prompt plus the first 44 generated tokens. Compare the HF and TT next-logit vectors for that exact prefix, preserving the intended incremental/cache execution path. Record at the sampler boundary: request ID, generated-history length and exact IDs, full raw logits before mutation, all sampling parameters, forbidden IDs, and complete CPU generator state. Replay both vectors through the audited CPU sampler with the same history and generator state.

- If the recorded serving history or RNG state differs from expected state after 44 draws, that localizes a serving-state defect.
- If history and RNG agree but the two raw-logit vectors yield the recorded different tokens, the divergence is upstream of the sampler. Inspect the penalized/top-k/top-p support and probability/noise winner margin before claiming a large model error: a small numerical change near a sampling boundary can change the token and then the entire free-running trajectory.
- If identical raw logits/history/RNG do not reproduce the serving token, capture the precise runtime sampler inputs before blaming model math.

Do not change the failed blind scores, disable thinking, change bans, or modify the sampling policy to make this comparison pass. No repair is justified by this audit alone.
