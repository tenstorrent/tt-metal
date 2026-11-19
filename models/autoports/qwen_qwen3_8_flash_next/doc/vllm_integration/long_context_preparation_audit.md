# P5 and maximum-context preparation audit

CPU/source/artifact inspection only, 2026-11-16. No HTTP, TTNN import, device
command, generation, or runtime source change was made for this audit.

**Use the planned explicit timeouts.** The shared P5 default request timeout is
3600 seconds; existing model timings forecast that a131072-token request can
take roughly5000–7450 seconds before its answer. The existing acceptance plan's
`--request-timeout 14400` and outer `timeout 43200` avoid that predictable
client-side interruption. These are estimates, not serving measurements or
upper bounds. There is no identified parser/tokenizer blocker requiring a
server restart, recipe change, or workload reduction.

## Exact P5 workload and scoring

Installed source:
`$TT_MODEL_BRINGUP_ROOT/runtime/readiness_check/run_long_context.py`, with
`served_probe.py` for HTTP bodies/stream handling.

- `--part accuracy` defaults to lengths8192,32768,131072 and depths0.1,0.5,0.9:
  nine sequential requests. The book exists at
  `models/ttt_compat/tests/tale-of-two-cities.txt.bz2`. Whitespace is collapsed;
  the book is repeated if needed and sliced by character count.
- Each fact is `The secret passphrase for the lighthouse keeper is WORD-NNNN.`
  where WORD is one of ten fixed words and the number is1000–9999. Seed is
  `length * 31 + depth_index`. Insertion depth is a fraction of **characters**,
  adjusted to a nearby sentence boundary, not an exact token-depth guarantee.
- The final question is: `What is the secret passphrase for the lighthouse keeper
  mentioned in the text above? Answer with the passphrase only.` The entire
  book/fact/question is one user message through `/v1/chat/completions`.
- Default thinking is off, with `chat_template_kwargs={"enable_thinking":false}`
  in both `/tokenize` and the actual generation request. Temperature is0 and
  the output cap512. EOS remains enabled. Loading `--profile lm_eval_reasoning`
  validates that profile but **does not apply top-k40** to accuracy: the local
  `chat_profile` has an empty request dictionary and explicitly overrides
  temperature0. Checkpoint top-k20 remains harmless for greedy device sampling.
- Token sizing uses the served chat tokenizer, so it counts the actual template.
  Four character/token fitting iterations plus up to12 shrink iterations aim
  for a prompt no larger than the target and within max(64,2% target) below it.
  The function does not assert final tolerance or positive count; inspect every
  recorded `prompt_tokens`, rather than treating nominal lengths as exact.
- At served C262144, reserve is512+64+floor(.02*262144)=5818, so the clamp ceiling
  is256326; all three default lengths remain unchanged. No request is intended
  to consume the advertised maximum. P5 does not replace maximum-context acceptance.
- Recall passes if the answer appears in either response content **or reasoning**,
  case-insensitively after removing hyphens and spaces. It is substring scoring,
  not exact-answer equality. A successful stream must still have a finish reason.
  Every depth must pass each length. A context-related HTTP400 is classified as
  a nonwaivable `probe_setup` defect, not a recall miss.
- The prefix-cache reset is best effort and its return status is ignored. That
  is compatible with this model's documented `recurrent_state` exclusion and
  disabled prefix cache; it does not prove cache reset or prefix-cache support.

The probe writes `doc/eval_readiness/long_context_accuracy.json` and
`long_context_accuracy/cases.jsonl` **only after all nine requests**. Its raw
case rows retain only the last200 response characters, the expected answer,
length/depth, error, finish reason and TTFT; they do not retain complete request
prompts or full streamed output. Preserve exact bodies/full responses and
per-case progress alongside the standard probe when collecting if possible,
without modifying its lengths/scoring. This avoids losing hours of evidence
on interruption and makes any required matched HF control unambiguous. A
model-local logging wrapper can delegate unchanged requests to the shared
probe; no change to its installed source is needed.

## Context, policy and capacity

`doc/context_contract.json` and selected
`doc/datatype_sweep/selected_precision_config.json` both retain262144. Selected
policy is `combined_shared4_mixed`, BF16 activations/residual/CCL/KV, page32,
TP4 local width256, routed/shared BFP4-LoFi with shared layer0/47 BFP8-HiFi4
exceptions, QSA/head BFP8-HiFi2 and GR/GDN-ZAB BFP8-LoFi. No KV layout change
invalidates the preserved context basis.

`capacity_full_v1.json` proves B2/C262144/8195 scheduler-owned blocks can allocate,
prepare and replay; it is not long-context execution evidence. The full-model
maximum artifact proves a real262143-token non-aligned prefill and one decode
to consumed context262144 under its historical recorded policy. Selected-policy
`datatype_sweep/final_validation_v1/context_smoke.json` proves32769 tokens and
two generated tokens with full262144 allocation, no captures, and unchanged
memory before/after. Stage9 must still perform its single served maximum run.

The current `tests/check_vllm_max_context.py` has appropriate guards: exact
context contract, one served model with matching `max_model_len`, valid token785,
exclusive persistent attempt marker, no retry, full response preservation,
usage262136+8=262144, eight in-vocabulary output IDs and finish_reason `length`.
Its prompt is non-aligned modulo32/256. Page demand fits the8195-block allocator
budget. The marker is deliberately created before `/v1/models`; a metadata
preflight failure leaves an attempt ledger but did not send the expensive request.
Do not silently delete it or auto-retry.

The HTTP request tests **public API prompt+output budget262144**. The final output
token is not fed back, so do not equate this with the full-model artifact's
`consumed_context=262144` internal position claim: that artifact emits two tokens
after262143 prompt tokens, a different generator/API length convention. The
helper's public serving boundary is correct; adding a ninth output would exceed
the advertised API budget.

## Time planning from existing evidence

| Existing evidence | Workload | Measured wall time |
| --- | --- | --- |
| Selected datatype context smoke |32769 input,2 generated, standalone B1|1248.605s|
| Historical full-model maximum phase |262143 input,2 generated, standalone B1|14895.095s|
| Current full48 tracked server log, short P2 prefills |105 input, one prefill row;22 observed submissions at audit|4.073s median submission-to-result|

Using the first two observed seconds/input-token rates solely as a planning
envelope gives roughly5.2–7.8 minutes per8K,20.8–31.0 minutes per32K, and
83.2–124.1 minutes per128K; nine P5 requests total approximately5.46–8.15 hours
before tokenization/answer overhead. The maximum request forecasts2.77–4.14h.
Neither linear extrapolation is an assured bound: policies, long-position
attention cost and serving dispatch differ. The max helper's21000-second read
timeout and21600-second outer timeout exceed the historical4.14h phase.

Do not estimate device runtime from intermediate `prefill_result` events:
the approved no-output path returns after host submission without waiting.
Current256-token intermediate submissions show about.017s while a later final
one-token result can absorb18.8s of queued work. These are asynchronous timings,
not miraculous long-prefill throughput. Use first chunk to terminal result for
whole requests, with actual starts and final sampling mode.

Before beginning long requests, finish P2/P3 work, establish no pending competing
load, and ensure the bounded86400-second server still has enough remaining
lifetime for approximately13h of conservative P5+max planning plus other work.
Do not start the single maximum near that timeout. Keep it on the same final
policy/server configuration; checkpoint prefill chunks do not authorize changing
context or scheduler workload midway through acceptance.

## Compatibility and completion checks

The served qualitative manifest already proves all six thinking-off prompts
match local/HF/server token IDs and actual generation IDs. P5 uses the same
request kwargs and the same renderer for counting/generation, so there is no
specific long-prompt BOS or reasoning-parser mismatch indicated by current
source. This is preparation evidence, not a P5 pass. The source's `/tokenize`
path returns `tokens` and `max_model_len` as expected.

After collection, inspect all nine actual lengths, recalls, finish reasons and
errors; count sampler routes/OOVs in the corresponding server interval; require
the maximum helper's completed passing artifact; preserve raw evidence; and run
`scripts/check_eval_readiness.py --model-dir <model> --stage vllm`. A recall miss
requires the skill's matched control/repair process; a timeout or server error
does not establish a hard physical capacity limit.

## Recording wrapper ready for collection

`tests/run_vllm_long_context_recorded.py` now delegates to the unchanged shared
P5 accuracy CLI. This is the replacement launch command for the shared command
in `acceptance_commands.md`; every argument following `--` passes through intact:

```bash
timeout 43200 python "$M/tests/run_vllm_long_context_recorded.py" \
  --record-dir "$M/readiness_vllm/P5_accuracy_raw" -- \
  --part accuracy --server-url "$SERVER" --model-dir "$M" --hf-model "$CKPT" \
  --server-log "$LOG" --lengths 8192,32768,131072 --depths 0.1,0.5,0.9 \
  --max-tokens 512 --no-thinking --request-timeout 14400 \
  > "$M/doc/vllm_integration/logs/P5_accuracy_final.log" 2>&1
```

The wrapper refuses an existing recording directory, never retries, records
shared module paths/source SHA256s and forwarded arguments, and preserves the
shared return code. Per request it retains the caller body, effective HTTP body,
HTTP metadata, raw response bytes as received, complete parsed response, and
timing/error state. `progress.jsonl` records starts, first bytes, progress at most
once per10 seconds of arriving data, and completions. Pending request metadata
and already received bytes remain available if the process is interrupted.
`/tokenize`, health and cache-reset behavior is untouched. Temporary hooks are
restored on normal completion and Python exceptions. A hard process kill may
leave the manifest marked running; process state remains authoritative.

CPU validation in `P5_recording_wrapper_validation.json` covers successful SSE,
an HTTP400 with a1200-byte body (full bytes retained while the shared summary
still truncates to300), partial-stream failure, uncaught exceptions, unchanged
arguments/result identity/exit code, hook restoration and no-overwrite behavior.
The actual shared parser accepts the full command above; no TTNN was imported.
No live HTTP or device request was made while implementing or validating it.
