# AutoDebug: Gemma 4 GPQA HF/TT divergence

## Finding

The available artifacts do not prove one source bug, but they narrow the
divergence to the generated autoport's long, concurrent decode path (including
its numerical policy) rather than the GPQA task definition or chat template.
The exact 10-document HF BF16 control scores 10/10 while TT scores 4/10.
No sample text is needed for this conclusion.

Both aggregate results use `gpqa_diamond_cot_zeroshot`, zero few-shot,
deterministic generation, seed 42, `max_gen_toks=32768`, and context 262144.
They record the same chat-template SHA
`ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`,
the same BOS/EOS/PAD IDs (2/1/0), and the same 10-of-198 subset. This strongly
demotes a template or dataset mismatch, although only a token-ID comparison can
fully close the request-serialization question.

## Ranked hypotheses

### 1. Long autoregressive numerical divergence (high likelihood)

The HF control is direct BF16 Transformers generation. The autoport instead
uses BFP8 attention/MLP weights, BFP4 packed decode MLP weights, LOFI full
attention and dense/expert compute, and a sharded on-device greedy sampler
(`doc/datatype_sweep/selected_precision_config.json`). GPQA chain-of-thought is
far more sensitive to an early greedy-token flip than short instruction tests:
one flip changes the entire remaining reasoning trajectory and final choice.
The preserved optimized-vLLM qualitative evidence exercises only about 13--209
generated tokens; it does not validate thousands of consecutive greedy steps.

Focused experiment: replay the same ten requests at concurrency one while
recording only token IDs and the first HF/TT divergence index (no decoded text).
At that index compare TT terminal logits against HF BF16, then A/B the existing
host-logits path and progressively safer policies (BF16 decode MLP first, then
HiFi attention/MLP). A first-token or very early mismatch implicates base
numerics; a divergence that grows with decode index implicates accumulated
precision error. Do not change prompts, context, or output limits.

### 2. Long-generation cache/position or active-row lifecycle defect (high likelihood)

The adapter combines vLLM-owned hybrid page tables with a 1024-token sliding
cache, 128-token full-attention blocks, traced device-feedback positions, and
active-row packing (`tt/generator_vllm.py`). Existing focused evidence covers
page boundaries and two-request churn only for short outputs (the integration
notes cite a 96-token long response). It does not establish correctness across
repeated 1024-token sliding-cache wraps or many thousands of decode iterations.

The definitive TT progress log is especially suspicious: six requests finish
over roughly 25 minutes, then the last four are reported together at 3:20:54.
That pattern is compatible with several requests reaching a common output cap
or with correctness/liveness changing after active-row compaction; it is not
proof of either. The adapter recaptures for non-identity `slot_remap`, but relies
on the worker having already packed tokens, positions, tables, and sampling
parameters into decode-row order. That cross-repository contract is not proven
for long request churn by the scoped artifacts.

Focused experiments:

1. Run the ten requests serially. If 10/10 returns, sweep concurrency 2, 4, 8,
   and 10 and correlate failures with completion/remap events.
2. Run a teacher-forced token-ID parity test through positions 1023/1024,
   2047/2048, 4095/4096, and the observed first divergence. At each boundary
   compare eager versus traced decode and inspect page-table rows and absolute
   positions without logging text.
3. Preserve per-request prompt-token count, completion-token count and
   `finish_reason`. Any failing row ending at exactly 32768 tokens should first
   be classified as no-EOS/cap exhaustion; do not treat it as a parser failure.
4. Add a long staggered two-request test where one row exits before and after a
   sliding-cache wrap; compare both token-ID streams with isolated controls.

### 3. HF-direct versus OpenAI API backend/version semantics (medium likelihood)

The HF control uses `--model hf`, Transformers 5.10.2 and batch size one. TT
uses `local-chat-completions`, Transformers 5.15.0, `tokenizer_backend=huggingface`,
and `num_concurrent=32`. The TT log also states that tokenized requests are
disabled, so lm-eval does not locally verify context plus generation length.
The identical recorded template SHA and special IDs make a gross template
difference unlikely, but they do not prove identical final token IDs or stop
handling after the API/server boundary.

Focused experiment: for each document, hash the final input token-ID vector
produced by the HF path and the server path and record its length, BOS count and
last eight token IDs. Also record the normalized OpenAI request fields and
response `finish_reason`. Repeat the HF control under Transformers 5.15.0 and,
if available, route an HF backend through the same OpenAI API adapter. This
separates tokenizer/version effects from TT model execution without exposing
prompt contents.

### 4. Metric/filter mismatch (low likelihood)

Both result files embed the same task prompt, `boxed_choice` flexible filter,
and task version 1.0. Both strict-match scores are zero, while flexible extract
is 1.0 for HF and 0.4 for TT. Therefore the score gap is not explained by the
known strict regex. Still, retained sample metadata should record only whether
a boxed choice was extracted, whether it matched, output length and finish
reason; this will distinguish wrong reasoning from missing/truncated boxes.

## Fastest discriminating sequence

1. Hash and compare input token IDs across HF-direct and API paths.
2. Re-run the same ten TT requests at concurrency one, preserving token counts,
   finish reasons, extraction-valid bits and first-divergence indices.
3. If concurrency changes results, isolate row compaction/page-table lifecycle.
   If it does not, A/B host logits and safer precision, then probe cache-wrap
   boundaries around the first divergence.
4. Only after the causal branch is known should the full TTI eval be repeated.

## Limits of this diagnosis

Raw samples were unavailable, so the current aggregates cannot say which six
documents failed, whether their boxes were absent versus wrong, or where token
streams first diverged. The 3:20:54 completion wave is a useful lead, not a
finding. No implementation edit or hardware run was performed.
