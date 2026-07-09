---
name: full-model
description: "Validate and harden the existing full DiffusionGemma block-diffusion model and generator: checkpoint construction, prefill, denoise, commit, multi-block state, tracing, correctness, context, and performance. Never build an autoregressive token-in/token-out generator."
---

# DiffusionGemma full model

Load `diffusion-gemma` first. The full model already exists; this skill
validates and hardens it instead of assembling a greenfield autoregressive
stack.

## Existing implementation

Use the current paths:

- `tt/model.py` — `DiffusionGemma4Model` over the reused Gemma-4 backbone;
- `checkpoint.py` / `weight_mapping.py` — checkpoint construction;
- `tt/denoise_forward.py` / `tt/denoise_loop.py` — denoise engine;
- `tt/sampling.py` — on-device decision path;
- `tt/self_conditioning.py` — denoise-only recurrent state;
- `tt/commit_decode.py` / `tt/commit_batched.py` — committed KV append;
- `tt/generate.py` — prompt→blocks→text entry points.

Do not create a second model wrapper, embedding stack, LM head, token-feedback
loop, or generic `Generator` contract. Do not edit `models/demos/gemma4/`.

## Generation contract

One request executes:

1. tokenize and causally prefill the prompt;
2. initialize a random-token canvas;
3. run up to 48 bidirectional denoise steps against frozen prompt/committed KV;
4. apply Gumbel-max, entropy-budget accept/renoise, and self-conditioning;
5. commit the clean 256-token argmax canvas once;
6. advance position by 256 and repeat for the next block;
7. trim EOS/length and detokenize at the API boundary.

There is no per-token `tt_out_tok`, greedy/top-k feedback loop, or position
increment within a canvas. The recurrent state is the canvas, previous logits
for self-conditioning, denoise step, frozen KV prefix, and block position.

## Full-model validation

Validate the normal constructor and entry points with real weights:

- all 30 layers on QB2 `(1,4)`, TP=4;
- causal prefill and frozen-prefix KV collection;
- full-vocab denoise logits and self-conditioning;
- one complete denoise→commit block;
- at least two committed blocks with position advancement;
- short, non-aligned, long, and 256K-allocation contexts;
- argmax and chunked-Gumbel memory strategies;
- deterministic injected-noise replay.

The context window is `prompt + generated <= 262144`; it is not a requirement
to prefill a 256K prompt. Preserve whole-block physical commit semantics when a
host-visible `max_new_tokens` request trims the final block.

## Correctness

Use the torch/HF diffusion path with identical initial canvas, Gumbel noise,
and renoise tokens. Record:

- per-step clean/Gumbel argmax agreement;
- entropy PCC and maximum absolute error;
- accept/renoise IoU;
- canvas agreement;
- final committed-token agreement.

Teacher-forcing top-k and AIME24 are not full-model gates. RUN-first evidence
may accept EOS-heavy or degenerate text, but a release-quality claim must
resolve or explicitly disposition #48291.

## Trace and control flow

The default optimized path replays a static traced denoise sequence. Every
captured trace keeps fixed shapes and operations; accept/renoise decisions stay
as tensors.

The opt-in early-halt controller may replay one-step/window traces and read one
halt scalar between replays. It must never branch inside capture. Under #48291
the current gate does not halt and remains default OFF.

Canvas feedback must stay device-resident between denoise steps. Do not
reconstruct the canvas, cutoff, or full logits on host.

## Performance

Report:

- prefill TTFT;
- milliseconds per denoise step;
- steps per block;
- commit latency;
- milliseconds per block;
- tokens-per-block/second;
- complete generation latency.

Never report `1000 / mean_tpot_ms` as DiffusionGemma throughput. Use the same
warmed traced workload for comparisons and distinguish shipping defaults from
opt-in candidates.

## Runtime and memory rules

- Keep runtime forwards free of hidden torch or TT↔host conversion.
- Use chunked/no-materialize Gumbel at 256K; a full materialized vocab-noise
  tensor does not fit.
- Include weights, frozen KV, canvas scratch, masks, trace buffers, CCL buffers,
  and fragmentation in capacity accounting.
- Preserve arbitrary valid prompt lengths through internal padding/chunking.
- Preserve larger-batch correctness where implemented; do not claim concurrent
  vLLM batching before paged cache ownership and batched canvas decode exist.
- Run watcher separately from profiler/device-time evidence.

## Required evidence

Use:

- `models/experimental/diffusion_gemma/tests/test_generate.py`;
- `models/experimental/diffusion_gemma/tests/test_tt_generate.py`;
- `models/experimental/diffusion_gemma/tests/test_device_text_demo_run.py`;
- `models/experimental/diffusion_gemma/demo/replay_hf_tt.py`;
- the appropriate `doc/<stage>/` directory and `doc/context_contract.json`.

The authoritative RUN outcome is `DG_TEXT_DEMO_SUCCESS` or
`DG_TEXT_DEMO_FAILURE`, not incidental fallback logging.

Done means the existing full model constructs from the real checkpoint,
executes prompt→multi-block text on QB2, preserves the context/state contract,
has trace and memory evidence, and reports diffusion-specific correctness and
performance without an autoregressive fallback.
