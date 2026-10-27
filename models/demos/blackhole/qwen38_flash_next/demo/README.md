# Qwen3.8-Flash-Next demo

`text_demo.py` is the end-to-end test: the vLLM adapter's `Qwen38ForCausalLM` on a 1x4 Blackhole mesh at batch 1.
It generates 128 greedy tokens (no stop on EOS) for one input sequence length (ISL).

## Test ids

`traced_128`, `traced_256`, `traced_1k`, `traced_2k`, `traced_4k`, `traced_8k`, `traced_16k`, `traced_32k`,
`traced_64k`, `traced_128k`, `traced_256k`, and `determinism_128` (two runs, tokens must match).

## How the context is chosen

The resident context is the smallest of 32768, 65536, 131072, 262144 that holds the ISL. The model gets
`max_seq_len = context - 64`. The prompt length is `min(ISL, max_seq_len - 128)`, so the 128 generated tokens fit.
For 32k, 64k, 128k and 256k the prompt is a little shorter than the id name.

## Prompt and pass-key check

The prompt is one chat turn with thinking off. It holds an excerpt of War and Peace (Project Gutenberg), with the line
"The secret pass key is 58213." placed in the middle. The question asks for the pass key and a summary.
The test fails if `58213` is not in the decoded output. The perf line and the JSON report are written before this check.

The book downloads once into `$TT_CACHE_PATH/demo-context`. The prompt is built before the model loads.

## Environment

    export TT_METAL_TRACE_ALLOC_TRACKING=1 MESH_DEVICE=P150x4           # mesh and box type, see below
    export HF_MODEL=<checkpoint directory or Qwen/Qwen3.8-Flash-Next>   # a repo id resolves in the local Hugging Face hub cache; nothing is downloaded
    export TT_CACHE_PATH=<cache root>                                   # caches/<label>/{components,model-io}, caches/bf4-experts, demo-context
    export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0                   # only on hosts that need it
    export QWEN38_BF4_CORPUS=<corpus> QWEN38_BF4_CORPUS_VERIFICATION=<verification.json>   # only until bf4-experts exists

Each new context builds its own component cache on first use (`c<context>-vllm`, about 10 to 23 GB). That first run is
slow.

The demo derives the chip order like the server does. `MESH_DEVICE` selects the box type: `P300x2` is the QuietBox 2
(2x P300c); `P150x4` or `P150x8` is a p150 host. A 1x4 mesh uses the host's own profile when
`tools/hardware_profiles.py` lists the host, else `p150-x8` (the first four chips). If `TT_MESH_GRAPH_DESC_PATH` is
not set, the demo sets it from the profile.

## Commands

    pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -s -k "traced_128 and not traced_128k"
    pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -s -k traced_256k

`CI=true` also writes the benchmark JSON.

## Batched decode (`test_demo_text_batched`)

Ids: `batched_128_b8`, `batched_128_b32`, `batched_1k_b8`, `batched_1k_b32`, `batched_4k_b8`, `batched_4k_b32`,
`batched_8k_b8`, `batched_16k_b8`, `batched_32k_b8`, `batched_64k_b8` (ISL and users B). The context choice is the same
as above. `batched_64k_b8` may not fit in device memory.

B users decode together on the plain lane body (`tools/qwen38_batch_decode.py`) with paged KV (page size 64, pool =
1 + the users' pages for prompt + 128 tokens + 8 spare). All users are admitted first (B=1 prefill each, then import
into the lane). Then each user gets 128 tokens: t0 from the prefill and 127 batched steps. No stop on EOS.

Each user has its own prompt: the same ISL, a different non-overlapping War and Peace excerpt, and its own 5-digit
pass key (58213 for user 0). One prompt for all users would route the MoE to few experts and give a wrong speed.

What the test checks, per user: the first 8 tokens equal the B=1 reference (adapter prefill + 127 decode steps,
computed before the lane capture), the user's own pass key is in its output, and no other user's key is in it. The
log and the JSON report also give the exact match of all 128 tokens and the first difference.

The log line gives the mean TTFT per user, the admission time (prefill and import), the decode step in ms (mean, first
16 and last 16 steps), t/s/u, aggregate tok/s, the B=1 ms/token and the model-ready time. With `CI=true` the benchmark
JSON has `batch_size=B`; `decode_t/s` is the aggregate, `decode_t/s/u` the per-user rate.

The step time depends on the text: a step is faster when the users emit the same tokens (fewer routed experts). The
first 16 steps (every user writes its pass key) are the fastest; the summary steps, where the users differ, are the
realistic rate.

Extra env: `QWEN38_LANE_INDEXER_FORM=wide|per_lane` (default `wide`) picks the QSA lane indexer form.

    pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -s -k batched_4k_b8
