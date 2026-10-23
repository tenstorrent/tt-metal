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

The book downloads once into `$QWEN38_CACHE_ROOT/demo-context`. The prompt is built before the model loads.

## Environment

    export TT_METAL_TRACE_ALLOC_TRACKING=1 MESH_DEVICE="(1, 4)"
    export QWEN38_HARDWARE_PROFILE=tt-quietbox-2              # box type, see below
    export MODEL_WEIGHTS_DIR=<checkpoint directory>          # or HF_MODEL=Qwen/Qwen3.8-Flash-Next with HF_HOME populated
    export QWEN38_CACHE_ROOT=<cache root>                    # caches/<label>/{components,model-io}, caches/bf4-experts, demo-context
    export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0        # only on hosts that need it
    export QWEN38_BF4_CORPUS=<corpus> QWEN38_BF4_CORPUS_VERIFICATION=<verification.json>   # only until bf4-experts exists
    unset QWEN38_CACHE_LABEL                                 # the adapter default is c<context>-vllm

If you set `QWEN38_CACHE_LABEL`, it must start with `c<context>-`. Each new context builds its own component cache
on first use (`c<context>-vllm`, about 10 to 23 GB). That first run is slow.

The demo derives the chip order like the server does. Set `QWEN38_HARDWARE_PROFILE` to the box type: `tt-quietbox-2`
(QuietBox 2, 2x P300c), `tt-quietbox` or `p150-line`. Only a host listed in `tools/hardware_profiles.py` can leave it
unset. If `TT_MESH_GRAPH_DESC_PATH` is not set, the demo sets it from the profile.

## Commands

    pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -s -k "traced_128 and not traced_128k"
    pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -s -k traced_256k

`CI=true` also writes the benchmark JSON.
