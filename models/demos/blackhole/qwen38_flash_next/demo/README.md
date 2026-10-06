# Qwen3.8-Flash-Next demo

`text_demo.py` is the tiered CI end-to-end test: the vLLM adapter's `Qwen38ForCausalLM` on a 1x4 Blackhole mesh at
batch 1, a raw-tokenized 128-token prompt (`sample_prompts/`), 50 greedy tokens; `determinism_128` generates twice.

    export TT_METAL_TRACE_ALLOC_TRACKING=1 MESH_DEVICE="(1, 4)"
    export MODEL_WEIGHTS_DIR=<checkpoint directory>          # or HF_MODEL=Qwen/Qwen3.8-Flash-Next with HF_HOME populated
    export QWEN38_CACHE_ROOT=<cache root>                    # caches/<label>/{components,model-io}, caches/bf4-experts
    export QWEN38_BF4_CORPUS=<corpus> QWEN38_BF4_CORPUS_VERIFICATION=<verification.json>   # only until bf4-experts exists
    pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -v -s -k traced_128   # CI=true writes the benchmark JSON
