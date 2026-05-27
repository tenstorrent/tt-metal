# Qwen3.5-27B Demo — 4×P150 (TP=4)

Prefill + decode demo with TTFT and throughput metrics.

## Requirements

- 4× P150 devices (`MESH_DEVICE=P150x4`)
- Model weights at `HF_MODEL` path (default: `/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654`)

## Run

```bash
# Setup
export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

# ISL=4096, 20 decode tokens (default parallel-scan GDN, prefill trace enabled)
MESH_DEVICE=P150x4 pytest models/demos/qwen35_27b/demo/demo.py -k isl4k -v -s

# ISL=4096, 100 decode tokens
MESH_DEVICE=P150x4 pytest models/demos/qwen35_27b/demo/demo.py -k isl4k_long -v -s

# ISL=1024
MESH_DEVICE=P150x4 pytest models/demos/qwen35_27b/demo/demo.py -k isl1k -v -s

# Run all ISL variants
MESH_DEVICE=P150x4 pytest models/demos/qwen35_27b/demo/demo.py -v -s

# Disable prefill trace (slower but easier to debug)
MESH_DEVICE=P150x4 GDN_PREFILL_TRACE=0 pytest models/demos/qwen35_27b/demo/demo.py -k isl4k -v -s

# Custom model path
MESH_DEVICE=P150x4 HF_MODEL=/path/to/Qwen3.5-27B \
    pytest models/demos/qwen35_27b/demo/demo.py -k isl4k -v -s
```

## Prompt

Uses the first 4096 tokens of the Frankenstein long-context sample from
`models/tt_transformers/demo/sample_prompts/input_data_long_4k.json`
(Gutenberg text + instruction, cached locally after first download).

## Performance (P150x4, batch=32, bfloat8_b weights)

| ISL  | TTFT   | Throughput (aggregate) |
|------|--------|------------------------|
| 1024 | ~900ms | ~442 tok/s             |
| 4096 | ~3275ms| ~441 tok/s             |
