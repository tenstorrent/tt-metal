# Qwen3.8-27B on QB2

TP4 implementation of `Qwen/Qwen3.8-27B` on four Blackhole devices with a
`(1, 4)` mesh. Supports prefill, traced decode, device sampling, and vLLM serving.

## Setup

Build tt-metal and activate its Python environment from the repository root:

```bash
./build_metal.sh --enable-ccache
source python_env/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_WEIGHTS_DIR=/path/to/Qwen3.8-27B/snapshot
```

The weights directory must contain `config.json`, `model.safetensors.index.json`,
and the checkpoint shards. Without `MODEL_WEIGHTS_DIR`, the model uses the local
Hugging Face cache at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Weights are not downloaded at runtime.

## Demo

```bash
python models/demos/qwen38_27b_qb2/demo/text_demo.py \
  --full --length 128 --generate 128 --output /tmp/qwen38.json
```

Runs twice to check deterministic output and saves generated tokens and timing.
Omit `--full` for a two-layer smoke test.

## Serving

Use the vLLM TT plugin with `EXTRA_MODELS_DIR` pointing to `models/demos`.
The model registers as `TTQwen38ForCausalLM`.

- Context capacity: 262,144 tokens per request.
- Up to 32 request slots; bucketed decode supports capacities of 1, 8, or 16.
- The TTI configuration uses 16 slots and a shared 1,050,592-token KV pool.
- Prefix caching is disabled.

## Tests

Host tests, using the tt-metal Python environment:

```bash
python -m unittest discover -s models/demos/qwen38_27b_qb2/tests/unit -t .
python -m unittest discover -s models/demos/qwen38_27b_qb2/tests/vllm -t .
```

Device test on Blackhole:

```bash
pytest models/demos/qwen38_27b_qb2/tests/test_decode_conv.py
```
