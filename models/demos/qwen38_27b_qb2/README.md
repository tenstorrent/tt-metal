# Qwen3.8-27B on QB2

TP4 implementation of `Qwen/Qwen3.8-27B` on four Blackhole devices with a
`(1, 4)` mesh. It supports prefill, traced decode, device sampling, and vLLM
serving.

## Capacity

- Maximum supported context: 262,144 tokens.
- Largest tested input sequence length: 261,892 tokens with 252 output tokens.
- Up to 16 concurrent users.
- Decode uses batch sizes 1, 8, and 16. Other active-user counts are padded to
  the next supported batch size: 2–8 use batch 8, and 9–16 use batch 16.

## Performance

Batch-1 vLLM performance on QB2/P300x2:

| Input / output tokens | Tokens/s/user | TTFT |
| --- | ---: | ---: |
| 128 / 128 | 39.4 | 67.9 ms |

## Evaluation

Evaluation results:

- GPQA Diamond: 9/10 (90%).
- Terminal-Bench 2.1: 4/5 (80%).
- SWE-bench Verified: 3/5 (60%).

## Run the demo

Build tt-metal and activate its Python environment:

```bash
./build_metal.sh --enable-ccache
source python_env/bin/activate
export PYTHONPATH="$PWD:$PWD/ttnn:$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_WEIGHTS_DIR=/path/to/Qwen3.8-27B/snapshot
```

Run the model:

```bash
python models/demos/qwen38_27b_qb2/demo/text_demo.py \
  --full --length 128 --generate 128 --output /tmp/qwen38.json
```

For vLLM serving, set `EXTRA_MODELS_DIR` to `models/demos`. The model is
registered as `TTQwen38ForCausalLM`.

## vLLM prefix caching

Set `QWEN_PREFIX_SNAPSHOTS` to keep that many device snapshots of the
linear-attention state (36.7 MiB per device each) and enable vLLM prefix
caching:

```bash
export QWEN_PREFIX_SNAPSHOTS=32
```

A snapshot is taken at the last multiple of 128 tokens at least 16 tokens
before the end of each prompt of at least 4096 tokens
(`QWEN_PREFIX_SNAPSHOT_MARGIN`, `QWEN_PREFIX_SNAPSHOT_MIN`). Do not pass
`--no-enable-prefix-caching`.
