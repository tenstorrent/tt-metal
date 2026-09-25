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
| 128 / 128 | 40.37 | 69.30 ms |

## Evaluation

CI subset results:

- GPQA Diamond: 9/10 (90%).
- Terminal-Bench 2.1: 4/5 (80%).
- SWE-bench Verified: 3/5 (60%).

## Run the demo

Build tt-metal and activate its Python environment:

```bash
./build_metal.sh --enable-ccache
source python_env/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_WEIGHTS_DIR=/path/to/Qwen3.8-27B/snapshot
```

Run the model:

```bash
python models/demos/qwen38_27b_qb2/demo/text_demo.py \
  --full --length 128 --generate 128 --output /tmp/qwen38.json
```

For vLLM serving, set `EXTRA_MODELS_DIR` to `models/demos`. The model is
registered as `TTQwen38ForCausalLM`.
