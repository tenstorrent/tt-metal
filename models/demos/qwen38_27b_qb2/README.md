# Qwen3.8-27B on QB2

`Qwen/Qwen3.8-27B` on four Blackhole devices (QB2/P300x2), using a `(1, 4)` mesh.
Supports text generation and vLLM serving with existing TTNN operations and no
custom kernels.

## Capacity

- Up to 16 concurrent requests.
- Maximum context capacity: 262,144 tokens per request.
- Shared KV pool: 1,050,592 tokens; longer contexts reduce available concurrency.
- Prefix caching is disabled.

## Batch-1 performance

One active request on a 16-slot vLLM server:

| Input / output tokens | Decode tokens/s/user | Mean TTFT (ms) |
| --- | --- | --- |
| 128 / 128 | 35.83 | 71.46 |
| 1024 / 128 | 35.56 | 189.79 |

[Benchmark results](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36117958569)
at model revision `7099adadc2fc`. Decode tokens/s/user is calculated as
`1000 / mean TPOT (ms)`.

## Setup and demo

From the repository root:

```bash
./build_metal.sh --enable-ccache
source python_env/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_WEIGHTS_DIR=/path/to/Qwen3.8-27B/snapshot

python models/demos/qwen38_27b_qb2/demo/text_demo.py \
  --full --length 128 --generate 128 --output /tmp/qwen38.json
```

Use checkpoint revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Weights must already be downloaded.

For vLLM, set `EXTRA_MODELS_DIR` to `models/demos` and use architecture
`TTQwen38ForCausalLM`. See [the CI recipe](tests/run_ci.sh) for serving
configuration and tests.
