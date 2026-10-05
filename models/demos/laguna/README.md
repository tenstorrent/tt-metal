<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-S-2.1 on a TT-QuietBox 2

This guide runs [`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1) (117.6B parameters, 8.45B
active per token, 1,048,576-token context) as an OpenAI-compatible vLLM server on four Blackhole ASICs: a
TT-QuietBox 2 (two internal P300c cards, two ASICs each) or four P150 cards.

## Set up

```bash
git clone --branch jerrywangTT/laguna-s --recurse-submodules https://github.com/tenstorrent/tt-metal.git
cd tt-metal
export MODEL_DIR="$PWD/models/demos/laguna"
sudo ./install_dependencies.sh            # only if this host has never built tt-metal
"$MODEL_DIR/setup_vllm.sh"                 # builds tt-metal (1-3 h fresh) and the vLLM environment (30-45 min)
"$MODEL_DIR/.venv/bin/hf" auth login       # after accepting the model's terms on Hugging Face
"$MODEL_DIR/.venv/bin/hf" download poolside/Laguna-S-2.1   # about 235 GB; otherwise downloaded at first start
```

## Start the server

```bash
"$MODEL_DIR/serve_vllm.sh"            # port 8000; `serve_vllm.sh config` prints the settings without opening the cards
tail -f ~/laguna-logs/latest.log      # ready when it prints "Application startup complete" (~3 min)
```

The first start also converts the weights for the device (about 20 minutes, cached under
`~/.cache/ttnn/laguna_s_2_1`). The default serves one request at a time with a 1,048,576-token context. The model
thinks before it answers; send `"chat_template_kwargs": {"enable_thinking": false}` for plain answers.

Optional features: put the variables in front of the start command. Experimental ones also need
`LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`.

| Feature | Variables | Limits |
|---|---|---|
| DFlash speculative decoding (experimental) | `TT_LAGUNA_DFLASH=1` | 1 request at a time, greedy only |
| N-gram speculative decoding (experimental) | `TT_LAGUNA_SPEC_DECODE=1` | 1 request at a time, greedy only |
| Prefix caching (experimental) | `TT_LAGUNA_PREFIX_CACHE=1` | 1 request at a time |
| Concurrent requests | `TT_LAGUNA_HYBRID_KV=0 LAGUNA_MAX_NUM_SEQS=8` | up to 8 requests, 131,072-token context |

Check it:

```bash
curl -fsS http://localhost:8000/health && echo ready
curl -fsS http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "poolside/Laguna-S-2.1", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}]}'
```

Stop it with `"$MODEL_DIR/serve_vllm.sh" stop`. This also resets every Tenstorrent ASIC in the system
(`tt-smi -r`), so do not run it while anything else uses a card.

## Performance

Measured 2026-10-03 with `demo/perf_demo.py` (default server, batch 1, 512 output tokens, one random prompt per length).

| Input tokens | TTFT, normal | Decode tok/s, normal | TTFT, DFlash | Decode tok/s, DFlash | DFlash speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.23 s | 18.3 | 0.36 s | 28.1 | 1.5x |
| 1,024 | 2.40 s | 18.1 | 2.54 s | 40.7 | 2.2x |
| 2,048 | 2.81 s | 18.0 | 2.95 s | 25.3 | 1.4x |
| 4,096 | 9.45 s | 18.0 | 9.61 s | 45.9 | 2.6x |
| 8,192 | 19.9 s | 18.0 | 20.0 s | 31.1 | 1.7x |
| 16,384 | 33.5 s | 17.9 | 33.7 s | 30.2 | 1.7x |
| 32,768 | 64.8 s | 17.7 | 65.0 s | 36.8 | 2.1x |
| 65,536 | 142 s | 17.3 | 142 s | 41.0 | 2.4x |
| 131,072 | 355 s | 16.6 | 356 s | 22.5 | 1.4x |

### Run the perf demo

With no server running, from the repository root:

```bash
python models/demos/laguna/demo/perf_demo.py --quick   # 128 .. 8K input tokens, normal + DFlash, ~20 min
python models/demos/laguna/demo/perf_demo.py           # 128 .. 128K input tokens, normal + DFlash, ~40 min
```

It starts and stops the server itself and prints the table above. Results are saved under
`generated/laguna_perf_demo/<UTC time>/`. Options: `--modes normal|dflash`, `--input-lens 128,4096`,
`--prompts N` (average N random prompts per length; DFlash varies from prompt to prompt), `--output-tokens N`.

## Accuracy

Compared with the original model in fp32 over an AIME24 prompt plus a fixed 100-token answer:

| Check | top-1 | top-5 | top-100 |
|---|---:|---:|---:|
| Prefill | 0.97 | 1.00 | 1.00 |
| Teacher-forced decode | 0.98 | 1.00 | 1.00 |

Pass bars: top-1 >= 0.90, top-5 >= 0.98, top-100 = 1.00.

### Run the accuracy test

With no server running, from the repository root:

```bash
REPO=$PWD MODEL_DIR=$PWD/models/demos/laguna
cd /tmp && env -u TT_METAL_HOME PYTHONPATH=$REPO \
  LAGUNA_PROFILE=p150x4 TT_VISIBLE_DEVICES=0,1,2,3 LAGUNA_FABRIC_CONFIG=FABRIC_1D_RING \
  TT_LAGUNA_CCL_TOPOLOGY=ring TT_LAGUNA_CCL_NUM_LINKS=2 TT_LAGUNA_DECODE_SDPA_PC=1 \
  $MODEL_DIR/.venv/bin/python $MODEL_DIR/tests/full_model_checks.py teacher --profile p150x4 --enforce-memory-margin
```

It prints top-1, top-5 and top-100 (plus TTFT and decode tok/s), ~2 min once the weights are converted.
Replace `teacher` with `prefill_autoreg --max-seq-len 131072 --outdir /tmp/laguna-full-model` for the prefill check.
