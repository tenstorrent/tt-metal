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

### Run the perf demo

With no server running, from the repository root:

```bash
python models/demos/laguna/demo/perf_demo.py   # 128 .. 8K input tokens, normal + DFlash, ~20 min
```

It starts and stops the server itself, prints the model's answers to two real prompts in each mode, and prints
the table above. Results are saved under
`generated/laguna_perf_demo/<UTC time>/`. Options: `--modes normal|dflash`, `--input-lens 128,4096`,
`--prompts N` (average N random prompts per length; DFlash varies from prompt to prompt), `--output-tokens N`.
Longer prompts work too (`--input-lens 16384,131072`; a 128K prefill takes about 6 minutes per mode).

## Accuracy

Laguna is compared with the original model run in fp32 on the CPU, over an AIME24 prompt plus a fixed 100-token
answer. At each of the 100 positions both predict the next token:

| Measure | Result | Bar |
|---|---:|---:|
| top-1 (Laguna's top token = the reference's) | 0.99 | >= 0.90 |
| top-5 (the reference's token in Laguna's top 5) | 1.00 | >= 0.98 |
| top-100 | 1.00 | = 1.00 |
| top-1 of the traced decode the server uses | 0.98 | >= 0.90 |
| PCC of all 100,352 next-token scores, mean over positions | 0.97 | >= 0.95 |
| PCC, worst position | 0.74 | - |

The experts are stored in 4-bit, so the scores carry rounding error (PCC 0.97) while the chosen tokens agree.

### Run the accuracy test

With no server running, from the repository root. The first command makes the fp32 reference scores once
(CPU only, about 5 minutes, about 30 GB of memory):

```bash
REPO=$PWD MODEL_DIR=$PWD/models/demos/laguna
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python -m models.demos.laguna.tests.gen_streamed_reference --dtype fp32 \
  --output generated/laguna_reference/readiness_aime24_chat_s.refpt \
  --save-logits generated/laguna_reference/Laguna-S-2.1-aime24-logits.pt
cd /tmp && env -u TT_METAL_HOME PYTHONPATH=$REPO \
  $MODEL_DIR/.venv/bin/python -m pytest -s $MODEL_DIR/tests/test_accuracy.py
```

It prints top-1, top-5, top-100, traced top-1 and PCC, and fails if any is below its bar (~3 min once the weights
are converted).
