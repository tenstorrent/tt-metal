<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-S-2.1 and Laguna-XS-2.1 on Blackhole (p150x2/P300, p150x4/P300x2)

This guide runs [`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1) and
[`poolside/Laguna-XS-2.1`](https://huggingface.co/poolside/Laguna-XS-2.1) as an OpenAI-compatible
server on Tenstorrent Blackhole hardware. It is written for people who want to use the model; no
model-optimization knowledge is required.

The two checkpoints share this code. `HF_MODEL` picks one (default: Laguna-S-2.1):

| Model | Size | Hardware | Maximum context |
|---|---|---|---:|
| `poolside/Laguna-S-2.1` | 117.6B parameters, 8.45B active per token (about 235 GB in bf16) | p150x4/P300x2 only (all four ASICs) | 1,048,576 tokens |
| `poolside/Laguna-XS-2.1` | about 63 GB of bf16 weights | p150x2/P300 (recommended) or p150x4/P300x2 | 131,072 tokens |

## Choose the hardware profile

A P150 card contains one Blackhole ASIC. A **TT-QuietBox 2 contains two internal P300c cards**, and
each P300c contains two Blackhole ASICs, each equivalent to one P150. P300c cards are not sold as
standalone products. `tt-smi` lists the individual ASICs, so the `x2` and `x4` in the launcher profiles
count ASICs rather than physical cards.

| Name used in this guide | Physical hardware | Launcher profile | ASICs selected |
|---|---|---|---:|
| **p150x2/P300** | Two P150 cards, or one internal P300c card in a TT-QuietBox 2 | `p150x2` | 2 |
| **p150x4/P300x2** | Four P150 cards, or both internal P300c cards in a TT-QuietBox 2 | `p150x4` | 4 |

The lowercase values in the **Launcher profile** column are internal configuration names. Use them
literally in commands. P300 and P300x2 are configuration shorthand in this guide, not standalone
product names or accepted launcher values.

Laguna-S-2.1 needs **p150x4/P300x2**: its weights take about 17.7 GB of each ASIC's memory when split
four ways. For Laguna-XS-2.1 the recommended profile is **p150x2/P300**; use **p150x4/P300x2** with four
P150 cards or the full QuietBox 2. Start with one active request at a time.

## Before you start

### 1. Get this tt-metal branch

For a new checkout:

```bash
git clone --branch jerrywangTT/laguna-s --recurse-submodules \
  https://github.com/tenstorrent/tt-metal.git
cd tt-metal
```

For an existing checkout of this branch:

```bash
git submodule update --init --recursive
```

In every new shell, enter the repository root and set the model directory:

```bash
cd /path/to/tt-metal
export MODEL_DIR="$PWD/models/demos/laguna"
```

If the host has not already been prepared for a tt-metal source build, install the repository
dependencies:

```bash
sudo ./install_dependencies.sh
```

### 2. Build the serving environment

From the repository root:

```bash
"$MODEL_DIR/setup_vllm.sh"
```

The script builds tt-metal when needed and creates a self-contained Python environment under
`$MODEL_DIR/.venv`. On a fresh checkout, allow roughly 1–3 hours for tt-metal and another 30–45 minutes
for vLLM. Reusing an existing build is much faster.

After setup, authenticate with Hugging Face. First accept the model's access terms in your browser,
then run:

```bash
"$MODEL_DIR/.venv/bin/hf" auth login
```

The model weights download automatically during the first server start (about 235 GB for
Laguna-S-2.1, about 63 GB for Laguna-XS-2.1). To download ahead of time:

```bash
"$MODEL_DIR/.venv/bin/hf" download poolside/Laguna-S-2.1
```

## Start the server

The server runs in the background on port `8000`. The `config` command below is optional; it prints the
resolved settings without opening the cards.

### Laguna-S-2.1 on p150x4/P300x2: full TT-QuietBox 2 or four P150 cards

```bash
HF_MODEL=poolside/Laguna-S-2.1 "$MODEL_DIR/serve_vllm.sh" config
HF_MODEL=poolside/Laguna-S-2.1 "$MODEL_DIR/serve_vllm.sh"
```

This selects `p150x4` on ASICs `0,1,2,3` and serves 1,048,576 tokens of context to one request at a time,
using hybrid KV (the 36 sliding-window layers keep only their 512-token window, so only the 12
full-attention layers store every past token) and chunked prefill. `TT_LAGUNA_HYBRID_KV=0` falls back to a
plain KV cache, which fits 131,072 tokens and up to 8 requests.

Laguna-S-2.1 thinks before it answers by default; the reasoning is returned separately from the
answer (`reasoning_content`). A client that wants plain answers can send
`"chat_template_kwargs": {"enable_thinking": false}`, or start the server with
`LAGUNA_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'`.

### Laguna-XS-2.1 on p150x2/P300: two P150 cards or one internal QuietBox P300c

Start with ASIC IDs `0,1`:

```bash
HF_MODEL=poolside/Laguna-XS-2.1 TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 "$MODEL_DIR/serve_vllm.sh" config
HF_MODEL=poolside/Laguna-XS-2.1 TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 "$MODEL_DIR/serve_vllm.sh"
```

On a QuietBox 2, you can use its other internal P300c by replacing `0,1` with `2,3` after confirming
the Board Number in `tt-smi -ls`. With two P150 cards, replace `0,1` with the IDs for those two cards
from `tt-smi`.

### Laguna-XS-2.1 on p150x4/P300x2: four P150 cards or full TT-QuietBox 2

Use all four ASICs:

```bash
HF_MODEL=poolside/Laguna-XS-2.1 TT_VISIBLE_DEVICES=0,1,2,3 LAGUNA_PROFILE=p150x4 "$MODEL_DIR/serve_vllm.sh" config
HF_MODEL=poolside/Laguna-XS-2.1 TT_VISIBLE_DEVICES=0,1,2,3 LAGUNA_PROFILE=p150x4 "$MODEL_DIR/serve_vllm.sh"
```

If your ASIC IDs differ, replace `0,1,2,3` with all four IDs reported by `tt-smi -ls`.

Do not run two servers at the same time.

### Optional serving features (Laguna-S-2.1)

Add the variables in front of the start command. Experimental ones also need `LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`.

| Feature | Variables | Limits |
|---|---|---|
| DFlash speculative decoding (experimental) | `TT_LAGUNA_DFLASH=1` | 1 request at a time, greedy only |
| N-gram speculative decoding (experimental) | `TT_LAGUNA_SPEC_DECODE=1` | 1 request at a time, greedy only |
| Prefix caching (experimental) | `TT_LAGUNA_PREFIX_CACHE=1` | 1 request at a time |
| Concurrent requests | `TT_LAGUNA_HYBRID_KV=0 LAGUNA_MAX_NUM_SEQS=8` | up to 8 requests, 131,072-token context |

### Wait for startup

Follow the current log:

```bash
tail -f ~/laguna-logs/latest.log
```

The server is ready only when the log says:

```text
Application startup complete
```

Once the environment and weights are available, a normal server start takes about 3 minutes for
Laguna-S-2.1 and about 10 minutes for Laguna-XS-2.1. The first Laguna-S-2.1 start also converts the
weights for the device (about 20 minutes, cached under `~/.cache/ttnn/laguna_s_2_1`).

## Expected performance

### Laguna-S-2.1 on p150x4/P300x2

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

### Laguna-XS-2.1 on p150x2/P300

The p150x2 profile enables automatic prefix caching. This sweep deliberately bypassed prefix reuse,
so the table shows cold requests.

| Input tokens requested | Output tokens | Concurrency | Decode tok/s/user | Aggregate output tok/s | Time to first token | End-to-end latency |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 512 | 1 | 19.97 | 19.84 | 0.215 s | 25.801 s |
| 1,024 | 512 | 1 | 19.82 | 18.29 | 2.213 s | 27.995 s |
| 2,048 | 512 | 1 | 19.80 | 18.05 | 2.563 s | 28.366 s |
| 4,096 | 512 | 1 | 19.78 | 14.70 | 8.984 s | 34.822 s |
| 8,192 | 512 | 1 | 19.72 | 11.23 | 19.680 s | 45.592 s |
| 16,384 | 512 | 1 | 19.61 | 8.53 | 33.931 s | 59.993 s |
| 32,768 | 512 | 1 | 19.38 | 5.44 | 67.812 s | 94.178 s |
| 65,536 | 512 | 1 | 18.95 | 2.79 | 156.630 s | 183.595 s |
| 130,048 | 512 | 1 | 18.15 | 1.25 | 380.812 s | 408.967 s |

### Run the perf demo

With no server running, from the repository root:

```bash
python models/demos/laguna/demo/perf_demo.py --quick   # 128 .. 8K input tokens, normal + DFlash, ~20 min
python models/demos/laguna/demo/perf_demo.py           # 128 .. 128K input tokens, normal + DFlash, ~40 min
```

It starts and stops the server itself and prints the table above. Results are saved under
`generated/laguna_perf_demo/<UTC time>/`. Options: `--modes normal|dflash`, `--input-lens 128,4096`,
`--prompts N` (average N random prompts per length; DFlash varies from prompt to prompt), `--output-tokens N`.

## Accuracy (Laguna-S-2.1, p150x4)

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

## Verify and use the model

Check the health endpoint:

```bash
curl -fsS http://localhost:8000/health && echo ready
```

Send a chat request (use the model name the server was started with):

```bash
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "poolside/Laguna-S-2.1",
    "messages": [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}],
    "temperature": 0,
    "max_tokens": 256
  }' | python3 -m json.tool
```

For an OpenAI-compatible client, use:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
```

## Use the `pool` coding agent

[`pool`](https://github.com/poolsideai/pool) is a separate Poolside terminal agent and is not bundled
with this repository. Follow its official installation instructions, verify it with `pool --version`,
and point it at the running server:

```bash
export POOLSIDE_STANDALONE_BASE_URL=http://localhost:8000/v1
export POOLSIDE_API_KEY=EMPTY
export POOLSIDE_STANDALONE_MODEL="poolside/Laguna-S-2.1"   # or poolside/Laguna-XS-2.1
export POOLSIDE_STANDALONE_CONTEXT_LENGTH=1048576   # 131072 for Laguna-XS-2.1

cd /path/to/your/project
pool
```

## Stop or restart the server

Always stop the server with the launcher:

```bash
"$MODEL_DIR/serve_vllm.sh" stop
```

This stops the server and runs `tt-smi -r all`. It resets **every Tenstorrent ASIC in the system**,
including P150 cards or the internal QuietBox P300c not used by a p150x2/P300 run. Do not run it while
another user or workload is using any other card.

To restart or switch profiles, stop the current server, wait for the reset to finish, and then run the
desired start command again.
