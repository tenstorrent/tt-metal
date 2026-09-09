<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-XS-2.1 on p150x2/P300 and p150x4/P300x2

This guide runs [`poolside/Laguna-XS-2.1`](https://huggingface.co/poolside/Laguna-XS-2.1) as an
OpenAI-compatible server on Tenstorrent Blackhole hardware. It is written for people who want to use
the model; no model-optimization knowledge is required.

## Choose the hardware profile

A P150 card contains one Blackhole ASIC. A **TT-QuietBox 2 contains two internal P300c cards**, and
each P300c contains two Blackhole ASICs, each equivalent to one P150. P300c cards are not sold as
standalone products. `tt-smi` lists the individual ASICs, so the `x2` and `x4` in the launcher profiles
count ASICs rather than physical cards.

| Name used in this guide | Physical hardware | Launcher profile | ASICs selected | Maximum context |
|---|---|---|---:|---:|
| **p150x2/P300** | Two P150 cards, or one internal P300c card in a TT-QuietBox 2 | `p150x2` | 2 | 131,072 tokens |
| **p150x4/P300x2** | Four P150 cards, or both internal P300c cards in a TT-QuietBox 2 | `p150x4` | 4 | 131,072 tokens |

The lowercase values in the **Launcher profile** column are internal configuration names. Use them
literally in commands. P300 and P300x2 are configuration shorthand in this guide, not standalone
product names or accepted launcher values.

The recommended default is **p150x2/P300**. Use **p150x4/P300x2** with four P150 cards or the full
QuietBox 2. Start with one active request at a time. The expected-performance table below is a
single-request measurement of the recommended p150x2/P300 profile.

## Before you start

### 1. Get this tt-metal branch

For a new checkout:

```bash
git clone --branch agentic-research/hous/laguna-xs-2.1 --recurse-submodules \
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
export MODEL_DIR="$PWD/models/autoports/poolside_laguna_xs_2_1"
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

The model weights download automatically during the first server start.

## Start the server

The server runs in the background on port `8000`. The `config` command below is optional; it prints the
resolved settings without opening the cards.

### p150x2/P300: two P150 cards or one internal QuietBox P300c

Start with ASIC IDs `0,1`:

```bash
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 "$MODEL_DIR/serve_vllm.sh" config
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 "$MODEL_DIR/serve_vllm.sh"
```

On a QuietBox 2, you can use its other internal P300c by replacing `0,1` with `2,3` after confirming
the Board Number in `tt-smi -ls`. With two P150 cards, replace `0,1` with the IDs for those two cards
from `tt-smi`.

### p150x4/P300x2: four P150 cards or full TT-QuietBox 2

Use all four ASICs:

```bash
TT_VISIBLE_DEVICES=0,1,2,3 LAGUNA_PROFILE=p150x4 "$MODEL_DIR/serve_vllm.sh" config
TT_VISIBLE_DEVICES=0,1,2,3 LAGUNA_PROFILE=p150x4 "$MODEL_DIR/serve_vllm.sh"
```

If your ASIC IDs differ, replace `0,1,2,3` with all four IDs reported by `tt-smi -ls`.

Do not run the two profiles at the same time.

### Wait for startup

Follow the current log:

```bash
tail -f ~/laguna-logs/latest.log
```

The server is ready only when the log says:

```text
Application startup complete
```

Once the environment and weights are available, a normal server start takes about 10 minutes. The
first start also includes the approximately 63 GB model download.

## Expected performance

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

## Verify and use the model

Check the health endpoint:

```bash
curl -fsS http://localhost:8000/health && echo ready
```

Send a chat request:

```bash
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "poolside/Laguna-XS-2.1",
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
export POOLSIDE_STANDALONE_MODEL="poolside/Laguna-XS-2.1"
export POOLSIDE_STANDALONE_CONTEXT_LENGTH=131072

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
