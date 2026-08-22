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
QuietBox 2. Start with one active request at a time; the current performance results for both
configurations are single-request measurements.

## Before you start

You need:

- Two or four P150 cards, or a TT-QuietBox 2, with the Tenstorrent driver, firmware, and `tt-smi`
  installed.
- A Linux x86-64 host and the tt-metal source-build prerequisites from the repository's
  [`INSTALLING.md`](../../../INSTALLING.md).
- Access to the gated
  [`poolside/Laguna-XS-2.1`](https://huggingface.co/poolside/Laguna-XS-2.1) repository on Hugging Face.
- At least 80 GB of free space for the approximately 63 GB model download, the tt-metal build, and
  the Python environment. Allow more if you retain build or package caches.

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

### 2. Check the cards

```bash
tt-smi -ls
```

For p150x2/P300, select exactly two ASIC IDs:

- With two P150 cards, select the one ASIC on each card. Their **Board Numbers are different**.
- With a TT-QuietBox 2, select the two ASICs with the same **Board Number**; those ASICs are on one of
  its internal P300c cards. On a normally enumerated QuietBox 2, IDs `0,1` form one pair and `2,3`
  form the other. Confirm this on your system instead of assuming the numbering.

For p150x4/P300x2, select the four ASIC IDs from four P150 cards or all four ASIC IDs in the QuietBox 2.

### 3. Build the serving environment

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

Use the two ASIC IDs identified above:

```bash
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 "$MODEL_DIR/serve_vllm.sh" config
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 "$MODEL_DIR/serve_vllm.sh"
```

On a QuietBox 2, you can use its other internal P300c by replacing `0,1` with `2,3` after confirming
the Board Number in `tt-smi -ls`. With two P150 cards, replace `0,1` with the IDs for those two cards.

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

## Verify and use the model

Check the health endpoint:

```bash
curl -fsS http://localhost:8000/health && echo ready
```

Confirm the served model:

```bash
curl -fsS http://localhost:8000/v1/models \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
```

Expected output:

```text
poolside/Laguna-XS-2.1
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

The server does not require an API key. Keep it on a trusted machine or network unless you add your
own authentication and access controls.

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

## Troubleshooting

### The launcher reports the wrong number of ASICs

Run `tt-smi -ls` again. A p150x2/P300 launch needs exactly two unique ASIC IDs: one from each of two
P150 cards, or both ASICs sharing a Board Number inside a QuietBox 2. A p150x4/P300x2 launch needs the
four ASIC IDs from four P150 cards or the full QuietBox 2.

### Hugging Face returns 401 or 403

Confirm that your Hugging Face account has access to the model, then authenticate again:

```bash
"$MODEL_DIR/.venv/bin/hf" auth login
```

### Startup appears stuck

The first build and first model download take a long time. Check the latest log instead of restarting:

```bash
tail -n 200 ~/laguna-logs/latest.log
```

Wait for `Application startup complete`. An older successful startup message in another log does not
mean the current run is ready.

### A card, server process, or port is busy

Make sure no other workload is using the selected cards or port `8000`, then run:

```bash
"$MODEL_DIR/serve_vllm.sh" stop
```

Start the desired profile again after the reset completes.

### The launcher rejects settings inherited from another environment

Clear stale overrides and retry the documented command:

```bash
unset TT_METAL_HOME TT_MESH_GRAPH_DESC_PATH MESH_DEVICE VLLM_PLUGINS
```

If the Python environment itself is damaged, rebuild it as a last resort:

```bash
"$MODEL_DIR/setup_vllm.sh" --force
```
