# DeepSeek-V4-Flash

Experimental TT-NN implementation of DeepSeek-V4-Flash for Tenstorrent Blackhole.
There is no prefill op: a prompt is replayed one traced decode step per token, so
follow-up turns only cost the tokens they add. KV caches are paged, so several
independent conversations share one captured trace and one token budget.

Three ways to run it:

| | Entry point |
| --- | --- |
| Single-shot demo test | `tests/test_full_model_decode_demo.py` |
| Interactive chat REPL | `demo/chat_cli.py` |
| OpenAI-compatible server | [tt-inference-server](https://github.com/tenstorrent/tt-inference-server) |

## Setup

Build tt-metal from the branch that carries this model
(`smanoj/ds_v4_flash`), then activate its venv:

```bash
git clone git@github.com:tenstorrent/tt-metal.git && cd tt-metal
git checkout smanoj/ds_v4_flash
git submodule update --init --recursive
./build_metal.sh
./create_venv.sh
source python_env/bin/activate
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD
```

See [`INSTALLING.md`](../../../INSTALLING.md) for the full build instructions and
dependencies. Check the device is visible with `tt-smi`.

## Weights

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-DSpark
```

The default checkpoint path is the HF hub cache
(`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark`; the
`snapshots/<hash>/` layout is resolved automatically). ~167 GB of fp8 weights,
plus room for the converted tile cache — budget 300 GB.

The first run converts every weight to `bfloat4_b` and can take over an hour.
Point `DEEPSEEK_V4_CACHE_DIR` at a persistent directory so later runs reuse it.

## Demo test

```bash
pytest -s models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo.py
```

Builds the full ttnn `DeepSeekV4Model`, seeds the caches with a chat prompt, and
greedily generates until EOS or the token cap. Throughput is logged every 10
tokens. The test is skipped if the checkpoint is missing.

| Variable | Default | Description |
| --- | --- | --- |
| `DEEPSEEK_V4_DECODE_LAYERS` | all (43) | Cap the layer count. The full bf4 stack does not fit one 32 GB Blackhole; start at `4` for bringup. |
| `DEEPSEEK_V4_CACHE_DIR` | `../cache` | Converted ttnn weight tiles. Reuse across runs. |
| `DEEPSEEK_V4_MAX_NEW_TOKENS` | `1024` | Generation cap. |
| `DEEPSEEK_V4_TRACED_DECODE` | `1` | `0` for eager host-bound decode instead of captured traces. |
| `DEEPSEEK_V4_POOL_EVERY_STEP` | `0` | Re-pool the CSA/HCA compressors every step instead of only on window closures. Bit-identical, slower, but collapses the traced path to one trace variant — use it if trace capture runs out of memory. |
| `DEEPSEEK_V4_SDPA_CAUSAL` | `1` | Bound compressor attention with a causal `cur_pos` rather than an additive mask once the sliding ring is full. `0` forces the mask everywhere. |

## Chat CLI

An interactive multi-user REPL on the same decode engine
([`demo/chat_cli.py`](demo/chat_cli.py)). The model, RoPE tables and traced
buffers are built once; each turn feeds only the new tokens.

```bash
DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \
python models/experimental/deepseek_v4_flash/demo/chat_cli.py \
    --num-users 4 --max-context 2048 --think \
    --system-prompt "You are a terse assistant."
```

Startup runs one throwaway decode step so the kernel compile and trace capture
(minutes of it) happen before the first prompt, and the reported per-turn
timings stay honest.

Each user is an independent conversation with its own messages, position, system
prompt and thinking mode, backed by its own pages out of a shared block pool.
Switching users rewrites a page table rather than the cache, so one trace serves
everyone and the users share one `--total-context` token budget.

```
you[0]> what is the capital of France?
bot> Paris.
you[0]> /user 1
[switched to user 1]
you[1]> write me a haiku about cache coherence
```

Commands: `/user N`, `/users`, `/reset [N|all]`, `/system TEXT`,
`/think [on|off]`, `/context`, `/help`, `/exit`.

Key flags (`--help` for all): `--num-layers`, `--num-users`, `--max-context`
(per user), `--total-context` (shared pool), `--max-new-tokens`, `--think`
(streams a `<think>` reasoning block inline), `--no-trace` (eager, single user
only), `--cache-dir`, `--model-dir`.

## Inference server

DeepSeek-V4-Flash is served through
[tt-inference-server](https://github.com/tenstorrent/tt-inference-server) as an
OpenAI-compatible endpoint. It is currently a **dev** model spec, so every
command needs `--dev-mode`.

### Branches

| Repo | Branch |
| --- | --- |
| tt-metal | `smanoj/ds_v4_flash` (must contain `models/experimental/deepseek_v4_flash`) |
| tt-inference-server | `smanoj/ds_v4_flash` |

No tt-metal or vLLM commit is pinned in the dev spec — the server runs against
whatever tt-metal tree you point it at.

```bash
git clone git@github.com:tenstorrent/tt-inference-server.git
cd tt-inference-server
git checkout smanoj/ds_v4_flash
```

The tt-metal venv also needs vLLM and the `vllm-tt-plugin` installed:

```bash
cd $TT_METAL_HOME/vllm && bash tt_metal/install-vllm-tt.sh
```

### Run

```bash
export HF_TOKEN=hf_...
export TT_METAL_HOME=/path/to/tt-metal

python3 run.py \
  --dev-mode \
  --model DeepSeek-V4-Flash-DSpark \
  --workflow server \
  --tt-device p150x8 \
  --local-server \
  --tt-metal-home $TT_METAL_HOME \
  --no-auth
```

Use `--tt-device blackhole_galaxy` for a BH Galaxy. For a Docker run swap
`--local-server` for `--docker-server --override-docker-image <image>`; the dev
spec pins no image, so it must be one you built against the same tt-metal
commit (`python3 scripts/build_docker_images.py --build-metal-commit <sha>`).

### Tensor cache path

The server sets the model's two cache env vars at startup, but only if they are
not already set, so an explicit export wins:

| Variable | Meaning |
| --- | --- |
| `DEEPSEEK_V4_HF_MODEL` | Checkpoint directory. Defaults to the weights symlink under `CACHE_ROOT`. |
| `DEEPSEEK_V4_CACHE_DIR` | Converted `bfloat4_b` tile cache. Defaults to `TT_CACHE_PATH`. |

By default `TT_CACHE_PATH` resolves to
`$CACHE_ROOT/tt_metal_cache/cache_DeepSeek-V4-Flash-DSpark/<mesh_device>/`
(e.g. `P150x8`). Set `CACHE_ROOT`, or pass `--host-volume /mnt/tt-cache`, to put
it on a disk with room; pass `--host-hf-cache ~/.cache/huggingface` to reuse
weights you already downloaded.

To share the tile cache with the demo/CLI runs above, export it explicitly
before launching:

```bash
export DEEPSEEK_V4_CACHE_DIR=/mnt/tt-cache/deepseek_v4_flash
export DEEPSEEK_V4_HF_MODEL=/path/to/DeepSeek-V4-Flash-DSpark/snapshot
```

The spec allows 7200 s for the first-run conversion before timing out.

### Query it

The server listens on port 8000 (`--service-port` / `SERVICE_PORT`) and the
model name is the HF repo id:

```bash
curl -s --no-buffer -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }'
```

Without `--no-auth` the request needs an `Authorization: Bearer <jwt>` header
signed with `JWT_SECRET`.
