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

## System profiles (per-machine tuning)

Every deployment knob — pipeline group size, prefetcher ring depth, the MoE L1
expert block size, socket sizes, weight precision, batch/session/context sizing —
lives in [`configs/system_configs.yaml`](configs/system_configs.yaml) as a named
profile per machine, loaded by [`tt/system_config.py`](tt/system_config.py).
Model geometry is deliberately *not* here: that comes from the checkpoint's own
config.

The profile is chosen automatically by the mesh's device count, so the same
command line tunes itself to the machine it lands on:

| Profile | Machine | Notes |
| --- | --- | --- |
| `p150x8` | 8x Blackhole P150 | The measured config: 16.2 tok/s/u at B=1, PGS=1 |
| `galaxy32` | 32x Blackhole Galaxy | Capacity-tuned starting point (see the file's note on why 32 chips needs TP/EP for per-user speed) |
| `single_chip` | 1x Blackhole | Capped layer stack for PCC tests and bring-up |
| `<machine>_server` | either | Many concurrent conversations, long contexts |
| `<machine>_throughput` | either | Batched decode: aggregate tok/s over per-user latency |

Selection precedence: an explicit `system_config=` / `--system-profile`, then
`$DEEPSEEK_V4_SYSTEM_PROFILE`, then the profile matching the device count, then
the file's `default_profile`. `demo/server.py` additionally asks for the `server`
variant of whichever machine matched, so `p150x8` becomes `p150x8_server`.

Profiles inherit with `extends` and are merged key-by-key, so a new machine only
states what it changes. Any single field can be overridden at runtime by its
environment variable (each is named in a comment beside the field), which beats
the file — handy for a one-off A/B:

```bash
# One experiment, no file edit:
DEEPSEEK_V4_EXPERTS_BLOCK_SIZE=1 DEEPSEEK_V4_PREFETCH_PAGES=24 \
python models/experimental/deepseek_v4_flash/demo/chat_cli.py

# Force a profile that the device count would not pick:
DEEPSEEK_V4_SYSTEM_PROFILE=p150x8_throughput python ...

# Point at a local tune instead of the shipped file:
DEEPSEEK_V4_SYSTEM_CONFIG=/path/to/my_profiles.yaml python ...
```

In code, the entry points take it directly and every knob left unset falls back
to the profile:

```python
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config

cfg = load_system_config(mesh_device=mesh_device)      # or profile="galaxy32"
cfg = cfg.with_overrides(pipeline={"group_size": 4})   # experiment in-process
gen = DeepSeekV4Generator.from_pretrained(mesh_device, system_config=cfg)
```

`load_system_config` rejects unknown keys with a suggestion, so a typo fails at
load rather than silently running an untuned machine. Host-only tests:
`pytest models/experimental/deepseek_v4_flash/tests/test_system_config.py`.

## Demo test

```bash
pytest -s models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo.py
```

Builds the full ttnn `DeepSeekV4Model`, seeds the caches with a chat prompt, and
greedily generates until EOS or the token cap. Throughput is logged every 10
tokens. The test is skipped if the checkpoint is missing.

These variables override the active system profile (see above for the full list,
which is documented field-by-field in `configs/system_configs.yaml`):

| Variable | Default | Description |
| --- | --- | --- |
| `DEEPSEEK_V4_SYSTEM_PROFILE` | by device count | Force a machine profile by name. |
| `DEEPSEEK_V4_DECODE_LAYERS` | all (43) | Cap the layer count. The full bf4 stack does not fit one 32 GB Blackhole; start at `4` for bringup. |
| `DEEPSEEK_V4_CACHE_DIR` | `../cache` | Converted ttnn weight tiles. Reuse across runs. |
| `DEEPSEEK_V4_MAX_NEW_TOKENS` | `1024` | Generation cap. |
| `DEEPSEEK_V4_TRACED_DECODE` | `1` | `0` for eager host-bound decode instead of captured traces. |
| `DEEPSEEK_V4_POOL_EVERY_STEP` | `0` | Re-pool the CSA/HCA compressors every step instead of only on window closures. Bit-identical, slower, but collapses the traced path to one trace variant — use it if trace capture runs out of memory. |
| `DEEPSEEK_V4_SDPA_CAUSAL` | `1` | Bound compressor attention with a causal `cur_pos` rather than an additive mask once the sliding ring is full. `0` forces the mask everywhere. |
| `DEEPSEEK_V4_EXPERTS_BLOCK_SIZE` | `2` | Experts whose activations `fused_experts` holds in L1 at once. Lower this when the op's static circular buffers clash with L1 buffers; only `1` and `2` actually save L1 (the block is double-buffered). |
| `DEEPSEEK_V4_PREFETCH_PAGES` | `16` | Weight-prefetcher GCB ring depth, i.e. how far ahead the DRISC senders may run. |
| `DEEPSEEK_V4_PIPELINE_GROUP_SIZE` | `1` | Devices per pipeline group. `1` gives each device one contiguous slice of layers (fewest socket hops per token). |
| `DEEPSEEK_V4_FUSE_QA_KV` | `0` | Run q_a and kv as one matmul over their concatenated weight. `auto` fuses wherever the prefetcher is off, which is the only place the fused weight works — it shares no GCB page size, so forcing `1` with the prefetcher on is rejected. |
| `DEEPSEEK_V4_PREFETCHER` | `auto` | Force the DRISC weight prefetcher on/off; `auto` uses it wherever the device supports it. |

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
only), `--cache-dir`, `--model-dir`. The sizing flags default to the active
system profile, and `--system-profile` / `--system-config-file` pick a different
one.

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
