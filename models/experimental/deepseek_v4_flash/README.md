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
| OpenAI-compatible server | `demo/server.py` |

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


## Inference server

OpenAI-compatible HTTP server ([`demo/server.py`](demo/server.py)) on the same
decode engine as the chat CLI. See `--help` for flags.
This runs on port 8000.

```bash
python models/experimental/deepseek_v4_flash/demo/server.py
```

Then query it:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```



## Chat CLI

An interactive multi-user REPL on the same decode engine
([`demo/chat_cli.py`](demo/chat_cli.py)). The model, RoPE tables and traced
buffers are built once; each turn feeds only the new tokens.

```bash
python models/experimental/deepseek_v4_flash/demo/chat_cli.py
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
