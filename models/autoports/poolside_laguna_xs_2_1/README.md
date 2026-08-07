<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-XS-2.1 on Tenstorrent (P150x4)

TTNN bring-up + vLLM serving of [`poolside/Laguna-XS-2.1`](https://huggingface.co/poolside/Laguna-XS-2.1),
a ~31B GLM/Qwen3-style MoE (256 experts, top-8, shared expert; 40 layers, 10 full-attention + 30
sliding-window(512); hybrid KV; router `sigmoid(logits)+e_bias`, `norm_topk_prob`, no router bias).

- **Target mesh:** Blackhole **P150x4** (1×4), TP=4 / EP=4, `FABRIC_1D_RING`.
- **Precision (selected policy):** BF16 activations/norms/router, BFP8 attn/dense/shared + KV + LM-head,
  BFP4 routed experts, fp32/HiFi4 SDPA. See `doc/datatype_sweep/`.
- **Serving:** stock upstream vLLM 0.24.0 + the public `tenstorrent/vllm-tt-plugin` + this model's
  `vllm_ext` package — **no vLLM fork**. Adapter: `tt/generator_vllm.py`.

This is the single doc for the model — everything to get it running is below.

## Context / capability

| Quantity | Value | Note |
|---|---|---|
| HF config context | **262,144** | what the checkpoint declares |
| **Advertised = servable context** | **131,072** | verified on P150x4; the value the server advertises |
| Verified-servable ISL (2026-07-31 sweep) | 128 … 131,072 | `doc/vllm_integration/sweep_vllm.tsv` |
| OOM | 262,144 | restoring it is Tier-2 (hybrid-KV + shared-RoPE frees) |

The advertised context **equals** the verified-servable context by construction (`ADVERTISED_MAX_CONTEXT`
in `tt/generator_vllm.py`): the model never accepts a context it cannot serve. See
`doc/context_contract.json` for the recorded limiting reason.

## Determinism contract

> **Prefix caching is ON by default (`TT_LAGUNA_PREFIX_CACHE=1`), and partial-hit reads are NOT
> bit-reproducible.** This is a deliberate, accepted property — read this before interpreting your own
> results (e.g. a pass@1 number or a replayed trajectory).

- **Full-hit and cold (no cache reuse) reads are bit-exact** vs a no-cache baseline. A prompt re-sent
  verbatim reproduces its previous generation exactly.
- **Partial-hit reads (cached prefix + new suffix) are NOT bit-exact.** The output matches the cold
  baseline for the deterministic head of generation, then can diverge at a high-entropy (near-tie)
  token. Cause: the suffix read (one chunked-SDPA call from `chunk_start_idx=K`) accumulates in a
  different floating-point order than the cold path. Same non-determinism prefix caching exhibits on
  GPUs, inherent to quantized hardware — **not a correctness bug**.
- **Bit-reproducible mode:** set **`TT_LAGUNA_PREFIX_CACHE=0`** to force every request onto the cold
  path (bit-exact, at the cost of no cache reuse — long prompts re-prefill in full).

## Serving (P150x4)

The serving env is **self-contained**: a dedicated venv holding `ttnn` built from this checkout, stock
`vllm==0.24.0`, the public `tenstorrent/vllm-tt-plugin`, and this model's `vllm_ext` package. Everything
is built from this repo — no hand-prepared environment.

### Quick start
From a fresh clone, one command — it builds the env if it isn't there, then serves:
```bash
models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh
tail -f /home/ttuser/laguna_serve.log
```
First run builds tt-metal (**~1–3 h**) then vLLM from sdist (**~30–45 min**); after that it is just the
**~10 min** server boot. Ready when the log says `Application startup complete` and the KV line reads
`GPU KV cache size: 131,584 tokens ... concurrency 1.00x`.

**Prerequisites:** Linux x86-64, Python 3.12, a tt-metal build toolchain (repo `INSTALLING.md` →
`./install_dependencies.sh`), submodules (`git submodule update --init --recursive`), a Tenstorrent
P150×4 with `tt-smi` on PATH, and the HF model cached (gated — `huggingface-cli login`, ~63 GB).

### The env (`setup_vllm.sh`)
`serve_vllm.sh` calls this for you; run it directly only to rebuild (`--force`) or build elsewhere
(`VLLM_ENV=/path ./setup_vllm.sh`). Default location `.venv/` in the model dir (gitignored). Pins live in
`requirements.txt`, dependency overrides in `overrides.txt`. What it does and why:

| Step | Why |
|---|---|
| `uv venv --python 3.12` | Matches the Python the C++ extension is built against. |
| `./build_metal.sh` (if `_ttnn.so` absent) then `uv pip install -e <repo root>` | **`ttnn` must come from this checkout, not PyPI** (see below). The editable install only wires the built tree in; the build step produces `_ttnn.so`. |
| `VLLM_TARGET_DEVICE=empty uv pip install --no-binary vllm --extra-index-url …/whl/cpu --index-strategy unsafe-best-match --override overrides.txt vllm==0.24.0` | PyPI vLLM is CUDA, so build from sdist against the `empty` target; the `tt` platform comes from the plugin at runtime. The CPU torch index is required or `torch==2.11.0` pulls ~4 GB of `nvidia-*-cu13` wheels. **Slow step.** |
| `uv pip uninstall torchaudio` | `transformers>=5.12` imports it if present, and the wheel pulled alongside CPU torch is unloadable. |
| `uv pip install vllm-tt-plugin @ git+…` | The `tt` platform + `EXTRA_MODELS_DIR` model registration. |
| `uv pip install -e vllm_ext` | The newline-tolerant `poolside_v1` tool-parser override — REQUIRED for `auto` tool-calling (see below). |

**Why `ttnn` is built here, not installed from PyPI:** the 30 sliding-window layers pass
`sliding_window_size` to SDPA. On a prefix-cache hit, prefill resumes at `start_pos > 0` and reads the
cached prefix through `ttnn.transformer.chunked_scaled_dot_product_attention`
(`tt/optimized_decoder.py:_prefill_attention`) — and **no released `ttnn` accepts `sliding_window_size`
on the chunked op** (absent from 0.74.0, 0.75.0, `origin/main`). This branch removes that restriction, so
the serving env must be built from this checkout; a stock wheel raises `TypeError` on the first cache
hit. **Why `uv` and not `pip`:** `ttnn` pins `numpy<2` while vLLM 0.24.0 wants
`opencv-python-headless>=4.13.0` (needs `numpy>=2`) — only `uv pip --override` resolves it (`overrides.txt`).

### Serve by hand (equivalent to `serve_vllm.sh`)
```bash
MODEL_DIR=<repo>/models/autoports/poolside_laguna_xs_2_1
export PYTHONPATH=<repo>                       # so EXTRA_MODELS_DIR's main_class resolves
export EXTRA_MODELS_DIR=$MODEL_DIR/vllm_ext/extra_models
export MESH_DEVICE=P150x4 HF_MODEL=poolside/Laguna-XS-2.1
export TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFIX_CACHE=1 TT_LAGUNA_PREFILL_FAST=1 TT_LAGUNA_HYBRID_KV=0

cd /tmp
$MODEL_DIR/.venv/bin/vllm serve poolside/Laguna-XS-2.1 \
  --trust-remote-code --max-model-len 131072 --max-num-seqs 8 --block-size 64 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 1500000000, "fabric_config": "FABRIC_1D_RING"}}' \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --port 8000 2>&1 | tee /home/ttuser/laguna_serve.log
```
> **Do not set `TT_METAL_HOME`.** `ttnn` self-locates its runtime root from the tree it was installed
> from; pointing `TT_METAL_HOME` at a different tt-metal tree mixes in another version's kernels.

### Verify
```bash
curl -s localhost:8000/health && echo OK
curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])'  # poolside/Laguna-XS-2.1

# Tool-calling (auto): expect finish_reason=tool_calls + get_weather({"city":"Paris","metric":true})
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"poolside/Laguna-XS-2.1",
  "messages":[{"role":"user","content":"What is the weather in Paris right now? Use the get_weather tool with metric units."}],
  "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"metric":{"type":"boolean"}},"required":["city"]}}}],
  "tool_choice":"auto","temperature":0,"max_tokens":1024,
  "chat_template_kwargs":{"enable_thinking":true}}' | python3 -m json.tool
```

### Teardown + mesh reset (ALWAYS between runs)
Hard-killing a `FABRIC_1D_RING` server dirties eth cores — reset the mesh before reopening it.
```bash
models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh stop     # TERM/KILL the group + tt-smi -r all
# by hand:  pkill -TERM -f "vllm serve poolside"; sleep 10; pkill -KILL -f "vllm serve poolside"; tt-smi -r all
```

### Notes / gotchas
- **`TT_LAGUNA_HYBRID_KV=0` is required on vLLM 0.24** to allocate the full 131072 pool. The real hybrid
  layout (10 full + 30 sliding-512) makes 0.24's uniform KV-sufficiency check reject 131072; the gate
  forces the uniform spec so the plugin's single `num_gpu_blocks_override` sizes the whole pool.
- **Agents / concurrent clients: serve batch-1.** Concurrent decode has crashed the engine (Bus error);
  keep one in-flight request per agent. `--max-num-seqs 8` is fine for `vllm bench` throughput runs.
- **`enable_thinking:true`** is the tool-calling default for this model (both true/false verified).
- Supported concurrency target is **8**; conc 32 collapses (TTFT into hundreds of seconds).
- `TT_LAGUNA_PREFILL_FAST=1` ≈ 1.43× prefill. All `TT_LAGUNA_*` env vars are passed through to the worker
  by the plugin — set them in the serve env.
- Decode is batch-flat ~27–29 t/s/u @1k; long-context E2EL is prefill-dominated. Bench ONLY with
  `.venv/bin/vllm bench serve`, never a custom loop.
- **Warmup:** every prefill program shape is compiled before the decode trace is captured
  (`warmup_model_prefill` warms the power-of-two bucket ladder up to the servable context). Do not set
  `TT_LAGUNA_PREFILL_WARM_CAP` below `--max-model-len` for serving.

## How it runs fork-free (`vllm_ext`)

The whole tt-metal-side delta to run on **stock** vLLM 0.24.0 + the **public** plugin is the `vllm_ext/`
package — no plugin edit, no vLLM/plugin PR:
- The engine-core/launcher hooks that used to require a vLLM fork are **upstreamed in vLLM 0.24.0**, so
  the public plugin runs on stock vLLM.
- Laguna registers via the plugin's **`EXTRA_MODELS_DIR`** mechanism — a bundle folder with
  `vllm_ext/extra_models/laguna/vllm_metadata.json` (`arch=LagunaForCausalLM` →
  `…tt.generator_vllm:LagunaForCausalLM`; the plugin prefixes `TT`). No plugin source change.
- The `poolside_v1` tool + reasoning parsers ship **in stock vLLM 0.24.0**. One catch fixed here: the
  stock tool parser's regex requires a newline after the function name (`<tool_call>NAME\n…`), but this
  checkpoint emits the arg tags immediately after the name (`<tool_call>NAME<arg_key>…`, no newline), so
  stock `auto` tool-calling silently returns `finish_reason=stop`. `vllm_ext/laguna_vllm_ext` is a
  `vllm.general_plugins` entry point that eagerly re-registers `poolside_v1` with a **newline-tolerant**
  regex (parses both grammars). `setup_vllm.sh` installs it; existing `--tool-call-parser poolside_v1`
  flags are unchanged.

Refresh just this package in an existing env:
`.venv/bin/pip install -e vllm_ext`.

## Coding agent (`pool`)

`pool` (Poolside's terminal agent, github.com/poolsideai/pool) talks to this server in **standalone mode**.

```bash
# 1. Serve Laguna (above), confirm /v1/models returns poolside/Laguna-XS-2.1.
# 2. Install pool:
curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh   # accept EULA; adds ~/.local/bin
# 3. Point pool at the local endpoint — POOLSIDE_STANDALONE_BASE_URL is the mode switch:
export POOLSIDE_STANDALONE_BASE_URL=http://localhost:8000/v1    # note the /v1 (without it: 404 default-agent)
export POOLSIDE_API_KEY=EMPTY                                   # any value; local server ignores it
export POOLSIDE_STANDALONE_MODEL="poolside/Laguna-XS-2.1"
export POOLSIDE_STANDALONE_CONTEXT_LENGTH=131072
# 4a. Interactive:      cd <project> && pool
# 4b. One-shot (CI):    pool exec --unsafe-auto-allow --sandbox disabled -p "…task…"
```
Tool-calling round-trips automatically through `poolside_v1`. The model is throughput-limited
(~25–29 t/s/u decode) and re-prefills the growing transcript each turn, so agentic tasks are minutes,
not seconds — prefer focused prompts and **one active `pool` session at a time**.

## Tests

```bash
cd /tmp && TT_METAL_HOME=<repo>/.local/… PYTHONPATH=<repo> \
  <model dir>/.venv/bin/python -m pytest <test> -q     # run from /tmp, not the repo cwd (JIT kernel path)
```
- `tests/test_prefill_buckets.py` — device-free prefill warm-set / advertised-context invariants (CI).
- `tests/test_optimized_decoder.py`, `tests/test_multichip_decoder.py` — layer PCC ≥ 0.995 vs HF.
  (`LAGUNA_MC_CLASS=optimized` exercises the packed gate+up path; default is the unpacked baseline.)
- `tests/full_model_checks.py` — prefill top-1/5/100 vs the AIME24 reference
  (`tests/reference_outputs/readiness_aime24_chat.refpt`).
