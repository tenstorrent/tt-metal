# Serving Laguna-XS-2.1 on vLLM (stock vLLM 0.24.0 + tenstorrent/vllm-tt-plugin)

Runbook for serving `poolside/Laguna-XS-2.1` on **P150×4**. The serving env is self-contained: a
dedicated venv built from a PyPI `ttnn` wheel plus stock upstream `vllm==0.24.0`, the public
`tenstorrent/vllm-tt-plugin`, and this model's `vllm_ext` package. **No tt-metal build is
required.** Device-verified 2026-08-06 (full 131072 KV pool, decode ~29 t/s/u @1k C1, `auto`
tool-calling green).

--------------------------------------------------------------------------------------------------------------
## Quick start
From a fresh clone of tt-metal, one command — it builds the env if it isn't there, then serves:

```bash
models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh
tail -f /home/ttuser/laguna_serve.log
```

Budget **~30–45 min for the first run** (vLLM is built from sdist; see §1) plus **~10 min** for
the server boot itself. Re-runs skip straight to the boot. Ready when the log says
`Application startup complete`.

Prerequisites: Linux x86-64 with glibc ≥ 2.34, Python 3.12 available, a Tenstorrent P150×4 with
`tt-smi` on PATH, and the HF model cached (gated — `huggingface-cli login`, ~63 GB).

--------------------------------------------------------------------------------------------------------------
## 1. The env (`setup_vllm.sh`)
`serve_vllm.sh` calls this for you; run it directly only to rebuild (`--force`) or to build into a
different location (`VLLM_ENV=/path ./setup_vllm.sh`). Default location is `.venv/` in the model
dir (gitignored).

Pins live in `requirements.txt`, the dependency overrides in `overrides.txt`. What the script
does and why:

| Step | Why |
|---|---|
| `uv venv --python 3.12` | The `ttnn` wheels are `cp312` with no stable-ABI tag, so the Python minor version must match. |
| `uv pip install ttnn==0.74.0` | 0.74.0 is the release cut from the `v0.74.0-dev` line this checkout sits on. Ships manylinux wheels, so no tt-metal build. Declares `numpy<2` + `loguru` only — no torch pin. |
| `VLLM_TARGET_DEVICE=empty uv pip install --no-binary vllm --extra-index-url …/whl/cpu --index-strategy unsafe-best-match --override overrides.txt vllm==0.24.0` | The PyPI vLLM wheel is CUDA, so it must be built from sdist against the `empty` target; the `tt` platform comes from the plugin at runtime. The CPU torch index is required — without it `torch==2.11.0` resolves to the CUDA build and pulls ~4 GB of `nvidia-*-cu13` wheels. **This is the slow step.** |
| `uv pip uninstall torchaudio` | `transformers>=5.12` imports it if merely installed, and the wheel pulled in alongside CPU torch is unloadable. |
| `uv pip install vllm-tt-plugin @ git+…` | Provides the `tt` platform and `EXTRA_MODELS_DIR` model registration. |
| `uv pip install -e vllm_ext` | The newline-tolerant `poolside_v1` tool-parser override — REQUIRED for `auto` tool-calling; see `../../vllm_ext/README.md`. |

**Why `uv` and not `pip`:** `ttnn` pins `numpy<2` while vLLM 0.24.0 wants
`opencv-python-headless>=4.13.0`, which needs `numpy>=2`. That is a hard conflict, and `pip -c`
constraints cannot resolve it — only `uv pip --override` can. See `overrides.txt` for the full
rationale.

The script finishes by verifying the env, and each assertion is there because it has a real
failure mode: `vllm` is the stock install; `ttnn` SDPA exposes `sliding_window_size` (every
sliding layer needs it); `models.autoports.…tt.generator_vllm` imports (what `EXTRA_MODELS_DIR`
resolves at serve time); and `poolside_v1` resolves to `laguna_vllm_ext`, not the stock parser.

--------------------------------------------------------------------------------------------------------------
## 2. Serve
`serve_vllm.sh` backgrounds the server with `setsid` and streams the full raw log. The equivalent
by hand:

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
> **Do not set `TT_METAL_HOME`.** A wheel `ttnn` self-locates its runtime root; pointing
> `TT_METAL_HOME` at some other tt-metal tree mixes in a different version's kernels.
>
> **Boot takes ~10 min** (weight convert/tilize + prefill+decode warmup). Ready at
> `Application startup complete`. The KV line should read
> `GPU KV cache size: 131,584 tokens ... concurrency 1.00x`.

--------------------------------------------------------------------------------------------------------------
## 3. Verify
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

--------------------------------------------------------------------------------------------------------------
## 4. Teardown + mesh reset (ALWAYS between runs)
Hard-killing a `FABRIC_1D_RING` server dirties eth cores — reset the mesh before reopening it.
```bash
models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh stop     # TERM/KILL the group + tt-smi -r all
```
By hand:
```bash
pkill -TERM -f "vllm serve poolside"; sleep 10; pkill -KILL -f "vllm serve poolside"; sleep 3
tt-smi -r all      # reset all P150s before the next boot
```

--------------------------------------------------------------------------------------------------------------
## Notes / gotchas
- **`TT_LAGUNA_HYBRID_KV=0` is required on vLLM 0.24** to allocate the full 131072 pool. The model's real
  hybrid layout (10 full + 30 sliding-512) makes 0.24's uniform KV-sufficiency check reject 131072; the gate
  forces the uniform spec so the plugin's single `num_gpu_blocks_override` sizes the whole pool.
- **Agents / concurrent clients: serve batch-1.** Concurrent decode has crashed the engine (Bus error) in this
  bringup; keep one in-flight request per agent. `--max-num-seqs 8` is fine for `vllm bench` throughput runs.
- **Tool-calling needs the `vllm_ext` override** (§1) — without it, stock `poolside_v1` misses this
  checkpoint's newline-free `<tool_call>` grammar and `auto` silently returns `finish_reason=stop`.
- **`enable_thinking:true`** is the tool-calling default for this model (both true/false verified working).
- `TT_LAGUNA_PREFILL_FAST=1` ≈ 1.43× prefill; `TT_LAGUNA_PREFIX_CACHE=1` enables APC warm reads. Both safe.
- All `TT_LAGUNA_*` env vars are passed through to the worker by the plugin — set them in the serve env.
- Decode is batch-flat ~27–29 t/s/u @1k; long-context E2EL is prefill-dominated. Bench ONLY with
  `vllm bench serve` (`.venv/bin/vllm bench serve`), never a custom loop.
