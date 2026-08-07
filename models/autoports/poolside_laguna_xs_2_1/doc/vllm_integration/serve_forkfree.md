# Serving Laguna-XS-2.1 — fork-free stack (stock vLLM 0.24.0 + public vllm-tt-plugin)

Full runbook for serving `poolside/Laguna-XS-2.1` on **P150×4** with **no Tenstorrent vLLM fork**: stock upstream
`vllm==0.24.0` + the public `tenstorrent/vllm-tt-plugin` + the tt-metal `vllm_ext` package. Device-verified
2026-08-06 (full 131072 KV pool, decode ~29 t/s/u @1k C1, `auto` tool-calling green).

--------------------------------------------------------------------------------------------------------------
## 0. Prerequisites (once)
- Built tt-metal at `/home/ttuser/.local/lib/model-bringup/tt-metal` (provides the built `ttnn`).
- HF model cached: `poolside/Laguna-XS-2.1` (gated; `huggingface-cli login` if not present).
- A **fork-free Python env** (Section 1). The working `.tenstorrent-venv` carries the FORK build — do NOT
  install into it; use a dedicated env so the fork stack stays intact.

--------------------------------------------------------------------------------------------------------------
## 1. Build the fork-free env (once)  →  `$FF` = its `bin/`
The env needs tt-metal's full dep closure (ttnn, loguru, …) which is why we clone the tt-metal venv and swap
ONLY vLLM, rather than hand-rolling a venv. (vLLM 0.24.0 hard-pins torchvision 0.26 → torch 2.11.0+cpu; the
built ttnn runs fine on torch 2.11.)

```bash
# 1a. Clone the working tt-metal venv into a durable location (leaves .tenstorrent-venv / the fork serve intact)
cp -a /home/ttuser/.tenstorrent-venv /home/ttuser/.venv_laguna_forkfree
FF=/home/ttuser/.venv_laguna_forkfree/bin

# 1b. Replace the fork vLLM + bundled plugin with STOCK vLLM 0.24.0 + the PUBLIC plugin
$FF/pip uninstall -y vllm vllm-tt-plugin
git clone https://github.com/tenstorrent/vllm-tt-plugin /home/ttuser/src/vllm-tt-plugin   # if not already cloned
cd /home/ttuser/src/vllm-tt-plugin
VLLM_TARGET_DEVICE=empty $FF/pip install vllm==0.24.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu       # stock PyPI vLLM, CPU torch
$FF/pip install -e .                                              # the public plugin (fork-free)

# 1c. Install the tt-metal vLLM extension (newline-tolerant poolside_v1 tool-parser override — REQUIRED for
#     `auto` tool-calling; see vllm_ext/README.md for why)
$FF/pip install -e /home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/vllm_ext
```

Verify the env is fork-free and correct:
```bash
$FF/python - <<'EOF'
import vllm, vllm_tt_plugin
print("vllm", vllm.__version__, vllm.__file__)                  # 0.24.0(+empty), NOT under .local/.../vllm (fork)
print("plugin", vllm_tt_plugin.__file__)                        # public package, NOT the fork bundle
from vllm.plugins import load_general_plugins; load_general_plugins()
from vllm.tool_parsers import ToolParserManager as M
print("poolside_v1 ->", M.get_tool_parser("poolside_v1").__module__)   # laguna_vllm_ext  (override active)
EOF
```

--------------------------------------------------------------------------------------------------------------
## 2. Serve
```bash
export FF=/home/ttuser/.venv_laguna_forkfree/bin
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export PYTHONPATH=/home/ttuser/dev/tt-metal                       # so main_class (generator_vllm) resolves
export EXTRA_MODELS_DIR=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/vllm_ext/extra_models
export MESH_DEVICE=P150x4 HF_MODEL=poolside/Laguna-XS-2.1

# Decode/perf + KV flags (see notes):
export TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFIX_CACHE=1 TT_LAGUNA_PREFILL_FAST=1 TT_LAGUNA_HYBRID_KV=0

cd /tmp
$FF/vllm serve poolside/Laguna-XS-2.1 \
  --trust-remote-code --max-model-len 131072 --max-num-seqs 8 --block-size 64 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 1500000000, "fabric_config": "FABRIC_1D_RING"}}' \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --port 8000 2>&1 | tee /home/ttuser/laguna_serve.log
```
> **Boot takes ~10 min** (weight convert/tilize + prefill+decode warmup). Watch it:
> ```
> tail -f /home/ttuser/laguna_serve.log
> ```
> Ready when you see `Application startup complete`. The KV line should read
> `GPU KV cache size: 131,584 tokens ... concurrency 1.00x`.

Or use the launcher (backgrounds it + streams a full log):
```bash
/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/vllm_integration/scripts/serve_forkfree.sh
tail -f /home/ttuser/laguna_serve.log
```

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
pkill -TERM -f "vllm serve poolside"; sleep 10; pkill -KILL -f "vllm serve poolside"; sleep 3
/home/ttuser/.tenstorrent-venv/bin/tt-smi -r all      # reset all P150s before the next boot
```

--------------------------------------------------------------------------------------------------------------
## Notes / gotchas
- **`TT_LAGUNA_HYBRID_KV=0` is required on vLLM 0.24** to allocate the full 131072 pool. The model's real
  hybrid layout (10 full + 30 sliding-512) makes 0.24's uniform KV-sufficiency check reject 131072; the gate
  forces the uniform spec (fork parity) so the plugin's single `num_gpu_blocks_override` sizes the whole pool.
- **Agents / concurrent clients: serve batch-1.** Concurrent decode has crashed the engine (Bus error) in this
  bringup; keep one in-flight request per agent. `--max-num-seqs 8` is fine for `vllm bench` throughput runs.
- **Tool-calling needs the `vllm_ext` override** (Section 1c) — without it, stock `poolside_v1` misses this
  checkpoint's newline-free `<tool_call>` grammar and `auto` silently returns `finish_reason=stop`.
- **`enable_thinking:true`** is the tool-calling default for this model (both true/false verified working).
- `TT_LAGUNA_PREFILL_FAST=1` ≈ 1.43× prefill; `TT_LAGUNA_PREFIX_CACHE=1` enables APC warm reads. Both safe.
- All `TT_LAGUNA_*` env vars are passed through to the worker by the plugin — set them in the serve env.
- Decode is batch-flat ~27–29 t/s/u @1k; long-context E2EL is prefill-dominated. Bench ONLY with
  `vllm bench serve` from `.venv_benchmarks_vllm` (never a custom loop).
