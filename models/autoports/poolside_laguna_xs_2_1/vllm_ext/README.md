# Laguna-XS-2.1 on the public `vllm-tt-plugin` + stock vLLM (fork-free)

This directory is the **entire tt-metal-side delta** to run Laguna-XS-2.1 on **stock upstream vLLM `0.24.0`**
plus the **public `tenstorrent/vllm-tt-plugin`** — with **no Tenstorrent vLLM fork**, no plugin edit, and no
vLLM/plugin PR. Everything Laguna-specific lives here in the tt-metal branch.

## Why this is so small
- The plugin hooks that used to require the fork (engine-core/launcher class injection, host-device handling)
  are **upstreamed into vLLM 0.24.0** — the public plugin runs on stock vLLM.
- The `poolside_v1` **tool + reasoning parsers are built into vLLM 0.24.0** (`vllm/tool_parsers/__init__.py`
  and `vllm/reasoning/__init__.py`), so nothing to vendor — just pass `--tool-call-parser poolside_v1
  --reasoning-parser poolside_v1`.
- The public plugin registers out-of-tree models via **`EXTRA_MODELS_DIR`**: a dir of bundle folders, each with
  a `vllm_metadata.json` (`arch` + `main_class`). It registers `TT<arch>` and appends the bundle to `sys.path`.
  So Laguna registration is just the bundle below — no plugin source change.

The only tt-metal artifact needed is `extra_models/laguna/vllm_metadata.json` (arch `LagunaForCausalLM` →
`models.autoports.poolside_laguna_xs_2_1.tt.generator_vllm:LagunaForCausalLM`; the plugin prefixes `TT`).

## One-time environment setup (fork-free stack)
In a tt-metal `python_env` (provides torch + ttnn). This installs **stock PyPI vLLM 0.24.0** + the public plugin:
```bash
git clone https://github.com/tenstorrent/vllm-tt-plugin && cd vllm-tt-plugin
source docs/install-vllm-tt.sh          # VLLM_TARGET_DEVICE=empty vllm==0.24.0 (stock) + the plugin
python -c "import vllm; print(vllm.__version__)"          # -> 0.24.0
python -c "from vllm.tool_parsers import ToolParserManager as M; print('poolside_v1' in M.tool_parsers)"  # True
```
> Do NOT install this into the current `.tenstorrent-venv` (which carries the fork build) — use a dedicated
> tt-metal env so the fork stack stays intact until cutover.

## Serve Laguna (fork-free)
```bash
export EXTRA_MODELS_DIR=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/vllm_ext/extra_models
export TT_VLLM_BUILTIN_MODELS=0          # optional: rely solely on EXTRA_MODELS_DIR
export PYTHONPATH=/home/ttuser/dev/tt-metal:$PYTHONPATH   # so main_class resolves
export MESH_DEVICE=P150x4

vllm serve poolside/Laguna-XS-2.1 \
  --trust-remote-code --max-model-len 131072 --max-num-seqs 8 --block-size 64 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 1500000000, "fabric_config": "FABRIC_1D_RING"}}' \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 --reasoning-parser poolside_v1
```
(TTPlatform auto-activates because `ttnn` is importable. `TT_LAGUNA_*` decode flags still apply; keep them in
the env — the plugin passes model-side env through.)

## Verify fork-free
```bash
python -c "import vllm,vllm_tt_plugin; print(vllm.__file__); print(vllm_tt_plugin.__file__)"
# neither path is under .local/.../tt-metal/vllm  (the fork); vllm is 0.24.0, plugin is the public package.
curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])'  # poolside/Laguna-XS-2.1
```

See `../doc/vllm_integration/pool_agent_getting_started.md` for the `pool` agent against this endpoint, and the
plan `~/.claude/plans/lovely-wibbling-clover.md`.
