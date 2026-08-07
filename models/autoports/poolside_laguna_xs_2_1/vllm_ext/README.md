# Laguna-XS-2.1 on the public `vllm-tt-plugin` + stock vLLM

This directory is the **entire tt-metal-side delta** to run Laguna-XS-2.1 on **stock upstream vLLM `0.24.0`**
plus the **public `tenstorrent/vllm-tt-plugin`** — with no plugin edit and no vLLM/plugin PR. Everything Laguna-specific lives here in the tt-metal branch.

## Why this is so small
- The plugin hooks that used to require a vLLM fork (engine-core/launcher class injection, host-device
  handling) are **upstreamed into vLLM 0.24.0** — the public plugin runs on stock vLLM.
- The `poolside_v1` **tool + reasoning parsers are built into vLLM 0.24.0** (`vllm/tool_parsers/__init__.py`
  and `vllm/reasoning/__init__.py`) — pass `--tool-call-parser poolside_v1 --reasoning-parser poolside_v1`.
  **One catch (fixed here):** the *stock* `poolside_v1` tool parser's detail regex requires a newline after the
  function name (`<tool_call>NAME\n...`), but this Laguna checkpoint emits the arg tags immediately after the
  name (`<tool_call>NAME<arg_key>...`, no newline) — so stock `auto` tool-calling silently returns
  `finish_reason=stop` with the raw `<tool_call>` left in `content`. The tiny installable package
  `laguna_vllm_ext/` (this dir) fixes it: a `vllm.general_plugins` entry point that eagerly re-registers
  `poolside_v1` with a **newline-tolerant** regex (parses both grammars). No stock/plugin edit, no vendored
  parser copy — just `pip install -e` this dir into the serving env (step below). The reasoning parser is
  unaffected.
- The public plugin registers out-of-tree models via **`EXTRA_MODELS_DIR`**: a dir of bundle folders, each with
  a `vllm_metadata.json` (`arch` + `main_class`). It registers `TT<arch>` and appends the bundle to `sys.path`.
  So Laguna registration is just the bundle below — no plugin source change.

The only tt-metal artifact needed is `extra_models/laguna/vllm_metadata.json` (arch `LagunaForCausalLM` →
`models.autoports.poolside_laguna_xs_2_1.tt.generator_vllm:LagunaForCausalLM`; the plugin prefixes `TT`).

## One-time environment setup
Use the model's own setup script — it builds a dedicated venv with stock vLLM 0.24.0, the public plugin,
and this package, and then verifies the result:
```bash
../setup_vllm.sh          # -> <model dir>/.venv ; see ../doc/vllm_integration/serve_vllm.md §1
```
That runbook is the single canonical recipe; don't hand-roll a second one here. To refresh only this
package inside an existing env:
```bash
<model dir>/.venv/bin/python -m pip install -e <model dir>/vllm_ext
<model dir>/.venv/bin/python - <<'EOF'   # override resolves via vLLM's real plugin loader
from vllm.plugins import load_general_plugins; load_general_plugins()
from vllm.tool_parsers import ToolParserManager as M
print(M.get_tool_parser("poolside_v1").__module__)   # -> laguna_vllm_ext
EOF
```

## Serve Laguna
Use `../serve_vllm.sh`, which sets all of this and backgrounds the server. By hand:
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

## Verify
```bash
python -c "import vllm,vllm_tt_plugin; print(vllm.__file__); print(vllm_tt_plugin.__file__)"
# neither path is inside a tt-metal tree; vllm is 0.24.0, plugin is the public package.
curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])'  # poolside/Laguna-XS-2.1
```

See `../doc/vllm_integration/serve_vllm.md` for the full serve runbook and
`../doc/vllm_integration/pool_agent_getting_started.md` for the `pool` agent against this endpoint.
