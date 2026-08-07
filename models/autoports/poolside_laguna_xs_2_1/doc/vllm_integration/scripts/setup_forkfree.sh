#!/bin/bash
# ONE-TIME setup of the fork-free serving env for Laguna-XS-2.1:
#   stock vLLM 0.24.0 (VLLM_TARGET_DEVICE=empty) + public tenstorrent/vllm-tt-plugin + tt-metal vllm_ext.
# Clones the working tt-metal venv and swaps ONLY vLLM, so .tenstorrent-venv (the fork serve) stays intact.
# Idempotent-ish: safe to re-run; it recreates the env dir. Run once, then use serve_forkfree.sh to serve.
set -e

FF_ENV="${FF_ENV:-/home/ttuser/.venv_laguna_forkfree}"
PLUGIN_SRC="${PLUGIN_SRC:-/home/ttuser/src/vllm-tt-plugin}"
VLLM_EXT=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/vllm_ext
BASE_VENV=/home/ttuser/.tenstorrent-venv

echo "==> [1/4] clone $BASE_VENV -> $FF_ENV (leaves the fork serve untouched)"
[ -e "$FF_ENV" ] && { echo "    $FF_ENV exists; remove it first or set FF_ENV=... to a new path"; exit 1; }
cp -a "$BASE_VENV" "$FF_ENV"
FF="$FF_ENV/bin"

echo "==> [2/4] swap fork vLLM -> stock vllm==0.24.0"
"$FF/pip" uninstall -y vllm vllm-tt-plugin >/dev/null 2>&1 || true
VLLM_TARGET_DEVICE=empty "$FF/pip" install "vllm==0.24.0" --extra-index-url https://download.pytorch.org/whl/cpu

echo "==> [3/4] install the PUBLIC vllm-tt-plugin"
[ -d "$PLUGIN_SRC" ] || git clone https://github.com/tenstorrent/vllm-tt-plugin "$PLUGIN_SRC"
"$FF/pip" install -e "$PLUGIN_SRC"

echo "==> [4/4] install the tt-metal vllm_ext (poolside_v1 tool-parser override)"
"$FF/pip" install -e "$VLLM_EXT"

echo "==> verify fork-free + override"
"$FF/python" - <<'EOF'
import vllm, vllm_tt_plugin
assert ".local/lib/model-bringup/tt-metal/vllm" not in vllm.__file__, "still on the FORK vllm!"
print("  vllm", vllm.__version__, "(fork-free)")
from vllm.plugins import load_general_plugins; load_general_plugins()
from vllm.tool_parsers import ToolParserManager as M
mod = M.get_tool_parser("poolside_v1").__module__
assert mod == "laguna_vllm_ext", f"poolside_v1 override NOT active (resolved to {mod})"
print("  poolside_v1 override active ->", mod)
EOF

echo
echo "DONE. Fork-free env ready at $FF_ENV"
echo "Serve with:  FF=$FF_ENV/bin $(dirname "$0")/serve_forkfree.sh"
