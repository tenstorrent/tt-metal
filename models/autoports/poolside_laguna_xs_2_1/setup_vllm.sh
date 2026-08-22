#!/bin/bash
# Build the self-contained vLLM serving env for Laguna-XS-2.1: stock vLLM 0.24.0 + the public
# tenstorrent/vllm-tt-plugin + this model's vllm_ext, with ttnn built and installed from this checkout.
#   Setup:  ./setup_vllm.sh            (into ./.venv; --force to rebuild from scratch)
#   Serve:  ./serve_vllm.sh            (runs this automatically if the env is missing)
# Pins live in requirements.txt; the numpy/opencv overrides live in overrides.txt.
set -euo pipefail

MODEL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$MODEL_DIR/../../.." && pwd)
VLLM_ENV="${VLLM_ENV:-$MODEL_DIR/.venv}"
PYTORCH_CPU_INDEX="${PYTORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# Pins are read from requirements.txt so there is one place to bump them.
req() { grep -m1 "^$1" "$MODEL_DIR/requirements.txt" | sed 's/[[:space:]]*#.*//'; }
VLLM_PIN=$(req 'vllm==')
PLUGIN_PIN=$(req 'vllm-tt-plugin @')
PYTEST_PIN=$(req 'pytest==')
PYTEST_TIMEOUT_PIN=$(req 'pytest-timeout==')
for p in "$VLLM_PIN" "$PLUGIN_PIN" "$PYTEST_PIN" "$PYTEST_TIMEOUT_PIN"; do
  [ -n "$p" ] || { echo "ERROR: could not read pins from $MODEL_DIR/requirements.txt"; exit 1; }
done
PLUGIN_SHA="${PLUGIN_PIN##*@}"
[[ "$PLUGIN_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "ERROR: vllm-tt-plugin must be pinned to a full commit SHA in $MODEL_DIR/requirements.txt"
  exit 1
}

echo "=============================================================================="
echo " Laguna-XS-2.1 vLLM env -> $VLLM_ENV"
echo "   ttnn: built from $REPO_ROOT | $VLLM_PIN"
echo " First run builds tt-metal (~1-3 h) then vLLM from sdist (~30-45 min)."
echo " Re-runs on an existing build/env are quick. Serving is a separate ~10 min boot."
echo "=============================================================================="

# ---- uv -----------------------------------------------------------------------------------
# uv, not pip: the numpy<2 (ttnn) vs opencv>=4.13 (vLLM) conflict needs --override, which pip
# has no equivalent for. Bootstrap with the repo's pinned installer.
if ! command -v uv >/dev/null 2>&1; then
  echo "==> [0/6] installing uv (repo-pinned)"
  bash "$REPO_ROOT/scripts/install-uv.sh"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || { echo "ERROR: uv still not on PATH after install"; exit 1; }
fi

# ---- venv ---------------------------------------------------------------------------------
# Python 3.12: the ttnn wheels are cp312 (no stable-ABI tag), so the minor version must match.
if [ "$FORCE" = 1 ] && [ -e "$VLLM_ENV" ]; then
  echo "==> [1/6] --force: removing $VLLM_ENV"
  rm -rf "$VLLM_ENV"
fi
if [ -x "$VLLM_ENV/bin/python" ]; then
  echo "==> [1/6] reusing existing env $VLLM_ENV (--force to rebuild)"
else
  echo "==> [1/6] creating venv $VLLM_ENV (python 3.12)"
  uv python install 3.12 >/dev/null 2>&1 || true
  uv venv --python 3.12 "$VLLM_ENV"
fi
PY="$VLLM_ENV/bin/python"

# ---- ttnn (built from this checkout) --------------------------------------------------------
# NOT the PyPI wheel: this checkout carries an SDPA change no release has — the chunked path
# accepts sliding_window_size, which the 30 sliding-window layers need whenever a prefix-cache
# hit makes prefill start at start_pos>0 (tt/optimized_decoder.py:_prefill_attention). A wheel
# ttnn raises TypeError there, so ttnn must come from this tree.
if [ ! -f "$REPO_ROOT/ttnn/ttnn/_ttnn.so" ]; then
  echo "==> [2/6] building tt-metal (no _ttnn.so yet) — this is the long one, ~1-3 h"
  ( cd "$REPO_ROOT" && ./build_metal.sh )
else
  echo "==> [2/6] reusing existing tt-metal build ($REPO_ROOT/ttnn/ttnn/_ttnn.so)"
fi
# Editable install: setup.py skips cmake for editable/srcdir installs and just wires the tree up,
# so the build above is what actually produces _ttnn.so. Needs setuptools-scm (git describe).
uv pip install --python "$PY" setuptools==80.10.2 setuptools-scm==8.1.0 wheel
uv pip install --python "$PY" --no-build-isolation -e "$REPO_ROOT"

echo "==> [3/6] $VLLM_PIN from sdist (VLLM_TARGET_DEVICE=empty, CPU torch) — the slow step"
# The CPU torch index is required: without it vLLM's torch==2.11.0 resolves to the default CUDA
# build and drags in ~4 GB of nvidia-*-cu13 wheels. There is no NVIDIA device here and the
# validated env is torch 2.11.0+cpu. --index-strategy unsafe-best-match lets uv consider the
# extra index for a package that also exists on PyPI (same pattern as the repo's create_venv.sh).
VLLM_TARGET_DEVICE=empty uv pip install --python "$PY" --no-binary vllm \
  --extra-index-url "$PYTORCH_CPU_INDEX" --index-strategy unsafe-best-match \
  --override "$MODEL_DIR/overrides.txt" "$VLLM_PIN"
# transformers>=5.12 imports torchaudio if it is merely installed, and the wheel pulled in
# alongside CPU torch is unloadable.
uv pip uninstall --python "$PY" torchaudio >/dev/null 2>&1 || true

echo "==> [4/6] public vllm-tt-plugin (tt platform + EXTRA_MODELS_DIR registration)"
uv pip install --python "$PY" "$PLUGIN_PIN"

echo "==> [5/6] this model's vllm_ext compatibility hooks + device-free test dependency"
uv pip install --python "$PY" -e "$MODEL_DIR/vllm_ext"
uv pip install --python "$PY" "$PYTEST_PIN" "$PYTEST_TIMEOUT_PIN"

# ---- verify -------------------------------------------------------------------------------
echo "==> [6/6] verifying the env"
EXPECTED_PLUGIN_SHA="$PLUGIN_SHA" PYTHONPATH="$REPO_ROOT" "$PY" - <<'EOF'

import importlib.metadata
import json
import os

import ttnn
import vllm
import vllm_tt_plugin

# vLLM must be the stock install, not a vLLM checkout vendored inside a tt-metal tree.
assert "/tt-metal/vllm" not in vllm.__file__, f"vllm resolves into a tt-metal tree: {vllm.__file__}"
print("  vllm", vllm.__version__)
print("  ttnn", getattr(ttnn, "__version__", "(no __version__)"), "->", ttnn.__file__)
print("  plugin", vllm_tt_plugin.__file__)

# The TT plugin monkeypatches vLLM internals, so a moving branch is unsafe. Verify
# the installed PEP 610 source metadata, not just the package's static version.
plugin_dist = importlib.metadata.distribution("vllm-tt-plugin")
direct_url_text = plugin_dist.read_text("direct_url.json")
assert direct_url_text, "vllm-tt-plugin has no direct_url.json; expected the pinned git install"
plugin_commit = json.loads(direct_url_text).get("vcs_info", {}).get("commit_id")
expected_plugin_commit = os.environ["EXPECTED_PLUGIN_SHA"]
assert plugin_commit == expected_plugin_commit, (
    f"vllm-tt-plugin commit mismatch: installed {plugin_commit!r}, expected {expected_plugin_commit}"
)
print("  plugin commit", plugin_commit)

# CPU torch, not the CUDA default: there is no NVIDIA device here, and the CUDA build drags in
# ~4 GB of nvidia-*-cu13 wheels. Catches a dropped --extra-index-url.
import torch
assert torch.__version__.endswith("+cpu"), f"expected CPU torch, got {torch.__version__}"
print("  torch", torch.__version__)

# The Laguna decoders pass sliding_window_size to three SDPA entry points (30 of 40 layers are
# sliding-window). These are nanobind functions, so inspect.signature() cannot see their named
# args — the signature is embedded in __doc__, which is what we check.
missing = [
    name
    for name in (
        "scaled_dot_product_attention",
        "chunked_scaled_dot_product_attention",
        "paged_scaled_dot_product_attention_decode",
    )
    if "sliding_window_size" not in (getattr(ttnn.transformer, name).__doc__ or "")
]
assert not missing, (
    "this ttnn build's "
    + ", ".join(missing)
    + " does not accept sliding_window_size, which the sliding-window layers require.\n"
    "  No released ttnn wheel supports it on the chunked op — see README.md 'Serving'."
)
print("  ttnn SDPA sliding_window_size: present on all three entry points")

# Prefix-resume replays update the suffix start through a device scalar without recompiling the
# chunked SDPA program. Verify the named runtime tensor survived the local TTNN build/install.
chunked_sdpa_doc = ttnn.transformer.chunked_scaled_dot_product_attention.__doc__ or ""
assert "chunk_start_idx_tensor" in chunked_sdpa_doc, (
    "this ttnn build's chunked_scaled_dot_product_attention does not accept "
    "chunk_start_idx_tensor; runtime-stable prefix resume is unavailable"
)
print("  ttnn chunked SDPA chunk_start_idx_tensor: present")

# Trace replay writes embeddings into a stable preallocated buffer. A build that lacks this argument
# silently forces a new output allocation and cannot satisfy the prefix-cache trace contract.
embedding_doc = ttnn.embedding.__doc__ or ""
assert "output_tensor" in embedding_doc, (
    "this ttnn build's embedding does not accept output_tensor; stable trace replay is unavailable"
)
print("  ttnn embedding output_tensor: present")

# Qualification checks program-cache cardinality across changing resume offsets. Keep setup
# device-free by verifying the bound MeshDevice methods rather than opening a device here.
program_cache_methods = (
    "enable_program_cache",
    "disable_and_clear_program_cache",
    "num_program_cache_entries",
    "set_program_cache_misses_allowed",
)
missing_program_cache_methods = [
    name for name in program_cache_methods if not hasattr(ttnn.MeshDevice, name)
]
assert not missing_program_cache_methods, (
    "this ttnn build is missing MeshDevice program-cache API(s): "
    + ", ".join(missing_program_cache_methods)
)
print("  ttnn MeshDevice program-cache APIs: present")

# What EXTRA_MODELS_DIR will resolve at serve time (needs PYTHONPATH=<repo root>).
from models.autoports.poolside_laguna_xs_2_1.tt.generator_vllm import LagunaForCausalLM
capabilities = LagunaForCausalLM.model_capabilities
assert "supports_prefix_caching_with_sliding_window" in capabilities
expected_prefix_capability = os.environ.get("TT_LAGUNA_PREFIX_CACHE", "0") == "1"
assert capabilities["supports_prefix_caching"] is expected_prefix_capability
assert capabilities["supports_prefix_caching_with_sliding_window"] is expected_prefix_capability
print("  generator_vllm: importable; prefix capability:", expected_prefix_capability)

# The stock poolside_v1 tool parser misses this checkpoint's newline-free <tool_call> grammar,
# so `auto` tool-calling silently returns finish_reason=stop unless vllm_ext wins registration.
from vllm.plugins import load_general_plugins; load_general_plugins()
from vllm.tool_parsers import ToolParserManager as M
mod = M.get_tool_parser("poolside_v1").__module__
assert mod == "laguna_vllm_ext", f"poolside_v1 override NOT active (resolved to {mod})"
print("  poolside_v1 override active ->", mod)

# The model-local general plugin must wrap the TT platform before VllmConfig's
# config hook runs. The wrapper is capability-gated, so other TT models retain
# the pinned public plugin's sliding-window policy.
from laguna_vllm_ext.prefix_cache import sliding_window_prefix_cache_patch_is_installed
assert sliding_window_prefix_cache_patch_is_installed(), "sliding-window prefix-cache wrapper NOT active"
print("  sliding-window prefix-cache capability wrapper: active")

# Canonical cache admission is a correctness boundary: the plugin must patch
# KVCacheManager even in this cache-off setup check, so a later cache-on launch
# cannot silently run stock arbitrary-block admission.
from laguna_vllm_ext.prefix_cache_quantum import (
    DEFAULT_PREFIX_QUANTUM,
    QUALIFIED_KV_BLOCK_SIZE,
    prefix_cache_quantum_patch_is_installed,
)
assert prefix_cache_quantum_patch_is_installed(), "canonical prefix-cache admission patch NOT active"
assert DEFAULT_PREFIX_QUANTUM == 8192
assert QUALIFIED_KV_BLOCK_SIZE == 64
print(
    "  canonical prefix-cache admission patch: active "
    f"(quantum={DEFAULT_PREFIX_QUANTUM}, block={QUALIFIED_KV_BLOCK_SIZE})"
)

import pytest
import pytest_timeout  # noqa: F401
print(
    "  pytest",
    pytest.__version__,
    "+ pytest-timeout",
    importlib.metadata.version("pytest-timeout"),
)
EOF

PYTHONPATH="$MODEL_DIR/vllm_ext:$REPO_ROOT" "$PY" -m pytest -q "$MODEL_DIR/vllm_ext/tests"

# Weights are gated; a missing snapshot only shows up ~10 min into a boot otherwise.
HF_HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
if [ ! -d "$HF_HUB/models--poolside--Laguna-XS-2.1" ]; then
  echo
  echo "NOTE: poolside/Laguna-XS-2.1 is not in $HF_HUB."
  echo "      It is a gated repo (~63 GB): run 'hf auth login', then serving will fetch it."
  echo "      ('huggingface-cli' is removed in huggingface_hub >= 1.0 — 'hf' replaces it.)"
fi

echo
echo "DONE. Env ready at $VLLM_ENV"
echo "Serve with:  $MODEL_DIR/serve_vllm.sh"
