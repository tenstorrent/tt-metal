# Shared configuration for the Mistral Small 4 PP=4 vs single-rank prefill perf harness.
#
# EVERY path below is overridable, because none of them are in the repo: the checkpoint, the TTNN
# weight caches and the golden traces are multi-GB artifacts that live wherever you put them. Export
# any of these before sourcing (or edit this file) to point the harness at your own copies.
#
# The defaults are the author's layout on the shared /data mount; they work as-is on the tt bh-glx
# boxes and will need overriding anywhere else.

# --- repo + build (usually correct as-is) -------------------------------------------------------
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)}"
export PYTHONPATH="$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/lib:${LD_LIBRARY_PATH:-}"
export PY="${PY:-$TT_METAL_HOME/python_env/bin/python}"

# --- model checkpoint ---------------------------------------------------------------------------
export MISTRAL4_HF_MODEL="${MISTRAL4_HF_MODEL:-/data/kmabee/models/Mistral-Small-4-119B-2603}"
export PREFILL_HF_MODEL="${PREFILL_HF_MODEL:-$MISTRAL4_HF_MODEL}"

# --- TTNN weight caches -------------------------------------------------------------------------
# Resolved by the runner as {name}_{arch}_{ttnn.get_num_devices()}dev/{sp}x{tp}. A single-rank run
# sees 32 devices -> 32dev/8x4; each PP rank sees 8 -> 8dev/8x1.
#
# NOTE the device-count component is NAMESPACING ONLY: the tensor content depends solely on the mesh
# shape, so 32dev/8x1 and 8dev/8x1 files are byte-identical. If you already have a 32dev/8x1 cache,
# hardlink it into an 8dev/8x1 tree rather than rebuilding 65 GB (`cp -al`). Cache keys are GLOBAL
# layer indices, so all four PP ranks share one directory and it must hold layers 0..35.
export M4_CACHE_8x4="${M4_CACHE_8x4:-/data/kmabee/mistral4_caches/ttnn_cache_8x4}"
export M4_CACHE_8x1="${M4_CACHE_8x1:-/data/kmabee/mistral4_caches/ttnn_cache_pp8dev_s4}"
export TT_MISTRAL4_PREFILL_HOST_REF_CACHE="${TT_MISTRAL4_PREFILL_HOST_REF_CACHE:-/data/kmabee/mistral4_caches/ref_cache}"

# --- golden traces (only needed for the single-rank KV-PCC correctness run) ----------------------
export GOLDEN_5120="${GOLDEN_5120:-/data/kmabee/mistral4_golden_traces/_timing_5120_36L}"
export GOLDEN_10240="${GOLDEN_10240:-/data/kmabee/mistral4_golden_traces/_timing_10240_36L}"
export GOLDEN_56320="${GOLDEN_56320:-/data/kmabee/mistral4_golden_traces/mistral4_56320_36L}"

# --- outputs (created under the repo root by default; both are large) ----------------------------
export M4_PERF_OUT="${M4_PERF_OUT:-$TT_METAL_HOME/mistral4_perf_$(hostname)}"
export M4_PROFILE_OUT="${M4_PROFILE_OUT:-$TT_METAL_HOME/mistral4_perf_profile}"
