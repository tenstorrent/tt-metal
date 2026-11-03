# source me: embeddinggemma-2 comparison env (Gio's tt-metal a12d330bb3c + TTIS e1fcaa29), chip 86:00.0 (x8), 160 W.
# Overrides (defaults = the u20-42 study): EMB_ROOT (work dir: caches, hf, logs, results), EMB_TT_METAL (tt-metal
# checkout with python_env), EMB_TTIS (tt-inference-server checkout), EMB_CHIP (TT_VISIBLE_DEVICES), EMB_DEV (device
# node the chain scripts check is free), DEVICE_IDS (media servers), UV_PYTHON_LIB (cpp_server, LESSONS: uv libpython).
E=${EMB_ROOT:-/mnt/tt-data/nkira/muse-glimmer-30b/emb}; ROOT=${EMB_TT_METAL:-$E/tt-metal}; TTIS=${EMB_TTIS:-$E/tt-inference-server}
EMB_SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); EMB_DEV=${EMB_DEV:-/dev/tenstorrent/31}; mkdir -p $E/logs $E/results
export PATH="$ROOT/python_env/bin:$PATH" VIRTUAL_ENV="$ROOT/python_env" TT_METAL_HOME="$ROOT" PYTHONPATH="$ROOT:$EMB_SCRIPTS"
export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
export TT_VISIBLE_DEVICES=${EMB_CHIP:-0000:86:00.0} TT_METAL_TDP_LIMIT_WATTS=${TT_METAL_TDP_LIMIT_WATTS:-160}
export TT_METAL_CACHE=$E/tt-metal-cache HF_HOME=$E/hf HF_HUB_OFFLINE=1 EMBEDDING_GEMMA2_CACHE=$E/weight-cache
export EMBEDDING_GEMMA2_REVISION=914f7f89142e33e77833254d9c9b90c3cef7303b TOKENIZERS_PARALLELISM=false
export TT_MM_THROTTLE_PERF=${TT_MM_THROTTLE_PERF:-0}  # parity across paths (bank P2)
