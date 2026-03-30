#!/usr/bin/env bash
set -euo pipefail

EXPECTED_GPUS=8
TP_SIZE=8
RUN_SMOKE=1
PYTHON_BIN=""
HF_HOME_PATH="${HF_HOME:-}"
SMOKE_MAX_NEW_TOKENS=8
SMOKE_PROMPT="Speculative decoding is"

usage() {
    cat <<EOF
GPU preflight checks for EAGLE3 DeepSeek runs.

Usage:
  $(basename "$0") [options]

Options:
  --expected-gpus N        Expected visible GPU count (default: 8)
  --tp-size N              TP size for smoke run (default: 8)
  --python PATH            Python interpreter path (default: auto-detect)
  --hf-home PATH           HF cache root (default: env HF_HOME or /proj_sw/user_dev/\$USER/hf_cache)
  --smoke-max-new-tokens N Smoke run generation length (default: 8)
  --skip-smoke             Run checks only, do not launch smoke command
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expected-gpus)
            EXPECTED_GPUS="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --hf-home)
            HF_HOME_PATH="$2"
            shift 2
            ;;
        --smoke-max-new-tokens)
            SMOKE_MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --skip-smoke)
            RUN_SMOKE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$PYTHON_BIN" ]]; then
    default_py="/proj_sw/user_dev/${USER}/tt-metal/python_env/bin/python"
    if [[ -x "$default_py" ]]; then
        PYTHON_BIN="$default_py"
    else
        PYTHON_BIN="$(command -v python3 || true)"
    fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: Could not resolve Python interpreter." >&2
    exit 1
fi

if [[ -z "$HF_HOME_PATH" ]]; then
    HF_HOME_PATH="/proj_sw/user_dev/${USER}/hf_cache"
fi

echo "== GPU Preflight =="
echo "python: $PYTHON_BIN"
echo "expected_gpus: $EXPECTED_GPUS"
echo "tp_size: $TP_SIZE"
echo "hf_home: $HF_HOME_PATH"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. GPU drivers/toolkit may be missing." >&2
    exit 1
fi

visible_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
echo "visible_gpus: $visible_gpus"
if [[ "$visible_gpus" -lt "$EXPECTED_GPUS" ]]; then
    echo "ERROR: Expected at least $EXPECTED_GPUS GPUs, found $visible_gpus." >&2
    exit 1
fi

if [[ "$TP_SIZE" -gt "$visible_gpus" ]]; then
    echo "ERROR: tp-size ($TP_SIZE) cannot exceed visible GPUs ($visible_gpus)." >&2
    exit 1
fi

mkdir -p "$HF_HOME_PATH"
df -h "$HF_HOME_PATH" || true

if ! command -v torchrun >/dev/null 2>&1; then
    echo "ERROR: torchrun not found in PATH." >&2
    exit 1
fi

echo "== Python dependency checks =="
"$PYTHON_BIN" - <<'PY'
import importlib
mods = ["torch", "transformers", "deepspeed"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing modules: {missing}")
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
print("python deps: OK")
print(f"torch version: {torch.__version__}")
print(f"cuda devices: {torch.cuda.device_count()}")
PY

if [[ "$RUN_SMOKE" -eq 1 ]]; then
    echo "== Launching TP smoke run =="
    export HF_HOME="$HF_HOME_PATH"
    unset TRANSFORMERS_CACHE
    torchrun --nproc_per_node="$TP_SIZE" \
      models/demos/speculative_deepseek_r1_broad/scripts/run_baseline_deepseek_cpu.py \
      --prompt "$SMOKE_PROMPT" \
      --base-model-preset tiny_gpt2 \
      --base-impl reference \
      --tp-size "$TP_SIZE" \
      --device cuda \
      --dtype bfloat16 \
      --max-new-tokens "$SMOKE_MAX_NEW_TOKENS"
fi

echo "Preflight OK."
