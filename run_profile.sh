#!/usr/bin/env bash
set -uo pipefail

# ─────────────────────────────────────────────────────────────────────
#  Galaxy Profiling — Llama 8B · Qwen 32B · Llama 70B
# ─────────────────────────────────────────────────────────────────────
#
#  Usage:
#    bash run_profile.sh <model> <mode>
#    bash run_profile.sh <model> <mode> --plot-only   # skip profiling, just re-plot
#
#  Models:  llama-8b   meta-llama/Llama-3.1-8B-Instruct  (DP=4, TP=8,  1 layer)
#           qwen-32b   Qwen/Qwen3-32B                    (DP=4, TP=8,  1 layer)
#           llama-70b  Llama3.3-70B-Instruct              (TP=32,       full model)
#
#  Modes:   coarse     Module-level timing (attention, mlp, …)
#           fine       Sub-operation timing (qkv, sdpa, ff1, …)
#
#  Output:  profiling/<model>/<mode>/            JSON data + logs
#           profiling/<model>/<model>_<mode>.png  plot image
#
#  Examples:
#    bash run_profile.sh llama-8b coarse
#    bash run_profile.sh qwen-32b fine
#    bash run_profile.sh llama-70b coarse --plot-only
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse arguments ──────────────────────────────────────────────────
PLOT_ONLY=false
if [[ "${3:-}" == "--plot-only" ]]; then PLOT_ONLY=true; fi

if [[ $# -lt 2 ]]; then
    echo "Usage: bash run_profile.sh <model> <mode> [--plot-only]"
    echo ""
    echo "  Models:  llama-8b | qwen-32b | llama-70b"
    echo "  Modes:   coarse   | fine"
    exit 1
fi

MODEL="$1"; MODE="$2"

# ── Model table ──────────────────────────────────────────────────────
#  Each model defines:  HF / env vars, demo script, test-ID pattern,
#  pytest extra args, ISL list, and max_seq_len overrides.
declare -A MODEL_CFG
case "$MODEL" in
    llama-8b)
        MODEL_CFG=(
            [label]="Llama 3.1-8B"  [short]="llama8b"
            [demo]="models/tt_transformers/demo/simple_text_demo.py"
            [filter]="performance-long-context-{ISL}"
            [args]="--data_parallel=4 --max_generated_tokens=2 --num_layers=1"
            [isls]="16k 32k"
            [max_seq_16k]=32768  [max_seq_32k]=65536
        )
        export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=TG
        ;;
    qwen-32b)
        MODEL_CFG=(
            [label]="Qwen3-32B"  [short]="qwen32b"
            [demo]="models/tt_transformers/demo/simple_text_demo.py"
            [filter]="performance-long-context-{ISL}"
            [args]="--data_parallel=4 --max_generated_tokens=2 --num_layers=1"
            [isls]="16k 32k"
            [max_seq_16k]=32768  [max_seq_32k]=32768
        )
        export HF_MODEL=Qwen/Qwen3-32B MESH_DEVICE=TG
        ;;
    llama-70b)
        MODEL_CFG=(
            [label]="Llama 3.3-70B"  [short]="llama70b"
            [demo]="models/demos/llama3_70b_galaxy/demo/text_demo.py"
            [filter]="performance-long-{ISL}-b1"
            [args]="--max_generated_tokens=2"
            [isls]="4k 8k 16k 32k 64k"
        )
        export TT_CACHE_PATH="$SCRIPT_DIR/model_cache/Llama3.3-70B-Instruct"
        export LLAMA_DIR=/home/ubuntu/Llama3.3-70B-Instruct
        ;;
    *)  echo "Unknown model: $MODEL"; exit 1 ;;
esac

case "$MODE" in
    coarse) export PROFILING_FINE=0 ;;
    fine)   export PROFILING_FINE=1 ;;
    *)      echo "Unknown mode: $MODE"; exit 1 ;;
esac

# ── Paths ────────────────────────────────────────────────────────────
SHORT="${MODEL_CFG[short]}"
LABEL="${MODEL_CFG[label]}"
OUTDIR="profiling/${SHORT}/${MODE}"
PLOT="profiling/${SHORT}/${SHORT}_${MODE}.png"
mkdir -p "$OUTDIR"

# ── Run profiling ────────────────────────────────────────────────────
if ! $PLOT_ONLY; then
    read -ra ISLS <<< "${MODEL_CFG[isls]}"
    for ISL in "${ISLS[@]}"; do
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  ${LABEL}  ISL=${ISL}  ${MODE}  $(date +%H:%M:%S)"
        echo "═══════════════════════════════════════════════════════════"

        export PROFILING_OUTPUT="${OUTDIR}/${ISL}.json"

        # Build test filter from pattern
        FILTER="${MODEL_CFG[filter]//\{ISL\}/$ISL}"

        # Build pytest command
        CMD="pytest ${MODEL_CFG[demo]} -k \"${FILTER}\" ${MODEL_CFG[args]}"

        # Apply max_seq_len cap if defined for this model:isl
        MAX_KEY="max_seq_${ISL}"
        if [[ -n "${MODEL_CFG[$MAX_KEY]:-}" ]]; then
            CMD="${CMD} --max_seq_len=${MODEL_CFG[$MAX_KEY]}"
        fi

        CMD="${CMD} --timeout=900 -s"
        echo "  $CMD"
        eval "$CMD" 2>&1 | tee "${OUTDIR}/log_${ISL}.txt" || true
    done
fi

# ── Plot ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Plotting ${LABEL} ${MODE}"
echo "═══════════════════════════════════════════════════════════"
python3 plot_profiling.py "$MODE" "$OUTDIR" --title "$LABEL" -o "$PLOT"

echo ""
echo "  JSON  →  ${OUTDIR}/"
echo "  Plot  →  ${PLOT}"
