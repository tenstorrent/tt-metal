#!/usr/bin/env bash
# DiffusionGemma SWE-Bench Verified — frozen baseline launcher.
#
# Points mini-swe-agent at any OpenAI-compatible DiffusionGemma server (the A100 reference
# vLLM server, or the TT vLLM plugin server) so both sides run a byte-identical harness.
# See README.md for the frozen baseline contract and the measured context requirement.
#
# usage: run_swebench.sh <outdir> <workers>
# env:
#   DG_BASE_URL   OpenAI base url          (default http://localhost:8000/v1)
#   DG_MODEL      served model name        (default google/diffusiongemma-26B-A4B-it)
#   DG_VENV       mini-swe-agent venv      (default /home/ttuser/zni/venvs/mini)
#   DG_WORK       dir holding dg_mini_model.py + the subset regex
#   SUBSET        path to the subset regex file; empty runs all 500
#   STEP_LIMIT    agent step limit         (default 250, the mini-swe-agent published value)
#   MAX_TOKENS    per-turn output cap      (default 16384)
set -uo pipefail
OUT="${1:?outdir}"
WORKERS="${2:-10}"

DG_BASE_URL="${DG_BASE_URL:-http://localhost:8000/v1}"
DG_MODEL="${DG_MODEL:-google/diffusiongemma-26B-A4B-it}"
DG_VENV="${DG_VENV:-/home/ttuser/zni/venvs/mini}"
DG_WORK="${DG_WORK:-$(cd "$(dirname "$0")" && pwd)}"
SUBSET="${SUBSET:-$DG_WORK/swebench_verified_subset100.regex}"

export HF_HOME="${HF_HOME:-/home/ttuser/zni/benchmarks/hfcache}"
export OPENAI_API_BASE="$DG_BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export MSWEA_COST_TRACKING=ignore_errors
export MSWEA_CONFIRM_EXIT=false
export PYTHONPATH="$DG_WORK"

source "$DG_VENV/bin/activate"

ARGS=(--subset verified --split test --workers "$WORKERS"
      -m "openai/$DG_MODEL"
      --model-class dg_mini_model.DGTextbasedModel
      -c swebench_backticks.yaml
      -c model.model_kwargs.max_tokens="${MAX_TOKENS:-16384}"
      -c agent.step_limit="${STEP_LIMIT:-250}"
      -o "$OUT")
if [ -n "$SUBSET" ] && [ -f "$SUBSET" ]; then ARGS+=(--filter "$(cat "$SUBSET")"); fi

echo "=== $(date -Is) START model=$DG_MODEL base=$DG_BASE_URL step_limit=${STEP_LIMIT:-250} workers=$WORKERS out=$OUT ==="
mini-extra swebench "${ARGS[@]}"
echo "=== $(date -Is) agent phase exit=$? ==="
echo "Now grade with: eval_swebench.sh $OUT <run-id>"
