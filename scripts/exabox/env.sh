# Exabox environment for op generation (source me). Exabox nodes have no /localdev and cannot reach the
# central eval Postgres on bgdepyc01 (no DNS), so keep the eval DB and run output on shared /data.
export EVAL_FORCE_SQLITE=1
export EVAL_SQLITE_PATH="/data/$USER/evaldb/eval_runs.db"
export EVAL_RUNS_DIR="/data/$USER/evalruns"   # pass as: run_eval.py --base-dir "$EVAL_RUNS_DIR"
mkdir -p "$(dirname "$EVAL_SQLITE_PATH")" "$EVAL_RUNS_DIR" 2>/dev/null || true
