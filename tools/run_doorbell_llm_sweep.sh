#!/bin/bash
# Sweep TT_DOORBELL_DELAY_NS across values and measure LLM decode tok/s/user.
#
# Requires: HF_MODEL set (or defaults to meta-llama/Llama-3.1-8B).
# Requires: tt-metal built with the doorbell delay + stats instrumentation.
#
# Usage:
#   ./tools/run_doorbell_llm_sweep.sh                    # default delays
#   ./tools/run_doorbell_llm_sweep.sh 0 400 1000 5000    # custom delays
#
# Output: one line per delay value with tok/s/user, TTFT, and doorbell counts.
# Also writes a CSV to /tmp/doorbell_llm_sweep_<timestamp>.csv.
# Full pytest output for each run is saved to /tmp/doorbell_run_<delay>.log.

set -uo pipefail

export HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.1-8B}"
PYTEST_FILTER="${PYTEST_FILTER:-performance and batch-1}"
DEMO_SCRIPT="models/tt_transformers/demo/simple_text_demo.py"

if [ ! -f "$DEMO_SCRIPT" ]; then
    echo "ERROR: $DEMO_SCRIPT not found. Run from tt-metal root." >&2
    exit 1
fi

# Verify pytest is available.
if ! command -v pytest &>/dev/null; then
    echo "ERROR: pytest not found. Activate your python environment first." >&2
    exit 1
fi

# Warm-up / sanity check: run baseline once and verify we can parse output.
echo "Sanity check: running baseline (delay=0) to verify setup..."
SANITY_LOG=$(mktemp /tmp/doorbell_sanity_XXXXXX.log)
if ! TT_DOORBELL_DELAY_NS=0 TT_DOORBELL_STATS=1 \
    pytest "$DEMO_SCRIPT" -k "$PYTEST_FILTER" -s 2>&1 | tee "$SANITY_LOG" | tail -5; then
    echo ""
    echo "ERROR: Baseline run failed. Full output in: $SANITY_LOG" >&2
    exit 1
fi

if ! grep -q "tok/s/user" "$SANITY_LOG"; then
    echo ""
    echo "ERROR: Could not find 'tok/s/user' in baseline output." >&2
    echo "       Check: $SANITY_LOG" >&2
    exit 1
fi
echo "Sanity check passed."
echo ""

if [ $# -gt 0 ]; then
    DELAYS=("$@")
else
    DELAYS=(0 100 400 800 1700 3000 5000 10000)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV="/tmp/doorbell_llm_sweep_${TIMESTAMP}.csv"
echo "delay_ns,tok_s_user,ttft_ms,fetch_q_writes,completion_q_writes,total_doorbells" > "$CSV"

parse_field() {
    local pattern="$1" file="$2"
    grep -oP "$pattern" "$file" | tail -1
}

printf "%-12s %-14s %-10s %-16s %-18s %-14s\n" \
    "delay_ns" "tok/s/user" "TTFT(ms)" "fetch_q_writes" "completion_q_writes" "total_doorbells"
printf "%-12s %-14s %-10s %-16s %-18s %-14s\n" \
    "--------" "----------" "--------" "--------------" "-------------------" "--------------"

# Use the sanity-check log as the delay=0 baseline if 0 is in the sweep.
for DELAY in "${DELAYS[@]}"; do
    LOGFILE="/tmp/doorbell_run_${DELAY}.log"

    if [ "$DELAY" = "0" ] && [ -f "$SANITY_LOG" ]; then
        cp "$SANITY_LOG" "$LOGFILE"
    else
        echo "--- Running with TT_DOORBELL_DELAY_NS=$DELAY (log: $LOGFILE) ---" >&2
        TT_DOORBELL_DELAY_NS="$DELAY" TT_DOORBELL_STATS=1 \
            pytest "$DEMO_SCRIPT" -k "$PYTEST_FILTER" -s \
            >"$LOGFILE" 2>&1 || true
    fi

    TOK_S_USER=$(parse_field '(?<=@ )\S+(?= tok/s/user)' "$LOGFILE" || true)
    TTFT_MS=$(parse_field '(?<=Average Time to First Token \(TTFT\): )\S+(?=ms)' "$LOGFILE" || true)
    FETCH_Q=$(parse_field '(?<=fetch_q_writes=)\d+' "$LOGFILE" || true)
    COMPL_Q=$(parse_field '(?<=completion_q_writes=)\d+' "$LOGFILE" || true)
    TOTAL_DB=$(parse_field '(?<=total=)\d+' "$LOGFILE" || true)

    TOK_S_USER="${TOK_S_USER:---}"
    TTFT_MS="${TTFT_MS:---}"
    FETCH_Q="${FETCH_Q:---}"
    COMPL_Q="${COMPL_Q:---}"
    TOTAL_DB="${TOTAL_DB:---}"

    if [ "$TOK_S_USER" = "--" ]; then
        echo "  WARNING: Could not parse tok/s/user for delay=$DELAY. Check $LOGFILE" >&2
    fi

    printf "%-12s %-14s %-10s %-16s %-18s %-14s\n" \
        "$DELAY" "$TOK_S_USER" "$TTFT_MS" "$FETCH_Q" "$COMPL_Q" "$TOTAL_DB"

    echo "$DELAY,$TOK_S_USER,$TTFT_MS,$FETCH_Q,$COMPL_Q,$TOTAL_DB" >> "$CSV"
done

echo ""
echo "CSV: $CSV"
echo "Logs: /tmp/doorbell_run_*.log"
echo ""
echo "To view: column -t -s, $CSV"
