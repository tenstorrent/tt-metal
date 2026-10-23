#!/usr/bin/env bash
# ccache_log_filter.sh — Analyze ccache S3/crsh remote storage stats from a CI log.
# Usage: ./ccache_log_filter.sh <logfile>
#    or: cat logfile | ./ccache_log_filter.sh

set -uo pipefail

INPUT="${1:-/dev/stdin}"

_TMP=$(mktemp)
trap 'rm -f "$_TMP"' EXIT
cp "$INPUT" "$_TMP"
INPUT="$_TMP"

echo "=== ccache Remote Storage Analysis (S3/crsh) ==="
echo ""

# --- Remote hit/miss summary from ccache log ---
count_matches() {
    local out rc
    out=$(grep -c "$1" "$INPUT" 2>/dev/null)
    rc=$?
    if [ "$rc" -gt 1 ]; then
        echo 0
    else
        echo "$out"
    fi
}
REMOTE_HITS=$(count_matches "Remote storage hit")
REMOTE_MISSES=$(count_matches "Remote storage miss")
REMOTE_PUTS=$(count_matches "Sending result.*to remote")

echo "Remote cache hits:   ${REMOTE_HITS}"
echo "Remote cache misses: ${REMOTE_MISSES}"
echo "Remote puts:         ${REMOTE_PUTS}"
echo ""

# --- crsh daemon / socket events ---
CRSH_EVENTS=$(grep -iE "crsh|ccache-storage-s3|storage helper|IPC|\.sock" "$INPUT" | head -20 || true)
if [[ -n "$CRSH_EVENTS" ]]; then
    echo "crsh daemon events:"
    echo "$CRSH_EVENTS" | sed 's/^/  /'
    echo ""
fi

# --- Remote retrieval timings ---
TIMINGS=$(grep -oP '(?<=Retrieved ).*?\([\d.]+ ms\)' "$INPUT" | grep -oP '[\d.]+(?= ms\))' || true)

if [[ -n "$TIMINGS" ]]; then
    COUNT=$(echo "$TIMINGS" | wc -l)
    STATS=$(echo "$TIMINGS" | awk '
    BEGIN { min = 1e9; max = 0; sum = 0; n = 0 }
    {
        v = $1 + 0
        sum += v; n++
        if (v < min) min = v
        if (v > max) max = v
    }
    END { printf "count=%d min=%.2f max=%.2f mean=%.2f total=%.2f\n", n, min, max, sum/n, sum }')
    eval "$STATS"
    echo "Remote retrievals: ${count}"
    echo "  Min:  ${min} ms  Max: ${max} ms  Mean: ${mean} ms  Total: ${total} ms"
    echo ""
fi

# --- Errors and warnings ---
echo "Errors / warnings:"

strip_ts() {
    sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z[[:space:]]*//' \
    | sed -E 's/^\[[-0-9T:. ]+ [0-9]+ \] //'
}

{
  grep -iE 'error|failed|timeout|refused|unreachable|denied|forbidden|could not|unable to' "$INPUT" \
    | grep -v 'errors=0' \
    | grep -v 'Writing to\|Object file:\|Source file:\|Executing \|Manifest file:\|Included file:' \
    | grep -vE '\.(h|hpp|cpp|cc|c|o|d|a|so)\b|/system_error\b|/experimental/' \
    | strip_ts
} | sort | uniq -c | sort -rn \
  | awk 'NR<=10 { count=$1; $1=""; printf "  %4dx  %s\n", count, substr($0,2) }
         END    { if (NR==0) print "  None"
                  else if (NR>10) printf "  ... and %d more\n", NR-10 }' \
  || true
