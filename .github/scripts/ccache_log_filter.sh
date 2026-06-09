#!/usr/bin/env bash
# ccache_log_filter.sh — Analyze ccache Redis retrieval stats from a CI job log.
# Usage: ./ccache_log_filter.sh <logfile>
#    or: cat logfile | ./ccache_log_filter.sh

set -uo pipefail

INPUT="${1:-/dev/stdin}"

# Sanitize input: mask any Redis credentials in URLs (redis://user:pass@host or redis://:pass@host)
# Write to a temp file so all downstream greps can use it as a normal path.
_TMP=$(mktemp)
trap 'rm -f "$_TMP"' EXIT
sed -E 's|redis://[^@]*@|redis://********@|g' "$INPUT" > "$_TMP"
INPUT="$_TMP"

# --- Redis connection check ---
if grep -q "Redis connection OK" "$INPUT"; then
    REDIS_OK="yes"
    REDIS_COUNT=$(grep -c "Redis connection OK" "$INPUT")
else
    REDIS_OK="no"
    REDIS_COUNT=0
fi

echo "=== ccache Redis Analysis ==="
echo ""
echo "Redis connection OK: ${REDIS_OK} (${REDIS_COUNT} worker(s))"
echo ""

# --- Extract retrieval timings ---
# Lines look like: Retrieved <hash> from redis://...  (1.09 ms)
TIMINGS=$(grep -oP '(?<=Retrieved ).*?\([\d.]+ ms\)' "$INPUT" | grep -oP '[\d.]+(?= ms\))' || true)

if [[ -z "$TIMINGS" ]]; then
    echo "Redis retrievals: 0"
    echo "No retrieval timing data found."
    exit 0
fi

COUNT=$(echo "$TIMINGS" | wc -l)

# Use awk for min/max/mean/total — pure bash can't do floats
STATS=$(echo "$TIMINGS" | awk '
BEGIN { min = 1e9; max = 0; sum = 0; n = 0 }
{
    v = $1 + 0
    sum += v
    n++
    if (v < min) min = v
    if (v > max) max = v
}
END {
    if (n > 0) {
        printf "count=%d min=%.2f max=%.2f mean=%.2f total=%.2f\n", n, min, max, sum/n, sum
    } else {
        printf "count=0 min=0 max=0 mean=0 total=0\n"
    }
}')

# Parse stats
eval "$STATS"

echo "Redis retrievals: ${count}"
echo ""
echo "  Min:   ${min} ms"
echo "  Max:   ${max} ms"
echo "  Mean:  ${mean} ms"
echo "  Total: ${total} ms ($(awk "BEGIN {printf \"%.2f\", ${total}/1000}") s)"

# --- Histogram (optional quick distribution) ---
echo ""
echo "Distribution:"
echo "$TIMINGS" | awk '
{
    v = $1 + 0
    if      (v <  1)  b[1]++
    else if (v <  2)  b[2]++
    else if (v <  5)  b[3]++
    else if (v < 10)  b[4]++
    else               b[5]++
}
END {
    split(" < 1 ms| 1-2 ms| 2-5 ms|5-10 ms|>=10 ms", labels, "|")
    for (i = 1; i <= 5; i++)
        printf "  %-8s  %5d\n", labels[i], (i in b ? b[i] : 0)
}'

# --- Real errors summary ---
echo ""
echo "Errors / warnings:"

strip_ts() {
    sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z[[:space:]]*//' \
    | sed -E 's/^\[[-0-9T:. ]+ [0-9]+ \] //'
}

# Mismatch summary: group by type (size vs hash), show top affected files
MISMATCH_LINES=$(grep -i 'Mismatch for' "$INPUT" | strip_ts \
    | sed -E 's/: size [0-9]+ != [0-9]+/: size mismatch/' \
    | sed -E 's/: hash [0-9a-f]+ != [0-9a-f]+/: hash mismatch/' \
    || true)

if [[ -n "$MISMATCH_LINES" ]]; then
    SIZE_MM=$(echo "$MISMATCH_LINES" | grep -c ': size mismatch' || true)
    HASH_MM=$(echo "$MISMATCH_LINES" | grep -c ': hash mismatch' || true)
    UNIQ_FILES=$(echo "$MISMATCH_LINES" | grep -oP '(?<=Mismatch for )[^ :]+' | sort -u | wc -l || true)
    echo "  Cache mismatches: ${SIZE_MM}x size, ${HASH_MM}x hash  (${UNIQ_FILES} unique files)"
    echo "  Top files:"
    echo "$MISMATCH_LINES" | grep -oP '(?<=Mismatch for )[^ :]+' \
        | sort | uniq -c | sort -rn | head -5 \
        | awk '{ printf "    %4dx  %s\n", $1, $2 }'
fi

# Other errors: non-mismatch, non-filename-noise
echo ""
echo "  Other:"
{
  grep -i 'error\|failed\|timeout' "$INPUT" \
    | grep -v 'connect timeout [0-9]* ms' \
    | grep -v 'timeout set to [0-9]* ms' \
    | grep -v 'errors=0' \
    | grep -v 'Writing to\|Object file:\|Source file:\|Executing \|Manifest file:\|Included file:\|Mismatch for' \
    | grep -vE '\.(h|hpp|cpp|cc|c|o|d|a|so)\b|/system_error\b|/experimental/' \
    | strip_ts
} | sort | uniq -c | sort -rn \
  | awk 'NR<=5 { count=$1; $1=""; printf "    %4dx  %s\n", count, substr($0,2) }
         END   { if (NR==0) print "    None"
                 else if (NR>5) printf "    ... and %d more\n", NR-5 }'
