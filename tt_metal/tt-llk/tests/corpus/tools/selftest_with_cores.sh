#!/usr/bin/env bash
# selftest for with-cores: a throwaway 2-slot pool proves (1) exclusion —
# two 2-slot jobs serialize instead of overlapping; (2) crash-release — a
# SIGKILLed holder's slots free automatically (kernel-owned flocks);
# (3) exit-status propagation; (4) loud clamp of oversized requests;
# (5) status reports holders.  Exits 0 on all-green, 1 with the first
# failing assertion otherwise.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WC="$HERE/with-cores"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/with-cores-selftest.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
export COREBUDGET_DIR="$TMP/pool" COREBUDGET_SLOTS=2 COREBUDGET_POLL_SECS=1

fail() { echo "SELFTEST FAIL: $*" >&2; exit 1; }

# (1) exclusion: job A holds both slots for 3s; job B (also 2 slots) must
# not START inside A's hold window.  Timestamps prove the serialization.
"$WC" 2 -- bash -c "date +%s.%N > '$TMP/a-start'; sleep 3; date +%s.%N > '$TMP/a-end'" &
A=$!
sleep 0.7   # let A acquire first
"$WC" 2 -- bash -c "date +%s.%N > '$TMP/b-start'" &
B=$!
wait "$A" "$B"
[ -s "$TMP/a-end" ] && [ -s "$TMP/b-start" ] || fail "exclusion: jobs left no timestamps"
awk -v ae="$(cat "$TMP/a-end")" -v bs="$(cat "$TMP/b-start")" \
    'BEGIN { exit !(bs >= ae - 0.05) }' \
    || fail "exclusion: B started at $(cat "$TMP/b-start") before A released at $(cat "$TMP/a-end")"
echo "ok 1 - two full-pool jobs serialized"

# (2) crash-release: SIGKILL a holder mid-run; a follow-up job must acquire.
"$WC" 2 -- sleep 300 &
H=$!
sleep 1
# Kill the whole tree (wrapper + sleep): the slot fds all close on death.
PIDS=$(pgrep -P "$H"; echo "$H")
kill -9 $PIDS 2>/dev/null
wait "$H" 2>/dev/null
timeout 15 "$WC" 2 -- true || fail "crash-release: slots still held after SIGKILL of holder"
echo "ok 2 - SIGKILLed holder released its slots (kernel-owned flock)"

# (3) exit status propagates unchanged.
"$WC" 1 -- bash -c 'exit 42'
[ $? -eq 42 ] || fail "exit status: expected 42"
echo "ok 3 - exit status propagated"

# (4) oversized request clamps loudly and still runs.
OUT=$("$WC" 99 -- bash -c 'echo ran-with=$WITH_CORES' 2>&1) || fail "clamp: run failed"
echo "$OUT" | grep -q 'clamping request 99 -> pool size 2' || fail "clamp: no clamp notice in: $OUT"
echo "$OUT" | grep -q 'ran-with=2' || fail "clamp: WITH_CORES not clamped in: $OUT"
echo "ok 4 - oversized request clamped to pool size"

# (5) status names the holder while a job runs.
"$WC" 1 -- sleep 5 &
S=$!
sleep 1
STATUS=$("$WC" status)
echo "$STATUS" | grep -q 'HELD by pid' || fail "status: no holder shown while a job runs: $STATUS"
echo "$STATUS" | grep -q '1/2 held' || fail "status: expected 1/2 held: $STATUS"
kill "$S" 2>/dev/null; wait "$S" 2>/dev/null
echo "ok 5 - status reports holders"

echo "with-cores selftest: ALL GREEN"
