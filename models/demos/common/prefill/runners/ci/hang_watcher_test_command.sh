#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# DO NOT MERGE -- hang-injection experiment: what the prefill stage hang watcher runs on a verdict, while the
# hung rank is still alive. $1 is CI's TRIAGE_OUTPUT_DIR, uploaded after the job.
# 1. tt-triage into $1/<host>, with CI's flags.
# 2. A detached sampler of the rank's process state, to see whether the watcher's SIGKILL ends it.
# 3. A detached producer kill a minute later; run_multirank_pcc.sh then cleans up and prints the sampler's log.
D="${1:-/tmp/tt_triage}/$(hostname)"
mkdir -p "$D" || D=/tmp

# The rank is our nearest ancestor running prefill_runner, a few hops up: sh, the watcher, the rank.
rank=$PPID
for _ in 1 2 3 4 5 6 7 8; do
    tr '\0' ' ' <"/proc/$rank/cmdline" 2>/dev/null | grep -q 'prefill[.]runners[.]prefill_runner' && break
    rank=$(sed -E 's/^.*\) [A-Za-z] ([0-9]+) .*$/\1/' "/proc/$rank/stat" 2>/dev/null || echo 1)
done
echo "HANG WATCHER COMMAND: $(date -u +%T.%N) rank pid ${rank}, tt-triage into ${D}" >&2

timeout -k 30 1800 "${TRIAGE_PYTHON:-/opt/venv/bin/python3}" "${TT_METAL_HOME:-/home/user/tt-metal}/tools/tt-triage.py" \
    --disable-progress --path="$D/debug_bus_signal_groups.json" --triage-summary-path="$D/triage_summary.txt" \
    --llm-output-path="$D/triage_llm_output.txt" --sqlite-output-path="$D/triage_sqlite_output.sqlite" \
    >"$D/triage_output.txt" 2>&1
echo "HANG WATCHER COMMAND: $(date -u +%T.%N) tt-triage exited $?" >&2

# Logs the rank's state, wchan and thread count, and the watcher's pid, whenever one changes.
sample() {
    local pid=$1 end=$((SECONDS + ${SAMPLER_SECONDS:-55})) last="" now st
    while [ "$SECONDS" -lt "$end" ]; do
        if [ -r "/proc/$pid/stat" ]; then
            st=$(sed -E 's/^.*\) ([A-Za-z]) .*$/\1/' "/proc/$pid/stat")
            now="rank $pid state=$st wchan=$(cat "/proc/$pid/wchan" 2>/dev/null) threads=$(ls "/proc/$pid/task" 2>/dev/null | wc -l)"
        else
            now="rank $pid gone"
        fi
        now="$now, watcher=$(pgrep -f 'hang_watcher[.]py' | tr '\n' ' ')"
        if [ "$now" != "$last" ]; then
            echo "$(date -u +%T.%N) $now"
            last=$now
        fi
        case "$now" in *gone*"watcher=") break ;; esac
        sleep 0.1
    done
    echo "$(date -u +%T.%N) sampler done"
}
export -f sample
# Detached and off our stdio, so neither holds the rank's pipes or dies with its process group.
setsid bash -c 'sample "$1"' _ "$rank" </dev/null >>"$D/postkill.txt" 2>&1 &
# The [_] keeps this pkill's own command line from matching.
setsid bash -c "sleep ${PRODUCER_KILL_DELAY_S:-60}; pkill -f 'prefill[_]producer'" </dev/null >/dev/null 2>&1 &
