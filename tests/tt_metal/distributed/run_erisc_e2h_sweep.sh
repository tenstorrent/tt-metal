#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# erisc_e2h_benchmark sweep, one process per case (the bridged channel is fixed at fabric init).
# Env: CHANS SIZES MIB WARMUP BATCH HANG RING, RAND=1 SEED N for random cases, STOP_ON_FAIL, OUT, B.
# MD=1 prints a Markdown table (throughput, latency, verify).
#
# Examples (run from anywhere; build target erisc_e2h_benchmark first):
#   MD=1 MIB=256 BATCH=8 OUT=e2h_pr_table.md tests/tt_metal/distributed/run_erisc_e2h_sweep.sh
#   MD=1 RAND=1 SEED=42 N=40 OUT=e2h_pr_random_42.md tests/tt_metal/distributed/run_erisc_e2h_sweep.sh
set -u
cd "$(dirname "$0")/../../.." || exit 2
B=${B:-./build_Release/test/tt_metal/distributed/erisc_e2h_benchmark}
MIB=${MIB:-1024}; WARMUP=${WARMUP:-10}; BATCH=${BATCH:-32}; HANG=${HANG:-90}
CHANS=${CHANS:-"4 5 6 7 8 9 10 11"}; SIZES=${SIZES:-"1024 2048 4096"}
RAND=${RAND:-0}; SEED=${SEED:-1337}; N=${N:-20}
OUT=${OUT:-e2h_sweep_$(date +%Y%m%d_%H%M%S).out}
[ -x "$B" ] || { echo "no benchmark at $B (build target erisc_e2h_benchmark)"; exit 2; }
[ -n "${RING:-}" ] && export TT_BRIDGE_RING_PAGES=$RING

# total_amt:16 is a registered case; TT_BRIDGE_E2H_MIB supplies the real amount.
filt() { echo "E2HLegFixture/Bridge/total_amt:16/warmup_pct:$WARMUP/packet_size:$1/verify:1/pace:0/sweep:0/batch:$2/"; }
KEYS='Failed to match.*|ERROR OCCURRED.*|TT_FATAL.*|drained=[0-9.]+[kMG]?|want=[0-9.]+[kMG]?|verify_fail=[0-9]+|bad_order=[0-9]+|router_declined=[0-9.]+[kMG]?|throughput_GBps=[0-9.]+|latency_us_(p50|p99|max)=[0-9.]+[kMG]?'
v() {  # a counter's value with google-benchmark's k/M/G suffix expanded, so printf can format it
    grep -oE "$1=[0-9.]+[kMG]?" <<<"$2" | head -1 | cut -d= -f2 |
        awk '{m = sub(/k$/, "") ? 1e3 : sub(/M$/, "") ? 1e6 : sub(/G$/, "") ? 1e9 : 1; printf "%.15g", $0 * m}'
}
md_header() {
    echo; echo "| # | eth chan | packet (B) | batch | MiB | throughput (GB/s) | latency p50 (µs) | latency p99 (µs) |" \
        "latency max (µs) | verify_fail | bad_order | frames |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
}
pass=0; fail=0; fails=()
LOG=$(mktemp); trap 'rm -f "$LOG"' EXIT

run() {  # $1 chan, $2 packet, $3 batch, $4 MiB, $5 case number
    local t0 line rc dr want p n=0
    t0=$(date +%s)
    [ -z "${MD:-}" ] && printf "[%s] chan=%-2s ps=%-4s batch=%-2s mib=%-5s " "$5" "$1" "$2" "$3" "$4"
    TT_BRIDGE_E2H_MIB=$4 TT_BRIDGE_E2H_FORCE_CHAN=$1 "$B" --benchmark_filter="$(filt "$2" "$3")" >"$LOG" 2>&1 &
    p=$!
    while kill -0 "$p" 2>/dev/null && [ $n -lt "$HANG" ]; do sleep 1; n=$((n + 1)); done
    if kill -0 "$p" 2>/dev/null; then  # hung: keep where each thread was waiting, then kill
        for t in /proc/$p/task/*; do echo "$(cat "$t/comm") $(cat "$t/wchan")"; done | sort | uniq -c >>"$LOG"
        kill -9 "$p"; wait "$p" 2>/dev/null; rc=124
    else
        wait "$p"; rc=$?
    fi
    line=$(grep -oE "$KEYS" "$LOG" | tr '\n' ' ')
    [ "$rc" != 0 ] && cp "$LOG" "${OUT%.out}_fail$((fail + 1))_ch$1_ps$2.log"
    dr=$(grep -oE 'drained=[^ ]+' <<<"$line"); want=$(grep -oE 'want=[^ ]+' <<<"$line")
    if [ -n "${MD:-}" ] && [ "$rc" = 0 ] && [ -n "$dr" ]; then
        printf "| %s | %s | %s | %s | %s | %.2f | %.2f | %.2f | %.1f | %s | %s | %s |\n" "$5" "$1" "$2" "$3" "$4" \
            "$(v throughput_GBps "$line")" "$(v latency_us_p50 "$line")" "$(v latency_us_p99 "$line")" \
            "$(v latency_us_max "$line")" "$(v verify_fail "$line")" "$(v bad_order "$line")" "$(v drained "$line")"
    elif [ -n "${MD:-}" ]; then
        echo "| $5 | $1 | $2 | $3 | $4 | FAILED (rc=$rc) | | | | | | |"
    else
        echo "$line rc=$rc $(($(date +%s) - t0))s"
    fi
    if [ "$rc" = 0 ] && [ -n "$dr" ] && [ "${dr#*=}" = "${want#*=}" ] &&
       grep -q 'verify_fail=0' <<<"$line" && grep -q 'bad_order=0' <<<"$line"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1)); fails+=("chan=$1 ps=$2 batch=$3 mib=$4 rc=$rc")
        [ -n "${STOP_ON_FAIL:-}" ] && { echo "# STOP_ON_FAIL: chips not reset"; exit 1; }
        tt-smi -r >/dev/null 2>&1  # a stalled producer would wedge the next case
    fi
}

{
    echo "# $(date)  host=$(hostname)  warmup=$WARMUP%  hang=${HANG}s  ring=${RING:-default}"
    if [ "$RAND" = 1 ]; then
        RANDOM=$SEED; read -ra C <<<"$CHANS"; read -ra S <<<"$SIZES"; BS=(1 8 32); MS=(1 4 16 64 256)
        echo "# random: seed=$SEED cases=$N"
        [ -n "${MD:-}" ] && md_header
        for i in $(seq 1 "$N"); do
            run "${C[RANDOM % ${#C[@]}]}" "${S[RANDOM % ${#S[@]}]}" "${BS[RANDOM % 3]}" "${MS[RANDOM % 5]}" "$i"
        done
    else
        echo "# sweep: chans=($CHANS) sizes=($SIZES) batch=$BATCH mib=$MIB"
        [ -n "${MD:-}" ] && md_header
        i=0; for ch in $CHANS; do for ps in $SIZES; do i=$((i + 1)); run "$ch" "$ps" "$BATCH" "$MIB" "$i"; done; done
    fi
    echo "# PASS $pass  FAIL $fail"
    for f in ${fails[@]+"${fails[@]}"}; do echo "# failed: $f"; done
} 2>&1 | tee "$OUT"
