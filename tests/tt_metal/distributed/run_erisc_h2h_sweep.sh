#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# erisc_h2h_benchmark sweep: one 2-rank mpirun per case, single host, no device.
# Env: ARENAS SIZES (registered: 1024 4096 16384) MIB WARMUP HANG RING VERIFY, RAND=1 SEED N, STOP_ON_FAIL, OUT, B, MPIRUN.
# MPIRUN defaults to the ULFM wrapper (the binary links ULFM) with a 32 KiB eager limit to match TT_H2H_MAX_PUT.
# MD=1 prints a Markdown table (throughput, round trip, verify).
#
# Examples (run from anywhere; build target erisc_h2h_benchmark first):
#   MD=1 MIB=256 OUT=h2h_pr_table.md tests/tt_metal/distributed/run_erisc_h2h_sweep.sh
#   MD=1 RAND=1 SEED=42 N=40 OUT=h2h_pr_random_42.md tests/tt_metal/distributed/run_erisc_h2h_sweep.sh
set -u
cd "$(dirname "$0")/../../.." || exit 2
B=${B:-./build_Release/test/tt_metal/distributed/erisc_h2h_benchmark}
MPIRUN=${MPIRUN:-tests/tt_metal/multihost/mpirun_wrapper.sh --oversubscribe -np 2 --mca btl_sm_eager_limit 32768}
MIB=${MIB:-1024}; WARMUP=${WARMUP:-10}; HANG=${HANG:-90}
ARENAS=${ARENAS:-"1 2 4"}; SIZES=${SIZES:-"1024 4096 16384"}
RAND=${RAND:-0}; SEED=${SEED:-1337}; N=${N:-20}
OUT=${OUT:-h2h_sweep_$(date +%Y%m%d_%H%M%S).out}
[ -x "$B" ] || { echo "no benchmark at $B (build target erisc_h2h_benchmark)"; exit 2; }
[ -n "${RING:-}" ] && export TT_BRIDGE_H2H_RING=$RING

# total_amt:4 is a registered case; TT_BRIDGE_H2H_MIB supplies the real amount.
filt() { echo "H2HFixture/Bridge/total_amt:4/warmup_pct:$WARMUP/packet_size:$1/verify:${VERIFY:-1}/pace:0/sweep:0/arenas:$2/ring:32/"; }
KEYS='Failed to match.*|ERROR OCCURRED.*|moved=[0-9.]+[kMG]?|want=[0-9.]+[kMG]?|verify_fail=[0-9]+|bad_order=[0-9]+|credit_stalls=[0-9.]+[kMG]?|posts_per_flush=[0-9.]+|throughput_GBps=[0-9.]+|rtt_us_(p50|p99|max)=[0-9.]+[kMG]?'
v() {  # a counter's value with google-benchmark's k/M/G suffix expanded, so printf can format it
    grep -oE "$1=[0-9.]+[kMG]?" <<<"$2" | head -1 | cut -d= -f2 |
        awk '{m = sub(/k$/, "") ? 1e3 : sub(/M$/, "") ? 1e6 : sub(/G$/, "") ? 1e9 : 1; printf "%.15g", $0 * m}'
}
md_header() {
    echo; echo "| # | arenas | packet (B) | MiB | throughput (GB/s) | rtt p50 (µs) | rtt p99 (µs) | rtt max (µs) |" \
        "posts/flush | verify_fail | bad_order | frames |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
}
pass=0; fail=0; fails=()
LOG=$(mktemp); trap 'rm -f "$LOG"' EXIT

run() {  # $1 arenas, $2 packet, $3 MiB, $4 case number
    local t0 line rc got want p r n=0
    t0=$(date +%s)
    [ -z "${MD:-}" ] && printf "[%s] arenas=%-2s ps=%-5s mib=%-5s " "$4" "$1" "$2" "$3"
    TT_BRIDGE_H2H_MIB=$3 $MPIRUN -x TT_BRIDGE_H2H_MIB ${RING:+-x TT_BRIDGE_H2H_RING} "$B" --benchmark_filter="$(filt "$2" "$1")" >"$LOG" 2>&1 &
    p=$!
    while kill -0 "$p" 2>/dev/null && [ $n -lt "$HANG" ]; do sleep 1; n=$((n + 1)); done
    if kill -0 "$p" 2>/dev/null; then  # hung: keep where each rank's threads were waiting, then kill
        for r in $(pgrep -f "$B"); do
            for t in /proc/$r/task/*; do echo "pid $r $(cat "$t/comm") $(cat "$t/wchan")"; done
        done | sort | uniq -c >>"$LOG"
        pkill -9 -f "$B"; kill -9 "$p"; wait "$p" 2>/dev/null; rc=124
    else
        wait "$p"; rc=$?
    fi
    line=$(grep -oE "$KEYS" "$LOG" | tr '\n' ' ')
    grep -q 'ERROR OCCURRED' <<<"$line" && [ "$rc" = 0 ] && rc=3
    [ "$rc" != 0 ] && cp "$LOG" "${OUT%.*}_fail$((fail + 1))_a$1_ps$2.log"
    got=$(grep -oE 'moved=[^ ]+' <<<"$line" | head -1); want=$(grep -oE 'want=[^ ]+' <<<"$line" | head -1)
    if [ -n "${MD:-}" ] && [ "$rc" = 0 ] && [ -n "$got" ]; then
        printf "| %s | %s | %s | %s | %.3f | %.2f | %.2f | %.1f | %.2f | %s | %s | %s |\n" "$4" "$1" "$2" "$3" \
            "$(v throughput_GBps "$line")" "$(v rtt_us_p50 "$line")" "$(v rtt_us_p99 "$line")" \
            "$(v rtt_us_max "$line")" "$(v posts_per_flush "$line")" "$(v verify_fail "$line")" \
            "$(v bad_order "$line")" "$(v moved "$line")"
    elif [ -n "${MD:-}" ]; then
        echo "| $4 | $1 | $2 | $3 | FAILED (rc=$rc) | | | | | | | |"
    else
        echo "$line rc=$rc $(($(date +%s) - t0))s"
    fi
    if [ "$rc" = 0 ] && [ -n "$got" ] && [ "${got#*=}" = "${want#*=}" ] &&
       grep -q 'verify_fail=0' <<<"$line" && grep -q 'bad_order=0' <<<"$line"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1)); fails+=("arenas=$1 ps=$2 mib=$3 rc=$rc")
        [ -n "${STOP_ON_FAIL:-}" ] && { echo "# STOP_ON_FAIL"; exit 1; }
    fi
}

{
    echo "# $(date)  host=$(hostname)  warmup=$WARMUP%  hang=${HANG}s  ring=${RING:-32}  mpirun=$MPIRUN"
    if [ "$RAND" = 1 ]; then
        RANDOM=$SEED; read -ra A <<<"$ARENAS"; read -ra S <<<"$SIZES"; MS=(1 4 16 64 256)
        echo "# random: seed=$SEED cases=$N"
        [ -n "${MD:-}" ] && md_header
        for i in $(seq 1 "$N"); do
            run "${A[RANDOM % ${#A[@]}]}" "${S[RANDOM % ${#S[@]}]}" "${MS[RANDOM % 5]}" "$i"
        done
    else
        echo "# sweep: arenas=($ARENAS) sizes=($SIZES) mib=$MIB"
        [ -n "${MD:-}" ] && md_header
        i=0; for a in $ARENAS; do for ps in $SIZES; do i=$((i + 1)); run "$a" "$ps" "$MIB" "$i"; done; done
    fi
    echo "# PASS $pass  FAIL $fail"
    for f in ${fails[@]+"${fails[@]}"}; do echo "# failed: $f"; done
} 2>&1 | tee "$OUT"
