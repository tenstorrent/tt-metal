#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# erisc_pipeline_benchmark sweep: one two-rank mpirun per case, one 1x2 mesh per rank on a 4-chip box.
# Env: CHANS (rank 0's inter-mesh channels) SIZES MIB WARMUP BATCH HANG VERIFY R0 R1, RAND=1 SEED N, STOP_ON_FAIL, OUT, B.
# MD=1 prints a Markdown table. R0/R1 are the chips per rank: chips 0 and 1 are not cabled on bh-qbae-12.
# HANG is total seconds per case. 14 KiB packets need TT_BRIDGE_PIPE_MAX_PAYLOAD=15232 (the router slot).
#
# Examples (run from anywhere; build target erisc_pipeline_benchmark first):
#   VERIFY=0 MD=1 MIB=256 OUT=pipe_perf.md tests/tt_metal/distributed/run_erisc_e2h2h2e_sweep.sh
#   MD=1 RAND=1 SEED=42 N=40 OUT=pipe_random_42.md tests/tt_metal/distributed/run_erisc_e2h2h2e_sweep.sh
set -u
cd "$(dirname "$0")/../../.." || exit 2
B=${B:-./build_Release/test/tt_metal/distributed/erisc_pipeline_benchmark}
MGD=tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_qb_dual_1x2_intermesh.textproto
MIB=${MIB:-256}; WARMUP=${WARMUP:-10}; BATCH=${BATCH:-32}; HANG=${HANG:-120}; VERIFY=${VERIFY:-1}
R0=${R0:-0,2}; R1=${R1:-1,3}; CHANS=${CHANS:-"9 8"}; SIZES=${SIZES:-"1024 2048 4096"}
RAND=${RAND:-0}; SEED=${SEED:-1337}; N=${N:-20}
OUT=${OUT:-pipe_sweep_$(date +%Y%m%d_%H%M%S).out}
[ -x "$B" ] || { echo "no benchmark at $B (build target erisc_pipeline_benchmark)"; exit 2; }

# total_amt:16 is a registered case; TT_BRIDGE_PIPE_MIB supplies the real amount.
filt() { echo "PipeFixture/Bridge/total_amt:16/warmup_pct:$WARMUP/packet_size:$1/verify:$VERIFY/pace:0/sweep:0/batch:$2/"; }
rank() {  # $1 mesh id, $2 chips; every rank gets the forced channel, only mesh 0 chip 0 bridges it
    echo "-np 1 -x TT_MESH_ID=$1 -x TT_MESH_GRAPH_DESC_PATH=$MGD -x TT_VISIBLE_DEVICES=$2 -x TT_METAL_HOME=$PWD" \
        "-x TT_BRIDGE_E2H_FORCE_CHAN -x TT_BRIDGE_PIPE_MIB ${TT_BRIDGE_PIPE_MAX_PAYLOAD:+-x TT_BRIDGE_PIPE_MAX_PAYLOAD} $B"
}
KEYS='Failed to match.*|ERROR OCCURRED.*|TT_FATAL.*|(want|forwarded|injected|landed)=[0-9.]+[kMG]?|verify_fail=[0-9]+|bad_order=[0-9]+|throughput_MBps=[0-9.]+[kMG]?|(e2e_us|rtt_us)_(p50|p99)=[0-9.]+[kMG]?'
v() {  # a counter's value with google-benchmark's k/M/G suffix expanded, so printf can format it
    grep -oE "$1=[0-9.]+[kMG]?" <<<"$2" | head -1 | cut -d= -f2 |
        awk '{m = sub(/k$/, "") ? 1e3 : sub(/M$/, "") ? 1e6 : sub(/G$/, "") ? 1e9 : 1; printf "%.15g", $0 * m}'
}
md_header() {
    echo; echo "| # | eth chan | packet (B) | batch | MiB | throughput (MB/s) | e2e p50 (µs) | e2e p99 (µs) |" \
        "h2h rtt p50 (µs) | verify_fail | bad_order | frames |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
}
pass=0; fail=0; fails=()
LOG=$(mktemp); trap 'rm -f "$LOG"' EXIT

run() {  # $1 chan, $2 packet, $3 batch, $4 MiB, $5 case number
    local t0 line rc want p n=0 f
    t0=$(date +%s)
    [ -z "${MD:-}" ] && printf "[%s] chan=%-2s ps=%-4s batch=%-2s mib=%-5s " "$5" "$1" "$2" "$3" "$4"
    f="--benchmark_filter=$(filt "$2" "$3")"
    # shellcheck disable=SC2046  # rank() is meant to split into words
    TT_BRIDGE_E2H_FORCE_CHAN=$1 TT_BRIDGE_PIPE_MIB=$4 mpirun --oversubscribe --bind-to none \
        $(rank 0 "$R0") "$f" : $(rank 1 "$R1") "$f" >"$LOG" 2>&1 &
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
    [ "$rc" != 0 ] && cp "$LOG" "${OUT%.*}_fail$((fail + 1))_ch$1_ps$2.log"
    want=$(v want "$line")
    if [ -n "${MD:-}" ] && [ "$rc" = 0 ] && [ -n "$want" ]; then
        printf "| %s | %s | %s | %s | %s | %.0f | %.1f | %.1f | %.1f | %s | %s | %s |\n" "$5" "$1" "$2" "$3" "$4" \
            "$(v throughput_MBps "$line")" "$(v e2e_us_p50 "$line")" "$(v e2e_us_p99 "$line")" "$(v rtt_us_p50 "$line")" \
            "$(v verify_fail "$line")" "$(v bad_order "$line")" "$(v landed "$line")"
    elif [ -n "${MD:-}" ]; then
        echo "| $5 | $1 | $2 | $3 | $4 | FAILED (rc=$rc) | | | | | | |"
    else
        echo "$line rc=$rc $(($(date +%s) - t0))s"
    fi
    if [ "$rc" = 0 ] && [ -n "$want" ] && [ "$(v landed "$line")" = "$want" ] &&
       [ "$(v forwarded "$line")" = "$want" ] && grep -q 'verify_fail=0' <<<"$line" && grep -q 'bad_order=0' <<<"$line"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1)); fails+=("chan=$1 ps=$2 batch=$3 mib=$4 rc=$rc")
        [ -n "${STOP_ON_FAIL:-}" ] && { echo "# STOP_ON_FAIL: chips not reset"; exit 1; }
        pkill -9 -f "$B" 2>/dev/null; tt-smi -r >/dev/null 2>&1  # a stalled router would wedge the next case
    fi
}

{
    echo "# $(date)  host=$(hostname)  warmup=$WARMUP%  hang=${HANG}s  verify=$VERIFY  ranks=($R0) ($R1)"
    if [ "$RAND" = 1 ]; then
        RANDOM=$SEED; read -ra C <<<"$CHANS"; read -ra S <<<"$SIZES"; BS=(8 32); MS=(1 4 16 64 256)
        echo "# random: seed=$SEED cases=$N"
        [ -n "${MD:-}" ] && md_header
        for i in $(seq 1 "$N"); do
            run "${C[RANDOM % ${#C[@]}]}" "${S[RANDOM % ${#S[@]}]}" "${BS[RANDOM % 2]}" "${MS[RANDOM % 5]}" "$i"
        done
    else
        echo "# sweep: chans=($CHANS) sizes=($SIZES) batch=$BATCH mib=$MIB"
        [ -n "${MD:-}" ] && md_header
        i=0; for ch in $CHANS; do for ps in $SIZES; do i=$((i + 1)); run "$ch" "$ps" "$BATCH" "$MIB" "$i"; done; done
    fi
    echo "# PASS $pass  FAIL $fail"
    for f in ${fails[@]+"${fails[@]}"}; do echo "# failed: $f"; done
} 2>&1 | tee "$OUT"
