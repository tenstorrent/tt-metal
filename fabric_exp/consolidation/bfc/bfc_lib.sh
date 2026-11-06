#!/bin/bash
# The bubble-flow-control injection rule's device runs (RING_JAM_FIX.md §10): a copy of the mainline worktree's
# .scratch/val/val_lib.sh (2026-11-04) with its results, yamls and capture tools here, so nothing is written into
# /home/aagarwal/tt-metal-mainline. A run's tree is $W at the time of the call (bfc_p150_runs.sh sets it per run; the
# default is the branch worktree). Results and logs: bfc/results/LABEL.log; a hang leaves a read-only device capture in
# bfc/results/LABEL_hangcap/.
#
# run_cap LABEL YAML|gtest:FILTER CACHE TIMEOUT-S [ENV=VALUE ...]
#   One test_tt_fabric run of YAML (or fabric_unit_tests --gtest_filter=FILTER; or vc1:YAML, two ranks on the P150x8
#   split into two 2x2 meshes, through tt-run), in its own session, polled. Past
#   TIMEOUT-S it is a hang: its processes are killed (TERM, then KILL), the device is captured (hang_capture.py with the
#   study tree's Python, which has tt-exalens) and decoded, and run_cap returns 2. Otherwise 0 when the run reports no
#   failure, 1 when it does. Settings: fast dispatch, TT_METAL_CLEAR_L1=1, profiler off, compile commands logged,
#   TT_FABRIC_NO_BW_TELEMETRY=1; extra ENV settings go after TIMEOUT-S.
# budget_ok TIMEOUT-S: whether a run of that timeout still fits in the job budget (JOB_BUDGET, default 1450 s).
set -u
# VAL_TREE: run another worktree's build and sources (e.g. the pushed tip's) with these scripts and harness files.
W=${VAL_TREE:-/home/aagarwal/tt-metal-mainline-bfc}
VAL=/home/aagarwal/tt-metal/fabric_exp/consolidation/bfc
HARNESS=/home/aagarwal/tt-metal-mainline/.scratch
out=$VAL/results
PY=/home/aagarwal/tt-metal/python_env/bin/python
OBJCOPY=$W/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy
JOB_BUDGET=${JOB_BUDGET:-1450}
CAPTURE_TIMEOUT=${CAPTURE_TIMEOUT:-260}
cd $W || exit 1
mkdir -p $out
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
RUN_ROOT=""
RUN_SID=""
RUN_PIDS=""

_tree() {  # sid root: PIDs in session SID or descending from ROOT (this shell excluded)
    local own_sid; own_sid=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')
    [ "$1" = "$own_sid" ] && set -- "" "$2"
    ps -e -o pid=,ppid=,sid= | awk -v sid="$1" -v root="$2" -v self="$$" '
        { n++; p[n] = $1; pp[$1] = $2; s[$1] = $3 }
        END { for (i = 1; i <= n; i++) { q = p[i]; keep = (sid != "" && s[q] == sid); k = 0
                  while (!keep && q > 1 && k < 64) { if (q == root) keep = 1; q = pp[q]; k++ }
                  if (keep && p[i] != self) print p[i] } }'
}
_alive() {
    local p
    for p in $(_tree "$1" "$2"); do
        [ "$(ps -o stat= -p "$p" 2>/dev/null | cut -c1)" = Z ] || [ -z "$(ps -o pid= -p "$p" 2>/dev/null)" ] || echo "$p"
    done
}
_kill() {  # sid root [term-wait-s]
    local sid=$1 root=$2 tw=${3:-15} pids t
    pids=$(_alive "$sid" "$root")
    RUN_PIDS="$RUN_PIDS $pids"
    if [ -n "$pids" ]; then
        echo "   kill: $(echo $pids | wc -w) processes"
        kill -TERM $pids 2>/dev/null
        for t in $(seq 1 "$tw"); do pids=$(_alive "$sid" "$root"); [ -z "$pids" ] && break; sleep 1; done
        if [ -n "$pids" ]; then
            kill -KILL $pids 2>/dev/null
            for t in $(seq 1 30); do pids=$(_alive "$sid" "$root"); [ -z "$pids" ] && break; sleep 1; done
        fi
        [ -n "$pids" ] && echo "   !!! still alive after SIGKILL: $pids"
    fi
    wait "$root" 2>/dev/null
}
_cleanup() {
    kill $HB 2>/dev/null
    if [ -n "$RUN_ROOT" ]; then
        echo "!!! job exiting with a run in progress: killing it"
        _kill "$RUN_SID" "$RUN_ROOT" 5
    fi
}
trap _cleanup EXIT

_capture() {  # capture-dir cache run-log
    local cap=$1 c=$2 runlog=$3 t0=$SECONDS d h k t p left
    for t in $(seq 1 60); do
        left=""
        for p in $RUN_PIDS; do
            [ -n "$(ps -o pid= -p "$p" 2>/dev/null)" ] && [ "$(ps -o stat= -p "$p" 2>/dev/null | cut -c1)" != Z ] && left="$left $p"
        done
        [ -z "$left" ] && break
        sleep 1
    done
    sleep 2
    rm -rf "${cap:?}"; mkdir -p "$cap"
    echo "   capture -> $cap"
    env -u TT_VISIBLE_DEVICES timeout "$CAPTURE_TIMEOUT" $PY $VAL/hang_capture.py "$cap" \
        --time-budget $((CAPTURE_TIMEOUT - 40)) --meta "cache=$c" 2>&1 | sed 's/^/   | /'
    echo "   capture exit ${PIPESTATUS[0]} after $((SECONDS - t0)) s"
    cp "$runlog" "$cap/run.log" 2>/dev/null
    grep -ah 'g++ compile cmd' "$runlog" | grep -a 'fabric_erisc_router/' > "$cap/compile_cmds.txt"
    for d in "$c"/*/kernels/fabric_erisc_router/*/; do
        [ -f "$d/named_ct_arg_map_generated.h" ] || continue
        h=$(basename "$d"); mkdir -p "$cap/cache/kernels/fabric_erisc_router/$h"
        cp "$d/named_ct_arg_map_generated.h" "$cap/cache/kernels/fabric_erisc_router/$h/"
        for k in active_erisc subordinate_active_erisc; do
            [ -f "$d/$k/$k.elf.xip.elf" ] || continue
            mkdir -p "$cap/cache/kernels/fabric_erisc_router/$h/$k"
            $OBJCOPY --strip-all "$d/$k/$k.elf.xip.elf" "$cap/cache/kernels/fabric_erisc_router/$h/$k/$k.elf.xip.elf"
        done
    done
    timeout 120 $PY $VAL/hang_decode.py "$cap" "$c" > "$cap/decode.txt" 2>&1
    echo "   decode exit $? -> $cap/decode.txt"
}

budget_ok() { [ $((SECONDS + $1)) -le "$JOB_BUDGET" ]; }

run_cap() {
    local label=$1 yaml=$2 c=$3 to=$4 t0=$SECONDS st child own_sid try hung=0 rc res
    shift 4
    local cmd=(./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "$yaml")
    if [[ $yaml == gtest:* ]]; then
        cmd=(./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${yaml#gtest:}")
    elif [[ $yaml == vc1:* ]]; then
        # Two processes, one 2x2 mesh each (the P150x8 split; VC1 on the links between them), through tt-run.
        cmd=($PY $W/ttnn/ttnn/distributed/ttrun.py --rank-binding $HARNESS/vc1/p150x8_2x2_split_rank_bindings.yaml
            ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "${yaml#vc1:}")
    fi
    mkdir -p "$c"
    RUN_PIDS=""
    rm -f generated/fabric/bandwidth_summary_results_blackhole.csv "$out/$label.csv"
    echo "=== $label: $yaml (timeout ${to} s) $*"
    setsid -w env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$W TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_CACHE=$c TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 "$@" "${cmd[@]}" \
        > "$out/$label.log" 2>&1 < /dev/null &
    RUN_ROOT=$!
    own_sid=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')
    RUN_SID=""
    for try in $(seq 1 10); do
        sleep 0.5
        RUN_SID=$(ps -o sid= -p "$RUN_ROOT" 2>/dev/null | tr -d ' ')
        if [ -n "$RUN_SID" ] && [ "$RUN_SID" != "$RUN_ROOT" ]; then
            child=$(ps -o pid= --ppid "$RUN_ROOT" 2>/dev/null | head -1 | tr -d ' ')
            RUN_SID=""
            [ -n "$child" ] && RUN_SID=$(ps -o sid= -p "$child" 2>/dev/null | tr -d ' ')
        fi
        [ -n "$RUN_SID" ] && [ "$RUN_SID" != "$own_sid" ] && break
        RUN_SID=""
    done
    while :; do
        st=$(ps -o stat= -p "$RUN_ROOT" 2>/dev/null | cut -c1)
        [ -z "$st" ] || [ "$st" = Z ] && break
        RUN_PIDS=$(_tree "$RUN_SID" "$RUN_ROOT" | tr '\n' ' ')
        if [ $((SECONDS - t0)) -ge "$to" ]; then hung=1; break; fi
        sleep 2
    done
    if [ $hung = 1 ]; then
        echo "   no exit after ${to} s: hung; last tests started:"
        sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -aE "Running Test|^\[ RUN      \]" | tail -2 | sed 's/^/   | /'
        _kill "$RUN_SID" "$RUN_ROOT" 15
        RUN_ROOT=""
        _capture "$out/${label}_hangcap" "$c" "$out/$label.log"
        return 2
    fi
    wait "$RUN_ROOT"; rc=$?
    RUN_ROOT=""
    [ -f generated/fabric/bandwidth_summary_results_blackhole.csv ] && cp generated/fabric/bandwidth_summary_results_blackhole.csv "$out/$label.csv"
    if [[ $yaml == gtest:* ]]; then
        res=$(grep -aE '^\[  PASSED  \]|^\[  FAILED  \] [0-9]+ test|^\[  SKIPPED \] [0-9]+ test' "$out/$label.log" | tr '\n' ' ')
        echo "$label: ${res:-NO RESULT} (exit $rc, $((SECONDS - t0)) s)"
        grep -a '^\[  FAILED  \] [A-Z]' "$out/$label.log" | head -8 | sed 's/^/   | /'
        [ $rc = 0 ] && [ -n "$res" ] && ! echo "$res" | grep -q 'FAILED'
        return
    fi
    if [[ $yaml == vc1:* ]]; then
        # Both ranks report; every report must show no failure.
        res=$(sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -aoE 'Tests: .*executed, [0-9]+ failed' | sort | uniq -c | tr '\n' ' ')
        echo "$label: ${res:-NO RESULT} (exit $rc, $((SECONDS - t0)) s)"
        sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -aE "FAILED|TT_FATAL|TT_THROW|mismatch" | head -5 | sed 's/^/   | /'
        [ $rc = 0 ] && [ -n "$res" ] && ! sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -aoE 'Tests: .*executed, [0-9]+ failed' | grep -qv ", 0 failed"
        return
    fi
    res=$(sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -aoE 'Tests: .*executed, [0-9]+ failed' | tail -1)
    echo "$label: ${res:-NO RESULT} (exit $rc, $((SECONDS - t0)) s)"
    sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -aE "FAILED|TT_FATAL|TT_THROW|mismatch" | head -5 | sed 's/^/   | /'
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
