#!/bin/bash
# Single-mesh test_tt_fabric runs with the read-only hang capture (source after jobs_lib.sh).
#
# run_cap LABEL YAML CACHE TIMEOUT-S [ENV=VALUE ...]
#   One run of YAML with jobs_lib.sh run's settings (v2 slot, BW telemetry off, keepalive 1000, flags 13129088, -O3,
#   compile commands logged), launched in its own session and polled. Past TIMEOUT-S it is a hang: the run's processes
#   are killed (vc1_capture_lib.sh's kill: TERM, then KILL), the device state is captured into
#   fabric_exp/results/LABEL_hangcap/ and decoded (hang_capture.py, hang_decode.py), and run_cap returns 2.
#   Otherwise returns 0 when the log reports "Tests: N executed, 0 failed", 1 if not. Results as jobs_lib.sh run:
#   fabric_exp/results/LABEL.{log,csv} (latency-mode yamls: LABEL_latency.csv). Extra ENV settings go after
#   TIMEOUT-S (e.g. TT_FABRIC_KERNEL_VERSION=1 for the tree's own router).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/vc1_capture_lib.sh
out=fabric_exp/results

run_cap() {
    local label=$1 yaml=$2 c=$3 to=$4 t0=$SECONDS st child own_sid try hung=0 rc res
    shift 4
    VC1CAP_LABEL=$label
    VC1CAP_PIDS=""
    rm -f generated/fabric/bandwidth_summary_results_blackhole.csv generated/fabric/latency_results_blackhole.csv \
        "$out/$label.csv" "$out/${label}_latency.csv"
    echo "=== $label: $yaml (timeout ${to} s) $*"
    _vc1cap_trap_on
    setsid -w env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal TT_METAL_CLEAR_L1=1 \
        TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_NO_BW_TELEMETRY=1 \
        ${KA_ENV-TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000} TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_CACHE=$c \
        TT_METAL_FABRIC_OPT_LEVEL=O3 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 $EXTRA_ENV "$@" \
        ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "$PWD/$yaml" \
        > "$out/$label.log" 2>&1 < /dev/null &
    VC1CAP_ROOT=$!
    own_sid=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')
    VC1CAP_SID=""
    for try in $(seq 1 10); do
        sleep 0.5
        VC1CAP_SID=$(ps -o sid= -p "$VC1CAP_ROOT" 2>/dev/null | tr -d ' ')
        if [ -n "$VC1CAP_SID" ] && [ "$VC1CAP_SID" != "$VC1CAP_ROOT" ]; then
            child=$(ps -o pid= --ppid "$VC1CAP_ROOT" 2>/dev/null | head -1 | tr -d ' ')
            VC1CAP_SID=""
            [ -n "$child" ] && VC1CAP_SID=$(ps -o sid= -p "$child" 2>/dev/null | tr -d ' ')
        fi
        [ -n "$VC1CAP_SID" ] && [ "$VC1CAP_SID" != "$own_sid" ] && break
        VC1CAP_SID=""
    done
    while :; do
        st=$(ps -o stat= -p "$VC1CAP_ROOT" 2>/dev/null | cut -c1)
        [ -z "$st" ] || [ "$st" = Z ] && break
        VC1CAP_PIDS=$(_vc1cap_tree "$VC1CAP_SID" "$VC1CAP_ROOT" | tr '\n' ' ')
        if [ $((SECONDS - t0)) -ge "$to" ]; then hung=1; break; fi
        sleep 2
    done
    if [ $hung = 1 ]; then
        echo "   no exit after ${to} s: hung; last tests started:"
        sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -a "Running Test" | tail -2 | sed 's/^/   | /'
        _vc1cap_kill "$VC1CAP_SID" "$VC1CAP_ROOT" 15
        VC1CAP_ROOT=""
        _vc1cap_capture "$out/${label}_hangcap" "$c" "$out/$label.log" --meta "cache=$c" --meta "label=$label"
        _vc1cap_trap_off
        return 2
    fi
    wait "$VC1CAP_ROOT"; rc=$?
    VC1CAP_ROOT=""
    _vc1cap_trap_off
    [ -f generated/fabric/bandwidth_summary_results_blackhole.csv ] && cp generated/fabric/bandwidth_summary_results_blackhole.csv "$out/$label.csv"
    [ -f generated/fabric/latency_results_blackhole.csv ] && cp generated/fabric/latency_results_blackhole.csv "$out/${label}_latency.csv"
    res=$(sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -aoE 'Tests: .*executed, [0-9]+ failed' | tail -1)
    echo "$label: ${res:-NO RESULT} (exit $rc, $((SECONDS - t0)) s)"
    sed 's/\x1b\[[0-9;]*m//g' "$out/$label.log" | grep -aE "FAILED|TT_FATAL|TT_THROW|mismatch" | head -5 | sed 's/^/   | /'
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
