#!/bin/bash
# The port's tree and one test run on an lb2 node (source after vc1_capture_lib.sh; needs out=fabric_exp/results):
#   port_tree_setup   copies the staged port tree (PORT_TREE_SRC, default /data/aagarwal/lb2/trees/port_slots_22cd:
#                     22cdb1e55c3 + slot_override.patch, lean) to /home/aagarwal/tt-metal-port-slots on the node (the
#                     binaries' RUNPATH) and checks the binaries' md5s; refuses to overwrite a09u32's worktree.
#   run_port LABEL YAML TIMEOUT-S [ENV=VALUE ...]   one test_tt_fabric run with the port's own settings (as its run
#                     library: fast dispatch, TT_METAL_CLEAR_L1=1, profiler off, BW telemetry off, compile commands
#                     logged), TT_FABRIC_EXP_LOG_SLOTS=1 and the given env; polled; past TIMEOUT-S killed and captured
#                     (results/LABEL_hangcap, hang_capture.py), returns 2; else 0 when "Tests: N executed, 0 failed".
#                     Results: results/LABEL.log, .csv (summary), _tests.csv (per test).
clean() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }

port_tree_setup() {
    SRC=${PORT_TREE_SRC:-/data/aagarwal/lb2/trees/port_slots_22cd}
    PT=${PS_PT:-/home/aagarwal/tt-metal-port-slots}   # PS_PT: another place, for the stub dry run only (RUNPATH)
    if [ "$(hostname)" = bh-lb-120-a09u32 ] && [ "$PT" = /home/aagarwal/tt-metal-port-slots ]; then
        echo "!!! a09u32 holds the port worktree at $PT: run this on an lb2 node"; exit 1
    fi
    mkdir -p $PT $out
    rsync -a --delete --exclude=/generated/ $SRC/ $PT/ || { echo "!!! copy of $SRC failed"; exit 1; }
    (cd $PT && md5sum -c --quiet MD5SUMS) || { echo "!!! the port tree's binaries differ from MD5SUMS"; exit 1; }
    echo "port tree $(cat $PT/PORT_COMMIT) + slot_override.patch at $PT ($(du -sh $PT | cut -f1))"
    C=${C:-/home/aagarwal/.tt-metal-cache-port}
}

run_port() {  # label yaml timeout ENV...: one run, polled; 2 on a hang (captured), else 0 when every test passed
    local label=$1 yaml=$2 to=$3 t0=$SECONDS child own_sid try st hung=0 rc res lib=$PT/build_Release
    shift 3
    VC1CAP_LABEL=$label
    VC1CAP_PIDS=""
    rm -f $PT/generated/fabric/bandwidth_summary_results_blackhole.csv $PT/generated/fabric/bandwidth_results_blackhole.csv \
        "$out/$label.csv" "$out/${label}_tests.csv"
    echo "=== $label: $yaml (timeout ${to} s) $*"
    _vc1cap_trap_on
    setsid -w env -C $PT -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PT \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_CACHE=$C \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_FABRIC_EXP_LOG_SLOTS=1 "$@" \
        $lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "$PWD/$yaml" \
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
        clean "$out/$label.log" | grep -aE "Running Test" | tail -2 | sed 's/^/   | /'
        _vc1cap_kill "$VC1CAP_SID" "$VC1CAP_ROOT" 15
        VC1CAP_ROOT=""
        _vc1cap_capture "$out/${label}_hangcap" "$C" "$out/$label.log" --meta "cache=$C" --meta "label=$label"
        _vc1cap_trap_off
        return 2
    fi
    wait "$VC1CAP_ROOT"; rc=$?
    VC1CAP_ROOT=""
    _vc1cap_trap_off
    [ -f $PT/generated/fabric/bandwidth_summary_results_blackhole.csv ] && cp $PT/generated/fabric/bandwidth_summary_results_blackhole.csv "$out/$label.csv"
    [ -f $PT/generated/fabric/bandwidth_results_blackhole.csv ] && cp $PT/generated/fabric/bandwidth_results_blackhole.csv "$out/${label}_tests.csv"
    res=$(clean "$out/$label.log" | grep -aoE 'Tests: .*executed, [0-9]+ failed' | tail -1)
    echo "$label: ${res:-NO RESULT} (exit $rc, $((SECONDS - t0)) s)"
    clean "$out/$label.log" | grep -aE "FAILED|TT_FATAL|TT_THROW|mismatch" | head -5 | sed 's/^/   | /'
    [ $rc = 0 ]
}

