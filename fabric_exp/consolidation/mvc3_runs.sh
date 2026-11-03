#!/bin/bash
# Multi-VC throughput on the VC1 split (report multi-VC page, 2026-11-02): rows of fabric_exp/vc1/mvc3/ (mvc3_yamls.py:
# one E/W link in mesh 1, [1,0] -> [1,1], next to the landing chip; VC0 and VC2 from [1,0]'s workers, VC1 forwarded in
# from [0,0]). Two MPI ranks, one mesh each (vc1_lib.sh's split).
#   mvc3_runs.sh PASS SET...     SETs in the order given:
#     A_d     v4rxg3v (VC1 + VC2 probe, VC2 compiled in), -O3, default slots (study row 19/8/16/4/4 + VC2 2/4)
#     (A_v8, VC2 8/10 via TT_FABRIC_EXP_SLOTS_VC1=19,4,16,4,4,8,10, was dropped: the host refused it at -O3, program
#     26,640 B against the 26,624 B kernel config buffer, job 479; user: three-VC rows at the default allocation only)
#     B_new   v4rxg3 (the receiver lead, no VC2), -O3, default slots (19,9,16,5,4)
#     B_main  main's router as shipped (kernel v1, flags 0, firmware keepalive), the host's opt level (-Os with VC1)
#   NEW (default v4rxg3) and NEWV (default v4rxg3v) name the variants B_new and A_d swap in (labels do not change).
#   Per set, three processes: base (no launch delay), lag1 (mesh 1 launches 200 ms later), lag0 (mesh 0 launches 200 ms
#   later: rank bindings with TT_RUN_RANK swapped). One fresh kernel cache per set. MATCH=1 adds the matched-count rows
#   (mvc3_matched.py: every flow of the combination finishing together, so one run gives the combined rate), run first
#   (match1) and last (match2) in the set: two measurements per job. Slots logged
#   (TT_FABRIC_EXP_LOG_SLOTS=1); kernel config buffer headroom logged at exit (TT_METAL_LOG_JIT_TELEMETRY=1).
# Labels mvc3<PASS>_<SET>_<group>: fabric_exp/results/<label>.log. A hang (no exit within the timeout) or a failed run
# stops the job; a hang is captured (vc1_capture_lib.sh) first.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_capture_lib.sh
P=$1; shift
[ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: mvc3_runs.sh PASS SET..."; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2 TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS
Y=fabric_exp/vc1/mvc3
RB_STD=fabric_exp/vc1/p150x8_2x2_split_rank_bindings.yaml
RB_LAG0=$Y/p150x8_2x2_split_rank_bindings_lag0.yaml
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }

# mvc3_run LABEL YAML BINDING CACHE OPT|auto TIMEOUT-S [ENV=VALUE ...]: one two-rank run (vc1_capture_lib.sh's _vc1cap_run
# with a chosen rank binding and yaml); returns 0 on a clean run with every test finished on both ranks, 1 on a failed
# run, 2 on a hang (captured).
mvc3_run() {
    local l=$1 yaml=$2 rb=$3 c=$4 o=$5 to=$6 t0=$SECONDS st child own_sid try hung=0 rc optenv="" log tests fins bad
    shift 6
    [ "$o" = auto ] || optenv="TT_METAL_FABRIC_OPT_LEVEL=$o"
    log=$out/$l.log
    tests=$(grep -c '^  - name:' $yaml)
    [ -f "$yaml" ] || { echo "!!! no yaml $yaml"; return 1; }
    local expect=$(( 2 * $(grep -oE 'size: \[[0-9, ]*\]' $yaml | tr -cd ',\n' | awk '{n += length($0) + 1} END {print n + 0}') ))
    echo "=== $l: $yaml, $rb, opt $o, $tests tests, $expect finishes expected (timeout ${to} s, job at ${SECONDS} s) $*"
    VC1CAP_LABEL=$l
    VC1CAP_PIDS=""
    _vc1cap_trap_on
    setsid -w env -u TT_METAL_SLOW_DISPATCH_MODE -u TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT TT_METAL_HOME=/home/aagarwal/tt-metal \
        TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_NO_BW_TELEMETRY=1 \
        ${KA_ENV-TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000} TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_CACHE=$c $optenv \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_FABRIC_EXP_LOG_SLOTS=1 TT_METAL_LOG_JIT_TELEMETRY=1 "$@" \
        python_env/bin/python ttnn/ttnn/distributed/ttrun.py --rank-binding $rb \
        ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $PWD/$yaml \
        > "$log" 2>&1 < /dev/null &
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
        sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -a "Running Test:" | tail -2 | sed 's/^/   | /'
        _vc1cap_kill "$VC1CAP_SID" "$VC1CAP_ROOT" 15
        VC1CAP_ROOT=""
        _vc1cap_capture "$out/${l}_hangcap" "$c" "$log" --meta "cache=$c" --meta "label=$l" --meta "yaml=$yaml"
        _vc1cap_trap_off
        return 2
    fi
    wait "$VC1CAP_ROOT"; rc=$?
    VC1CAP_ROOT=""
    _vc1cap_trap_off
    fins=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -ac 'Finished\. (test_tt_fabric')
    bad=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -acE 'TT_FATAL|TT_THROW|HUNG|timed out|Aborted|mismatch')
    echo "$l: exit $rc after $((SECONDS - t0)) s; $fins test finishes over both ranks ($expect expected); $bad error lines"
    sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE "VC[0-9] sender slots \{[^}]*\} receiver slots \{[^}]*\}" | sort | uniq -c \
        | sed 's/^/   slots: /'
    sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -a "config buffer, least headroom" | sed 's/^.*summary | /   /' | sort -u
    [ $rc = 0 ] && [ $bad = 0 ] && [ "$fins" = "$expect" ] && return 0
    sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aE 'TT_FATAL|TT_THROW|HUNG|what\(\)' | head -5 | sed 's/^/   | /'
    return 1
}

for s in "$@"; do
    echo "##### set $s (job at ${SECONDS} s)"
    o=O3; X=""; R=""; MAINRUN=0
    case $s in
        A_d) R=${NEWV:-v4rxg3v}; X="TT_METAL_ENABLE_FABRIC_VC2=1" ;;
        B_new) R=${NEW:-v4rxg3} ;;
        B_main) MAINRUN=1; o=auto; X="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0" ;;
        *) echo "unknown set $s"; stop ;;
    esac
    [ -n "$R" ] && { python3 $CON/swap.py apply fabric_exp/variants/$R $BK || stop; }
    c=/home/aagarwal/.tt-metal-cache-mvc3-$s
    rm -rf "${c:?}"; mkdir -p $c
    GRPS="base lag1 lag0"  # (not GROUPS: a bash special variable, assignments ignored)
    [ -n "${MATCH:-}" ] && GRPS="match1 base lag1 lag0 match2"
    for g in $GRPS; do
        rb=$RB_STD; D=""; yml=$Y/mvc3_${s}_$g.yaml
        [ $g = lag0 ] && rb=$RB_LAG0
        case $g in lag0|lag1) D="TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS=200" ;; match*) yml=$Y/mvc3_${s}_match.yaml ;; esac
        if [ $MAINRUN = 1 ]; then
            KA_ENV= mvc3_run mvc3${P}_${s}_$g $yml $rb $c $o 400 $X $D; rc=$?
        else
            mvc3_run mvc3${P}_${s}_$g $yml $rb $c $o 400 $X $D; rc=$?
        fi
        [ $rc = 0 ] || { echo "!!! mvc3${P}_${s}_$g: rc $rc"; stop; }
    done
    first=${GRPS%% *}
    echo "   router compile levels: $(sed 's/\x1b\[[0-9;]*m//g' $out/mvc3${P}_${s}_$first.log | grep -a 'g++ compile cmd' \
        | grep -a 'fabric_erisc_router' | grep -oE ' -O[0-9s]+ ' | sort | uniq -c | tr '\n' ' ')"
    [ -n "$R" ] && { python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
