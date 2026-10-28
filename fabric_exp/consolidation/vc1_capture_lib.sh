#!/bin/bash
# VC1 repeat runs with a read-only hang-state capture (source after jobs_lib.sh and vc1_lib.sh).
#
# vc1_repeat_capture NAME VARIANT-DIR|tree LABEL LEVEL N
#   vc1_repeat's protocol (same env, swap and restore, logs LABEL_r<i>_13129088_vc1.log, a fresh cache
#   /home/aagarwal/.tt-metal-cache-p1vc1cap-NAME-LEVEL whose first run compiles, stop at the first failure), plus:
#   - each run is launched in its own session (setsid) and polled; past VC1CAP_RUN_TIMEOUT (120 s) it is a hang;
#   - compile commands are logged (TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1, log output only) for the decoder;
#   - on a hang (timeout, or fewer than 24 test finishes) the run's processes are killed: SIGTERM, then SIGKILL after
#     15 s, to exactly the processes of that session plus the launched process's descendants (never pkill); then
#     wait until they are gone (other holders of /dev/tenstorrent are reported, not waited for); then hang_capture.py writes
#     fabric_exp/results/LABEL_hangcap/ (plus the run log, compile commands and stripped router ELFs + named args
#     under LABEL_hangcap/cache/, so it decodes on its own), hang_decode.py writes LABEL_hangcap/decode.txt, and
#     the job restores the tree and stops;
#   - a failure that is not a hang (an error with all 24 finishes) stops the job without a capture.
#   Returns 0 when all N runs are clean, 1 otherwise. New runs are not started when the job (bash $SECONDS) could
#   no longer fit a hang, its kill, the capture and the decode under VC1CAP_JOB_BUDGET (1450 s; the broker's cap
#   is 1500 s).
#
# vc1_capture_selftest NAME VARIANT-DIR|tree LEVEL
#   One normal sanity run (fresh cache, compiles), then the same capture on the now idle device into
#   fabric_exp/results/NAME_<level>_capselftest_hangcap/ and its decode: checks the capture's mechanics and the
#   decoder on real data (routers are torn down; binaries, runtime args and buffers are still in L1).
#
# Clean sanity runs take 3-6 s, 16 s for the first (compile) (p1l2be_r*). Budget for one job: 10 clean runs
# ~80 s + hang 120 s + kill <= 45 s + release wait <= 60 s + capture <= 260 s + decode <= 120 s ~ 690 s worst case
# (typical: capture 10-30 s, decode 1 s, so ~4-5 min for a job that ends in a hang).
VC1CAP_RUN_TIMEOUT=${VC1CAP_RUN_TIMEOUT:-120}
VC1CAP_JOB_BUDGET=${VC1CAP_JOB_BUDGET:-1450}
VC1CAP_CAPTURE_TIMEOUT=${VC1CAP_CAPTURE_TIMEOUT:-260}
VC1CAP_CACHE_ROOT=${VC1CAP_CACHE_ROOT:-/home/aagarwal/.tt-metal-cache-p1vc1cap}
VC1CAP_ROOT=""   # PID of the running test's launcher (session leader), "" when none
VC1CAP_SID=""
VC1CAP_PIDS=""   # every process of the run, as last seen
VC1CAP_OBJCOPY=/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy

_vc1cap_tree() {  # sid root: PIDs in session SID or descending from ROOT (this shell excluded)
    local own_sid; own_sid=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')
    [ "$1" = "$own_sid" ] && set -- "" "$2"  # never select by this job's own session
    ps -e -o pid=,ppid=,sid= | awk -v sid="$1" -v root="$2" -v self="$$" '
        { n++; p[n] = $1; pp[$1] = $2; s[$1] = $3 }
        END { for (i = 1; i <= n; i++) { q = p[i]; keep = (sid != "" && s[q] == sid); k = 0
                  while (!keep && q > 1 && k < 64) { if (q == root) keep = 1; q = pp[q]; k++ }
                  if (keep && p[i] != self) print p[i] } }'
}

_vc1cap_alive() {  # sid root: the tree's PIDs that are not zombies
    local pids p
    pids=$(_vc1cap_tree "$1" "$2")
    for p in $pids; do
        [ "$(ps -o stat= -p "$p" 2>/dev/null | cut -c1)" = Z ] || [ -z "$(ps -o pid= -p "$p" 2>/dev/null)" ] || echo "$p"
    done
}

_vc1cap_kill() {  # sid root [term-wait-s]: TERM the run's processes, KILL survivors, reap the launcher
    local sid=$1 root=$2 tw=${3:-15} pids t
    pids=$(_vc1cap_alive "$sid" "$root")
    VC1CAP_PIDS="$VC1CAP_PIDS $pids"
    if [ -n "$pids" ]; then
        echo "   kill: session $sid, $(echo $pids | wc -w) processes:" \
            "$(ps -o pid=,comm= -p "$(echo $pids | tr ' ' ,)" 2>/dev/null | awk '{printf "%s(%s) ", $2, $1}')"
        kill -TERM $pids 2>/dev/null
        for t in $(seq 1 "$tw"); do
            pids=$(_vc1cap_alive "$sid" "$root"); [ -z "$pids" ] && break; sleep 1
        done
        if [ -n "$pids" ]; then
            echo "   kill: SIGKILL to $(echo $pids | wc -w) survivors after ${tw} s: $pids"
            kill -KILL $pids 2>/dev/null
            for t in $(seq 1 30); do
                pids=$(_vc1cap_alive "$sid" "$root"); [ -z "$pids" ] && break; sleep 1
            done
        fi
        [ -n "$pids" ] && echo "   !!! still alive after SIGKILL: $(ps -o pid=,stat=,comm= -p "$(echo $pids | tr ' ' ,)" 2>/dev/null | tr '\n' ' ')"
    fi
    wait "$root" 2>/dev/null
}

_vc1cap_device_users() {  # PIDs (readable /proc) holding /dev/tenstorrent open
    find /proc/[0-9]*/fd -lname '/dev/tenstorrent*' 2>/dev/null | cut -d/ -f3 | sort -un | tr '\n' ' '
}

_vc1cap_wait_released() {  # wait (<= 60 s) until the run's processes are gone (their device fds closed)
    local t p left users
    for t in $(seq 1 60); do
        left=""
        for p in $VC1CAP_PIDS; do
            [ -n "$(ps -o pid= -p "$p" 2>/dev/null)" ] && [ "$(ps -o stat= -p "$p" 2>/dev/null | cut -c1)" != Z ] && left="$left $p"
        done
        [ -z "$left" ] && break
        sleep 1
    done
    users=$(_vc1cap_device_users)
    if [ -n "$left" ]; then
        echo "   !!! run processes still alive after 60 s: [$left] (capturing anyway: the capture only reads)"
    else
        echo "   run processes gone after ${t} s"
    fi
    # Anything else holding the device is reported, not waited for (the broker gives this job the device).
    [ -n "$users" ] && echo "   note: /dev/tenstorrent open in: $(for p in $users; do ps -o pid=,user=,etime=,comm= -p $p 2>/dev/null; done | tr -s ' \n' ' ')"
    sleep 2
    [ -z "$left" ]
}

_vc1cap_cleanup() {  # EXIT trap: never leave a run of ours holding the device
    if [ -n "$VC1CAP_ROOT" ]; then
        echo "!!! job exiting with a run in progress: killing it"
        _vc1cap_kill "$VC1CAP_SID" "$VC1CAP_ROOT" 5
        VC1CAP_ROOT=""
    fi
}

_vc1cap_trap_on() {  # chain _vc1cap_cleanup in front of the caller's EXIT trap (jobs_lib.sh's tree restore)
    local prev
    prev=$(trap -p EXIT); prev=${prev#trap -- }; prev=${prev% EXIT}
    eval "VC1CAP_PREV_TRAP=${prev:-\"\"}"
    trap "_vc1cap_cleanup; $VC1CAP_PREV_TRAP" EXIT
}

_vc1cap_trap_off() {
    if [ -n "$VC1CAP_PREV_TRAP" ]; then trap -- "$VC1CAP_PREV_TRAP" EXIT; else trap - EXIT; fi
}

# _vc1cap_run LOG CACHE OPT: one sanity run; sets VC1CAP_RC (exit code, 124 = killed as hung), VC1CAP_HUNG,
# VC1CAP_FIN (test finishes over both ranks), VC1CAP_BAD (error lines), VC1CAP_SECS.
# VC1CAP_EXTRA_ENV="VAR=value ..." goes after the defaults, so it overrides them (e.g. main's router: kernel v1, flags 0,
# with KA_ENV= for the firmware keepalive).
_vc1cap_run() {
    local log=$1 c=$2 o=$3 optenv="" t0=$SECONDS st child
    [ "$o" = auto ] || optenv="TT_METAL_FABRIC_OPT_LEVEL=$o"
    VC1CAP_PIDS=""
    VC1CAP_HUNG=0
    setsid -w env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal \
        TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_NO_BW_TELEMETRY=1 \
        ${KA_ENV-TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000} TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_CACHE=$c $optenv \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 ${VC1CAP_EXTRA_ENV:-} \
        python_env/bin/python ttnn/ttnn/distributed/ttrun.py \
        --rank-binding fabric_exp/vc1/p150x8_2x2_split_rank_bindings.yaml \
        ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $PWD/${VC1CAP_YAML:-fabric_exp/vc1/vc1_sanity.yaml} \
        > "$log" 2>&1 < /dev/null &
    VC1CAP_ROOT=$!
    # The run's session: never this job's own (setsid may not have run yet, or may have forked). Retry for up to 5 s;
    # if no separate session shows up, the kill falls back to the launched process's descendants only.
    local own_sid try
    own_sid=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')
    VC1CAP_SID=""
    for try in $(seq 1 10); do
        sleep 0.5
        VC1CAP_SID=$(ps -o sid= -p "$VC1CAP_ROOT" 2>/dev/null | tr -d ' ')
        if [ -n "$VC1CAP_SID" ] && [ "$VC1CAP_SID" != "$VC1CAP_ROOT" ]; then  # setsid forked: the session is its child's
            child=$(ps -o pid= --ppid "$VC1CAP_ROOT" 2>/dev/null | head -1 | tr -d ' ')
            VC1CAP_SID=""
            [ -n "$child" ] && VC1CAP_SID=$(ps -o sid= -p "$child" 2>/dev/null | tr -d ' ')
        fi
        [ -n "$VC1CAP_SID" ] && [ "$VC1CAP_SID" != "$own_sid" ] && break
        VC1CAP_SID=""
    done
    [ -z "$VC1CAP_SID" ] && echo "   note: no separate session for the run; kill limited to the launcher's descendants"
    while :; do
        st=$(ps -o stat= -p "$VC1CAP_ROOT" 2>/dev/null | cut -c1)
        [ -z "$st" ] || [ "$st" = Z ] && break
        VC1CAP_PIDS=$(_vc1cap_tree "$VC1CAP_SID" "$VC1CAP_ROOT" | tr '\n' ' ')
        if [ $((SECONDS - t0)) -ge "$VC1CAP_RUN_TIMEOUT" ]; then VC1CAP_HUNG=1; break; fi
        sleep 2
    done
    if [ $VC1CAP_HUNG = 1 ]; then
        echo "   no exit after ${VC1CAP_RUN_TIMEOUT} s: hung"
        _vc1cap_kill "$VC1CAP_SID" "$VC1CAP_ROOT" 15
        VC1CAP_RC=124
    else
        wait "$VC1CAP_ROOT"; VC1CAP_RC=$?
        local left; left=$(_vc1cap_alive "$VC1CAP_SID" "$VC1CAP_ROOT")
        if [ -n "$left" ]; then echo "   launcher exited, processes of its session remain"; _vc1cap_kill "$VC1CAP_SID" "$VC1CAP_ROOT" 15; fi
    fi
    VC1CAP_ROOT=""
    VC1CAP_SECS=$((SECONDS - t0))
    VC1CAP_FIN=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -ac 'Finished\. (test_tt_fabric')
    VC1CAP_BAD=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -acE 'TT_FATAL|TT_THROW|HUNG|timed out|Aborted|mismatch')
}

# _vc1cap_capture CAPDIR CACHE RUNLOG META...: release wait, capture, cache copy, decode
_vc1cap_capture() {
    local cap=$1 c=$2 runlog=$3 t0=$SECONDS d h k; shift 3
    _vc1cap_wait_released
    rm -rf "${cap:?}"; mkdir -p "$cap"
    echo "   capture -> $cap"
    env -u TT_VISIBLE_DEVICES timeout "$VC1CAP_CAPTURE_TIMEOUT" python_env/bin/python $CON/hang_capture.py "$cap" \
        --time-budget $((VC1CAP_CAPTURE_TIMEOUT - 40)) "$@" 2>&1 | sed 's/^/   | /'
    echo "   capture exit ${PIPESTATUS[0]} after $((SECONDS - t0)) s"
    # What the decoder needs from this run, kept with the capture: the run log, the router compile commands of
    # every run of this job, and per router config its named args and stripped ELF (load image only).
    cp "$runlog" "$cap/run.log" 2>/dev/null
    grep -ah 'g++ compile cmd' $out/${VC1CAP_LABEL}_r*_13129088_vc1.log "$runlog" 2>/dev/null \
        | grep -a 'fabric_erisc_router_v2/' > "$cap/compile_cmds.txt"
    for d in "$c"/*/kernels/fabric_erisc_router_v2/*/; do
        [ -f "$d/named_ct_arg_map_generated.h" ] || continue
        h=$(basename "$d"); mkdir -p "$cap/cache/kernels/fabric_erisc_router_v2/$h"
        cp "$d/named_ct_arg_map_generated.h" "$cap/cache/kernels/fabric_erisc_router_v2/$h/"
        for k in active_erisc subordinate_active_erisc; do
            [ -f "$d/$k/$k.elf.xip.elf" ] || continue
            mkdir -p "$cap/cache/kernels/fabric_erisc_router_v2/$h/$k"
            $VC1CAP_OBJCOPY --strip-all "$d/$k/$k.elf.xip.elf" "$cap/cache/kernels/fabric_erisc_router_v2/$h/$k/$k.elf.xip.elf"
        done
    done
    echo "   kept $(grep -c . "$cap/compile_cmds.txt") compile commands, $(ls "$cap/cache/kernels/fabric_erisc_router_v2" 2>/dev/null | wc -l) router configs"
    timeout 120 python_env/bin/python $CON/hang_decode.py "$cap" "$c" > "$cap/decode.txt" 2>&1
    echo "   decode exit $? -> $cap/decode.txt; summary:"
    sed -n '/^== Traffic waiting/,$p' "$cap/decode.txt" | head -60 | sed 's/^/   | /'
    echo "   re-decode: python_env/bin/python $CON/hang_decode.py $cap $cap/cache   (or the full cache $c)"
}

vc1_repeat_capture() {  # NAME VARIANT-DIR|tree LABEL LEVEL N
    local n=$1 v=$2 l=$3 o=$4 N=$5 i c ok=0 lab res=0 reason need
    c=$VC1CAP_CACHE_ROOT-$n-$o
    VC1CAP_LABEL=$l
    rm -rf "${c:?}"; mkdir -p $c
    _vc1cap_trap_on
    if [ "$v" != tree ]; then python3 $CON/swap.py apply $v $BK || { echo STOPPED; exit 1; }; fi
    for i in $(seq 1 $N); do
        lab=${l}_r${i}
        need=$((VC1CAP_RUN_TIMEOUT + 45 + 60 + VC1CAP_CAPTURE_TIMEOUT + 120 + 20))
        if [ $((SECONDS + need)) -gt "$VC1CAP_JOB_BUDGET" ]; then
            echo "=== stopping before run $i: job at ${SECONDS} s, a hung run could need ${need} s more (budget ${VC1CAP_JOB_BUDGET} s)"
            res=1; break
        fi
        echo "=== ${lab}: $n, opt $o, run $i/$N (timeout ${VC1CAP_RUN_TIMEOUT} s, job at ${SECONDS} s) ==="
        _vc1cap_run $out/${lab}_13129088_vc1.log $c $o
        echo "exit $VC1CAP_RC after ${VC1CAP_SECS} s; $VC1CAP_FIN test finishes over both ranks (24 expected); $VC1CAP_BAD error lines"
        if [ $VC1CAP_HUNG = 1 ] || [ "$VC1CAP_FIN" -ne 24 ] || [ $VC1CAP_RC -ne 0 ] || [ "$VC1CAP_BAD" -ne 0 ]; then
            echo "!!! $n run $i failed: last tests $(sed 's/\x1b\[[0-9;]*m//g' $out/${lab}_13129088_vc1.log | grep -a 'Running Test:' | tail -2 | grep -o 'Running Test: [A-Za-z0-9_]*' | tr '\n' ' ')"
            res=1
            if [ $VC1CAP_HUNG = 1 ] || [ "$VC1CAP_FIN" -ne 24 ]; then
                reason=$([ $VC1CAP_HUNG = 1 ] && echo "timeout ${VC1CAP_RUN_TIMEOUT}s" || echo "exit $VC1CAP_RC with $VC1CAP_FIN finishes")
                echo "=== hang capture (${reason}) ==="
                _vc1cap_capture $out/${l}_hangcap $c $out/${lab}_13129088_vc1.log --meta label=$lab --meta name=$n \
                    --meta variant=$v --meta level=$o --meta run=$i --meta cache=$c --meta "reason=$reason"
            else
                echo "    (not a hang: no capture)"
            fi
            break
        fi
        ok=$((ok+1))
    done
    echo "vc1_repeat_capture $n -$o: $ok clean runs of $N (job at ${SECONDS} s)"
    if [ "$v" != tree ]; then python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }; fi
    _vc1cap_trap_off
    return $res
}

vc1_capture_selftest() {  # NAME VARIANT-DIR|tree LEVEL
    local n=$1 v=$2 o=$3 c lab
    lab=${n}_$(echo $o | tr A-Z a-z)_capselftest
    c=$VC1CAP_CACHE_ROOT-$n-$o
    VC1CAP_LABEL=$lab
    rm -rf "${c:?}"; mkdir -p $c
    _vc1cap_trap_on
    if [ "$v" != tree ]; then python3 $CON/swap.py apply $v $BK || { echo STOPPED; exit 1; }; fi
    echo "=== ${lab}: $n, opt $o, one sanity run, then a capture of the idle device ==="
    _vc1cap_run $out/${lab}_13129088_vc1.log $c $o
    echo "exit $VC1CAP_RC after ${VC1CAP_SECS} s; $VC1CAP_FIN test finishes over both ranks (24 expected); $VC1CAP_BAD error lines"
    _vc1cap_capture $out/${lab}_hangcap $c $out/${lab}_13129088_vc1.log --meta label=$lab --meta name=$n \
        --meta variant=$v --meta level=$o --meta run=selftest --meta cache=$c \
        --meta "reason=selftest (run exit $VC1CAP_RC, $VC1CAP_FIN finishes)"
    if [ "$v" != tree ]; then python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }; fi
    _vc1cap_trap_off
    echo "vc1_capture_selftest $n -$o done (job at ${SECONDS} s)"
}

# run_vc1_cap LABEL CACHE OPT|auto: vc1_lib.sh's run_vc1 (the full VC1 split suite, fabric_exp/vc1/vc1_split.yaml,
# 34 finishes, a fresh cache, the per-test CSV kept) with the hang capture armed: a run that does not exit within
# VC1CAP_SPLIT_TIMEOUT (240 s; a clean run takes ~25-35 s) or ends with fewer than 34 finishes is captured into
# fabric_exp/results/LABEL_hangcap/ as in vc1_repeat_capture. Used by vc1_variant when VC1_RUN=run_vc1_cap.
# VC1CAP_SPLIT_YAML and VC1CAP_SPLIT_FINISHES run another yaml of the same kind (e.g. vc1/vc1_alloc.yaml, 50).
run_vc1_cap() {  # label cache opt|auto
    local l=$1 c=$2 o=$3 res=0 reason
    rm -rf "${c:?}"; mkdir -p $c
    rm -f generated/fabric/bandwidth_summary_results_blackhole.csv generated/fabric/bandwidth_results_blackhole.csv \
        $out/${l}_13129088_vc1.csv $out/${l}_13129088_vc1_tests.csv
    echo "=== ${l}_13129088_vc1: opt $o, cache $c (hang capture armed) ==="
    VC1CAP_LABEL=$l
    _vc1cap_trap_on
    local VC1CAP_YAML=${VC1CAP_SPLIT_YAML:-fabric_exp/vc1/vc1_split.yaml} VC1CAP_RUN_TIMEOUT=${VC1CAP_SPLIT_TIMEOUT:-240}
    local fins=${VC1CAP_SPLIT_FINISHES:-34}
    _vc1cap_run $out/${l}_13129088_vc1.log $c $o
    echo "exit $VC1CAP_RC after ${VC1CAP_SECS} s; $VC1CAP_FIN test finishes over both ranks ($fins expected); $VC1CAP_BAD error lines"
    [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_13129088_vc1_tests.csv
    if [ $VC1CAP_HUNG = 1 ] || [ "$VC1CAP_FIN" -ne $fins ] || [ $VC1CAP_RC -ne 0 ] || [ "$VC1CAP_BAD" -ne 0 ]; then
        res=1
        echo "!!! ${l} failed: last tests $(sed 's/\x1b\[[0-9;]*m//g' $out/${l}_13129088_vc1.log | grep -a 'Running Test:' | tail -2 | grep -o 'Running Test: [A-Za-z0-9_]*' | tr '\n' ' ')"
        if [ $VC1CAP_HUNG = 1 ] || [ "$VC1CAP_FIN" -ne $fins ]; then
            reason=$([ $VC1CAP_HUNG = 1 ] && echo "timeout ${VC1CAP_RUN_TIMEOUT}s" || echo "exit $VC1CAP_RC with $VC1CAP_FIN finishes")
            echo "=== hang capture (${reason}) ==="
            _vc1cap_capture $out/${l}_hangcap $c $out/${l}_13129088_vc1.log --meta label=$l --meta level=$o \
                --meta cache=$c --meta yaml=$VC1CAP_YAML --meta "reason=$reason"
        fi
    fi
    _vc1cap_trap_off
    return $res
}
