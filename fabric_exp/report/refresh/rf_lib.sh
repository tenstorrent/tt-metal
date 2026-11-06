#!/bin/bash
# Shared parts of the report data refresh's lb2 jobs (rf_lb_stock.sh, rf_lb_harness.sh). Source after cd to the study tree;
# needs out=fabric_exp/results. Trees are staged as one file each in /data/aagarwal/lb2/trees (pr_perf/stage_pr_tree.sh,
# refresh/stage_rf_tree.sh) and unpacked at their build paths (the binaries' RUNPATH); a path that holds something no
# refresh or PR harness put there (no .lb2_pr_owned) is never touched.
#   rf_unpack NAME PATH TARBALL WANT_COMMIT
#   rf_run LABEL TREE CACHE YAML TIMEOUT [ENV=VALUE ...]        one test_tt_fabric run (port_lib.sh's run_port: fast
#       dispatch, TT_METAL_CLEAR_L1=1, profiler off, BW telemetry off, compile commands logged); 2 on a hang (captured)
#   rf_run2 LABEL TREE CACHE YAML BINDINGS TIMEOUT [ENV=VALUE ...] one two-rank run through tt-run (the VC1 split);
#       0 when both ranks finish every test with no error line, 1 on a failure, 2 on a hang (captured)
CON=fabric_exp/consolidation
RF=fabric_exp/report/refresh
G=/data/aagarwal/lb2/trees
STUDY_PY=/home/aagarwal/tt-metal/python_env/bin/python
source $CON/vc1_capture_lib.sh
source fabric_exp/port_slots/port_lib.sh

rf_unpack() {  # name path tarball want
    local r=$1 T=$2 f=$3 want=$4
    if [ -e $T ] && [ ! -e $T/.lb2_pr_owned ]; then echo "!!! $T exists and is not this harness's: not touched"; exit 1; fi
    (cd $(dirname $f) && sha256sum -c --quiet $(basename $f).sha256) || { echo "!!! $f fails its sha256"; exit 1; }
    rm -rf "${T:?}.new" && mkdir -p $T.new && tar -I zstd -xf $f -C $T.new || { echo "!!! unpack of $f failed"; exit 1; }
    (cd $T.new && md5sum -c --quiet MD5SUMS) || { echo "!!! $r: binaries differ from MD5SUMS"; exit 1; }
    case $(cat $T.new/COMMIT) in $want*) ;; *) echo "!!! $r tree at $(cat $T.new/COMMIT), want $want"; exit 1 ;; esac
    ln -sfn build_Release $T.new/build
    echo "created by $(hostname) job ${SLURM_JOB_ID:-?} $(date -u +%FT%TZ) from $f (report refresh)" > $T.new/.lb2_pr_owned
    rm -rf "${T:?}" && mv $T.new $T || { echo "!!! could not put the tree at $T"; exit 1; }
    echo "$r: $(cut -c1-11 $T/COMMIT) + $(grep -c '^[+-][^+-]' $T/PATCH) patch lines at $T ($(du -sh $T | cut -f1))"
}

rf_run() {  # label tree cache yaml timeout env...
    local label=$1 yaml=$4 to=$5
    PT=$2 C=$3
    shift 5
    mkdir -p $C
    run_port $label $yaml $to TT_METAL_RUNTIME_ROOT=$PT "$@"
}

rf_run2() {  # label tree cache yaml bindings timeout env...
    local l=$1 PT=$2 c=$3 yaml=$4 rb=$5 to=$6 t0=$SECONDS st child own_sid try hung=0 rc log fins bad expect lib
    shift 6
    lib=$PT/build_Release
    log=$out/$l.log
    mkdir -p $c
    [ -f "$yaml" ] || { echo "!!! no yaml $yaml"; return 1; }
    expect=$(( 2 * $(grep -oE 'size: \[[0-9, ]*\]' $yaml | tr -cd ',\n' | awk '{n += length($0) + 1} END {print n + 0}') ))
    echo "=== $l: $yaml, $(basename $rb), $expect finishes expected (timeout ${to} s, job at ${SECONDS} s) $*"
    VC1CAP_LABEL=$l
    VC1CAP_PIDS=""
    _vc1cap_trap_on
    setsid -w env -C $PT -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PT TT_METAL_RUNTIME_ROOT=$PT \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_CACHE=$c \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_LOG_JIT_TELEMETRY=1 "$@" \
        $STUDY_PY $PT/ttnn/ttnn/distributed/ttrun.py --rank-binding $rb \
        $lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $PWD/$yaml > "$log" 2>&1 < /dev/null &
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
    echo "   router compile levels: $(sed 's/\x1b\[[0-9;]*m//g' $log | grep -a 'g++ compile cmd' | grep -a 'fabric_erisc_router' \
        | grep -oE ' -O[0-9s]+ ' | sort | uniq -c | tr -s ' \n' ' ')"
    sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -a "config buffer, least headroom" | sed 's/^.*summary | /   /' | sort -u | head -2
    [ $rc = 0 ] && [ $bad = 0 ] && [ "$fins" = "$expect" ] && return 0
    sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aE 'TT_FATAL|TT_THROW|HUNG|what\(\)' | head -5 | sed 's/^/   | /'
    return 1
}

rf_keep_cache() {  # cache tag: the router kernels' ELFs (with the JIT's -g) and generated headers, kept as a result
    local c=$1 tag=$2
    (cd $c && find . -path '*/kernels/fabric_erisc_router*/*' \( -name '*.elf' -o -name '*.h' -o -name '*.hpp' \) -print \
        | tar czf $OLDPWD/$out/${tag}_routers.tgz -T -) && echo "   kept $c's router builds: $out/${tag}_routers.tgz ($(du -k $out/${tag}_routers.tgz | cut -f1) KB)"
}
rf_keep_src() {  # tree tag: the router sources as the JIT compiled them (members start with tt_metal/)
    (cd $1 && tar czf $OLDPWD/$out/${2}_src.tgz tt_metal/fabric/hw/inc tt_metal/fabric/impl/kernels/edm_fabric tt_metal/hw/inc \
        tt_metal/hostdevcommon) && echo "   sources: $out/${2}_src.tgz"
}
