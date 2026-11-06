#!/bin/bash
# The per-channel transaction-ID A/B on one P150 LoudBox (an lb2 job; never a09u32; TRID_ALLOCATION.md section 7, user
# 2026-11-05: option (ii), one P150 A/B decides).
#   RULE     aagarwal/fabric-router-mainline-trid at f9c258d280b (6dad5fec638 + the staged allocator + per-channel IDs) with
#            fabric_exp/port_trid/trid_uniform_switch.patch (local, never committed) built in, switch off
#   UNIFORM  the same build with TT_FABRIC_EXP_TRID_UNIFORM=1: every channel the earlier 16 / 8 / 4 IDs, same group sizes
# Router and host: built in /home/aagarwal/tt-metal-port-trid, staged as one file (PORT_TREE_SRC, .tar.zst) that this job
# unpacks at the same path on the node (the binaries' RUNPATH); md5s and commit checked. No test-infra patch: these rows
# run on the port's own test binary.
# Pass a: the data-checked rows in both arms first (val2d: pval2d_4352; valring: v4/val_ring_p150.yaml; valtorus:
# torus/val_torus_xy_p150.yaml on the 4x2 XY torus), then per suite RULE then UNIFORM; pass b: suites reversed, UNIFORM
# then RULE. One kernel cache per arm, fresh at the start of the job.
#   t42ns  the P150 as a 4x2 XY torus: N/S 1-hop, bidir, 3-hop, wrap and E/W 1-hop, writes and scatter, 256-4352 B
#   ring   v4/exp_ring.yaml          c2d  the 2D census suite (exp_2d_4352.yaml)
#   c1lin  the 1D census suite's Linear rows        c1ne  its NeighborExchange rows
# Each run: the port's own settings (fast dispatch, TT_METAL_CLEAR_L1=1, profiler off, BW telemetry off, compile commands
# logged), TT_FABRIC_EXP_LOG_SLOTS=1 (the host logs every receiver's IDs) and the arm's switch. Checked: every test run
# with 0 failed or skipped by the test binary (one "Skipping:" line each), every logged ID line names the arm's rule, and the distinct ID rows equal the mock's for that
# suite and arm (yamls/trid_expected_ids.json). A failure, a mismatch or a hang (killed, captured) stops the job.
#   trid_runs.sh [SUITE...]    labels ${TR_PREFIX:-tr}<a|b>_<rule|uni>_<suite>: results/<label>.log, .csv, _tests.csv
#   DRY=1 trid_runs.sh ...     prints the runs only
cd /home/aagarwal/tt-metal || exit 1
CON=fabric_exp/consolidation
out=fabric_exp/results
D=fabric_exp/port_trid
E=$D/yamls/trid_expected_ids.json
SRC=${PORT_TREE_SRC:-/data/aagarwal/lb2/trees/port_trid_f9c2.tar.zst}
PT=/home/aagarwal/tt-metal-port-trid
WANT_COMMIT=f9c258d280b
MGD=TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto
DRY=${DRY:-0}
# suite | yaml | extra env
SUITE_TABLE="val2d|fabric_exp/port_slots/yamls/pval2d_4352.yaml|-
valring|fabric_exp/v4/val_ring_p150.yaml|-
valtorus|fabric_exp/torus/val_torus_xy_p150.yaml|$MGD
t42ns|fabric_exp/port_slots/yamls/pvs_t42ns.yaml|$MGD
ring|fabric_exp/v4/exp_ring.yaml|-
c2d|fabric_exp/port_slots/yamls/pvs_c2d.yaml|-
c1lin|fabric_exp/port_slots/yamls/pvs_c1lin.yaml|-
c1ne|fabric_exp/port_slots/yamls/pvs_c1ne.yaml|-"
SUITES=(${*:-t42ns ring c2d c1lin c1ne})
VAL=(val2d valring valtorus)
row() { echo "$SUITE_TABLE" | awk -F'|' -v s=$1 '$1 == s'; }
ntests() {
    python3 -c "
import sys, yaml
n = 0
for t in yaml.safe_load(open(sys.argv[1]))['Tests']:
    m = 1
    for v in (t.get('parametrization_params') or {}).values():
        m *= len(v) if isinstance(v, list) else 1
    n += m
print(n)" $1
}
for s in "${SUITES[@]}" "${VAL[@]}"; do
    r=$(row $s); [ -n "$r" ] || { echo "unknown suite $s"; exit 1; }
    [ -f "$(echo "$r" | cut -d'|' -f2)" ] || { echo "no yaml for $s"; exit 1; }
    python3 -c "import json,sys; e=json.load(open('$E'))[sys.argv[1]]; assert e['rule'] and e['uniform']" $s ||
        { echo "no expected IDs for $s"; exit 1; }
done
if [ "$DRY" != 1 ]; then
    [ "$(hostname)" != bh-lb-120-a09u32 ] || { echo "!!! a09u32 holds the build worktree at $PT: run this on an lb2 node"; exit 1; }
    source $CON/vc1_capture_lib.sh
    mkdir -p $out
    rm -rf "${PT:?}.new" && mkdir -p $PT.new && tar -I zstd -xf $SRC -C $PT.new || { echo "!!! unpack of $SRC failed"; exit 1; }
    rm -rf "${PT:?}" && mv $PT.new $PT || { echo "!!! could not put the unpacked tree at $PT"; exit 1; }
    (cd $PT && md5sum -c --quiet MD5SUMS) || { echo "!!! the port tree's binaries differ from MD5SUMS"; exit 1; }
    case $(cat $PT/PORT_COMMIT) in $WANT_COMMIT*) ;; *) echo "!!! port tree commit $(cat $PT/PORT_COMMIT), want $WANT_COMMIT"; exit 1 ;; esac
    echo "port tree $(cat $PT/PORT_COMMIT) + trid_uniform_switch.patch at $PT ($(du -sh $PT | cut -f1))"
fi
clean() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
CR=/home/aagarwal/.tt-metal-cache-ptrid-rule; CU=/home/aagarwal/.tt-metal-cache-ptrid-uni
[ "$DRY" = 1 ] || { rm -rf "${CR:?}" "${CU:?}"; mkdir -p $CR $CU; }

run_port() {  # label yaml timeout ENV...: one run, polled; 2 on a hang (captured), 1 on a failure, else 0
    local label=$1 yaml=$2 to=$3 t0=$SECONDS child own_sid try st hung=0 rc res lib=$PT/build_Release cmd
    shift 3
    cmd=($lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "$PWD/$yaml")
    if [ "$DRY" = 1 ]; then echo "   $label: $* ${cmd[*]} (timeout $to s, cache $C)"; return 0; fi
    VC1CAP_LABEL=$label
    VC1CAP_PIDS=""
    rm -f $PT/generated/fabric/bandwidth_summary_results_blackhole.csv $PT/generated/fabric/bandwidth_results_blackhole.csv \
        "$out/$label.csv" "$out/${label}_tests.csv"
    echo "=== $label: $yaml (timeout ${to} s) $*"
    _vc1cap_trap_on
    setsid -w env -C $PT -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PT \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_CACHE=$C \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_FABRIC_EXP_LOG_SLOTS=1 "$@" "${cmd[@]}" > "$out/$label.log" 2>&1 < /dev/null &
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
    res=$(clean "$out/$label.log" | grep -aoE 'Tests: .*executed, [0-9]+ failed' | sort | uniq -c | tr -s ' \n' ' ')
    echo "$label: ${res:-NO RESULT} (exit $rc, $((SECONDS - t0)) s)"
    clean "$out/$label.log" | grep -aE "FAILED|TT_FATAL|TT_THROW|mismatch" | head -5 | sed 's/^/   | /'
    [ $rc = 0 ]
}

stop() { echo "!!! $1: stopping"; echo STOPPED; exit 1; }
arm_run() {  # pass arm suite
    local P=$1 a=$2 s=$3 l y n r env=() rc
    r=$(row $s); y=$(echo "$r" | cut -d'|' -f2); [ "$(echo "$r" | cut -d'|' -f3)" = - ] || env+=($(echo "$r" | cut -d'|' -f3))
    n=$(ntests $y)
    l=${TR_PREFIX:-tr}${P}_${a}_$s
    if [ $a = uni ]; then C=$CU; env+=(TT_FABRIC_EXP_TRID_UNIFORM=1); else C=$CR; fi
    echo "##### $l ($n tests; job at ${SECONDS} s)"
    run_port $l $y 600 "${env[@]}"; rc=$?
    [ "$DRY" = 1 ] && return 0
    [ $rc = 2 ] && stop "$l: hang (captured)"
    # The test binary skips a test it cannot place (e.g. more links than the host's routing planes), one "Skipping:"
    # line each: every test must have run and passed, or been skipped.
    local got skips
    got=$(clean $out/$l.log | grep -aoE "Tests: [0-9]+ executed, 0 failed" | tail -1 | grep -oE "[0-9]+" | head -1)
    skips=$(clean $out/$l.log | grep -ac "Skipping: ")
    [ -n "$got" ] && [ "$got" -gt 0 ] && [ $((got + skips)) = $n ] ||
        stop "$l failed (rc $rc; want $n tests run or skipped, 0 failed; got ${got:-none} run, $skips skipped)"
    [ "$skips" = 0 ] || echo "   $got of $n tests run, $skips skipped by the test binary"
    clean $out/$l.log | grep -aoE 'receiver transaction IDs \((uniform rule|rule)\): RX_CH_[0-9] G [0-9]+ ids [0-9]+ start [0-9]+' |
        sort -u > $out/${l}_ids.txt
    if python3 - $E $s $a $out/${l}_ids.txt <<'PY'
import json, re, sys
e, s, a, f = sys.argv[1:5]
arm = "uniform" if a == "uni" else "rule"
label = "uniform rule" if arm == "uniform" else "rule"
lines = [l.strip() for l in open(f) if l.strip()]
bad = [l for l in lines if f"({label})" not in l]
got = {re.sub(r"^.*\): ", "", l) for l in lines}
want = set(json.load(open(e))[s][arm])
if bad or got != want or not lines:
    print("   ids: want", sorted(want), "got", sorted(got), "other-arm lines", len(bad)); sys.exit(1)
print("   ids as captured: " + " | ".join(sorted(got)))
PY
    then :; else stop "$l: receiver IDs differ from the mock"; fi
}

mapfile -t SUITES_B < <(printf '%s\n' "${SUITES[@]}" | tac)
for P in a b; do
    if [ $P = a ]; then
        for v in "${VAL[@]}"; do arm_run a rule $v; arm_run a uni $v; done
        order=("${SUITES[@]}"); arms="rule uni"
    else
        order=("${SUITES_B[@]}"); arms="uni rule"
    fi
    for s in "${order[@]}"; do for a in $arms; do arm_run $P $a $s; done; done
done
echo "ALL_DONE trid ${SUITES[*]} (job at ${SECONDS} s)"
