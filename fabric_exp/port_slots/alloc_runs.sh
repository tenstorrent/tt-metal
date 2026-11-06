#!/bin/bash
# The staged slot allocator's device A/B on one P150 LoudBox (an lb2 job; never a09u32; SLOT_ALLOCATOR.md section 12.14,
# user 2026-11-05: only the configs whose slot rows change, plus the 2D mesh, at 4352 and 3072 B; no census, no 8704 B).
#   STAGED  the port-alloc build (c53b5a5cbbd + alloc_tier_switch.patch, never committed) as is: the staged allocator.
#           The A/B harness only (local, never committed, both arms): fabric_exp/streams/port/test_infra_6dad.patch (the
#           study's test infra: inter-mesh bandwidth profiling) and adapter_vc2_credits_6dad.patch (VC2 benchmark senders).
#   TIER    the same build with TT_FABRIC_EXP_TIER_SLOTS=1: main's tier table
# Router and host: built in /home/aagarwal/tt-metal-port-alloc, staged lean at PORT_TREE_SRC (a directory, or one
# .tar.zst file); this job copies or unpacks it to the same path on the node (the binaries' RUNPATH) and checks the md5s
# and the commit.
# Pass a: the data-checked 1-hop rows on STAGED first (pval2d_4352, pval2d_3072, pval1d_3072), then per suite STAGED then
# TIER; pass b: suites reversed, TIER then STAGED. One kernel cache per arm, fresh at the start of the job.
# Each run: the port's own settings (fast dispatch, TT_METAL_CLEAR_L1=1, profiler off, BW telemetry off, compile commands
# logged), TT_FABRIC_EXP_LOG_SLOTS=1, the suite's env (VC2 on for vc2_*) and the arm's. Checked: "Tests: N executed, 0
# failed" from every rank (vc1_*: two ranks through tt-run), and the slot rows the run logged against the mock vet's for
# that suite (yamls/alloc_expected_rows.json, alloc_mock_vet.sh): every logged row is one the vet logged for this arm, and
# every vetted row that tells this arm from the other is logged. A failure, a row mismatch or a
# hang (past the timeout: killed, captured with hang_capture.py into results/<label>_hangcap) stops the job.
#   alloc_runs.sh [SUITE...]     (default: every suite of yamls/alloc_manifest.json, 4352 B first)
#     PA_VAL="<data-checked rows>" replaces the pass-a validation rows (default pval2d_4352 pval2d_3072 pval1d_3072)
#     labels ${PA_PREFIX:-pa}<a|b>_<staged|tier>_<suite>: results/<label>.log, .csv, _tests.csv
#   DRY=1 alloc_runs.sh ...      prints the runs only (no copy, no device)
cd /home/aagarwal/tt-metal || exit 1
CON=fabric_exp/consolidation
out=fabric_exp/results
D=fabric_exp/port_slots
M=$D/yamls/alloc_manifest.json
E=$D/yamls/alloc_expected_rows.json
SRC=${PORT_TREE_SRC:-/data/aagarwal/lb2/trees/port_alloc_c53bh.tar.zst}
PT=/home/aagarwal/tt-metal-port-alloc
WANT_COMMIT=${PORT_COMMIT_WANT:-c53b5a5cbbd}
PY=/home/aagarwal/tt-metal/python_env/bin/python
RB=$PWD/fabric_exp/port_tune/cov/vc1/p150x8_2x2_split_rank_bindings.yaml
DRY=${DRY:-0}
SUITES=(${*:-2d_4352 1d_4352 ring_4352 vc1_4352 vc2_4352 ne_4352 2d_3072 1d_3072 ring_3072 vc1_3072 vc2_3072 ne_3072})
VAL=(${PA_VAL:-pval2d_4352 pval2d_3072 pval1d_3072})  # data-checked rows (yamls/manifest.json), staged arm, pass a first
for f in $M $E $RB; do [ -f $f ] || { echo "no $f"; exit 1; }; done
mf() { python3 -c "import json,sys; print(json.load(open('$M'))[sys.argv[1]][sys.argv[2]])" $1 $2; }
for s in "${SUITES[@]}"; do
    [ -f "$(mf $s yaml)" ] || { echo "no yaml for $s"; exit 1; }
    python3 -c "import json,sys; e=json.load(open('$E'))[sys.argv[1]]; assert e['staged'] and e['tier']" $s ||
        { echo "no expected rows for $s (alloc_mock_vet.sh)"; exit 1; }
done
if [ "$DRY" != 1 ]; then
    [ "$(hostname)" != bh-lb-120-a09u32 ] || { echo "!!! a09u32 holds the build worktree at $PT: run this on an lb2 node"; exit 1; }
    source $CON/vc1_capture_lib.sh
    mkdir -p $PT $out
    if [[ $SRC == *.tar.zst ]]; then
        # A one-file stage (2026-11-05: /data short of inodes): unpacked fresh in place of the node's copy.
        rm -rf "${PT:?}.new" && mkdir -p $PT.new && tar -I zstd -xf $SRC -C $PT.new || { echo "!!! unpack of $SRC failed"; exit 1; }
        rm -rf "${PT:?}" && mv $PT.new $PT || { echo "!!! could not put the unpacked tree at $PT"; exit 1; }
    else
        rsync -a --delete --exclude=/generated/ $SRC/ $PT/ || { echo "!!! copy of $SRC failed"; exit 1; }
    fi
    (cd $PT && md5sum -c --quiet MD5SUMS) || { echo "!!! the port tree's binaries differ from MD5SUMS"; exit 1; }
    case $(cat $PT/PORT_COMMIT) in $WANT_COMMIT*) ;; *) echo "!!! port tree commit $(cat $PT/PORT_COMMIT), want $WANT_COMMIT"; exit 1 ;; esac
    echo "port tree $(cat $PT/PORT_COMMIT) + alloc_tier_switch.patch at $PT ($(du -sh $PT | cut -f1))"
fi
clean() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
CS=/home/aagarwal/.tt-metal-cache-palloc-staged; CT=/home/aagarwal/.tt-metal-cache-palloc-tier
[ "$DRY" = 1 ] || { rm -rf "${CS:?}" "${CT:?}"; mkdir -p $CS $CT; }

run_port() {  # label yaml ranks timeout ENV...: one run, polled; 2 on a hang (captured), 1 on a failure, else 0
    local label=$1 yaml=$2 ranks=$3 to=$4 t0=$SECONDS child own_sid try st hung=0 rc res lib=$PT/build_Release cmd
    shift 4
    cmd=($lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "$PWD/$yaml")
    [ $ranks = 2 ] && cmd=($PY $PT/ttnn/ttnn/distributed/ttrun.py --rank-binding $RB "${cmd[@]}")
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
    local P=$1 a=$2 s=$3 l y n ranks to env=() got
    l=${PA_PREFIX:-pa}${P}_${a}_$s
    if [[ $s == pv* ]]; then
        y=$(python3 -c "import json,sys; print(json.load(open('$D/yamls/manifest.json'))[sys.argv[1]]['yaml'])" $s)
        n=$(python3 -c "import json,sys; print(json.load(open('$D/yamls/manifest.json'))[sys.argv[1]]['tests'])" $s)
        ranks=1; to=300
    else
        y=$(mf $s yaml); n=$(mf $s tests); ranks=$(mf $s ranks); read -ra env <<< "$(mf $s env)"
        to=600; [ $ranks = 2 ] && to=900
    fi
    [ $a = tier ] && env+=(TT_FABRIC_EXP_TIER_SLOTS=1)
    if [ $a = tier ]; then C=$CT; else C=$CS; fi
    echo "##### $l ($n tests x$ranks; job at ${SECONDS} s)"
    run_port $l $y $ranks $to "${env[@]}"; local rc=$?
    [ "$DRY" = 1 ] && return 0
    [ $rc = 2 ] && stop "$l: hang (captured)"
    got=$(clean $out/$l.log | grep -aoE "Tests: $n executed, 0 failed" | wc -l)
    [ "$got" -ge $ranks ] || stop "$l failed (rc $rc; want '$n executed, 0 failed' from $ranks rank(s), got $got)"
    [[ $s == pv* ]] && return 0
    clean $out/$l.log | grep -aoE 'VC[0-9] sender slots \{[^}]*\} receiver slots \{[^}]*\} \(channel buffer [0-9]+ B\)' |
        sort -u > $out/${l}_rows.txt
    if python3 - $E $s $a $out/${l}_rows.txt <<'PY'
import json, sys
e, s, a, f = sys.argv[1:5]
exp = json.load(open(e))[s]
want = set(exp[a]); other = set().union(*(set(v) for k, v in exp.items() if k != a))
got = {l.strip() for l in open(f) if l.strip()}
# every logged row is one the mock vet logged for this arm, and every row that tells this arm from the other is logged
extra, missing = got - want, (want - other) - got
if extra or missing:
    print("   rows: not vetted for this arm", sorted(extra), "; missing", sorted(missing)); sys.exit(1)
print("   rows as vetted: " + " | ".join(sorted(x.split(" (channel")[0] for x in got)))
PY
    then :; else stop "$l: slot rows differ from the mock vet"; fi
}

mapfile -t SUITES_B < <(printf '%s\n' "${SUITES[@]}" | tac)
for P in a b; do
    if [ $P = a ]; then
        for v in "${VAL[@]}"; do arm_run a staged $v; done
        order=("${SUITES[@]}"); arms="staged tier"
    else
        order=("${SUITES_B[@]}"); arms="tier staged"
    fi
    for s in "${order[@]}"; do for a in $arms; do arm_run $P $a $s; done; done
done
echo "ALL_DONE alloc ${SUITES[*]} (job at ${SECONDS} s)"
