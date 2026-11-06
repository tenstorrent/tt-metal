#!/bin/bash
# Receiver tuning sweep on the port (RECEIVER_TUNING.md), on one P150 LoudBox (an lb2 job; never a09u32): the arms of
# the given sets in two passes, pass a in table order and pass b reversed, all in this job (one host per comparison).
# Router and host: the port, commit 22cdb1e55c3 plus fabric_exp/port_tune/port_tune.patch (never committed), built in
# /home/aagarwal/tt-metal-port-tune and staged lean at PORT_TREE_SRC; this job copies it to the same path on the node
# (the binaries' RUNPATH) and checks the md5s.
#   Sets (tune_arms below):
#     idle   TT_FABRIC_EXP_IDLE_PASSES 1, 2, 4, 8 at TT_FABRIC_EXP_PROOF_CHECKS 32
#     proof  TT_FABRIC_EXP_PROOF_CHECKS 8, 32, 128 at TT_FABRIC_EXP_IDLE_PASSES 4
#     pert   idle passes 1, 2, 4, 8, each with no delay, TT_FABRIC_EXP_TX_DELAY 8 / 16 / 32 (a spin after each packet
#            sent) and a receiver-pass spin of 8 / 16 / 32 (TT_FABRIC_EXP_DRISC_DELAY high 16 bits): pert rows only
#     census base (4, 32) and cand (TUNE_CAND="IDLE,PROOF"): every suite whose router images the arms change (the
#            offline gate): the idle / proof suites, one ERISC (2d_1e, 1d_1e), Tensix mux, VC2 on (vc2ew, vc02ew,
#            vc02bew). Rings and tori are left out: their images are identical across arms.
#     idlealloc  idle passes 2, 4, 8 at proof checks 32 on 2d, vc1, vc1i: the allocator follow-up (PORT_TREE /
#            PORT_TREE_SRC point at the allocator-based tree)
#     idlesw idle passes 4 (default), 8, 16, 32, 64 at proof checks 32 on e7fe1899049 (RECEIVER_TUNING.md section 8):
#            2d 1d burst mix vc1 nx4096 ring lat; PORT_TREE=/home/aagarwal/tt-metal-port-nxab with
#            PORT_TREE_TAR=/data/aagarwal/lb2/trees/port_tune_e7fe.tar.zst (one staged file, unpacked at PORT_TREE)
#   SUITES (default "2d 1d mix burst burst612 vc1 vc1i"; pert: "pert") limits the suites of idle / proof.
# Each arm: its own fresh kernel cache per pass; in pass a a data-checked run first (val, val1d; not for pert). Each run:
# the port's own settings (fast dispatch, TT_METAL_CLEAR_L1=1, profiler off, BW telemetry off, compile commands
# logged) and the arm's env; checked: "Tests: N executed, 0 failed" (both ranks for vc1 / vc1i) and the router
# defines the host logged ("port_tune: router define ..."). A failure, a missing define or a hang (past the timeout:
# killed, captured with hang_capture.py into results/<label>_hangcap) stops the job.
#   tune_runs.sh SET...     labels ${PT_PREFIX:-pt}<a|b>_<set>_<arm>_<suite>: results/<label>.log, .csv, _tests.csv
#   DRY=1 tune_runs.sh ...  prints the runs only (no copy, no device)
cd /home/aagarwal/tt-metal || exit 1
CON=fabric_exp/consolidation
out=fabric_exp/results
D=fabric_exp/port_tune
[ $# -gt 0 ] || { echo "usage: [DRY=1] [SUITES=...] tune_runs.sh idle|proof|pert..."; exit 1; }
SRC=${PORT_TREE_SRC:-/data/aagarwal/lb2/trees/port_tune_22cd}
# PORT_TREE: where the staged tree goes on the node; it must be the path the tree was built at (the binaries' RUNPATH),
# e.g. /home/aagarwal/tt-metal-port-tune-alloc for the allocator follow-up (RECEIVER_TUNING.md §6).
PT=${PORT_TREE:-/home/aagarwal/tt-metal-port-tune}
PY=/home/aagarwal/tt-metal/python_env/bin/python
RB=$PWD/$D/cov/vc1/p150x8_2x2_split_rank_bindings.yaml
DRY=${DRY:-0}
if [ "$DRY" != 1 ]; then
    [ "$(hostname)" != bh-lb-120-a09u32 ] || { echo "!!! a09u32 holds the build worktree at $PT: run this on an lb2 node"; exit 1; }
    source $CON/vc1_capture_lib.sh
    mkdir -p $out
    if [ -n "${PORT_TREE_TAR:-}" ]; then  # one staged file (stage as pr_perf/stage_pr_tree.sh; /data is short of inodes)
        if [ -e $PT ] && [ ! -e $PT/.tune_owned ]; then echo "!!! $PT exists and is not this harness's: not touched"; exit 1; fi
        (cd $(dirname $PORT_TREE_TAR) && sha256sum -c --quiet $(basename $PORT_TREE_TAR).sha256) || { echo "!!! $PORT_TREE_TAR fails its sha256"; exit 1; }
        rm -rf "${PT:?}.new" && mkdir -p $PT.new && tar -I zstd -xf $PORT_TREE_TAR -C $PT.new || { echo "!!! unpack failed"; exit 1; }
        (cd $PT.new && md5sum -c --quiet MD5SUMS) || { echo "!!! the port tree's binaries differ from MD5SUMS"; exit 1; }
        ln -sfn build_Release $PT.new/build && cp $PT.new/COMMIT $PT.new/PORT_COMMIT
        echo "unpacked by $(hostname) job ${SLURM_JOB_ID:-?} from $PORT_TREE_TAR" > $PT.new/.tune_owned
        rm -rf "${PT:?}" && mv $PT.new $PT || { echo "!!! could not put the tree at $PT"; exit 1; }
    else
        mkdir -p $PT
        rsync -a --delete --exclude=/generated/ $SRC/ $PT/ || { echo "!!! copy of $SRC failed"; exit 1; }
        (cd $PT && md5sum -c --quiet MD5SUMS) || { echo "!!! the port tree's binaries differ from MD5SUMS"; exit 1; }
    fi
    echo "port tree $(cat $PT/PORT_COMMIT) + its local patch at $PT ($(du -sh $PT | cut -f1))"
fi

tune_arms() {  # "SET ARM ENV..." per line (ENV: NAME=VALUE words)
    local i c d
    for i in 1 2 4 8; do echo "idle i$i TT_FABRIC_EXP_IDLE_PASSES=$i TT_FABRIC_EXP_PROOF_CHECKS=32"; done
    for c in 8 32 128; do echo "proof c$c TT_FABRIC_EXP_IDLE_PASSES=4 TT_FABRIC_EXP_PROOF_CHECKS=$c"; done
    # the allocator follow-up (RECEIVER_TUNING.md §6): idle passes 2 / 4 / 8 on the staged allocator's slot rows
    for i in 2 4 8; do echo "idlealloc i$i TT_FABRIC_EXP_IDLE_PASSES=$i TT_FABRIC_EXP_PROOF_CHECKS=32"; done
    # the IDLE_PASSES sweep on e7fe1899049 (RECEIVER_TUNING.md section 8): 4 (default) to 64
    for i in 4 8 16 32 64; do echo "idlesw i$i TT_FABRIC_EXP_IDLE_PASSES=$i TT_FABRIC_EXP_PROOF_CHECKS=32"; done
    if [ -n "${TUNE_CAND:-}" ]; then
        echo "census base TT_FABRIC_EXP_IDLE_PASSES=4 TT_FABRIC_EXP_PROOF_CHECKS=32"
        echo "census cand TT_FABRIC_EXP_IDLE_PASSES=${TUNE_CAND%,*} TT_FABRIC_EXP_PROOF_CHECKS=${TUNE_CAND#*,}"
    fi
    for i in 1 2 4 8; do
        echo "pert i${i}_d0 TT_FABRIC_EXP_IDLE_PASSES=$i TT_FABRIC_EXP_PROOF_CHECKS=32"
        for d in 8 16 32; do echo "pert i${i}_tx$d TT_FABRIC_EXP_IDLE_PASSES=$i TT_FABRIC_EXP_PROOF_CHECKS=32 TT_FABRIC_EXP_TX_DELAY=$d"; done
        for d in 8 16 32; do echo "pert i${i}_rp$d TT_FABRIC_EXP_IDLE_PASSES=$i TT_FABRIC_EXP_PROOF_CHECKS=32 TT_FABRIC_EXP_DRISC_DELAY=$((d << 16))"; done
    done
}
tests_of() { python3 -c "import json,sys; print(json.load(open('$D/yamls/manifest.json'))[sys.argv[1]]['tests'])" $1; }
yaml_of() { python3 -c "import json,sys; print(json.load(open('$D/yamls/manifest.json'))[sys.argv[1]]['yaml'])" $1; }
suite_env() { case $1 in 2d_1e|1d_1e) echo TT_METAL_DISABLE_FABRIC_TWO_ERISC=1 ;; vc2ew|vc02ew|vc02bew) echo TT_METAL_ENABLE_FABRIC_VC2=1 ;; esac; }
timeout_of() { case $1 in vc1) echo 600 ;; vc1i) echo 300 ;; burst|mix) echo 420 ;; *) echo 300 ;; esac; }
clean() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }

run_port() {  # label suite timeout ENV...: one run, polled; 2 on a hang (captured), 1 on a failure, else 0
    local label=$1 suite=$2 to=$3 t0=$SECONDS child own_sid try st hung=0 rc res lib=$PT/build_Release yaml cmd
    shift 3
    yaml=$(yaml_of $suite)
    cmd=($lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "$PWD/$yaml")
    case $suite in vc1|vc1i) cmd=($PY $PT/ttnn/ttnn/distributed/ttrun.py --rank-binding $RB "${cmd[@]}") ;; esac
    if [ "$DRY" = 1 ]; then echo "   $label: $* ${cmd[*]} (timeout $to s)"; return 0; fi
    VC1CAP_LABEL=$label
    VC1CAP_PIDS=""
    rm -f $PT/generated/fabric/bandwidth_summary_results_blackhole.csv $PT/generated/fabric/bandwidth_results_blackhole.csv \
        $PT/generated/fabric/latency_results_blackhole.csv "$out/$label.csv" "$out/${label}_tests.csv" "$out/${label}_latency.csv"
    echo "=== $label: $yaml (timeout ${to} s) $*"
    _vc1cap_trap_on
    setsid -w env -C $PT -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PT \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_CACHE=$C \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 "$@" "${cmd[@]}" > "$out/$label.log" 2>&1 < /dev/null &
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
    [ -f $PT/generated/fabric/latency_results_blackhole.csv ] && cp $PT/generated/fabric/latency_results_blackhole.csv "$out/${label}_latency.csv"
    res=$(clean "$out/$label.log" | grep -aoE 'Tests: .*executed, [0-9]+ failed' | sort | uniq -c | tr -s ' \n' ' ')
    echo "$label: ${res:-NO RESULT} (exit $rc, $((SECONDS - t0)) s)"
    clean "$out/$label.log" | grep -aE "FAILED|TT_FATAL|TT_THROW|mismatch" | head -5 | sed 's/^/   | /'
    [ $rc = 0 ]
}

arm_run() {  # pass set arm suite ENV...
    local P=$1 s=$2 a=$3 su=$4 l n want rc ranks=1 e v got
    shift 4
    l=${PT_PREFIX:-pt}${P}_${s}_${a}_$su
    n=$(tests_of $su)
    case $su in vc1|vc1i) ranks=2 ;; esac
    run_port $l $su $(timeout_of $su) "$@" $(suite_env $su); rc=$?
    [ "$DRY" = 1 ] && return 0
    [ $rc = 2 ] && { echo "!!! $l: hang (captured): stopping"; exit 1; }
    want=$(clean $out/$l.log | grep -aoE "Tests: $n executed, 0 failed" | wc -l)
    [ "$want" -ge $ranks ] || { echo "!!! $l failed (rc $rc; want '$n executed, 0 failed' from $ranks rank(s))"; exit 1; }
    for e in "$@"; do  # every arm setting must have reached the router build as a define
        v=${e#TT_FABRIC_EXP_}; v="FABRIC_EXP_${v%%=*}=${e#*=}"
        got=$(clean $out/$l.log | grep -acF "port_tune: router define $v")
        [ "$got" -ge $ranks ] || { echo "!!! $l: define $v not logged ($got of $ranks)"; exit 1; }
    done
}

sets=" $* "
mapfile -t ARMS < <(tune_arms | while read -r s a env; do [[ $sets == *" $s "* ]] && echo "$s $a $env"; done)
[ ${#ARMS[@]} -gt 0 ] || { echo "no arms for sets: $*"; exit 1; }
mapfile -t ARMS_B < <(printf '%s\n' "${ARMS[@]}" | tac)
for P in a b; do
    if [ $P = a ]; then order=("${ARMS[@]}"); else order=("${ARMS_B[@]}"); fi
    for line in "${order[@]}"; do
        read -r s a env <<< "$line"
        read -ra ENVV <<< "$env"
        C=/home/aagarwal/.tt-metal-cache-ptune-$s-$a
        [ "$DRY" = 1 ] || { rm -rf "${C:?}"; mkdir -p $C; }
        echo "##### pass $P $s $a: ${ENVV[*]} (job at ${SECONDS} s)"
        case $s in
            pert) su_list=${SUITES:-pert} ;;
            census) su_list=${SUITES:-2d 1d mix burst burst612 vc1 vc1i 2d_1e 1d_1e mux vc2ew vc02ew vc02bew} ;;
            idlealloc) su_list=${SUITES:-2d vc1 vc1i} ;;
            idlesw) su_list=${SUITES:-2d 1d burst mix vc1 nx4096 ring lat} ;;
            *) su_list=${SUITES:-2d 1d mix burst burst612 vc1 vc1i} ;;
        esac
        if [ $P = a ] && [ $s != pert ]; then
            for su in val val1d; do arm_run $P $s $a $su "${ENVV[@]}"; done
        fi
        for su in $su_list; do arm_run $P $s $a $su "${ENVV[@]}"; done
        [ "$DRY" = 1 ] || rm -rf "${C:?}"
    done
done
echo "ALL_DONE port tune sets $* (job at ${SECONDS} s)"
