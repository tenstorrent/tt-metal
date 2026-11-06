#!/bin/bash
# Report data refresh at 400G, stock test infra (one lb2 job; never a09u32; user 2026-11-05): main as shipped against the
# port (PR tip) on one P150x8 LoudBox, same job, two passes in reversed router order, telemetry off for both.
#   main    d2bd12fcf1f + the test harness's TT_FABRIC_NO_BW_TELEMETRY switch (/data/aagarwal/lb2/trees/pr_main_d2bd)
#   mainka  the same tree with fabric_exp/report/refresh/main_ka1000_kernel.patch on main's router kernel (JIT source
#           only; ERISC0 sets ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT to 1000 cycles on both TX queues, as the study's
#           TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 did): main's router at the study keepalive, for the keepalive table only
#   port    d51e42cd78a (origin/aagarwal/fabric-router-mainline tip) + the same switch (/data/aagarwal/lb2/trees/rf_port_d51e)
# Pass a: a data-checked validation run per router (port_slots/yamls/pval2d_4352, pval1d_4352), then per suite main,
# (mainka,) port; then the breakdown rows main, port. Pass b: breakdown port, main; suites reversed, port, (mainka,) main.
#   suites (fabric_exp/report/refresh/yamls, mk_rf_yamls.py): rf_2d rf_1d rf_ring rf_torusy rf_txy24 rf_txy42 (size
#   sweep, 256-4352 B), rf_ka and fabric_exp/exp_2d_trim_4352.yaml (keepalive table; three arms)
#   breakdown (report/BREAKDOWN_PLAN.md): pcs3/compile2d.yaml, then pcs3/ew4352.yaml, rf_bd_ew1024.yaml and
#   consolidation/bd_rows/mcast.yaml each with pc_sampler.py beside it (debug-bus PC reads, both ERISCs, every chip);
#   own kernel caches, whose router ELFs and the router sources are kept as results (rfbd_*_routers.tgz, _src.tgz)
# Labels rf1<a|b>_<arm>_<suite>; breakdown rf1<a|b>_<arm>_bd_<row> (+ _samples.{bin,json}). A hang (past the timeout:
# killed, captured into results/<label>_hangcap) stops the job; a failed run is reported and the job goes on.
#   rf_lb_stock.sh          DRY=1: print the runs only
cd /home/aagarwal/tt-metal || exit 1
out=fabric_exp/results
CON=fabric_exp/consolidation
DRY=${DRY:-0}
Y=fabric_exp/report/refresh/yamls
P=rf1
# main unpacks at its own path, not the PR harness's /home/aagarwal/tt-metal-main (another harness may hold that path on a
# node, lb2 180460); LD_LIBRARY_PATH (over the binaries' RUNPATH) and TT_METAL_HOME/RUNTIME_ROOT point every run at its tree
declare -A TREE=([main]=/home/aagarwal/tt-metal-rf-main [mainka]=/home/aagarwal/tt-metal-rf-mainka [port]=/home/aagarwal/tt-metal-rf-port)
SUITES=(2d 1d ring torusy txy24 txy42 ka trim)
suite_of() {
    case $1 in
        2d | 1d | ring | torusy | txy24) echo "$Y/rf_$1.yaml 600" ;;
        txy42) echo "$Y/rf_txy42.yaml 600 TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto" ;;
        ka) echo "$Y/rf_ka.yaml 300" ;; trim) echo "fabric_exp/exp_2d_trim_4352.yaml 400" ;;
        val2d) echo "fabric_exp/port_slots/yamls/pval2d_4352.yaml 300" ;; val1d) echo "fabric_exp/port_slots/yamls/pval1d_4352.yaml 300" ;;
    esac
}
arms_of() { case $1 in ka | trim) echo "main mainka port" ;; *) echo "main port" ;; esac; }
for s in "${SUITES[@]}" val2d val1d; do y=$(suite_of $s | cut -d' ' -f1); [ -f $y ] || { echo "!!! no $y"; exit 1; }; done
for f in fabric_exp/pcs3/compile2d.yaml fabric_exp/pcs3/ew4352.yaml $Y/rf_bd_ew1024.yaml $CON/bd_rows/mcast.yaml \
    fabric_exp/report/refresh/main_ka1000_kernel.patch; do [ -f $f ] || { echo "!!! no $f"; exit 1; }; done
source fabric_exp/report/refresh/rf_lib.sh 2>/dev/null || [ "$DRY" = 1 ] || { echo "!!! rf_lib.sh"; exit 1; }
cache() { echo /home/aagarwal/.tt-metal-cache-rf-$1-$2; }

if [ "$DRY" != 1 ]; then
    [ "$(hostname)" != bh-lb-120-a09u32 ] || { echo "!!! a09u32 holds the build worktrees: run this on an lb2 node"; exit 1; }
    mkdir -p $out
    rf_unpack main ${TREE[main]} $G/pr_main_d2bd.tar.zst d2bd12fcf1f
    rf_unpack port ${TREE[port]} $G/rf_port_d51e.tar.zst d51e42cd78a
    # mainka: hard links of main's tree, the router kernel replaced by a patched copy (main's file untouched)
    K=${TREE[mainka]} F=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
    [ ! -e $K ] || [ -e $K/.lb2_pr_owned ] || { echo "!!! $K exists and is not this harness's"; exit 1; }
    rm -rf "${K:?}" && cp -al ${TREE[main]} $K && rm -f "${K:?}/${F:?}" && cp ${TREE[main]}/$F $K/$F &&
        patch -s -p1 -d $K < fabric_exp/report/refresh/main_ka1000_kernel.patch && rm -rf "${K:?}/generated" && mkdir -p $K/generated ||
        { echo "!!! mainka tree failed"; exit 1; }
    echo "mainka: $K = main's tree with main_ka1000_kernel.patch ($(grep -c refresh_seq_update_timeout $K/$F) refs; main's" \
        "file $(grep -c refresh_seq_update_timeout ${TREE[main]}/$F) refs)"
    for pass in a b; do for r in main mainka port; do for k in sw bd; do c=$(cache $r-$k $pass); rm -rf "${c:?}"; mkdir -p $c; done; done; done
fi

one() {  # pass arm suite
    local pass=$1 r=$2 s=$3 y t env label rc
    read -r y t env <<< "$(suite_of $s)"
    label=${P}${pass}_${r}_$s
    if [ "$DRY" = 1 ]; then echo "   $label: ${TREE[$r]} $y (timeout $t) ${env:-}"; return 0; fi
    rf_run $label ${TREE[$r]} $(cache $r-sw $pass) $y $t $env; rc=$?
    if [ $r = mainka ]; then  # the patched kernel is what compiled (the JIT reads the tree's own sources)
        local n; n=$(sed 's/\x1b\[[0-9;]*m//g' $out/$label.log | grep -a 'g++ compile cmd' | grep -a 'fabric_erisc_router' | grep -c 'tt-metal-rf-mainka/')
        echo "   $label: $n router compile commands from the mainka tree"
    fi
    case $s in txy24 | txy42)
        local want=$([ $s = txy24 ] && echo "Y=LINE X=RING" || echo "Y=RING X=LINE") axes
        axes=$(sed 's/\x1b\[[0-9;]*m//g' $out/$label.log | grep -aoE "axis topologies derived as Y=[A-Z]+ X=[A-Z]+" | sort -u | sed 's/.* as //' | tr '\n' ' ' | sed 's/ $//')
        printf 'axes=%s want=%s\n' "${axes:-not logged}" "$want" > $out/${label}_axes.txt
        echo "   $label: derived axes ${axes:-not logged} (want $want)" ;;
    esac
    [ $rc = 2 ] && { echo "!!! $label hung (captured): stopping"; exit 1; }
    [ $rc = 0 ] || echo "!!! $label failed (rc $rc): going on"
    return 0
}
bd() {  # pass arm: the breakdown rows with the sampler beside each
    local pass=$1 r=$2 c lab row y rp rc
    c=$(cache $r-bd $pass)
    for row in compile ew4352 ew1024 mcast; do
        case $row in compile) y=fabric_exp/pcs3/compile2d.yaml ;; ew4352) y=fabric_exp/pcs3/ew4352.yaml ;;
            ew1024) y=$Y/rf_bd_ew1024.yaml ;; mcast) y=$CON/bd_rows/mcast.yaml ;; esac
        lab=${P}${pass}_${r}_bd_$row
        if [ "$DRY" = 1 ]; then echo "   $lab: ${TREE[$r]} $y$([ $row = compile ] || echo ', sampled')"; continue; fi
        if [ $row = compile ]; then rf_run $lab ${TREE[$r]} $c $y 600; rc=$?
        else
            rm -f $out/${lab}_samples.bin $out/${lab}_samples.json
            ( trap - EXIT; rf_run $lab ${TREE[$r]} $c $y 900 ) & rp=$!
            timeout 400 $STUDY_PY $CON/pc_sampler.py $out/${lab}_samples --start-log $out/$lab.log --start-delay-s 2 \
                --max-s 60 > $out/${lab}_sampler.log 2>&1
            echo "   sampler: exit $?; $(grep -E '^sampler: ' $out/${lab}_sampler.log | tail -2 | tr '\n' ' ')"
            wait $rp; rc=$?
        fi
        awk -F, -v l=$lab 'NR>1 {printf "   %s: %s %s B %.2f cycles/packet\n", l, $1, $7, 1.35e9/$10}' $out/$lab.csv 2>/dev/null
        [ $rc = 2 ] && { echo "!!! $lab hung (captured): stopping"; exit 1; }
        [ $rc = 0 ] || echo "!!! $lab failed (rc $rc): going on"
    done
    [ "$DRY" = 1 ] || { rf_keep_cache $c ${P}${pass}_${r}_bd; [ $pass = a ] && rf_keep_src ${TREE[$r]} ${P}_${r}_bd; }
}
echo "##### pass a: validation, then ${SUITES[*]}, main (mainka) port; breakdown main, port (job at ${SECONDS} s)"
for v in val2d val1d; do one a main $v; one a port $v; done
for s in "${SUITES[@]}"; do for r in $(arms_of $s); do one a $r $s; done; done
bd a main; bd a port
echo "##### pass b: breakdown port, main; suites reversed, port (mainka) main (job at ${SECONDS} s)"
bd b port; bd b main
for ((i = ${#SUITES[@]} - 1; i >= 0; i--)); do for r in $(arms_of ${SUITES[$i]} | tr ' ' '\n' | tac); do one b $r ${SUITES[$i]}; done; done
[ "$DRY" = 1 ] || rm -rf "${TREE[mainka]:?}"
echo "ALL_DONE (job at ${SECONDS} s)"
