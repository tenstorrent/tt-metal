#!/bin/bash
# PR perf refresh on one P150 LoudBox (an lb2 job; never a09u32), user 2026-11-05: main as shipped against the port tip,
# telemetry off for both, every suite of the group on both routers in this job, in two passes: pass a in suite order with
# main first, pass b in reversed suite order with the port first.
#   main   d2bd12fcf1f (tt-metal-main, main's router as shipped: kernel v1, no study flags, firmware keepalive)
#   port   e7fe1899049 (origin/aagarwal/fabric-router-mainline tip)
#   Both trees carry only fabric_exp/port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch (the stock
#   test_tt_fabric ignores TT_FABRIC_NO_BW_TELEMETRY). Staged by stage_pr_tree.sh as one file each in
#   /data/aagarwal/lb2/trees; this job unpacks them at their build paths (the binaries' RUNPATH) and checks sha256,
#   MD5SUMS and COMMIT. A path that holds something this harness didn't put there (no .lb2_pr_owned) is never touched.
# Groups (report/data/bandwidth_p150.json's census suites; stream mixes and bursts need the study's test infra and are
# left out):
#   two  two ERISCs: 4352 (exp_2d_4352), 1d (exp_1d_4352: lines, half ring, NeighborExchange), ring (v4/exp_ring),
#        torusy (v4/exp_torus_y)
#   one  one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1): 4352_1e, 1d_1e, rse (pr_ring_se: v4/exp_ring_se + 4096 B);
#        mux (pr_mux: v4/exp_mux + 4096 B); trim (exp_2d_trim_4352: capture, then replay)
#   k4   4096 B: 2d_4096, 1d_4096, ring_4096 (pr_*_4096: the two-ERISC suites at 4096 B); sweep (pr_size_sweep: Mesh /
#        Linear 1-hop E/W write and NeighborExchange write, 256-4352 B)
#   gapa the review page's gaps (coordinator, 2026-11-05; mk_pr_gap_yamls.py), two ERISCs: ptype (pr_ptype_sweep: 2D 1-hop
#        E/W scatter_write and fused_atomic_inc at 256-4352 B, atomic_inc on 1-hop E/W and N/S, 2 hops merging with a
#        1-hop flow)
#   gapb txw24 (pr_txy_wrap_2x4: XY torus 2x4, 1 hop E/W across the wrap), txw42 (pr_txy_wrap_4x2: XY torus 4x2 with the
#        study descriptor torus/p150_x8_4x2_torus_xy_mgd.textproto, 1 hop N/S across the wrap), ring1 (pr_ring1_sweep:
#        ring 1 hop E/W), write, 256-4352 B. Torus runs record in results/<label>_axes.txt the control plane's derived
#        axes when logged (want 2x4 Y=LINE X=RING, 4x2 Y=RING X=LINE; a mismatch is reported, the rows stay) and the
#        routers' ENABLE_DEADLOCK_AVOIDANCE per direction from the kernel cache.
#   lat  lat (fabric_exp/exp_latency.yaml as is: one-packet round trips, 1-hop E/W and N/S, 3-hop E/W, write and
#        fused_atomic_inc, 16 and 4096 B; the same-job set of report/data/latency_samejob.json). Per-sample results:
#        results/<label>_latency.csv (the test's latency_results_blackhole.csv).
# Each run: the port library's run_port (fast dispatch, TT_METAL_CLEAR_L1=1, profiler off, TT_FABRIC_NO_BW_TELEMETRY=1,
# compile commands logged), one fresh kernel cache per router and pass. Labels cen<PASS>_<main|port>_<suite> with PASS
# ${PR_PASS:-pr1}a / b: results/<label>.log, .csv (summary), _tests.csv (census_report.py / report builders' format).
# A hang (past the timeout: killed, captured into results/<label>_hangcap) stops the job; a failed run is reported and
# the job goes on.
#   pr_perf_runs.sh GROUP...        DRY=1: print the runs only (no unpack, no device)
cd /home/aagarwal/tt-metal || exit 1
CON=fabric_exp/consolidation
out=fabric_exp/results
D=fabric_exp/pr_perf
DRY=${DRY:-0}
P=${PR_PASS:-pr1}
G=/data/aagarwal/lb2/trees
declare -A TREE=([main]=/home/aagarwal/tt-metal-main [port]=/home/aagarwal/tt-metal-port-pr)
declare -A TAR=([main]=$G/pr_main_d2bd.tar.zst [port]=$G/pr_port_e7fe.tar.zst)
declare -A WANT=([main]=d2bd12fcf1f [port]=e7fe1899049)
E1="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
# suite -> yaml timeout env
suite_of() {
    case $1 in
        4352) echo "fabric_exp/exp_2d_4352.yaml 300" ;; 1d) echo "fabric_exp/exp_1d_4352.yaml 300" ;;
        ring) echo "fabric_exp/v4/exp_ring.yaml 240" ;; torusy) echo "fabric_exp/v4/exp_torus_y.yaml 240" ;;
        4352_1e) echo "fabric_exp/exp_2d_4352.yaml 300 $E1" ;; 1d_1e) echo "fabric_exp/exp_1d_4352.yaml 300 $E1" ;;
        rse) echo "$D/pr_ring_se.yaml 300 $E1" ;; mux) echo "$D/pr_mux.yaml 420" ;;
        trim) echo "fabric_exp/exp_2d_trim_4352.yaml 300" ;;
        2d_4096) echo "$D/pr_2d_4096.yaml 300" ;; 1d_4096) echo "$D/pr_1d_4096.yaml 300" ;;
        ring_4096) echo "$D/pr_ring_4096.yaml 240" ;; sweep) echo "$D/pr_size_sweep.yaml 300" ;;
        ptype) echo "$D/pr_ptype_sweep.yaml 300" ;; ring1) echo "$D/pr_ring1_sweep.yaml 240" ;;
        txw24) echo "$D/pr_txy_wrap_2x4.yaml 300" ;;
        lat) echo "fabric_exp/exp_latency.yaml 900" ;;
        txw42) echo "$D/pr_txy_wrap_4x2.yaml 300 TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto" ;;
        *) return 1 ;;
    esac
}
SUITES=()
for g in "$@"; do
    case $g in
        two) SUITES+=(4352 1d ring torusy) ;; one) SUITES+=(4352_1e 1d_1e rse mux trim) ;;
        k4) SUITES+=(2d_4096 1d_4096 ring_4096 sweep) ;; gapa) SUITES+=(ptype) ;; gapb) SUITES+=(txw24 txw42 ring1) ;; lat) SUITES+=(lat) ;;
        *) echo "usage: [DRY=1] pr_perf_runs.sh two|one|k4|gapa|gapb|lat..."; exit 1 ;;
    esac
done
[ ${#SUITES[@]} -gt 0 ] || { echo "usage: [DRY=1] pr_perf_runs.sh two|one|k4|gapa|gapb|lat..."; exit 1; }
for s in "${SUITES[@]}"; do y=$(suite_of $s | cut -d' ' -f1); [ -f $y ] || { echo "!!! no $y"; exit 1; }; done
for r in main port; do [ -f ${TAR[$r]} ] && [ -f ${TAR[$r]}.sha256 ] || { echo "!!! no ${TAR[$r]}"; exit 1; }; done

unpack() {  # router
    local r=$1 T=${TREE[$1]} f=${TAR[$1]}
    if [ -e $T ] && [ ! -e $T/.lb2_pr_owned ]; then echo "!!! $T exists and is not this harness's: not touched"; exit 1; fi
    (cd $(dirname $f) && sha256sum -c --quiet $(basename $f).sha256) || { echo "!!! $f fails its sha256"; exit 1; }
    rm -rf "${T:?}.new" && mkdir -p $T.new && tar -I zstd -xf $f -C $T.new || { echo "!!! unpack of $f failed"; exit 1; }
    (cd $T.new && md5sum -c --quiet MD5SUMS) || { echo "!!! $r: binaries differ from MD5SUMS"; exit 1; }
    case $(cat $T.new/COMMIT) in ${WANT[$r]}*) ;; *) echo "!!! $r tree at $(cat $T.new/COMMIT), want ${WANT[$r]}"; exit 1 ;; esac
    ln -sfn build_Release $T.new/build
    echo "created by $(hostname) job ${SLURM_JOB_ID:-?} $(date -u +%FT%TZ) from $f" > $T.new/.lb2_pr_owned
    rm -rf "${T:?}" && mv $T.new $T || { echo "!!! could not put the tree at $T"; exit 1; }
    echo "$r: $(cat $T/COMMIT | cut -c1-11) + $(grep -c '^[+-][^+-]' $T/PATCH) patch lines at $T ($(du -sh $T | cut -f1))"
}
cache() { echo /home/aagarwal/.tt-metal-cache-pr-$1-$2; }

if [ "$DRY" != 1 ]; then
    [ "$(hostname)" != bh-lb-120-a09u32 ] || { echo "!!! a09u32 holds the build worktrees: run this on an lb2 node"; exit 1; }
    source $CON/vc1_capture_lib.sh
    source fabric_exp/port_slots/port_lib.sh
    mkdir -p $out
    for r in main port; do unpack $r; done
    for pass in a b; do for r in main port; do c=$(cache $r $pass); rm -rf "${c:?}"; mkdir -p $c; done; done
fi
one() {  # pass router suite
    local pass=$1 r=$2 s=$3 y t env label rc
    read -r y t env <<< "$(suite_of $s)"
    label=cen${P}${pass}_${r}_$s
    if [ "$DRY" = 1 ]; then echo "   $label: ${TREE[$r]} $y (timeout $t) ${env:-}"; return 0; fi
    PT=${TREE[$r]} C=$(cache $r $pass)
    rm -f $PT/generated/fabric/latency_results_blackhole.csv "$out/${label}_latency.csv"
    run_port $label $y $t $env; rc=$?
    [ -f $PT/generated/fabric/latency_results_blackhole.csv ] && cp $PT/generated/fabric/latency_results_blackhole.csv "$out/${label}_latency.csv"
    case $s in txw24 | txw42)  # realized torus: the derived axes (logged only when a multicast map is built) and the
        # routers' deadlock-avoidance arg per direction from the kernel cache (on for every direction on a torus)
        local want=$([ $s = txw24 ] && echo "Y=LINE X=RING" || echo "Y=RING X=LINE") axes da
        axes=$(sed 's/\x1b\[[0-9;]*m//g' $out/$label.log | grep -aoE "axis topologies derived as Y=[A-Z]+ X=[A-Z]+" | sort -u | sed 's/.* as //' | tr '\n' ' ' | sed 's/ $//')
        da=$(python3 - $C <<'PY'
import collections, glob, re, sys
da = collections.defaultdict(collections.Counter)
for f in glob.glob(sys.argv[1] + "/*/kernels/*router*/*/named_ct_arg_map_generated.h"):
    s = open(f).read()
    d, e = re.search(r'\{"MY_DIRECTION",(\d+)\}', s), re.search(r'\{"ENABLE_DEADLOCK_AVOIDANCE",(\d+)\}', s)
    if d and e:
        da["EWNS"[int(d.group(1))]][e.group(1)] += 1
print(", ".join(f"{k}: {dict(v)}" for k, v in sorted(da.items())) or "not found")
PY
)
        printf 'axes=%s want=%s\ndeadlock_avoidance=%s\n' "${axes:-not logged}" "$want" "$da" > $out/${label}_axes.txt
        echo "   $label: derived axes ${axes:-not logged} (want $want); deadlock avoidance by direction: $da"
        [ -z "$axes" ] || [ "$axes" = "$want" ] || echo "!!! $label: derived axes '$axes', want '$want'" ;;
    esac
    [ $rc = 2 ] && { echo "!!! $label hung (captured): stopping"; exit 1; }
    [ $rc = 0 ] || echo "!!! $label failed (rc $rc): going on"
    return 0
}
echo "##### pass a: ${SUITES[*]}, main then port (job at ${SECONDS} s)"
for s in "${SUITES[@]}"; do one a main $s; one a port $s; done
echo "##### pass b: reversed, port then main (job at ${SECONDS} s)"
for ((i = ${#SUITES[@]} - 1; i >= 0; i--)); do one b port ${SUITES[$i]}; one b main ${SUITES[$i]}; done
echo "ALL_DONE (job at ${SECONDS} s)"
