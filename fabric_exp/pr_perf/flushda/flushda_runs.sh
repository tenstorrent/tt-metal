#!/bin/bash
# Flushed-atomic check of 3efd636e8df ("one counter check for a flushed atomic on rings and tori too"; coordinator
# 2026-11-05; local, never committed) on one P150 LoudBox (an lb2 job): the patched router (3efd636e8df + the test
# harness's TT_FABRIC_NO_BW_TELEMETRY switch) against the unpatched one (the same tree with d51e42cd78a's
# fabric_edm_packet_transmission.hpp, the commit's only change: d51e_fabric_edm_packet_transmission.hpp beside this
# script), same job, pass a unpatched first, pass b reversed.
#   P  /home/aagarwal/tt-metal-gate-3efd, unpacked from /data/aagarwal/lb2/trees/flushda_3efd.tar.zst (sha256, MD5SUMS)
#   U  /home/aagarwal/tt-metal-gate-3efd-unp: a farm of P (top-level links) with tt_metal/fabric/hw/inc/edm_fabric real
#      and the unpatched header in it; host binaries are P's (the commit changes no host code)
# Suites per arm:
#   data-checked (benchmark off; flushda_val_*.yaml: atomic_inc, fused_atomic_inc and scatter + atomic_inc, flush off
#   and on, all-to-all, unicast and multicast, 1024 / 4096 B): ring, XY torus 2x4, XY torus 4x2 (study MGD), each at
#   two ERISCs and at one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1)
#   perf (benchmark): flushda_bench_{ring,torus,torus4x2}.yaml (one flow: write, fused flush off / on, scatter +
#   atomic_inc flush off / on, 1024 / 4096 B) and the census rows: v4/exp_ring.yaml, v4/exp_torus_y.yaml,
#   torus/torusxy_2x4_sizes.yaml, torus/torusxy_4x2_sizes.yaml (4x2 MGD)
# Each run: run_port (fast dispatch, TT_METAL_CLEAR_L1=1, profiler off, TT_FABRIC_NO_BW_TELEMETRY=1), a kernel cache per
# arm and pass. A hang (past the timeout: killed, captured) stops the job: a hang rules the change out. A failed
# data-checked run stops the job; a failed perf run is reported and the job goes on.
#   flushda_runs.sh        DRY=1: print the runs only
cd /home/aagarwal/tt-metal || exit 1
CON=fabric_exp/consolidation
out=fabric_exp/results
D=fabric_exp/pr_perf/flushda
DRY=${DRY:-0}
P=/home/aagarwal/tt-metal-gate-3efd
U=/home/aagarwal/tt-metal-gate-3efd-unp
TAR=/data/aagarwal/lb2/trees/flushda_3efd.tar.zst
MGD42="TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto"
E1="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
EDM=tt_metal/fabric/hw/inc/edm_fabric
# suite -> yaml timeout kind(val|perf) env...
suite_of() {
    case $1 in
        val_ring) echo "$D/flushda_val_ring.yaml 900 val" ;;
        val_ring_1e) echo "$D/flushda_val_ring.yaml 900 val $E1" ;;
        val_txy24) echo "$D/flushda_val_torus.yaml 900 val" ;;
        val_txy24_1e) echo "$D/flushda_val_torus.yaml 900 val $E1" ;;
        val_txy42) echo "$D/flushda_val_torus.yaml 900 val $MGD42" ;;
        val_txy42_1e) echo "$D/flushda_val_torus.yaml 900 val $E1 $MGD42" ;;
        b_ring) echo "$D/flushda_bench_ring.yaml 300 perf" ;;
        b_txy24) echo "$D/flushda_bench_torus.yaml 300 perf" ;;
        b_txy42) echo "$D/flushda_bench_torus4x2.yaml 300 perf $MGD42" ;;
        c_ring) echo "fabric_exp/v4/exp_ring.yaml 300 perf" ;;
        c_torusy) echo "fabric_exp/v4/exp_torus_y.yaml 300 perf" ;;
        c_txy24) echo "fabric_exp/torus/torusxy_2x4_sizes.yaml 600 perf" ;;
        c_txy42) echo "fabric_exp/torus/torusxy_4x2_sizes.yaml 600 perf $MGD42" ;;
        *) return 1 ;;
    esac
}
SUITES=(${FLUSHDA_SUITES:-val_ring val_ring_1e val_txy24 val_txy24_1e val_txy42 val_txy42_1e b_ring b_txy24 b_txy42 c_ring c_torusy c_txy24 c_txy42})
for s in "${SUITES[@]}"; do y=$(suite_of $s | cut -d' ' -f1); [ -f $y ] || { echo "!!! no $y"; exit 1; }; done
[ -f $D/d51e_fabric_edm_packet_transmission.hpp ] || { echo "!!! no unpatched header"; exit 1; }
if [ "$DRY" != 1 ]; then
    [ "$(hostname)" != bh-lb-120-a09u32 ] || { echo "!!! run this on an lb2 node"; exit 1; }
    source $CON/vc1_capture_lib.sh
    source fabric_exp/port_slots/port_lib.sh
    mkdir -p $out
    if [ -e $P ] && [ ! -e $P/.flushda_owned ]; then echo "!!! $P exists and is not this harness's: not touched"; exit 1; fi
    (cd $(dirname $TAR) && sha256sum -c --quiet $(basename $TAR).sha256) || { echo "!!! $TAR fails its sha256"; exit 1; }
    rm -rf "${P:?}.new" && mkdir -p $P.new && tar -I zstd -xf $TAR -C $P.new || { echo "!!! unpack failed"; exit 1; }
    (cd $P.new && md5sum -c --quiet MD5SUMS) || { echo "!!! binaries differ from MD5SUMS"; exit 1; }
    case $(cat $P.new/COMMIT) in 3efd636e8df*) ;; *) echo "!!! tree at $(cat $P.new/COMMIT)"; exit 1 ;; esac
    ln -sfn build_Release $P.new/build
    echo "unpacked by $(hostname) job ${SLURM_JOB_ID:-?} from $TAR" > $P.new/.flushda_owned
    rm -rf "${P:?}" && mv $P.new $P || exit 1
    # U: P's top-level entries as links, tt_metal/fabric/hw/inc/edm_fabric real with the unpatched header
    rm -rf "${U:?}" && mkdir -p $U
    for e in $P/* $P/.[!.]*; do b=$(basename $e); [ "$b" = tt_metal ] || [ "$b" = generated ] || ln -s $e $U/$b; done
    mkdir -p $U/generated
    python3 - $P $U $EDM <<'PY'
import os, sys
P, U, rel = sys.argv[1:4]
parts = rel.split("/")
for k in range(len(parts) + 1):  # real directories along the path, links for everything else
    d = "/".join(parts[:k])
    os.makedirs(os.path.join(U, d), exist_ok=True)
    nxt = parts[k] if k < len(parts) else None
    for e in os.listdir(os.path.join(P, d)):
        if e == nxt:
            continue
        dst = os.path.join(U, d, e)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(P, d, e), dst)
PY
    rm -f "${U:?}/${EDM:?}/fabric_edm_packet_transmission.hpp" && cp $D/d51e_fabric_edm_packet_transmission.hpp $U/$EDM/fabric_edm_packet_transmission.hpp
    diff -q $P/$EDM/fabric_edm_packet_transmission.hpp $U/$EDM/fabric_edm_packet_transmission.hpp > /dev/null && { echo "!!! U's header equals P's"; exit 1; }
    grep -q "all_writes_counted && !enable_deadlock_avoidance" $U/$EDM/fabric_edm_packet_transmission.hpp || { echo "!!! U's header is not d51e's"; exit 1; }
    echo "P: $(cat $P/COMMIT | cut -c1-11) + $(grep -c '^[+-][^+-]' $P/PATCH) patch lines; U: P with d51e42cd78a's fabric_edm_packet_transmission.hpp"
fi
declare -A TREE=([unp]=$U [pat]=$P)
one() {  # pass arm suite
    local pass=$1 a=$2 s=$3 y t kind env label rc
    read -r y t kind env <<< "$(suite_of $s)"
    label=fda${pass}_${a}_$s
    if [ "$DRY" = 1 ]; then echo "   $label: ${TREE[$a]} $y (timeout $t, $kind) ${env:-}"; return 0; fi
    PT=${TREE[$a]} C=/home/aagarwal/.tt-metal-cache-flushda-$a-$pass
    run_port $label $y $t $env; rc=$?
    [ $rc = 2 ] && { echo "!!! $label HUNG (captured): stopping; a hang rules the change out"; exit 1; }
    if [ $rc != 0 ]; then
        [ $kind = val ] && { echo "!!! $label: data-checked run failed (rc $rc): stopping"; exit 1; }
        echo "!!! $label failed (rc $rc): going on"
    fi
    return 0
}
for pass in a b; do
    if [ $pass = a ]; then arms=(unp pat); else arms=(pat unp); fi
    for a in "${arms[@]}"; do
        C=/home/aagarwal/.tt-metal-cache-flushda-$a-$pass
        [ "$DRY" = 1 ] || { rm -rf "${C:?}"; mkdir -p $C; }
        echo "##### pass $pass, $a (job at ${SECONDS} s)"
        for s in "${SUITES[@]}"; do one $pass $a $s; done
        [ "$DRY" = 1 ] || rm -rf "${C:?}"
    done
done
echo "ALL_DONE (job at ${SECONDS} s)"
