#!/bin/bash
# Slot allocation sweep on v4us (= v4u + transaction ids by slot index for rings that are a whole number of id rounds;
# identical code to v4u at the default depths). Host switches TT_FABRIC_EXP_SLOTS_1D (Linear, Ring) and
# TT_FABRIC_EXP_SLOTS_2D (Mesh, Torus) = worker,forwarding,receiver (VC0; worker 0 = every slot left, up to 127).
# Capacity: 83 slots on 1D and rings (4400 B), 81 on 2D and tori (4448 B).
#   v4us_slots_runs.sh a   1D (16 ids) and ring (8 ids)
#   v4us_slots_runs.sh b   2D mesh (16 ids) and Y-torus (8 ids) (job 071, with the one switch TT_FABRIC_EXP_SLOTS)
#   v4us_slots_runs.sh c   2D and Y-torus follow-up: forwarding 17-20 for receiver 8; receiver 32 on the torus
#   v4us_slots_runs.sh d   2D worker/forwarding at receiver 16 on the stream mixes; 1D around 27/24/32
# Each part first data-checks its least usual depths; a failed check or a hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
part=${1:?part a, b, c or d}
v=v4us
lp=$v; [ $part != b ] && lp=$v$part  # result labels: v4us_* (part b, job 071), v4usa_*, v4usc_*
C=/home/aagarwal/.tt-metal-cache-$v-$part
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply fabric_exp/variants/$v $BK || { echo STOPPED; exit 1; }

tag() { [ -z "$1" ] && echo def || echo "s${1//,/-}"; }
val() {  # 1D|2D slots suite-label yaml
    run_cap ${lp}_$(tag $2)_val_$3 $4 $C 300 ${2:+TT_FABRIC_EXP_SLOTS_$1=$2} TT_FABRIC_EXP_LOG_SLOTS=1 || { echo STOPPED; exit 1; }
}
bench() {  # 1D|2D suite-label yaml slots...
    local f=$1 s=$2 y=$3 slots rc; shift 3
    for slots in "$@"; do
        [ "$slots" = def ] && slots=""
        run_cap ${lp}_$(tag $slots)_$s $y $C 150 ${slots:+TT_FABRIC_EXP_SLOTS_$f=$slots} TT_FABRIC_EXP_LOG_SLOTS=1
        rc=$?; [ $rc = 2 ] && { echo STOPPED; exit 1; }
    done
}

case $part in
a)
    val 1D 0,24,32 1d $(suite_yaml val1d_m1)
    val 1D 0,16,40 ring fabric_exp/v4/val_ring_p150.yaml
    bench 1D 1d $(suite_yaml 1d_m1) def 0,16,32 0,24,32 0,16,48 0,32,32 0,16,16
    bench 1D ring fabric_exp/v4/exp_ring.yaml def 0,16,32 0,24,32 0,16,40 0,16,48 0,24,40 0,16,24
    ;;
b)  # as run in job 071 (TT_FABRIC_EXP_SLOTS then covered both families; these yamls are single-family)
    val 2D 0,12,16 mesh $(suite_yaml mesh)
    val 2D 0,12,24 torusy fabric_exp/v4/val_torus_y_p150.yaml
    bench 2D 4352 $(suite_yaml 4352) def 0,12,16 0,8,16 0,14,16 0,10,16 0,12,32
    bench 2D torusy fabric_exp/v4/exp_torus_y.yaml def 0,12,16 0,8,16 0,14,16 0,12,24 0,10,24
    ;;
c)
    val 2D 0,20,8 mesh $(suite_yaml mesh)
    bench 2D 4352 $(suite_yaml 4352) def 0,16,8 0,17,8 0,18,8 0,19,8 0,20,8
    bench 2D torusy fabric_exp/v4/exp_torus_y.yaml def 0,16,8 0,18,8 0,12,32 0,8,32
    ;;
d)  # the 2D worker/forwarding trade at receiver 16 on the stream mixes too; the 1D pick against the default
    bench 2D 4352 $(suite_yaml 4352) def 0,15,16
    bench 2D mix $(suite_yaml mix) def 0,15,16 0,14,16
    bench 1D 1d $(suite_yaml 1d_m1) 0,20,32 0,28,32
    ;;
esac
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
