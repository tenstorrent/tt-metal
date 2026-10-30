#!/bin/bash
# Which side a row is sensitive to (2026-10-29; V4_PROFILE.md): v4trm's rows with a fixed spin per packet on one side,
# variants/v4trmd (v4trm + the study spins: TT_FABRIC_EXP_TX_DELAY after each send, TT_FABRIC_EXP_RX_DELAY after each
# packet received; at 0 every image is byte-identical to v4trm's, 600 of 600). Per suite, one job, one cache:
#   base (no spin), tx32, tx96, rx32, rx96
# A spin of N cycles costs about N+12 on its path. A side with no slack shows the full cost already at 32 (+44), a side
# with slack S >= ~108 shows nothing at 96; the 32 -> 96 step (both runs carry the same spin code, only the constant
# differs) is the sensitivity without the hook's own code effect.
#   prof_delay_runs.sh two   two ERISCs: exp_2d_4352 (4352), exp_1d_4352 (1d), p3_mixes (mix), v4/exp_ring (ring),
#                            v4/exp_torus_y (torusy)
#   prof_delay_runs.sh one   one ERISC: exp_2d_4352, exp_1d_4352, v4/exp_ring_se; mux mode v4/exp_mux
#   prof_delay_runs.sh vc1   the VC1 split (vc1_alloc.yaml, two MPI ranks)
# Labels prof_<suite>_<arm> (VC1: profvc1_<arm>_default_p1_o3). Hang capture armed; a hang stops the job.
# A second argument "pass" spins once per pass instead of once per packet (V4_PROFILE.md §6; TT_FABRIC_EXP_DRISC_DELAY:
# low 16 bits after each sender pass, high 16 bits after each receiver pass): arms base, sp16, sp48, rp16, rp48; labels
# profp_<suite>_<arm> (VC1: profpvc1_<arm>_default_p1_o3). The 16 -> 48 step over 32 is how many passes of that kind
# sit on the row's critical path per packet.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
G=$1 MODE=${2:-packet}
case $G in two|one|vc1) ;; *) echo "usage: prof_delay_runs.sh two|one|vc1 [pass]"; exit 1 ;; esac
case $MODE in packet) P=prof ;; pass) P=profp ;; *) echo "usage: prof_delay_runs.sh two|one|vc1 [pass]"; exit 1 ;; esac
[ $G = vc1 ] && { source $CON/vc1_lib.sh; source $CON/vc1_capture_lib.sh; }
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
C=/home/aagarwal/.tt-metal-cache-$P-$G
rm -rf "${C:?}"; mkdir -p $C
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
ARMS="base tx32 tx96 rx32 rx96"
[ $MODE = pass ] && ARMS="base sp16 sp48 rp16 rp48"
arm_env() {
    case $1 in
        base) echo "" ;; tx*) echo "TT_FABRIC_EXP_TX_DELAY=${1#tx}" ;; rx*) echo "TT_FABRIC_EXP_RX_DELAY=${1#rx}" ;;
        sp*) echo "TT_FABRIC_EXP_DRISC_DELAY=${1#sp}" ;; rp*) echo "TT_FABRIC_EXP_DRISC_DELAY=$(( ${1#rp} << 16 ))" ;;
    esac
}
sweep() {  # suite yaml timeout [ENV...]
    local s=$1 y=$2 t=$3 a; shift 3
    for a in $ARMS; do
        run_cap ${P}_${s}_$a $y $C $t $(arm_env $a) "$@"; [ $? = 2 ] && stop
    done
}
python3 $CON/swap.py apply fabric_exp/variants/v4trmd $BK || stop
case $G in
    two)
        sweep 4352 $(suite_yaml 4352) 300
        sweep 1d $(suite_yaml 1d) 300
        sweep mix $(suite_yaml mix) 300
        sweep ring fabric_exp/v4/exp_ring.yaml 200
        sweep torusy fabric_exp/v4/exp_torus_y.yaml 200 ;;
    one)
        sweep 4352_1e $(suite_yaml 4352) 300 $E
        sweep 1d_1e $(suite_yaml 1d) 300 $E
        sweep rse fabric_exp/v4/exp_ring_se.yaml 300 $E
        sweep mux fabric_exp/v4/exp_mux.yaml 400 ;;
    vc1)
        for a in $ARMS; do
            env_a=$(arm_env $a)
            VC1CAP_EXTRA_ENV="$env_a" VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
                run_vc1_cap ${P}vc1_${a}_default_p1_o3 $C O3 || stop
        done ;;
esac
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
