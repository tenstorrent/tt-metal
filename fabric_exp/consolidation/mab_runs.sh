#!/bin/bash
# Toolchain-only A/B on the merged tree (2026-10-28): after the merge of main, the VC1 split's inter-mesh rows and main's
# router's VC1 rows moved (jobs 114-117). Same tree, same placement, only the device compiler differs: sfpi 7.91.0
# (runtime/sfpi as the build installed it) against 7.86.0 (the saved copy ~/.sfpi-dl/7.86.0/sfpi, put in place as a
# symlink for its runs; the EXIT trap puts 7.91.0 back). Fresh kernel caches per run. Order 791, 786, 786, 791.
# Per arm: v4gi_m on the VC1 split (vc1_alloc.yaml with hang capture, vc1_inter.yaml), then main's router as shipped
# (kernel v1, flags 0, firmware keepalive) on the VC1 split (builder's level) and the 2D 4352 and 1D suites (-O3).
# Labels mab<tc><p>vc1[i]_{v4gi,v1}_default_p1_{o3,auto}, mab<tc><p>_v1_13129088_{4352,1d} (tc 791 / 786, p a / b).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
export TT_FABRIC_EXP_LOG_SLOTS=1
EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1"
SF=runtime/sfpi
OLD=/home/aagarwal/.sfpi-dl/7.86.0/sfpi
[ -d $SF ] && [ ! -L $SF ] && [ ! -e $SF.791 ] || { echo "unexpected runtime/sfpi state"; ls -la runtime; exit 1; }
toolchain() {  # 791 | 786
    if [ $1 = 786 ]; then
        [ -L $SF ] || { mv $SF $SF.791 && ln -s $OLD $SF; }
    else
        [ -L $SF ] && { rm $SF && mv $SF.791 $SF; }
    fi
    echo "   toolchain: $($SF/compiler/bin/riscv-tt-elf-g++ --version | head -1)"
}
trap 'kill $HB 2>/dev/null; [ -L $SF ] && rm $SF && mv $SF.791 $SF; python3 $CON/swap.py restore $BK' EXIT
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
arm() {  # tc pass
    local tc=$1 p=$2 L=mab$1$2 c=/home/aagarwal/.tt-metal-cache-mab C=/home/aagarwal/.tt-metal-cache-mab-v1
    echo "##### toolchain $tc pass $p (job at ${SECONDS} s)"
    toolchain $tc
    python3 $CON/swap.py apply fabric_exp/variants/v4gi_m $BK || { echo STOPPED; exit 1; }
    VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 run_vc1_cap ${L}vc1_v4gi_default_p1_o3 $c O3 \
        || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 ${L}vc1i_v4gi_default_p1_o3 $c O3 || echo "!!! inter rows failed"
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    KA_ENV= VC1CAP_EXTRA_ENV="$MAIN" VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
        run_vc1_cap ${L}vc1_v1_default_p1_auto $c auto || { echo STOPPED; exit 1; }
    rm -rf "${C:?}"; mkdir -p $C
    for s in 4352 1d; do
        KA_ENV= run_cap ${L}_v1_13129088_$s $(suite_yaml $s) $C 200 $MAIN; [ $? = 2 ] && { echo STOPPED; exit 1; }
    done
}
arm 791 a
arm 786 a
arm 786 b
arm 791 b
toolchain 791
rm -f $M
echo ALL_DONE
