#!/bin/bash
# Standalone mux on v4 (2026-10-28): the mux kernel ops launch on their own Tensix cores (tt_fabric_mux, FabricMuxConfig)
# is a worker to the router, so v4 should carry it as is. fabric_unit_tests --gtest_filter="Fabric1DMuxFixture.*" (a
# plain 1D fabric; not the fabric's Tensix mux mode, FabricTensixConfig::MUX) with main's router in the default mode
# first (same-job control; jobs 119-120 ran it in every mode), then v4se in each MODE given:
#   two   the default (two ERISCs per router)
#   aer1  TT_METAL_DISABLE_MULTI_AERISC=1 (router on ERISC 0)
#   fab1  TT_METAL_DISABLE_FABRIC_TWO_ERISC=1 (router on ERISC 1)
# Fresh caches; per run the router and mux builds made. A timeout stops the job; failures do not.
# Logs mux4_two_v1_unit.log, mux4_<mode>_v4se_unit.log.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
[ $# -gt 0 ] || { echo "usage: mux4_runs.sh MODE..."; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
declare -A MODE_ENV=([two]="" [aer1]="TT_METAL_DISABLE_MULTI_AERISC=1" [fab1]="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1")
builds() {  # cache: the router and mux builds in it, by kernel and RISC
    find $1 -name '*.elf' ! -name '*.xip.elf' -path '*kernels/*' 2>/dev/null | grep -E 'fabric_erisc_router|mux' \
        | awk -F/ '{for(i=1;i<=NF;i++) if($i=="kernels") print $(i+1), $(NF-1)}' | sort | uniq -c | tr '\n' ';'
}
unit() {  # label cache env...: run the mux fixture, report; returns 2 on a timeout
    local l=$1 c=$2; shift 2
    rm -rf "${c:?}"; mkdir -p $c
    echo "=== $l: fabric_unit_tests Fabric1DMuxFixture.* $*"
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_METAL_CACHE=$c "$@" timeout 600 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
        --gtest_filter="Fabric1DMuxFixture.*" > $out/$l.log 2>&1
    local rc=$?
    echo "exit $rc; $(grep -aE '^\[  PASSED  \]|^\[  FAILED  \] [0-9]+ test|^\[  SKIPPED \] [0-9]+ test' $out/$l.log | tr '\n' ' ')"
    grep -a '^\[  FAILED  \] [A-Z]' $out/$l.log | head -8 | sed 's/^/   | /'
    echo "   builds: $(builds $c)"
    [ $rc = 124 ] && { echo "!!! $l timed out"; return 2; }
    return 0
}
unit mux4_two_v1_unit /home/aagarwal/.tt-metal-cache-mux4-v1 TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0 \
    || { echo STOPPED; exit 1; }
python3 $CON/swap.py apply fabric_exp/variants/v4se $BK || { echo STOPPED; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for mode in "$@"; do
    [ -n "${MODE_ENV[$mode]+x}" ] || { echo "unknown mode $mode"; stop; }
    echo "##### mode $mode: ${MODE_ENV[$mode]:-default} (job at ${SECONDS} s)"
    unit mux4_${mode}_v4se_unit /home/aagarwal/.tt-metal-cache-mux4-$mode TT_FABRIC_KERNEL_VERSION=2 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_FABRIC_OPT_LEVEL=O3 \
        ${MODE_ENV[$mode]} || stop
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
