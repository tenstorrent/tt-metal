#!/bin/bash
# v4tm Tensix mux mode bring-up on Blackhole (2026-10-28; V4_TENSIX_MUX.md). v4tm = v4se with mux mode: the receiver
# credits the far worker channel without reading the header's source channel, waits for the downstream muxes before
# opening its downstream connections, and the gate admits mux mode on one ERISC without VC1 (offline: every non-mux mock
# config loads the same image as v4se, 202 of 202; mux mode builds for 1D, 2D and ring). Main's router first as the
# same-job control (kernel v1, flags 0, firmware keepalive), then v4tm; fresh caches, hang capture armed:
#   val_mux.yaml (the mux cases of test_fabric_sanity_at_least_2x2_mesh.yaml: linear, full-ring and mesh multicast,
#   the ring deadlock check), then fabric_unit_tests on the Tensix fixtures (Fabric1DTensixFixture, the nightly 1D and 2D
#   Tensix fixtures); v4tm also runs the standalone mux v2 smoke fixture (default mode).
# A hang or a unit-test timeout stops the job; failures do not. Labels tm_{v1,v4tm}_*.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
STUDY="TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_FABRIC_OPT_LEVEL=O3"
TENSIX="Fabric1DTensixFixture.*:NightlyFabric1DTensixFixture.*:NightlyFabric2DTensixFixture.*"
procs() {  # cache: which ERISCs the router binaries were built for, and the mux builds
    find $1 -name '*.elf' ! -name '*.xip.elf' -path '*kernels/*' 2>/dev/null | grep -E 'fabric_erisc_router|mux' \
        | awk -F/ '{for(i=1;i<=NF;i++) if($i=="kernels") print $(i+1), $(NF-1)}' | sort | uniq -c | tr '\n' ';'
}
unit() {  # label cache filter env...: fabric_unit_tests, report; returns 2 on a timeout
    local l=$1 c=$2 f=$3; shift 3
    echo "=== $l: fabric_unit_tests $f $*"
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_METAL_CACHE=$c "$@" timeout 600 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
        --gtest_filter="$f" > $out/$l.log 2>&1
    local rc=$?
    echo "exit $rc; $(grep -aE '^\[  PASSED  \]|^\[  FAILED  \] [0-9]+ test|^\[  SKIPPED \] [0-9]+ test' $out/$l.log | tr '\n' ' ')"
    grep -a '^\[  FAILED  \] [A-Z]' $out/$l.log | head -8 | sed 's/^/   | /'
    [ $rc = 124 ] && { echo "!!! $l timed out"; return 2; }
    return 0
}
C=/home/aagarwal/.tt-metal-cache-tm-v1
rm -rf "${C:?}"; mkdir -p $C
KA_ENV= run_cap tm_v1_val_mux fabric_exp/v4/val_mux.yaml $C 300 $MAIN $L; [ $? = 2 ] && { echo STOPPED; exit 1; }
unit tm_v1_unit_tensix $C "$TENSIX" $MAIN || { echo STOPPED; exit 1; }
echo "   builds: $(procs $C)"

python3 $CON/swap.py apply fabric_exp/variants/v4tm $BK || { echo STOPPED; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-tm-v4tm
rm -rf "${C:?}"; mkdir -p $C
run_cap tm_v4tm_val_mux fabric_exp/v4/val_mux.yaml $C 300 $L; [ $? = 2 ] && stop
unit tm_v4tm_unit_tensix $C "$TENSIX" $STUDY || stop
echo "   builds: $(procs $C)"
C=/home/aagarwal/.tt-metal-cache-tm-v4tm-muxv2
rm -rf "${C:?}"; mkdir -p $C
unit tm_v4tm_unit_muxv2 $C "*FabricMuxV2Smoke*Fixture.*" $STUDY || stop
echo "   builds: $(procs $C)"
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
