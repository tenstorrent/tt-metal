#!/bin/bash
# UDM mode on v4 (2026-10-29; consolidation/V4_UDM.md, variants/v4udm = v4rs + the relay as VC0's local downstream).
# fabric_unit_tests's UDM fixtures (FabricTensixConfig::UDM: one ERISC, a mux and a relay per router direction, 2D):
# Fabric2DUDMModeFixture.* (the default mesh) and NightlyFabric2DUDMModeFixture.* (2x4, 8 devices), with ROUTER:
#   v1     main's router as shipped (kernel v1, flags 0, firmware keepalive): the control
#   v4udm  variants/v4udm in the v2 slot (the study's settings)
# Hang capture armed (cap_lib.sh run_cap, gtest:FILTER); a hang stops the job, test failures do not. Fresh cache per
# router. Offline: the 28 UDM configs build; every other config's image equals v4rs's (612 configs).
# Logs fabric_exp/results/udm_<router>_{unit,nightly}.log.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
r=$1
c=/home/aagarwal/.tt-metal-cache-udm-$r
rm -rf "${c:?}"; mkdir -p $c
fixtures() {  # env...
    run_cap udm_${r}_unit "gtest:Fabric2DUDMModeFixture.*" $c 600 "$@"; [ $? = 2 ] && return 2
    run_cap udm_${r}_nightly "gtest:NightlyFabric2DUDMModeFixture.*" $c 900 "$@"; [ $? = 2 ] && return 2
    return 0
}
case $r in
    v1)
        KA_ENV= fixtures TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0 || { echo STOPPED; exit 1; } ;;
    v4udm)
        python3 $CON/swap.py apply fabric_exp/variants/v4udm $BK || { echo STOPPED; exit 1; }
        fixtures; rc=$?
        python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
        [ $rc = 2 ] && { echo STOPPED; exit 1; } ;;
    *)
        echo "usage: udm_runs.sh v1|v4udm"; exit 1 ;;
esac
echo "   builds: $(find $c -name '*.elf' ! -name '*.xip.elf' -path '*kernels/*' | awk -F/ '{for(i=1;i<=NF;i++) if($i=="kernels") print $(i+1)}' | sort | uniq -c | tr '\n' ';')"
rm -f $M
echo ALL_DONE
