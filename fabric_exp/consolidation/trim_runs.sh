#!/bin/bash
# Channel trimming on v4 (2026-10-29; consolidation/V4_TRIM.md, variants/v4trm = v4tel + the forwarded-to capture, the
# forwarding-off receiver and the host's downstream open gates). Main's channel-trimming gtests (fabric_unit_tests):
#   - capture: ChannelTrimming1DFixture.* / ChannelTrimming2DFixture.* (the recorded bitfields: sender channels used,
#     receiver channels that forwarded, packet sizes, the 2D forwarded-to masks) and the override fixtures (traffic
#     with a fixed override profile);
#   - replay: ChannelTrimmingScenarios/ChannelTrimmingReplayTest.* (11 scenarios, Mesh and Torus X: capture a traffic
#     shape, check the captured shape, rebuild the fabric from it and run the traffic again);
#   - the host-only ChannelTrimming* unit tests run too (no device).
# One run per job (each is up to ~20 min): trim_runs.sh ROUTER MODE, ROUTER main (main's router as shipped: kernel v1,
# flags 0, firmware keepalive) or v4trm, MODE two or one (ERISCs). Label trim_<router>_<mode>. Main first, as control.
# Hang capture armed; a hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
R=$1 MODE=$2
G="gtest:ChannelTrimming*"
E=""; [ "$MODE" = one ] && E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
[ "$MODE" = one ] || [ "$MODE" = two ] || { echo "MODE two|one"; exit 1; }
C=/home/aagarwal/.tt-metal-cache-trim-$R-$MODE
rm -rf "${C:?}"; mkdir -p $C
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }

echo "##### trimming gtests: $R, $MODE ERISC(s) (job at ${SECONDS} s)"
case $R in
    main)
        KA_ENV= run_cap trim_main_$MODE "$G" $C 1300 TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0 $E
        [ $? = 2 ] && stop ;;
    v4trm)
        python3 $CON/swap.py apply fabric_exp/variants/v4trm $BK || stop
        run_cap trim_v4trm_$MODE "$G" $C 1300 $E; [ $? = 2 ] && stop
        python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;;
    *) echo "ROUTER main|v4trm"; exit 1 ;;
esac
grep -aE '^\[ +(OK|FAILED|SKIPPED) +\] [A-Za-z]' $out/trim_${R}_$MODE.log | sed 's/ ([0-9]* ms)$//' | head -80
rm -f $M
echo ALL_DONE
