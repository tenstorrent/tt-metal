#!/bin/bash
# Telemetry on v4 (2026-10-29; consolidation/V4_TELEMETRY.md, variants/v4tel = v4spd + the host's telemetry options:
# the perf recorder and the dynamic stats). Functional checks of telemetry itself: the runs below have telemetry on
# and none of their numbers are scored or compared with telemetry-off runs (memory feedback-no-bw-telemetry).
# Offline: the 240 telemetry configs build; every telemetry-off config's image equals v4spd's (764).
#   1. Dynamic stats + perf recorder: test_bandwidth_telemetry_validation (FabricBandwidthTelemetry.*: the dynamic
#      bandwidth telemetry within 15% of the measured bandwidth, several sizes; TT_METAL_FABRIC_TELEMETRY=1,
#      TT_METAL_FABRIC_BW_TELEMETRY=1): main's router as shipped (kernel v1, flags 0, firmware keepalive), then v4tel;
#      two ERISCs, then one. Labels tel_<router>_<mode>_gtest.
#   2. Perf recorder: test_tt_fabric benchmarks with its bandwidth telemetry on (CAP_NO_BW_ENV=): exp_4352 (2D) and
#      exp_1d_4352 (1D, NeighborExchange included), main then v4tel; the harness logs the telemetry's bandwidth
#      ("Measured BW (min/avg/max)") and writes generated/fabric/<test>_raw_telemetry.csv (copied to
#      results/<label>_raw/). Labels telb_<router>_<suite>.
# Hang capture armed; a hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
TEL="TT_METAL_FABRIC_TELEMETRY=1 TT_METAL_FABRIC_BW_TELEMETRY=1"
G="gtest:test_bandwidth_telemetry_validation:FabricBandwidthTelemetry.*"
C1=/home/aagarwal/.tt-metal-cache-tel-v1
CT=/home/aagarwal/.tt-metal-cache-tel-v4tel
rm -rf "${C1:?}-two" "${C1:?}-one" "${CT:?}-two" "${CT:?}-one"; mkdir -p $C1-two $C1-one $CT-two $CT-one
main_env="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
raw() {  # label: keep the run's raw telemetry and the telemetry bandwidth lines
    mkdir -p $out/${1}_raw
    mv generated/fabric/*_raw_telemetry.csv $out/${1}_raw/ 2>/dev/null
    grep -a 'Measured BW' $out/$1.log | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^.*Measured/   Measured/' | head -20
}
rm -f generated/fabric/*_raw_telemetry.csv

echo "##### 1. telemetry validation gtest (job at ${SECONDS} s)"
KA_ENV= run_cap tel_v1_two_gtest "$G" $C1-two 400 $main_env $TEL; [ $? = 2 ] && stop
KA_ENV= run_cap tel_v1_one_gtest "$G" $C1-one 400 $main_env $TEL $E; [ $? = 2 ] && stop
python3 $CON/swap.py apply fabric_exp/variants/v4tel $BK || stop
run_cap tel_v4tel_two_gtest "$G" $CT-two 400 $TEL; [ $? = 2 ] && stop
run_cap tel_v4tel_one_gtest "$G" $CT-one 400 $TEL $E; [ $? = 2 ] && stop
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }

echo "##### 2. perf recorder in test_tt_fabric benchmarks (job at ${SECONDS} s)"
for s in 4352 1d; do
    y=$(suite_yaml 4352); [ $s = 1d ] && y=fabric_exp/exp_1d_4352.yaml
    CAP_NO_BW_ENV= KA_ENV= run_cap telb_v1_$s $y $C1-two 300 $main_env; [ $? = 2 ] && stop
    raw telb_v1_$s
done
python3 $CON/swap.py apply fabric_exp/variants/v4tel $BK || stop
for s in 4352 1d; do
    y=$(suite_yaml 4352); [ $s = 1d ] && y=fabric_exp/exp_1d_4352.yaml
    CAP_NO_BW_ENV= run_cap telb_v4tel_$s $y $CT-two 300; [ $? = 2 ] && stop
    raw telb_v4tel_$s
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
