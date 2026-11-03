#!/bin/bash
# tel_txo_runs.sh ROUTER...: tel_runs.sh's checks for the sender pass's telemetry stage (X6, V4_SENDER_OWNERSHIP.md §12),
# main's router first, then each ROUTER (variant), two ERISCs and one. The perf recorder's per-core totals
# ("packets_sent=", "eth_words_sent=") are compared across routers by tel_txo_report.py.
# (From tel_runs.sh: telemetry on v4, 2026-10-29; consolidation/V4_TELEMETRY.md, variants/v4tel = v4spd + the host's telemetry options:
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
C1=/home/aagarwal/.tt-metal-cache-teltxo-v1
rm -rf "${C1:?}-two" "${C1:?}-one"; mkdir -p $C1-two $C1-one
main_env="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
raw() {  # label: keep the run's raw telemetry and the telemetry bandwidth lines
    mkdir -p $out/${1}_raw
    mv generated/fabric/*_raw_telemetry.csv $out/${1}_raw/ 2>/dev/null
    grep -a 'Measured BW' $out/$1.log | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^.*Measured/   Measured/' | head -20
}
rm -f generated/fabric/*_raw_telemetry.csv

for r in "$@"; do [ -d fabric_exp/variants/$r ] || { echo "no variant $r"; exit 1; }; done
echo "##### main's router (job at ${SECONDS} s)"
for m in two one; do
    me=""; [ $m = one ] && me=$E
    KA_ENV= run_cap teltxo_v1_${m}_gtest "$G" $C1-$m 400 $main_env $TEL $me; [ $? = 2 ] && stop
    for s in 4352 1d; do
        y=$(suite_yaml 4352); [ $s = 1d ] && y=fabric_exp/exp_1d_4352.yaml
        CAP_NO_BW_ENV= KA_ENV= run_cap teltxob_v1_${m}_$s $y $C1-$m 300 $main_env $me; [ $? = 2 ] && stop
        raw teltxob_v1_${m}_$s
    done
done
for r in "$@"; do
    echo "##### $r (job at ${SECONDS} s)"
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    for m in two one; do
        me=""; [ $m = one ] && me=$E
        c=/home/aagarwal/.tt-metal-cache-teltxo-$r-$m; rm -rf "${c:?}"; mkdir -p $c
        run_cap teltxo_${r}_${m}_gtest "$G" $c 400 $TEL $me; [ $? = 2 ] && stop
        for s in 4352 1d; do
            y=$(suite_yaml 4352); [ $s = 1d ] && y=fabric_exp/exp_1d_4352.yaml
            CAP_NO_BW_ENV= run_cap teltxob_${r}_${m}_$s $y $c 300 $me; [ $? = 2 ] && stop
            raw teltxob_${r}_${m}_$s
        done
    done
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo ALL_DONE
