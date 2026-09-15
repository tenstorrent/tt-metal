#!/bin/bash
# T2.2 ablations at the causal anchor and at q128 k256, q128 k512:
#   A4  reader stub (SDPA_ABL_READER_STUB=1): reader reserves/pushes K and V CBs, no NoC reads
#   A4b barrier threshold raised to 64 (SDPA_ABL_BARRIER_THR=64): all 16 K (or V) reads in flight per chunk
#   A2  mask bracket off (SDPA_ABL_MASK_OFF=1)
#   A6  exp stub (SDPA_ABL_EXP_STUB=1): softmax exp_packthread_tile removed, STALLWAIT kept
# Each ablation: zones off (clean wall) and zones on (which parts moved); anchor zones-on also with counters.
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"   # DD is the data directory in both layouts (campaign_paths.sh)
cd $DD
POINTS=${POINTS:-"128:128 128:256 128:512"}
ABL=${ABL:-"a4:1:0:0:0 a4b:0:0:0:64 a2:0:1:0:0 a6:0:0:1:0"}
for A in $ABL; do
  NAME=${A%%:*}; REST=${A#*:}; RS=${REST%%:*}; REST=${REST#*:}; MO=${REST%%:*}; REST=${REST#*:}; ES=${REST%%:*}; BT=${REST#*:}
  for QK in $POINTS; do
    QC=${QK%%:*}; KC=${QK##*:}
    $SD/set_zone_config.sh 0 $RS $MO $ES $BT >/dev/null
    SDPA_CAUSAL=1 SDPA_QCHUNK=$QC SDPA_KCHUNK=$KC $SD/run_zone.sh t22_abl_${NAME}_causal_q${QC}k${KC}_zoff
    $SD/set_zone_config.sh 1 $RS $MO $ES $BT >/dev/null
    SDPA_CAUSAL=1 SDPA_QCHUNK=$QC SDPA_KCHUNK=$KC $SD/run_zone.sh t22_abl_${NAME}_causal_q${QC}k${KC}_zon
    if [ "$QK" = "128:128" ] && [ "${WITH_MP:-1}" = 1 ]; then
      SDPA_CAUSAL=1 SDPA_QCHUNK=$QC SDPA_KCHUNK=$KC $SD/run_zone.sh t22_abl_${NAME}_causal_q${QC}k${KC}_zon_mp mp
    fi
  done
done
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
echo "### T2.2 campaign done $(date -u +%H:%M:%SZ)"
