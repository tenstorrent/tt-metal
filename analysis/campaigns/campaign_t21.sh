#!/bin/bash
# T2.1 grid 1: anchor + seven (q,k) points, causal and non-causal, zones on and off, 3 iterations each.
# Zones-on runs also get a perf-counter multipass capture (counter view cross-check, section 2.4).
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"   # DD is the data directory in both layouts (campaign_paths.sh)
cd $DD
POINTS=${POINTS:-"128:128 64:128 256:128 128:256 128:512 512:128 512:512"}
MODES=${MODES:-"causal noncausal"}
for QK in $POINTS; do
  QC=${QK%%:*}; KC=${QK##*:}
  for MODE in $MODES; do
    C=1; [ $MODE = noncausal ] && C=0
    $SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
    SDPA_CAUSAL=$C SDPA_QCHUNK=$QC SDPA_KCHUNK=$KC $SD/run_zone.sh t21_${MODE}_q${QC}k${KC}_zoff
    $SD/set_zone_config.sh 1 0 0 0 0 >/dev/null
    SDPA_CAUSAL=$C SDPA_QCHUNK=$QC SDPA_KCHUNK=$KC $SD/run_zone.sh t21_${MODE}_q${QC}k${KC}_zon
    if [ "${WITH_MP:-1}" = 1 ]; then
      SDPA_CAUSAL=$C SDPA_QCHUNK=$QC SDPA_KCHUNK=$KC $SD/run_zone.sh t21_${MODE}_q${QC}k${KC}_zon_mp mp
    fi
  done
done
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
echo "### T2.1 campaign done $(date -u +%H:%M:%SZ)"
