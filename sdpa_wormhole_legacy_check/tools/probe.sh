#!/bin/bash
# usage: CARD=n probe.sh <tag> "<args>" ["<args>" ...]; each probe in its own process, reset card on failure
TAG=$1; shift
export TT_METAL_OPERATION_TIMEOUT_SECONDS=10
for a in "$@"; do
  n=$(echo "$a" | tr ' ' '_')
  flock /tmp/tt-device-card${CARD}.lock /localdev/cglagovich/whcheck/run.sh $TAG ${TAG}_probe_$n timeout 300 python /localdev/cglagovich/whcheck/tools/lofi_probe.py $a
  if ! grep -q PROBE_OK /localdev/cglagovich/whcheck/logs/${TAG}_probe_$n.log; then
    echo "PROBE_FAIL $a" >> /localdev/cglagovich/whcheck/logs/${TAG}_probe_$n.log
    flock /tmp/tt-device-card${CARD}.lock tt-smi -r $CARD > /dev/null 2>&1
  fi
done
