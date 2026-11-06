#!/bin/bash
# Latency size sweep, old vs new router in one broker job, in the given order (pass label first).
# usage: run_lat.sh <pass-label> <router> [<router>...]   (router: new | old -> /home/aagarwal/tt-metal-lat-<router>)
{
set -u
pass=$1; shift
out=/home/aagarwal/tt-metal/fabric_exp/latsweep/results; mkdir -p $out
Y=/home/aagarwal/tt-metal/fabric_exp/latsweep/lat_sweep_p150.yaml
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
for r in "$@"; do
  T=/home/aagarwal/tt-metal-lat-$r; label=${pass}_${r}
  export TT_METAL_HOME=$T TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0; unset TT_METAL_SLOW_DISPATCH_MODE
  cd $T; rm -f generated/fabric/latency_results_blackhole.csv
  echo "=== $label ($(git log --oneline -1 | cut -c1-60)) ==="
  TT_FABRIC_NO_BW_TELEMETRY=1 timeout 600 ./build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $Y 2>&1 |
    tee $out/$label.log | grep --line-buffered -E "Running Test:|FATAL|error:|Tests:"
  st=${PIPESTATUS[0]}
  cp generated/fabric/latency_results_blackhole.csv $out/$label.csv 2>/dev/null || echo "!!! no latency csv for $label"
  echo "=== $label: exit $st ==="
  [ $st -eq 0 ] || exit $st
done
exit 0
}
