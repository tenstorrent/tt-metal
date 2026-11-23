#!/bin/bash
# Start driver_perf.sh on this container board only if nothing else is using the board.
# Usage: board_launch.sh <board tag> <version>...
P=/proj_sw/user_dev/ctr-sbratic/tools/perf_cycles
busy=$(ps -eo pid,cmd | grep -E "[p]ytest|[c]laude -p|[r]un_hal_gen|[d]river_|[r]un_full|[g]db|[t]t-smi")
if [[ -n "$busy" ]]; then echo "BUSY, not launched: $busy"; exit 1; fi
nohup $P/driver_perf.sh "$@" > $P/out/driver_$1.nohup 2>&1 &
echo "launched $1 on $(ls /dev/tenstorrent/)"
