#!/usr/bin/env bash
# Sweep the buffering depth (NUM_TRIDS = outstanding reads per core) and report
# DRAM read bandwidth at each. Start double-buffered and work up.
set -u
source python_env/bin/activate
CSV="generated/profiler/.logs/profile_log_device.csv"

echo "depth | per-core B/cyc | aggregate B/cyc | GB/s | util"
echo "------|----------------|-----------------|------|-----"
for n in 2 3 4 6 8 12 15; do
    rm -f "$CSV"
    out=$(BH_DRAM_READ_NUM_TRIDS=$n TT_METAL_DEVICE_PROFILER=1 python3 measure_bh_dram_read_bw.py 2>/dev/null)
    pc=$(echo "$out"  | sed -n 's/.*per-core bytes\/cycle  : \([0-9.]*\).*/\1/p')
    agg=$(echo "$out" | sed -n 's/.*AGGREGATE bytes\/cycle : \([0-9.]*\).*/\1/p')
    gbs=$(echo "$out" | sed -n 's/.*AGGREGATE bandwidth   : \([0-9.]*\).*/\1/p')
    ut=$(echo "$out"  | sed -n 's/.*UTILIZATION           : \([0-9.]*\).*/\1/p')
    printf "%5s | %14s | %15s | %4s | %s%%\n" "$n" "$pc" "$agg" "$gbs" "$ut"
done
