#!/bin/bash
# Compares host-socket throughput on an x8 PCIe chip against an x1 chip.
#
#   sbatch -p <partition> --nodes=2 run_pcie_width_compare.sh [x8_chip] [x1_chip]
#
# Only 4 of a Galaxy's 32 chips have an x8 link (one per tray, ASIC location 6);
# the rest are x1. The link widths are printed first so the premise is checked
# rather than assumed.
#SBATCH --job-name=pcie-width
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail

X8_CHIP="${1:-5}"
X1_CHIP="${2:-0}"
# sbatch copies this script to its spool dir, which exists but holds nothing
# else -- so test for the launcher itself, not for the directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$HERE/run_host_socket_tests.sh" ]] || HERE="${TT_METAL_HOME}/tests/tt_metal/multihost/host_socket"
[[ -x "$HERE/run_host_socket_tests.sh" ]] || { echo "error: cannot find run_host_socket_tests.sh (set TT_METAL_HOME)" >&2; exit 2; }
RESULTS="${TT_METAL_HOME}/tests/tt_metal/multihost/host_socket/results"
mkdir -p "$RESULTS"

echo "########## PCIe link widths (rank 0 host) ##########"
srun --nodes=1 --ntasks=1 bash -c '
  for d in /sys/bus/pci/devices/*/; do
    [[ "$(cat $d/vendor 2>/dev/null)" == "0x1e52" ]] || continue
    echo "  $(basename $d)  width=x$(cat $d/current_link_width 2>/dev/null) speed=$(cat $d/current_link_speed 2>/dev/null)"
  done | sort' 2>&1 | grep -v '^srun:'
echo

for spec in "x8:$X8_CHIP" "x1:$X1_CHIP"; do
    label=${spec%%:*}; chip=${spec#*:}
    for cores in 1 8; do
        echo "########## $label (chip $chip), $cores core(s) ##########"
        HOST_SOCKET_VISIBLE_DEVICES="$chip" \
        TT_HOST_SOCKET_NUM_CORES="$cores" \
        HOST_SOCKET_TIMEOUT=900 \
        HOST_SOCKET_CSV="$RESULTS/pcie_${label}_${cores}c_${SLURM_JOB_ID}.csv" \
            "$HERE/run_host_socket_tests.sh" perf
        echo
    done
done

echo "########## summary ##########"
grep -h . "$RESULTS"/pcie_*_"${SLURM_JOB_ID}".csv 2>/dev/null || echo "(no csv rows)"
