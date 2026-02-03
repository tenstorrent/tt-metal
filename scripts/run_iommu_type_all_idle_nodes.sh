#!/usr/bin/env bash
# Runs IOMMU group type check on all idle nodes in the cluster.
# Uses sinfo for idle nodelists and squeue to exclude nodes with running jobs.
# Logs output to a file, grouped per node with partition info.

set -euo pipefail

LOG_FILE="${1:-iommu_type_idle_nodes_$(date +%Y%m%d_%H%M%S).log}"
CMD="for dev in /sys/bus/pci/devices/*; do cat \"\$dev/iommu_group/type\" 2>/dev/null; done"

echo "Logging to: $LOG_FILE"
echo "Started at: $(date -Iseconds)" | tee "$LOG_FILE"

# Busy nodes: nodes that have running (R) or configuring (CF) jobs
busy_nodes_file=$(mktemp)
trap 'rm -f "$busy_nodes_file"' EXIT
squeue -h -t R,CF -o "%N" 2>/dev/null | while read -r nodelist; do
  [[ -z "$nodelist" ]] && continue
  scontrol show hostnames "$nodelist" 2>/dev/null || true
done | sort -u > "$busy_nodes_file"
echo "Nodes with running/configuring jobs: $(wc -l < "$busy_nodes_file")" | tee -a "$LOG_FILE"

# All up nodes: from sinfo take every partition line that is not down/drain (idle, alloc, mix, etc.),
# then we exclude busy nodes from squeue so we run only on nodes that are free.
idle_partitions=$(sinfo -h -o "%P %N %T" | awk '$3 != "down" && $3 != "down*" && $3 != "drain" { print $1, $2 }')
if [[ -z "$idle_partitions" ]]; then
  echo "No partition lines from sinfo (excluding down/drain)." | tee -a "$LOG_FILE"
  exit 0
fi

count=0
skipped=0
while read -r partition nodelist; do
  [[ -z "$nodelist" ]] && continue
  for node in $(scontrol show hostnames "$nodelist" 2>/dev/null); do
    [[ -z "$node" ]] && continue
    if grep -Fxq "$node" "$busy_nodes_file" 2>/dev/null; then
      skipped=$((skipped + 1))
      continue
    fi
    count=$((count + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "========== node=$node partition=$partition ==========" | tee -a "$LOG_FILE"
    if srun -w "$node" -p "$partition" --pty bash -c "$CMD" >> "$LOG_FILE" 2>&1; then
      echo "(exit 0)" | tee -a "$LOG_FILE"
    else
      echo "(exit $?)" | tee -a "$LOG_FILE"
    fi
  done
done <<< "$idle_partitions"

echo "" | tee -a "$LOG_FILE"
echo "Finished at: $(date -Iseconds). Ran on $count idle nodes, skipped $skipped busy nodes." | tee -a "$LOG_FILE"
