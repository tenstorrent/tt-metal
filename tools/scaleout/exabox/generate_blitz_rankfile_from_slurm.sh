#!/usr/bin/env bash
# Generate blitz_decode_pipeline rankfile from SLURM allocation (4 nodes, 4 ranks per node, 1 slot per rank).
# Use this when running under SLURM so hostnames in the rankfile match the allocation and avoid
# "host was not allocated or oversubscribed" from Open MPI.
#
# If you still see that error, your cluster may require full hostnames (FQDN). This script
# tries to use FQDNs when the launch node's hostname -f differs from the short name.
#
# Usage:
#   source python_env/bin/activate  # or your env
#   . tools/scaleout/exabox/generate_blitz_rankfile_from_slurm.sh
#   tt-run --mpi-args "--rankfile $RANKFILE --host $HOSTS_SLOTS --tag-output" ...
#
# If it still fails, try adding --oversubscribe to MPI args.

set -e
if [ -z "${SLURM_NODELIST}" ]; then
  echo "SLURM_NODELIST is not set. Run this script inside a SLURM allocation (srun/sbatch)." >&2
  exit 1
fi

RANKFILE="${1:-}"
if [ -z "$RANKFILE" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  TT_METAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
  RANKFILE="$TT_METAL_ROOT/blitz_decode_pipeline_rank_file_single_pod"
fi
RANKFILE="$(cd "$(dirname "$RANKFILE")" && pwd)/$(basename "$RANKFILE")"
RANKS_PER_NODE=4

mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
NUM_NODES=${#NODES[@]}
if [ "$NUM_NODES" -ne 4 ]; then
  echo "Expected 4 nodes from SLURM_NODELIST, got $NUM_NODES." >&2
  exit 1
fi

# If launch node's FQDN differs from short name, use FQDN for all nodes (some clusters require this).
USE_FQDN=false
MY_FQDN=$(hostname -f 2>/dev/null || true)
MY_SHORT=$(hostname -s 2>/dev/null || echo "${NODES[0]}")
if [ -n "$MY_FQDN" ] && [ "$MY_FQDN" != "$MY_SHORT" ]; then
  case "$MY_FQDN" in
    ${MY_SHORT}.*) DOMAIN_SUFFIX="${MY_FQDN#${MY_SHORT}}"; USE_FQDN=true ;;
    *) DOMAIN_SUFFIX="" ;;
  esac
fi

if [ "$USE_FQDN" = true ] && [ -n "$DOMAIN_SUFFIX" ]; then
  echo "Using FQDN for rankfile (hostname -f differs from short name). Suffix: $DOMAIN_SUFFIX"
  for i in "${!NODES[@]}"; do
    NODES[i]="${NODES[i]}${DOMAIN_SUFFIX}"
  done
fi

{
  rank=0
  for node in "${NODES[@]}"; do
    for slot in 0 1 2 3; do
      echo "rank $rank=$node slot=$slot"
      rank=$((rank + 1))
    done
  done
} > "$RANKFILE"

export HOSTS
HOSTS=$(IFS=,; echo "${NODES[*]}")
# Optional: export host list with slot count for --host node1:4,node2:4,...
export HOSTS_SLOTS
HOSTS_SLOTS=$(printf "%s:${RANKS_PER_NODE}," "${NODES[@]}" | sed 's/,$//')

echo "Wrote rankfile: $RANKFILE"
echo "HOSTS=$HOSTS"
echo "HOSTS_SLOTS=$HOSTS_SLOTS"
echo "Use: tt-run --mpi-args \"--rankfile $RANKFILE --host \$HOSTS_SLOTS --tag-output\" ..."
echo "If allocation error persists, add: --oversubscribe"
