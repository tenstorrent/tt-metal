#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# run_3tier_demo.sh
#
# Copy nano_gpt 3-tier binaries and config to remote hosts and launch via MPI.
#
# Usage: ./run_3tier_demo.sh [-h] [-m METAL_HOME] [-c CONFIG]
# -----------------------------------------------------------------------------

# Defaults (customize as needed)
METAL_HOME="${TT_METAL_HOME:-/home/ttuser/git/tt-metal}"
CONFIG="training_shakespeare_nanogpt_3tier_mpi.yaml"
BIN_DIR="${METAL_HOME}/tt-train/build/sources/examples/nano_gpt"
CFG_DIR="${METAL_HOME}/tt-train/configs"
HOSTFILE="/tmp/mpi_hosts.$$"
BINARIES=(nano_gpt nano_gpt_aggregator nano_gpt_optimizer)
SSH_USER="ttuser"
MESH_GRAPH_DESC_PATH="${METAL_HOME}/tests/tt_metal/tt_fabric/custom_mesh_descriptors/nano_exabox_1x8_mesh_graph_descriptor.yaml"
SCP_OPTS="-p"    # preserve modification times & modes

# Your cluster hosts, in the order MPI should assign ranks:
HOSTS=(
  "11.228.0.10"
  "11.228.0.11"
  "11.228.0.14"
  "11.228.0.15"
  "11.228.0.16"
)

# One MESH_ID per *global* MPI rank (workers..., aggregator, optimizer)
# If fewer entries than total ranks, ranks beyond the end will default to their rank id.
MESH_IDS=(1 4 3 0 2)

print_usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -h            Show this help and exit
  -m METAL_HOME Override TT_METAL_HOME (default: $METAL_HOME)
  -c CONFIG     Config filename (default: $CONFIG)
EOF
}

# parse flags
while getopts "hm:c:" opt; do
  case "$opt" in
    h) print_usage; exit 0 ;;
    m) METAL_HOME="$OPTARG"
       BIN_DIR="${METAL_HOME}/tt-train/build/sources/examples/nano_gpt"
       CFG_DIR="${METAL_HOME}/tt-train/configs" ;;
    c) CONFIG="$OPTARG" ;;
    *) print_usage; exit 1 ;;
  esac
done

# derive MPI process counts
NUM_HOSTS=${#HOSTS[@]}
if (( NUM_HOSTS < 3 )); then
  echo "ERROR: need at least 3 hosts (2 workers + 1 aggregator + 1 optimizer)" >&2
  exit 1
fi
AGG_COUNT=1
OPT_COUNT=1
WORKER_COUNT=$(( NUM_HOSTS - AGG_COUNT - OPT_COUNT ))
TOTAL_RANKS=$(( WORKER_COUNT + AGG_COUNT + OPT_COUNT ))

# verify binaries & config exist locally
for bin in "${BINARIES[@]}"; do
  if [[ ! -x "${BIN_DIR}/${bin}" ]]; then
    echo "ERROR: Missing or not executable: ${BIN_DIR}/${bin}" >&2
    exit 1
  fi
done
if [[ ! -f "${CFG_DIR}/${CONFIG}" ]]; then
  echo "ERROR: Config file not found: ${CFG_DIR}/${CONFIG}" >&2
  exit 1
fi

# build hostfile in OpenMPI format with slots=1
{
  for h in "${HOSTS[@]}"; do
    printf "%s slots=1\n" "$h"
  done
} > "${HOSTFILE}"

# copy to all remote hosts (skip index 0)
echo "Copying binaries and config to remote hosts..."
for host in "${HOSTS[@]:1}"; do
  echo " -> $host"
  ssh "${SSH_USER}@${host}" "mkdir -p '${BIN_DIR}' '${CFG_DIR}'"
  for bin in "${BINARIES[@]}"; do
    scp ${SCP_OPTS} "${BIN_DIR}/${bin}" "${SSH_USER}@${host}:${BIN_DIR}/"
  done
  scp ${SCP_OPTS} "${CFG_DIR}/${CONFIG}" "${SSH_USER}@${host}:${CFG_DIR}/"
done
echo "✔ Remote copy complete."

# --- Per-rank TT_MESH_ID wiring ---

# Serialize MESH_IDS to pass via env
MESH_IDS_STR="$(IFS=,; echo "${MESH_IDS[*]:-}")"
echo "DEBUG: MESH_IDS_STR='${MESH_IDS_STR}'"

# Count entries without awk (robust, no external deps)
MESH_COUNT=0
if [[ -n "${MESH_IDS_STR}" ]]; then
  IFS=, read -r -a __mids_tmp <<< "${MESH_IDS_STR}"
  MESH_COUNT=${#__mids_tmp[@]}
fi

if (( MESH_COUNT < TOTAL_RANKS )); then
  echo "WARN: MESH_IDS has $MESH_COUNT entries but TOTAL_RANKS is $TOTAL_RANKS."
  echo "      Ranks >= $MESH_COUNT will default to TT_MESH_ID=\$OMPI_COMM_WORLD_RANK."
fi

# Snippet evaluated on *each* rank to set TT_MESH_ID (built with a safe heredoc)
mesh_id_snippet="$(cat <<'EOSNIP'
  IDX=${OMPI_COMM_WORLD_RANK:-0}
  echo "[rank=${IDX}] DEBUG: MESH_IDS_STR='${MESH_IDS_STR:-}'" >&2
  if [[ -n "${MESH_IDS_STR:-}" ]]; then
    IFS=, read -r -a __mids <<< "${MESH_IDS_STR}"
    echo "[rank=${IDX}] DEBUG: __mids array has ${#__mids[@]} elements: ${__mids[*]}" >&2
    if [[ ${IDX} -lt ${#__mids[@]} ]]; then
      export TT_MESH_ID="${__mids[${IDX}]}"
      echo "[rank=${IDX}] DEBUG: Using __mids[${IDX}]=${__mids[${IDX}]}" >&2
    else
      export TT_MESH_ID="${IDX}"
      echo "[rank=${IDX}] DEBUG: Using fallback IDX=${IDX}" >&2
    fi
  else
    export TT_MESH_ID="${IDX}"
    echo "[rank=${IDX}] DEBUG: MESH_IDS_STR is empty, using fallback IDX=${IDX}" >&2
  fi
  echo "[rank=${IDX}] TT_MESH_ID=${TT_MESH_ID}" >&2
EOSNIP
)"

# Pretty-print planned mapping
echo "Planned mapping:"
for ((i=0;i<TOTAL_RANKS;++i)); do
  mid="${MESH_IDS[$i]:-$i}"
  echo "  rank $i -> TT_MESH_ID=$mid"
done

# launch MPI job
echo "Launching MPI 3-tier demo..."
mpirun --hostfile "${HOSTFILE}" \
  -x MESH_IDS_STR="${MESH_IDS_STR}" \
  -np "${WORKER_COUNT}"   bash -lc "export TT_METAL_HOME='${METAL_HOME}' TT_LOGGER_LEVEL=FATAL TT_HOST_RANK=0 MESH_IDS_STR='${MESH_IDS_STR}'; ${mesh_id_snippet}; \"${BIN_DIR}/nano_gpt\" -c \"${CFG_DIR}/${CONFIG}\"" \
  : -x MESH_IDS_STR="${MESH_IDS_STR}" -np "${AGG_COUNT}"    bash -lc "export TT_METAL_HOME='${METAL_HOME}' TT_LOGGER_LEVEL=FATAL TT_HOST_RANK=0 MESH_IDS_STR='${MESH_IDS_STR}'; ${mesh_id_snippet}; \"${BIN_DIR}/nano_gpt_aggregator\" -c \"${CFG_DIR}/${CONFIG}\""  \
  : -x MESH_IDS_STR="${MESH_IDS_STR}" -np "${OPT_COUNT}"    bash -lc "export TT_METAL_HOME='${METAL_HOME}' TT_LOGGER_LEVEL=FATAL TT_HOST_RANK=0 MESH_IDS_STR='${MESH_IDS_STR}'; ${mesh_id_snippet}; \"${BIN_DIR}/nano_gpt_optimizer\" -c \"${CFG_DIR}/${CONFIG}\""

# cleanup
rm -f "${HOSTFILE}"
echo "✔ MPI job finished."
