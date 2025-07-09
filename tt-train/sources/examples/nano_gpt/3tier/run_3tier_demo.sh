#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# run_3tier_demo.sh
#
# Copy nano_gpt 3-tier binaries and config to remote hosts and launch via MPI.
#
# Usage: ./run_3tier_demo.sh [-h] [-m METAL_HOME] [-c CONFIG] [-p N] [-d N]
# -----------------------------------------------------------------------------

# Defaults (customize as needed)
METAL_HOME="${TT_METAL_HOME:-/home/ttuser/git/tt-metal}"
CONFIG="training_shakespear_nanogpt_3tier.yaml"
BIN_DIR="${METAL_HOME}/tt-train/build/sources/examples/nano_gpt"
CFG_DIR="${METAL_HOME}/tt-train/configs"
HOSTFILE="/tmp/mpi_hosts.$$"
BINARIES=(nano_gpt nano_gpt_aggregator nano_gpt_optimizer)
SSH_USER="ttuser"
SCP_OPTS="-p"    # preserve modification times & modes

# Your cluster hosts, in the order MPI should assign ranks:
HOSTS=(
  "11.228.0.10"   # worker #1 (this host—skip copy)
  "11.228.0.11"   # worker #2
  "11.228.0.14"   # aggregator
  "11.228.0.16"   # optimizer
  # "11.228.0.15" # disabled host (slots=1) – uncomment to re-enable
)

print_usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -h            Show this help and exit
  -m METAL_HOME Override TT_METAL_HOME (default: $METAL_HOME)
  -c CONFIG     Config filename (default: $CONFIG)
  -p N          Pass "-p N" to each nano_gpt invocation
  -d N          Pass "-d N" to each nano_gpt invocation
                (cannot use -p and -d together)
EOF
}

# parse flags
P_FLAG="" D_FLAG=""
while getopts "hm:c:p:d:" opt; do
  case "$opt" in
    h) print_usage; exit 0 ;;
    m) METAL_HOME="$OPTARG"
       BIN_DIR="${METAL_HOME}/tt-train/build/sources/examples/nano_gpt"
       CFG_DIR="${METAL_HOME}/tt-train/configs" ;;
    c) CONFIG="$OPTARG" ;;
    p) P_FLAG="-p $OPTARG" ;;
    d) D_FLAG="-d $OPTARG" ;;
    *) print_usage; exit 1 ;;
  esac
done

# ensure -p and -d are not both set
if [[ -n "$P_FLAG" && -n "$D_FLAG" ]]; then
  echo "ERROR: -p and -d are mutually exclusive" >&2
  exit 1
fi

# decide which flag to use (or none)
RUN_FLAG="${P_FLAG:-${D_FLAG:-}}"

# derive MPI process counts
NUM_HOSTS=${#HOSTS[@]}
if (( NUM_HOSTS < 3 )); then
  echo "ERROR: need at least 3 hosts (2 workers + 1 aggregator + 1 optimizer)" >&2
  exit 1
fi
AGG_COUNT=1
OPT_COUNT=1
WORKER_COUNT=$(( NUM_HOSTS - AGG_COUNT - OPT_COUNT ))

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

# launch MPI job
echo "Launching MPI 3-tier demo..."
mpirun --hostfile "${HOSTFILE}" \
  -np "${WORKER_COUNT}" bash -lc "export TT_METAL_HOME='${METAL_HOME}' && export TT_LOGGER_LEVEL=FATAL && export TT_MESH_ID=0 && export TT_HOST_RANK=0 && \"${BIN_DIR}/nano_gpt\" -c \"${CFG_DIR}/${CONFIG}\" $RUN_FLAG" \
  : -np "${AGG_COUNT}"    bash -lc "export TT_METAL_HOME='${METAL_HOME}' && export TT_LOGGER_LEVEL=FATAL && export TT_MESH_ID=0 && export TT_HOST_RANK=0 && \"${BIN_DIR}/nano_gpt_aggregator\" -c \"${CFG_DIR}/${CONFIG}\" $RUN_FLAG" \
  : -np "${OPT_COUNT}"    bash -lc "export TT_METAL_HOME='${METAL_HOME}' && export TT_LOGGER_LEVEL=FATAL && export TT_MESH_ID=0 && export TT_HOST_RANK=0 && \"${BIN_DIR}/nano_gpt_optimizer\" -c \"${CFG_DIR}/${CONFIG}\" $RUN_FLAG"

# cleanup
rm -f "${HOSTFILE}"
echo "✔ MPI job finished."
