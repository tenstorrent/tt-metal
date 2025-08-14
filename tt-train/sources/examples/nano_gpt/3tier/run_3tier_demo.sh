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
DRY_RUN=false


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
MESH_IDS=(0 0 0 0 0)
if grep -q "socket_type: fabric" "${CFG_DIR}/${CONFIG}"; then
  MESH_IDS=(1 4 3 0 2)
fi

print_usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -h            Show this help and exit
  -m METAL_HOME Override TT_METAL_HOME (default: $METAL_HOME)
  -c CONFIG     Config filename (default: $CONFIG)
  -n            Dry run - show commands without executing
EOF
}

# parse flags
while getopts "hm:c:n" opt; do
  case "$opt" in
    h) print_usage; exit 0 ;;
    m) METAL_HOME="$OPTARG"
       BIN_DIR="${METAL_HOME}/tt-train/build/sources/examples/nano_gpt"
       CFG_DIR="${METAL_HOME}/tt-train/configs" ;;
    c) CONFIG="$OPTARG" ;;
    n) DRY_RUN=true ;;
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
echo "Verifying local files..."
missing_files=0
for bin in "${BINARIES[@]}"; do
  bin_path="${BIN_DIR}/${bin}"
  if [[ ! -f "$bin_path" ]]; then
    echo "ERROR: Binary not found: $bin_path" >&2
    ((missing_files++))
  elif [[ ! -x "$bin_path" ]]; then
    echo "ERROR: Binary not executable: $bin_path" >&2
    ((missing_files++))
  else
    echo "✓ Found: $bin_path"
  fi
done

if [[ ! -f "${CFG_DIR}/${CONFIG}" ]]; then
  echo "ERROR: Config file not found: ${CFG_DIR}/${CONFIG}" >&2
  ((missing_files++))
else
  echo "✓ Found: ${CFG_DIR}/${CONFIG}"
fi

if (( missing_files > 0 )); then
  echo "ERROR: $missing_files required files are missing or invalid" >&2
  echo "Build directory: $BIN_DIR" >&2
  echo "Config directory: $CFG_DIR" >&2
  echo "" >&2
  echo "To build the missing binaries, try:" >&2
  echo "  cd $METAL_HOME" >&2
  echo "  make -C tt-train build" >&2
  echo "  # or" >&2
  echo "  cd tt-train && mkdir -p build && cd build" >&2
  echo "  cmake .. && make -j\$(nproc)" >&2
  exit 1
fi
echo "✓ All local files verified"

# build hostfile in OpenMPI format with slots=1
{
  for h in "${HOSTS[@]}"; do
    printf "%s slots=1\n" "$h"
  done
} > "${HOSTFILE}"

# copy to all remote hosts (skip index 0)
copy_to_remote_hosts() {
  local unique_hosts
  IFS=' ' read -r -a unique_hosts <<< "$(printf '%s\n' "${HOSTS[@]}" | sort -u | tr '\n' ' ')"

  if [[ ${#unique_hosts[@]} -eq 1 ]]; then
    echo "All hosts are the same (${unique_hosts[0]}) - skipping remote copy"
    return 0
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: Would copy binaries and config to remote hosts..."
    for host in "${HOSTS[@]:1}"; do
      echo "  -> $host (skipped in dry run)"
    done
    return 0
  fi

  echo "Copying binaries and config to remote hosts..."
  local copy_errors=0

  for host in "${HOSTS[@]:1}"; do
    echo " -> $host"

    # Test SSH connectivity first
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${SSH_USER}@${host}" "echo 'SSH test successful'" >/dev/null 2>&1; then
      echo "   ERROR: SSH connection failed to $host" >&2
      echo "   Check: 1) Host is reachable, 2) SSH keys are set up, 3) User '$SSH_USER' exists" >&2
      ((copy_errors++))
      continue
    fi

    # Create directories with error handling
    if ! ssh "${SSH_USER}@${host}" "mkdir -p '${BIN_DIR}' '${CFG_DIR}'" 2>/dev/null; then
      echo "   ERROR: Failed to create directories on $host" >&2
      ((copy_errors++))
      continue
    fi

    # Copy binaries
    for bin in "${BINARIES[@]}"; do
      local src_file="${BIN_DIR}/${bin}"
      local dest="${SSH_USER}@${host}:${BIN_DIR}/"

      # Check if source file exists and is executable
      if [[ ! -f "$src_file" ]]; then
        echo "   ERROR: Source binary not found: $src_file" >&2
        ((copy_errors++))
        continue
      elif [[ ! -x "$src_file" ]]; then
        echo "   WARNING: Source binary not executable: $src_file" >&2
      fi

      # Attempt copy with detailed error output
      echo "   Copying $bin..."
      local scp_output
      if scp_output=$(scp ${SCP_OPTS} "$src_file" "$dest" 2>&1); then
        echo "   ✓ $bin copied successfully"
      else
        echo "   ERROR: Failed to copy $bin to $host" >&2
        echo "   Source: $src_file" >&2
        echo "   Destination: $dest" >&2
        echo "   SCP output: $scp_output" >&2
        ((copy_errors++))
      fi
    done

    # Copy config
    if ! scp ${SCP_OPTS} "${CFG_DIR}/${CONFIG}" "${SSH_USER}@${host}:${CFG_DIR}/" 2>/dev/null; then
      echo "   ERROR: Failed to copy config to $host" >&2
      ((copy_errors++))
    fi
  done

  if (( copy_errors > 0 )); then
    echo "WARNING: $copy_errors copy operations failed" >&2
  else
    echo "✔ Remote copy complete."
  fi
}

copy_to_remote_hosts

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

# Build common environment variables
build_env_vars() {
  local with_fabric="$1"
  local env_vars=(
    "TT_METAL_HOME='${METAL_HOME}'"
    "TT_LOGGER_LEVEL=FATAL"
    "TT_HOST_RANK=0"
    "MESH_IDS_STR='${MESH_IDS_STR}'"
  )

  if [[ "$with_fabric" == "true" ]]; then
    env_vars+=("TT_MESH_GRAPH_DESC_PATH=${MESH_GRAPH_DESC_PATH}")
  fi

  printf "export %s; " "${env_vars[@]}"
}

# Build MPI command for a specific binary
build_mpi_command() {
  local binary="$1"
  local count="$2"
  local env_setup="$3"

  echo "-x MESH_IDS_STR=\"\${MESH_IDS_STR}\" -np \"${count}\" bash -lc \"${env_setup}\${mesh_id_snippet}; \\\"${BIN_DIR}/${binary}\\\" -c \\\"${CFG_DIR}/${CONFIG}\\\"\""
}

# Check if fabric configuration is enabled
USE_FABRIC=false
if grep -q "socket_type: fabric" "${CFG_DIR}/${CONFIG}"; then
  USE_FABRIC=true
  echo "Fabric configuration detected - using mesh graph descriptor"
else
  echo "Standard configuration detected - fabric disabled"
fi

# Build environment setup
ENV_SETUP=$(build_env_vars "$USE_FABRIC")

# Build MPI commands for each process type
WORKER_CMD=$(build_mpi_command "nano_gpt" "${WORKER_COUNT}" "${ENV_SETUP}")
AGG_CMD=$(build_mpi_command "nano_gpt_aggregator" "${AGG_COUNT}" "${ENV_SETUP}")
OPT_CMD=$(build_mpi_command "nano_gpt_optimizer" "${OPT_COUNT}" "${ENV_SETUP}")

# Validate MPI setup
validate_mpi_setup() {
  echo "Validating MPI setup..."
  echo "  Total processes: $((WORKER_COUNT + AGG_COUNT + OPT_COUNT))"
  echo "  Available hosts: ${#HOSTS[@]}"
  echo "  Workers: ${WORKER_COUNT}, Aggregators: ${AGG_COUNT}, Optimizers: ${OPT_COUNT}"

  if (( WORKER_COUNT + AGG_COUNT + OPT_COUNT > ${#HOSTS[@]} )); then
    echo "WARNING: More processes ($(( WORKER_COUNT + AGG_COUNT + OPT_COUNT ))) than available hosts (${#HOSTS[@]})" >&2
    echo "  Some hosts will run multiple processes"
  fi

  # Test MPI is available
  if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun command not found!" >&2
    exit 1
  fi

  echo "✔ MPI setup validation passed"
}

validate_mpi_setup

# launch MPI job
echo "Launching MPI 3-tier demo with ${WORKER_COUNT} workers, ${AGG_COUNT} aggregators, ${OPT_COUNT} optimizers..."
echo "Environment: USE_FABRIC=${USE_FABRIC}"

MPI_COMMAND="mpirun --hostfile \"${HOSTFILE}\" \\
  ${WORKER_CMD} \\
  : ${AGG_CMD} \\
  : ${OPT_CMD}"

if [[ "$DRY_RUN" == "true" ]]; then
  echo ""
  echo "=== DRY RUN MODE - Command that would be executed ==="
  echo "$MPI_COMMAND"
  echo "=== END DRY RUN ==="
  exit 0
fi

eval "$MPI_COMMAND"

# cleanup
cleanup() {
  rm -f "${HOSTFILE}"
  echo "✔ Cleanup completed."
}

# Set up cleanup trap
trap cleanup EXIT

# Launch with error handling
echo "Executing MPI command..."
if ! eval "$MPI_COMMAND"; then
  echo "ERROR: MPI job failed!" >&2
  exit 1
fi

echo "✔ MPI job finished successfully."
