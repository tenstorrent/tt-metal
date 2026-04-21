#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  b4-mtp-ci.sh [TT_METAL_HOME] [smoke|reference|tests|module|all|reset|hosts]

Examples:
  ./models/demos/deepseek_v3/demo/b4-mtp-ci.sh /data/$USER/tt-metal smoke
  TT_METAL_HOME=/data/$USER/tt-metal ./models/demos/deepseek_v3/demo/b4-mtp-ci.sh all
  B4_MTP_FORCE_REFERENCE=1 ./models/demos/deepseek_v3/demo/b4-mtp-ci.sh reference

Modes:
  smoke      Reset B4 with /data/deepseek/scripts/tt-reset, then run the dual-host MTP demo smoke test.
  reference  Reset B4, then regenerate the MTP reference payload.
  tests      Reset B4, ensure the reference payload exists, then run models/demos/deepseek_v3/tests/test_mtp.py.
  module     Alias for tests.
  all        Reset B4, run the smoke test, reset B4 again, then run the MTP test suite.
  reset      Run the cluster-provided B4 reset and health check only.
  hosts      Verify SSH reachability for both B4 hosts only.

Notes:
  - This script is B4-specific and uses CLUSTER=B4 from /data/deepseek/scripts/cluster-config.sh.
  - CI defaults to false here. The reference-generator test is marked skip in CI.
  - Set B4_MTP_FORCE_REFERENCE=1 to force reference regeneration in any mode.
EOF
}

is_mode() {
  case "${1:-}" in
    smoke|reference|tests|module|all|reset|hosts)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -n "${1:-}" ]] && ! is_mode "${1:-}" && [[ -d "${1:-}" ]]; then
  export TT_METAL_HOME="$1"
  shift
fi

MODE="${1:-all}"

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  export TT_METAL_HOME="$(pwd)"
fi

if [[ ! -d "${TT_METAL_HOME}" ]]; then
  echo "TT_METAL_HOME is not a directory: ${TT_METAL_HOME}" >&2
  exit 2
fi

if ! is_mode "${MODE}"; then
  echo "Unknown mode: ${MODE}" >&2
  usage >&2
  exit 2
fi

CLUSTER_SCRIPTS_DIR="${CLUSTER_SCRIPTS_DIR:-/data/deepseek/scripts}"
ENV_METAL_PATH="${CLUSTER_SCRIPTS_DIR}/env-metal"
CLUSTER_CONFIG_PATH="${CLUSTER_SCRIPTS_DIR}/cluster-config.sh"
TT_RESET_PATH="${CLUSTER_SCRIPTS_DIR}/tt-reset"
export CLUSTER="B4"

if [[ ! -f "${ENV_METAL_PATH}" ]]; then
  echo "Cluster env helper not found: ${ENV_METAL_PATH}" >&2
  exit 2
fi

if [[ ! -f "${CLUSTER_CONFIG_PATH}" ]]; then
  echo "Cluster config helper not found: ${CLUSTER_CONFIG_PATH}" >&2
  exit 2
fi

if [[ ! -x "${TT_RESET_PATH}" ]]; then
  echo "Cluster reset helper not found or not executable: ${TT_RESET_PATH}" >&2
  exit 2
fi

source "${ENV_METAL_PATH}"
source "${CLUSTER_CONFIG_PATH}"
unset USE_TORUS_MODE

if [[ "${CLUSTER_TOPOLOGY:-}" != "dual" ]]; then
  echo "Expected CLUSTER=B4 to resolve to a dual-host topology, got: ${CLUSTER_TOPOLOGY:-unset}" >&2
  exit 2
fi

if [[ "${MESH_DEVICE:-}" != "DUAL" ]]; then
  echo "Expected CLUSTER=B4 to export MESH_DEVICE=DUAL, got: ${MESH_DEVICE:-unset}" >&2
  exit 2
fi

export PATH="${TT_METAL_HOME}/python_env/bin:${PATH}"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-${TT_METAL_HOME}}"
export TT_RUN_ORIGINAL_CWD="${TT_RUN_ORIGINAL_CWD:-${TT_METAL_HOME}}"
export CI="${CI:-false}"
export DEEPSEEK_V3_HF_MODEL="${DEEPSEEK_V3_HF_MODEL:-/data/deepseek/DeepSeek-R1-0528-dequantized}"
DEFAULT_DEEPSEEK_V3_CACHE="/data/deepseek/DeepSeek-R1-0528-Cache"
export DEEPSEEK_V3_CACHE="${DEEPSEEK_V3_CACHE:-${DEFAULT_DEEPSEEK_V3_CACHE}}"
export B4_MTP_ALLOW_COLD_CACHE="${B4_MTP_ALLOW_COLD_CACHE:-0}"

SSH_BIN="${SSH_BIN:-ssh}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
ARTIFACT_DIR="${ARTIFACT_DIR:-${TT_METAL_HOME}/generated/artifacts}"
DEMO_LOG_PATH="${ARTIFACT_DIR}/b4_dual_demo_mtp_output.log"
MODULE_LOG_PATH="${ARTIFACT_DIR}/b4_dual_test_mtp_output.log"
REFERENCE_LOG_PATH="${ARTIFACT_DIR}/b4_dual_test_mtp_reference_output.log"
REFERENCE_STEPS="${DEEPSEEK_V3_MTP_REF_STEPS:-128}"
REFERENCE_PATH="${DEEPSEEK_V3_CACHE}/test_io_cache/mtp_full_model_seq${REFERENCE_STEPS}.pt"
IFS=',' read -r -a dual_hosts <<< "${CLUSTER_HOSTS}"
if [[ ${#dual_hosts[@]} -ne 2 ]]; then
  echo "Expected CLUSTER=B4 to resolve to exactly 2 hosts, got: ${CLUSTER_HOSTS}" >&2
  exit 2
fi

mkdir -p "${ARTIFACT_DIR}"

if [[ ! -f "${CLUSTER_RANKFILE}" ]]; then
  echo "Cluster rankfile not found: ${CLUSTER_RANKFILE}" >&2
  exit 2
fi

if [[ ! -f "${CLUSTER_RANK_BINDING}" ]]; then
  echo "Cluster rank-binding config not found: ${CLUSTER_RANK_BINDING}" >&2
  exit 2
fi

CURRENT_CACHE_CONFIG_PATH="${DEEPSEEK_V3_CACHE}/61_layers/mesh_8x8/config.json"
DEFAULT_CACHE_CONFIG_PATH="${DEFAULT_DEEPSEEK_V3_CACHE}/61_layers/mesh_8x8/config.json"
if [[ "${B4_MTP_ALLOW_COLD_CACHE}" != "1" ]] \
  && [[ "${DEEPSEEK_V3_CACHE}" != "${DEFAULT_DEEPSEEK_V3_CACHE}" ]] \
  && [[ ! -f "${CURRENT_CACHE_CONFIG_PATH}" ]] \
  && [[ -f "${DEFAULT_CACHE_CONFIG_PATH}" ]]; then
  echo "Selected cache is missing a full 61-layer dual-host config: ${CURRENT_CACHE_CONFIG_PATH}"
  echo "Falling back to populated cache: ${DEFAULT_DEEPSEEK_V3_CACHE}"
  export DEEPSEEK_V3_CACHE="${DEFAULT_DEEPSEEK_V3_CACHE}"
  REFERENCE_PATH="${DEEPSEEK_V3_CACHE}/test_io_cache/mtp_full_model_seq${REFERENCE_STEPS}.pt"
fi

if ! command -v tt-run >/dev/null 2>&1; then
  echo "tt-run is not available in PATH. Build the repo and create python_env first." >&2
  exit 2
fi

if [[ ! -x "${TT_METAL_HOME}/python_env/bin/pytest" ]]; then
  echo "pytest not found in ${TT_METAL_HOME}/python_env/bin. Build/create python_env first." >&2
  exit 2
fi

print_run_context() {
  echo "TT_METAL_HOME=${TT_METAL_HOME}"
  echo "CLUSTER=${CLUSTER}"
  echo "CLUSTER_HOSTS=${CLUSTER_HOSTS}"
  echo "CLUSTER_RANKFILE=${CLUSTER_RANKFILE}"
  echo "CLUSTER_RANK_BINDING=${CLUSTER_RANK_BINDING}"
  echo "CLUSTER_TCP_INTERFACE=${CLUSTER_TCP_INTERFACE}"
  echo "MESH_DEVICE=${MESH_DEVICE}"
  echo "CI=${CI}"
  echo "DEEPSEEK_V3_HF_MODEL=${DEEPSEEK_V3_HF_MODEL}"
  echo "DEEPSEEK_V3_CACHE=${DEEPSEEK_V3_CACHE}"
  echo "REFERENCE_PATH=${REFERENCE_PATH}"
}

build_mpi_exports() {
  local -n out_args=$1
  out_args=(-x CI -x MESH_DEVICE)

  local var_name
  while IFS='=' read -r var_name _; do
    case "${var_name}" in
      DEEPSEEK_*)
        out_args+=(-x "${var_name}")
        ;;
    esac
  done < <(env)

  if [[ -n "${PYTEST_ADDOPTS:-}" ]]; then
    out_args+=(-x PYTEST_ADDOPTS)
  fi

  if [[ -n "${USE_TORUS_MODE:-}" ]]; then
    out_args+=(-x USE_TORUS_MODE)
  fi
}

build_mpi_args_string() {
  local -a mpi_args
  local -a mpi_exports
  build_mpi_exports mpi_exports
  mpi_args=(
    --host "${CLUSTER_HOSTS}"
    --map-by "rankfile:file=${CLUSTER_RANKFILE}"
    --bind-to none
    "${mpi_exports[@]}"
  )
  printf '%q ' "${mpi_args[@]}"
}

probe_hosts() {
  local host
  for host in "${dual_hosts[@]}"; do
    echo "Checking SSH reachability for ${host}"
    "${SSH_BIN}" "${SSH_OPTS[@]}" "${host}" "hostname -s"
  done
}

reset_cluster() {
  echo
  echo "=== Resetting ${CLUSTER} with ${TT_RESET_PATH} ==="
  CLUSTER="${CLUSTER}" TT_METAL_HOME="${TT_METAL_HOME}" "${TT_RESET_PATH}"
}

run_tt_pytest() {
  local test_path="$1"
  local log_path="$2"
  local mpi_args
  mpi_args="$(build_mpi_args_string)"

  tt-run \
    --rank-binding "${CLUSTER_RANK_BINDING}" \
    --tcp-interface "${CLUSTER_TCP_INTERFACE}" \
    --mpi-args "${mpi_args% }" \
    bash -lc "set -o pipefail; cd '${TT_METAL_HOME}'; pytest -svvv -rs '${test_path}' 2>&1 | tee '${log_path}'"
}

run_demo_smoke() {
  echo
  echo "=== Running dual demo MTP smoke test ==="
  run_tt_pytest "models/demos/deepseek_v3/demo/test_mtp_demo.py" "${DEMO_LOG_PATH}"
}

ensure_reference_payload() {
  local force_reference="${B4_MTP_FORCE_REFERENCE:-0}"
  local backup_reference=""

  if [[ "${MODE}" == "reference" ]]; then
    force_reference=1
  fi

  if [[ "${force_reference}" != "1" ]] && [[ -f "${REFERENCE_PATH}" ]]; then
    echo "Using existing MTP reference payload: ${REFERENCE_PATH}"
    return
  fi

  echo
  if [[ -f "${REFERENCE_PATH}" ]]; then
    backup_reference="${REFERENCE_PATH}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "=== Regenerating MTP reference payload (${REFERENCE_PATH}) ==="
    mv "${REFERENCE_PATH}" "${backup_reference}"
  else
    echo "=== Generating missing MTP reference payload (${REFERENCE_PATH}) ==="
  fi

  rm -f "${REFERENCE_PATH%.*}.tmp"
  local saved_ci="${CI}"
  export CI=false
  export DEEPSEEK_V3_MTP_GENERATE_REFERENCE=1
  if ! run_tt_pytest "models/demos/deepseek_v3/tests/test_mtp.py::test_generate_mtp_reference_io" "${REFERENCE_LOG_PATH}"; then
    unset DEEPSEEK_V3_MTP_GENERATE_REFERENCE
    export CI="${saved_ci}"
    if [[ -n "${backup_reference}" ]] && [[ ! -f "${REFERENCE_PATH}" ]]; then
      mv "${backup_reference}" "${REFERENCE_PATH}"
    fi
    return 1
  fi
  unset DEEPSEEK_V3_MTP_GENERATE_REFERENCE
  export CI="${saved_ci}"

  if [[ ! -f "${REFERENCE_PATH}" ]]; then
    if [[ -n "${backup_reference}" ]] && [[ -f "${backup_reference}" ]]; then
      mv "${backup_reference}" "${REFERENCE_PATH}"
    fi
    echo "Expected reference payload was not created: ${REFERENCE_PATH}" >&2
    exit 1
  fi

  if [[ -n "${backup_reference}" ]] && [[ -f "${backup_reference}" ]]; then
    rm -f "${backup_reference}"
  fi
}

run_module_suite() {
  echo
  echo "=== Running DeepSeek V3 MTP test suite (CI=${CI}) ==="
  ensure_reference_payload
  run_tt_pytest "models/demos/deepseek_v3/tests/test_mtp.py" "${MODULE_LOG_PATH}"
}

cd "${TT_METAL_HOME}"
print_run_context

case "${MODE}" in
  hosts)
    probe_hosts
    ;;
  smoke)
    probe_hosts
    reset_cluster
    run_demo_smoke
    ;;
  reference)
    probe_hosts
    reset_cluster
    ensure_reference_payload
    ;;
  tests|module)
    probe_hosts
    reset_cluster
    run_module_suite
    ;;
  all)
    probe_hosts
    reset_cluster
    run_demo_smoke
    reset_cluster
    run_module_suite
    ;;
  reset)
    probe_hosts
    reset_cluster
    ;;
esac
