#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${1:-}" ]]; then
  export TT_METAL_HOME="$1"
  shift
fi

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  echo "TT_METAL_HOME is not set. Export TT_METAL_HOME to your tt-metal checkout, or pass it as the first argument." >&2
  echo "  Example: TT_METAL_HOME=/path/to/tt-metal $0" >&2
  echo "  Example: $0 /path/to/tt-metal" >&2
  exit 2
fi

if [[ ! -d "${TT_METAL_HOME}" ]]; then
  echo "TT_METAL_HOME is not a directory: ${TT_METAL_HOME}" >&2
  exit 2
fi

export PATH="${TT_METAL_HOME}/python_env/bin:${PATH}"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"

TEST_PATH="models/demos/deepseek_v3/demo/test_demo.py"
TEACHER_FORCED_TEST_PATH="models/demos/deepseek_v3/demo/test_demo_teacher_forced.py::test_demo_teacher_forcing_accuracy"
TCP_INTERFACE="${TT_RUN_TCP_INTERFACE:-ens5f0np0}"
DUAL_RANK_BINDING="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
QUAD_RANK_BINDING="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
USER_NAME="${USER:-$(id -un)}"
DEFAULT_DUAL_HOSTS_CSV="UF-EV-B8-GWH01,UF-EV-B8-GWH02"
DEFAULT_QUAD_HOSTS_CSV="UF-EV-B8-GWH01,UF-EV-B8-GWH02,UF-EV-B9-GWH01,UF-EV-B9-GWH02"

run_demo_case() {
  local rank_binding="$1"
  local rankfile="$2"
  local hosts="$3"
  local case_id="$4"

  echo
  echo "=== Running ${case_id} ==="
  tt-run \
    --tcp-interface "${TCP_INTERFACE}" \
    --rank-binding "${rank_binding}" \
    --mpi-args "--host ${hosts} --map-by rankfile:file=${rankfile} --bind-to none -x DEEPSEEK_V3_HF_MODEL -x DEEPSEEK_V3_CACHE -x MESH_DEVICE" \
    pytest -svvv "${TEST_PATH}" -k "${case_id}"
}

run_teacher_forced_case() {
  local rank_binding="$1"
  local rankfile="$2"
  local hosts="$3"
  local mesh_name="$4"

  echo
  echo "=== Running ${mesh_name} teacher-forced demo ==="
  tt-run \
    --tcp-interface "${TCP_INTERFACE}" \
    --rank-binding "${rank_binding}" \
    --mpi-args "--host ${hosts} --map-by rankfile:file=${rankfile} --bind-to none -x DEEPSEEK_V3_HF_MODEL -x DEEPSEEK_V3_CACHE -x MESH_DEVICE" \
    pytest -svvv "${TEACHER_FORCED_TEST_PATH}"
}

ensure_non_default_hosts_on_other_system() {
  local hosts_csv="$1"
  local default_hosts_csv="$2"
  local env_var_name="$3"

  if [[ "${hosts_csv}" != "${default_hosts_csv}" ]]; then
    return
  fi

  local current_host
  current_host="$(hostname -s 2>/dev/null || hostname)"
  local current_host_upper="${current_host^^}"
  IFS=',' read -r -a default_hosts <<< "${default_hosts_csv}"

  local host
  for host in "${default_hosts[@]}"; do
    if [[ "${current_host_upper}" == "${host^^}" ]]; then
      return
    fi
  done

  echo "Default hosts (${default_hosts_csv}) target the B8/B9 systems, but this machine is '${current_host}'." >&2
  echo "Set ${env_var_name} to the correct comma-separated hosts for your system and re-run this script." >&2
  exit 2
}

cd "${TT_METAL_HOME}"

### DUAL TESTS ###
export DEEPSEEK_V3_HF_MODEL="${DEEPSEEK_V3_HF_MODEL_DUAL:-/data/deepseek/DeepSeek-R1-0528-dequantized}"
export DEEPSEEK_V3_CACHE="${DEEPSEEK_V3_CACHE_DUAL:-/data/deepseek/DeepSeek-R1-0528-Cache}"
export MESH_DEVICE="DUAL"

DUAL_HOSTS_CSV="${DUAL_MPI_HOSTS:-${MPI_HOSTS:-${DEFAULT_DUAL_HOSTS_CSV}}}"
ensure_non_default_hosts_on_other_system "${DUAL_HOSTS_CSV}" "${DEFAULT_DUAL_HOSTS_CSV}" "DUAL_MPI_HOSTS (or MPI_HOSTS)"
DUAL_RANKFILE_PATH="${DUAL_RANKFILE_PATH:-${TMPDIR:-/tmp}/rankfile_b8_dual_demo_${USER_NAME}}"
IFS=',' read -r -a dual_hosts <<< "${DUAL_HOSTS_CSV}"
if [[ ${#dual_hosts[@]} -ne 2 ]]; then
  echo "DUAL_MPI_HOSTS must contain exactly 2 comma-separated hosts." >&2
  exit 2
fi

cat > "${DUAL_RANKFILE_PATH}" <<EOF
# mpirun rankfile
rank 0=${dual_hosts[0]} slot=0:*
rank 1=${dual_hosts[1]} slot=0:*
EOF

for case_id in \
  dual_full_demo_32upr \
  dual_full_demo_8upr \
  dual_stress_demo_32upr \
  dual_stress_demo_8upr; do
  run_demo_case "${DUAL_RANK_BINDING}" "${DUAL_RANKFILE_PATH}" "${DUAL_HOSTS_CSV}" "${case_id}"
done

run_teacher_forced_case "${DUAL_RANK_BINDING}" "${DUAL_RANKFILE_PATH}" "${DUAL_HOSTS_CSV}" "DUAL"

### QUAD TESTS ###
export DEEPSEEK_V3_HF_MODEL="${DEEPSEEK_V3_HF_MODEL_QUAD:-/data/deepseek/DeepSeek-R1-0528-dequantized}"
export DEEPSEEK_V3_CACHE="${DEEPSEEK_V3_CACHE_QUAD:-/data/deepseek/DeepSeek-R1-0528-Cache-pprajapati}"
export MESH_DEVICE="QUAD"

QUAD_HOSTS_CSV="${QUAD_MPI_HOSTS:-${DEFAULT_QUAD_HOSTS_CSV}}"
ensure_non_default_hosts_on_other_system "${QUAD_HOSTS_CSV}" "${DEFAULT_QUAD_HOSTS_CSV}" "QUAD_MPI_HOSTS"
QUAD_RANKFILE_PATH="${QUAD_RANKFILE_PATH:-/tmp/rankfile_b89_quad_working}"
IFS=',' read -r -a quad_hosts <<< "${QUAD_HOSTS_CSV}"
if [[ ${#quad_hosts[@]} -ne 4 ]]; then
  echo "QUAD_MPI_HOSTS must contain exactly 4 comma-separated hosts." >&2
  exit 2
fi

cat > "${QUAD_RANKFILE_PATH}" <<EOF
# mpirun rankfile
rank 0=${quad_hosts[0]} slot=0:*
rank 1=${quad_hosts[1]} slot=0:*
rank 2=${quad_hosts[2]} slot=0:*
rank 3=${quad_hosts[3]} slot=0:*
EOF

for case_id in \
  quad_full_demo_32upr \
  quad_full_demo_8upr \
  quad_stress_demo_32upr \
  quad_stress_demo_8upr; do
  run_demo_case "${QUAD_RANK_BINDING}" "${QUAD_RANKFILE_PATH}" "${QUAD_HOSTS_CSV}" "${case_id}"
done

run_teacher_forced_case "${QUAD_RANK_BINDING}" "${QUAD_RANKFILE_PATH}" "${QUAD_HOSTS_CSV}" "QUAD"
