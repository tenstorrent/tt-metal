#!/bin/bash
set -eo pipefail

# Default ARCH_NAME for local runs when not pre-set by CI.
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"

readonly DEFAULT_DUAL_HOSTS_CSV="g02glx01,g02glx02"
readonly DEFAULT_QUAD_HOSTS_CSV="g05glx04,g05glx03,g05glx02,g05glx01"
readonly DEFAULT_SOURCE_RANKFILE="${MPI_SOURCE_RANKFILE:-/etc/mpirun/rankfile}"
readonly DEFAULT_TCP_INTERFACE="${TT_RUN_TCP_INTERFACE:-ens5f0np0}"
TMP_RANKFILES=()

usage() {
  cat <<'EOF'
Usage:
  run_quad_galaxy_tests.sh <test_function> [upr_mode]

Notes:
  Run from the tt-metal repository root, or set TT_METAL_HOME to the repo root.
  ARCH_NAME defaults to wormhole_b0 when unset (this script exports it).

Arguments:
  test_function  One of:
                 unit_tests
                 dual_deepseekv3_unit_tests
                 quad_deepseekv3_unit_tests
                 dual_deepseekv3_module_tests
                 quad_deepseekv3_module_tests
                 dual_teacher_forced
                 quad_teacher_forced
                 dual_demo
                 quad_demo
                 dual_demo_mtp
                 quad_demo_mtp
                 dual_demo_stress
                 quad_demo_stress
                 dual_deepseekv3_integration_tests
                 quad_deepseekv3_integration_tests
                 all_needed_local_tests
                 all
  upr_mode       Optional demo UPR mode: all | 32 | 8 (default: all).
                 Ignored for all_needed_local_tests (demos always run 8 and 32 UPR).

Examples:
  bash ./tests/scripts/multihost/run_quad_galaxy_tests.sh unit_tests
  QUAD_MPI_HOSTS="h1,h2,h3,h4" bash ./tests/scripts/multihost/run_quad_galaxy_tests.sh quad_demo_stress 32
  bash ./tests/scripts/multihost/run_quad_galaxy_tests.sh all_needed_local_tests
EOF
}

cleanup_temp_rankfiles() {
  local rankfile_path
  for rankfile_path in "${TMP_RANKFILES[@]}"; do
    rm -f "${rankfile_path}"
  done
}
trap cleanup_temp_rankfiles EXIT

###############################################################################
# Infrastructure unit tests (quad galaxy only)
###############################################################################

run_quad_galaxy_unit_tests() {
  fail=0

  # tt-run --tcp-interface handles tcp and tag flags
  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile"
  local tcp_interface="cnx1"
  local mpi_host="--host g05glx04,g05glx03,g05glx02,g05glx01"
  local mpi_args="$mpi_host $mpi_args_base"

  local mpirun_args_base="$mpi_args_base --mca btl self,tcp --mca btl_tcp_if_include cnx1 --tag-output"
  local mpirun_args="$mpi_host $mpirun_args_base"

  local rank_binding="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
  local descriptor_path="${DESCRIPTOR_PATH:-/etc/mpirun}"

  # TODO: Currently failing
  #mpirun-ulfm $mpi_run_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?

  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation --send-traffic --cabling-descriptor-path ${descriptor_path}/cabling_descriptor.textproto --deployment-descriptor-path ${descriptor_path}/deployment_descriptor.textproto ; fail+=$?

  tt-run --tcp-interface $tcp_interface --rank-binding "$rank_binding" --mpi-args "$mpi_args" pytest -svv "tests/ttnn/unit_tests/base_functionality/test_multi_host_clusters.py::test_quad_galaxy_mesh_device_trace" ; fail+=$?

  # TODO: Currently failing on 1D/2D tests
  #tt-run --tcp-interface $tcp_interface --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=\"MultiHost.TestQuadGalaxy*\"" ; fail+=$?

  tt-run --tcp-interface $tcp_interface --rank-binding "$rank_binding" --mpi-args "$mpi_args" pytest -svv tests/nightly/tg/ccl/ -k "quad_host_mesh" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

###############################################################################
# Environment setup helpers (DeepSeek multihost: rankfile + hosts)
###############################################################################

_resolve_deepseekv3_cache() {
  local ci_cache="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"
  if [[ -n "${DEEPSEEK_V3_CACHE_OVERRIDE:-}" ]]; then
    local resolved
    resolved="$(realpath -m "${DEEPSEEK_V3_CACHE_OVERRIDE}")"
    local ci_resolved
    ci_resolved="$(realpath -m "${ci_cache}")"
    if [[ "${resolved}" == "${ci_resolved}" || "${resolved}" == "${ci_resolved}/"* ]]; then
      echo "Error: DEEPSEEK_V3_CACHE_OVERRIDE must not point to or inside the production CI cache (${ci_cache})." >&2
      exit 1
    fi
    export DEEPSEEK_V3_CACHE="${DEEPSEEK_V3_CACHE_OVERRIDE}"
  else
    export DEEPSEEK_V3_CACHE="${ci_cache}"
  fi
}

_extract_hosts_csv_from_rankfile() {
  local rankfile="$1"
  local expected_count="$2"

  if [[ ! -f "${rankfile}" ]]; then
    return 1
  fi

  local -a rank_hosts=()
  local line
  while IFS= read -r line; do
    if [[ "${line}" =~ ^[[:space:]]*rank[[:space:]]+[0-9]+=([^[:space:]]+) ]]; then
      rank_hosts+=("${BASH_REMATCH[1]}")
    fi
  done < "${rankfile}"

  if [[ ${#rank_hosts[@]} -lt ${expected_count} ]]; then
    return 1
  fi

  local -a selected_hosts=("${rank_hosts[@]:0:${expected_count}}")
  local hosts_csv
  hosts_csv="$(IFS=','; echo "${selected_hosts[*]}")"
  echo "${hosts_csv}"
}

_build_rankfile_from_hosts_csv() {
  local hosts_csv="$1"
  local expected_count="$2"
  local rankfile_path="$3"
  local hosts_error_hint="$4"

  local -a hosts=()
  IFS=',' read -r -a hosts <<< "${hosts_csv}"
  if [[ ${#hosts[@]} -ne ${expected_count} ]]; then
    echo "${hosts_error_hint} must contain exactly ${expected_count} comma-separated hosts." >&2
    exit 1
  fi

  : > "${rankfile_path}"
  echo "# mpirun rankfile" >> "${rankfile_path}"
  local rank_idx
  for rank_idx in "${!hosts[@]}"; do
    echo "rank ${rank_idx}=${hosts[$rank_idx]} slot=0:*" >> "${rankfile_path}"
  done
}

_resolve_dual_hosts_csv() {
  if [[ -n "${DUAL_MPI_HOSTS:-}" ]]; then
    echo "${DUAL_MPI_HOSTS}"
    return
  fi

  if [[ -n "${MPI_HOSTS:-}" ]]; then
    echo "${MPI_HOSTS}"
    return
  fi

  local source_rankfile="${DUAL_SOURCE_RANKFILE:-${DEFAULT_SOURCE_RANKFILE}}"
  local hosts_from_rankfile
  if hosts_from_rankfile="$(_extract_hosts_csv_from_rankfile "${source_rankfile}" 2)"; then
    echo "${hosts_from_rankfile}"
    return
  fi

  echo "${DEFAULT_DUAL_HOSTS_CSV}"
}

_resolve_quad_hosts_csv() {
  if [[ -n "${QUAD_MPI_HOSTS:-}" ]]; then
    echo "${QUAD_MPI_HOSTS}"
    return
  fi

  local source_rankfile="${QUAD_SOURCE_RANKFILE:-${DEFAULT_SOURCE_RANKFILE}}"
  local hosts_from_rankfile
  if hosts_from_rankfile="$(_extract_hosts_csv_from_rankfile "${source_rankfile}" 4)"; then
    echo "${hosts_from_rankfile}"
    return
  fi

  echo "${DEFAULT_QUAD_HOSTS_CSV}"
}

_resolve_upr_mode() {
  local upr_mode="${DEEPSEEK_DEMO_UPR_MODE:-all}"
  upr_mode="${upr_mode,,}"
  case "${upr_mode}" in
    ""|all|both)
      echo "both"
      ;;
    32|32upr|upr32)
      echo "32"
      ;;
    8|8upr|upr8)
      echo "8"
      ;;
    *)
      echo "Unsupported DEEPSEEK_DEMO_UPR_MODE='${DEEPSEEK_DEMO_UPR_MODE:-}'." >&2
      echo "Supported values: all|both|32|8" >&2
      exit 1
      ;;
  esac
}

_demo_case_selector() {
  local setup_name="$1"
  local profile_name="$2"
  local upr_mode
  upr_mode="$(_resolve_upr_mode)"

  case "${upr_mode}" in
    both)
      echo "${setup_name}_${profile_name}_demo_32upr or ${setup_name}_${profile_name}_demo_8upr"
      ;;
    32)
      echo "${setup_name}_${profile_name}_demo_32upr"
      ;;
    8)
      echo "${setup_name}_${profile_name}_demo_8upr"
      ;;
  esac
}

# Compute pytest --timeout value.
# When DEEPSEEK_V3_CACHE_OVERRIDE is set (cache recalculation), add 6 hours.
_demo_timeout() {
  local base_timeout="$1"
  local cache_extra=21600  # 6 hours
  if [[ -n "${DEEPSEEK_V3_CACHE_OVERRIDE:-}" ]]; then
    echo $((base_timeout + cache_extra))
  else
    echo "${base_timeout}"
  fi
}

setup_dual_galaxy_env() {
  export RANK_BINDING_YAML="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
  export HOSTS="$(_resolve_dual_hosts_csv)"
  export RANKFILE="$(mktemp "${TMPDIR:-/tmp}/rankfile_dual_deepseek_ci.XXXXXX")"
  TMP_RANKFILES+=("${RANKFILE}")
  _build_rankfile_from_hosts_csv "${HOSTS}" 2 "${RANKFILE}" "DUAL_MPI_HOSTS (or MPI_HOSTS)"
  export MPI_ARGS="--host ${HOSTS} --map-by rankfile:file=${RANKFILE} --bind-to none --output-filename logs/mpi_job"
  export TCP_INTERFACE="${TT_RUN_TCP_INTERFACE:-${DEFAULT_TCP_INTERFACE}}"
  export DUAL_MPI_HOSTS="${HOSTS}"
  mkdir -p logs
  mkdir -p generated/artifacts

  echo "Using dual Galaxy hosts: ${HOSTS}"
  echo "Using dual Galaxy rankfile: ${RANKFILE}"

  if ! test -f "${RANK_BINDING_YAML}"; then
    echo "File '${RANK_BINDING_YAML}' does not exist." >&2
    exit 1
  fi

  export DEEPSEEK_V3_HF_MODEL="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized"
  _resolve_deepseekv3_cache
  export MESH_DEVICE="DUAL"
  unset USE_TORUS_MODE
}

setup_quad_galaxy_env() {
  export RANK_BINDING_YAML="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
  export HOSTS="$(_resolve_quad_hosts_csv)"
  export RANKFILE="$(mktemp "${TMPDIR:-/tmp}/rankfile_quad_deepseek_ci.XXXXXX")"
  TMP_RANKFILES+=("${RANKFILE}")
  _build_rankfile_from_hosts_csv "${HOSTS}" 4 "${RANKFILE}" "QUAD_MPI_HOSTS"
  export MPI_ARGS="--host ${HOSTS} --map-by rankfile:file=${RANKFILE} --bind-to none --output-filename logs/mpi_job"
  export TCP_INTERFACE="${TT_RUN_TCP_INTERFACE:-${DEFAULT_TCP_INTERFACE}}"
  export QUAD_MPI_HOSTS="${HOSTS}"
  mkdir -p logs
  mkdir -p generated/artifacts

  if ! test -f "${RANK_BINDING_YAML}"; then
    echo "File '${RANK_BINDING_YAML}' does not exist." >&2
    exit 1
  fi

  export DEEPSEEK_V3_HF_MODEL="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized"
  _resolve_deepseekv3_cache
  export MESH_DEVICE="QUAD"
  export USE_TORUS_MODE=1
}

_run_deepseekv3_tt() {
  local mpi_args="${MPI_ARGS} -x DEEPSEEK_V3_HF_MODEL -x DEEPSEEK_V3_CACHE -x MESH_DEVICE"
  if [[ -n "${USE_TORUS_MODE:-}" ]]; then
    mpi_args="${mpi_args} -x USE_TORUS_MODE"
  fi

  tt-run \
    --tcp-interface "${TCP_INTERFACE}" \
    --rank-binding "${RANK_BINDING_YAML}" \
    --mpi-args "${mpi_args}" \
    "$@"
}

###############################################################################
# DeepSeek V3 unit tests (models/demos/deepseek_v3/tests/unit)
###############################################################################

run_dual_deepseekv3_unit_tests() {
  fail=0
  setup_dual_galaxy_env

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv models/demos/deepseek_v3/tests/unit 2>&1 | tee generated/artifacts/dual_deepseekv3_unit_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_quad_deepseekv3_unit_tests() {
  fail=0
  setup_quad_galaxy_env

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv models/demos/deepseek_v3/tests/unit 2>&1 | tee generated/artifacts/quad_deepseekv3_unit_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

###############################################################################
# DeepSeek V3 module tests (models/demos/deepseek_v3/tests)
###############################################################################

run_dual_deepseekv3_module_tests() {
  fail=0
  setup_dual_galaxy_env

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv models/demos/deepseek_v3/tests --ignore=models/demos/deepseek_v3/tests/unit --ignore=models/demos/deepseek_v3/tests/fused_op_unit_tests 2>&1 | tee generated/artifacts/dual_deepseekv3_module_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_quad_deepseekv3_module_tests() {
  fail=0
  setup_quad_galaxy_env

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv models/demos/deepseek_v3/tests --ignore=models/demos/deepseek_v3/tests/unit --ignore=models/demos/deepseek_v3/tests/fused_op_unit_tests 2>&1 | tee generated/artifacts/quad_deepseekv3_module_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

###############################################################################
# Teacher forced accuracy tests
###############################################################################

run_dual_teacher_forced_test() {
  fail=0
  setup_dual_galaxy_env
  local timeout=$(_demo_timeout 3600)

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo_teacher_forced.py::test_demo_teacher_forcing_accuracy 2>&1 | tee generated/artifacts/dual_teacher_forced_output.log" ; fail+=$?

  if [[ -f generated/artifacts/dual_teacher_forced_output.log ]]; then
    echo "Extracting accuracy metrics from test output..."
    grep -E "Top-1 accuracy:|Top-5 accuracy:" generated/artifacts/dual_teacher_forced_output.log > generated/artifacts/dual_teacher_forced_accuracy.txt || true
    echo "Accuracy metrics saved to generated/artifacts/dual_teacher_forced_accuracy.txt"
  fi

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_quad_teacher_forced_test() {
  fail=0
  setup_quad_galaxy_env
  local timeout=$(_demo_timeout 3600)

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo_teacher_forced.py::test_demo_teacher_forcing_accuracy 2>&1 | tee generated/artifacts/quad_teacher_forced_output.log" ; fail+=$?

  if [[ -f generated/artifacts/quad_teacher_forced_output.log ]]; then
    echo "Extracting accuracy metrics from test output..."
    grep -E "Top-1 accuracy:|Top-5 accuracy:" generated/artifacts/quad_teacher_forced_output.log > generated/artifacts/quad_teacher_forced_accuracy.txt || true
    echo "Accuracy metrics saved to generated/artifacts/quad_teacher_forced_accuracy.txt"
  fi

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

###############################################################################
# Demo tests (full)
###############################################################################

run_dual_demo_test() {
  fail=0
  setup_dual_galaxy_env
  local timeout=$(_demo_timeout 2400)
  local selector
  selector="$(_demo_case_selector "dual" "full")"

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo.py -k '${selector}' 2>&1 | tee generated/artifacts/dual_demo_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_dual_demo_mtp_test() {
  fail=0
  setup_dual_galaxy_env
  local timeout=$(_demo_timeout 2400)

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_mtp_demo.py 2>&1 | tee generated/artifacts/dual_demo_mtp_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_quad_demo_test() {
  fail=0
  setup_quad_galaxy_env
  local timeout=$(_demo_timeout 3600)
  local selector
  selector="$(_demo_case_selector "quad" "full")"

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo.py -k '${selector}' 2>&1 | tee generated/artifacts/quad_demo_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_quad_demo_mtp_test() {
  fail=0
  setup_quad_galaxy_env
  local timeout=$(_demo_timeout 3600)

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_mtp_demo.py 2>&1 | tee generated/artifacts/quad_demo_mtp_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

###############################################################################
# Demo stress tests
###############################################################################

run_dual_demo_stress_test() {
  fail=0
  setup_dual_galaxy_env
  local timeout=$(_demo_timeout 5400)
  local selector
  selector="$(_demo_case_selector "dual" "stress")"

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo.py -k '${selector}' 2>&1 | tee generated/artifacts/dual_demo_stress_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_quad_demo_stress_test() {
  fail=0
  setup_quad_galaxy_env
  local timeout=$(_demo_timeout 5400)
  local selector
  selector="$(_demo_case_selector "quad" "stress")"

  _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo.py -k '${selector}' 2>&1 | tee generated/artifacts/quad_demo_stress_output.log" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

###############################################################################
# Composite runners
###############################################################################

run_all_needed_local_tests() {
  local _saved_upr="${DEEPSEEK_DEMO_UPR_MODE:-}"
  export DEEPSEEK_DEMO_UPR_MODE=all
  run_dual_teacher_forced_test
  run_dual_demo_test
  run_dual_demo_stress_test
  run_quad_teacher_forced_test
  run_quad_demo_test
  run_quad_demo_stress_test
  if [[ -n "${_saved_upr}" ]]; then
    export DEEPSEEK_DEMO_UPR_MODE="${_saved_upr}"
  else
    unset DEEPSEEK_DEMO_UPR_MODE
  fi
}

run_dual_deepseekv3_integration_tests() {
  run_dual_deepseekv3_module_tests
  run_dual_teacher_forced_test
  run_dual_demo_test
  run_dual_demo_mtp_test
  run_dual_demo_stress_test
}

run_quad_deepseekv3_integration_tests() {
  run_quad_deepseekv3_module_tests
  run_quad_teacher_forced_test
  run_quad_demo_test
  run_quad_demo_mtp_test
  run_quad_demo_stress_test
}

run_quad_galaxy_tests() {
  run_quad_galaxy_unit_tests
  run_dual_deepseekv3_unit_tests
  run_quad_deepseekv3_unit_tests
  run_dual_deepseekv3_integration_tests
  run_quad_deepseekv3_integration_tests
}

###############################################################################
# Main dispatcher
###############################################################################

main() {
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  if [[ -z "${TT_METAL_HOME:-}" ]]; then
    local repo_root
    repo_root="$(pwd)"
    if [[ ! -f "${repo_root}/tests/scripts/multihost/run_quad_galaxy_tests.sh" ]]; then
      echo "Run this script from the tt-metal repository root or set TT_METAL_HOME." >&2
      exit 1
    fi
    export TT_METAL_HOME="${repo_root}"
  fi

  cd "$TT_METAL_HOME"
  export PYTHONPATH="$TT_METAL_HOME"

  local test_function="${1:-all}"
  local upr_mode_arg="${2:-}"
  if [[ -n "${upr_mode_arg}" && "${test_function}" != "all_needed_local_tests" ]]; then
    export DEEPSEEK_DEMO_UPR_MODE="${upr_mode_arg}"
    _resolve_upr_mode >/dev/null
    echo "Using demo UPR mode: $(_resolve_upr_mode)"
  fi

  case "$test_function" in
    "unit_tests")
      run_quad_galaxy_unit_tests
      ;;
    "dual_deepseekv3_unit_tests")
      run_dual_deepseekv3_unit_tests
      ;;
    "quad_deepseekv3_unit_tests")
      run_quad_deepseekv3_unit_tests
      ;;
    "dual_deepseekv3_module_tests")
      run_dual_deepseekv3_module_tests
      ;;
    "quad_deepseekv3_module_tests")
      run_quad_deepseekv3_module_tests
      ;;
    "dual_teacher_forced")
      run_dual_teacher_forced_test
      ;;
    "quad_teacher_forced")
      run_quad_teacher_forced_test
      ;;
    "dual_demo")
      run_dual_demo_test
      ;;
    "dual_demo_mtp")
      run_dual_demo_mtp_test
      ;;
    "quad_demo")
      run_quad_demo_test
      ;;
    "quad_demo_mtp")
      run_quad_demo_mtp_test
      ;;
    "dual_demo_stress")
      run_dual_demo_stress_test
      ;;
    "quad_demo_stress")
      run_quad_demo_stress_test
      ;;
    "dual_deepseekv3_integration_tests")
      run_dual_deepseekv3_integration_tests
      ;;
    "quad_deepseekv3_integration_tests")
      run_quad_deepseekv3_integration_tests
      ;;
    "all_needed_local_tests")
      run_all_needed_local_tests
      ;;
    "all")
      run_quad_galaxy_tests
      ;;
    "-h"|"--help")
      usage
      ;;
    *)
      echo "Unknown test function: $test_function" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
