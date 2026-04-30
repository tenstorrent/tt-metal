#!/bin/bash
set -eo pipefail

# Default ARCH_NAME for local runs when not set by CI.
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"

# Prefer tt-run on PATH; otherwise run ttrun.py directly.
tt_run() {
    if command -v tt-run >/dev/null 2>&1; then
        command tt-run "$@"
        return
    fi
    if [[ -z "${TT_METAL_HOME:-}" ]]; then
        echo "tt_run: tt-run not on PATH and TT_METAL_HOME is unset; cannot locate ttrun.py" >&2
        return 1
    fi
    local ttrun_py="${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py"
    if [[ ! -f "${ttrun_py}" ]]; then
        echo "tt_run: expected launcher at ${ttrun_py} (missing); install ttnn or set TT_METAL_HOME" >&2
        return 1
    fi
    "${PYTHON:-python3}" "${ttrun_py}" "$@"
}

# Pick cnx1 when present, else first up non-virtual NIC.
default_mpi_tcp_interface() {
    if [[ -d /sys/class/net/cnx1 ]]; then
        echo "cnx1"
        return 0
    fi
    local n state
    for n in /sys/class/net/*; do
        n="${n##*/}"
        case "${n}" in
            lo | docker* | br-* | veth* | tailscale*) continue ;;
        esac
        state="$(cat "/sys/class/net/${n}/operstate" 2>/dev/null || true)"
        if [[ "${state}" == "up" ]]; then
            echo "${n}"
            return 0
        fi
    done
    echo "cnx1"
}

export_tcp_interface_for_multihost() {
    export TCP_INTERFACE="${TCP_INTERFACE:-$(default_mpi_tcp_interface)}"
}

extract_hosts_from_hostfile() {
    local host_count="$1"
    local hostfile="${2:-/etc/mpirun/hostfile}"

    awk '!/^#/ && NF {print $1}' "$hostfile" | head -n "$host_count" | paste -sd,
}

###############################################################################
# Infrastructure unit tests (quad galaxy only)
###############################################################################

run_quad_galaxy_unit_tests() {
  fail=0

  export_tcp_interface_for_multihost
  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile"
  local tcp_interface="${TCP_INTERFACE}"
  local hosts="$(extract_hosts_from_hostfile 4)"
  local rank_binding_yaml="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
  local tt_mpi_args="--host $hosts --map-by rankfile:file=/etc/mpirun/rankfile --bind-to none --output-filename logs/mpi_job"
  local mpi_host="--host $hosts"
  local mpirun_args_base="$mpi_args_base --mca btl self,tcp --mca btl_tcp_if_include ${tcp_interface} --tag-output"
  local mpirun_args="$mpi_host $mpirun_args_base"

  local mesh_graph="tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto"
  local descriptor_path="${DESCRIPTOR_PATH:-/etc/mpirun}"

  # TODO: Currently failing
  #mpirun-ulfm $mpi_run_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?

  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation ; fail+=$?

  tt_run --tcp-interface "$tcp_interface" --rank-binding "$rank_binding_yaml" --mpi-args "$tt_mpi_args" pytest -svv "tests/ttnn/unit_tests/base_functionality/test_multi_host_clusters.py::test_quad_galaxy_mesh_device_trace" ; fail+=$?

  # TODO: Currently failing on 1D/2D tests
  #tt_run --tcp-interface $tcp_interface --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" bash -c "./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=\"MultiHost.TestQuadGalaxy*\"" ; fail+=$?

  tt_run --tcp-interface "$tcp_interface" --rank-binding "$rank_binding_yaml" --mpi-args "$tt_mpi_args" pytest -svv tests/nightly/tg/ccl/ -k "quad_host_mesh" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

###############################################################################
# Environment setup helpers
###############################################################################

# Kernel cache must be local per host (not NFS/shared home) to avoid multihost races.
_ensure_local_tt_metal_cache() {
    unset TT_METAL_CACHE 2>/dev/null || true
    local rid="${GITHUB_RUN_ID:-$$}"
    export TT_METAL_CACHE="${TMPDIR:-/tmp}/tt_metal_kernel_cache_${rid}"
    mkdir -p "${TT_METAL_CACHE}"
}

# MLPerf weight cache is read-only in CI. Module-test jobs set MULTIHOST_DS_V3_WEIGHT_CACHE=1;
# demos omit it so DEEPSEEK_V3_CACHE stays unset unless explicitly overridden elsewhere.
_resolve_deepseekv3_cache() {
    local ci_cache="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"

    if [[ "${MULTIHOST_DS_V3_WEIGHT_CACHE:-0}" != "1" ]]; then
        unset DEEPSEEK_V3_CACHE 2>/dev/null || true
        unset DEEPSEEK_V3_CACHE_OVERRIDE 2>/dev/null || true
        return 0
    fi

    unset DEEPSEEK_V3_CACHE 2>/dev/null || true

    if [[ -n "${DEEPSEEK_V3_CACHE_OVERRIDE:-}" ]]; then
        local resolved
        resolved=$(realpath -m "${DEEPSEEK_V3_CACHE_OVERRIDE}")
        local ci_resolved
        ci_resolved=$(realpath -m "${ci_cache}")
        if [[ "${resolved}" == "${ci_resolved}" || "${resolved}" == "${ci_resolved}/"* ]]; then
            echo "Error: DEEPSEEK_V3_CACHE_OVERRIDE must not point to or inside the production CI cache (${ci_cache})." >&2
            exit 1
        fi
        export DEEPSEEK_V3_CACHE="${DEEPSEEK_V3_CACHE_OVERRIDE}"
    else
        export DEEPSEEK_V3_CACHE="${ci_cache}"
    fi
}

resolve_deepseekv3_model() {
    local default_model="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked"
    local model_path="${DEEPSEEK_V3_HF_MODEL_OVERRIDE:-${DEEPSEEK_V3_HF_MODEL:-${default_model}}}"

    if [[ ! -d "${model_path}" ]]; then
        echo "Warning: DeepSeek V3 model directory not visible from orchestrator: ${model_path}" >&2
        echo "  This is expected in CI Docker containers; model must exist on Galaxy hosts." >&2
        echo "  For local testing, pass --model-path <path> (or set DEEPSEEK_V3_HF_MODEL_OVERRIDE)." >&2
    fi

    export DEEPSEEK_V3_HF_MODEL="${model_path}"
    echo "Using DeepSeek V3 model: ${DEEPSEEK_V3_HF_MODEL}"
}

setup_dual_galaxy_env() {
    _ensure_local_tt_metal_cache
    export RANK_BINDING_YAML="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
    export MESH_GRAPH_DESCRIPTOR="tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto"
    export HOSTS="$(extract_hosts_from_hostfile 2)"
    export RANKFILE=/etc/mpirun/rankfile
    export MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --bind-to none --output-filename logs/mpi_job"
    export_tcp_interface_for_multihost
    mkdir -p logs
    mkdir -p generated/artifacts

    echo "Using dual Galaxy hosts: ${HOSTS}"
    echo "Using MPI TCP interface (tt-run / Open MPI): ${TCP_INTERFACE}"
    echo "Using dual Galaxy rankfile: ${RANKFILE}"

    if ! test -f "$RANKFILE"; then
        echo "File '$RANKFILE' does not exist."
        exit 1
    fi
    if ! test -f "$RANK_BINDING_YAML"; then
        echo "File '$RANK_BINDING_YAML' does not exist."
        exit 1
    fi
    if ! test -f "$MESH_GRAPH_DESCRIPTOR"; then
        echo "File '$MESH_GRAPH_DESCRIPTOR' does not exist."
        exit 1
    fi

    resolve_deepseekv3_model
    _resolve_deepseekv3_cache
    export MESH_DEVICE="DUAL"
    export USE_TORUS_MODE=0
    echo "Dual Galaxy: USE_TORUS_MODE=0 (torus/ring mode disabled)."
}

setup_quad_galaxy_env() {
    _ensure_local_tt_metal_cache
    export RANK_BINDING_YAML="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
    export MESH_GRAPH_DESCRIPTOR="tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto"
    export HOSTS="$(extract_hosts_from_hostfile 4)"
    export RANKFILE=/etc/mpirun/rankfile
    export MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --bind-to none --output-filename logs/mpi_job"
    export_tcp_interface_for_multihost
    mkdir -p logs
    mkdir -p generated/artifacts

    if ! test -f "$RANKFILE"; then
        echo "File '$RANKFILE' does not exist."
        exit 1
    fi
    if ! test -f "$RANK_BINDING_YAML"; then
        echo "File '$RANK_BINDING_YAML' does not exist."
        exit 1
    fi
    if ! test -f "$MESH_GRAPH_DESCRIPTOR"; then
        echo "File '$MESH_GRAPH_DESCRIPTOR' does not exist."
        exit 1
    fi

    echo "Using quad Galaxy hosts: ${HOSTS}"
    echo "Using MPI TCP interface (tt-run / Open MPI): ${TCP_INTERFACE}"
    echo "Using quad Galaxy rankfile: ${RANKFILE}"

    resolve_deepseekv3_model
    _resolve_deepseekv3_cache
    export MESH_DEVICE="QUAD"

    # DS_QUAD_USE_TORUS_MODE defaults to 1; set 0 or --no-torus to disable torus mode.
    local ds_quad_torus="${DS_QUAD_USE_TORUS_MODE:-1}"
    ds_quad_torus="${ds_quad_torus,,}"
    case "${ds_quad_torus}" in
        ""|1|true|yes|on)
            export USE_TORUS_MODE=1
            echo "Quad Galaxy: USE_TORUS_MODE=1 (DeepSeek V3 torus/ring mode enabled)."
            ;;
        0|false|no|off)
            export USE_TORUS_MODE=0
            echo "Quad Galaxy: USE_TORUS_MODE=0 (DeepSeek V3 torus/ring mode disabled)."
            ;;
        *)
            echo "Error: unsupported DS_QUAD_USE_TORUS_MODE='${DS_QUAD_USE_TORUS_MODE:-}' (use 1|0|true|false|yes|no|on|off)." >&2
            exit 1
            ;;
    esac
}

# Compute pytest --timeout value.
# When DEEPSEEK_V3_CACHE_OVERRIDE is set (custom DeepSeek cache dir), add 6 hours.
_demo_timeout() {
    local base_timeout=$1
    local cache_extra=21600  # 6 hours
    if [[ -n "${DEEPSEEK_V3_CACHE_OVERRIDE:-}" ]]; then
        echo $(( base_timeout + cache_extra ))
    else
        echo "$base_timeout"
    fi
}

resolve_upr_mode() {
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

demo_case_selector() {
    local setup_name="$1"
    local profile_name="$2"
    local upr_mode
    upr_mode="$(resolve_upr_mode)"

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

teacher_forced_pytest_args() {
    local test_node="models/demos/deepseek_v3/demo/test_demo_teacher_forced.py::test_demo_teacher_forcing_accuracy"
    local upr_mode
    upr_mode="$(resolve_upr_mode)"

    if [[ "${upr_mode}" == "both" ]]; then
        echo "${test_node}"
        return 0
    fi

    # Parametrize values from test_demo_teacher_forced.py:
    #   max_users_per_row: [8, 32]  ids=["8", "32"]
    #   max_new_tokens:    [128, 2048, 8192]  ids=["128", "2048", "8192"]
    #   reference_file:    [REFERENCE_FILE]
    # Node ID format: test_demo_teacher_forcing_accuracy[{upr}-{tokens}-reference_file0]
    local token_count
    for token_count in 128 2048 8192; do
        echo "${test_node}[${upr_mode}-${token_count}-reference_file0]"
    done
}

# Helper: run a test command via tt-run using the current environment
_run_deepseekv3_tt() {
    local -a tt_run_args=(--tcp-interface "$TCP_INTERFACE" --rank-binding "$RANK_BINDING_YAML")
    if [[ -n "${MPI_ARGS:-}" ]]; then
        tt_run_args+=(--mpi-args "$MPI_ARGS")
    fi

    local runtime_root="${TT_METAL_RUNTIME_ROOT:-${TT_METAL_HOME:-$(pwd -P)}}"
    local torus_mode="${USE_TORUS_MODE-__UNSET__}"

    tt_run "${tt_run_args[@]}" env \
        _DS_MESH_DEVICE="${MESH_DEVICE:-}" \
        _DS_USE_TORUS_MODE="${torus_mode}" \
        _DS_RUNTIME_ROOT="${runtime_root}" \
        bash -c '
            if [[ -n "${_DS_MESH_DEVICE}" ]]; then
                export MESH_DEVICE="${_DS_MESH_DEVICE}"
            fi
            if [[ "${_DS_USE_TORUS_MODE}" == "0" || "${_DS_USE_TORUS_MODE}" == "__UNSET__" ]]; then
                unset USE_TORUS_MODE
            elif [[ -n "${_DS_USE_TORUS_MODE}" ]]; then
                export USE_TORUS_MODE="${_DS_USE_TORUS_MODE}"
            fi
            if [[ -n "${_DS_RUNTIME_ROOT}" ]]; then
                export TT_METAL_RUNTIME_ROOT="${_DS_RUNTIME_ROOT}"
            fi
            exec "$@"
        ' _ "$@"
}

###############################################################################
# DeepSeek V3 unit tests (models/demos/deepseek_v3/tests/unit)
###############################################################################

run_dual_deepseekv3_unit_tests() {
    fail=0
    setup_dual_galaxy_env

    _run_deepseekv3_tt pytest -svvv models/demos/deepseek_v3/tests/unit ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

run_quad_deepseekv3_unit_tests() {
    fail=0
    setup_quad_galaxy_env

    _run_deepseekv3_tt pytest -svvv models/demos/deepseek_v3/tests/unit ; fail+=$?

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

    _run_deepseekv3_tt pytest -svvv models/demos/deepseek_v3/tests --ignore=models/demos/deepseek_v3/tests/unit --ignore=models/demos/deepseek_v3/tests/fused_op_unit_tests ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

run_quad_deepseekv3_module_tests() {
    fail=0
    setup_quad_galaxy_env

    _run_deepseekv3_tt pytest -svvv models/demos/deepseek_v3/tests --ignore=models/demos/deepseek_v3/tests/unit --ignore=models/demos/deepseek_v3/tests/fused_op_unit_tests ; fail+=$?

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
    local tf_args
    tf_args="$(teacher_forced_pytest_args | paste -sd' ')"

    _run_deepseekv3_tt bash -c "set -f -o pipefail; pytest -svvv --timeout=$timeout ${tf_args} 2>&1 | tee generated/artifacts/dual_teacher_forced_output.log" ; fail+=$?

    # Extract accuracy metrics from logs and save to artifact file
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
    local tf_args
    tf_args="$(teacher_forced_pytest_args | paste -sd' ')"

    _run_deepseekv3_tt bash -c "set -f -o pipefail; pytest -svvv --timeout=$timeout ${tf_args} 2>&1 | tee generated/artifacts/quad_teacher_forced_output.log" ; fail+=$?

    # Extract accuracy metrics from logs and save to artifact file
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
    setup_dual_galaxy_env
    local timeout=$(_demo_timeout 2400)
    local selector
    selector="$(demo_case_selector "dual" "full")"

    _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo.py -k '$selector' 2>&1 | tee generated/artifacts/dual_demo_output.log"
}

run_dual_demo_mtp_test() {
    setup_dual_galaxy_env
    local timeout=$(_demo_timeout 2400)

    _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_mtp_demo.py 2>&1 | tee generated/artifacts/dual_demo_mtp_output.log"

}

run_quad_demo_test() {
    setup_quad_galaxy_env
    local timeout=$(_demo_timeout 3600)
    local selector
    selector="$(demo_case_selector "quad" "full")"

    _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo.py -k '$selector' 2>&1 | tee generated/artifacts/quad_demo_output.log"
}

run_quad_demo_mtp_test() {
    setup_quad_galaxy_env
    local timeout=$(_demo_timeout 3600)

    _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_mtp_demo.py 2>&1 | tee generated/artifacts/quad_demo_mtp_output.log"
}

###############################################################################
# Demo stress tests
###############################################################################

run_dual_demo_stress_test() {
    setup_dual_galaxy_env
    local timeout=$(_demo_timeout 5400)
    local selector
    selector="$(demo_case_selector "dual" "stress")"

    _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo.py -k '$selector' 2>&1 | tee generated/artifacts/dual_demo_stress_output.log"
}

run_quad_demo_stress_test() {
    setup_quad_galaxy_env
    local timeout=$(_demo_timeout 5400)
    local selector
    selector="$(demo_case_selector "quad" "stress")"

    _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo.py -k '$selector' 2>&1 | tee generated/artifacts/quad_demo_stress_output.log"
}

###############################################################################
# Composite runners
###############################################################################

# All dual galaxy deepseek v3 integration tests
run_dual_deepseekv3_integration_tests() {
    run_dual_deepseekv3_module_tests
    run_dual_teacher_forced_test
    run_dual_demo_test
    run_dual_demo_mtp_test
    run_dual_demo_stress_test
}

# All quad galaxy deepseek v3 integration tests
run_quad_deepseekv3_integration_tests() {
    run_quad_deepseekv3_module_tests
    run_quad_teacher_forced_test
    run_quad_demo_test
    run_quad_demo_mtp_test
    run_quad_demo_stress_test
}

run_all_needed_local_tests() {
    local saved_upr_mode="${DEEPSEEK_DEMO_UPR_MODE:-}"
    export DEEPSEEK_DEMO_UPR_MODE="all"

    run_dual_teacher_forced_test
    run_dual_demo_test
    run_dual_demo_stress_test
    run_quad_teacher_forced_test
    run_quad_demo_test
    run_quad_demo_stress_test

    if [[ -n "${saved_upr_mode}" ]]; then
        export DEEPSEEK_DEMO_UPR_MODE="${saved_upr_mode}"
    else
        unset DEEPSEEK_DEMO_UPR_MODE
    fi
}

# Run everything
run_quad_galaxy_tests() {
    run_quad_galaxy_unit_tests
    run_dual_deepseekv3_unit_tests
    run_quad_deepseekv3_unit_tests
    run_dual_deepseekv3_integration_tests
    run_quad_deepseekv3_integration_tests
}

###############################################################################
# Environment setup for local and CI-compatible multihost runs.
###############################################################################

set_multihost_pythonhome_if_needed() {
    # Force PYTHONHOME when copied venv metadata points to host-local interpreters.
    if [[ "${MULTIHOST_FORCE_PYTHONHOME:-1}" == "0" ]]; then
        return 0
    fi

    local pyhome="${TT_METAL_HOME}/python_env"
    local -a encodings_candidates=()
    shopt -s nullglob
    encodings_candidates=("${pyhome}"/lib/python*/encodings/__init__.py)
    shopt -u nullglob

    if [[ ${#encodings_candidates[@]} -eq 0 ]]; then
        return 0
    fi

    if [[ "${PYTHONHOME:-}" != "${pyhome}" ]]; then
        export PYTHONHOME="${pyhome}"
        echo "Using PYTHONHOME override for multihost ranks: ${PYTHONHOME}"
    fi
}

init_multihost_test_env() {
    export TT_METAL_HOME="$(cd -- "${TT_METAL_HOME}" && pwd -P)"
    cd "${TT_METAL_HOME}"

    export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
    export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

    local reports_dir="${TT_METAL_HOME}/generated/test_reports"
    export GTEST_OUTPUT="${GTEST_OUTPUT:-xml:${reports_dir}/}"
    mkdir -p "${reports_dir}"

    if [[ "${MULTIHOST_MATCH_CI_HOME:-}" == "1" ]]; then
        export HOME="${TT_METAL_HOME}"
    fi

    if [[ "${MULTIHOST_SKIP_SHARED_VENV:-0}" == "1" ]]; then
        set_multihost_pythonhome_if_needed
        return 0
    fi

    local setup_venv_script="${TT_METAL_HOME}/tests/scripts/multihost/setup_shared_venv.sh"
    local py_env_dir="${TT_METAL_HOME}/python_env"
    local source_venv="${MULTIHOST_SOURCE_VENV:-/opt/venv}"
    if [[ ! -x "${setup_venv_script}" ]]; then
        echo "Warning: ${setup_venv_script} is missing; skipping shared venv activation." >&2
        set_multihost_pythonhome_if_needed
        return 0
    fi
    if [[ -d "${py_env_dir}" ]] || [[ -d "${source_venv}" ]]; then
        eval "$("${setup_venv_script}" --activate "${source_venv}" "${py_env_dir}")"
    else
        echo "Warning: neither ${py_env_dir} nor ${source_venv} exists; skipping setup_shared_venv.sh. Use a venv with ttnn installed or set MULTIHOST_SOURCE_VENV." >&2
    fi

    set_multihost_pythonhome_if_needed
}

###############################################################################
# Main dispatcher
###############################################################################

main() {
    # For CI pipeline - source func commands but don't execute tests if not invoked directly
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        echo "Script is being sourced, not executing main function"
        return 0
    fi

    if [[ -z "${TT_METAL_HOME:-}" ]]; then
        export TT_METAL_HOME="$(pwd -P)"
        echo "TT_METAL_HOME not set; defaulting to current directory: ${TT_METAL_HOME}"
    fi

    init_multihost_test_env

    # Args: [test_function] [upr_mode] plus --no-torus/--model-path/--cache-path.
    local test_function="all"
    if [[ $# -gt 0 ]]; then
        test_function="$1"
        shift
    fi

    local upr_mode_arg=""
    if [[ $# -gt 0 && "${1}" != --* ]]; then
        upr_mode_arg="$1"
        shift
    fi
    if [[ -n "${upr_mode_arg}" ]]; then
        export DEEPSEEK_DEMO_UPR_MODE="${upr_mode_arg}"
        resolve_upr_mode >/dev/null
        echo "Using demo UPR mode: $(resolve_upr_mode)"
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-torus)
                export DS_QUAD_USE_TORUS_MODE=0
                shift
                ;;
            --model-path)
                if [[ $# -lt 2 ]]; then
                    echo "Error: --model-path requires a value." >&2
                    exit 1
                fi
                export DEEPSEEK_V3_HF_MODEL_OVERRIDE="$2"
                shift 2
                ;;
            --cache-path)
                if [[ $# -lt 2 ]]; then
                    echo "Error: --cache-path requires a value." >&2
                    exit 1
                fi
                export DEEPSEEK_V3_CACHE_OVERRIDE="$2"
                shift 2
                ;;
            *)
                echo "Unknown argument: $1" >&2
                echo "Usage: $0 [test_function] [upr_mode] [--no-torus] [--model-path <path>] [--cache-path <path>]" >&2
                exit 1
                ;;
        esac
    done

    if [[ -n "${DEEPSEEK_V3_HF_MODEL_OVERRIDE:-}" ]]; then
        echo "Using local DeepSeek model override: ${DEEPSEEK_V3_HF_MODEL_OVERRIDE}"
    fi
    if [[ -n "${DEEPSEEK_V3_CACHE_OVERRIDE:-}" ]]; then
        echo "Using local DeepSeek cache override: ${DEEPSEEK_V3_CACHE_OVERRIDE}"
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
        *)
            echo "Unknown test function: $test_function" 1>&2
            echo "Available options: unit_tests, dual_deepseekv3_unit_tests, quad_deepseekv3_unit_tests, dual_deepseekv3_module_tests, quad_deepseekv3_module_tests, dual_teacher_forced, quad_teacher_forced, dual_demo, dual_demo_mtp, quad_demo, quad_demo_mtp, dual_demo_stress, quad_demo_stress, dual_deepseekv3_integration_tests, quad_deepseekv3_integration_tests, all_needed_local_tests, all" 1>&2
            echo "Optional second argument: UPR mode (all|32|8)" 1>&2
            echo "Optional flags: --no-torus  --model-path <path>  --cache-path <path>" 1>&2
            echo "Example: $0 quad_demo 32 --no-torus --model-path /data/deepseek/DeepSeek-R1-0528-dequantized-stacked --cache-path /data/deepseek/DeepSeek-R1-0528-Cache/CI" 1>&2
            exit 1
            ;;
    esac
}

main "$@"
