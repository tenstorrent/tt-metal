#!/bin/bash
set -eo pipefail

# Default ARCH_NAME for local runs when not set by CI.
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"

# Prefer `tt-run` on PATH (from pip/editable install). Otherwise run `ttrun.py` directly so
# multihost scripts work when console scripts are not installed (avoids importing `ttnn` package
# root, which is unnecessary for the launcher).
_tt_run() {
    if command -v tt-run >/dev/null 2>&1; then
        command tt-run "$@"
        return
    fi
    if [[ -z "${TT_METAL_HOME:-}" ]]; then
        echo "_tt_run: tt-run not on PATH and TT_METAL_HOME is unset; cannot locate ttrun.py" >&2
        return 1
    fi
    local _ttrun_py="${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py"
    if [[ ! -f "${_ttrun_py}" ]]; then
        echo "_tt_run: expected launcher at ${_ttrun_py} (missing); install ttnn or set TT_METAL_HOME" >&2
        return 1
    fi
    "${PYTHON:-python3}" "${_ttrun_py}" "$@"
}

# MPI/OpenFabrics TCP binding: CI / many Galaxy boxes use "cnx1". If it is missing, pick the
# first non-virtual interface in operstate up (skips lo, docker, tailscale, …) so local clusters
# like UF-* hosts get a valid btl_tcp_if_include without manual export.
_default_mpi_tcp_interface() {
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

_export_tcp_interface_for_multihost() {
    export TCP_INTERFACE="${TCP_INTERFACE:-$(_default_mpi_tcp_interface)}"
}

_extract_hosts_from_hostfile() {
    local host_count="$1"
    local hostfile="${2:-/etc/mpirun/hostfile}"

    awk '!/^#/ && NF {print $1}' "$hostfile" | head -n "$host_count" | paste -sd,
}

###############################################################################
# Infrastructure unit tests (quad galaxy only)
###############################################################################

run_quad_galaxy_unit_tests() {
  fail=0

  _export_tcp_interface_for_multihost
  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile"
  local tcp_interface="${TCP_INTERFACE}"
  local hosts="$(_extract_hosts_from_hostfile 4)"
  local rank_binding_yaml="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
  local tt_mpi_args="--host $hosts --map-by rankfile:file=/etc/mpirun/rankfile --bind-to none --output-filename logs/mpi_job"
  local mpi_host="--host $hosts"
  local mpirun_args_base="$mpi_args_base --mca btl self,tcp --mca btl_tcp_if_include ${tcp_interface} --tag-output"
  local mpirun_args="$mpi_host $mpirun_args_base"

  local mesh_graph="tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto"
  local descriptor_path="${DESCRIPTOR_PATH:-/etc/mpirun}"

  # TODO: Currently failing
  #mpirun-ulfm $mpi_run_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?

  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation --send-traffic --cabling-descriptor-path ${descriptor_path}/cabling_descriptor.textproto --deployment-descriptor-path ${descriptor_path}/deployment_descriptor.textproto ; fail+=$?

  _tt_run --tcp-interface "$tcp_interface" --rank-binding "$rank_binding_yaml" --mpi-args "$tt_mpi_args" pytest -svv "tests/ttnn/unit_tests/base_functionality/test_multi_host_clusters.py::test_quad_galaxy_mesh_device_trace" ; fail+=$?

  # TODO: Currently failing on 1D/2D tests
  #_tt_run --tcp-interface $tcp_interface --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" bash -c "./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=\"MultiHost.TestQuadGalaxy*\"" ; fail+=$?

  _tt_run --tcp-interface "$tcp_interface" --rank-binding "$rank_binding_yaml" --mpi-args "$tt_mpi_args" pytest -svv tests/nightly/tg/ccl/ -k "quad_host_mesh" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

###############################################################################
# Environment setup helpers
###############################################################################

_resolve_deepseekv3_cache() {
    local ci_cache="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"
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

_resolve_deepseekv3_model() {
    local default_model="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized"
    local model_path="${DEEPSEEK_V3_HF_MODEL_OVERRIDE:-${DEEPSEEK_V3_HF_MODEL:-${default_model}}}"

    if [[ ! -d "${model_path}" ]]; then
        echo "Error: DeepSeek V3 model directory does not exist: ${model_path}" >&2
        echo "For local testing, pass --model-path <path> (or set DEEPSEEK_V3_HF_MODEL_OVERRIDE)." >&2
        exit 1
    fi

    export DEEPSEEK_V3_HF_MODEL="${model_path}"
    echo "Using DeepSeek V3 model: ${DEEPSEEK_V3_HF_MODEL}"
}

setup_dual_galaxy_env() {
    export RANK_BINDING_YAML="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
    export MESH_GRAPH_DESCRIPTOR="tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto"
    export HOSTS="$(_extract_hosts_from_hostfile 2)"
    export RANKFILE=/etc/mpirun/rankfile
    export MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --bind-to none --output-filename logs/mpi_job"
    _export_tcp_interface_for_multihost
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

    _resolve_deepseekv3_model
    _resolve_deepseekv3_cache
    export MESH_DEVICE="DUAL"
    export USE_TORUS_MODE=0
    echo "Dual Galaxy: USE_TORUS_MODE=0 (torus/ring mode disabled)."
}

setup_quad_galaxy_env() {
    export RANK_BINDING_YAML="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
    export MESH_GRAPH_DESCRIPTOR="tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto"
    export HOSTS="$(_extract_hosts_from_hostfile 4)"
    export RANKFILE=/etc/mpirun/rankfile
    export MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --bind-to none --output-filename logs/mpi_job"
    _export_tcp_interface_for_multihost
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

    _resolve_deepseekv3_model
    _resolve_deepseekv3_cache
    export MESH_DEVICE="QUAD"

    # DeepSeek V3 currently interprets any set USE_TORUS_MODE as torus/ring.
    # This script uses USE_TORUS_MODE=0 as an explicit OFF sentinel; _run_deepseekv3_tt
    # unsets it for child test processes so OFF works while still overriding ambient env.
    # Default is ON; set DS_QUAD_USE_TORUS_MODE=0 or pass --no-torus to disable.
    local _ds_quad_torus="${DS_QUAD_USE_TORUS_MODE:-1}"
    _ds_quad_torus="${_ds_quad_torus,,}"
    case "${_ds_quad_torus}" in
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
# When DEEPSEEK_V3_CACHE_OVERRIDE is set (cache recalculation), add 6 hours.
_demo_timeout() {
    local base_timeout=$1
    local cache_extra=21600  # 6 hours
    if [[ -n "${DEEPSEEK_V3_CACHE_OVERRIDE:-}" ]]; then
        echo $(( base_timeout + cache_extra ))
    else
        echo "$base_timeout"
    fi
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

# Helper: run a test command via tt-run using the current environment
_run_deepseekv3_tt() {
    # DeepSeek uses presence-only checks on USE_TORUS_MODE today.
    # Treat "0" as explicit OFF by unsetting for the launched process.
    if [[ "${USE_TORUS_MODE:-}" == "0" ]]; then
        if [[ -n "${MPI_ARGS:-}" ]]; then
            ( unset USE_TORUS_MODE; _tt_run --tcp-interface "$TCP_INTERFACE" --rank-binding "$RANK_BINDING_YAML" --mpi-args "$MPI_ARGS" "$@" )
        else
            ( unset USE_TORUS_MODE; _tt_run --tcp-interface "$TCP_INTERFACE" --rank-binding "$RANK_BINDING_YAML" "$@" )
        fi
        return
    fi

    if [[ -n "${MPI_ARGS:-}" ]]; then
        _tt_run --tcp-interface "$TCP_INTERFACE" --rank-binding "$RANK_BINDING_YAML" --mpi-args "$MPI_ARGS" "$@"
    else
        _tt_run --tcp-interface "$TCP_INTERFACE" --rank-binding "$RANK_BINDING_YAML" "$@"
    fi
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

    _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo_teacher_forced.py::test_demo_teacher_forcing_accuracy 2>&1 | tee generated/artifacts/dual_teacher_forced_output.log" ; fail+=$?

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

    _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo_teacher_forced.py::test_demo_teacher_forcing_accuracy 2>&1 | tee generated/artifacts/quad_teacher_forced_output.log" ; fail+=$?

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
    selector="$(_demo_case_selector "dual" "full")"

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
    selector="$(_demo_case_selector "quad" "full")"

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
    selector="$(_demo_case_selector "dual" "stress")"

    _run_deepseekv3_tt bash -c "set -o pipefail; pytest -svvv --timeout=$timeout models/demos/deepseek_v3/demo/test_demo.py -k '$selector' 2>&1 | tee generated/artifacts/dual_demo_stress_output.log"
}

run_quad_demo_stress_test() {
    setup_quad_galaxy_env
    local timeout=$(_demo_timeout 5400)
    local selector
    selector="$(_demo_case_selector "quad" "stress")"

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
# Environment - prepare a runnable multihost test environment
#
# Goal: make local runs work out of the box while keeping CI compatibility.
# CI sets most of these variables externally; local runs can rely on this
# script to fill in missing defaults.
#
# Optional overrides:
#   MULTIHOST_SKIP_SHARED_VENV=1     - skip setup_shared_venv.sh activation
#   MULTIHOST_SOURCE_VENV=/path      - source venv for setup_shared_venv.sh
#   MULTIHOST_MATCH_CI_HOME=1        - export HOME=\$TT_METAL_HOME (CI parity)
#   MULTIHOST_FORCE_PYTHONHOME=0     - disable PYTHONHOME override for ranks
#   DS_QUAD_USE_TORUS_MODE=0         - quad DeepSeek runs: set USE_TORUS_MODE=0 (debug /
#                                    linear fabric). Default is 1. Also:
#                                    pass --no-torus on the command line.
###############################################################################

_set_multihost_pythonhome_if_needed() {
    # Some physical hosts cannot resolve the base interpreter from copied venv metadata
    # (for example when pyvenv.cfg points to a host-local uv install path). Explicitly
    # setting PYTHONHOME to the shared venv keeps Python startup deterministic on all ranks.
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

_init_multihost_test_env() {
    export TT_METAL_HOME="$(cd -- "${TT_METAL_HOME}" && pwd -P)"
    cd "${TT_METAL_HOME}"

    export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
    export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

    local _reports="${TT_METAL_HOME}/generated/test_reports"
    export GTEST_OUTPUT="${GTEST_OUTPUT:-xml:${_reports}/}"
    mkdir -p "${_reports}"

    if [[ "${MULTIHOST_MATCH_CI_HOME:-}" == "1" ]]; then
        export HOME="${TT_METAL_HOME}"
    fi

    if [[ "${MULTIHOST_SKIP_SHARED_VENV:-0}" == "1" ]]; then
        _set_multihost_pythonhome_if_needed
        return 0
    fi

    local _setup_venv="${TT_METAL_HOME}/tests/scripts/multihost/setup_shared_venv.sh"
    local _py_env="${TT_METAL_HOME}/python_env"
    local _src_venv="${MULTIHOST_SOURCE_VENV:-/opt/venv}"
    if [[ ! -x "${_setup_venv}" ]]; then
        echo "Warning: ${_setup_venv} is missing; skipping shared venv activation." >&2
        _set_multihost_pythonhome_if_needed
        return 0
    fi
    if [[ -d "${_py_env}" ]] || [[ -d "${_src_venv}" ]]; then
        eval "$("${_setup_venv}" --activate "${_src_venv}" "${_py_env}")"
    else
        echo "Warning: neither ${_py_env} nor ${_src_venv} exists; skipping setup_shared_venv.sh. Use a venv with ttnn installed or set MULTIHOST_SOURCE_VENV." >&2
    fi

    _set_multihost_pythonhome_if_needed
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

    _init_multihost_test_env

    # Support running specific test function via argument.
    # Positional compatibility:
    #   $1 test function (default: all)
    #   $2 UPR mode (all|32|8), optional when it is not an option flag
    # Optional local-testing flags:
    #   --no-torus                       - quad DeepSeek: set USE_TORUS_MODE=0 (see DS_QUAD_USE_TORUS_MODE)
    #   --model-path <path>
    #   --cache-path <path>
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
        _resolve_upr_mode >/dev/null
        echo "Using demo UPR mode: $(_resolve_upr_mode)"
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
            echo "Example: $0 quad_demo 32 --no-torus --model-path /data/deepseek/DeepSeek-R1-0528-dequantized --cache-path /data/deepseek/DeepSeek-R1-0528-Cache/CI" 1>&2
            exit 1
            ;;
    esac
}

main "$@"
