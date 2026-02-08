#!/bin/bash
# Script to run TTNN unit tests with TTSim using pytest collection
# Based on: .github/workflows/ttnn-post-commit.yaml with TTSim skip list

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Default values
ARCH="${ARCH:-wormhole_b0}"  # Can be: wormhole_b0 or blackhole
TTSIM_VERSION="${TTSIM_VERSION:-v1.3.3}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
TTSIM_HOME="${TT_METAL_SIMULATOR_HOME:-${WORKSPACE_DIR}/sim}"
TEST_GROUP="${TEST_GROUP:-all}"  # Which test group to run
SKIP_LIST_FILE="${SKIP_LIST_FILE:-${WORKSPACE_DIR}/tests/pipeline_reorg/ttsim-skip-list.yaml}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-auto}"  # Number of parallel workers for pytest-xdist

# ============================================================================
# Functions
# ============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run TTNN unit tests with TTSim using pytest collection and skip list.

OPTIONS:
    -a, --arch ARCH         Architecture to test (wormhole_b0 or blackhole)
                            Default: wormhole_b0
    -v, --version VERSION   TTSim version to use (e.g., v1.3.3)
                            Default: v1.3.3
    -g, --group GROUP       Test group to run (see below for options)
                            Default: all
    -p, --parallel [N]      Enable parallel test execution with pytest-xdist
                            N = number of workers (default: auto)
                            NOTE: Requires sufficient /dev/shm space (--shm-size=512m)
                            May have import issues with complex test environments
                            If you see ImportError, run without --parallel
    -l, --list-groups       List available test groups
    -s, --skip-download     Skip TTSim download (if already installed)
    --skip-list FILE        Path to skip list file
                            Default: ./ttsim_skip_list.txt
    --dry-run              Show what would be run without running tests
    -h, --help             Show this help message

TEST GROUPS:
    all                     Run all test groups
    examples                TTNN example tests
    eltwise                 All eltwise operations (groups 1-5 combined)
    eltwise-1 through eltwise-5   Individual eltwise groups
    conv                    Convolution operations
    pool                    Pooling operations
    matmul                  Matrix multiplication operations
    fused                   Fused operations
    transformers            Transformer operations
    reduce                  Reduce and misc operations
    core                    Core TTNN unit tests

ENVIRONMENT VARIABLES:
    ARCH                    Architecture (same as --arch)
    TTSIM_VERSION          TTSim version (same as --version)
    WORKSPACE_DIR          Root workspace directory (default: current dir)
    TT_METAL_SIMULATOR_HOME TTSim installation directory (default: \$WORKSPACE_DIR/sim)
    TEST_GROUP             Test group to run (same as --group)
    SKIP_LIST_FILE         Path to skip list file

SKIP LIST FORMAT:
    The skip list file should contain one test path per line:
        tests/ttnn/unit_tests/operations/test_example.py
        tests/ttnn/unit_tests/operations/test_another.py::test_specific_function

EXAMPLES:
    # Run all test groups
    $0 --arch wormhole_b0

    # Run only eltwise tests
    $0 --group eltwise

    # Run with parallel execution (auto-detect CPU count)
    $0 --group core --parallel

    # Run with specific number of parallel workers
    $0 --group core --parallel 4

    # Run specific eltwise group in parallel
    $0 --group eltwise-3 --parallel

    # List available test groups
    $0 --list-groups

    # Dry run to see what would be executed
    $0 --group matmul --dry-run

    # Use custom skip list
    $0 --skip-list my_custom_skip_list.txt

EOF
    exit 0
}

log_info() {
    echo "ℹ️  $*"
}

log_success() {
    echo "✅ $*"
}

log_error() {
    echo "❌ $*" >&2
}

log_warning() {
    echo "⚠️  $*"
}

list_test_groups() {
    cat << EOF
Available test groups:
  all          - Run all test groups
  examples     - TTNN example tests
  eltwise      - All eltwise operations (groups 1-5 combined)
  eltwise-1    - Eltwise operations group 1
  eltwise-2    - Eltwise operations group 2
  eltwise-3    - Eltwise operations group 3
  eltwise-4    - Eltwise operations group 4
  eltwise-5    - Eltwise operations group 5
  conv         - Convolution operations
  pool         - Pooling operations
  matmul       - Matrix multiplication operations
  fused        - Fused operations
  transformers - Transformer operations
  reduce       - Reduce and misc operations
  core         - Core TTNN unit tests
EOF
    exit 0
}

get_soc_descriptor() {
    local arch=$1
    case "$arch" in
        wormhole_b0)
            echo "wormhole_b0_80_arch.yaml"
            ;;
        blackhole)
            echo "blackhole_140_arch.yaml"
            ;;
        *)
            log_error "Unknown architecture: $arch"
            exit 1
            ;;
    esac
}

get_build_suffix() {
    local arch=$1
    case "$arch" in
        wormhole_b0)
            echo "release_wh"
            ;;
        blackhole)
            echo "release_bh"
            ;;
        *)
            log_error "Unknown architecture: $arch"
            exit 1
            ;;
    esac
}

download_ttsim() {
    local version=$1
    local build_suffix=$2
    local asset_suffix="${build_suffix#release_}"
    local asset_name="libttsim_${asset_suffix}.so"
    local download_url="https://github.com/tenstorrent/ttsim/releases/download/${version}/${asset_name}"

    log_info "Downloading TTSim ${version} for ${ARCH}..."
    log_info "URL: ${download_url}"

    mkdir -p /tmp/ttsim

    if command -v curl &> /dev/null; then
        curl -L -o "/tmp/ttsim/${asset_name}" "${download_url}"
    elif command -v wget &> /dev/null; then
        wget -O "/tmp/ttsim/${asset_name}" "${download_url}"
    else
        log_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    log_success "Downloaded ${asset_name}"
}

install_ttsim() {
    local build_suffix=$1
    local soc_desc=$2
    local asset_suffix="${build_suffix#release_}"
    local asset_name="libttsim_${asset_suffix}.so"

    log_info "Installing TTSim to ${TTSIM_HOME}..."

    mkdir -p "${TTSIM_HOME}"
    mv "/tmp/ttsim/${asset_name}" "${TTSIM_HOME}/libttsim.so"

    # Copy SOC descriptor
    local soc_desc_path="${WORKSPACE_DIR}/tt_metal/soc_descriptors/${soc_desc}"
    if [[ ! -f "$soc_desc_path" ]]; then
        log_error "SOC descriptor not found: $soc_desc_path"
        log_error "Make sure you're running this from the tt-metal repository root"
        exit 1
    fi

    cp "$soc_desc_path" "${TTSIM_HOME}/soc_descriptor.yaml"

    log_success "TTSim installed successfully"
}

setup_environment() {
    log_info "Setting up environment variables..."

    export ARCH_NAME="${ARCH}"
    export TT_METAL_SIMULATOR_HOME="${TTSIM_HOME}"
    export TT_METAL_SIMULATOR="${TTSIM_HOME}/libttsim.so"
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export LOGURU_LEVEL=INFO
    export PYTHONPATH="${WORKSPACE_DIR}"
    export PYTHONHASHSEED=0

    log_info "Environment configuration:"
    log_info "  ARCH_NAME: ${ARCH_NAME}"
    log_info "  TT_METAL_SIMULATOR_HOME: ${TT_METAL_SIMULATOR_HOME}"
    log_info "  TT_METAL_SIMULATOR: ${TT_METAL_SIMULATOR}"
    log_info "  TT_METAL_SLOW_DISPATCH_MODE: ${TT_METAL_SLOW_DISPATCH_MODE}"
    log_info "  LOGURU_LEVEL: ${LOGURU_LEVEL}"
    log_info "  PYTHONPATH: ${PYTHONPATH}"
    log_info "  PYTHONHASHSEED: ${PYTHONHASHSEED}"
}

build_skip_args() {
    local skip_args=""

    if [[ -f "$SKIP_LIST_FILE" ]]; then
        log_info "Loading skip list from: $SKIP_LIST_FILE (arch: $ARCH)" >&2

        # Parse YAML skip list using Python (more portable than yq)
        local skip_paths
        skip_paths=$(python3 << EOF
import yaml
with open("$SKIP_LIST_FILE") as f:
    data = yaml.safe_load(f)
common = data.get("common", []) or []
arch_specific = data.get("$ARCH", []) or []
for path in common + arch_specific:
    print(path)
EOF
)
        local skip_count=0
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            skip_args="${skip_args} --deselect='${line}'"
            ((skip_count++))
        done <<< "$skip_paths"
        log_info "Loaded ${skip_count} test(s) to skip" >&2
    else
        log_warning "Skip list file not found: $SKIP_LIST_FILE" >&2
        log_warning "Proceeding without skip list" >&2
    fi

    echo "$skip_args"
}

run_test_group() {
    local group=$1
    local skip_args=$2
    local dry_run=$3

    log_info "Running test group: $group"

    cd "${WORKSPACE_DIR}"

    # Check if pytest is available (unless running examples)
    if [[ "$group" != "examples" ]] && ! command -v pytest &> /dev/null; then
        log_error "pytest not found. Please install pytest first."
        exit 1
    fi

    # Build pytest command as array to avoid eval issues with special characters
    local pytest_cmd=(pytest --timeout 600 -v -m "not disable_fast_runtime_mode")

    # Add parallel execution if enabled
    if [[ "${USE_PARALLEL}" == "true" ]]; then
        if command -v pytest &> /dev/null && python3 -c "import xdist" 2>/dev/null; then
            pytest_cmd+=(-n "${PARALLEL_WORKERS}")
            log_info "Parallel execution enabled with ${PARALLEL_WORKERS} workers"
        else
            log_warning "pytest-xdist not found. Install with: pip install pytest-xdist"
            log_warning "Continuing with sequential execution"
        fi
    fi

    case "$group" in
        examples)
            if [[ "$dry_run" == "true" ]]; then
                log_info "DRY RUN - Would execute:"
                echo "  ./tests/scripts/run_ttnn_examples.sh"
                return 0
            fi
            log_info "Executing: ./tests/scripts/run_ttnn_examples.sh"
            ./tests/scripts/run_ttnn_examples.sh
            return $?
            ;;
        eltwise)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/eltwise)
            ;;
        eltwise-1)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/eltwise --splits 5 --group 1)
            ;;
        eltwise-2)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/eltwise --splits 5 --group 2)
            ;;
        eltwise-3)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/eltwise --splits 5 --group 3)
            ;;
        eltwise-4)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/eltwise --splits 5 --group 4)
            ;;
        eltwise-5)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/eltwise --splits 5 --group 5)
            ;;
        conv)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/conv)
            ;;
        pool)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/pool)
            ;;
        matmul)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/matmul)
            ;;
        fused)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/fused)
            ;;
        transformers)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/transformers)
            ;;
        reduce)
            pytest_cmd+=(tests/ttnn/unit_tests/operations/reduce tests/ttnn/unit_tests/operations/debug)
            ;;
        core)
            pytest_cmd+=(tests/ttnn/unit_tests/base_functionality tests/ttnn/unit_tests/benchmarks tests/ttnn/unit_tests/tensor)
            ;;
        *)
            log_error "Unknown test group: $group"
            exit 1
            ;;
    esac

    # Add skip arguments (already properly formatted with --deselect=)
    if [[ -n "$skip_args" ]]; then
        # Read skip args into array, splitting on spaces but preserving quoted strings
        eval "pytest_cmd+=($skip_args)"
    fi

    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY RUN - Would execute:"
        echo "  ${pytest_cmd[*]}"
        return 0
    fi

    log_info "Executing: ${pytest_cmd[*]}"
    "${pytest_cmd[@]}"
}

# ============================================================================
# Main Script
# ============================================================================

# Parse command line arguments
SKIP_DOWNLOAD=false
DRY_RUN=false
USE_PARALLEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--arch)
            ARCH="$2"
            shift 2
            ;;
        -v|--version)
            TTSIM_VERSION="$2"
            shift 2
            ;;
        -g|--group)
            TEST_GROUP="$2"
            shift 2
            ;;
        -p|--parallel)
            USE_PARALLEL=true
            if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                PARALLEL_WORKERS="$2"
                shift 2
            else
                shift
            fi
            ;;
        -l|--list-groups)
            list_test_groups
            ;;
        -s|--skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-list)
            SKIP_LIST_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate architecture
if [[ "$ARCH" != "wormhole_b0" && "$ARCH" != "blackhole" ]]; then
    log_error "Invalid architecture: $ARCH"
    log_error "Valid options are: wormhole_b0, blackhole"
    exit 1
fi

# Get configuration for selected architecture
BUILD_SUFFIX=$(get_build_suffix "$ARCH")
SOC_DESC=$(get_soc_descriptor "$ARCH")

log_info "Configuration:"
log_info "  Architecture: ${ARCH}"
log_info "  TTSim Version: ${TTSIM_VERSION}"
log_info "  Test Group: ${TEST_GROUP}"
log_info "  Build Suffix: ${BUILD_SUFFIX}"
log_info "  SOC Descriptor: ${SOC_DESC}"
log_info "  Workspace: ${WORKSPACE_DIR}"
log_info "  Skip List: ${SKIP_LIST_FILE}"
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "  Mode: DRY RUN"
fi
echo ""

# Download and install TTSim if needed
if [[ "$SKIP_DOWNLOAD" == false ]]; then
    if [[ -f "${TTSIM_HOME}/libttsim.so" ]]; then
        log_warning "TTSim already exists at ${TTSIM_HOME}/libttsim.so"
        read -p "Do you want to re-download? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            download_ttsim "$TTSIM_VERSION" "$BUILD_SUFFIX"
            install_ttsim "$BUILD_SUFFIX" "$SOC_DESC"
        else
            log_info "Using existing TTSim installation"
        fi
    else
        download_ttsim "$TTSIM_VERSION" "$BUILD_SUFFIX"
        install_ttsim "$BUILD_SUFFIX" "$SOC_DESC"
    fi
else
    log_info "Skipping TTSim download (--skip-download flag set)"
    if [[ ! -f "${TTSIM_HOME}/libttsim.so" ]]; then
        log_error "TTSim not found at ${TTSIM_HOME}/libttsim.so"
        log_error "Run without --skip-download to download it first"
        exit 1
    fi
fi

echo ""

# Setup environment
setup_environment

echo ""

# Build skip arguments
SKIP_ARGS=$(build_skip_args)

# Run tests
if [[ "$TEST_GROUP" == "all" ]]; then
    log_info "Running all test groups..."
    GROUPS=("examples" "eltwise-1" "eltwise-2" "eltwise-3" "eltwise-4" "eltwise-5" "conv" "pool" "matmul" "fused" "transformers" "reduce" "core")
    for group in "${GROUPS[@]}"; do
        echo ""
        log_info "========================================"
        run_test_group "$group" "$SKIP_ARGS" "$DRY_RUN"
        log_info "========================================"
    done
else
    run_test_group "$TEST_GROUP" "$SKIP_ARGS" "$DRY_RUN"
fi

if [[ "$DRY_RUN" == "false" ]]; then
    log_success "All tests completed!"
else
    log_info "Dry run completed!"
fi
