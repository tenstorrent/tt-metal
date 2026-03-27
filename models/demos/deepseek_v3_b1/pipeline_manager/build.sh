#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${JOBS:-$(nproc)}"
TSAN="${TSAN:-OFF}"

usage() {
    cat <<EOF
Usage: $0 [--standalone | --with-metal]

Build modes:
  --standalone   Build only the core library (MockPipeline, no hardware).
                 Produces: libpipeline_manager_core.so, test_pipeline_manager
                 Output dir: build-standalone/

  --with-metal   Build both core and full libraries (with SocketPipeline).
                 Requires tt-metal to be built. Provide the paths via:
                   TT_METAL_SOURCE_DIR  — path to tt-metal source tree
                   CMAKE_PREFIX_PATH    — path(s) to tt-metal install + deps
                 Produces: libpipeline_manager_core.so, libpipeline_manager.so,
                           test_pipeline_manager, test_device_pipeline,
                           pipeline_launcher
                 Output dir: build-full/

Environment variables:
  TT_METAL_SOURCE_DIR  tt-metal source root (for --with-metal)
  CMAKE_PREFIX_PATH    CMake search paths for installed packages
  JOBS                 Parallel build jobs (default: nproc)
  TSAN                 ON/OFF — enable thread sanitizer (default: OFF)

Examples:
  # Standalone (no hardware, no tt-metal):
  $0 --standalone

  # With tt-metal (assumes tt-metal is built at ../../../..):
  export TT_METAL_SOURCE_DIR=\$(realpath ../../../../)
  export CMAKE_PREFIX_PATH="\${TT_METAL_SOURCE_DIR}/build/lib/cmake;\${TT_METAL_SOURCE_DIR}/build_Release/_deps/nlohmann_json-build"
  $0 --with-metal
EOF
    exit 1
}

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    usage
fi

case "$MODE" in
    --standalone)
        BUILD_DIR="${SCRIPT_DIR}/build-standalone"
        CMAKE_ARGS=(
            -S "${SCRIPT_DIR}"
            -B "${BUILD_DIR}"
            -DCMAKE_BUILD_TYPE=RelWithDebInfo
            -DPM_ENABLE_TSAN="${TSAN}"
        )
        cmake "${CMAKE_ARGS[@]}"
        cmake --build "${BUILD_DIR}" --target pipeline_manager_tests -j "${JOBS}"

        echo ""
        echo "=== Standalone build complete ==="
        echo "  ${BUILD_DIR}/libpipeline_manager_core.so"
        echo "  ${BUILD_DIR}/test_pipeline_manager"
        echo ""
        echo "Run mock tests:"
        echo "  LD_LIBRARY_PATH=${BUILD_DIR} ${BUILD_DIR}/test_pipeline_manager"
        ;;

    --with-metal)
        BUILD_DIR="${SCRIPT_DIR}/build-full"

        if [[ -z "${TT_METAL_SOURCE_DIR:-}" ]]; then
            # Default: assume we're inside the tt-metal tree
            TT_METAL_SOURCE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
            echo "TT_METAL_SOURCE_DIR not set, defaulting to: ${TT_METAL_SOURCE_DIR}"
        fi

        if [[ -z "${CMAKE_PREFIX_PATH:-}" ]]; then
            # Default: standard tt-metal build layout
            CMAKE_PREFIX_PATH="${TT_METAL_SOURCE_DIR}/build/lib/cmake;${TT_METAL_SOURCE_DIR}/build_Release/_deps/nlohmann_json-build"
            echo "CMAKE_PREFIX_PATH not set, defaulting to: ${CMAKE_PREFIX_PATH}"
        fi

        CMAKE_ARGS=(
            -S "${SCRIPT_DIR}"
            -B "${BUILD_DIR}"
            -DCMAKE_BUILD_TYPE=RelWithDebInfo
            -DPM_ENABLE_TSAN="${TSAN}"
            -DTT_METAL_SOURCE_DIR="${TT_METAL_SOURCE_DIR}"
            -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
        )
        cmake "${CMAKE_ARGS[@]}"
        cmake --build "${BUILD_DIR}" --target pipeline_manager_tests -j "${JOBS}"

        echo ""
        echo "=== Full build complete (with TT::Metalium) ==="
        echo "  ${BUILD_DIR}/libpipeline_manager_core.so"
        echo "  ${BUILD_DIR}/libpipeline_manager.so"
        echo "  ${BUILD_DIR}/test_pipeline_manager"
        [[ -f "${BUILD_DIR}/test_device_pipeline" ]] && echo "  ${BUILD_DIR}/test_device_pipeline"
        [[ -f "${BUILD_DIR}/pipeline_launcher" ]] && echo "  ${BUILD_DIR}/pipeline_launcher"
        echo ""
        echo "Run mock tests:"
        echo "  LD_LIBRARY_PATH=${BUILD_DIR} ${BUILD_DIR}/test_pipeline_manager"
        if [[ -f "${BUILD_DIR}/test_device_pipeline" ]]; then
            echo ""
            echo "Run device tests (requires hardware):"
            echo "  LD_LIBRARY_PATH=${BUILD_DIR} ${BUILD_DIR}/test_device_pipeline"
        fi
        ;;

    *)
        usage
        ;;
esac
