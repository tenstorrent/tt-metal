#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Run fabric tests on 4x32 or 8x16 cluster configuration.

Required Options:
    --hosts <host-list>                 Comma-separated list of hosts
    --image <docker-image>              Docker image to use

Optional:
    --config <4x32|8x16>                Mesh configuration (default: 4x32)
    --output <directory>                Output directory for log files (default: fabric_test_logs)
    --mesh-graph-desc-path <path>       Path to mesh graph descriptor file (overrides --config)
                                        4x32 default: tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        8x16 default: tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
    --test-binary <path>                Path to test binary
                                        (default: ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric)
    --test-config <path>                Path to test configuration file
                                        (default: tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml)
    --help                              Display this help message and exit

Example:
    $0 --hosts bh-glx-c01u02,bh-glx-c01u08 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-wh-6u:latest
EOF
}

# Parse command line arguments
HOSTS=""
DOCKER_IMAGE=""
OUTPUT_DIR="fabric_test_logs"
MESH_GRAPH_DESC_PATH_4x32="tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_8x16="tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
CONFIG="4x32"
MESH_GRAPH_DESC_PATH=""
MESH_GRAPH_DESC_PATH_EXPLICIT=false
TEST_BINARY="./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric"
TEST_CONFIG="tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --hosts)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --hosts requires a non-empty value"
                exit 1
            fi
            HOSTS="$2"
            shift 2
            ;;
        --image)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --image requires a non-empty value"
                exit 1
            fi
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --config)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --config requires a non-empty value"
                exit 1
            fi
            CONFIG="$2"
            if [[ "$CONFIG" != "4x32" && "$CONFIG" != "8x16" ]]; then
                echo "Error: --config must be either '4x32' or '8x16'"
                echo ""
                show_help
                exit 1
            fi
            shift 2
            ;;
        --output)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --output requires a non-empty value"
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mesh-graph-desc-path)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --mesh-graph-desc-path requires a non-empty value"
                exit 1
            fi
            MESH_GRAPH_DESC_PATH="$2"
            MESH_GRAPH_DESC_PATH_EXPLICIT=true
            shift 2
            ;;
        --test-binary)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --test-binary requires a non-empty value"
                exit 1
            fi
            TEST_BINARY="$2"
            shift 2
            ;;
        --test-config)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --test-config requires a non-empty value"
                exit 1
            fi
            TEST_CONFIG="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$HOSTS" ]]; then
    echo "Error: --hosts is required"
    echo ""
    show_help
    exit 1
fi

if [[ -z "$DOCKER_IMAGE" ]]; then
    echo "Error: --image is required"
    echo ""
    show_help
    exit 1
fi

# Set mesh graph descriptor path based on config if not explicitly provided
if [[ "$MESH_GRAPH_DESC_PATH_EXPLICIT" == false ]]; then
    if [[ "$CONFIG" == "4x32" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_4x32"
    elif [[ "$CONFIG" == "8x16" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_8x16"
    fi
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/fabric_tests_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Running fabric tests..."
echo "Using hosts: $HOSTS"
echo "Using docker image: $DOCKER_IMAGE"
echo "Configuration: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Mesh graph descriptor: $MESH_GRAPH_DESC_PATH"
echo "Test binary: $TEST_BINARY"
echo "Test config: $TEST_CONFIG"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
    --empty-entrypoint \
    --bind-to none \
    --host "$HOSTS" \
    -np 1 \
    -x TT_MESH_ID=0 \
    -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
    -x TT_MESH_HOST_RANK=0 "$TEST_BINARY" \
    --test_config "$TEST_CONFIG" : \
    -np 1 \
    -x TT_MESH_ID=0 \
    -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
    -x TT_MESH_HOST_RANK=1 "$TEST_BINARY" \
    --test_config "$TEST_CONFIG" : \
    -np 1 \
    -x TT_MESH_ID=0 \
    -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
    -x TT_MESH_HOST_RANK=2 "$TEST_BINARY" \
    --test_config "$TEST_CONFIG" : \
    -np 1 \
    -x TT_MESH_ID=0 \
    -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
    -x TT_MESH_HOST_RANK=3 "$TEST_BINARY" \
    --test_config "$TEST_CONFIG" |& tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Tests completed at $(date)"
echo "Results logged to: $LOG_FILE"
echo "=========================================="
