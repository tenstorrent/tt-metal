#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Run fabric tests on a cluster configuration.

Supported configurations:
    4x8   - Single Galaxy, 1 host, 8x4 mesh
    4x32  - Single pod, 4 hosts, 32x4 mesh (quad Galaxy linear)
    8x16  - Single pod, 4 hosts, 16x8 mesh (quad Galaxy 2x2 grid)
    sp4   - SuperPod 4, 16 hosts (4 pods x 4 hosts), 4 meshes with 2D torus XY

Required Options:
    --hosts <host-list>                 Comma-separated list of hosts (1 for 4x8, 4 for 4x32/8x16, 16 for sp4)
    --image <docker-image>              Docker image to use

Optional:
    --config <4x8|4x32|8x16|sp4>       Mesh configuration (default: 4x32)
    --output <directory>                Output directory for log files (default: fabric_test_logs)
    --mesh-graph-desc-path <path>       Path to mesh graph descriptor file (overrides --config)
                                        4x8 default:  tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        4x32 default: tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        8x16 default: tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        sp4 default:  tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sp4_torus_xy_graph_descriptor.textproto
    --test-binary <path>                Path to test binary
                                        (default: ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric)
    --test-config <path>                Path to test configuration file
                                        4x8/4x32/8x16 default: tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml
                                        sp4 default:            tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_multi_mesh.yaml
    --filter <pattern>                  Filter pattern passed to test_tt_fabric --filter
    --no-show-progress                  Disable real-time progress monitoring (enabled by default)
    --no-show-workers                   Disable per-device worker/location logging (enabled by default)
    --help                              Display this help message and exit

Example (single pod):
    $0 --hosts bh-glx-c05u02,bh-glx-c05u08,bh-glx-c06u02,bh-glx-c06u08 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest

Example (sp4 - 16 hosts across 4 pods):
    $0 --config sp4 \\
       --hosts pod0-g0,pod0-g1,pod0-g2,pod0-g3,pod1-g0,...,pod3-g3 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest
EOF
}

# Parse command line arguments
HOSTS=""
DOCKER_IMAGE=""
OUTPUT_DIR="fabric_test_logs"
MESH_GRAPH_DESC_PATH_4x8="tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_4x32="tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_8x16="tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_SP4="tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sp4_torus_xy_graph_descriptor.textproto"
CONFIG="4x32"
MESH_GRAPH_DESC_PATH=""
MESH_GRAPH_DESC_PATH_EXPLICIT=false
TEST_BINARY="./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric"
TEST_CONFIG=""
TEST_CONFIG_EXPLICIT=false
TEST_CONFIG_DEFAULT="tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml"
TEST_CONFIG_SP4="tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_multi_mesh.yaml"
FILTER=""
SHOW_PROGRESS=true
SHOW_WORKERS=true

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
            if [[ "$CONFIG" != "4x8" && "$CONFIG" != "4x32" && "$CONFIG" != "8x16" && "$CONFIG" != "sp4" ]]; then
                echo "Error: --config must be one of '4x8', '4x32', '8x16', or 'sp4'"
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
            TEST_CONFIG_EXPLICIT=true
            shift 2
            ;;
        --filter)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --filter requires a non-empty value"
                exit 1
            fi
            FILTER="$2"
            shift 2
            ;;
        --no-show-progress)
            SHOW_PROGRESS=false
            shift
            ;;
        --no-show-workers)
            SHOW_WORKERS=false
            shift
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
    if [[ "$CONFIG" == "4x8" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_4x8"
    elif [[ "$CONFIG" == "4x32" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_4x32"
    elif [[ "$CONFIG" == "8x16" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_8x16"
    elif [[ "$CONFIG" == "sp4" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_SP4"
    fi
fi

# Set test config based on config if not explicitly provided
if [[ "$TEST_CONFIG_EXPLICIT" == false ]]; then
    if [[ "$CONFIG" == "sp4" ]]; then
        TEST_CONFIG="$TEST_CONFIG_SP4"
    else
        TEST_CONFIG="$TEST_CONFIG_DEFAULT"
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
if [[ -n "$FILTER" ]]; then
    echo "Filter: $FILTER"
fi
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

EXTRA_BINARY_ARGS=""
if [[ "$TEST_BINARY" == *test_tt_fabric ]]; then
    if [[ "$SHOW_PROGRESS" == true ]]; then
        EXTRA_BINARY_ARGS+=" --show-progress"
    fi
    if [[ "$SHOW_WORKERS" == true ]]; then
        EXTRA_BINARY_ARGS+=" --show-workers"
    fi
fi
if [[ -n "$FILTER" ]]; then
    EXTRA_BINARY_ARGS="$EXTRA_BINARY_ARGS --filter $FILTER"
fi

if [[ "$CONFIG" == "4x8" ]]; then
    SINGLE_HOST="${HOSTS%%,*}"
    echo "Running single-host 4x8 on: $SINGLE_HOST"
    echo ""

    ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
        --empty-entrypoint \
        --bind-to none \
        --host "$SINGLE_HOST" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
elif [[ "$CONFIG" == "sp4" ]]; then
    ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
        --empty-entrypoint \
        --bind-to none \
        --host "$HOSTS" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=0 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=1 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=2 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=3 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=0 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=1 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=2 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=3 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=0 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=1 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=2 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=3 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=0 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=1 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=2 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        -x TT_MESH_HOST_RANK=3 "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
else
    ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
        --empty-entrypoint \
        --bind-to none \
        --host "$HOSTS" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
fi

echo ""
echo "=========================================="
echo "Tests completed at $(date)"
echo "Results logged to: $LOG_FILE"
echo "=========================================="
