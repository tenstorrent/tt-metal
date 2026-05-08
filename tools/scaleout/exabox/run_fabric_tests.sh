#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Run fabric tests on 4x8, 4x32, 8x16, or multi-pod (sc8/sc12/sc16/sc20/sc36) cluster configurations.

Required Options:
    --hosts <host-list>                 Comma-separated list of hosts. Host count must match --config:
                                        4x8: 1 host
                                        4x32 / 8x16: 4 hosts (single pod)
                                        sc8: 8 hosts (2 pods)    sc12: 12 hosts (3 pods)
                                        sc16: 16 hosts (4 pods)  sc20: 20 hosts (5 pods)
                                        sc36: 36 hosts (9 pods)
    --image <docker-image>              Docker image to use ("none" to use local build)

Optional:
    --config <4x8|4x32|8x16|sc8|sc12|sc16|sc20|sc36>
                                        Mesh configuration (default: 4x32)
    --output <directory>                Output directory for log files (default: fabric_test_logs)
    --mesh-graph-desc-path <path>       Path to mesh graph descriptor file (overrides --config)
                                        4x8 default:  tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        4x32 default: tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        8x16 default: tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        sc<N> default: tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sc<N>_torus_xy_graph_descriptor.textproto
    --test-binary <path>                Path to test binary
                                        (default: ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric)
    --test-config <path>                Path to test configuration file
                                        (default: tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml)
    --filter <pattern>                  Filter pattern passed to test_tt_fabric --filter
    --mpi-if <interface>                Network interface for MPI TCP transport (default: ens5f0np0)
    --mpi-args <args>                   Extra arguments passed directly to mpirun (quoted string)
                                        e.g. --mpi-args "--tag-output"
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
MESH_GRAPH_DESC_PATH_4x8="tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_4x32="tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_8x16="tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_sc8="tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sc8_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_sc12="tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sc12_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_sc16="tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sc16_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_sc20="tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sc20_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_sc36="tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sc36_torus_xy_graph_descriptor.textproto"
CONFIG="4x32"
MESH_GRAPH_DESC_PATH=""
MESH_GRAPH_DESC_PATH_EXPLICIT=false
TEST_BINARY="./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric"
TEST_CONFIG="tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml"
FILTER=""
MPI_IF="ens5f0np0"
MPI_EXTRA_ARGS=()

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
            case "$CONFIG" in
                4x8|4x32|8x16|sc8|sc12|sc16|sc20|sc36) ;;
                *)
                    echo "Error: --config must be one of '4x8', '4x32', '8x16', 'sc8', 'sc12', 'sc16', 'sc20', or 'sc36'"
                    echo ""
                    show_help
                    exit 1
                    ;;
            esac
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
        --filter)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --filter requires a non-empty value"
                exit 1
            fi
            FILTER="$2"
            shift 2
            ;;
        --mpi-if)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --mpi-if requires a non-empty value"
                exit 1
            fi
            MPI_IF="$2"
            shift 2
            ;;
        --mpi-args)
            if [[ -z "$2" ]]; then
                echo "Error: --mpi-args requires a non-empty value"
                exit 1
            fi
            read -ra _extra <<< "$2"
            MPI_EXTRA_ARGS+=("${_extra[@]}")
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
    if [[ "$CONFIG" == "4x8" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_4x8"
    elif [[ "$CONFIG" == "4x32" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_4x32"
    elif [[ "$CONFIG" == "8x16" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_8x16"
    elif [[ "$CONFIG" == "sc8" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_sc8"
    elif [[ "$CONFIG" == "sc12" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_sc12"
    elif [[ "$CONFIG" == "sc16" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_sc16"
    elif [[ "$CONFIG" == "sc20" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_sc20"
    elif [[ "$CONFIG" == "sc36" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_sc36"
    fi
fi

run_mpi() {
    local hosts="$1"
    shift
    if [[ "$DOCKER_IMAGE" == "none" ]]; then
        mpirun-ulfm --host "$hosts" \
            --tag-output \
            --prtemca oob_tcp_if_include "$MPI_IF" \
            --mca plm_ssh_args "-o StrictHostKeyChecking=false -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
            --mca btl_tcp_if_include "$MPI_IF" \
            "${MPI_EXTRA_ARGS[@]}" \
            --bind-to none \
            "$@"
    else
        ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
            --empty-entrypoint \
            --mpi-interface "$MPI_IF" \
            "${MPI_EXTRA_ARGS[@]}" \
            --bind-to none \
            --host "$hosts" \
            "$@"
    fi
}

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
echo "MPI interface: $MPI_IF"
if [[ "${#MPI_EXTRA_ARGS[@]}" -gt 0 ]]; then
    echo "MPI extra args: ${MPI_EXTRA_ARGS[*]}"
fi
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

EXTRA_BINARY_ARGS=""
if [[ "$TEST_BINARY" == *test_tt_fabric ]]; then
    EXTRA_BINARY_ARGS="--show-progress --show-workers --progress-interval 1"
fi
if [[ -n "$FILTER" ]]; then
    EXTRA_BINARY_ARGS="$EXTRA_BINARY_ARGS --filter $FILTER"
fi

# Marker used to detect reports written during this run (vs. stale ones from a
# previous run). We compare report mtimes against this file with bash's `-nt`.
RUN_START_MARKER="$(mktemp)"
trap 'rm -f "$RUN_START_MARKER"' EXIT

if [[ "$CONFIG" == "4x8" ]]; then
    SINGLE_HOST="${HOSTS%%,*}"
    echo "Running single-host 4x8 on: $SINGLE_HOST"
    echo ""

    run_mpi "$SINGLE_HOST" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
elif [[ "$CONFIG" == "sc8" ]]; then
    # sc8: 2 pods x 4 hosts = 8 ranks across 8 hosts.
    # rank r -> (TT_MESH_ID = r/4, TT_MESH_HOST_RANK = r%4)
    echo "Running multi-pod sc8 on 8 hosts: $HOSTS"
    echo ""

    run_mpi "$HOSTS" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
elif [[ "$CONFIG" == "sc12" ]]; then
    # sc12: 3 pods x 4 hosts = 12 ranks across 12 hosts.
    # rank r -> (TT_MESH_ID = r/4, TT_MESH_HOST_RANK = r%4)
    echo "Running multi-pod sc12 on 12 hosts: $HOSTS"
    echo ""

    run_mpi "$HOSTS" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
elif [[ "$CONFIG" == "sc16" ]]; then
    # sc16: 4 pods x 4 hosts = 16 ranks across 16 hosts.
    # rank r -> (TT_MESH_ID = r/4, TT_MESH_HOST_RANK = r%4)
    echo "Running multi-pod sc16 on 16 hosts: $HOSTS"
    echo ""

    run_mpi "$HOSTS" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
elif [[ "$CONFIG" == "sc20" ]]; then
    # sc20: 5 pods x 4 hosts = 20 ranks across 20 hosts.
    # rank r -> (TT_MESH_ID = r/4, TT_MESH_HOST_RANK = r%4)
    echo "Running multi-pod sc20 on 20 hosts: $HOSTS"
    echo ""

    run_mpi "$HOSTS" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=4 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=4 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=4 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=4 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
elif [[ "$CONFIG" == "sc36" ]]; then
    # sc36: 9 pods x 4 hosts = 36 ranks across 36 hosts.
    # rank r -> (TT_MESH_ID = r/4, TT_MESH_HOST_RANK = r%4)
    echo "Running multi-pod sc36 on 36 hosts: $HOSTS"
    echo ""

    run_mpi "$HOSTS" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=1 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=2 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=3 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=4 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=4 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=4 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=4 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=5 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=5 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=5 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=5 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=6 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=6 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=6 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=6 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=7 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=7 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=7 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=7 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=8 \
        -x TT_MESH_HOST_RANK=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=8 \
        -x TT_MESH_HOST_RANK=1 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=8 \
        -x TT_MESH_HOST_RANK=2 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=8 \
        -x TT_MESH_HOST_RANK=3 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
else
    run_mpi "$HOSTS" \
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

# Copy any pairwise-validation reports written by test_tt_fabric (only rank 0
# writes them, and only when a hang is detected) into the user's --output dir
# so all artifacts for this run live in one place. Only copy reports that were
# written during this run (newer than $RUN_START_MARKER) so we don't pick up
# stale files from a previous invocation.
REPORT_SRC_DIR="${TT_METAL_HOME:-.}/generated/fabric"
for report in pairwise_validation_summary.log pairwise_validation_detailed.log; do
    if [[ -f "$REPORT_SRC_DIR/$report" && "$REPORT_SRC_DIR/$report" -nt "$RUN_START_MARKER" ]]; then
        cp "$REPORT_SRC_DIR/$report" "$OUTPUT_DIR/"
        echo "Copied report: $OUTPUT_DIR/$report"
    fi
done
echo "=========================================="
