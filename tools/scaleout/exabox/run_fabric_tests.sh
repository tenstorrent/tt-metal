#!/bin/bash

# Run TT-Fabric tests on BH Galaxy clusters.
#
# Single-pod (4x32, 8x16): use --config. Multi-pod: use --num-meshes +
# --hosts-per-mesh + --mesh-graph-desc-path.
#
# MPI host ordering encodes the rank-to-mesh mapping implicitly: hosts are
# assigned sequentially, so the first hosts_per_mesh hosts land on mesh 0,
# the next on mesh 1, etc. No separate rankfile or rank-bindings YAML is needed.

show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Run fabric tests on BH Galaxy clusters (single-pod or multi-pod).
Hosts must be in mesh order: first hosts_per_mesh hosts -> mesh 0, next -> mesh 1, etc.

Required Options:
    --hosts <host-list>                 Comma-separated list of hosts in mesh order
    --image <docker-image>              Docker image to use

Optional:
    --config <preset>                   Named single-pod topology preset (default: 4x32)
                                        4x32, 8x16
    --num-meshes <N>                    Number of meshes for multi-pod runs; requires
                                        --mesh-graph-desc-path
    --hosts-per-mesh <N>                Hosts per mesh (default: 4)
    --mesh-graph-desc-path <path>       Path to mesh graph descriptor; overrides --config default
    --test-binary <path>                Path to test binary
                                        (default: ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric)
    --test-config <path>                Path to test configuration YAML; overrides --config default
    --output <directory>                Output directory for log files; overrides default
    --help                              Display this help message and exit

Presets:
    4x32   1 mesh, 4 hosts, stability tests   (32x4 mesh topology, Aisle B)
    8x16   1 mesh, 4 hosts, stability tests   (16x8 mesh topology, Aisle C)

MGD defaults per preset:
    4x32 -> tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
    8x16 -> tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto

Examples:
    # Single-pod (default: 4x32)
    $0 --hosts bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest

    $0 --config 8x16 \\
       --hosts bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest

    # Multi-pod (4 meshes, 16 hosts, requires MGD)
    $0 --num-meshes 4 \\
       --mesh-graph-desc-path tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sp4_torus_xy_graph_descriptor.textproto \\
       --hosts bh-glx-d03u02,bh-glx-d03u08,bh-glx-d04u08,bh-glx-d04u02,\\
               bh-glx-d05u02,bh-glx-d05u08,bh-glx-d06u08,bh-glx-d06u02,\\
               bh-glx-d07u02,bh-glx-d07u08,bh-glx-d08u08,bh-glx-d08u02,\\
               bh-glx-d09u02,bh-glx-d09u08,bh-glx-d10u08,bh-glx-d10u02 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest
EOF
}

# --- Preset table (single-pod configs only) ---------------------------------
declare -A PRESET_NUM_MESHES=( [4x32]=1  [8x16]=1  )
declare -A PRESET_HOSTS_PER=(  [4x32]=4  [8x16]=4  )
declare -A PRESET_MGD=(
    [4x32]="tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
    [8x16]="tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
)
declare -A PRESET_TEST_CONFIG=(
    [4x32]="tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml"
    [8x16]="tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml"
)
declare -A PRESET_OUTPUT=( [4x32]="fabric_test_logs"  [8x16]="fabric_test_logs"  )

# --- Defaults (resolved after parsing) --------------------------------------
CONFIG="4x32"
HOSTS=""
DOCKER_IMAGE=""
NUM_MESHES=""
HOSTS_PER_MESH=""
MESH_GRAPH_DESC_PATH=""
TEST_BINARY="./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric"
TEST_CONFIG=""
OUTPUT_DIR=""

# --- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --hosts)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --hosts requires a non-empty value"; exit 1
            fi
            HOSTS="$2"; shift 2 ;;
        --image)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --image requires a non-empty value"; exit 1
            fi
            DOCKER_IMAGE="$2"; shift 2 ;;
        --config)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --config requires a non-empty value"; exit 1
            fi
            CONFIG="$2"
            if [[ -z "${PRESET_NUM_MESHES[$CONFIG]}" ]]; then
                echo "Error: unknown --config '$CONFIG'. Known presets: ${!PRESET_NUM_MESHES[*]}"
                echo "       For multi-pod runs use --num-meshes + --mesh-graph-desc-path."
                exit 1
            fi
            shift 2 ;;
        --num-meshes)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --num-meshes requires a non-empty value"; exit 1
            fi
            NUM_MESHES="$2"; shift 2 ;;
        --hosts-per-mesh)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --hosts-per-mesh requires a non-empty value"; exit 1
            fi
            HOSTS_PER_MESH="$2"; shift 2 ;;
        --mesh-graph-desc-path)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --mesh-graph-desc-path requires a non-empty value"; exit 1
            fi
            MESH_GRAPH_DESC_PATH="$2"; shift 2 ;;
        --test-binary)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --test-binary requires a non-empty value"; exit 1
            fi
            TEST_BINARY="$2"; shift 2 ;;
        --test-config)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --test-config requires a non-empty value"; exit 1
            fi
            TEST_CONFIG="$2"; shift 2 ;;
        --output)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --output requires a non-empty value"; exit 1
            fi
            OUTPUT_DIR="$2"; shift 2 ;;
        --help)
            show_help; exit 0 ;;
        *)
            echo "Error: Unknown option: $1"
            echo ""
            show_help
            exit 1 ;;
    esac
done

# --- Validate required arguments --------------------------------------------
if [[ -z "$HOSTS" ]]; then
    echo "Error: --hosts is required"; echo ""; show_help; exit 1
fi
if [[ -z "$DOCKER_IMAGE" ]]; then
    echo "Error: --image is required"; echo ""; show_help; exit 1
fi

# --- Resolve topology -------------------------------------------------------
# --num-meshes triggers multi-pod mode; --config drives single-pod mode.

if [[ -n "$NUM_MESHES" ]]; then
    [[ -z "$HOSTS_PER_MESH" ]] && HOSTS_PER_MESH=4
    if [[ -z "$MESH_GRAPH_DESC_PATH" ]]; then
        echo "Error: --mesh-graph-desc-path is required for multi-pod runs (--num-meshes)"
        exit 1
    fi
    [[ -z "$TEST_CONFIG" ]] && TEST_CONFIG="tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_multi_mesh.yaml"
    [[ -z "$OUTPUT_DIR" ]]  && OUTPUT_DIR="multi_mesh_fabric_test_logs"
else
    NUM_MESHES="${PRESET_NUM_MESHES[$CONFIG]}"
    HOSTS_PER_MESH="${PRESET_HOSTS_PER[$CONFIG]}"
    [[ -z "$MESH_GRAPH_DESC_PATH" ]] && MESH_GRAPH_DESC_PATH="${PRESET_MGD[$CONFIG]}"
    [[ -z "$TEST_CONFIG" ]]          && TEST_CONFIG="${PRESET_TEST_CONFIG[$CONFIG]}"
    [[ -z "$OUTPUT_DIR" ]]           && OUTPUT_DIR="${PRESET_OUTPUT[$CONFIG]}"
fi

# --- Validate host count ----------------------------------------------------
EXPECTED_HOSTS=$(( NUM_MESHES * HOSTS_PER_MESH ))
IFS=',' read -ra HOST_ARRAY <<< "$HOSTS"
NUM_HOSTS=${#HOST_ARRAY[@]}
if [[ "$NUM_HOSTS" -ne "$EXPECTED_HOSTS" ]]; then
    echo "Error: expected $EXPECTED_HOSTS hosts ($NUM_MESHES mesh(es) x $HOSTS_PER_MESH hosts/mesh), got $NUM_HOSTS"
    exit 1
fi

# --- Run --------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/fabric_tests_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Running fabric tests..."
echo "Meshes: $NUM_MESHES  Hosts/mesh: $HOSTS_PER_MESH  Total ranks: $EXPECTED_HOSTS"
echo "Hosts: $HOSTS"
echo "Docker image: $DOCKER_IMAGE"
echo "Mesh graph descriptor: $MESH_GRAPH_DESC_PATH"
echo "Test binary: $TEST_BINARY"
echo "Test config: $TEST_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

CMD=(
    ./tools/scaleout/exabox/mpi-docker
    --image "$DOCKER_IMAGE"
    --empty-entrypoint
    --bind-to none
    --host "$HOSTS"
)

for ((mesh_id = 0; mesh_id < NUM_MESHES; mesh_id++)); do
    for ((mesh_host_rank = 0; mesh_host_rank < HOSTS_PER_MESH; mesh_host_rank++)); do
        CMD+=(
            -np 1
            -x TT_MESH_ID=$mesh_id
            -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH"
            -x TT_MESH_HOST_RANK=$mesh_host_rank
            "$TEST_BINARY"
            --test_config "$TEST_CONFIG"
        )
        if [[ $mesh_id -lt $((NUM_MESHES - 1)) || $mesh_host_rank -lt $((HOSTS_PER_MESH - 1)) ]]; then
            CMD+=(:)
        fi
    done
done

"${CMD[@]}" |& tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Tests completed at $(date)"
echo "Results logged to: $LOG_FILE"
echo "=========================================="
