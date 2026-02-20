#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Run dispatch tests on cluster.

Required Options:
    --hosts <host-list>      Comma-separated list of hosts
    --image <docker-image>   Docker image to use

Optional:
    --output <directory>                Output directory for log files (default: dispatch_test_logs)
    --mesh-graph-desc-path <path>       Path to mesh graph descriptor file
                                        (default: tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto)
    --help                              Display this help message and exit

Example:
    $0 --hosts bh-glx-c01u02,bh-glx-c01u08 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-wh-6u:latest
EOF
}

# Parse command line arguments
HOSTS=""
DOCKER_IMAGE=""
OUTPUT_DIR="dispatch_test_logs"
MESH_GRAPH_DESC_PATH="tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"

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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/dispatch_tests_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Running dispatch tests..."
echo "Using hosts: $HOSTS"
echo "Using docker image: $DOCKER_IMAGE"
echo "Output directory: $OUTPUT_DIR"
echo "Mesh graph descriptor: $MESH_GRAPH_DESC_PATH"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

./tools/scaleout/exabox/mpi-docker \
    --image "$DOCKER_IMAGE" \
    --empty-entrypoint \
    --host "$HOSTS" \
    -x TT_MESH_ID=0 \
    -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
    ./build/test/tt_metal/unit_tests_dispatch \
    --gtest_filter="\
UnitMeshCQProgramFixture.TensixTestRandomizedProgram:\
UnitMeshRandomProgramFixture.TensixTestLargeProgramInBetweenFiveSmallPrograms:\
UnitMeshRandomProgramTraceFixture.TensixTestLargeProgramInBetweenFiveSmallProgramsTrace:\
UnitMeshRandomProgramTraceFixture.TensixTestSimpleProgramsTrace:\
UnitMeshCQTraceFixture.TensixEnqueueMultiProgramTraceBenchmark:\
UnitMeshCQTraceFixture.TensixEnqueueTwoProgramTrace:\
UnitMeshCQSingleCardBufferFixture.ShardedBufferLargeL1ReadWrites:\
UnitMeshCQSingleCardBufferFixture.ShardedBufferLargeDRAMReadWrites:\
UnitMeshCQSingleCardFixture.TensixTestSubDeviceAllocations:\
UnitMeshMultiCQMultiDeviceEventFixture.*:\
UnitMeshCQSingleCardFixture.TensixTestReadWriteMultipleCoresL1" \
    |& tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Tests completed at $(date)"
echo "Results logged to: $LOG_FILE"
echo "=========================================="
