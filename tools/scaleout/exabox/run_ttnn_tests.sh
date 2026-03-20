#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Runs ttnn validation tests across multiple hosts using a specified Docker image.

Required Options:
    --hosts <host-list>                     Comma-separated list of hosts
    --image <docker-image>                  Docker image to use
    --script <script-path>                  Path to the validation script to run inside the container

Optional:
    --output <directory>                    Output directory for log files (default: validation_output)
    --help                                  Display this help message and exit

Example:
    $0 --hosts bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-wh-6u:latest \\
       --script test_ttnn.py \\
EOF
}

# Parse command line arguments
HOSTS=""
DOCKER_IMAGE=""
SCRIPT_PATH=""
OUTPUT_DIR="ttnn_validation_output"

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
        --script)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --script requires a non-empty value"
                exit 1
            fi
            SCRIPT_PATH="$2"
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

if [[ -z "$SCRIPT_PATH" ]]; then
    echo "Error: --script is required"
    echo ""
    show_help
    exit 1
fi

echo "Using hosts: $HOSTS"
echo "Using docker image: $DOCKER_IMAGE"
echo "Using script: $SCRIPT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

echo "Slurm partition: $PARTITION"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/ttnn_tests_$(date +%Y%m%d_%H%M%S).log"

WORK_DIR=/workspace

{
    echo "Starting ttnn tests at $(date)"
    srun \
    --partition=$PARTITION \
    --nodelist=$HOSTS \
    docker run \
    --rm --device /dev/tenstorrent \
    -v $(pwd):/workspace \
    --entrypoint="" \
    $DOCKER_IMAGE \
    python /workspace/ttnn/ttnn/examples/usage/run_op_on_device.py
} 2>&1 | tee "$LOG_FILE"
echo "Done."
