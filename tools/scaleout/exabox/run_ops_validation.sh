#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Run cluster validation commands for multiple iterations.

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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
WORK_DIR=/workspace

LOG_FILE="$OUTPUT_DIR/ttnn_validation.log"
# {
#     # echo "Running tt-smi -glx_reset..."
#     # mpirun --host "$HOSTS" --mca btl_tcp_if_exclude docker0,lo,tailscale0 tt-smi -glx_reset
#     # sleep 5

#     echo ""
#     echo "Running TTNN validation..."
#     ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
#         --empty-entrypoint \
#         --host "$HOSTS" \
#         -x ARCH_NAME=blackhole \
#         -x TT_METAL_LOGS_PATH=/tmp \
#         -x TT_METAL_HOME=/ \
#         python $WORK_DIR/$SCRIPT_PATH
#         # $WORK_DIR/tests/scripts/tg/run_tg_frequent_tests.sh --model unit

#     echo "=========================================="
# } 2>&1 | tee "$LOG_FILE"

echo "Done."
