#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Run cluster validation commands for multiple iterations.

Required Options:
    --hosts <host-list>                     Comma-separated list of hosts
    --image <docker-image>                  Docker image to use

Optional:
    --cabling-descriptor-path <path>        Path to cabling descriptor file
                                            (default: /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto)
    --deployment-descriptor-path <path>     Path to deployment descriptor file
                                            (default: /data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto)
    --iterations <number>                   Number of times to run the full validation sequence (default: 50)
                                            Each iteration runs run_cluster_validation with 10 internal iterations
    --output <directory>                    Output directory for log files (default: validation_output)
    --help                                  Display this help message and exit

Example:
    $0 --hosts bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-wh-6u:latest \\
       --iterations 100
EOF
}

# Parse command line arguments
HOSTS=""
DOCKER_IMAGE=""
CABLING_DESCRIPTOR_PATH="/data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto"
DEPLOYMENT_DESCRIPTOR_PATH="/data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto"
ITERATIONS=50
OUTPUT_DIR="validation_output"

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
        --cabling-descriptor-path)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --cabling-descriptor-path requires a non-empty value"
                exit 1
            fi
            CABLING_DESCRIPTOR_PATH="$2"
            shift 2
            ;;
        --deployment-descriptor-path)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --deployment-descriptor-path requires a non-empty value"
                exit 1
            fi
            DEPLOYMENT_DESCRIPTOR_PATH="$2"
            shift 2
            ;;
        --iterations)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --iterations requires a non-empty value"
                exit 1
            fi
            if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: --iterations must be a positive integer, got '$2'"
                exit 1
            fi
            ITERATIONS="$2"
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

echo "Using hosts: $HOSTS"
echo "Using docker image: $DOCKER_IMAGE"
echo "Cabling descriptor path: $CABLING_DESCRIPTOR_PATH"
echo "Deployment descriptor path: $DEPLOYMENT_DESCRIPTOR_PATH"
echo "Number of iterations: $ITERATIONS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Main testing loop
for ((i=1; i<=ITERATIONS; i++)); do
    echo "Starting iteration $i of $ITERATIONS..."

    LOG_FILE="$OUTPUT_DIR/cluster_validation_iteration_${i}.log"

    {
        echo "=========================================="
        echo "Iteration: $i"
        echo "Timestamp: $(date)"
        echo "=========================================="
        echo ""

        echo "Running tt-smi -glx_reset..."
        # mpirun --host "$HOSTS" --mca btl_tcp_if_exclude docker0,lo,tailscale0 tt-smi -glx_reset

        sleep 5

        echo ""
        echo "Running cluster validation..."
        ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
            --empty-entrypoint \
            --host "$HOSTS" \
            ./build/tools/scaleout/run_cluster_validation \
            --cabling-descriptor-path "$CABLING_DESCRIPTOR_PATH" \
            --deployment-descriptor-path "$DEPLOYMENT_DESCRIPTOR_PATH" \
            --send-traffic \
            --num-iterations 10
        echo "Iteration $i completed at $(date)"
        echo "=========================================="
    } 2>&1 | tee "$LOG_FILE"

    echo "Iteration $i logged to $LOG_FILE"
    echo ""
done

echo "All $ITERATIONS iterations completed!"
