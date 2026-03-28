#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Run cluster validation commands for multiple iterations.

Required Options:
    --hosts <host-list>                     Comma-separated list of hosts
    --image <docker-image>                  Docker image to use ("none" to use local build)

Optional:
    --cabling-descriptor-path <path>        Path to cabling descriptor file
                                            (default: /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto)
    --deployment-descriptor-path <path>     Path to deployment descriptor file
                                            (default: /data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto)
    --iterations <number>                   Number of times to run the full validation sequence (default: 50)
                                            Each iteration runs run_cluster_validation with 10 internal iterations

    --factory-descriptor-path <path>        Path to pregenerated factory system descriptor (FSD) file (.textproto)
                                            (if provided, cabling and deployment descriptors are ignored)
    --output <directory>                    Output directory for log files (default: validation_output)
    --volume <host-path>                    Extra volume mount for Docker containers (can be repeated)
                                            The host path is mounted at the same path inside the container
    --rerun-on-retrain                      Rerun validation when Ethernet links are retrained
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

FACTORY_DESCRIPTOR_PATH=""
OUTPUT_DIR="validation_output"
EXTRA_VOLUMES=(/data/scaleout_configs)
RERUN_ON_RETRAIN=false

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
        --factory-descriptor-path)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --factory-descriptor-path requires a non-empty value"
                exit 1
            fi
            FACTORY_DESCRIPTOR_PATH="$2"
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
        --volume)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --volume requires a non-empty value"
                exit 1
            fi
            EXTRA_VOLUMES+=("$2")
            shift 2
            ;;
        --rerun-on-retrain)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                echo "Error: --rerun-on-retrain does not accept a value"
                exit 1
            fi
            RERUN_ON_RETRAIN=true
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

run_cluster_validation() {
    if [[ -n "$FACTORY_DESCRIPTOR_PATH" ]]; then
        local descriptor_args=(--factory-descriptor-path "$FACTORY_DESCRIPTOR_PATH")
    else
        local descriptor_args=(--cabling-descriptor-path "$CABLING_DESCRIPTOR_PATH" --deployment-descriptor-path "$DEPLOYMENT_DESCRIPTOR_PATH")
    fi

    local volume_args=()
    for vol in "${EXTRA_VOLUMES[@]}"; do
        volume_args+=(--volume "$vol")
    done

    if [[ $DOCKER_IMAGE == "none" ]]; then
        mpirun --host "$HOSTS" \
            --mca btl_tcp_if_exclude docker0,lo,tailscale0 \
            --tag-output \
            ./build/tools/scaleout/run_cluster_validation \
            "${descriptor_args[@]}" \
            --send-traffic \
            --num-iterations 10
    else
        ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
            --empty-entrypoint \
            "${volume_args[@]}" \
            --host "$HOSTS" \
            ./build/tools/scaleout/run_cluster_validation \
            "${descriptor_args[@]}" \
            --send-traffic \
            --num-iterations 10
    fi
}

echo "Using hosts: $HOSTS"
echo "Using docker image: $DOCKER_IMAGE"
if [[ -n "$FACTORY_DESCRIPTOR_PATH" ]]; then
    echo "Factory descriptor path: $FACTORY_DESCRIPTOR_PATH"
else
    echo "Cabling descriptor path: $CABLING_DESCRIPTOR_PATH"
    echo "Deployment descriptor path: $DEPLOYMENT_DESCRIPTOR_PATH"
fi
echo "Number of iterations: $ITERATIONS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

FAILED_ITERATIONS=()

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

        echo "Running tt-smi -r..."
        mpirun --host "$HOSTS" --mca btl_tcp_if_exclude docker0,lo,tailscale0 tt-smi -r

        sleep 5

        echo ""
        echo "Running cluster validation..."
        run_cluster_validation
        VALIDATION_EXIT_CODE=${PIPESTATUS[0]:-$?}
        echo "VALIDATION_EXIT_CODE=$VALIDATION_EXIT_CODE"
        echo "Iteration $i completed at $(date)"
        echo "=========================================="
    } 2>&1 | tee "$LOG_FILE"

    if grep -q "FAULTY LINKS REPORT" "$LOG_FILE"; then
        echo ""
        echo "*** ITERATION $i: FAULTY CONNECTIONS DETECTED ***"
        # Extract the faulty links summary table from the log
        sed -n '/FAULTY LINKS REPORT/,/^$/p' "$LOG_FILE"
        echo ""
    fi

    if grep -q "Workload execution timed out" "$LOG_FILE"; then
        echo ""
        echo "*** ITERATION $i: WORKLOAD TIMED OUT - cluster in unhealthy state ***"
        grep "timed out" "$LOG_FILE"
        echo ""
    fi

    if grep -q "Total Faulty Links:" "$LOG_FILE"; then
        FAULTY_COUNT=$(grep "Total Faulty Links:" "$LOG_FILE" | head -1 | grep -o '[0-9]\+')
        echo "*** Iteration $i: $FAULTY_COUNT faulty link(s) detected. See $LOG_FILE for full details. ***"
        FAILED_ITERATIONS+=("$i")
    fi

    if grep -q "missing" "$LOG_FILE" 2>/dev/null || grep -q "MISSING" "$LOG_FILE" 2>/dev/null; then
        echo "*** Iteration $i: Missing connections detected during connectivity validation. ***"
        grep -i "missing" "$LOG_FILE" | head -20
    fi

    if [[ "$RERUN_ON_RETRAIN" == true ]] && grep -q "Ethernet Links were Retrained" "$LOG_FILE"; then
        OUTPUT_DIR_RETRY="${OUTPUT_DIR}_retry"

        mkdir -p "$OUTPUT_DIR_RETRY"

        LOG_FILE_RETRY="$OUTPUT_DIR_RETRY/cluster_validation_iteration_${i}_retry.log"

        {
            echo "=========================================="
            echo "Iteration: $i - retry due to retrained links"
            echo "Timestamp: $(date)"
            echo "=========================================="
            echo ""

            echo "Re-running cluster validation..."
            run_cluster_validation
            VALIDATION_EXIT_CODE=${PIPESTATUS[0]:-$?}
            echo "VALIDATION_EXIT_CODE=$VALIDATION_EXIT_CODE"
            echo "Iteration $i retry completed at $(date)"
            echo "=========================================="
        } 2>&1 | tee "$LOG_FILE_RETRY"

        if grep -q "FAULTY LINKS REPORT" "$LOG_FILE_RETRY"; then
            echo ""
            echo "*** ITERATION $i RETRY: FAULTY CONNECTIONS DETECTED ***"
            sed -n '/FAULTY LINKS REPORT/,/^$/p' "$LOG_FILE_RETRY"
            echo ""
        fi

        if grep -q "Total Faulty Links:" "$LOG_FILE_RETRY"; then
            FAULTY_COUNT=$(grep "Total Faulty Links:" "$LOG_FILE_RETRY" | head -1 | grep -o '[0-9]\+')
            echo "*** Iteration $i retry: $FAULTY_COUNT faulty link(s) detected. See $LOG_FILE_RETRY for full details. ***"
        fi
    fi

    echo "Iteration $i logged to $LOG_FILE"
    echo ""
done

echo "All $ITERATIONS iterations completed!"

if [[ ${#FAILED_ITERATIONS[@]} -gt 0 ]]; then
    echo ""
    echo "=========================================="
    echo "FAILURE SUMMARY"
    echo "=========================================="
    echo "${#FAILED_ITERATIONS[@]} of $ITERATIONS iterations had faulty links: ${FAILED_ITERATIONS[*]}"
    echo ""
    echo "Faulty connection details per failed iteration:"
    for fail_iter in "${FAILED_ITERATIONS[@]}"; do
        FAIL_LOG="$OUTPUT_DIR/cluster_validation_iteration_${fail_iter}.log"
        echo ""
        echo "--- Iteration $fail_iter ($FAIL_LOG) ---"
        sed -n '/FAULTY LINKS REPORT/,/^-\+$/p' "$FAIL_LOG"
    done
    echo "=========================================="
else
    echo "All iterations passed with no faulty links detected."
fi
