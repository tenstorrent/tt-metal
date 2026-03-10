#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> [OPTIONS]

Recover cluster: distributed tt-smi reset + cluster validation.

Required Options:
    --hosts <host-list>                     Comma-separated list of hosts

Optional:
    --config <4x32|8x16>                    Mesh configuration (default: 4x32)
    --use-docker <docker-image>             Run validation via mpi-docker with the given image
                                            (if not provided, uses plain mpirun with local build)
    --num-iterations <number>               Number of validation iterations (default: 5)
    --sleep-duration <seconds>              Sleep duration after reset, before validation (default: 5)
    --skip-reset                            Skip tt-smi reset, only run validation
    --skip-validation                       Skip validation, only run tt-smi reset
    --no-send-traffic                       Disable --send-traffic in cluster validation

    --cabling-descriptor-path <path>        Path to cabling descriptor file (4x32 only, overrides --config default)
                                            (default: /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto)
    --deployment-descriptor-path <path>     Path to deployment descriptor file (4x32 only, overrides --config default)
                                            (default: /data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto)
    --factory-descriptor-path <path>        Path to factory system descriptor file (overrides --config defaults;
                                            when provided, cabling and deployment descriptors are ignored)
                                            (8x16 default: /data/scaleout_configs/5xBH_8x16_intrapod/fsd.textproto)
    --help                                  Display this help message and exit

Example:
    $0 --hosts bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08

    $0 --hosts bh-glx-c01u02,bh-glx-c01u08 --config 8x16 --num-iterations 10

    $0 --hosts bh-glx-c01u02,bh-glx-c01u08 --skip-reset
EOF
}

HOSTS=""
CONFIG="4x32"
DOCKER_IMAGE=""
NUM_ITERATIONS=5
SLEEP_DURATION=5
SKIP_RESET=false
SKIP_VALIDATION=false
SEND_TRAFFIC=true

CABLING_DESCRIPTOR_PATH_DEFAULT="/data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto"
DEPLOYMENT_DESCRIPTOR_PATH_DEFAULT="/data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto"
FACTORY_DESCRIPTOR_PATH_8x16_DEFAULT="/data/scaleout_configs/5xBH_8x16_intrapod/fsd.textproto"

CABLING_DESCRIPTOR_PATH=""
DEPLOYMENT_DESCRIPTOR_PATH=""
FACTORY_DESCRIPTOR_PATH=""

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
        --use-docker)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --use-docker requires a non-empty value"
                exit 1
            fi
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --num-iterations)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --num-iterations requires a non-empty value"
                exit 1
            fi
            if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: --num-iterations must be a positive integer, got '$2'"
                exit 1
            fi
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --sleep-duration)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --sleep-duration requires a non-empty value"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --sleep-duration must be a non-negative integer, got '$2'"
                exit 1
            fi
            SLEEP_DURATION="$2"
            shift 2
            ;;
        --skip-reset)
            SKIP_RESET=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --no-send-traffic)
            SEND_TRAFFIC=false
            shift
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

if [[ "$SKIP_RESET" == true && "$SKIP_VALIDATION" == true ]]; then
    echo "Error: cannot use both --skip-reset and --skip-validation"
    exit 1
fi

# Resolve descriptor paths based on config when not explicitly provided
if [[ -n "$FACTORY_DESCRIPTOR_PATH" ]]; then
    : # explicit factory descriptor overrides everything
elif [[ "$CONFIG" == "8x16" ]]; then
    FACTORY_DESCRIPTOR_PATH="$FACTORY_DESCRIPTOR_PATH_8x16_DEFAULT"
else
    # 4x32 config uses cabling + deployment descriptors
    [[ -z "$CABLING_DESCRIPTOR_PATH" ]] && CABLING_DESCRIPTOR_PATH="$CABLING_DESCRIPTOR_PATH_DEFAULT"
    [[ -z "$DEPLOYMENT_DESCRIPTOR_PATH" ]] && DEPLOYMENT_DESCRIPTOR_PATH="$DEPLOYMENT_DESCRIPTOR_PATH_DEFAULT"
fi

# Build descriptor args array for run_cluster_validation
DESCRIPTOR_ARGS=()
if [[ -n "$FACTORY_DESCRIPTOR_PATH" ]]; then
    DESCRIPTOR_ARGS+=(--factory-descriptor-path "$FACTORY_DESCRIPTOR_PATH")
else
    DESCRIPTOR_ARGS+=(--cabling-descriptor-path "$CABLING_DESCRIPTOR_PATH" --deployment-descriptor-path "$DEPLOYMENT_DESCRIPTOR_PATH")
fi

# Print summary
echo "=========================================="
echo "Cluster recovery"
echo "Using hosts: $HOSTS"
echo "Configuration: $CONFIG"
if [[ -n "$DOCKER_IMAGE" ]]; then
    echo "Docker image: $DOCKER_IMAGE"
else
    echo "Using local build (no docker)"
fi
if [[ -n "$FACTORY_DESCRIPTOR_PATH" ]]; then
    echo "Factory descriptor: $FACTORY_DESCRIPTOR_PATH"
else
    echo "Cabling descriptor: $CABLING_DESCRIPTOR_PATH"
    echo "Deployment descriptor: $DEPLOYMENT_DESCRIPTOR_PATH"
fi
echo "Num iterations: $NUM_ITERATIONS"
echo "Send traffic: $SEND_TRAFFIC"
echo "Sleep after reset: ${SLEEP_DURATION}s"
echo "Skip reset: $SKIP_RESET"
echo "Skip validation: $SKIP_VALIDATION"
echo "=========================================="
echo ""

# Step 1: tt-smi reset
if [[ "$SKIP_RESET" == false ]]; then
    echo "Running tt-smi -glx_reset..."
    mpirun --host "$HOSTS" --mca btl_tcp_if_exclude docker0,lo,tailscale0 tt-smi -glx_reset

    echo ""
    echo "Sleeping ${SLEEP_DURATION}s..."
    sleep "$SLEEP_DURATION"
else
    echo "Skipping tt-smi reset (--skip-reset)"
fi

# Step 2: Cluster validation
if [[ "$SKIP_VALIDATION" == false ]]; then
    VALIDATION_ARGS=("${DESCRIPTOR_ARGS[@]}")
    if [[ "$SEND_TRAFFIC" == true ]]; then
        VALIDATION_ARGS+=(--send-traffic)
    fi
    VALIDATION_ARGS+=(--num-iterations "$NUM_ITERATIONS")

    echo ""
    echo "Running cluster validation..."
    if [[ -n "$DOCKER_IMAGE" ]]; then
        ./tools/scaleout/exabox/mpi-docker --host "$HOSTS" \
            --image "$DOCKER_IMAGE" \
            --empty-entrypoint \
            ./build/tools/scaleout/run_cluster_validation \
            "${VALIDATION_ARGS[@]}"
    else
        mpirun --host "$HOSTS" \
            --mca btl_tcp_if_exclude docker0,lo,tailscale0 \
            --tag-output \
            ./build/tools/scaleout/run_cluster_validation \
            "${VALIDATION_ARGS[@]}"
    fi
else
    echo "Skipping validation (--skip-validation)"
fi

echo ""
echo "=========================================="
echo "Recovery completed at $(date)"
echo "=========================================="
