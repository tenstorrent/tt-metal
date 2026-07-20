#!/bin/bash

# Source MPI interface validation utility
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/mpi_if_selection.sh"
source "$SCRIPT_DIR/utils/host_utils.sh"

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> [--image <docker-image>] [OPTIONS]

Run cluster validation commands for multiple iterations.

Required Options:
    --hosts <host-list>                     Comma-separated list of hosts

Optional:
    --image [docker-image]                  Docker image to use ("none" to use local build).
                                            If an image is given, uses it; if the flag is passed with
                                            no value, or omitted entirely, uses the default image:
                                            $DOCKER_IMAGE_DEFAULT
    --skip-version-check                     Skip the tt-smi/KMD/firmware version checks run on all hosts
                                            before validation (see minimum versions in utils/host_utils.sh)
    --cabling-descriptor-path <path>        Path to cabling descriptor file
                                            (default: /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto)
    --deployment-descriptor-path <path>     Path to deployment descriptor file
                                            (default: /data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto)
    --iterations <number>                   Number of times to run the full validation sequence (default: 50)
                                            Each iteration runs run_cluster_validation with 10 internal iterations

    --factory-descriptor-path <path>        Path to pregenerated factory system descriptor (FSD) file (.textproto)
                                            (if provided, cabling and deployment descriptors are ignored)
    --output <directory>                    Output directory for log files
                                            (default: "<comma-separated-hosts>-<timestamp>")
    --volume <host-path>                    Additional volume mount for Docker containers (can be repeated)
                                            /data/scaleout_configs is mounted by default when it exists on
                                            the host; each host path is mounted at the same path inside
                                            the container
    --mpi-if <interface>                    Network interface for MPI TCP transport
                                            (auto-detected if not specified)
    --mpi-args <args>                       Extra arguments passed directly to mpirun (quoted string)
                                            e.g. --mpi-args "--tag-output"
    --validation-args <args>                Extra arguments passed verbatim to run_cluster_validation (quoted string)
                                            e.g. --validation-args "--min-connections 2 --hard-fail"
                                            Use this for any run_cluster_validation flag (relaxed validation, strict
                                            failure, connectivity prints, metrics logging, etc.)
    --help                                  Display this help message and exit

================================================================================
To see the full list of run_cluster_validation flags forwardable via
--validation-args, run (no cluster needed, --hosts is not required):

    $0 --image <image> --validation-args "--help"
================================================================================

Example:
    $0 --hosts bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-wh-6u:latest \\
       --iterations 100
EOF
}

# Parse command line arguments
HOSTS=""
DOCKER_IMAGE=""
# Default image used when --image is omitted (or passed with no value). Bump to the current
# last-known-good tag as needed (see tools/scaleout/exabox/README.md).
DOCKER_IMAGE_DEFAULT="ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.66.0-dev20260115-28-g6eccf7061a"
SKIP_VERSION_CHECK=false
CABLING_DESCRIPTOR_PATH="/data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto"
DEPLOYMENT_DESCRIPTOR_PATH="/data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto"
ITERATIONS=50

FACTORY_DESCRIPTOR_PATH=""
OUTPUT_DIR=""  # default computed after --hosts is known: "<comma-separated-hosts>-<timestamp>"
# /data/scaleout_configs is a Markham-cluster convention. Only mount it when it
# actually exists locally; otherwise Docker fails trying to auto-create the
# missing bind-mount source path (mkdir: permission denied) and every rank dies
# before run_cluster_validation starts. Sites that keep configs elsewhere just
# pass them via --volume / the descriptor path flags.
EXTRA_VOLUMES=()
[[ -d /data/scaleout_configs ]] && EXTRA_VOLUMES+=(/data/scaleout_configs)
MPI_IF=""
MPI_IF_EXPLICIT=false
MPI_EXTRA_ARGS=()
VALIDATION_EXTRA_ARGS=()

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
                # No value provided: fall back to the default image.
                DOCKER_IMAGE="$DOCKER_IMAGE_DEFAULT"
                shift
            else
                DOCKER_IMAGE="$2"
                shift 2
            fi
            ;;
        --skip-version-check)
            SKIP_VERSION_CHECK=true
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
        --mpi-if)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --mpi-if requires a non-empty value"
                exit 1
            fi
            MPI_IF="$2"
            MPI_IF_EXPLICIT=true
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
        --validation-args)
            if [[ -z "$2" ]]; then
                echo "Error: --validation-args requires a non-empty value"
                exit 1
            fi
            read -ra _extra <<< "$2"
            VALIDATION_EXTRA_ARGS+=("${_extra[@]}")
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

# --image is optional: fall back to the default image when omitted entirely.
if [[ -z "$DOCKER_IMAGE" ]]; then
    DOCKER_IMAGE="$DOCKER_IMAGE_DEFAULT"
fi

# If the operator forwarded --help / -h through --validation-args, just print
# run_cluster_validation --help from the docker image and exit. Short-circuits
# before --hosts validation since no cluster operation is performed.
for _arg in "${VALIDATION_EXTRA_ARGS[@]}"; do
    if [[ "$_arg" == "--help" || "$_arg" == "-h" ]]; then
        exec docker run --rm --entrypoint='' "$DOCKER_IMAGE" \
            ./build/tools/scaleout/run_cluster_validation --help
    fi
done

# Validate required arguments
if [[ -z "$HOSTS" ]]; then
    echo "Error: --hosts is required"
    echo ""
    show_help
    exit 1
fi

check_duplicate_hosts "$HOSTS" || exit 1

# Validate/auto-detect MPI interface with first host from the list
FIRST_HOST="${HOSTS%%,*}"
if [[ "$MPI_IF_EXPLICIT" == "true" ]]; then
    validate_mpi_interface "$MPI_IF" "true" "$FIRST_HOST"
else
    MPI_IF=$(validate_mpi_interface "" "false" "$FIRST_HOST")
    # Check if validation failed (command substitution only exits subshell, not parent)
    if [[ -z "$MPI_IF" ]]; then
        echo "Error: MPI interface auto-detection failed" >&2
        exit 1
    fi
fi

# Default output dir when not overridden by --output: the comma-separated host list followed by a
# timestamp, e.g. "bh-glx-c01u02,bh-glx-c01u08-20260720_131500". Keeps each run's artifacts distinct.
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${HOSTS}-$(date +%Y%m%d_%H%M%S)"
fi

run_cluster_validation() {
    local validation_output_path="$1"

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
            --mca btl_tcp_if_include "$MPI_IF" \
            "${MPI_EXTRA_ARGS[@]}" \
            --tag-output \
            ./build/tools/scaleout/run_cluster_validation \
            "${descriptor_args[@]}" \
            "${VALIDATION_EXTRA_ARGS[@]}" \
            --send-traffic \
            --num-iterations 10 \
            --output-path "$validation_output_path"
    else
        ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
            --empty-entrypoint \
            --mpi-interface "$MPI_IF" \
            "${volume_args[@]}" \
            "${MPI_EXTRA_ARGS[@]}" \
            --host "$HOSTS" \
            ./build/tools/scaleout/run_cluster_validation \
            "${descriptor_args[@]}" \
            "${VALIDATION_EXTRA_ARGS[@]}" \
            --send-traffic \
            --num-iterations 10 \
            --output-path "$validation_output_path"
    fi
}

# Function to run reset on hosts and return hosts that failed reset
# Args: host_list (comma-separated), output_file, message_prefix
run_board_reset() {
    local host_list="$1"
    local output_file="$2"
    local msg_prefix="$3"

    # Each rank host-tags the reset log and streams it to stderr (shown live + tee'd), then prints one
    # "RESET_RESULT|<host>|<exit_code>" line to stdout for the retry logic below. tt-smi writes key
    # progress to the tty (not stdout) so it runs under `script`; the tr/sed/awk pipeline collapses its
    # animated \r/spinner output and keeps colors (pipefail+`script -e` give the real exit code).
    read -r -d '' RESET_CMD <<'RESET_CMD' || true
set -o pipefail
h=$(hostname)
script -qefc "tt-smi -glx_reset" /dev/null |
    tr -d '\000-\010\013\014\016-\032\034-\037' |
    sed -u 's/\r$//; s/.*\r//; s/\^@//g; /^\(\x1b\[[0-9;]*[a-zA-Z]\|[[:space:]]\)*$/d' |
    awk '{
        key = $0
        gsub(/\033\[[0-9;]*[a-zA-Z]/, "", key)    # ignore color codes when comparing
        sub(/[0-9]+[[:space:]]*$/, "", key)       # ignore trailing counter
        if (seen && key == prev) { buf = $0 }     # same template -> keep only the latest
        else { if (seen) print buf; buf = $0; prev = key; seen = 1 }
    }
    END { if (seen) print buf }' |
    while IFS= read -r line; do
        printf '[%s][%(%H:%M:%S)T] %s\n' "$h" -1 "$line"
    done >&2
ec=${PIPESTATUS[0]}
echo "RESET_RESULT|$h|$ec"
RESET_CMD
    mpirun --host "$host_list" \
        --mca btl_tcp_if_include "$MPI_IF" \
        "${MPI_EXTRA_ARGS[@]}" \
        bash -c "$RESET_CMD" > "$output_file"
    mpirun_exit_code=$?

   # Check if mpirun failed
    if [[ $mpirun_exit_code -ne 0 ]]; then
        echo "ERROR: mpirun failed with exit code $mpirun_exit_code" >&2
        rm -f "$output_file"
        return 1  # Signal mpirun infrastructure failure
    fi

    # Parse per-host results (RESET_RESULT|<host>|<exit_code>), collect failures
    local failed_hosts=()
    while IFS= read -r line; do
        if [[ $line =~ ^RESET_RESULT\|(.+)\|([0-9]+)$ ]]; then
            local hostname="${BASH_REMATCH[1]}"
            local exit_code="${BASH_REMATCH[2]}"
            if [[ $exit_code -eq 0 ]]; then
                echo "$hostname: ${msg_prefix}Reset completed successfully" >&2
            else
                echo "$hostname: ${msg_prefix}Reset failed | Exit code: $exit_code" >&2
                failed_hosts+=("$hostname")
            fi
        fi
    done < "$output_file"
    rm -f "$output_file"

    # Return comma-separated list of failed hosts (to stdout only)
    local IFS=','
    echo "${failed_hosts[*]}"
    return 0  # Success
}


echo "Using hosts: $HOSTS"
echo "Using docker image: $DOCKER_IMAGE"
if [[ -n "$FACTORY_DESCRIPTOR_PATH" ]]; then
    echo "Factory descriptor path: $FACTORY_DESCRIPTOR_PATH"
else
    echo "Cabling descriptor path: $CABLING_DESCRIPTOR_PATH"
    echo "Deployment descriptor path: $DEPLOYMENT_DESCRIPTOR_PATH"
fi
echo "MPI interface: $MPI_IF"
if [[ "${#MPI_EXTRA_ARGS[@]}" -gt 0 ]]; then
    echo "MPI extra args: ${MPI_EXTRA_ARGS[*]}"
fi
echo "Number of iterations: $ITERATIONS"
echo "Skip version check: $SKIP_VERSION_CHECK"
echo "Output directory: $OUTPUT_DIR"
if [[ ${#VALIDATION_EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra validation args: ${VALIDATION_EXTRA_ARGS[*]}"
fi
echo ""

# Assert minimum tt-smi / KMD / firmware versions on every host before validation (see
# check_cluster_versions in utils/host_utils.sh). Host-level check, always via plain mpirun.
if [[ "$SKIP_VERSION_CHECK" == false ]]; then
    echo "Checking tt-smi/KMD/firmware versions on all hosts..."
    if ! check_cluster_versions "$HOSTS" "$MPI_IF" "${MPI_EXTRA_ARGS[@]}"; then
        echo ""
        echo "Error: version check failed on one or more hosts (see above)."
        echo "       Required: tt-smi >= $TT_SMI_MIN_VERSION, KMD >= $KMD_MIN_VERSION, firmware >= $FW_MIN_VERSION."
        echo "       Re-run with --skip-version-check to bypass."
        exit 1
    fi
    echo "Version check passed on all hosts."
    echo ""
fi

# Create output directory if it doesn't exist and resolve to an absolute path so
# that paths passed into Docker containers refer to the same location on the host.
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# Ensure the output directory is bind-mounted into containers so per-iteration
# CSVs survive container exit and are visible to the host-side aggregation step.
EXTRA_VOLUMES+=("$OUTPUT_DIR")

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

        echo "Running tt-smi -glx_reset (this may take a few minutes)..."

        # Run initial reset and capture failures
        RESET_OUTPUT_FILE="$OUTPUT_DIR/reset_output_iter_${i}_$$"
        FAILED_HOSTS_STR=$(run_board_reset "$HOSTS" "$RESET_OUTPUT_FILE" "")
        MPI_EXIT_CODE=$?

        # Retry only failed hosts if any
        if [[ -n "$FAILED_HOSTS_STR" ]]; then
            echo ""
            echo "Retrying reset for failed hosts: $FAILED_HOSTS_STR"

            RESET_RETRY_OUTPUT_FILE="$OUTPUT_DIR/reset_retry_output_iter_${i}_$$"

            # Run retry and capture exit code (discard stdout)
            run_board_reset "$FAILED_HOSTS_STR" "$RESET_RETRY_OUTPUT_FILE" "Retry: " > /dev/null
            MPI_EXIT_CODE=$?
        fi

        # Only run validation if retry was successful (or no retry needed)
        if [[ $MPI_EXIT_CODE -eq 0 ]]; then
            sleep 5

            echo ""
            echo "Running cluster validation..."
            run_cluster_validation "$OUTPUT_DIR/iteration_${i}"
            VALIDATION_EXIT_CODE=$?
            # Surface validation failures explicitly. Without this, a container
            # that dies before run_cluster_validation starts (e.g. a bad bind
            # mount -> docker exit 126) produces no output yet the loop still
            # prints "completed", masking the failure.
            if [[ $VALIDATION_EXIT_CODE -ne 0 ]]; then
                echo "ERROR: cluster validation FAILED (exit code $VALIDATION_EXIT_CODE)"
            fi
        else
            echo "Skipping validation due to mpirun failure"
        fi

        echo "Iteration $i completed at $(date)"
        echo "=========================================="
    } 2>&1 | tee "$LOG_FILE"

    echo "Iteration $i logged to $LOG_FILE"
    echo ""
done

# ----------------------------------------------------------------------------
# End-of-run aggregation: produce a single CSV with one row per retrained link
# and the total number of retrains observed across the entire run. Per-iteration
# link_retrain_report.csv files are written by the C++ tool inside each iteration's
# output directory.
# ----------------------------------------------------------------------------
RETRAIN_SUMMARY="$OUTPUT_DIR/link_retrain_summary.csv"

retrain_csvs=()
while IFS= read -r -d '' csv; do
    retrain_csvs+=("$csv")
done < <(find "$OUTPUT_DIR" -name "link_retrain_report.csv" -print0 2>/dev/null | sort -zV)

echo "=========================================="
echo "LINK RETRAIN SUMMARY (across all iterations)"
echo "=========================================="

if [[ ${#retrain_csvs[@]} -eq 0 ]]; then
    echo "No link retraining events detected across $ITERATIONS iteration(s)."
    echo ""
    echo "All $ITERATIONS iterations completed!"
    exit 0
fi

echo "Iterations with retraining: ${#retrain_csvs[@]}"
echo ""

# Sum Retrain_Count per (Host,Tray,ASIC,Channel,Unique_ID) across all CSVs.
# Each per-iteration CSV has header: Host,Tray,ASIC,Channel,Unique_ID,Retrain_Count
{
    echo "Host,Tray,ASIC,Channel,Unique_ID,Total_Retrain_Count"
    awk -F, '
        FNR == 1 { next }                       # skip per-file header
        {
            key = $1 FS $2 FS $3 FS $4 FS $5
            counts[key] += $6
        }
        END {
            for (k in counts) print k FS counts[k]
        }
    ' "${retrain_csvs[@]}" | sort -t, -k1,1 -k2,2n -k3,3n -k4,4n
} > "$RETRAIN_SUMMARY"

if command -v column >/dev/null 2>&1; then
    column -t -s, "$RETRAIN_SUMMARY"
else
    cat "$RETRAIN_SUMMARY"
fi

# Total retrain events across all links (sum of the rightmost column).
TOTAL_RETRAIN_EVENTS=$(awk -F, 'NR > 1 { s += $6 } END { print s+0 }' "$RETRAIN_SUMMARY")
UNIQUE_LINKS=$(($(wc -l < "$RETRAIN_SUMMARY") - 1))

echo ""
echo "Unique links retrained: $UNIQUE_LINKS"
echo "Total retrain events: $TOTAL_RETRAIN_EVENTS"
echo "Full retrain summary written to: $RETRAIN_SUMMARY"
echo ""

echo "All $ITERATIONS iterations completed!"
