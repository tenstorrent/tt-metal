#!/bin/bash

set -eo pipefail

# Source MPI interface validation utility
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/mpi_if_selection.sh"
source "$SCRIPT_DIR/utils/host_utils.sh"

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> [OPTIONS]

Recover cluster: distributed tt-smi reset + cluster validation.

Required Options:
    --hosts <host-list>                     Comma-separated list of hosts

Optional:
    --config <4x32|8x16>                    Mesh configuration (default: 4x32)
    --use-docker [docker-image]             Run validation via mpi-docker. If an image is given, uses it;
                                            if the flag is passed with no image, uses the default:
                                            $DOCKER_IMAGE_DEFAULT
                                            (if the flag is omitted entirely, uses plain mpirun with local build)
    --num-iterations <number>               Number of validation iterations (default: 5)
                                            This is the inner per-run validation loop.
    --max-attempts <number>                 Number of times to run the full recovery (reset + validation)
                                            before giving up, or until it succeeds (default: 1).
                                            This is the outer loop wrapping the whole recovery.
    --sleep-duration <seconds>              Sleep duration after reset, before validation (default: 5)
    --skip-reset                            Skip tt-smi reset, only run validation
    --skip-validation                       Skip validation, only run tt-smi reset
    --skip-version-check                     Skip the tt-smi/KMD/firmware version checks run on all hosts
                                            before recovery (see minimum versions at top of this script)
    --no-send-traffic                       Disable --send-traffic in cluster validation
    --check                                 Dry run: verify MPI can reach all hosts via hostname, then exit
    --mpi-if <interface>                    Network interface for MPI TCP transport
                                            (auto-detected if not specified)
    --mpi-args <args>                       Extra arguments passed directly to mpirun (quoted string)
                                            e.g. --mpi-args "--tag-output"
    --output <directory>                    Output directory for logs and validation artifacts
                                            (default: "<comma-separated-hosts>-<timestamp>").
                                            Passed to run_cluster_validation as --output-path so the
                                            unretrainable_channels.yaml artifact lands here too.

    --cabling-descriptor-path <path>        Path to cabling descriptor file (4x32 only, overrides --config default)
                                            (default: /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto)
    --deployment-descriptor-path <path>     Path to deployment descriptor file (4x32 only, overrides --config default)
                                            (default: /data/scaleout_configs/bh_glx_exabox/deployment_descriptor.textproto)
    --factory-descriptor-path <path>        Path to factory system descriptor file (overrides --config defaults;
                                            when provided, cabling and deployment descriptors are ignored)
                                            (8x16 default: /data/scaleout_configs/5xBH_8x16_intrapod/fsd.textproto)
    --rerun-on-retrain                      Rerun validation when Ethernet links are retrained
                                            (the underlying tool early-exits after a successful retrain without sending
                                            traffic; this reruns it to actually validate the cluster)
    --validation-args <args>                Extra arguments passed verbatim to run_cluster_validation (quoted string)
                                            e.g. --validation-args "--min-connections 2 --hard-fail"
                                            Use this for any run_cluster_validation flag (relaxed validation, strict
                                            failure, connectivity prints, metrics logging, etc.)
    --no-regenerate-on-failure              Disable automatic descriptor regeneration after an unrecoverable
                                            validation failure. By default, when run_cluster_validation
                                            exhausts its retrain budget and emits unretrainable_channels.yaml,
                                            recover.sh invokes run_regen_descriptors to write a degraded
                                            descriptor set (FSD + cabling + deployment) to <output>/regenerated.
                                            In --use-docker mode regen runs inside the image on the first host.
                                            Regen is skipped automatically when only --factory-descriptor-path is
                                            in use (cabling+deployment are required inputs).
    --help                                  Display this help message and exit

================================================================================
To see the full list of run_cluster_validation flags forwardable via
--validation-args, run (no cluster needed, --hosts is not required):

    $0 --use-docker <image> --validation-args "--help"
================================================================================

Example:
    $0 --hosts bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08

    $0 --hosts bh-glx-c01u02,bh-glx-c01u08 --config 8x16 --num-iterations 10

    $0 --hosts bh-glx-c01u02,bh-glx-c01u08 --skip-reset

    $0 --hosts bh-glx-d03u02,bh-glx-d03u08 --check

    $0 --hosts bh-glx-d03u02,bh-glx-d03u08 --mpi-if ens5f0np0 --mpi-args "--tag-output"
EOF
}

HOSTS=""
CONFIG="4x32"
DOCKER_IMAGE=""
DOCKER_IMAGE_DEFAULT="ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.74.0-dev20260620-6-gd9d52dfe7b6"
NUM_ITERATIONS=5
MAX_ATTEMPTS=1
SLEEP_DURATION=5
SKIP_RESET=false
SKIP_VALIDATION=false
SKIP_VERSION_CHECK=false
SEND_TRAFFIC=true
CHECK=false
MPI_IF=""
MPI_IF_EXPLICIT=false
MPI_EXTRA_ARGS=()
OUTPUT_DIR=""  # default computed after --hosts is known: "<comma-separated-hosts>-<timestamp>"
RERUN_ON_RETRAIN=false
VALIDATION_EXTRA_ARGS=()
REGENERATE_ON_FAILURE=true

# Minimum required versions, checked on every host before recovery (see --skip-version-check).
# Bump these as the required baseline moves.
TT_SMI_MIN_VERSION="5.2.0"   # `tt-smi --version`
KMD_MIN_VERSION="2.9.0"      # `cat /sys/module/tenstorrent/version`
FW_MIN_VERSION="19.11"       # `cat /sys/class/tenstorrent/tenstorrent!<n>/tt_fw_bundle_ver`

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
                # No value provided: fall back to the default image.
                DOCKER_IMAGE="$DOCKER_IMAGE_DEFAULT"
                shift
            else
                DOCKER_IMAGE="$2"
                shift 2
            fi
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
        --max-attempts)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --max-attempts requires a non-empty value"
                exit 1
            fi
            if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: --max-attempts must be a positive integer, got '$2'"
                exit 1
            fi
            MAX_ATTEMPTS="$2"
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
        --skip-version-check)
            SKIP_VERSION_CHECK=true
            shift
            ;;
        --no-send-traffic)
            SEND_TRAFFIC=false
            shift
            ;;
        --check)
            CHECK=true
            shift
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
        --output)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --output requires a non-empty value"
                exit 1
            fi
            OUTPUT_DIR="$2"
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
        --rerun-on-retrain)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                echo "Error: --rerun-on-retrain does not accept a value"
                exit 1
            fi
            RERUN_ON_RETRAIN=true
            shift
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
        --no-regenerate-on-failure)
            REGENERATE_ON_FAILURE=false
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

# If the operator forwarded --help / -h through --validation-args, just print
# run_cluster_validation --help from the docker image and exit. Short-circuits
# before --hosts validation since no cluster operation is performed.
for _arg in "${VALIDATION_EXTRA_ARGS[@]}"; do
    if [[ "$_arg" == "--help" || "$_arg" == "-h" ]]; then
        if [[ -z "$DOCKER_IMAGE" ]]; then
            echo "Error: --validation-args \"--help\" requires --use-docker <image>"
            echo "Example: $0 --use-docker <ghcr-image> --validation-args \"--help\""
            exit 1
        fi
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

if [[ "$SKIP_RESET" == true && "$SKIP_VALIDATION" == true ]]; then
    echo "Error: cannot use both --skip-reset and --skip-validation"
    exit 1
fi

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

# Set log file path inside output directory (captures actual start time).
# Resolve to an absolute path so it can be bind-mounted into Docker containers
# (regen runs inside the image in --use-docker mode) and referenced identically
# inside and outside the container.
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
LOG_FILE="$OUTPUT_DIR/recover_$(date +%Y%m%d_%H%M%S).log"

# Redirect all output: terminal sees colors, log file gets ANSI/CR stripped
exec > >(tee >(sed 's/\x1b\[[0-9;]*[mJKHABCDfsuGMF]//g; s/\r//g' > "$LOG_FILE")) 2>&1
echo "Logging to: $LOG_FILE"

# --check: dry run to verify MPI can reach all hosts, then exit
if [[ "$CHECK" == true ]]; then
    echo "=========================================="
    echo "MPI connectivity check"
    echo "Using hosts: $HOSTS"
    echo "MPI interface: $MPI_IF"
    echo "=========================================="
    mpirun --host "$HOSTS" \
        --mca btl_tcp_if_include "$MPI_IF" \
        "${MPI_EXTRA_ARGS[@]}" \
        hostname
    echo "=========================================="
    echo "Check complete at $(date)"
    echo "=========================================="
    exit 0
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
echo "MPI interface: $MPI_IF"
if [[ "${#MPI_EXTRA_ARGS[@]}" -gt 0 ]]; then
    echo "MPI extra args: ${MPI_EXTRA_ARGS[*]}"
fi
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
echo "Max attempts: $MAX_ATTEMPTS"
echo "Send traffic: $SEND_TRAFFIC"
echo "Sleep after reset: ${SLEEP_DURATION}s"
echo "Skip reset: $SKIP_RESET"
echo "Skip validation: $SKIP_VALIDATION"
echo "Skip version check: $SKIP_VERSION_CHECK"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Rerun on retrain: $RERUN_ON_RETRAIN"
if [[ ${#VALIDATION_EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra validation args: ${VALIDATION_EXTRA_ARGS[*]}"
fi
echo "Regenerate on failure: $REGENERATE_ON_FAILURE"
echo "=========================================="
echo ""

# Step 0: assert minimum tt-smi / KMD / firmware versions on every host.
# These are host-level (independent of --use-docker), so the check always runs via plain mpirun.
if [[ "$SKIP_VERSION_CHECK" == false ]]; then
    echo "Checking tt-smi/KMD/firmware versions on all hosts..."
    # Single-quoted heredoc: nothing expands locally. Minimum versions are passed as positional
    # args ($1/$2/$3) so the script body stays literal and runs identically on every rank.
    read -r -d '' VERSION_CHECK_CMD <<'VERSION_CHECK_CMD' || true
set -o pipefail
h=$(hostname)
tt_smi_min="$1"
kmd_min="$2"
fw_min="$3"
fail=0

# version_ge A B -> true (0) if A >= B, using version-aware ordering.
version_ge() {
    [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]
}

# 1) tt-smi version
tt_smi_ver=$(tt-smi --version 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)
if [[ -z "$tt_smi_ver" ]]; then
    printf '[%s] ERROR: could not determine tt-smi version (is tt-smi installed?)\n' "$h"
    fail=1
elif ! version_ge "$tt_smi_ver" "$tt_smi_min"; then
    printf '[%s] ERROR: tt-smi version %s < required %s\n' "$h" "$tt_smi_ver" "$tt_smi_min"
    fail=1
else
    printf '[%s] OK: tt-smi version %s (>= %s)\n' "$h" "$tt_smi_ver" "$tt_smi_min"
fi

# 2) KMD (kernel driver) version
kmd_ver=$(cat /sys/module/tenstorrent/version 2>/dev/null)
if [[ -z "$kmd_ver" ]]; then
    printf '[%s] ERROR: could not read /sys/module/tenstorrent/version (is the tenstorrent KMD loaded?)\n' "$h"
    fail=1
elif ! version_ge "$kmd_ver" "$kmd_min"; then
    printf '[%s] ERROR: KMD version %s < required %s\n' "$h" "$kmd_ver" "$kmd_min"
    fail=1
else
    printf '[%s] OK: KMD version %s (>= %s)\n' "$h" "$kmd_ver" "$kmd_min"
fi

# 3) Firmware bundle version, checked on every tenstorrent device present on the host
fw_found=0
for f in /sys/class/tenstorrent/tenstorrent!*/tt_fw_bundle_ver; do
    [[ -e "$f" ]] || continue
    fw_found=1
    fw_ver=$(cat "$f" 2>/dev/null)
    if [[ -z "$fw_ver" ]]; then
        printf '[%s] ERROR: could not read %s\n' "$h" "$f"
        fail=1
    elif ! version_ge "$fw_ver" "$fw_min"; then
        printf '[%s] ERROR: firmware version %s on %s < required %s\n' "$h" "$fw_ver" "$f" "$fw_min"
        fail=1
    fi
done
if [[ "$fw_found" -eq 0 ]]; then
    printf '[%s] ERROR: no tenstorrent devices found under /sys/class/tenstorrent\n' "$h"
    fail=1
elif [[ "$fail" -eq 0 ]]; then
    printf '[%s] OK: firmware version >= %s on all devices\n' "$h" "$fw_min"
fi

exit "$fail"
VERSION_CHECK_CMD
    # `if !` suspends `set -e`, so a failing rank is handled here instead of aborting abruptly.
    if ! mpirun --host "$HOSTS" \
        --mca btl_tcp_if_include "$MPI_IF" \
        "${MPI_EXTRA_ARGS[@]}" \
        bash -c "$VERSION_CHECK_CMD" _ "$TT_SMI_MIN_VERSION" "$KMD_MIN_VERSION" "$FW_MIN_VERSION"; then
        echo ""
        echo "Error: version check failed on one or more hosts (see above)."
        echo "       Required: tt-smi >= $TT_SMI_MIN_VERSION, KMD >= $KMD_MIN_VERSION, firmware >= $FW_MIN_VERSION."
        echo "       Re-run with --skip-version-check to bypass."
        exit 1
    fi
    echo "Version check passed on all hosts."
    echo ""
else
    echo "Skipping version check (--skip-version-check)"
    echo ""
fi

# Outer recovery loop: run the full reset + validation up to MAX_ATTEMPTS times, or until
# validation succeeds. --num-iterations controls the inner validation loop; this controls how
# many times the whole recovery is retried. Descriptor regeneration (Step 3) runs once after the
# loop, only if every attempt failed.
UNRETRAINABLE_YAML="$OUTPUT_DIR/unretrainable_channels.yaml"
VALIDATION_EXIT=0
for (( ATTEMPT=1; ATTEMPT<=MAX_ATTEMPTS; ATTEMPT++ )); do
echo "=========================================="
echo "Recovery attempt $ATTEMPT of $MAX_ATTEMPTS"
echo "=========================================="

# All attempts share $OUTPUT_DIR, but unretrainable_channels.yaml is only (re)written on one
# specific validation failure path. Clear any artifact from a prior attempt so Step 3 can only
# regenerate descriptors from evidence produced by the current (latest post-reset) attempt.
rm -f "$UNRETRAINABLE_YAML"

# Step 1: tt-smi reset
# Note: tt-smi -glx_reset is deprecated as of tt-smi 3.1.1; use tt-smi -r if available
RESET_EXIT=0
if [[ "$SKIP_RESET" == false ]]; then
    echo "Running tt-smi -glx_reset..."
    # Host-tag every reset line for attribution. tt-smi writes key progress to the tty (not stdout) so
    # run under `script`; the tr/sed/awk pipeline collapses its animated \r/spinner output, keeps colors.
    read -r -d '' RESET_CMD <<'RESET_CMD' || true
set -o pipefail
h=$(hostname)
script -qefc "tt-smi -glx_reset" /dev/null |
    tr -d '\000' |
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
    done
ec=${PIPESTATUS[0]}
if [[ $ec -eq 0 ]]; then
    printf '[%s][%(%H:%M:%S)T] Reset completed successfully\n' "$h" -1
else
    printf '[%s][%(%H:%M:%S)T] Reset failed | Exit code: %s\n' "$h" -1 "$ec"
fi
# Propagate tt-smi's status so mpirun (and the caller) sees per-host reset failures.
exit "$ec"
RESET_CMD
    # Capture mpirun's status without tripping `set -e` (the `if` context suspends it) so a reset
    # failure retries the attempt instead of aborting the whole script. The remote command exits with
    # tt-smi's status, so mpirun returns non-zero if any host's reset failed.
    if mpirun --host "$HOSTS" \
        --mca btl_tcp_if_include "$MPI_IF" \
        "${MPI_EXTRA_ARGS[@]}" \
        bash -c "$RESET_CMD"; then RESET_EXIT=0; else RESET_EXIT=$?; fi

    if [[ $RESET_EXIT -ne 0 ]]; then
        echo ""
        echo "Reset failed on one or more hosts (exit code $RESET_EXIT)."
    else
        echo ""
        echo "Sleeping ${SLEEP_DURATION}s..."
        sleep "$SLEEP_DURATION"
    fi
else
    echo "Skipping tt-smi reset (--skip-reset)"
fi

# Step 2: Cluster validation
# VALIDATION_EXIT carries the whole attempt's outcome: a failed reset short-circuits validation and
# fails the attempt so the outer loop retries (or the script exits non-zero once attempts run out).
VALIDATION_EXIT=0
if [[ $RESET_EXIT -ne 0 ]]; then
    echo ""
    echo "Skipping validation because reset failed on this attempt."
    VALIDATION_EXIT=$RESET_EXIT
elif [[ "$SKIP_VALIDATION" == false ]]; then
    VALIDATION_ARGS=("${DESCRIPTOR_ARGS[@]}")
    if [[ "$SEND_TRAFFIC" == true ]]; then
        VALIDATION_ARGS+=(--send-traffic)
    fi
    VALIDATION_ARGS+=(--num-iterations "$NUM_ITERATIONS")
    if [[ ${#VALIDATION_EXTRA_ARGS[@]} -gt 0 ]]; then
        VALIDATION_ARGS+=("${VALIDATION_EXTRA_ARGS[@]}")
    fi
    VALIDATION_ARGS+=(--output-path "$OUTPUT_DIR")

    run_cluster_validation() {
        if [[ -n "$DOCKER_IMAGE" ]]; then
            ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
                --empty-entrypoint \
                --mpi-interface "$MPI_IF" \
                --volume /data/scaleout_configs \
                "${MPI_EXTRA_ARGS[@]}" \
                --host "$HOSTS" \
                ./build/tools/scaleout/run_cluster_validation \
                "${VALIDATION_ARGS[@]}"
        else
            mpirun --host "$HOSTS" \
                --mca btl_tcp_if_include "$MPI_IF" \
                "${MPI_EXTRA_ARGS[@]}" \
                --tag-output \
                ./build/tools/scaleout/run_cluster_validation \
                "${VALIDATION_ARGS[@]}"
        fi
    }

    echo ""
    echo "Running cluster validation..."
    VALIDATION_LOG=$(mktemp)
    # Capture the exit code without tripping `set -e` (the `if` context suspends it) so regen
    # can run on failure. pipefail makes the pipeline reflect run_cluster_validation's status.
    if run_cluster_validation 2>&1 | tee "$VALIDATION_LOG"; then VALIDATION_EXIT=0; else VALIDATION_EXIT=$?; fi

    if [[ "$RERUN_ON_RETRAIN" == true ]] && grep -q "Ethernet Links were Retrained" "$VALIDATION_LOG"; then
        echo ""
        echo "Ethernet links were retrained — rerunning validation to issue traffic..."
        if run_cluster_validation 2>&1 | tee "$VALIDATION_LOG"; then VALIDATION_EXIT=0; else VALIDATION_EXIT=$?; fi
    fi
    rm -f "$VALIDATION_LOG"
else
    echo "Skipping validation (--skip-validation)"
fi

# Outer-loop control: stop as soon as an attempt succeeds; otherwise retry until attempts exhausted.
if [[ $VALIDATION_EXIT -eq 0 ]]; then
    echo ""
    echo "Recovery succeeded on attempt $ATTEMPT of $MAX_ATTEMPTS."
    break
fi
echo ""
echo "Recovery attempt $ATTEMPT of $MAX_ATTEMPTS failed (exit code $VALIDATION_EXIT)."
if [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; then
    echo "Retrying full recovery..."
    echo ""
else
    echo "Exhausted all $MAX_ATTEMPTS recovery attempts."
fi
done

# Step 3: Regenerate descriptors if validation hit unrecoverable state.
# UNRETRAINABLE_YAML is cleared before each attempt, so any file present here was produced by the
# final (failed) attempt and reflects the latest post-reset state.
if [[ "$REGENERATE_ON_FAILURE" == true && $VALIDATION_EXIT -ne 0 ]]; then
    if [[ -f "$UNRETRAINABLE_YAML" ]]; then
        if [[ -z "$CABLING_DESCRIPTOR_PATH" || -z "$DEPLOYMENT_DESCRIPTOR_PATH" ]]; then
            echo ""
            echo "Skipping descriptor regeneration: requires --cabling-descriptor-path and"
            echo "--deployment-descriptor-path (cannot regenerate from --factory-descriptor-path alone)."
        else
            REGEN_DIR="$OUTPUT_DIR/regenerated"
            REGEN_ARGS=(
                --cabling "$CABLING_DESCRIPTOR_PATH"
                --deployment "$DEPLOYMENT_DESCRIPTOR_PATH"
                --unretrainable-channels "$UNRETRAINABLE_YAML"
                --output-dir "$REGEN_DIR"
            )
            echo ""
            echo "Validation exited unrecoverable; regenerating descriptors without unretrainable cables..."
            if [[ -n "$DOCKER_IMAGE" ]]; then
                # run_regen_descriptors is a single-host offline tool. In docker mode the binary
                # only exists inside the image, so run one rank on the first host, mounting the
                # descriptor inputs and the output dir so paths resolve identically in-container.
                # Relative input paths resolve against the container's working dir, which differs
                # from the host cwd, so warn the operator to use absolute paths.
                if [[ "$CABLING_DESCRIPTOR_PATH" != /* || "$DEPLOYMENT_DESCRIPTOR_PATH" != /* ]]; then
                    echo "Warning: --cabling-descriptor-path / --deployment-descriptor-path are relative;"
                    echo "         in --use-docker mode they may not resolve inside the container."
                    echo "         Use absolute paths if regeneration fails to find the descriptors."
                fi
                FIRST_HOST="${HOSTS%%,*}"
                REGEN_VOLUMES=(--volume /data/scaleout_configs --volume "$OUTPUT_DIR")
                # Mount the directories holding the input descriptors too, in case they live
                # outside /data/scaleout_configs (custom --cabling/--deployment paths).
                CABLING_DIR="$(cd "$(dirname "$CABLING_DESCRIPTOR_PATH")" && pwd)"
                DEPLOYMENT_DIR="$(cd "$(dirname "$DEPLOYMENT_DESCRIPTOR_PATH")" && pwd)"
                REGEN_VOLUMES+=(--volume "$CABLING_DIR" --volume "$DEPLOYMENT_DIR")
                ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
                    --empty-entrypoint \
                    --mpi-interface "$MPI_IF" \
                    "${REGEN_VOLUMES[@]}" \
                    --host "$FIRST_HOST" -np 1 \
                    ./build/tools/scaleout/run_regen_descriptors \
                    "${REGEN_ARGS[@]}" || echo "Warning: descriptor regeneration failed (see error above)"
            else
                ./build/tools/scaleout/run_regen_descriptors \
                    "${REGEN_ARGS[@]}" || echo "Warning: descriptor regeneration failed (see error above)"
            fi
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Recovery completed at $(date)"
echo "=========================================="

# Propagate validation's exit code so callers still see the failure
if [[ $VALIDATION_EXIT -ne 0 ]]; then
    exit "$VALIDATION_EXIT"
fi
