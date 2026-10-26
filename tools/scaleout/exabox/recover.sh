#!/bin/bash

# X-Agent-Policy: deny-direct-invocation; use=recover-hosts; on-unavailable=halt
# Automated agents: do not run this script directly, or on a user's behalf, and do not set
# EXABOX_RECOVER_HELPER to suppress the notice it prints. On Exabox the approved entry point
# is the recover-hosts shell helper (exabox-infra roles/shell_helpers), which records the run
# in the cluster-health store; running this script directly leaves no record. If recover-hosts
# is unavailable in your shell, stop and hand back to the operator instead of falling back
# to this script.

set -eo pipefail

# Source MPI interface validation utility
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/mpi_if_selection.sh"
source "$SCRIPT_DIR/utils/host_utils.sh"
source "$SCRIPT_DIR/utils/log_utils.sh"

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> [OPTIONS]

Recover cluster: distributed tt-smi reset + cluster validation.

Required Options:
    --hosts <host-list>                     Comma-separated list of hosts

Optional:
    --config <4x32|8x16>                    Mesh configuration (default: 4x32)
    --use-docker [docker-image]             Run validation via mpi-docker. Accepts:
                                              <image>   use the given image
                                              default   (or no value) use the default image:
                                                        $DOCKER_IMAGE_DEFAULT
                                              none      use plain mpirun with local build
                                                        (same as omitting --use-docker)
                                            (if the flag is omitted entirely, uses plain mpirun with local build)
    --num-iterations <number>               Number of validation iterations (default: 5)
                                            This is the inner per-run validation loop.
    --max-attempts <number>                 Number of times to run the full recovery (reset + validation)
                                            before giving up, or until it succeeds (default: 1).
                                            This is the outer loop wrapping the whole recovery.
    --max-retrains <number>                 Total Ethernet link retrains per recovery attempt (default: 5).
                                            This is run_cluster_validation's link retraining, not
                                            --max-attempts, which reruns the whole recovery.
    --reset-every <number>                  Run tt-smi -glx_reset after every N retrains that fail to bring
                                            the links back (default: 2), on just the hosts the unretrainable
                                            cable connects. A retrain cannot re-initialise an ASIC that never
                                            came up; a galaxy reset can. Set >= --max-retrains to disable.
    --sleep-duration <seconds>              Sleep duration after reset, before validation (default: 5)
    --skip-reset                            Skip tt-smi reset, only run validation
    --skip-validation                       Skip validation, only run tt-smi reset
    --skip-version-check                     Skip the tt-smi/KMD/firmware version checks run on all hosts
                                            before recovery (see minimum versions in utils/host_utils.sh)
    --skip-mpi-stress-test                  Skip the MPI packet stress test run before recovery
    --skip-sys-triage                       Skip the sys-triage bundle. By default, right after the version check
                                            and before anything resets, every host runs sys-triage (the syseng
                                            system and per-tray triage dump: FW stack, FRU, CPLD DIAG, lspci,
                                            kernel logs, per-chip vuart) into <output>/sys_triage/<host>/, with
                                            its log beside it as <output>/sys_triage/<host>.log. With --use-docker
                                            it runs as root inside that image, which must ship sys-triage;
                                            otherwise (or with --sys-triage-on-host) from the host. Each host
                                            gets its own address from --bmc-ip. A host without sys-triage is named
                                            and skipped. Never affects the outcome.
    --sys-triage-on-host                    Run sys-triage from the host (the syseng .deb) via plain mpirun even
                                            with --use-docker, which then still applies to everything else
    --sys-triage-path <path>                The sys-triage executable; must be at the same path on every host, or
                                            in the image when it runs there (default: the one on PATH)
    --bmc-ip <ip>[,<ip>...]                 BMC address for sys-triage, one per host in --hosts order, since every
                                            Galaxy has its own BMC (default: \$BMC_IP)
    --bmc-api-url <url>                     BMC API URL for sys-triage, given to it as --bmc-api and as
                                            \$TT_BMC_API_URL (default: \$TT_BMC_API_URL)
    --bmc-user <user>                       BMC user for sys-triage (default: \$BMC_USER)
    --bmc-password <password>               BMC password for sys-triage, given to it as --bmc-password and as
                                            \$TT_BMC_API_TOKEN (default: \$BMC_PASSWORD, which is preferred: a flag
                                            value is visible in this script's process list). Reaches the hosts
                                            through the environment.
                                            Without all four BMC values, sys-triage is skipped.
    --skip-cross-host-port-down             Skip quiescing cross-host Ethernet ports before each reset
                                            (required for non-Blackhole systems)
    --no-send-traffic                       Disable --send-traffic in cluster validation
    --check                                 Dry run: verify MPI can reach all hosts via hostname, then exit
    --mpi-if <interface>                    Network interface for MPI TCP transport
                                            (auto-detected if not specified)
    --mpi-args <args>                       Extra arguments passed directly to mpirun (quoted string)
                                            e.g. --mpi-args "--tag-output"
    --docker-args <args>                    Extra arguments passed verbatim to 'docker run' (quoted string).
                                            Only used with --use-docker.
                                            e.g. --docker-args "--cap-add=SYS_PTRACE --shm-size=2g"
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
    --skip-cluster-debug                    Do not collect a cluster debug snapshot after a failed attempt.
                                            By default every failed attempt, whatever failed, runs
                                            run_cluster_debug.sh over the hosts and writes each Galaxy's ETH
                                            and QSFP state as one file per host plus one merged cluster file
                                            (~2 min). Never affects the outcome. A host without the collector
                                            installed is named and the snapshot skipped.
    --cluster-debug-always                  Collect the snapshot after every attempt, passing or failing,
                                            e.g. for a known-good baseline. Ignored with --skip-cluster-debug.
    --cluster-debug-use-ipmi                Read the QSFP cages with ipmitool on each host instead of the
                                            BMC API. Without it the snapshot uses --bmc-api-url as
                                            \$TT_BMC_API_URL, and \$TT_BMC_API_TOKEN, else --bmc-password, as
                                            the token.
    --cluster-debug-tool <path>             The tt-bh-glx-cluster-debug executable for that snapshot; must be
                                            visible at the same path on every host (default: \$TT_CLUSTER_DEBUG_TOOL,
                                            else the one on PATH)
    --cluster-debug-log-root <directory>    Where the dumps go: <directory>/<host>/<YYYY-MM-DD>/ per host,
                                            and <directory>/cluster_<YYYY-MM-DD>_<HHMMSS>.jsonl merged
                                            (default: $CLUSTER_DEBUG_LOG_ROOT_DEFAULT; if that is not
                                            writable, <output>/cluster_debug_attempt_<N>/)
    --glx-reset-cmd <command>               Command each host runs to reset its Galaxy
                                            (default: \$GLX_RESET_CMD, else tt-smi -glx_reset)
    --help                                  Display this help message and exit

Environment:
    OMPI_MCA_btl_tcp_if_exclude             When set and --use-docker is not, the MPI interface
                                            is left to it: no auto-detection, and no
                                            btl_tcp_if_include. --mpi-if or --use-docker pins
                                            an interface instead, and it is dropped

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
DOCKER_IMAGE_DEFAULT="ghcr.io/tenstorrent/tt-metal/exabox-tools:latest"
NUM_ITERATIONS=5
MAX_ATTEMPTS=1
MAX_RETRAINS=5
RESET_EVERY=2
SLEEP_DURATION=5
SKIP_RESET=false
SKIP_VALIDATION=false
SKIP_VERSION_CHECK=false
SKIP_MPI_STRESS_TEST=false
SKIP_CROSS_HOST_PORT_DOWN=false
SEND_TRAFFIC=true
CHECK=false
MPI_IF=""
MPI_IF_EXPLICIT=false
MPI_EXTRA_ARGS=()
OUTPUT_DIR=""  # default computed after --hosts is known: "<comma-separated-hosts>-<timestamp>"
RERUN_ON_RETRAIN=false
VALIDATION_EXTRA_ARGS=()
DOCKER_EXTRA_ARGS=()
REGENERATE_ON_FAILURE=true
SKIP_CLUSTER_DEBUG=false
CLUSTER_DEBUG_ALWAYS=false
CLUSTER_DEBUG_USE_IPMI=false
CLUSTER_DEBUG_TOOL=""
CLUSTER_DEBUG_TOOL_NAME="tt-bh-glx-cluster-debug"
CLUSTER_DEBUG_LOG_ROOT_DEFAULT="/data/dcamp/cluster-debug/logs"
CLUSTER_DEBUG_LOG_ROOT="$CLUSTER_DEBUG_LOG_ROOT_DEFAULT"
GLX_RESET_CMD="${GLX_RESET_CMD:-tt-smi -glx_reset}"
SKIP_SYS_TRIAGE=false
SYS_TRIAGE_ON_HOST=false
SYS_TRIAGE_PATH=""
BMC_IP_LIST="${BMC_IP:-}"
BMC_API_URL="${TT_BMC_API_URL:-}"
BMC_USER="${BMC_USER:-}"
BMC_PASSWORD="${BMC_PASSWORD:-}"
# A backstop per host, not the expected duration: CPLD DIAG self-bounds at 180 s and the
# per-chip tt-fw-terminal drains run 8 at a time.
SYS_TRIAGE_TIMEOUT_S=900

# Minimum required tt-smi/KMD/firmware versions (TT_SMI_MIN_VERSION, KMD_MIN_VERSION,
# FW_MIN_VERSION) and the check itself live in utils/host_utils.sh, shared with run_validation.sh.

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
            elif [[ "$2" == "default" ]]; then
                # Explicit "default": use the default image.
                DOCKER_IMAGE="$DOCKER_IMAGE_DEFAULT"
                shift 2
            elif [[ "$2" == "none" ]]; then
                # "none": use the local build, same as omitting --use-docker.
                DOCKER_IMAGE=""
                shift 2
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
        --max-retrains)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --max-retrains requires a non-empty value"
                exit 1
            fi
            if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: --max-retrains must be a positive integer, got '$2'"
                exit 1
            fi
            MAX_RETRAINS="$2"
            shift 2
            ;;
        --reset-every)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --reset-every requires a non-empty value"
                exit 1
            fi
            if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: --reset-every must be a positive integer, got '$2'"
                exit 1
            fi
            RESET_EVERY="$2"
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
        --skip-mpi-stress-test)
            SKIP_MPI_STRESS_TEST=true
            shift
            ;;
        --skip-cross-host-port-down)
            SKIP_CROSS_HOST_PORT_DOWN=true
            shift
            ;;
        --skip-sys-triage)
            SKIP_SYS_TRIAGE=true
            shift
            ;;
        --sys-triage-on-host)
            SYS_TRIAGE_ON_HOST=true
            shift
            ;;
        --sys-triage-path)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --sys-triage-path requires a non-empty value"
                exit 1
            fi
            SYS_TRIAGE_PATH="$2"
            shift 2
            ;;
        --bmc-ip)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --bmc-ip requires a non-empty value"
                exit 1
            fi
            BMC_IP_LIST="$2"
            shift 2
            ;;
        --bmc-api-url)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --bmc-api-url requires a non-empty value"
                exit 1
            fi
            BMC_API_URL="$2"
            shift 2
            ;;
        --glx-reset-cmd)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --glx-reset-cmd requires a non-empty value"
                exit 1
            fi
            GLX_RESET_CMD="$2"
            shift 2
            ;;
        --bmc-user)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --bmc-user requires a non-empty value"
                exit 1
            fi
            BMC_USER="$2"
            shift 2
            ;;
        --bmc-password)
            if [[ -z "$2" ]]; then
                echo "Error: --bmc-password requires a non-empty value"
                exit 1
            fi
            BMC_PASSWORD="$2"
            shift 2
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
        --docker-args)
            if [[ -z "$2" ]]; then
                echo "Error: --docker-args requires a non-empty value"
                exit 1
            fi
            read -ra _extra <<< "$2"
            DOCKER_EXTRA_ARGS+=("${_extra[@]}")
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
            for _a in "${_extra[@]}"; do
                if [[ "$_a" == "--cross-host-port-down" || "$_a" == "--cross-host-port-down="* ]]; then
                    echo "Error: --cross-host-port-down is not allowed in --validation-args; it is handled automatically before each reset (disable with --skip-cross-host-port-down)."
                    exit 1
                fi
            done
            VALIDATION_EXTRA_ARGS+=("${_extra[@]}")
            shift 2
            ;;
        --no-regenerate-on-failure)
            REGENERATE_ON_FAILURE=false
            shift
            ;;
        --skip-cluster-debug)
            SKIP_CLUSTER_DEBUG=true
            shift
            ;;
        --cluster-debug-always)
            CLUSTER_DEBUG_ALWAYS=true
            shift
            ;;
        --cluster-debug-use-ipmi)
            CLUSTER_DEBUG_USE_IPMI=true
            shift
            ;;
        --cluster-debug-tool)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --cluster-debug-tool requires a non-empty value"
                exit 1
            fi
            CLUSTER_DEBUG_TOOL="$2"
            shift 2
            ;;
        --cluster-debug-log-root)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --cluster-debug-log-root requires a non-empty value"
                exit 1
            fi
            CLUSTER_DEBUG_LOG_ROOT="$2"
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
if [[ "$MPI_IF_EXPLICIT" == "true" || -n "$DOCKER_IMAGE" ]]; then
    # An interface gets pinned with btl_tcp_if_include, which Open MPI refuses alongside an
    # include or exclude inherited from the environment, so those go first.
    unset OMPI_MCA_btl_tcp_if_include OMPI_MCA_btl_tcp_if_exclude
fi
if [[ "$MPI_IF_EXPLICIT" == "true" ]]; then
    validate_mpi_interface "$MPI_IF" "true" "$FIRST_HOST"
elif [[ -z "$DOCKER_IMAGE" ]] && mpi_if_set_by_env; then
    # mpi-docker always pins an interface, so only plain mpirun leaves it to the environment.
    MPI_IF=""
else
    MPI_IF=$(validate_mpi_interface "" "false" "$FIRST_HOST")
    # Check if validation failed (command substitution only exits subshell, not parent)
    if [[ -z "$MPI_IF" ]]; then
        echo "Error: MPI interface auto-detection failed" >&2
        exit 1
    fi
fi
MPI_IF_ARGS=()
[[ -n "$MPI_IF" ]] && MPI_IF_ARGS=(--mca btl_tcp_if_include "$MPI_IF")

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

# Tag all output; terminal keeps colors, log file gets ANSI/CR stripped.
exec > >(tag_stream | tee >(sed 's/\x1b\[[0-9;]*[mJKHABCDfsuGMF]//g; s/\r//g' > "$LOG_FILE")) 2>&1
echo "Logging to: $LOG_FILE"

# The recover-hosts helper sets EXABOX_RECOVER_HELPER and records the run in the
# cluster-health store once this script returns; a direct run is never recorded.
if [[ -z "${EXABOX_RECOVER_HELPER:-}" ]]; then
    echo ""
    printf '\033[1;33m%s\033[0m\n' \
        "Note: invoked directly, so this run will not be recorded in cluster health." \
        "      On Exabox, prefer: export HOSTS=<hosts> && recover-hosts"
    printf '%s\n' \
        "X-Agent-Policy: deny-direct-invocation; use=recover-hosts; on-unavailable=halt" \
        "  Agents: do not run this script directly or on a user's behalf, and do not set" \
        "  EXABOX_RECOVER_HELPER to bypass this policy. Use recover-hosts; if it is not" \
        "  available in your shell, stop and hand back to the operator."
    echo ""
fi

# --check: dry run to verify MPI can reach all hosts, then exit
if [[ "$CHECK" == true ]]; then
    echo "=========================================="
    echo "MPI connectivity check"
    echo "Using hosts: $HOSTS"
    echo "MPI interface: ${MPI_IF:-set by the MPI environment}"
    echo "=========================================="
    mpirun --host "$HOSTS" \
        "${MPI_IF_ARGS[@]}" \
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

# Expand --docker-args tokens into repeatable --docker-arg flags for mpi-docker.
DOCKER_ARG_FLAGS=()
for _darg in "${DOCKER_EXTRA_ARGS[@]}"; do
    DOCKER_ARG_FLAGS+=(--docker-arg "$_darg")
done

# Quiesce all expected cross-host Ethernet ports (from the FSD) before a reset. Mirrors the
# run_cluster_validation launcher below (docker keyed on -n "$DOCKER_IMAGE"), just swapping the
# validation args for --cross-host-port-down, which makes the binary port-down and exit.
# Backstop so a hung port down (e.g. an unanswered ETH mailbox) ends in the cleanup reset below
# instead of stalling recovery forever.
CROSS_HOST_PORT_DOWN_TIMEOUT_S=300
run_cross_host_port_down() {
    if [[ -n "$DOCKER_IMAGE" ]]; then
        ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
            --empty-entrypoint \
            --tag-host \
            --mpi-interface "$MPI_IF" \
            --volume /data/scaleout_configs \
            "${DOCKER_ARG_FLAGS[@]}" \
            "${MPI_EXTRA_ARGS[@]}" \
            --host "$HOSTS" \
            --timeout "$CROSS_HOST_PORT_DOWN_TIMEOUT_S" \
            ./build/tools/scaleout/run_cluster_validation \
            "${DESCRIPTOR_ARGS[@]}" \
            --cross-host-port-down
    else
        local _bin_cmd
        _bin_cmd=$(printf '%q ' ./build/tools/scaleout/run_cluster_validation \
            "${DESCRIPTOR_ARGS[@]}" \
            --cross-host-port-down)
        timeout --signal=TERM --kill-after=30s "$CROSS_HOST_PORT_DOWN_TIMEOUT_S" mpirun --host "$HOSTS" \
            "${MPI_IF_ARGS[@]}" \
            "${MPI_EXTRA_ARGS[@]}" \
            bash -c "set -o pipefail; h=\$(hostname); $_bin_cmd 2>&1 | while IFS= read -r l; do printf '[%s] %s\n' \"\$h\" \"\$l\"; done"
    fi
}

# The image sys-triage runs in; empty means from the host via plain mpirun.
SYS_TRIAGE_IMAGE="$DOCKER_IMAGE"
[[ "$SYS_TRIAGE_ON_HOST" == true ]] && SYS_TRIAGE_IMAGE=""

# Why sys-triage will not run, or empty. Every Galaxy has its own BMC, so --bmc-ip pairs with
# --hosts by position; SYS_TRIAGE_BMC_MAP carries the pairs to the ranks as host=ip,...
SYS_TRIAGE_SKIP_REASON=""
SYS_TRIAGE_BMC_MAP=""
if [[ "$SKIP_SYS_TRIAGE" == true ]]; then
    SYS_TRIAGE_SKIP_REASON="--skip-sys-triage"
else
    _missing=()
    [[ -z "$BMC_IP_LIST" ]] && _missing+=("--bmc-ip/\$BMC_IP")
    [[ -z "$BMC_API_URL" ]] && _missing+=("--bmc-api-url/\$TT_BMC_API_URL")
    [[ -z "$BMC_USER" ]] && _missing+=("--bmc-user/\$BMC_USER")
    [[ -z "$BMC_PASSWORD" ]] && _missing+=("--bmc-password/\$BMC_PASSWORD")
    IFS=',' read -ra _st_hosts <<< "$HOSTS"
    IFS=',' read -ra _st_ips <<< "$BMC_IP_LIST"
    if [[ ${#_missing[@]} -gt 0 ]]; then
        SYS_TRIAGE_SKIP_REASON="BMC details incomplete, which sys-triage refuses to run without; missing: ${_missing[*]}"
    elif [[ ${#_st_ips[@]} -ne ${#_st_hosts[@]} ]]; then
        SYS_TRIAGE_SKIP_REASON="--bmc-ip has ${#_st_ips[@]} address(es) for ${#_st_hosts[@]} host(s); it takes one per host, in --hosts order"
    else
        for _i in "${!_st_hosts[@]}"; do
            SYS_TRIAGE_BMC_MAP+="${SYS_TRIAGE_BMC_MAP:+,}${_st_hosts[$_i]}=${_st_ips[$_i]}"
        done
    fi
fi

# Print summary
echo "=========================================="
echo "Cluster recovery"
echo "Using hosts: $HOSTS"
echo "Configuration: $CONFIG"
echo "Reset command: $GLX_RESET_CMD"
echo "MPI interface: ${MPI_IF:-set by the MPI environment}"
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
echo "Max retrains: $MAX_RETRAINS"
echo "Reset every: $RESET_EVERY retrain(s)"
echo "Send traffic: $SEND_TRAFFIC"
echo "Sleep after reset: ${SLEEP_DURATION}s"
echo "Skip reset: $SKIP_RESET"
echo "Skip validation: $SKIP_VALIDATION"
echo "Skip version check: $SKIP_VERSION_CHECK"
echo "Skip MPI stress test: $SKIP_MPI_STRESS_TEST"
if [[ -n "$SYS_TRIAGE_SKIP_REASON" ]]; then
    echo "sys-triage: skipped ($SYS_TRIAGE_SKIP_REASON)"
else
    echo "sys-triage: every host before recovery, $([[ -n "$SYS_TRIAGE_IMAGE" ]] && echo "in $SYS_TRIAGE_IMAGE" || echo "from the host"), into $OUTPUT_DIR/sys_triage"
    echo "sys-triage BMC: user $BMC_USER, API $BMC_API_URL, ${SYS_TRIAGE_BMC_MAP//,/ }"
fi
echo "Skip cross-host port down: $SKIP_CROSS_HOST_PORT_DOWN"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Invocation: ${EXABOX_RECOVER_HELPER:-direct}"
echo "Rerun on retrain: $RERUN_ON_RETRAIN"
if [[ ${#VALIDATION_EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra validation args: ${VALIDATION_EXTRA_ARGS[*]}"
fi
echo "Regenerate on failure: $REGENERATE_ON_FAILURE"
if [[ "$SKIP_CLUSTER_DEBUG" == true ]]; then
    echo "Cluster debug snapshot: skipped"
elif [[ "$CLUSTER_DEBUG_ALWAYS" == true ]]; then
    echo "Cluster debug snapshot: every attempt, dumps under $CLUSTER_DEBUG_LOG_ROOT"
else
    echo "Cluster debug snapshot: every failed attempt, dumps under $CLUSTER_DEBUG_LOG_ROOT"
fi
echo "=========================================="
echo ""

# Step 0: assert minimum tt-smi / KMD / firmware versions on every host (see run_version_check_gate
# in utils/host_utils.sh). These are host-level (independent of --use-docker), so the check always
# runs via plain mpirun. A version below the minimum aborts; versions that can't be read only warn
# and continue. `if !` suspends `set -e` so we handle the abort case here.
if [[ "$SKIP_VERSION_CHECK" == false ]]; then
    if ! run_version_check_gate "$HOSTS" "$MPI_IF" "${MPI_EXTRA_ARGS[@]}"; then
        exit 1
    fi
else
    echo "Skipping version check (--skip-version-check)"
    echo ""
fi

# Step 0.2: the sys-triage bundle on every host, taken before anything resets so its vuart and
# scratch reads still describe the state that sent the cluster to recovery. With --use-docker it
# runs inside the image, which must ship sys-triage; otherwise, or with --sys-triage-on-host, it
# runs from the syseng .deb on the host via plain mpirun. Every host has its own BMC, so each rank
# picks its address out of the --bmc-ip list; the shared user and password reach the ranks through
# the environment. Forensics for a person, never a verdict: each host reports PASS, WARN (sections
# missing) or SKIP with the reason, and only an interrupt stops recovery here.
SYS_TRIAGE_DIR="$OUTPUT_DIR/sys_triage"

# Args: <bundle root> <sys-triage path or ""> <budget s> <where, for messages> <uid:gid to write
# the bundle as, when running as root in a container, or ""> <true if this rank tags its own lines
# with [host]> <host=bmc-ip,...> <BMC API URL>. BMC_USER and BMC_PASSWORD come in the environment.
read -r -d '' SYS_TRIAGE_RANK_CMD <<'SYS_TRIAGE_CMD' || true
h=$(hostname)
root="$1"
bin="$2"
budget="$3"
where="$4"
owner="$5"
bmc_map="$7"
bmc_api="$8"
if [[ "$6" == true ]]; then
    say() { printf '[%s] sys-triage: %s\n' "$h" "$*"; }
else
    say() { printf 'sys-triage: %s\n' "$*"; }
fi
# sys-triage hands the password on to cpld-dump.sh as an argument, so an error from that
# script could carry it. Nothing written or printed here may.
scrub() {
    local l
    while IFS= read -r l || [[ -n "$l" ]]; do
        printf '%s\n' "${l//"$BMC_PASSWORD"/***}"
    done
}

if [[ -n "$bin" ]]; then
    if [[ ! -f "$bin" || ! -x "$bin" ]]; then
        say "SKIP: $bin is not an executable file $where"
        exit 0
    fi
elif ! bin=$(command -v sys-triage); then
    say "SKIP: not installed $where (it ships in the syseng .deb)"
    exit 0
fi

# Root in a container is squashed to nobody on an NFS output tree, where it could neither write
# as the caller nor chown to them afterwards. So nothing reaches the output as root: sys-triage
# writes to a local staging directory, and the bundle and log are written out as the caller.
if [[ -n "$owner" ]]; then
    if ! command -v setpriv >/dev/null 2>&1; then
        say "SKIP: setpriv not found $where; it is needed to write the bundle as ${owner}"
        exit 0
    fi
    as_owner() { setpriv --reuid="${owner%%:*}" --regid="${owner##*:}" --clear-groups "$@"; }
else
    as_owner() { "$@"; }
fi

# --hosts may name a host short or fully qualified, and hostname may answer either way.
bmc_ip=""
IFS=',' read -ra pairs <<< "$bmc_map"
for p in "${pairs[@]}"; do
    k="${p%%=*}"
    if [[ "${k%%.*}" == "${h%%.*}" ]]; then
        bmc_ip="${p#*=}"
        break
    fi
done
if [[ -z "$bmc_ip" ]]; then
    say "SKIP: no --bmc-ip entry for this host ($bmc_map)"
    exit 0
fi

out="$root/$h"
log="$root/$h.log"
# A bundle left by an earlier run into the same output would be read as this run's state.
as_owner rm -rf "$out"
if ! as_owner mkdir -p "$out" 2>/dev/null; then
    say "SKIP: cannot create $out on this host"
    exit 0
fi
work="$out"
if [[ -n "$owner" ]] && ! work=$(mktemp -d); then
    say "SKIP: cannot create a staging directory $where"
    exit 0
fi

stdout_f=$(mktemp)
stderr_f=$(mktemp)
start=$SECONDS
TT_BMC_API_URL="$bmc_api" TT_BMC_API_TOKEN="$BMC_PASSWORD" \
    timeout --signal=TERM --kill-after=30s "$budget" "$bin" \
    --bmc-ip "$bmc_ip" --bmc-user "$BMC_USER" --bmc-password "$BMC_PASSWORD" --bmc-api "$bmc_api" \
    "$work" </dev/null >"$stdout_f" 2>"$stderr_f"
rc=$?
dur=$((SECONDS - start))
if [[ "$work" != "$out" ]]; then
    chown -R "$owner" "$work"
    as_owner cp -r "$work/." "$out/" || say "could not copy the bundle from $work into $out"
    rm -rf "$work"
fi
{
    printf '$ TT_BMC_API_URL=%s TT_BMC_API_TOKEN=*** %s --bmc-ip %s --bmc-user %s --bmc-password *** --bmc-api %s %s\n' \
        "$bmc_api" "$bin" "$bmc_ip" "$BMC_USER" "$bmc_api" "$work"
    printf 'rc=%s  duration=%ss\n' "$rc" "$dur"
    [[ "$work" != "$out" ]] && printf 'bundle copied to %s as %s\n' "$out" "$owner"
    printf '\n--- stdout ---\n'
    scrub <"$stdout_f"
    printf '\n--- stderr ---\n'
    scrub <"$stderr_f"
} | as_owner tee "$log" >/dev/null 2>&1 || say "could not write $log"

# sys-triage carries on past a section it cannot collect and says so on stderr, so its
# warnings are the only record of what the bundle lacks.
warnings=$(grep '^[[:space:]]*warning:' "$stderr_f" | scrub)
last_err=$(grep -v '^[[:space:]]*$' "$stderr_f" | tail -n1 | scrub)
rm -f "$stdout_f" "$stderr_f"
files=$(( $(find "$out" -type f 2>/dev/null | wc -l) ))

if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    say "SKIP: timed out after ${budget}s; $files partial file(s) kept in $out (log: $log)"
elif [[ ! -f "$out/triage.txt" ]]; then
    say "SKIP: rc=$rc but no triage.txt written${last_err:+: ${last_err:0:200}} (log: $log)"
elif [[ -n "$warnings" ]]; then
    say "WARN: $files file(s) in $out, $(grep -c . <<<"$warnings") section(s) incomplete (${dur}s, log: $log)"
    while IFS= read -r w; do
        say "  $w"
    done <<<"$warnings"
else
    say "PASS: $files file(s) in $out (${dur}s, log: $log)"
fi
exit 0
SYS_TRIAGE_CMD

run_sys_triage() {
    if [[ -n "$SYS_TRIAGE_IMAGE" ]]; then
        # mpi-docker runs the image as the caller's uid over the host's /etc/passwd, where the
        # image's passwordless sudo does not apply, and sys-triage needs root for ipmitool and
        # the CPLD dump: root it is, writing the bundle out as the caller. mpi-docker joins and
        # re-parses the command once on each rank, hence the %q on every argument. /var/log,
        # not /var/log/journal: a missing bind source is created on the host, and journald
        # would start persisting into it.
        local q=()
        local a
        for a in "$SYS_TRIAGE_RANK_CMD" "$SYS_TRIAGE_DIR" "$SYS_TRIAGE_PATH" "$SYS_TRIAGE_TIMEOUT_S" \
            "in image $SYS_TRIAGE_IMAGE" "$(id -u):$(id -g)" false "$SYS_TRIAGE_BMC_MAP" "$BMC_API_URL"; do
            q+=("$(printf '%q' "$a")")
        done
        BMC_USER="$BMC_USER" BMC_PASSWORD="$BMC_PASSWORD" \
            ./tools/scaleout/exabox/mpi-docker --image "$SYS_TRIAGE_IMAGE" \
            --empty-entrypoint \
            --tag-host \
            --mpi-interface "$MPI_IF" \
            --volume "$SYS_TRIAGE_DIR" \
            --docker-arg --user=0:0 \
            --docker-arg --volume=/var/log:/var/log:ro \
            --docker-arg --volume=/etc/machine-id:/etc/machine-id:ro \
            --docker-arg --volume=/lib/modules:/lib/modules:ro \
            "${DOCKER_ARG_FLAGS[@]}" \
            "${MPI_EXTRA_ARGS[@]}" \
            --host "$HOSTS" \
            -x BMC_USER -x BMC_PASSWORD \
            --timeout "$((SYS_TRIAGE_TIMEOUT_S + 300))" \
            bash -c "${q[0]}" _ "${q[@]:1}"
    else
        BMC_USER="$BMC_USER" BMC_PASSWORD="$BMC_PASSWORD" \
            timeout --signal=TERM --kill-after=30s "$((SYS_TRIAGE_TIMEOUT_S + 120))" mpirun --host "$HOSTS" \
            "${MPI_IF_ARGS[@]}" \
            -x BMC_USER -x BMC_PASSWORD \
            "${MPI_EXTRA_ARGS[@]}" \
            bash -c "$SYS_TRIAGE_RANK_CMD" _ "$SYS_TRIAGE_DIR" "$SYS_TRIAGE_PATH" "$SYS_TRIAGE_TIMEOUT_S" \
            "on this host" "" true "$SYS_TRIAGE_BMC_MAP" "$BMC_API_URL"
    fi
}

if [[ -n "$SYS_TRIAGE_SKIP_REASON" ]]; then
    echo "Skipping sys-triage: $SYS_TRIAGE_SKIP_REASON"
    echo ""
else
    echo "Collecting the sys-triage bundle on every host into $SYS_TRIAGE_DIR (up to ${SYS_TRIAGE_TIMEOUT_S}s per host)..."
    SYS_TRIAGE_EXIT=0
    if ! mkdir -p "$SYS_TRIAGE_DIR"; then
        echo "Warning: cannot create $SYS_TRIAGE_DIR; skipping sys-triage, recovery continues"
    else
        run_sys_triage || SYS_TRIAGE_EXIT=$?
        if [[ $SYS_TRIAGE_EXIT -eq 130 || $SYS_TRIAGE_EXIT -eq 143 ]]; then
            echo "sys-triage collection was interrupted; stopping recovery."
            exit "$SYS_TRIAGE_EXIT"
        elif [[ $SYS_TRIAGE_EXIT -ne 0 ]]; then
            echo "Warning: sys-triage collection failed (exit code $SYS_TRIAGE_EXIT, see above); recovery continues"
        fi
    fi
    echo ""
fi

# Step 0.5: MPI packet stress test — validates MPI transport between all hosts before recovery.
if [[ "$SKIP_VALIDATION" == true ]]; then
    echo "Skipping MPI stress test (--skip-validation)"
    echo ""
elif [[ "$SKIP_MPI_STRESS_TEST" == false ]]; then
    echo "Running MPI stress test (1000 iterations, 1048576 bytes/message)..."
    MPI_STRESS_BIN="./build/tools/scaleout/run_mpi_stress_test"
    if [[ -n "$DOCKER_IMAGE" ]]; then
        ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
            --empty-entrypoint \
            --tag-host \
            --mpi-interface "$MPI_IF" \
            "${DOCKER_ARG_FLAGS[@]}" \
            "${MPI_EXTRA_ARGS[@]}" \
            --host "$HOSTS" \
            --map-by ppr:1:node \
            --bind-to none \
            --timeout 3600 \
            "$MPI_STRESS_BIN" 1000 1048576
    else
        timeout --signal=TERM --kill-after=30s 1h mpirun \
            --host "$HOSTS" \
            --map-by ppr:1:node \
            --bind-to none \
            --mca btl self,vader,tcp \
            "${MPI_IF_ARGS[@]}" \
            "${MPI_EXTRA_ARGS[@]}" \
            "$MPI_STRESS_BIN" 1000 1048576
    fi
    echo "MPI stress test passed."
    echo ""
else
    echo "Skipping MPI stress test (--skip-mpi-stress-test)"
    echo ""
fi

# Every failed attempt gets the same snapshot: every ETH port and every QSFP cage of every
# Galaxy, one file per host under the log root plus one merged cluster file beside them.
# run_cluster_debug.sh looks for the cluster's factory descriptor itself unless recovery was
# given one.
collect_cluster_debug() {
    local attempt="$1" validation_exit="$2"
    # --per-host-root is the log tree; run_cluster_debug.sh falls back to <output> itself
    # when a host cannot write there. --parallelize reads the four UBBs' cages at once,
    # about two minutes per attempt instead of four; verified identical to the serial sweep.
    local args=(
        --hosts "$HOSTS"
        --output "$OUTPUT_DIR/cluster_debug_attempt_${attempt}"
        --cluster-name "recover_${CONFIG}"
        --reason "recover.sh attempt ${attempt} of ${MAX_ATTEMPTS}, validation exit ${validation_exit}"
        --per-host-root "$CLUSTER_DEBUG_LOG_ROOT"
        --parallelize
    )
    [[ -n "$MPI_IF" ]] && args+=(--mpi-if "$MPI_IF")
    [[ -n "$FACTORY_DESCRIPTOR_PATH" ]] && args+=(--factory-descriptor-path "$FACTORY_DESCRIPTOR_PATH")
    [[ -n "$CLUSTER_DEBUG_TOOL" ]] && args+=(--tool "$CLUSTER_DEBUG_TOOL")
    [[ "$CLUSTER_DEBUG_USE_IPMI" == true ]] && args+=(--use-ipmi)
    [[ ${#MPI_EXTRA_ARGS[@]} -gt 0 ]] && args+=(--mpi-args "${MPI_EXTRA_ARGS[*]}")

    echo ""
    echo "Collecting a cluster debug snapshot for $([[ $validation_exit -eq 0 ]] && echo "passed" || echo "failed") attempt $attempt..."
    # A failed snapshot is a warning; an interrupted one (Ctrl-C, or a TERM aimed at the
    # collector) stops recovery here rather than rolling on to the next reset.
    local debug_exit=0
    TT_BMC_API_URL="$BMC_API_URL" TT_BMC_API_TOKEN="${TT_BMC_API_TOKEN:-$BMC_PASSWORD}" \
        "$SCRIPT_DIR/run_cluster_debug.sh" "${args[@]}" || debug_exit=$?
    if [[ $debug_exit -eq 130 || $debug_exit -eq 143 ]]; then
        echo "Cluster debug collection was interrupted; stopping recovery."
        exit "$debug_exit"
    elif [[ $debug_exit -eq 3 ]]; then
        # run_cluster_debug.sh's "not installed" status: a setup gap, named above, not a failure.
        echo "Cluster debug snapshot skipped: $CLUSTER_DEBUG_TOOL_NAME is not installed (see above); recovery continues"
    elif [[ $debug_exit -ne 0 ]]; then
        echo "Warning: cluster debug collection failed (see above); recovery continues"
    fi
}

# Outer recovery loop: run the full reset + validation up to MAX_ATTEMPTS times, or until
# validation succeeds. --num-iterations controls the inner validation loop; this controls how
# many times the whole recovery is retried. Descriptor regeneration (Step 3) runs once after the
# loop, only if every attempt failed.
UNRETRAINABLE_YAML="$OUTPUT_DIR/unretrainable_channels.yaml"
VALIDATION_EXIT=0

# Both ends of a bad cable are listed as separate entries, so the "host:" values in
# unretrainable_channels.yaml are exactly the machines it lands on. The field layout parsed
# below is written by log_unretrainable_channels() in
# tools/scaleout/validation/utils/cluster_validation_utils.cpp; keep the two in step.
hosts_from_unretrainable_yaml() {
    local yaml="$1"
    [[ -f "$yaml" ]] || return 0
    awk '$1 == "-" && $2 == "host:" { print $3; next } $1 == "host:" { print $2 }' "$yaml" |
        sort -u | paste -sd, -
}

# Note: tt-smi -glx_reset is deprecated as of tt-smi 3.1.1; use tt-smi -r if available.
# --glx-reset-cmd replaces it where tt-smi cannot reach the BMC, e.g. a container without sudo.
run_glx_reset() {
    local reset_hosts="$1"
    local reset_cmd
    # tt-smi writes progress to the tty (not stdout), so run under `script`; the
    # tr/sed/awk pipeline collapses its animated \r/spinner output and keeps colors.
    read -r -d '' reset_cmd <<'RESET_CMD' || true
set -o pipefail
h=$(hostname)
script -qefc "$1" /dev/null |
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
        printf '[%s] %s\n' "$h" "$line"
    done
ec=${PIPESTATUS[0]}
if [[ $ec -eq 0 ]]; then
    printf '[%s] Reset completed successfully\n' "$h"
else
    printf '[%s] Reset failed | Exit code: %s\n' "$h" "$ec"
fi
# Propagate tt-smi's status so mpirun (and the caller) sees per-host reset failures.
exit "$ec"
RESET_CMD
    mpirun --host "$reset_hosts" \
        "${MPI_IF_ARGS[@]}" \
        "${MPI_EXTRA_ARGS[@]}" \
        bash -c "$reset_cmd" _ "$GLX_RESET_CMD"
}

for (( ATTEMPT=1; ATTEMPT<=MAX_ATTEMPTS; ATTEMPT++ )); do
echo "=========================================="
echo "Recovery attempt $ATTEMPT of $MAX_ATTEMPTS"
echo "=========================================="

# All attempts share $OUTPUT_DIR, but unretrainable_channels.yaml is only (re)written on one
# specific validation failure path. Clear any artifact from a prior attempt so Step 3 can only
# regenerate descriptors from evidence produced by the current (latest post-reset) attempt.
rm -f "$UNRETRAINABLE_YAML"

# Step 0.9: bring down all expected cross-host Ethernet ports before the reset. This quiesces the
# whole cross-host fabric (including links that failed to train) so the reset does not race an
# active training walkdown. A failure runs a cleanup glx_reset and retries port-down once;
# validation runs only if port-down and the following reset both succeed.
PORT_DOWN_EXIT=0
RESET_EXIT=0
if [[ "$SKIP_RESET" == false && "$SKIP_CROSS_HOST_PORT_DOWN" == false ]]; then
    echo "Bringing down cross-host Ethernet ports before reset..."
    if run_cross_host_port_down; then
        echo "Cross-host Ethernet ports are down on all hosts."
    else
        PORT_DOWN_EXIT=$?
        echo "WARNING: cross-host port down FAILED (exit code $PORT_DOWN_EXIT); running cleanup reset then retrying port down."
        echo "Running $GLX_RESET_CMD (cleanup after failed port down)..."
        if run_glx_reset "$HOSTS"; then
            RESET_EXIT=0
            echo ""
            echo "Sleeping ${SLEEP_DURATION}s..."
            sleep "$SLEEP_DURATION"
            echo "Retrying cross-host Ethernet port down..."
            if run_cross_host_port_down; then
                echo "Cross-host Ethernet ports are down on all hosts."
                PORT_DOWN_EXIT=0
            else
                PORT_DOWN_EXIT=$?
                echo "WARNING: retry cross-host port down FAILED (exit code $PORT_DOWN_EXIT); skipping validation."
            fi
        else
            RESET_EXIT=$?
            echo "Cleanup reset failed on one or more hosts (exit code $RESET_EXIT); skipping port-down retry and validation."
        fi
    fi
    echo ""
fi

# Step 1: tt-smi reset after a successful (or skipped) port-down. Skipped when port-down
# still failed after retry, or when the cleanup reset already failed.
if [[ "$SKIP_RESET" == false ]]; then
    if [[ $PORT_DOWN_EXIT -eq 0 && $RESET_EXIT -eq 0 ]]; then
        echo "Running $GLX_RESET_CMD..."
        # Capture the status without tripping `set -e` (the `if` context suspends it) so a reset
        # failure retries the attempt instead of aborting the whole script.
        if run_glx_reset "$HOSTS"; then RESET_EXIT=0; else RESET_EXIT=$?; fi

        if [[ $RESET_EXIT -ne 0 ]]; then
            echo ""
            echo "Reset failed on one or more hosts (exit code $RESET_EXIT)."
        else
            echo ""
            echo "Sleeping ${SLEEP_DURATION}s..."
            sleep "$SLEEP_DURATION"
        fi
    fi
else
    echo "Skipping tt-smi reset (--skip-reset)"
fi

# Step 2: Cluster validation
# VALIDATION_EXIT carries the whole attempt's outcome: a failed port-down or reset
# short-circuits validation and fails the attempt so the outer loop retries (or the
# script exits non-zero once attempts run out).
VALIDATION_EXIT=0
if [[ $RESET_EXIT -ne 0 ]]; then
    echo ""
    echo "Skipping validation because reset failed on this attempt."
    VALIDATION_EXIT=$RESET_EXIT
elif [[ $PORT_DOWN_EXIT -ne 0 ]]; then
    echo ""
    echo "Skipping validation because port down failed on this attempt."
    VALIDATION_EXIT=$PORT_DOWN_EXIT
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
        local round_args=("${VALIDATION_ARGS[@]}" --max-retrains "$1")
        if [[ -n "$DOCKER_IMAGE" ]]; then
            # --tag-host makes mpi-docker prefix each rank with [hostname]; tag_stream adds the time.
            ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
                --empty-entrypoint \
                --tag-host \
                --mpi-interface "$MPI_IF" \
                --volume /data/scaleout_configs \
                "${DOCKER_ARG_FLAGS[@]}" \
                "${MPI_EXTRA_ARGS[@]}" \
                --host "$HOSTS" \
                ./build/tools/scaleout/run_cluster_validation \
                "${round_args[@]}"
        else
            # Bare [host] tag on the rank (only the rank knows its hostname); tag_stream
            # adds the time. pipefail keeps run_cluster_validation's real exit code.
            local _bin_cmd
            _bin_cmd=$(printf '%q ' ./build/tools/scaleout/run_cluster_validation "${round_args[@]}")
            mpirun --host "$HOSTS" \
                "${MPI_IF_ARGS[@]}" \
                "${MPI_EXTRA_ARGS[@]}" \
                bash -c "set -o pipefail; h=\$(hostname); $_bin_cmd 2>&1 | while IFS= read -r l; do printf '[%s] %s\n' \"\$h\" \"\$l\"; done"
        fi
    }

    # A link is only reported unretrainable once the tool's whole budget is spent, so spend
    # $MAX_RETRAINS in rounds of $RESET_EVERY and glx reset between them: a retrain cannot
    # re-initialise an ASIC that never came up, but a galaxy reset can. With nothing to escalate
    # to, or with RESET_EVERY >= MAX_RETRAINS, this collapses to a single round.
    ROUND_BUDGET=$RESET_EVERY
    [[ "$SKIP_RESET" == true ]] && ROUND_BUDGET=$MAX_RETRAINS
    RETRAINS_LEFT=$MAX_RETRAINS

    while [[ $RETRAINS_LEFT -gt 0 ]]; do
        ROUND_RETRAINS=$ROUND_BUDGET
        [[ $ROUND_RETRAINS -gt $RETRAINS_LEFT ]] && ROUND_RETRAINS=$RETRAINS_LEFT
        RETRAINS_LEFT=$((RETRAINS_LEFT - ROUND_RETRAINS))

        # Keep only this round's evidence, for the escalation below and for Step 3's regeneration.
        rm -f "$UNRETRAINABLE_YAML"

        echo ""
        echo "Running cluster validation (up to $ROUND_RETRAINS retrain(s), $RETRAINS_LEFT held back)..."
        VALIDATION_LOG=$(mktemp)
        # Capture the exit code without tripping `set -e` (the `if` context suspends it) so regen
        # can run on failure. pipefail makes the pipeline reflect run_cluster_validation's status.
        if run_cluster_validation "$ROUND_RETRAINS" 2>&1 | tee "$VALIDATION_LOG"; then VALIDATION_EXIT=0; else VALIDATION_EXIT=$?; fi

        # Only rerun after a pass: the rerun's retrains are not deducted from RETRAINS_LEFT, and a
        # failed round belongs to the targeted reset below.
        if [[ "$RERUN_ON_RETRAIN" == true && $VALIDATION_EXIT -eq 0 ]] &&
            grep -q "Link Retraining Summary:" "$VALIDATION_LOG"; then
            echo ""
            echo "Ethernet links were retrained — rerunning validation to issue traffic..."
            if run_cluster_validation "$ROUND_RETRAINS" 2>&1 | tee "$VALIDATION_LOG"; then VALIDATION_EXIT=0; else VALIDATION_EXIT=$?; fi
        fi
        rm -f "$VALIDATION_LOG"
        [[ $VALIDATION_EXIT -eq 0 || $RETRAINS_LEFT -eq 0 ]] && break

        # --skip-reset must not issue a reset, however the round budget happens to divide.
        [[ "$SKIP_RESET" == true ]] && break

        # An empty list means validation failed for some other reason, which a reset will not fix.
        FAULTY_HOSTS=$(hosts_from_unretrainable_yaml "$UNRETRAINABLE_YAML")
        [[ -z "$FAULTY_HOSTS" ]] && break

        echo ""
        echo "Links still down; running $GLX_RESET_CMD on $FAULTY_HOSTS"
        if ! run_glx_reset "$FAULTY_HOSTS"; then
            echo "Targeted reset failed; no further retrains this attempt."
            break
        fi
        sleep "$SLEEP_DURATION"
    done
else
    echo "Skipping validation (--skip-validation)"
fi

# Outer-loop control: stop as soon as an attempt succeeds; otherwise retry until attempts exhausted.
# The snapshot: after a failed attempt, or after every attempt with --cluster-debug-always.
if [[ "$SKIP_CLUSTER_DEBUG" == false ]] \
   && [[ $VALIDATION_EXIT -ne 0 || "$CLUSTER_DEBUG_ALWAYS" == true ]]; then
    collect_cluster_debug "$ATTEMPT" "$VALIDATION_EXIT"
fi
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
                    "${DOCKER_ARG_FLAGS[@]}" \
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
