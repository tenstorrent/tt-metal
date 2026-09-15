#!/bin/bash

# Collect a cluster-wide tt-bh-glx-cluster-debug snapshot: run `collect` on every host
# under mpirun, then `merge` the per-Galaxy dumps into one cluster file.
#
# Three stages:
#     1. every rank runs the collector against its own Galaxy
#     2. every rank writes its dump straight into the shared output directory
#     3. the dumps are merged here into one cluster.jsonl
#
# The collector is a single self-contained executable. It is not installed on the hosts:
# one copy on a mount every host shares is run by every rank. Where there is no shared
# mount, --no-shared-mount copies the binary to each host first and the dumps back after,
# over ssh, and everything in between is the same. Loading the merged file into a database
# and browsing it is a separate step, done afterwards on whatever machine you like:
# `tt-bh-glx-cluster-debug db --db cluster.db --dumps cluster.jsonl --serve`.
#
# Built to be called from recover.sh after a failed attempt, so: it never needs root, it
# runs without a descriptor, a host that fails to collect does not fail the run, and the
# collection has a hard time limit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/mpi_if_selection.sh"
source "$SCRIPT_DIR/utils/host_utils.sh"
source "$SCRIPT_DIR/utils/log_utils.sh"

TOOL_NAME="tt-bh-glx-cluster-debug"

# How long the collection stage may take before mpirun is killed. The collector bounds its
# own cage sweep (--qsfp-budget, 10 min by default) and every ipmitool call it makes; this
# is the outer guard, for the case where a host has stopped answering altogether.
COLLECT_TIMEOUT_DEFAULT="20m"

show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> [OPTIONS]

Collect a cluster-wide $TOOL_NAME snapshot and merge it into one cluster.jsonl.

Required:
    --hosts <host-list>                     Comma-separated list of hosts (one Galaxy each)

Tool:
    --tool <path>                           The $TOOL_NAME executable. Must be visible at the
                                            same path on every host, e.g. on the shared /data mount.
                                            (default: \$TT_CLUSTER_DEBUG_TOOL, else $TOOL_NAME on PATH)

Collection:
    --factory-descriptor-path <path>        Factory system descriptor (FSD). Gives cage-attached links
                                            an expected partner. Optional: without it the dump still
                                            has every port, cage and module, but no expected far end
                                            for cable links.
    --skip-qsfp                             Skip the QSFP cage sweep: ~10 s per host instead of ~4 min,
                                            at the cost of all cage and module data.
    --parallelize                           Read the four UBBs' cages at once (about twice as fast).
    --qsfp-budget <seconds>                 Time allowed for the cage sweep per host (collector default 600)
    --collect-timeout <duration>            Kill the collection stage after this long
                                            (default: $COLLECT_TIMEOUT_DEFAULT; any \`timeout\` duration)
    --reason <text>                         Free text recorded in each dump's envelope, e.g. a repro id

Output:
    --output <directory>                    Output directory. Must be on a mount every host shares,
                                            unless --no-shared-mount. (default: cluster_debug_<cluster-name>_<timestamp>)
    --no-shared-mount                       The hosts share no filesystem: copy the tool to each host over
                                            ssh before collecting, and copy each dump back here afterwards.
                                            Needs passwordless ssh to every host, which mpirun needs anyway.
    --cluster-name <name>                   Name for the merged cluster; a cluster has no serial to read.
                                            (default: cluster_debug_dump)

MPI:
    --mpi-if <interface>                    Network interface for MPI TCP transport (auto-detected)
    --mpi-args <args>                       Extra arguments passed directly to mpirun (quoted string)

    --help                                  Display this help message and exit

Example:
    $0 --hosts bh-glx-110-a07u02,bh-glx-110-a07u08 --reason "link flap triage"

Exit status: 0 when a merged cluster.jsonl was written (even if some hosts failed to collect;
those are listed), 1 when nothing could be collected or merged.
EOF
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

HOSTS=""
TOOL="${TT_CLUSTER_DEBUG_TOOL:-}"
FACTORY_DESCRIPTOR_PATH=""
SKIP_QSFP=false
PARALLELIZE=false
QSFP_BUDGET=""
COLLECT_TIMEOUT="$COLLECT_TIMEOUT_DEFAULT"
REASON=""
OUTPUT_DIR=""
CLUSTER_NAME="cluster_debug_dump"
SHARED_MOUNT=true
MPI_IF=""
MPI_IF_EXPLICIT=false
MPI_EXTRA_ARGS=()

require_value() {
    # require_value <flag> <value>
    if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
        echo "Error: $1 requires a non-empty value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --hosts)                    require_value "$1" "$2"; HOSTS="$2"; shift 2 ;;
        --tool)                     require_value "$1" "$2"; TOOL="$2"; shift 2 ;;
        --factory-descriptor-path)  require_value "$1" "$2"; FACTORY_DESCRIPTOR_PATH="$2"; shift 2 ;;
        --skip-qsfp)                SKIP_QSFP=true; shift ;;
        --parallelize)              PARALLELIZE=true; shift ;;
        --qsfp-budget)              require_value "$1" "$2"; QSFP_BUDGET="$2"; shift 2 ;;
        --collect-timeout)          require_value "$1" "$2"; COLLECT_TIMEOUT="$2"; shift 2 ;;
        --reason)                   require_value "$1" "$2"; REASON="$2"; shift 2 ;;
        --output)                   require_value "$1" "$2"; OUTPUT_DIR="$2"; shift 2 ;;
        --cluster-name)             require_value "$1" "$2"; CLUSTER_NAME="$2"; shift 2 ;;
        --no-shared-mount)          SHARED_MOUNT=false; shift ;;
        --mpi-if)                   require_value "$1" "$2"; MPI_IF="$2"; MPI_IF_EXPLICIT=true; shift 2 ;;
        --mpi-args)
            require_value "$1" "$2"
            read -ra _extra <<< "$2"
            MPI_EXTRA_ARGS+=("${_extra[@]}")
            shift 2
            ;;
        --help)                     show_help; exit 0 ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo "" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

if [[ -z "$HOSTS" ]]; then
    echo "Error: --hosts is required" >&2
    echo "" >&2
    show_help >&2
    exit 1
fi
check_duplicate_hosts "$HOSTS" || exit 1

IFS=',' read -ra HOST_ARRAY <<< "$HOSTS"
HOST_LIST=()
for h in "${HOST_ARRAY[@]}"; do
    [[ -n "$h" ]] && HOST_LIST+=("$h")
done
NUM_HOSTS=${#HOST_LIST[@]}
if [[ $NUM_HOSTS -eq 0 ]]; then
    echo "Error: --hosts contained no hosts" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# The collector: one copy, visible at the same absolute path on every host.
# ---------------------------------------------------------------------------

if [[ -z "$TOOL" ]]; then
    TOOL="$(command -v "$TOOL_NAME" || true)"
fi
if [[ -z "$TOOL" ]]; then
    echo "Error: no $TOOL_NAME found. Pass --tool <path>, set TT_CLUSTER_DEBUG_TOOL," >&2
    echo "       or install the tt-syseng-diag package that provides it." >&2
    exit 1
fi
if [[ ! -x "$TOOL" ]]; then
    echo "Error: tool '$TOOL' is not an executable file" >&2
    exit 1
fi
TOOL="$(cd "$(dirname "$TOOL")" && pwd)/$(basename "$TOOL")"
if ! "$TOOL" --help >/dev/null 2>&1; then
    echo "Error: '$TOOL' does not run on this host" >&2
    exit 1
fi

# A descriptor is an improvement, not a requirement: without one the cage-attached links
# have no expected partner and the collector says so in each dump's findings.
DESCRIPTOR_NOTE=""
if [[ -n "$FACTORY_DESCRIPTOR_PATH" && ! -f "$FACTORY_DESCRIPTOR_PATH" ]]; then
    echo "WARNING: factory descriptor not found, collecting without one: $FACTORY_DESCRIPTOR_PATH" >&2
    DESCRIPTOR_NOTE="none ($FACTORY_DESCRIPTOR_PATH was not found)"
    FACTORY_DESCRIPTOR_PATH=""
fi

# ---------------------------------------------------------------------------
# MPI interface and output directory
# ---------------------------------------------------------------------------

FIRST_HOST="${HOST_LIST[0]}"
if [[ "$MPI_IF_EXPLICIT" == "true" ]]; then
    validate_mpi_interface "$MPI_IF" "true" "$FIRST_HOST"
else
    MPI_IF=$(validate_mpi_interface "" "false" "$FIRST_HOST")
    if [[ -z "$MPI_IF" ]]; then
        echo "Error: MPI interface auto-detection failed" >&2
        exit 1
    fi
fi

# The default embeds the cluster name, not the host list as run_validation.sh does: sixteen
# hostnames exceed the 255-character filename limit.
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="cluster_debug_${CLUSTER_NAME}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR" || exit 1
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
DUMP_DIR="$OUTPUT_DIR/dumps"
mkdir -p "$DUMP_DIR" || exit 1

CLUSTER_FILE="$OUTPUT_DIR/cluster.jsonl"
RUN_LOG="$OUTPUT_DIR/run_cluster_debug.log"

# Everything below is tagged [host][time] and teed to the run log. Ranks prepend their own
# bare [host]; tag_stream keeps it and adds the time. The tagging runs as a background
# pipeline, so on exit our end of it is closed and it is waited for: otherwise the shell
# prompt comes back while the last lines are still being written, and they land on top of it.
exec > >(tag_stream | tee "$RUN_LOG") 2>&1
LOG_PID=$!
trap 'exec >&- 2>&-; wait "$LOG_PID" 2>/dev/null' EXIT

echo "=========================================="
echo "Cluster debug collection"
echo "=========================================="
echo "Hosts ($NUM_HOSTS): $HOSTS"
echo "Cluster name: $CLUSTER_NAME"
echo "Tool: $TOOL"
echo "Factory descriptor: ${FACTORY_DESCRIPTOR_PATH:-${DESCRIPTOR_NOTE:-none}}"
if [[ "$SKIP_QSFP" == true ]]; then
    echo "QSFP cage sweep: skipped"
else
    echo "QSFP cage sweep: included$([[ "$PARALLELIZE" == true ]] && echo ", all UBBs at once")${QSFP_BUDGET:+, budget ${QSFP_BUDGET}s}"
fi
echo "Collection timeout: $COLLECT_TIMEOUT"
echo "MPI interface: $MPI_IF"
if [[ ${#MPI_EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "MPI extra args: ${MPI_EXTRA_ARGS[*]}"
fi
echo "Output directory: $OUTPUT_DIR"
[[ "$SHARED_MOUNT" == false ]] && echo "Shared mount: none; the tool and the dumps travel over ssh"
echo ""

# ---------------------------------------------------------------------------
# Where a rank finds the tool and writes its dump. With a shared mount, the paths here;
# without one, a scratch directory on each host, filled before the pre-flight and emptied
# after the dumps are copied back. Everything between reads these two variables only.
# ---------------------------------------------------------------------------

RANK_TOOL="$TOOL"
RANK_DUMP_DIR="$DUMP_DIR"
SELF="$(hostname)"

# ssh and scp never prompt: a host wanting a password fails at once and is named below.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

# Run `cp` when the host is this one, else scp; likewise for a remote command. The launching
# host is usually in the host list, and ssh to itself is not something every box allows.
copy_to_host() { local h="$1" src="$2" dst="$3"; if [[ "$h" == "$SELF" ]]; then cp "$src" "$dst"; else scp -q "${SSH_OPTS[@]}" "$src" "$h:$dst"; fi; }
run_on_host()  { local h="$1"; shift; if [[ "$h" == "$SELF" ]]; then bash -c "$*"; else ssh "${SSH_OPTS[@]}" "$h" "$@"; fi; }
# The rank names its dump after its own hostname, which need not be the name in --hosts
# (short against fully qualified), so the fetch takes whatever cluster_dump_*.jsonl is there.
fetch_dumps_from_host() {
    local h="$1"
    if [[ "$h" == "$SELF" ]]; then cp "$RANK_DUMP_DIR"/cluster_dump_*.jsonl "$DUMP_DIR/"
    else scp -q "${SSH_OPTS[@]}" "$h:$RANK_DUMP_DIR/cluster_dump_*.jsonl" "$DUMP_DIR/"; fi
}

if [[ "$SHARED_MOUNT" == false ]]; then
    REMOTE_DIR="/tmp/tt-cluster-debug-${USER:-$(id -un)}/$(basename "$OUTPUT_DIR")"
    RANK_TOOL="$REMOTE_DIR/$(basename "$TOOL")"
    RANK_DUMP_DIR="$REMOTE_DIR/dumps"

    echo "Copying the tool to $NUM_HOSTS hosts ($REMOTE_DIR)..."
    COPY_FAIL_FILE="$OUTPUT_DIR/.copy_failed_$$"
    : > "$COPY_FAIL_FILE"
    # One copy per host at once, waited for by PID: a bare `wait` would also wait for the
    # log pipeline this script writes through, which cannot end while the script runs.
    copy_pids=()
    for h in "${HOST_LIST[@]}"; do
        (
            if run_on_host "$h" "mkdir -p '$RANK_DUMP_DIR'" && copy_to_host "$h" "$TOOL" "$RANK_TOOL" \
               && run_on_host "$h" "chmod 755 '$RANK_TOOL'"; then
                :
            else
                echo "$h" >> "$COPY_FAIL_FILE"
                echo "  ERROR: could not copy the tool to $h" >&2
            fi
        ) &
        copy_pids+=($!)
    done
    wait "${copy_pids[@]}"
    mapfile -t copy_failed < "$COPY_FAIL_FILE"
    rm -f "$COPY_FAIL_FILE"
    if [[ ${#copy_failed[@]} -gt 0 ]]; then
        echo "Error: the tool could not be copied to: ${copy_failed[*]}" >&2
        echo "       --no-shared-mount needs passwordless ssh and scp to every host." >&2
        exit 1
    fi
    echo "  copied to all $NUM_HOSTS hosts"
    echo ""
fi

# ---------------------------------------------------------------------------
# Pre-flight: the tool and the output directory must be reachable from every rank.
# ---------------------------------------------------------------------------

echo "Checking the tool and output directory are visible on all $NUM_HOSTS hosts..."
PREFLIGHT_FILE="$OUTPUT_DIR/.preflight_$$"

# Each rank reports PREFLIGHT|<host>|<non-empty if it failed>|<ipmi>, so the hosts that
# cannot see the shared mount are named rather than the run just failing. <ipmi> says
# whether the QSFP half of the collection can happen there: the collector reads the cages
# through ipmitool under passwordless sudo, and without that it still writes every ETH
# port but no cages or modules. Known before collecting, so the operator is told which
# hosts will come back ETH-only rather than finding out from the dumps.
tool_q=$(printf '%q' "$RANK_TOOL")
out_q=$(printf '%q' "$RANK_DUMP_DIR")
PREFLIGHT_CMD="h=\$(hostname); bad=\"\"
test -x $tool_q || { echo \"[\$h] ERROR: tool not executable here: $RANK_TOOL\" >&2; bad=1; }
test -w $out_q || { echo \"[\$h] ERROR: dump directory not writable here: $RANK_DUMP_DIR\" >&2; bad=1; }
if ! command -v ipmitool >/dev/null 2>&1; then ipmi=no-ipmitool
elif sudo -n ipmitool help >/dev/null 2>&1; then ipmi=ok
else ipmi=no-sudo; fi
echo \"PREFLIGHT|\$h|\$bad|\$ipmi\""

if ! mpirun --host "$HOSTS" \
        --mca btl_tcp_if_include "$MPI_IF" \
        "${MPI_EXTRA_ARGS[@]}" \
        bash -c "$PREFLIGHT_CMD" > "$PREFLIGHT_FILE"; then
    echo "Error: mpirun failed during pre-flight; cannot reach all hosts" >&2
    rm -f "$PREFLIGHT_FILE"
    exit 1
fi

preflight_failed=()
preflight_seen=0
no_sudo_hosts=()
no_ipmitool_hosts=()
while IFS= read -r line; do
    if [[ $line =~ ^PREFLIGHT\|([^|]+)\|([^|]*)\|(.*)$ ]]; then
        ((preflight_seen++))
        [[ -n "${BASH_REMATCH[2]}" ]] && preflight_failed+=("${BASH_REMATCH[1]}")
        case "${BASH_REMATCH[3]}" in
            no-sudo)     no_sudo_hosts+=("${BASH_REMATCH[1]}") ;;
            no-ipmitool) no_ipmitool_hosts+=("${BASH_REMATCH[1]}") ;;
        esac
    fi
done < "$PREFLIGHT_FILE"
rm -f "$PREFLIGHT_FILE"

if [[ ${#preflight_failed[@]} -gt 0 ]]; then
    echo "Error: pre-flight failed on: ${preflight_failed[*]}" >&2
    if [[ "$SHARED_MOUNT" == true ]]; then
        echo "       The tool and the output directory must both be on a mount every host shares," >&2
        echo "       or pass --no-shared-mount to have them copied over ssh." >&2
    fi
    exit 1
fi
if [[ $preflight_seen -ne $NUM_HOSTS ]]; then
    echo "Error: only $preflight_seen of $NUM_HOSTS hosts reported in; check the host list" >&2
    exit 1
fi
echo "  all $NUM_HOSTS hosts OK"

# The sudo situation, stated before the four minutes are spent. A host without it is not
# an error: its dump is still worth having, and says in its findings what it lacks.
cage_hosts=$((NUM_HOSTS - ${#no_sudo_hosts[@]} - ${#no_ipmitool_hosts[@]}))
if [[ "$SKIP_QSFP" == true ]]; then
    echo "  QSFP cages: not requested (--skip-qsfp)"
elif [[ $cage_hosts -eq $NUM_HOSTS ]]; then
    echo "  QSFP cages: all $NUM_HOSTS hosts can read them (passwordless sudo for ipmitool)"
else
    echo "  QSFP cages: $cage_hosts of $NUM_HOSTS hosts can read them"
    if [[ ${#no_sudo_hosts[@]} -gt 0 ]]; then
        echo "  ETH only, no passwordless sudo for ipmitool: ${no_sudo_hosts[*]}"
    fi
    if [[ ${#no_ipmitool_hosts[@]} -gt 0 ]]; then
        echo "  ETH only, ipmitool not installed: ${no_ipmitool_hosts[*]}"
    fi
    if [[ $cage_hosts -eq 0 ]]; then
        echo "  WARNING: no host can read the cages, so the snapshot will carry no module or cable data."
        echo "           The ETH ports, their training state and partners are still collected."
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Stages 1 and 2: collect on every host, straight into the shared output directory.
# ---------------------------------------------------------------------------

collect_args=(collect)
[[ -n "$FACTORY_DESCRIPTOR_PATH" ]] && collect_args+=(--factory-descriptor-path "$FACTORY_DESCRIPTOR_PATH")
[[ "$SKIP_QSFP" == true ]]          && collect_args+=(--skip-qsfp)
[[ "$PARALLELIZE" == true ]]        && collect_args+=(--parallelize)
[[ -n "$QSFP_BUDGET" ]]             && collect_args+=(--qsfp-budget "$QSFP_BUDGET")
[[ -n "$REASON" ]]                  && collect_args+=(--reason "$REASON")

# %q-quote the command here, then one `bash -c` on the rank. Only --out differs per rank,
# because only the rank knows its own hostname. Each rank tags its log lines with [host]
# on stderr and prints one COLLECT_RESULT line to stdout for the summary below.
collect_bin=$(printf '%q ' "$RANK_TOOL" "${collect_args[@]}")
dump_prefix=$(printf '%q' "$RANK_DUMP_DIR/cluster_dump_")
COLLECT_CMD="set -o pipefail; h=\$(hostname); $collect_bin --out $dump_prefix\$h.jsonl 2>&1 | while IFS= read -r l; do printf '[%s] %s\n' \"\$h\" \"\$l\"; done >&2; echo \"COLLECT_RESULT|\$h|\${PIPESTATUS[0]}\""

# The descriptor may live on a "latest" path that changes under us, so record which bytes
# this collection was resolved against. The dumps carry their own SNAPSHOT_IDs.
{
    echo "cluster_name: $CLUSTER_NAME"
    echo "hosts: $HOSTS"
    echo "collected_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tool: $TOOL"
    echo "skip_qsfp: $SKIP_QSFP"
    echo "parallelize: $PARALLELIZE"
    echo "shared_mount: $SHARED_MOUNT"
    echo "hosts_reading_cages: $cage_hosts of $NUM_HOSTS"
    [[ ${#no_sudo_hosts[@]} -gt 0 ]] && echo "hosts_without_ipmitool_sudo: ${no_sudo_hosts[*]}"
    [[ ${#no_ipmitool_hosts[@]} -gt 0 ]] && echo "hosts_without_ipmitool: ${no_ipmitool_hosts[*]}"
    [[ -n "$QSFP_BUDGET" ]] && echo "qsfp_budget_s: $QSFP_BUDGET"
    [[ -n "$REASON" ]] && echo "reason: $REASON"
    if [[ -n "$FACTORY_DESCRIPTOR_PATH" ]]; then
        echo "factory_descriptor: $FACTORY_DESCRIPTOR_PATH"
        echo "factory_descriptor_sha256: $(sha256sum "$FACTORY_DESCRIPTOR_PATH" | cut -d' ' -f1)"
    else
        echo "factory_descriptor: ${DESCRIPTOR_NOTE:-none}"
    fi
} > "$OUTPUT_DIR/run_metadata.txt"

if [[ "$SKIP_QSFP" == true ]]; then
    echo "Collecting on $NUM_HOSTS hosts (~10 s, cages skipped)..."
else
    echo "Collecting on $NUM_HOSTS hosts (~4 min; the QSFP cage sweep dominates)..."
fi
COLLECT_START=$SECONDS
COLLECT_RESULT_FILE="$OUTPUT_DIR/.collect_results_$$"

timeout --signal=TERM --kill-after=30s "$COLLECT_TIMEOUT" \
    mpirun --host "$HOSTS" \
        --mca btl_tcp_if_include "$MPI_IF" \
        "${MPI_EXTRA_ARGS[@]}" \
        bash -c "$COLLECT_CMD" > "$COLLECT_RESULT_FILE"
COLLECT_MPI_EXIT=$?
# timeout(1) exits 124 when it had to stop the command, 137 when it had to kill it.
COLLECT_TIMED_OUT=false
[[ $COLLECT_MPI_EXIT -eq 124 || $COLLECT_MPI_EXIT -eq 137 ]] && COLLECT_TIMED_OUT=true

collect_ok=()
collect_failed=()
while IFS= read -r line; do
    if [[ $line =~ ^COLLECT_RESULT\|(.+)\|([0-9]+)$ ]]; then
        if [[ "${BASH_REMATCH[2]}" -eq 0 ]]; then
            collect_ok+=("${BASH_REMATCH[1]}")
        else
            collect_failed+=("${BASH_REMATCH[1]} (exit ${BASH_REMATCH[2]})")
        fi
    fi
done < "$COLLECT_RESULT_FILE"
rm -f "$COLLECT_RESULT_FILE"

echo ""
if [[ "$COLLECT_TIMED_OUT" == true ]]; then
    echo "WARNING: collection hit the $COLLECT_TIMEOUT limit and was stopped; the dumps below are what arrived"
fi
echo "Collection finished in $((SECONDS - COLLECT_START))s: ${#collect_ok[@]} of $NUM_HOSTS hosts OK"
if [[ ${#collect_failed[@]} -gt 0 ]]; then
    echo "Hosts that failed to collect:"
    printf '  %s\n' "${collect_failed[@]}"
fi
if [[ $COLLECT_MPI_EXIT -ne 0 && "$COLLECT_TIMED_OUT" == false && ${#collect_failed[@]} -eq 0 ]]; then
    echo "WARNING: mpirun exited $COLLECT_MPI_EXIT but every rank that reported succeeded"
fi

# Without a shared mount the dumps are on the hosts; bring back whatever each one wrote and
# clear its scratch directory. A host with no dump is named; the run goes on with the rest,
# as it does when a rank fails on a shared mount.
if [[ "$SHARED_MOUNT" == false ]]; then
    echo "Copying the dumps back from $NUM_HOSTS hosts..."
    FETCH_FAIL_FILE="$OUTPUT_DIR/.fetch_failed_$$"
    : > "$FETCH_FAIL_FILE"
    fetch_pids=()
    for h in "${HOST_LIST[@]}"; do
        (
            if ! fetch_dumps_from_host "$h" 2>/dev/null; then
                echo "$h" >> "$FETCH_FAIL_FILE"
            fi
            run_on_host "$h" "rm -rf '$REMOTE_DIR'" 2>/dev/null || true
        ) &
        fetch_pids+=($!)
    done
    wait "${fetch_pids[@]}"
    mapfile -t fetch_failed < "$FETCH_FAIL_FILE"
    rm -f "$FETCH_FAIL_FILE"
    if [[ ${#fetch_failed[@]} -gt 0 ]]; then
        echo "Hosts with no dump to copy back: ${fetch_failed[*]}"
    fi
fi

# The ranks wrote straight into the shared directory, so gathering is checking they arrived.
dumps=()
while IFS= read -r -d '' d; do
    dumps+=("$d")
done < <(find "$DUMP_DIR" -maxdepth 1 -name 'cluster_dump_*.jsonl' -size +0 -print0 2>/dev/null | sort -z)

echo "Dumps in $DUMP_DIR: ${#dumps[@]}"

# A dump that exists can still say a lot went wrong: the collector exits 0 whenever it
# collected anything, and puts what it could not do in the envelope's FINDINGS. Show
# those here, so a host with sixty findings is not reported the same as a clean one.
if command -v python3 >/dev/null 2>&1 && [[ ${#dumps[@]} -gt 0 ]]; then
    python3 - "${dumps[@]}" << 'EOF'
import json, os, sys
for path in sys.argv[1:]:
    try:
        with open(path) as handle:
            envelope = json.loads(handle.readline())
        findings = envelope.get("FINDINGS") or []
    except (OSError, ValueError) as err:
        print(f"  {os.path.basename(path)}: unreadable envelope: {err}")
        continue
    if findings:
        host = (envelope.get("HOST") or {}).get("HOSTNAME") or os.path.basename(path)
        print(f"  {host}: {len(findings)} finding(s)")
        for text in findings[:3]:
            print(f"      {text}")
        if len(findings) > 3:
            print(f"      ... and {len(findings) - 3} more; see the dump's envelope")
EOF
fi
echo ""

if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "Error: no dumps were collected; nothing to merge" >&2
    exit 1
fi
if [[ ${#dumps[@]} -ne $NUM_HOSTS ]]; then
    echo "WARNING: merging ${#dumps[@]} of $NUM_HOSTS dumps. Links leaving a missing chassis"
    echo "         will have no far end resolved."
    echo ""
fi

# ---------------------------------------------------------------------------
# Stage 3: merge the dumps here into one cluster file, even for a single Galaxy.
# ---------------------------------------------------------------------------

echo "Merging ${#dumps[@]} dump(s) into $CLUSTER_FILE ..."
if ! "$TOOL" merge --cluster-name "$CLUSTER_NAME" --out "$CLUSTER_FILE" --dumps "${dumps[@]}"; then
    echo "Error: merge failed" >&2
    exit 1
fi
echo ""

echo "=========================================="
echo "Collected cluster: $CLUSTER_NAME"
echo "  dumps:    $DUMP_DIR (${#dumps[@]} file(s))"
echo "  merged:   $CLUSTER_FILE"
echo "  metadata: $OUTPUT_DIR/run_metadata.txt"
echo "  log:      $RUN_LOG"
echo "To browse: $TOOL_NAME db --db cluster.db --dumps $CLUSTER_FILE --serve"
echo "=========================================="
