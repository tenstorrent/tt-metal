#!/bin/bash

# Collect a cluster-wide tt-bh-glx-cluster-debug snapshot: run `collect` on every host
# under mpirun, merge the per-Galaxy dumps into one cluster file, load it into a SQLite
# database and serve the page.
#
# The tool is a single self-contained executable (it carries its own Python), so rather
# than installing it on all N hosts (which needs root on each), one copy lives on the
# shared network mount and every rank runs that same binary over NFS. No sudo, one copy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/mpi_if_selection.sh"
source "$SCRIPT_DIR/utils/host_utils.sh"
source "$SCRIPT_DIR/utils/log_utils.sh"

# Default binary. It sits on the shared mount so every rank can exec the same copy;
# point --tool elsewhere to try a different build.
TOOL_DEFAULT="/data/bingli/public/tt-bh-glx-cluster-debug"

# Descriptors for the Markham exabox clusters. The factory descriptor is what gives
# cage-attached links an expected partner, and it has to name each collecting host or
# that host's cage links come back with no expectation (a finding, not an error).
DESCRIPTOR_DIR_DEFAULT="/data/scaleout_configs/tt-cluster-config-sources/exabox-latest-staging"

# Names the pod set this script is currently pointed at. A cluster has no serial to read,
# so an operator supplies the name; it lands on the merged cluster record and in the
# default output directory. Override with --cluster-name for a different set of hosts.
CLUSTER_NAME_DEFAULT="cluster_debug_dump"

show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> [OPTIONS]

Collect a cluster-wide tt-bh-glx-cluster-debug snapshot and merge it.

Stages, in order:
    1. mpirun 'collect' on every host, one Galaxy each
    2. every rank writes its dump straight into the shared output directory
    3. 'merge' the dumps here into one cluster file

Required Options:
    --hosts <host-list>                     Comma-separated list of hosts (one Galaxy each)

Tool:
    --tool <path>                           tt-bh-glx-cluster-debug executable to run. Must be visible on
                                            every host. (default: $TOOL_DEFAULT)

Collection:
    --factory-descriptor-path <path>        Factory system descriptor (FSD)
                                            (default: $DESCRIPTOR_DIR_DEFAULT/factory_system_descriptor.textproto)
    --skip-qsfp                             Skip the QSFP cage sweep. Collection drops from ~4 min to
                                            ~10 s per host, at the cost of all module/cage data (so no
                                            miscabled or module_moved answers).
    --reason <text>                         Free text recorded in each dump's envelope, e.g. a repro id

Output:
    --output <directory>                    Output directory (default: cluster_debug_<cluster>_<timestamp>)
                                            Must be on a mount all hosts share; ranks write dumps into it.
    --cluster-name <name>                   Name for the merged cluster; a cluster has no serial to read.
                                            (default: $CLUSTER_NAME_DEFAULT)

MPI:
    --mpi-if <interface>                    Network interface for MPI TCP transport (auto-detected)
    --mpi-args <args>                       Extra arguments passed directly to mpirun (quoted string)

    --help                                  Display this help message and exit

Example:
    $0 --hosts bh-glx-110-a07u02,bh-glx-110-a07u08,bh-glx-110-a08u02 \\
       --reason "link flap triage"

Everything above defaults to the $CLUSTER_NAME_DEFAULT pod set, so a
full collection is just --hosts with the 16 hostnames.
EOF
}

HOSTS=""
TOOL="$TOOL_DEFAULT"
FACTORY_DESCRIPTOR_PATH="$DESCRIPTOR_DIR_DEFAULT/factory_system_descriptor.textproto"
SKIP_QSFP=false
REASON=""
OUTPUT_DIR=""
CLUSTER_NAME=""
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
        --hosts)                        require_value "$1" "$2"; HOSTS="$2"; shift 2 ;;
        --tool)                         require_value "$1" "$2"; TOOL="$2"; shift 2 ;;
        --factory-descriptor-path)      require_value "$1" "$2"; FACTORY_DESCRIPTOR_PATH="$2"; shift 2 ;;
        --skip-qsfp)                    SKIP_QSFP=true; shift ;;
        --reason)                       require_value "$1" "$2"; REASON="$2"; shift 2 ;;
        --output)                       require_value "$1" "$2"; OUTPUT_DIR="$2"; shift 2 ;;
        --cluster-name)                 require_value "$1" "$2"; CLUSTER_NAME="$2"; shift 2 ;;
        --mpi-if)                       require_value "$1" "$2"; MPI_IF="$2"; MPI_IF_EXPLICIT=true; shift 2 ;;
        --mpi-args)
            if [[ -z "$2" ]]; then echo "Error: --mpi-args requires a non-empty value" >&2; exit 1; fi
            read -ra _extra <<< "$2"
            MPI_EXTRA_ARGS+=("${_extra[@]}")
            shift 2
            ;;
        --help)                         show_help; exit 0 ;;
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

if [[ -z "$CLUSTER_NAME" ]]; then
    CLUSTER_NAME="$CLUSTER_NAME_DEFAULT"
fi

# Default output dir. Unlike run_validation.sh this does not embed the host list: 16
# hostnames exceed the 255-character filename limit.
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="cluster_debug_${CLUSTER_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

# ---------------------------------------------------------------------------
# The tool: one copy on the shared mount, exec'd by every rank.
# ---------------------------------------------------------------------------

if [[ ! -x "$TOOL" ]]; then
    echo "Error: tool '$TOOL' is not an executable file" >&2
    echo "       Pass --tool <path> to point at a different build." >&2
    exit 1
fi
TOOL="$(cd "$(dirname "$TOOL")" && pwd)/$(basename "$TOOL")"

if ! "$TOOL" --help >/dev/null 2>&1; then
    echo "Error: '$TOOL' does not run on this host" >&2
    exit 1
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

mkdir -p "$OUTPUT_DIR" || exit 1
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
DUMP_DIR="$OUTPUT_DIR/dumps"
mkdir -p "$DUMP_DIR" || exit 1

CLUSTER_FILE="$OUTPUT_DIR/cluster.jsonl"
RUN_LOG="$OUTPUT_DIR/run_cluster_debug.log"

# Everything below is tagged [host][time] and teed to the run log. Ranks prepend their own
# bare [host]; tag_stream keeps it and adds the time.
exec > >(tag_stream | tee "$RUN_LOG") 2>&1

echo "=========================================="
echo "Cluster debug collection"
echo "=========================================="
echo "Hosts ($NUM_HOSTS): $HOSTS"
echo "Cluster name: $CLUSTER_NAME"
echo "Tool: $TOOL"
echo "Factory descriptor: $FACTORY_DESCRIPTOR_PATH"
echo "QSFP cage sweep: $([[ "$SKIP_QSFP" == true ]] && echo "skipped (~10 s/host)" || echo "included (~4 min/host)")"
echo "MPI interface: $MPI_IF"
if [[ ${#MPI_EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "MPI extra args: ${MPI_EXTRA_ARGS[*]}"
fi
echo "Output directory: $OUTPUT_DIR"
echo ""

if [[ ! -f "$FACTORY_DESCRIPTOR_PATH" ]]; then
    echo "Error: factory descriptor not found: $FACTORY_DESCRIPTOR_PATH" >&2
    exit 1
fi

# The descriptor has to name each collecting host or that host's cage-attached links come
# back with no expected partner. Warn up front rather than letting it surface as a finding
# buried in 16 envelopes.
missing_from_descriptor=()
for h in "${HOST_LIST[@]}"; do
    grep -q "\"$h\"" "$FACTORY_DESCRIPTOR_PATH" || missing_from_descriptor+=("$h")
done
if [[ ${#missing_from_descriptor[@]} -gt 0 ]]; then
    echo "WARNING: the factory descriptor names no host ${missing_from_descriptor[*]};"
    echo "         cage-attached links on those hosts will have no expected partner."
    echo ""
fi

# ---------------------------------------------------------------------------
# Pre-flight: the tool and the output directory must be reachable from every rank.
# ---------------------------------------------------------------------------
echo "Checking the tool and output directory are visible on all $NUM_HOSTS hosts..."
PREFLIGHT_FILE="$OUTPUT_DIR/.preflight_$$"

# Each rank reports PREFLIGHT|<host>|<non-empty if it failed>, so the parent can name the
# hosts that could not see the shared mount rather than just failing the run.
tool_q=$(printf '%q' "$TOOL")
out_q=$(printf '%q' "$OUTPUT_DIR")
PREFLIGHT_CMD="h=\$(hostname); bad=\"\"
test -x $tool_q || { echo \"[\$h] ERROR: tool not executable here: $TOOL\" >&2; bad=1; }
test -w $out_q || { echo \"[\$h] ERROR: output directory not writable here: $OUTPUT_DIR\" >&2; bad=1; }
echo \"PREFLIGHT|\$h|\$bad\""

mpirun --host "$HOSTS" \
    --mca btl_tcp_if_include "$MPI_IF" \
    "${MPI_EXTRA_ARGS[@]}" \
    bash -c "$PREFLIGHT_CMD" > "$PREFLIGHT_FILE"
if [[ $? -ne 0 ]]; then
    echo "Error: mpirun failed during pre-flight; cannot reach all hosts" >&2
    rm -f "$PREFLIGHT_FILE"
    exit 1
fi

preflight_failed=()
preflight_seen=0
while IFS= read -r line; do
    if [[ $line =~ ^PREFLIGHT\|([^|]+)\|(.*)$ ]]; then
        ((preflight_seen++))
        [[ -n "${BASH_REMATCH[2]}" ]] && preflight_failed+=("${BASH_REMATCH[1]}")
    fi
done < "$PREFLIGHT_FILE"
rm -f "$PREFLIGHT_FILE"

if [[ ${#preflight_failed[@]} -gt 0 ]]; then
    echo "Error: pre-flight failed on: ${preflight_failed[*]}" >&2
    echo "       The tool and the output directory must both be on a mount every host shares." >&2
    exit 1
fi
if [[ $preflight_seen -ne $NUM_HOSTS ]]; then
    echo "Error: only $preflight_seen of $NUM_HOSTS hosts reported in; check the host list" >&2
    exit 1
fi
echo "  all $NUM_HOSTS hosts OK"
echo ""

# ---------------------------------------------------------------------------
# Stage 1+2: collect on every host, straight into the shared output directory
# ---------------------------------------------------------------------------
collect_args=(collect --factory-descriptor-path "$FACTORY_DESCRIPTOR_PATH")
[[ "$SKIP_QSFP" == true ]] && collect_args+=(--skip-qsfp)
[[ -n "$REASON" ]] && collect_args+=(--reason "$REASON")

# Same shape as run_validation.sh: %q-quote the command here, then one `bash -c` on the
# rank. Only --out differs, because only the rank knows its own hostname. Each rank tags
# its log with [host] on stderr and prints one COLLECT_RESULT line to stdout, which the
# per-host summary below reads.
collect_bin=$(printf '%q ' "$TOOL" "${collect_args[@]}")
dump_prefix=$(printf '%q' "$DUMP_DIR/cluster_dump_")
COLLECT_CMD="set -o pipefail; h=\$(hostname); $collect_bin --out $dump_prefix\$h.jsonl 2>&1 | while IFS= read -r l; do printf '[%s] %s\n' \"\$h\" \"\$l\"; done >&2; echo \"COLLECT_RESULT|\$h|\${PIPESTATUS[0]}\""

# The descriptor lives on a shared "latest staging" path that moves under us, so record
# which bytes this collection was actually resolved against. The dumps keep their own
# SNAPSHOT_IDs; this is the missing half.
{
    echo "cluster_name: $CLUSTER_NAME"
    echo "hosts: $HOSTS"
    echo "collected_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tool: $TOOL"
    echo "skip_qsfp: $SKIP_QSFP"
    [[ -n "$REASON" ]] && echo "reason: $REASON"
    echo "factory_descriptor: $FACTORY_DESCRIPTOR_PATH"
    echo "factory_descriptor_sha256: $(sha256sum "$FACTORY_DESCRIPTOR_PATH" | cut -d' ' -f1)"
} > "$OUTPUT_DIR/run_metadata.txt"

if [[ "$SKIP_QSFP" == true ]]; then
    echo "Collecting on $NUM_HOSTS hosts (~10 s, cages skipped)..."
else
    echo "Collecting on $NUM_HOSTS hosts (~4 min; the QSFP cage sweep dominates)..."
fi
COLLECT_START=$SECONDS
COLLECT_RESULT_FILE="$OUTPUT_DIR/.collect_results_$$"

mpirun --host "$HOSTS" \
    --mca btl_tcp_if_include "$MPI_IF" \
    "${MPI_EXTRA_ARGS[@]}" \
    bash -c "$COLLECT_CMD" > "$COLLECT_RESULT_FILE"
COLLECT_MPI_EXIT=$?

collect_failed=()
collect_ok=()
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
echo "Collection finished in $((SECONDS - COLLECT_START))s: ${#collect_ok[@]} of $NUM_HOSTS hosts OK"
if [[ ${#collect_failed[@]} -gt 0 ]]; then
    echo "Hosts that failed to collect:"
    for f in "${collect_failed[@]}"; do
        echo "  $f"
    done
fi
if [[ $COLLECT_MPI_EXIT -ne 0 && ${#collect_failed[@]} -eq 0 ]]; then
    echo "WARNING: mpirun exited $COLLECT_MPI_EXIT but every rank reported success"
fi

# Dumps land in the shared output directory as the ranks write them, so "funnelling" is
# just checking they all arrived.
dumps=()
while IFS= read -r -d '' d; do
    dumps+=("$d")
done < <(find "$DUMP_DIR" -maxdepth 1 -name 'cluster_dump_*.jsonl' -size +0 -print0 2>/dev/null | sort -z)

echo "Dumps in $DUMP_DIR: ${#dumps[@]}"
echo ""

if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "Error: no dumps were collected; nothing to merge" >&2
    exit 1
fi
if [[ ${#dumps[@]} -ne $NUM_HOSTS ]]; then
    echo "WARNING: merging ${#dumps[@]} of $NUM_HOSTS dumps. Links leaving a missing chassis"
    echo "         will have no far end resolved. Re-run without the failing hosts, or fix them."
    echo ""
fi

# ---------------------------------------------------------------------------
# Stage 3: merge the dumps here into one cluster file, even for a single Galaxy.
# ---------------------------------------------------------------------------
echo "Merging ${#dumps[@]} dump(s) into $CLUSTER_FILE ..."
merge_args=(merge --cluster-name "$CLUSTER_NAME" --out "$CLUSTER_FILE")
merge_args+=(--dumps "${dumps[@]}")
if ! "$TOOL" "${merge_args[@]}"; then
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
echo "=========================================="
