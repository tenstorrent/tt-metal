#!/bin/bash
# Purpose: Standalone launcher for the experimental ZMQ distributed-context
# backend. Forks N local processes of a test binary on localhost, wiring up the
# environment the ZMQ backend expects. No MPI dependency.
#
# Each spawned process gets:
#   TT_DISTRIBUTED_BACKEND=zmq
#   TT_ZMQ_RANK=<rank>
#   TT_ZMQ_ENDPOINTS=tcp://127.0.0.1:<base_port+0>,...,tcp://127.0.0.1:<base_port+N-1>
# plus any global env (-g) and per-rank env (-e) overrides. This is the ZMQ
# analogue of a tt-run rank binding: use -e to give each rank its own
# TT_VISIBLE_DEVICES, and -g for values shared by all ranks (e.g. TT_MESH_ID,
# TT_MESH_GRAPH_DESC_PATH).
#
# Usage:
#   zmq_launcher.sh -n <num_ranks> [-p <base_port>] \
#       [-g "KEY VALUE"]... [-e "<rank> KEY VALUE"]... -- <binary> [binary args...]
#
# Examples:
#   zmq_launcher.sh -n 4 -- ./build/test/tt_metal/single_host_mp_tests
#   zmq_launcher.sh -n 2 \
#       -g "TT_MESH_ID 0" -g "TT_MESH_GRAPH_DESC_PATH /path/mgd.textproto" \
#       -e "0 TT_VISIBLE_DEVICES 0,1,2,3" -e "1 TT_VISIBLE_DEVICES 4,5,6,7" \
#       -- ./build/test/.../distributed_multiprocess_tests --gtest_filter='Foo.*'
#
# Note: -g / -e values are parsed as whitespace-separated tokens, so KEY and
# VALUE must not contain spaces (comma-separated device lists are fine).
#
# Exit code: non-zero if any rank exits non-zero (max of all rank exit codes).

set -u

NUM_RANKS=4
BASE_PORT=$(( 20000 + (RANDOM % 20000) ))
declare -a GLOBAL_ENV
declare -a RANK_ENV_SPECS

usage() {
    echo "Usage: $0 -n <num_ranks> [-p <base_port>] [-g \"KEY VALUE\"]... [-e \"<rank> KEY VALUE\"]... -- <binary> [binary args...]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-ranks)
            NUM_RANKS="$2"; shift 2 ;;
        -p|--base-port)
            BASE_PORT="$2"; shift 2 ;;
        -g|--global-env)
            GLOBAL_ENV+=("$2"); shift 2 ;;
        -e|--rank-env)
            RANK_ENV_SPECS+=("$2"); shift 2 ;;
        --)
            shift; break ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown option: $1" >&2; usage ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Error: no binary specified after '--'." >&2
    usage
fi

if ! [[ "$NUM_RANKS" =~ ^[0-9]+$ ]] || [[ "$NUM_RANKS" -lt 1 ]]; then
    echo "Error: num_ranks must be a positive integer (got '$NUM_RANKS')." >&2
    exit 2
fi

BINARY="$1"; shift
BINARY_ARGS=("$@")

# Build the comma-separated endpoint list (indexed by rank).
ENDPOINTS=""
for (( r=0; r<NUM_RANKS; r++ )); do
    port=$(( BASE_PORT + r ))
    ep="tcp://127.0.0.1:${port}"
    if [[ -z "$ENDPOINTS" ]]; then
        ENDPOINTS="$ep"
    else
        ENDPOINTS="${ENDPOINTS},${ep}"
    fi
done

echo "ZMQ launcher: spawning ${NUM_RANKS} ranks of '${BINARY}'" >&2
echo "ZMQ launcher: TT_ZMQ_ENDPOINTS=${ENDPOINTS}" >&2

declare -a PIDS
for (( r=0; r<NUM_RANKS; r++ )); do
    # Base ZMQ env for this rank.
    env_pairs=(
        "TT_DISTRIBUTED_BACKEND=zmq"
        "TT_ZMQ_RANK=${r}"
        "TT_ZMQ_ENDPOINTS=${ENDPOINTS}"
    )
    # Global env shared by all ranks.
    for g in ${GLOBAL_ENV[@]+"${GLOBAL_ENV[@]}"}; do
        read -r gk gv <<< "$g"
        env_pairs+=("${gk}=${gv}")
    done
    # Per-rank env overrides ("<rank> KEY VALUE").
    for spec in ${RANK_ENV_SPECS[@]+"${RANK_ENV_SPECS[@]}"}; do
        read -r sr sk sv <<< "$spec"
        if [[ "$sr" == "$r" ]]; then
            env_pairs+=("${sk}=${sv}")
        fi
    done

    env "${env_pairs[@]}" "$BINARY" ${BINARY_ARGS[@]+"${BINARY_ARGS[@]}"} &
    PIDS[$r]=$!
done

# Wait for all ranks; track the worst exit code. Capture the status directly
# from `wait` (do not use `if ! wait`, which would overwrite $? with the
# negation result and mask non-zero rank exits).
MAX_RC=0
for (( r=0; r<NUM_RANKS; r++ )); do
    wait "${PIDS[$r]}"
    rc=$?
    if [[ "$rc" -ne 0 ]]; then
        echo "ZMQ launcher: rank ${r} (pid ${PIDS[$r]}) exited with ${rc}" >&2
        if [[ "$rc" -gt "$MAX_RC" ]]; then
            MAX_RC=$rc
        fi
    fi
done

if [[ "$MAX_RC" -eq 0 ]]; then
    echo "ZMQ launcher: all ${NUM_RANKS} ranks exited 0" >&2
fi
exit "$MAX_RC"
