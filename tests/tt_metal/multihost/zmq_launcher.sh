#!/bin/bash
# Purpose: Standalone launcher for the experimental ZMQ distributed-context
# backend. Forks N local processes of a test binary on localhost, wiring up the
# environment the ZMQ backend expects. No MPI dependency.
#
# Each spawned process gets:
#   TT_DISTRIBUTED_BACKEND=zmq
#   TT_ZMQ_RANK=<rank>
#   TT_ZMQ_ENDPOINTS=tcp://127.0.0.1:<base_port+0>,...,tcp://127.0.0.1:<base_port+N-1>
#
# Usage:
#   zmq_launcher.sh -n <num_ranks> [-p <base_port>] -- <binary> [binary args...]
#
# Examples:
#   zmq_launcher.sh -n 4 -- ./build/test/tt_metal/single_host_mp_tests
#   zmq_launcher.sh -n 4 -p 6000 -- ./bin --gtest_filter='Foo.*'
#
# Exit code: non-zero if any rank exits non-zero (max of all rank exit codes).

set -u

NUM_RANKS=4
BASE_PORT=$(( 20000 + (RANDOM % 20000) ))

usage() {
    echo "Usage: $0 -n <num_ranks> [-p <base_port>] -- <binary> [binary args...]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-ranks)
            NUM_RANKS="$2"; shift 2 ;;
        -p|--base-port)
            BASE_PORT="$2"; shift 2 ;;
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
    TT_DISTRIBUTED_BACKEND=zmq \
    TT_ZMQ_RANK="$r" \
    TT_ZMQ_ENDPOINTS="$ENDPOINTS" \
        "$BINARY" "${BINARY_ARGS[@]}" &
    PIDS[$r]=$!
done

# Wait for all ranks; track the worst exit code.
MAX_RC=0
for (( r=0; r<NUM_RANKS; r++ )); do
    if ! wait "${PIDS[$r]}"; then
        rc=$?
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
