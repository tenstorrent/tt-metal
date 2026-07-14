#!/bin/bash
# Purpose: Run the distributed-context primitive tests (single_host_mp_tests)
# against the experimental ZMQ backend, on a single host, with no MPI.
#
# This reuses tests/tt_metal/multihost/single_host_mp_tests/test_context.cpp.
# The ZMQ backend does not yet implement the reduction/gather/scatter/all_to_all
# collectives, so those cases are excluded via gtest_filter. The remaining cases
# (send/recv, barrier, broadcast, all_gather, isend/irecv, duplicate, split,
# create_sub_context) exercise the transport end-to-end.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)

NUM_RANKS="${NUM_RANKS:-4}"
BINARY="$TT_METAL_HOME/build/test/tt_metal/single_host_mp_tests"

# Cases relying on collectives the ZMQ backend has not implemented yet
# (all_reduce, gather, scatter, all_to_all, reduce, reduce_scatter, scan).
EXCLUDE_FILTER="-DistributedContextTest.AllReduceInt"
EXCLUDE_FILTER="${EXCLUDE_FILTER}:DistributedContextExtraTest.GatherIntToRoot"
EXCLUDE_FILTER="${EXCLUDE_FILTER}:DistributedContextExtraTest.ScatterIntFromRoot"
EXCLUDE_FILTER="${EXCLUDE_FILTER}:DistributedContextExtraTest.AllToAllInt"
EXCLUDE_FILTER="${EXCLUDE_FILTER}:DistributedContextExtraTest.ReduceSumIntToRoot"
EXCLUDE_FILTER="${EXCLUDE_FILTER}:DistributedContextExtraTest.ReduceScatterSumInt"
EXCLUDE_FILTER="${EXCLUDE_FILTER}:DistributedContextExtraTest.PrefixScanSumInt"

exec "$SCRIPT_DIR/zmq_launcher.sh" -n "$NUM_RANKS" -- \
    "$BINARY" --gtest_filter="$EXCLUDE_FILTER" "$@"
