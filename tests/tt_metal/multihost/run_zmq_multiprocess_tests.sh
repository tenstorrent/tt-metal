#!/bin/bash
# Purpose: Run the distributed multiprocess tests against the experimental ZMQ
# backend, on a single host, with no MPI. This is the ZMQ analogue of the
# tt-run launched multiprocess suite (run_t3000_tt_metal_multiprocess_tests).
#
# It exercises the metal control-plane bringup (physical discovery, topology
# mapper, subcontext creation) and the global distributed context across two
# ranks entirely over the ZMQ transport -- a real end-to-end trust signal for
# the ZMQ DistributedContext implementation on actual hardware.
#
# Hardware assumptions (override via env if needed):
#   - A single Blackhole Loudbox (8x p150), fully ethernet connected.
#   - Rank 0 owns TT_VISIBLE_DEVICES 0,1,2,3; rank 1 owns 4,5,6,7.
#   - Mesh graph: one 1x8 line across 2 host ranks (see the MGD below).
#
# On a T3K (2x4 Wormhole) you would instead point MGD at the t3k dual-host
# descriptor and use TT_VISIBLE_DEVICES 0,1 / 2,3; the shape-specific test
# bodies (BigMeshDualRankTest2x4 etc.) only pass on that topology, so by default
# we run the arch-agnostic BigMeshDualRankTest.DistributedContext case.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)

BINARY="$TT_METAL_HOME/build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests"
MGD="${TT_MESH_GRAPH_DESC_PATH:-tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_lb_1x8_dual_rank_mesh_graph_descriptor.textproto}"
GTEST_FILTER="${GTEST_FILTER:-BigMeshDualRankTest.DistributedContext}"

cd "$TT_METAL_HOME"

exec "$SCRIPT_DIR/zmq_launcher.sh" -n 2 \
    -g "TT_MESH_GRAPH_DESC_PATH ${MGD}" \
    -g "TT_MESH_ID 0" \
    -e "0 TT_MESH_HOST_RANK 0" \
    -e "1 TT_MESH_HOST_RANK 1" \
    -e "0 TT_VISIBLE_DEVICES 0,1,2,3" \
    -e "1 TT_VISIBLE_DEVICES 4,5,6,7" \
    -- "$BINARY" --gtest_filter="$GTEST_FILTER" "$@"
