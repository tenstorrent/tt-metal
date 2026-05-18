#!/usr/bin/env bash
# Usage: ./probe.sh <W_tiles_per_shard>
set -u
W="${1:?usage: $0 <W_tiles_per_shard>}"

MGD=/localdev/bklockiewicz/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.textproto
BIN=../build_Release/tt-train/tests/ttml_tests
FILTER='SubtractFp32ColBBcastTest.ColBBroadcast_Fp32Output_ShardedLhs_BigW_Hangs'

echo "== probe W_tiles=$W =="
TT_MESH_GRAPH_DESC_PATH="$MGD" \
REPRO_W_TILES_PER_SHARD="$W" \
timeout 60 "$BIN" --gtest_filter="$FILTER"
rc=$?
echo "== W=$W exit=$rc  (0=PASS, 124=HANG) =="
exit $rc
