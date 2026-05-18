#!/usr/bin/env bash
# Usage: ./probe.sh <W_tiles_per_shard>
#
# Runs the full SubtractFp32ColBBcastTest suite in one gtest invocation:
#   1. ColBBroadcast_DefaultDtype_NoHang       (control: BF16 out, replicated)
#   2. NoBroadcast_Fp32Output_NoHang           (control: FP32 out, no bcast)
#   3. ColBBroadcast_Fp32Output_ReplicatedLhs_NoHang  (control: replicated LHS)
#   4. ColBBroadcast_Fp32Output_ShardedLhs_Hangs      (parametric, uses $W)
#
# The controls run first (gtest preserves registration order), so if the
# device is wedged you'll see it on a control test, NOT on the parametric one
# — keeping the parametric outcome trustworthy.
set -u
W="${1:?usage: $0 <W_tiles_per_shard>}"

MGD=/localdev/bklockiewicz/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.textproto
BIN=../build_Release/tt-train/tests/ttml_tests
FILTER='SubtractFp32ColBBcastTest.*'

echo "== probe W_tiles=$W (controls + parametric) =="
TT_MESH_GRAPH_DESC_PATH="$MGD" \
REPRO_W_TILES_PER_SHARD="$W" \
timeout 60 "$BIN" --gtest_filter="$FILTER"
rc=$?
echo "== W=$W exit=$rc  (0=PASS, 124=HANG) =="
exit $rc
