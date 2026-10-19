#!/usr/bin/env bash
# Runs the fabric manifest tests that need no fabric hardware:
#   - the device-free unit tests
#   - the manifest fixtures on mock clusters
#   - the schema check on every manifest the above runs write
#   - the route decoder's unit tests
#   - the route decoder's check on the same manifests, and that it finds Z peers on the express mesh's
# CI runs it on a CPU runner, as fabric-cpu-manifest in tests/pipeline_reorg/fabric_cpu_only_unit_tests.yaml. The 1D
# and 2D manifest fixtures also run on hardware, through run_cpp_fabric_tests.sh's Fabric1D/2D fixture filters; the
# express one runs only here.
#
# Run from the repository root with a built tree: tests/scripts/run_fabric_manifest_tests.sh
# BUILD_DIR selects the tree (default ./build). Manifests are written under generated/fabric_manifest_tests/<config>.
set -euo pipefail

BUILD_DIR=${BUILD_DIR:-./build}
BIN=$BUILD_DIR/test/tt_metal/tt_fabric/fabric_unit_tests
OUT_DIR=$PWD/generated/fabric_manifest_tests
CLUSTER_DESCS=tt_metal/third_party/tt-cluster-descriptors
T3K=$CLUSTER_DESCS/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml
T3K_ON_TWO_RANKS=$CLUSTER_DESCS/wormhole/t3k_cluster_desc/t3k_2x4_big_mesh_cluster_desc_mapping.yaml
BH_4XP150=tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_4xP150.yaml
SPLIT_T3K_RANK_BINDINGS=tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml
BH_32X4_ON_FOUR_RANKS=$CLUSTER_DESCS/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC4_32x4_revC_subtorus_aisleC_mapping.yaml
EXPRESS_32X4_RANK_BINDINGS=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/express_links_32x4_rank_bindings.yaml
CHECK_SCHEMA=tests/tt_metal/tt_fabric/fabric_manifest/check_manifest_schema.py
DECODE_TESTS=tt_metal/fabric/debug/visualizer/decode/tests
# A module, run with python3 -m: it imports the route decoder by its package path.
CHECK_ROUTES=tt_metal.fabric.debug.visualizer.decode.tests.check_manifest_routes
# tt-run, without needing it installed.
TT_RUN=ttnn/ttnn/distributed/ttrun.py

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Fails unless the config wrote `count` manifests: a fixture that skips writes none.
expect_manifests() {
    local name=$1 count=$2
    local found
    found=$(find "$OUT_DIR/$name" -name 'fabric_manifest_rank_*.json' 2>/dev/null | wc -l)
    if [ "$found" -ne "$count" ]; then
        echo "Fabric manifest: $name wrote $found manifests, expected $count"
        exit 1
    fi
}

# Runs the manifest fixtures matching `filter` on the mock cluster `desc`.
run_mock() {
    local name=$1 desc=$2 filter=$3
    echo "== $name"
    TT_METAL_LOGS_PATH="$OUT_DIR/$name" TT_METAL_MOCK_CLUSTER_DESC_PATH="$desc" "$BIN" --gtest_filter="$filter"
    expect_manifests "$name" 1
}

# Runs the manifest fixtures matching `filter` on `ranks` ranks, with the mock cluster each rank's binding names.
run_mock_ranks() {
    local name=$1 mock_rank_bindings=$2 rank_bindings=$3 filter=$4 ranks=$5
    echo "== $name"
    TT_METAL_LOGS_PATH="$OUT_DIR/$name" python3 "$TT_RUN" \
        --mock-cluster-rank-binding "$mock_rank_bindings" \
        --rank-binding "$rank_bindings" \
        --mpi-args "--allow-run-as-root --oversubscribe" \
        "$BIN" --gtest_filter="$filter"
    expect_manifests "$name" "$ranks"
}

echo "== Device-free tests"
"$BIN" --gtest_filter='ManifestNames.*:ManifestArgs.*:ManifestDocs.*:StructLayout.*:PacketHeaderLayouts.*:RoutingEncodings.*'

run_mock t3k_1d "$T3K" 'Fabric1DManifestFixture.*'
run_mock t3k_2d "$T3K" 'Fabric2DManifestFixture.*'
run_mock bh_4xp150_1d "$BH_4XP150" 'Fabric1DManifestFixture.*'
run_mock bh_4xp150_2d "$BH_4XP150" 'Fabric2DManifestFixture.*'
# Single-ERISC mode, where a router's one processor runs on physical ERISC1.
TT_METAL_DISABLE_MULTI_AERISC=1 run_mock bh_4xp150_2d_single_erisc "$BH_4XP150" 'Fabric2DManifestFixture.*'

run_mock_ranks split_t3k "$T3K_ON_TWO_RANKS" "$SPLIT_T3K_RANK_BINDINGS" 'Fabric2DManifestFixture.*' 2
run_mock_ranks express_32x4 "$BH_32X4_ON_FOUR_RANKS" "$EXPRESS_32X4_RANK_BINDINGS" 'FabricExpressManifestFixture.*' 4

echo "== Schema check"
python3 "$CHECK_SCHEMA" "$OUT_DIR"

echo "== Route decoder unit tests"
python3 -m unittest discover -s "$DECODE_TESTS" -t . 2>&1 | tee "$OUT_DIR/route_decoder_unit_tests.log"
# Before Python 3.12, unittest passes when it finds no tests.
if ! grep -q '^Ran [1-9]' "$OUT_DIR/route_decoder_unit_tests.log"; then
    echo "Fabric manifest: found no route decoder unit tests under $DECODE_TESTS"
    exit 1
fi

echo "== Route decoder check"
python3 -m "$CHECK_ROUTES" "$OUT_DIR"
python3 -m "$CHECK_ROUTES" --expect-z-peers "$OUT_DIR/express_32x4"
