#!/usr/bin/env bash
# Runs the fabric manifest tests that need no fabric hardware, in the order the visualizer's tools use each other's
# files: the manifest, then capture, then decode.
#   Manifest:
#     - the device-free unit tests
#     - the manifest fixtures on mock clusters, which write the manifests the later checks read
#     - the schema check on every manifest they write
#   The visualizer's shared code: its unit tests
#   Capture:
#     - its unit tests
#     - its check, which takes a snapshot of each manifest with a fake device
#   Decode:
#     - its unit tests
#     - the route decoder's check on each manifest, and that it finds Z peers on the express mesh's
#     - its check, which decodes each manifest's types and a snapshot of it with a fake device
# CI runs it on a CPU runner, as fabric-cpu-manifest in tests/pipeline_reorg/fabric_cpu_only_unit_tests.yaml. The 1D
# and 2D manifest fixtures also run on hardware, through run_cpp_fabric_tests.sh's Fabric1D/2D fixture filters; the
# express one runs only here.
#
# Run from the repository root with a built tree: tests/scripts/run_fabric_manifest_tests.sh
# BUILD_DIR selects the tree (default ./build). Manifests are written under generated/fabric_manifest_tests/<config>.
set -euo pipefail

# The build, and where the manifests and logs go.
BUILD_DIR=${BUILD_DIR:-./build}
BIN=$BUILD_DIR/test/tt_metal/tt_fabric/fabric_unit_tests
OUT_DIR=$PWD/generated/fabric_manifest_tests

# The mock clusters, and the rank bindings of the multi-rank configs.
CLUSTER_DESCS=tt_metal/third_party/tt-cluster-descriptors
T3K=$CLUSTER_DESCS/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml
T3K_ON_TWO_RANKS=$CLUSTER_DESCS/wormhole/t3k_cluster_desc/t3k_2x4_big_mesh_cluster_desc_mapping.yaml
BH_4XP150=tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_4xP150.yaml
SPLIT_T3K_RANK_BINDINGS=tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml
BH_32X4_ON_FOUR_RANKS=$CLUSTER_DESCS/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC4_32x4_revC_subtorus_aisleC_mapping.yaml
EXPRESS_32X4_RANK_BINDINGS=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/express_links_32x4_rank_bindings.yaml
# tt-run
TT_RUN=ttnn/ttnn/distributed/ttrun.py

# The tests and checks. The visualizer's checks are modules, run with python3 -m: they import their package by its path.
CHECK_SCHEMA=tests/tt_metal/tt_fabric/fabric_manifest/check_manifest_schema.py
COMMON_TESTS=tt_metal/fabric/debug/visualizer/common/tests
CAPTURE_TESTS=tt_metal/fabric/debug/visualizer/capture/tests
CHECK_CAPTURE=tt_metal.fabric.debug.visualizer.capture.tests.check_capture
DECODE_TESTS=tt_metal/fabric/debug/visualizer/decode/tests
CHECK_ROUTES=tt_metal.fabric.debug.visualizer.decode.tests.check_manifest_routes
CHECK_DECODE=tt_metal.fabric.debug.visualizer.decode.tests.check_decode

#############################################
# HELPERS                                   #
#############################################

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

# Runs the unit tests under `dir`, logging to <name>_unit_tests.log. Fails unless at least one ran: before Python
# 3.12, unittest passes when it finds no tests.
run_unit_tests() {
    local name=$1 dir=$2
    local log="$OUT_DIR/${name}_unit_tests.log"
    python3 -m unittest discover -s "$dir" -t . 2>&1 | tee "$log"
    if ! grep -q '^Ran [1-9]' "$log"; then
        echo "Fabric manifest: found no unit tests under $dir"
        exit 1
    fi
}

#############################################
# MANIFEST                                  #
#############################################

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "== Manifest: device-free tests"
"$BIN" --gtest_filter='ManifestNames.*:ManifestArgs.*:ManifestDocs.*:StructLayout.*:PacketHeaderLayouts.*:RoutingEncodings.*'

# One rank each.
run_mock t3k_1d "$T3K" 'Fabric1DManifestFixture.*'
run_mock t3k_2d "$T3K" 'Fabric2DManifestFixture.*'
run_mock bh_4xp150_1d "$BH_4XP150" 'Fabric1DManifestFixture.*'
run_mock bh_4xp150_2d "$BH_4XP150" 'Fabric2DManifestFixture.*'
# Single-ERISC mode, where a router's one processor runs on physical ERISC1.
TT_METAL_DISABLE_MULTI_AERISC=1 run_mock bh_4xp150_2d_single_erisc "$BH_4XP150" 'Fabric2DManifestFixture.*'

# Several ranks, through tt-run: a manifest per rank.
run_mock_ranks split_t3k "$T3K_ON_TWO_RANKS" "$SPLIT_T3K_RANK_BINDINGS" 'Fabric2DManifestFixture.*' 2
run_mock_ranks express_32x4 "$BH_32X4_ON_FOUR_RANKS" "$EXPRESS_32X4_RANK_BINDINGS" 'FabricExpressManifestFixture.*' 4

echo "== Manifest: schema check"
python3 "$CHECK_SCHEMA" "$OUT_DIR"

#############################################
# VISUALIZER SHARED CODE                    #
#############################################

echo "== Shared code: unit tests"
run_unit_tests common "$COMMON_TESTS"

#############################################
# CAPTURE                                   #
#############################################

echo "== Capture: unit tests"
run_unit_tests capture "$CAPTURE_TESTS"

echo "== Capture: check"
python3 -m "$CHECK_CAPTURE" "$OUT_DIR"

#############################################
# DECODE                                    #
#############################################

echo "== Decode: unit tests"
run_unit_tests decode "$DECODE_TESTS"

echo "== Decode: route decoder check"
python3 -m "$CHECK_ROUTES" "$OUT_DIR"
python3 -m "$CHECK_ROUTES" --expect-z-peers "$OUT_DIR/express_32x4"

echo "== Decode: check"
python3 -m "$CHECK_DECODE" "$OUT_DIR"
