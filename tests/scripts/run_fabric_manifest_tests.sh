#!/usr/bin/env bash
# Runs the fabric manifest tests that need no fabric hardware:
#   - the device-free unit tests
#   - the manifest fixtures on mock clusters
#   - the schema check on every manifest the above runs write
# CI runs it on a CPU runner, as fabric-cpu-manifest in tests/pipeline_reorg/fabric_cpu_only_unit_tests.yaml. The
# manifest fixtures also run on hardware, through run_cpp_fabric_tests.sh's Fabric1D/2D fixture filters.
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
CHECK_SCHEMA=tests/tt_metal/tt_fabric/fabric_manifest/check_manifest_schema.py
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

echo "== Device-free tests"
"$BIN" --gtest_filter='ManifestNames.*:ManifestArgs.*:ManifestDocs.*:StructLayout.*'

run_mock t3k_1d "$T3K" 'Fabric1DManifestFixture.*'
run_mock t3k_2d "$T3K" 'Fabric2DManifestFixture.*'
run_mock bh_4xp150_1d "$BH_4XP150" 'Fabric1DManifestFixture.*'
run_mock bh_4xp150_2d "$BH_4XP150" 'Fabric2DManifestFixture.*'

echo "== split_t3k"
TT_METAL_LOGS_PATH="$OUT_DIR/split_t3k" python3 "$TT_RUN" \
    --mock-cluster-rank-binding "$T3K_ON_TWO_RANKS" \
    --rank-binding "$SPLIT_T3K_RANK_BINDINGS" \
    --mpi-args "--allow-run-as-root --oversubscribe" \
    "$BIN" --gtest_filter='Fabric2DManifestFixture.*'
expect_manifests split_t3k 2

echo "== Schema check"
python3 "$CHECK_SCHEMA" "$OUT_DIR"
