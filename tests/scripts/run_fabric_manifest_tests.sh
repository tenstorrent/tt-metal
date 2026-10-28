#!/usr/bin/env bash
# Runs the fabric manifest tests that need no fabric hardware:
#   - the device-free unit tests
#   - the manifest fixtures on mock clusters, one rank or several
#   - the schema check on every manifest the above runs write
# CI runs it on a CPU runner, as fabric-cpu-manifest in tests/pipeline_reorg/fabric_cpu_only_unit_tests.yaml. The 1D
# and 2D manifest fixtures also run on hardware, through run_cpp_fabric_tests.sh's Fabric1D/2D fixture filters; the
# other fixtures are named outside those filters and run only here.
#
# Run from the repository root with a built tree: tests/scripts/run_fabric_manifest_tests.sh
# BUILD_DIR selects the tree (default ./build). Manifests are written under generated/fabric_manifest_tests/<config>.
set -euo pipefail

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
BUILD_DIR=${BUILD_DIR:-./build}
BIN=$BUILD_DIR/test/tt_metal/tt_fabric/fabric_unit_tests
OUT_DIR=$PWD/generated/fabric_manifest_tests
CHECK_SCHEMA=tests/tt_metal/tt_fabric/fabric_manifest/check_manifest_schema.py
# tt-run, without needing it installed.
TT_RUN=ttnn/ttnn/distributed/ttrun.py

# Mock clusters, one rank each.
CLUSTER_DESCS=tt_metal/third_party/tt-cluster-descriptors
T3K=$CLUSTER_DESCS/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml
BH_4XP150=tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_4xP150.yaml
WH_GALAXY_Y_WRAP=$CLUSTER_DESCS/wormhole/wh_galaxy_y_wrap_only_cluster_desc/wh_galaxy_y_wrap_only_cluster_desc.yaml
BH_GALAXY_TORUS_XY=$CLUSTER_DESCS/blackhole/bh_galaxy_torus_xy_cluster_desc/bh_x32_torus_xy_cluster_desc.yaml

# Mock clusters across ranks, with what binds them.
T3K_ON_TWO_RANKS=$CLUSTER_DESCS/wormhole/t3k_cluster_desc/t3k_2x4_big_mesh_cluster_desc_mapping.yaml
SPLIT_T3K_RANK_BINDINGS=tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml
SC4_SUBTORUS=$CLUSTER_DESCS/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC4_32x4_revC_subtorus_aisleC_mapping.yaml
SC4_EXPRESS_MGD=tests/tt_metal/tt_fabric/custom_mesh_descriptors/subtorus/express_links_quad_8x4_mesh_graph_descriptor.textproto

# A channel trimming profile for the T3K mock.
T3K_TRIMMING_PROFILE=$PWD/tests/tt_metal/tt_fabric/fabric_manifest/t3k_trimming_profile.yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

# Runs the manifest fixtures matching `filter` on the mock cluster `desc`, in one rank.
run_mock() {
    local name=$1 desc=$2 filter=$3
    echo "== $name"
    TT_METAL_LOGS_PATH="$OUT_DIR/$name" TT_METAL_MOCK_CLUSTER_DESC_PATH="$desc" "$BIN" --gtest_filter="$filter"
    expect_manifests "$name" 1
}

# Runs the manifest fixtures matching `filter` under tt-run, which takes the rest of the arguments, and expects one
# manifest per rank.
run_ranks() {
    local name=$1 ranks=$2 filter=$3
    shift 3
    echo "== $name"
    TT_METAL_LOGS_PATH="$OUT_DIR/$name" python3 "$TT_RUN" "$@" \
        --mpi-args "--allow-run-as-root --oversubscribe" \
        "$BIN" --gtest_filter="$filter"
    expect_manifests "$name" "$ranks"
}

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Device-free tests
# ---------------------------------------------------------------------------
echo "== Device-free tests"
"$BIN" --gtest_filter='ManifestNames.*:ManifestTypes.*:ManifestArgs.*:ManifestPublishedChips.*:ManifestWriter.*:ManifestDocs.*:StructLayout.*'

# ---------------------------------------------------------------------------
# Single rank
# ---------------------------------------------------------------------------

# 1D and 2D, on Wormhole and Blackhole.
run_mock t3k_1d "$T3K" 'Fabric1DManifestFixture.*'
run_mock t3k_2d "$T3K" 'Fabric2DManifestFixture.*'
run_mock bh_4xp150_1d "$BH_4XP150" 'Fabric1DManifestFixture.*'
run_mock bh_4xp150_2d "$BH_4XP150" 'Fabric2DManifestFixture.*'

# Single-ERISC mode, where a router's one processor runs on physical ERISC1.
TT_METAL_DISABLE_MULTI_AERISC=1 run_mock bh_4xp150_2d_single_erisc "$BH_4XP150" 'Fabric2DManifestFixture.*'

# Torus, on galaxies whose cabling wraps.
run_mock wh_galaxy_torus_y "$WH_GALAXY_Y_WRAP" 'FabricTorusYManifestFixture.*'
run_mock bh_galaxy_torus_xy "$BH_GALAXY_TORUS_XY" 'FabricTorusXYManifestFixture.*'

# The tensix mux in front of the routers' worker channels.
run_mock bh_4xp150_mux "$BH_4XP150" 'FabricMuxManifestFixture.*'

# VC2's channels.
TT_METAL_ENABLE_FABRIC_VC2=1 run_mock bh_4xp150_2d_vc2 "$BH_4XP150" 'Fabric2DManifestFixture.*'

# Channel trimming: the routers' capture, then a profile that trims one chip's channels.
TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE=1 run_mock t3k_2d_trimming_capture "$T3K" 'Fabric2DManifestFixture.*'
TT_METAL_FABRIC_TRIMMING_PROFILE="$T3K_TRIMMING_PROFILE" run_mock t3k_2d_trimmed "$T3K" 'Fabric2DManifestFixture.*'

# ---------------------------------------------------------------------------
# Multi ranks
# ---------------------------------------------------------------------------

# One T3K mesh split across two ranks: remote chips and cross-host links.
run_ranks split_t3k 2 'Fabric2DManifestFixture.*' \
    --mock-cluster-rank-binding "$T3K_ON_TWO_RANKS" \
    --rank-binding "$SPLIT_T3K_RANK_BINDINGS"

# Four Blackhole galaxies on four ranks, as four meshes joined over N/S links, with express chords: express links,
# Blackhole multi-mesh and its idle VC1 senders on inter-mesh routers, and cross-host links.
run_ranks sc4_express 4 'FabricTorusXManifestFixture.*' \
    --mesh-graph-descriptor "$SC4_EXPRESS_MGD" \
    --mock-cluster-rank-binding "$SC4_SUBTORUS"

# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------
echo "== Schema check"
python3 "$CHECK_SCHEMA" "$OUT_DIR"
