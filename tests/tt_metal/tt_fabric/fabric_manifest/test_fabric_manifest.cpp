// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks the fabric manifest written during fabric init against the live fabric. Each check covers one part of the
// manifest and runs under 1D and 2D.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <set>
#include <string>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using json = nlohmann::json;

// ============ Helpers ============

using manifest::lower_enum_name;

std::set<std::string> keys_of(const json& object) {
    std::set<std::string> keys;
    for (const auto& [key, _] : object.items()) {
        keys.insert(key);
    }
    return keys;
}

// ============ Fixture ============

// Brings up fabric with manifest generation on, then reads this rank's manifest.
template <FabricConfig kFabricConfig>
class FabricManifestFixture : public BaseFabricFixture {
protected:
    static constexpr FabricConfig fabric_config = kFabricConfig;

    static void SetUpTestSuite() {
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        generate_manifest_before_ = rtoptions.get_generate_fabric_manifest();
        rtoptions.set_generate_fabric_manifest(true);
        suite_start_ = std::filesystem::file_time_type::clock::now();
        DoSetUpTestSuite(kFabricConfig);
    }

    static void TearDownTestSuite() {
        DoTearDownTestSuite();
        tt::tt_metal::MetalContext::instance().rtoptions().set_generate_fabric_manifest(generate_manifest_before_);
    }

    void SetUp() override {
        BaseFabricFixture::SetUp();
        if (IsSkipped()) {
            return;
        }

        manifest_path_ = fabric_manifest_path(tt::tt_metal::MetalContext::instance().rtoptions());
        std::ifstream stream(manifest_path_);
        ASSERT_TRUE(stream.is_open()) << manifest_path_;
        manifest_ = json::parse(stream);
    }

    // Used for restoring the generate_manifest runtime arg value to its initial value after the test.
    inline static bool generate_manifest_before_ = false;
    // When this suite started bringing up fabric. A manifest written before it is a stale one.
    inline static std::filesystem::file_time_type suite_start_;
    std::filesystem::path manifest_path_;
    json manifest_;
};

using Fabric1DManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_1D>;
using Fabric2DManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_2D>;

// ============ Checks ============

// The manifest has exactly the top-level blocks the writer emits, was written by this run, and the write left no
// temporary file. Values are checked against the fixture's own settings, not the getters the writer reads.
void check_top_level(
    const json& manifest,
    const std::filesystem::path& manifest_path,
    FabricConfig fabric_config,
    std::filesystem::file_time_type suite_start) {
    EXPECT_EQ(keys_of(manifest), (std::set<std::string>{"manifest_version", "kind", "run", "fabric_context"}));
    EXPECT_EQ(manifest.at("manifest_version"), FABRIC_MANIFEST_VERSION);
    EXPECT_EQ(manifest.at("kind"), "fabric_manifest");
    EXPECT_GE(std::filesystem::last_write_time(manifest_path), suite_start);

    const auto& run = manifest.at("run");
    EXPECT_EQ(
        keys_of(run),
        (std::set<std::string>{
            "arch",
            "fabric_config",
            "reliability_mode",
            "tensix_config",
            "udm_mode",
            "host_rank",
            "mpi_rank",
            "world_size",
            "written_at"}));
    EXPECT_EQ(run.at("fabric_config"), lower_enum_name(fabric_config));
    EXPECT_EQ(run.at("arch"), lower_enum_name(BaseFabricFixture::arch_));
    EXPECT_FALSE(run.at("written_at").get<std::string>().empty());

    const bool is_2d = fabric_config == FabricConfig::FABRIC_2D;
    const auto& block = manifest.at("fabric_context");
    EXPECT_EQ(
        keys_of(block),
        (std::set<std::string>{
            "topology",
            "is_2d_routing",
            "packet_header_size_bytes",
            "max_payload_size_bytes",
            "channel_buffer_size_bytes",
            is_2d ? "routing_2d_route_buffer_size" : "routing_1d_extension_words"}));
    EXPECT_EQ(block.at("is_2d_routing"), is_2d);

    for (const auto& entry : std::filesystem::directory_iterator(manifest_path.parent_path())) {
        EXPECT_EQ(entry.path().string().find(".tmp."), std::string::npos) << entry.path();
    }
}

}  // namespace

// ============ Tests ============

TEST_F(Fabric1DManifestFixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config, suite_start_); }
TEST_F(Fabric2DManifestFixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config, suite_start_); }

}  // namespace tt::tt_fabric::fabric_router_tests
