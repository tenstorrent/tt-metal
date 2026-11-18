// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The Keraunos line from a manifest, SPEC-0011 § The line test: the topology and the link kinds come from the manifest
// at VCLUSTER_MANIFEST, and the descriptors in the environment must be the ones generated from it (env.emu on the
// emulator, env.mock on Mock). Run with TT_METAL_SLOW_DISPATCH_MODE=1; skipped when VCLUSTER_MANIFEST is unset. The
// traffic, the verdict and the path proof (steps 5 and 6) are run_line_unicast()'s; this test adds steps 1 to 4.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_fabric/keraunos/co_launched_line.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/descriptors.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/manifest.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::MetalContext;
namespace vc = tt::tt_fabric::vcluster;

const char* env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

// The router slot of `device` at Ethernet channel `port`, if any.
std::optional<vc::Slot> router_at(const vc::Device& device, uint32_t port) {
    for (const auto& slot : device.slots) {
        if (slot.role == vc::SlotRole::Router && slot.port == port) {
            return slot;
        }
    }
    return std::nullopt;
}

// The device of `manifest` named `name`; check_manifest() refuses a link end naming no device, so a miss is a bug.
const vc::Device& device_named(const vc::Manifest& manifest, const std::string& name) {
    for (const auto& device : manifest.devices) {
        if (device.name == name) {
            return device;
        }
    }
    TT_THROW("the manifest has no device named {}", name);
}

const vc::Device& device_of_chip(const vc::Manifest& manifest, uint32_t fabric_chip) {
    for (const auto& device : manifest.devices) {
        if (device.fabric_chip == fabric_chip) {
            return device;
        }
    }
    TT_THROW("the manifest has no device of fabric_chip {}", fabric_chip);
}

// Step 3: the opened cluster is the manifest's, and the failure names the first difference.
// The core keraunos.cores of the package SoC descriptor gives CCE `cce` of HSIO tile `tile`, as "x-y"; empty when it
// lists none.
std::string package_core(const YAML::Node& package, uint32_t tile, uint32_t cce) {
    for (const auto& entry : package["keraunos"]["cores"]) {
        if (entry["hsio"].as<uint32_t>() == tile && entry["cce"].as<uint32_t>() == cce) {
            return entry["core"].as<std::string>();
        }
    }
    return "";
}

void expect_cluster_is_manifest(const vc::Manifest& manifest, const YAML::Node& package) {
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto num_chips = static_cast<uint32_t>(tt::tt_metal::GetNumAvailableDevices());
    ASSERT_EQ(num_chips, manifest.devices.size()) << "the cluster's chips are not the manifest's devices";

    // Physical chip c is fabric chip c. A line the topology mapper laid the other way would put a link's kind on the
    // wrong link and still pass where Ethernet is up.
    const MeshId mesh = control_plane.get_fabric_node_id_from_physical_chip_id(0).mesh_id;
    for (uint32_t c = 0; c < num_chips; ++c) {
        const FabricNodeId node = control_plane.get_fabric_node_id_from_physical_chip_id(c);
        ASSERT_EQ(node.chip_id, c) << "physical chip " << c << " is fabric chip " << node.chip_id;
        ASSERT_EQ(node.mesh_id, mesh) << "physical chip " << c << " is in another mesh";
    }

    // Each device's chip has one fabric router per router slot, on the core of that slot's CCE, at channel `port`.
    for (const auto& device : manifest.devices) {
        const ChipId chip = device.fabric_chip;
        std::set<uint32_t> ports;
        for (const auto& slot : device.slots) {
            if (slot.role == vc::SlotRole::Router) {
                ports.insert(*slot.port);
            }
        }
        std::set<uint32_t> chans;
        for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(FabricNodeId(mesh, chip))) {
            chans.insert(chan);
        }
        ASSERT_EQ(chans, ports) << "device " << device.name << ": fabric router channels are not its router ports";
        for (const uint32_t port : ports) {
            const vc::Slot slot = *router_at(device, port);
            const CoreCoord core = cluster.get_soc_desc(chip).get_eth_core_for_channel(port, tt::CoordSystem::NOC0);
            const std::string expected = package_core(package, slot.tile, slot.cce);
            ASSERT_EQ(fmt::format("{}-{}", core.x, core.y), expected)
                << "device " << device.name << " port " << port << ": router core " << core.str()
                << ", the package's core for the slot's tile " << slot.tile << " cce " << slot.cce << " is '"
                << expected << "'";
        }
    }

    // The cluster's connection of each link's two ports joins exactly those two.
    for (size_t i = 0; i < manifest.links.size(); ++i) {
        const auto& link = manifest.links[i];
        const uint32_t a_chip = device_named(manifest, link.a.device).fabric_chip;
        const uint32_t b_chip = device_named(manifest, link.b.device).fabric_chip;
        const auto [peer, peer_chan] =
            control_plane.get_connected_mesh_chip_chan_ids(FabricNodeId(mesh, a_chip), link.a.port);
        ASSERT_EQ(peer, FabricNodeId(mesh, b_chip))
            << "links[" << i << "]: " << link.a.device << " port " << link.a.port << " is connected to chip "
            << peer.chip_id << ", the manifest says " << link.b.device;
        ASSERT_EQ(peer_chan, link.b.port)
            << "links[" << i << "]: " << link.a.device << " port " << link.a.port << " is connected to channel "
            << peer_chan << " of " << link.b.device << ", the manifest says port " << link.b.port;
    }
}

// Step 4: the manifest's worker slots are exactly hart 1 of the CCE of the router that faces the line on the first
// chip and on the last, the two the line's traffic uses.
void expect_workers_are_the_lines(const vc::Manifest& manifest) {
    const Line line = line_ends();
    std::set<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> expected;  // (fabric_chip, (tile, cce))
    for (const Worker& end : {line.sender, line.receiver}) {
        const vc::Device& device = device_of_chip(manifest, end.node.chip_id);
        const auto router = router_at(device, end.router_chan);
        ASSERT_TRUE(router.has_value()) << "device " << device.name << " has no router slot at channel "
                                        << static_cast<uint32_t>(end.router_chan) << ", which faces the line";
        expected.insert({device.fabric_chip, {router->tile, router->cce}});
    }
    std::set<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> workers;
    size_t num_workers = 0;
    for (const auto& device : manifest.devices) {
        for (const auto& slot : device.slots) {
            if (slot.role != vc::SlotRole::Worker) {
                continue;
            }
            ++num_workers;
            EXPECT_EQ(slot.hart, 1u) << "device " << device.name << " has a worker on hart " << slot.hart
                                     << "; the line's workers are on hart 1";
            workers.insert({device.fabric_chip, {slot.tile, slot.cce}});
        }
    }
    EXPECT_EQ(num_workers, 2u) << "the line uses two workers";
    EXPECT_EQ(workers, expected) << "the workers are not beside the routers that face the line on its two ends";
}

// With VCLUSTER_MANIFEST set, every skip but Mock's is a failure, so this fixture does not skip in SetUp when fewer
// than two devices are open (BaseFabricFixture::SetUp); FromManifest checks the device count itself.
class VclusterLine : public KeraunosCoLaunchedWorkers {
protected:
    void SetUp() override {}
};

bool is_mock() { return MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock; }

TEST_F(VclusterLine, FromManifest) {
    const char* manifest_path = env("VCLUSTER_MANIFEST");
    if (manifest_path == nullptr) {
        GTEST_SKIP() << "VCLUSTER_MANIFEST is not set";
    }
    const char* package = env("VCLUSTER_PACKAGE_SOC_DESC");
    ASSERT_NE(package, nullptr) << "VCLUSTER_PACKAGE_SOC_DESC is not set";

    // Step 1, before the devices open: the manifest loads and passes every row, with the package's facts (the
    // generated per-chip descriptors each hold one chip's cores and cannot give them).
    const vc::Manifest manifest = vc::load_manifest(manifest_path);
    const auto findings = vc::check_manifest(manifest, vc::package_facts(package));
    for (const auto& finding : findings) {
        ADD_FAILURE() << "row " << static_cast<uint32_t>(finding.check) << ": " << finding.detail;
    }
    ASSERT_TRUE(findings.empty()) << "the manifest fails the checks";
    const YAML::Node package_desc = YAML::LoadFile(package);

    // run_line_unicast() skips off Quasar and when a NOC link's chip is beyond the open chips; with a manifest set,
    // either means the environment is not the manifest's (env.emu or env.mock not exported), so it fails here instead.
    ASSERT_EQ(MetalContext::instance().get_cluster().arch(), tt::ARCH::QUASAR) << "the open cluster is not Keraunos";
    ASSERT_EQ(tt::tt_metal::GetNumAvailableDevices(), manifest.devices.size())
        << "the open chips are not the manifest's devices; export the generated env.emu or env.mock";

    // Step 2, with steps 3 and 4 after the open and before any traffic; steps 5 and 6 follow in run_line_unicast().
    bool checked_after_open = false;
    run_line_unicast(vc::noc_links(manifest), [&] {
        checked_after_open = true;
        expect_cluster_is_manifest(manifest, package_desc);
        if (HasFatalFailure()) {
            return;
        }
        expect_workers_are_the_lines(manifest);
    });
    if (IsSkipped() && !is_mock()) {
        ADD_FAILURE() << "run_line_unicast() skipped off Mock with a manifest set";
    }
    if (!HasFatalFailure()) {
        EXPECT_TRUE(checked_after_open) << "run_line_unicast() did not run steps 3 and 4";
    }
}

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
