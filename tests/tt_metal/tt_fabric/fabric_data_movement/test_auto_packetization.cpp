// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "fabric_fixture.hpp"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp"

// Forward-declare runner function from unicast_runner.cpp
namespace tt::tt_fabric::test {
void run_raw_unicast_write_test(
    tt::tt_fabric::fabric_router_tests::BaseFabricFixture* fixture,
    const RawTestParams& p);
}  // namespace tt::tt_fabric::test

namespace tt::tt_fabric::fabric_router_tests {

// Compile-only test for 2D (mesh) API kernels.
// Verifies mesh/api.h and linear/api.h headers compile with the device toolchain
// when FABRIC_2D is defined. Does NOT run the kernels on hardware.
TEST_F(Fabric2DFixture, CompileOnlyAutoPacketization2D) {
    auto device = get_devices()[0]->get_devices()[0];
    tt::tt_metal::Program program;
    auto core = CoreCoord{0, 0};
    std::map<std::string, std::string> defines = {{"FABRIC_2D", "1"}};

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    tt::tt_metal::detail::CompileProgram(device, program);

    // Second program for additional compile probes (unicast + multicast families)
    tt::tt_metal::Program program2;

    tt::tt_metal::CreateKernel(
        program2,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/compile_probe_unicast_families.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt::tt_metal::CreateKernel(
        program2,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/compile_probe_multicast_families.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    tt::tt_metal::detail::CompileProgram(device, program2);
}

// Compile-only test for 1D (linear) API kernels.
// Uses a separate kernel that only includes linear/api.h (no mesh/api.h).
// Verifies linear/api.h headers compile without FABRIC_2D defined.
TEST_F(Fabric1DFixture, CompileOnlyAutoPacketization1D) {
    auto device = get_devices()[0]->get_devices()[0];
    tt::tt_metal::Program program;
    auto core = CoreCoord{0, 0};

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/linear_unicast_tx_writer_raw.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default});

    tt::tt_metal::detail::CompileProgram(device, program);

    // Second program for linear compile probes covering all missing families
    tt::tt_metal::Program program2;

    tt::tt_metal::CreateKernel(
        program2,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/linear_compile_probe_all_families.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default});

    tt::tt_metal::detail::CompileProgram(device, program2);
}

// ============================================================================
// Silicon data-validation tests for unicast auto-packetization families.
// These tests actually run on hardware and verify byte-for-byte correctness.
// ============================================================================

namespace {

// Helper: pick two distinct physical devices from BaseFabricFixture.
// Returns {src_chip, dst_chip} as fabric-logical chip IDs.
struct ChipPair {
    uint32_t mesh_id;
    ChipId src_chip;
    ChipId dst_chip;
};

ChipPair pick_chip_pair(BaseFabricFixture* fixture) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& devices = fixture->get_devices();
    // Use first two available devices
    auto src_phys = devices[0]->get_devices()[0]->id();
    auto dst_phys = devices[1]->get_devices()[0]->id();
    auto src_node = cp.get_fabric_node_id_from_physical_chip_id(src_phys);
    auto dst_node = cp.get_fabric_node_id_from_physical_chip_id(dst_phys);
    return {*src_node.mesh_id, src_node.chip_id, dst_node.chip_id};
}

}  // anonymous namespace

// --- Unicast basic write: 3 payload sizes ---
TEST_F(Fabric2DFixture, AutoPacketizationUnicastWriteSilicon) {
    auto [mesh_id, src_chip, dst_chip] = pick_chip_pair(this);
    const uint32_t max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());

    // 3 payload sizes: sub-MAX, multi-chunk, large
    std::vector<uint32_t> sizes = {
        max_payload / 2,
        2 * max_payload + 512,
        1u << 20,  // 1 MiB
    };

    for (uint32_t sz : sizes) {
        // Round to 4-byte alignment
        sz = (sz + 3u) & ~3u;
        tt::tt_fabric::test::RawTestParams p{
            .mesh_id = mesh_id,
            .src_chip = src_chip,
            .dst_chip = dst_chip,
            .tensor_bytes = sz,
            .sender_core = CoreCoord{0, 0},
            .receiver_core = CoreCoord{1, 0},
            .family = tt::tt_fabric::test::AutoPacketFamily::UnicastWrite,
        };
        tt::tt_fabric::test::run_raw_unicast_write_test(this, p);
    }
}

// --- Unicast scatter write: small payload only ---
TEST_F(Fabric2DFixture, AutoPacketizationUnicastScatterSilicon) {
    auto [mesh_id, src_chip, dst_chip] = pick_chip_pair(this);
    const uint32_t max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());

    // Scatter payloads must fit in a single packet. Use small sizes.
    // The payload is split in half: each half goes to a different destination address.
    // Must be even (split in half) and word-aligned.
    std::vector<uint32_t> sizes = {256u, 1024u, max_payload & ~3u};

    for (uint32_t sz : sizes) {
        // Ensure even split and word-aligned
        sz = (sz / 8) * 8;  // 8-byte aligned for even 4-byte-word halves
        if (sz == 0) sz = 8;
        tt::tt_fabric::test::RawTestParams p{
            .mesh_id = mesh_id,
            .src_chip = src_chip,
            .dst_chip = dst_chip,
            .tensor_bytes = sz,
            .sender_core = CoreCoord{0, 0},
            .receiver_core = CoreCoord{1, 0},
            .family = tt::tt_fabric::test::AutoPacketFamily::UnicastScatter,
        };
        tt::tt_fabric::test::run_raw_unicast_write_test(this, p);
    }
}

// --- Unicast fused atomic inc: 3 payload sizes ---
TEST_F(Fabric2DFixture, AutoPacketizationUnicastFusedAtomicIncSilicon) {
    auto [mesh_id, src_chip, dst_chip] = pick_chip_pair(this);
    const uint32_t max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());

    std::vector<uint32_t> sizes = {
        max_payload / 2,
        2 * max_payload + 512,
        1u << 20,
    };

    for (uint32_t sz : sizes) {
        sz = (sz + 3u) & ~3u;
        tt::tt_fabric::test::RawTestParams p{
            .mesh_id = mesh_id,
            .src_chip = src_chip,
            .dst_chip = dst_chip,
            .tensor_bytes = sz,
            .sender_core = CoreCoord{0, 0},
            .receiver_core = CoreCoord{1, 0},
            .family = tt::tt_fabric::test::AutoPacketFamily::UnicastFusedAtomicInc,
        };
        tt::tt_fabric::test::run_raw_unicast_write_test(this, p);
    }
}

// --- Unicast fused scatter + atomic inc: small payload only ---
TEST_F(Fabric2DFixture, AutoPacketizationUnicastFusedScatterAtomicIncSilicon) {
    auto [mesh_id, src_chip, dst_chip] = pick_chip_pair(this);
    const uint32_t max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());

    std::vector<uint32_t> sizes = {256u, 1024u, max_payload & ~3u};

    for (uint32_t sz : sizes) {
        sz = (sz / 8) * 8;
        if (sz == 0) sz = 8;
        tt::tt_fabric::test::RawTestParams p{
            .mesh_id = mesh_id,
            .src_chip = src_chip,
            .dst_chip = dst_chip,
            .tensor_bytes = sz,
            .sender_core = CoreCoord{0, 0},
            .receiver_core = CoreCoord{1, 0},
            .family = tt::tt_fabric::test::AutoPacketFamily::UnicastFusedScatterAtomicInc,
        };
        tt::tt_fabric::test::run_raw_unicast_write_test(this, p);
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
