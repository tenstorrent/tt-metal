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
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "fabric_fixture.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp"

// Forward-declare runner functions from unicast_runner.cpp and multicast_runner.cpp
namespace tt::tt_fabric::test {
void run_raw_unicast_write_test(
    tt::tt_fabric::fabric_router_tests::BaseFabricFixture* fixture,
    const RawTestParams& p);
void run_raw_multicast_write_test(
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

// ============================================================================
// Silicon data-validation tests for multicast auto-packetization families.
// These tests run on hardware and verify byte-for-byte correctness
// across ALL receiver chips.
// ============================================================================

// --- Multicast basic write: 3 payload sizes ---
TEST_F(Fabric2DFixture, AutoPacketizationMulticastWriteSilicon) {
    auto [mesh_id, src_chip, dst_chip] = pick_chip_pair(this);
    const uint32_t max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());

    std::vector<uint32_t> sizes = {
        max_payload / 2,
        2 * max_payload + 512,
        1u << 20,  // 1 MiB
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
            .family = tt::tt_fabric::test::AutoPacketFamily::MulticastWrite,
        };
        tt::tt_fabric::test::run_raw_multicast_write_test(this, p);
    }
}

// --- Multicast scatter write: small payload only ---
TEST_F(Fabric2DFixture, AutoPacketizationMulticastScatterSilicon) {
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
            .family = tt::tt_fabric::test::AutoPacketFamily::MulticastScatter,
        };
        tt::tt_fabric::test::run_raw_multicast_write_test(this, p);
    }
}

// --- Multicast fused atomic inc: 3 payload sizes ---
TEST_F(Fabric2DFixture, AutoPacketizationMulticastFusedAtomicIncSilicon) {
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
            .family = tt::tt_fabric::test::AutoPacketFamily::MulticastFusedAtomicInc,
        };
        tt::tt_fabric::test::run_raw_multicast_write_test(this, p);
    }
}

// --- Multicast fused scatter + atomic inc: small payload only ---
TEST_F(Fabric2DFixture, AutoPacketizationMulticastFusedScatterAtomicIncSilicon) {
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
            .family = tt::tt_fabric::test::AutoPacketFamily::MulticastFusedScatterAtomicInc,
        };
        tt::tt_fabric::test::run_raw_multicast_write_test(this, p);
    }
}

// ============================================================================
// Fabric1DFixture (linear) silicon data-validation tests.
// Tests linear API variants. At minimum: LinearUnicastWrite,
// LinearMulticastWrite, and SparseMulticast MUST NOT be skipped.
// ============================================================================

// --- Linear unicast basic write: uses existing unicast_runner with 1D fixture ---
// The unicast_runner detects 1D vs 2D via is_2D_routing_enabled() and selects
// the appropriate kernel. For 1D, uses unicast_tx_writer_raw.cpp with no FABRIC_2D define.
TEST_F(Fabric1DFixture, AutoPacketizationLinearUnicastWriteSilicon) {
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
            .family = tt::tt_fabric::test::AutoPacketFamily::UnicastWrite,
        };
        tt::tt_fabric::test::run_raw_unicast_write_test(this, p);
    }
}

// --- Linear multicast basic write: uses multicast_runner with 1D fixture ---
// The multicast_runner handles the bounding-box computation and per-direction
// fanout which works for both 2D and 1D topologies. In 1D, only one direction
// is active (E or W), so the per-direction pattern degenerates to single-direction.
TEST_F(Fabric1DFixture, AutoPacketizationLinearMulticastWriteSilicon) {
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
            .family = tt::tt_fabric::test::AutoPacketFamily::MulticastWrite,
        };
        tt::tt_fabric::test::run_raw_multicast_write_test(this, p);
    }
}

// --- Sparse multicast (linear-only): targets subset of devices ---
// Sparse multicast uses a bitmask to select which devices in the linear chain
// receive the data. This test verifies data arrives at targeted devices.
TEST_F(Fabric1DFixture, AutoPacketizationSparseMulticastSilicon) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& devices = get_devices();
    if (devices.size() < 3) {
        GTEST_SKIP() << "Sparse multicast needs at least 3 devices";
    }

    // Source = first device. Targets = devices at hop distances encoded in sparse_mask.
    auto src_phys = devices[0]->get_devices()[0]->id();
    auto src_node = cp.get_fabric_node_id_from_physical_chip_id(src_phys);
    uint32_t mesh_id = *src_node.mesh_id;

    // Build sparse_mask targeting all other devices
    // For a 4-device linear chain with source at device 0:
    // Devices 1,2,3 are at hops 1,2,3 respectively.
    // sparse_mask bit N means "deliver to device N hops away".
    uint16_t sparse_mask = 0;
    for (size_t i = 1; i < devices.size(); ++i) {
        sparse_mask |= (1u << i);  // hop distance = i (device index in linear chain)
    }

    // Sparse multicast is passthrough -- must fit in single packet
    std::vector<uint32_t> sizes = {256u, 1024u};

    for (uint32_t sz : sizes) {
        sz = (sz + 3u) & ~3u;
        // Use SparseMulticast family -- the multicast_runner does NOT handle this;
        // sparse multicast needs special handling (single sender, sparse_mask).
        // For this test, we use the unicast_runner pattern adapted for sparse multicast.
        // The sparse kernel (sparse_multicast_tx_writer_raw.cpp) takes sparse_mask as RT arg.
        tt::tt_fabric::test::RawTestParams p{
            .mesh_id = mesh_id,
            .src_chip = src_node.chip_id,
            .dst_chip = src_node.chip_id,  // Not used for sparse; all targets encoded in mask
            .tensor_bytes = sz,
            .sender_core = CoreCoord{0, 0},
            .receiver_core = CoreCoord{1, 0},
            .family = tt::tt_fabric::test::AutoPacketFamily::SparseMulticast,
        };

        // Sparse multicast needs custom dispatch since it uses a different kernel contract.
        // We inline the test logic here for sparse multicast.
        const auto& fabric_context = cp.get_fabric_context();
        const auto topology = fabric_context.get_fabric_topology();

        auto src_mesh = get_device(src_phys);
        auto* src_dev = src_mesh->get_devices()[0];
        auto src_mem_map = BaseFabricFixture::generate_worker_mem_map(src_mesh, topology);
        const uint32_t src_l1_addr = src_mem_map.source_l1_buffer_address;

        // Collect receivers (all non-source devices)
        struct SpRx {
            std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh;
            tt::tt_metal::IDevice* dev;
            tt::tt_fabric::FabricNodeId node;
        };
        std::vector<SpRx> rx_devs;
        for (size_t i = 1; i < devices.size(); ++i) {
            auto* d = devices[i]->get_devices()[0];
            auto node = cp.get_fabric_node_id_from_physical_chip_id(d->id());
            rx_devs.push_back(SpRx{devices[i], d, node});
        }

        auto rx_mem_map = BaseFabricFixture::generate_worker_mem_map(rx_devs[0].mesh, topology);
        const uint32_t dst_l1_addr = rx_mem_map.target_address;

        // Verify NOC XY consistency
        tt::tt_metal::CoreCoord rx_xy = rx_devs[0].dev->worker_core_from_logical_core(p.receiver_core);

        const size_t n_words = sz / 4;
        std::vector<uint32_t> tx(n_words);
        for (size_t i = 0; i < n_words; ++i) tx[i] = 0xA5A50000u + (uint32_t)i;
        std::vector<uint32_t> zeros(n_words, 0u);

        tt::tt_metal::detail::WriteToDeviceL1(src_dev, p.sender_core, src_l1_addr, tx, CoreType::WORKER);
        for (auto& r : rx_devs) {
            tt::tt_metal::detail::WriteToDeviceL1(r.dev, p.receiver_core, dst_l1_addr, zeros, CoreType::WORKER);
        }

        // Receiver programs + semaphores
        std::vector<tt::tt_metal::Program> rx_progs;
        std::vector<tt::tt_metal::GlobalSemaphore> rx_sems;
        const std::string RX_KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";
        for (auto& r : rx_devs) {
            tt::tt_metal::CoreRangeSet cs(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
            rx_sems.push_back(tt::tt_metal::CreateGlobalSemaphore(r.mesh.get(), cs, 0));
            rx_progs.emplace_back(tt::tt_metal::CreateProgram());
            auto k = tt::tt_metal::CreateKernel(
                rx_progs.back(), RX_KDIR + "rx_addrgen.cpp", p.receiver_core,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default});
            tt::tt_metal::SetRuntimeArgs(rx_progs.back(), k, p.receiver_core, {rx_sems.back().address(), 1u});
        }

        // Sender program
        tt::tt_metal::Program tx_prog = tt::tt_metal::CreateProgram();
        auto tx_k = tt::tt_metal::CreateKernel(
            tx_prog,
            tt::tt_fabric::test::family_kernel_path(tt::tt_fabric::test::AutoPacketFamily::SparseMulticast),
            p.sender_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default});

        // RT args: src_l1, size, dst_base, rx_noc_x, rx_noc_y, sem_addr, sparse_mask, ...fabric_conn
        std::vector<uint32_t> tx_rt = {
            src_l1_addr, sz, dst_l1_addr,
            rx_xy.x, rx_xy.y,
            rx_sems[0].address(),
            static_cast<uint32_t>(sparse_mask),
        };

        // For sparse multicast: single sender, use forwarding link to first receiver
        auto fwd_links = tt::tt_fabric::get_forwarding_link_indices(src_node, rx_devs[0].node);
        if (fwd_links.empty()) {
            ADD_FAILURE() << "No forwarding link from src to first receiver";
            continue;
        }
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_node, rx_devs[0].node, fwd_links[0], tx_prog, p.sender_core, tx_rt);
        tt::tt_metal::SetRuntimeArgs(tx_prog, tx_k, p.sender_core, tx_rt);

        // Dispatch
        for (size_t i = 0; i < rx_devs.size(); ++i) {
            RunProgramNonblocking(rx_devs[i].mesh, rx_progs[i]);
        }
        RunProgramNonblocking(src_mesh, tx_prog);
        WaitForSingleProgramDone(src_mesh, tx_prog);
        for (size_t i = 0; i < rx_devs.size(); ++i) {
            WaitForSingleProgramDone(rx_devs[i].mesh, rx_progs[i]);
        }

        // Verify
        for (auto& r : rx_devs) {
            std::vector<uint32_t> rx(n_words, 0u);
            tt::tt_metal::detail::ReadFromDeviceL1(
                r.dev, p.receiver_core, dst_l1_addr, sz, rx, CoreType::WORKER);
            for (size_t w = 0; w < n_words; ++w) {
                if (rx[w] != tx[w]) {
                    ADD_FAILURE() << "Sparse mcast data mismatch at word " << w
                                  << " on device " << r.dev->id()
                                  << ": got 0x" << std::hex << rx[w]
                                  << ", exp 0x" << tx[w] << std::dec;
                    break;
                }
            }
        }
    }
}

// --- Linear scatter write: GTEST_SKIP (no linear-specific scatter kernel) ---
TEST_F(Fabric1DFixture, AutoPacketizationLinearUnicastScatterSilicon) {
    GTEST_SKIP() << "Linear-specific unicast scatter kernel not yet implemented";
}

// --- Linear fused atomic inc: GTEST_SKIP ---
TEST_F(Fabric1DFixture, AutoPacketizationLinearUnicastFusedAtomicIncSilicon) {
    GTEST_SKIP() << "Linear-specific unicast fused atomic inc kernel not yet implemented";
}

// --- Linear fused scatter + atomic inc: GTEST_SKIP ---
TEST_F(Fabric1DFixture, AutoPacketizationLinearUnicastFusedScatterAtomicIncSilicon) {
    GTEST_SKIP() << "Linear-specific unicast fused scatter+atomic inc kernel not yet implemented";
}

// --- Linear multicast scatter: GTEST_SKIP ---
TEST_F(Fabric1DFixture, AutoPacketizationLinearMulticastScatterSilicon) {
    GTEST_SKIP() << "Linear-specific multicast scatter kernel not yet implemented";
}

// --- Linear multicast fused atomic inc: GTEST_SKIP ---
TEST_F(Fabric1DFixture, AutoPacketizationLinearMulticastFusedAtomicIncSilicon) {
    GTEST_SKIP() << "Linear-specific multicast fused atomic inc kernel not yet implemented";
}

// --- Linear multicast fused scatter + atomic inc: GTEST_SKIP ---
TEST_F(Fabric1DFixture, AutoPacketizationLinearMulticastFusedScatterAtomicIncSilicon) {
    GTEST_SKIP() << "Linear-specific multicast fused scatter+atomic inc kernel not yet implemented";
}

}  // namespace tt::tt_fabric::fabric_router_tests
