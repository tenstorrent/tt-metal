// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Does a fabric unicast write sent from an IDLE ethernet core land in HOST
// MEMORY?  (tt-coremon eth-aggregator transport, PLAN_ETH_AGGREGATOR.md 3.3)
//
// Every piece below this is already established by code or by test:
//   - the fabric receiver passes header.noc_address through unvalidated
//     (fabric_edm_packet_transmission.hpp:154)
//   - a kernel can write host memory over PCIe (cq_realtime_profiler_push.cpp)
//   - an eth core can be a fabric client (VC2 runtime-arg path)
//   - TestSetUnicastRouteIdleEth passes on a T3K
// Nobody has joined them, which is what this does.

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

constexpr uint32_t kSentinelBase = 0x77CAFE00u;
constexpr uint32_t kPayloadBytes = 32;
constexpr uint32_t kNumSends = 16;

// ~100 ms at 1 GHz -- the aggregator's real duty cycle: a small journal at a low
// rate. Deliberately not a tight loop: a telemetry stream that keeps the core
// hot raises AICLK, perturbing both the power envelope and the very utilization
// figures the aggregator exists to report. riscv_wait() on ERISC yields via
// risc_context_switch(), so eth routing is not starved either.
constexpr uint32_t kSendIntervalCycles = 100000000u;

TEST_F(Fabric1DFixture, TestFabricWriteReachesHostMemory) {
    if (!slow_dispatch_) {
        GTEST_SKIP() << "IDLE_ETH launch needs TT_METAL_SLOW_DISPATCH_MODE; fast dispatch is "
                        "unsupported on IDLE_ETH (impl/program/dispatch.cpp)";
    }

    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 2u) << "need at least two chips";

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // Sender must be a REMOTE chip; the receiver is the MMIO chip that owns the
    // PCIe tile the write has to land on.
    tt::tt_metal::IDevice* sender = nullptr;
    for (const auto& d : devices) {
        auto* dev = d->get_devices()[0];
        if (cluster.get_associated_mmio_device(dev->id()) != dev->id()) {
            sender = dev;
            break;
        }
    }
    if (sender == nullptr) {
        GTEST_SKIP() << "no remote (non-MMIO) chip in this cluster";
    }
    const auto mmio_id = cluster.get_associated_mmio_device(sender->id());

    tt::tt_metal::IDevice* receiver = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> sender_mesh;
    for (const auto& d : devices) {
        auto* dev = d->get_devices()[0];
        if (dev->id() == mmio_id) {
            receiver = dev;
        }
        if (dev->id() == sender->id()) {
            sender_mesh = d;
        }
    }
    ASSERT_NE(receiver, nullptr) << "sender's MMIO chip is not in this mesh";

    auto idle_eth = sender->get_inactive_ethernet_cores();
    if (idle_eth.empty()) {
        GTEST_SKIP() << "no inactive ethernet cores on the remote chip";
    }
    const tt::tt_metal::CoreCoord eth_core = *idle_eth.begin();
    log_info(tt::LogTest, "sender chip {} (remote), idle eth core {}", sender->id(), eth_core.str());
    log_info(tt::LogTest, "receiver chip {} (mmio)", receiver->id());

    // Host memory the device can reach. Whether get_noc_addr() is encoded for
    // the NOC the EDM actually writes on (edm_to_local_chip_noc = 1) is exactly
    // what this test settles.
    auto sysmem = cluster.allocate_sysmem_buffer(mmio_id, 4096, /*map_to_noc=*/true);
    ASSERT_NE(sysmem, nullptr);
    const auto noc_addr_opt = sysmem->get_noc_addr();
    ASSERT_TRUE(noc_addr_opt.has_value()) << "sysmem buffer was not mapped to NOC";
    const uint64_t dest_noc_addr = noc_addr_opt.value();
    log_info(tt::LogTest, "host buffer device-visible NOC addr = 0x{:x} (NOC0-encoded)", dest_noc_addr);

    // NOTE: get_noc_addr() is a COMPLETE address produced by the kernel driver
    // (TENSTORRENT_IOCTL_PIN_PAGES with TENSTORRENT_PIN_PAGES_NOC_DMA). It must
    // NOT be decomposed and rebuilt -- an earlier attempt masked its low 36 bits
    // and OR'd in a locally-computed PCIe XY, producing a meaningless address
    // that hung the fabric router. Diagnose the encoding, do not reconstruct it.
    // The driver's noc_addr carries the PCIe window marker and an offset, but NO
    // XY (measured: 0x0000000880000000 -- high bits zero). The consumer ORs in
    // the tile XY, as cq_prefetch.cpp does: `pcie_noc_xy | pcie_read_ptr`.
    //
    // Crucially the XY must be encoded for the NOC the EDM writes on, which is
    // NOC1 (edm_to_local_chip_noc = 1). An earlier attempt let the kernel compute
    // it via NOC_X_PHYS_COORD(), which resolves against the KERNEL's noc_index
    // (0) -- yielding NOC0 coordinates issued on NOC1, i.e. a different and
    // probably nonexistent tile. That hung the fabric router. So compute the
    // whole destination here and hand the kernel a finished address.
    constexpr uint32_t kNocAddrLocalBits = 36;
    constexpr uint32_t kNocAddrNodeIdBits = 6;
    constexpr uint32_t kNocCoordRegOffset = 4;
    constexpr uint64_t kPcieMarker = 0x800000000ull;
    auto pcie_encoding = [&](uint32_t x, uint32_t y) -> uint64_t {
        const uint32_t xy = (y << ((kNocAddrLocalBits % 32) + kNocAddrNodeIdBits)) | (x << (kNocAddrLocalBits % 32));
        return ((uint64_t)xy << (kNocAddrLocalBits - kNocCoordRegOffset)) | kPcieMarker;
    };

    const auto& soc = cluster.get_soc_desc(mmio_id);
    const auto& pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    ASSERT_FALSE(pcie_cores.empty()) << "no PCIE core in the SOC descriptor";
    const auto grid = soc.grid_size;
    // NOC1 mirrors coordinates: phys = (size - 1) - logical.
    const uint32_t px1 = (grid.x - 1) - pcie_cores.front().x;
    const uint32_t py1 = (grid.y - 1) - pcie_cores.front().y;
    const uint64_t dest_full = pcie_encoding(px1, py1) | dest_noc_addr;
    log_info(
        tt::LogTest,
        "PCIe NOC0 ({},{}) -> NOC1 ({},{});  destination = 0x{:016x}",
        pcie_cores.front().x, pcie_cores.front().y, px1, py1, dest_full);

    auto* host_va = static_cast<volatile uint32_t*>(sysmem->get_buffer_va());
    ASSERT_NE(host_va, nullptr);
    for (uint32_t i = 0; i < kPayloadBytes / 4; i++) {
        host_va[i] = 0xDEADBEEFu;  // clear, so any arrival is unambiguous
    }

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t payload_l1 = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);

    auto program = tt::tt_metal::CreateProgram();
    auto kernel = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_fabric_pcie_target.cpp",
        eth_core,
        tt::tt_metal::EthernetConfig{
            .eth_mode = tt::tt_metal::Eth::IDLE, .processor = tt::tt_metal::DataMovementProcessor::RISCV_0});

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto src_node = control_plane.get_fabric_node_id_from_physical_chip_id(sender->id());
    const auto dst_node = control_plane.get_fabric_node_id_from_physical_chip_id(receiver->id());

    std::vector<uint32_t> rt_args = {
        static_cast<uint32_t>(dest_full & 0xFFFFFFFFull),
        static_cast<uint32_t>(dest_full >> 32),
        payload_l1,
        kPayloadBytes,
        kNumSends,
        kSendIntervalCycles,
        1u,  // one fabric hop: remote chip -> its own MMIO chip
    };
    tt::tt_fabric::append_fabric_connection_rt_args(
        src_node, dst_node, /*link_idx=*/0, program, eth_core, rt_args, CoreType::ETH);
    tt::tt_metal::SetRuntimeArgs(program, kernel, eth_core, rt_args);

    this->RunProgramNonblocking(sender_mesh, program);

    // Poll host memory. A destination encoded for the wrong NOC lands on some
    // other tile: no fault, no error, nothing ever arrives.
    bool arrived = false;
    uint32_t last_seq = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        const uint32_t w0 = host_va[0];
        if ((w0 & 0xFFFFFF00u) == kSentinelBase) {
            arrived = true;
            last_seq = host_va[1];
            if (last_seq + 1 >= kNumSends) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Read the kernel's own progress markers before judging. They distinguish
    // "kernel never ran" / "connection never opened" / "sent but nothing landed".
    std::vector<uint32_t> dbg(4, 0);
    tt::tt_metal::detail::ReadFromDeviceL1(
        sender, eth_core, payload_l1 + 64, dbg.size() * sizeof(uint32_t), dbg, CoreType::ETH);
    log_info(
        tt::LogTest,
        "kernel markers: alive=0x{:08x} sends_done={} dest=0x{:08x}{:08x}",
        dbg[0], dbg[1], dbg[3], dbg[2]);
    EXPECT_EQ(dbg[0], 0x09E00000u) << "fabric connection never opened (0xA11E0000 = kernel ran but open() blocked; "
                                      "0 = kernel never ran)";
    EXPECT_GT(dbg[1], 0u) << "kernel opened the connection but completed no sends";

    EXPECT_TRUE(arrived) << "no fabric packet reached host memory. Most likely the destination is "
                            "encoded for the wrong NOC: the EDM issues its local write on NOC1 "
                            "(edm_to_local_chip_noc = 1) and the PCIe tile XY encoding is per-NOC.";
    if (arrived) {
        log_info(tt::LogTest, "host memory got sentinel 0x{:08x}, seq {}", host_va[0], last_seq);
        EXPECT_GT(last_seq, 0u) << "only one packet arrived; the stream did not advance";
    }
    this->WaitForSingleProgramDone(sender_mesh, program);
}


// Read-only. Allocates a NOC-mapped host buffer and decodes the address the
// KERNEL DRIVER produced, comparing it against the NOC0 and NOC1 PCIe encodings
// computed host-side. Answers "which NOC is the driver's address encoded for?"
// without issuing a single device write -- the question that has to be settled
// before aiming another fabric packet at it.
TEST_F(Fabric1DFixture, TestSysmemNocAddressEncoding) {
    constexpr uint32_t kNocAddrLocalBits = 36;
    constexpr uint32_t kNocAddrNodeIdBits = 6;
    constexpr uint32_t kNocCoordRegOffset = 4;
    constexpr uint64_t kPcieMarker = 0x800000000ull;

    auto xy_encoding = [](uint32_t x, uint32_t y) -> uint32_t {
        return (y << ((kNocAddrLocalBits % 32) + kNocAddrNodeIdBits)) | (x << (kNocAddrLocalBits % 32));
    };
    auto pcie_encoding = [&](uint32_t x, uint32_t y) -> uint64_t {
        return ((uint64_t)xy_encoding(x, y) << (kNocAddrLocalBits - kNocCoordRegOffset)) | kPcieMarker;
    };

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& devices = this->get_devices();
    ASSERT_FALSE(devices.empty());

    const auto chip = devices[0]->get_devices()[0]->id();
    const auto mmio_id = cluster.get_associated_mmio_device(chip);

    auto sysmem = cluster.allocate_sysmem_buffer(mmio_id, 4096, /*map_to_noc=*/true);
    ASSERT_NE(sysmem, nullptr);
    const auto noc_addr_opt = sysmem->get_noc_addr();
    ASSERT_TRUE(noc_addr_opt.has_value()) << "buffer was not NOC-mapped (needs IOMMU + KMD >= 2.0.0)";
    const uint64_t drv = noc_addr_opt.value();

    const auto& soc = cluster.get_soc_desc(mmio_id);
    const auto& pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    ASSERT_FALSE(pcie_cores.empty());
    const uint32_t px = pcie_cores.front().x;
    const uint32_t py = pcie_cores.front().y;
    const auto grid = soc.grid_size;

    // NOC1 mirrors coordinates: phys = (size - 1) - logical.
    const uint32_t px1 = (grid.x - 1) - px;
    const uint32_t py1 = (grid.y - 1) - py;

    const uint64_t enc0 = pcie_encoding(px, py);
    const uint64_t enc1 = pcie_encoding(px1, py1);
    const uint64_t offset_mask = (1ull << kNocAddrLocalBits) - 1;

    log_info(tt::LogTest, "mmio chip {}  grid {}x{}  PCIe NOC0 core ({},{})", mmio_id, grid.x, grid.y, px, py);
    log_info(tt::LogTest, "driver noc_addr = 0x{:016x}", drv);
    log_info(tt::LogTest, "  high (xy+marker) = 0x{:016x}   offset = 0x{:x}", drv & ~offset_mask, drv & offset_mask);
    log_info(tt::LogTest, "NOC0 PCIe encoding = 0x{:016x}  match={}", enc0, ((drv & ~offset_mask) == (enc0 & ~offset_mask)));
    log_info(tt::LogTest, "NOC1 PCIe encoding = 0x{:016x}  match={}", enc1, ((drv & ~offset_mask) == (enc1 & ~offset_mask)));
    log_info(tt::LogTest, "PCIe marker bit (0x8_0000_0000) present in driver addr: {}", (drv & kPcieMarker) != 0);
}


// Discriminator for the fabric test's failure. Writes to the SAME PCIe
// destination from a kernel on the MMIO chip itself, no fabric involved.
//   bytes land  -> the address is right, the EDM receive path is the problem
//   nothing     -> the address is wrong, fabric is exonerated
// Also diffs the host-computed address against the one a kernel computes the
// way cq_realtime_profiler_push.cpp does.
TEST_F(Fabric1DFixture, TestDirectPcieWriteFromMmioChip) {
    if (!slow_dispatch_) {
        GTEST_SKIP() << "run under TT_METAL_SLOW_DISPATCH_MODE for parity with the fabric test";
    }
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& devices = this->get_devices();
    ASSERT_FALSE(devices.empty());

    tt::tt_metal::IDevice* mmio_dev = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mmio_mesh;
    for (const auto& d : devices) {
        auto* dev = d->get_devices()[0];
        if (cluster.get_associated_mmio_device(dev->id()) == dev->id()) {
            mmio_dev = dev;
            mmio_mesh = d;
            break;
        }
    }
    ASSERT_NE(mmio_dev, nullptr);
    const auto mmio_id = mmio_dev->id();

    auto sysmem = cluster.allocate_sysmem_buffer(mmio_id, 4096, /*map_to_noc=*/true);
    ASSERT_NE(sysmem, nullptr);
    const auto noc_addr_opt = sysmem->get_noc_addr();
    ASSERT_TRUE(noc_addr_opt.has_value());
    const uint64_t driver_off = noc_addr_opt.value();

    constexpr uint32_t kLocalBits = 36, kNodeIdBits = 6, kCoordOff = 4;
    constexpr uint64_t kPcieMarker = 0x800000000ull;
    auto pcie_encoding = [&](uint32_t x, uint32_t y) -> uint64_t {
        const uint32_t xy = (y << ((kLocalBits % 32) + kNodeIdBits)) | (x << (kLocalBits % 32));
        return ((uint64_t)xy << (kLocalBits - kCoordOff)) | kPcieMarker;
    };
    const auto& soc = cluster.get_soc_desc(mmio_id);
    const auto& pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    ASSERT_FALSE(pcie_cores.empty());
    const uint32_t px = pcie_cores.front().x, py = pcie_cores.front().y;
    const auto grid = soc.grid_size;
    const uint64_t host_dest = pcie_encoding((grid.x - 1) - px, (grid.y - 1) - py) | driver_off;

    auto* host_va = static_cast<volatile uint32_t*>(sysmem->get_buffer_va());
    for (uint32_t i = 0; i < 64; i++) {
        host_va[i] = 0xDEADBEEFu;
    }

    // Run on an IDLE ETH core, on NOC1 -- the NOC the EDM issues its local write
    // on. (A Tensix core would need a real L1 allocation: hal.get_dev_addr(TENSIX,
    // UNRESERVED) is explicitly forbidden, whereas IDLE_ETH exposes an unreserved
    // base -- the same one the fabric sender kernel already uses.)
    auto mmio_idle_eth = mmio_dev->get_inactive_ethernet_cores();
    if (mmio_idle_eth.empty()) {
        GTEST_SKIP() << "no inactive ethernet cores on the MMIO chip";
    }
    const tt::tt_metal::CoreCoord core = *mmio_idle_eth.begin();
    auto program = tt::tt_metal::CreateProgram();
    auto kernel = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_direct_pcie_write.cpp",
        core,
        tt::tt_metal::EthernetConfig{
            .eth_mode = tt::tt_metal::Eth::IDLE,
            .noc = tt::tt_metal::NOC::NOC_1,
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0});

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t src_l1 = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);

    tt::tt_metal::SetRuntimeArgs(
        program, kernel, core,
        {static_cast<uint32_t>(host_dest & 0xFFFFFFFFull), static_cast<uint32_t>(host_dest >> 32), px, py,
         static_cast<uint32_t>(driver_off & 0xFFFFFFFFull), static_cast<uint32_t>(driver_off >> 32), src_l1, 32u});

    log_info(tt::LogTest, "mmio chip {}  driver_off=0x{:x}  host_dest=0x{:016x}", mmio_id, driver_off, host_dest);
    this->RunProgramNonblocking(mmio_mesh, program);
    this->WaitForSingleProgramDone(mmio_mesh, program);

    std::vector<uint32_t> dbg(4, 0);
    tt::tt_metal::detail::ReadFromDeviceL1(
        mmio_dev, core, src_l1 + 64, dbg.size() * sizeof(uint32_t), dbg, CoreType::ETH);
    const uint64_t kernel_dest = ((uint64_t)dbg[2] << 32) | dbg[1];
    log_info(tt::LogTest, "kernel stage=0x{:08x} noc_index={} kernel_dest=0x{:016x}", dbg[0], dbg[3], kernel_dest);
    log_info(tt::LogTest, "host_dest == kernel_dest ? {}", host_dest == kernel_dest);

    const bool landed_host_addr = (host_va[0] == 0x51DECAFEu);
    const bool landed_kernel_addr = (host_va[32] == 0x51DECAFEu);  // +128 bytes
    log_info(
        tt::LogTest, "host[0]=0x{:08x}  host[+128]=0x{:08x}", (uint32_t)host_va[0], (uint32_t)host_va[32]);

    EXPECT_TRUE(landed_host_addr || landed_kernel_addr)
        << "NEITHER address delivered from a kernel on the MMIO chip itself. The destination "
           "encoding is wrong -- fabric is not the problem.";
    if (landed_host_addr) {
        log_info(tt::LogTest, "host-computed address WORKS -> fabric EDM receive path is the fault");
    }
    if (landed_kernel_addr && !landed_host_addr) {
        log_info(tt::LogTest, "only the kernel-computed address works -> host encoding is wrong");
    }
}

// CONTROL for TestFabricWriteReachesHostMemory.
//
// That test blames the PCIe destination, but it never established that this test
// harness delivers ANYTHING. A broken route, hop count or connection arg would
// produce the identical symptom (sends complete, nothing arrives) and the PCIe
// conclusion would be wrong. So: same sender, same route, same everything --
// destination changed to ordinary L1 on the MMIO chip.
//
//   lands   -> harness is sound, the PCIe destination really is the fault
//   nothing -> the harness is broken and the PCIe conclusion is unfounded
TEST_F(Fabric1DFixture, TestFabricWriteReachesRemoteL1_Control) {
    if (!slow_dispatch_) {
        GTEST_SKIP() << "IDLE_ETH launch needs TT_METAL_SLOW_DISPATCH_MODE";
    }
    const auto& devices = this->get_devices();
    ASSERT_GE(devices.size(), 2u);
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    tt::tt_metal::IDevice* sender = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> sender_mesh;
    for (const auto& d : devices) {
        auto* dev = d->get_devices()[0];
        if (cluster.get_associated_mmio_device(dev->id()) != dev->id()) {
            sender = dev;
            sender_mesh = d;
            break;
        }
    }
    if (sender == nullptr) {
        GTEST_SKIP() << "no remote chip";
    }
    const auto mmio_id = cluster.get_associated_mmio_device(sender->id());
    tt::tt_metal::IDevice* receiver = nullptr;
    for (const auto& d : devices) {
        if (d->get_devices()[0]->id() == mmio_id) {
            receiver = d->get_devices()[0];
        }
    }
    ASSERT_NE(receiver, nullptr);

    auto s_eth = sender->get_inactive_ethernet_cores();
    auto r_eth = receiver->get_inactive_ethernet_cores();
    if (s_eth.empty() || r_eth.empty()) {
        GTEST_SKIP() << "need an idle eth core on both chips";
    }
    const tt::tt_metal::CoreCoord send_core = *s_eth.begin();
    const tt::tt_metal::CoreCoord recv_core = *r_eth.begin();

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t l1 =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);

    // Clear the landing spot on the receiver.
    std::vector<uint32_t> clear(8, 0xDEADBEEFu);
    tt::tt_metal::detail::WriteToDeviceL1(receiver, recv_core, l1 + 256, clear, CoreType::ETH);

    // Ordinary L1 destination: translated coords of the receiver's eth core.
    const auto& rsoc = cluster.get_soc_desc(mmio_id);
    const auto rt = rsoc.translate_coord_to(
        tt::umd::CoreCoord(recv_core.x, recv_core.y, CoreType::ETH, CoordSystem::LOGICAL), CoordSystem::TRANSLATED);
    constexpr uint32_t kLocalBits = 36, kNodeIdBits = 6, kCoordOff = 4;
    const uint32_t xy = ((uint32_t)rt.y << ((kLocalBits % 32) + kNodeIdBits)) | ((uint32_t)rt.x << (kLocalBits % 32));
    const uint64_t dest_l1 = ((uint64_t)xy << (kLocalBits - kCoordOff)) | (uint64_t)(l1 + 256);
    log_info(
        tt::LogTest,
        "control: receiver eth {} translated ({},{}) dest=0x{:016x}",
        recv_core.str(),
        rt.x,
        rt.y,
        dest_l1);

    auto program = tt::tt_metal::CreateProgram();
    auto kernel = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_fabric_pcie_target.cpp",
        send_core,
        tt::tt_metal::EthernetConfig{
            .eth_mode = tt::tt_metal::Eth::IDLE, .processor = tt::tt_metal::DataMovementProcessor::RISCV_0});

    auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto src_node = cp.get_fabric_node_id_from_physical_chip_id(sender->id());
    const auto dst_node = cp.get_fabric_node_id_from_physical_chip_id(receiver->id());

    std::vector<uint32_t> rt_args = {
        static_cast<uint32_t>(dest_l1 & 0xFFFFFFFFull),
        static_cast<uint32_t>(dest_l1 >> 32),
        l1,
        kPayloadBytes,
        kNumSends,
        kSendIntervalCycles,
        1u};
    tt::tt_fabric::append_fabric_connection_rt_args(
        src_node, dst_node, /*link_idx=*/0, program, send_core, rt_args, CoreType::ETH);
    tt::tt_metal::SetRuntimeArgs(program, kernel, send_core, rt_args);

    this->RunProgramNonblocking(sender_mesh, program);
    std::this_thread::sleep_for(std::chrono::seconds(8));

    std::vector<uint32_t> got(2, 0);
    tt::tt_metal::detail::ReadFromDeviceL1(receiver, recv_core, l1 + 256, 8, got, CoreType::ETH);
    log_info(tt::LogTest, "control: receiver L1 = 0x{:08x} seq={}", got[0], got[1]);

    std::vector<uint32_t> dbg(4, 0);
    tt::tt_metal::detail::ReadFromDeviceL1(sender, send_core, l1 + 64, 16, dbg, CoreType::ETH);
    log_info(tt::LogTest, "control: sender markers alive=0x{:08x} sends_done={}", dbg[0], dbg[1]);

    EXPECT_EQ(got[0] & 0xFFFFFF00u, kSentinelBase)
        << "fabric did not deliver even to ORDINARY L1 -- this harness is broken, and the "
           "PCIe conclusion in TestFabricWriteReachesHostMemory is unfounded.";
    this->WaitForSingleProgramDone(sender_mesh, program);
}

}  // namespace tt::tt_fabric::fabric_router_tests
