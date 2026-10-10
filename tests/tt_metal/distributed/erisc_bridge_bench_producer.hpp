// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The T6 traffic source: a leg that only drains measures nothing, which reads as a clean
// drained=0. It does not divert packets -- that is the router's job, on a bridged link only.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include "tt_metal/fabric/erisc_bridge_block.hpp"
#include "tt_metal/distributed/erisc_bridge_doorbell.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_fabric::erisc_bridge::bench {

// Worker L1 from the allocator base: below it is the kernel-config ring, which dispatch rewrites per launch.
inline uint32_t src_l1(tt::tt_metal::IDevice* dev) {
    return static_cast<uint32_t>(dev->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
}
// {stage, frames sent}, past the largest payload; read back when a run stalls.
inline uint32_t status_l1(tt::tt_metal::IDevice* dev) { return src_l1(dev) + 0x4000; }

struct Producer {
    tt::tt_metal::distributed::MeshWorkload wl;
    bool ok = false;
    std::string err;
};

inline std::string env_str(const char* n, const char* d) {
    const char* v = std::getenv(n);
    return v != nullptr ? std::string(v) : std::string(d);
}

// Is the router on this core actually bridged, read from its own status block: build_stamp says
// this binary is running, magic says its init ran. Either missing and a run reports drained=0.
inline bool bridge_router_armed(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& eth_logical,
    std::uint32_t status_addr,
    tt::tt_fabric::erisc_bridge::BridgeStatus& out,
    std::string& why) {
    out = {};
    if (!tt::tt_fabric::erisc_bridge::read_bytes_from_l1(
            dev, eth_logical, status_addr, &out, sizeof(out), tt::CoreType::ETH)) {
        why = "could not read the bridge status block at 0x" + std::to_string(status_addr);
        return false;
    }
    if (out.build_stamp != tt::tt_fabric::erisc_bridge::kBridgeBuildStamp) {
        why = "the router on this core is the stock router: name its channel with TT_BRIDGE_E2H_FORCE_CHAN.";
        return false;
    }
    if (out.magic != tt::tt_fabric::erisc_bridge::kBridgeStatusMagic) {
        why = "the bridge build is on the core but its init block did not run -- enable_e2h was false here";
        return false;
    }
    return true;
}

// Builds and enqueues the sender NON-BLOCKING, so the caller's drain overlaps it -- a drain
// that runs after the worker records when we got round to reading, not when frames arrived.
inline Producer launch_t6_producer(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh,
    tt::tt_metal::IDevice* dev,  // the chip at mesh (0, 0), which owns the bridged link
    const tt::tt_fabric::FabricNodeId& src_node,
    const tt::tt_fabric::FabricNodeId& dst_node,  // the bridged link's peer, so frames cross it
    uint32_t link_idx,
    uint32_t payload_bytes,
    uint32_t frames) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;
    Producer p;

    const CoreCoord worker{0, 0};
    const CoreCoord dst_worker{0, 0};

    const MeshCoordinateRange range(MeshCoordinate(0, 0), MeshCoordinate(0, 0));
    p.wl.add_program(range, CreateProgram());
    auto& prog = p.wl.get_programs().at(range);

    const std::vector<uint32_t> ct{payload_bytes, frames, src_l1(dev), status_l1(dev)};
    auto k = CreateKernel(
        prog,
        env_str("TT_BRIDGE_T6_KERNEL", "tests/tt_metal/distributed/kernels/erisc_bridge_t6_sender.cpp"),
        worker,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = ct});

    const auto dst_virt = dev->worker_core_from_logical_core(dst_worker);
    std::vector<uint32_t> rt{
        static_cast<uint32_t>(dst_virt.x),
        static_cast<uint32_t>(dst_virt.y),
        src_l1(dev),  // same layout on the destination chip
        static_cast<uint32_t>(*dst_node.mesh_id),
        static_cast<uint32_t>(dst_node.chip_id)};
    tt::tt_fabric::append_fabric_connection_rt_args(src_node, dst_node, link_idx, prog, worker, rt);
    SetRuntimeArgs(prog, k, worker, rt);

    std::vector<uint32_t> zero{0u, 0u};  // stage 0 = the kernel never started
    tt::tt_metal::detail::WriteToDeviceL1(dev, worker, status_l1(dev), zero, tt::CoreType::WORKER);
    EnqueueMeshWorkload(mesh->mesh_command_queue(), p.wl, /*blocking=*/false);
    p.ok = true;
    return p;
}

// Before the socket is constructed: the previous run's cursor is still in L1 and the constructor
// does not reset it, so the router would write mid-ring while the reader starts at zero.
inline bool zero_socket_config(
    tt::tt_metal::IDevice* dev, const tt::tt_metal::CoreCoord& eth_logical, uint32_t block_addr) {
    std::vector<uint32_t> zeros(16, 0);  // 64 B
    return tt::tt_metal::detail::WriteToDeviceL1(dev, eth_logical, block_addr, zeros, tt::CoreType::ETH);
}

}  // namespace tt::tt_fabric::erisc_bridge::bench
