// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <device_impl.hpp>
#include <tt-metalium/program.hpp>
#include <stdint.h>
#include <map>
#include <string>
#include <vector>

#include "assert.hpp"
#include "core_coord.hpp"
#include "mesh_graph.hpp"
#include "system_memory_manager.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include <umd/device/tt_xy_pair.h>
#include "utils.hpp"

enum class CoreType;
namespace tt {
namespace tt_metal {
class IDevice;
class Program;
enum DispatchWorkerType : uint32_t;
enum NOC : uint8_t;
}  // namespace tt_metal
}  // namespace tt

#define UNUSED_LOGICAL_CORE tt_cxy_pair(device_->id(), 0, 0)
#define UNUSED_SEM_ID 0

typedef struct {
    tt::tt_metal::NOC non_dispatch_noc;  // For communicating with workers/DRAM/host
    tt::tt_metal::NOC upstream_noc;      // For communicating with upstream dispatch modules
    tt::tt_metal::NOC downstream_noc;    // For communicating with downstream dispatch modules
} noc_selection_t;

static std::vector<string> dispatch_kernel_file_names = {
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",        // PREFETCH
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",        // PREFETCH_HD
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",        // PREFETCH_H
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",        // PREFETCH_D
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",        // DISPATCH
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",        // DISPATCH_HD
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",        // DISPATCH_H
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",        // DISPATCH_D
    "tt_metal/impl/dispatch/kernels/cq_dispatch_slave.cpp",  // DISPATCH_S
    "",                                                      // MUX
    "tt_metal/impl/dispatch/kernels/packet_mux.cpp",         // MUX_D
    "tt_metal/impl/dispatch/kernels/packet_demux.cpp",       // DEMUX
    "",                                                      // DEMUX_D
    "tt_metal/impl/dispatch/kernels/vc_eth_tunneler.cpp",    // US_TUNNELER_LOCAL
    "tt_metal/impl/dispatch/kernels/vc_eth_tunneler.cpp",    // US_TUNNELER_REMOTE
    "tt_metal/impl/dispatch/kernels/vc_packet_router.cpp",   // PACKET_ROUTER_MUX
    "tt_metal/impl/dispatch/kernels/vc_packet_router.cpp",   // PACKET_ROUTER_DEMUX
    "",                                                      // FABRIC_ROUTER_VC
    ""                                                       // COUNT
};

// Top-level class describing a Fast Dispatch Kernel (kernel running on a specific core). All FD kernels should inherit
// from this class and implement the virtual functions as required.
class FDKernel {
public:
    FDKernel(
        int node_id, chip_id_t device_id, chip_id_t servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection) :
        node_id_(node_id),
        device_id_(device_id),
        servicing_device_id_(servicing_device_id),
        cq_id_(cq_id),
        noc_selection_(noc_selection) {}
    virtual ~FDKernel() = default;

    // Populate the static configs for this kernel (ones that do not depend on configs from other kernels), including
    // the logical core placement. Is called after AddDeviceAndProgram and AddUpstreamKernel/AddDownstreamKernel.
    virtual void GenerateStaticConfigs() = 0;

    // Populate the dependent configs for this kernel (ones that depend on static configs from other kernels). Is called
    // after GenerateStaticConfigs for all upstream/downstream kernels.
    virtual void GenerateDependentConfigs() = 0;

    // Use all configs and add this kernel to its Program. Called after GenerateStaticConfigs/GenerateDependentConfigs.
    virtual void CreateKernel() = 0;

    // Override for specific kernels that need host-side configureation (special values written to l1, etc.). Is called
    // after above functions and before FD kernels are launched.
    virtual void ConfigureCore() {}

    // Override for specific kernels that can be configured for fabric. Will be called by the FABRIC_ROUTER_VC, which is
    // an intermediary FDKernel for indicating a fabric router path needs to be found.
    virtual void UpdateArgsForFabric(
        const CoreCoord& fabric_router_virtual,
        tt::tt_fabric::mesh_id_t upstream_mesh_id,
        chip_id_t upstream_chip_id,
        tt::tt_fabric::mesh_id_t downstream_mesh_id,
        chip_id_t downstream_chip_id) {}

    // Generator function to create a kernel of a given type. New kernels need to be added here.
    static FDKernel* Generate(
        int node_id,
        chip_id_t device_id,
        chip_id_t servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        tt::tt_metal::DispatchWorkerType type);

    // Register another kernel as upstream/downstream of this one
    void AddUpstreamKernel(FDKernel* upstream) { upstream_kernels_.push_back(upstream); }
    void AddDownstreamKernel(FDKernel* downstream) { downstream_kernels_.push_back(downstream); }

    virtual CoreType GetCoreType() {
        return tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    }
    tt_cxy_pair GetLogicalCore() { return logical_core_; }
    tt_cxy_pair GetVirtualCore() {
        return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            logical_core_, GetCoreType());
    }
    chip_id_t GetDeviceId() { return device_id_; }  // Since this->device may not exist yet

    // Get the port index for which a given kernel is upstream/downstream of this one
    int GetUpstreamPort(FDKernel* other) { return GetPort(other, this->upstream_kernels_); }
    int GetDownstreamPort(FDKernel* other) { return GetPort(other, this->downstream_kernels_); }
    void AddDevice(tt::tt_metal::IDevice* device) { device_ = device; }
    void AddProgram(tt::tt_metal::Program* program) { program_ = program; }

protected:
    void configure_kernel_variant(
        const string& path,
        const std::vector<uint32_t>& compile_args,
        std::map<string, string> defines_in,
        bool is_active_eth_core,
        bool send_to_brisc,
        bool force_watcher_no_inline,
        tt::tt_metal::KernelBuildOptLevel opt_level = tt::tt_metal::KernelBuildOptLevel::Os);
    int GetPort(FDKernel* other, std::vector<FDKernel*>& kernels) {
        for (int idx = 0; idx < kernels.size(); idx++) {
            if (kernels[idx] == other) {
                return idx;
            }
        }
        TT_ASSERT(false);
        return -1;
    }

    // Some static helper functions commonly used by FD kernels
    static chip_id_t GetUpstreamDeviceId(chip_id_t device_id);
    static chip_id_t GetDownstreamDeviceId(chip_id_t device_id);
    static uint32_t GetTunnelStop(chip_id_t device_id);

    tt::tt_metal::IDevice* device_ = nullptr;  // Set at configuration time by AddDeviceAndProgram()
    tt::tt_metal::Program* program_ = nullptr;
    tt_cxy_pair logical_core_;
    chip_id_t device_id_;
    chip_id_t servicing_device_id_;  // Remote chip that this PREFETCH_H/DISPATCH_H is servicing
    int node_id_;
    uint8_t cq_id_;
    noc_selection_t noc_selection_;

    std::vector<FDKernel*> upstream_kernels_;
    std::vector<FDKernel*> downstream_kernels_;
};
