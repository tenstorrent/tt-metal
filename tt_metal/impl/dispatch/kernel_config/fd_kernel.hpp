// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <stdint.h>
#include <map>
#include <string>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <vector>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <tt_stl/tt_stl/reflection.hpp>

namespace tt {
class Cluster;
namespace llrt {
class RunTimeOptions;
}
}  // namespace tt

namespace tt::tt_fabric {
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt::tt_metal {

class IDevice;
class Program;
class Hal;
class dispatch_core_manager;
class DispatchMemMap;
class DispatchQueryManager;
class DeviceManager;
class DPrintServer;
enum DispatchWorkerType : uint32_t;
enum NOC : uint8_t;

// Context struct holding all dependencies needed for dispatch kernel initialization.
// This decouples the kernel config classes from MetalContext.
struct DispatchInitContext {
    tt::Cluster& cluster;
    dispatch_core_manager& dispatch_core_mgr;
    const Hal& hal;
    const DispatchMemMap& dispatch_mem_map;
    DispatchQueryManager& dispatch_query_mgr;
    tt::tt_fabric::ControlPlane& control_plane;
    DeviceManager& device_manager;
    llrt::RunTimeOptions& rtoptions;
    DPrintServer* dprint_server;
};

#define UNUSED_LOGICAL_CORE tt_cxy_pair(device_->id(), 0, 0)
#define UNUSED_SEM_ID 0

struct noc_selection_t {
    tt::tt_metal::NOC non_dispatch_noc;  // For communicating with workers/DRAM/host
    tt::tt_metal::NOC upstream_noc;      // For communicating with upstream dispatch modules
    tt::tt_metal::NOC downstream_noc;    // For communicating with downstream dispatch modules
};

enum class FDKernelType : uint32_t {
    UNSET = 0,
    VIRTUAL,   // Not a real kernel
    DISPATCH,  // Dispatch kernels
    ROUTING,   // Routing/Tunneling kernels
};

struct TerminationInfo {
    CoreCoord logical_core;  // Logical core coordination
    CoreType core_type;      // Core Type
    uint32_t address;        // Termination signal address in L1
    uint32_t val;            // Termination signal value

    bool operator==(const TerminationInfo& other) const {
        return logical_core == other.logical_core && core_type == other.core_type && address == other.address &&
               val == other.val;
    }
};

static std::vector<std::string> dispatch_kernel_file_names = {
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",              // PREFETCH
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",              // PREFETCH_HD
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",              // PREFETCH_H
    "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",              // PREFETCH_D
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",              // DISPATCH
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",              // DISPATCH_HD
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",              // DISPATCH_H
    "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",              // DISPATCH_D
    "tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp",  // DISPATCH_S
    "",                                                            // MUX
    "tt_metal/impl/dispatch/kernels/packet_mux.cpp",               // MUX_D
    "tt_metal/impl/dispatch/kernels/packet_demux.cpp",             // DEMUX
    "",                                                            // DEMUX_D
    "tt_metal/impl/dispatch/kernels/vc_eth_tunneler.cpp",          // US_TUNNELER_LOCAL
    "tt_metal/impl/dispatch/kernels/vc_eth_tunneler.cpp",          // US_TUNNELER_REMOTE
    "tt_metal/impl/dispatch/kernels/vc_packet_router.cpp",         // PACKET_ROUTER_MUX
    "tt_metal/impl/dispatch/kernels/vc_packet_router.cpp",         // PACKET_ROUTER_DEMUX
    "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",              // FABRIC_MUX
    "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",              // FABRIC_RETURN_MUX
    ""                                                             // COUNT
};

// Top-level class describing a Fast Dispatch Kernel (kernel running on a specific core). All FD kernels should inherit
// from this class and implement the virtual functions as required.
class FDKernel {
public:
    FDKernel(int node_id, ChipId device_id, ChipId servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection) :
        device_id_(device_id),
        servicing_device_id_(servicing_device_id),
        node_id_(node_id),
        cq_id_(cq_id),
        noc_selection_(noc_selection) {}
    virtual ~FDKernel() = default;

    // Populate the static configs for this kernel (ones that do not depend on configs from other kernels), including
    // the logical core placement. Is called after AddDeviceAndProgram and AddUpstreamKernel/AddDownstreamKernel.
    virtual void GenerateStaticConfigs() = 0;

    // Populate the dependent configs for this kernel (ones that depend on static configs from other kernels). Is called
    // after GenerateStaticConfigs for all upstream/downstream kernels.
    virtual void GenerateDependentConfigs() = 0;

    virtual void InitializeRuntimeArgsValues() {}
    // Populate the runtime args for this kernel.
    virtual void SetRuntimeArgs();

    // Use all configs and add this kernel to its Program. Called after GenerateStaticConfigs/GenerateDependentConfigs.
    virtual void CreateKernel() = 0;

    // Override for specific kernels that need host-side configureation (special values written to l1, etc.). Is called
    // after above functions and before FD kernels are launched.
    virtual void ConfigureCore() {}

    // Generator function to create a kernel of a given type. New kernels need to be added here.
    static FDKernel* Generate(
        int node_id,
        ChipId device_id,
        ChipId servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        tt::tt_metal::DispatchWorkerType type,
        int tunnel_index = -1);

    // Translate DispatchCoreType to programmable core type index
    uint32_t get_programmable_core_type_index(CoreType dispatch_core_type, bool is_active_eth_core = false) const;

    // Translate core coord using the chip_id from the logical_cxy
    //
    // IDevice::virtual_core_from_logical_core uses the chip_id of the device instance whereas this function uses the
    // chip_id specified in the logical coordinate.
    CoreCoord get_virtual_core_coord(const tt_cxy_pair& logical_cxy, const CoreType& core_type) const;

    // Register another kernel as upstream/downstream of this one
    void AddUpstreamKernel(FDKernel* upstream) { upstream_kernels_.push_back(upstream); }
    void AddDownstreamKernel(FDKernel* downstream) { downstream_kernels_.push_back(downstream); }

    virtual CoreType GetCoreType() const;
    FDKernelType GetKernelType() const { return kernel_type_; }
    tt_cxy_pair GetLogicalCore() const { return logical_core_; }
    tt_cxy_pair GetVirtualCore() const;
    ChipId GetDeviceId() const { return device_id_; }  // Since this->device may not exist yet
    int GetNodeId() const { return node_id_; }
    virtual std::optional<tt::tt_metal::TerminationInfo> GetTerminationInfo() const { return std::nullopt; }

    // Get the port index for which a given kernel is upstream/downstream of this one
    int GetUpstreamPort(FDKernel* other) const { return GetPort(other, this->upstream_kernels_); }
    int GetDownstreamPort(FDKernel* other) const { return GetPort(other, this->downstream_kernels_); }
    void AddDevice(tt::tt_metal::IDevice* device) { device_ = device; }
    void AddProgram(tt::tt_metal::Program* program) { program_ = program; }
    void SetContext(DispatchInitContext* ctx) { ctx_ = ctx; }
    DispatchInitContext& ctx() const {
        TT_ASSERT(ctx_ != nullptr, "DispatchInitContext not set");
        return *ctx_;
    }

protected:
    // Attributes for an EDM client to connect to the router
    struct FDKernelEdmConnectionAttributes {
        size_t worker_flow_control_sem{0};
        size_t worker_teardown_sem{0};
        size_t worker_buffer_index_sem{0};
    };

    [[maybe_unused]] KernelHandle configure_kernel_variant(
        const std::string& path,
        const std::vector<uint32_t>& compile_args,
        std::map<std::string, std::string> defines_in,
        bool is_active_eth_core,
        bool send_to_brisc,
        bool force_watcher_no_inline,
        tt::tt_metal::KernelBuildOptLevel opt_level = tt::tt_metal::KernelBuildOptLevel::Os);
    int GetPort(const FDKernel* other, const std::vector<FDKernel*>& kernels) const {
        for (int idx = 0; idx < kernels.size(); idx++) {
            if (kernels[idx] == other) {
                return idx;
            }
        }
        TT_ASSERT(false);
        return -1;
    }

    // Helper function to get upstream device in the tunnel from current device, not valid for mmio
    ChipId GetUpstreamDeviceId(ChipId device_id) const;
    // Helper function to get downstream device in the tunnel from current device
    ChipId GetDownstreamDeviceId(ChipId device_id, int tunnel = -1) const;
    // Helper function to get the tunnel stop index of current device
    uint32_t GetTunnelStop(ChipId device_id) const;
    // Create and populate semaphores for the EDM connection
    void create_edm_connection_sems(FDKernelEdmConnectionAttributes& attributes);

    DispatchInitContext* ctx_ = nullptr;  // Set by SetContext(), must be set before GenerateStaticConfigs()
    IDevice* device_ = nullptr;           // Set at configuration time by AddDeviceAndProgram()
    Program* program_ = nullptr;
    tt_cxy_pair logical_core_;
    FDKernelType kernel_type_ = FDKernelType::UNSET;
    KernelHandle kernel_handle_{std::numeric_limits<KernelHandle>::max()};  // Invalid handle
    ChipId device_id_;
    ChipId servicing_device_id_;  // Remote chip that this PREFETCH_H/DISPATCH_H is servicing
    int node_id_;
    uint8_t cq_id_;
    noc_selection_t noc_selection_;

    std::vector<FDKernel*> upstream_kernels_;
    std::vector<FDKernel*> downstream_kernels_;

    std::vector<uint32_t> runtime_args_;
};

}  // namespace tt::tt_metal

namespace std {
template <>
struct hash<tt::tt_metal::TerminationInfo> {
    std::size_t operator()(const tt::tt_metal::TerminationInfo& info) const {
        size_t hash = 0;
        ttsl::hash::hash_combine(hash, info.logical_core);
        ttsl::hash::hash_combine(hash, info.core_type);
        ttsl::hash::hash_combine(hash, info.address);
        ttsl::hash::hash_combine(hash, info.val);
        return hash;
    }
};
}  // namespace std
