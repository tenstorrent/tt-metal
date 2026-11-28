// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <hostdevcommon/fabric_common.h>  // chan_id_t
#include "tt_metal/fabric/hw/inc/edm_fabric/host_rdma_service_abi.hpp"
#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

namespace tt {
class Cluster;
}

namespace tt::tt_metal {
class MetalContext;
class Program;
namespace distributed {
class CoreWriter;
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_fabric::host_rdma {

// Added to the 32-bit counter at `addr` on the destination chip once every packet of the write has landed.
struct Completion {
    tt_metal::CoreCoord virtual_core;
    uint64_t addr = 0;
    uint32_t increment = 1;
};

struct DeviceEndpoint {
    tt_metal::CoreCoord virtual_core;
    uint32_t addr = 0;
};

// Addresses and the size are 16 B aligned.
struct WriteRequest {
    DeviceEndpoint src;
    FabricNodeId dst{MeshId{0}, 0};
    DeviceEndpoint dst_at;
    uint32_t size = 0;
    std::optional<Completion> completion;
};

// Where the request service runs. For Dram, `logical_core` is a DRAM-logical coordinate (x = bank, y = endpoint).
enum class ServiceCore { Tensix, Dram };

// Owns the L1 of its core from the unreserved base. One per chip; not thread safe.
class HostRdmaService : public std::enable_shared_from_this<HostRdmaService> {
public:
    static std::shared_ptr<HostRdmaService> create(
        std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
        tt_metal::CoreCoord logical_core,
        ServiceCore core = ServiceCore::Dram);
    // Does not wait for posted requests: drain() first.
    ~HostRdmaService() noexcept;

    HostRdmaService(const HostRdmaService&) = delete;
    HostRdmaService& operator=(const HostRdmaService&) = delete;

    // Blocks while the queue is full. Returns once posted, not once the data has landed (see Completion).
    void write(const WriteRequest& request);
    void write_batch(std::span<const WriteRequest> requests);
    void drain();

    FabricNodeId node() const { return node_; }
    std::string describe() const;

private:
    struct Channel {
        uint32_t entry = 0;
        chan_id_t eth_chan = 0;
        fabric_connection_info_t router;
        uint32_t produced = 0;
        uint32_t consumed_as_last_read = 0;
    };

    HostRdmaService(
        std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
        tt_metal::CoreCoord logical_core,
        ServiceCore core);
    void barrier_service_core() const;
    size_t channel_toward(FabricNodeId dst);
    size_t channel_on(chan_id_t eth_chan);
    service_abi::Request encode(const WriteRequest& request) const;
    void post(Channel& ch, std::span<const service_abi::Request> requests);
    uint32_t channel_addr(uint32_t entry) const;
    uint32_t read_word(uint32_t addr) const;
    uint32_t read_consumed(Channel& ch) const;
    void publish_word(uint32_t addr, uint32_t value);

    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device_;
    tt::tt_metal::MetalContext& context_;
    std::unique_ptr<tt_metal::distributed::CoreWriter> writer_;
    const tt::Cluster& cluster_;
    ChipId chip_ = 0;
    FabricNodeId node_{MeshId{0}, 0};
    tt_metal::CoreCoord virtual_core_;
    ServiceCore core_ = ServiceCore::Tensix;
    uint64_t l1_noc_offset_ = 0;
    std::unique_ptr<tt_metal::Program> dram_program_;
    uint32_t layout_addr_ = 0;
    std::vector<Channel> channels_;
    std::unordered_map<FabricNodeId, size_t> channel_index_by_destination_;
};

}  // namespace tt::tt_fabric::host_rdma
