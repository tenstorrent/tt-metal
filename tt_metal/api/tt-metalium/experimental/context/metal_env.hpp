// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tt_stl/span.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

namespace tt::tt_metal {

// Describes the fabric topology and routing configuration for the devices in the environment.
// These parameters determine how devices are interconnected and how data is routed between them.
struct FabricConfigDescriptor {
    tt_fabric::FabricConfig fabric_config = tt_fabric::FabricConfig::DISABLED;
    tt_fabric::FabricReliabilityMode reliability_mode =
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    std::optional<uint8_t> num_routing_planes = std::nullopt;
    tt_fabric::FabricTensixConfig fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED;
    tt_fabric::FabricUDMMode fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED;
    tt_fabric::FabricManagerMode fabric_manager = tt_fabric::FabricManagerMode::DEFAULT;
    tt_fabric::FabricRouterConfig router_config = {};
};

// The cluster a MetalEnv binds to.
//
// Only one MetalEnv for the physical cluster may exist at a time due to UMD limitations. There is no limit on
// the number of mock clusters.
class MetalEnvTarget {
public:
    // The physical cluster present in the system.
    static MetalEnvTarget silicon();

    // A mock cluster described by a cluster descriptor YAML: either a path, or a bare filename that is searched for
    // in the known cluster descriptor directories. Fatal if `cluster_desc` is empty.
    static MetalEnvTarget mock(std::string cluster_desc);

    // A mock cluster of `num_chips` chips of `arch`. Fatal if no cluster descriptor exists for the combination.
    static MetalEnvTarget mock(tt::ARCH arch, uint32_t num_chips);

    bool is_mock() const { return mock_cluster_desc_.has_value(); }

    // Fatal if this is not a mock target.
    const std::string& mock_cluster_desc() const;

private:
    MetalEnvTarget() = default;

    std::optional<std::string> mock_cluster_desc_;
};

// Configuration for a MetalEnv. The default targets the physical cluster with fabric disabled.
//
//     MetalEnv env({.target = MetalEnvTarget::mock(tt::ARCH::BLACKHOLE, 2),
//                   .fabric = {.fabric_config = tt_fabric::FabricConfig::FABRIC_2D}});
struct MetalEnvDescriptor {
    MetalEnvTarget target = MetalEnvTarget::silicon();
    FabricConfigDescriptor fabric = {};
};

// Options for creating a MeshDevice from a MetalEnv.
//
// num_command_queues, dispatch_core_config and worker_l1_size apply to the whole MetalEnv: while a MeshDevice created
// from it is open, other create_* calls must pass the same values.
struct CreateMeshDeviceOptions {
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
    uint8_t num_command_queues = 1;
    DispatchCoreConfig dispatch_core_config = {};
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
};

class MetalEnvImpl;

// A MetalEnv provides an interface for the runtime environment to access a homogeneous cluster of Tenstorrent devices.
// It exposes several query functions for the hardware capabilities and cluster configuration.
//
// The FabricConfigDescriptor in the MetalEnvDescriptor describes the topology of the devices — how they are
// interconnected and how traffic is routed between them. From this topology the MetalEnv constructs the
// system mesh, which virtualizes and partitions the physical hardware for placement queries.
//
// Note, MetalEnv is a RAII object. As such, it must outlive every object that uses it (e.g. MeshDevice).
// The MetalEnv should be destroyed before forking to avoid undefined behavior.
class MetalEnv {
public:
    // Construct and initialize a MetalEnv using the provided descriptor.
    explicit MetalEnv(MetalEnvDescriptor descriptor = {});
    ~MetalEnv();

    MetalEnv(const MetalEnv&) = delete;
    MetalEnv& operator=(const MetalEnv&) = delete;
    MetalEnv(MetalEnv&&) = delete;
    MetalEnv& operator=(MetalEnv&&) = delete;

    /// @return The descriptor used to construct this MetalEnv.
    const MetalEnvDescriptor& get_descriptor() const;

    /// @return Architecture of this environment.
    tt::ARCH get_arch() const;

    /// @return Human-readable name of the architecture of this environment.
    std::string get_arch_name() const;

    /// @return Total number of PCIe devices in this environment.
    uint32_t get_num_pcie_devices() const;

    /// @return Number of available devices in this environment.
    uint32_t get_num_available_devices() const;

    /// @return Size in bytes of each Tensix core's L1 SRAM of this environment.
    uint32_t get_l1_size() const;

    /// @return Required address alignment in bytes for DRAM allocations of this environment.
    uint32_t get_dram_alignment() const;

    /// @return Required address alignment in bytes for L1 allocations of this environment.
    uint32_t get_l1_alignment() const;

    /// @return Maximum number of circular buffers per core of this environment.
    uint32_t get_arch_num_circular_buffers() const;

    /// @return Maximum usable L1 size in bytes when the ring-buffer size is 0 of this environment.
    uint32_t get_max_worker_l1_unreserved_size() const;

    /// @return Representable SFPU epsilon value of this environment.
    float get_eps() const;

    /// @return Representable SFPU NaN value of this environment.
    float get_nan() const;

    /// @return Representable SFPU Infinity value of this environment.
    float get_inf() const;

    /// @return The system mesh, lazily initialized.
    /// The system mesh provides a virtualized coordinate system over the physical devices, allowing
    /// MeshDevice instances to map logical coordinates to physical device IDs.
    distributed::SystemMesh& get_system_mesh();

    // Create a MeshDevice which will use this MetalEnv
    std::shared_ptr<distributed::MeshDevice> create_mesh_device(
        const distributed::MeshDeviceConfig& config, const CreateMeshDeviceOptions& options = {});

    // Create a unit mesh for the physical device ID which will use this MetalEnv
    std::shared_ptr<distributed::MeshDevice> create_unit_mesh(
        ChipId device_id, const CreateMeshDeviceOptions& options = {});

    // Create a unit mesh for each physical device ID in the list which will use this MetalEnv
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> create_unit_meshes(
        ttsl::Span<const ChipId> device_ids, const CreateMeshDeviceOptions& options = {});

private:
    friend class MetalEnvAccessor;
    std::unique_ptr<MetalEnvImpl> impl_;

    MetalEnvImpl& impl() { return *impl_; }
};

}  // namespace tt::tt_metal
