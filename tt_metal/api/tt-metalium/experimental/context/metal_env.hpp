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
#include <tt-metalium/device_types.hpp>
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

// Configuration for a MetalEnv. The default targets the physical cluster.
//
// Set mock_cluster_desc_path to bind a mock cluster instead: a path, or a bare filename that is searched for in the
// known cluster descriptor directories. nullopt is the physical cluster. An empty path is not a mock cluster.
//
// Only one MetalEnv for the physical cluster may exist at a time due to UMD limitations. There is no limit on the
// number of mock clusters.
struct MetalEnvDescriptor {
    std::optional<std::string> mock_cluster_desc_path = std::nullopt;

    bool is_mock_device() const { return mock_cluster_desc_path.has_value() && !mock_cluster_desc_path->empty(); }
};

// Options for creating a MeshDevice from a MetalEnv.
//
// num_command_queues, dispatch_core_config and worker_l1_size apply to the whole MetalEnv: while a MeshDevice created
// from it is open, other create_* calls must pass the same values.
struct CreateMeshDeviceOptions {
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
    uint8_t num_command_queues = 1;
    DispatchCoreConfig dispatch_core_config;
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
};

class MetalEnvImpl;

// A MetalEnv provides an interface for the runtime environment to access a homogeneous cluster of Tenstorrent devices.
// It exposes several query functions for the hardware capabilities and cluster configuration.
//
// Fabric configuration describes the topology of the devices — how they are interconnected and how traffic is
// routed between them. Set it with configure_fabric() before the first get_system_mesh() or create_* call; those
// calls materialize the topology and freeze the configuration. From this topology the MetalEnv constructs the
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

    // Configure fabric for this environment. May be called repeatedly (last call wins) until the topology is
    // materialized by get_system_mesh() or a create_* call; afterwards it throws. Never calling it leaves fabric
    // disabled. num_routing_planes must be greater than 0 when set; leaving it unset uses every available plane.
    void configure_fabric(const FabricConfigDescriptor& fabric);

    /// @return The fabric configuration requested via configure_fabric. Disabled by default. This is the requested
    /// configuration: the runtime may still enable fabric for dispatch on remote devices without changing it.
    const FabricConfigDescriptor& get_fabric_config_descriptor() const;

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

    /// @return Maximum number of dataflow buffers per core of this environment.
    uint32_t get_num_dataflow_buffers() const;

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
