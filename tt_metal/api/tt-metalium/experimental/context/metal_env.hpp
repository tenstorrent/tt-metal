// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_fabric {
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt::tt_metal::distributed {
class SystemMesh;
}  // namespace tt::tt_metal::distributed

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

// Configuration for a MetalEnv.
//
// The default descriptor discovers and connects to the physical cluster present in the system.
// A custom MetalEnvDescriptor can be supplied to target a mock/simulated cluster instead.
//
// Only one MetalEnv for the physical cluster may exist at a time  due to UMD limitations.
class MetalEnvDescriptor {
public:
    MetalEnvDescriptor() = default;

    explicit MetalEnvDescriptor(const std::string& mock_cluster_desc_path);

    explicit MetalEnvDescriptor(std::optional<std::string> mock_cluster_desc_path);

    MetalEnvDescriptor(std::optional<std::string> mock_cluster_desc_path, FabricConfigDescriptor fabric_config_desc);

    bool is_mock_device() const { return mock_cluster_desc_path_.has_value(); }
    const std::string& mock_cluster_desc_path() const { return *mock_cluster_desc_path_; }
    const FabricConfigDescriptor& fabric_config_descriptor() const { return fabric_config_desc_; }

protected:
    std::optional<std::string> mock_cluster_desc_path_ = std::nullopt;
    FabricConfigDescriptor fabric_config_desc_;
};

class MetalEnvImpl;

// A MetalEnv provides an interface for the runtime environment to access a homogeneous cluster of Tenstorrent devices.
// It exposes several query functions for the hardware capabilities and cluster configuration.
//
// The FabricConfigDescriptor in the MetalEnvDescriptor describes the topology of the devices — how they are
// interconnected and how traffic is routed between them. From this topology the MetalEnv constructs the fabric
// control plane and the system mesh, which virtualize and partition the physical hardware.
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

    /// @return The fabric control plane, lazily initialized.
    /// The control plane manages routing tables and fabric channels based on the device topology
    /// described by the environment's FabricConfigDescriptor.
    tt::tt_fabric::ControlPlane& get_control_plane();

    /// @return The system mesh, lazily initialized.
    /// The system mesh provides a virtualized coordinate system over the physical devices, allowing
    /// MeshDevice instances to map logical coordinates to physical device IDs.
    distributed::SystemMesh& get_system_mesh();

private:
    friend class MetalEnvAccessor;
    std::unique_ptr<MetalEnvImpl> impl_;

    MetalEnvImpl& impl() { return *impl_; }
    MetalEnvDescriptor descriptor_;
};

}  // namespace tt::tt_metal
