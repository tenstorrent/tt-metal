// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include "accessor/tensor_accessor.h"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "dataflow_api_addrgen.h"
#endif

// Forward declaration
template <std::size_t CTA_OFFSET>
struct MeshGcoreAccessorArgs;

/**
 * @brief Accessor for direct core-to-core access across devices in a mesh
 *
 * MeshGcoreAccessor allows a kernel on global core (x0, y0) to send requests to
 * global core (x1, y1) in a multi-device mesh. It converts global core coordinates
 * to grid_id and local bank_id.
 *
 * **Process:**
 * 1. Compute mesh_strides and grid_strides (done in constructor)
 * 2. Convert gcore_coord to (grid_id, bank_id):
 *    - grid_id = sum((gcore_coord[i] / grid.dim[i]) * mesh_strides[i])
 *    - bank_id = sum((gcore_coord[i] % grid.dim[i]) * grid_strides[i])
 * 3. Convert bank_id to NOC address using get_noc_addr_from_bank_id<L1>()
 *
 * **Example:**
 * @code
 *   auto args = MeshGcoreAccessorArgs<0>();
 *   MeshGcoreAccessor accessor(args);
 *   uint32_t gcore[3] = {1, 5, 10};
 *   uint64_t noc_addr = accessor.get_noc_addr(gcore);
 * @endcode
 *
 * **Note:** When mesh and grid have different ranks, the host side automatically
 * adjusts them to the same rank by prepending 1s to the shorter shape before passing
 * as compile-time args. Example: mesh (2,2,2) + grid (16,16) → grid becomes (1,16,16)
 */
struct MeshGcoreAccessor {
    static constexpr uint32_t MAX_DIMS = tensor_accessor::MAX_RANK;

    /**
     * @brief Configuration for mesh topology
     */
    struct MeshConfig {
        uint32_t num_dims;
        std::array<uint32_t, MAX_DIMS> dim;      // mesh dimensions (e.g., 2x2x2)
        std::array<uint32_t, MAX_DIMS> strides;  // pre-computed mesh strides for indexing

        constexpr MeshConfig() : num_dims(0), dim{}, strides{} {}
    };

    /**
     * @brief Configuration for grid topology (cores per device)
     */
    struct GridConfig {
        uint32_t num_dims;
        std::array<uint32_t, MAX_DIMS> dim;      // grid dimensions (e.g., 16,16)
        std::array<uint32_t, MAX_DIMS> strides;  // pre-computed grid strides for indexing

        constexpr GridConfig() : num_dims(0), dim{}, strides{} {}
    };

    /**
     * @brief Mapping of global core coordinate to grid_id and local bank_id
     */
    struct GridBankMapping {
        size_t grid_id;  // which device/grid
        size_t bank_id;  // which bank within the device
    };

    /**
     * @brief NOC address with fabric node information
     */
    struct NocAddrWithFabricNode {
        uint64_t noc_addr;        // NOC address for accessing memory
        uint32_t fabric_mesh_id;  // Fabric mesh ID of the device
        uint32_t fabric_chip_id;  // Fabric chip ID of the device
    };

    /**
     * @brief Configuration for grid to fabric node id mapping
     */
    struct GridToFabricNodeMapping {
        uint32_t num_grids;
        std::array<std::pair<uint32_t, uint32_t>, MAX_DIMS> fabric_node_ids;  // (mesh_id, chip_id) for each grid

        constexpr GridToFabricNodeMapping() : num_grids(0), fabric_node_ids{} {}
    };

private:
    MeshConfig mesh_config_;
    GridConfig grid_config_;
    GridToFabricNodeMapping fabric_node_mapping_;

public:
    /**
     * @brief Construct a MeshGcoreAccessor from MeshGcoreAccessorArgs
     *
     * Extracts mesh and grid configuration from compile-time arguments.
     *
     * @tparam CTA_OFFSET Offset of compile-time arguments
     * @param args MeshGcoreAccessorArgs containing mesh/grid CTA arg offsets
     */
    template <std::size_t CTA_OFFSET>
    MeshGcoreAccessor(const MeshGcoreAccessorArgs<CTA_OFFSET>& args) :
        mesh_config_(), grid_config_(), fabric_node_mapping_() {
        // Populate mesh config from args (runtime initialization)
        args.populate_mesh_config(mesh_config_);
        compute_strides(mesh_config_);

        // Populate grid config from args (runtime initialization)
        args.populate_grid_config(grid_config_);
        compute_strides(grid_config_);

        // Populate fabric node mapping from args (runtime initialization)
        args.populate_fabric_node_mapping(fabric_node_mapping_);
    }

    // ==================== Methods ====================

    /**
     * @brief Compute strides from configuration dimensions
     * Strides are computed in row-major order (last dimension has stride 1)
     *
     * @tparam ConfigT Configuration type with num_dims, dim[], and strides[]
     * @param config Configuration to compute strides for (modified in place)
     */
    template <typename ConfigT>
    static void compute_strides(ConfigT& config) {
        uint32_t stride = 1;
        for (int i = config.num_dims - 1; i >= 0; --i) {
            config.strides[i] = stride;
            stride *= config.dim[i];
        }
    }

    /**
     * @brief Get grid_id and bank_id from global core coordinate
     *
     * Formula:
     *   grid_id = sum((gcore_coord[i] / grid.dim[i]) * mesh_strides[i])
     *   bank_id = sum((gcore_coord[i] % grid.dim[i]) * grid_strides[i])
     *
     * @tparam ArrType Array-like type with operator[] (e.g., array, span)
     * @param gcore_coord Global core coordinate
     * @return GridBankMapping with grid_id and bank_id
     */
    template <typename ArrType>
    FORCE_INLINE GridBankMapping get_grid_and_bank_id(const ArrType& gcore_coord) const {
        size_t grid_id = 0;
        size_t bank_id = 0;

        // Dimensions should match (host side adjusts ranks before passing compile-time args)
        uint32_t num_dims = mesh_config_.num_dims;
        ASSERT(num_dims == grid_config_.num_dims);

        for (size_t i = 0; i < num_dims; ++i) {
            grid_id += (gcore_coord[i] / grid_config_.dim[i]) * mesh_config_.strides[i];
            bank_id += (gcore_coord[i] % grid_config_.dim[i]) * grid_config_.strides[i];
        }

        return GridBankMapping{grid_id, bank_id};
    }

    /**
     * @brief Get NOC address with fabric node info from global core coordinate for L1 access
     *
     * **WARNING: This method can ONLY be called from kernel/firmware code!**
     *
     * @tparam ArrType Array-like type with operator[]
     * @param gcore_coord Global core coordinate
     * @param offset Additional offset within the bank
     * @param noc NOC index to use (0 or 1)
     * @return NocAddrWithFabricNode containing NOC address and fabric node IDs
     */
    template <typename ArrType>
    FORCE_INLINE NocAddrWithFabricNode
    get_fabric_node_and_noc_addr(const ArrType& gcore_coord, uint32_t offset = 0, uint8_t noc = noc_index) const {
        GridBankMapping mapping = get_grid_and_bank_id(gcore_coord);

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        // Get NOC address from bank_id using hardware address generation (L1 access)
        uint64_t noc_addr = get_noc_addr_from_bank_id<false>(static_cast<uint32_t>(mapping.bank_id), offset, noc);

        // Get fabric node IDs for the grid
        const auto& fabric_node = fabric_node_mapping_.fabric_node_ids[mapping.grid_id];

        return NocAddrWithFabricNode{noc_addr, fabric_node.first, fabric_node.second};
#else
        // Compile-time error: this method cannot be called from host code
        static_assert(false, "get_noc_addr() can only be called from kernel/firmware code (KERNEL_BUILD or FW_BUILD)");
        return NocAddrWithFabricNode{0, 0, 0};
#endif
    }

    // ==================== Getters ====================

    const MeshConfig& mesh_config() const { return mesh_config_; }
    const GridConfig& grid_config() const { return grid_config_; }
    const GridToFabricNodeMapping& fabric_node_mapping() const { return fabric_node_mapping_; }
};

// ==================== MeshGcoreAccessorArgs ====================

/**
 * @brief Arguments parser for constructing MeshGcoreAccessor from compile-time args
 *
 * This class reads compile-time arguments passed from the host side and provides
 * methods to extract mesh and grid configuration.
 *
 * The compile-time args layout from host side is:
 * 1. mesh_num_dims (1 uint32_t)
 * 2. mesh_dims[mesh_num_dims]
 * 3. mesh_strides[mesh_num_dims]
 * 4. grid_num_dims (1 uint32_t)
 * 5. grid_dims[grid_num_dims]
 * 6. grid_strides[grid_num_dims]
 * 7. num_grids (1 uint32_t)
 * 8. fabric_mesh_ids[num_grids]
 * 9. fabric_chip_ids[num_grids]
 *
 * @tparam CTA_OFFSET Offset where MeshGcoreAccessorArgs start
 */
template <std::size_t CTA_OFFSET>
struct MeshGcoreAccessorArgs {
    static constexpr uint32_t MAX_DIMS = tensor_accessor::MAX_RANK;

    // Offset calculations
    static constexpr uint32_t MeshNumDimsOffset = CTA_OFFSET;

    // Helper to get mesh_num_dims at compile time
    static constexpr uint32_t get_mesh_num_dims() { return get_compile_time_arg_val(MeshNumDimsOffset); }

    static constexpr uint32_t mesh_num_dims = get_mesh_num_dims();
    static constexpr uint32_t MeshDimsOffset = MeshNumDimsOffset + 1;
    static constexpr uint32_t MeshStridesOffset = MeshDimsOffset + mesh_num_dims;
    static constexpr uint32_t GridNumDimsOffset = MeshStridesOffset + mesh_num_dims;

    // Helper to get grid_num_dims at compile time
    static constexpr uint32_t get_grid_num_dims() { return get_compile_time_arg_val(GridNumDimsOffset); }

    static constexpr uint32_t grid_num_dims = get_grid_num_dims();
    static constexpr uint32_t GridDimsOffset = GridNumDimsOffset + 1;
    static constexpr uint32_t GridStridesOffset = GridDimsOffset + grid_num_dims;
    static constexpr uint32_t NumGridsOffset = GridStridesOffset + grid_num_dims;

    // Helper to get num_grids at compile time
    static constexpr uint32_t get_num_grids() { return get_compile_time_arg_val(NumGridsOffset); }

    static constexpr uint32_t num_grids = get_num_grids();
    static constexpr uint32_t FabricMeshIdsOffset = NumGridsOffset + 1;
    static constexpr uint32_t FabricChipIdsOffset = FabricMeshIdsOffset + num_grids;

    /**
     * @brief Populate MeshConfig from compile-time args (runtime function)
     */
    template <typename MeshConfigT>
    static void populate_mesh_config(MeshConfigT& config) {
        config.num_dims = mesh_num_dims;
        // Use runtime loop to read compile-time args array with runtime index
        for (uint32_t i = 0; i < mesh_num_dims; ++i) {
            config.dim[i] = kernel_compile_time_args[MeshDimsOffset + i];
            config.strides[i] = kernel_compile_time_args[MeshStridesOffset + i];
        }
    }

    /**
     * @brief Populate GridConfig from compile-time args (runtime function)
     */
    template <typename GridConfigT>
    static void populate_grid_config(GridConfigT& config) {
        config.num_dims = grid_num_dims;
        // Use runtime loop to read compile-time args array with runtime index
        for (uint32_t i = 0; i < grid_num_dims; ++i) {
            config.dim[i] = kernel_compile_time_args[GridDimsOffset + i];
            config.strides[i] = kernel_compile_time_args[GridStridesOffset + i];
        }
    }

    /**
     * @brief Populate GridToFabricNodeMapping from compile-time args (runtime function)
     */
    template <typename MappingT>
    static void populate_fabric_node_mapping(MappingT& mapping) {
        mapping.num_grids = num_grids;
        // Use runtime loop to read compile-time args array with runtime index
        for (uint32_t i = 0; i < num_grids; ++i) {
            uint32_t mesh_id = kernel_compile_time_args[FabricMeshIdsOffset + i];
            uint32_t chip_id = kernel_compile_time_args[FabricChipIdsOffset + i];
            mapping.fabric_node_ids[i] = std::make_pair(mesh_id, chip_id);
        }
    }

    /**
     * @brief Total number of compile-time args for MeshGcoreAccessor
     */
    static constexpr uint32_t num_compile_time_args() { return FabricChipIdsOffset + num_grids - CTA_OFFSET; }

    /**
     * @brief Next available offset after MeshGcoreAccessor args
     */
    static constexpr uint32_t next_compile_time_args_offset() { return CTA_OFFSET + num_compile_time_args(); }
};
