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
 *   auto args = MeshGcoreAccessorArgs();
 *   MeshGcoreAccessor accessor(args);
 *   uint32_t gcore[3] = {1, 5, 10};
 *   auto result = accessor.get_fabric_node_and_noc_addr(gcore);
 * @endcode
 *
 * **Note:** Configuration is read from preprocessor defines as constexpr arrays (MESH_DIMS, GRID_DIMS, etc.)
 * provided at compile time. When mesh and grid have different ranks, the host side automatically
 * adjusts them to the same rank by prepending 1s to the shorter shape before generating defines.
 * Example: mesh (2,2,2) + grid (16,16) → grid becomes (1,16,16)
 */
struct MeshGcoreAccessor {
    static constexpr uint32_t MAX_DIMS = tensor_accessor::MAX_RANK;

    /**
     * @brief Configuration for mesh topology
     */
    struct MeshConfig {
        uint32_t num_dims;
        uint32_t dim[MAX_DIMS];      // mesh dimensions (e.g., 2x2x2)
        uint32_t strides[MAX_DIMS];  // pre-computed mesh strides for indexing
        // No constructor - trivially default constructible (uninitialized)
    };

    /**
     * @brief Configuration for grid topology (cores per device)
     */
    struct GridConfig {
        uint32_t num_dims;
        uint32_t dim[MAX_DIMS];      // grid dimensions (e.g., 16,16)
        uint32_t strides[MAX_DIMS];  // pre-computed grid strides for indexing
        // No constructor - trivially default constructible (uninitialized)
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
        uint32_t fabric_mesh_ids[MAX_DIMS];  // mesh_id for each grid
        uint32_t fabric_chip_ids[MAX_DIMS];  // chip_id for each grid
        // No constructor - trivially default constructible (uninitialized)
    };

    /**
     * @brief Worker core to NOC coordinate mapping
     * Maps linearized worker core index to NOC (x, y) coordinates
     * Note: Actual size is determined by NUM_WORKER_CORES compile-time define
     */
    struct WorkerCoreNocMapping {
        uint32_t num_cores;
        uint8_t noc_x[NUM_WORKER_CORES];  // NOC x coordinate for each core
        uint8_t noc_y[NUM_WORKER_CORES];  // NOC y coordinate for each core
        // No constructor - trivially default constructible (uninitialized)
    };

private:
    MeshConfig mesh_config_;
    GridConfig grid_config_;
    GridToFabricNodeMapping fabric_node_mapping_;
    WorkerCoreNocMapping worker_core_noc_mapping_;

public:
    /**
     * @brief Default constructor - trivially default constructible
     *
     * Must call init() before use. No member initialization to avoid
     * generating .init_array entries (bare-metal compatible).
     */
    MeshGcoreAccessor() = default;

    /**
     * @brief Initialize accessor from MeshGcoreAccessorArgs
     *
     * @param args MeshGcoreAccessorArgs containing preprocessor define readers
     */
    void init(const MeshGcoreAccessorArgs& args);

    /**
     * @brief Construct a MeshGcoreAccessor from MeshGcoreAccessorArgs
     *
     * @param args MeshGcoreAccessorArgs containing preprocessor define readers
     */
    MeshGcoreAccessor(const MeshGcoreAccessorArgs& args);

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
        // Get NOC coordinates from worker core mapping (bank_id is the linearized core index)
        uint32_t core_idx = static_cast<uint32_t>(mapping.bank_id);
        uint32_t noc_x = worker_core_noc_mapping_.noc_x[core_idx];
        uint32_t noc_y = worker_core_noc_mapping_.noc_y[core_idx];

        // Build NOC address using get_noc_addr which handles coordinate transformations
        uint64_t noc_addr = get_noc_addr(noc_x, noc_y, offset, noc);

        // Get fabric node IDs for the grid
        uint32_t fabric_mesh_id = fabric_node_mapping_.fabric_mesh_ids[mapping.grid_id];
        uint32_t fabric_chip_id = fabric_node_mapping_.fabric_chip_ids[mapping.grid_id];

        return NocAddrWithFabricNode{noc_addr, fabric_mesh_id, fabric_chip_id};
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
 * @brief Arguments parser for constructing MeshGcoreAccessor from preprocessor defines
 *
 * This class reads configuration from preprocessor defines provided at compile time.
 * The CTA_OFFSET template parameter is kept for API compatibility but is not used.
 *
 * The defines expected from host side are constexpr arrays:
 * 1. MESH_NUM_DIMS - number of mesh dimensions
 * 2. MESH_DIMS - constexpr array of mesh dimensions {dim0, dim1, ...}
 * 3. MESH_STRIDES - constexpr array of mesh strides {stride0, stride1, ...}
 * 4. GRID_NUM_DIMS - number of grid dimensions
 * 5. GRID_DIMS - constexpr array of grid dimensions {dim0, dim1, ...}
 * 6. GRID_STRIDES - constexpr array of grid strides {stride0, stride1, ...}
 * 7. NUM_GRIDS - number of grids
 * 8. FABRIC_MESH_IDS - constexpr array of fabric mesh IDs {id0, id1, ...}
 * 9. FABRIC_CHIP_IDS - constexpr array of fabric chip IDs {id0, id1, ...}
 *
 */
struct MeshGcoreAccessorArgs {
    static constexpr uint32_t MAX_DIMS = tensor_accessor::MAX_RANK;

    /**
     * @brief Populate MeshConfig from preprocessor defines
     */
    template <typename MeshConfigT>
    static void populate_mesh_config(MeshConfigT& config) {
#ifndef MESH_NUM_DIMS
        static_assert(false, "MESH_NUM_DIMS must be defined");
#endif
#ifndef MESH_DIMS
        static_assert(false, "MESH_DIMS must be defined");
#endif
#ifndef MESH_STRIDES
        static_assert(false, "MESH_STRIDES must be defined");
#endif

        config.num_dims = MESH_NUM_DIMS;

        constexpr uint32_t mesh_dims[] = MESH_DIMS;
        for (uint32_t i = 0; i < MESH_NUM_DIMS; ++i) {
            config.dim[i] = mesh_dims[i];
        }

        constexpr uint32_t mesh_strides[] = MESH_STRIDES;
        for (uint32_t i = 0; i < MESH_NUM_DIMS; ++i) {
            config.strides[i] = mesh_strides[i];
        }
    }

    /**
     * @brief Populate GridConfig from preprocessor defines
     */
    template <typename GridConfigT>
    static void populate_grid_config(GridConfigT& config) {
#ifndef GRID_NUM_DIMS
        static_assert(false, "GRID_NUM_DIMS must be defined");
#endif
#ifndef GRID_DIMS
        static_assert(false, "GRID_DIMS must be defined");
#endif
#ifndef GRID_STRIDES
        static_assert(false, "GRID_STRIDES must be defined");
#endif

        config.num_dims = GRID_NUM_DIMS;

        constexpr uint32_t grid_dims[] = GRID_DIMS;
        for (uint32_t i = 0; i < GRID_NUM_DIMS; ++i) {
            config.dim[i] = grid_dims[i];
        }

        constexpr uint32_t grid_strides[] = GRID_STRIDES;
        for (uint32_t i = 0; i < GRID_NUM_DIMS; ++i) {
            config.strides[i] = grid_strides[i];
        }
    }

    /**
     * @brief Populate GridToFabricNodeMapping from preprocessor defines
     */
    template <typename MappingT>
    static void populate_fabric_node_mapping(MappingT& mapping) {
#ifndef NUM_GRIDS
        static_assert(false, "NUM_GRIDS must be defined");
#endif
#ifndef FABRIC_MESH_IDS
        static_assert(false, "FABRIC_MESH_IDS must be defined");
#endif
#ifndef FABRIC_CHIP_IDS
        static_assert(false, "FABRIC_CHIP_IDS must be defined");
#endif

        mapping.num_grids = NUM_GRIDS;

        constexpr uint32_t fabric_mesh_ids_arr[] = FABRIC_MESH_IDS;
        constexpr uint32_t fabric_chip_ids_arr[] = FABRIC_CHIP_IDS;

        for (uint32_t i = 0; i < NUM_GRIDS; ++i) {
            mapping.fabric_mesh_ids[i] = fabric_mesh_ids_arr[i];
            mapping.fabric_chip_ids[i] = fabric_chip_ids_arr[i];
        }
    }

    /**
     * @brief Populate WorkerCoreNocMapping from preprocessor defines
     */
    template <typename MappingT>
    static void populate_worker_core_noc_mapping(MappingT& mapping) {
#ifndef NUM_WORKER_CORES
        static_assert(false, "NUM_WORKER_CORES must be defined");
#endif
#ifndef WORKER_CORE_NOC_X
        static_assert(false, "WORKER_CORE_NOC_X must be defined");
#endif
#ifndef WORKER_CORE_NOC_Y
        static_assert(false, "WORKER_CORE_NOC_Y must be defined");
#endif

        mapping.num_cores = NUM_WORKER_CORES;

        constexpr uint32_t noc_x_arr[] = WORKER_CORE_NOC_X;
        constexpr uint32_t noc_y_arr[] = WORKER_CORE_NOC_Y;

        for (uint32_t i = 0; i < NUM_WORKER_CORES; ++i) {
            mapping.noc_x[i] = static_cast<uint8_t>(noc_x_arr[i]);
            mapping.noc_y[i] = static_cast<uint8_t>(noc_y_arr[i]);
        }
    }
};

// ==================== MeshGcoreAccessor method definitions ====================
// (defined after MeshGcoreAccessorArgs to avoid incomplete type errors)

inline void MeshGcoreAccessor::init(const MeshGcoreAccessorArgs& args) {
    args.populate_mesh_config(mesh_config_);
    args.populate_grid_config(grid_config_);
    args.populate_fabric_node_mapping(fabric_node_mapping_);
    args.populate_worker_core_noc_mapping(worker_core_noc_mapping_);
}

inline MeshGcoreAccessor::MeshGcoreAccessor(const MeshGcoreAccessorArgs& args) :
    mesh_config_(), grid_config_(), fabric_node_mapping_(), worker_core_noc_mapping_() {
    init(args);
}
