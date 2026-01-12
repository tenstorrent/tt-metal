// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include "api/tensor/tensor_accessor.h"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "internal/dataflow/dataflow_api_addrgen.h"
#endif

// Forward declaration
template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
struct MeshTensorAccessorArgs;

/**
 * @brief Mapping of global page coordinate to grid_id and local page_id
 */
struct PageMapping {
    size_t grid_id;        // which device/grid
    size_t local_page_id;  // which page within the device/grid
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
 * @brief Distribution specification for mesh tensor (multi-device tensor)
 * @tparam Rank Compile-time rank of the tensor
 */
template <uint32_t Rank>
struct MeshDSpec {
    static constexpr uint32_t rank = Rank;
    std::array<uint32_t, Rank> mesh_tensor_shape;    // global shape across all devices
    std::array<uint32_t, Rank> mesh_tensor_strides;  // strides for global tensor
    std::array<uint32_t, Rank> tensor_shape;         // local shape on single device
    std::array<uint32_t, Rank> tensor_strides;       // strides for local tensor
    std::array<uint32_t, Rank> mesh_shape;           // mesh device shape (mesh_tensor_shape / tensor_shape)
    std::array<uint32_t, Rank> mesh_strides;         // strides for mesh device space

    constexpr MeshDSpec() :
        mesh_tensor_shape{}, mesh_tensor_strides{}, tensor_shape{}, tensor_strides{}, mesh_shape{}, mesh_strides{} {}
};

/**
 * @brief Configuration for grid to fabric node id mapping
 * @tparam NumGrids Compile-time number of grids
 */
template <uint32_t NumGrids>
struct GridToFabricNodeMapping {
    static constexpr uint32_t num_grids = NumGrids;
    std::array<std::pair<uint32_t, uint32_t>, NumGrids> fabric_node_ids;  // (mesh_id, chip_id) for each grid

    constexpr GridToFabricNodeMapping() : fabric_node_ids{} {}
};

/**
 * @brief Accessor for distributed tensor page access across devices in a mesh
 *
 * MeshTensorAccessor allows access via global page IDs that span across the entire
 * mesh tensor. It converts global page IDs to grid_id and local page_id, enabling
 * seamless access to tensor data distributed across multiple devices.
 *
 * **Process:**
 * 1. Compute mesh_tensor_strides and tensor_strides (done in constructor)
 * 2. Convert global_page_id to global_page_coord:
 *    - Decompose linear ID using mesh_tensor_shape
 * 3. Convert global_page_coord to (grid_id, local_page_id):
 *    - grid_id = sum((coord[i] / tensor_shape[i]) * mesh_tensor_strides[i])
 *    - local_page_id = sum((coord[i] % tensor_shape[i]) * tensor_strides[i])
 * 4. Use local TensorAccessor to get NOC address from local_page_id
 *
 * **Example:**
 * @code
 *   auto args = MeshTensorAccessorArgs<0, 0>();
 *   auto accessor = MeshTensorAccessor(args);  // Type deduced automatically!
 *   uint64_t noc_addr = accessor.get_noc_addr(global_page_id);
 * @endcode
 *
 * @tparam TensorAccessorT The underlying TensorAccessor type (auto-deduced)
 * @tparam Rank Compile-time rank of the tensor
 * @tparam NumGrids Compile-time number of grids in the mesh
 */
template <typename TensorAccessorT, uint32_t Rank, uint32_t NumGrids>
struct MeshTensorAccessor {
    using TensorAccessorType = TensorAccessorT;
    static constexpr uint32_t rank = Rank;
    static constexpr uint32_t num_grids = NumGrids;

private:
    TensorAccessorType tensor_accessor_;
    MeshDSpec<Rank> mesh_dspec_;
    GridToFabricNodeMapping<NumGrids> fabric_node_mapping_;

public:
    /**
     * @brief Construct a MeshTensorAccessor from MeshTensorAccessorArgs
     *
     * Extracts distribution spec and buffer address from compile-time arguments.
     *
     * @tparam CTA_OFFSET Offset of compile-time arguments
     * @tparam CRTA_OFFSET Offset of counted runtime-time arguments
     * @param args MeshTensorAccessorArgs containing all CTA arg offsets
     */
    template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
    MeshTensorAccessor(const MeshTensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>& args) :
        tensor_accessor_(args.get_tensor_accessor_args(), args.get_buffer_address(), args.get_aligned_page_size()),
        mesh_dspec_(),
        fabric_node_mapping_() {
        // Populate mesh dspec from args (runtime initialization)
        args.populate_mesh_dspec(mesh_dspec_);

        // Populate fabric node mapping from args (runtime initialization)
        args.populate_fabric_node_mapping(fabric_node_mapping_);
    }

    /**
     * @brief Get the underlying tensor accessor
     */
    constexpr const TensorAccessorType& tensor_accessor() const { return tensor_accessor_; }
    constexpr uint32_t page_size() const { return tensor_accessor_.page_size; }

    // ==================== Global Page Access Methods ====================

    /**
     * @brief Compute global page coordinate from global page id
     *
     * Converts linear global page ID to multi-dimensional coordinate
     * using mesh tensor shape.
     *
     * @tparam ArrType Array-like type to store coordinate
     * @param global_page_id Linear global page ID
     * @param global_page_coord Output coordinate (modified in place)
     */
    template <typename ArrType>
    FORCE_INLINE void get_global_page_coord(uint32_t global_page_id, ArrType& global_page_coord) const {
        for (int i = Rank - 1; i >= 0; --i) {
            global_page_coord[i] = global_page_id % mesh_dspec_.mesh_tensor_shape[i];
            global_page_id /= mesh_dspec_.mesh_tensor_shape[i];
        }
    }

    /**
     * @brief Get grid_id and local_page_id from global page coordinate
     *
     * Formula:
     *   grid_id = sum((global_page_coord[i] / tensor_shape[i]) * mesh_tensor_strides[i])
     *   local_page_id = sum((global_page_coord[i] % tensor_shape[i]) * tensor_strides[i])
     *
     * @tparam ArrType Array-like type with operator[]
     * @param global_page_coord Global page coordinate
     * @return PageMapping with grid_id and local_page_id
     */
    template <typename ArrType>
    FORCE_INLINE PageMapping get_grid_and_local_page(const ArrType& global_page_coord) const {
        size_t grid_id = 0;
        size_t local_page_id = 0;

        for (size_t i = 0; i < Rank; ++i) {
            // Use mesh_strides (from compile-time args) to map to grid_id correctly
            grid_id += (global_page_coord[i] / mesh_dspec_.tensor_shape[i]) * mesh_dspec_.mesh_strides[i];
            local_page_id += (global_page_coord[i] % mesh_dspec_.tensor_shape[i]) * mesh_dspec_.tensor_strides[i];
        }

        return PageMapping{grid_id, local_page_id};
    }

    /**
     * @brief Get NOC address with fabric node info from global page ID
     *
     * **WARNING: This method can ONLY be called from kernel/firmware code!**
     *
     * @param global_page_id Linear global page ID
     * @param offset Additional offset within the page
     * @param noc NOC index to use (0 or 1)
     * @return NocAddrWithFabricNode containing NOC address and fabric node IDs
     */
    FORCE_INLINE NocAddrWithFabricNode
    get_fabric_node_and_noc_addr(uint32_t global_page_id, uint32_t offset = 0, uint8_t noc = noc_index) const {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        // Convert global page ID to coordinate - use compile-time sized array
        uint32_t global_page_coord[Rank];
        get_global_page_coord(global_page_id, global_page_coord);

        // Get mapping to local page
        PageMapping mapping = get_grid_and_local_page(global_page_coord);

        // Use local tensor accessor to get NOC address for local page
        uint64_t noc_addr = tensor_accessor_.get_noc_addr(static_cast<uint32_t>(mapping.local_page_id), offset, noc);

        // Get fabric node IDs for the grid
        const auto& fabric_node = fabric_node_mapping_.fabric_node_ids[mapping.grid_id];

        return NocAddrWithFabricNode{noc_addr, fabric_node.first, fabric_node.second};
#else
        // Compile-time error: this method cannot be called from host code
        static_assert(false, "get_noc_addr() can only be called from kernel/firmware code (KERNEL_BUILD or FW_BUILD)");
        return NocAddrWithFabricNode{0, 0, 0};
#endif
    }

    /**
     * @brief Get NOC address with fabric node info from global page coordinate
     *
     * **WARNING: This method can ONLY be called from kernel/firmware code!**
     *
     * @tparam ArrType Array-like type with operator[]
     * @param global_page_coord Global page coordinate
     * @param offset Additional offset within the page
     * @param noc NOC index to use (0 or 1)
     * @return NocAddrWithFabricNode containing NOC address and fabric node IDs
     */
    template <typename ArrType>
    FORCE_INLINE NocAddrWithFabricNode
    get_fabric_node_and_noc_addr(const ArrType& global_page_coord, uint32_t offset = 0, uint8_t noc = noc_index) const {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        // Get mapping to local page
        PageMapping mapping = get_grid_and_local_page(global_page_coord);

        // Use local tensor accessor to get NOC address for local page
        uint64_t noc_addr = tensor_accessor_.get_noc_addr(static_cast<uint32_t>(mapping.local_page_id), offset, noc);

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

    const MeshDSpec<Rank>& mesh_dspec() const { return mesh_dspec_; }
    const GridToFabricNodeMapping<NumGrids>& fabric_node_mapping() const { return fabric_node_mapping_; }
};

// ==================== Deduction Guide ====================

/**
 * @brief Deduction guide for MeshTensorAccessor from MeshTensorAccessorArgs
 *
 * Allows the compiler to deduce the complete DSpec from MeshTensorAccessorArgs.
 *
 * Usage (kernel side - no template parameters needed):
 *   auto args = MeshTensorAccessorArgs<0, 0>();
 *   auto accessor = MeshTensorAccessor(args);
 *   // DSpec and buffer address are automatically deduced!
 */
// Deduction guide: auto-deduce TensorAccessor type, Rank, and NumGrids from MeshTensorAccessorArgs
template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
MeshTensorAccessor(const MeshTensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>& args) -> MeshTensorAccessor<
    decltype(TensorAccessor(
        typename MeshTensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::TensorAccessorArgsT(),
        args.get_buffer_address(),
        args.get_aligned_page_size())),
    MeshTensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::mesh_dspec_rank,
    MeshTensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::num_grids>;

// ==================== MeshTensorAccessorArgs ====================

/**
 * @brief Arguments parser for constructing MeshTensorAccessor from compile-time args
 *
 * This class reads compile-time arguments passed from the host side and provides
 * methods to extract buffer address and mesh distribution spec.
 *
 * The compile-time args layout from host side is:
 * 1. TensorAccessorArgs (variable size)
 * 2. buffer_address (1 uint32_t)
 * 3. aligned_page_size (1 uint32_t)
 * 4. mesh_dspec_rank (1 uint32_t)
 * 5. mesh_tensor_shape[mesh_dspec_rank]
 * 6. mesh_tensor_strides[mesh_dspec_rank]
 * 7. tensor_shape[mesh_dspec_rank]
 * 8. tensor_strides[mesh_dspec_rank]
 * 9. mesh_shape[mesh_dspec_rank]
 * 10. mesh_strides[mesh_dspec_rank]
 * 11. num_grids (1 uint32_t)
 * 12. fabric_mesh_ids[num_grids]
 * 13. fabric_chip_ids[num_grids]
 *
 * @tparam TENSOR_ACCESSOR_CTA_OFFSET Offset where TensorAccessorArgs start
 * @tparam TENSOR_ACCESSOR_CRTA_OFFSET Offset where TensorAccessor CRTA args start
 */
template <std::size_t TENSOR_ACCESSOR_CTA_OFFSET, std::size_t TENSOR_ACCESSOR_CRTA_OFFSET>
struct MeshTensorAccessorArgs {
    // TensorAccessorArgs for the underlying local tensor accessor
    using TensorAccessorArgsT = TensorAccessorArgs<TENSOR_ACCESSOR_CTA_OFFSET, TENSOR_ACCESSOR_CRTA_OFFSET>;
    static constexpr uint32_t tensor_accessor_args_size = TensorAccessorArgsT::num_compile_time_args();

    // Offset calculations for MeshTensorAccessor-specific args
    static constexpr uint32_t BufferAddressOffset = TENSOR_ACCESSOR_CTA_OFFSET + tensor_accessor_args_size;
    static constexpr uint32_t AlignedPageSizeOffset = BufferAddressOffset + 1;
    static constexpr uint32_t MeshDSpecRankOffset = AlignedPageSizeOffset + 1;

    // Helper to get mesh_dspec_rank at compile time
    static constexpr uint32_t get_mesh_dspec_rank() { return get_compile_time_arg_val(MeshDSpecRankOffset); }

    static constexpr uint32_t mesh_dspec_rank = get_mesh_dspec_rank();
    static constexpr uint32_t MeshTensorShapeOffset = MeshDSpecRankOffset + 1;
    static constexpr uint32_t MeshTensorStridesOffset = MeshTensorShapeOffset + mesh_dspec_rank;
    static constexpr uint32_t TensorShapeOffset = MeshTensorStridesOffset + mesh_dspec_rank;
    static constexpr uint32_t TensorStridesOffset = TensorShapeOffset + mesh_dspec_rank;
    static constexpr uint32_t MeshShapeOffset = TensorStridesOffset + mesh_dspec_rank;
    static constexpr uint32_t MeshStridesOffset = MeshShapeOffset + mesh_dspec_rank;
    static constexpr uint32_t NumGridsOffset = MeshStridesOffset + mesh_dspec_rank;

    // Helper to get num_grids at compile time
    static constexpr uint32_t get_num_grids() { return get_compile_time_arg_val(NumGridsOffset); }

    static constexpr uint32_t num_grids = get_num_grids();
    static constexpr uint32_t FabricMeshIdsOffset = NumGridsOffset + 1;
    static constexpr uint32_t FabricChipIdsOffset = FabricMeshIdsOffset + num_grids;

    /**
     * @brief Get TensorAccessorArgs for the underlying local tensor accessor
     */
    static constexpr TensorAccessorArgsT get_tensor_accessor_args() { return TensorAccessorArgsT(); }

    /**
     * @brief Get buffer address from compile-time args
     */
    static constexpr uint64_t get_buffer_address() { return get_compile_time_arg_val(BufferAddressOffset); }

    /**
     * @brief Get aligned page size
     */
    static constexpr uint32_t get_aligned_page_size() { return get_compile_time_arg_val(AlignedPageSizeOffset); }

    /**
     * @brief Populate MeshDSpec from compile-time args
     * Uses compile-time rank for loop bounds
     */
    static void populate_mesh_dspec(MeshDSpec<mesh_dspec_rank>& dspec) {
        // Use compile-time loop bounds for exact sizing
        for (uint32_t i = 0; i < mesh_dspec_rank; ++i) {
            dspec.mesh_tensor_shape[i] = kernel_compile_time_args[MeshTensorShapeOffset + i];
            dspec.mesh_tensor_strides[i] = kernel_compile_time_args[MeshTensorStridesOffset + i];
            dspec.tensor_shape[i] = kernel_compile_time_args[TensorShapeOffset + i];
            dspec.tensor_strides[i] = kernel_compile_time_args[TensorStridesOffset + i];
            dspec.mesh_shape[i] = kernel_compile_time_args[MeshShapeOffset + i];
            dspec.mesh_strides[i] = kernel_compile_time_args[MeshStridesOffset + i];
        }
    }

    /**
     * @brief Populate GridToFabricNodeMapping from compile-time args
     * Uses compile-time num_grids for loop bounds
     */
    static void populate_fabric_node_mapping(GridToFabricNodeMapping<num_grids>& mapping) {
        // Use compile-time loop bounds for exact sizing
        for (uint32_t i = 0; i < num_grids; ++i) {
            uint32_t mesh_id = kernel_compile_time_args[FabricMeshIdsOffset + i];
            uint32_t chip_id = kernel_compile_time_args[FabricChipIdsOffset + i];
            mapping.fabric_node_ids[i] = std::make_pair(mesh_id, chip_id);
        }
    }

    /**
     * @brief Total number of compile-time args for MeshTensorAccessor
     */
    static constexpr uint32_t num_compile_time_args() {
        return FabricChipIdsOffset + num_grids - TENSOR_ACCESSOR_CTA_OFFSET;
    }

    /**
     * @brief Next available offset after MeshTensorAccessor args
     */
    static constexpr uint32_t next_compile_time_args_offset() {
        return TENSOR_ACCESSOR_CTA_OFFSET + num_compile_time_args();
    }
};
