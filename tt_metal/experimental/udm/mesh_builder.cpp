// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/udm/mesh_builder.hpp"
#include "tt_metal/experimental/udm/mesh_utils.hpp"
#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <unordered_map>

namespace tt::tt_metal::experimental::udm {

class MeshBuilder::Impl {
public:
    Impl(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const tt::tt_metal::distributed::MeshShape& mesh_shape,
        const std::vector<tt::tt_metal::distributed::MeshCoordinate>& mesh_coords,
        std::optional<std::pair<uint32_t, uint32_t>> grid_shape_override = std::nullopt) :
        mesh_device_(mesh_device),
        mesh_shape_(mesh_shape),
        mesh_coords_(mesh_coords),
        grid_shape_override_(grid_shape_override) {
        log_debug(
            tt::LogOp,
            "MeshBuilder::Impl constructor - mesh_shape: {}, num_coords: {}",
            mesh_shape_,
            mesh_coords_.size());

        // Create mesh object with the mesh_shape passed in as its dims
        create_mesh();
        log_debug(tt::LogOp, "  Created mesh with dims: {}", mesh_.shape());

        // Create grids for each mesh coordinate
        create_grids();
        if (!grids_.empty()) {
            log_debug(tt::LogOp, "  Created {} grids with shape: {}", grids_.size(), grids_[0].shape());
        }

        // Build grid lookup maps
        build_grid_to_fabric_node_id_map();
        log_debug(tt::LogOp, "  Built grid fabric node id mapping");

        // Create flattened mesh shape by combining grid and mesh dimensions
        create_flattened_mesh();
        log_debug(tt::LogOp, "  Created flattened mesh with dims: {}", flattened_mesh_.shape());

        // Create flattened grid shape aligned to same rank as flattened mesh
        create_flattened_grid();
        log_debug(tt::LogOp, "  Created flattened grid with dims: {}", flattened_grid_);

        // Create gcores using flattened mesh
        create_gcores();
        log_debug(tt::LogOp, "  Created {} gcores total", gcores_.size());

        // Build lookup maps for gcores
        build_gcore_maps();
        log_debug(tt::LogOp, "  Built gcore lookup maps");
    }

    // Getters
    const Mesh& get_mesh() const { return mesh_; }

    const Mesh& get_flattened_mesh() const { return flattened_mesh_; }

    const Shape& get_flattened_grid() const { return flattened_grid_; }

    const std::vector<Grid>& get_all_grids_in_mesh() const { return grids_; }

    const std::vector<GlobalCore>& get_all_gcores_in_grid(const Grid& grid) const {
        auto it = grid_to_gcores_.find(grid.id);
        TT_FATAL(it != grid_to_gcores_.end(), "Grid id {} not found", grid.id);
        return it->second;
    }

    const std::vector<GlobalCore>& get_all_gcores_in_mesh() const { return gcores_; }

    const Grid& get_grid_from_coord(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
        for (const auto& grid : grids_) {
            if (grid.coord == coord) {
                return grid;
            }
        }
        TT_FATAL(false, "Mesh coordinate not found: {}", coord);
    }

    const GlobalCore& get_gcore_with_local_coord(
        const tt::tt_metal::distributed::MeshCoordinate& grid_coord,
        const tt::tt_metal::distributed::MeshCoordinate& gcore_local_coord) const {
        const auto& grid = get_grid_from_coord(grid_coord);
        auto it = local_coord_to_gcore_.find({grid.id, gcore_local_coord});
        TT_FATAL(it != local_coord_to_gcore_.end(), "Local gcore coordinate not found");
        return gcores_[it->second];
    }

    const GlobalCore& get_gcore_with_global_coord(
        const tt::tt_metal::distributed::MeshCoordinate& gcore_global_coord) const {
        for (const auto& gcore : gcores_) {
            if (gcore.global_coord == gcore_global_coord) {
                return gcore;
            }
        }
        TT_FATAL(false, "Global gcore coordinate not found: {}", gcore_global_coord);
    }

    std::unordered_map<uint32_t, tt::tt_metal::CoreRangeSet> get_grid_core_range_set_from_gcores(
        const std::vector<GlobalCore>& gcores) const {
        std::unordered_map<uint32_t, std::vector<tt::tt_metal::CoreCoord>> grid_to_core_coords;

        // Group gcores by grid using precomputed map
        for (const auto& gcore : gcores) {
            auto grid_it = gcore_global_id_to_grid_id_.find(gcore.global_id);
            TT_FATAL(
                grid_it != gcore_global_id_to_grid_id_.end(),
                "GlobalCore with global_id {} not found in grid mapping",
                gcore.global_id);
            uint32_t grid_id = grid_it->second;

            // Convert gcore to CoreCoord
            tt::tt_metal::CoreCoord core_coord = gcore.to_core_coord();

            grid_to_core_coords[grid_id].push_back(core_coord);
        }

        // Convert vectors of CoreCoords to CoreRangeSets
        std::unordered_map<uint32_t, tt::tt_metal::CoreRangeSet> result;
        for (const auto& [grid_id, core_coords] : grid_to_core_coords) {
            result[grid_id] = tt::tt_metal::CoreRangeSet(tt::stl::Span<const tt::tt_metal::CoreCoord>(core_coords));
        }

        return result;
    }

    // Public accessor for mesh_device
    tt::tt_metal::distributed::MeshDevice* get_mesh_device() const { return mesh_device_; }

    struct MeshGridDimensions {
        uint32_t mesh_num_dims;
        std::array<uint32_t, MAX_RANK> mesh_dims;
        std::array<uint32_t, MAX_RANK> mesh_strides;
        uint32_t grid_num_dims;
        std::array<uint32_t, MAX_RANK> grid_dims;
        std::array<uint32_t, MAX_RANK> grid_strides;
    };

    MeshGridDimensions get_aligned_mesh_grid_dimensions() const {
        MeshGridDimensions result{};

        // Extract mesh dimensions
        result.mesh_num_dims = mesh_.rank();
        for (uint32_t i = 0; i < result.mesh_num_dims; ++i) {
            result.mesh_dims[i] = mesh_[i];
        }

        // Extract grid dimensions from first grid
        TT_FATAL(!grids_.empty(), "No grids in mesh");
        const auto& grid_shape = grids_[0].shape();
        result.grid_num_dims = grid_shape.rank();
        for (uint32_t i = 0; i < result.grid_num_dims; ++i) {
            result.grid_dims[i] = grid_shape[i];
        }

        // Align ranks for kernel args (prepend 1s for mesh, grid)
        adjust_shape_ranks(result.mesh_dims, result.mesh_num_dims, result.grid_dims, result.grid_num_dims, 1);

        // Compute strides after alignment (only once)
        compute_strides(result.mesh_dims, result.mesh_num_dims, result.mesh_strides);
        compute_strides(result.grid_dims, result.grid_num_dims, result.grid_strides);

        return result;
    }

    std::vector<uint32_t> get_compile_time_args() const {
        std::vector<uint32_t> compile_time_args;

        // MeshGlobalCoreAccessor args layout:
        // 1. mesh_num_dims
        // 2. mesh_dims[mesh_num_dims]
        // 3. mesh_strides[mesh_num_dims]
        // 4. grid_num_dims
        // 5. grid_dims[grid_num_dims]
        // 6. grid_strides[grid_num_dims]
        // 7. num_grids
        // 8. fabric_mesh_ids[num_grids]
        // 9. fabric_chip_ids[num_grids]

        auto dims = get_aligned_mesh_grid_dimensions();

        compile_time_args.push_back(dims.mesh_num_dims);
        for (uint32_t i = 0; i < dims.mesh_num_dims; ++i) {
            compile_time_args.push_back(dims.mesh_dims[i]);
        }
        for (uint32_t i = 0; i < dims.mesh_num_dims; ++i) {
            compile_time_args.push_back(dims.mesh_strides[i]);
        }

        compile_time_args.push_back(dims.grid_num_dims);
        for (uint32_t i = 0; i < dims.grid_num_dims; ++i) {
            compile_time_args.push_back(dims.grid_dims[i]);
        }
        for (uint32_t i = 0; i < dims.grid_num_dims; ++i) {
            compile_time_args.push_back(dims.grid_strides[i]);
        }

        // Add fabric node id mapping
        compile_time_args.push_back(static_cast<uint32_t>(grids_.size()));
        for (size_t i = 0; i < grids_.size(); ++i) {
            compile_time_args.push_back(*grid_to_fabric_node_id_[i].mesh_id);
        }
        for (size_t i = 0; i < grids_.size(); ++i) {
            compile_time_args.push_back(grid_to_fabric_node_id_[i].chip_id);
        }

        return compile_time_args;
    }

    std::vector<uint32_t> get_fabric_nodes_compile_args() const {
        std::vector<uint32_t> compile_time_args;

        // Fabric node args layout:
        // 1. num_grids
        // 2. fabric_mesh_ids[num_grids]
        // 3. fabric_chip_ids[num_grids]

        compile_time_args.push_back(static_cast<uint32_t>(grids_.size()));
        for (size_t i = 0; i < grids_.size(); ++i) {
            compile_time_args.push_back(*grid_to_fabric_node_id_[i].mesh_id);
        }
        for (size_t i = 0; i < grids_.size(); ++i) {
            compile_time_args.push_back(grid_to_fabric_node_id_[i].chip_id);
        }

        return compile_time_args;
    }

    std::map<std::string, std::string> get_compile_time_defines() const {
        std::map<std::string, std::string> defines;

        // MeshGlobalCoreAccessor defines layout using constexpr arrays:
        // 1. MESH_NUM_DIMS
        // 2. MESH_DIMS - constexpr array {dim0, dim1, ...}
        // 3. MESH_STRIDES - constexpr array {stride0, stride1, ...}
        // 4. GRID_NUM_DIMS
        // 5. GRID_DIMS - constexpr array {dim0, dim1, ...}
        // 6. GRID_STRIDES - constexpr array {stride0, stride1, ...}
        // 7. NUM_GRIDS
        // 8. FABRIC_MESH_IDS - constexpr array {id0, id1, ...}
        // 9. FABRIC_CHIP_IDS - constexpr array {id0, id1, ...}

        auto dims = get_aligned_mesh_grid_dimensions();

        // Mesh configuration
        defines["MESH_NUM_DIMS"] = std::to_string(dims.mesh_num_dims);

        // Build MESH_DIMS array string: {dim0, dim1, ...}
        std::string mesh_dims_str = "{";
        for (uint32_t i = 0; i < dims.mesh_num_dims; ++i) {
            if (i > 0) {
                mesh_dims_str += ", ";
            }
            mesh_dims_str += std::to_string(dims.mesh_dims[i]);
        }
        mesh_dims_str += "}";
        defines["MESH_DIMS"] = mesh_dims_str;

        // Build MESH_STRIDES array string: {stride0, stride1, ...}
        std::string mesh_strides_str = "{";
        for (uint32_t i = 0; i < dims.mesh_num_dims; ++i) {
            if (i > 0) {
                mesh_strides_str += ", ";
            }
            mesh_strides_str += std::to_string(dims.mesh_strides[i]);
        }
        mesh_strides_str += "}";
        defines["MESH_STRIDES"] = mesh_strides_str;

        // Grid configuration
        defines["GRID_NUM_DIMS"] = std::to_string(dims.grid_num_dims);

        // Build GRID_DIMS array string: {dim0, dim1, ...}
        std::string grid_dims_str = "{";
        for (uint32_t i = 0; i < dims.grid_num_dims; ++i) {
            if (i > 0) {
                grid_dims_str += ", ";
            }
            grid_dims_str += std::to_string(dims.grid_dims[i]);
        }
        grid_dims_str += "}";
        defines["GRID_DIMS"] = grid_dims_str;

        // Build GRID_STRIDES array string: {stride0, stride1, ...}
        std::string grid_strides_str = "{";
        for (uint32_t i = 0; i < dims.grid_num_dims; ++i) {
            if (i > 0) {
                grid_strides_str += ", ";
            }
            grid_strides_str += std::to_string(dims.grid_strides[i]);
        }
        grid_strides_str += "}";
        defines["GRID_STRIDES"] = grid_strides_str;

        // Fabric node id mapping
        defines["NUM_GRIDS"] = std::to_string(grids_.size());

        // Build FABRIC_MESH_IDS array string: {id0, id1, ...}
        std::string fabric_mesh_ids_str = "{";
        for (size_t i = 0; i < grids_.size(); ++i) {
            if (i > 0) {
                fabric_mesh_ids_str += ", ";
            }
            fabric_mesh_ids_str += std::to_string(*grid_to_fabric_node_id_[i].mesh_id);
        }
        fabric_mesh_ids_str += "}";
        defines["FABRIC_MESH_IDS"] = fabric_mesh_ids_str;

        // Build FABRIC_CHIP_IDS array string: {id0, id1, ...}
        std::string fabric_chip_ids_str = "{";
        for (size_t i = 0; i < grids_.size(); ++i) {
            if (i > 0) {
                fabric_chip_ids_str += ", ";
            }
            fabric_chip_ids_str += std::to_string(grid_to_fabric_node_id_[i].chip_id);
        }
        fabric_chip_ids_str += "}";
        defines["FABRIC_CHIP_IDS"] = fabric_chip_ids_str;

        // Worker core to NOC coordinate mapping
        // This maps linearized worker core index (row-major) to packed NOC (x,y) coordinates
        // The mapping is the same for all devices in the mesh
        // Get any device to query the core mapping (all devices have the same mapping)
        auto* device = mesh_device_->get_devices()[0];

        // Compute total number of cores in grid
        uint32_t num_cores = 1;
        for (uint32_t i = 0; i < dims.grid_num_dims; ++i) {
            num_cores *= dims.grid_dims[i];
        }
        defines["NUM_WORKER_CORES"] = std::to_string(num_cores);

        // Build WORKER_CORE_NOC_X and WORKER_CORE_NOC_Y arrays
        // For the accessor: bank_id = sum((coord[i] % grid.dim[i]) * grid_strides[i])
        // With grid_strides computed row-major (last dim has stride 1):
        //   For 2D with dims [height, width]: bank_id = y * width + x
        // So to reverse: x = bank_id % width, y = bank_id / width
        std::string noc_x_str = "{";
        std::string noc_y_str = "{";

        // Use dims.grid_dims which matches the accessor's linearization
        // Row-major linearization: last dim varies fastest
        // For worker cores, we map to 2D (x, y) where x is the last dim and y is all others fused
        uint32_t x_dim = dims.grid_dims[dims.grid_num_dims - 1];  // Last dim = x

        for (uint32_t idx = 0; idx < num_cores; ++idx) {
            // Reverse the row-major linearization to get (x, y)
            // x is the last dim, y is all other dims fused
            uint32_t x = idx % x_dim;
            uint32_t y = idx / x_dim;

            // Get physical NOC coordinates from device
            CoreCoord logical_core(x, y);
            CoreCoord physical_core = device->worker_core_from_logical_core(logical_core);

            if (idx > 0) {
                noc_x_str += ", ";
                noc_y_str += ", ";
            }
            noc_x_str += std::to_string(physical_core.x);
            noc_y_str += std::to_string(physical_core.y);
        }
        noc_x_str += "}";
        noc_y_str += "}";
        defines["WORKER_CORE_NOC_X"] = noc_x_str;
        defines["WORKER_CORE_NOC_Y"] = noc_y_str;

        return defines;
    }

private:
    void create_mesh() {
        // Create mesh object with the mesh_shape passed in as its dims
        std::vector<uint32_t> dims;
        dims.reserve(mesh_shape_.dims());
        for (size_t i = 0; i < mesh_shape_.dims(); ++i) {
            dims.push_back(mesh_shape_[i]);
        }
        mesh_.dims = Shape(std::move(dims));
    }

    void create_flattened_mesh() {
        // Flatten grid dimensions with mesh dimensions to form expanded mesh shape
        // The flattened mesh combines grid[i] with mesh[mesh_rank - grid_rank + i]
        // For example, if grid is [Y, X] and mesh is [M0, M1]:
        //   flattened_mesh = [Y * M0, X * M1] (when mesh_rank == grid_rank == 2)
        // Or if grid is [Y, X] and mesh is [M0, M1, M2]:
        //   flattened_mesh = [M0, Y * M1, X * M2] (when mesh_rank == 3, grid_rank == 2)

        TT_FATAL(!grids_.empty(), "Grids must be created before flattened mesh");

        const auto& grid_shape = grids_[0].shape();
        uint32_t mesh_rank = mesh_.rank();
        uint32_t grid_rank = grids_[0].rank();

        // Determine the rank of the flattened mesh
        uint32_t flattened_rank = std::max(mesh_rank, grid_rank);

        // Initialize flattened mesh dimensions with mesh dimensions aligned to the trailing positions
        // Prepend 1s at the beginning if mesh_rank < flattened_rank
        std::vector<uint32_t> flattened_dims(flattened_rank);
        for (uint32_t i = 0; i < flattened_rank; ++i) {
            if (i < flattened_rank - mesh_rank) {
                // Prepend 1s at the beginning
                flattened_dims[i] = 1;
            } else {
                // Copy mesh dims to trailing positions
                flattened_dims[i] = mesh_[i - (flattened_rank - mesh_rank)];
            }
        }

        // Multiply the last grid_rank dimensions of flattened_mesh with corresponding grid dimensions
        // This aligns grid dimensions with the trailing dimensions of the mesh
        for (uint32_t i = 0; i < grid_rank; ++i) {
            uint32_t flattened_idx = flattened_rank - grid_rank + i;
            flattened_dims[flattened_idx] *= grid_shape[i];
        }

        flattened_mesh_.dims = Shape(std::move(flattened_dims));
    }

    void create_flattened_grid() {
        // Create flattened_grid_ with same rank as flattened_mesh_
        // Prepend 1s to grid dimensions to align with flattened_mesh_ rank
        TT_FATAL(!grids_.empty(), "Grids must be created before flattened grid");

        const auto& grid_shape = grids_[0].shape();
        uint32_t grid_rank = grid_shape.rank();
        uint32_t flattened_rank = flattened_mesh_.rank();

        std::vector<uint32_t> flattened_grid_dims(flattened_rank);
        for (uint32_t i = 0; i < flattened_rank; ++i) {
            if (i < flattened_rank - grid_rank) {
                flattened_grid_dims[i] = 1;
            } else {
                flattened_grid_dims[i] = grid_shape[i - (flattened_rank - grid_rank)];
            }
        }

        flattened_grid_ = Shape(std::move(flattened_grid_dims));
    }

    void create_grids() {
        // Get grid size - use override if provided, otherwise use full compute grid
        std::vector<uint32_t> grid_dims;
        if (grid_shape_override_.has_value()) {
            // Use the provided grid shape override {height, width}
            grid_dims = {grid_shape_override_->first, grid_shape_override_->second};
            log_debug(
                tt::LogOp, "MeshBuilder::create_grids - using override grid shape: {}x{}", grid_dims[0], grid_dims[1]);
        } else {
            // Use full compute grid from mesh device
            auto compute_grid_size = mesh_device_->compute_with_storage_grid_size();
            grid_dims = {static_cast<uint32_t>(compute_grid_size.y), static_cast<uint32_t>(compute_grid_size.x)};
            log_debug(
                tt::LogOp, "MeshBuilder::create_grids - using full compute grid: {}x{}", grid_dims[0], grid_dims[1]);
        }

        grids_.reserve(mesh_coords_.size());

        for (size_t i = 0; i < mesh_coords_.size(); ++i) {
            Grid grid(i, Shape(grid_dims), mesh_coords_[i]);
            grids_.push_back(grid);
        }
    }

    void build_grid_to_fabric_node_id_map() {
        grid_to_fabric_node_id_.reserve(grids_.size());
        for (const auto& grid : grids_) {
            auto fabric_node_id = mesh_device_->get_fabric_node_id(grid.coord);
            grid_to_fabric_node_id_.push_back(fabric_node_id);
        }
    }

    tt::tt_metal::distributed::MeshCoordinate compute_global_coord_from_flat_index(uint32_t flat_index) const {
        const auto& flattened_shape = flattened_mesh_.shape();
        uint32_t flattened_rank = flattened_shape.rank();

        std::vector<uint32_t> global_coord_vals(flattened_rank);
        uint32_t idx = flat_index;
        for (int dim = flattened_rank - 1; dim >= 0; --dim) {
            global_coord_vals[dim] = idx % flattened_shape[dim];
            idx /= flattened_shape[dim];
        }

        return tt::tt_metal::distributed::MeshCoordinate(
            tt::stl::Span<const uint32_t>(global_coord_vals.data(), global_coord_vals.size()));
    }

    tt::tt_metal::distributed::MeshCoordinate compute_local_coord_from_global_coord(
        const tt::tt_metal::distributed::MeshCoordinate& global_coord) const {
        uint32_t grid_rank = grids_[0].rank();
        uint32_t flattened_rank = flattened_mesh_.rank();

        // Extract local coordinate from the last grid_rank dimensions of global_coord
        std::vector<uint32_t> local_coord_vals(grid_rank);
        for (uint32_t i = 0; i < grid_rank; ++i) {
            uint32_t global_idx = flattened_rank - grid_rank + i;
            local_coord_vals[i] = global_coord[global_idx] % flattened_grid_[global_idx];
        }

        return tt::tt_metal::distributed::MeshCoordinate(
            tt::stl::Span<const uint32_t>(local_coord_vals.data(), local_coord_vals.size()));
    }

    uint32_t compute_local_gcore_id_from_local_coord(
        const tt::tt_metal::distributed::MeshCoordinate& local_coord) const {
        const auto& grid_shape = grids_[0].shape();
        uint32_t grid_rank = grid_shape.rank();

        uint32_t local_gcore_id = 0;
        uint32_t stride = 1;
        for (int i = grid_rank - 1; i >= 0; --i) {
            local_gcore_id += local_coord[i] * stride;
            stride *= grid_shape[i];
        }

        return local_gcore_id;
    }

    void create_gcores() {
        uint32_t total_num_gcores = flattened_mesh_.volume();
        gcores_.reserve(total_num_gcores);

        for (uint32_t global_gcore_id = 0; global_gcore_id < total_num_gcores; ++global_gcore_id) {
            auto global_coord = compute_global_coord_from_flat_index(global_gcore_id);
            auto local_coord = compute_local_coord_from_global_coord(global_coord);
            uint32_t local_gcore_id = compute_local_gcore_id_from_local_coord(local_coord);

            GlobalCore gcore(local_gcore_id, global_gcore_id, local_coord, global_coord);
            gcores_.push_back(gcore);
        }
    }

    uint32_t compute_grid_id_from_global_coord(const tt::tt_metal::distributed::MeshCoordinate& global_coord) const {
        uint32_t flattened_rank = flattened_mesh_.rank();
        uint32_t mesh_rank = mesh_.rank();

        std::vector<uint32_t> grid_coord_vals(mesh_rank, 0);

        // Compute grid coordinate: grid_coord[i] = global_coord[i] / flattened_grid_[i]
        // Align to mesh_rank by taking the last mesh_rank dimensions
        uint32_t offset = flattened_rank - mesh_rank;
        for (uint32_t i = 0; i < mesh_rank; ++i) {
            grid_coord_vals[i] = global_coord[offset + i] / flattened_grid_[offset + i];
        }

        auto grid_coord = tt::tt_metal::distributed::MeshCoordinate(
            tt::stl::Span<const uint32_t>(grid_coord_vals.data(), grid_coord_vals.size()));

        for (const auto& grid : grids_) {
            if (grid.coord == grid_coord) {
                return grid.id;
            }
        }
        TT_FATAL(false, "Grid coordinate not found: {}", grid_coord);
    }

    void build_grid_to_gcores_map() {
        for (const auto& gcore : gcores_) {
            uint32_t grid_id = compute_grid_id_from_global_coord(gcore.global_coord);
            grid_to_gcores_[grid_id].push_back(gcore);
        }
    }

    void build_local_coord_to_gcore_map() {
        for (const auto& gcore : gcores_) {
            uint32_t grid_id = compute_grid_id_from_global_coord(gcore.global_coord);
            local_coord_to_gcore_[{grid_id, gcore.local_coord}] = gcore.global_id;
        }
    }

    void build_gcore_id_to_grid_id_map() {
        for (const auto& gcore : gcores_) {
            uint32_t grid_id = compute_grid_id_from_global_coord(gcore.global_coord);
            gcore_global_id_to_grid_id_[gcore.global_id] = grid_id;
        }
    }

    void build_gcore_maps() {
        build_grid_to_gcores_map();
        build_local_coord_to_gcore_map();
        build_gcore_id_to_grid_id_map();
    }

    // Hash function for coordinate pair in local_coord_to_gcore_ map
    struct CoordPairHash {
        size_t operator()(const std::pair<uint32_t, tt::tt_metal::distributed::MeshCoordinate>& p) const {
            return std::hash<uint32_t>{}(p.first) ^ std::hash<tt::tt_metal::distributed::MeshCoordinate>{}(p.second);
        }
    };

    tt::tt_metal::distributed::MeshDevice* mesh_device_;
    tt::tt_metal::distributed::MeshShape mesh_shape_;
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords_;
    std::optional<std::pair<uint32_t, uint32_t>>
        grid_shape_override_;  // Optional override for grid shape {height, width}

    // Constructed objects
    Mesh mesh_;
    Mesh flattened_mesh_;   // Flattened mesh shape combining grid and mesh dimensions
    Shape flattened_grid_;  // Grid shape aligned to same rank as flattened_mesh_
    std::vector<Grid> grids_;
    std::vector<GlobalCore> gcores_;

    // Lookup maps
    std::unordered_map<uint32_t, std::vector<GlobalCore>> grid_to_gcores_;
    std::unordered_map<std::pair<uint32_t, tt::tt_metal::distributed::MeshCoordinate>, size_t, CoordPairHash>
        local_coord_to_gcore_;

    // Efficient mapping for finding grid_id from gcore global_id
    std::unordered_map<uint32_t, uint32_t> gcore_global_id_to_grid_id_;

    // Grid to fabric node id mapping
    std::vector<tt::tt_fabric::FabricNodeId> grid_to_fabric_node_id_;
};

MeshBuilder::MeshBuilder(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    const std::vector<tt::tt_metal::distributed::MeshCoordinate>& mesh_coords) :
    impl_(std::make_unique<Impl>(mesh_device, mesh_shape, mesh_coords)) {}

MeshBuilder::MeshBuilder(const tt::tt_metal::distributed::MeshBuffer& mesh_buffer) {
    // Get mesh device from the mesh buffer
    auto* mesh_device = mesh_buffer.device();

    // Get mesh shape from the device
    auto mesh_shape = mesh_device->shape();

    // Collect all mesh coordinates in row-major order
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords;
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        mesh_coords.push_back(coord);
    }

    // Extract grid shape from mesh buffer's shard spec if present
    std::optional<std::pair<uint32_t, uint32_t>> grid_shape = std::nullopt;
    const auto& device_local_config = mesh_buffer.device_local_config();
    const auto& shard_spec_opt = device_local_config.sharding_args.shard_spec();
    if (shard_spec_opt.has_value()) {
        auto bbox = shard_spec_opt->grid().bounding_box();
        uint32_t grid_height = bbox.end_coord.y - bbox.start_coord.y + 1;
        uint32_t grid_width = bbox.end_coord.x - bbox.start_coord.x + 1;
        grid_shape = std::make_pair(grid_height, grid_width);
        log_debug(tt::LogOp, "MeshBuilder: extracted grid shape from shard spec: {}x{}", grid_height, grid_width);
    }

    // Create the impl with extracted grid shape
    impl_ = std::make_unique<Impl>(mesh_device, mesh_shape, mesh_coords, grid_shape);
}

MeshBuilder::~MeshBuilder() = default;

MeshBuilder::MeshBuilder(MeshBuilder&&) noexcept = default;
MeshBuilder& MeshBuilder::operator=(MeshBuilder&&) noexcept = default;

const Mesh& MeshBuilder::get_mesh() const { return impl_->get_mesh(); }

const Mesh& MeshBuilder::get_flattened_mesh() const { return impl_->get_flattened_mesh(); }

const Shape& MeshBuilder::get_flattened_grid() const { return impl_->get_flattened_grid(); }

const std::vector<Grid>& MeshBuilder::get_all_grids_in_mesh() const { return impl_->get_all_grids_in_mesh(); }

const std::vector<GlobalCore>& MeshBuilder::get_all_gcores_in_grid(const Grid& grid) const {
    return impl_->get_all_gcores_in_grid(grid);
}

const std::vector<GlobalCore>& MeshBuilder::get_all_gcores_in_mesh() const { return impl_->get_all_gcores_in_mesh(); }

const Grid& MeshBuilder::get_grid_from_coord(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    return impl_->get_grid_from_coord(coord);
}

const GlobalCore& MeshBuilder::get_gcore_with_local_coord(
    const tt::tt_metal::distributed::MeshCoordinate& grid_coord,
    const tt::tt_metal::distributed::MeshCoordinate& gcore_coord) const {
    return impl_->get_gcore_with_local_coord(grid_coord, gcore_coord);
}

const GlobalCore& MeshBuilder::get_gcore_with_global_coord(
    const tt::tt_metal::distributed::MeshCoordinate& gcore_coord) const {
    return impl_->get_gcore_with_global_coord(gcore_coord);
}

std::unordered_map<uint32_t, tt::tt_metal::CoreRangeSet> MeshBuilder::get_grid_core_range_set_from_gcores(
    const std::vector<GlobalCore>& gcores) const {
    return impl_->get_grid_core_range_set_from_gcores(gcores);
}

tt::tt_metal::distributed::MeshDevice* MeshBuilder::mesh_device() const { return impl_->get_mesh_device(); }

std::vector<uint32_t> MeshBuilder::get_compile_time_args() const { return impl_->get_compile_time_args(); }

std::vector<uint32_t> MeshBuilder::get_fabric_nodes_compile_args() const {
    return impl_->get_fabric_nodes_compile_args();
}

std::map<std::string, std::string> MeshBuilder::get_compile_time_defines() const {
    return impl_->get_compile_time_defines();
}

}  // namespace tt::tt_metal::experimental::udm
