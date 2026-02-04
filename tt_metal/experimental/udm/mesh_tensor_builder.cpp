// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/udm/mesh_tensor_builder.hpp"
#include "tt_metal/experimental/udm/mesh_builder.hpp"
#include "tt_metal/experimental/udm/mesh_utils.hpp"
#include "tt_metal/api/tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>
#include <array>

namespace tt::tt_metal::experimental::udm {

// Align mesh and grid ranks by prepending 1s to the shorter shape
class MeshTensorBuilder::Impl {
public:
    explicit Impl(
        const tt::tt_metal::distributed::MeshBuffer& mesh_buffer,
        const tt::tt_metal::Shape& tensor_shape_in_pages,
        const tt::tt_metal::distributed::MeshShape& distribution_shape,
        const std::vector<std::optional<int>>& shard_dims) :
        mesh_buffer_(mesh_buffer), distribution_shape_(distribution_shape), shard_dims_(shard_dims) {
        // tensor_shape_in_pages is already computed externally using TensorLayout
        // Extract and reconstruct mesh tensor shape from local tensor shape (already in pages)
        extract_tensor_shapes(tensor_shape_in_pages);

        accessor_args_ = tt::tt_metal::TensorAccessorArgs(mesh_buffer_);

        // Create mesh builder to get fabric node id mapping
        mesh_builder_ = std::make_unique<MeshBuilder>(mesh_buffer_);
    }

    std::vector<uint32_t> get_compile_time_args() const {
        auto compile_time_args = accessor_args_.get_compile_time_args();

        // MeshTensorAccessor args layout:
        // 1. TensorAccessorArgs (variable size) - already in compile_time_args
        // 2. buffer_address (uint32_t)
        // 3. aligned_page_size
        // 4. mesh_dspec_rank
        // 5. mesh_tensor_shape_in_pages[mesh_dspec_rank]
        // 6. mesh_tensor_strides_in_pages[mesh_dspec_rank]
        // 7. tensor_shape_in_pages[mesh_dspec_rank]
        // 8. tensor_strides_in_pages[mesh_dspec_rank]
        // 9. mesh_shape[mesh_dspec_rank]
        // 10. mesh_strides[mesh_dspec_rank]
        // 11. num_grids
        // 12. fabric_mesh_ids[num_grids]
        // 13. fabric_chip_ids[num_grids]

        compile_time_args.push_back(static_cast<uint32_t>(mesh_buffer_.address()));
        compile_time_args.push_back(mesh_buffer_.get_reference_buffer()->aligned_page_size());

        compile_time_args.push_back(mesh_tensor_rank_);
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(mesh_tensor_shape_in_pages_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(mesh_tensor_strides_in_pages_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(tensor_shape_in_pages_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(tensor_strides_in_pages_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(mesh_shape_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(mesh_strides_[i]);
        }

        // Get reordered fabric node IDs (tensor-space order, not mesh row-major order)
        auto [reordered_mesh_ids, reordered_chip_ids] = get_reordered_fabric_node_ids();

        // Fabric node args layout:
        // 1. num_grids
        // 2. fabric_mesh_ids[num_grids]
        // 3. fabric_chip_ids[num_grids]
        compile_time_args.push_back(static_cast<uint32_t>(reordered_mesh_ids.size()));
        for (uint32_t id : reordered_mesh_ids) {
            compile_time_args.push_back(id);
        }
        for (uint32_t id : reordered_chip_ids) {
            compile_time_args.push_back(id);
        }

        return compile_time_args;
    }

    uint64_t get_buffer_address() const { return mesh_buffer_.address(); }

    uint32_t get_aligned_page_size() const { return mesh_buffer_.get_reference_buffer()->aligned_page_size(); }

    tt::tt_metal::Shape get_mesh_tensor_shape_in_pages() const {
        return tt::tt_metal::Shape(
            tt::stl::Span<const uint32_t>(mesh_tensor_shape_in_pages_.data(), mesh_tensor_rank_));
    }

    tt::tt_metal::Shape get_local_tensor_shape_in_pages() const {
        return tt::tt_metal::Shape(tt::stl::Span<const uint32_t>(tensor_shape_in_pages_.data(), mesh_tensor_rank_));
    }

    tt::tt_metal::Shape get_mesh_shape() const {
        return tt::tt_metal::Shape(tt::stl::Span<const uint32_t>(mesh_shape_.data(), mesh_tensor_rank_));
    }

    const tt::tt_metal::distributed::MeshShape& get_distribution_shape() const { return distribution_shape_; }

    const std::vector<std::optional<int>>& get_shard_dims() const { return shard_dims_; }

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> get_reordered_fabric_node_ids() const {
        auto [orig_mesh_ids, orig_chip_ids] = get_original_fabric_ids();
        auto tensor_to_mesh = build_tensor_to_mesh_mapping();
        auto distribution_strides = compute_distribution_strides();
        uint32_t total_tensor_grids = get_total_tensor_grids();

        std::vector<uint32_t> reordered_mesh_ids(total_tensor_grids);
        std::vector<uint32_t> reordered_chip_ids(total_tensor_grids);

        for (uint32_t tensor_grid_id = 0; tensor_grid_id < total_tensor_grids; ++tensor_grid_id) {
            uint32_t mesh_grid_id =
                tensor_grid_id_to_mesh_grid_id(tensor_grid_id, tensor_to_mesh, distribution_strides);
            reordered_mesh_ids[tensor_grid_id] = orig_mesh_ids[mesh_grid_id];
            reordered_chip_ids[tensor_grid_id] = orig_chip_ids[mesh_grid_id];
        }

        return {reordered_mesh_ids, reordered_chip_ids};
    }

    const tt::tt_metal::distributed::MeshBuffer& mesh_buffer() const { return mesh_buffer_; }

    MeshBuilder& mesh_builder() { return *mesh_builder_; }

private:
    // Get original fabric IDs from mesh builder (in mesh row-major order)
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> get_original_fabric_ids() const {
        auto fabric_node_args = mesh_builder_->get_fabric_nodes_compile_args();
        uint32_t num_grids = fabric_node_args[0];

        std::vector<uint32_t> mesh_ids(num_grids);
        std::vector<uint32_t> chip_ids(num_grids);
        for (uint32_t i = 0; i < num_grids; ++i) {
            mesh_ids[i] = fabric_node_args[1 + i];
            chip_ids[i] = fabric_node_args[1 + num_grids + i];
        }
        return {mesh_ids, chip_ids};
    }

    // Build mapping from tensor dims to mesh dims
    // Replicated dims map to tensor dim 0 (since we multiply dim 0 by replica count)
    // Sharded dims map their tensor dim to their mesh dim
    // Non-sharded dims map to -1
    std::vector<int> build_tensor_to_mesh_mapping() const {
        std::vector<int> tensor_to_mesh(mesh_tensor_rank_, -1);

        // Sharded dims: tensor_dim -> mesh_dim
        for (size_t mesh_dim = 0; mesh_dim < shard_dims_.size(); ++mesh_dim) {
            if (shard_dims_[mesh_dim].has_value()) {
                int tensor_dim = shard_dims_[mesh_dim].value();
                tensor_to_mesh[tensor_dim] = static_cast<int>(mesh_dim);
            }
        }

        // Replicated dims: all map to tensor dim 0
        // Note: if multiple replicated dims exist, they all contribute to dim 0
        for (size_t mesh_dim = 0; mesh_dim < shard_dims_.size(); ++mesh_dim) {
            if (!shard_dims_[mesh_dim].has_value()) {
                // Replicated dim maps to tensor dim 0
                // If dim 0 is already mapped to a sharded mesh dim, this replicated dim
                // shares the same tensor dim (they're multiplied together)
                if (tensor_to_mesh[0] == -1) {
                    tensor_to_mesh[0] = static_cast<int>(mesh_dim);
                }
                // If already mapped, the replicated dims are folded into dim 0
            }
        }

        return tensor_to_mesh;
    }

    // Compute row-major strides for distribution_shape_ using compute_strides from mesh_utils
    std::array<uint32_t, MAX_RANK> compute_distribution_strides() const {
        std::array<uint32_t, MAX_RANK> shape{};
        for (size_t i = 0; i < distribution_shape_.dims(); ++i) {
            shape[i] = distribution_shape_[i];
        }
        std::array<uint32_t, MAX_RANK> strides{};
        compute_strides(shape, static_cast<uint32_t>(distribution_shape_.dims()), strides);
        return strides;
    }

    // Get total number of tensor grid positions
    uint32_t get_total_tensor_grids() const {
        uint32_t total = 1;
        for (uint32_t d = 0; d < mesh_tensor_rank_; ++d) {
            total *= mesh_shape_[d];
        }
        return total;
    }

    // Convert tensor_grid_id to mesh_grid_id
    // Handles the case where tensor dim 0 encodes multiple mesh dims (sharded + replicated)
    uint32_t tensor_grid_id_to_mesh_grid_id(
        uint32_t tensor_grid_id,
        const std::vector<int>& tensor_to_mesh,
        const std::array<uint32_t, MAX_RANK>& distribution_strides) const {
        // Convert tensor_grid_id to tensor-dim coordinates using mesh_strides_
        std::array<uint32_t, MAX_RANK> tensor_coord{};
        uint32_t remaining = tensor_grid_id;
        for (uint32_t d = 0; d < mesh_tensor_rank_; ++d) {
            tensor_coord[d] = remaining / mesh_strides_[d];
            remaining %= mesh_strides_[d];
        }

        // Build mesh coordinates
        // For tensor dims > 0 with sharded mesh dims: direct mapping
        // For tensor dim 0: may encode multiple mesh dims (Shard{0} + all Replicate dims)
        std::array<uint32_t, MAX_RANK> mesh_coord{};

        // Handle tensor dims > 0 (simple 1-to-1 mapping for sharded dims)
        for (uint32_t tensor_dim = 1; tensor_dim < mesh_tensor_rank_; ++tensor_dim) {
            int mesh_dim = tensor_to_mesh[tensor_dim];
            if (mesh_dim >= 0) {
                mesh_coord[mesh_dim] = tensor_coord[tensor_dim];
            }
        }

        // Handle tensor dim 0: decompose into mesh coords for Shard{0} and Replicate dims
        // Order of factors applied to dim 0: Replicate first (outer), then Shard{0} (inner)
        // To decompose, process in reverse order (last applied = innermost = stride 1)
        uint32_t dim0_remaining = tensor_coord[0];

        // Shard{0} was applied last (inner), so extract it first
        for (size_t mesh_dim = 0; mesh_dim < shard_dims_.size(); ++mesh_dim) {
            if (shard_dims_[mesh_dim].has_value() && shard_dims_[mesh_dim].value() == 0) {
                uint32_t extent = distribution_shape_[mesh_dim];
                mesh_coord[mesh_dim] = dim0_remaining % extent;
                dim0_remaining /= extent;
            }
        }

        // Replicate dims were applied first (outer), so extract them last
        // Process in reverse mesh dim order to maintain correct decomposition
        for (int mesh_dim = static_cast<int>(shard_dims_.size()) - 1; mesh_dim >= 0; --mesh_dim) {
            if (!shard_dims_[mesh_dim].has_value()) {
                uint32_t extent = distribution_shape_[mesh_dim];
                mesh_coord[mesh_dim] = dim0_remaining % extent;
                dim0_remaining /= extent;
            }
        }

        // Convert mesh coordinates to mesh_grid_id
        uint32_t mesh_grid_id = 0;
        for (size_t d = 0; d < distribution_shape_.dims(); ++d) {
            mesh_grid_id += mesh_coord[d] * distribution_strides[d];
        }
        return mesh_grid_id;
    }

    // Helper: Expand sharded tensor dimensions by multiplying by mesh extent
    // (doesn't change rank, so tensor dimension indices remain stable)
    void expand_sharded_tensor_dims(std::array<uint32_t, MAX_RANK>& mesh_tensor_shape) const {
        for (int mesh_dim = static_cast<int>(shard_dims_.size()) - 1; mesh_dim >= 0; --mesh_dim) {
            if (shard_dims_[mesh_dim].has_value()) {
                int tensor_dim = shard_dims_[mesh_dim].value();
                mesh_tensor_shape[tensor_dim] *= distribution_shape_[mesh_dim];
            }
        }
    }

    // Helper: Expand replicated tensor dimensions by multiplying the first tensor dim
    // For each replicated mesh dim, multiply mesh_tensor_shape[0] by that mesh dim's extent.
    // This keeps tensor rank unchanged.
    void expand_replicated_tensor_dims(std::array<uint32_t, MAX_RANK>& mesh_tensor_shape) const {
        for (size_t mesh_dim = 0; mesh_dim < shard_dims_.size(); ++mesh_dim) {
            if (!shard_dims_[mesh_dim].has_value()) {
                // Replicated dim: multiply first tensor dim by mesh extent
                mesh_tensor_shape[0] *= distribution_shape_[mesh_dim];
            }
        }
    }

    std::pair<std::array<uint32_t, MAX_RANK>, uint32_t> reconstruct_mesh_tensor_shape(
        const Shape& local_tensor_shape) const {
        uint32_t rank = local_tensor_shape.rank();
        TT_FATAL(rank <= MAX_RANK, "Tensor rank exceeds MAX_RANK");

        std::array<uint32_t, MAX_RANK> mesh_tensor_shape{};
        for (size_t i = 0; i < rank; ++i) {
            mesh_tensor_shape[i] = local_tensor_shape[i];
        }

        // Apply distribution in two passes:
        // 1. First expand replicated dims (outer/slower varying)
        // 2. Then expand sharded dims (inner/faster varying)
        // This ensures shard indices are contiguous, matching mesh row-major order
        expand_replicated_tensor_dims(mesh_tensor_shape);
        expand_sharded_tensor_dims(mesh_tensor_shape);

        return {mesh_tensor_shape, rank};
    }

    void extract_tensor_shapes(const tt::tt_metal::Shape& tensor_shape_in_pages) {
        // Extract local tensor shape dimensions
        uint32_t local_rank = tensor_shape_in_pages.rank();
        TT_FATAL(local_rank <= MAX_RANK, "Tensor rank exceeds MAX_RANK");
        for (size_t i = 0; i < local_rank; ++i) {
            tensor_shape_in_pages_[i] = tensor_shape_in_pages[i];
        }

        // Apply distribution sharding and replication to get global mesh tensor shape
        auto [reconstructed_shape, reconstructed_rank] = reconstruct_mesh_tensor_shape(tensor_shape_in_pages);
        mesh_tensor_shape_in_pages_ = reconstructed_shape;
        mesh_tensor_rank_ = reconstructed_rank;

        // Adjust tensor_shape_in_pages to match mesh_tensor_rank
        adjust_shape_ranks(tensor_shape_in_pages_, local_rank, mesh_tensor_shape_in_pages_, mesh_tensor_rank_, 1);

        // Compute strides
        compute_strides(mesh_tensor_shape_in_pages_, mesh_tensor_rank_, mesh_tensor_strides_in_pages_);
        compute_strides(tensor_shape_in_pages_, mesh_tensor_rank_, tensor_strides_in_pages_);

        // Compute mesh shape (mesh_tensor_shape / tensor_shape) and mesh strides
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            mesh_shape_[i] = mesh_tensor_shape_in_pages_[i] / tensor_shape_in_pages_[i];
        }
        compute_strides(mesh_shape_, mesh_tensor_rank_, mesh_strides_);
    }

    const tt::tt_metal::distributed::MeshBuffer& mesh_buffer_;
    tt::tt_metal::TensorAccessorArgs accessor_args_;

    tt::tt_metal::distributed::MeshShape distribution_shape_;
    std::vector<std::optional<int>> shard_dims_;  // nullopt = replicate, value = shard on that tensor dim

    uint32_t mesh_tensor_rank_ = 0;
    std::array<uint32_t, MAX_RANK> mesh_tensor_shape_in_pages_{};
    std::array<uint32_t, MAX_RANK> mesh_tensor_strides_in_pages_{};
    std::array<uint32_t, MAX_RANK> tensor_shape_in_pages_{};
    std::array<uint32_t, MAX_RANK> tensor_strides_in_pages_{};
    std::array<uint32_t, MAX_RANK> mesh_shape_{};    // mesh device shape (mesh_tensor_shape / tensor_shape)
    std::array<uint32_t, MAX_RANK> mesh_strides_{};  // strides for mesh device space

    // Mesh builder for fabric node id mapping
    std::unique_ptr<MeshBuilder> mesh_builder_;
};

MeshTensorBuilder::MeshTensorBuilder(
    const tt::tt_metal::distributed::MeshBuffer& mesh_buffer,
    const tt::tt_metal::Shape& tensor_shape_in_pages,
    const tt::tt_metal::distributed::MeshShape& distribution_shape,
    const std::vector<std::optional<int>>& shard_dims) :
    impl_(std::make_unique<Impl>(mesh_buffer, tensor_shape_in_pages, distribution_shape, shard_dims)) {}

MeshTensorBuilder::~MeshTensorBuilder() = default;

MeshTensorBuilder::MeshTensorBuilder(MeshTensorBuilder&&) noexcept = default;
MeshTensorBuilder& MeshTensorBuilder::operator=(MeshTensorBuilder&&) noexcept = default;

std::vector<uint32_t> MeshTensorBuilder::get_compile_time_args() const { return impl_->get_compile_time_args(); }

uint64_t MeshTensorBuilder::get_buffer_address() const { return impl_->get_buffer_address(); }

uint32_t MeshTensorBuilder::get_aligned_page_size() const { return impl_->get_aligned_page_size(); }

tt::tt_metal::Shape MeshTensorBuilder::get_mesh_tensor_shape_in_pages() const {
    return impl_->get_mesh_tensor_shape_in_pages();
}

tt::tt_metal::Shape MeshTensorBuilder::get_local_tensor_shape_in_pages() const {
    return impl_->get_local_tensor_shape_in_pages();
}

tt::tt_metal::Shape MeshTensorBuilder::get_mesh_shape() const { return impl_->get_mesh_shape(); }

const tt::tt_metal::distributed::MeshShape& MeshTensorBuilder::get_distribution_shape() const {
    return impl_->get_distribution_shape();
}

const std::vector<std::optional<int>>& MeshTensorBuilder::get_shard_dims() const { return impl_->get_shard_dims(); }

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> MeshTensorBuilder::get_reordered_fabric_node_ids() const {
    return impl_->get_reordered_fabric_node_ids();
}

const tt::tt_metal::distributed::MeshBuffer& MeshTensorBuilder::mesh_buffer() const { return impl_->mesh_buffer(); }

MeshBuilder& MeshTensorBuilder::mesh_builder() { return impl_->mesh_builder(); }

}  // namespace tt::tt_metal::experimental::udm
