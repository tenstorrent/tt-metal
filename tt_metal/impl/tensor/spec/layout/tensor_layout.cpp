// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>

#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

#include "page_config_impl.hpp"
#include "tensor_layout_impl.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

size_t round_up(size_t value, size_t multiple) {
    if (multiple == 0) {
        return value;
    }

    return ((value + multiple - 1) / multiple) * multiple;
};

void validate_alignment(const TensorLayoutImpl& tensor_layout) {
    const auto& alignment = tensor_layout.get_alignment();
    const auto& memory_config = tensor_layout.get_memory_config();
    TT_FATAL(
        alignment.size() <= 2 || !memory_config.shard_spec().has_value(),
        "Tensor must be interleaved if alignment has rank greater than 2!");

    const auto& page_config = tensor_layout.get_page_config();
    const auto& dtype = tensor_layout.get_data_type();
    validate_alignment(page_config, alignment, dtype, memory_config);
}

std::optional<std::string> get_shard_align_error(
    const MemoryConfig& memory_config, const Layout& layout, const Tile& tile) {
    if (!memory_config.is_sharded() || layout != Layout::TILE) {
        return std::nullopt;
    }
    const auto& tile_shape = tile.get_tile_shape();
    if (memory_config.shard_spec().has_value()) {
        const auto& physical_shard_shape = Shape2D(memory_config.shard_spec().value().shape);
        if (!(physical_shard_shape.height() % tile_shape[0] == 0 &&
              physical_shard_shape.width() % tile_shape[1] == 0)) {
            return fmt::format("Physical shard shape {} must be tile {} sized!", physical_shard_shape, tile_shape);
        }
    } else {
        const auto& shard_shape = memory_config.nd_shard_spec().value().shard_shape;
        if (!(shard_shape[-2] % tile_shape[0] == 0 && shard_shape[-1] % tile_shape[1] == 0)) {
            return fmt::format("Physical shard shape {} must be tile {} sized!", shard_shape, tile_shape);
        }
    }
    return std::nullopt;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

bool can_shard_align(const MemoryConfig& memory_config, const Layout& layout, const Tile& tile) {
    return !CMAKE_UNIQUE_NAMESPACE::get_shard_align_error(memory_config, layout, tile).has_value();
}

// ------------------------------------------------------------------------------------------------
// TensorLayoutImpl: the internal layout-computation API, reachable from within tt_metal via impl().
// ------------------------------------------------------------------------------------------------

TensorLayoutImpl::TensorLayoutImpl(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config) :
    dtype_(dtype), page_config_(page_config), memory_config_(memory_config) {
    initialize_alignment();
    CMAKE_UNIQUE_NAMESPACE::validate_alignment(*this);

    if (get_layout() == Layout::TILE) {
        auto shard_align_error =
            CMAKE_UNIQUE_NAMESPACE::get_shard_align_error(memory_config_, get_layout(), get_tile());
        TT_FATAL(!shard_align_error.has_value(), "{}", shard_align_error);
    }
}

void TensorLayoutImpl::set_custom_alignment(const Alignment& alignment) {
    // initialize_alignment() merges alignment_ into the Alignment derived from page_config_ / dtype_ /
    // memory_config_, so overwriting alignment_ first yields the same result as constructing with **alignment**.
    alignment_ = alignment;
    initialize_alignment();
    CMAKE_UNIQUE_NAMESPACE::validate_alignment(*this);
}

void TensorLayoutImpl::initialize_alignment() {
    auto default_alignment = create_default_alignment(page_config_, dtype_, memory_config_);
    if (alignment_.empty()) {
        alignment_ = default_alignment;
        return;
    }

    ttsl::SmallVector<uint32_t> result(std::max(alignment_.size(), default_alignment.size()), 1);
    for (size_t i = 0; i < alignment_.size(); i++) {
        result[i + result.size() - alignment_.size()] = alignment_[i];
    }
    for (size_t i = 0; i < default_alignment.size(); i++) {
        size_t result_idx = i + result.size() - default_alignment.size();
        result[result_idx] = CMAKE_UNIQUE_NAMESPACE::round_up(result[result_idx], default_alignment[i]);
    }
    alignment_ = Alignment(std::move(result));
}

BufferShardingArgs TensorLayoutImpl::compute_buffer_sharding_args(const tt::tt_metal::Shape& shape) const {
    if (!memory_config_.is_sharded()) {
        return {};
    }

    TT_FATAL(
        memory_config_.shard_spec().has_value() || memory_config_.nd_shard_spec().has_value(),
        "MemoryConfig must have Shard Spec specified for sharded memory layout");

    const Shape2D physical_size = compute_physical_shape(shape);
    const Shape2D page_shape = compute_page_shape(physical_size);

    TT_FATAL(
        physical_size.width() % page_shape.width() == 0,
        "Physical width {} must be multiple of page width {}",
        physical_size.width(),
        page_shape.width());
    TT_FATAL(
        physical_size.height() % page_shape.height() == 0,
        "Physical height {} must be multiple of page height {}",
        physical_size.height(),
        page_shape.height());

    std::optional<ShardSpecBuffer> shard_spec_buffer;
    std::optional<BufferDistributionSpec> distribution_spec;

    if (auto shard_spec = memory_config_.shard_spec()) {
        const auto width_in_pages = physical_size.width() / page_shape.width();
        const auto height_in_pages = physical_size.height() / page_shape.height();
        const std::array<uint32_t, 2> tensor2d_shape_in_pages{
            static_cast<uint32_t>(height_in_pages), static_cast<uint32_t>(width_in_pages)};
        shard_spec_buffer = ShardSpecBuffer(*shard_spec, std::array<uint32_t, 2>(page_shape), tensor2d_shape_in_pages);
        auto padded_shape = compute_padded_shape(shape);
        if (padded_shape.rank() < 2) {  // Edge Case: For 1-D tensors and scalars, we need to make its shape 2D to
                                        // construct the buffer distribution spec, since the tensor rank cannot be less
                                        // than the shard rank (always 2 for 2D sharding).
            padded_shape = Shape({1, padded_shape[0]});
        }
        distribution_spec = BufferDistributionSpec::from_shard_spec(
            padded_shape,
            Shape(shard_spec->shape),
            page_shape,
            shard_spec->grid,
            shard_spec->orientation,
            memory_config_.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED
                ? ShardDistributionStrategy::GRID_2D
                : ShardDistributionStrategy::ROUND_ROBIN_1D);
    }

    if (const auto& nd_shard_spec = memory_config_.nd_shard_spec()) {
        auto padded_shape = compute_padded_shape(shape);
        distribution_spec = BufferDistributionSpec::from_shard_spec(
            padded_shape,
            nd_shard_spec->shard_shape,
            page_shape,
            nd_shard_spec->grid,
            nd_shard_spec->orientation,
            nd_shard_spec->shard_distribution_strategy);
    }
    auto sharding_args =
        BufferShardingArgs(std::move(distribution_spec), std::move(shard_spec_buffer), memory_config_.memory_layout());
    if (tt::tt_metal::experimental::per_core_allocation::is_per_core_allocation(memory_config_)) {
        tt::tt_metal::experimental::per_core_allocation::set_per_core_allocation(sharding_args, true);
    }
    return sharding_args;
}

size_t TensorLayoutImpl::compute_packed_buffer_size_bytes(const tt::tt_metal::Shape& shape) const {
    const Shape2D physical_size = compute_physical_shape(shape);
    const Shape2D page_shape = compute_page_shape(physical_size);
    const auto width_remainder = physical_size.width() % page_shape.width();
    const auto height_remainder = physical_size.height() % page_shape.height();
    TT_FATAL(
        (width_remainder == 0 && height_remainder == 0) || ((physical_size.width() * physical_size.height()) == 0),
        "Physical size {} must be multiple of page size {}",
        physical_size,
        page_shape);

    const size_t physical_area = physical_size.height() * physical_size.width();
    const size_t page_area = page_shape.height() * page_shape.width();

    const size_t page_count = physical_area / page_area;
    const size_t page_size_bytes = compute_page_size_bytes(page_shape);

    return page_count * page_size_bytes;
}

size_t TensorLayoutImpl::compute_page_size_bytes(const tt::tt_metal::Shape& shape) const {
    const auto physical_size = compute_physical_shape(shape);
    const auto page_shape = compute_page_shape(physical_size);
    return compute_page_size_bytes(page_shape);
}

size_t TensorLayoutImpl::compute_page_size_bytes(const Shape2D& page_size) const {
    return get_page_size_bytes(page_config_, page_size, dtype_);
}

size_t TensorLayoutImpl::compute_consumed_memory_bytes_per_bank(
    const tt::tt_metal::Shape& shape, size_t page_alignment, size_t num_banks) const {
    const Shape2D physical_shape = compute_physical_shape(shape);
    const Shape2D page_shape = compute_page_shape(physical_shape);

    size_t num_pages_per_bank = 0;
    if (!memory_config_.is_sharded()) {
        const size_t num_pages =
            physical_shape.height() * physical_shape.width() / page_shape.height() / page_shape.width();
        num_pages_per_bank = div_up(num_pages, num_banks);
    } else if (const auto& shard_spec = memory_config_.shard_spec()) {
        Shape2D shard_shape = Shape2D(shard_spec->shape);
        num_pages_per_bank =
            div_up(shard_shape.height(), page_shape.height()) * div_up(shard_shape.width(), page_shape.width());
    } else {
        auto sharding_args = compute_buffer_sharding_args(shape);
        const auto& dist_spec = sharding_args.buffer_distribution_spec().value();
        num_pages_per_bank = dist_spec.max_num_dev_pages_per_core();
    }

    const size_t aligned_page_size = round_up(compute_page_size_bytes(page_shape), page_alignment);
    return num_pages_per_bank * aligned_page_size;
}

Shape2D TensorLayoutImpl::get_logical_shard_shape() const {
    TT_FATAL(
        memory_config_.shard_spec().has_value(),
        "Shard spec must have value for TensorLayout::get_logical_shard_shape!");

    // In physical mode, shape in shard spec is logical shard shape if no padding
    // Otherwise, not possible to infer logical shard shape in general
    return Shape2D(memory_config_.shard_spec().value().shape);
}

Shape2D TensorLayoutImpl::get_physical_shard_shape() const {
    TT_FATAL(
        memory_config_.shard_spec().has_value(),
        "Shard spec must have value for TensorLayout::get_physical_shard_shape!");
    const auto& shard_spec = memory_config_.shard_spec().value();
    return shard_spec.shape;
}

Shape2D TensorLayoutImpl::compute_logical_2d_shape(const tt::tt_metal::Shape& shape) const {
    if (shape.rank() < 2) {
        return Shape2D{1, shape[-1]};
    }
    size_t width = shape[-1];
    size_t height = shape[-2];
    for (int i = -3; i >= -shape.rank(); --i) {
        height *= shape[i];
    }
    return Shape2D{height, width};
}

Shape2D TensorLayoutImpl::compute_physical_shape(const tt::tt_metal::Shape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignment_rank = static_cast<int>(alignment_.size());

    size_t width = 1;
    size_t height = 1;

    const int max_rank = std::max(rank, alignment_rank);

    // Iterate dims in reverse order and ensure alignment
    // Even tensor of rank 0 or 1 must be aligned (to Tile / Page / Shard)
    for (int i = -1; i >= -max_rank; --i) {
        auto& dim = i == -1 ? width : height;
        if (i >= -rank) {
            dim *= shape[i];
        }

        // Align the current dimension if alignment is available
        if (i >= -alignment_rank) {
            dim = CMAKE_UNIQUE_NAMESPACE::round_up(dim, alignment_[i]);
        }
    }

    Shape2D size{height, width};
    return size;
}

Shape2D TensorLayoutImpl::compute_page_shape(const Shape2D& physical_size) const {
    std::optional<Shape2D> physical_shard_shape = std::nullopt;
    if (memory_config_.shard_spec().has_value()) {
        physical_shard_shape = get_physical_shard_shape();
    }

    return get_page_shape(page_config_, physical_size, dtype_, memory_config_, physical_shard_shape);
}

Strides TensorLayoutImpl::compute_strides(const tt::tt_metal::Shape& logical_shape) const {
    const int rank = static_cast<int>(logical_shape.rank());
    const int alignment_rank = static_cast<int>(alignment_.size());
    Strides strides(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * logical_shape[i + 1];
        const int alignment_index = i - (rank - alignment_rank) + 1;
        if (alignment_index >= 0) {
            strides[i] = CMAKE_UNIQUE_NAMESPACE::round_up(strides[i], alignment_[alignment_index]);
        }
    }
    return strides;
}

tt::tt_metal::Shape TensorLayoutImpl::compute_padded_shape(const tt::tt_metal::Shape& shape) const {
    ttsl::SmallVector<uint32_t> padded_shape(std::max(shape.rank(), alignment_.size()));
    int rank_index = static_cast<int>(shape.rank()) - 1;
    int alignment_index = static_cast<int>(alignment_.size()) - 1;
    int padded_shape_index = static_cast<int>(padded_shape.size() - 1);
    size_t accum_alignment = 1;

    for (; alignment_index >= 0; rank_index--, alignment_index--, padded_shape_index--) {
        uint32_t shape_value = rank_index >= 0 ? shape[rank_index] : 1;
        uint32_t alignment_value = alignment_[alignment_index];
        uint32_t& padded_shape_value = padded_shape[padded_shape_index];
        // The last 2 dimensions of a shape are special
        if (rank_index >= static_cast<int>(shape.rank()) - 2) {
            padded_shape_value = CMAKE_UNIQUE_NAMESPACE::round_up(shape_value, alignment_value);
        } else {
            if (accum_alignment % alignment_value == 0) {
                // Alignment for this dimension is redundant, ignoring
                padded_shape_value = shape_value;
            } else if (alignment_value % accum_alignment == 0) {
                padded_shape_value = CMAKE_UNIQUE_NAMESPACE::round_up(shape_value, alignment_value / accum_alignment);
            } else {
                TT_THROW(
                    "Padded shape can't be deducted from TensorLayout parameters {} and Shape {}", alignment_, shape);
            }
        }

        // Alignment doesn't accumulate on the last dimension of a shape
        if (rank_index != static_cast<int>(shape.rank()) - 1) {
            accum_alignment *= padded_shape_value;
        }
    }
    for (; rank_index >= 0; rank_index--, padded_shape_index--) {
        padded_shape[padded_shape_index] = shape[rank_index];
    }
    return tt::tt_metal::Shape(std::move(padded_shape));
}

// ------------------------------------------------------------------------------------------------
// TensorLayout: public facade forwarding to TensorLayoutImpl.
// ------------------------------------------------------------------------------------------------

TensorLayout::TensorLayout(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config) :
    impl_(std::make_unique<TensorLayoutImpl>(dtype, page_config, memory_config)) {}

TensorLayout::~TensorLayout() = default;

TensorLayout::TensorLayout(const TensorLayout& other) :
    impl_(other.impl_ ? std::make_unique<TensorLayoutImpl>(*other.impl_) : nullptr) {}

TensorLayout& TensorLayout::operator=(const TensorLayout& other) {
    if (this == &other) {
        return *this;
    }
    impl_ = other.impl_ ? std::make_unique<TensorLayoutImpl>(*other.impl_) : nullptr;
    return *this;
}

TensorLayout::TensorLayout(TensorLayout&& other) noexcept = default;
TensorLayout& TensorLayout::operator=(TensorLayout&& other) noexcept = default;

TensorLayoutImpl& TensorLayout::impl() {
    TT_FATAL(impl_ != nullptr, "TensorLayout is in a moved-from state.");
    return *impl_;
}

const TensorLayoutImpl& TensorLayout::impl() const {
    TT_FATAL(impl_ != nullptr, "TensorLayout is in a moved-from state.");
    return *impl_;
}

Layout TensorLayout::get_layout() const { return impl().get_layout(); }
Tile TensorLayout::get_tile() const { return impl().get_tile(); }
PageConfig TensorLayout::get_page_config() const { return impl().get_page_config(); }
DataType TensorLayout::get_data_type() const { return impl().get_data_type(); }
const MemoryConfig& TensorLayout::get_memory_config() const { return impl().get_memory_config(); }
const Alignment& TensorLayout::get_alignment() const { return impl().get_alignment(); }

tt::tt_metal::Shape TensorLayout::compute_padded_shape(const tt::tt_metal::Shape& shape) const {
    return impl().compute_padded_shape(shape);
}

TensorLayout TensorLayout::with_memory_config(MemoryConfig memory_config) const {
    TensorLayout result = *this;
    result.impl().set_memory_config(std::move(memory_config));
    return result;
}

bool TensorLayout::operator==(const TensorLayout& other) const { return impl() == other.impl(); }
bool TensorLayout::operator!=(const TensorLayout& other) const { return impl() != other.impl(); }

std::tuple<const DataType&, const PageConfig&, const MemoryConfig&, const Alignment&> TensorLayout::attribute_values()
    const {
    return impl().attribute_values();
}

}  // namespace tt::tt_metal
