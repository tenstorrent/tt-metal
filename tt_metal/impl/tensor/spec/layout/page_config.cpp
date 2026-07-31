// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/fmt.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>

#include <tt-metalium/shape2d.hpp>
#include <numeric>
#include <type_traits>
#include <utility>
#include <variant>

#include "impl/tensor/spec/layout/page_config_impl.hpp"

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

size_t rm_element_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::FP8_E4M3: return sizeof(float8_e4m3);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
            // To store block floats in RowMajor layout, we use a fallback and store full floats instead
            return sizeof(float);

        default: TT_THROW("Unsupported data type!");
    }
}

// Maximum possible device memory alignment for all devices and buffer types.
constexpr uint32_t RECOMMENDED_MEMORY_ALIGNMENT_BYTES = 64;

Alignment create_default_alignment_tile(const TilePageConfig& config, DataType, const MemoryConfig&) {
    return Alignment({config.tile.get_height(), config.tile.get_width()});
}

Alignment create_default_alignment_rm(const RowMajorPageConfig&, DataType, const MemoryConfig& memory_config) {
    if (memory_config.shard_spec().has_value()) {
        const auto& shard_spec = memory_config.shard_spec().value();
        return Alignment({shard_spec.shape[1]});
    }
    if (memory_config.nd_shard_spec().has_value()) {
        const auto& nd_shard_spec = *memory_config.nd_shard_spec();
        return Alignment({nd_shard_spec.shard_shape[-1]});
    }
    return Alignment({1});
}

void validate_alignment_tile(const TilePageConfig& config, const Alignment& alignment, DataType, const MemoryConfig&) {
    TT_FATAL(alignment.size() >= 2, "Alignment should have at least 2 dimensions for Tile layout");
    const auto widthAlignment = alignment[-1];
    TT_FATAL(
        widthAlignment % config.tile.get_width() == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout innermost dimension should be multiple of tile width "
        "{}.",
        alignment,
        config.tile.get_width());
    auto heightAlignment = alignment[-2];
    TT_FATAL(
        (heightAlignment % config.tile.get_height()) == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout second innermost dimension should be multiple of "
        "tile height {}.",
        alignment,
        config.tile.get_height());
}

void validate_alignment_rm(
    const RowMajorPageConfig&, const Alignment& alignment, DataType, const MemoryConfig& memory_config) {
    TT_FATAL(!alignment.empty(), "Alignment must contain at least one dimension for Row Major layout.");
    const uint32_t width_alignment = alignment[-1];

    // TODO: Do we need to validate sharded width here if we are guaranteed that physical_shard_width is set as
    // width_alignment
    if (memory_config.shard_spec().has_value() && memory_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto& physical_shard_shape = memory_config.shard_spec().value().shape;
        const auto physical_shard_width = physical_shard_shape[1];
        TT_FATAL(
            physical_shard_width % width_alignment == 0,
            "Alignment mismatch for sharded tensor: Expected physical shard shape {} to be aligned to {} along the "
            "width for Row Major layout.",
            physical_shard_width,
            width_alignment);
    }
}

Shape2D get_page_shape_tile(
    const TilePageConfig& config, const Shape2D&, DataType, const MemoryConfig&, const std::optional<Shape2D>&) {
    return Shape2D(config.tile.get_height(), config.tile.get_width());
}

Shape2D get_page_shape_rm(
    const RowMajorPageConfig&,
    const Shape2D& physical_size,
    DataType dtype,
    const MemoryConfig& memory_config,
    const std::optional<Shape2D>& physical_shard_size) {
    if (physical_size.height() == 0 || physical_size.width() == 0) {
        return Shape2D(1, sizeof(uint32_t) / rm_element_size_bytes(dtype));
    }

    if (memory_config.shard_spec().has_value() && memory_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_FATAL(
            physical_shard_size.has_value(),
            "For width or block sharded tensors, Row Major page width comes from physical shard size so it must be "
            "provided!");

        return Shape2D(1, physical_shard_size.value().width());
    }

    if (memory_config.is_sharded() && memory_config.nd_shard_spec().has_value()) {
        const auto& nd_shard_spec = *memory_config.nd_shard_spec();
        return Shape2D(1, nd_shard_spec.shard_shape[-1]);
    }

    return Shape2D(1, physical_size.width());
}

size_t get_page_size_bytes_tile(const TilePageConfig& config, const Shape2D& page_shape, DataType dtype) {
    const auto tiles_count =
        page_shape.height() / config.tile.get_height() * page_shape.width() / config.tile.get_width();
    return tiles_count * config.tile.get_tile_size(datatype_to_dataformat_converter(dtype));
}

size_t get_page_size_bytes_rm(const RowMajorPageConfig&, const Shape2D& page_shape, DataType dtype) {
    return page_shape.height() * page_shape.width() * rm_element_size_bytes(dtype);
}

Alignment get_required_shard_shape_alignment_tile(const TilePageConfig& config) {
    return Alignment({config.tile.get_height(), config.tile.get_width()});
}

Alignment get_required_shard_shape_alignment_rm(const RowMajorPageConfig&) { return Alignment({1}); }

Alignment get_recommended_shard_shape_alignment_tile(const TilePageConfig& config, DataType) {
    return get_required_shard_shape_alignment_tile(config);
}

Alignment get_recommended_shard_shape_alignment_rm(const RowMajorPageConfig&, DataType dtype) {
    auto element_size_bytes = rm_element_size_bytes(dtype);
    auto alignment_bytes = std::lcm(RECOMMENDED_MEMORY_ALIGNMENT_BYTES, element_size_bytes);
    return Alignment({static_cast<uint32_t>(alignment_bytes / element_size_bytes)});
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

PageConfig::PageConfig(Config config) : config_(config) {}

PageConfig::PageConfig(Layout layout) : PageConfig(layout, std::nullopt) {}

PageConfig::PageConfig(Layout layout, const std::optional<Tile>& tile) {
    if (layout == Layout::ROW_MAJOR) {
        TT_FATAL(
            !tile.has_value() || *tile == Tile{},
            "Configuring a ROW MAJOR page config with a custom tile configuration is not supported.");
        config_ = RowMajorPageConfig{};
    } else {
        config_ = TilePageConfig{tile.value_or(Tile())};
    }
}

Layout PageConfig::get_layout() const {
    if (std::holds_alternative<RowMajorPageConfig>(config_)) {
        return Layout::ROW_MAJOR;
    }
    return Layout::TILE;
}

Tile PageConfig::get_tile() const {
    if (const auto* tile_config = std::get_if<TilePageConfig>(&config_)) {
        return tile_config->tile;
    }
    return Tile{};
}

Alignment PageConfig::get_recommended_shard_shape_alignment(DataType dtype) const {
    return tt::tt_metal::get_recommended_shard_shape_alignment(*this, dtype);
}

Alignment create_default_alignment(const PageConfig& page_config, DataType dtype, const MemoryConfig& memory_config) {
    return std::visit(
        [&](const auto& config) -> Alignment {
            using T = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<T, TilePageConfig>) {
                return CMAKE_UNIQUE_NAMESPACE::create_default_alignment_tile(config, dtype, memory_config);
            } else {
                return CMAKE_UNIQUE_NAMESPACE::create_default_alignment_rm(config, dtype, memory_config);
            }
        },
        page_config.get_config());
}

void validate_alignment(
    const PageConfig& page_config, const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) {
    std::visit(
        [&](const auto& config) {
            using T = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<T, TilePageConfig>) {
                CMAKE_UNIQUE_NAMESPACE::validate_alignment_tile(config, alignment, dtype, memory_config);
            } else {
                CMAKE_UNIQUE_NAMESPACE::validate_alignment_rm(config, alignment, dtype, memory_config);
            }
        },
        page_config.get_config());
}

Shape2D get_page_shape(
    const PageConfig& page_config,
    const Shape2D& physical_size,
    DataType dtype,
    const MemoryConfig& memory_config,
    const std::optional<Shape2D>& physical_shard_size) {
    return std::visit(
        [&](const auto& config) -> Shape2D {
            using T = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<T, TilePageConfig>) {
                return CMAKE_UNIQUE_NAMESPACE::get_page_shape_tile(
                    config, physical_size, dtype, memory_config, physical_shard_size);
            } else {
                return CMAKE_UNIQUE_NAMESPACE::get_page_shape_rm(
                    config, physical_size, dtype, memory_config, physical_shard_size);
            }
        },
        page_config.get_config());
}

size_t get_page_size_bytes(const PageConfig& page_config, const Shape2D& page_shape, DataType dtype) {
    return std::visit(
        [&](const auto& config) -> size_t {
            using T = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<T, TilePageConfig>) {
                return CMAKE_UNIQUE_NAMESPACE::get_page_size_bytes_tile(config, page_shape, dtype);
            } else {
                return CMAKE_UNIQUE_NAMESPACE::get_page_size_bytes_rm(config, page_shape, dtype);
            }
        },
        page_config.get_config());
}

Alignment get_required_shard_shape_alignment(const PageConfig& page_config) {
    return std::visit(
        [&](const auto& config) -> Alignment {
            using T = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<T, TilePageConfig>) {
                return CMAKE_UNIQUE_NAMESPACE::get_required_shard_shape_alignment_tile(config);
            } else {
                return CMAKE_UNIQUE_NAMESPACE::get_required_shard_shape_alignment_rm(config);
            }
        },
        page_config.get_config());
}

Alignment get_recommended_shard_shape_alignment(const PageConfig& page_config, DataType dtype) {
    return std::visit(
        [&](const auto& config) -> Alignment {
            using T = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<T, TilePageConfig>) {
                return CMAKE_UNIQUE_NAMESPACE::get_recommended_shard_shape_alignment_tile(config, dtype);
            } else {
                return CMAKE_UNIQUE_NAMESPACE::get_recommended_shard_shape_alignment_rm(config, dtype);
            }
        },
        page_config.get_config());
}

}  // namespace tt::tt_metal
