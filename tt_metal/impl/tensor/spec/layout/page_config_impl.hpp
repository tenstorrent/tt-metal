// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/fmt.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>

#include <tt-metalium/shape2d.hpp>
#include <memory>
#include <numeric>

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

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

// ------------------------------------------------------------------------------------------------
// PageConfigImpl: the internal page-config API, reachable from within tt_metal via impl().
// ------------------------------------------------------------------------------------------------

class PageConfigImpl {
public:
    explicit PageConfigImpl(const PageConfig::Config& config) : config_(config) {}

    PageConfigImpl(Layout layout, const std::optional<Tile>& tile) {
        if (layout == Layout::ROW_MAJOR) {
            // TODO: add TT_FATAL(!tile.has_value(), "Specifying tile shape for a row major layout is not supported")
            config_ = RowMajorPageConfig(tile.value_or(Tile()));
        } else {
            config_ = TilePageConfig(tile.value_or(Tile()));
        }
    }

    bool operator==(const PageConfigImpl&) const = default;
    bool operator!=(const PageConfigImpl&) const = default;

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const {
        return std::visit(
            [&](const auto& config) constexpr { return config.create_default_alignment(dtype, memory_config); },
            config_);
    }

    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const {
        std::visit(
            [&](const auto& config) constexpr { config.validate_alignment(alignment, dtype, memory_config); }, config_);
    }

    Shape2D get_page_shape(
        const Shape2D& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Shape2D>& physical_shard_size) const {
        return std::visit(
            [&](const auto& config) constexpr {
                return config.get_page_shape(physical_size, dtype, memory_config, physical_shard_size);
            },
            config_);
    }

    size_t get_page_size_bytes(const Shape2D& page_shape, DataType dtype) const {
        return std::visit(
            [&](const auto& config) constexpr { return config.get_page_size_bytes(page_shape, dtype); }, config_);
    }

    Layout get_layout() const {
        if (std::holds_alternative<RowMajorPageConfig>(config_)) {
            return Layout::ROW_MAJOR;
        }
        return Layout::TILE;
    }

    Tile get_tile() const {
        return std::visit([&](const auto& config) { return config.get_tile(); }, config_);
    }

    Alignment get_required_shard_shape_alignment() const {
        return std::visit(
            [&](const auto& config) constexpr { return config.get_required_shard_shape_alignment(); }, config_);
    }

    Alignment get_recommended_shard_shape_alignment(DataType dtype) const {
        return std::visit(
            [&](const auto& config) constexpr { return config.get_recommended_shard_shape_alignment(dtype); }, config_);
    }

    const PageConfig::Config& config() const { return config_; }

private:
    PageConfig::Config config_;
};

}  // namespace tt::tt_metal
