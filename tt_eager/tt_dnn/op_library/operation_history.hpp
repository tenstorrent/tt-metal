// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_eager/tensor/tensor.hpp>
#include "tt_dnn/op_library/operation.hpp"

#include <sstream>
#include <fmt/ostream.h>
#include <string_view>

template <>
struct fmt::formatter<tt::tt_metal::ShardOrientation> : formatter<string_view> {
    template <typename FormatContext>
    auto format(tt::tt_metal::ShardOrientation c, FormatContext& ctx) {
        string_view name = "Unknown";
        switch (c) {
            case tt::tt_metal::ShardOrientation::ROW_MAJOR: name = "Row Major"; break;
            case tt::tt_metal::ShardOrientation::COL_MAJOR: name = "Column Major"; break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};

template <>
struct fmt::formatter<ShardSpec> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator { return ctx.end(); }

    template <typename FormatContext>
    auto format(const ShardSpec &spec, FormatContext &ctx) {
        std::string halo_str = spec.halo ? "true" : "false";
        return fmt::format_to(
            ctx.out(),
            "Grid: {}, Shape: [{} x {}], Orientation: {}, Halo: {}",
            spec.grid, spec.shape[0], spec.shape[1], spec.orientation, halo_str);
    }
};

template <>
struct fmt::formatter<ShardSpecBuffer> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator { return ctx.end(); }

    template <typename FormatContext>
    auto format(const ShardSpecBuffer &buffer, FormatContext &ctx) {
        return fmt::format_to(
            ctx.out(),
            "ShardSpec: {{ {} }}, Page Shape: [{} x {}], Tensor2D Shape: [{} x {}]",
            buffer.tensor_shard_spec,
            buffer.page_shape[0], buffer.page_shape[1],
            buffer.tensor2d_shape[0], buffer.tensor2d_shape[1]);
    }
};

namespace tt {

namespace tt_metal {

#ifdef DEBUG

namespace operation_history {

struct TensorRecord {
    const StorageType storage_type;
    const Shape shape;
    const DataType data_type;
    const Layout layout;
    const std::optional<MemoryConfig> memory_config;
    const std::vector<ShardSpecBuffer> shard_spec_buffers{};

    static constexpr auto attribute_names = std::make_tuple("storage_type", "shape", "data_type", "layout", "memory_config","shard_spec_buffers");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->storage_type), std::cref(this->shape), std::cref(this->data_type), std::cref(this->layout), std::cref(this->memory_config), std::cref(this->shard_spec_buffers));
    }
};

struct OperationRecord {
    const std::string opcode;
    const tt::stl::reflection::Attributes attributes;
    const std::vector<TensorRecord> input_tensor_records;
    const std::vector<const char*> composite_parent_names{};
};

namespace detail {

struct OperationHistory {

    ~OperationHistory();

    void append(OperationRecord&& record);
    void dump_to_csv();

  private:
    std::vector<OperationRecord> records;
};

inline OperationHistory OPERATION_HISTORY{};

}

template<typename ... Args>
inline void append(Args&& ... args) {
    detail::OPERATION_HISTORY.append(std::forward<Args>(args)...);
}

const char* csv_file_name();

bool enabled();

}  // namespace operation_history

#endif

}  // namespace tt_metal

}  // namespace tt
