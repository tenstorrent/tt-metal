// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/borrowed_buffer.hpp"
#include "tensor/owned_buffer.hpp"

#include "common/bfloat16.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

#include "tt_stl/concepts.hpp"
#include "tt_stl/reflection.hpp"

#include <memory>
#include <variant>
#include <vector>


namespace tt {

namespace tt_metal {

enum class Layout {
    ROW_MAJOR = 0,
    TILE = 1
};

enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3
};

enum class StorageType {
    OWNED,
    DEVICE,
    BORROWED,  // for storing torch/numpy/etc tensors
};

enum class TensorMemoryLayout {
    INTERLEAVED,
    SINGLE_BANK,
    HEIGHT_SHARDED,
    WIDTH_SHARDED,
    BLOCK_SHARDED,
};

enum class ShardOrientation {
    ROW_MAJOR,
    COL_MAJOR,
};


tt::DataFormat datatype_to_dataformat_converter(DataType datatype);


static constexpr std::size_t MAX_NUM_DIMENSIONS = 8;

struct Padding {
    enum class PadValue {
        Any,
        Zero,
        Infinity,
        NegativeInfinity
    };

    struct PadDimension {
        std::size_t front;
        std::size_t back;

        static constexpr auto attribute_names = std::make_tuple("front", "back");
        const auto attribute_values() const { return std::make_tuple(std::cref(this->front), std::cref(this->back)); }
    };

    std::size_t rank_;
    std::array<PadDimension, MAX_NUM_DIMENSIONS> pad_dimensions_;
    PadValue pad_value_;

    Padding(const std::size_t rank);
    Padding(const std::initializer_list<PadDimension> pad_dimensions, PadValue pad_value);
    Padding(const std::vector<PadDimension>& pad_dimensions, PadValue pad_value);

    PadDimension& operator[](const std::int64_t index);
    const PadDimension& operator[](const std::int64_t index) const;

    PadValue pad_value() const;

    static constexpr auto attribute_names = std::make_tuple("rank", "pad_dimensions", "pad_value");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->rank_), std::cref(this->pad_dimensions_), std::cref(this->pad_value_));
    }
};

class Shape {
    std::size_t rank_;
    std::array<uint32_t, MAX_NUM_DIMENSIONS> dimensions_;
    Padding padding_;

   public:
    Shape(const std::initializer_list<uint32_t>);
    Shape(const std::array<uint32_t, 4>&);
    Shape(const std::vector<uint32_t>&);

    Shape(const std::initializer_list<uint32_t>, const Padding&);
    Shape(const std::vector<uint32_t>&, const Padding&);
    Shape(const Shape&, const Padding&);

    uint32_t rank() const;

    uint32_t& operator[](const std::int64_t index);
    const uint32_t& operator[](const std::int64_t index) const;

    const uint32_t* begin() const;
    const uint32_t* end() const;

    const Padding& padding() const;
    const Shape without_padding() const;

    const uint32_t get_normalized_index(std::int64_t index) const;

    static constexpr auto attribute_names = std::make_tuple("rank", "dimensions", "padding");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->rank_), std::cref(this->dimensions_), std::cref(this->padding_));
    }
};

bool operator==(const Shape& shape_a, const Shape& shape_b);
bool operator!=(const Shape& shape_a, const Shape& shape_b);

struct MemoryConfig {
    TensorMemoryLayout memory_layout = TensorMemoryLayout::INTERLEAVED;    // Interleave the data across multiple banks
    BufferType buffer_type = BufferType::DRAM; // Can be either DRAM or L1
    bool is_sharded() const;

    static constexpr auto attribute_names = std::make_tuple("memory_layout", "buffer_type");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->memory_layout), std::cref(this->buffer_type));
    }
};

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b);
bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b);

using OwnedBuffer = std::variant<
    owned_buffer::Buffer<uint32_t>,
    owned_buffer::Buffer<float>,
    owned_buffer::Buffer<bfloat16>
>;
struct OwnedStorage {
    OwnedBuffer buffer;

    static constexpr auto attribute_names = std::make_tuple();
    const auto attribute_values() const { return std::make_tuple(); }
};

using DeviceBuffer = std::shared_ptr<Buffer>;
struct DeviceStorage {
    DeviceBuffer buffer;
    Device* device;
    MemoryConfig memory_config;

    static constexpr auto attribute_names = std::make_tuple("memory_config");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->memory_config)); }
};

using BorrowedBuffer = std::variant<
    borrowed_buffer::Buffer<uint32_t>,
    borrowed_buffer::Buffer<float>,
    borrowed_buffer::Buffer<bfloat16>
>;
struct BorrowedStorage {
    BorrowedBuffer buffer;
    std::function<void()> on_creation_callback = []{};
    std::function<void()> on_destruction_callback = []{};


    explicit BorrowedStorage(const BorrowedBuffer& buffer, const std::function<void()>& on_creation_callback, const std::function<void()>& on_destruction_callback)
    : buffer(buffer), on_creation_callback(on_creation_callback), on_destruction_callback(on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage(const BorrowedStorage& other)
    : buffer(other.buffer), on_creation_callback(other.on_creation_callback), on_destruction_callback(other.on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage operator=(const BorrowedStorage& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        this->on_creation_callback();
        return *this;
    }

    BorrowedStorage(BorrowedStorage&& other)
    : buffer(other.buffer), on_creation_callback(other.on_creation_callback), on_destruction_callback(other.on_destruction_callback) {
        other.on_creation_callback = []{};
        other.on_destruction_callback = []{};
    }

    BorrowedStorage operator=(BorrowedStorage&& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        other.on_creation_callback = []{};
        other.on_destruction_callback = []{};
        return *this;
    }

    ~BorrowedStorage() {
        this->on_destruction_callback();
    }

    static constexpr auto attribute_names = std::make_tuple();
    const auto attribute_values() const { return std::make_tuple(); }
};

using Storage = std::variant<
    OwnedStorage,
    DeviceStorage,
    BorrowedStorage
>;

template<typename T>
constexpr void raise_unsupported_storage() {
    static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported Storage");
}

struct ShardSpec {
    CoreRangeSet shard_grid;
    std::array<uint32_t, 2> shard_shape;
    ShardOrientation shard_orientation;
    bool halo = false;
    const uint32_t num_cores() const { return this->shard_grid.num_cores(); }
    const uint32_t numel() const { return this->shard_shape[0] * this->shard_shape[1]; }

    static constexpr auto attribute_names = std::make_tuple("shard_grid", "shard_shape", "shard_orientation", "halo");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->shard_grid),
            std::cref(this->shard_shape),
            std::cref(this->shard_orientation),
            std::cref(this->halo));
    }
};

bool operator==(const ShardSpec& spec_a, const ShardSpec& spec_b);
bool operator!=(const ShardSpec& spec_a, const ShardSpec& spec_b);


}  // namespace tt_metal

}  // namespace tt
