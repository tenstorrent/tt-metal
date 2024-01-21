// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "common/bfloat16.hpp"
#include "tensor/borrowed_buffer.hpp"
#include "tensor/owned_buffer.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/tt_stl/concepts.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

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
    BFLOAT8_B = 3,
    UINT16 = 4,
};

enum class StorageType {
    OWNED,
    DEVICE,
    BORROWED,  // for storing torch/numpy/etc tensors
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

    Padding(const Padding&) = default;
    Padding& operator=(const Padding&) = default;
    Padding(Padding&&) = default;
    Padding& operator=(Padding&&) = default;
    ~Padding() = default;

    Padding(const std::size_t rank);
    Padding(const std::initializer_list<PadDimension> pad_dimensions, PadValue pad_value);
    Padding(const std::vector<PadDimension>& pad_dimensions, PadValue pad_value);

    template <std::size_t Rank>
    Padding(const std::array<std::array<uint32_t, 2>, Rank> pad_dimensions, PadValue pad_value) :
        rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
        for (auto index = 0; index < Rank; index++) {
            this->pad_dimensions_[index] = {.front = pad_dimensions[index][0], .back = pad_dimensions[index][1]};
        }
    }

    PadDimension& operator[](const std::int64_t index);
    const PadDimension& operator[](const std::int64_t index) const;

    PadValue pad_value() const;

    static constexpr auto attribute_names = std::make_tuple("rank", "pad_dimensions", "pad_value");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->rank_), std::cref(this->pad_dimensions_), std::cref(this->pad_value_));
    }
};

bool operator==(const Padding&, const Padding&);
bool operator!=(const Padding&, const Padding&);

class Shape {
    std::size_t rank_;
    std::array<uint32_t, MAX_NUM_DIMENSIONS> dimensions_;
    Padding padding_;

   public:
    Shape(const Shape&) = default;
    Shape& operator=(const Shape&) = default;
    Shape(Shape&&) = default;
    Shape& operator=(Shape&&) = default;
    ~Shape() = default;

    Shape(const std::initializer_list<uint32_t>);
    Shape(const std::vector<uint32_t>&);
    Shape(const std::initializer_list<uint32_t>, const Padding&);
    Shape(const std::vector<uint32_t>&, const Padding&);

    explicit Shape(const Shape&, const Padding&);

    template <std::size_t Rank>
    explicit Shape(
        const std::array<uint32_t, Rank>& shape,
        const std::optional<std::array<uint32_t, Rank>>& padded_shape = std::nullopt) :
        rank_(Rank), dimensions_{}, padding_{Rank} {
        if (padded_shape.has_value()) {
            TT_ASSERT(shape.size() == padded_shape.value().size());
            for (auto index = 0; index < Rank; index++) {
                auto padded_dimension = padded_shape.value()[index];
                this->dimensions_[index] = padded_dimension;
                this->padding_[index] = {.front = 0, .back = padded_dimension - shape[index]};
            }
        } else {
            for (auto index = 0; index < Rank; index++) {
                this->dimensions_[index] = shape[index];
            }
        }
    }

    // Add an implicit constructor from 4D array due to legacy code
    Shape(const std::array<uint32_t, 4>& shape) : Shape(shape, std::optional<std::array<uint32_t, 4>>{std::nullopt}) {}

    std::size_t rank() const;

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

bool operator==(const Shape&, const Shape&);
bool operator!=(const Shape&, const Shape&);

struct MemoryConfig {
    TensorMemoryLayout memory_layout = TensorMemoryLayout::INTERLEAVED;    // Interleave the data across multiple banks
    BufferType buffer_type = BufferType::DRAM; // Can be either DRAM or L1
    std::optional <ShardSpec>  shard_spec = std::nullopt;
    bool is_sharded() const;

    static constexpr auto attribute_names = std::make_tuple("memory_layout", "buffer_type");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->memory_layout), std::cref(this->buffer_type));
    }
    ~MemoryConfig(){;}
};

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b);
bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b);

using OwnedBuffer = std::variant<
    owned_buffer::Buffer<uint16_t>,
    owned_buffer::Buffer<uint32_t>,
    owned_buffer::Buffer<float>,
    owned_buffer::Buffer<bfloat16>>;
struct OwnedStorage {
    OwnedBuffer buffer;

    static constexpr auto attribute_names = std::make_tuple();
    const auto attribute_values() const { return std::make_tuple(); }
};

using DeviceBuffer = std::shared_ptr<Buffer>;
struct DeviceStorage {
    DeviceBuffer buffer;

    const MemoryConfig memory_config() const {
        const auto& buffer = this->buffer;

        std::optional<ShardSpec> shard_spec_opt = std::nullopt;
        if(is_sharded(buffer->buffer_layout())){
            shard_spec_opt = buffer->shard_spec().tensor_shard_spec;
        }
        return MemoryConfig{
            .memory_layout = buffer->buffer_layout(),
            .buffer_type = buffer->buffer_type(),
            .shard_spec = shard_spec_opt
        };
    }

    static constexpr auto attribute_names = std::make_tuple("memory_config");
    const auto attribute_values() const { return std::make_tuple(this->memory_config()); }
};

using BorrowedBuffer = std::variant<
    borrowed_buffer::Buffer<uint16_t>,
    borrowed_buffer::Buffer<uint32_t>,
    borrowed_buffer::Buffer<float>,
    borrowed_buffer::Buffer<bfloat16>>;
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


bool operator==(const ShardSpec& spec_a, const ShardSpec& spec_b);
bool operator!=(const ShardSpec& spec_a, const ShardSpec& spec_b);


}  // namespace tt_metal

}  // namespace tt


namespace std {

template <>
struct hash<tt::tt_metal::MemoryConfig> {
    uint64_t operator()(const tt::tt_metal::MemoryConfig &mem_config) const {
        return tt::stl::hash::hash_objects(0,
             typeid(tt::tt_metal::MemoryConfig).hash_code(),
             mem_config.memory_layout,
             mem_config.buffer_type
             );
    }
};


template <>
struct hash<tt::tt_metal::DeviceStorage> {
    uint64_t operator()(const tt::tt_metal::DeviceStorage &storage) const {
        return tt::stl::hash::hash_objects(0,
             typeid(tt::tt_metal::DeviceStorage).hash_code(),
             storage.buffer,
             storage.memory_config()
             );
    }
};


template <>
struct hash<tt::tt_metal::Padding::PadDimension> {
    uint64_t operator()(const tt::tt_metal::Padding::PadDimension &pad_dimension) const {
        return tt::stl::hash::hash_objects(0,
             typeid(tt::tt_metal::Padding::PadDimension).hash_code(),
             pad_dimension.front,
             pad_dimension.back
             );
    }
};


template <>
struct hash<tt::tt_metal::Padding> {
    uint64_t operator()(const tt::tt_metal::Padding &padding) const {

        uint64_t hash = tt::stl::hash::hash_objects(0, typeid(tt::tt_metal::Padding).hash_code(),
            padding.rank_, padding.pad_value_);
        for (const auto& pad_dim : padding.pad_dimensions_) {
            hash = tt::stl::hash::hash_objects(hash, pad_dim);
        }
        return hash;
    }
};
template <>
struct hash<tt::tt_metal::Shape> {
    uint64_t operator()(const tt::tt_metal::Shape &shape) const {

        uint64_t hash = tt::stl::hash::hash_objects(0, typeid(tt::tt_metal::Shape).hash_code(),
            shape.rank(), shape.padding());
        for(int idx=0; idx < shape.rank(); idx++){
            hash = tt::stl::hash::hash_objects(hash, shape[idx]);
        }
        return hash;

    }
};

}
