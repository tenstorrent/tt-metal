// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/tensor/details/storage.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}

template <typename Storage>
struct TensorImpl {
    Storage storage_;
    TensorSpec tensor_spec_;
    TensorTopology tensor_topology_;
};

template struct TensorImpl<HostStorage>;
template struct TensorImpl<DeviceStorage>;

namespace tensor_impl {

// Empty structs to facilitate Tensor template logic.
struct bfloat4_b {};
struct bfloat8_b {};

// ======================================================================================
//                                  Layout converters
// ======================================================================================

template <typename T>
std::vector<T> convert_layout_tile_to_row_major(
    const Shape2D& shape, const Tile& tile, tt::stl::Span<const T> data_to_convert) {
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    return convert_layout(
        data_to_convert,
        shape,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR,
        tile_shape,
        face_shape,
        transpose_within_face,
        transpose_of_faces);
}

// Converts logical data into physical data based on tensor spec
// - Logical data: Flat container of row major data corresponding to some ND logical shape
// - Physical data: Flat container of physical data corresponding to tensor spec. It takes into account:
//   * Sharding: Each shard will be padded to nearest page (if needed)
//     ** This is mostly for logical sharding, since logical shards may not be aligned to page in general
//     ** For interleaved, it will be handled as a "logically sharded" tensor with same shard shard height/width
//        as the original tensor dims at -2 and -1. In the future, interleaved may be generalized as sharded.
//     ** This means padding may be inserted in the middle of logical data (if needed)
//   * Layout: Each aligned shard will be tilized (if needed)
//     ** Tilization happens after first inserting padding to align shards (if needed)
//     ** For the last shard, we only align to nearest page instead of full shard size for partial shards
//   * After conversion, size of physical data will match 2D physical size indicated by tensor_spec.physical_shape()
template <typename T>
std::vector<T> encode_tensor_data(tt::stl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value = 0);

// Converts physical data into logical data based on tensor spec (see encode_tensor_data for details)
// - Physical data: Flat container of physical data corresponding to tensor spec
//   * Assumes that the physical data already matches tensor spec
//   * There is a bare minimum check that size of physical data matches size indicated by tensor_spec.physical_shape()
// - Logical data: Flat container of row major data corresponding to some ND logical shape
//   * To get logical data, perform the exact inverse process of encode_tensor_data
//   * Resulting data is safe to be converted to python tensors or general consumption with just a ND logical shape
template <typename T>
std::vector<T> decode_tensor_data(tt::stl::Span<const T> physical_data, const TensorSpec& tensor_spec);

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

std::shared_ptr<distributed::MeshBuffer> allocate_device_buffer(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec);

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec);

// Utility to convert runtime DataType to compile-time constant and dispatch the function call
template <typename Func, typename... Args>
auto dispatch(DataType dtype, Func&& func, Args&&... args) {
    switch (dtype) {
        case DataType::BFLOAT16:
            return (std::forward<Func>(func)).template operator()<bfloat16>(std::forward<Args>(args)...);
        case DataType::FLOAT32:
            return (std::forward<Func>(func)).template operator()<float>(std::forward<Args>(args)...);
        case DataType::INT32:
            return (std::forward<Func>(func)).template operator()<int32_t>(std::forward<Args>(args)...);
        case DataType::UINT32:
            return (std::forward<Func>(func)).template operator()<uint32_t>(std::forward<Args>(args)...);
        case DataType::UINT16:
            return (std::forward<Func>(func)).template operator()<uint16_t>(std::forward<Args>(args)...);
        case DataType::UINT8:
            return (std::forward<Func>(func)).template operator()<uint8_t>(std::forward<Args>(args)...);
        case DataType::BFLOAT8_B:
            return (std::forward<Func>(func)).template operator()<bfloat8_b>(std::forward<Args>(args)...);
        case DataType::BFLOAT4_B:
            return (std::forward<Func>(func)).template operator()<bfloat4_b>(std::forward<Args>(args)...);
        default: TT_THROW("Unsupported data type");
    }
}

}  // namespace tensor_impl

}  // namespace tt::tt_metal
