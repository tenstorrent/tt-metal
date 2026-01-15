// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/device_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/spec/spec_fwd.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshCommandQueue;
}

// TODO:
// Aggregate all of the free function tensor APIs currently spread across
// - tensor_impl.hpp,
// - tensor_utils.hpp,
// - tensor.hpp.
// (Use the opportunity to remove stale / unused APIs.)

// tensor_impl.hpp
// Looks very internal, there's cleanup effort going.
// https://github.com/tenstorrent/tt-metal/issues/32606
//
// TODO: This list is not completed due to the current cleanup effort.
// Not tensor related:
// - cast_vec (not applicable)
// - element_size_bytes(DataType)
// - packed_buffer_size_bytes(size_t volume_unpacked_data)
// - convert_layout_row_major_to_tile(Shape2D, Tile, Span<T>)
// - convert_layout_tile_to_row_major(same signature as above)
//
// Remotely tensor related?
// - encode_tensor_data(span<T>, TensorSpec, pad)
// - decode_tensor_data(span<T>, TensorSpec)
// - logical_matches_physical(TensorSpec)
// - allocate_device_buffeer
//   River: is this implementation

// tensor_utils.hpp
// - is_cpu_tensor(tensor) (not applicable)
// - is_device_tensor(tensor) (not applicable)
// - cb_descriptor_from_sharded_tensor(cb_index, tensor) <- what is this??

// tensor.hpp

// TODO: should these mutate inplace?
void to_layout(HostTensor&, Layout);
// TODO: why is the pad value a float???, shouldn't this be a T?
void pad(HostTensor&, const Shape&, const Shape&, float);
void unpad(HostTensor&, const Shape&, const Shape&);
// TODO: shouldn't be a flot
void pad_to_tile(HostTensor&, float);
void unpad_to_tile(HostTensor&, const Shape&);

// New APIs
DeviceTensor EnqueueWriteTensor(distributed::MeshCommandQueue&, const HostTensor&);
// Should this return a HostTensor or take a HostTensor& as dst??
void EnqueueReadTensor(distributed::MeshCommandQueue&, const DeviceTensor&);

}  // namespace tt::tt_metal
