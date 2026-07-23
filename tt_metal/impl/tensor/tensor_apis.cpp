// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <string_view>
#include <type_traits>
#include <unordered_set>

#include "host_tensor_impl.hpp"
#include "mesh_tensor_impl.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include "tt_metal/distributed/pinned_memory_cache.hpp"
#include "tt_metal/distributed/mesh_device_view_impl.hpp"
#include <tt_stl/concepts.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr size_t k_pin_write_threshold_bytes = 32 * 1024 * 1024;

bool should_use_pinned_write_path(distributed::MeshDevice& mesh_device, size_t size_bytes) {
    if (size_bytes <= k_pin_write_threshold_bytes) {
        return false;
    }
    const auto params = experimental::GetMemoryPinningParameters(mesh_device);
    return params.max_pins > 0 && params.can_map_to_noc;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

}  // namespace

// ======================================================================================
//                                Uniform Data movement APIs
// ======================================================================================

HostTensor enqueue_read_tensor(distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, bool blocking) {
    auto mesh_buffer = device_tensor.impl().raw_mesh_buffer();
    const auto& device = device_tensor.device();

    auto distributed_host_buffer = DistributedHostBuffer::create(device.get_view());

    distributed::MeshCoordinateRange all_coords(device.shape());
    std::vector<distributed::MeshCoordinate> coords(all_coords.begin(), all_coords.end());
    distributed_host_buffer.emplace_shards(
        coords,
        [&](const distributed::MeshCoordinate&) {
            return tensor_impl::allocate_host_buffer(device_tensor.tensor_spec());
        },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    cq.enqueue_read(mesh_buffer, distributed_host_buffer, /*shards=*/std::nullopt, blocking);

    return host_tensor_from_buffer_with_topology(
        std::move(distributed_host_buffer), device_tensor.tensor_spec(), get_tensor_topology(device_tensor));
}

MeshTensor enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    distributed::MeshDevice& mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config) {
    TT_FATAL(
        is_uniform_write(host_tensor, mesh_device),
        "Incompatible shape between source host tensor and target MeshDevice. For non-uniform transfers, use the "
        "non-uniform data movement APIs.");
    std::optional<TensorSpec> tensor_spec_overriden_memory_config;
    if (memory_config) {
        const auto& old_spec = host_tensor.tensor_spec();
        tensor_spec_overriden_memory_config = TensorSpec(
            old_spec.logical_shape(),
            TensorLayout(
                old_spec.tensor_layout().get_data_type(),
                old_spec.tensor_layout().get_page_config(),
                *memory_config,
                old_spec.tensor_layout().get_alignment()));
    }

    const auto* tensor_spec = tensor_spec_overriden_memory_config.has_value()
                                  ? &tensor_spec_overriden_memory_config.value()
                                  : &host_tensor.tensor_spec();

    auto result =
        allocate_mesh_tensor_on_device_with_topology(mesh_device, *tensor_spec, get_tensor_topology(host_tensor));
    enqueue_write_tensor(cq, host_tensor, result);
    return result;
}

void enqueue_read_tensor(
    distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor, HostTensor& host_tensor, bool blocking) {
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    auto mesh_buffer = device_tensor.impl().raw_mesh_buffer();
    auto* device = mesh_buffer->device();

    DistributedHostBuffer dst_distributed_host_buffer = DistributedHostBuffer::create(device->get_view());
    const size_t expected_per_shard_size_bytes = device_tensor.tensor_spec().compute_packed_buffer_size_bytes();

    distributed::MeshCoordinateRange all_coords(device->shape());
    std::vector<distributed::MeshCoordinate> coords(all_coords.begin(), all_coords.end());
    for (const auto& coord : coords) {
        dst_distributed_host_buffer.emplace_shard(coord, [&]() {
            auto host_buffer = host_tensor.buffer().get_shard(coord);
            TT_FATAL(host_buffer.has_value(), "Host shard for device shard {} is not populated.", coord);
            TT_FATAL(
                host_buffer->view_bytes().size() >= expected_per_shard_size_bytes,
                "Host shard for device shard {} has invalid size: {} < {}",
                coord,
                host_buffer->view_bytes().size(),
                expected_per_shard_size_bytes);

            auto coord_range = distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(coord, coord));
            if (auto pinned = experimental::PinnedMemoryCache::instance().try_pin(
                    *device, coord_range, *host_buffer, /*map_to_noc=*/true)) {
                experimental::HostBufferSetPinnedMemory(*host_buffer, std::move(pinned));
            }
            return *host_buffer;
        });
    }

    cq.enqueue_read(mesh_buffer, dst_distributed_host_buffer, /*shards=*/std::nullopt, blocking);
    update_tensor_topology(host_tensor, get_tensor_topology(device_tensor));
}

void enqueue_write_tensor(distributed::MeshCommandQueue& cq, const HostTensor& host_tensor, MeshTensor& device_tensor) {
    TT_FATAL(
        is_uniform_write(host_tensor, device_tensor.device()),
        "Incompatible shape between source host tensor and target MeshDevice. For non-uniform transfers, use the "
        "non-uniform data movement APIs.");
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    auto mesh_buffer = device_tensor.impl().raw_mesh_buffer();
    const auto& distributed_host_buffer = host_tensor.buffer();

    size_t total_size = 0;
    for (const auto& coord : distributed_host_buffer.shard_coords()) {
        auto buf = distributed_host_buffer.get_shard(coord);
        if (buf) {
            total_size += buf->view_bytes().size();
        }
    }

    const bool use_pinned = CMAKE_UNIQUE_NAMESPACE::should_use_pinned_write_path(*cq.device(), total_size);

    if (use_pinned) {
        auto* mesh_device = mesh_buffer->device();
        const auto& view = mesh_device->get_view();
        std::vector<distributed::ShardDataTransfer> transfers;
        transfers.reserve(distributed_host_buffer.shard_coords().size());
        bool any_pinned = false;

        for (const auto& coord : distributed_host_buffer.shard_coords()) {
            // get_shard yields a buffer only for shards owned by this host, so remote chips are
            // never pinned or added to the transfer list -- the transfer is a no-op for them here.
            auto buf = distributed_host_buffer.get_shard(coord);
            if (buf) {
                // The host buffer's distribution must agree with the device's: host memory can only
                // be pinned to MMIO devices local to this process, so a populated shard for a coord
                // the device owns on another host must never reach try_pin (which would fault while
                // resolving the remote device).
                TT_FATAL(
                    view.impl().is_local(coord),
                    "Host buffer holds a shard for device coordinate {}, but that device is not local "
                    "to this host; host memory can only be pinned to MMIO devices owned by this process.",
                    coord);
                auto coord_range = distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(coord, coord));
                HostBuffer pinned_buf(*buf);
                auto pinned_memory = experimental::PinnedMemoryCache::instance().try_pin(
                    *mesh_device, coord_range, pinned_buf, /*map_to_noc=*/true);

                auto xfer = distributed::ShardDataTransfer{distributed::MeshCoordinate(coord)}
                                .host_data(buf->view_bytes().data())
                                .region(BufferRegion(0, buf->view_bytes().size()));
                if (pinned_memory) {
                    experimental::ShardDataTransferSetPinnedMemory(xfer, std::move(pinned_memory));
                    any_pinned = true;
                }
                transfers.push_back(std::move(xfer));
            }
        }
        if (any_pinned) {
            cq.enqueue_write_shards(mesh_buffer, transfers, /*blocking=*/true);
        } else {
            cq.enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);
        }
    } else {
        cq.enqueue_write(mesh_buffer, distributed_host_buffer, /*blocking=*/false);
    }

    device_tensor = mesh_tensor_from_buffer_with_topology(
        std::move(*mesh_buffer),
        TensorSpec(
            host_tensor.tensor_spec().logical_shape(),
            TensorLayout(
                host_tensor.tensor_spec().tensor_layout().get_data_type(),
                host_tensor.tensor_spec().tensor_layout().get_page_config(),
                device_tensor.memory_config(),
                host_tensor.tensor_spec().tensor_layout().get_alignment())),
        get_tensor_topology(host_tensor));
}

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

void assert_host_shards_match_packed_size(
    const DistributedHostBuffer& buffer, const TensorSpec& spec, std::string_view op_name) {
    const size_t expected_shard_size = spec.compute_packed_buffer_size_bytes();
    for (const auto& coord : buffer.shard_coords()) {
        auto shard = buffer.get_shard(coord);
        if (shard) {
            TT_FATAL(
                shard->view_bytes().size() == expected_shard_size,
                "{} shard size mismatch after conversion: actual {} != expected {}",
                op_name,
                shard->view_bytes().size(),
                expected_shard_size);
        }
    }
}

template <typename T>
HostTensor to_row_major_layout_impl(const HostTensor& tensor) {
    if (tensor.layout() == Layout::ROW_MAJOR) {
        return tensor;
    }

    TT_FATAL(tensor.layout() == Layout::TILE, "Converting from {} to Row Major is unsupported.", tensor.layout());

    // Do not pass a non-default tile into ROW_MAJOR PageConfig (deprecation #18536).
    auto output_spec = TensorSpec(
        tensor.logical_shape(),
        TensorLayout(
            tensor.dtype(),
            PageConfig(Layout::ROW_MAJOR),
            tensor.memory_config(),
            tensor.tensor_spec().tensor_layout().get_alignment()));

    TT_FATAL(
        output_spec.physical_shape() == tensor.tensor_spec().physical_shape(),
        "to_layout: Converting layout to {} implicitly changed physical shape from {} to {} due to alignment "
        "constraints. This is currently unsupported. Please pad the tensor explicitly before conversion.",
        Layout::ROW_MAJOR,
        tensor.tensor_spec().physical_shape(),
        output_spec.physical_shape());

    auto tile = tensor.tensor_spec().tile();
    auto physical_shape = tensor.tensor_spec().physical_shape();

    auto transformed_buffer = tensor.buffer().transform(
        [&](const HostBuffer& buffer) {
            auto input_data = buffer.view_as<T>();
            auto rm_data = tensor_impl::to_row_major_layout(physical_shape, tile, input_data);
            return HostBuffer(std::move(rm_data));
        },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    assert_host_shards_match_packed_size(transformed_buffer, output_spec, "to_row_major_layout");

    return host_tensor_from_buffer_with_topology(
        std::move(transformed_buffer), output_spec, get_tensor_topology(tensor));
}

template <typename T>
HostTensor to_tile_layout_impl(const HostTensor& tensor, Tile tile) {
    if (tensor.layout() == Layout::TILE) {
        return tensor;
    }

    if constexpr (std::is_same_v<T, float8_e4m3>) {
        // FP8_E4M3 is constrained to ROW_MAJOR, so tilizing it is a caller error.
        TT_THROW("to_layout: FP8_E4M3 only supports ROW_MAJOR layout (got target {})", Layout::TILE);
    } else {
        TT_FATAL(tensor.layout() == Layout::ROW_MAJOR, "Converting from {} to Tile is unsupported.", tensor.layout());

        auto output_spec = TensorSpec(
            tensor.logical_shape(),
            TensorLayout(
                tensor.dtype(),
                PageConfig(Layout::TILE, tile),
                tensor.memory_config(),
                tensor.tensor_spec().tensor_layout().get_alignment()));

        TT_FATAL(
            output_spec.physical_shape() == tensor.tensor_spec().physical_shape(),
            "to_layout: Converting layout to {} implicitly changed physical shape from {} to {} due to alignment "
            "constraints. This is currently unsupported. Please pad the tensor explicitly before conversion.",
            Layout::TILE,
            tensor.tensor_spec().physical_shape(),
            output_spec.physical_shape());

        auto physical_shape = tensor.tensor_spec().physical_shape();

        auto transformed_buffer = tensor.buffer().transform(
            [&](const HostBuffer& buffer) {
                auto input_data = buffer.view_as<T>();
                auto tilized_data = tensor_impl::to_tile_major_layout(physical_shape, tile, input_data);
                return HostBuffer(std::move(tilized_data));
            },
            DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

        assert_host_shards_match_packed_size(transformed_buffer, output_spec, "to_tile_layout");

        return host_tensor_from_buffer_with_topology(
            std::move(transformed_buffer), output_spec, get_tensor_topology(tensor));
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostTensor to_row_major_layout(const HostTensor& tensor) {
    return tensor_impl::dispatch(tensor.dtype(), [&]<typename T>() {
        if constexpr (
            std::is_same_v<T, tensor_impl::bfloat4_b> || std::is_same_v<T, tensor_impl::bfloat8_b> ||
            std::is_same_v<T, float8_e4m3>) {
            // bfloat4_b / bfloat8_b: TODO(#43763):
            // Flipping this assert to TT_FATAL triggers multiple failures in **sanity** test suite.
            // This silent fail has a high impact area and should be studied and addressed asap.
            //
            // Original comment:
            // TODO: Flip to assert when we remove use cases in python and c++
            //
            // FP8_E4M3 is constrained to ROW_MAJOR at construction, so it is already row-major.
            return tensor;
        } else {
            return CMAKE_UNIQUE_NAMESPACE::to_row_major_layout_impl<T>(tensor);
        }
    });
}

HostTensor to_tile_layout(const HostTensor& tensor, const Tile& tile) {
    // Reject mismatched retile before dtype dispatch so BFP cannot silently identity-return.
    if (tensor.layout() == Layout::TILE) {
        TT_FATAL(
            tile == tensor.tensor_spec().tile(),
            "to_tile_layout: requested tile {} does not match input tile {}. Retile is not supported.",
            tile,
            tensor.tensor_spec().tile());
    }
    return tensor_impl::dispatch(tensor.dtype(), [&]<typename T>() {
        if constexpr (std::is_same_v<T, tensor_impl::bfloat4_b> || std::is_same_v<T, tensor_impl::bfloat8_b>) {
            // Block-float formats are natively TILE — no conversion needed.
            return tensor;
        } else {
            return CMAKE_UNIQUE_NAMESPACE::to_tile_layout_impl<T>(tensor, tile);
        }
    });
}

HostTensor to_layout(const HostTensor& tensor, Layout target_layout) {
    switch (target_layout) {
        case Layout::ROW_MAJOR: return to_row_major_layout(tensor);
        case Layout::TILE: return to_tile_layout(tensor, tensor.tensor_spec().tile());
        default: TT_THROW("Target layout {} is not supported", target_layout);
    }
}

// ======================================================================================
//                                  .to_dtype()
// ======================================================================================

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

struct bfloat4_tag {};
struct bfloat8_tag {};

// Preprocess the storage to unpack the bfloat8/4 tiles into float32.
tt::tt_metal::DistributedHostBuffer preprocess_buffers(
    const tt::tt_metal::DistributedHostBuffer& input_storage,
    const DataType input_dtype,
    const tt::tt_metal::Tile& tile) {
    constexpr bool row_major_output = false;
    constexpr bool is_exp_a = false;

    if (input_dtype == DataType::BFLOAT8_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            ttsl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a, tile);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    }
    if (input_dtype == DataType::BFLOAT4_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            ttsl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp4_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a, tile);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    }
    return input_storage;
}

template <typename SrcType, typename DstType>
tt::tt_metal::DistributedHostBuffer transform_buffers(
    const tt::tt_metal::TensorSpec& input_tensor_spec,
    const tt::tt_metal::TensorSpec& output_spec,
    const tt::tt_metal::DistributedHostBuffer& input_buffer) {
    if constexpr (std::is_same_v<SrcType, DstType>) {
        return input_buffer;
    } else if constexpr (std::is_same_v<SrcType, float8_e4m3> || std::is_same_v<DstType, float8_e4m3>) {
        // FP8_E4M3 only has a direct bridge to/from FLOAT32 (operator float() and the float
        // constructor in float8.hpp). Other dtypes would need a float pivot, which is not wired
        // up yet because the only host-side consumer today is the print path in tensor_impl.cpp,
        // which already converts through FLOAT32. Add the broader lattice when a use case appears.
        if constexpr (
            (std::is_same_v<SrcType, float8_e4m3> && std::is_same_v<DstType, float>) ||
            (std::is_same_v<SrcType, float> && std::is_same_v<DstType, float8_e4m3>)) {
            auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
                auto data = buffer.view_as<const SrcType>();
                std::vector<DstType> output_vector(data.size());
                std::transform(data.begin(), data.end(), output_vector.begin(), [](SrcType value) {
                    return static_cast<DstType>(value);
                });
                return tt::tt_metal::HostBuffer(std::move(output_vector));
            };
            return input_buffer.transform(transform_fn);
        } else {
            TT_THROW("to_dtype: FP8_E4M3 cross-type conversion is only supported to/from FLOAT32");
            return input_buffer;  // unreachable, satisfies return type
        }
    } else if constexpr (std::is_same_v<DstType, bfloat4_tag> || std::is_same_v<DstType, bfloat8_tag>) {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            ttsl::Span<const SrcType> data = buffer.view_as<const SrcType>();
            std::vector<SrcType> tilized_data;  // empty if `data` is already in tile layout.
            if (input_tensor_spec.layout() == Layout::ROW_MAJOR) {
                tilized_data =
                    tensor_impl::to_tile_major_layout(output_spec.physical_shape(), output_spec.tile(), data);
                data = ttsl::make_const_span(tilized_data);
            }

            auto float_packed_data = [&]() {
                constexpr bool row_major_input = false;
                constexpr bool is_exp_a = false;
                if constexpr (std::is_same_v<DstType, bfloat8_tag>) {
                    return pack_as_bfp8_tiles(data, row_major_input, is_exp_a, output_spec.tile());
                } else if constexpr (std::is_same_v<DstType, bfloat4_tag>) {
                    return pack_as_bfp4_tiles(data, row_major_input, is_exp_a, output_spec.tile());
                } else {
                    static_assert(ttsl::concepts::always_false_v<DstType>, "Unsupported data type");
                }
            }();
            return tt::tt_metal::HostBuffer(std::move(float_packed_data));
        };

        return input_buffer.transform(transform_fn);
    } else {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            auto data = buffer.view_as<const SrcType>();
            std::vector<DstType> output_vector(data.size());
            std::transform(data.begin(), data.end(), output_vector.begin(), [](SrcType value) {
                return static_cast<DstType>(value);
            });
            return tt::tt_metal::HostBuffer(std::move(output_vector));
        };

        return input_buffer.transform(transform_fn);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostTensor to_dtype(const HostTensor& input_tensor, DataType dtype) {
    const auto src_type = input_tensor.dtype();
    if (src_type == dtype) {
        return input_tensor;
    }

    const size_t expected_input_shard_size = input_tensor.tensor_spec().compute_packed_buffer_size_bytes();
    for (const auto& coord : input_tensor.buffer().shard_coords()) {
        auto shard = input_tensor.buffer().get_shard(coord);
        if (shard) {
            TT_FATAL(
                shard->view_bytes().size() == expected_input_shard_size,
                "to_dtype input shard size mismatch before conversion: actual {} != expected {}",
                shard->view_bytes().size(),
                expected_input_shard_size);
        }
    }

    auto input_buffer =
        CMAKE_UNIQUE_NAMESPACE::preprocess_buffers(input_tensor.buffer(), src_type, input_tensor.tensor_spec().tile());

    const auto layout =
        (dtype == DataType::BFLOAT4_B || dtype == DataType::BFLOAT8_B) ? Layout::TILE : input_tensor.layout();

    tt::tt_metal::PageConfig page_config(layout, input_tensor.tensor_spec().tile());

    auto output_spec = TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            dtype,
            page_config,
            input_tensor.tensor_spec().memory_config(),
            input_tensor.tensor_spec().tensor_layout().get_alignment()));

    TT_FATAL(
        input_tensor.tensor_spec().physical_shape() == output_spec.physical_shape(),
        "to_dtype: Converting layout to {} implicitly changed physical shape from {} to {} due to alignment "
        "constraints. This is currently unsupported. Please pad the tensor explicitly before conversion.",
        layout,
        input_tensor.tensor_spec().physical_shape(),
        output_spec.physical_shape());

    auto output_storage = [src_type, dst_type = dtype, &input_tensor, &input_buffer, &output_spec]() {
        auto with_src_and_dst = [&]<typename SrcType, typename DstType>() {
            return CMAKE_UNIQUE_NAMESPACE::transform_buffers<SrcType, DstType>(
                input_tensor.tensor_spec(), output_spec, input_buffer);
        };

        auto with_src = [dst_type, &with_src_and_dst]<typename SrcType>() {
            switch (dst_type) {
                case DataType::BFLOAT4_B:
                    return with_src_and_dst.operator()<SrcType, CMAKE_UNIQUE_NAMESPACE::bfloat4_tag>();
                case DataType::BFLOAT8_B:
                    return with_src_and_dst.operator()<SrcType, CMAKE_UNIQUE_NAMESPACE::bfloat8_tag>();
                case DataType::FLOAT32: return with_src_and_dst.operator()<SrcType, float>();
                case DataType::BFLOAT16: return with_src_and_dst.operator()<SrcType, bfloat16>();
                case DataType::UINT8: return with_src_and_dst.operator()<SrcType, uint8_t>();
                case DataType::UINT16: return with_src_and_dst.operator()<SrcType, uint16_t>();
                case DataType::UINT32: return with_src_and_dst.operator()<SrcType, uint32_t>();
                case DataType::INT32: return with_src_and_dst.operator()<SrcType, int32_t>();
                case DataType::FP8_E4M3: return with_src_and_dst.operator()<SrcType, float8_e4m3>();
                case DataType::INVALID: TT_THROW("Unsupported data type conversion requested. Source type is invalid!");
            }
            TT_THROW("Unreachable");
        };

        switch (src_type) {
            case DataType::BFLOAT4_B:
            case DataType::BFLOAT8_B:
            case DataType::FLOAT32: return with_src.operator()<float>();
            case DataType::BFLOAT16: return with_src.operator()<bfloat16>();
            case DataType::UINT8: return with_src.operator()<uint8_t>();
            case DataType::UINT16: return with_src.operator()<uint16_t>();
            case DataType::UINT32: return with_src.operator()<uint32_t>();
            case DataType::INT32: return with_src.operator()<int32_t>();
            case DataType::FP8_E4M3: return with_src.operator()<float8_e4m3>();
            case DataType::INVALID: TT_THROW("Unsupported data type conversion requested. Source type is invalid!");
        }
        TT_THROW("Unreachable");
    }();

    auto result_buffer = std::move(output_storage);
    const size_t expected_shard_size = output_spec.compute_packed_buffer_size_bytes();
    for (const auto& coord : result_buffer.shard_coords()) {
        auto shard = result_buffer.get_shard(coord);
        if (shard) {
            TT_FATAL(
                shard->view_bytes().size() == expected_shard_size,
                "to_dtype shard size mismatch after conversion: actual {} != expected {}",
                shard->view_bytes().size(),
                expected_shard_size);
        }
    }

    return host_tensor_from_buffer_with_topology(
        std::move(result_buffer), output_spec, get_tensor_topology(input_tensor));
}

// ======================================================================================
//                                  Utility functions
// ======================================================================================

bool logical_matches_physical(const TensorSpec& tensor_spec) {
    return tensor_spec.layout() == Layout::ROW_MAJOR && tensor_spec.logical_2d_shape() == tensor_spec.physical_shape();
}

namespace host_buffer {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <typename T>
void validate_datatype(DataType dtype) {
    using BaseType = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<BaseType, uint32_t>) {
        TT_FATAL(
            dtype == DataType::UINT32 or dtype == DataType::BFLOAT8_B or dtype == DataType::BFLOAT4_B,
            "Incorrect data type {}",
            dtype);
    } else if constexpr (std::is_same_v<BaseType, int32_t>) {
        TT_FATAL(dtype == DataType::INT32, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, float>) {
        TT_FATAL(dtype == DataType::FLOAT32, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, bfloat16>) {
        TT_FATAL(dtype == DataType::BFLOAT16, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, uint16_t>) {
        TT_FATAL(dtype == DataType::UINT16, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, uint8_t>) {
        TT_FATAL(dtype == DataType::UINT8, "Incorrect data type {}", dtype);
    } else {
        static_assert(sizeof(BaseType) == 0, "Unsupported DataType");
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostBuffer get_host_buffer(const HostTensor& tensor) {
    std::vector<HostBuffer> buffers;
    tensor.buffer().apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
    TT_FATAL(
        buffers.size() == 1,
        "Can't get a single buffer from host storage distributed over mesh shape {}",
        tensor.buffer().shape());
    return buffers.front();
}

template <typename T>
ttsl::Span<const T> get_as(const HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
ttsl::Span<T> get_as(HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
ttsl::Span<const T> get_as(const HostTensor& tensor) {
    CMAKE_UNIQUE_NAMESPACE::validate_datatype<T>(tensor.dtype());
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

template <typename T>
ttsl::Span<T> get_as(HostTensor& tensor) {
    CMAKE_UNIQUE_NAMESPACE::validate_datatype<T>(tensor.dtype());
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

// Explicit template instantiations
#define INSTANTIATE_HOST_BUFFER_FUNCTIONS(T)                         \
    template ttsl::Span<const T> get_as<T>(const HostBuffer&);       \
    template ttsl::Span<const T> get_as<const T>(const HostBuffer&); \
    template ttsl::Span<T> get_as<T>(HostBuffer&);                   \
    template ttsl::Span<const T> get_as<const T>(HostBuffer&);       \
    template ttsl::Span<const T> get_as<T>(const HostTensor&);       \
    template ttsl::Span<const T> get_as<const T>(const HostTensor&); \
    template ttsl::Span<T> get_as<T>(HostTensor&);                   \
    template ttsl::Span<const T> get_as<const T>(HostTensor&);

INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(int32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(float)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(bfloat16)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint16_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint8_t)

#undef INSTANTIATE_HOST_BUFFER_FUNCTIONS

}  // namespace host_buffer

}  // namespace tt::tt_metal
