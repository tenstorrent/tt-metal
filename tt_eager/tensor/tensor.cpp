// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor.hpp"
#include <memory>

#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_impl_wrapper.hpp"
#include "tensor/tensor_utils.hpp"
#include "common/bfloat16.hpp"
#include "llrt/llrt.hpp"
#include "tensor/types.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"

#include "tt_metal/tt_stl/reflection.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "queue/queue.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor::Tensor(const Storage storage, const ttnn::Shape shape, DataType dtype, Layout layout) : tensor_attributes(std::make_shared<TensorAttributes>(storage, shape, dtype, layout)) {
    this->set_metadata_populated();
    std::visit(
        [&] (auto&& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                // do nothing
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_ASSERT(storage.buffer->device() != nullptr);
                workers = {device()};
                tensor_impl::validate_on_device_dtype_and_layout(storage.buffer->device(), dtype, layout);
                // Increment main thread ref count for all tensors on device
                this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                // do nothing
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                workers.reserve(storage.buffers.size());
                for (const auto& buffer : storage.buffers) {
                    TT_ASSERT(buffer->device() != nullptr);
                    tensor_impl::validate_on_device_dtype_and_layout(buffer->device(), dtype, layout);
                    workers.push_back(buffer->device());
                }
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                // do nothing
            }else {
                raise_unsupported_storage<StorageType>();
            }
        },
        this->get_storage()
    );
}

Tensor::Tensor(const Storage storage, const Shape shape, DataType dtype, Layout layout) :
    Tensor(storage, ttnn::Shape{shape}, dtype, layout) {}

Tensor::~Tensor() {
    this->deallocate_through_destructor = true;
    this->deallocate();
    // Decrement main thread ref count for all tensors on device
    if (this->workers.size()) {
        this->tensor_attributes->decrement_main_thread_ref_count(this->workers.at(0));
    }
    tensor_attributes.reset();
}

void Tensor::deallocate(bool force) {
    if (this->tensor_attributes.use_count()) {
        // Check if the attributes didn't get moved to another tensor.
        // If not, we can deallocate this tensor.
        if (this->tensor_attributes->dynamic_storage) {
            // Tensor was populated with autoformat. Storage type can
            // change based on op behaviour. Wait for tensor populated.
            this->wait_for_metadata_populated();
        }
        std::visit(
                [force, this](auto& storage) {
                    using T = std::decay_t<decltype(storage)>;
                    if constexpr (std::is_same_v<T, OwnedStorage>) {
                        if (this->tensor_attributes.use_count() == 1) {
                            std::visit([](auto&& buffer) { buffer.reset(); }, storage.buffer);
                        }
                    } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                        if (this->workers.at(0)->in_main_thread()) {
                            // If owned by the main thread, deallocate this tensor only from the main thread
                            uint32_t ref_count_to_use = (this->workers.at(0)->get_worker_mode() == WorkExecutorMode::SYNCHRONOUS) ? this->tensor_attributes.use_count() : this->tensor_attributes->main_thread_ref_count;
                            if ((force or ref_count_to_use == 1) and not this->tensor_attributes->deallocated) {
                                this->tensor_attributes->deallocated = true;
                                this->workers.at(0)->push_work([force, *this] () mutable {
                                    std::visit([force, this] (auto&& s) {
                                        using type = std::decay_t<decltype(s)>;
                                        if constexpr (std::is_same_v<type, DeviceStorage>) {
                                            if (force or s.buffer.use_count() == 1) {
                                                DeallocateBuffer(*(s.buffer));
                                            }
                                            // Safe to reset this buf object since this is the last reference (in the main thread) to the tensor attr object holding this buffer.
                                            // If any other tensor handles hold this buffer, it will not be deleted, until the last handle goes out of scope
                                            // or is deallocated.
                                            s.buffer.reset();
                                        }
                                    }, this->tensor_attributes->storage);
                                });
                            }
                        } else {
                            TT_FATAL(this->deallocate_through_destructor, "Device tensors cannot be explictly deallocated in worker threads.");
                        }
                    } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                        if (force) {
                            TT_THROW("Cannot deallocate tensor with borrowed storage!");
                        }
                    } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                        for (auto& buffer : storage.buffers) {
                            if (force or (this->tensor_attributes.use_count() == 1 and buffer.use_count() == 1)) {
                                // Same logic as above for device buffers
                                DeallocateBuffer(*buffer);
                            }
                            if (force or this->tensor_attributes.use_count() == 1) {
                                // Same logic as above for host buffer copies
                                buffer.reset();
                            }
                        }
                    } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                        if (this->tensor_attributes.use_count() == 1) {
                            // Same logic as above for host tensors
                            for (auto& current_buffer : storage.buffers) {
                                std::visit([](auto&& buffer) { buffer.reset(); }, current_buffer);
                            }
                        }
                    } else {
                        raise_unsupported_storage<T>();
                    }
                },
        this->tensor_attributes->storage);
    }
}

bool Tensor::metadata_populated() const {
    // Make a list of boolens for storage
    return (this->tensor_attributes->metadata_populated).load();
}

void Tensor::set_metadata_populated() {
    (this->tensor_attributes->metadata_populated).store(true);
}

void Tensor::wait_for_metadata_populated() const {
    while (not this->metadata_populated());
}

void Tensor::deepcopy(const Tensor& other) {
    // Wait until the tensor being copied is populated
    other.wait_for_metadata_populated();
    // Populate tensor metadata
    this->set_shape(other.get_shape());
    this->set_storage(other.get_storage());
    this->set_dtype(other.get_dtype());
    this->set_layout(other.get_layout());
    // Set metadata populated flag for getters
    this->set_metadata_populated();
}

void Tensor::populate_buffers_and_metadata(const Tensor& other) {
    // Similar to deepcopy, but to be applied on a tensor that has an empty storage
    // container initialized. Require tensor storage to be correctly initialized.
    // Populate storage container with buffers + shapes
    std::visit([this] (auto&& storage) {
        using StorageType = std::decay_t<decltype(storage)>;
        if constexpr(std::is_same_v<StorageType, OwnedStorage> or std::is_same_v<StorageType, DeviceStorage>) {
            std::get<StorageType>(this->tensor_attributes->storage).buffer = storage.buffer;
        } else if constexpr(std::is_same_v<StorageType, MultiDeviceHostStorage> or std::is_same_v<StorageType, MultiDeviceStorage>) {
            std::get<StorageType>(this->tensor_attributes->storage).buffers = storage.buffers;
            std::get<StorageType>(this->tensor_attributes->storage).shapes = storage.shapes;
        }
    }, other.get_storage()); // Non blocking storage query, since this is done for tensors that get created inside the worker thread
    // Populate remaining MD
    this->set_shape(other.get_shape());
    this->set_dtype(other.get_dtype());
    this->set_layout(other.get_layout());
    this->set_metadata_populated();
}

std::vector<Device*> Tensor::get_workers(bool blocking) const {
    // Initialize an empty worker vector (remains empty for host side storage)
    std::vector<Device*> workers = {};

    if (this->tensor_attributes->dynamic_storage) {
        // Tensor is populated by launch_with_autoformat
        // Storage type can change based on op behaviour, wait until tensor populated.
        this->wait_for_metadata_populated();
    }

    std::visit([this, blocking, &workers] (auto&& storage) {
        using StorageType = std::decay_t<decltype(storage)>;
        // Assign workers only to device tensors
        if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
            // Either explictly syncing or workers are pre-populated (this will happen for device tensors if using the correct APIs).
            TT_FATAL(blocking or (this->workers.size() == 1), "Worker Handles for tensor must be populated or blocking = true must be set in get_workers().");
            if (this->workers.size() != 1) {
                // Not populated - sync.
                this->wait_for_metadata_populated();
                workers = {this->device()};
            } else {
                // Already populated.
                workers = this->workers;
            }
        } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
            // Either explictly syncing or workers are pre-populated (this will happen for device tensors if using the correct APIs).
            TT_FATAL(blocking or (this->workers.size() == storage.buffers.size()), "Worker Handles for tensor must be populated or blocking = true must be set in get_workers().");
            if (this->workers.size() != storage.buffers.size()) {
                // Not populated - sync.
                this->wait_for_metadata_populated();
                workers.reserve(storage.buffers.size());
                for (const auto& buffer : storage.buffers) {
                    // Already populated.
                    workers.push_back(buffer->device());
                }
            } else {
                workers = this->workers;
            }
        }
    }, this->tensor_attributes->storage);
    return workers;
}

// Getters - Spin until tensor is populated before querying tensor metadata
const Tensor::TensorAttributes& Tensor::get_attr() const {
    this->wait_for_metadata_populated();
    return *tensor_attributes;
}

const Shape& Tensor::get_legacy_shape() const {
    return this->get_attr().shape.value();
}

const ttnn::Shape& Tensor::get_shape() const {
    return this->get_attr().shape;
}
const DataType& Tensor::get_dtype() const {
    return this->get_attr().dtype;
}
const Layout& Tensor::get_layout() const {
    return this->get_attr().layout;
}

const Storage& Tensor::get_storage() const {
    // Per device bool for storage population for multidevice
    return this->get_attr().storage;
}

Tensor Tensor::to(CommandQueue & queue, const MemoryConfig & mem_config) const {
    ZoneScoped;
    if (storage_type() == StorageType::DEVICE) {
        TT_ASSERT(this->device() == queue.device() && "Currently do not support moving between devices");
        return *this;
    }
    tensor_impl::validate_on_device_dtype_and_layout(queue.device(), this->get_dtype(), this->get_layout());
    return tensor_impl::to_device_wrapper(*this, queue.device(), mem_config, queue);
}

Tensor Tensor::to(Device *target_device, const MemoryConfig &mem_config) const {
    ZoneScoped;
    // Populate device storage outside of thread, so that downstream
    // functions running in main can get storage type without blocking
    Tensor device_tensor({target_device});
    device_tensor.tensor_attributes->storage = DeviceStorage();
    // Record main thread ref count for tensors before pushing to queue.
    uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t original_tensor_ref_count =this->tensor_attributes->record_main_thread_ref_count();
    target_device->push_work([*this, device_tensor, mem_config, target_device] () mutable {
        if (this->storage_type() == StorageType::DEVICE) {
            TT_ASSERT(this->device() == target_device && "Currently do not support moving between devices");
            device_tensor.populate_buffers_and_metadata(*this);
        }
        else {
            tensor_impl::validate_on_device_dtype_and_layout(target_device, this->get_dtype(), this->get_layout());
            auto local_tensor = tensor_impl::to_device_wrapper(*this, target_device, mem_config);
            // Populate device tensor
            device_tensor.populate_buffers_and_metadata(local_tensor);
        }
    });
    // Update main thread ref count for tensors after pushing to queue (update original tensor and returned tensor,
    // since both can be on device).
    device_tensor.tensor_attributes->update_main_thread_ref_count(device_tensor.workers.at(0), device_tensor_ref_count);
    this->tensor_attributes->update_main_thread_ref_count(device_tensor.workers.at(0), original_tensor_ref_count);
    return device_tensor;
}

Tensor Tensor::to(DeviceMesh *device_mesh, const MemoryConfig &mem_config) const {
    ZoneScoped;
    TT_FATAL(device_mesh->get_devices().at(0)->get_worker_mode() == WorkExecutorMode::SYNCHRONOUS, "Async mode is not currently supported for multi-device tensors");
    if (storage_type() == StorageType::MULTI_DEVICE_HOST) {
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(this->get_storage());
        std::vector<DeviceBuffer> device_buffers;

        for (int i = 0; i < host_storage.buffers.size(); ++i) {
            Device& target_device = device_mesh->get_device(i);
            auto shard = Tensor{OwnedStorage{host_storage.buffers[i]},  host_storage.shapes[i], this->get_dtype(), this->get_layout()};
            shard = shard.to(&target_device, mem_config);
            device_buffers.push_back(std::get<DeviceStorage>(shard.get_storage()).buffer);
        }
        Tensor multi_device_tensor = Tensor(
            MultiDeviceStorage{std::move(device_buffers), host_storage.shapes},
            this->get_shape(),
            this->get_dtype(),
            this->get_layout());
        multi_device_tensor.workers = device_mesh->get_devices();
        return multi_device_tensor;
    } else if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(this->get_storage())) {
        return *this; // already on device
    }

    TT_THROW("Tensor::to(...) requires the tensor the be multi-device tensor.");
    return *this;
}

Tensor Tensor::cpu(bool blocking) const {
    ZoneScoped;
    auto workers = this->get_workers(blocking);
    if (not workers.size()) {
        // Tensor is on host and does not have a worker group.
        // Return immediately. If this is a result of .cpu() called twice,
        // tensor accessors will stall until tensor is populated.
        return *this;
    }

    Tensor host_tensor;
    // Populate host storage outside of thread, so that downstream
    // functions running in main can get storage type without blocking
    if (workers.size() == 1) {
        host_tensor.tensor_attributes->storage = OwnedStorage();
    } else {
        host_tensor.tensor_attributes->storage = MultiDeviceHostStorage();
    }
    // Record main_thread_ref_count for tensor before pushing to queue.
    uint32_t original_tensor_ref_count = this->tensor_attributes->record_main_thread_ref_count();
    workers.at(0)->push_work([*this, host_tensor, blocking] () mutable {
        TT_ASSERT(this->storage_type() == StorageType::DEVICE or this->storage_type() == StorageType::MULTI_DEVICE, "Can only use worker queue for cpu call if tensor is on device.");
        auto local_tensor = tensor_impl::to_host_wrapper(*this, blocking);
        // Populate host tensor
        host_tensor.populate_buffers_and_metadata(local_tensor);
    }, blocking);

    // Update main_thread_ref_count for tensor after pushing to queue.
    this->tensor_attributes->update_main_thread_ref_count(workers.at(0), original_tensor_ref_count);
    return host_tensor;
}

Tensor Tensor::cpu_sharded() const {
    ZoneScoped;
    return tensor_impl::to_host_wrapper_sharded(*this);
}


Tensor Tensor::extract_shard(const CoreCoord & core) const{

    auto buffer= this->buffer();
    uint32_t core_id = buffer->core_to_core_id().at(core);
    return this->extract_shard(core_id);
}

Tensor Tensor::extract_shard(const uint32_t & core_id) const{

    return tensor_impl::to_extract_shard_wrapper(*this, core_id);

}

Tensor Tensor::to(Layout target_layout) const {
    ZoneScoped;
    TT_ASSERT(this->storage_type() != StorageType::DEVICE && "Bring tensor to host before converting to target layout");
    return tensor_impl::to_layout_wrapper(*this, target_layout);
}

const std::string Tensor::write_to_string() const { return tensor_impl::to_string_wrapper(*this); }

void Tensor::print() const { std::cout << write_to_string() << std::endl; }

Tensor Tensor::pad(const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value) const {
    ZoneScoped;
    TT_ASSERT(
        this->storage_type() == StorageType::OWNED or
        this->storage_type() == StorageType::MULTI_DEVICE_HOST or
        this->storage_type() == StorageType::BORROWED && "Tensor must be on host for padding");
    TT_ASSERT(this->get_layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for padding");

    auto input_shape = this->get_legacy_shape();
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape.rank(); index++) {
        auto front = input_tensor_start[index];
        auto back = output_tensor_shape[index] - (input_tensor_start[index] + input_shape[index]);
        dimensions_pads.push_back(Padding::PadDimension{.front=front, .back=back});
    }
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto output_shape_with_padding = Shape(output_tensor_shape, padding);

    return tensor_impl::pad_wrapper(*this, output_shape_with_padding, input_tensor_start, pad_value);
}

Tensor Tensor::unpad(const Shape &output_tensor_start, const Shape &output_tensor_end) const {
    ZoneScoped;
    TT_ASSERT(this->get_layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for unpadding");
    return tensor_impl::unpad_wrapper(*this, output_tensor_start, output_tensor_end);
}

Tensor Tensor::pad_to_tile(float pad_value) const {
    ZoneScoped;
    uint32_t h = this->get_legacy_shape()[2];
    uint32_t w = this->get_legacy_shape()[3];
    uint32_t padded_h = round_up(h, TILE_HEIGHT);
    uint32_t padded_w = round_up(w, TILE_WIDTH);

    auto padding = Padding({{0, 0}, {0, 0}, {0, padded_h - h}, {0, padded_w - w}}, Padding::PadValue::Any);

    Shape output_tensor_shape = Shape({this->get_legacy_shape()[0], this->get_legacy_shape()[1], padded_h, padded_w}, padding);
    Shape input_tensor_start = {0, 0, 0, 0};

    return this->pad(output_tensor_shape, input_tensor_start, pad_value);
}

Tensor Tensor::unpad_from_tile(const Shape &output_tensor_shape) const {
    ZoneScoped;

    TT_ASSERT(this->get_legacy_shape()[0] == output_tensor_shape[0] && this->get_legacy_shape()[1] == output_tensor_shape[1], "Input shape must match output shape apart from last 2 dims");
    TT_ASSERT(this->get_legacy_shape()[2] % TILE_HEIGHT == 0 && this->get_legacy_shape()[3] % TILE_WIDTH==0, "Last 2 dims of input shape must be multiples of 32");
    TT_ASSERT(this->get_legacy_shape()[2] - TILE_HEIGHT < output_tensor_shape[2] && this->get_legacy_shape()[3] - TILE_WIDTH < output_tensor_shape[3], "Last 2 dims of output must be within range to have been padded to input");
    Shape output_tensor_start = {0, 0, 0, 0};
    Shape output_tensor_end = {output_tensor_shape[0] - 1, output_tensor_shape[1] - 1, output_tensor_shape[2] - 1, output_tensor_shape[3] - 1};
    return this->unpad(output_tensor_start, output_tensor_end);
}

uint32_t Tensor::element_size() const {
    return tensor_impl::element_size_bytes_wrapper(this->get_dtype());
}

Tensor Tensor::reshape(int N, int C, int H, int W) const {
    auto new_shape = infer_dims_for_reshape(N, C, H, W, this->volume());
    return this->reshape(new_shape);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    TT_ASSERT(
        this->volume() == tt::tt_metal::compute_volume(new_shape),
        "{} != {}",
        this->volume(),
        tt::tt_metal::compute_volume(new_shape));
    if (this->get_layout() == Layout::TILE) {
        TT_ASSERT(new_shape[-2] % TILE_HEIGHT == 0 && new_shape[-1] % TILE_WIDTH == 0 && "Expected a multiple of 32 for H, W (or -1 evaluating to such) in Tensor::reshape()!");
    }

    const auto& tensor = *this;
    return Tensor(tensor.get_storage(), new_shape, tensor.get_dtype(), tensor.get_layout());
}

bool Tensor::is_allocated() const {
    return std::visit(
        [](auto&& storage) -> bool
        {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return std::visit([](auto&& buffer) -> bool { return buffer.is_allocated(); }, storage.buffer);
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return bool(storage.buffer) and storage.buffer->size() > 0;
            }
            else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return true;
            }
            else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                bool is_allocated = true;
                for (const auto& buffer : storage.buffers) {
                    is_allocated &= std::visit([](auto&& buffer) -> bool { return buffer.is_allocated(); }, buffer);
                }
                return is_allocated;
            }
            else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                bool is_allocated = true;
                for (const auto& buffer : storage.buffers) {
                    is_allocated &= bool(buffer) and buffer->size() > 0;
                }
                return is_allocated;
            }
            else {
                raise_unsupported_storage<T>();
            }
        },
        this->get_storage()
    );
}

std::vector<uint32_t> Tensor::host_page_ordering(){
    auto cores = buffer()->all_cores();
    auto shard_size = buffer()->shard_spec().size();
    auto num_pages = cores.size() * shard_size;

    std::vector<uint32_t> ret_vec;
    ret_vec.reserve(num_pages);
    for(int page_id = 0; page_id <num_pages ; page_id++){
        ret_vec.push_back(buffer()->get_dev_to_host_mapped_page_id(page_id));
    }
    return ret_vec;
}

StorageType Tensor::storage_type() const {
    return std::visit(
        [] (auto&& storage) -> StorageType
        {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return StorageType::OWNED;
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return StorageType::DEVICE;
            }
            else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return StorageType::BORROWED;
            }
            else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                return StorageType::MULTI_DEVICE;
            }
            else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                return StorageType::MULTI_DEVICE_HOST;
            }
            else {
                raise_unsupported_storage<T>();
            }
        },
        this->get_storage()
    );
}

namespace detail {
const Shape compute_strides(const Shape& shape) {
    auto num_elements = compute_volume(shape);
    std::vector<std::uint32_t> strides;
    for (std::int32_t index = 0; index < shape.rank(); index++) {
        num_elements /= shape[index];
        strides.push_back(num_elements);
    }
    return strides;
}
}

const Shape Tensor::strides() const {
    return detail::compute_strides(this->get_legacy_shape());
}

uint32_t Tensor::volume() const { return tt::tt_metal::compute_volume(this->get_legacy_shape()); }

Tensor create_device_tensor(const Shape& shape, DataType data_type, Layout layout, Device *device, const MemoryConfig& memory_config) {
    ZoneScoped;
    uint32_t packed_size_in_bytes = tensor_impl::packed_buffer_size_bytes_wrapper(data_type, compute_buffer_size(shape, data_type));
    auto device_buffer = tensor_impl::allocate_buffer_on_device(packed_size_in_bytes, device, shape, data_type, layout, memory_config);
    return Tensor(DeviceStorage{device_buffer}, shape, data_type, layout);
}

Tensor create_sharded_device_tensor(const Shape& shape, DataType data_type, Layout layout, Device *device, const MemoryConfig& memory_config, bool pad_to_same_shard_size) {
    ZoneScoped;
    TT_ASSERT(memory_config.is_sharded());
    TT_ASSERT(memory_config.shard_spec.has_value());
    TT_ASSERT(memory_config.buffer_type == BufferType::L1);
    auto shard_spec = memory_config.shard_spec.value();
    auto& shard_shape = shard_spec.shape;

    uint32_t num_cores = shard_spec.num_cores();

    uint32_t num_shards;
    uint32_t total_height = tt_metal::compute_volume(shape) / shape[-1];
    uint32_t total_width = shape[-1];
    if (memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_ASSERT(total_width == shard_shape[1], "Shard shape {} does not divide tensor shape {} correctly according to sharding scheme", shard_shape[1], total_width);
        num_shards = div_up(total_height, shard_shape[0]);
    } else if (memory_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        TT_ASSERT(total_height == shard_shape[0], "Shard shape does not divide tensor shape correctly according to sharding scheme");
        num_shards = div_up(total_width, shard_shape[1]);
    } else if (memory_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        num_shards = div_up(total_height, shard_shape[0]) * div_up(total_width, shard_shape[1]);
    } else {
        TT_FATAL(false, "Unsupported sharding scheme");
    }

    TT_ASSERT(num_shards == num_cores, "Number of shards {} must match number of cores {}", num_shards, num_cores);

    if (layout == Layout::TILE) {
        TT_ASSERT((shard_shape[0] % TILE_HEIGHT == 0 && shard_shape[1] % TILE_WIDTH == 0), "Shard shape must be tile sized");
    } else if (layout == Layout::ROW_MAJOR) {
        // Require alignment for now
        // TT_ASSERT(shard_shape[1] * tensor_impl::element_size_bytes_wrapper(data_type) % ADDRESS_ALIGNMENT == 0);
    }

    auto element_size = tensor_impl::element_size_bytes_wrapper(data_type);
    auto page_shape = tensor_impl::get_sharded_page_shape(layout, data_type, shard_spec.shape);
    std::array<uint32_t,2> tensor2d_size = {shape[0]*shape[1] * shape[2]/page_shape[0],
                                                shape[3]/page_shape[1]
                                            };
    ShardSpecBuffer shard_spec_buffer(shard_spec, page_shape, tensor2d_size);
    uint32_t packed_size_in_bytes;

    // Investigate if this padding is correct for other shard orientations
    // Falcon40B was showing that this didn't work for Width Sharding
    // Currently need this as interleaved_to_sharded needs this padding
    // #6029: looks at either updating interleaved_to_sharded s.t we can remove this padding, or update padding for other shard orientations
    if(pad_to_same_shard_size){
        uint32_t shard_size;
        if(layout == Layout::TILE)
            shard_size = tensor_impl::packed_buffer_size_bytes_wrapper(data_type, compute_buffer_size(Shape({shard_shape[0], shard_shape[1]}), data_type));
        else{
            shard_size = shard_shape[0] * shard_shape[1] * element_size;
        }
        packed_size_in_bytes = shard_size * num_cores;
    }
    else {
        packed_size_in_bytes = tensor_impl::packed_buffer_size_bytes_wrapper(data_type, compute_buffer_size(shape, data_type));
    }
    auto device_buffer = tensor_impl::allocate_buffer_on_device(packed_size_in_bytes, device, shape,
                                                            data_type, layout, memory_config,
                                                            std::make_optional<ShardSpecBuffer>(shard_spec_buffer)
                                                            );
    return Tensor(DeviceStorage{device_buffer}, shape, data_type, layout);
}

void* get_raw_host_data_ptr(const Tensor& tensor) {
    const static std::unordered_map<DataType, std::function<void*(const Tensor&)>> dispatch_map = {
        {DataType::BFLOAT16, &tensor_impl::get_raw_host_data_ptr<bfloat16>},
        {DataType::FLOAT32, &tensor_impl::get_raw_host_data_ptr<float>},
        {DataType::UINT32, &tensor_impl::get_raw_host_data_ptr<uint32_t>},
        {DataType::BFLOAT8_B, &tensor_impl::get_raw_host_data_ptr<uint32_t>},
        {DataType::BFLOAT4_B, &tensor_impl::get_raw_host_data_ptr<uint32_t>},
        {DataType::UINT16, &tensor_impl::get_raw_host_data_ptr<uint16_t>},
    };
    return dispatch_map.at(tensor.get_dtype())(tensor);
}

void memcpy(CommandQueue& queue, void* dst, const Tensor& src, const std::optional<std::size_t> transfer_size) {
    if (not transfer_size.has_value()) {
        TT_ASSERT("transfer_size is not supported for memcpy right now!");
    }
    if (not is_device_tensor(src)) {
        TT_THROW("memcpy: src tensor must be on device");
    }

    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }
    EnqueueReadBuffer(queue, src.device_buffer(), dst, true);
}

void memcpy(void* dst, const Tensor& src, const std::optional<std::size_t> transfer_size) {
    memcpy(src.device()->command_queue(), dst, src, transfer_size);
}

void memcpy(CommandQueue& queue, Tensor& dst, const void* src, const std::optional<std::size_t> transfer_size) {
    if (not transfer_size.has_value()) {
        TT_ASSERT("transfer_size is not supported for memcpy right now!");
    }
    if (not is_device_tensor(dst)) {
        TT_THROW("memcpy: memcpy to non-device tensor is not supported!");
    }
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }
    EnqueueWriteBuffer(queue, dst.device_buffer(), src, false);
}

void memcpy(Tensor& dst, const void* src, const std::optional<std::size_t> transfer_size) {
    memcpy(dst.device()->command_queue(), dst, src, transfer_size);
}

void memcpy(CommandQueue& queue, Tensor& dst, const Tensor& src, const std::optional<std::size_t> transfer_size) {
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    TT_ASSERT(dst.get_dtype() == src.get_dtype());
    TT_ASSERT(dst.get_layout() == src.get_layout());

    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        memcpy(queue, get_raw_host_data_ptr(dst), src, transfer_size);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        memcpy(queue, dst, get_raw_host_data_ptr(src), transfer_size);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

void memcpy(Tensor& dst, const Tensor& src, const std::optional<std::size_t> transfer_size) {
    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        memcpy(src.device()->command_queue(), dst, src, transfer_size);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        memcpy(dst.device()->command_queue(), dst, src, transfer_size);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

}  // namespace tt_metal

}  // namespace tt
