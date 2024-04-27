// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "tensor/tensor.hpp"
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

Tensor::Tensor(const Storage storage, const ttnn::Shape shape, DataType dtype, Layout layout) :
    tensor_id{std::nullopt},
    tensor_attributes(std::make_shared<TensorAttributes>(storage, shape, dtype, layout)),
    deallocate_through_destructor(false) {
    ZoneScoped;
    std::visit(
        [&] (auto&& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                this->tensor_attributes->tensor_populated = {true};
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_ASSERT(storage.buffer->device() != nullptr);
                workers = {storage.buffer->device()};
                tensor_impl::validate_on_device_dtype_and_layout(storage.buffer->device(), dtype, layout);
                // Increment main thread ref count for all tensors on device
                this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
                this->tensor_attributes->tensor_populated = {true};
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                this->tensor_attributes->tensor_populated = {true};
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                workers.reserve(storage.buffers.size());
                for (const auto& device_buf_pair : storage.buffers) {
                    auto [device_id, buffer] = device_buf_pair;
                    TT_ASSERT(buffer->device() != nullptr);
                    TT_ASSERT(buffer->device()->id() == device_id);
                    tensor_impl::validate_on_device_dtype_and_layout(buffer->device(), dtype, layout);
                    workers.push_back(buffer->device());
                }
                // Increment main thread ref count for all tensors on cluster
                this->tensor_attributes->increment_main_thread_ref_count(this->workers.at(0));
                this->tensor_attributes->tensor_populated = std::vector<bool>(storage.buffers.size(), true);
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                this->tensor_attributes->tensor_populated = std::vector<bool>(storage.buffers.size(), true);
            } else {
                raise_unsupported_storage<StorageType>();
            }
        }, storage);
}

Tensor::Tensor(const Storage storage, const Shape shape, DataType dtype, Layout layout) :
    Tensor(storage, ttnn::Shape{shape}, dtype, layout) {}

Tensor::~Tensor() {
    ZoneScoped;
    this->deallocate_through_destructor = true;
    this->deallocate();
    // Decrement main thread ref count for all tensors on device
    if (this->workers.size() and this->tensor_attributes) {
        this->tensor_attributes->decrement_main_thread_ref_count(this->workers.at(0));
    }
    tensor_attributes.reset();
}

void Tensor::deallocate(bool force) {
    ZoneScopedN("TensorDeallocate");
    if (this->tensor_attributes.use_count()) {
        // Check if the attributes didn't get moved to another tensor.
        // If not, we can deallocate this tensor.
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
                                // Record ref count before sending to worker
                                uint32_t device_tensor_ref_count = this->tensor_attributes->record_main_thread_ref_count();
                                this->workers.at(0)->push_work([force, *this] () mutable {
                                    // Cross worker synchronization: If the tensor being deallocated is shared across workers (ex: all_gather op),
                                    // wait until all workers are done with this tensor before deallocating.
                                    bool num_threads_sharing_tensor = this->tensor_attributes->num_sibling_workers_sharing_tensor;
                                    if (num_threads_sharing_tensor) {
                                        while (num_threads_sharing_tensor) {
                                           num_threads_sharing_tensor = this->tensor_attributes->num_sibling_workers_sharing_tensor;;
                                        }
                                    }
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
                                        } else if  constexpr(std::is_same_v<type, OwnedStorage>) {
                                            // Manage Dynamic Storage (due to autoformat in async mode): Main thread sees this tensor as a device tensor, since worker has not updated
                                            // storage time. When the worker executes the dealloc request, the storage type has been appropriately updated to Owned.
                                            TT_ASSERT(this->tensor_attributes->dynamic_storage, "Tensor storage type changed during runtime (device -> host), but dynamic storage was not marked.");
                                            std::visit([] (auto&& buffer) { buffer.reset(); }, s.buffer);
                                        }
                                    }, this->tensor_attributes->storage);
                                });
                                // Update ref count after sending to worker
                                this->tensor_attributes->update_main_thread_ref_count(this->workers.at(0), device_tensor_ref_count);
                            }
                        } else {
                            TT_FATAL(this->deallocate_through_destructor, "Device tensors cannot be explictly deallocated in worker threads.");
                        }
                    } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                        if (force) {
                            TT_THROW("Cannot deallocate tensor with borrowed storage!");
                        }
                    } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                        if (this->workers.at(0)->in_main_thread()) {
                            uint32_t ref_count_to_use = (this->workers.at(0)->get_worker_mode() == WorkExecutorMode::SYNCHRONOUS) ? this->tensor_attributes.use_count() : this->tensor_attributes->main_thread_ref_count;
                            if ((force or ref_count_to_use == 1) and not this->tensor_attributes->deallocated) {
                                this->tensor_attributes->deallocated = true;
                                // Record ref count before sending to workers
                                uint32_t device_tensor_ref_count = this->tensor_attributes->record_main_thread_ref_count();
                                for (auto worker : this->workers) {
                                    worker->push_work([force, *this, worker] () mutable {
                                        std::visit([force, worker] (auto&& s) {
                                            using type = std::decay_t<decltype(s)>;
                                            if constexpr (std::is_same_v<type, MultiDeviceStorage>) {
                                                if (s.buffers.find(worker->id()) != s.buffers.end()) {
                                                    if (force or s.buffers.at(worker->id()).use_count() == 1) {
                                                        DeallocateBuffer(*(s.buffers.at(worker->id())));
                                                    }
                                                    s.buffers.at(worker->id()).reset();
                                                }
                                            }
                                        }, this->tensor_attributes->storage);
                                    });
                                }
                                // Update ref count after sending to workers
                                this->tensor_attributes->update_main_thread_ref_count(this->workers.at(0), device_tensor_ref_count);
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

void Tensor::perform_cleanup_for_async_mode() {
    // Used when tensor attributes object for this is reassigned by copy
    // or move assignment operator
    if (this->tensor_attributes) {
        // Object has tensor_attributes that will be reassigned
        if (this->workers.size() and this->workers.at(0)->in_main_thread() and this->workers.at(0)->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS) {
            // Operator called in main thread with async mode. Main thread Ref Count must be decremented.
            // This is the last tensor in the main thread holding these attributes. Deallocate the buffer
            // for this tensor.
            if (this->tensor_attributes->main_thread_ref_count == 1) {
                this->deallocate();
            }
            this->tensor_attributes->main_thread_ref_count--;
        }
    }
}

// Main Thread - Wait for all workers in this tensor to populate the entire tensor
void Tensor::wait_for_tensor_data_populated() const {
    ZoneScoped;
    // Stall until all the workers for this tensor
    // have populated the full tensor
    for (int i = 0; i < this->tensor_attributes->tensor_populated.size(); i++) {
        while (true) {
            std::scoped_lock<std::mutex> lock(this->tensor_attributes->populated_mutex);
            if (this->tensor_attributes->tensor_populated.at(i)) break;
        }
    }
}

// Main Thread - Wait for the first worker in this tensor to populate the global metadata fields
void Tensor::wait_for_tensor_metadata_populated() const {
    ZoneScoped;
    // First worker is responsible for updating all metadata fields
    // Stall until this worker is done
    while (true) {
        std::scoped_lock<std::mutex> lock(this->tensor_attributes->populated_mutex);
        if (this->tensor_attributes->tensor_populated.at(0)) break;
    };
}

// Worker Thread - Set populated flag to true, once worker has completed it's task for this tensor
void Tensor::set_populated(Device* worker) {
    // If worker is not specified, set entry for all workers to true
    std::scoped_lock<std::mutex> lock(this->tensor_attributes->populated_mutex);
    if (not worker) {
        for (int i = 0; i < this->tensor_attributes->tensor_populated.size(); i++) {
            this->tensor_attributes->tensor_populated.at(i) = true;
        }
    }
    else {
        this->tensor_attributes->tensor_populated.at(worker->id()) = true;
    }
}

void Tensor::deepcopy(const Tensor& other) {
    ZoneScoped;
    // Wait until the tensor being copied is populated
    other.wait_for_tensor_data_populated();
    // Populate tensor metadata
    this->set_shape(other.get_shape());
    this->set_storage(other.get_storage());
    this->set_dtype(other.get_dtype());
    this->set_layout(other.get_layout());
    // Set metadata populated flag for getters
    this->set_populated();
}

void Tensor::populate_buffers_and_metadata(const Tensor& other) {
    ZoneScoped;
    // Similar to deepcopy, but to be applied on a tensor that has an empty storage
    // container initialized. Require tensor storage to be correctly initialized.
    this->set_shape(other.get_shape());
    this->set_dtype(other.get_dtype());
    this->set_layout(other.get_layout());
    // Populate storage container with buffers + shapes
    std::visit([this] (auto&& storage) {
        using StorageType = std::decay_t<decltype(storage)>;
        if constexpr(std::is_same_v<StorageType, OwnedStorage> or std::is_same_v<StorageType, DeviceStorage>) {
            std::get<StorageType>(this->tensor_attributes->storage).insert_buffer(storage.get_buffer());
            this->tensor_attributes->tensor_populated = {true};
        } else if constexpr(std::is_same_v<StorageType, MultiDeviceHostStorage> or std::is_same_v<StorageType, MultiDeviceStorage>) {
            std::get<StorageType>(this->tensor_attributes->storage).buffers = storage.buffers;
            std::get<StorageType>(this->tensor_attributes->storage).shapes = storage.shapes;
            this->tensor_attributes->tensor_populated = std::vector<bool>(storage.buffers.size(), true);
        }
    }, other.get_storage()); // Non blocking storage query, since this is done for tensors that get created inside the worker thread
}

std::vector<Device*> Tensor::get_workers(bool blocking) const {
    ZoneScoped;
    // Initialize an empty worker vector (remains empty for host side storage)
    std::vector<Device*> workers = {};

    if (this->tensor_attributes->dynamic_storage) {
        // Tensor is populated by launch_with_autoformat
        // Storage type can change based on op behaviour, wait until tensor populated.
        this->wait_for_tensor_metadata_populated();
    }

    std::visit([this, blocking, &workers] (auto&& storage) {
        using StorageType = std::decay_t<decltype(storage)>;
        // Assign workers only to device tensors
        if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
            // Either explictly syncing or workers are pre-populated (this will happen for device tensors if using the correct APIs).
            TT_FATAL(blocking or (this->workers.size() == 1), "Worker Handles for tensor must be populated or blocking = true must be set in get_workers().");
            if (this->workers.size() != 1) {
                // Not populated - sync.
                this->wait_for_tensor_data_populated();
                workers = {this->device()};
            } else {
                // Already populated.
                workers = this->workers;
            }
        } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
            // Either explictly syncing or workers are pre-populated (this will happen for device tensors if using the correct APIs).
            TT_FATAL(blocking or (this->workers.size()), "Worker Handles for tensor must be populated or blocking = true must be set in get_workers().");
            if (not this->workers.size()) {
                // Not populated - sync.
                this->wait_for_tensor_data_populated();
                workers.reserve(storage.buffers.size());
                for (auto& device_buf_pair : storage.buffers) {
                    // Already populated.
                    workers.push_back(device_buf_pair.second->device());
                }
            } else {
                workers = this->workers;
            }
        }
    }, this->tensor_attributes->storage);
    return workers;
}

// Getters - Spin until tensor is populated before querying tensor metadata
const Shape& Tensor::get_legacy_shape() const {
    this->wait_for_tensor_metadata_populated();
    return this->tensor_attributes->shape.value();
}

const ttnn::Shape& Tensor::get_shape() const {
    this->wait_for_tensor_metadata_populated();
    return this->tensor_attributes->shape;
}
const DataType& Tensor::get_dtype() const {
    this->wait_for_tensor_metadata_populated();
    return this->tensor_attributes->dtype;
}
const Layout& Tensor::get_layout() const {
    this->wait_for_tensor_metadata_populated();
    return this->tensor_attributes->layout;
}

const Storage& Tensor::get_storage() const {
    this->wait_for_tensor_data_populated();
    return this->tensor_attributes->storage;
}

Tensor Tensor::to(CommandQueue & queue, const MemoryConfig & mem_config) const {
    ZoneScoped;
    auto target_device = queue.device();
    // Tensor can be using borrowed storage. If so, when running in async mode, copy this tensor to owned storage.
    Tensor async_safe_tensor = copy_borrowed_tensor_in_async_mode(target_device, *this);
    // Populate device storage outside of thread, so that downstream
    // functions running in main can get storage type without blocking
    Tensor device_tensor({target_device});
    // Record main thread ref count for tensors before pushing to queue.
    uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t original_tensor_ref_count = async_safe_tensor.tensor_attributes->record_main_thread_ref_count();
    queue.device()->push_work([async_safe_tensor, device_tensor, mem_config, target_device] () mutable {
        if (async_safe_tensor.storage_type() == StorageType::DEVICE) {
            TT_ASSERT(async_safe_tensor.device() == target_device && "Currently do not support moving between devices");
            device_tensor.populate_buffers_and_metadata(async_safe_tensor);
        }
        else {
            tensor_impl::validate_on_device_dtype_and_layout(target_device, async_safe_tensor.get_dtype(), async_safe_tensor.get_layout());
            auto local_tensor = tensor_impl::to_device_wrapper(async_safe_tensor, target_device, mem_config);
            // Populate device tensor
            device_tensor.populate_buffers_and_metadata(local_tensor);
        }
    });
    // Update main thread ref count for tensors after pushing to queue (update original tensor and returned tensor,
    // since both can be on device).
    device_tensor.tensor_attributes->update_main_thread_ref_count(device_tensor.workers.at(0), device_tensor_ref_count);
    async_safe_tensor.tensor_attributes->update_main_thread_ref_count(device_tensor.workers.at(0), original_tensor_ref_count);
    return device_tensor;
}

Tensor Tensor::to(Device *target_device, const MemoryConfig &mem_config) const {
    ZoneScoped;
    // Tensor can be using borrowed storage. If so, when running in async mode, copy this tensor to owned storage.
    Tensor async_safe_tensor = copy_borrowed_tensor_in_async_mode(target_device, *this);
    // Populate device storage outside of thread, so that downstream
    // functions running in main can get storage type without blocking
    Tensor device_tensor({target_device});
    // Record main thread ref count for tensors before pushing to queue.
    uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t original_tensor_ref_count = async_safe_tensor.tensor_attributes->record_main_thread_ref_count();
    target_device->push_work([async_safe_tensor, device_tensor, mem_config, target_device] () mutable {
        if (async_safe_tensor.storage_type() == StorageType::DEVICE) {
            TT_ASSERT(async_safe_tensor.device() == target_device && "Currently do not support moving between devices");
            device_tensor.populate_buffers_and_metadata(async_safe_tensor);
        }
        else {
            tensor_impl::validate_on_device_dtype_and_layout(target_device, async_safe_tensor.get_dtype(), async_safe_tensor.get_layout());
            auto local_tensor = tensor_impl::to_device_wrapper(async_safe_tensor, target_device, mem_config);
            // Populate device tensor
            device_tensor.populate_buffers_and_metadata(local_tensor);
        }
    });
    // Update main thread ref count for tensors after pushing to queue (update original tensor and returned tensor,
    // since both can be on device).
    device_tensor.tensor_attributes->update_main_thread_ref_count(device_tensor.workers.at(0), device_tensor_ref_count);
    async_safe_tensor.tensor_attributes->update_main_thread_ref_count(device_tensor.workers.at(0), original_tensor_ref_count);
    return device_tensor;
}

Tensor Tensor::to(DeviceMesh *device_mesh, const MemoryConfig &mem_config) const {
    ZoneScoped;
    return this->to(device_mesh->get_devices(), mem_config);
}

Tensor Tensor::to(const std::vector<Device*>& workers, const MemoryConfig &mem_config) const {
    ZoneScoped;
    TT_FATAL(validate_worker_modes(workers), "All device threads/workers must be running in the same mode (ASYNC or SYNC)");
    // When broadcasting a single shard to all devices, we use all workers.
    // When sending a MultiDeviceHost tensor to the cluster, send it only to devices for which shards exist
    auto workers_to_use = workers;
    if (std::holds_alternative<MultiDeviceStorage>(this->get_storage()) or std::holds_alternative<MultiDeviceHostStorage>(this->get_storage())) {
        workers_to_use = std::vector<Device*>(workers.begin(), workers.begin() + num_buffers_in_tensor(*this));
    }
    Tensor device_tensor = Tensor(workers_to_use);
    uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t original_tensor_ref_count = this->tensor_attributes->record_main_thread_ref_count();
    uint32_t num_workers = workers_to_use.size();
    for (auto& worker : workers_to_use) {
        worker->push_work(
            [worker, *this, device_tensor, mem_config, num_workers] () mutable {
                auto shard = get_shard_for_device(*this, worker);
                if (shard.storage_type() == StorageType::OWNED) {
                    shard = tensor_impl::to_device_wrapper(shard, worker, mem_config);
                }
                insert_buffer_and_shape_for_device(worker, shard, device_tensor);
                if (not worker->id()) {
                    device_tensor.set_shape(this->get_shape());
                    device_tensor.set_dtype(this->get_dtype());
                    device_tensor.set_layout(this->get_layout());
                }
                if (num_workers > 1) device_tensor.set_populated(worker);
                else device_tensor.set_populated();
        });
    }
    device_tensor.tensor_attributes->update_main_thread_ref_count(workers.at(0), device_tensor_ref_count);
    this->tensor_attributes->update_main_thread_ref_count(workers.at(0), original_tensor_ref_count);
    return device_tensor;
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
    TT_FATAL(validate_worker_modes(workers), "All device threads/workers must be running in the same mode (ASYNC or SYNC)");
    Tensor host_tensor({}, workers.size());
    uint32_t original_tensor_ref_count = this->tensor_attributes->record_main_thread_ref_count();
    for (auto target_device : workers) {
        target_device->push_work([host_tensor, blocking, target_device, *this, workers] () mutable {
            TT_ASSERT(this->storage_type() == StorageType::DEVICE or this->storage_type() == StorageType::MULTI_DEVICE, "Can only use worker queue for cpu call if tensor is on device.");
            auto shard = get_shard_for_device(*this, target_device);
            shard = tensor_impl::to_host_wrapper(shard, blocking);
            insert_buffer_and_shape_for_device(target_device, shard, host_tensor);
            if (not target_device->id() or workers.size() == 1) {
                host_tensor.set_shape(this->get_shape());
                host_tensor.set_dtype(this->get_dtype());
                host_tensor.set_layout(this->get_layout());
            }
            if (workers.size() == 1) {
                host_tensor.set_populated();
            }
            else {
                host_tensor.set_populated(target_device);
            }
        });
    }
    if (blocking) {
        for (auto target_device : workers) {
            target_device->synchronize();
        }
    }
    // Update main_thread_ref_count for tensor after pushing to queue.
    this->tensor_attributes->update_main_thread_ref_count(workers.at(0), original_tensor_ref_count);
    return host_tensor;
}

Tensor Tensor::cpu_sharded() const {
    ZoneScoped;
    return tensor_impl::to_host_wrapper_sharded(*this);
}


Tensor Tensor::extract_shard(const CoreCoord & core) const{
    ZoneScoped;
    auto buffer_page_mapping = generate_buffer_page_mapping(*this->buffer());
    auto core_id = buffer_page_mapping.core_to_core_id_.at(core);
    return this->extract_shard(core_id);
}

Tensor Tensor::extract_shard(const uint32_t & core_id) const{

    return tensor_impl::to_extract_shard_wrapper(*this, core_id);

}

Tensor Tensor::to(Layout target_layout, Device* worker) const {
    ZoneScoped;
    // Only push layout conversion to worker if running in async mode
    if (worker and worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS) {
        // Tensor can be using borrowed storage. If so, when running in async mode, copy this tensor to owned storage.
        Tensor async_safe_tensor = copy_borrowed_tensor_in_async_mode(worker, *this);
        Tensor tensor_modified_layout = Tensor({}, 1);
        worker->push_work([async_safe_tensor, tensor_modified_layout, target_layout] () mutable {
            TT_ASSERT(async_safe_tensor.storage_type() != StorageType::DEVICE && "Bring tensor to host before converting to target layout");
            auto local_tensor = tensor_impl::to_layout_wrapper(async_safe_tensor, target_layout);
            // Populate modified layout tensor
            tensor_modified_layout.populate_buffers_and_metadata(local_tensor);
        });
        return tensor_modified_layout;
    } else {
        TT_ASSERT(this->storage_type() != StorageType::DEVICE && "Bring tensor to host before converting to target layout");
        return tensor_impl::to_layout_wrapper(*this, target_layout);
    }
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
    ZoneScoped;
    auto new_shape = infer_dims_for_reshape(N, C, H, W, this->volume());
    return this->reshape(new_shape);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    ZoneScoped;
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
    ZoneScoped;
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
                for (const auto& device_buf_pair : storage.buffers) {
                    auto buffer = device_buf_pair.second;
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
    auto buffer_page_mapping = generate_buffer_page_mapping(*this->buffer());
    auto cores = buffer_page_mapping.all_cores_;
    auto shard_size = buffer()->shard_spec().size();
    auto num_pages = cores.size() * shard_size;

    std::vector<uint32_t> ret_vec;
    ret_vec.reserve(num_pages);
    for(int page_id = 0; page_id <num_pages ; page_id++){
        if(buffer_page_mapping.dev_page_to_host_page_mapping_[page_id].has_value()) {
            ret_vec.push_back(buffer_page_mapping.dev_page_to_host_page_mapping_[page_id].value());
        }
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

Tensor create_sharded_device_tensor(const Shape& shape, DataType data_type, Layout layout, Device *device, const MemoryConfig& memory_config) {
    ZoneScoped;
    TT_ASSERT(memory_config.is_sharded());
    TT_ASSERT(memory_config.shard_spec.has_value());
    TT_ASSERT(memory_config.is_l1());

    auto shard_spec = memory_config.shard_spec.value();
    auto& shard_shape = shard_spec.shape;

    auto width = shape[-1];
    auto other_dims = 1;
    for (int i = 0; i < shape.rank() - 1; i++) {
        other_dims *= shape[i];
    }

    auto element_size = tensor_impl::element_size_bytes_wrapper(data_type);
    auto page_shape = tensor_impl::get_sharded_page_shape(layout, data_type, shard_spec.shape);
    std::array<uint32_t,2> tensor2d_size = {other_dims/page_shape[0], width/page_shape[1]};
    ShardSpecBuffer shard_spec_buffer(shard_spec, page_shape, tensor2d_size);
    uint32_t packed_size_in_bytes;

    packed_size_in_bytes = tensor_impl::packed_buffer_size_bytes_wrapper(data_type, compute_buffer_size(shape, data_type));
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
        {DataType::INT32, &tensor_impl::get_raw_host_data_ptr<int32_t>},
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
