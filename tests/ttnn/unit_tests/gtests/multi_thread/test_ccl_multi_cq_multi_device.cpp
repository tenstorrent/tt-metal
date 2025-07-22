// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "test_utils.hpp"
#include "ttnn_test_fixtures.hpp"

#include <cmath>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <future>
#include <memory>

#define BOOST_ASIO_HAS_STD_INVOKE_RESULT
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>

#include "ttnn/async_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/erisc_datamover_builder_helper.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/distributed/types.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/event.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>

#include "gtest/gtest.h"

namespace ttnn::distributed::test {

using namespace tt;
using namespace tt_metal;

using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshDeviceView;
using tt::tt_metal::distributed::MeshShape;

// Custom Fixture using 1D Fabric on a Multi-CQ MeshDevice
class T3000MultiCQFabricMeshDeviceFixture : public T3000MultiCQMeshDeviceFixture {
protected:
    T3000MultiCQFabricMeshDeviceFixture() {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    }
    void TearDown() override {
        T3000MultiCQMeshDeviceFixture::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

TEST_F(T3000MultiCQFabricMeshDeviceFixture, AsyncExecutionWorksCQ0) {
    const size_t dim = 0;
    const size_t num_links = 1;
    constexpr auto layout = Layout::TILE;
    constexpr size_t test_expected_num_devices = 4;

    MeshDevice* mesh_device = this->mesh_device_.get();

    auto view = mesh_device->get_view();

    // build a line of devices
    std::vector<IDevice*> devices = {
        view.get_device(MeshCoordinate(0, 0)),
        view.get_device(MeshCoordinate(0, 1)),
        view.get_device(MeshCoordinate(0, 2)),
        view.get_device(MeshCoordinate(0, 3))};
    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);

    log_info(LogTest, "Creating Global Semaphore for Ccl Ops");
    auto
        [from_remote_multi_device_global_semaphore,
         to_remote_multi_device_global_semaphore,
         multi_device_global_semaphore] = create_global_semaphores(this->mesh_device_, devices[0]);

    const int batch_size = 8;
    const int sequence_length = 1024;
    const int embedding_dim = 768;

    const ttnn::Shape input_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    uint8_t op_cq_id = 0;  // operation command queue id
    boost::asio::thread_pool pool(devices.size());

    TensorSpec tensor_spec(input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());

        log_info(LogTest, "Enqueue Operations before AllGather");
        std::vector<std::future<void>> futures;
        for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
            auto device = devices[dev_idx];
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, device, promise]() mutable {
                // Generate input data for each device
                auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    host_data[j] = bfloat16(static_cast<float>(dev_idx));
                }

                auto input_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device, tensor_spec);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
                Tensor input_tensor = Tensor(input_storage, tensor_spec, ReplicateTensor{});

                // Enqueue write_buffer to the read/write command queue and record the event
                ttnn::write_buffer(ttnn::QueueId(op_cq_id), input_tensor, {host_data});

                // Enqueue multiple operations to the operation command queue
                // Set output_tensor into device_tensor for allreduce
                device_tensors[dev_idx] = dispatch_ops_to_device(device, input_tensor, ttnn::QueueId(op_cq_id));

                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        log_info(LogTest, "Enqueue AllGather");

        // Enqueue the all_gather_async operation on each device.
        // It does not support command queue ID as a parameter and internally uses command queue 0.
        std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> multi_dev_semaphore = {
            multi_device_global_semaphore};
        const std::vector<Tensor> gathered_tensors = ttnn::experimental::all_gather_async(
            device_tensors,
            0,
            multi_dev_semaphore,
            1,
            operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            ttnn::ccl::Topology::Linear,
            SubDeviceId(0));

        log_info(LogTest, "Enqueue dummy ops");
        for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
            auto device = devices[dev_idx];
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, device, promise]() mutable {
                auto dummy_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    dummy_data[j] = bfloat16(static_cast<float>(dev_idx));
                }
                auto dummy_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device, tensor_spec);
                auto dummy_storage = tt::tt_metal::DeviceStorage{dummy_buffer};
                Tensor dummy_tensor = Tensor(dummy_storage, tensor_spec, ReplicateTensor{});
                ttnn::write_buffer(ttnn::QueueId(op_cq_id), dummy_tensor, {dummy_data});
                dispatch_ops_to_device(device, dummy_tensor, ttnn::QueueId(op_cq_id));
                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        log_info(LogTest, "EnqueueReadBuffer");
        // Read the values from each device and compare them with the results calculated on the host
        for (size_t i = 0; i < devices.size(); ++i) {
            auto device = devices[i];
            auto device_tensor = gathered_tensors[i];
            boost::asio::post(pool, [&, i, device, num_elems, device_tensor]() mutable {
                auto output_data = std::shared_ptr<bfloat16[]>(new bfloat16[device_tensor.physical_volume()]);
                ttnn::read_buffer(ttnn::QueueId(op_cq_id), device_tensor, {output_data});

                for (int j = 0; j < device_tensor.physical_volume(); j++) {
                    int base = j / num_elems;  // dev_idx
                    ASSERT_EQ(output_data[j].to_float(), (-1.0 * base * 32.0 + 128));
                }
                log_info(LogTest, "Device{} Compare Success", device->id());
            });
        }
    }

    pool.join();

    for (auto device : devices) {
        ttnn::queue_synchronize(device->command_queue(op_cq_id));
    }

    log_info(tt::LogTest, "Finished");
}

TEST_F(T3000MultiCQFabricMeshDeviceFixture, AsyncExecutionWorksCQ0CQ1) {
    const size_t dim = 0;
    const size_t num_links = 1;
    constexpr auto layout = Layout::TILE;
    constexpr size_t test_expected_num_devices = 4;

    MeshDevice* mesh_device = this->mesh_device_.get();
    auto view = mesh_device->get_view();

    // build a line of devices
    std::vector<IDevice*> devices = {
        view.get_device(MeshCoordinate(0, 0)),
        view.get_device(MeshCoordinate(0, 1)),
        view.get_device(MeshCoordinate(0, 2)),
        view.get_device(MeshCoordinate(0, 3))};

    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);

    log_info(LogTest, "Creating Global Semaphore for Ccl Ops");
    auto
        [from_remote_multi_device_global_semaphore,
         to_remote_multi_device_global_semaphore,
         multi_device_global_semaphore] = create_global_semaphores(this->mesh_device_, devices[0]);

    const int batch_size = 8;
    const int sequence_length = 1024;
    const int embedding_dim = 768;

    const ttnn::Shape input_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    uint8_t ccl_cq_id = 0;  // ccl operation command queue id
    uint8_t op_cq_id = 1;   // device operation, read/write command queue id

    boost::asio::thread_pool pool(devices.size());

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());

        TensorSpec tensor_spec(
            input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));

        log_info(LogTest, "Enqueue Operations before AllGather");
        std::vector<std::future<void>> futures;
        for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
            auto device = devices[dev_idx];
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, device, promise]() mutable {
                // Generate input data for each device
                auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    host_data[j] = bfloat16(static_cast<float>(dev_idx));
                }

                auto input_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device, tensor_spec);
                auto dummy_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device, tensor_spec);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
                Tensor input_tensor = Tensor(input_storage, tensor_spec, ReplicateTensor{});

                // Enqueue write_buffer to the operation`s command queue and record the event
                ttnn::write_buffer(ttnn::QueueId(op_cq_id), input_tensor, {host_data});

                // Enqueue multiple operations to the operation command queue
                // Set output_tensor into device_tensor for allgather
                device_tensors[dev_idx] = dispatch_ops_to_device(device, input_tensor, ttnn::QueueId(op_cq_id));

                auto operation_event = std::make_shared<Event>();
                ttnn::record_event(device->command_queue(op_cq_id), operation_event);
                // Enqueue the task waiting for the operation_event to the ccl`s command queue
                ttnn::wait_for_event(device->command_queue(ccl_cq_id), operation_event);

                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        log_info(LogTest, "Enqueue AllGather");

        // Enqueue the all_gather_async operation on each device.
        // It does not support command queue ID as a parameter and internally uses command queue 0.
        std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> multi_dev_semaphore = {
            multi_device_global_semaphore};
        const std::vector<Tensor> gathered_tensors = ttnn::experimental::all_gather_async(
            device_tensors,
            0,
            multi_dev_semaphore,
            1,
            operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            ttnn::ccl::Topology::Linear,
            SubDeviceId(0));

        log_info(LogTest, "Enqueue dummy ops");
        for (size_t dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
            auto device = devices[dev_idx];
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, device, promise]() mutable {
                // TODO: investigate why other OPs can't be scheduled on a different command queue until CCL is finished
                ttnn::queue_synchronize(device->command_queue(ccl_cq_id));

                auto dummy_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    dummy_data[j] = bfloat16(static_cast<float>(dev_idx));
                }
                auto dummy_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device, tensor_spec);
                auto dummy_storage = tt::tt_metal::DeviceStorage{dummy_buffer};
                Tensor dummy_tensor = Tensor(dummy_storage, tensor_spec, ReplicateTensor{});
                ttnn::write_buffer(ttnn::QueueId(op_cq_id), dummy_tensor, {dummy_data});
                dispatch_ops_to_device(device, dummy_tensor, ttnn::QueueId(op_cq_id));
                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
            auto device = devices[dev_idx];
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, device, promise]() mutable {
                auto ccl_event = std::make_shared<Event>();
                ttnn::record_event(device->command_queue(ccl_cq_id), ccl_event);
                // Enqueue the task waiting for the operation_event to the ccl`s command queue
                ttnn::wait_for_event(device->command_queue(op_cq_id), ccl_event);

                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        log_info(LogTest, "EnqueueReadBuffer");
        // Read the values from each device and compare them with the results calculated on the host
        for (size_t i = 0; i < devices.size(); ++i) {
            auto device = devices[i];
            auto device_tensor = gathered_tensors[i];

            boost::asio::post(pool, [&, i, device, num_elems, device_tensor]() mutable {
                auto output_data = std::shared_ptr<bfloat16[]>(new bfloat16[device_tensor.physical_volume()]);
                ttnn::read_buffer(ttnn::QueueId(op_cq_id), device_tensor, {output_data});

                for (int j = 0; j < device_tensor.physical_volume(); j++) {
                    int base = j / num_elems;  // dev_idx
                    ASSERT_EQ(output_data[j].to_float(), (-1.0 * base * 32.0 + 128));
                }
                log_info(LogTest, "Device{} Compare Success", device->id());
            });
        }
    }

    pool.join();

    for (auto device : devices) {
        ttnn::queue_synchronize(device->command_queue(op_cq_id));
    }

    log_info(tt::LogTest, "Finished");
}

}  // namespace ttnn::distributed::test
