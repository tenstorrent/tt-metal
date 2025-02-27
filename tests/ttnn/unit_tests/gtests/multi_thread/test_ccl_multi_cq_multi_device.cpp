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
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/lockfree/queue.hpp>

#include "ttnn/async_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/erisc_datamover_builder.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/distributed/types.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

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

TEST_F(T3kMultiDeviceMultiQueueFixture, AsyncExecutionWorksCQ0) {
    const size_t dim = 0;
    const size_t num_links = 1;
    constexpr auto layout = Layout::TILE;
    constexpr size_t test_expected_num_devices = 4;

    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(true);
    mesh_device->enable_program_cache();

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

    // FABRIC setup
    const bool enable_persistent_fabric = true;

    std::vector<Program> dummy_worker_programs;
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle;
    setup_test_with_persistent_fabric(
        devices,
        dummy_worker_programs,
        subdevice_managers,
        fabric_programs,
        fabric_program_ptrs,
        fabric_handle,
        enable_persistent_fabric,
        num_links);

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
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);

    uint8_t op_cq_id = 0;  // operation command queue id
    boost::asio::thread_pool pool(devices.size());

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());

        int dev_idx = 0;
        log_info(LogTest, "Enqueue Operations before AllGather", outer_loop);
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < devices.size(); ++i) {
            auto device = devices[i];
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, device, promise]() mutable {
                // Generate input data for each device
                for (int j = 0; j < num_elems; j++) {
                    host_data[j] = bfloat16(static_cast<float>(dev_idx));
                }

                TensorSpec tensor_spec(
                    input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));
                auto input_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device, tensor_spec);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
                Tensor input_tensor = Tensor(input_storage, input_shape, DataType::BFLOAT16, Layout::TILE);

                // Enqueue write_buffer to the read/write command queue and record the event
                ttnn::write_buffer(ttnn::QueueId(op_cq_id), input_tensor, {host_data});

                // Enqueue multiple operations to the operation command queue
                // Set output_tensor into device_tensor for allreduce
                device_tensors[dev_idx] = dispatch_ops_to_device(device, input_tensor, ttnn::QueueId(op_cq_id));

                promise->set_value();
            });
            // If you remove below comment (perform sleep), the final output value will be correct.
            // std::this_thread::sleep_for(std::chrono::seconds(1));
            dev_idx++;
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }

        log_info(LogTest, "Enqueue AllGather", outer_loop);
        // Create a multi-device tensor for allgather
        const Tensor mesh_tensor_for_ag = ttnn::distributed::aggregate_as_tensor(device_tensors, AllGatherTensor{});

        // Enqueue the all_gather_async operation on each device.
        // It does not support command queue ID as a parameter and internally uses command queue 0.
        const Tensor gathered_tensor = ttnn::experimental::all_gather_async(
            mesh_tensor_for_ag,
            0,
            multi_device_global_semaphore,
            1,
            operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            ttnn::ccl::Topology::Linear,
            SubDeviceId(0),
            true);

        log_info(LogTest, "EnqueueReadBuffer", outer_loop);
        // Read the values from each device and compare them with the results calculated on the host
        for (size_t i = 0; i < devices.size(); ++i) {
            auto device = devices[i];
            auto device_tensor = ttnn::distributed::get_device_tensor(gathered_tensor, device);
            boost::asio::post(pool, [&, i, device, num_elems, device_tensor]() mutable {
                auto output_data = std::shared_ptr<bfloat16[]>(new bfloat16[device_tensor.volume()]);
                ttnn::read_buffer(ttnn::QueueId(op_cq_id), device_tensor, {output_data});

                for (int j = 0; j < device_tensor.volume(); j++) {
                    int base = j / num_elems;  // dev_idx
                    ASSERT_EQ(output_data[j].to_float(), (-1.0 * base * 32.0 + 128));
                }
                log_info(LogTest, "Device{} Compare Success", device->id(), outer_loop);
            });
        }
    }

    pool.join();

    if (enable_persistent_fabric) {
        persistent_fabric_teardown_sequence(
            devices, subdevice_managers, fabric_handle.value(), tt::fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    }
    for (auto device : devices) {
        ttnn::queue_synchronize(device->command_queue(op_cq_id));
    }

    log_info(tt::LogTest, "Finished");
}

TEST_F(T3kMultiDeviceMultiQueueFixture, AsyncExecutionWorksCQ0CQ1) {
    const size_t dim = 0;
    const size_t num_links = 1;
    constexpr auto layout = Layout::TILE;
    constexpr size_t test_expected_num_devices = 4;

    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(true);
    mesh_device->enable_program_cache();

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

    // FABRIC setup
    const bool enable_persistent_fabric = true;

    std::vector<Program> dummy_worker_programs;
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle;
    setup_test_with_persistent_fabric(
        devices,
        dummy_worker_programs,
        subdevice_managers,
        fabric_programs,
        fabric_program_ptrs,
        fabric_handle,
        enable_persistent_fabric,
        num_links);

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
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);

    uint8_t ccl_cq_id = 0;  // ccl operation command queue id
    uint8_t op_cq_id = 1;   // device operation, read/write command queue id

    boost::asio::thread_pool pool(devices.size());

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());

        int dev_idx = 0;
        log_info(LogTest, "Enqueue Operations before AllGather", outer_loop);
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < devices.size(); ++i) {
            auto device = devices[i];
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, device, promise]() mutable {
                // Generate input data for each device
                for (int j = 0; j < num_elems; j++) {
                    host_data[j] = bfloat16(static_cast<float>(dev_idx));
                }

                TensorSpec tensor_spec(
                    input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));
                auto input_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device, tensor_spec);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
                Tensor input_tensor = Tensor(input_storage, input_shape, DataType::BFLOAT16, Layout::TILE);

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
            dev_idx++;
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }

        log_info(LogTest, "Enqueue AllGather", outer_loop);
        // Create a multi-device tensor for allgather
        const Tensor mesh_tensor_for_ag = ttnn::distributed::aggregate_as_tensor(device_tensors, AllGatherTensor{});

        // Enqueue the all_gather_async operation on each device.
        // It does not support command queue ID as a parameter and internally uses command queue 0.
        const Tensor gathered_tensor = ttnn::experimental::all_gather_async(
            mesh_tensor_for_ag,
            0,
            multi_device_global_semaphore,
            1,
            operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            ttnn::ccl::Topology::Linear,
            SubDeviceId(0),
            true);

        futures.clear();  // Clear futures for the next set of tasks
        for (size_t i = 0; i < devices.size(); ++i) {
            auto device = devices[i];
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, device, promise]() mutable {
                auto ccl_event = std::make_shared<Event>();
                ttnn::record_event(device->command_queue(ccl_cq_id), ccl_event);
                // Enqueue the task waiting for the operation_event to the ccl`s command queue
                ttnn::wait_for_event(device->command_queue(op_cq_id), ccl_event);

                promise->set_value();
            });
            dev_idx++;
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }

        log_info(LogTest, "EnqueueReadBuffer", outer_loop);
        // Read the values from each device and compare them with the results calculated on the host
        for (size_t i = 0; i < devices.size(); ++i) {
            auto device = devices[i];
            auto device_tensor = ttnn::distributed::get_device_tensor(gathered_tensor, device);

            boost::asio::post(pool, [&, i, device, num_elems, device_tensor]() mutable {
                auto output_data = std::shared_ptr<bfloat16[]>(new bfloat16[device_tensor.volume()]);
                ttnn::read_buffer(ttnn::QueueId(op_cq_id), device_tensor, {output_data});

                for (int j = 0; j < device_tensor.volume(); j++) {
                    int base = j / num_elems;  // dev_idx
                    ASSERT_EQ(output_data[j].to_float(), (-1.0 * base * 32.0 + 128));
                }
                log_info(LogTest, "Device{} Compare Success", device->id(), outer_loop);
            });
        }
    }

    pool.join();

    if (enable_persistent_fabric) {
        persistent_fabric_teardown_sequence(
            devices, subdevice_managers, fabric_handle.value(), tt::fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    }
    for (auto device : devices) {
        ttnn::queue_synchronize(device->command_queue(op_cq_id));
    }

    log_info(tt::LogTest, "Finished");
}

}  // namespace ttnn::distributed::test
