// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_coord.hpp"
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

#include "common_test_utils.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_command_processor_async/all_gather_command_processor_async.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/distributed/types.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/event.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/host_api.hpp>

#include "gtest/gtest.h"

namespace ttnn::distributed::test {

using namespace tt;
using namespace tt_metal;

using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshShape;

// Custom Fixture using 1D Fabric on a Multi-CQ MeshDevice
class MultiCQFabricMeshDevice2x4Fixture : public MeshDeviceFixtureBase {
protected:
    MultiCQFabricMeshDevice2x4Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{2, 4}, .num_cqs = 2}) {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    }
    void TearDown() override {
        MeshDeviceFixtureBase::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

TEST_F(MultiCQFabricMeshDevice2x4Fixture, AsyncExecutionWorksCQ0) {
    constexpr size_t test_expected_num_devices = 4;

    MeshDevice* mesh_device = this->mesh_device_.get();

    // build a line of devices
    std::vector<std::shared_ptr<distributed::MeshDevice>> single_meshes = {
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0)),
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 1)),
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 2)),
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 3)),
    };
    std::vector<IDevice*> devices = {
        single_meshes[0].get(),
        single_meshes[1].get(),
        single_meshes[2].get(),
        single_meshes[3].get(),
    };

    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);

    log_info(LogTest, "Creating Global Semaphore for Ccl Ops");
    auto multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore_with_same_address(
        devices,
        devices[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                             // initial value
        tt::tt_metal::BufferType::L1,  // buffer type
        10                             // attempts
    );

    const int batch_size = 8;
    const int sequence_length = 1024;
    const int embedding_dim = 768;

    const ttnn::Shape input_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    QueueId op_cq_id(0);  // operation command queue id
    boost::asio::thread_pool pool(devices.size());

    TensorSpec tensor_spec(input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());

        log_info(LogTest, "Enqueue Operations before AllGather");
        std::vector<std::future<void>> futures;
        for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, promise]() mutable {
                // Generate input data for each device
                auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    host_data[j] = bfloat16(static_cast<float>(dev_idx));
                }

                // TODO (#25340): Switch to use create_device_tensor? (TensorTopology logic should mirror
                // create_device_tensor)
                auto& single_mesh = single_meshes[dev_idx];
                auto input_buffer = tt::tt_metal::tensor_impl::allocate_device_buffer(single_mesh.get(), tensor_spec);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer, {MeshCoordinate(0, 0)}};
                Tensor input_tensor = Tensor(input_storage, tensor_spec, TensorTopology{});

                // Enqueue write_buffer to the read/write command queue and record the event
                ttnn::write_buffer(QueueId(op_cq_id), input_tensor, {host_data});

                // Enqueue multiple operations to the operation command queue
                // Set output_tensor into device_tensor for allreduce
                device_tensors[dev_idx] = ttnn::test_utils::dispatch_ops_to_device(input_tensor, QueueId(op_cq_id));

                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        log_info(LogTest, "Enqueue AllGather");

        // Enqueue the all_gather_command_processor_async operation on each device.
        // It does not support command queue ID as a parameter and internally uses command queue 0.
        std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> multi_dev_semaphore = {
            multi_device_global_semaphore};
        const std::vector<Tensor> gathered_tensors = ttnn::experimental::all_gather_command_processor_async(
            device_tensors,
            /* dim */ 0,
            multi_dev_semaphore,
            /* persistent_output_buffer */ std::nullopt,
            /* num_links */ 1,
            operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            ttnn::ccl::Topology::Linear,
            /* cluster_axis */ std::nullopt,
            SubDeviceId(0));

        log_info(LogTest, "Enqueue dummy ops");
        for (int dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, promise]() mutable {
                auto dummy_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    dummy_data[j] = bfloat16(static_cast<float>(dev_idx));
                }
                auto& single_mesh = single_meshes[dev_idx];
                auto dummy_buffer = tt::tt_metal::tensor_impl::allocate_device_buffer(single_mesh.get(), tensor_spec);
                auto dummy_storage = tt::tt_metal::DeviceStorage{dummy_buffer, {MeshCoordinate(0, 0)}};

                Tensor dummy_tensor = Tensor(dummy_storage, tensor_spec, TensorTopology{});
                ttnn::write_buffer(ttnn::QueueId(op_cq_id), dummy_tensor, {dummy_data});
                ttnn::test_utils::dispatch_ops_to_device(dummy_tensor, ttnn::QueueId(op_cq_id));

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
                ttnn::read_buffer(QueueId(op_cq_id), device_tensor, {output_data});

                for (int j = 0; j < device_tensor.physical_volume(); j++) {
                    int base = j / num_elems;  // dev_idx
                    ASSERT_EQ(static_cast<float>(output_data[j]), (-1.0 * base * 32.0 + 128));
                }
                log_info(LogTest, "Device{} Compare Success", device->id());
            });
        }
    }

    pool.join();

    for (auto& single_mesh : single_meshes) {
        auto& op_cq_1 = single_mesh->mesh_command_queue(op_cq_id.get());
        ttnn::queue_synchronize(op_cq_1);
    }

    log_info(tt::LogTest, "Finished");
}

TEST_F(MultiCQFabricMeshDevice2x4Fixture, AsyncExecutionWorksCQ0CQ1) {
    constexpr size_t test_expected_num_devices = 4;

    MeshDevice* mesh_device = this->mesh_device_.get();

    // build a line of devices
    std::vector<std::shared_ptr<distributed::MeshDevice>> single_meshes = {
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0)),
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 1)),
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 2)),
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 3)),
    };
    std::vector<IDevice*> devices = {
        single_meshes[0].get(),
        single_meshes[1].get(),
        single_meshes[2].get(),
        single_meshes[3].get(),
    };

    // https://github.com/tenstorrent/tt-metal/issues/24235
    for (auto& device : single_meshes) {
        device->disable_and_clear_program_cache();
    }

    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);

    log_info(LogTest, "Creating Global Semaphore for Ccl Ops");

    auto multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore_with_same_address(
        devices,
        devices[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                             // initial value
        tt::tt_metal::BufferType::L1,  // buffer type
        10                             // attempts
    );

    const int batch_size = 8;
    const int sequence_length = 1024;
    const int embedding_dim = 768;

    const ttnn::Shape input_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    QueueId ccl_cq_id(0);  // ccl operation command queue id
    QueueId op_cq_id(1);   // device operation, read/write command queue id

    boost::asio::thread_pool pool(devices.size());

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());

        TensorSpec tensor_spec(
            input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));

        log_info(LogTest, "Enqueue Operations before AllGather");
        std::vector<std::future<void>> futures;
        for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, promise]() mutable {
                // Generate input data for each device
                auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    host_data[j] = bfloat16(static_cast<float>(dev_idx));
                }

                auto& single_mesh = single_meshes[dev_idx];
                auto input_buffer = tt::tt_metal::tensor_impl::allocate_device_buffer(single_mesh.get(), tensor_spec);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer, {MeshCoordinate(0, 0)}};

                // TODO (#25340): Switch to use create_device_tensor? (TensorTopology logic should mirror
                // create_device_tensor)
                Tensor input_tensor = Tensor(input_storage, tensor_spec, TensorTopology{});

                // Enqueue write_buffer to the operation`s command queue and record the event
                ttnn::write_buffer(op_cq_id, input_tensor, {host_data});

                // Enqueue multiple operations to the operation command queue
                // Set output_tensor into device_tensor for allgather
                device_tensors[dev_idx] = ttnn::test_utils::dispatch_ops_to_device(input_tensor, op_cq_id);

                auto& op_cq_2 = single_mesh->mesh_command_queue(op_cq_id.get());
                auto operation_event = ttnn::record_event(op_cq_2);
                // Enqueue the task waiting for the operation_event to the ccl`s command queue
                auto& ccl_cq = single_mesh->mesh_command_queue(ccl_cq_id.get());
                ttnn::wait_for_event(ccl_cq, operation_event);

                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        log_info(LogTest, "Enqueue AllGather");

        // Enqueue the all_gather_command_processor_async operation on each device.
        // It does not support command queue ID as a parameter and internally uses command queue 0.
        std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> multi_dev_semaphore = {
            multi_device_global_semaphore};
        const std::vector<Tensor> gathered_tensors = ttnn::experimental::all_gather_command_processor_async(
            device_tensors,
            /* dim */ 0,
            multi_dev_semaphore,
            /* persistent_output_buffer */ std::nullopt,
            /* num_links */ 1,
            operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            ttnn::ccl::Topology::Linear,
            /* cluster_axis */ std::nullopt,
            SubDeviceId(0));

        log_info(LogTest, "Enqueue dummy ops");
        for (size_t dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, promise]() mutable {
                auto& single_mesh = single_meshes[dev_idx];
                // Wait for the CCL operation to finish before enqueueing the dummy ops, because ownership of the
                // workers needs to be transferred between CQs.
                ttnn::queue_synchronize(single_mesh->mesh_command_queue(ccl_cq_id.get()));

                auto dummy_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    dummy_data[j] = bfloat16(static_cast<float>(dev_idx));
                }
                auto dummy_buffer = tt::tt_metal::tensor_impl::allocate_device_buffer(single_mesh.get(), tensor_spec);
                auto dummy_storage = tt::tt_metal::DeviceStorage{dummy_buffer, {MeshCoordinate(0, 0)}};

                Tensor dummy_tensor = Tensor(dummy_storage, tensor_spec, TensorTopology{});
                ttnn::write_buffer(op_cq_id, dummy_tensor, {dummy_data});
                ttnn::test_utils::dispatch_ops_to_device(dummy_tensor, op_cq_id);

                promise->set_value();
            });
            futures.back().wait();
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
                auto& single_mesh = single_meshes[dev_idx];
                auto ccl_event = ttnn::record_event(single_mesh->mesh_command_queue(ccl_cq_id.get()));
                // Enqueue the task waiting for the operation_event to the ccl`s command queue
                ttnn::wait_for_event(single_mesh->mesh_command_queue(op_cq_id.get()), ccl_event);

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
                ttnn::read_buffer(op_cq_id, device_tensor, {output_data});

                for (int j = 0; j < device_tensor.physical_volume(); j++) {
                    int base = j / num_elems;  // dev_idx
                    ASSERT_EQ(static_cast<float>(output_data[j]), (-1.0 * base * 32.0 + 128));
                }
                log_info(LogTest, "Device{} Compare Success", device->id());
            });
        }
    }

    pool.join();

    for (auto& single_mesh : single_meshes) {
        auto& op_cq_1 = single_mesh->mesh_command_queue(op_cq_id.get());
        ttnn::queue_synchronize(op_cq_1);
    }

    log_info(tt::LogTest, "Finished");
}

TEST_F(MultiCQFabricMeshDevice2x4Fixture, AsyncExecutionWorksMultithreadCQ0) {
    constexpr size_t test_expected_num_devices = 4;

    MeshDevice* mesh_device = this->mesh_device_.get();

    // build a line of devices
    std::vector<std::shared_ptr<distributed::MeshDevice>> single_meshes = {
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0)),
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 1)),
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 2)),
        mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 3)),
    };
    std::vector<IDevice*> devices = {
        single_meshes[0].get(),
        single_meshes[1].get(),
        single_meshes[2].get(),
        single_meshes[3].get(),
    };

    // https://github.com/tenstorrent/tt-metal/issues/24235
    // Remove when https://github.com/tenstorrent/tt-metal/issues/25418 is fixed.
    for (auto& device : single_meshes) {
        device->disable_and_clear_program_cache();
    }

    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);

    log_info(LogTest, "Creating Global Semaphore for Ccl Ops");

    auto multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore_with_same_address(
        devices,
        devices[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                             // initial value
        tt::tt_metal::BufferType::L1,  // buffer type
        10                             // attempts
    );

    const int batch_size = 8;
    const int sequence_length = 1024;
    const int embedding_dim = 768;

    const ttnn::Shape input_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    QueueId op_ccl_cq_id(0);  // device operation, ccl command queue id
    QueueId mem_cq_id(1);     // read/write command queue id

    boost::asio::thread_pool pool(devices.size());

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());

        TensorSpec tensor_spec(
            input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));

        log_info(LogTest, "Enqueue Operations before AllGather");
        std::vector<std::future<void>> futures;
        for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, promise]() mutable {
                // Generate input data for each device
                auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    host_data[j] = bfloat16(static_cast<float>(dev_idx));
                }

                auto& single_mesh = single_meshes[dev_idx];
                auto input_buffer = tt::tt_metal::tensor_impl::allocate_device_buffer(single_mesh.get(), tensor_spec);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer, {MeshCoordinate(0, 0)}};

                // TODO (#25340): Switch to use create_device_tensor? (TensorTopology logic should mirror
                // create_device_tensor)
                Tensor input_tensor = Tensor(input_storage, tensor_spec, TensorTopology{});

                // Enqueue write_buffer to the operation`s command queue and record the event
                ttnn::write_buffer(mem_cq_id, input_tensor, {host_data});

                auto& mem_cq = single_mesh->mesh_command_queue(mem_cq_id.get());
                auto operation_event = ttnn::record_event(mem_cq);
                // Enqueue the task waiting for the operation_event to the ccl`s command queue
                auto& ccl_cq = single_mesh->mesh_command_queue(op_ccl_cq_id.get());
                ttnn::wait_for_event(ccl_cq, operation_event);

                // Enqueue multiple operations to the operation command queue
                // Set output_tensor into device_tensor for allgather
                device_tensors[dev_idx] = ttnn::test_utils::dispatch_ops_to_device(input_tensor, op_ccl_cq_id);

                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        log_info(LogTest, "Enqueue AllGather");

        // Enqueue the all_gather_command_processor_async operation on each device.
        // It does not support command queue ID as a parameter and internally uses command queue 0.
        std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> multi_dev_semaphore = {
            multi_device_global_semaphore};
        const std::vector<Tensor> gathered_tensors = ttnn::experimental::all_gather_command_processor_async(
            device_tensors,
            /* dim */ 0,
            multi_dev_semaphore,
            /* persistent_output_buffer */ std::nullopt,
            /* num_links */ 1,
            operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            ttnn::ccl::Topology::Linear,
            /* cluster_axis */ std::nullopt,
            SubDeviceId(0));

        log_info(LogTest, "Enqueue dummy ops");
        for (size_t dev_idx = 0; dev_idx < devices.size(); dev_idx++) {
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, promise]() mutable {
                // Enqueue the dummy ops at the same time as the AllGather is running.
                auto& single_mesh = single_meshes[dev_idx];

                auto dummy_data = std::shared_ptr<bfloat16[]>(new bfloat16[num_elems]);
                for (int j = 0; j < num_elems; j++) {
                    dummy_data[j] = bfloat16(static_cast<float>(dev_idx));
                }
                auto dummy_buffer = tt::tt_metal::tensor_impl::allocate_device_buffer(single_mesh.get(), tensor_spec);
                auto dummy_storage = tt::tt_metal::DeviceStorage{dummy_buffer, {MeshCoordinate(0, 0)}};

                Tensor dummy_tensor = Tensor(dummy_storage, tensor_spec, TensorTopology{});
                ttnn::write_buffer(op_ccl_cq_id, dummy_tensor, {dummy_data});
                ttnn::test_utils::dispatch_ops_to_device(dummy_tensor, op_ccl_cq_id);

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
                auto& single_mesh = single_meshes[dev_idx];
                auto ccl_event = ttnn::record_event(single_mesh->mesh_command_queue(op_ccl_cq_id.get()));
                // Enqueue the task waiting for the operation_event to the ccl`s command queue
                ttnn::wait_for_event(single_mesh->mesh_command_queue(mem_cq_id.get()), ccl_event);

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
                ttnn::read_buffer(mem_cq_id, device_tensor, {output_data});

                for (int j = 0; j < device_tensor.physical_volume(); j++) {
                    int base = j / num_elems;  // dev_idx
                    ASSERT_EQ(static_cast<float>(output_data[j]), (-1.0 * base * 32.0 + 128));
                }
                log_info(LogTest, "Device{} Compare Success", device->id());
            });
        }
    }

    pool.join();

    for (auto& single_mesh : single_meshes) {
        auto& mem_cq_3 = single_mesh->mesh_command_queue(mem_cq_id.get());
        ttnn::queue_synchronize(mem_cq_3);
    }

    log_info(tt::LogTest, "Finished");
}

}  // namespace ttnn::distributed::test
