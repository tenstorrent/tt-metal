// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/device_context.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/distributed/types.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/span.hpp>

#include "gtest/gtest.h"

namespace ttnn::distributed::test {

using namespace tt;
using namespace tt_metal;

using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshShape;

// Custom Fixture using 1D Fabric on a Multi-CQ MeshDevice
class MultiCQFabricMeshDevice2x4Fixture : public MeshDeviceFixtureBase {
protected:
    MultiCQFabricMeshDevice2x4Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 4}, .num_cqs = 2}) {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    }
    void TearDown() override {
        MeshDeviceFixtureBase::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

// TODO(#30692): Re-enable after migrating to aggregated tensor + semaphore-free all-gather APIs.
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

    const int batch_size = 8;
    const int sequence_length = 1024;
    const int embedding_dim = 768;

    uint32_t dim = 0;
    const ttnn::Shape input_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    ttnn::Shape output_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    output_shape[dim] *= devices.size();
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    QueueId op_cq_id(0);  // operation command queue id
    boost::asio::thread_pool pool(devices.size());

    TensorSpec tensor_spec(input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));
    TensorSpec output_tensor_spec(
        output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());
        std::vector<Tensor> output_tensors(devices.size());

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

                auto output_buffer =
                    tt::tt_metal::tensor_impl::allocate_device_buffer(single_mesh.get(), output_tensor_spec);
                auto output_storage = tt::tt_metal::DeviceStorage{output_buffer, {MeshCoordinate(0, 0)}};
                Tensor output_tensor = Tensor(output_storage, output_tensor_spec, TensorTopology{});

                // Enqueue write_buffer to the read/write command queue and record the event
                ttnn::write_buffer(QueueId(op_cq_id), input_tensor, {host_data});

                // Enqueue multiple operations to the operation command queue
                // Set output_tensor into device_tensor for allreduce
                device_tensors[dev_idx] = ttnn::test_utils::dispatch_ops_to_device(input_tensor, QueueId(op_cq_id));
                output_tensors[dev_idx] = output_tensor;

                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        log_info(LogTest, "Enqueue AllGather");

        auto aggregated_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(device_tensors);
        auto aggregated_output_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(output_tensors);
        // Quiesce parent mesh before all gather
        mesh_device_->quiesce_devices();

        auto all_gathered_tensor = ttnn::all_gather(
            aggregated_tensor,
            /* dim */ 0,
            std::nullopt,
            std::nullopt,
            aggregated_output_tensor);

        // Quiesce parent mesh after all gather
        mesh_device_->quiesce_devices();

        auto gathered_tensors = output_tensors;

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

        log_info(LogTest, "read_buffer");
        // Read the values from each device and compare them with the results calculated on the host
        for (size_t i = 0; i < devices.size(); ++i) {
            auto* device = devices[i];
            auto device_tensor = gathered_tensors[i];
            boost::asio::post(pool, [&, device, num_elems, device_tensor]() mutable {
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

    const int batch_size = 8;
    const int sequence_length = 1024;
    const int embedding_dim = 768;

    uint32_t dim = 0;
    const ttnn::Shape input_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    ttnn::Shape output_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    output_shape[dim] *= devices.size();
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    QueueId ccl_cq_id(0);  // ccl operation command queue id
    QueueId op_cq_id(1);   // device operation, read/write command queue id

    boost::asio::thread_pool pool(devices.size());

    TensorSpec tensor_spec(input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));
    TensorSpec output_tensor_spec(
        output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());
        std::vector<Tensor> output_tensors(devices.size());

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

                auto output_buffer =
                    tt::tt_metal::tensor_impl::allocate_device_buffer(single_mesh.get(), output_tensor_spec);
                auto output_storage = tt::tt_metal::DeviceStorage{output_buffer, {MeshCoordinate(0, 0)}};
                Tensor output_tensor = Tensor(output_storage, output_tensor_spec, TensorTopology{});

                // Enqueue write_buffer to the operation`s command queue and record the event
                ttnn::write_buffer(op_cq_id, input_tensor, {host_data});

                // Enqueue multiple operations to the operation command queue
                // Set output_tensor into device_tensor for allgather
                device_tensors[dev_idx] = ttnn::test_utils::dispatch_ops_to_device(input_tensor, op_cq_id);
                output_tensors[dev_idx] = output_tensor;

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

        auto aggregated_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(device_tensors);
        auto aggregated_output_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(output_tensors);

        // Quiesce parent mesh before all gather
        mesh_device_->quiesce_devices();

        auto all_gathered_tensor = ttnn::all_gather(
            aggregated_tensor,
            /* dim */ dim,
            std::nullopt,
            std::nullopt,
            aggregated_output_tensor);

        // Quiesce parent mesh after all gather
        mesh_device_->quiesce_devices();

        auto gathered_tensors = output_tensors;

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
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, promise]() mutable {
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

        log_info(LogTest, "read_buffer");
        // Read the values from each device and compare them with the results calculated on the host
        for (size_t i = 0; i < devices.size(); ++i) {
            auto* device = devices[i];
            auto device_tensor = gathered_tensors[i];
            boost::asio::post(pool, [&, device, num_elems, device_tensor]() mutable {
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

    const int batch_size = 8;
    const int sequence_length = 1024;
    const int embedding_dim = 768;

    uint32_t dim = 0;
    const ttnn::Shape input_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    ttnn::Shape output_shape = ttnn::Shape{1, batch_size, sequence_length, embedding_dim};
    output_shape[dim] *= devices.size();
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    QueueId op_ccl_cq_id(0);  // device operation, ccl command queue id
    QueueId mem_cq_id(1);     // read/write command queue id

    boost::asio::thread_pool pool(devices.size());

    TensorSpec tensor_spec(input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));
    TensorSpec output_tensor_spec(
        output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in_memory_config));

    for (int outer_loop = 0; outer_loop < 1; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        std::vector<Tensor> device_tensors(devices.size());
        std::vector<Tensor> output_tensors(devices.size());

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

                auto output_buffer =
                    tt::tt_metal::tensor_impl::allocate_device_buffer(single_mesh.get(), output_tensor_spec);
                auto output_storage = tt::tt_metal::DeviceStorage{output_buffer, {MeshCoordinate(0, 0)}};
                Tensor output_tensor = Tensor(output_storage, output_tensor_spec, TensorTopology{});

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
                output_tensors[dev_idx] = output_tensor;

                promise->set_value();
            });
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        log_info(LogTest, "Enqueue AllGather");

        auto aggregated_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(device_tensors);
        auto aggregated_output_tensor = tt::tt_metal::experimental::unit_mesh::aggregate(output_tensors);

        // Quiesce parent mesh before all gather
        mesh_device_->quiesce_devices();

        auto all_gathered_tensor = ttnn::all_gather(
            aggregated_tensor,
            /* dim */ dim,
            std::nullopt,
            std::nullopt,
            aggregated_output_tensor);

        // Quiesce parent mesh after all gather
        mesh_device_->quiesce_devices();

        auto gathered_tensors = output_tensors;

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
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            boost::asio::post(pool, [&, dev_idx, promise]() mutable {
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

        log_info(LogTest, "read_buffer");
        // Read the values from each device and compare them with the results calculated on the host
        for (size_t i = 0; i < devices.size(); ++i) {
            auto* device = devices[i];
            auto device_tensor = gathered_tensors[i];

            boost::asio::post(pool, [&, device, num_elems, device_tensor]() mutable {
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

// Sets up 2 subdevices (IDs 0, 1), each with 2 cores in one row, so CCL and matmul can run on different subdevices.
static tt::tt_metal::SubDeviceManagerId setup_two_sub_devices_for_parallel(MeshDevice* device) {
    constexpr int cores_per_subdevice = 2;
    constexpr int num_sub_devices = 2;
    std::vector<tt::tt_metal::CoreRangeSet> core_range_sets;
    core_range_sets.reserve(num_sub_devices);
    for (int row = 0; row < num_sub_devices; ++row) {
        tt::tt_metal::CoreRange range(
            tt::tt_metal::CoreCoord(0, row), tt::tt_metal::CoreCoord(cores_per_subdevice - 1, row));
        core_range_sets.push_back(tt::tt_metal::CoreRangeSet(range));
    }
    std::vector<tt::tt_metal::SubDevice> sub_devices;
    sub_devices.reserve(num_sub_devices);
    for (int i = 0; i < num_sub_devices; ++i) {
        sub_devices.push_back(
            tt::tt_metal::SubDevice(ttsl::Span<const tt::tt_metal::CoreRangeSet>(&core_range_sets[i], 1)));
    }
    const auto id = device->create_sub_device_manager(
        ttsl::Span<const tt::tt_metal::SubDevice>(sub_devices.data(), sub_devices.size()), tt::tt_metal::DeviceAddr{0});
    device->load_sub_device_manager(id);
    const std::array<tt::tt_metal::SubDeviceId, num_sub_devices> ids = {
        tt::tt_metal::SubDeviceId{0}, tt::tt_metal::SubDeviceId{1}};
    device->set_sub_device_stall_group(ttsl::Span<const tt::tt_metal::SubDeviceId>(ids.data(), ids.size()));
    return id;
}

// Runs CCL all_gather (CQ 0, subdevice 0) and matmul (CQ 1, subdevice 1) in parallel so both execute on different
// subdevices, matching the Python test pattern.
TEST_F(MultiCQFabricMeshDevice2x4Fixture, ParallelCclAndMatmul) {
    constexpr size_t num_devices = 4;
    MeshDevice* mesh_device = this->mesh_device_.get();

    const tt::tt_metal::SubDeviceManagerId sub_device_manager_id = setup_two_sub_devices_for_parallel(mesh_device);

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

    const uint32_t dim = 0;
    const ttnn::Shape ag_input_shape = ttnn::Shape{1, 1, 32, 32};
    ttnn::Shape ag_output_shape = ag_input_shape;
    ag_output_shape[dim] *= num_devices;
    const MemoryConfig mem_cfg(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto ag_num_elems = ag_input_shape.volume();

    TensorSpec ag_tensor_spec(ag_input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));
    TensorSpec ag_output_spec(ag_output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));

    std::vector<Tensor> ag_input_tensors(num_devices), ag_output_tensors(num_devices);
    for (size_t i = 0; i < num_devices; i++) {
        std::vector<bfloat16> in_data(ag_num_elems, bfloat16(static_cast<float>(i)));
        ag_input_tensors[i] = Tensor::from_vector(std::move(in_data), ag_tensor_spec).to_device(single_meshes[i].get());
        std::vector<bfloat16> out_data(ag_output_spec.logical_shape().volume(), bfloat16(-1));
        ag_output_tensors[i] =
            Tensor::from_vector(std::move(out_data), ag_output_spec).to_device(single_meshes[i].get());
    }
    auto aggregated_input = tt::tt_metal::experimental::unit_mesh::aggregate(ag_input_tensors);
    auto aggregated_output = tt::tt_metal::experimental::unit_mesh::aggregate(ag_output_tensors);

    const ttnn::Shape matmul_shape = ttnn::Shape{1, 1, 32, 32};
    TensorSpec matmul_spec(matmul_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));
    std::vector<bfloat16> a_data(matmul_shape.volume(), bfloat16(1.0f));
    std::vector<bfloat16> b_data(matmul_shape.volume(), bfloat16(1.0f));
    Tensor tensor_a = Tensor::from_vector(a_data, matmul_spec).to_device(single_meshes[0].get());
    Tensor tensor_b = Tensor::from_vector(b_data, matmul_spec).to_device(single_meshes[0].get());

    mesh_device_->quiesce_devices();

    QueueId ccl_cq(0);
    QueueId matmul_cq(1);
    constexpr tt::tt_metal::SubDeviceId ccl_subdevice{0};
    constexpr tt::tt_metal::SubDeviceId matmul_subdevice{1};

    auto ccl_future = std::async(std::launch::async, [&]() {
        auto sub_guard = ttnn::DeviceContext(mesh_device).set_current_sub_device(ccl_subdevice);
        auto cq_guard = ttnn::with_command_queue_id(ccl_cq);
        return ttnn::all_gather(
            aggregated_input, static_cast<int32_t>(dim), std::nullopt, std::nullopt, aggregated_output);
    });

    Tensor matmul_result;
    auto matmul_future = std::async(std::launch::async, [&]() {
        auto sub_guard = ttnn::DeviceContext(mesh_device).set_current_sub_device(matmul_subdevice);
        auto cq_guard = ttnn::with_command_queue_id(matmul_cq);
        matmul_result = ttnn::matmul(
            tensor_a,
            tensor_b,
            false,
            false,
            std::nullopt,
            std::nullopt,
            ttnn::operations::matmul::MatmulMultiCoreProgramConfig{});
    });

    ASSERT_NO_THROW(ccl_future.get());
    ASSERT_NO_THROW(matmul_future.get());

    mesh_device_->quiesce_devices();

    for (size_t i = 0; i < num_devices; i++) {
        auto data = ag_output_tensors[i].to_vector<bfloat16>();
        for (size_t j = 0; j < data.size(); j++) {
            size_t slice = j / ag_num_elems;
            EXPECT_EQ(static_cast<float>(data[j]), static_cast<float>(slice))
                << "all_gather output device " << i << " idx " << j;
        }
    }

    auto matmul_cpu = matmul_result.cpu().to_vector<bfloat16>();
    const float expected_val = 32.0f;  // 32x32 matmul of ones: each output = sum of 32 ones
    for (size_t i = 0; i < matmul_cpu.size(); i++) {
        EXPECT_NEAR(static_cast<float>(matmul_cpu[i]), expected_val, 2.0f) << "matmul output idx " << i;
    }

    mesh_device->reset_sub_device_stall_group();
    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager_id);

    log_info(tt::LogTest, "ParallelCclAndMatmul finished");
}

}  // namespace ttnn::distributed::test
