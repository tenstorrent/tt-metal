// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_coord.hpp"
#include "ttnn_test_fixtures.hpp"

#include <cmath>
#include <cstdlib>
#include <string_view>
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
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/distributed/types.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_builder_context.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/host_api.hpp>

#include "gtest/gtest.h"

namespace ttnn::distributed::test {

using namespace tt;
using namespace tt_metal;

using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshShape;

// Custom Fixture using 1D Fabric on a Multi-CQ MeshDevice.
//
// NOTE: Despite the class name ("2x4"), this fixture uses MeshShape{1, 4} (a 4-device line).
// The "2x4" in the name refers to the T3K hardware topology, not the mesh shape passed to
// MeshDevice. Renaming is deferred because several CI scripts + a reproducer script filter on
// this exact string (see tests/scripts/t3000/run_t3000_unit_tests.sh,
// tests/scripts/t3000/repro_ccl_cq0_hang.sh). A separate, distinct class with the same name
// (but actually 2x4) exists in tests/ttnn/unit_tests/gtests/ccl/test_multi_tensor_ccl.cpp.
//
// Also: because `config_.fabric_config` is left at its default (DISABLED), the base
// MeshDeviceFixtureBase::SetUp does NOT call the 5-arg SetFabricConfig that would set
// `fabric_tensix_config`. The ctor calls the 1-arg SetFabricConfig(FABRIC_1D) below, so
// `FabricTensixConfig` stays at DISABLED for this fixture. Device::quiesce_and_restart_fabric_workers()
// therefore early-returns on L431 for every device here. Any diagnostics assuming that
// function actually ran will be misleading for this test.
class MultiCQFabricMeshDevice2x4Fixture : public MeshDeviceFixtureBase {
protected:
    MultiCQFabricMeshDevice2x4Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 4}, .num_cqs = 2}) {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    }
    void SetUp() override {
        // Escape hatch for CI while the chip-3 CQ0 AllGather hang is under investigation.
        // Setting TT_METAL_DISABLE_ASYNC_CQ0_T3K_TEMP=1 skips the body of these tests but
        // leaves them present so local reproducers and bisects continue to work. Remove
        // once the underlying hang is fixed.
        //
        // ERISC race condition fixes are confirmed working (branch nsexton/0-racecondition-hunt):
        // CclAsyncOp.ReduceScatterSmall_PersistentFabric and all 19 FabricSendRecv2x4Tests now
        // pass consistently. The remaining hang here is a distinct underlying issue: Tensix
        // workers on the far N300 chips (device IDs 4-7, accessed via non-MMIO ETH fabric)
        // perform an unsafe NOC access at 0x880030060 during "Enqueue dummy ops" after
        // ttnn::all_gather. The hang manifests at dispatch_thread_pool_->wait() in
        // enqueue_write_shards_nolock() and is not related to the ERISC firmware init race.
        if (const char* disable = std::getenv("TT_METAL_DISABLE_ASYNC_CQ0_T3K_TEMP");
            disable != nullptr && std::string_view(disable) == "1") {
            GTEST_SKIP() << "Temporarily disabled via TT_METAL_DISABLE_ASYNC_CQ0_T3K_TEMP=1 "
                            "while chip-3 AllGather hang is being root-caused.";
        }
        setenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART", "1", /*overwrite=*/0);
        MeshDeviceFixtureBase::SetUp();
    }
    void TearDown() override {
        MeshDeviceFixtureBase::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }

    // Diagnostic: for every device in the mesh, read and log fabric ETH router status on every
    // active fabric ETH channel. Gated on env TT_METAL_FABRIC_HEALTH_PROBE=1 to avoid log noise
    // in normal CI runs. Used to isolate whether ETH router state decays across
    // quiesce_devices() iterations (OQ2) or AllGather operations (plan Experiment D).
    void log_fabric_eth_health_for_all_devices(const std::string& label) const {
        const char* env_flag = std::getenv("TT_METAL_FABRIC_HEALTH_PROBE");
        if (env_flag == nullptr || env_flag[0] == '\0' || env_flag[0] == '0') {
            return;
        }
        if (!mesh_device_) {
            return;
        }
        const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
        if (!tt::tt_fabric::is_tt_fabric_config(fabric_config)) {
            log_info(tt::LogMetal, "[fabric_eth_health:{}] skipped: fabric_config not a tt_fabric config", label);
            return;
        }
        auto& metal_context = tt::tt_metal::MetalContext::instance();
        auto& control_plane = metal_context.get_control_plane();
        const auto& builder_ctx = control_plane.get_fabric_context().get_builder_context();
        const auto [edm_status_address, expected_status] = builder_ctx.get_fabric_router_sync_address_and_status();
        auto& cluster = metal_context.get_cluster();
        for (auto* idev : mesh_device_->get_devices()) {
            const auto chip_id = idev->id();
            if (builder_ctx.get_num_fabric_initialized_routers(chip_id) == 0) {
                log_info(
                    tt::LogMetal,
                    "[fabric_eth_health:{}] Device {} skipped: no initialized fabric routers",
                    label,
                    chip_id);
                continue;
            }
            const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(chip_id);
            const auto active_eth_channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);
            for (const auto& [chan_id, _direction] : active_eth_channels) {
                const auto eth_logical_core =
                    cluster.get_soc_desc(chip_id).get_eth_core_for_channel(chan_id, CoordSystem::LOGICAL);
                uint32_t status_word = 0;
                try {
                    const auto status =
                        cluster.read_core<uint32_t>(chip_id, eth_logical_core, edm_status_address, sizeof(uint32_t));
                    status_word = status.empty() ? 0u : status[0];
                } catch (const std::exception& e) {
                    log_info(
                        tt::LogMetal,
                        "[fabric_eth_health:{}] Device {} chan {} read FAILED: {}",
                        label,
                        chip_id,
                        chan_id,
                        e.what());
                    continue;
                }
                const bool ready = (status_word == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC));
                const bool terminated = (status_word == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED));
                const char* state = ready ? "READY_FOR_TRAFFIC" : (terminated ? "TERMINATED" : "OTHER");
                log_info(
                    tt::LogMetal,
                    "[fabric_eth_health:{}] Device {} chan {} state={} status=0x{:08x} expected=0x{:08x}",
                    label,
                    chip_id,
                    chan_id,
                    state,
                    status_word,
                    expected_status);
            }
        }
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

                auto& single_mesh = single_meshes[dev_idx];
                Tensor input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, single_mesh.get());
                Tensor output_tensor = tt::tt_metal::create_device_tensor(output_tensor_spec, single_mesh.get());

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
        log_info(LogTest, "[AsyncExecutionWorksCQ0] pre-AllGather quiesce_devices() begin");
        log_fabric_eth_health_for_all_devices("pre-allgather-pre-quiesce");
        mesh_device_->quiesce_devices();
        log_fabric_eth_health_for_all_devices("pre-allgather-post-quiesce");
        log_info(LogTest, "[AsyncExecutionWorksCQ0] pre-AllGather quiesce_devices() done; calling ttnn::all_gather");

        auto all_gathered_tensor = ttnn::all_gather(
            aggregated_tensor,
            /* dim */ 0,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            aggregated_output_tensor);

        log_info(LogTest, "[AsyncExecutionWorksCQ0] ttnn::all_gather returned; post-AllGather quiesce_devices() begin");
        log_fabric_eth_health_for_all_devices("post-allgather-pre-quiesce");
        mesh_device_->quiesce_devices();
        log_fabric_eth_health_for_all_devices("post-allgather-post-quiesce");
        log_info(LogTest, "[AsyncExecutionWorksCQ0] post-AllGather quiesce_devices() done");

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
                Tensor dummy_tensor = tt::tt_metal::create_device_tensor(tensor_spec, single_mesh.get());
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
                Tensor input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, single_mesh.get());
                Tensor output_tensor = tt::tt_metal::create_device_tensor(output_tensor_spec, single_mesh.get());

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

        log_info(LogTest, "[AsyncExecutionWorksCQ0CQ1] pre-AllGather quiesce_devices() begin");
        log_fabric_eth_health_for_all_devices("cq0cq1-pre-allgather-pre-quiesce");
        mesh_device_->quiesce_devices();
        log_fabric_eth_health_for_all_devices("cq0cq1-pre-allgather-post-quiesce");
        log_info(LogTest, "[AsyncExecutionWorksCQ0CQ1] pre-AllGather quiesce_devices() done; calling ttnn::all_gather");

        auto all_gathered_tensor = ttnn::all_gather(
            aggregated_tensor,
            /* dim */ dim,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            aggregated_output_tensor);

        log_info(
            LogTest, "[AsyncExecutionWorksCQ0CQ1] ttnn::all_gather returned; post-AllGather quiesce_devices() begin");
        log_fabric_eth_health_for_all_devices("cq0cq1-post-allgather-pre-quiesce");
        mesh_device_->quiesce_devices();
        log_fabric_eth_health_for_all_devices("cq0cq1-post-allgather-post-quiesce");
        log_info(LogTest, "[AsyncExecutionWorksCQ0CQ1] post-AllGather quiesce_devices() done");

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
                Tensor dummy_tensor = tt::tt_metal::create_device_tensor(tensor_spec, single_mesh.get());
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
                Tensor input_tensor = tt::tt_metal::create_device_tensor(tensor_spec, single_mesh.get());
                Tensor output_tensor = tt::tt_metal::create_device_tensor(output_tensor_spec, single_mesh.get());

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

        log_info(LogTest, "[AsyncExecutionWorksMultithreadCQ0] pre-AllGather quiesce_devices() begin");
        log_fabric_eth_health_for_all_devices("mt-pre-allgather-pre-quiesce");
        mesh_device_->quiesce_devices();
        log_fabric_eth_health_for_all_devices("mt-pre-allgather-post-quiesce");
        log_info(
            LogTest,
            "[AsyncExecutionWorksMultithreadCQ0] pre-AllGather quiesce_devices() done; calling ttnn::all_gather");

        auto all_gathered_tensor = ttnn::all_gather(
            aggregated_tensor,
            /* dim */ dim,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            aggregated_output_tensor);

        log_info(
            LogTest,
            "[AsyncExecutionWorksMultithreadCQ0] ttnn::all_gather returned; post-AllGather quiesce_devices() begin");
        log_fabric_eth_health_for_all_devices("mt-post-allgather-pre-quiesce");
        mesh_device_->quiesce_devices();
        log_fabric_eth_health_for_all_devices("mt-post-allgather-post-quiesce");
        log_info(LogTest, "[AsyncExecutionWorksMultithreadCQ0] post-AllGather quiesce_devices() done");

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
                Tensor dummy_tensor = tt::tt_metal::create_device_tensor(tensor_spec, single_mesh.get());
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

}  // namespace ttnn::distributed::test
