// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.hpp"
#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/cpp/ttnn/multi_device.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn_multi_command_queue_fixture.hpp"

using namespace tt;
using namespace tt_metal;

// We use this to dispatch a single device operation asynchronously
// Needed to reproduce the deadlock scenario with a very specific pattern of commands
// This can go away once device_operation::run will be made async and ccl op is moved to the new tmp-based DeviceOperation
namespace async_detail {
template<typename OpConfig>
std::vector<Tensor> run_operation(
    uint8_t cq_id,
    OpConfig devop,
    const tt::tt_metal::operation::Tensors& input_tensors,
    const tt::tt_metal::operation::OptionalConstTensors& optional_input_tensors = {},
    const tt::tt_metal::operation::OptionalTensors& optional_output_tensors = {}) {
    static_assert(tt::tt_metal::operation::detail::is_device_operation<OpConfig>(), "ttnn::run_operation can only dispatch Device Operations!");
    // Create output tensor vector by examining the number of output shapes created by the device operation
    std::vector<Tensor> outputs(tt::tt_metal::operation::DeviceOperation<tt::tt_metal::operation::Tensors>(devop).compute_output_shapes(input_tensors).size());
    // Populate the workers of the output tensors, based on the input tensors. This is needed for the async engine.
    for (int i = 0; i < outputs.size(); i++) {
        outputs[i] = Tensor(tt::tt_metal::operation::get_workers_for_op_output(std::move(input_tensors), std::move(optional_input_tensors)));
    }
    // Send the operation to the async engine, which will populate the output tensors.
    for (auto worker : outputs.at(0).workers) {
        tt::tt_metal::operation::launch_op(
            [devop, worker, cq_id] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                return operation::run(std::move(devop), input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
            }, input_tensors, outputs, optional_input_tensors, optional_output_tensors);
    }
    return outputs;
}
} // namespace async_detail

TEST(TGTests, TestAllGatherDeadlock) {
    if (not tt::Cluster::instance().is_galaxy_cluster()) {
        GTEST_SKIP() << "Skipping Galaxy test, since this is not a Galaxy System";
    }
    // Construct the remote devices in this cluster. TTNN Device Mesh APIs need this to be passed in.
    // Do this using TT Cluster APIs, since device IDs may change in the future.
    uint32_t num_devices_in_tunnel = tt::Cluster::instance().get_mmio_device_max_tunnel_depth(0);
    uint32_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
    uint32_t cluster_tunnel_count = tt::Cluster::instance().get_mmio_device_tunnel_count(0);
    TT_FATAL(num_devices_in_tunnel == 4, "Expected Galaxy to have tunnel depth of 4");
    TT_FATAL(num_mmio_devices * cluster_tunnel_count == 8, "Expected 8 tunnels in a Galaxy");

    std::vector<chip_id_t> all_device_ids = {};
    for (uint32_t mmio_idx = 0; mmio_idx < num_mmio_devices; mmio_idx++) {
        auto tunnels_from_mmio = tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_idx);
        for (uint32_t tunnel_idx = 0; tunnel_idx < tunnels_from_mmio.size(); tunnel_idx++) {
            auto remote_devices_in_tunnel = tunnels_from_mmio.at(tunnel_idx);
            all_device_ids.insert(all_device_ids.end(), remote_devices_in_tunnel.begin(), remote_devices_in_tunnel.end());
        }
    }

    // Create the device mesh: Grid size is <num_tunnels, tunnel_depth>.
    auto mesh = ttnn::multi_device::open_mesh_device({cluster_tunnel_count * num_mmio_devices, num_devices_in_tunnel}, all_device_ids, 0, 0, 1, DispatchCoreType::WORKER);

    // Setup input data and output data containers
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};
    ttnn::Shape shape = ttnn::Shape(tt::tt_metal::LegacyShape({1, 1, 32, 16384}));
    uint32_t buf_size_datums = 32 * 16384;
    uint32_t datum_size_bytes = 2;
    auto host_data = std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums * num_devices_in_tunnel]);
    uint32_t outer_loops = 200;


    // Input to CCL is a tensor of 1s. The output should contain 4x the amount of data, but all values should be 1.
    for (int j = 0; j < buf_size_datums; j++) {
        host_data[j] = bfloat16(static_cast<float>(1));
    }
    // Iterate over each tunnel and run line all-gather multiple times.
    // For each tunnel, send adversarial traffic to the first chip, that can hang the network if the CCL is not tagged.
    for (uint32_t row = 0; row < 8; row++) {
        auto devs = mesh.get_devices_on_row(row);
        std::vector<uint32_t> device_ids = {};
        for (auto dev : devs) {
            device_ids.push_back(dev->id());
        }
        log_info(LogTest, "Running CCL Op on row {} for {} loops", row, outer_loops);
        log_info(LogTest, "Devices on row: {}", device_ids);
        for (int i = 0; i < outer_loops; i++) {
            std::vector<Tensor> output_tensors = {};
            uint32_t dev_idx = 0;
            if (i % 100 == 0) {
                log_info(LogTest, "Running iteration {}", i);
            }
            for (auto& dev : devs) {
                auto input_buffer = ttnn::allocate_buffer_on_device(buf_size_datums * datum_size_bytes, dev, shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
                Tensor input_tensor = Tensor(input_storage, shape, DataType::BFLOAT16, Layout::TILE);
                // Push inputs.
                ttnn::write_buffer(0, input_tensor, {host_data});
                // Configure CCL running on this device.
                uint32_t receiver_device_id = device_ids[(dev_idx) + 1 % num_devices_in_tunnel];
                uint32_t sender_device_id = device_ids[(dev_idx + num_devices_in_tunnel - 1) % num_devices_in_tunnel];
                auto all_gather_op = ttnn::LineAllGather{
                                        3, 2, num_devices_in_tunnel, dev_idx, std::nullopt, std::nullopt, receiver_device_id, sender_device_id, input_tensor.memory_config(), ttnn::all_gather_op::Topology::Linear};
                // Send CCL to this device. All CCLs will complete simultaneously.
                output_tensors.push_back(async_detail::run_operation(0, all_gather_op, {input_tensor}).at(0));
                // Expose deadlock: After the CCL is sent to the first device in the tunnel, send enough data to it to backpressure prefetch_h. This will block the
                // demux, which will prevent the CCL from being sent to additional chips. If the CCL has been tagged as having multi-device dependencies, deadlock should
                // get bypassed.
                if (!dev_idx) {
                    ttnn::write_buffer(0, input_tensor, {host_data});
                }
                dev_idx++;
            }
            // Readback data and verify correctness.
            for (auto& tensor : output_tensors) {
                ASSERT_EQ(tensor.get_shape(), ttnn::Shape(tt::tt_metal::LegacyShape({1, 1, 32, static_cast<uint32_t>(16384 * device_ids.size())})));
                ttnn::read_buffer(0, tensor, {readback_data});
                for (int j = 0; j < device_ids.size() * 32 * 16384; j++) {
                    ASSERT_EQ(readback_data[j].to_float(), 1);
                }
            }
        }
    }
    ttnn::multi_device::close_mesh_device(mesh);
}

TEST(TGTests, TestReduceScatterDeadlock) {
    if (not tt::Cluster::instance().is_galaxy_cluster()) {
        GTEST_SKIP() << "Skipping Galaxy test, since this is not a Galaxy System";
    }
    // Construct the remote devices in this cluster. TTNN Device Mesh APIs need this to be passed in.
    // Do this using TT Cluster APIs, since device IDs may change in the future.
    uint32_t num_devices_in_tunnel = tt::Cluster::instance().get_mmio_device_max_tunnel_depth(0);
    uint32_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
    uint32_t cluster_tunnel_count = tt::Cluster::instance().get_mmio_device_tunnel_count(0);
    TT_FATAL(num_devices_in_tunnel == 4, "Expected Galaxy to have tunnel depth of 4");
    TT_FATAL(num_mmio_devices * cluster_tunnel_count == 8, "Expected 8 tunnels in a Galaxy");

    std::vector<chip_id_t> all_device_ids = {};
    for (uint32_t mmio_idx = 0; mmio_idx < num_mmio_devices; mmio_idx++) {
        auto tunnels_from_mmio = tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_idx);
        for (uint32_t tunnel_idx = 0; tunnel_idx < tunnels_from_mmio.size(); tunnel_idx++) {
            auto remote_devices_in_tunnel = tunnels_from_mmio.at(tunnel_idx);
            all_device_ids.insert(all_device_ids.end(), remote_devices_in_tunnel.begin(), remote_devices_in_tunnel.end());
        }
    }

    // Create the device mesh: Grid size is <num_tunnels, tunnel_depth>.
    auto mesh = ttnn::multi_device::open_mesh_device({cluster_tunnel_count * num_mmio_devices, num_devices_in_tunnel}, all_device_ids, 0, 0, 1, DispatchCoreType::WORKER);
    // Create the outer ring on which Reduce Scatter will be run. This allows us to verify that there are no deadlocks when we send CCLs to the
    // first tunnel (forward path).
    std::vector<Device*> ring_devices = mesh.get_devices_on_row(0); // Tunnel 0
    std::vector<Device*> ring_devices_1 = mesh.get_devices_on_column(3); // Orthogonal to tunnel .. no deadlocks
    ring_devices_1 = std::vector<Device*>(ring_devices_1.begin() + 1, ring_devices_1.end());
    std::vector<Device*> ring_devices_2 = mesh.get_devices_on_row(7); // Tunnel 7 .. potential deadlocks with lack of buffering
    std::reverse(ring_devices_2.begin(), ring_devices_2.end());
    ring_devices_2 = std::vector<Device*>(ring_devices_2.begin() + 1, ring_devices_2.end());
    std::vector<Device*> ring_devices_3 = mesh.get_devices_on_column(0); // Orthogonal to tunnel .. no deadlocks
    std::reverse(ring_devices_3.begin(), ring_devices_3.end());
    ring_devices_3 = std::vector<Device*>(ring_devices_3.begin() + 1, ring_devices_3.end() - 1);

    ring_devices.insert(ring_devices.end(), ring_devices_1.begin(), ring_devices_1.end());
    ring_devices.insert(ring_devices.end(), ring_devices_2.begin(), ring_devices_2.end());
    ring_devices.insert(ring_devices.end(), ring_devices_3.begin(), ring_devices_3.end());

    // Setup input data and output data containers
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};
    ttnn::Shape shape = ttnn::Shape(tt::tt_metal::LegacyShape({1, 2, 256, static_cast<uint32_t>(256 * ring_devices.size())}));
    uint32_t buf_size_datums = 2 * 256 * 256 * 20;
    uint32_t datum_size_bytes = 2;
    // Output of reduce scatter is input_numel / num_devices_used_in_scatter_op
    auto host_data = std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums / ring_devices.size()]);
    uint32_t scatter_dim = 3;
    uint32_t outer_loops = 500;

    // Input to CCL is a tensor of 1s. The output will contain 20x (ring size) less data along the innermost dim,
    // with each entry == 20 * 1
    for (int j = 0; j < buf_size_datums; j++) {
        host_data[j] = bfloat16(static_cast<float>(1));
    }
    std::vector<uint32_t> device_ids = {};

    for (auto dev : ring_devices) {
        dev->enable_program_cache();
        device_ids.push_back(dev->id());
    }

    log_info(LogTest, "Running Reduce Scatter Op for {} loops", outer_loops);
    log_info(LogTest, "Devices in Ring: {}", device_ids);
    // Run reduce scatter multiple times.
    // For the first tunnel, send adversarial traffic that can clog the forward path, if the op is not tagged correctly.
    for (int i = 0; i < outer_loops; i++) {
        std::vector<Tensor> output_tensors = {};
        uint32_t dev_idx = 0;
        if (i % 100 == 0) {
            log_info(LogTest, "Running iteration {}", i);
        }
        for (auto& dev : ring_devices) {
            auto input_buffer = ttnn::allocate_buffer_on_device(buf_size_datums * datum_size_bytes, dev, shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
            auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
            Tensor input_tensor = Tensor(input_storage, shape, DataType::BFLOAT16, Layout::TILE);
            // Push inputs.
            ttnn::write_buffer(0, input_tensor, {host_data});
            // Configure CCL running on this device.
            uint32_t receiver_device_id = device_ids[(dev_idx + 1) % ring_devices.size()];
            uint32_t sender_device_id = device_ids[(dev_idx + ring_devices.size() - 1) % ring_devices.size()];
            auto all_gather_op = ttnn::ReduceScatter{
                                    ttnn::operations::binary::BinaryOpType::ADD, scatter_dim, 1, static_cast<uint32_t>(ring_devices.size()), dev_idx, receiver_device_id, sender_device_id, input_tensor.memory_config(), ttnn::all_gather_op::Topology::Ring};
            // Send CCL to this device. All CCLs will complete simultaneously.
            output_tensors.push_back(async_detail::run_operation(0, all_gather_op, {input_tensor}).at(0));
            // Expose deadlock: After the CCL is sent to a device in the first tunnel, send enough data to it to backpressure prefetch_h. This will block the
            // demux, which will prevent the CCL from being sent to additional chips on the tunnel. If the CCL has been tagged as having multi-device dependencies, deadlock should
            // get bypassed.
            // if (dev_idx < 3) {
                for (int j = 0; j < 16; j++) {
                    ttnn::write_buffer(0, input_tensor, {host_data});
                }
            // }
            dev_idx++;
        }
        // Readback data and verify correctness.
        for (auto& tensor : output_tensors) {
            ASSERT_EQ(tensor.get_shape(), ttnn::Shape(tt::tt_metal::LegacyShape({1, 2, 256, 256})));
            ttnn::read_buffer(0, tensor, {readback_data});
            for (int j = 0; j < 512 * 256; j++) {
                ASSERT_EQ(readback_data[j].to_float(), 20);
            }
        }
    }
    ttnn::multi_device::close_mesh_device(mesh);
}
