// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/distributed/api.hpp"
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
    const operation::Tensors& input_tensors,
    const operation::OptionalConstTensors& optional_input_tensors = {},
    const operation::OptionalTensors& optional_output_tensors = {}) {
    static_assert(operation::detail::is_device_operation<OpConfig>(), "ttnn::run_operation can only dispatch Device Operations!");
    // Create output tensor vector by examining the number of output shapes created by the device operation
    auto output_shapes = operation::DeviceOperation<operation::Tensors>(devop).compute_output_shapes(input_tensors);
    size_t output_shapes_size = 0;
    if (std::holds_alternative<std::vector<ttnn::SimpleShape>>(output_shapes)) {
        output_shapes_size = std::get<std::vector<ttnn::SimpleShape>>(output_shapes).size();
    } else {
        output_shapes_size = std::get<std::vector<tt::tt_metal::LegacyShape>>(output_shapes).size();
    }
    std::vector<Tensor> outputs(output_shapes_size);
    // Populate the workers of the output tensors, based on the input tensors. This is needed for the async engine.
    for (int i = 0; i < outputs.size(); i++) {
        outputs[i] = Tensor(operation::get_workers_for_op_output(std::move(input_tensors), std::move(optional_input_tensors)));
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

bool is_tg_system()
{
    const bool is_galaxy_system = tt::Cluster::instance().is_galaxy_cluster();
    const size_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
    const size_t num_devices = tt::Cluster::instance().number_of_user_devices();
    return is_galaxy_system && (num_mmio_devices == 4) && (num_devices == 32);
}

bool is_tgg_system()
{
    const bool is_galaxy_system = tt::Cluster::instance().is_galaxy_cluster();
    const size_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
    const size_t num_devices = tt::Cluster::instance().number_of_user_devices();
    return is_galaxy_system && (num_mmio_devices == 8) && (num_devices == 64);
}

ttnn::MeshShape get_mesh_shape()
{
    ttnn::MeshShape shape;
    if (is_tg_system())
    {
        shape = {8, 4};
    }
    else {
        TT_FATAL(is_tgg_system(), "Unsupported Galaxy system");
        shape = {8, 8};
    }
    return shape;
}

void validate_num_tunnels_and_tunnel_depth()
{
    const uint32_t num_devices_in_tunnel = tt::Cluster::instance().get_mmio_device_max_tunnel_depth(0);
    const uint32_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
    const uint32_t cluster_tunnel_count = tt::Cluster::instance().get_mmio_device_tunnel_count(0);
    TT_FATAL(num_devices_in_tunnel == 4, "Expected Galaxy to have tunnel depth of 4, detected tunnel depth of {}", num_devices_in_tunnel);
    const uint32_t num_tunnels = num_mmio_devices * cluster_tunnel_count;
    if (is_tg_system())
    {
        TT_FATAL(num_tunnels == 8, "Expected 8 tunnels in a TG system, detected {} tunnels", num_tunnels);
    }
    else if (is_tgg_system())
    {
        TT_FATAL(num_tunnels == 16, "Expected 16 tunnels in a TGG system, detected {} tunnels", num_tunnels);
    }
}

std::shared_ptr<bfloat16 []> create_container_for_readback_data(const uint32_t buf_size_datums)
{
    if (is_tg_system())
    {
        return std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums * 4]);
    }
    else
    {
        TT_FATAL(is_tgg_system(), "Unsupported Galaxy system");
        return std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums * 8]);
    }
}

TEST(GalaxyTests, TestAllGatherDeadlock) {
    if (not tt::Cluster::instance().is_galaxy_cluster()) {
        GTEST_SKIP() << "Skipping Galaxy test, since this is not a Galaxy System";
    }
    validate_num_tunnels_and_tunnel_depth();

    ttnn::MeshShape mesh_shape = get_mesh_shape();
    std::shared_ptr<ttnn::MeshDevice> mesh = ttnn::distributed::open_mesh_device(mesh_shape, 0, 0, 1, DispatchCoreType::WORKER);

    // Setup input data and output data containers
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};
    ttnn::SimpleShape shape{1, 1, 32, 16384};
    const uint32_t buf_size_datums = 32 * 16384;
    const uint32_t datum_size_bytes = 2;
    auto host_data = std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums]);
    std::shared_ptr<bfloat16 []> readback_data = create_container_for_readback_data(buf_size_datums);
    const uint32_t outer_loops = 200;

    // Input to CCL is a tensor of 1s. The output should contain 4x the amount of data, but all values should be 1.
    for (int j = 0; j < buf_size_datums; j++) {
        host_data[j] = bfloat16(static_cast<float>(1));
    }
    // Iterate over each row and run line all-gather multiple times.
    // For each row, send adversarial traffic to the first chip, that can hang the network if the CCL is not tagged.
    auto view = ttnn::MeshDeviceView(*mesh);
    for (uint32_t row = 0; row < 8; row++) {
        auto devs = view.get_devices_on_row(row);
        std::vector<uint32_t> device_ids = {};
        for (auto dev : devs) {
            device_ids.push_back(dev->id());
        }
        const uint32_t num_devices_in_row = device_ids.size();
        log_info(LogTest, "Running CCL Op on row {} for {} loops", row, outer_loops);
        log_info(LogTest, "Devices on row: {}", device_ids);
        for (int i = 0; i < outer_loops; i++) {
            std::vector<Tensor> output_tensors = {};
            uint32_t dev_idx = 0;
            if (i % 100 == 0) {
                log_info(LogTest, "Running iteration {}", i);
            }
            for (auto& dev : devs) {
                auto input_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(buf_size_datums * datum_size_bytes, dev, shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
                auto input_storage = DeviceStorage{input_buffer};
                Tensor input_tensor = Tensor(input_storage, shape, DataType::BFLOAT16, Layout::TILE);
                // Push inputs.
                ttnn::write_buffer(0, input_tensor, {host_data});
                // Configure CCL running on this device.
                uint32_t receiver_device_id = device_ids[(dev_idx) + 1 % num_devices_in_row];
                uint32_t sender_device_id = device_ids[(dev_idx + num_devices_in_row - 1) % num_devices_in_row];
                auto all_gather_op = ttnn::AllGather{
                                        3, 2, num_devices_in_row, dev_idx, std::nullopt, std::nullopt, receiver_device_id, sender_device_id, input_tensor.memory_config(), ttnn::ccl::Topology::Linear};
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
                ASSERT_EQ(tensor.get_shape(), ttnn::Shape(LegacyShape({1, 1, 32, static_cast<uint32_t>(16384 * device_ids.size())})));
                ttnn::read_buffer(0, tensor, {readback_data});
                for (int j = 0; j < device_ids.size() * 32 * 16384; j++) {
                    ASSERT_EQ(readback_data[j].to_float(), 1);
                }
            }
        }
    }
    ttnn::distributed::close_mesh_device(mesh);
}

TEST(GalaxyTests, TestReduceScatterDeadlock) {
    if (not tt::Cluster::instance().is_galaxy_cluster()) {
        GTEST_SKIP() << "Skipping Galaxy test, since this is not a Galaxy System";
    }
    validate_num_tunnels_and_tunnel_depth();

    ttnn::MeshShape mesh_shape = get_mesh_shape();
    std::shared_ptr<ttnn::MeshDevice> mesh = ttnn::distributed::open_mesh_device(mesh_shape, 0, 0, 1, DispatchCoreType::WORKER);
    // Create the outer ring on which Reduce Scatter will be run. This allows us to verify that there are no deadlocks when we send CCLs to the
    // first tunnel (forward path).
    auto view = ttnn::MeshDeviceView(*mesh);
    std::vector<Device*> ring_devices = view.get_devices_on_row(0); // Tunnel 0
    std::vector<Device*> ring_devices_1 = view.get_devices_on_column(mesh_shape.second - 1); // Orthogonal to tunnel .. no deadlocks
    ring_devices_1 = std::vector<Device*>(ring_devices_1.begin() + 1, ring_devices_1.end());
    std::vector<Device*> ring_devices_2 = view.get_devices_on_row(7); // Tunnel 7 .. potential deadlocks with lack of buffering
    std::reverse(ring_devices_2.begin(), ring_devices_2.end());
    ring_devices_2 = std::vector<Device*>(ring_devices_2.begin() + 1, ring_devices_2.end());
    std::vector<Device*> ring_devices_3 = view.get_devices_on_column(0); // Orthogonal to tunnel .. no deadlocks
    std::reverse(ring_devices_3.begin(), ring_devices_3.end());
    ring_devices_3 = std::vector<Device*>(ring_devices_3.begin() + 1, ring_devices_3.end() - 1);

    ring_devices.insert(ring_devices.end(), ring_devices_1.begin(), ring_devices_1.end());
    ring_devices.insert(ring_devices.end(), ring_devices_2.begin(), ring_devices_2.end());
    ring_devices.insert(ring_devices.end(), ring_devices_3.begin(), ring_devices_3.end());

    // Setup input data and output data containers
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};
    ttnn::SimpleShape shape{1, 2, 256, static_cast<uint32_t>(256 * ring_devices.size())};
    const uint32_t buf_size_datums = 2 * 256 * 256 * ring_devices.size();
    const uint32_t datum_size_bytes = 2;
    // Output of reduce scatter is input_numel / num_devices_used_in_scatter_op
    auto host_data = std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16 []>(new bfloat16[buf_size_datums / ring_devices.size()]);
    uint32_t scatter_dim = 3;
    uint32_t outer_loops = 500;

    // Input to CCL is a tensor of 1s. The output will contain (ring size) times less data along the innermost dim,
    // with each entry == ring size
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
            auto input_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(buf_size_datums * datum_size_bytes, dev, shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
            auto input_storage = DeviceStorage{input_buffer};
            Tensor input_tensor = Tensor(input_storage, shape, DataType::BFLOAT16, Layout::TILE);
            // Push inputs.
            ttnn::write_buffer(0, input_tensor, {host_data});
            // Configure CCL running on this device.
            uint32_t receiver_device_id = device_ids[(dev_idx + 1) % ring_devices.size()];
            uint32_t sender_device_id = device_ids[(dev_idx + ring_devices.size() - 1) % ring_devices.size()];
            auto all_gather_op = ttnn::ReduceScatter{
                                    ttnn::operations::binary::BinaryOpType::ADD, scatter_dim, 1, static_cast<uint32_t>(ring_devices.size()), dev_idx, receiver_device_id, sender_device_id, input_tensor.memory_config(), ttnn::ccl::Topology::Ring};
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
            ASSERT_EQ(tensor.get_shape(), ttnn::Shape(LegacyShape({1, 2, 256, 256})));
            ttnn::read_buffer(0, tensor, {readback_data});
            for (int j = 0; j < 512 * 256; j++) {
                ASSERT_EQ(readback_data[j].to_float(), ring_devices.size());
            }
        }
    }
    ttnn::distributed::close_mesh_device(mesh);
}
