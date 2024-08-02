// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.hpp"
#include "ttnn/cpp/ttnn/multi_device.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn_multi_command_queue_fixture.hpp"

using namespace tt;
using namespace tt_metal;

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
    auto mesh = ttnn::multi_device::open_device_mesh({cluster_tunnel_count * num_mmio_devices, num_devices_in_tunnel}, all_device_ids, 0, 0, 1);

    // Setup input data and output data containers
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};
    ttnn::Shape shape = ttnn::Shape(Shape({1, 1, 32, 16384}));
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
                                        3, 2, num_devices_in_tunnel, dev_idx, receiver_device_id, sender_device_id, input_tensor.memory_config(), ttnn::all_gather_op::Topology::Linear};
                // Send CCL to this device. All CCLs will complete simultaneously.
                output_tensors.push_back(ttnn::run_operation(0, all_gather_op, {input_tensor}).at(0));
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
                ASSERT_EQ(tensor.get_shape(), ttnn::Shape(Shape({1, 1, 32, 16384 * device_ids.size()})));
                ttnn::read_buffer(0, tensor, {readback_data});
                for (int j = 0; j < device_ids.size() * 32 * 16384; j++) {
                    ASSERT_EQ(readback_data[j].to_float(), 1);
                }
            }
        }
    }
    ttnn::multi_device::close_device_mesh(mesh);
}
