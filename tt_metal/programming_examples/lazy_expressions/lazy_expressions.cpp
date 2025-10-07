// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <random>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main(int argc, char** argv) {
    bool pass = true;

    try {
        using namespace tt::tt_metal;

        constexpr auto device_id = 0;
        const auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        auto& cq = mesh_device->mesh_command_queue();

        distributed::MeshWorkload workload;
        Program program = CreateProgram();

        // This example program will use 24 Tensix cores. So we set the end core to {7, 2}.
        const CoreRange cores = {{0, 0}, {7, 2}};

        // Define some constants that will be used throughout the program.
        // * Processing 64 tiles
        // * Each tile is 32x32 elements
        // * Each element is a bfloat16 (2 bytes)
        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
        constexpr uint32_t tiles_per_cycle = 2;

        auto
            [num_cores,                   // number of cores utilized
             all_cores,                   // set of all cores used
             core_group_1,                // Primary core group
             core_group_2,                // Secondary core group
             num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
             num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
        ] = tt::tt_metal::split_work_to_cores(cores, n_tiles);

        // Create 4 buffers on L1. These will hold the input and output data. src0, src1, src2 are the input buffers,
        // dst is the output buffer.
        const auto device_config = distributed::DeviceLocalBufferConfig{
            .page_size = tile_size_bytes,   // The page size of the buffer in bytes. Unlike the `loopback` example, we
                                            // need the page size to be the same as the tile size for a large portion of
                                            // the NoC transfer APIs to work.
            .buffer_type = BufferType::L1,  // This is an L1 buffer.
        };
        const auto mesh_config = distributed::ReplicatedBufferConfig{
            .size = tile_size_bytes * n_tiles,
        };
        const auto device_ptr = mesh_device.get();

        using distributed::MeshBuffer;

        auto c0_buffer = MeshBuffer::create(mesh_config, device_config, device_ptr);
        auto c1_buffer = MeshBuffer::create(mesh_config, device_config, device_ptr);
        // c2 is reserved for intermediate result
        auto c3_buffer = MeshBuffer::create(mesh_config, device_config, device_ptr);
        auto c4_buffer = MeshBuffer::create(mesh_config, device_config, device_ptr);

        // Initialize the input buffers with random data.
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::vector<bfloat16> a_data;
        std::vector<bfloat16> b_data;
        std::vector<bfloat16> c_data;
        bfloat16 value = distribution(rng);

        a_data.reserve(elements_per_tile * n_tiles);
        b_data.reserve(elements_per_tile * n_tiles);
        c_data.reserve(elements_per_tile * n_tiles);

        for (std::size_t i = 0; i < elements_per_tile * n_tiles; ++i) {
            a_data.push_back(distribution(rng));
            b_data.push_back(distribution(rng));
            c_data.push_back(distribution(rng));
        }

        // Upload the data from host to the device.
        distributed::EnqueueWriteMeshBuffer(cq, c0_buffer, a_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, c1_buffer, b_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, c3_buffer, c_data, false);

        // Create 3 circular buffers. Think them like pipes moving data from one core to another. cb_src0 and cb_src1
        // are used to move data from the reader kernel to the compute kernel. cb_dst is used to move data from the
        // compute kernel to the writer kernel. Each circular buffer is made up of 2 tiles. Thus when one tile is pushed
        // and being used by the receiving end, the sending end can get the next piece of data ready to be pushed.
        // Overlapping the operations. Leading to better performance. However there is a trade off, The more tiles in a
        // circular buffer, the more memory is used. And Circular buffers are backed by L1(SRAM) memory and L1 is a
        // precious resource. The hardware supports up to 32 circular buffers and they all act the same.
        constexpr uint32_t tiles_per_cb = 2;

        // tensor A
        CreateCircularBuffer(
            program,
            cores,
            CircularBufferConfig(
                /*total_size=*/tiles_per_cb * tile_size_bytes,  // The total size of the circular buffer in bytes
                                                                /*data_format_spec=*/
                {{tt::c_0, tt::DataFormat::Float16_b}})         // The circular buffer index and data format it'll hold
                .set_page_size(tt::c_0, tile_size_bytes));      // Since we will be sending one tile at a time, we set
                                                                // the page size to the tile size (and thus
                                                                // total_size / page_size = tiles_per is the number of
                                                                // entries in the circular buffer)

        // tensor B
        CreateCircularBuffer(
            program,
            cores,
            CircularBufferConfig(
                /*total_size=*/tiles_per_cb * tile_size_bytes,
                /*data_format_spec=*/{{tt::c_1, tt::DataFormat::Float16_b}})
                .set_page_size(tt::c_1, tile_size_bytes));
        // intermediate result
        CreateCircularBuffer(
            program,
            cores,
            CircularBufferConfig(
                /*total_size=*/tiles_per_cb * tile_size_bytes,
                /*data_format_spec=*/{{tt::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(tt::c_2, tile_size_bytes));
        // tensor C
        CreateCircularBuffer(
            program,
            cores,
            CircularBufferConfig(
                /*total_size=*/tiles_per_cb * tile_size_bytes,
                /*data_format_spec=*/{{tt::c_3, tt::DataFormat::Float16_b}})
                .set_page_size(tt::c_3, tile_size_bytes));
        // result tensor
        CreateCircularBuffer(
            program,
            cores,
            CircularBufferConfig(
                /*total_size=*/tiles_per_cb * tile_size_bytes,
                /*data_format_spec=*/{{tt::c_4, tt::DataFormat::Float16_b}})
                .set_page_size(tt::c_4, tile_size_bytes));

        // Create the reader, writer and compute kernels. The kernels do the following:
        // * Reader: Reads data from the DRAM buffer and pushes it into the circular buffer.
        // * Compute: Waits for data to be available in the circular buffer, pops it, adds the two inputs together and
        // pushes the result
        //   into the output circular buffer.
        // * Writer: Waits for data to be available in the output circular buffer, pops it and writes it back into DRAM.
        // These kernels work together to form a pipeline. The reader reads data from the DRAM buffer and makes them
        // available in the compute kernel. The compute kernel does math and pushes the result into the writer kernel.
        // The writer kernel writes the result back to DRAM.
        std::vector<uint32_t> reader_compile_time_args;
        std::vector<uint32_t> reader_common_runtime_args;
        std::vector<uint32_t> writer_compile_time_args;
        std::vector<uint32_t> writer_common_runtime_args;
        std::vector<uint32_t> compute_compile_time_args;

        constexpr auto pack_into = [](const MeshBuffer& buffer,
                                      tt::CBIndex cb_index,
                                      std::vector<uint32_t>& compile_time_args,
                                      std::vector<uint32_t>& common_runtime_args) {
            using enum tensor_accessor::ArgConfig;
            constexpr auto args_config = RuntimeNumBanks | RuntimeTensorShape | RuntimeShardShape | RuntimeBankCoords;
            compile_time_args.push_back(static_cast<uint32_t>(cb_index));
            compile_time_args.push_back(common_runtime_args.size());
            TensorAccessorArgs(buffer, args_config).append_to(compile_time_args, common_runtime_args);
        };

        reader_compile_time_args.push_back(tiles_per_cycle);
        pack_into(*c0_buffer, tt::c_0, reader_compile_time_args, reader_common_runtime_args);
        pack_into(*c1_buffer, tt::c_1, reader_compile_time_args, reader_common_runtime_args);
        pack_into(*c3_buffer, tt::c_3, reader_compile_time_args, reader_common_runtime_args);

        writer_compile_time_args.push_back(tiles_per_cycle);
        pack_into(*c4_buffer, tt::c_4, writer_compile_time_args, writer_common_runtime_args);

        compute_compile_time_args.push_back(tiles_per_cycle);

        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "lazy_expressions/kernels/dataflow/read_tiles.cpp",
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "lazy_expressions/kernels/dataflow/write_tiles.cpp",
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args});
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "lazy_expressions/kernels/compute/compute_tiles.cpp",
            cores,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

        SetCommonRuntimeArgs(program, reader, reader_common_runtime_args);
        SetCommonRuntimeArgs(program, writer, writer_common_runtime_args);

        // Set the runtime arguments for the kernels. This also registers
        // the kernels with the program.
        const auto scalar = std::bit_cast<uint32_t>(static_cast<float>(value));
        const auto set_runtime_args_for = [&](const CoreRangeSet& group,
                                              uint32_t num_tiles,
                                              uint32_t group_start_id = 0) {
            for (const auto& range : group.ranges()) {
                for (const auto& core : range) {
                    SetRuntimeArgs(
                        program,
                        reader,
                        core,
                        {num_tiles, group_start_id, c0_buffer->address(), c1_buffer->address(), c3_buffer->address()});
                    SetRuntimeArgs(program, writer, core, {num_tiles, group_start_id, c4_buffer->address()});
                    SetRuntimeArgs(program, compute, core, {num_tiles, scalar});
                    group_start_id += num_tiles;
                }
            }

            return group_start_id;
        };
        const auto start_id_group_2 = set_runtime_args_for(core_group_1, num_tiles_per_core_group_1);
        set_runtime_args_for(core_group_2, num_tiles_per_core_group_2, start_id_group_2);

        std::cout << "Enqueuing program" << std::endl;

        // We have setup the program. Now we queue the kernel for execution. The final argument is set to false. This
        // indicates to Metalium that the operation is non-blocking. The function is allowed to return upon the kernel
        // being queued. We must ensure that the kernel is finished before we read the output buffer. This is done by
        // calling Finish(cq) which waits until all operations in the command queue are finished. This is equivalent to
        // calling EnqueueProgram(cq, program, true); telling Metalium to wait until the program is finished before
        // returning.
        const auto device_range = distributed::MeshCoordinateRange{mesh_device->shape()};
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);

        std::cout << "Enqueued program" << std::endl;

        distributed::Finish(cq);

        std::cout << "Finished" << std::endl;

        // Read the output buffer and compare it with the expected output.
        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, c4_buffer, true);

        constexpr float eps = 1e-2f;  // loose tolerance because of the nature of bfloat16
        TT_FATAL(result_vec.size() == a_data.size(), "Result vector size mismatch");
        for (size_t i = 0; i < result_vec.size(); ++i) {
            const float expected = a_data[i] + b_data[i] * value * c_data[i];
            const float actual = result_vec[i];

            if (std::abs(expected - actual) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
                break;
            }
        }

        // Finally, we close the device.
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
