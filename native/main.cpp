#include "common/core_coord.h"
#include "detail/tt_metal.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt;
using namespace tt_metal;

int main() {
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 4 * 1024;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    // auto src0_dram_noc_coord = src0_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();
    uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
    uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;

    auto grid = device->compute_with_storage_grid_size();
    uint32_t units_to_divide = 1;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    std::cout << "All coressss: " << all_cores.num_cores() << std::endl;

    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Int32}})
            .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float32}})
            .set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/uniform/device/kernels/";
    const std::vector<uint32_t> reader_compile_time_args{};
    const std::string reader_file_path = kernels_dir_path + "reader.cpp";
    const std::vector<uint32_t> writer_compile_time_args{};
    const std::string writer_file_path = kernels_dir_path + "writer.cpp";
    const std::vector<uint32_t> compute_compile_time_args{};
    const std::string compute_file_path = kernels_dir_path + "uniform.cpp";

    KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program, reader_file_path, all_cores, ReaderDataMovementConfig(reader_compile_time_args));
    KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program, writer_file_path, all_cores, WriterDataMovementConfig(writer_compile_time_args));
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(
        device->arch(), ttnn::init_device_compute_kernel_config(device->arch(), std::nullopt, MathFidelity::HiFi4));
    std::cout << "fp32_dest_acc_en" << fp32_dest_acc_en << " " << "math_approx_mode" << math_approx_mode << std::endl;
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        compute_file_path,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = true,
            .compile_args = compute_compile_time_args,
        });

    SetRuntimeArgs(program, reader_kernel_id, core, {});
    SetRuntimeArgs(program, compute_kernel_id, core, {});
    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address()});

    EnqueueProgram(cq, program, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<float> result_vec(1024);
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec.data(), true);
    std::map<float, int> mp;

    for (uint32_t i = 0; i < 1024; ++i) {
        float a = result_vec[i];
        mp[result_vec[i]] += 1;
        std::cout << result_vec[i] << " ";
        if ((i & 31) == 31)
            std::cout << std::endl;
    }

    std::cout << mp.size() << std::endl;
    // for (const auto &pair : mp) {
    //     std::cout << std::bitset<32>(pair.first) << " " << pair.second << std::endl;
    // }

    CloseDevice(device);
}
