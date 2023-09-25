// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_softmax/helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"

#include "tt_metal/detail/util.hpp"

namespace tt {
namespace operations {
namespace primary {

inline bool is_dram(const Tensor &input_tensor) { return input_tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
    return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer *b) { return b->buffer_type() == BufferType::DRAM; }

inline std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet> add_core_offset(
    CoreRangeSet all_cores,
    CoreRangeSet core_group_1,
    CoreRangeSet core_group_2,
    uint32_t offset_x,
    uint32_t offset_y) {
    std::set<CoreRange> new_all_cores_set;
    std::set<CoreRange> new_core_group_1_set;
    std::set<CoreRange> new_core_group_2_set;

    for (auto core : all_cores.ranges()) {
        new_all_cores_set.insert((CoreRange){
            .start = {core.start.x + offset_x, core.start.y + offset_y},
            .end = {core.end.x + offset_x, core.end.y + offset_y}});
    }

    for (auto core : core_group_1.ranges()) {
        new_core_group_1_set.insert((CoreRange){
            .start = {core.start.x + offset_x, core.start.y + offset_y},
            .end = {core.end.x + offset_x, core.end.y + offset_y}});
    }

    for (auto core : core_group_2.ranges()) {
        new_core_group_2_set.insert((CoreRange){
            .start = {core.start.x + offset_x, core.start.y + offset_y},
            .end = {core.end.x + offset_x, core.end.y + offset_y}});
    }

    CoreRangeSet new_all_cores(new_all_cores_set);
    CoreRangeSet new_core_group_1(new_core_group_1_set);
    CoreRangeSet new_core_group_2(new_core_group_2_set);

    return std::make_tuple(new_all_cores, new_core_group_1, new_core_group_2);
}

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    CoreRange core_range, uint32_t units_to_divide) {
    uint32_t core_w = core_range.end.x - core_range.start.x + 1;
    uint32_t core_h = core_range.end.y - core_range.start.y + 1;
    CoreCoord grid_size = {core_w, core_h};
    auto
        [num_cores,
         all_cores_t,
         core_group_1_t,
         core_group_2_t,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, units_to_divide);

    auto core_x_offset = core_range.start.x;
    auto core_y_offset = core_range.start.y;

    auto [all_cores, core_group_1, core_group_2] =
        add_core_offset(all_cores_t, core_group_1_t, core_group_2_t, core_x_offset, core_y_offset);

    {
        auto iter = core_group_1.ranges();
        for_each(
            iter.begin(), iter.end(), [](CoreRange core) { log_info(LogTest, "Use core_group_1 {}", core.str()); });
    }
    log_info(LogTest, "num_tiles_per_core_group_1 {}", num_tiles_per_core_group_1);

    {
        auto iter = core_group_2.ranges();
        for_each(
            iter.begin(), iter.end(), [](CoreRange core) { log_info(LogTest, "Use core_group_2 {}", core.str()); });
    }
    log_info(LogTest, "num_tiles_per_core_group_2 {}", num_tiles_per_core_group_2);

    return std::make_tuple(
        num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2);
}

KernelID CreateReadKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core,
    const std::vector<uint32_t> &compile_args,
    std::map<string, string> defines) {
    const string dir_path = "tt_metal/kernels/dataflow/";
    string kernel_file = dir_path + file_name;

    return tt_metal::CreateDataMovementKernel(
        program,
        kernel_file,
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = compile_args,
            .defines = defines});
}

KernelID CreateWriteKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core,
    const std::vector<uint32_t> &compile_args,
    std::map<string, string> defines) {
    const string dir_path = "tt_metal/kernels/dataflow/";
    string kernel_file = dir_path + file_name;

    return tt_metal::CreateDataMovementKernel(
        program,
        kernel_file,
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_args,
            .defines = defines});
}

void CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    std::vector<ComputeKernelArg> args,
    std::map<std::string, std::string> defines,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    const string dir_path = "kernels/compute/";

    auto full_path = dir_path + file_name;

    for (auto arg : args) {
        if (arg.num_tile_per_core_group > 0) {
            auto coumpute_kernel = CreateComputeKernel(
                program,
                full_path,
                arg.core_range,
                tt_metal::ComputeConfig{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .math_approx_mode = math_approx_mode,
                    .compile_args = arg.compile_args,
                    .defines = defines});
        }
    }
}

void CreateCircularBuffers(
    Program &program,
    const CoreRangeSet &core_range,
    tt::DataFormat data_format,
    std::vector<CircularBufferArg> args,
    std::optional<uint32_t> l1_address) {
    for (auto arg : args) {
        auto _buffer_index = arg.buffer_index;
        auto _num_tiles = arg.num_tiles;
        auto _data_format = (arg.data_format != tt::DataFormat::Invalid) ? arg.data_format : data_format;
        auto _core_range = (arg.core_range != nullptr) ? *arg.core_range : core_range;

        CreateCircularBuffers(
            program,
            std::set<u32>({_buffer_index}),
            CoreRangeSet({_core_range}),
            _num_tiles,
            _num_tiles * tt_metal::detail::TileSize(_data_format),
            _data_format,
            l1_address);
    }
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
