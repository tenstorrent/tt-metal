// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"

#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"

namespace tt {
namespace operations {
namespace primary {

std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet> add_core_offset(
    CoreRangeSet all_cores,
    CoreRangeSet core_group_1,
    CoreRangeSet core_group_2,
    uint32_t offset_x,
    uint32_t offset_y) {
    std::set<CoreRange> new_all_cores_set;
    std::set<CoreRange> new_core_group_1_set;
    std::set<CoreRange> new_core_group_2_set;

    for (auto core : all_cores.ranges()) {
        new_all_cores_set.insert(CoreRange(
            {core.start.x + offset_x, core.start.y + offset_y},
            {core.end.x + offset_x, core.end.y + offset_y}));
    }

    for (auto core : core_group_1.ranges()) {
        new_core_group_1_set.insert(CoreRange(
            {core.start.x + offset_x, core.start.y + offset_y},
            {core.end.x + offset_x, core.end.y + offset_y}));
    }

    for (auto core : core_group_2.ranges()) {
        new_core_group_2_set.insert(CoreRange(
            {core.start.x + offset_x, core.start.y + offset_y},
            {core.end.x + offset_x, core.end.y + offset_y}));
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
            iter.begin(), iter.end(), [](CoreRange core) { log_debug(LogTest, "Use core_group_1 {}", core.str()); });
    }
    log_debug(LogTest, "num_tiles_per_core_group_1 {}", num_tiles_per_core_group_1);

    {
        auto iter = core_group_2.ranges();
        for_each(
            iter.begin(), iter.end(), [](CoreRange core) { log_debug(LogTest, "Use core_group_2 {}", core.str()); });
    }
    log_debug(LogTest, "num_tiles_per_core_group_2 {}", num_tiles_per_core_group_2);

    return std::make_tuple(
        num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2);
}

[[maybe_unused]] KernelHandle CreateReadKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args,
    std::map<string, string> defines) {
    return tt_metal::CreateKernel(
        program,
        file_name,
        core_spec,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch()),
            .compile_args = compile_args,
            .defines = defines});
}

[[maybe_unused]] KernelHandle CreateWriteKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args,
    std::map<string, string> defines) {
    return tt_metal::CreateKernel(
        program,
        file_name,
        core_spec,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch()),
            .compile_args = compile_args,
            .defines = defines});
}

[[maybe_unused]] std::vector<KernelHandle> CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    std::vector<ComputeKernelArg> args,
    std::map<std::string, std::string> defines,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    std::vector<KernelHandle> compute_kernel_ids{};
    KernelHandle compute_kernel_id{};
    for (auto arg : args) {
        compute_kernel_id =
            CreateComputeKernel(program, file_name, arg, defines, math_fidelity, fp32_dest_acc_en, math_approx_mode);
        compute_kernel_ids.push_back(compute_kernel_id);
    }
    return compute_kernel_ids;
}

[[maybe_unused]] KernelHandle CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    ComputeKernelArg arg,
    std::map<std::string, std::string> defines,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    KernelHandle compute_kernel_id{0};
    if (arg.num_tile_per_core_group > 0) {
        compute_kernel_id = CreateKernel(
            program,
            file_name,
            arg.core_spec,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = arg.compile_args,
                .defines = defines});
    }
    return compute_kernel_id;
}

[[maybe_unused]] std::vector<CBHandle> CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_range,
    tt::DataFormat data_format,
    std::vector<CircularBufferArg> args) {
    std::vector<CBHandle> cb_ids{};
    CBHandle cb_id{};
    for (auto arg : args) {
        cb_id = CreateCircularBuffer(program, core_range, data_format, arg);
        cb_ids.push_back(cb_id);
    }
    return cb_ids;
}

[[maybe_unused]] CBHandle CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_range,
    tt::DataFormat data_format,
    CircularBufferArg arg) {
    CBHandle cb_id{0};
    if (arg.num_tiles > 0) {
        auto _buffer_index = arg.buffer_index;
        auto _num_tiles = arg.num_tiles;
        auto _data_format = (arg.data_format != tt::DataFormat::Invalid) ? arg.data_format : data_format;
        auto _core_range = (arg.core_range != std::nullopt) ? arg.core_range : core_range;

        tt_metal::CircularBufferConfig cb_config =
            tt_metal::CircularBufferConfig(
                _num_tiles * tt_metal::detail::TileSize(_data_format), {{_buffer_index, _data_format}})
                .set_page_size(_buffer_index, tt_metal::detail::TileSize(_data_format));

        cb_id = tt_metal::CreateCircularBuffer(program, _core_range.value(), cb_config);
    }
    return cb_id;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
