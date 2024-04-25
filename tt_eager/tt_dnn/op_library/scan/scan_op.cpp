// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/scan/scan_op.hpp"

#include <optional>

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt::tt_metal {

using TCores = const std::variant<CoreCoord, CoreRange, CoreRangeSet>;

template <size_t N = 1>
std::array<CB, N> create_cb(
    Program &program,
    const TCores &core_spec,
    uint32_t page_size,
    uint32_t num_pages,
    const tt::DataFormat data_format,
    std::array<CB, N> cbs,
    Buffer *buffer = nullptr) {
    std::map<uint8_t, tt::DataFormat> data_format_spec = {};
    for (auto cb : cbs) {
        data_format_spec[cb] = data_format;
    }

    auto cb_config = CircularBufferConfig(num_pages * page_size, {data_format_spec});
    for (auto cb : cbs) {
        cb_config.set_page_size(cb, page_size);
    }

    if (buffer != nullptr) {
        cb_config.set_globally_allocated_address(*buffer);
    }

    tt_metal::CreateCircularBuffer(program, core_spec, cb_config);
    return cbs;
}

template <typename T, size_t N, typename... Rest>
auto aggregate_arrays(const std::array<T, N> &first, const Rest &...rest) {
    std::vector<uint32_t> result;

    result.reserve(N + (rest.size() + ...));
    result.insert(result.end(), first.begin(), first.end());
    (result.insert(result.end(), rest.begin(), rest.end()), ...);
    return result;
}

void ScanBase::validate(const std::vector<Tensor> &input_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    const Shape &input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Scan: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Scan: Expect input tensor to be allocated on a device buffer.");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Scan: Expect input tensor in tile layout.");
    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Scan: Expect input tensor to be height sharded.");
    auto shard_spec = input_tensor.shard_spec().value();
    TT_FATAL(shard_spec.grid.ranges().size() == 1, "Scan: Expect input tensor to be sharded in one core range.");

    auto core_range = *shard_spec.grid.ranges().begin();
    auto start_coord = core_range.start;
    auto end_coord = core_range.end;
    TT_FATAL(start_coord.x == end_coord.x, "Scan: Expect input tensor to be sharded along a single core column.");
}

operation::ProgramWithCallbacks scan_impl(const Tensor &input) {
    Program program = Program();
    tt_metal::Device *device = input.device();
    Buffer *src_buffer = input.buffer();
    auto all_cores = input.shard_spec()->grid;

    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t tile_size = tt_metal::detail::TileSize(data_format);
    uint32_t total_tiles = input.shard_spec()->numel() / TILE_HW;
    uint32_t tiles_per_row = input.shard_spec()->shape[1] / TILE_WIDTH;
    uint32_t tiles_per_col = input.shard_spec()->shape[0] / TILE_HEIGHT;
    uint32_t reshapes_per_row = input.shard_spec()->shape[1] / TILE_HW;

    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t *>(&one);

    auto ct_args = aggregate_arrays(
        create_cb<2>(program, all_cores, tile_size, total_tiles, data_format, {CB::c_in0, CB::c_out0}, src_buffer),
        create_cb(program, all_cores, tile_size, 32, data_format, {CB::c_intermed0}),
        create_cb<3>(
            program, all_cores, tile_size, 32, data_format, {CB::c_intermed1, CB::c_intermed2, CB::c_intermed3}),
        create_cb(program, all_cores, tile_size, 8, data_format, {CB::c_intermed4}),
        create_cb(program, all_cores, tile_size, 8, data_format, {CB::c_intermed5}),
        create_cb<2>(program, all_cores, tile_size, reshapes_per_row, data_format, {CB::c_intermed6, CB::c_intermed7}));
    ct_args.push_back(bf16_one_u32);

    std::vector<uint32_t> runtime_args = {tiles_per_col, reshapes_per_row, total_tiles};

    // Reader kernel
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/dataflow/reader_scan_sharded.cpp",
        all_cores,
        WriterDataMovementConfig(ct_args));
    SetRuntimeArgs(program, reader_kernel_id, all_cores, runtime_args);

    // Compute kernel
    tt_metal::KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/compute/untilize_scan_tilize.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = ct_args});
    tt_metal::SetRuntimeArgs(program, compute_kernel_id, all_cores, runtime_args);

    return {std::move(program), std::nullopt, std::nullopt};
}

operation::ProgramWithCallbacks Scan::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    return scan_impl(input_tensor);
}

Tensor scan(Tensor &a) {
    operation::run(Scan{}, {a});
    return a;
}

operation::ProgramWithCallbacks retile_to_row_major_impl(const Tensor &input) {
    Program program = Program();
    tt_metal::Device *device = input.device();
    Buffer *src_buffer = input.buffer();
    auto all_cores = input.shard_spec()->grid;

    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t tile_size = tt_metal::detail::TileSize(data_format);
    uint32_t total_tiles = input.shard_spec()->numel() / TILE_HW;
    uint32_t tiles_per_row = input.shard_spec()->shape[1] / TILE_WIDTH;
    uint32_t tiles_per_col = input.shard_spec()->shape[0] / TILE_HEIGHT;
    uint32_t reshapes_per_row = input.shard_spec()->shape[1] / TILE_HW;

    auto ct_args = aggregate_arrays(
        create_cb<2>(program, all_cores, tile_size, total_tiles, data_format, {CB::c_in0, CB::c_out0}, src_buffer),
        create_cb(program, all_cores, tile_size, 32, data_format, {CB::c_intermed0}),
        create_cb(program, all_cores, tile_size, 8, data_format, {CB::c_intermed1}));

    std::vector<uint32_t> runtime_args = {tiles_per_col, reshapes_per_row, total_tiles};

    // Reader kernel
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/dataflow/reader_retilize.cpp",
        all_cores,
        WriterDataMovementConfig(ct_args));
    SetRuntimeArgs(program, reader_kernel_id, all_cores, runtime_args);

    // Compute kernel
    tt_metal::KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/compute/untilize_32_tiles.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = ct_args});
    tt_metal::SetRuntimeArgs(program, compute_kernel_id, all_cores, runtime_args);

    return {std::move(program), std::nullopt, std::nullopt};
}

operation::ProgramWithCallbacks RetileToRowMajor::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    return retile_to_row_major_impl(input_tensor);
}

Tensor retile_to_row_major(Tensor &a) {
    operation::run(RetileToRowMajor{}, {a});
    return a;
}

operation::ProgramWithCallbacks undo_retile_to_row_major_impl(const Tensor &input) {
    Program program = Program();
    tt_metal::Device *device = input.device();
    Buffer *src_buffer = input.buffer();
    auto all_cores = input.shard_spec()->grid;

    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t tile_size = tt_metal::detail::TileSize(data_format);
    uint32_t total_tiles = input.shard_spec()->numel() / TILE_HW;
    uint32_t tiles_per_row = input.shard_spec()->shape[1] / TILE_WIDTH;
    uint32_t tiles_per_col = input.shard_spec()->shape[0] / TILE_HEIGHT;
    uint32_t reshapes_per_row = input.shard_spec()->shape[1] / TILE_HW;

    auto ct_args = aggregate_arrays(
        create_cb<2>(program, all_cores, tile_size, total_tiles, data_format, {CB::c_in0, CB::c_out0}, src_buffer),
        create_cb(program, all_cores, tile_size, 32, data_format, {CB::c_intermed0}),
        create_cb(program, all_cores, tile_size, 8, data_format, {CB::c_intermed1}));

    std::vector<uint32_t> runtime_args = {tiles_per_col, reshapes_per_row, total_tiles};

    // Reader kernel
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/dataflow/reader_undo_retilize.cpp",
        all_cores,
        WriterDataMovementConfig(ct_args));
    SetRuntimeArgs(program, reader_kernel_id, all_cores, runtime_args);

    // Compute kernel
    tt_metal::KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/compute/tilize_32_tiles.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = ct_args});
    tt_metal::SetRuntimeArgs(program, compute_kernel_id, all_cores, runtime_args);

    return {std::move(program), std::nullopt, std::nullopt};
}

operation::ProgramWithCallbacks UndoRetileToRowMajor::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    return undo_retile_to_row_major_impl(input_tensor);
}

Tensor undo_retile_to_row_major(Tensor &a) {
    operation::run(UndoRetileToRowMajor{}, {a});
    return a;
}

operation::ProgramWithCallbacks scan_only_impl(const Tensor &input) {
    Program program = Program();
    tt_metal::Device *device = input.device();
    Buffer *src_buffer = input.buffer();
    auto all_cores = input.shard_spec()->grid;

    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t tile_size = tt_metal::detail::TileSize(data_format);
    uint32_t total_tiles = input.shard_spec()->numel() / TILE_HW;
    uint32_t tiles_per_row = input.shard_spec()->shape[1] / TILE_WIDTH;
    uint32_t tiles_per_col = input.shard_spec()->shape[0] / TILE_HEIGHT;
    uint32_t reshapes_per_row = input.shard_spec()->shape[1] / TILE_HW;

    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t *>(&one);

    auto ct_args = aggregate_arrays(
        create_cb<2>(program, all_cores, tile_size, total_tiles, data_format, {CB::c_in0, CB::c_out0}, src_buffer),
        create_cb<2>(program, all_cores, tile_size, reshapes_per_row, data_format, {CB::c_intermed0, CB::c_intermed1}));
    ct_args.push_back(bf16_one_u32);

    std::vector<uint32_t> runtime_args = {tiles_per_col, reshapes_per_row, total_tiles};

    // Reader kernel
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/dataflow/reader_push_factors.cpp",
        all_cores,
        WriterDataMovementConfig(ct_args));
    SetRuntimeArgs(program, reader_kernel_id, all_cores, runtime_args);

    // Compute kernel
    tt_metal::KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/compute/scan.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = ct_args});
    tt_metal::SetRuntimeArgs(program, compute_kernel_id, all_cores, runtime_args);

    return {std::move(program), std::nullopt, std::nullopt};
}

operation::ProgramWithCallbacks ScanOnly::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    return scan_only_impl(input_tensor);
}

Tensor scan_only(Tensor &a) {
    operation::run(ScanOnly{}, {a});
    return a;
}

operation::ProgramWithCallbacks scan_communicate_impl(const Tensor &input) {
    Program program = Program();
    tt_metal::Device *device = input.device();
    Buffer *src_buffer = input.buffer();
    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t tile_size = tt_metal::detail::TileSize(data_format);
    uint32_t total_tiles = input.shard_spec()->numel() / TILE_HW;
    uint32_t tiles_per_row = input.shard_spec()->shape[1] / TILE_WIDTH;
    uint32_t tiles_per_col = input.shard_spec()->shape[0] / TILE_HEIGHT;
    uint32_t reshapes_per_row = input.shard_spec()->shape[1] / TILE_HW;

    auto ct_args = aggregate_arrays(
        create_cb<2>(program, all_cores, tile_size, total_tiles, data_format, {CB::c_in0, CB::c_out0}, src_buffer),
        create_cb<2>(
            program,
            all_cores,
            tile_size,
            reshapes_per_row * num_cores,
            data_format,
            {CB::c_intermed0, CB::c_intermed1}));

    auto core0_ready_to_receive = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto core0_ready_to_send = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto work_done = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    auto reader_ct_args = ct_args;
    reader_ct_args.push_back(core0_ready_to_receive);
    reader_ct_args.push_back(core0_ready_to_send);
    reader_ct_args.push_back(work_done);

    auto first_core = device->worker_core_from_logical_core(cores[0]);
    auto cores_except_first = CoreRange(cores[1], cores.back());
    auto mcast_start = device->worker_core_from_logical_core(cores[1]);
    auto mcast_end = device->worker_core_from_logical_core(cores.back());

    uint32_t tiles_per_reshape = 32;
    uint32_t first_tile_offset = (total_tiles - tiles_per_row + tiles_per_reshape - 1) * tile_size;

    std::vector<uint32_t> rt_args_common = {tiles_per_col, reshapes_per_row, total_tiles, first_tile_offset};
    std::vector<uint32_t> rt_args_core0 = rt_args_common;
    rt_args_core0.push_back(mcast_start.x);
    rt_args_core0.push_back(mcast_start.y);
    rt_args_core0.push_back(mcast_end.x);
    rt_args_core0.push_back(mcast_end.y);
    rt_args_core0.push_back(num_cores * tile_size);
    rt_args_core0.push_back(num_cores - 1);

    // Reader kernel for first core
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/dataflow/reader_receive_then_send_last_row.cpp",
        cores[0],
        WriterDataMovementConfig(reader_ct_args));
    SetRuntimeArgs(program, reader_kernel_id, cores[0], rt_args_core0);

    /*
        // Compute kernel for first core
        KernelHandle compute_kernel_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/scan/kernels/compute/scan_last_row.cpp",
            cores[0],
            ComputeConfig{.compile_args = ct_args});
        SetRuntimeArgs(program, compute_kernel_id, cores[0], rt_args_common);
    */
    std::vector<uint32_t> rt_args_rest = rt_args_common;
    rt_args_rest.push_back((uint32_t)first_core.x);
    rt_args_rest.push_back((uint32_t)first_core.y);
    rt_args_rest.push_back(num_cores * tile_size);
    rt_args_rest.push_back(0);  // placeholder for offset

    // Reader kernel for all other cores
    KernelHandle reader_kernel_id_except_first = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/dataflow/reader_send_then_receive_last_row.cpp",
        cores_except_first,
        WriterDataMovementConfig(reader_ct_args));
    /*
        // Compute kernel for all other cores
        KernelHandle compute_kernel_id_except_first = CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/scan/kernels/compute/mul_tiles_by_last_row.cpp",
            cores_except_first,
            ComputeConfig{.compile_args = ct_args});
    */
    // Runtime args for all other cores
    for (uint32_t core_idx = 1; core_idx < cores.size(); ++core_idx) {
        // Reader kernel
        rt_args_rest.back() = core_idx * tile_size;
        SetRuntimeArgs(program, reader_kernel_id_except_first, cores[core_idx], rt_args_rest);

        // Compute kernel
        // SetRuntimeArgs(program, compute_kernel_id_except_first, cores[core_idx], rt_args_common);
    }

    return {std::move(program), std::nullopt, std::nullopt};
}

operation::ProgramWithCallbacks ScanCommunicate::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    return scan_communicate_impl(input_tensor);
}

Tensor scan_communicate(Tensor &a) {
    operation::run(ScanCommunicate{}, {a});
    return a;
}
}  // namespace tt::tt_metal
