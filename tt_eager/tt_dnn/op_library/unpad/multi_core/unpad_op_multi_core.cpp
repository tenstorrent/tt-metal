// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optional"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

namespace unpad_impl::multi_core {
typedef std::tuple<std::vector<CoreCoord>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>>
    RuntimeArgs;

namespace row_major {

RuntimeArgs get_runtime_args(const Tensor &input_tensor, Tensor &output_tensor, const Shape &output_tensor_start) {
    tt_metal::Device *device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.shape();
    auto output_shape = output_tensor.shape();

    uint32_t num_unpadded_sticks = output_tensor.volume() / output_shape[-1];

    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    vector<uint32_t> common_reader_kernel_args = {
        input_tensor.buffer()->address() + output_tensor_start[-1] * output_tensor.element_size(),
        padded_row_size_bytes,
        unpadded_row_size_bytes,
        num_dims,
        0,
        0};
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());

    uint32_t start_offset = get_rm_start_offset(input_tensor, output_tensor_start);

    CoreRangeSet all_cores({}), core_group_1({}), core_group_2({});
    uint32_t num_cores, num_sticks_per_core_group_1, num_sticks_per_core_group_2;
    uint32_t num_sticks_per_core_last = 0;
    bool row_major = true;

    std::vector<CoreCoord> cores;
    if (output_tensor.is_sharded()) {
        row_major = output_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
        auto output_mem_config = output_tensor.memory_config();
        ShardSpec shard_spec = output_mem_config.shard_spec.value();
        num_cores = shard_spec.num_cores();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_sticks_per_core_group_1 = shard_spec.shape[0];
        num_sticks_per_core_group_2 = 0;
        num_sticks_per_core_last = num_sticks_per_core_group_1 -
                                   (round_up(num_unpadded_sticks, num_sticks_per_core_group_1) - num_unpadded_sticks);
        auto bbox = core_group_1.bounding_box();
        cores = grid_to_cores_with_noop(bbox.end.x, bbox.end.y, num_cores_x, num_cores_y, row_major);
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_sticks_per_core_group_1,
            num_sticks_per_core_group_2) = split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);
        cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    }

    std::vector<uint32_t> num_sticks_per_core(num_cores_total);

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    for (uint32_t i = 0; i < num_cores; ++i) {
        auto core = cores[i];
        if (i < g1_numcores) {
            num_sticks_per_core[i] = num_sticks_per_core_group_1;
        } else {
            num_sticks_per_core[i] = num_sticks_per_core_group_2;
        }
    }
    if (output_tensor.is_sharded()) {
        num_sticks_per_core[num_cores - 1] = num_sticks_per_core_last;
    }

    std::vector<std::vector<uint32_t>> reader_runtime_args = {
        num_cores_total, std::vector<uint32_t>(common_reader_kernel_args.size() + id_per_dim.size())};
    std::vector<std::vector<uint32_t>> writer_runtime_args = {
        num_cores_total, output_tensor.is_sharded() ? std::vector<uint32_t>(1) : std::vector<uint32_t>(5)};

    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores; ++i) {
        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; ++j) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        uint32_t addr_offset = 4;  // input buffer addr, padded_row_size_bytes, unpadded_row_size_bytes, num_dims
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset] = num_sticks_per_core[i];
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());
        if (output_tensor.is_sharded()) {
            writer_runtime_args[i] = {num_sticks_per_core[i]};
        } else {
            writer_runtime_args[i] = {
                output_buffer->address(), unpadded_row_size_bytes, num_sticks_per_core[i], num_sticks_written, 0};
        }
        reader_runtime_args[i] = std::move(reader_kernel_args);
        num_sticks_written += num_sticks_per_core[i];
    }

    return std::make_tuple(std::move(cores), std::move(reader_runtime_args), std::move(writer_runtime_args));
}

// row-major
operation::ProgramWithCallbacks get_program(
    const Tensor &a, Tensor &output, const Shape &output_tensor_start, const Shape &output_tensor_end) {
    const Shape output_shape = output.shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    uint32_t num_unpadded_sticks = output.volume() / output.shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});
    uint32_t num_cores_total = num_cores_x*num_cores_y;

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t padded_row_size_bytes = a.shape()[-1] * a.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * a.element_size();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src_stick_size = padded_row_size_bytes;
    uint32_t dst_stick_size = unpadded_row_size_bytes;

    uint32_t src0_cb_index = 0;

    uint32_t cb_page_size = round_up(unpadded_row_size_bytes, TILE_WIDTH);
    tt_metal::CBHandle cb_src0;

    std::optional<uint32_t> num_pages_per_shard_height;
    std::optional<uint32_t> num_pages_per_shard_height_last;
    std::optional<uint32_t> num_pages_per_shard_width;

    if (output.is_sharded()) {
        auto shard_shape = output.shard_spec().value().shape;
        num_pages_per_shard_height = shard_shape[0];
        uint32_t num_pages_height = output.volume() / output_shape[-1];
        num_pages_per_shard_height_last =
            num_pages_per_shard_height.value() -
            (round_up(num_pages_height, num_pages_per_shard_height.value()) - num_pages_height);
        num_pages_per_shard_width = shard_shape[1] / output_shape[-1];

        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_pages_per_shard_height.value() * num_pages_per_shard_width.value() * cb_page_size,
                {{src0_cb_index, cb_data_format}})
                .set_page_size(src0_cb_index, cb_page_size)
                .set_globally_allocated_address(*output.buffer());
        cb_src0 = tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    } else {
        uint32_t num_input_pages = 2;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(num_input_pages * cb_page_size, {{src0_cb_index, cb_data_format}})
                .set_page_size(src0_cb_index, cb_page_size);
        cb_src0 = tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    }

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args_vec = {(std::uint32_t)src0_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> writer_compile_time_args_vec;

    if (output.is_sharded()) {
        writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index};

    } else {
        writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};
    }
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/unpad/kernels/dataflow/reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        total_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        (output.is_sharded())
            ? "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp"
            : "tt_eager/tt_dnn/op_library/unpad/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        total_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    auto [cores, unary_reader_args, unary_writer_args] = get_runtime_args(a, output, output_tensor_start);

    SetRuntimeArgs(program, unary_reader_kernel_id, cores, unary_reader_args);
    SetRuntimeArgs(program, unary_writer_kernel_id, cores, unary_writer_args);

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, src0_cb_index](
                                              const void *operation,
                                              Program &program,
                                              const std::vector<Tensor> &input_tensors,
                                              const std::vector<std::optional<const Tensor>> &,
                                              const std::vector<Tensor> &output_tensors) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        auto compute_with_storage_grid_size = src_tensor.device()->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_cores_total = num_cores_x * num_cores_y;

        tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(src_tensor.dtype());

        uint32_t unpadded_row_size_bytes = dst_tensor.shape()[-1] * dst_tensor.element_size();
        uint32_t cb_page_size = round_up(unpadded_row_size_bytes, TILE_WIDTH);

        std::optional<CoreCoord> end_core;
        if (dst_tensor.is_sharded()) {
            auto dst_buffer = dst_tensor.buffer();
            UpdateDynamicCircularBufferAddress(program, cb_src0, *dst_buffer);
            auto shard_shape = dst_tensor.shard_spec().value().shape;
            UpdateCircularBufferTotalSize(program, cb_src0, shard_shape[0] * cb_page_size);
            UpdateCircularBufferPageSize(program, cb_src0, src0_cb_index, cb_page_size);
        } else {
            UpdateCircularBufferTotalSize(program, cb_src0, 2 * cb_page_size);
            UpdateCircularBufferPageSize(program, cb_src0, src0_cb_index, cb_page_size);
        }

        const auto tensor_start = static_cast<const Unpad *>(operation)->output_tensor_start;

        auto [cores, unary_reader_args, unary_writer_args] = get_runtime_args(src_tensor, dst_tensor, tensor_start);

        SetRuntimeArgs(program, unary_reader_kernel_id, cores, unary_reader_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, cores, unary_writer_args);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace row_major

namespace tile {

// Each element of outer vector corresponds to a core
// Each core has a pair of std::vector<uint32_t>
// First of pair is reader args
// Second of pair is writer args
RuntimeArgs get_runtime_args(const Tensor &input_tensor, Tensor &output_tensor, const Shape &output_tensor_start) {
    uint32_t num_unpadded_tiles = output_tensor.volume() / TILE_HW;
    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.shape();
    auto output_shape = output_tensor.shape();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_padded_tiles_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    for (int32_t i = 2; i < num_dims; ++i) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    vector<uint32_t> common_reader_kernel_args = {input_buffer->address(), num_dims, 0, 0};
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_unpadded_tiles_per_dim.begin(), num_unpadded_tiles_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_padded_tiles_per_dim.begin(), num_padded_tiles_per_dim.end());

    uint32_t start_offset = get_tiled_start_offset(input_tensor, output_tensor_start);

    CoreRangeSet all_cores({}), core_group_1({}), core_group_2({});
    uint32_t num_cores, num_tiles_per_core_group_1, num_tiles_per_core_group_2;
    uint32_t num_tiles_per_core_last = 0;
    bool row_major = true;

    std::vector<CoreCoord> cores;
    if (output_tensor.is_sharded()) {
        row_major = output_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
        auto output_mem_config = output_tensor.memory_config();
        ShardSpec shard_spec = output_mem_config.shard_spec.value();
        num_cores = shard_spec.num_cores();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_tiles_per_core_group_1 = shard_spec.shape[0] * shard_spec.shape[1] / TILE_HW;
        num_tiles_per_core_group_2 = 0;
        num_tiles_per_core_last = num_tiles_per_core_group_1 -
                                  (round_up(num_unpadded_tiles, num_tiles_per_core_group_1) - num_unpadded_tiles);
        auto bbox = core_group_1.bounding_box();
        cores = grid_to_cores_with_noop(bbox.end.x, bbox.end.y, num_cores_x, num_cores_y, row_major);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);
        cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    }
    std::vector<uint32_t> num_tiles_per_core(num_cores_total);

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    for (uint32_t i = 0; i < num_cores; ++i) {
        auto core = cores[i];
        if (i < g1_numcores) {
            num_tiles_per_core[i] = num_tiles_per_core_group_1;
        } else {
            num_tiles_per_core[i] = num_tiles_per_core_group_2;
        }
    }
    if (output_tensor.is_sharded()) {
        num_tiles_per_core[num_cores - 1] = num_tiles_per_core_last;
    }

    std::vector<std::vector<uint32_t>> reader_runtime_args = {
        num_cores_total, std::vector<uint32_t>(common_reader_kernel_args.size() + id_per_dim.size())};
    std::vector<std::vector<uint32_t>> writer_runtime_args = {
        num_cores_total, output_tensor.is_sharded() ? std::vector<uint32_t>(1) : std::vector<uint32_t>(4)};

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; ++i) {
        id_per_dim[0] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; ++j) {
            id_per_dim[j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        uint32_t addr_offset = 2;  // input buffer addr, num_dims
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset] = num_tiles_per_core[i];
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());
        if (output_tensor.is_sharded()) {
            writer_runtime_args[i] = {num_tiles_per_core[i]};
        } else {
            writer_runtime_args[i] = {output_buffer->address(), num_tiles_per_core[i], num_tiles_written, 0};
        }
        reader_runtime_args[i] = std::move(reader_kernel_args);
        num_tiles_written += num_tiles_per_core[i];
    }

    return std::make_tuple(std::move(cores), std::move(reader_runtime_args), std::move(writer_runtime_args));
}

// tile
operation::ProgramWithCallbacks get_program(
    const Tensor &a, Tensor &output, const Shape &output_tensor_start, const Shape &output_tensor_end) {
    const Shape output_shape = output.shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cores_total = num_cores_x*num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    tt_metal::CBHandle cb_src0;
    uint32_t dst_cb_index = src0_cb_index;
    std::optional<uint32_t> num_tiles_per_shard_height;
    std::optional<uint32_t> num_tiles_per_shard_height_last;
    std::optional<uint32_t> num_tiles_per_shard_width;

    if (output.is_sharded()) {
        auto shard_shape = output.shard_spec().value().shape;
        num_tiles_per_shard_height = shard_shape[0] / TILE_HEIGHT;
        uint32_t num_tiles_height = output.volume() / output.shape()[-1] / TILE_HEIGHT;
        num_tiles_per_shard_height_last =
            num_tiles_per_shard_height.value() -
            (round_up(num_tiles_height, num_tiles_per_shard_height.value()) - num_tiles_height);
        num_tiles_per_shard_width = shard_shape[1] / TILE_WIDTH;

        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_tiles_per_shard_height.value() * num_tiles_per_shard_width.value() * single_tile_size,
                {{src0_cb_index, cb_data_format}})
                .set_page_size(src0_cb_index, single_tile_size)
                .set_globally_allocated_address(*output.buffer());
        cb_src0 = tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    } else {
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
                .set_page_size(src0_cb_index, single_tile_size);
        cb_src0 = tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    }

    // Reader compile-time args
    // Data is 32 byte aligned
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_is_dram};

    std::vector<uint32_t> writer_compile_time_args;

    if (output.is_sharded()) {
        writer_compile_time_args = {dst_cb_index};
    } else {
        writer_compile_time_args = {// interleaved accessor args
                                    (std::uint32_t)src0_cb_index,
                                    (std::uint32_t)dst_is_dram};
    }

    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/unpad/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp",
        total_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        (output.is_sharded()) ? "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp"
                              : "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto [cores, unary_reader_args, unary_writer_args] = get_runtime_args(a, output, output_tensor_start);

    SetRuntimeArgs(program, unary_reader_kernel_id, cores, unary_reader_args);
    SetRuntimeArgs(program, unary_writer_kernel_id, cores, unary_writer_args);

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cb_src0](
                                              const void *operation,
                                              Program &program,
                                              const std::vector<Tensor> &input_tensors,
                                              const std::vector<std::optional<const Tensor>> &,
                                              const std::vector<Tensor> &output_tensors) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        const auto tensor_start = static_cast<const Unpad *>(operation)->output_tensor_start;

        tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(src_tensor.dtype());
        uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

        std::optional<CoreCoord> end_core;
        bool row_major = true;
        if (dst_tensor.is_sharded()) {
            auto dst_buffer = dst_tensor.buffer();
            UpdateDynamicCircularBufferAddress(program, cb_src0, *dst_buffer);
            auto shard_shape = dst_tensor.shard_spec().value().shape;
            uint32_t num_tiles_per_shard = shard_shape[0] * shard_shape[1] / TILE_HW;
            UpdateCircularBufferTotalSize(program, cb_src0, num_tiles_per_shard * single_tile_size);
        }

        auto [cores, unary_reader_args, unary_writer_args] = get_runtime_args(src_tensor, dst_tensor, tensor_start);

        SetRuntimeArgs(program, unary_reader_kernel_id, cores, unary_reader_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, cores, unary_writer_args);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace tile

operation::ProgramWithCallbacks get_program(
    const Tensor &a, Tensor &output, const Shape &output_tensor_start, const Shape &output_tensor_end) {
    switch (a.layout()) {
        case Layout::ROW_MAJOR: return row_major::get_program(a, output, output_tensor_start, output_tensor_end);
        case Layout::TILE: return tile::get_program(a, output, output_tensor_start, output_tensor_end);
        default: TT_ASSERT(false, "Unsupported Layout");
    }
    return {};
}

}  // namespace unpad_impl::multi_core

}  // namespace tt_metal

}  // namespace tt
