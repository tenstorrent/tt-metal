// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tensor/owned_buffer_functions.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

inline void log_rt_args(const CoreCoord& core,  vector<uint32_t>& args) {
    for (auto v : args) {
        log_debug(LogOp, "{},{} :: {}", core.x, core.y, v);
    }
}

// This is currently mostly hardcoded for resnet shapes
inline std::tuple<uint32_t, uint32_t, uint32_t, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t, uint32_t, uint32_t>
    split_across_cores(CoreCoord grid_size, uint32_t nbatch, uint32_t nchannel, uint32_t ntiles_h, uint32_t ntiles_w) {

    uint32_t ncores, ncores_h, ncores_w, ntiles_per_core_h, ntiles_per_core_w, nbatch_per_core_h, ncores_per_batch_h;

    ncores_h = 1;

    // each batch needs to be padded independently
    switch (nbatch) {
        case 1:
            ncores_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2: ncores_h = 2; ntiles_per_core_h = 1; break;
                case 4: ncores_h = 4; ntiles_per_core_h = 1; break;
                case 8: ncores_h = 8; ntiles_per_core_h = 1; break;
                case 64: ncores_h = 8; ntiles_per_core_h = 8; break;
            }
            ncores_per_batch_h = ncores_h;
            break;

        case 2:
            ncores_h = 1;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2: ncores_per_batch_h = 2; ncores_h = ncores_per_batch_h * nbatch; ntiles_per_core_h = 1; break;
                case 4: ncores_per_batch_h = 4; ncores_h = ncores_per_batch_h * nbatch; ntiles_per_core_h = 1; break;
                case 8: ncores_per_batch_h = 4; ncores_h = ncores_per_batch_h * nbatch; ntiles_per_core_h = 2; break;
                case 64: ncores_per_batch_h = 4; ncores_h = ncores_per_batch_h * nbatch; ntiles_per_core_h = 16; break;
            }
            break;

        case 8:
            ncores_h = 8;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = ntiles_h;
            break;

        default:
            TT_ASSERT(false, "unhandled nbatch. TODO");

            // generic case -- TODO

            // one of the following will be 0 when grid_size.y != nbatch
            uint32_t nbatch_per_core_h = nbatch / grid_size.y;  // floor
            uint32_t ncores_per_batch_h = grid_size.y / nbatch; // floor
            if (nbatch == grid_size.y) {
                nbatch_per_core_h = 1;
                ncores_per_batch_h = 1;
            }

            // currently uses hardcoded values for resnet50
            // TT_ASSERT(ntiles_h == 1 || ntiles_h == 2 || ntiles_h == 4 || ntiles_h == 16, "Only Resnet50 shapes are supported in multicore version for now.");
            // TT_ASSERT(ntiles_w == 64, "Only Resnet50 shapes are supported in multicore version for now.");

            TT_ASSERT(nbatch <= grid_size.y, "Unsupported case with nbatch > grid_size.y!");

            uint32_t ncores_h = 1;
            uint32_t ntiles_per_core_h = ntiles_h / ncores_h;
            if (nbatch_per_core_h == 0) {
                // there are multiple cores along h per batch
                nbatch_per_core_h = 1;
                ncores_h = ncores_per_batch_h * nbatch;
                ntiles_per_core_h = ntiles_h / ncores_per_batch_h;
            } else if (ncores_per_batch_h == 0) {
                // unsupported case. TODO.
                TT_ASSERT(false);
                // there are multiple batch per core along h
                // ncores_per_batch_h = 1;
                // ncores_h = (uint32_t) ceil((float) nbatch / nbatch_per_core_h);
                // ntiles_per_core_h = nbatch_per_core_h * ntiles_h;
            } else {
                TT_ASSERT("Something went terribly wrong in splitting acrtoss cores");
            }
            break;
    }

    ncores_w = 1;
    switch (ntiles_w) {
        case 2: ncores_w = 2; break;
        case 4: ncores_w = 4; break;
        case 8: ncores_w = 8; break;
        case 64: ncores_w = 8; break;
    }
    ncores = ncores_h * ncores_w;
    ntiles_per_core_w = ntiles_w / ncores_w;
    std::set<CoreRange> all_cores;
    std::set<CoreRange> core_range;

    all_cores.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));
    core_range.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));

    return std::make_tuple(ncores, ncores_h, ncores_w, all_cores, core_range, ntiles_per_core_h, ntiles_per_core_w, nbatch_per_core_h, ncores_per_batch_h);
}

operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core(const Tensor &a,
                                                                Tensor &output,
                                                                const Shape &output_tensor_shape,
                                                                const Shape &input_tensor_start,
                                                                const float pad_value) {
    Program program{};

    auto output_shape = output_tensor_shape;

    uint32_t unpadded_row_size_nbytes = a.get_legacy_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();   // Assuming output is same datatype as input
    TT_ASSERT(unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    Device *device = a.device();

    // construct const buffer with the pad_value
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();
    auto pad_value_const_buffer = owned_buffer::create(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    // NOTE: The const buffer is always in L1
    // TODO: make a local buffer for each core?
    const Tensor pad_value_const_tensor = Tensor(OwnedStorage{pad_value_const_buffer},
                                                 Shape({1, 1, 1, pad_value_const_buffer_size}),
                                                 DataType::BFLOAT16, Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1});
    auto pad_value_const_tensor_addr = pad_value_const_tensor.buffer()->address();

    // uint32_t ntiles_h = output_tensor_shape[0] * output_tensor_shape[1] * output_tensor_shape[2] / TILE_HEIGHT;
    uint32_t ntiles_h = output_tensor_shape[2] / TILE_HEIGHT;
    uint32_t ntiles_w = output_tensor_shape[3] / TILE_WIDTH;

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t nbatch = output_tensor_shape[0];
    uint32_t nchannel = output_tensor_shape[1];
    // first the batch dim is distributed along H, and within each batch then the tiles are distributed.
    auto [ncores, ncores_h, ncores_w, all_cores, core_range, ntiles_per_core_h, ntiles_per_core_w, nbatch_per_core_h, ncores_per_batch_h] = split_across_cores(grid_size, nbatch, nchannel, ntiles_h, ntiles_w);

    int32_t src_nbytes_per_core_w = ntiles_per_core_w * TILE_WIDTH * a.element_size();
    int32_t dst_nbytes_per_core_w = ntiles_per_core_w * TILE_WIDTH * output.element_size();

    uint32_t cb_id = CB::c_in0;
    uint32_t cb_npages = 16; // multibuffering for perf
    // uint32_t cb_npages = 1; // multibuffering for perf
    uint32_t cb_pagesize = (uint32_t) ceil((float) dst_nbytes_per_core_w / constants::TILE_WIDTH) * constants::TILE_WIDTH;
    DataFormat in_df = datatype_to_dataformat_converter(a.get_dtype());
    tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(cb_npages * cb_pagesize, {{cb_id, in_df}})
		.set_page_size(cb_id, cb_pagesize);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(unpadded_row_size_nbytes);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t) std::log2(unpadded_row_size_nbytes) : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(padded_row_size_nbytes);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t) std::log2(padded_row_size_nbytes) : 0;
    std::vector<uint32_t> reader_ct_args = {(std::uint32_t) src0_is_dram,
                                            (std::uint32_t) dst_is_dram,
                                            (std::uint32_t) src_stick_size_is_power_of_two,
                                            (std::uint32_t) src_log2_stick_size,
                                            (std::uint32_t) dst_stick_size_is_power_of_two,
                                            (std::uint32_t) dst_log2_stick_size};
    std::vector<uint32_t> writer_ct_args = reader_ct_args;

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    bfloat16 bfloat_zero = bfloat16(0.0f);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_zero, bfloat_pad_value});

    KernelHandle reader_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp",
                                                        all_cores,
                                                        ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp",
                                                        all_cores,
                                                        WriterDataMovementConfig(writer_ct_args));
    // int32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;
    log_rt_args(CoreCoord{0, 0}, reader_ct_args);

    #if 1
    {
        log_debug("ncores: {}", ncores);
        log_debug("ncores_h: {}", ncores_h);
        log_debug("ncores_w: {}", ncores_w);
        log_debug("ntiles_per_core_h: {}", ntiles_per_core_h);
        log_debug("ntiles_per_core_w: {}", ntiles_per_core_w);
        log_debug("src0_buffer_addr: {}", src0_buffer->address());
        log_debug("dst_buffer_addr: {}", dst_buffer->address());
        log_debug("a.shape[0]: {}", a.get_legacy_shape()[0]);
        log_debug("out.shape[0]: {}", output_shape[0]);
        log_debug("a.shape[1]: {}", a.get_legacy_shape()[1]);
        log_debug("out.shape[1]: {}", output_shape[1]);
        log_debug("a.shape[2]: {}", a.get_legacy_shape()[2]);
        log_debug("out.shape[2]: {}", output_shape[2]);
        log_debug("s.shape[3]: {}", a.get_legacy_shape()[3]);
        log_debug("out.shape[3]: {}", output_shape[3]);
        log_debug("unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        log_debug("padded_row_size_nbytes: {}", padded_row_size_nbytes);
        // log_debug("padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        log_debug("pad_value_const_tensor_addr: {}", pad_value_const_tensor_addr);
        log_debug("pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        log_debug("packed_pad_value: {}", packed_pad_value);
        log_debug("src_nbytes_per_core_w: {}", src_nbytes_per_core_w);
        log_debug("dst_nbytes_per_core_w: {}", dst_nbytes_per_core_w);
        log_debug("nbatch_per_core_h: {}", nbatch_per_core_h);
        log_debug("ncores_per_batch_h: {}", ncores_per_batch_h);
    }
    #endif

    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;
    uint32_t start_src_stick_wi = 0; // start of stick segment for 2d decomp
    uint32_t start_dst_stick_wi = 0;
    int32_t local_nsticks = ntiles_per_core_h * TILE_HEIGHT;
    int32_t rem_nbatch = nbatch;    // per core h, there are ncores_per_batch_h cores, ie each batch ncores_h = ncores_per_batch_h
    for (int32_t b = 0; b < nbatch; ++ b) {
        int32_t rem_src_nsticks = a.get_legacy_shape()[2];
        for (uint32_t j = 0; j < ncores_per_batch_h; ++ j) {
            uint32_t num_local_unpadded_nsticks = local_nsticks;
            if (rem_src_nsticks - local_nsticks >= 0) {
                // not reached padding sticks yet
                rem_src_nsticks -= local_nsticks;
            } else {
                num_local_unpadded_nsticks = rem_src_nsticks;
                rem_src_nsticks = 0;
            }
            start_src_stick_wi = 0;
            start_dst_stick_wi = 0;
            int32_t rem_src_stick_size_nbytes = unpadded_row_size_nbytes;
            for (uint32_t i = 0; i < ncores_w; ++ i) {
                CoreCoord core = {i, b * ncores_per_batch_h + j};
                uint32_t curr_stick_size_nbytes = 0;
                int32_t curr_stick_diff_nbytes = 0;
                if (rem_src_stick_size_nbytes - dst_nbytes_per_core_w >= 0) {
                    // no padding on this core
                    curr_stick_size_nbytes = dst_nbytes_per_core_w;
                    rem_src_stick_size_nbytes -= dst_nbytes_per_core_w;
                } else {
                    // this core has padding
                    curr_stick_size_nbytes = rem_src_stick_size_nbytes;
                    curr_stick_diff_nbytes = dst_nbytes_per_core_w - curr_stick_size_nbytes;
                    rem_src_stick_size_nbytes = 0;
                }
                vector<uint32_t> reader_rt_args = {src0_buffer->address(),
                                                    dst_buffer->address(),
                                                    a.get_legacy_shape()[0],
                                                    output_shape[0],
                                                    a.get_legacy_shape()[1],
                                                    output_shape[1],
                                                    a.get_legacy_shape()[2],
                                                    output_shape[2],
                                                    a.get_legacy_shape()[3],
                                                    output_shape[3],
                                                    curr_stick_size_nbytes,
                                                    (uint32_t) dst_nbytes_per_core_w,
                                                    (uint32_t) curr_stick_diff_nbytes,
                                                    pad_value_const_tensor_addr,
                                                    pad_value_const_buffer_nbytes,
                                                    packed_pad_value,
                                                    start_src_stick_id,
                                                    start_dst_stick_id,
                                                    start_src_stick_wi,
                                                    start_dst_stick_wi,
                                                    start_src_stick_wi * a.element_size(),
                                                    (uint32_t) local_nsticks,
                                                    num_local_unpadded_nsticks,
                                                    unpadded_row_size_nbytes,
                                                    padded_row_size_nbytes,
                                                    start_dst_stick_wi * output.element_size(),
                                                    nbatch_per_core_h
                                                    };
                // if (core.x == 0) log_rt_args(core, reader_rt_args);
                // if (core.x == 0) {
                //     log_debug("{} :: start_src_stick_id: {}", core.y, start_src_stick_id);
                //     log_debug("{} :: start_dst_stick_id: {}", core.y, start_dst_stick_id);
                //     log_debug("{} :: local_nsticks: {}", core.y, local_nsticks);
                //     log_debug("{} :: num_local_unpadded_nsticks: {}", core.y, num_local_unpadded_nsticks);
                //     log_debug("{} :: nbatch_per_core_h: {}", core.y, nbatch_per_core_h);
                //     log_debug("{} :: ncores_per_batch_h: {}", core.y, ncores_per_batch_h);
                // }
                vector<uint32_t> writer_rt_args = reader_rt_args;
                SetRuntimeArgs(program,
                                reader_kernel_id,
                                core,
                                reader_rt_args);
                SetRuntimeArgs(program,
                                writer_kernel_id,
                                core,
                                writer_rt_args);
                start_src_stick_wi += ntiles_per_core_w * TILE_WIDTH;
                start_dst_stick_wi += ntiles_per_core_w * TILE_WIDTH;
            } // for ncores_w
            start_src_stick_id += num_local_unpadded_nsticks;
            start_dst_stick_id += local_nsticks;
        } // for ncores_h
    }

    auto override_runtime_args_callback = [reader_kernel_id = reader_kernel_id,
                                           writer_kernel_id = writer_kernel_id,
                                           ncores_h = ncores_h,
                                           ncores_w = ncores_w](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t j = 0; j < ncores_h; ++ j) {
            for (uint32_t i = 0; i < ncores_w; ++ i) {
                CoreCoord core = {i, j};
                {
                    auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_buffer->address();
                    runtime_args[1] = dst_buffer->address();
                }
                {
                    auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = src_buffer->address();
                    runtime_args[1] = dst_buffer->address();
                }
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal
}  // namespace tt
