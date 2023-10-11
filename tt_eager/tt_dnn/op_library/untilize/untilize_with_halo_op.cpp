// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tensor/owned_buffer_functions.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

namespace untilize_with_halo_helpers {

// reader noc coords for left and right neighbors
std::map<CoreCoord, CoreCoord> left_neighbor_noc_xy, right_neighbor_noc_xy;
void init_neighbor_noc_xy_mapping(CoreCoord grid_size, uint32_t noc = 0) {
    TT_ASSERT(grid_size.x == 12 && grid_size.y == 9);
    if (noc == 0) {
        for (uint32_t y = 1; y <= grid_size.y; ++ y) {
            for (uint32_t x = 1; x <= grid_size.x; ++ x) {
                CoreCoord local_noc = {x, y > 5 ? y + 1 : y};
                CoreCoord left_noc, right_noc;
                // calculate left neighbor
                left_noc.x = local_noc.x;
                left_noc.y = local_noc.y;
                if (left_noc.x > 1) {
                    left_noc.x -= 1;
                } else {
                    left_noc.x = grid_size.x;
                    left_noc.y -= 1;
                }
                if (left_noc.y < 1) {
                    // there is no left neighbor
                } else {
                    if (left_noc.y == 6) {
                        // y = 6 is to be skipped
                        left_noc.y -= 1;
                    }
                    left_neighbor_noc_xy[local_noc] = {left_noc.x, left_noc.y};
                }
                // calculate right neighbor
                right_noc.x = local_noc.x;
                right_noc.y = local_noc.y;
                if (right_noc.x < grid_size.x) {
                    right_noc.x += 1;
                } else {
                    right_noc.y += 1;
                    right_noc.x = 1;
                }
                // NOTE: y = 6 is to be skipped. Hence go till y + 1
                if (right_noc.y > grid_size.y + 1) {
                    // there is no right neighbor
                } else {
                    if (right_noc.y == 6) {
                        // y = 6 is to be skipped
                        right_noc.y += 1;
                    }
                    right_neighbor_noc_xy[local_noc] = {right_noc.x, right_noc.y};
                }
            }
        }
    } else {
        // noc == 1
        TT_ASSERT(noc == 0, "noc = 1 for reader is not yet handled in sharded input case.");
    }
}

} // namespace untilize_with_halo_helpers

operation::ProgramWithCallbacks untilize_with_halo_multi_core(const Tensor& a, Tensor& output, uint32_t pad_val, const uint32_t &in_b, const uint32_t &in_h, const uint32_t &in_w, const uint32_t &out_shard_size_max_per_core) {
    Program program = Program();

    Device *device = a.device();
    Buffer *src_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    Shape input_shape = a.shape();
    Shape output_shape = output.shape();

    DataFormat in_df = datatype_to_dataformat_converter(a.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);

    DataFormat cb_df = in_df;
    uint32_t tile_size = tt_metal::detail::TileSize(cb_df);

    uint32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = input_shape[3] / TILE_WIDTH;
    TT_ASSERT(a.shard_spec().value().shard_shape[1] == input_shape[3], "Input shape in W should be same as shard width!");

    uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = input_shape[3] * a.element_size();

    {
        log_debug(LogOp, "ntiles: {}", ntiles);
        log_debug(LogOp, "ntiles_per_block: {}", ntiles_per_block);
        log_debug(LogOp, "nblocks: {}", nblocks);
    }

    // TODO: hard coded for testing only. need to pass these args in.
    uint32_t nbatch = in_b;
    uint32_t in_c = input_shape[3];
    uint32_t pad_h = 1;
    uint32_t pad_w = 1;
    uint32_t window_h = 3;
    uint32_t window_w = 3;

    {
        log_debug(LogOp, "nbatch: {}", nbatch);
        log_debug(LogOp, "in_h: {}", in_h);
        log_debug(LogOp, "in_w: {}", in_w);
        log_debug(LogOp, "in_c: {}", in_c);
        log_debug(LogOp, "pad_h: {}", pad_h);
        log_debug(LogOp, "pad_w: {}", pad_w);
        log_debug(LogOp, "window_h: {}", window_h);
        log_debug(LogOp, "window_w: {}", window_w);
    }

    auto grid_size = device->compute_with_storage_grid_size();

    untilize_with_halo_helpers::init_neighbor_noc_xy_mapping(grid_size);

    int32_t ncores_x = grid_size.x;
    int32_t ncores_y = grid_size.y;
    CoreRangeSet all_cores = a.shard_spec().value().shard_grid;
    int32_t ncores = 0;
    for (const auto& core_range : all_cores.ranges()) {
        ncores += core_range.size();
    }
    CoreRangeSet core_range_cliff = CoreRangeSet({});
    uint32_t nblocks_per_core = a.shard_spec().value().shard_shape[0] / TILE_HEIGHT;
    uint32_t nblocks_per_core_cliff = 0;

    uint32_t in_hw = in_h * in_w;
    uint32_t in_nhw = nbatch * in_hw;
    uint32_t in_stick_nbytes = in_c * in_nbytes;
    uint32_t in_nsticks = in_nhw;
    uint32_t in_nsticks_per_batch = in_hw;
    uint32_t in_nsticks_per_core = in_nhw / ncores;

    {
        log_debug(LogOp, "shard_shape: {},{}", a.shard_spec().value().shard_shape[0], a.shard_spec().value().shard_shape[1]);
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "ncores_x: {}", ncores_x);
        log_debug(LogOp, "ncores_y: {}", ncores_y);
        log_debug(LogOp, "nblocks_per_core: {}", nblocks_per_core);
        log_debug(LogOp, "nblocks_per_core_cliff: {}", nblocks_per_core_cliff);
        log_debug(LogOp, "in_hw: {}", in_hw);
        log_debug(LogOp, "in_nhw: {}", in_nhw);
        log_debug(LogOp, "in_stick_nbytes: {}", in_stick_nbytes);
        log_debug(LogOp, "in_nsticks_per_batch: {}", in_nsticks_per_batch);
        log_debug(LogOp, "in_nsticks_per_core: {}", in_nsticks_per_core);
    }

    uint32_t src_cb_id = CB::c_in0;
    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    auto src_cb_config = CircularBufferConfig(num_input_tiles * tile_size, {{src_cb_id, cb_df}})
                            .set_page_size(src_cb_id, tile_size)
                            .set_globally_allocated_address(a.buffer()->address());
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);

    // output of untilize from compute kernel goes into this CB
    uint32_t untilize_out_cb_id = CB::c_out0;
    uint32_t num_output_tiles = ntiles_per_block * nblocks_per_core;
    auto untilize_out_cb_config = CircularBufferConfig(num_output_tiles * tile_size, {{untilize_out_cb_id, cb_df}})
                                    .set_page_size(untilize_out_cb_id, tile_size);
    auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);

    // output after concatenating halo and padding goes into this CB, as input to next op.
    uint32_t out_cb_id = CB::c_out1;
    uint32_t out_cb_pagesize = out_nbytes * in_c;
    uint32_t out_nsticks_per_core = out_shard_size_max_per_core;
    {
        log_debug(LogOp, "out_cb_pagesize: {}", out_cb_pagesize);
        log_debug(LogOp, "out_nsticks_per_core: {}", out_nsticks_per_core);
    }
    auto out_cb_config = CircularBufferConfig(out_shard_size_max_per_core * out_cb_pagesize, {{out_cb_id, cb_df}})
                            .set_page_size(out_cb_id, out_cb_pagesize)
                            .set_globally_allocated_address(output.buffer()->address());
    auto cb_out = CreateCircularBuffer(program, all_cores, out_cb_config);

    /** reader
     */

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t) src_cb_id
    };

    KernelID reader_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t) untilize_out_cb_id,
        (std::uint32_t) out_cb_id
    };
    KernelID writer_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_sharded_with_halo.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    /** compute
     */
    TT_ASSERT(core_range_cliff.ranges().size() == 0);
    vector<uint32_t> compute_args = {
        (uint32_t) nblocks_per_core,    // per_core_block_cnt
        (uint32_t) ntiles_per_block,    // per_block_ntiles
    };
    KernelID untilize_kernel_id = CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        all_cores,
        ComputeConfig{
            .compile_args = compute_args});

    // const buffer with pad value (-INF)
    uint32_t const_buffer_size = input_shape[3];
    auto const_buffer = owned_buffer::create(std::vector<bfloat16>(const_buffer_size, bfloat16(uint16_t(pad_val))));
    const Tensor const_tensor = Tensor(OwnedStorage{const_buffer},
                                       Shape({1, 1, 1, const_buffer_size}),
                                       DataType::BFLOAT16,
                                       Layout::ROW_MAJOR)
                                    .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED,
                                                             .buffer_type = BufferType::L1});
    auto const_tensor_addr = const_tensor.buffer()->address();

    // 1D distribution of blocks across all cores
    uint32_t ncores_full = ncores;
    // cliff core not yet supported
    TT_ASSERT(nblocks_per_core_cliff == 0);

    // reader runtime args
    vector<uint32_t> reader_rt_args = {
        ntiles_per_block * nblocks_per_core // ntiles
    };

    TT_ASSERT(in_nhw % ncores == 0);

    vector<uint32_t> writer_rt_args = {
        ntiles_per_block * nblocks_per_core,  // in_nsticks,                     // 0
        in_nsticks_per_core,  // UNUSED
        0,  // partial_first_row_nsticks,
        pad_w,
        in_w,
        0,  // partial_top_image_nrows,        // 5
        pad_h,
        in_h,
        0,  // full_nimages,
        0,  // partial_bottom_image_nrows,
        0,  // partial_last_row_nsticks,       // 10
        0,  // halo_for_left_left_nsticks,
        0,  // halo_for_left_nsticks,
        0,  // halo_for_right_nsticks,
        0,  // halo_for_right_right_nsticks,
        0,  // local_in_stick_start,           // 15
        0,  // local_in_stick_end,
        in_nsticks_per_batch,
        in_nsticks_per_core,
        0,  // has_left,
        0,  // left_noc_x,                     // 20
        0,  // left_noc_y,
        0,  // has_right,
        0,  // right_noc_x,
        0,  // right_noc_y,
        0,  // has_left_left,                  // 25
        0,  // left_left_noc_x,
        0,  // left_left_noc_y,
        0,  // has_right_right,
        0,  // right_right_noc_x,
        0,  // right_right_noc_y,              // 30
        in_stick_nbytes,
        0,  // left_left_nsticks,
        0,  // left_nsticks,
        0,  // right_nsticks,
        0,  // right_right_nsticks,            // 35
        0,  // right_right_halo_offset,
        0,  // right_halo_offset,
        0,  // left_halo_offset,
        0,  // left_left_halo_offset,
        0,  // left_halo_pad_i_offset          // 40
        0,  // right_halo_pad_i_offset
        0,  // partial_first_row_skip
        0,  // partial_top_image_skip
        0,  // full_image_skip
        0,  // initial_pad_nsticks             // 45
        const_tensor_addr,
    };

    uint32_t writer_noc = 0;

    TT_ASSERT(window_h == 3 && window_w == 3);
    // int32_t halo_in_nsticks = (in_w + (window_w / 2)) * (window_h / 2);
    int32_t halo_in_nsticks = (in_w + 1) * 1;
    int32_t halo_out_nsticks = (in_w + 2 * pad_w) * pad_h + window_w / 2;

    log_debug(LogOp, "halo_in_nsticks: {}", halo_in_nsticks);
    log_debug(LogOp, "halo_out_nsticks: {}", halo_out_nsticks);

    // NOTE: Irrespective of batch boundary, always ass the left/right halo to the output shards.
    // IE: output shards ALWAYS have halo region on left and right as long as they are not the start/end of the input
    // TODO: Ensure this produces the correct output after padding is inserted.

    // For each core, calculate how many sticks are coming from left left/left/right/right right neighbors:
    // These are used by the neighbor cores to set their runtime args for the data to be pushed to me.
    // Also calculate each of these segments start offset in my local l1: these need to take all my padding (left/right/top) into account
    map<int32_t, int32_t> my_left_halo, my_left_left_halo, my_right_halo, my_right_right_halo;
    map<int32_t, uint32_t> my_left_halo_offset, my_left_left_halo_offset, my_right_halo_offset, my_right_right_halo_offset;
    map<int32_t, uint32_t> my_left_halo_pad_i_offset, my_right_halo_pad_i_offset;
    int32_t in_stick_start = 0;

    for (int32_t i = 0; i < ncores_full; ++ i) {
        int32_t in_stick_batch_start = in_stick_start / in_nsticks_per_batch;
        int32_t in_stick_batch_end = (in_stick_start + in_nsticks_per_core) / in_nsticks_per_batch;

        // left side halo
        my_left_halo[i] = 0;
        my_left_left_halo[i] = 0;
        int32_t halo_stick_start = in_stick_start - halo_in_nsticks;
        int32_t stick_id = halo_stick_start < 0 ? 0 : halo_stick_start;
        while(stick_id < in_stick_start) {
            int32_t core = stick_id / in_nsticks_per_core;
            switch (core - i) {
                case - 2:
                    TT_ASSERT(false, "This case not correctly handled yet. Needs fixing");
                    ++ my_left_left_halo[i];
                    break;
                case - 1:
                    ++ my_left_halo[i];
                    break;
                default:
                    TT_ASSERT(false, "Encountered an unsupported case!!!!");
            }
            ++ stick_id;
        }
        my_left_left_halo_offset[i] = 0;    // always starts at the beginning
        if ((halo_stick_start / in_w) == ((halo_stick_start + my_left_left_halo[i]) / in_w)) {
            // left left and left halo start on the same row, there is no additional offset
            my_left_halo_offset[i] = my_left_left_halo_offset[i] + my_left_left_halo[i] * in_stick_nbytes;
        } else {
            // left left halo and left halo are in different rows, so there's 2 additional padding sticks (right/left edge)
            my_left_halo_offset[i] = my_left_left_halo_offset[i] + (my_left_left_halo[i] + 2) * in_stick_nbytes;
        }
        my_left_halo_pad_i_offset[i] = in_w - (in_stick_start % in_w);

        // right side halo
        my_right_halo[i] = 0;
        my_right_right_halo[i] = 0;
        int32_t halo_stick_end = in_stick_start + in_nsticks_per_core + halo_in_nsticks;
        stick_id = in_stick_start + in_nsticks_per_core;
        while(stick_id < halo_stick_end) {
            int32_t core = stick_id / in_nsticks_per_core;
            switch (core - i) {
                case 2:
                    TT_ASSERT(false, "This case not correctly handled yet. Needs fixing");
                    ++ my_right_right_halo[i];
                    break;
                case 1:
                    ++ my_right_halo[i];
                    break;
                default:
                    TT_ASSERT(false, "Something went really wrong!!");
            }
            ++ stick_id;
        }

        // uint32_t my_batch = in_stick_start / in_nsticks_per_batch;
        // uint32_t my_core = i;

        ShardingConfig sc = get_specs_for_sharding_partition(in_stick_start, in_stick_start + in_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        uint32_t partial_first_row_nsticks = sc.first_partial_right_aligned_row_width;
        uint32_t partial_top_image_nrows = sc.first_partial_image_num_rows;
        uint32_t full_nimages = sc.num_full_images;
        uint32_t partial_bottom_image_nrows = sc.last_partial_image_num_rows;
        uint32_t partial_last_row_nsticks = sc.last_partial_left_aligned_row_width;
        uint32_t partial_first_row_skip = sc.skip_after_partial_right_aligned_row;
        uint32_t partial_top_image_skip = sc.skip_after_first_partial_image_row;
        uint32_t full_image_skip = sc.skip_after_full_image;
        uint32_t initial_pad_nsticks = 0;
        if (in_stick_start % (in_h * in_w) == 0) {
            // This is start of image. Insert initial padding worth halo size
            initial_pad_nsticks = halo_out_nsticks;
        }

        // uint32_t local_nsticks = in_nsticks_per_core                                                // data sticks
        //                             + (in_nsticks_per_core / in_w) * 2                              // left/right edge padding
        //                             + (in_nsticks_per_core / in_nsticks_per_batch) * (in_w + 2);    // padding rows
        uint32_t local_nsticks = partial_first_row_nsticks + partial_first_row_skip
                                    + partial_top_image_nrows * (in_w + 2 * pad_w) + partial_top_image_skip
                                    + full_nimages * (in_w + 2 * pad_w) * in_h
                                    + full_image_skip
                                    + partial_bottom_image_nrows * (in_w + 2 * pad_w)
                                    + partial_last_row_nsticks;

        // NOTE: this is always the same for all cores
        uint32_t halo_nsticks = (in_w + window_w / 2 + 2 * pad_w);  // left or right halo
        uint32_t out_nsticks_per_core = local_nsticks + 2 * halo_nsticks;
        {
            log_debug(LogOp, "local_nsticks: {}", local_nsticks);
            log_debug(LogOp, "halo_nsticks: {}", halo_nsticks);
            log_debug(LogOp, "out_nsticks_per_core: {}", out_nsticks_per_core);
        }

        my_right_halo_offset[i] = my_left_halo_offset[i] + (((initial_pad_nsticks > 0) ? initial_pad_nsticks : halo_nsticks) + local_nsticks) * in_stick_nbytes;
                                    // (in_nsticks_per_core /* local sticks */ + (in_nsticks_per_core / in_w) * 2 /* 2 padding sticks per row */) * in_stick_nbytes;
        if ((in_stick_start + in_nsticks_per_core) / in_w == (in_stick_start + in_nsticks_per_core + my_right_halo[i]) / in_w) {
            my_right_right_halo_offset[i] = my_right_halo_offset[i] + my_right_halo[i] * in_stick_nbytes;
        } else {
            my_right_right_halo_offset[i] = my_right_halo_offset[i] + (my_right_halo[i] + 2) * in_stick_nbytes;
        }
        my_right_halo_pad_i_offset[i] = in_w - ((in_stick_start + in_nsticks_per_core) % in_w);

        if (0)
        {
            log_debug(LogOp, "==== Core {}", i);
            log_debug(LogOp, "in_stick_start {}", in_stick_start);
            log_debug(LogOp, "my_left_halo = {}", my_left_halo[i]);
            log_debug(LogOp, "my_left_halo_offset = {}", my_left_halo_offset[i]);
            log_debug(LogOp, "my_left_left_halo = {}", my_left_left_halo[i]);
            log_debug(LogOp, "my_left_left_halo_offset = {}", my_left_left_halo_offset[i]);
            log_debug(LogOp, "my_left_halo_pad_i_offset = {}", my_left_halo_pad_i_offset[i]);
            log_debug(LogOp, "my_right_halo = {}", my_right_halo[i]);
            log_debug(LogOp, "my_right_halo_offset = {}", my_right_halo_offset[i]);
            log_debug(LogOp, "my_right_right_halo = {}", my_right_right_halo[i]);
            log_debug(LogOp, "my_right_right_halo_offset = {}", my_right_right_halo_offset[i]);
            log_debug(LogOp, "my_right_halo_pad_i_offset = {}", my_right_halo_pad_i_offset[i]);
        }

        in_stick_start += in_nsticks_per_core;
    }

    in_stick_start = 0;
    uint32_t out_stick_start = 0;
    for (uint32_t i = 0; i < ncores_full; ++ i) {
        CoreCoord core = {i % ncores_x, i / ncores_x};  // logical
        CoreCoord noc_core = core;  // calculate physical
        if (writer_noc == 0) {
            noc_core.x += 1;
            noc_core.y += 1;
            if (noc_core.y > 5) {
                noc_core.y += 1;
            }
        } else {
            TT_ASSERT(false);
        }

        // reader rt args
        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        // writer rt args
        writer_rt_args[15] = in_stick_start;    // unused
        writer_rt_args[16] = in_stick_start + in_nsticks_per_core;  // unused

        // int32_t partial_first_row_nsticks = (in_w - (in_stick_start % in_w)) % in_w;
        // int32_t batch = in_stick_start / in_hw;
        // int32_t partial_top_image_nrows = ((batch + 1) * in_h - (int32_t) ceil((float) in_stick_start / in_w)) % in_h;
        // int32_t full_nimages = ((int32_t) in_nsticks_per_core - (partial_first_row_nsticks + (partial_top_image_nrows * in_w))) / in_nsticks_per_batch;
        // int32_t rem_nsticks = (int32_t) in_nsticks_per_core - (partial_first_row_nsticks + partial_top_image_nrows * in_w + full_nimages * in_hw);
        // int32_t partial_bottom_image_nrows = rem_nsticks / in_w;
        // int32_t partial_last_row_nsticks = rem_nsticks % in_w;

        // ShardingConfig sc = get_specs_for_sharding_partition(out_stick_start, out_stick_start + out_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        ShardingConfig sc = get_specs_for_sharding_partition(in_stick_start, in_stick_start + in_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        uint32_t partial_first_row_nsticks = sc.first_partial_right_aligned_row_width;
        uint32_t partial_top_image_nrows = sc.first_partial_image_num_rows;
        uint32_t full_nimages = sc.num_full_images;
        uint32_t partial_bottom_image_nrows = sc.last_partial_image_num_rows;
        uint32_t partial_last_row_nsticks = sc.last_partial_left_aligned_row_width;
        uint32_t partial_first_row_skip = sc.skip_after_partial_right_aligned_row;
        uint32_t partial_top_image_skip = sc.skip_after_first_partial_image_row;
        uint32_t full_image_skip = sc.skip_after_full_image;
        uint32_t initial_pad_nsticks = 0;
        if (in_stick_start % (in_h * in_w) == 0) {
            // This is start of image. Insert initial padding worth halo size
            initial_pad_nsticks = halo_out_nsticks;
        }

        writer_rt_args[45] = initial_pad_nsticks;

        writer_rt_args[2] = partial_first_row_nsticks;
        writer_rt_args[5] = partial_top_image_nrows;
        writer_rt_args[8] = full_nimages;
        writer_rt_args[9] = partial_bottom_image_nrows;
        writer_rt_args[10] = partial_last_row_nsticks;

        writer_rt_args[42] = partial_first_row_skip;
        writer_rt_args[43] = partial_top_image_skip;
        writer_rt_args[44] = full_image_skip;

        if (untilize_with_halo_helpers::left_neighbor_noc_xy.count(noc_core) > 0) {
            CoreCoord left_noc = untilize_with_halo_helpers::left_neighbor_noc_xy.at(noc_core);
            writer_rt_args[19] = 1;
            writer_rt_args[20] = left_noc.x;
            writer_rt_args[21] = left_noc.y;
            writer_rt_args[33] = my_right_halo[i - 1];      // sticks to left neighbor core's right halo
            writer_rt_args[37] = my_right_halo_offset[i - 1];
            if (untilize_with_halo_helpers::left_neighbor_noc_xy.count(left_noc) > 0) {
                CoreCoord left_left_noc = untilize_with_halo_helpers::left_neighbor_noc_xy.at(left_noc);
                writer_rt_args[25] = 1;
                writer_rt_args[26] = left_left_noc.x;
                writer_rt_args[27] = left_left_noc.y;
                writer_rt_args[32] = my_right_right_halo[i - 2];    // sticks to left left neighbor core's right right halo
                writer_rt_args[36] = my_right_right_halo_offset[i - 2];
            } else {
                writer_rt_args[25] = 0;
            }
            writer_rt_args[40] = my_right_halo_pad_i_offset[i - 1];
        } else {
            writer_rt_args[19] = 0;
            writer_rt_args[25] = 0;
        }
        if (untilize_with_halo_helpers::right_neighbor_noc_xy.count(noc_core) > 0 && (i < ncores_full - 1)) {
            CoreCoord right_noc = untilize_with_halo_helpers::right_neighbor_noc_xy.at(noc_core);
            writer_rt_args[22] = 1;
            writer_rt_args[23] = right_noc.x;
            writer_rt_args[24] = right_noc.y;
            writer_rt_args[34] = my_left_halo[i + 1];       // sticks to right neighbor core's left halo
            writer_rt_args[38] = my_left_halo_offset[i + 1];
            if (untilize_with_halo_helpers::right_neighbor_noc_xy.count(noc_core) > 0 && (i < ncores_full - 2)) {
                CoreCoord right_right_noc = untilize_with_halo_helpers::right_neighbor_noc_xy.at(right_noc);
                writer_rt_args[28] = 1;
                writer_rt_args[29] = right_right_noc.x;
                writer_rt_args[30] = right_right_noc.y;
                writer_rt_args[35] = my_left_left_halo[i + 2];      // sticks to right right neighbor core's left left halo
                writer_rt_args[39] = my_left_left_halo_offset[i + 2];
            } else {
                writer_rt_args[28] = 0;
            }
            writer_rt_args[41] = my_left_halo_pad_i_offset[i + 1];
        } else {
            writer_rt_args[22] = 0;
            writer_rt_args[28] = 0;
        }

        if (0)
        {
            log_debug(LogOp, "++++ Core: {}", i);
            log_debug(LogOp, "out_stick_start: {}", out_stick_start);
            log_debug(LogOp, "halo::has_left: {}", writer_rt_args[19]);
            // log_debug(LogOp, "halo::has_left_left: {}", writer_rt_args[25]);
            log_debug(LogOp, "halo::has_right: {}", writer_rt_args[22]);
            // log_debug(LogOp, "halo::has_right_right: {}", writer_rt_args[28]);
            log_debug(LogOp, "local_in_stick_start: {}", writer_rt_args[15]);
            log_debug(LogOp, "partial_first_row_nsticks: {}", writer_rt_args[2]);
            log_debug(LogOp, "partial_top_image_nrows: {}", writer_rt_args[5]);
            log_debug(LogOp, "full_nimages: {}", writer_rt_args[8]);
            log_debug(LogOp, "partial_bottom_image_nrows: {}", writer_rt_args[9]);
            log_debug(LogOp, "partial_last_row_nsticks: {}", writer_rt_args[10]);
            log_debug(LogOp, "skip_after_partial_right_aligned_row: {}", sc.skip_after_partial_right_aligned_row);
            log_debug(LogOp, "skip_after_first_partial_image_row: {}", sc.skip_after_first_partial_image_row);
            log_debug(LogOp, "skip_after_full_image: {}", sc.skip_after_full_image);
            // log_debug(LogOp, "halo_for_left_left_nsticks: {}", writer_rt_args[11]);
            log_debug(LogOp, "halo_for_left_nsticks: {}", writer_rt_args[12]);
            log_debug(LogOp, "halo_for_right_nsticks: {}", writer_rt_args[13]);
            // log_debug(LogOp, "halo_for_right_right_nsticks: {}", writer_rt_args[14]);
            // log_debug(LogOp, "left_left_core_nsticks: {}", writer_rt_args[32]);
            log_debug(LogOp, "left_core_nsticks: {}", writer_rt_args[33]);
            log_debug(LogOp, "right_core_nsticks: {}", writer_rt_args[34]);
            // log_debug(LogOp, "right_right_core_nsticks: {}", writer_rt_args[35]);
            // log_debug(LogOp, "left_left_core_halo_offset: {}", writer_rt_args[36]);
            log_debug(LogOp, "left_core_halo_offset: {}", writer_rt_args[37]);
            log_debug(LogOp, "right_core_halo_offset: {}", writer_rt_args[38]);
            // log_debug(LogOp, "right_right_core_halo_offset: {}", writer_rt_args[39]);
            // log_debug(LogOp, "left_going_halo_pad_i_offset: {}", writer_rt_args[40]);
            // log_debug(LogOp, "right_going_halo_pad_i_offset: {}", writer_rt_args[41]);
        }

        SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);

        in_stick_start += in_nsticks_per_core;
        out_stick_start += out_nsticks_per_core;
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        // TODO: support cliff core
        TT_ASSERT(false);
    }

    auto override_runtime_args_callback = [
        reader_kernel_id=reader_kernel_id,
        writer_kernel_id=writer_kernel_id,
        ncores=ncores,
        ncores_x=ncores_x
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < ncores; ++ i) {
            CoreCoord core = {i % ncores_x, i / ncores_x};
            // in and out are sharded
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void UntilizeWithHalo::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to untilize need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Input tensor is not TILE for untilize");
    TT_ASSERT(input_tensor_a.memory_config().is_sharded());
    TT_ASSERT(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Only works for sharded input");
    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
}

std::vector<Shape> UntilizeWithHalo::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.shape();
    Shape output_shape = input_shape;
    // pad_h, pad_w
    // calculate the sizes (num sticks) for each of the 7 sections (5 local, 2 halo)
    // output num sticks:
    // local sections:
    // 1. (partial first row width + pad_w)
    // 2. (out_w + pad_w * 2) * (num full rows partial top image)
    // 3. (out_w + pad_w * 2) * (pad_h + out_h) * num full images
    // 4. (out_w + pad_w * 2) * (pad_h + num full rows partial bottom image)
    // 5. (partial last row width + pad_w)
    // halo sections on local core:
    // A. left halo: out_w + pad_w + 1
    // B. right halo: out_w + pad_w + 1
    // corresponding halo sections on neighbors
    // Aa. left left halo:
    // Ab. left halo:
    // Ba. right halo:
    // Bb. right right halo:

    uint32_t in_nhw = this->in_b * this->in_h * this->in_w;

    // get ncores from shard shape and input shape
    auto shard_shape = input.shard_spec().value().shard_shape;
    uint32_t ncores = in_nhw / shard_shape[0];

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[3] remains same
    // output_shape[2] changes
    output_shape[2] = this->out_shard_size_max_per_core * ncores;

    log_debug(LogOp, "output_shape: {} {} {} {}", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(LogOp, "derived ncores: {}", ncores);

    return {output_shape};
}

std::vector<Tensor> UntilizeWithHalo::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shard_spec = input_tensor.shard_spec().value();
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    uint32_t ncores = input_tensor.shape()[0] * input_tensor.shape()[2] / shard_spec.shard_shape[0];
    shard_spec.shard_shape[0] = output_shape[2] / ncores;
    // log_debug(LogOp, "derived ncores: {}", ncores);
    return {create_sharded_device_tensor(output_shape, input_tensor.dtype(), Layout::ROW_MAJOR, input_tensor.device(), this->output_mem_config, shard_spec)};
}

operation::ProgramWithCallbacks UntilizeWithHalo::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {untilize_with_halo_multi_core(input_tensor_a, output_tensor, pad_val_, this->in_b, this->in_h, this->in_w, this->out_shard_size_max_per_core)};
}

tt::stl::reflection::Attributes UntilizeWithHalo::attributes() const {
    return {
        {"pad_val", pad_val_},
        {"in_b", this->in_b},
        {"in_h", this->in_h},
        {"in_w", this->in_w},
        {"out_shard_size_max_per_core", this->out_shard_size_max_per_core},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor untilize_with_halo(const Tensor &input_tensor_a, const uint32_t pad_val, const uint32_t &in_b, const uint32_t &in_h, const uint32_t &in_w, const MemoryConfig& mem_config) {
    TT_ASSERT(input_tensor_a.memory_config().is_sharded()); // TODO: Remove from validate?

    uint32_t pad_h = 1;
    uint32_t pad_w = 1;
    uint32_t window_w = 3;
    uint32_t ncores = 0;
    CoreRangeSet all_cores = input_tensor_a.shard_spec().value().shard_grid;
    for (const auto& core_range : all_cores.ranges()) {
        ncores += core_range.size();
    }

    // TODO: Uplift to support different num of sticks per core
    uint32_t in_nsticks_per_core = (in_b * in_h * in_w) / ncores;
    // NOTE: This is always the same for all cores
    uint32_t halo_nsticks = (in_w + window_w / 2 + 2 * pad_w);  // left or right halo

    uint32_t out_nsticks_max_per_core = 0;
    for (int32_t i = 0; i < ncores; ++ i) {
        uint32_t in_stick_start = in_nsticks_per_core * i;
        ShardingConfig sc = get_specs_for_sharding_partition(in_stick_start, in_stick_start + in_nsticks_per_core, in_h, in_w, window_w, pad_h, pad_w);
        uint32_t partial_first_row_nsticks = sc.first_partial_right_aligned_row_width;
        uint32_t partial_top_image_nrows = sc.first_partial_image_num_rows;
        uint32_t full_nimages = sc.num_full_images;
        uint32_t partial_bottom_image_nrows = sc.last_partial_image_num_rows;
        uint32_t partial_last_row_nsticks = sc.last_partial_left_aligned_row_width;
        uint32_t partial_first_row_skip = sc.skip_after_partial_right_aligned_row;
        uint32_t partial_top_image_skip = sc.skip_after_first_partial_image_row;
        uint32_t full_image_skip = sc.skip_after_full_image;

        uint32_t local_nsticks = partial_first_row_nsticks + partial_first_row_skip
                                    + partial_top_image_nrows * (in_w + 2 * pad_w) + partial_top_image_skip
                                    + full_nimages * (in_w + 2 * pad_w) * in_h
                                    + full_image_skip
                                    + partial_bottom_image_nrows * (in_w + 2 * pad_w)
                                    + partial_last_row_nsticks;

        uint32_t out_nsticks = local_nsticks + 2 * halo_nsticks;
        out_nsticks_max_per_core = std::max(out_nsticks_max_per_core, out_nsticks);
    }
    return operation::run_without_autoformat(UntilizeWithHalo{pad_val, in_b, in_h, in_w, out_nsticks_max_per_core, mem_config}, {input_tensor_a}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
