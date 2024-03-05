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

using range_t = std::array<int32_t, 2>;
const int32_t NEIGHBORHOOD_DIST = 2;    // => ncores to left and ncores to right

namespace untilize_with_halo_helpers {

range_t calculate_in_range(const range_t& out_range, const PoolConfig& pc) {
    // given out stick range, calculate corresponding window's center stick input coords
    range_t in_range;
    // start of the range
    {
        uint32_t out_w_i = out_range[0] % pc.out_w;
        uint32_t out_h_i = out_range[0] / pc.out_w;
        uint32_t in_w_i = out_w_i * pc.stride_w;
        uint32_t in_h_i = out_h_i * pc.stride_h;
        in_range[0] = in_h_i * pc.in_w + in_w_i;
    }
    // end of the range
    {
        uint32_t out_w_i = out_range[1] % pc.out_w;
        uint32_t out_h_i = out_range[1] / pc.out_w;
        // corresponding window's center stick input coords:
        uint32_t in_w_i = out_w_i * pc.stride_w;
        uint32_t in_h_i = out_h_i * pc.stride_h;
        in_range[1] = in_h_i * pc.in_w + in_w_i;
    }
    return in_range;
}

std::map<CoreCoord, CoreCoord> left_neighbor_core, right_neighbor_core;
void init_neighbor_core_xy_mapping(CoreCoord grid_size, bool is_twod = false) {
    TT_ASSERT(grid_size.x == 12 && grid_size.y == 9);   // grayskull
    if (is_twod) {
        // 2d decomposition case (block sharded)
        // left-right neighbors are calculated along the x dim
        // first the left neighbors (x = 0 has no left neighbor)
        for (int32_t x = 1; x < grid_size.x; ++ x) {
            int32_t left_x = x - 1;
            for (int32_t y = 0; y < grid_size.y; ++ y) {
                CoreCoord core = {(uint32_t) x, (uint32_t) y};
                left_neighbor_core[core] = {(uint32_t) left_x, (uint32_t) y};
            }
        }
        // then the neighbors (x = grid_size.x - 1 has no left neighbor)
        for (int32_t x = 0; x < grid_size.x - 1; ++ x) {
            int32_t right_x = x + 1;
            for (int32_t y = 0; y < grid_size.y; ++ y) {
                CoreCoord core = {(uint32_t) x, (uint32_t) y};
                right_neighbor_core[core] = {(uint32_t) right_x, (uint32_t) y};
            }
        }
    } else {
        // default 1d distribution case (height sharded)
        for (int32_t y = 0; y < grid_size.y; ++ y) {
            for (int32_t x = 0; x < grid_size.x; ++ x) {
                CoreCoord core = {(uint32_t) x, (uint32_t) y};
                // calculate left neighbor
                int32_t left_x = x - 1, left_y = y;
                if (left_x < 0) {
                    left_x = grid_size.x - 1;
                    left_y -= 1;
                }
                if (left_y < 0) {
                    // there is no left neighbor
                } else {
                    left_neighbor_core[core] = {(uint32_t) left_x, (uint32_t) left_y};
                }
                // calculate right neighbor
                int32_t right_x = x + 1, right_y = y;
                if (right_x == grid_size.x) {
                    right_x = 0;
                    right_y += 1;
                }
                if (right_y == grid_size.y) {
                    // there is no right neighbor
                } else {
                    right_neighbor_core[core] = {(uint32_t) right_x, (uint32_t) right_y};
                }
            }
        }
    }
}

} // namespace untilize_with_halo_helpers

// The case of stride = 2
operation::ProgramWithCallbacks untilize_with_halo_multi_core_s2(const Tensor& input, Tensor& output, uint32_t pad_val, uint32_t in_b, uint32_t in_h, uint32_t in_w, uint32_t max_out_nsticks_per_core, const PoolConfig& pc) {
    Program program = CreateProgram();

    Device *device = input.device();
    Buffer *src_buffer = input.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    Shape input_shape = input.get_legacy_shape();
    Shape output_shape = output.get_legacy_shape();

    DataFormat in_df = datatype_to_dataformat_converter(input.get_dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t out_nbytes = datum_size(out_df);

    uint32_t in_tile_size = tt_metal::detail::TileSize(in_df);
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);

    uint32_t ntiles = input.volume() / TILE_HW;
    uint32_t ntiles_per_block = input_shape[3] / TILE_WIDTH;
    uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = input_shape[3] * output.element_size();

    // TODO: hard coded for testing only. need to pass these args in.
    // These are the input values before inserting and padding or halo
    TT_ASSERT(in_h * in_w == input_shape[2] || in_b * in_h * in_w == input_shape[2]);
    uint32_t in_c = input_shape[3];

    if (1) {
        log_debug(LogOp, "ntiles: {}", ntiles);
        log_debug(LogOp, "ntiles_per_block: {}", ntiles_per_block);
        log_debug(LogOp, "nblocks: {}", nblocks);
        log_debug(LogOp, "in_b: {}", in_b);
        log_debug(LogOp, "in_h: {}", pc.in_h);
        log_debug(LogOp, "in_w: {}", pc.in_w);
        log_debug(LogOp, "in_c: {}", in_c);
        log_debug(LogOp, "pad_h: {}", pc.pad_h);
        log_debug(LogOp, "pad_w: {}", pc.pad_w);
        log_debug(LogOp, "window_h: {}", pc.window_h);
        log_debug(LogOp, "window_w: {}", pc.window_w);
        log_debug(LogOp, "stride_h: {}", pc.stride_h);
        log_debug(LogOp, "stride_w: {}", pc.stride_w);
    }

    auto grid_size = device->compute_with_storage_grid_size();

    untilize_with_halo_helpers::init_neighbor_core_xy_mapping(grid_size);

    int32_t ncores_x = grid_size.x;     // distributing data to cores row-wise
    CoreRangeSet all_cores = input.shard_spec().value().grid;
    int32_t ncores = 0;
    for (const auto& core_range : all_cores.ranges()) {
        ncores += core_range.size();
    }
    CoreRangeSet core_range_cliff = CoreRangeSet({});
    uint32_t nblocks_per_core = input.shard_spec().value().shape[0] / TILE_HEIGHT;
    uint32_t nblocks_per_core_cliff = 0;

    uint32_t in_hw = pc.in_h * pc.in_w;
    uint32_t in_nhw = in_b * in_hw;
    uint32_t in_stick_nbytes = in_c * out_nbytes;
    uint32_t in_nsticks = in_nhw;
    uint32_t in_nsticks_per_batch = in_hw;
    uint32_t in_nsticks_per_core = in_nhw / ncores;

    int32_t halo_in_nsticks = (pc.in_w + (pc.window_w / 2)) * (pc.window_h / 2);     // input sticks to the writer
    int32_t halo_out_nsticks = (pc.in_w + 2 * pc.pad_w) * pc.pad_h + pc.window_w / 2;   // output sticks from the writer

    if (1) {
        log_debug(LogOp, "shard_shape: {},{}", input.shard_spec().value().shape[0], input.shard_spec().value().shape[1]);
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "ncores_x: {}", ncores_x);
        log_debug(LogOp, "nblocks_per_core: {}", nblocks_per_core);
        log_debug(LogOp, "nblocks_per_core_cliff: {}", nblocks_per_core_cliff);
        log_debug(LogOp, "in_hw: {}", in_hw);
        log_debug(LogOp, "in_nhw: {}", in_nhw);
        log_debug(LogOp, "in_stick_nbytes: {}", in_stick_nbytes);
        log_debug(LogOp, "in_nsticks_per_batch: {}", in_nsticks_per_batch);
        log_debug(LogOp, "in_nsticks_per_core: {}", in_nsticks_per_core);
        log_debug(LogOp, "halo_in_nsticks: {}", halo_in_nsticks);
        log_debug(LogOp, "halo_out_nsticks: {}", halo_out_nsticks);
        log_debug(LogOp, "max nsticks across all cores = {}", max_out_nsticks_per_core);
    }

    // For each core, calculate the desired resharding with halo and pad inserted in order to
    // to obtain resulting **output after the pooling/downsampling op to be equally distributed across all cores**.
    // The resulting shards with halo and pad could be different across cores.
    // The resharding is represented as a map:
    // map :: core -> [l_halo_start, l_halo_end) [local_start, local_end) [r_halo_start, r_halo_end)
    // (these are global input stick indices, [0, (in_nhw - 1)])
    // (and l_halo_end == local_start, local_end == r_halo_start)
    // calculate this core's [left halo, local owned, right halo]. These halo data could be "anywhere".
    std::map<uint32_t, std::array<range_t, 3>> my_shard;
    int32_t out_stick_start = 0;   // global "output" stick (after downsample/pool)
    int32_t pool_out_nsticks_per_core = in_b * pc.out_h * pc.out_w / ncores;
    for (uint32_t core = 0; core < ncores; ++ core) {
        range_t out_range = {out_stick_start, out_stick_start + pool_out_nsticks_per_core};
        range_t in_range = untilize_with_halo_helpers::calculate_in_range(out_range, pc);  // this represents the "window" center input sticks
        int32_t l_halo_start = in_range[0] - halo_in_nsticks;
        int32_t batch_start = (in_range[0] / (in_h * in_w)) * (in_h * in_w);
        l_halo_start = l_halo_start < batch_start ? batch_start : l_halo_start;
        int32_t r_halo_end = in_range[1] + halo_in_nsticks;
        r_halo_end = r_halo_end >= in_nhw ? in_nhw : r_halo_end;
        my_shard[core] = {{
            { l_halo_start, in_range[0] }, // l_halo
            { in_range[0], in_range[1] },                   // local
            { in_range[1], r_halo_end }  // r_halo
        }};
        out_stick_start += pool_out_nsticks_per_core;
    }

    // CBs

    uint32_t src_cb_id = CB::c_in0;
    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    auto src_cb_config = CircularBufferConfig(num_input_tiles * in_tile_size, {{src_cb_id, in_df}})
                            .set_page_size(src_cb_id, in_tile_size)
                            .set_globally_allocated_address(*input.buffer());
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);

    // output of untilize from compute kernel goes into this CB
    uint32_t untilize_out_cb_id = CB::c_out0;
    uint32_t num_output_tiles = ntiles_per_block * nblocks_per_core;
    auto untilize_out_cb_config = CircularBufferConfig(num_output_tiles * out_tile_size, {{untilize_out_cb_id, out_df}})
                                    .set_page_size(untilize_out_cb_id, out_tile_size);
    auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);

    // output after concatenating halo and padding goes into this CB, as input to next op.
    uint32_t out_cb_id = CB::c_out1;
    uint32_t out_cb_pagesize = out_nbytes * in_c;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    auto out_cb_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
                            .set_page_size(out_cb_id, out_cb_pagesize)
                            .set_globally_allocated_address(*output.buffer());
    auto out_cb = CreateCircularBuffer(program, all_cores, out_cb_config);

    // CB for pad val buffer (stick sized)
    uint32_t pad_cb_id = CB::c_in1;
    uint32_t pad_cb_pagesize = in_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    auto pad_cb_config = CircularBufferConfig(pad_cb_pagesize * pad_cb_npages, {{pad_cb_id, out_df}})
                            .set_page_size(pad_cb_id, pad_cb_pagesize);
    auto pad_cb = CreateCircularBuffer(program, all_cores, pad_cb_config);

    if (0) {
        log_debug(LogOp, "src cb: id = {}, pagesize = {}, npages = {}", src_cb_id, in_tile_size, num_input_tiles);
        log_debug(LogOp, "untilize cb: id = {}, pagesize = {}, npages = {}", untilize_out_cb_id, out_tile_size, num_output_tiles);
        log_debug(LogOp, "out cb: id = {}, pagesize = {}, npages = {}", out_cb_id, out_cb_pagesize, out_cb_npages);
    }

    /** reader
     */

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t) src_cb_id
    };

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        ReaderDataMovementConfig{reader_ct_args});

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t) untilize_out_cb_id,
        (std::uint32_t) out_cb_id,
        (std::uint32_t) pad_cb_id,
        (std::uint32_t) pad_val,
        (std::uint32_t) in_c,    // stick len
        (std::uint32_t) in_stick_nbytes,    // bytes per stick (in RM, after untilize)
        (std::uint32_t) pc.in_w,
        (std::uint32_t) pc.in_h,
        (std::uint32_t) pc.pad_w,
        (std::uint32_t) pc.pad_h,
    };
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/writer_unary_sharded_with_halo_s2.cpp",
        all_cores,
        WriterDataMovementConfig{writer_ct_args});

    /** compute
     */
    TT_ASSERT(core_range_cliff.ranges().size() == 0);
    vector<uint32_t> compute_args = {
        (uint32_t) nblocks_per_core,    // per_core_block_cnt
        (uint32_t) ntiles_per_block,    // per_block_ntiles
    };
    KernelHandle untilize_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp",
        all_cores,
        ComputeConfig{.compile_args=compute_args});

    // 1D distribution of blocks across all cores
    uint32_t ncores_full = ncores;
    // cliff core not yet supported
    TT_ASSERT(nblocks_per_core_cliff == 0);

    // reader runtime args
    vector<uint32_t> reader_rt_args = {
        ntiles_per_block * nblocks_per_core // ntiles
    };

    TT_ASSERT(in_nhw % ncores == 0);

    // writer inserts halo and padding where needed
    vector<uint32_t> writer_rt_args = {
        ntiles_per_block * nblocks_per_core,  // in_nsticks,                     // 0
        in_nsticks_per_core,  // UNUSED
        0,  // partial_first_row_nsticks,
        pc.pad_w,
        pc.in_w,
        0,  // partial_top_image_nrows,        // 5
        pc.pad_h,
        pc.in_h,
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
        0,  // UNUSED const_tensor_addr,
        // sharding config
        0,
        0,
        0,
        0,                                     // 50
        0,
        0,
        0,
        0,
        0,                                     // 55
        0,
        0,
        0,
        0,
        // halo config
        0,                                     // 60
        0,
        0,
        0,
        0,
        0,                                     // 65
        0,
        0,
        0,
        0,
        0,                                     // 70
        0,
    };

    uint32_t writer_noc = 0;

    TT_ASSERT(pc.window_h == 3 && pc.window_w == 3);

    // Calculate data shuffle config for each core using its shard size:
    // Given equally distributed input across all cores,
    // for each core, identify the data to be scattered across its neighborhood:
    // construct the map (subset of alltoallv primitive)
    // map :: core -> [ll_start, ll_end] [l_start, l_end] [start, end] [r_start, r_end] [rr_start, rr_end]
    // where, ll_end == l_start, l_end == start, end == r_start, r_end == rr_start
    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> count_to_send;    // nsticks to push to each neighbor
    std::map<uint32_t, std::array<range_t, (NEIGHBORHOOD_DIST * 2 + 1)>> range_to_send;     // range to push to each neighbor
    uint32_t in_stick_start = 0;    // global input stick
    for (uint32_t core = 0; core < ncores; ++ core) {
        uint32_t in_stick_end = in_stick_start + in_nsticks_per_core;
        // calculate the shuffle config [excl. padding] <-- TODO: see if padding should be handled here or later
        //   - for each core in the neighborhood:
        //         calculate range to push to this core

        // first, cores on the left neighborhood (starting left->right):
        uint32_t count = 0;
        int32_t range_start = 0, range_end = 0;
        for (int32_t l_neighbor = NEIGHBORHOOD_DIST; l_neighbor > 0; -- l_neighbor) {
            int32_t l_core = core - l_neighbor;
            if (l_core >= 0) {  // this is a valid neighbor
                count = 0;
                // see if this neighbor needs anything from me
                // neighbor's right halo end > in_stick_start
                uint32_t stick = in_stick_start;
                range_start = stick;
                while (stick < my_shard[l_core][2][1]) {
                    ++ count;
                    ++ stick;
                }
                range_end = stick;
                count_to_send[core][NEIGHBORHOOD_DIST - l_neighbor] = count;
                range_to_send[core][NEIGHBORHOOD_DIST - l_neighbor] = { range_start, range_end };
            }
        }

        // check if there is data to keep local:
        // that is, if my full new shard range (incl. halos) intersects with my current shard
        count = 0;
        range_start = 0, range_end = 0;
        int32_t my_shard_start = my_shard[core][0][0];
        int32_t my_shard_end = my_shard[core][2][1];
        if (my_shard_end > in_stick_start && my_shard_start < in_stick_end) {
            uint32_t stick = in_stick_start < my_shard_start ? my_shard_start : in_stick_start;
            range_start = stick;
            while (stick < my_shard_end && stick < in_stick_end) {
                ++ count;
                ++ stick;
            }
            range_end = stick;
        }
        count_to_send[core][NEIGHBORHOOD_DIST] = count; // keep local
        range_to_send[core][NEIGHBORHOOD_DIST] = { range_start, range_end }; // keep local

        // cores on the right neighborhood (starting left->right)
        range_start = 0, range_end = 0;
        for (int32_t r_neighbor = 1; r_neighbor <= NEIGHBORHOOD_DIST; ++ r_neighbor) {
            int32_t r_core = core + r_neighbor;
            if (r_core < ncores) {  // this is a valid neighbor
                count = 0;
                // see if this neighbor needs anything from me
                // neighbor's left halo start < in_stick_end
                uint32_t stick = my_shard[r_core][0][0];
                stick = stick < in_stick_start ? in_stick_start : stick;
                range_start = stick;
                while (stick < in_stick_end) {
                    ++ count;
                    ++ stick;
                }
                range_end = stick;
                count_to_send[core][NEIGHBORHOOD_DIST + r_neighbor] = count;
                range_to_send[core][NEIGHBORHOOD_DIST + r_neighbor] = { range_start, range_end };
            }
        }

        in_stick_start += in_nsticks_per_core;
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> count_to_receive; // nsticks pushed from each neighbor
    for (uint32_t core = 0; core < ncores; ++ core) {
        // calculate the nsticks I receive from each of my neighbors
        for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            int32_t neighbor_core = core + neighbor;
            uint32_t count = 0;
            if (neighbor_core >= 0 && neighbor_core < ncores) {
                count = count_to_send[neighbor_core][NEIGHBORHOOD_DIST - neighbor];
            }
            count_to_receive[core][NEIGHBORHOOD_DIST + neighbor] = count;
        }
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> updated_count_to_send;    // nsticks to push to each neighbor
    for (uint32_t core = 0; core < ncores; ++ core) {
        // for each core, calculate the offsets where data is to be pushed to
        // the push will be from locally constructed re-sharded data, so no additional need to insert padding when pushing.
        for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            // calculate the total nsticks incl. padding to be sent to this neighbor
            range_t to_send = range_to_send[core][NEIGHBORHOOD_DIST + neighbor];
            NewShardingConfig sc = get_shard_specs(to_send[0], to_send[1], pc);
            uint32_t count = 0;
            count += sc.first_partial_right_aligned_row_width + sc.skip_after_partial_right_aligned_row;
            count += sc.first_partial_image_num_rows * (pc.in_w + 2 * pc.pad_w) + sc.skip_after_first_partial_image_row;
            count += sc.num_full_images * (pc.in_h * (pc.in_w + 2 * pc.pad_w) + sc.skip_after_full_image);
            count += sc.last_partial_image_num_rows * (pc.in_w + 2 * pc.pad_w);
            count += sc.last_partial_left_aligned_row_width;
            updated_count_to_send[core][NEIGHBORHOOD_DIST + neighbor] = count;
        }
        in_stick_start += in_nsticks_per_core;
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> updated_count_to_receive; // nsticks pushed from each neighbor
    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> receive_at_offset_nsticks;  // data is to be received at these offsets
    std::map<uint32_t, int32_t> initial_pad_nsticks;  // any padding to be inserted at the beginning (for left most cores)
    for (uint32_t core = 0; core < ncores; ++ core) {
        // record any inital skip to be used in offset calculations
        initial_pad_nsticks[core] = 0;
        if (my_shard[core][0][1] - my_shard[core][0][0] < halo_in_nsticks) {
            initial_pad_nsticks[core] = halo_out_nsticks - (my_shard[core][0][1] - my_shard[core][0][0]);
        }
        uint32_t cumulative_count = initial_pad_nsticks[core];
        // calculate the nsticks I receive from each of my neighbors
        for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            int32_t neighbor_core = core + neighbor;
            uint32_t count = 0;
            if (neighbor_core >= 0 && neighbor_core < ncores) {
                count = updated_count_to_send[neighbor_core][NEIGHBORHOOD_DIST - neighbor];
            }
            updated_count_to_receive[core][NEIGHBORHOOD_DIST + neighbor] = count;
            receive_at_offset_nsticks[core][NEIGHBORHOOD_DIST + neighbor] = cumulative_count;
            cumulative_count += count;
        }
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> send_to_offset_nsticks;  // data is to be received at these offsets
    for (uint32_t core = 0; core < ncores; ++ core) {
        for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            int32_t neighbor_core = core + neighbor;
            uint32_t offset = 0;
            if (neighbor_core >= 0 && neighbor_core < ncores) {
                offset = receive_at_offset_nsticks[neighbor_core][NEIGHBORHOOD_DIST - neighbor];
            }
            send_to_offset_nsticks[core][NEIGHBORHOOD_DIST + neighbor] = offset;
        }
    }

    std::map<uint32_t, std::array<uint32_t, (NEIGHBORHOOD_DIST * 2 + 1)>> send_from_offset_nsticks; // after adding pad etc, the offset to start send data from
    in_stick_start = 0;
    for (uint32_t core = 0; core < ncores; ++ core) {
        for (int32_t neighbor = - NEIGHBORHOOD_DIST; neighbor <= NEIGHBORHOOD_DIST; ++ neighbor) {
            bool toprint = core == 1 && neighbor == 2;
            int32_t neighbor_core = core + neighbor;
            uint32_t offset = 0;
            if (neighbor_core >= 0 && neighbor_core < ncores) {
                NewShardingConfig sc = get_shard_specs(in_stick_start, range_to_send[core][NEIGHBORHOOD_DIST + neighbor][0], pc);
                offset += sc.first_partial_right_aligned_row_width + sc.skip_after_partial_right_aligned_row;
                offset += sc.first_partial_image_num_rows * (pc.in_w + 2 * pc.pad_w) + sc.skip_after_first_partial_image_row;
                offset += sc.num_full_images * (pc.in_h * (pc.in_w + 2 * pc.pad_w) + sc.skip_after_full_image);
                offset += sc.last_partial_image_num_rows * (pc.in_w + 2 * pc.pad_w);
                offset += sc.last_partial_left_aligned_row_width;
            }
            send_from_offset_nsticks[core][NEIGHBORHOOD_DIST + neighbor] = offset + receive_at_offset_nsticks[core][NEIGHBORHOOD_DIST];
        }
        in_stick_start += in_nsticks_per_core;
    }

    // Calculate all information needed to copy locally owned data to output shard.
    in_stick_start = 0;
    for (uint32_t core = 0; core < ncores; ++ core) {
        CoreCoord core_coord = {core % ncores_x, core / ncores_x};  // logical

        // set reader rt args
        SetRuntimeArgs(program, reader_kernel_id, core_coord, reader_rt_args);

        // for each core, identify its sections with padding for the data it locally owns
        // that is, compute the sharding config
        uint32_t in_stick_end = in_stick_start + in_nsticks_per_core;

        // left neighbor args
        if (untilize_with_halo_helpers::left_neighbor_core.count(core_coord) > 0) {
            CoreCoord left_core = untilize_with_halo_helpers::left_neighbor_core.at(core_coord);
            CoreCoord left_noc = device->worker_core_from_logical_core(left_core);
            writer_rt_args[19] = 1;
            writer_rt_args[20] = left_noc.x;
            writer_rt_args[21] = left_noc.y;
            if (untilize_with_halo_helpers::left_neighbor_core.count(left_core) > 0) {
                CoreCoord left_left_core = untilize_with_halo_helpers::left_neighbor_core.at(left_core);
                CoreCoord left_left_noc = device->worker_core_from_logical_core(left_left_core);
                writer_rt_args[25] = 1;
                writer_rt_args[26] = left_left_noc.x;
                writer_rt_args[27] = left_left_noc.y;
            } else {
                // no left-left neighbor
                writer_rt_args[25] = 0;
            }
        } else {
            // no left neighbors
            writer_rt_args[19] = 0;
            writer_rt_args[25] = 0;
        }
        // right neighbor args
        if (untilize_with_halo_helpers::right_neighbor_core.count(core_coord) > 0) {
            CoreCoord right_core = untilize_with_halo_helpers::right_neighbor_core.at(core_coord);
            CoreCoord right_noc = device->worker_core_from_logical_core(right_core);
            writer_rt_args[22] = 1;
            writer_rt_args[23] = right_noc.x;
            writer_rt_args[24] = right_noc.y;
            if (untilize_with_halo_helpers::right_neighbor_core.count(right_core) > 0) {
                CoreCoord right_right_core = untilize_with_halo_helpers::right_neighbor_core.at(right_core);
                CoreCoord right_right_noc = device->worker_core_from_logical_core(right_right_core);
                writer_rt_args[28] = 1;
                writer_rt_args[29] = right_right_noc.x;
                writer_rt_args[30] = right_right_noc.y;
            } else {
                // no right-right neighbor
                writer_rt_args[28] = 0;
            }
        } else {
            // no right neighbors
            writer_rt_args[22] = 0;
            writer_rt_args[28] = 0;
        }

        // logically insert padding into locally owned data and generate the sharding config
        // information to calculate for a core:
        //  + partial_first_row_nsticks (no pad)
        //  + skip_after_partial_first_row (pad)
        //  + partial_first_image_nrows (no pad) + padding per row (pad)
        //  + skip_after_partial_first_image (pad)
        //  + full_nimages (no pad) + padding per row (pad) + padding per image (pad)
        //  + skip_after_full_images (pad)
        //  + partial_last_image_nrows (no pad) + padding per row (pad)
        //  + partial_last_row_nsticks (no pad)
        NewShardingConfig sc = get_shard_specs(in_stick_start, in_stick_end, pc);
        int32_t partial_first_row_nsticks = sc.first_partial_right_aligned_row_width;
        int32_t skip_after_partial_first_row = sc.skip_after_partial_right_aligned_row;
        int32_t partial_first_image_nrows = sc.first_partial_image_num_rows;
        int32_t skip_after_partial_first_image = sc.skip_after_first_partial_image_row;
        int32_t full_nimages = sc.num_full_images;
        int32_t skip_after_full_images = sc.skip_after_full_image;
        int32_t partial_last_image_nrows = sc.last_partial_image_num_rows;
        int32_t partial_last_row_nsticks = sc.last_partial_left_aligned_row_width;
        int32_t initial_skip = sc.initial_skip;

        // args for handling local data movement:
        //     intial_pad_nsticks
        //     receive_at_offset_nsticks[core][NEIGHBORHOOD_DIST]; // local_offset_nsticks
        //     partial_first_row_nsticks
        //     partial_first_row_skip
        //     partial_top_image_nrows, partial_top_image_skip_per_row
        //     partial_top_image_skip
        //     full_nimages, full_image_skip_per_row
        //     full_image_skip
        //     partial_bottom_image_nrows, partial_bottom_image_skip_per_row
        //     partial_last_row_nsticks
        writer_rt_args[47] = initial_pad_nsticks[core];
        writer_rt_args[48] = receive_at_offset_nsticks[core][NEIGHBORHOOD_DIST]; // local_offset_nsticks
        uint32_t partial_first_row_nbytes = partial_first_row_nsticks * in_stick_nbytes;
        writer_rt_args[49] = partial_first_row_nbytes;
        writer_rt_args[50] = skip_after_partial_first_row;
        writer_rt_args[51] = partial_first_image_nrows;
        writer_rt_args[52] = partial_first_image_nrows > 0 ? 2 * pc.pad_w : 0;                          // partial_top_image_skip_per_row
        writer_rt_args[53] = partial_first_image_nrows > 0 ? pc.pad_h * (pc.in_w + 2 * pc.pad_w) : 0;   // skip_after_partial_first_image
        writer_rt_args[54] = full_nimages;
        writer_rt_args[55] = full_nimages > 0 ? 2 * pc.pad_w : 0;                          // full_nimage_skip_per_row
        writer_rt_args[56] = full_nimages > 0 ? pc.pad_h * (pc.in_w + 2 * pc.pad_w) : 0;   // full_image_skip
        writer_rt_args[57] = partial_last_image_nrows;
        writer_rt_args[58] = partial_last_image_nrows > 0 ? 2 * pc.pad_w : 0;                          // partial_bottom_image_skip_per_row
        uint32_t partial_last_row_nbytes = partial_last_row_nsticks * in_stick_nbytes;
        writer_rt_args[59] = partial_last_row_nbytes;

        // args for handling remote data shuffle:
        //     NEIGHBORHOOD_DIST
        //     ll_send_count
        //     ll_send_from_offset
        //     ll_send_at_offset
        //     l_send_count
        //     l_send_from_offset
        //     l_send_at_offset
        //     r_send_count
        //     r_send_from_offset
        //     r_send_at_offset
        //     rr_send_count
        //     rr_send_from_offset
        //     rr_send_at_offset

        // LL
        writer_rt_args[60] = updated_count_to_send[core][0] * in_stick_nbytes;
        writer_rt_args[61] = send_from_offset_nsticks[core][0] * in_stick_nbytes;
        writer_rt_args[62] = send_to_offset_nsticks[core][0] * in_stick_nbytes;
        // L
        writer_rt_args[63] = updated_count_to_send[core][1] * in_stick_nbytes;
        writer_rt_args[64] = send_from_offset_nsticks[core][1] * in_stick_nbytes;
        writer_rt_args[65] = send_to_offset_nsticks[core][1] * in_stick_nbytes;
        // R
        writer_rt_args[66] = updated_count_to_send[core][3] * in_stick_nbytes;
        writer_rt_args[67] = send_from_offset_nsticks[core][3] * in_stick_nbytes;
        writer_rt_args[68] = send_to_offset_nsticks[core][3] * in_stick_nbytes;
        // RR
        writer_rt_args[69] = updated_count_to_send[core][4] * in_stick_nbytes;
        writer_rt_args[70] = send_from_offset_nsticks[core][4] * in_stick_nbytes;
        writer_rt_args[71] = send_to_offset_nsticks[core][4] * in_stick_nbytes;

        SetRuntimeArgs(program, writer_kernel_id, core_coord, writer_rt_args);

        in_stick_start += in_nsticks_per_core;
    }

    // print stuff for debug
    if (0) {
        in_stick_start = 0;
        for (uint32_t core = 0; core < ncores; ++ core) {
            uint32_t in_stick_end = in_stick_start + in_nsticks_per_core;
            log_debug("==== Core {}:", core);
            log_debug(" in shard = [{},{})", in_stick_start, in_stick_end);
            log_debug(" re shard = [{},{}) [{},{}) [{},{})", my_shard[core][0][0], my_shard[core][0][1],
                                                            my_shard[core][1][0], my_shard[core][1][1],
                                                            my_shard[core][2][0], my_shard[core][2][1]);
            for (uint32_t neighbor = 0; neighbor < 2 * NEIGHBORHOOD_DIST + 1; ++ neighbor) {
                log_debug(" + N {} :: recv: (count = {}, offset = {})\t send: (count = {}, from = {}, to = {}) : [{},{})", neighbor,
                                                                    updated_count_to_receive[core][neighbor],
                                                                    receive_at_offset_nsticks[core][neighbor],
                                                                    updated_count_to_send[core][neighbor],
                                                                    send_from_offset_nsticks[core][neighbor],
                                                                    send_to_offset_nsticks[core][neighbor],
                                                                    range_to_send[core][neighbor][0], range_to_send[core][neighbor][1]);
            }
            in_stick_start += in_nsticks_per_core;
        }
    }


    auto override_runtime_arguments_callback = [
        reader_kernel_id=reader_kernel_id,
        writer_kernel_id=writer_kernel_id,
        src_cb=src_cb,
        out_cb=out_cb
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);

        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

// The case with stride = 1
operation::ProgramWithCallbacks untilize_with_halo_multi_core_s1(const Tensor& a, Tensor& output, uint32_t pad_val, const uint32_t &in_b, const uint32_t &in_h, const uint32_t &in_w, const uint32_t &max_out_nsticks_per_core) {
    Program program = CreateProgram();

    Device *device = a.device();
    Buffer *src_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    Shape input_shape = a.get_legacy_shape();
    Shape output_shape = output.get_legacy_shape();

    DataFormat in_df = datatype_to_dataformat_converter(a.get_dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t out_nbytes = datum_size(out_df);

    uint32_t in_tile_size = tt_metal::detail::TileSize(in_df);
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);

    auto grid_size = device->compute_with_storage_grid_size();
    untilize_with_halo_helpers::init_neighbor_core_xy_mapping(grid_size, a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

    int32_t ncores_x = grid_size.x;
    int32_t ncores_y = grid_size.y;
    CoreRangeSet all_cores = a.shard_spec().value().grid;
    int32_t ncores = all_cores.num_cores();
    int32_t ncores_col = 1;
    if (a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(all_cores.ranges().begin());
        ncores = core_range.end.x - core_range.start.x + 1;
        ncores_col = core_range.end.y - core_range.start.y + 1;
    }

    CoreRangeSet core_range_cliff = CoreRangeSet({});

    uint32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = input_shape[3] / TILE_WIDTH;
    uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = input_shape[3] * output.element_size();

    auto shard_shape = a.shard_spec().value().shape;

    if (a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        ntiles = input_shape[0] * input_shape[1] * shard_shape[0] * shard_shape[1];
        ntiles_per_block = shard_shape[1] / TILE_WIDTH;
        nblocks = ceil((float) ntiles / ntiles_per_block);
        block_size_nbytes = shard_shape[1] * output.element_size();
        auto core_range = *(all_cores.ranges().begin());
        int32_t ncores_y = core_range.end.y - core_range.start.y + 1;
        TT_ASSERT(a.shard_spec().value().shape[1] * ncores_y == input_shape[3], "Input shape in W should be same as shard width * num cores along each row!");
    } else {
        TT_ASSERT(a.shard_spec().value().shape[1] == input_shape[3], "Input shape in W should be same as shard width!");
    }

    // TODO: hard coded for now. need to pass these args in.
    uint32_t nbatch = in_b;
    uint32_t in_c = shard_shape[1]; // input_shape[3];  // this is per core row
    uint32_t pad_h = 1;
    uint32_t pad_w = 1;
    uint32_t window_h = 3;
    uint32_t window_w = 3;

    uint32_t nblocks_per_core = shard_shape[0] / TILE_HEIGHT;
    uint32_t nblocks_per_core_cliff = 0;

    uint32_t in_hw = in_h * in_w;
    uint32_t in_nhw = nbatch * in_hw;
    uint32_t in_stick_nbytes = in_c * out_nbytes;
    uint32_t in_nsticks = in_nhw;
    if (a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        in_nsticks = shard_shape[0] * ncores;
    }
    uint32_t in_nsticks_per_batch = in_hw;
    uint32_t in_nsticks_per_core = in_nsticks / ncores;

    if (1) {
        log_debug(LogOp, "ntiles: {}", ntiles);
        log_debug(LogOp, "ntiles_per_block: {}", ntiles_per_block);
        log_debug(LogOp, "nblocks: {}", nblocks);
        log_debug(LogOp, "nbatch: {}", nbatch);
        log_debug(LogOp, "in_h: {}", in_h);
        log_debug(LogOp, "in_w: {}", in_w);
        log_debug(LogOp, "in_c: {}", in_c);
        log_debug(LogOp, "pad_h: {}", pad_h);
        log_debug(LogOp, "pad_w: {}", pad_w);
        log_debug(LogOp, "window_h: {}", window_h);
        log_debug(LogOp, "window_w: {}", window_w);
        log_debug(LogOp, "shard_shape: {},{}", a.shard_spec().value().shape[0], a.shard_spec().value().shape[1]);
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "ncores_col: {}", ncores_col);
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
    auto src_cb_config = CircularBufferConfig(num_input_tiles * in_tile_size, {{src_cb_id, in_df}})
                            .set_page_size(src_cb_id, in_tile_size)
                            .set_globally_allocated_address(*a.buffer());
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);

    // output of untilize from compute kernel goes into this CB
    uint32_t untilize_out_cb_id = CB::c_out0;
    uint32_t num_output_tiles = ntiles_per_block * nblocks_per_core;
    auto untilize_out_cb_config = CircularBufferConfig(num_output_tiles * out_tile_size, {{untilize_out_cb_id, out_df}})
                                    .set_page_size(untilize_out_cb_id, out_tile_size);
    auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);

    // output after concatenating halo and padding goes into this CB, as input to next op.
    uint32_t out_cb_id = CB::c_out1;
    uint32_t out_cb_pagesize = out_nbytes * in_c;
    uint32_t out_nsticks_per_core = max_out_nsticks_per_core;
    auto out_cb_config = CircularBufferConfig(max_out_nsticks_per_core * out_cb_pagesize, {{out_cb_id, out_df}})
                            .set_page_size(out_cb_id, out_cb_pagesize)
                            .set_globally_allocated_address(*output.buffer());
    auto out_cb = CreateCircularBuffer(program, all_cores, out_cb_config);

    // CB for pad val buffer (stick sized)
    uint32_t pad_cb_id = CB::c_in1;
    uint32_t pad_cb_pagesize = in_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    auto pad_cb_config = CircularBufferConfig(pad_cb_pagesize * pad_cb_npages, {{pad_cb_id, out_df}})
                            .set_page_size(pad_cb_id, pad_cb_pagesize);
    auto pad_cb = CreateCircularBuffer(program, all_cores, pad_cb_config);

    if (0) {
        log_debug(LogOp, "out_cb_pagesize: {}", out_cb_pagesize);
        log_debug(LogOp, "out_nsticks_per_core: {}", out_nsticks_per_core);
    }

    /** reader
     */

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t) src_cb_id
    };

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        ReaderDataMovementConfig{reader_ct_args});

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t) untilize_out_cb_id,
        (std::uint32_t) out_cb_id,
        (std::uint32_t) pad_cb_id,
        (std::uint32_t) pad_val,
        (std::uint32_t) in_c,    // stick len
        (std::uint32_t) in_stick_nbytes,    // bytes per stick (in RM, after untilize)
        (std::uint32_t) in_w,
        (std::uint32_t) in_h,
        (std::uint32_t) pad_w,
        (std::uint32_t) pad_h,
    };
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/writer_unary_sharded_with_halo.cpp",
        all_cores,
        WriterDataMovementConfig{writer_ct_args});

    /** compute
     */
    TT_ASSERT(core_range_cliff.ranges().size() == 0);
    vector<uint32_t> compute_args = {
        (uint32_t) nblocks_per_core,    // per_core_block_cnt
        (uint32_t) ntiles_per_block,    // per_block_ntiles
    };
    KernelHandle untilize_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp",
        all_cores,
        ComputeConfig{.compile_args=compute_args});

    // 1D distribution of blocks across all cores
    // cliff core not yet supported
    TT_ASSERT(nblocks_per_core_cliff == 0);

    // reader runtime args
    vector<uint32_t> reader_rt_args = {
        ntiles_per_block * nblocks_per_core // ntiles
    };

    if (a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_ASSERT(in_nhw % ncores == 0);
    }

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
        0,  // UNUSED
    };

    uint32_t writer_noc = 0;

    TT_ASSERT(window_h == 3 && window_w == 3);
    int32_t halo_in_nsticks = (in_w + (window_w / 2)) * (window_h / 2);
    int32_t halo_out_nsticks = (in_w + 2 * pad_w) * pad_h + window_w / 2;

    if (0) {
        log_debug(LogOp, "halo_in_nsticks: {}", halo_in_nsticks);
        log_debug(LogOp, "halo_out_nsticks: {}", halo_out_nsticks);
    }

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

    for (int32_t i = 0; i < ncores; ++ i) {
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

        uint32_t local_nsticks = partial_first_row_nsticks + partial_first_row_skip
                                    + partial_top_image_nrows * (in_w + 2 * pad_w) + partial_top_image_skip
                                    + full_nimages * (in_w + 2 * pad_w) * in_h
                                    + full_image_skip
                                    + partial_bottom_image_nrows * (in_w + 2 * pad_w)
                                    + partial_last_row_nsticks;

        // NOTE: this is always the same for all cores
        uint32_t halo_nsticks = (in_w + window_w / 2 + 2 * pad_w);  // left or right halo
        uint32_t out_nsticks_per_core = local_nsticks + 2 * halo_nsticks;

        my_right_halo_offset[i] = my_left_halo_offset[i] + (((initial_pad_nsticks > 0) ? initial_pad_nsticks : halo_nsticks) + local_nsticks) * in_stick_nbytes;
        if ((in_stick_start + in_nsticks_per_core) / in_w == (in_stick_start + in_nsticks_per_core + my_right_halo[i]) / in_w) {
            my_right_right_halo_offset[i] = my_right_halo_offset[i] + my_right_halo[i] * in_stick_nbytes;
        } else {
            my_right_right_halo_offset[i] = my_right_halo_offset[i] + (my_right_halo[i] + 2) * in_stick_nbytes;
        }
        my_right_halo_pad_i_offset[i] = in_w - ((in_stick_start + in_nsticks_per_core) % in_w);

        if (0) {
            log_debug(LogOp, "==== Core {}", i);
            log_debug(LogOp, "local_nsticks: {}", local_nsticks);
            log_debug(LogOp, "halo_nsticks: {}", halo_nsticks);
            log_debug(LogOp, "out_nsticks_per_core: {}", out_nsticks_per_core);
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
    for (uint32_t i = 0; i < ncores; ++ i) {
        // writer rt args
        writer_rt_args[15] = in_stick_start;    // unused
        writer_rt_args[16] = in_stick_start + in_nsticks_per_core;  // unused

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

        uint32_t partial_first_row_nbytes = partial_first_row_nsticks * in_stick_nbytes;
        writer_rt_args[2] = partial_first_row_nbytes;
        writer_rt_args[5] = partial_top_image_nrows;
        writer_rt_args[8] = full_nimages;
        writer_rt_args[9] = partial_bottom_image_nrows;

        uint32_t partial_last_row_nbytes = partial_last_row_nsticks * in_stick_nbytes;
        writer_rt_args[10] = partial_last_row_nbytes;
        writer_rt_args[42] = partial_first_row_skip;
        writer_rt_args[43] = partial_top_image_skip;
        writer_rt_args[44] = full_image_skip;

        for (uint32_t j = 0; j < ncores_col; ++ j) {
            CoreCoord core;

            if (a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                core = {i, j};
            } else {
                core = {i % ncores_x, i / ncores_x};
            }

            if (untilize_with_halo_helpers::left_neighbor_core.count(core) > 0) {
                CoreCoord left_core = untilize_with_halo_helpers::left_neighbor_core.at(core);
                CoreCoord left_noc = device->worker_core_from_logical_core(left_core);
                writer_rt_args[19] = 1;
                writer_rt_args[20] = left_noc.x;
                writer_rt_args[21] = left_noc.y;
                // my_right_halo[i - 1] == sticks to left neighbor core's right halo
                uint32_t left_core_nbytes = (my_right_halo[i - 1] + 2*pad_w)*in_stick_nbytes;
                writer_rt_args[33] = left_core_nbytes;
                writer_rt_args[37] = my_right_halo_offset[i - 1];
                if (untilize_with_halo_helpers::left_neighbor_core.count(left_core) > 0) {
                    CoreCoord left_left_core = untilize_with_halo_helpers::left_neighbor_core.at(left_core);
                    CoreCoord left_left_noc = device->worker_core_from_logical_core(left_left_core);
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
            if (untilize_with_halo_helpers::right_neighbor_core.count(core) > 0) {
                CoreCoord right_core = untilize_with_halo_helpers::right_neighbor_core.at(core);
                CoreCoord right_noc = device->worker_core_from_logical_core(right_core);
                writer_rt_args[22] = 1;
                writer_rt_args[23] = right_noc.x;
                writer_rt_args[24] = right_noc.y;
                // my_left_halo[i + 1]  == sticks to right neighbor core's left halo
                uint32_t right_core_nbytes = (my_left_halo[i + 1] + 2*pad_w)*in_stick_nbytes;
                writer_rt_args[34] = right_core_nbytes;
                writer_rt_args[38] = my_left_halo_offset[i + 1];
                if (untilize_with_halo_helpers::right_neighbor_core.count(right_core) > 0) {
                    CoreCoord right_right_core = untilize_with_halo_helpers::right_neighbor_core.at(right_core);
                    CoreCoord right_right_noc = device->worker_core_from_logical_core(right_right_core);
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

            // reader rt args
            SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);
            // writer rt args
            SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);
        }
        if (0) {
            log_debug(LogOp, "++++ Core: {}", i);
            log_debug(LogOp, "out_stick_start: {}", out_stick_start);
            log_debug(LogOp, "halo::has_left: {}", writer_rt_args[19]);
            log_debug(LogOp, "halo::has_right: {}", writer_rt_args[22]);
            log_debug(LogOp, "local_in_stick_start: {}", writer_rt_args[15]);
            log_debug(LogOp, "partial_first_row_nsticks: {}", writer_rt_args[2]);
            log_debug(LogOp, "partial_top_image_nrows: {}", writer_rt_args[5]);
            log_debug(LogOp, "full_nimages: {}", writer_rt_args[8]);
            log_debug(LogOp, "partial_bottom_image_nrows: {}", writer_rt_args[9]);
            log_debug(LogOp, "partial_last_row_nsticks: {}", writer_rt_args[10]);
            log_debug(LogOp, "skip_after_partial_right_aligned_row: {}", sc.skip_after_partial_right_aligned_row);
            log_debug(LogOp, "skip_after_first_partial_image_row: {}", sc.skip_after_first_partial_image_row);
            log_debug(LogOp, "skip_after_full_image: {}", sc.skip_after_full_image);
            log_debug(LogOp, "halo_for_left_nsticks: {}", writer_rt_args[12]);
            log_debug(LogOp, "halo_for_right_nsticks: {}", writer_rt_args[13]);
            log_debug(LogOp, "left_core_nsticks: {}", writer_rt_args[33]);
            log_debug(LogOp, "right_core_nsticks: {}", writer_rt_args[34]);
            log_debug(LogOp, "left_core_halo_offset: {}", writer_rt_args[37]);
            log_debug(LogOp, "right_core_halo_offset: {}", writer_rt_args[38]);
        }

        in_stick_start += in_nsticks_per_core;
        out_stick_start += out_nsticks_per_core;
    }

    auto override_runtime_arguments_callback = [
        src_cb=src_cb,
        out_cb=out_cb
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);

        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

void UntilizeWithHalo::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Input tensor is not TILE for untilize");
    TT_FATAL(input_tensor_a.memory_config().is_sharded());
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED or input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED, "Only works for sharded input");
    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);

    // Only stride 1 and 2 mode are supported
    TT_FATAL(stride_ == 1 || stride_ == 2, "Only stride 1 and stride 2 modes are supported.");
}

std::vector<Shape> UntilizeWithHalo::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_legacy_shape();
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
    uint32_t nbatch = input_shape[0];

    // get ncores from shard shape and input shape
    auto shard_shape = input.shard_spec().value().shape;
    uint32_t ncores = in_nhw / shard_shape[0];
    if (input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(input.shard_spec().value().grid.ranges().begin());
        ncores = core_range.end.x - core_range.start.x + 1;
    }

    uint32_t total_nsticks = ncores * max_out_nsticks_per_core_;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    if (stride_ == 1) {
        output_shape[2] = total_nsticks;
    } else {
        total_nsticks = ncores * (max_out_nsticks_per_core_ + 2);   // TODO [AS]: Need to debug why using exact number (without + 2) makes it fail.
        output_shape[2] = (uint32_t) ceil((float) total_nsticks / output_shape[0]);
    }

    if (1) {
        log_debug(LogOp, "output_shape: {} {} {} {}", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
        log_debug(LogOp, "max_out_nsticks_per_core: {}", max_out_nsticks_per_core_);
        log_debug(LogOp, "derived ncores: {}", ncores);
    }

    return {output_shape};
}

std::vector<Tensor> UntilizeWithHalo::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype = input_tensor.get_dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.get_dtype();
    auto shard_spec = input_tensor.shard_spec().value();
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    uint32_t ncores = input_tensor.get_legacy_shape()[0] * input_tensor.get_legacy_shape()[2] / shard_spec.shape[0];
    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(input_tensor.shard_spec().value().grid.ranges().begin());
        ncores = core_range.end.x - core_range.start.x + 1;
    }
    shard_spec.shape[0] = output_shape[0] * div_up(output_shape[2], ncores);
    shard_spec.halo = true;
    // log_debug(LogOp, "derived ncores: {}", ncores);
    auto mem_config = this->output_mem_config;
    mem_config.shard_spec = shard_spec;
    return {create_sharded_device_tensor(output_shape, output_dtype, Layout::ROW_MAJOR, input_tensor.device(), mem_config)};
}

operation::ProgramWithCallbacks UntilizeWithHalo::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    switch (stride_) {
        case 1:
            log_debug(LogOp, "Using stride 1 kernel");
            return { untilize_with_halo_multi_core_s1(input_tensor_a, output_tensor, pad_val_, this->in_b, this->in_h, this->in_w, this->max_out_nsticks_per_core_) };
        case 2:
            log_debug(LogOp, "Using stride 2 kernel");
            return { untilize_with_halo_multi_core_s2(input_tensor_a, output_tensor, pad_val_, in_b, in_h, in_w, max_out_nsticks_per_core_, pc_) };
        default:
            TT_ASSERT(false, "Unsupported stride value");
    };
    return {};
}

Tensor untilize_with_halo(const Tensor &input_tensor_a, const uint32_t pad_val, const uint32_t &in_b, const uint32_t &in_h, const uint32_t &in_w, const uint32_t stride, const MemoryConfig& mem_config) {
    TT_ASSERT(input_tensor_a.memory_config().is_sharded()); // TODO: Remove from validate?

    // TODO: Pass these attributes in instead of hardcoding
    uint32_t pad_h = 1;
    uint32_t pad_w = 1;
    uint32_t window_h = 3;
    uint32_t window_w = 3;
    uint32_t dilation_h = 1;
    uint32_t dilation_w = 1;

    CoreRangeSet all_cores = input_tensor_a.shard_spec().value().grid;
    uint32_t ncores = all_cores.num_cores();

    // TODO: Uplift to support different num of sticks per core
    uint32_t in_nsticks_per_core = (in_b * in_h * in_w) / ncores;
    // NOTE: This is always the same for all cores
    uint32_t halo_out_nsticks = (in_w + window_w / 2 + 2 * pad_w);  // left or right halo

    if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_ASSERT(input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR);
        TT_ASSERT(all_cores.ranges().size() == 1);
        auto core_range = *(all_cores.ranges().begin());
        ncores = core_range.end.x - core_range.start.x + 1;
        in_nsticks_per_core = input_tensor_a.shard_spec().value().shape[0];
    }

    auto input_shape = input_tensor_a.get_legacy_shape();
    uint32_t in_hw = in_h * in_w;
    uint32_t in_nhw = in_b * in_hw;

    PoolConfig pc {
        .in_w = in_w,
        .in_h = in_h,
        .out_w = 0,         // set later in the following
        .out_h = 0,         // set later in the following
        .stride_w = stride, // assumes same stride in both dims
        .stride_h = stride,
        .pad_w = pad_w,
        .pad_h = pad_h,
        .window_w = window_w,
        .window_h = window_h,
        .dilation_w = dilation_w,
        .dilation_h = dilation_h
    };

    // Calculate the max output nsticks across all cores
    int32_t max_out_nsticks_per_core = 0;
    int32_t halo_in_nsticks = 0;
    int32_t out_stick_start = 0;
    int32_t pool_out_nsticks_per_core = 0;
    switch (stride) {
        case 1:
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

                int32_t out_nsticks = local_nsticks + 2 * halo_out_nsticks;
                max_out_nsticks_per_core = std::max(max_out_nsticks_per_core, out_nsticks);
            }
            break;

        case 2:

            TT_ASSERT(pc.in_h * pc.in_w == input_shape[2] || in_b * pc.in_h * pc.in_w == input_shape[2]);

            // resuting output shape of subsequent pooling op
            pc.out_h = ((pc.in_h + 2 * pc.pad_h - (pc.dilation_h * pc.window_h - 1) - 1) / pc.stride_h) + 1;
            pc.out_w = ((pc.in_w + 2 * pc.pad_w - (pc.dilation_w * pc.window_w - 1) - 1) / pc.stride_w) + 1;
            halo_in_nsticks = (pc.in_w + (pc.window_w / 2)) * (pc.window_h / 2);     // input sticks to the writer
            pool_out_nsticks_per_core = in_b * pc.out_h * pc.out_w / ncores;

            out_stick_start = 0;   // global "output" stick (after downsample/pool)
            for (uint32_t core = 0; core < ncores; ++ core) {
                range_t out_range = {out_stick_start, out_stick_start + pool_out_nsticks_per_core};
                range_t in_range = untilize_with_halo_helpers::calculate_in_range(out_range, pc);  // this represents the "window" center input sticks
                int32_t l_halo_start = in_range[0] - halo_in_nsticks;
                int32_t batch_start = (in_range[0] / (pc.in_h * pc.in_w)) * (pc.in_h * pc.in_w);
                l_halo_start = l_halo_start < batch_start ? batch_start : l_halo_start;
                int32_t r_halo_end = in_range[1] + halo_in_nsticks;
                r_halo_end = r_halo_end >= in_nhw ? in_nhw : r_halo_end;
                max_out_nsticks_per_core = std::max(max_out_nsticks_per_core, (r_halo_end - l_halo_start));
                out_stick_start += pool_out_nsticks_per_core;
            }
            break;

        default:
            TT_ASSERT(false, "Invalid stride value!");
    }
    log_debug("max nsticks across all cores = {}", max_out_nsticks_per_core);

    return operation::run_without_autoformat(UntilizeWithHalo{pad_val, in_b, in_h, in_w, max_out_nsticks_per_core, stride, pc, mem_config}, {input_tensor_a}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
