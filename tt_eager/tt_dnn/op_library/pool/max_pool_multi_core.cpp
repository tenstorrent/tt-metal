// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "tt_dnn/op_library/pool/max_pool.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"   // for reduce_op_utils
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_metal/host_api.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "detail/util.hpp"

namespace tt {
namespace tt_metal {

namespace max_pool_helpers {

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

void print_neighbor_noc_xy_mapping() {
    for (auto left : left_neighbor_noc_xy) {
        auto local_noc = left.first;
        auto left_noc = left.second;
        log_debug("({},{}) --left--> ({},{})", local_noc.x, local_noc.y, left_noc.x, left_noc.y);
    }
    for (auto right : right_neighbor_noc_xy) {
        auto local_noc = right.first;
        auto right_noc = right.second;
        log_debug("({},{}) --right--> ({},{})", local_noc.x, local_noc.y, right_noc.x, right_noc.y);
    }

}

CoreCoord get_ncores_hw(uint32_t h, uint32_t w, uint32_t avail_cores_h, uint32_t avail_cores_w) {
    CoreCoord cores_shape(0, 0);
    uint32_t total_cores = avail_cores_h * avail_cores_w;
    if (h >= total_cores) {
        TT_ASSERT("Too many cores ({}) :P. Case not yet implemented", total_cores);
    } else {
        // h < total_cores
        if (h == 56) {
            // NOTE: hardcoded for resnet50 maxpool output shape. TODO [AS]: Generalize
            // 56 = 7 * 8
            cores_shape.x = 7;
            cores_shape.y = 8;
        } else {
            TT_ASSERT("This case not handled yet!!");
        }
    }
    return cores_shape;
}

std::tuple<CoreRange, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> get_decomposition_h(uint32_t out_h, uint32_t ncores_h, uint32_t ncores_w) {
    uint32_t out_h_per_core = out_h / (ncores_h * ncores_w);
    uint32_t out_h_per_core_cliff = out_h % (ncores_h * ncores_w);
    std::set<CoreRange> core_range, core_range_cliff;
    if (out_h_per_core_cliff == 0) {
        // no cliff, distribute evenly, corerange is full core rectangle
        core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_w - 1, ncores_h - 1)});
    } else {
        // all but last row
        core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_w - 2, ncores_h - 1)});
        // last row but last core, only the last core is cliff (1D, not 2D)
        core_range.insert(CoreRange{.start = CoreCoord(0, ncores_h - 1), .end = CoreCoord(ncores_w - 2, ncores_h - 1)});
        core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_w - 1, ncores_h - 1), .end = CoreCoord(ncores_w - 1, ncores_h - 1)});
    }
    CoreRange all_cores{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_w - 1, ncores_h - 1)};
    return std::make_tuple(all_cores, core_range, core_range_cliff, out_h_per_core, out_h_per_core_cliff);
}

uint32_t get_num_cores(CoreCoord grid_size, uint32_t out_nhw) {
    uint32_t avail_ncores = grid_size.x * grid_size.y;
    uint32_t ncores;
    switch (out_nhw) {
        case 1024:  // test case
            ncores = 32;
            break;
        case 2048:  // test case
        case 4096:  // test case
        case 8192:  // test case
        case 16384:  // test case
        case 32768:  // test case
            ncores = 64;
            break;
        case 3136:  // nbatch = 1
        case 6272:  // nbatch = 2
        case 12544: // nbatch = 4
        case 25088: // nbatch = 8
        case 50176: // nbatch = 16
        case 62720: // nbatch = 20
            ncores = 98;
            break;
        case 784:   // test case
            ncores = 49;
            break;
        default:
            TT_ASSERT(false, "General case is not yet handled! Only RN50 shapes supported in multicore.");
            uint32_t out_nhw_per_core = (uint32_t) ceil((float) out_nhw / avail_ncores);
            ncores = out_nhw / out_nhw_per_core;
            break;
    }
    return ncores;
}

// decompose along height = N * H * W
std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t, uint32_t, uint32_t>
get_decomposition_nhw(CoreCoord grid_size, uint32_t in_nhw, uint32_t out_nhw) {
    std::set<CoreRange> all_cores, core_range, core_range_cliff;
    uint32_t avail_ncores = grid_size.x * grid_size.y;
    // // generic decomposition:
    // uint32_t ncores = out_nhw / out_nhw_per_core;

    // hardcoded for resnet shapes:
    uint32_t ncores = 0, out_nhw_per_core = 0, in_nhw_per_core = 0;
    ncores = get_num_cores(grid_size, out_nhw);

    out_nhw_per_core = out_nhw / ncores;
    in_nhw_per_core = in_nhw / ncores;
    uint32_t ncores_w = grid_size.x;    // 12
    uint32_t ncores_h = ncores / ncores_w;
    uint32_t ncores_cliff_h = 0;
    if (ncores % ncores_w != 0) ncores_cliff_h = 1;
    uint32_t ncores_cliff_w = ncores % ncores_w;
    // NOTE: Cliff core is not yet handled, assuming (out_nhw / ncores) is a whole number.
    uint32_t in_nhw_per_core_cliff = 0;
    uint32_t out_nhw_per_core_cliff = 0;

    // all but last row
    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_w - 1, ncores_h - 1)});
    all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_w - 1, ncores_h - 1)});
    // last row
    if (ncores_cliff_h > 0) {
        core_range.insert(CoreRange{.start = CoreCoord(0, ncores_h), .end = CoreCoord(ncores_cliff_w - 1, ncores_h)});
        all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_h), .end = CoreCoord(ncores_cliff_w - 1, ncores_h)});
    }

    return std::make_tuple(ncores, all_cores, core_range, core_range_cliff, in_nhw_per_core, in_nhw_per_core_cliff, out_nhw_per_core, out_nhw_per_core_cliff);
}

} // namespacce max_pool_helpers

// this version uses distribution along height = N * H * W
operation::ProgramWithCallbacks max_pool_2d_multi_core_generic(const Tensor &input, Tensor& output,
                                                                uint32_t in_h, uint32_t in_w,
                                                                uint32_t out_h, uint32_t out_w,
                                                                uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                                uint32_t stride_h, uint32_t stride_w,
                                                                uint32_t pad_h, uint32_t pad_w,
                                                                uint32_t dilation_h, uint32_t dilation_w,
                                                                const MemoryConfig& out_mem_config,
                                                                uint32_t nblocks) {
    Program program = CreateProgram();

    // This should allocate a DRAM buffer on the device
    Device *device = input.device();
    Buffer *src_dram_buffer = input.buffer();
    Buffer *dst_dram_buffer = output.buffer();

    Shape input_shape = input.shape();
    Shape output_shape = output.shape();

    // NOTE: input is assumed to be in {N, 1, H * W, C }

    // TODO [AS]: Support other data formats??
    DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);
    uint32_t in_nbytes_c = input_shape[3] * in_nbytes;      // row of input (channels)
    uint32_t out_nbytes_c = output_shape[3] * out_nbytes;   // row of output (channels)
    TT_ASSERT((in_nbytes_c & (in_nbytes_c - 1)) == 0, "in_nbytes_c should be power of 2");    // in_nbytes_c is power of 2
    TT_ASSERT((out_nbytes_c & (out_nbytes_c - 1)) == 0, "out_nbytes_c should be power of 2"); // out_nbytes_c is power of 2

    uint32_t nbatch = input_shape[0];
    TT_ASSERT(nbatch == output_shape[0], "Mismatch in N for input and output!!");

    uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;    // number of valid rows, to read
    uint32_t kernel_size_hw_padded = ceil_multiple_of(kernel_size_hw, constants::TILE_HEIGHT);
    uint32_t in_ntiles_hw = (uint32_t) ceil((float) kernel_size_hw_padded / constants::TILE_HEIGHT);
    uint32_t in_ntiles_c = (uint32_t) ceil((float) input_shape[3] / constants::TILE_WIDTH);
    uint32_t out_ntiles_hw = (uint32_t) ceil((float) output_shape[2] / constants::TILE_HEIGHT);
    uint32_t out_ntiles_c = (uint32_t) ceil((float) output_shape[3] / constants::TILE_WIDTH);

    uint32_t out_nelems = nblocks;  // TODO [AS]: Remove hard coding after identifying optimal param val
                                    // Also ensure the calculated ncores is good
    uint32_t out_w_loop_count = ceil((float) out_w / out_nelems);

    uint32_t in_hw = in_h * in_w;
    uint32_t in_nhw = in_hw * nbatch;
    uint32_t out_hw = out_h * out_w;
    uint32_t out_nhw = out_hw * nbatch;

    // distributing out_hw across the grid
    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, in_nhw_per_core, in_nhw_per_core_cliff, out_nhw_per_core, out_nhw_per_core_cliff] = max_pool_helpers::get_decomposition_nhw(grid_size, in_nhw, out_nhw);
    if (input.memory_config().is_sharded()) {
        all_cores = input.shard_spec().value().grid;
        uint32_t ncores = all_cores.num_cores();
        core_range = all_cores;
        core_range_cliff = CoreRangeSet({});
        in_nhw_per_core = input.shard_spec().value().shape[0];
        in_nhw_per_core_cliff = 0;
        out_nhw_per_core = out_nhw / ncores;
        out_nhw_per_core_cliff = 0;
    }
    uint32_t ncores_w = grid_size.x;

    // TODO: support generic nblocks
    TT_ASSERT(out_nhw_per_core % nblocks == 0, "number of sticks per core ({}) should be divisible by nblocks ({})", out_nhw_per_core, nblocks);
    // TODO: support generic values for in_nhw_per_core
    TT_ASSERT((in_nhw_per_core & (in_nhw_per_core - 1)) == 0, "in_nhw_per_core {} needs to be power of 2!", in_nhw_per_core);

    uint32_t in_nhw_per_core_rem_mask = in_nhw_per_core - 1;    // NOTE: assuming in_nhw_per_core is power of 2

    // CBs
    uint32_t multi_buffering_factor = 2;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_id = CB::c_in1;
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    CircularBufferConfig in_scalar_cb_config = CircularBufferConfig(
                                                    in_scalar_cb_npages * in_scalar_cb_pagesize,
                                                    {{in_scalar_cb_id, in_df}})
		                                        .set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);
    auto in_scalar_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_scalar_cb_config);

    CBHandle raw_in_cb = 0;
    if (input.memory_config().is_sharded()) {
        // incoming data is the input cb instead of raw l1/dram addr
        auto raw_in_cb_id = CB::c_in2;
        uint32_t raw_in_cb_npages = in_nhw_per_core;
        uint32_t raw_in_cb_pagesize = in_nbytes_c;
        CircularBufferConfig raw_in_cb_config = CircularBufferConfig(
                                                    raw_in_cb_npages * raw_in_cb_pagesize,
                                                    {{raw_in_cb_id, in_df}})
                                                .set_page_size(raw_in_cb_id, raw_in_cb_pagesize)
                                                .set_globally_allocated_address(*input.buffer());
        raw_in_cb = CreateCircularBuffer(program, all_cores, raw_in_cb_config);
    }

    // reader output == input to tilize
    uint32_t in_cb_id = CB::c_in0;          // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_page_nelems_padded = ceil_multiple_of(input_shape[3] * kernel_size_hw_padded, constants::TILE_HW);    // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_nelems_padded;
    uint32_t in_cb_npages = multi_buffering_factor * out_nelems;
    CircularBufferConfig in_cb_config = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id, in_df}})
		.set_page_size(in_cb_id, in_cb_pagesize);
    auto in_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);

    // output of tilize == input to reduce
    uint32_t in_tiled_cb_id = CB::c_intermed0;  // tiled input
    uint32_t in_tiled_cb_pagesize = tile_size(in_df);
    uint32_t in_tiled_cb_npages = in_ntiles_c * in_ntiles_hw * out_nelems;
    CircularBufferConfig in_tiled_cb_config = CircularBufferConfig(in_tiled_cb_npages * in_tiled_cb_pagesize, {{in_tiled_cb_id, in_df}})
		.set_page_size(in_tiled_cb_id, in_tiled_cb_pagesize);
    auto in_tiled_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_tiled_cb_config);

    // output of reduce == writer to write
    uint32_t out_cb_id = CB::c_out0;            // output rows in RM
    uint32_t out_cb_pagesize = tile_size(out_df);
    uint32_t out_cb_npages = out_ntiles_c * out_nelems * multi_buffering_factor;    // there is just one row of channels after reduction
    CircularBufferConfig cb_out_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
		.set_page_size(out_cb_id, out_cb_pagesize);
    auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    CBHandle cb_sharded_out = 0;
    if (output.memory_config().is_sharded()) {
        uint32_t sharded_out_cb_id = CB::c_out1;            // output rows in RM

        uint32_t sharded_out_num_pages = output.shard_spec().value().shape[0];

        uint32_t sharded_out_cb_page_size = output.shard_spec().value().shape[1] * out_nbytes;    // there is just one row of channels after reduction
        CircularBufferConfig cb_sharded_out_config = CircularBufferConfig(sharded_out_num_pages * sharded_out_cb_page_size, {{sharded_out_cb_id, out_df}})
            .set_page_size(sharded_out_cb_id, sharded_out_cb_page_size).set_globally_allocated_address(*output.buffer());
        cb_sharded_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_sharded_out_config);
    }

    // Construct const buffer with -INF
    // uint32_t const_buffer_size = 32;
    uint32_t const_buffer_size = input_shape[3];    // set it equal to 1 row
    auto minus_inf_const_buffer = owned_buffer::create(std::vector<bfloat16>(const_buffer_size, bfloat16(0xf7ff)));
    const Tensor minus_inf_const_tensor = Tensor(OwnedStorage{minus_inf_const_buffer},
                                                 Shape({1, 1, 1, const_buffer_size}),
                                                 DataType::BFLOAT16,
                                                 Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED,
                                                                     .buffer_type = BufferType::L1});
    auto minus_inf_const_tensor_addr = minus_inf_const_tensor.buffer()->address();

    #if 1
    {   // debug
        log_debug("in_cb :: PS = {}, NP = {}", in_cb_pagesize, in_cb_npages);
        log_debug("in_scalar_cb :: PS = {}, NP = {}", in_scalar_cb_pagesize, in_scalar_cb_npages);
        log_debug("in_tiled_cb :: PS = {}, NP = {}", in_tiled_cb_pagesize, in_tiled_cb_npages);
        log_debug("out_cb :: PS = {}, NP = {}", out_cb_pagesize, out_cb_npages);
        log_debug("in_addr: {}", src_dram_buffer->address());
        log_debug("out_addr: {}", dst_dram_buffer->address());
        log_debug("nbatch: {}", nbatch);
        log_debug("kernel_size_h: {}", kernel_size_h);
        log_debug("kernel_size_w: {}", kernel_size_w);
        log_debug("kernel_size_hw: {}", kernel_size_hw);
        log_debug("kernel_size_hw_padded: {}", kernel_size_hw_padded);
        log_debug("stride_h: {}", stride_h);
        log_debug("stride_w: {}", stride_w);
        log_debug("pad_h: {}", pad_h);
        log_debug("pad_w: {}", pad_w);
        log_debug("out_h: {}", out_h);
        log_debug("out_w: {}", out_w);
        log_debug("out_hw: {}", output_shape[2]);
        log_debug("out_c: {}", output_shape[3]);
        log_debug("in_nbytes_c: {}", in_nbytes_c);
        log_debug("out_nbytes_c: {}", out_nbytes_c);
        log_debug("in_h: {}", in_h);
        log_debug("in_w: {}", in_w);
        log_debug("in_hw: {}", input_shape[2]);
        log_debug("in_hw_padded: {}", in_hw);
        log_debug("in_c: {}", input_shape[3]);
        log_debug("out_hw_padded: {}", out_hw);
        log_debug("out_ntiles_hw: {}", out_ntiles_hw);
        log_debug("out_ntiles_c: {}", out_ntiles_c);
        log_debug("out_nelems: {}", out_nelems);
        log_debug("out_w_loop_count: {}", out_w_loop_count);
        log_debug("out_hw: {}", out_hw);
        log_debug("minus_inf_const_tensor_addr: {}", minus_inf_const_tensor_addr);
        log_debug("minus_inf_const_tensor_size: {}", minus_inf_const_buffer.size());
        log_debug("ncores: {}", ncores);
        log_debug("in_nhw_per_core: {}", in_nhw_per_core);
        log_debug("out_nhw_per_core: {}", out_nhw_per_core);
        log_debug("in_nhw_per_core_rem_mask: {}", in_nhw_per_core_rem_mask);
        log_debug("is_in_sharded: {}", input.memory_config().is_sharded());
        log_debug("is_out_sharded: {}", output.memory_config().is_sharded());
    }
    #endif

    const uint32_t reader_noc = 0;
    const uint32_t writer_noc = 1;

    if (input.memory_config().is_sharded()) {
        max_pool_helpers::init_neighbor_noc_xy_mapping(grid_size, reader_noc);
        // max_pool_helpers::print_neighbor_noc_xy_mapping();
    }

    /**
     * Reader Kernel: input rows -> input cb
     */
    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t*>(&one);
    std::vector<uint32_t> reader_ct_args = {input.memory_config().buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            out_mem_config.buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            bf16_one_u32,
                                            out_nelems,
                                            static_cast<uint32_t>(((in_nbytes_c & (in_nbytes_c - 1)) == 0) ? 1 : 0),    // is in_nbytes_c power of 2
                                            stride_h,
                                            stride_w,
                                            reader_noc,
                                            writer_noc};
    uint32_t in_log_base_2_of_page_size = (uint32_t) std::log2((float) in_nbytes_c);
    std::vector<uint32_t> reader_rt_args = {src_dram_buffer->address(),
                                            dst_dram_buffer->address(),
                                            kernel_size_h, kernel_size_w, kernel_size_hw, kernel_size_hw_padded,
                                            stride_h, stride_w,
                                            pad_h, pad_w,
                                            out_h, out_w, output_shape[2], output_shape[3],
                                            in_nbytes_c, out_nbytes_c,
                                            in_h, in_w, input_shape[2], input_shape[3],
                                            out_ntiles_hw, out_ntiles_c,
                                            in_cb_pagesize, out_cb_pagesize,
                                            in_cb_page_nelems_padded, out_w_loop_count,
                                            in_log_base_2_of_page_size,
                                            nbatch,
                                            in_hw,
                                            out_hw,
                                            // these are set later in the following
                                            0,          // start_out_h_i
                                            0,          // end_out_h_i
                                            0,          // base_start_h
                                            0,          // start_out_row_id
                                            minus_inf_const_tensor_addr,
                                            const_buffer_size * in_nbytes,
                                            (in_cb_page_nelems_padded * out_nelems * 2) >> 5,    // TODO: generalize num rows to fill in in_cb
                                            0,          // core_offset_in_row_id
                                            0,          // core_out_w_i_start
                                            0,          // core_out_h_i_start
                                            out_nhw_per_core,    // nsticks_per_core
                                            0,          // core_offset_out_row_id
                                            out_nhw_per_core / nblocks,     // loop count with blocks
                                            // the following are for sharded input
                                            0,                  // 43: local_out_stick_start
                                            out_hw,             // out_nsticks_per_batch
                                            0,                  // local_in_stick_start
                                            0,                  // local_in_stick_end
                                            in_hw,              // in_nsticks_per_batch
                                            in_nhw_per_core,    // in_nsticks_per_core
                                            0,                  // has_left
                                            0,                  // left_noc_x
                                            0,                  // left_noc_y
                                            0,                  // has_right
                                            0,                  // right_noc_x
                                            0,                  // right_noc_y
                                            in_nhw_per_core_rem_mask,
                                            0,                  // 56: has_left_left,
                                            0,                  // left_left_noc_x,
                                            0,                  // left_left_noc_y,
                                            0,                  // has_right_right,
                                            0,                  // right_right_noc_x,
                                            0,                  // right_right_noc_y,
                                            0,                  // left_in_stick_start,
                                            0,                  // right_in_stick_end,
                                            0,                  // my_core
                                            };
    auto reader_config = ReaderDataMovementConfig{.compile_args = reader_ct_args};
    std::string reader_kernel_fname;
    if (input.memory_config().is_sharded()) {
        reader_kernel_fname = std::string("tt_eager/tt_dnn/op_library/pool/kernels/dataflow/reader_max_pool_2d_multi_core_sharded.cpp");
    } else {
        reader_kernel_fname = std::string("tt_eager/tt_dnn/op_library/pool/kernels/dataflow/reader_max_pool_2d_multi_core.cpp");
    }
    auto reader_kernel = CreateKernel(program,
                                                  reader_kernel_fname,
                                                  all_cores,
                                                  reader_config);

    /**
     * Writer Kernel: output cb -> output rows
     */
    std::map<string, string> writer_defines;
    if (output.memory_config().is_sharded()) {
        writer_defines["SHARDED_OUT"] = "1";
    }
    std::vector<uint32_t> writer_ct_args = reader_ct_args;
    auto writer_config = WriterDataMovementConfig{.compile_args = writer_ct_args, .defines = writer_defines};
    std::string writer_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/dataflow/writer_max_pool_2d_multi_core.cpp");
    auto writer_kernel = CreateKernel(program,
                                                  writer_kernel_fname,
                                                  all_cores,
                                                  writer_config);

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block -> output cb
     */
    std::vector<uint32_t> compute_ct_args = {in_ntiles_hw,
                                            in_ntiles_c,
                                            in_ntiles_hw * in_ntiles_c,
                                            kernel_size_hw_padded,
                                            out_h,
                                            out_w,
                                            (uint32_t) ceil((float) output_shape[2] / constants::TILE_HEIGHT),
                                            (uint32_t) ceil((float) output_shape[3] / constants::TILE_WIDTH),
                                            out_nelems,
                                            out_w_loop_count,
                                            nbatch,
                                            out_nhw_per_core,
                                            out_nhw_per_core,
                                            out_nhw_per_core / nblocks,     // loop count with blocks
                                            };
    auto compute_ct_args_cliff = compute_ct_args;
    auto reduce_op = ReduceOpMath::MAX;
    auto reduce_dim = ReduceOpDim::H;
    auto compute_config = ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                        .fp32_dest_acc_en = false,
                                        .math_approx_mode = false,
                                        .compile_args = compute_ct_args,
                                        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    std::string compute_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/compute/max_pool_multi_core.cpp");
    auto compute_kernel = CreateKernel(program,
                                              compute_kernel_fname,
                                              core_range,
                                              compute_config);

    if (out_nhw_per_core_cliff > 0) {
        TT_ASSERT(false, "The cliff core case is not yet handled"); // TODO
        // there is a cliff core
        compute_ct_args_cliff[11] = out_nhw_per_core_cliff;
        auto compute_config_cliff = ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                                    .fp32_dest_acc_en = false,
                                                    .math_approx_mode = false,
                                                    .compile_args = compute_ct_args_cliff,
                                                    .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
        auto compute_kernel_cliff = CreateKernel(program,
                                                        compute_kernel_fname,
                                                        core_range_cliff,
                                                        compute_config);
    }

    // calculate and set the start/end h_i for each core
    // for all but last core (cliff)
    uint32_t core_out_h_i = 0;
    uint32_t core_out_w_i = 0;
    int32_t curr_start_h = - pad_h;
    if (out_nhw_per_core_cliff > 0) {
        // TODO: ... not yet handled
        TT_ASSERT(false, "The cliff core case is not yet handled"); // TODO
    } else {
        uint32_t core_batch_offset = 0;
        uint32_t curr_out_stick_id = 0; // track output sticks with batch folded in
        int32_t curr_in_stick_id = 0; // track input sticks with batch folded in
        uint32_t core_out_w_i_start = 0;
        uint32_t core_out_h_i_start = 0;
        for (int32_t i = 0; i < ncores; ++ i) {
            CoreCoord core(i % ncores_w, i / ncores_w); // logical
            reader_rt_args[37] = (curr_in_stick_id / in_hw) * in_hw;
            core_out_w_i_start = curr_out_stick_id % out_w;
            core_out_h_i_start = (curr_out_stick_id / out_w) % out_h;
            reader_rt_args[38] = core_out_w_i_start;
            reader_rt_args[39] = core_out_h_i_start;
            reader_rt_args[41] = curr_out_stick_id;

            if (input.memory_config().is_sharded()) {
                reader_rt_args[43] = curr_out_stick_id;
                reader_rt_args[45] = curr_in_stick_id;
                reader_rt_args[46] = curr_in_stick_id + in_nhw_per_core;

                reader_rt_args[64] = i; // my_core

                CoreCoord noc_core = core;  // physical
                if (reader_noc == 0) {
                    noc_core.x += 1;
                    noc_core.y += 1;
                    if (noc_core.y > 5) {
                        noc_core.y += 1;
                    }
                } else {
                    TT_ASSERT(false, "reader noc == 1 not yet handled");
                }
                if (max_pool_helpers::left_neighbor_noc_xy.count(noc_core) > 0) {
                    CoreCoord left_noc = max_pool_helpers::left_neighbor_noc_xy.at(noc_core);
                    reader_rt_args[49] = 1;
                    reader_rt_args[50] = (uint32_t) left_noc.x;
                    reader_rt_args[51] = (uint32_t) left_noc.y;
                    // log_debug("Local NOC: ({},{}), left: ({},{})", noc_core.x, noc_core.y, left_noc.x, left_noc.y);

                    // left-left
                    if (max_pool_helpers::left_neighbor_noc_xy.count(left_noc) > 0) {
                        CoreCoord left_left_noc = max_pool_helpers::left_neighbor_noc_xy.at(left_noc);
                        reader_rt_args[56] = 1;
                        reader_rt_args[57] = (uint32_t) left_left_noc.x;
                        reader_rt_args[58] = (uint32_t) left_left_noc.y;
                        reader_rt_args[62] = (uint32_t) (curr_in_stick_id - (int32_t) in_nhw_per_core);
                    } else {
                        reader_rt_args[56] = 0;
                    }
                } else {
                    reader_rt_args[49] = 0;
                }
                if (max_pool_helpers::right_neighbor_noc_xy.count(noc_core) > 0) {
                    CoreCoord right_noc = max_pool_helpers::right_neighbor_noc_xy.at(noc_core);
                    reader_rt_args[52] = 1;
                    reader_rt_args[53] = (uint32_t) right_noc.x;
                    reader_rt_args[54] = (uint32_t) right_noc.y;
                    // log_debug("Local NOC: ({},{}), right: ({},{})", noc_core.x, noc_core.y, right_noc.x, right_noc.y);

                    // right-right
                    if (max_pool_helpers::right_neighbor_noc_xy.count(right_noc) > 0) {
                        CoreCoord right_right_noc = max_pool_helpers::right_neighbor_noc_xy.at(right_noc);
                        reader_rt_args[59] = 1;
                        reader_rt_args[60] = (uint32_t) right_right_noc.x;
                        reader_rt_args[61] = (uint32_t) right_right_noc.y;
                        reader_rt_args[63] = (uint32_t) (curr_in_stick_id + 2 * in_nhw_per_core);
                    } else {
                        reader_rt_args[59] = 0;
                    }
                } else {
                    reader_rt_args[52] = 0;
                }
            }

            // log_debug("CORE: {},{} :: 37 = {}, 38 = {}, 39 = {}, 41 = {}", core.x, core.y, reader_rt_args[37], reader_rt_args[38], reader_rt_args[39], reader_rt_args[41]);
            SetRuntimeArgs(program, reader_kernel, core, reader_rt_args);
            std::vector<uint32_t> writer_rt_args = reader_rt_args;
            SetRuntimeArgs(program, writer_kernel, core, writer_rt_args);

            curr_out_stick_id += out_nhw_per_core;
            curr_in_stick_id += in_nhw_per_core;
        }
    }

    auto override_runtime_arguments_callback = [
            reader_kernel, writer_kernel, raw_in_cb, cb_sharded_out, ncores, ncores_w
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer = input_tensors.at(0).buffer();
        bool input_sharded = input_tensors.at(0).is_sharded();

        auto dst_buffer = output_tensors.at(0).buffer();
        bool out_sharded = output_tensors.at(0).is_sharded();

        for (uint32_t i = 0; i < ncores; ++ i) {
            CoreCoord core{i % ncores_w, i / ncores_w };
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
            }
            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
            }
        }
        if (input_sharded) {
            UpdateDynamicCircularBufferAddress(program, raw_in_cb, *src_buffer);
        }
        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_sharded_out, *dst_buffer);
        }
    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

// This version uses only output row distribution along H
operation::ProgramWithCallbacks max_pool_2d_multi_core(const Tensor &input, Tensor& output,
                                                       uint32_t in_h, uint32_t in_w,
                                                       uint32_t out_h, uint32_t out_w,
                                                       uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                       uint32_t stride_h, uint32_t stride_w,
                                                       uint32_t pad_h, uint32_t pad_w,
                                                       uint32_t dilation_h, uint32_t dilation_w,
                                                       const MemoryConfig& out_mem_config,
                                                       uint32_t nblocks) {
    Program program = CreateProgram();

    // This should allocate a DRAM buffer on the device
    Device *device = input.device();
    Buffer *src_dram_buffer = input.buffer();
    Buffer *dst_dram_buffer = output.buffer();

    Shape input_shape = input.shape();
    Shape output_shape = output.shape();

    // NOTE: input is assumed to be in {N, 1, H * W, C }

    // TODO [AS]: Support other data formats??
    DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);
    uint32_t in_nbytes_c = input_shape[3] * in_nbytes;      // row of input (channels)
    uint32_t out_nbytes_c = output_shape[3] * out_nbytes;   // row of output (channels)
    TT_ASSERT((in_nbytes_c & (in_nbytes_c - 1)) == 0, "in_nbytes_c should be power of 2");    // in_nbytes_c is power of 2
    TT_ASSERT((out_nbytes_c & (out_nbytes_c - 1)) == 0, "out_nbytes_c should be power of 2"); // out_nbytes_c is power of 2

    uint32_t nbatch = input_shape[0];
    TT_ASSERT(nbatch == output_shape[0], "Mismatch in N for input and output!!");

    uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;    // number of valid rows, to read
    uint32_t kernel_size_hw_padded = ceil_multiple_of(kernel_size_hw, constants::TILE_HEIGHT);
    uint32_t in_ntiles_hw = (uint32_t) ceil((float) kernel_size_hw_padded / constants::TILE_HEIGHT);
    uint32_t in_ntiles_c = (uint32_t) ceil((float) input_shape[3] / constants::TILE_WIDTH);
    uint32_t out_ntiles_hw = (uint32_t) ceil((float) output_shape[2] / constants::TILE_HEIGHT);
    uint32_t out_ntiles_c = (uint32_t) ceil((float) output_shape[3] / constants::TILE_WIDTH);

    uint32_t out_nelems = nblocks;     // TODO [AS]: Remove hard coding after identifying optimal param val
    uint32_t out_w_loop_count = ceil((float) out_w / out_nelems);

    uint32_t in_hw = in_h * in_w;
    uint32_t out_hw = out_h * out_w;

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t total_ncores_w = grid_size.x;
    uint32_t total_ncores_h = grid_size.y;
    // distributing out_hw across the grid
    CoreCoord ncores_coord_hw = max_pool_helpers::get_ncores_hw(out_h, out_w, total_ncores_h, total_ncores_w);
    uint32_t ncores_h = ncores_coord_hw.y;
    uint32_t ncores_w = ncores_coord_hw.x;
    uint32_t ncores_hw = ncores_h * ncores_w;

    auto [all_cores, core_range, core_range_cliff, out_h_per_core, out_h_per_core_cliff] = max_pool_helpers::get_decomposition_h(out_h, ncores_h, ncores_w);

    // CBs
    uint32_t multi_buffering_factor = 2;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_id = CB::c_in1;
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    CircularBufferConfig in_scalar_cb_config = CircularBufferConfig(in_scalar_cb_npages * in_scalar_cb_pagesize, {{in_scalar_cb_id, in_df}})
		.set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);
    auto in_scalar_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_scalar_cb_config);

    // reader output == input to tilize
    uint32_t in_cb_id = CB::c_in0;          // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_page_nelems_padded = ceil_multiple_of(input_shape[3] * kernel_size_hw_padded, constants::TILE_HW);    // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_nelems_padded;
    uint32_t in_cb_npages = multi_buffering_factor * out_nelems;
    CircularBufferConfig in_cb_config = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id, in_df}})
		.set_page_size(in_cb_id, in_cb_pagesize);
    auto in_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);

    // output of tilize == input to reduce
    uint32_t in_tiled_cb_id = CB::c_intermed0;  // tiled input
    uint32_t in_tiled_cb_pagesize = tile_size(in_df);
    uint32_t in_tiled_cb_npages = in_ntiles_c * in_ntiles_hw * out_nelems;
    CircularBufferConfig in_tiled_cb_config = CircularBufferConfig(in_tiled_cb_npages * in_tiled_cb_pagesize, {{in_tiled_cb_id, in_df}})
		.set_page_size(in_tiled_cb_id, in_tiled_cb_pagesize);
    auto in_tiled_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_tiled_cb_config);

    // output of reduce == writer to write
    uint32_t out_cb_id = CB::c_out0;            // output rows in RM
    uint32_t out_cb_pagesize = tile_size(out_df);
    uint32_t out_cb_npages = out_ntiles_c * out_nelems * multi_buffering_factor;    // there is just one row of channels after reduction
    CircularBufferConfig cb_out_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
		.set_page_size(out_cb_id, out_cb_pagesize);
    auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    // Construct const buffer with -INF
    // uint32_t const_buffer_size = 32;
    uint32_t const_buffer_size = input_shape[3];    // set it equal to 1 row
    auto minus_inf_const_buffer = owned_buffer::create(std::vector<bfloat16>(const_buffer_size, bfloat16(0xf7ff)));
    const Tensor minus_inf_const_tensor = Tensor(OwnedStorage{minus_inf_const_buffer},
                                                 Shape({1, 1, 1, const_buffer_size}),
                                                 DataType::BFLOAT16,
                                                 Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED,
                                                                     .buffer_type = BufferType::L1});
    auto minus_inf_const_tensor_addr = minus_inf_const_tensor.buffer()->address();

    #if 0
    {   // debug
        log_debug("in_cb :: PS = {}, NP = {}", in_cb_pagesize, in_cb_npages);
        log_debug("in_scalar_cb :: PS = {}, NP = {}", in_scalar_cb_pagesize, in_scalar_cb_npages);
        log_debug("in_tiled_cb :: PS = {}, NP = {}", in_tiled_cb_pagesize, in_tiled_cb_npages);
        log_debug("out_cb :: PS = {}, NP = {}", out_cb_pagesize, out_cb_npages);
        log_debug("in_addr: {}", src_dram_buffer->address());
        log_debug("out_addr: {}", dst_dram_buffer->address());
        log_debug("nbatch: {}", nbatch);
        log_debug("kernel_size_h: {}", kernel_size_h);
        log_debug("kernel_size_w: {}", kernel_size_w);
        log_debug("kernel_size_hw: {}", kernel_size_hw);
        log_debug("kernel_size_hw_padded: {}", kernel_size_hw_padded);
        log_debug("stride_h: {}", stride_h);
        log_debug("stride_w: {}", stride_w);
        log_debug("pad_h: {}", pad_h);
        log_debug("pad_w: {}", pad_w);
        log_debug("out_h: {}", out_h);
        log_debug("out_w: {}", out_w);
        log_debug("out_hw: {}", output_shape[2]);
        log_debug("out_c: {}", output_shape[3]);
        log_debug("in_nbytes_c: {}", in_nbytes_c);
        log_debug("out_nbytes_c: {}", out_nbytes_c);
        log_debug("in_h: {}", in_h);
        log_debug("in_w: {}", in_w);
        log_debug("in_hw: {}", input_shape[2]);
        log_debug("in_hw_padded: {}", in_hw);
        log_debug("in_c: {}", input_shape[3]);
        log_debug("out_hw_padded: {}", out_hw);
        log_debug("out_ntiles_hw: {}", out_ntiles_hw);
        log_debug("out_ntiles_c: {}", out_ntiles_c);
        log_debug("out_nelems: {}", out_nelems);
        log_debug("out_w_loop_count: {}", out_w_loop_count);

        log_debug("out_hw: {}", out_hw);
        log_debug("available total_ncores_h: {}, total_ncores_w: {}", total_ncores_h, total_ncores_w);
        log_debug("using ncores_h: {}", ncores_h);
        log_debug("using ncores_w: {}", ncores_w);
        log_debug("using ncores_hw: {}", ncores_hw);
        log_debug("out_h_per_core: {}", out_h_per_core);
        log_debug("out_h_per_core_cliff: {}", out_h_per_core_cliff);

        log_debug("minus_inf_const_tensor_addr: {}", minus_inf_const_tensor_addr);
        log_debug("minus_inf_const_tensor_size: {}", minus_inf_const_buffer.size());
    }
    #endif


    /**
     * Reader Kernel: input rows -> input cb
     */
    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t*>(&one);
    std::vector<uint32_t> reader_ct_args = {input.memory_config().buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            out_mem_config.buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            bf16_one_u32,
                                            out_nelems,
                                            static_cast<uint32_t>(((in_nbytes_c & (in_nbytes_c - 1)) == 0) ? 1 : 0),    // is in_nbytes_c power of 2
                                            stride_h,
                                            stride_w};
    uint32_t in_log_base_2_of_page_size = (uint32_t) std::log2((float) in_nbytes_c);
    std::vector<uint32_t> reader_rt_args = {src_dram_buffer->address(),
                                            dst_dram_buffer->address(),
                                            kernel_size_h, kernel_size_w, kernel_size_hw, kernel_size_hw_padded,
                                            stride_h, stride_w,
                                            pad_h, pad_w,
                                            out_h, out_w, output_shape[2], output_shape[3],
                                            in_nbytes_c, out_nbytes_c,
                                            in_h, in_w, input_shape[2], input_shape[3],
                                            out_ntiles_hw, out_ntiles_c,
                                            in_cb_pagesize, out_cb_pagesize,
                                            in_cb_page_nelems_padded, out_w_loop_count,
                                            in_log_base_2_of_page_size,
                                            nbatch,
                                            in_hw,
                                            out_hw,
                                            // these are set later in the following
                                            0,          // start_out_h_i
                                            0,          // end_out_h_i
                                            0,          // base_start_h
                                            0,          // start_out_row_id
                                            minus_inf_const_tensor_addr,
                                            const_buffer_size * in_nbytes,
                                            (in_cb_page_nelems_padded * out_nelems * 2) >> 5    // TODO: generalize num rows to fill in in_cb
                                            };
    auto reader_config = ReaderDataMovementConfig{.compile_args = reader_ct_args};
    std::string reader_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/dataflow/reader_max_pool_2d_single_core.cpp");
    auto reader_kernel = CreateKernel(program,
                                                  reader_kernel_fname,
                                                  all_cores,
                                                  reader_config);

    /**
     * Writer Kernel: output cb -> output rows
     */
    std::vector<uint32_t> writer_ct_args = reader_ct_args;
    auto writer_config = WriterDataMovementConfig{.compile_args = writer_ct_args};
    std::string writer_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/dataflow/writer_max_pool_2d_single_core.cpp");
    auto writer_kernel = CreateKernel(program,
                                                  writer_kernel_fname,
                                                  all_cores,
                                                  writer_config);

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block -> output cb
     */
    std::vector<uint32_t> compute_ct_args = {in_ntiles_hw,
                                            in_ntiles_c,
                                            in_ntiles_hw * in_ntiles_c,
                                            kernel_size_hw_padded,
                                            out_h,
                                            out_w,
                                            (uint32_t) ceil((float) output_shape[2] / constants::TILE_HEIGHT),
                                            (uint32_t) ceil((float) output_shape[3] / constants::TILE_WIDTH),
                                            out_nelems,
                                            out_w_loop_count,
                                            nbatch,
                                            out_h_per_core};
    auto compute_ct_args_cliff = compute_ct_args;
    auto reduce_op = ReduceOpMath::MAX;
    auto reduce_dim = ReduceOpDim::H;
    auto compute_config = ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                        .fp32_dest_acc_en = false,
                                        .math_approx_mode = false,
                                        .compile_args = compute_ct_args,
                                        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    std::string compute_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/compute/max_pool.cpp");
    auto compute_kernel = CreateKernel(program,
                                              compute_kernel_fname,
                                              core_range,
                                              compute_config);

    if (out_h_per_core_cliff > 0) {
        // there is a cliff core
        compute_ct_args_cliff[11] = out_h_per_core_cliff;
        auto compute_config_cliff = ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                                    .fp32_dest_acc_en = false,
                                                    .math_approx_mode = false,
                                                    .compile_args = compute_ct_args_cliff,
                                                    .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
        auto compute_kernel_cliff = CreateKernel(program,
                                                        compute_kernel_fname,
                                                        core_range_cliff,
                                                        compute_config);
    }

    // calculate and set the start/end h_i for each core
    // for all but last core (cliff)
    uint32_t curr_out_h_i = 0;
    int32_t curr_start_h = - pad_h;
    if (out_h_per_core_cliff > 0) {
        // have a cliff core
        for (int32_t i = 0; i < ncores_hw - 1; ++ i) {
            CoreCoord core(i % ncores_w, i / ncores_w);
            reader_rt_args[30] = curr_out_h_i; curr_out_h_i += out_h_per_core;  // start for reader
            reader_rt_args[31] = curr_out_h_i;
            reader_rt_args[32] = curr_start_h; curr_start_h += stride_h;

            SetRuntimeArgs(program, reader_kernel, core, reader_rt_args);
            std::vector<uint32_t> writer_rt_args = reader_rt_args;
            SetRuntimeArgs(program, writer_kernel, core, writer_rt_args);
        }
        // last core (cliff)
        CoreCoord core_cliff(ncores_w - 1, ncores_h - 1);
        reader_rt_args[30] = curr_out_h_i;
        reader_rt_args[31] = curr_out_h_i + out_h_per_core_cliff;
        reader_rt_args[32] = curr_start_h;
        SetRuntimeArgs(program, reader_kernel, core_cliff, reader_rt_args);
        std::vector<uint32_t> writer_rt_args = reader_rt_args;
        SetRuntimeArgs(program, writer_kernel, core_cliff, writer_rt_args);
    } else {
        // no cliff core
        for (int32_t i = 0; i < ncores_hw; ++ i) {
            CoreCoord core(i % ncores_w, i / ncores_w);
            reader_rt_args[30] = curr_out_h_i;          // start out_h i
            reader_rt_args[33] = curr_out_h_i * out_w;
            curr_out_h_i += out_h_per_core;
            reader_rt_args[31] = curr_out_h_i;
            reader_rt_args[32] = static_cast<uint32_t>(curr_start_h);
            curr_start_h += stride_h;
            // log_debug("CORE: ({},{}), RT ARGS 32: {}", core.x, core.y, reader_rt_args[31]);
            SetRuntimeArgs(program, reader_kernel, core, reader_rt_args);
            std::vector<uint32_t> writer_rt_args = reader_rt_args;
            SetRuntimeArgs(program, writer_kernel, core, writer_rt_args);
        }
    }

    auto override_runtime_args_callback =
        [reader_kernel, writer_kernel, ncores_hw, ncores_w](const Program& program,
                                                            const std::vector<Buffer*>& input_buffers,
                                                            const std::vector<Buffer*>& output_buffers) {
        auto src_dram_buffer = input_buffers.at(0);
        auto dst_dram_buffer = output_buffers.at(0);
        for (uint32_t i = 0; i < ncores_hw; ++ i) {
            CoreCoord core{i % ncores_w, i / ncores_w };
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel, core);
                runtime_args[0] = src_dram_buffer->address();
                runtime_args[1] = dst_dram_buffer->address();
            }
            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel, core);
                runtime_args[0] = src_dram_buffer->address();
                runtime_args[1] = dst_dram_buffer->address();
            }
        }
    };
    return {std::move(program), override_runtime_args_callback};
}

// this version uses distribution along height = N * H * W
operation::ProgramWithCallbacks max_pool_2d_multi_core_sharded_with_halo(const Tensor &input, Tensor& output,
                                                                        uint32_t in_n, uint32_t in_h, uint32_t in_w,
                                                                        uint32_t out_h, uint32_t out_w,
                                                                        uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                                        uint32_t stride_h, uint32_t stride_w,
                                                                        uint32_t pad_h, uint32_t pad_w,
                                                                        uint32_t dilation_h, uint32_t dilation_w,
                                                                        const MemoryConfig& out_mem_config,
                                                                        uint32_t nblocks) {
    Program program = CreateProgram();

    // This should allocate a DRAM buffer on the device
    Device *device = input.device();
    Buffer *src_dram_buffer = input.buffer();
    Buffer *dst_dram_buffer = output.buffer();

    Shape input_shape = input.shape();
    Shape output_shape = output.shape();

    // NOTE: input is assumed to be in {N, 1, H * W, C }

    // TODO [AS]: Support other data formats??
    DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);
    uint32_t in_nbytes_c = input_shape[3] * in_nbytes;      // row of input (channels)
    uint32_t out_nbytes_c = output_shape[3] * out_nbytes;   // row of output (channels)
    TT_ASSERT((in_nbytes_c & (in_nbytes_c - 1)) == 0, "in_nbytes_c should be power of 2");    // in_nbytes_c is power of 2
    TT_ASSERT((out_nbytes_c & (out_nbytes_c - 1)) == 0, "out_nbytes_c should be power of 2"); // out_nbytes_c is power of 2

    uint32_t nbatch = in_n;
    TT_ASSERT(nbatch == output_shape[0], "Mismatch in N for input and output!!");

    uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;    // number of valid rows, to read
    uint32_t kernel_size_hw_padded = ceil_multiple_of(kernel_size_hw, constants::TILE_HEIGHT);
    uint32_t in_ntiles_hw = (uint32_t) ceil((float) kernel_size_hw_padded / constants::TILE_HEIGHT);
    uint32_t in_ntiles_c = (uint32_t) ceil((float) input_shape[3] / constants::TILE_WIDTH);
    uint32_t out_ntiles_hw = (uint32_t) ceil((float) output_shape[2] / constants::TILE_HEIGHT);
    uint32_t out_ntiles_c = (uint32_t) ceil((float) output_shape[3] / constants::TILE_WIDTH);

    TT_ASSERT(nblocks == 1, "Multiple blocks not yet supported");

    uint32_t out_nelems = nblocks;  // TODO [AS]: Remove hard coding after identifying optimal param val
                                    // Also ensure the calculated ncores is good
    uint32_t out_w_loop_count = ceil((float) out_w / out_nelems);

    uint32_t in_hw = in_h * in_w;
    uint32_t in_nhw = in_hw * nbatch;
    uint32_t out_hw = out_h * out_w;
    uint32_t out_nhw = out_hw * nbatch;

    // distributing out_hw across the grid
    auto grid_size = device->compute_with_storage_grid_size();
    // auto [ncores, all_cores, core_range, core_range_cliff, in_nhw_per_core, in_nhw_per_core_cliff, out_nhw_per_core, out_nhw_per_core_cliff] = max_pool_helpers::get_decomposition_nhw(grid_size, in_nhw, out_nhw);
    auto all_cores = input.shard_spec().value().grid;
    uint32_t ncores = all_cores.num_cores();
    auto core_range = all_cores;
    auto core_range_cliff = CoreRangeSet({});
    uint32_t shard_size_per_core = input.shard_spec().value().shape[0];
    uint32_t in_nhw_per_core = in_h * in_w / ncores;
    uint32_t in_nhw_per_core_cliff = 0;
    uint32_t out_nhw_per_core = out_nhw / ncores;
    uint32_t out_nhw_per_core_cliff = 0;

    uint32_t ncores_w = grid_size.x;

    // TODO: support generic nblocks
    TT_ASSERT(out_nhw_per_core % nblocks == 0, "number of sticks per core ({}) should be divisible by nblocks ({})", out_nhw_per_core, nblocks);
    // TODO: support generic values for in_nhw_per_core
    // TT_ASSERT((in_nhw_per_core & (in_nhw_per_core - 1)) == 0, "in_nhw_per_core {} needs to be power of 2!", in_nhw_per_core);

    uint32_t in_nhw_per_core_rem_mask = in_nhw_per_core - 1;    // NOTE: assuming in_nhw_per_core is power of 2

    // CBs
    uint32_t multi_buffering_factor = 2;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_id = CB::c_in1;
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    CircularBufferConfig in_scalar_cb_config = CircularBufferConfig(
                                                    in_scalar_cb_npages * in_scalar_cb_pagesize,
                                                    {{in_scalar_cb_id, in_df}})
		                                        .set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);
    auto in_scalar_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_scalar_cb_config);

    // incoming data is the input cb instead of raw l1/dram addr
    // this input shard has halo and padding inserted.
    auto raw_in_cb_id = CB::c_in2;
    // uint32_t raw_in_cb_npages = in_nhw_per_core;
    uint32_t raw_in_cb_npages = input.shard_spec().value().shape[0];
    uint32_t raw_in_cb_pagesize = in_nbytes_c;
    CircularBufferConfig raw_in_cb_config = CircularBufferConfig(
                                                raw_in_cb_npages * raw_in_cb_pagesize,
                                                {{raw_in_cb_id, in_df}})
                                            .set_page_size(raw_in_cb_id, raw_in_cb_pagesize)
                                            .set_globally_allocated_address(*input.buffer());
    auto raw_in_cb = CreateCircularBuffer(program, all_cores, raw_in_cb_config);

    // reader output == input to tilize
    uint32_t in_cb_id = CB::c_in0;          // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_page_nelems_padded = ceil_multiple_of(input_shape[3] * kernel_size_hw_padded, constants::TILE_HW);    // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_nelems_padded;
    uint32_t in_cb_npages = multi_buffering_factor * out_nelems;
    CircularBufferConfig in_cb_config = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id, in_df}})
		.set_page_size(in_cb_id, in_cb_pagesize);
    auto in_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);

    // output of tilize == input to reduce
    uint32_t in_tiled_cb_id = CB::c_intermed0;  // tiled input
    uint32_t in_tiled_cb_pagesize = tile_size(in_df);
    uint32_t in_tiled_cb_npages = in_ntiles_c * in_ntiles_hw * out_nelems;
    CircularBufferConfig in_tiled_cb_config = CircularBufferConfig(in_tiled_cb_npages * in_tiled_cb_pagesize, {{in_tiled_cb_id, in_df}})
		.set_page_size(in_tiled_cb_id, in_tiled_cb_pagesize);
    auto in_tiled_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_tiled_cb_config);

    // output of reduce == writer to write
    uint32_t out_cb_id = CB::c_out0;            // output rows in RM
    uint32_t out_cb_pagesize = tile_size(out_df);
    uint32_t out_cb_npages = out_ntiles_c * out_nelems * multi_buffering_factor;    // there is just one row of channels after reduction
    CircularBufferConfig cb_out_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
		.set_page_size(out_cb_id, out_cb_pagesize);
    auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    CBHandle cb_sharded_out = 0;
    if (output.memory_config().is_sharded()) {
        uint32_t sharded_out_cb_id = CB::c_out1;            // output rows in RM

        auto shard_shape = output.shard_spec().value().shape;
        uint32_t sharded_out_num_pages = output.shard_spec().value().shape[0];

        uint32_t sharded_out_cb_page_size = output.shard_spec().value().shape[1] * out_nbytes;    // there is just one row of channels after reduction
        CircularBufferConfig cb_sharded_out_config = CircularBufferConfig(sharded_out_num_pages * sharded_out_cb_page_size, {{sharded_out_cb_id, out_df}})
            .set_page_size(sharded_out_cb_id, sharded_out_cb_page_size).set_globally_allocated_address(*output.buffer());
        cb_sharded_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_sharded_out_config);

        log_debug(LogOp, "OUTPUT SHARD: {} {}", shard_shape[0], shard_shape[1]);
        log_debug(LogOp, "OUTPUT CB: {} {}", sharded_out_cb_page_size, sharded_out_num_pages);
    }

    uint32_t reader_indices_cb_id = CB::c_intermed1;
    uint32_t reader_indices_cb_pagesize = 4;    // uint32_t
    uint32_t reader_indices_cb_npages = shard_size_per_core;
    CircularBufferConfig cb_reader_indices_config = CircularBufferConfig(
                                                        reader_indices_cb_pagesize * reader_indices_cb_npages,
                                                        {{reader_indices_cb_id, in_df}})
		                                            .set_page_size(reader_indices_cb_id, reader_indices_cb_pagesize);
    auto reader_indices_cb = tt_metal::CreateCircularBuffer(program, all_cores, cb_reader_indices_config);

    // Construct const buffer with -INF
    // uint32_t const_buffer_size = 32;
    uint32_t const_buffer_size = input_shape[3];    // set it equal to 1 row
    auto minus_inf_const_buffer = owned_buffer::create(std::vector<bfloat16>(const_buffer_size, bfloat16(0xf7ff)));
    const Tensor minus_inf_const_tensor = Tensor(OwnedStorage{minus_inf_const_buffer},
                                                 Shape({1, 1, 1, const_buffer_size}),
                                                 DataType::BFLOAT16,
                                                 Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED,
                                                                     .buffer_type = BufferType::L1});
    auto minus_inf_const_tensor_addr = minus_inf_const_tensor.buffer()->address();

    #if 1
    {   // debug
        log_debug("raw_in_cb :: PS = {}, NP = {}", raw_in_cb_pagesize, raw_in_cb_npages);
        log_debug("in_cb :: PS = {}, NP = {}", in_cb_pagesize, in_cb_npages);
        log_debug("in_scalar_cb :: PS = {}, NP = {}", in_scalar_cb_pagesize, in_scalar_cb_npages);
        log_debug("in_tiled_cb :: PS = {}, NP = {}", in_tiled_cb_pagesize, in_tiled_cb_npages);
        log_debug("out_cb :: PS = {}, NP = {}", out_cb_pagesize, out_cb_npages);
        log_debug("in_addr: {}", src_dram_buffer->address());
        log_debug("out_addr: {}", dst_dram_buffer->address());
        log_debug("nbatch: {}", nbatch);
        log_debug("kernel_size_h: {}", kernel_size_h);
        log_debug("kernel_size_w: {}", kernel_size_w);
        log_debug("kernel_size_hw: {}", kernel_size_hw);
        log_debug("kernel_size_hw_padded: {}", kernel_size_hw_padded);
        log_debug("stride_h: {}", stride_h);
        log_debug("stride_w: {}", stride_w);
        log_debug("pad_h: {}", pad_h);
        log_debug("pad_w: {}", pad_w);
        log_debug("out_h: {}", out_h);
        log_debug("out_w: {}", out_w);
        log_debug("out_hw: {}", output_shape[2]);
        log_debug("out_c: {}", output_shape[3]);
        log_debug("in_nbytes_c: {}", in_nbytes_c);
        log_debug("out_nbytes_c: {}", out_nbytes_c);
        log_debug("in_h: {}", in_h);
        log_debug("in_w: {}", in_w);
        log_debug("in_hw_padded: {}", in_hw);
        log_debug("in_c: {}", input_shape[3]);
        log_debug("out_hw_padded: {}", out_hw);
        log_debug("out_ntiles_hw: {}", out_ntiles_hw);
        log_debug("out_ntiles_c: {}", out_ntiles_c);
        log_debug("out_nelems: {}", out_nelems);
        log_debug("out_w_loop_count: {}", out_w_loop_count);
        log_debug("out_hw: {}", out_hw);
        log_debug("minus_inf_const_tensor_addr: {}", minus_inf_const_tensor_addr);
        log_debug("minus_inf_const_tensor_size: {}", minus_inf_const_buffer.size());
        log_debug("ncores: {}", ncores);
        log_debug("in_nhw_per_core: {}", in_nhw_per_core);
        log_debug("out_nhw_per_core: {}", out_nhw_per_core);
        log_debug("in_nhw_per_core_rem_mask: {}", in_nhw_per_core_rem_mask);
        log_debug("is_in_sharded: {}", input.memory_config().is_sharded());
        log_debug("is_out_sharded: {}", output.memory_config().is_sharded());
    }
    #endif

    const uint32_t reader_noc = 0;
    const uint32_t writer_noc = 1;

    max_pool_helpers::init_neighbor_noc_xy_mapping(grid_size, reader_noc);

    /**
     * Reader Kernel: input rows -> input cb
     */
    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t*>(&one);
    std::vector<uint32_t> reader_ct_args = {input.memory_config().buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            out_mem_config.buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            bf16_one_u32,
                                            out_nelems,
                                            static_cast<uint32_t>(((in_nbytes_c & (in_nbytes_c - 1)) == 0) ? 1 : 0),    // is in_nbytes_c power of 2
                                            stride_h,
                                            stride_w,
                                            reader_noc,
                                            writer_noc};
    uint32_t in_log_base_2_of_page_size = (uint32_t) std::log2((float) in_nbytes_c);
    std::vector<uint32_t> reader_rt_args = {src_dram_buffer->address(),         // 0
                                            dst_dram_buffer->address(),
                                            kernel_size_h,
                                            kernel_size_w,
                                            kernel_size_hw,
                                            kernel_size_hw_padded,              // 5
                                            stride_h,
                                            stride_w,
                                            pad_h,
                                            pad_w,
                                            out_h,                              // 10
                                            out_w,
                                            output_shape[2],
                                            output_shape[3],
                                            in_nbytes_c,
                                            out_nbytes_c,                       // 15
                                            in_h,
                                            in_w,
                                            input_shape[2],
                                            input_shape[3],
                                            out_ntiles_hw,                      // 20
                                            out_ntiles_c,
                                            in_cb_pagesize,
                                            out_cb_pagesize,
                                            in_cb_page_nelems_padded,
                                            out_w_loop_count,                   // 25
                                            in_log_base_2_of_page_size,
                                            nbatch,
                                            in_hw,
                                            out_hw,
                                            // these are set later in the following
                                            0,          // start_out_h_i        // 30
                                            0,          // end_out_h_i
                                            0,          // base_start_h
                                            0,          // start_out_row_id
                                            minus_inf_const_tensor_addr,
                                            const_buffer_size * in_nbytes,      // 35
                                            (in_cb_page_nelems_padded * out_nelems * 2) >> 5,    // TODO: generalize num rows to fill in in_cb
                                            0,          // core_offset_in_row_id
                                            0,          // core_out_w_i_start
                                            0,          // core_out_h_i_start
                                            out_nhw_per_core,    // nsticks_per_core    // 40
                                            0,          // core_offset_out_row_id
                                            out_nhw_per_core / nblocks,     // loop count with blocks
                                            // the following are for sharded input
                                            0,                  // 43: local_out_stick_start
                                            out_hw,             // out_nsticks_per_batch
                                            0,                  // local_in_stick_start // 45
                                            0,                  // local_in_stick_end
                                            in_hw,              // in_nsticks_per_batch
                                            in_nhw_per_core,    // in_nsticks_per_core
                                            0,                  // has_left
                                            0,                  // left_noc_x           // 50
                                            0,                  // left_noc_y
                                            0,                  // has_right
                                            0,                  // right_noc_x
                                            0,                  // right_noc_y
                                            in_nhw_per_core_rem_mask,                   // 55
                                            0,                  // 56: has_left_left,
                                            0,                  // left_left_noc_x,
                                            0,                  // left_left_noc_y,
                                            0,                  // has_right_right,
                                            0,                  // right_right_noc_x,   // 60
                                            0,                  // right_right_noc_y,
                                            0,                  // left_in_stick_start,
                                            0,                  // right_in_stick_end,
                                            0,                  // my_core
                                            0,                  // initial_skip         // 65
                                            0,                  // partial_first_row_nsticks
                                            0,                  // partial_first_row_skip
                                            0,                  // partial_top_image_nrows
                                            0,                  // partial_top_image_skip
                                            0,                  // full_nimages         // 70
                                            0,                  // full_images_skip
                                            0,                  // partial_bottom_image_nrows
                                            0,                  // partial_last_row_nsticks
                                            0,                  // start_stick
                                            0,                  // 75
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,                  // 80
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,                  // 85
                                            0,                  // in_skip_after_each_full_row
                                            0,                  // skip_after_each_stick
                                            };
    auto reader_config = ReaderDataMovementConfig{.compile_args = reader_ct_args};
    std::string reader_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/dataflow/reader_max_pool_2d_multi_core_sharded_with_halo.cpp");
    auto reader_kernel = CreateKernel(program,
                                                  reader_kernel_fname,
                                                  all_cores,
                                                  reader_config);

    /**
     * Writer Kernel: output cb -> output rows
     */
    std::map<string, string> writer_defines;
    writer_defines["SHARDED_OUT"] = "1";
    std::vector<uint32_t> writer_ct_args = reader_ct_args;
    auto writer_config = WriterDataMovementConfig{.compile_args = writer_ct_args, .defines = writer_defines};
    std::string writer_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/dataflow/writer_max_pool_2d_multi_core.cpp");
    auto writer_kernel = CreateKernel(program,
                                                  writer_kernel_fname,
                                                  all_cores,
                                                  writer_config);

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block -> output cb
     */
    std::vector<uint32_t> compute_ct_args = {in_ntiles_hw,
                                            in_ntiles_c,
                                            in_ntiles_hw * in_ntiles_c,
                                            kernel_size_hw_padded,
                                            out_h,
                                            out_w,
                                            (uint32_t) ceil((float) output_shape[2] / constants::TILE_HEIGHT),
                                            (uint32_t) ceil((float) output_shape[3] / constants::TILE_WIDTH),
                                            out_nelems,
                                            out_w_loop_count,
                                            nbatch,
                                            out_nhw_per_core,
                                            out_nhw_per_core,
                                            out_nhw_per_core / nblocks,     // loop count with blocks
                                            };
    auto compute_ct_args_cliff = compute_ct_args;
    auto reduce_op = ReduceOpMath::MAX;
    auto reduce_dim = ReduceOpDim::H;
    auto compute_config = ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                        .fp32_dest_acc_en = false,
                                        .math_approx_mode = false,
                                        .compile_args = compute_ct_args,
                                        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    std::string compute_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/compute/max_pool_multi_core.cpp");
    auto compute_kernel = CreateKernel(program,
                                              compute_kernel_fname,
                                              core_range,
                                              compute_config);

    if (out_nhw_per_core_cliff > 0) {
        TT_ASSERT(false, "The cliff core case is not yet handled"); // TODO
        // there is a cliff core
        compute_ct_args_cliff[11] = out_nhw_per_core_cliff;
        auto compute_config_cliff = ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                                    .fp32_dest_acc_en = false,
                                                    .math_approx_mode = false,
                                                    .compile_args = compute_ct_args_cliff,
                                                    .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
        auto compute_kernel_cliff = CreateKernel(program,
                                                        compute_kernel_fname,
                                                        core_range_cliff,
                                                        compute_config);
    }

    PoolConfig pc {
        .in_w = in_w,
        .in_h = in_h,
        .out_w = out_w,
        .out_h = out_h,
        .stride_w = stride_w,
        .stride_h = stride_h,
        .pad_w = pad_w,
        .pad_h = pad_h,
        .window_w = kernel_size_w,
        .window_h = kernel_size_h,
        .dilation_w = dilation_w,
        .dilation_h = dilation_h
    };

    // calculate and set the start/end h_i for each core
    // for all but last core (cliff)
    uint32_t core_out_h_i = 0;
    uint32_t core_out_w_i = 0;
    int32_t curr_start_h = - pad_h;
    if (out_nhw_per_core_cliff > 0) {
        // TODO: ... not yet handled
        TT_ASSERT(false, "The cliff core case is not yet handled"); // TODO
    } else {
        uint32_t core_batch_offset = 0;
        uint32_t curr_out_stick_id = 0; // track output sticks with batch folded in
        int32_t curr_in_stick_id = 0; // track input sticks with batch folded in
        uint32_t core_out_w_i_start = 0;
        uint32_t core_out_h_i_start = 0;
        for (int32_t i = 0; i < ncores; ++ i) {
            CoreCoord core(i % ncores_w, i / ncores_w); // logical
            reader_rt_args[37] = (curr_in_stick_id / in_hw) * in_hw;
            core_out_w_i_start = curr_out_stick_id % out_w;
            core_out_h_i_start = (curr_out_stick_id / out_w) % out_h;
            reader_rt_args[38] = core_out_w_i_start;
            reader_rt_args[39] = core_out_h_i_start;
            reader_rt_args[41] = curr_out_stick_id;

            reader_rt_args[43] = curr_out_stick_id;
            reader_rt_args[45] = curr_in_stick_id;
            reader_rt_args[46] = curr_in_stick_id + in_nhw_per_core;

            reader_rt_args[64] = i; // my_core

            CoreCoord noc_core = core;  // physical
            if (reader_noc == 0) {
                noc_core.x += 1;
                noc_core.y += 1;
                if (noc_core.y > 5) {
                    noc_core.y += 1;
                }
            } else {
                TT_ASSERT(false, "reader noc == 1 not yet handled");
            }
            if (max_pool_helpers::left_neighbor_noc_xy.count(noc_core) > 0) {
                CoreCoord left_noc = max_pool_helpers::left_neighbor_noc_xy.at(noc_core);
                reader_rt_args[49] = 1;
                reader_rt_args[50] = (uint32_t) left_noc.x;
                reader_rt_args[51] = (uint32_t) left_noc.y;
                // log_debug("Local NOC: ({},{}), left: ({},{})", noc_core.x, noc_core.y, left_noc.x, left_noc.y);

                // left-left
                if (max_pool_helpers::left_neighbor_noc_xy.count(left_noc) > 0) {
                    CoreCoord left_left_noc = max_pool_helpers::left_neighbor_noc_xy.at(left_noc);
                    reader_rt_args[56] = 1;
                    reader_rt_args[57] = (uint32_t) left_left_noc.x;
                    reader_rt_args[58] = (uint32_t) left_left_noc.y;
                    reader_rt_args[62] = (uint32_t) (curr_in_stick_id - (int32_t) in_nhw_per_core);
                } else {
                    reader_rt_args[56] = 0;
                }
            } else {
                reader_rt_args[49] = 0;
            }
            if (max_pool_helpers::right_neighbor_noc_xy.count(noc_core) > 0) {
                CoreCoord right_noc = max_pool_helpers::right_neighbor_noc_xy.at(noc_core);
                reader_rt_args[52] = 1;
                reader_rt_args[53] = (uint32_t) right_noc.x;
                reader_rt_args[54] = (uint32_t) right_noc.y;
                // log_debug("Local NOC: ({},{}), right: ({},{})", noc_core.x, noc_core.y, right_noc.x, right_noc.y);

                // right-right
                if (max_pool_helpers::right_neighbor_noc_xy.count(right_noc) > 0) {
                    CoreCoord right_right_noc = max_pool_helpers::right_neighbor_noc_xy.at(right_noc);
                    reader_rt_args[59] = 1;
                    reader_rt_args[60] = (uint32_t) right_right_noc.x;
                    reader_rt_args[61] = (uint32_t) right_right_noc.y;
                    reader_rt_args[63] = (uint32_t) (curr_in_stick_id + 2 * in_nhw_per_core);
                } else {
                    reader_rt_args[59] = 0;
                }
            } else {
                reader_rt_args[52] = 0;
            }

            // given the start,end out stick id, calculate the start,end in stick id.
            int32_t start_out_stick_id = curr_out_stick_id;
            int32_t start_batch_i = start_out_stick_id / out_hw;
            int32_t start_out_w_i = start_out_stick_id % out_w;
            int32_t start_out_h_i = (start_out_stick_id % out_hw) / out_w;
            int32_t start_in_w_i = start_out_w_i * stride_w;
            int32_t start_in_h_i = start_out_h_i * stride_h;
            int32_t start_center_in_stick_id = start_in_h_i * in_w + start_in_w_i;

            int32_t end_out_stick_id = start_out_stick_id + out_nhw_per_core - 1;
            int32_t end_batch_i = end_out_stick_id / out_hw;
            int32_t end_out_w_i = end_out_stick_id % out_w;
            int32_t end_out_h_i = (end_out_stick_id % out_hw) / out_w;
            int32_t end_in_w_i = end_out_w_i * stride_w;
            int32_t end_in_h_i = end_out_h_i * stride_h;
            int32_t end_center_in_stick_id = end_in_h_i * in_w + end_in_w_i + (end_batch_i - start_batch_i) * in_hw;

            auto [in_sc, out_sc] = get_inout_shard_specs(start_out_stick_id, start_out_stick_id + out_nhw_per_core, pc);

            if (1) {
                uint32_t in_w_padded = in_w + 2 * pad_w;
                log_debug(LogOp, "++++ CORE: {}", i);
                log_debug(LogOp, " + out_stick_id range: {} {}", start_out_stick_id, end_out_stick_id);
                log_debug(LogOp, " + start_out: {} {}", start_out_w_i, start_out_h_i);
                log_debug(LogOp, " + end_out: {} {}", end_out_w_i, end_out_h_i);
                log_debug(LogOp, " + partial_first_row_nsticks: {}", out_sc.first_partial_right_aligned_row_width);
                log_debug(LogOp, " + partial_top_image_nrows: {}", out_sc.first_partial_image_num_rows);
                log_debug(LogOp, " + full_nimages: {}", out_sc.num_full_images);
                log_debug(LogOp, " + partial_bottom_image_nrows: {}", out_sc.last_partial_image_num_rows);
                log_debug(LogOp, " + partial_last_row_nsticks: {}", out_sc.last_partial_left_aligned_row_width);
                log_debug(LogOp, " + skip_after_partial_right_aligned_row: {}", out_sc.skip_after_partial_right_aligned_row);
                log_debug(LogOp, " + skip_after_first_partial_image_row: {}", out_sc.skip_after_first_partial_image_row);
                log_debug(LogOp, " + skip_after_full_image: {}", out_sc.skip_after_full_image);
                log_debug(LogOp, " + initial_skip: {}", out_sc.initial_skip);
                log_debug(LogOp, " + start_stick: {}", out_sc.start_stick);
                log_debug(LogOp, " + ++++++++++++++++++++++");
                log_debug(LogOp, " + partial_first_row_nsticks: {}", in_sc.first_partial_right_aligned_row_width);
                log_debug(LogOp, " + partial_top_image_nrows: {}", in_sc.first_partial_image_num_rows);
                log_debug(LogOp, " + full_nimages: {}", in_sc.num_full_images);
                log_debug(LogOp, " + partial_bottom_image_nrows: {}", in_sc.last_partial_image_num_rows);
                log_debug(LogOp, " + partial_last_row_nsticks: {}", in_sc.last_partial_left_aligned_row_width);
                log_debug(LogOp, " + skip_after_partial_right_aligned_row: {} ({})", in_sc.skip_after_partial_right_aligned_row, 2 * pad_w + (stride_h - 1) * in_w_padded);
                log_debug(LogOp, " + skip_after_first_partial_image_row: {} ({})", in_sc.skip_after_first_partial_image_row, pad_h * in_w_padded);
                log_debug(LogOp, " + skip_after_full_image: {} ({})", in_sc.skip_after_full_image, pad_h * in_w_padded);
                log_debug(LogOp, " + skip_after_each_full_row: {} ({})", in_sc.skip_after_each_full_row, 2 * pad_w + (stride_h - 1) * in_w_padded);
                log_debug(LogOp, " + skip_after_each_stick: {} ({})", in_sc.skip_after_each_stick, stride_w);
                log_debug(LogOp, " + initial_skip: {}", in_sc.initial_skip);
                log_debug(LogOp, " + start_stick: {}", in_sc.start_stick);
            }

            reader_rt_args[65] = out_sc.initial_skip;
            reader_rt_args[66] = out_sc.first_partial_right_aligned_row_width;
            reader_rt_args[67] = out_sc.skip_after_partial_right_aligned_row;
            reader_rt_args[68] = out_sc.first_partial_image_num_rows;
            reader_rt_args[69] = out_sc.skip_after_first_partial_image_row;
            reader_rt_args[70] = out_sc.num_full_images;
            reader_rt_args[71] = out_sc.skip_after_full_image;
            reader_rt_args[72] = out_sc.last_partial_image_num_rows;
            reader_rt_args[73] = out_sc.last_partial_left_aligned_row_width;
            reader_rt_args[74] = out_sc.start_stick;

            reader_rt_args[75] = in_sc.start_stick;
            reader_rt_args[76] = in_sc.first_partial_right_aligned_row_width;
            reader_rt_args[77] = in_sc.first_partial_image_num_rows;
            reader_rt_args[78] = in_sc.num_full_images;
            reader_rt_args[79] = in_sc.last_partial_image_num_rows;
            reader_rt_args[80] = in_sc.last_partial_left_aligned_row_width;
            reader_rt_args[81] = in_sc.initial_skip;
            reader_rt_args[82] = in_sc.skip_after_stick;
            reader_rt_args[83] = in_sc.skip_after_partial_right_aligned_row;
            reader_rt_args[84] = in_sc.skip_after_first_partial_image_row;
            reader_rt_args[85] = in_sc.skip_after_full_image;
            reader_rt_args[86] = in_sc.skip_after_each_full_row;
            reader_rt_args[87] = in_sc.skip_after_each_stick;

            SetRuntimeArgs(program, reader_kernel, core, reader_rt_args);
            std::vector<uint32_t> writer_rt_args = reader_rt_args;
            SetRuntimeArgs(program, writer_kernel, core, writer_rt_args);

            curr_out_stick_id += out_nhw_per_core;
            curr_in_stick_id += in_nhw_per_core;
        }
    }

    auto override_runtime_arguments_callback = [
            reader_kernel, writer_kernel, raw_in_cb, cb_sharded_out, ncores, ncores_w
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer = input_tensors.at(0).buffer();
        bool input_sharded = input_tensors.at(0).is_sharded();

        auto dst_buffer = output_tensors.at(0).buffer();
        bool out_sharded = output_tensors.at(0).is_sharded();

        for (uint32_t i = 0; i < ncores; ++ i) {
            CoreCoord core{i % ncores_w, i / ncores_w };
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
            }
            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
            }
        }
        if (input_sharded) {
            UpdateDynamicCircularBufferAddress(program, raw_in_cb, *src_buffer);
        }
        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_sharded_out, *dst_buffer);
        }
    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

// this version uses distribution along height = N * H * W
operation::ProgramWithCallbacks max_pool_2d_multi_core_sharded_with_halo_v2(const Tensor &input, const Tensor &reader_indices,
                                                                        Tensor& output,
                                                                        uint32_t in_n, uint32_t in_h, uint32_t in_w,
                                                                        uint32_t out_h, uint32_t out_w,
                                                                        uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                                        uint32_t stride_h, uint32_t stride_w,
                                                                        uint32_t pad_h, uint32_t pad_w,
                                                                        uint32_t dilation_h, uint32_t dilation_w,
                                                                        const MemoryConfig& out_mem_config,
                                                                        uint32_t nblocks) {
    Program program = CreateProgram();

    // This should allocate a DRAM buffer on the device
    Device *device = input.device();
    Buffer *src_dram_buffer = input.buffer();
    Buffer *reader_indices_buffer = reader_indices.buffer();
    Buffer *dst_dram_buffer = output.buffer();

    Shape input_shape = input.shape();
    Shape output_shape = output.shape();

    // NOTE: input is assumed to be in {N, 1, H * W, C }

    DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);
    uint32_t in_nbytes_c = input_shape[3] * in_nbytes;      // row of input (channels)
    uint32_t out_nbytes_c = output_shape[3] * out_nbytes;   // row of output (channels)
    TT_ASSERT((in_nbytes_c & (in_nbytes_c - 1)) == 0, "in_nbytes_c should be power of 2");    // in_nbytes_c is power of 2
    TT_ASSERT((out_nbytes_c & (out_nbytes_c - 1)) == 0, "out_nbytes_c should be power of 2"); // out_nbytes_c is power of 2

    DataFormat indices_df = DataFormat::RawUInt16; //datatype_to_dataformat_converter(reader_indices.dtype());
    uint32_t indices_nbytes = datum_size(indices_df);

    uint32_t nbatch = in_n;
    TT_ASSERT(nbatch == output_shape[0], "Mismatch in N for input and output!!");

    uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;    // number of valid rows, to read
    uint32_t kernel_size_hw_padded = ceil_multiple_of(kernel_size_hw, constants::TILE_HEIGHT);
    uint32_t in_ntiles_hw = (uint32_t) ceil((float) kernel_size_hw_padded / constants::TILE_HEIGHT);
    uint32_t in_ntiles_c = (uint32_t) ceil((float) input_shape[3] / constants::TILE_WIDTH);
    uint32_t out_ntiles_hw = (uint32_t) ceil((float) output_shape[2] / constants::TILE_HEIGHT);
    uint32_t out_ntiles_c = (uint32_t) ceil((float) output_shape[3] / constants::TILE_WIDTH);

    TT_ASSERT(nblocks == 1, "Multiple blocks not yet supported");

    uint32_t out_nelems = nblocks;  // TODO [AS]: Remove hard coding after identifying optimal param val
                                    // Also ensure the calculated ncores is good
    uint32_t out_w_loop_count = ceil((float) out_w / out_nelems);

    uint32_t in_hw = in_h * in_w;
    uint32_t in_nhw = in_hw * nbatch;
    uint32_t out_hw = out_h * out_w;
    uint32_t out_nhw = out_hw * nbatch;

    // distributing out_hw across the grid
    auto grid_size = device->compute_with_storage_grid_size();
    auto all_cores = input.shard_spec().value().grid;
    uint32_t ncores = all_cores.num_cores();
    auto core_range = all_cores;
    auto core_range_cliff = CoreRangeSet({});
    uint32_t shard_size_per_core = input.shard_spec().value().shape[0];
    uint32_t in_nhw_per_core = in_h * in_w / ncores;
    uint32_t in_nhw_per_core_cliff = 0;
    uint32_t out_nhw_per_core = out_nhw / ncores;

    uint32_t ncores_w = grid_size.x;

    // TODO: support generic nblocks
    TT_ASSERT(out_nhw_per_core % nblocks == 0, "number of sticks per core ({}) should be divisible by nblocks ({})", out_nhw_per_core, nblocks);
    // TODO: support generic values for in_nhw_per_core
    // TT_ASSERT((in_nhw_per_core & (in_nhw_per_core - 1)) == 0, "in_nhw_per_core {} needs to be power of 2!", in_nhw_per_core);

    uint32_t in_nhw_per_core_rem_mask = in_nhw_per_core - 1;    // NOTE: assuming in_nhw_per_core is power of 2

    // CBs
    uint32_t multi_buffering_factor = 2;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_id = CB::c_in1;
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    CircularBufferConfig in_scalar_cb_config = CircularBufferConfig(
                                                    in_scalar_cb_npages * in_scalar_cb_pagesize,
                                                    {{in_scalar_cb_id, in_df}})
		                                        .set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);
    auto in_scalar_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_scalar_cb_config);

    // incoming data is the input cb instead of raw l1/dram addr
    // this input shard has halo and padding inserted.
    auto raw_in_cb_id = CB::c_in2;
    // uint32_t raw_in_cb_npages = in_nhw_per_core;
    uint32_t raw_in_cb_npages = input.shard_spec().value().shape[0];
    uint32_t raw_in_cb_pagesize = in_nbytes_c;
    CircularBufferConfig raw_in_cb_config = CircularBufferConfig(
                                                raw_in_cb_npages * raw_in_cb_pagesize,
                                                {{raw_in_cb_id, in_df}})
                                            .set_page_size(raw_in_cb_id, raw_in_cb_pagesize)
                                            .set_globally_allocated_address(*input.buffer());
    auto raw_in_cb = CreateCircularBuffer(program, all_cores, raw_in_cb_config);

    // reader indices
    auto in_reader_indices_cb_id = CB::c_in3;
    uint32_t in_reader_indices_cb_pagesize = out_nhw_per_core * indices_nbytes;
    uint32_t in_reader_indices_cb_npages = 1;
    CircularBufferConfig in_reader_indices_cb_config = CircularBufferConfig(
                                                            in_reader_indices_cb_npages * in_reader_indices_cb_pagesize,
                                                            {{in_reader_indices_cb_id, indices_df}})
                                                        .set_page_size(in_reader_indices_cb_id, in_reader_indices_cb_pagesize)
                                                        .set_globally_allocated_address(*reader_indices_buffer);
    auto in_reader_indices_cb = CreateCircularBuffer(program, all_cores, in_reader_indices_cb_config);

    // reader output == input to tilize
    uint32_t in_cb_id = CB::c_in0;          // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_page_nelems_padded = ceil_multiple_of(input_shape[3] * kernel_size_hw_padded, constants::TILE_HW);    // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_nelems_padded;
    uint32_t in_cb_npages = multi_buffering_factor * out_nelems;
    CircularBufferConfig in_cb_config = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id, in_df}})
		.set_page_size(in_cb_id, in_cb_pagesize);
    auto in_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);

    // output of tilize == input to reduce
    uint32_t in_tiled_cb_id = CB::c_intermed0;  // tiled input
    uint32_t in_tiled_cb_pagesize = tile_size(in_df);
    uint32_t in_tiled_cb_npages = in_ntiles_c * in_ntiles_hw * out_nelems;
    CircularBufferConfig in_tiled_cb_config = CircularBufferConfig(in_tiled_cb_npages * in_tiled_cb_pagesize, {{in_tiled_cb_id, in_df}})
		.set_page_size(in_tiled_cb_id, in_tiled_cb_pagesize);
    auto in_tiled_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_tiled_cb_config);

    // output of reduce == writer to write
    uint32_t out_cb_id = CB::c_out0;            // output rows in RM
    uint32_t out_cb_pagesize = tile_size(out_df);
    uint32_t out_cb_npages = out_ntiles_c * out_nelems * multi_buffering_factor;    // there is just one row of channels after reduction
    CircularBufferConfig cb_out_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
		.set_page_size(out_cb_id, out_cb_pagesize);
    auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    CBHandle cb_sharded_out = 0;
    TT_FATAL(output.memory_config().is_sharded());

    uint32_t sharded_out_cb_id = CB::c_out1;            // output rows in RM

    auto shard_shape = output.shard_spec().value().shape;
    uint32_t sharded_out_num_pages = output.shard_spec().value().shape[0];

    uint32_t sharded_out_cb_page_size = output.shard_spec().value().shape[1] * out_nbytes;    // there is just one row of channels after reduction
    CircularBufferConfig cb_sharded_out_config = CircularBufferConfig(sharded_out_num_pages * sharded_out_cb_page_size, {{sharded_out_cb_id, out_df}})
        .set_page_size(sharded_out_cb_id, sharded_out_cb_page_size).set_globally_allocated_address(*output.buffer());
    cb_sharded_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_sharded_out_config);

    log_debug(LogOp, "OUTPUT SHARD: {} {}", shard_shape[0], shard_shape[1]);
    log_debug(LogOp, "OUTPUT CB: {} {}", sharded_out_cb_page_size, sharded_out_num_pages);

    // Construct const buffer with -INF
    // uint32_t const_buffer_size = 32;
    uint32_t const_buffer_size = input_shape[3];    // set it equal to 1 row
    auto minus_inf_const_buffer = owned_buffer::create(std::vector<bfloat16>(const_buffer_size, bfloat16(0xf7ff)));
    const Tensor minus_inf_const_tensor = Tensor(OwnedStorage{minus_inf_const_buffer},
                                                 Shape({1, 1, 1, const_buffer_size}),
                                                 DataType::BFLOAT16,
                                                 Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED,
                                                                     .buffer_type = BufferType::L1});
    auto minus_inf_const_tensor_addr = minus_inf_const_tensor.buffer()->address();

    #if 1
    {   // debug
        log_debug("raw_in_cb :: PS = {}, NP = {}", raw_in_cb_pagesize, raw_in_cb_npages);
        log_debug("in_cb :: PS = {}, NP = {}", in_cb_pagesize, in_cb_npages);
        log_debug("in_reader_indices_cb :: PS = {}, NP = {}", in_reader_indices_cb_pagesize, in_reader_indices_cb_npages);
        log_debug("in_scalar_cb :: PS = {}, NP = {}", in_scalar_cb_pagesize, in_scalar_cb_npages);
        log_debug("in_tiled_cb :: PS = {}, NP = {}", in_tiled_cb_pagesize, in_tiled_cb_npages);
        log_debug("out_cb :: PS = {}, NP = {}", out_cb_pagesize, out_cb_npages);
        log_debug("in_addr: {}", src_dram_buffer->address());
        log_debug("in_reader_indices_addr: {}", reader_indices_buffer->address());
        log_debug("out_addr: {}", dst_dram_buffer->address());
        log_debug("nbatch: {}", nbatch);
        log_debug("kernel_size_h: {}", kernel_size_h);
        log_debug("kernel_size_w: {}", kernel_size_w);
        log_debug("kernel_size_hw: {}", kernel_size_hw);
        log_debug("kernel_size_hw_padded: {}", kernel_size_hw_padded);
        log_debug("stride_h: {}", stride_h);
        log_debug("stride_w: {}", stride_w);
        log_debug("pad_h: {}", pad_h);
        log_debug("pad_w: {}", pad_w);
        log_debug("out_h: {}", out_h);
        log_debug("out_w: {}", out_w);
        log_debug("out_hw: {}", output_shape[2]);
        log_debug("out_c: {}", output_shape[3]);
        log_debug("in_nbytes_c: {}", in_nbytes_c);
        log_debug("out_nbytes_c: {}", out_nbytes_c);
        log_debug("in_h: {}", in_h);
        log_debug("in_w: {}", in_w);
        log_debug("in_hw_padded: {}", in_hw);
        log_debug("in_c: {}", input_shape[3]);
        log_debug("out_hw_padded: {}", out_hw);
        log_debug("out_ntiles_hw: {}", out_ntiles_hw);
        log_debug("out_ntiles_c: {}", out_ntiles_c);
        log_debug("out_nelems: {}", out_nelems);
        log_debug("out_w_loop_count: {}", out_w_loop_count);
        log_debug("out_hw: {}", out_hw);
        log_debug("minus_inf_const_tensor_addr: {}", minus_inf_const_tensor_addr);
        log_debug("minus_inf_const_tensor_size: {}", minus_inf_const_buffer.size());
        log_debug("ncores: {}", ncores);
        log_debug("in_nhw_per_core: {}", in_nhw_per_core);
        log_debug("out_nhw_per_core: {}", out_nhw_per_core);
        log_debug("in_nhw_per_core_rem_mask: {}", in_nhw_per_core_rem_mask);
        log_debug("is_in_sharded: {}", input.memory_config().is_sharded());
        log_debug("is_out_sharded: {}", output.memory_config().is_sharded());
    }
    #endif

    const uint32_t reader_noc = 0;
    const uint32_t writer_noc = 1;

    max_pool_helpers::init_neighbor_noc_xy_mapping(grid_size, reader_noc);

    /**
     * Reader Kernel: input rows -> input cb
     */
    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t*>(&one);
    std::vector<uint32_t> reader_ct_args = {input.memory_config().buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            out_mem_config.buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            bf16_one_u32,
                                            nblocks};
    uint32_t in_nbytes_c_log2 = (uint32_t) std::log2((float) in_nbytes_c);
    std::vector<uint32_t> reader_rt_args = {
                                            out_nhw_per_core, // TODO: check ...
                                            kernel_size_h,
                                            kernel_size_w,
                                            pad_w,
                                            in_nbytes_c,
                                            in_nbytes_c_log2,
                                            in_w,
                                            (in_cb_page_nelems_padded * out_nelems * 2) >> 5,    // TODO: generalize num rows to fill in in_cb
                                            };
    auto reader_config = DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                            .noc = NOC::RISCV_0_default,
                                            .compile_args = reader_ct_args};
    std::string reader_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/dataflow/reader_max_pool_2d_multi_core_sharded_with_halo_v2.cpp");
    auto reader_kernel = CreateKernel(program,
                                        reader_kernel_fname,
                                        all_cores,
                                        reader_config);

    /**
     * Writer Kernel: output cb -> output rows
     */
    std::map<string, string> writer_defines;
    writer_defines["SHARDED_OUT"] = "1";
    std::vector<uint32_t> writer_ct_args = reader_ct_args;
    std::vector<uint32_t> writer_rt_args = {
                                            dst_dram_buffer->address(),
                                            out_nbytes_c,                       // 15
                                            out_ntiles_c,
                                            out_nhw_per_core,    // nsticks_per_core    // 40
                                            0,          // TODO: core_offset_out_row_id
                                            out_nhw_per_core / nblocks,     // loop count with blocks
                                            };
    auto writer_config = DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                            .noc = NOC::RISCV_1_default,
                                            .compile_args = writer_ct_args,
                                            .defines = writer_defines};
    std::string writer_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/dataflow/writer_max_pool_2d_multi_core_v2.cpp");
    auto writer_kernel = CreateKernel(program,
                                        writer_kernel_fname,
                                        all_cores,
                                        writer_config);

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block -> output cb
     */
    std::vector<uint32_t> compute_ct_args = {in_ntiles_hw,
                                            in_ntiles_c,
                                            in_ntiles_hw * in_ntiles_c,
                                            kernel_size_hw_padded,
                                            out_h,
                                            out_w,
                                            (uint32_t) ceil((float) output_shape[2] / constants::TILE_HEIGHT),
                                            (uint32_t) ceil((float) output_shape[3] / constants::TILE_WIDTH),
                                            out_nelems,
                                            out_w_loop_count,
                                            nbatch,
                                            out_nhw_per_core,
                                            out_nhw_per_core,
                                            out_nhw_per_core / nblocks,     // loop count with blocks
                                            };
    auto compute_ct_args_cliff = compute_ct_args;
    auto reduce_op = ReduceOpMath::MAX;
    auto reduce_dim = ReduceOpDim::H;
    auto compute_config = ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                        .fp32_dest_acc_en = false,
                                        .math_approx_mode = false,
                                        .compile_args = compute_ct_args,
                                        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    std::string compute_kernel_fname("tt_eager/tt_dnn/op_library/pool/kernels/compute/max_pool_multi_core.cpp");
    auto compute_kernel = CreateKernel(program,
                                              compute_kernel_fname,
                                              core_range,
                                              compute_config);

    /**
     * Set runtime args
     */

    SetRuntimeArgs(program, reader_kernel, all_cores, reader_rt_args);

    uint32_t curr_out_stick_id = 0; // track output sticks with batch folded in
    for (int32_t i = 0; i < ncores; ++ i) {
        CoreCoord core(i % ncores_w, i / ncores_w); // logical
        writer_rt_args[4] = curr_out_stick_id;
        SetRuntimeArgs(program, writer_kernel, core, writer_rt_args);
        curr_out_stick_id += out_nhw_per_core;
    }

    auto override_runtime_arguments_callback = [
            reader_kernel, writer_kernel, raw_in_cb, in_reader_indices_cb, cb_sharded_out, ncores, ncores_w
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer = input_tensors.at(0).buffer();
        bool input_sharded = input_tensors.at(0).is_sharded();
        auto reader_indices_buffer = input_tensors.at(1).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();
        bool out_sharded = output_tensors.at(0).is_sharded();

        for (uint32_t i = 0; i < ncores; ++ i) {
            CoreCoord core{i % ncores_w, i / ncores_w };
            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
        if (input_sharded) {
            UpdateDynamicCircularBufferAddress(program, raw_in_cb, *src_buffer);
            UpdateDynamicCircularBufferAddress(program, in_reader_indices_cb, *reader_indices_buffer);
        }
        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_sharded_out, *dst_buffer);
        }
    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

} // namespace tt_metal
} // namespace tt
