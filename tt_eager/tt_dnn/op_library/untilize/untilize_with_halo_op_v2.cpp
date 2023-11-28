// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <algorithm>

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

namespace untilize_with_halo_v2_helpers {

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

} // namespace untilize_with_halo_v2_helpers

operation::ProgramWithCallbacks untilize_with_halo_multi_core_v2(
    const Tensor& input_tensor,
    const Tensor& local_pad_start_and_size,
    const Tensor& ll_data_start_and_size,
    const Tensor& l_data_start_and_size,
    const Tensor& local_data_start_and_size,
    const Tensor& r_data_start_and_size,
    const Tensor& rr_data_start_and_size,
    const uint32_t pad_val,
    const uint32_t ncores_height,
    const uint32_t max_out_nsticks_per_core,
    const std::vector<int32_t>& local_pad_nsegments_per_core,
    const std::vector<int32_t>& ll_data_nsegments_per_core,
    const std::vector<int32_t>& l_data_nsegments_per_core,
    const std::vector<int32_t>& local_data_nsegments_per_core,
    const std::vector<int32_t>& r_data_nsegments_per_core,
    const std::vector<int32_t>& rr_data_nsegments_per_core,
    const std::vector<int32_t>& local_data_src_start_offsets_per_core,
    const std::vector<int32_t>& ll_data_src_start_offsets_per_core,
    const std::vector<int32_t>& l_data_src_start_offsets_per_core,
    const std::vector<int32_t>& r_data_src_start_offsets_per_core,
    const std::vector<int32_t>& rr_data_src_start_offsets_per_core,
    Tensor& output_tensor) {

    Program program = CreateProgram();

    Device *device = input_tensor.device();
    Buffer *src_buffer = input_tensor.buffer();
    Buffer *dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    Shape input_shape = input_tensor.shape();
    Shape output_shape = output_tensor.shape();

    DataFormat in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t out_nbytes = datum_size(out_df);

    uint32_t in_tile_size = detail::TileSize(in_df);
    uint32_t out_tile_size = detail::TileSize(out_df);

    auto grid_size = device->compute_with_storage_grid_size();
    untilize_with_halo_v2_helpers::init_neighbor_core_xy_mapping(grid_size, input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = grid_size.y;

    CoreRangeSet all_cores = input_tensor.shard_spec().value().shard_grid;
    uint32_t ncores = all_cores.num_cores();
    uint32_t ncores_width = 1;
    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(all_cores.ranges().begin());
        ncores = core_range.end.x - core_range.start.x + 1;
        ncores_width = core_range.end.y - core_range.start.y + 1;
    }
    TT_ASSERT(ncores_height == ncores);

    auto shard_shape = input_tensor.shard_spec().value().shard_shape;
    uint32_t ntiles_per_block = shard_shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = shard_shape[0] / TILE_HEIGHT;
    uint32_t out_stick_nbytes = shard_shape[1] * out_nbytes;

    // Construct CBs
    // //

    uint32_t src_cb_id = CB::c_in0;
    uint32_t pad_cb_id = CB::c_in1;
    uint32_t untilize_out_cb_id = CB::c_out0;
    uint32_t out_cb_id = CB::c_out1;

    uint32_t input_ntiles = ntiles_per_block * nblocks_per_core;
    auto src_cb_config = CircularBufferConfig(input_ntiles * in_tile_size, {{src_cb_id, in_df}})
                            .set_page_size(src_cb_id, in_tile_size)
                            .set_globally_allocated_address(*src_buffer);
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);

    // output of untilize from compute kernel goes into this CB
    uint32_t output_ntiles = ntiles_per_block * nblocks_per_core;
    auto untilize_out_cb_config = CircularBufferConfig(output_ntiles * out_tile_size, {{untilize_out_cb_id, out_df}})
                                    .set_page_size(untilize_out_cb_id, out_tile_size);
    auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);

    // output shard, after inserting halo and padding, goes into this CB as input to next op.
    uint32_t out_cb_pagesize = out_stick_nbytes;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    auto out_cb_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
                            .set_page_size(out_cb_id, out_cb_pagesize)
                            .set_globally_allocated_address(*dst_buffer);
    auto out_cb = CreateCircularBuffer(program, all_cores, out_cb_config);

    // CB for pad val buffer (stick sized)
    uint32_t pad_cb_pagesize = out_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    auto pad_cb_config = CircularBufferConfig(pad_cb_pagesize * pad_cb_npages, {{pad_cb_id, out_df}})
                            .set_page_size(pad_cb_id, pad_cb_pagesize);
    auto pad_cb = CreateCircularBuffer(program, all_cores, pad_cb_config);

    // Additional CBs for sharded data kernel configs
    // //

    uint32_t local_pad_ss_cb_id = CB::c_in2;
    uint32_t local_data_ss_cb_id = CB::c_in3;
    uint32_t ll_data_ss_cb_id = CB::c_in4;
    uint32_t l_data_ss_cb_id = CB::c_in5;
    uint32_t r_data_ss_cb_id = CB::c_in6;
    uint32_t rr_data_ss_cb_id = CB::c_in7;

    DataFormat kernel_config_df = DataFormat::UInt16;
    uint32_t pagesize = 0;

    // TODO: Create the following CBs conditionally, only when corresponding data exists

    // local_pad_start_and_size
    pagesize = *std::max(local_pad_nsegments_per_core.cbegin(), local_pad_nsegments_per_core.cend());
    bool local_pad_ss_exists = pagesize > 0;
    CBHandle local_pad_ss_cb = 0;
    if (local_pad_ss_exists) {
        Buffer *local_pad_ss_buffer = local_pad_start_and_size.buffer();
        auto local_pad_ss_cb_config = CircularBufferConfig(pagesize * 1, {{local_pad_ss_cb_id, kernel_config_df}})
                                        .set_page_size(local_pad_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*local_pad_ss_buffer);
        local_pad_ss_cb = CreateCircularBuffer(program, all_cores, local_pad_ss_cb_config);
    }

    // ll_data_start_and_size
    pagesize = *std::max(ll_data_nsegments_per_core.cbegin(), ll_data_nsegments_per_core.cend());
    bool ll_data_ss_exists = pagesize > 0;
    CBHandle ll_data_ss_cb = 0;
    if (ll_data_ss_exists) {
        Buffer *ll_data_ss_buffer = ll_data_start_and_size.buffer();
        auto ll_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{ll_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(ll_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*ll_data_ss_buffer);
        ll_data_ss_cb = CreateCircularBuffer(program, all_cores, ll_data_ss_cb_config);
    }

    // l_data_start_and_size
    pagesize = *std::max(l_data_nsegments_per_core.cbegin(), l_data_nsegments_per_core.cend());
    bool l_data_ss_exists = pagesize > 0;
    CBHandle l_data_ss_cb = 0;
    if (l_data_ss_exists) {
        Buffer *l_data_ss_buffer = l_data_start_and_size.buffer();
        auto l_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{l_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(l_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*l_data_ss_buffer);
        l_data_ss_cb = CreateCircularBuffer(program, all_cores, l_data_ss_cb_config);
    }

    // local_data_start_and_size
    pagesize = *std::max(local_data_nsegments_per_core.cbegin(), local_data_nsegments_per_core.cend());
    bool local_data_ss_exists = pagesize > 0;
    CBHandle local_data_ss_cb = 0;
    if (local_data_ss_exists) {
        Buffer *local_data_ss_buffer = local_data_start_and_size.buffer();
        auto local_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{local_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(local_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*local_data_ss_buffer);
        local_data_ss_cb = CreateCircularBuffer(program, all_cores, local_data_ss_cb_config);
    }

    // r_data_start_and_size
    pagesize = *std::max(r_data_nsegments_per_core.cbegin(), r_data_nsegments_per_core.cend());
    bool r_data_ss_exists = pagesize > 0;
    CBHandle r_data_ss_cb = 0;
    if (r_data_ss_exists) {
        Buffer *r_data_ss_buffer = r_data_start_and_size.buffer();
        auto r_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{r_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(r_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*r_data_ss_buffer);
        r_data_ss_cb = CreateCircularBuffer(program, all_cores, r_data_ss_cb_config);
    }

    // rr_data_start_and_size
    pagesize = *std::max(rr_data_nsegments_per_core.cbegin(), rr_data_nsegments_per_core.cend());
    bool rr_data_ss_exists = pagesize > 0;
    CBHandle rr_data_ss_cb = 0;
    if (rr_data_ss_exists) {
        Buffer *rr_data_ss_buffer = rr_data_start_and_size.buffer();
        auto rr_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{rr_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(rr_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*rr_data_ss_buffer);
        rr_data_ss_cb = CreateCircularBuffer(program, all_cores, rr_data_ss_cb_config);
    }

    // Construct kernels
    // //

    // reader kernel
    std::vector<uint32_t> reader_ct_args = { src_cb_id };
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    // writer kernel
    std::vector<uint32_t> writer_ct_args = {
        untilize_out_cb_id,
        out_cb_id,
        pad_cb_id,
        local_pad_ss_cb_id,
        local_data_ss_cb_id,
        ll_data_ss_cb_id,
        l_data_ss_cb_id,
        r_data_ss_cb_id,
        rr_data_ss_cb_id,
        pad_val,
        shard_shape[1],         // pad stick length == output stick size in nelems
        out_stick_nbytes };     // output stick size in bytes
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/writer_unary_sharded_with_halo_v2.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    // compute kernel
    std::vector<uint32_t> compute_ct_args = {
        nblocks_per_core,
        ntiles_per_block };
    KernelHandle untilize_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp",
        all_cores,
        ComputeConfig{
            .compile_args = compute_ct_args});

    // runtime args for reader
    std::vector<uint32_t> reader_rt_args = { ntiles_per_block * nblocks_per_core };
    SetRuntimeArgs(program, reader_kernel_id, all_cores, reader_rt_args);

    // runtime args for writer
    std::vector<uint32_t> writer_rt_args = {
        ntiles_per_block * nblocks_per_core,            // 0
        0,  // has ll
        0,  // ll noc.x
        0,  // ll noc.y
        0,  // has l
        0,  // l noc.x                                  // 5
        0,  // l noc.y
        0,  // has r
        0,  // r noc.x
        0,  // r noc.y
        0,  // has rr                                   // 10
        0,  // rr noc.x
        0,  // rr noc.y
        0,  // local_pad_nsegments
        0,  // local_data_src_start_offset
        0,  // local_data_nsegments                     // 15
        0,  // ll_data_src_start_offset
        0,  // ll_data_nsegments
        0,  // l_data_src_start_offset
        0,  // l_data_nsegments
        0,  // r_data_src_start_offset                  // 20
        0,  // r_data_nsegments
        0,  // rr_data_src_start_offset
        0,  // rr_data_nsegments
    };
    for (uint32_t core = 0; core < ncores_height; ++ core) {
        CoreCoord core_coord = { core % ncores_x, core / ncores_x };    // logical
        // left neighbor args
        if (untilize_with_halo_v2_helpers::left_neighbor_core.count(core_coord) > 0) {
            CoreCoord left_core = untilize_with_halo_v2_helpers::left_neighbor_core.at(core_coord);
            CoreCoord left_noc = device->worker_core_from_logical_core(left_core);
            writer_rt_args[4] = 1;
            writer_rt_args[5] = left_noc.x;
            writer_rt_args[6] = left_noc.y;
            if (untilize_with_halo_v2_helpers::left_neighbor_core.count(left_core) > 0) {
                CoreCoord left_left_core = untilize_with_halo_v2_helpers::left_neighbor_core.at(left_core);
                CoreCoord left_left_noc = device->worker_core_from_logical_core(left_left_core);
                writer_rt_args[1] = 1;
                writer_rt_args[2] = left_left_noc.x;
                writer_rt_args[3] = left_left_noc.y;
            } else {
                // no left-left neighbor
                writer_rt_args[1] = 0;
            }
        } else {
            // no left neighbors
            writer_rt_args[1] = 0;
            writer_rt_args[4] = 0;
        }
        // right neighbor args
        if (untilize_with_halo_v2_helpers::right_neighbor_core.count(core_coord) > 0) {
            CoreCoord right_core = untilize_with_halo_v2_helpers::right_neighbor_core.at(core_coord);
            CoreCoord right_noc = device->worker_core_from_logical_core(right_core);
            writer_rt_args[7] = 1;
            writer_rt_args[8] = right_noc.x;
            writer_rt_args[9] = right_noc.y;
            if (untilize_with_halo_v2_helpers::right_neighbor_core.count(right_core) > 0) {
                CoreCoord right_right_core = untilize_with_halo_v2_helpers::right_neighbor_core.at(right_core);
                CoreCoord right_right_noc = device->worker_core_from_logical_core(right_right_core);
                writer_rt_args[10] = 1;
                writer_rt_args[11] = right_right_noc.x;
                writer_rt_args[12] = right_right_noc.y;
            } else {
                // no right-right neighbor
                writer_rt_args[10] = 0;
            }
        } else {
            // no right neighbors
            writer_rt_args[7] = 0;
            writer_rt_args[10] = 0;
        }

        writer_rt_args[13] = local_pad_nsegments_per_core[core];
        writer_rt_args[14] = local_data_src_start_offsets_per_core[core];
        writer_rt_args[15] = local_data_nsegments_per_core[core];
        writer_rt_args[16] = ll_data_src_start_offsets_per_core[core];
        writer_rt_args[17] = ll_data_nsegments_per_core[core];
        writer_rt_args[18] = l_data_src_start_offsets_per_core[core];
        writer_rt_args[19] = l_data_nsegments_per_core[core];
        writer_rt_args[20] = r_data_src_start_offsets_per_core[core];
        writer_rt_args[21] = r_data_nsegments_per_core[core];
        writer_rt_args[22] = rr_data_src_start_offsets_per_core[core];
        writer_rt_args[23] = rr_data_nsegments_per_core[core];
    }


    auto override_runtime_arguments_callback = [
        reader_kernel_id=reader_kernel_id,
        writer_kernel_id=writer_kernel_id,
        src_cb=src_cb,
        out_cb=out_cb,
        local_pad_ss_exists=local_pad_ss_exists,
        ll_data_ss_exists=ll_data_ss_exists,
        l_data_ss_exists=l_data_ss_exists,
        local_data_ss_exists=local_data_ss_exists,
        r_data_ss_exists=r_data_ss_exists,
        rr_data_ss_exists=rr_data_ss_exists,
        local_pad_ss_cb=local_pad_ss_cb,
        ll_data_ss_cb=ll_data_ss_cb,
        l_data_ss_cb=l_data_ss_cb,
        local_data_ss_cb=local_data_ss_cb,
        r_data_ss_cb=r_data_ss_cb,
        rr_data_ss_cb=rr_data_ss_cb
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

        // kernel config data
        if (local_pad_ss_exists) {
            auto local_pad_ss_buffer = input_tensors.at(1).buffer();
            UpdateDynamicCircularBufferAddress(program, local_pad_ss_cb, *local_pad_ss_buffer);
        }
        if (ll_data_ss_exists) {
            auto ll_data_ss_buffer = input_tensors.at(2).buffer();
            UpdateDynamicCircularBufferAddress(program, ll_data_ss_cb, *ll_data_ss_buffer);
        }
        if (l_data_ss_exists) {
            auto l_data_ss_buffer = input_tensors.at(3).buffer();
            UpdateDynamicCircularBufferAddress(program, l_data_ss_cb, *l_data_ss_buffer);
        }
        if (local_data_ss_exists) {
            auto local_data_ss_buffer = input_tensors.at(4).buffer();
            UpdateDynamicCircularBufferAddress(program, local_data_ss_cb, *local_data_ss_buffer);
        }
        if (r_data_ss_exists) {
            auto r_data_ss_buffer = input_tensors.at(5).buffer();
            UpdateDynamicCircularBufferAddress(program, r_data_ss_cb, *r_data_ss_buffer);
        }
        if (rr_data_ss_exists) {
            auto rr_data_ss_buffer = input_tensors.at(6).buffer();
            UpdateDynamicCircularBufferAddress(program, rr_data_ss_cb, *rr_data_ss_buffer);
        }
    };

    return { .program = std::move(program),
             .override_runtime_arguments_callback = override_runtime_arguments_callback };
}

void UntilizeWithHaloV2::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& local_pad_start_and_size = input_tensors.at(1);
    const auto& ll_data_start_and_size = input_tensors.at(2);
    const auto& l_data_start_and_size = input_tensors.at(3);
    const auto& local_data_start_and_size = input_tensors.at(4);
    const auto& r_data_start_and_size = input_tensors.at(5);
    const auto& rr_data_start_and_size = input_tensors.at(6);

    // validate input data tensor
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor should be TILE for untilize");
    TT_FATAL(input_tensor.volume() % TILE_HW == 0);
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

    // validate all other config tensors
    for (auto tensor : {local_pad_start_and_size,
                        ll_data_start_and_size,
                        l_data_start_and_size,
                        local_data_start_and_size,
                        r_data_start_and_size,
                        rr_data_start_and_size}) {
        TT_FATAL(tensor.buffer() != nullptr, "Input tensors need to be allocated buffers on device");
        TT_FATAL(tensor.layout() == Layout::ROW_MAJOR);
        TT_FATAL(tensor.memory_config().is_sharded());
        TT_FATAL(tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
    }
}

std::vector<Shape> UntilizeWithHaloV2::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.shape();
    Shape output_shape = input_shape;

    uint32_t nbatch = input_shape[0];
    uint32_t total_nsticks = ncores_height_ * max_out_nsticks_per_core_;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    output_shape[2] = (uint32_t) ceil((float) total_nsticks / nbatch);

    log_debug(LogOp, "output_shape: [{} {} {} {}]", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(LogOp, "max_out_nsticks_per_core: {}", max_out_nsticks_per_core_);
    log_debug(LogOp, "ncores_height: {}", ncores_height_);

    return {output_shape};
}

std::vector<Tensor> UntilizeWithHaloV2::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    // NOTE: output is always ROW_MAJOR
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();
    auto shard_spec = input_tensor.shard_spec().value();
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);

    TT_ASSERT(ncores_height_ == input_tensor.shape()[0] * input_tensor.shape()[2] / shard_spec.shard_shape[0]);

    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(shard_spec.shard_grid.ranges().begin());
        TT_ASSERT(ncores_height_ == core_range.end.x - core_range.start.x + 1);
    }
    auto out_shard_spec = shard_spec;
    out_shard_spec.shard_shape[0] = output_shape[0] * div_up(output_shape[2], ncores_height_);
    out_shard_spec.halo = true;
    return {create_sharded_device_tensor(output_shape, output_dtype, Layout::ROW_MAJOR, input_tensor.device(), out_mem_config_, out_shard_spec)};
}

operation::ProgramWithCallbacks UntilizeWithHaloV2::create_program(const std::vector<Tensor>& inputs, std::vector<Tensor> &outputs) const {
    const auto& input_tensor = inputs.at(0);
    const auto& local_pad_start_and_size = inputs.at(1);
    const auto& ll_data_start_and_size = inputs.at(2);
    const auto& l_data_start_and_size = inputs.at(3);
    const auto& local_data_start_and_size = inputs.at(4);
    const auto& r_data_start_and_size = inputs.at(5);
    const auto& rr_data_start_and_size = inputs.at(6);
    auto& output_tensor = outputs.at(0);

    return { untilize_with_halo_multi_core_v2(input_tensor,
                                              local_pad_start_and_size,
                                              ll_data_start_and_size,
                                              l_data_start_and_size,
                                              local_data_start_and_size,
                                              r_data_start_and_size,
                                              rr_data_start_and_size,
                                              pad_val_,
                                              ncores_height_,
                                              max_out_nsticks_per_core_,
                                              local_pad_nsegments_per_core_,
                                              ll_data_nsegments_per_core_,
                                              l_data_nsegments_per_core_,
                                              local_data_nsegments_per_core_,
                                              r_data_nsegments_per_core_,
                                              rr_data_nsegments_per_core_,
                                              local_data_src_start_offsets_per_core_,
                                              ll_data_src_start_offsets_per_core_,
                                              l_data_src_start_offsets_per_core_,
                                              r_data_src_start_offsets_per_core_,
                                              rr_data_src_start_offsets_per_core_,
                                              output_tensor) };
}

Tensor untilize_with_halo_v2(const Tensor& input_tensor,
                             const Tensor& local_pad_start_and_size,
                             const Tensor& ll_data_start_and_size,
                             const Tensor& l_data_start_and_size,
                             const Tensor& local_data_start_and_size,
                             const Tensor& r_data_start_and_size,
                             const Tensor& rr_data_start_and_size,
                             const uint32_t pad_val,
                             const uint32_t ncores_height,
                             const uint32_t max_out_nsticks_per_core,
                             const std::vector<int32_t>& local_pad_nsegments_per_core,
                             const std::vector<int32_t>& ll_data_nsegments_per_core,
                             const std::vector<int32_t>& l_data_nsegments_per_core,
                             const std::vector<int32_t>& local_data_nsegments_per_core,
                             const std::vector<int32_t>& r_data_nsegments_per_core,
                             const std::vector<int32_t>& rr_data_nsegments_per_core,
                             const std::vector<int32_t>& local_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& ll_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& l_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& r_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& rr_data_src_start_offsets_per_core,
                             const MemoryConfig& mem_config) {
    TT_ASSERT(input_tensor.memory_config().is_sharded());
    TT_ASSERT(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    TT_ASSERT(ncores_height == local_data_nsegments_per_core.size());
    // NOTE: for HEIGHT_SHARDED, ncores_height == ncores
    //       for BLOCK_SHARDED, ncores_height is just the ncores along height dim (last tensor dim is split along width)

    // auto input_shape = input_tensor.shape();
    // auto input_shard_shape = input_tensor.shard_spec().value().shard_shape;

    // Calculate the max output nsticks across all coresfrom the resharded global indices
    // uint32_t max_out_nsticks_per_core = 0;
    // for (auto [shard_start, shard_end] : resharded_start_and_end) { // NOTE: start and end are inclusive
    //     uint32_t shard_nsticks = shard_end - shard_start + 1;
    //     max_out_nsticks_per_core = std::max(max_out_nsticks_per_core, shard_nsticks);
    // }
    // log_debug("max out nsticks across all shards = {}", max_out_nsticks_per_core);

    return operation::run_without_autoformat(UntilizeWithHaloV2{
                                                pad_val,
                                                ncores_height,
                                                max_out_nsticks_per_core,
                                                local_pad_nsegments_per_core,
                                                ll_data_nsegments_per_core,
                                                l_data_nsegments_per_core,
                                                local_data_nsegments_per_core,
                                                r_data_nsegments_per_core,
                                                rr_data_nsegments_per_core,
                                                local_data_src_start_offsets_per_core,
                                                ll_data_src_start_offsets_per_core,
                                                l_data_src_start_offsets_per_core,
                                                r_data_src_start_offsets_per_core,
                                                rr_data_src_start_offsets_per_core,
                                                mem_config},
                                             {input_tensor,
                                              local_pad_start_and_size,
                                              ll_data_start_and_size,
                                              l_data_start_and_size,
                                              local_data_start_and_size,
                                              r_data_start_and_size,
                                              rr_data_start_and_size})
                                            .at(0);

}

}  // namespace tt_metal

}  // namespace tt
