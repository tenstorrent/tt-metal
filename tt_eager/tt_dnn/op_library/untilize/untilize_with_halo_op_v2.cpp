// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <algorithm>

#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/sliding_window_op_infra/utils.hpp"
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

int32_t my_max(const std::vector<int32_t>& in) {
    int32_t mmax = 0;
    for (int32_t v : in) {
        mmax = mmax > v ? mmax : v;
    }
    return mmax;
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
    const uint32_t ncores_nhw,
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

    bool skip_untilize = input_tensor.layout() == Layout::ROW_MAJOR;

    Shape input_shape = input_tensor.shape();
    Shape output_shape = output_tensor.shape();

    DataFormat in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t out_nbytes = datum_size(out_df);

    auto grid_size = device->compute_with_storage_grid_size();
    std::map<CoreCoord, CoreCoord> left_neighbor_core, right_neighbor_core;
    utils::init_neighbor_core_xy_mapping(grid_size, left_neighbor_core, right_neighbor_core, input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = grid_size.y;

    CoreRangeSet all_cores = input_tensor.shard_spec().value().shard_grid;
    uint32_t ncores = all_cores.num_cores();
    uint32_t ncores_c = 1;
    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(all_cores.ranges().begin());
        ncores = core_range.end.x - core_range.start.x + 1;
        ncores_c = core_range.end.y - core_range.start.y + 1;
    }
    log_debug(LogOp, "ncores_c: {}", ncores_c);
    TT_ASSERT(ncores_nhw == ncores);

    auto shard_shape = input_tensor.shard_spec().value().shard_shape;
    uint32_t ntiles_per_block = shard_shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = shard_shape[0] / TILE_HEIGHT;
    uint32_t input_npages = ntiles_per_block * nblocks_per_core;

    uint32_t out_stick_nbytes = shard_shape[1] * out_nbytes;

    uint32_t in_page_size = detail::TileSize(in_df);
    uint32_t out_tile_size = detail::TileSize(out_df);

    if (skip_untilize) {
        uint32_t in_nbytes = datum_size(in_df);
        in_page_size = shard_shape[1] * in_nbytes;
        input_npages = shard_shape[0];
    }

    // Construct CBs
    // //

    uint32_t src_cb_id = CB::c_in0;
    uint32_t pad_cb_id = CB::c_in1;
    uint32_t untilize_out_cb_id = CB::c_out0;
    uint32_t out_cb_id = CB::c_out1;

    // input CB (sharded)
    auto src_cb_config = CircularBufferConfig(input_npages * in_page_size, {{src_cb_id, in_df}})
                            .set_page_size(src_cb_id, in_page_size)
                            .set_globally_allocated_address(*src_buffer);
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);
    log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", src_cb_id, input_npages, in_page_size);

    uint32_t input_to_writer_cb_id = src_cb_id;
    if (!skip_untilize) {
        input_to_writer_cb_id = untilize_out_cb_id;

        // output of untilize from compute kernel goes into this CB
        uint32_t output_ntiles = ntiles_per_block * nblocks_per_core;
        auto untilize_out_cb_config = CircularBufferConfig(output_ntiles * out_tile_size, {{untilize_out_cb_id, out_df}})
                                        .set_page_size(untilize_out_cb_id, out_tile_size);
        auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);
        log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", untilize_out_cb_id, output_ntiles, out_tile_size);
    }

    // output shard, after inserting halo and padding, goes into this CB as input to next op.
    uint32_t out_cb_pagesize = out_stick_nbytes;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    auto out_cb_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
                            .set_page_size(out_cb_id, out_cb_pagesize)
                            .set_globally_allocated_address(*dst_buffer);
    auto out_cb = CreateCircularBuffer(program, all_cores, out_cb_config);
    log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", out_cb_id, out_cb_npages, out_cb_pagesize);

    // CB for pad val buffer (stick sized)
    uint32_t pad_cb_pagesize = out_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    auto pad_cb_config = CircularBufferConfig(pad_cb_pagesize * pad_cb_npages, {{pad_cb_id, out_df}})
                            .set_page_size(pad_cb_id, pad_cb_pagesize);
    auto pad_cb = CreateCircularBuffer(program, all_cores, pad_cb_config);
    log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", pad_cb_id, pad_cb_npages, pad_cb_pagesize);

    // Additional CBs for sharded data kernel configs
    // //

    uint32_t local_pad_ss_cb_id = CB::c_in2;
    uint32_t local_data_ss_cb_id = CB::c_in3;
    uint32_t ll_data_ss_cb_id = CB::c_in4;
    uint32_t l_data_ss_cb_id = CB::c_in5;
    uint32_t r_data_ss_cb_id = CB::c_in6;
    uint32_t rr_data_ss_cb_id = CB::c_in7;

    DataFormat kernel_config_df = DataFormat::RawUInt16;        // NOTE: UInt16 is not supported for CB types
    uint32_t config_nbytes = datum_size(kernel_config_df) * 2;  // each config is a pair "start, size", so double the size
    uint32_t pagesize = 0;

    // local_pad_start_and_size
    pagesize = config_nbytes * untilize_with_halo_v2_helpers::my_max(local_pad_nsegments_per_core);
    bool local_pad_ss_exists = pagesize > 0;
    CBHandle local_pad_ss_cb = 0;
    if (local_pad_ss_exists) {
        TT_ASSERT(local_pad_start_and_size.dtype() == DataType::UINT16);
        log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", local_pad_ss_cb_id, 1, pagesize);
        Buffer *local_pad_ss_buffer = local_pad_start_and_size.buffer();
        auto local_pad_ss_cb_config = CircularBufferConfig(pagesize * 1, {{local_pad_ss_cb_id, kernel_config_df}})
                                        .set_page_size(local_pad_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*local_pad_ss_buffer);
        local_pad_ss_cb = CreateCircularBuffer(program, all_cores, local_pad_ss_cb_config);
    }

    // ll_data_start_and_size
    pagesize = config_nbytes * untilize_with_halo_v2_helpers::my_max(ll_data_nsegments_per_core);
    bool ll_data_ss_exists = pagesize > 0;
    CBHandle ll_data_ss_cb = 0;
    if (ll_data_ss_exists) {
        TT_ASSERT(ll_data_start_and_size.dtype() == DataType::UINT16);
        log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", ll_data_ss_cb_id, 1, pagesize);
        Buffer *ll_data_ss_buffer = ll_data_start_and_size.buffer();
        auto ll_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{ll_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(ll_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*ll_data_ss_buffer);
        ll_data_ss_cb = CreateCircularBuffer(program, all_cores, ll_data_ss_cb_config);
    }

    // l_data_start_and_size
    pagesize = config_nbytes * untilize_with_halo_v2_helpers::my_max(l_data_nsegments_per_core);
    bool l_data_ss_exists = pagesize > 0;
    CBHandle l_data_ss_cb = 0;
    if (l_data_ss_exists) {
        TT_ASSERT(l_data_start_and_size.dtype() == DataType::UINT16);
        log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", l_data_ss_cb_id, 1, pagesize);
        Buffer *l_data_ss_buffer = l_data_start_and_size.buffer();
        auto l_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{l_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(l_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*l_data_ss_buffer);
        l_data_ss_cb = CreateCircularBuffer(program, all_cores, l_data_ss_cb_config);
    }

    // local_data_start_and_size
    pagesize = config_nbytes * untilize_with_halo_v2_helpers::my_max(local_data_nsegments_per_core);
    bool local_data_ss_exists = pagesize > 0;
    CBHandle local_data_ss_cb = 0;
    if (local_data_ss_exists) {
    TT_ASSERT(local_data_start_and_size.dtype() == DataType::UINT16);
        log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", local_data_ss_cb_id, 1, pagesize);
        Buffer *local_data_ss_buffer = local_data_start_and_size.buffer();
        auto local_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{local_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(local_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*local_data_ss_buffer);
        local_data_ss_cb = CreateCircularBuffer(program, all_cores, local_data_ss_cb_config);
    }

    // r_data_start_and_size
    pagesize = config_nbytes * untilize_with_halo_v2_helpers::my_max(r_data_nsegments_per_core);
    bool r_data_ss_exists = pagesize > 0;
    CBHandle r_data_ss_cb = 0;
    if (r_data_ss_exists) {
        TT_ASSERT(r_data_start_and_size.dtype() == DataType::UINT16);
        log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", r_data_ss_cb_id, 1, pagesize);
        Buffer *r_data_ss_buffer = r_data_start_and_size.buffer();
        auto r_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{r_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(r_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*r_data_ss_buffer);
        r_data_ss_cb = CreateCircularBuffer(program, all_cores, r_data_ss_cb_config);
    }

    // rr_data_start_and_size
    pagesize = config_nbytes * untilize_with_halo_v2_helpers::my_max(rr_data_nsegments_per_core);
    bool rr_data_ss_exists = pagesize > 0;
    CBHandle rr_data_ss_cb = 0;
    if (rr_data_ss_exists) {
        TT_ASSERT(rr_data_start_and_size.dtype() == DataType::UINT16);
        log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}", rr_data_ss_cb_id, 1, pagesize);
        Buffer *rr_data_ss_buffer = rr_data_start_and_size.buffer();
        auto rr_data_ss_cb_config = CircularBufferConfig(pagesize * 1, {{rr_data_ss_cb_id, kernel_config_df}})
                                        .set_page_size(rr_data_ss_cb_id, pagesize)
                                        .set_globally_allocated_address(*rr_data_ss_buffer);
        rr_data_ss_cb = CreateCircularBuffer(program, all_cores, rr_data_ss_cb_config);
    }

    // Construct kernels
    // //

    // reader kernel
    std::vector<uint32_t> reader_ct_args = {
        input_to_writer_cb_id,
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
        out_stick_nbytes,
        (uint32_t) std::log2(out_stick_nbytes),
        src_cb_id };     // output stick size in bytes
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/reader_unary_sharded_with_halo_v2.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    // writer kernel
    std::vector<uint32_t> writer_ct_args = reader_ct_args;
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/writer_unary_sharded_with_halo_v2.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    if (!skip_untilize) {
        // compute kernel
        std::vector<uint32_t> compute_ct_args = {
            nblocks_per_core,
            ntiles_per_block };
        std::string compute_kernel("tt_eager/tt_dnn/op_library/untilize/kernels/compute/pack_untilize.cpp");
        if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH) {
            log_debug(LogOp, "Falling back to slow untilize since ntiles_per_block {} > MAX_PACK_UNTILIZE_WIDTH {}", ntiles_per_block, MAX_PACK_UNTILIZE_WIDTH);
            compute_kernel = std::string("tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp");
        }
        KernelHandle untilize_kernel_id = CreateKernel(
            program,
            compute_kernel,
            all_cores,
            ComputeConfig{
                .compile_args = compute_ct_args});
    }

    // runtime args for reader
    std::vector<uint32_t> reader_rt_args = { input_npages };

    // runtime args for writer
    std::vector<uint32_t> writer_rt_args = {
        input_npages,                                   // 0
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

    log_debug(LogOp, "local_pad_nsegments: {}", local_pad_nsegments_per_core);
    log_debug(LogOp, "local_data_nsegments: {}", local_data_nsegments_per_core);
    log_debug(LogOp, "local_data_src_start_offsets: {}", local_data_src_start_offsets_per_core);
    log_debug(LogOp, "ll_data_nsegments: {}", ll_data_nsegments_per_core);
    log_debug(LogOp, "ll_data_src_start_offsets: {}", ll_data_src_start_offsets_per_core);
    log_debug(LogOp, "l_data_nsegments: {}", l_data_nsegments_per_core);
    log_debug(LogOp, "l_data_src_start_offsets: {}", l_data_src_start_offsets_per_core);
    log_debug(LogOp, "r_data_nsegments: {}", r_data_nsegments_per_core);
    log_debug(LogOp, "r_data_src_start_offsets: {}", r_data_src_start_offsets_per_core);
    log_debug(LogOp, "rr_data_nsegments: {}", rr_data_nsegments_per_core);
    log_debug(LogOp, "rr_data_src_start_offsets: {}", rr_data_src_start_offsets_per_core);

    for (uint32_t core = 0; core < ncores_nhw; ++ core) {
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

        for (uint32_t core_c = 0; core_c < ncores_c; ++ core_c) {
            CoreCoord core_coord;   // logical
            if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                core_coord = { core, core_c };
            } else {
                core_coord = { core % ncores_x, core / ncores_x};
            }
            // left neighbor args
            if (left_neighbor_core.count(core_coord) > 0) {
                CoreCoord left_core = left_neighbor_core.at(core_coord);
                CoreCoord left_noc = device->worker_core_from_logical_core(left_core);
                writer_rt_args[4] = 1;
                writer_rt_args[5] = left_noc.x;
                writer_rt_args[6] = left_noc.y;
                if (left_neighbor_core.count(left_core) > 0) {
                    CoreCoord left_left_core = left_neighbor_core.at(left_core);
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
            if (right_neighbor_core.count(core_coord) > 0) {
                CoreCoord right_core = right_neighbor_core.at(core_coord);
                CoreCoord right_noc = device->worker_core_from_logical_core(right_core);
                writer_rt_args[7] = 1;
                writer_rt_args[8] = right_noc.x;
                writer_rt_args[9] = right_noc.y;
                if (right_neighbor_core.count(right_core) > 0) {
                    CoreCoord right_right_core = right_neighbor_core.at(right_core);
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

            SetRuntimeArgs(program, reader_kernel_id, core_coord, writer_rt_args);
            SetRuntimeArgs(program, writer_kernel_id, core_coord, writer_rt_args);
        }
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

void validate_untilize_with_halo_v2_config_tensor(const Tensor& tensor) {
    TT_FATAL(tensor.buffer() != nullptr, "Input tensors need to be allocated buffers on device");
    TT_FATAL(tensor.layout() == Layout::ROW_MAJOR);
    TT_FATAL(tensor.memory_config().is_sharded());
    TT_FATAL(tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
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
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        // skip the untilize, only do halo
        log_debug(LogOp, "Input is ROW_MAJOR, no need to untilize.");
    } else {
        TT_FATAL(input_tensor.volume() % TILE_HW == 0);
    }
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    TT_FATAL(input_tensor.shard_spec().has_value());

    // validate all other config tensors
    int32_t max_size = untilize_with_halo_v2_helpers::my_max(local_data_nsegments_per_core_);
    log_debug(LogOp, "max local data nsegments: {}", max_size);
    if (max_size > 0) validate_untilize_with_halo_v2_config_tensor(local_data_start_and_size);

    max_size = untilize_with_halo_v2_helpers::my_max(local_pad_nsegments_per_core_);
    log_debug(LogOp, "max local pad nsegments: {}", max_size);
    if (max_size > 0) validate_untilize_with_halo_v2_config_tensor(local_pad_start_and_size);

    max_size = untilize_with_halo_v2_helpers::my_max(ll_data_nsegments_per_core_);
    log_debug(LogOp, "max ll data nsegments: {}", max_size);
    if (max_size > 0) validate_untilize_with_halo_v2_config_tensor(ll_data_start_and_size);

    max_size = untilize_with_halo_v2_helpers::my_max(l_data_nsegments_per_core_);
    log_debug(LogOp, "max l data nsegments: {}", max_size);
    if (max_size > 0) validate_untilize_with_halo_v2_config_tensor(l_data_start_and_size);

    max_size = untilize_with_halo_v2_helpers::my_max(r_data_nsegments_per_core_);
    log_debug(LogOp, "max r data nsegments: {}", max_size);
    if (max_size > 0) validate_untilize_with_halo_v2_config_tensor(r_data_start_and_size);

    max_size = untilize_with_halo_v2_helpers::my_max(rr_data_nsegments_per_core_);
    log_debug(LogOp, "max rr data nsegments: {}", max_size);
    if (max_size > 0) validate_untilize_with_halo_v2_config_tensor(rr_data_start_and_size);
}

std::vector<Shape> UntilizeWithHaloV2::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.shape();
    Shape output_shape = input_shape;

    uint32_t nbatch = input_shape[0];
    uint32_t total_nsticks = ncores_nhw_ * max_out_nsticks_per_core_;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    output_shape[2] = (uint32_t) ceil((float) total_nsticks / nbatch);

    log_debug(LogOp, "output_shape: [{} {} {} {}]", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(LogOp, "max_out_nsticks_per_core: {}", max_out_nsticks_per_core_);
    log_debug(LogOp, "ncores_nhw: {}", ncores_nhw_);

    return {output_shape};
}

std::vector<Tensor> UntilizeWithHaloV2::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    // NOTE: output is always ROW_MAJOR
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();
    auto shard_spec = input_tensor.shard_spec().value();
    // log_debug(LogOp, "INPUT SHARD SPEC: {}", shard_spec);
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);

    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(shard_spec.shard_grid.ranges().begin());
        TT_FATAL(ncores_nhw_ == core_range.end.x - core_range.start.x + 1);
    } else {
        TT_FATAL(ncores_nhw_ == shard_spec.shard_grid.num_cores());
    }
    auto out_shard_spec = shard_spec;
    out_shard_spec.shard_shape[0] = output_shape[0] * div_up(output_shape[2], ncores_nhw_);
    out_shard_spec.halo = true;
    // log_debug(LogOp, "OUTPUT SHARD SPEC: {}", out_shard_spec);
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
                                              ncores_nhw_,
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
                             const uint32_t ncores_nhw,
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
    TT_ASSERT(ncores_nhw == local_data_nsegments_per_core.size());
    // NOTE: for HEIGHT_SHARDED, ncores_nhw == ncores
    //       for BLOCK_SHARDED, ncores_nhw is just the ncores along height dim (last tensor dim is split along width)

    return operation::run_without_autoformat(UntilizeWithHaloV2{
                                                pad_val,
                                                ncores_nhw,
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
