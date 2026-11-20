// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "k_split_gram_matmul_program_factory.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <utility>
#include <vector>

#include "k_split_gram_matmul_device_operation.hpp"
#include "metal/common/program_utils.hpp"

namespace ttml::metal::ops::k_split_gram_matmul::device {

TTML_ASSERT_DESCRIPTOR_OVERRIDE(KSplitGramMatmulDeviceOperation, KSplitGramMatmulProgramFactory);

namespace {

// Kernel handles in the built Program: the push order of ProgramDescriptor::kernels, followed by the three compute
// kernels. The device grid is at least 4x3 (validated), so every kernel has cores.
constexpr uint32_t kRowSenderReduceKernel = 0U;
constexpr uint32_t kRowSenderKernel = 1U;
constexpr uint32_t kColSenderKernel = 2U;
constexpr uint32_t kRowLowerOffdiagKernel = 3U;
constexpr uint32_t kRowLowerDiagKernel = 4U;
constexpr uint32_t kRowUpperRecvKernel = 5U;
constexpr uint32_t kColLowerRecvKernel = 6U;
constexpr uint32_t kColLowerRecvEdgeKernel = 7U;
constexpr uint32_t kColUpperRecvKernel = 8U;
constexpr uint32_t kHelperRecvKernel = 9U;
constexpr uint32_t kHelperDramReaderKernel = 10U;

// Semaphore ids, in the order legacy CreateSemaphore handed them out; the kernels get them as compile-time args.
constexpr uint32_t kRowSenderSemId = 0U;
constexpr uint32_t kRowReceiverSemId = 1U;
constexpr uint32_t kRowSenderSem2Id = 2U;
constexpr uint32_t kRowReceiverSem2Id = 3U;
constexpr uint32_t kColSenderSemId = 4U;
constexpr uint32_t kColReceiverSemId = 5U;
constexpr uint32_t kColSenderSem2Id = 6U;
constexpr uint32_t kColReceiverSem2Id = 7U;
constexpr uint32_t kReduceSemId = 8U;
// Consumption credit for the reduce channel: the receiver increments the sender's
// copy after reserving its c_5 slot; the sender waits for it before each NOC write.
constexpr uint32_t kReduceAckSemId = 9U;

// Runtime arg slots that override_runtime_arguments() rewrites on every cache hit: the only buffer addresses. Every
// other runtime arg follows from the device grid and the input shape, which the program hash keys on.
constexpr uint32_t kSenderInAddrIdx = 0U;      // row sender+reduce, row senders, col senders
constexpr uint32_t kUpperRecvOutAddrIdx = 2U;  // row upper receivers, which write the combined output
constexpr uint32_t kHelperInAddrIdx = 0U;      // diagonal helpers' DRAM reader
constexpr uint32_t kHelperOutAddrIdx = 3U;

tt::tt_metal::CoreRangeSet make_core_range_set(const std::vector<tt::tt_metal::CoreCoord>& cores) {
    std::set<tt::tt_metal::CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.insert(tt::tt_metal::CoreRange(c, c));
    }
    return tt::tt_metal::CoreRangeSet(ranges);
}

// Largest square grid with room for a helper column at x=grid_dim
uint32_t gram_grid_dim(const ttnn::distributed::MeshDevice& device) {
    const auto device_grid = device.compute_with_storage_grid_size();
    return static_cast<uint32_t>(std::min(device_grid.x - 1, device_grid.y));
}

}  // namespace

tt::tt_metal::ProgramDescriptor KSplitGramMatmulProgramFactory::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    ProgramDescriptor desc;
    const auto& input = tensor_args.input_tensor;
    auto* const device = input.device();

    const uint32_t grid_dim = gram_grid_dim(*device);

    // Tile counts from padded_shape (always tile-aligned, matches DRAM layout)
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const uint32_t M_tiles = input.padded_shape()[-2] / tile_height;
    const uint32_t K_tiles = input.padded_shape()[-1] / tile_width;
    const uint32_t K_half = K_tiles / 2;

    const uint32_t padded_M_tiles = tt::round_up(M_tiles, grid_dim);
    const uint32_t Mpc = padded_M_tiles / grid_dim;

    constexpr auto tile_format = tt::DataFormat::Float16_b;
    constexpr auto tile_sz = tt::tile_size(tile_format);

    const auto full_grid = tt::tt_metal::CoreRange({0, 0}, {grid_dim - 1, grid_dim - 1});

    // Diagonal helper column: x=grid_dim cores compute odd-K partial for diagonal blocks.
    const tt::tt_metal::CoreRange helper_range({grid_dim, 0}, {grid_dim, grid_dim - 1});
    const auto all_cores = tt::tt_metal::CoreRangeSet(std::set<tt::tt_metal::CoreRange>{full_grid, helper_range});
    const tt::tt_metal::CoreRangeSet helper_cores(helper_range);
    // Upper x-bound for row multicast (includes helper column)
    const uint32_t upper_x_end = grid_dim;

    constexpr uint32_t subblock_h = 2;
    const uint32_t subblock_w = std::min(Mpc, 2u);

    constexpr auto intermed_format = tt::DataFormat::Float32;
    constexpr auto intermed_tile_sz = tt::tile_size(intermed_format);
    constexpr auto out_tile_format = tt::DataFormat::Float16_b;
    constexpr auto out_tile_sz = tt::tile_size(out_tile_format);

    const bool mirror_active = (attrs.output_mode == ttml::metal::OutputMode::Full);

    // Find the largest M_block and K_block_tiles (kb) that fit all CBs in L1.
    // Minimizes num_m_blocks (fewer subblock passes), then maximizes kb (fewer DRAM reads).
    const uint32_t L1_BUDGET =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t mirror_out_overhead = mirror_active ? 2 * out_tile_sz : 0;  // per-mb extra for c_4 + c_7

    // Find optimal (kb, mb) that minimizes num_m_blocks (= ceil(Mpc/mb)), then maximizes kb.
    // Input CBs (c_0, c_1) hold 2 blocks each for double-buffered DRAM streaming.
    constexpr uint32_t input_cb_num_blocks = 2;

    auto find_max_mb = [&](uint32_t kb) -> uint32_t {
        for (uint32_t mb = Mpc; mb >= 1; mb--) {
            // c_0 + c_1: two input CBs, each holds input_cb_num_blocks × kb × mb tiles
            uint32_t input_cbs = 2 * input_cb_num_blocks * kb * mb * tile_sz;
            // c_2: matmul intermediate accumulator (FP32), mb × mb tiles
            uint32_t intermed_cb = mb * mb * intermed_tile_sz;
            // c_5: output/reduce CB, mb × mb tiles (BF16).
            //
            // Precision/L1 tradeoff (currently BF16):
            //   The lower-triangle partial is packed to BF16 before being NOC-written to the upper
            //   partner. This is the dominant truncation step for off-diagonal precision. Promoting
            //   c_5 to FP32 would roughly halve off-diagonal error (diagonal stays BF16-limited by
            //   the final c_6/DRAM output). However at typical mb=13 it adds ~338 KB to L1 — pushes
            //   over the per-core budget and forces mb→12, roughly doubling (m_sub, n_sub) iterations.
            //
            // Future optimization possibility: stream c_5 row-by-row (size mb instead of mb²) — sender packs+sends
            //   one row at a time, receiver consumes incrementally. Frees ~mb²−mb tiles of L1, which
            //   could absorb FP32 c_5 (still smaller than today's BF16 mb²) and improves sender/receiver pipelining.
            uint32_t output_cb = mb * mb * out_tile_sz;
            // c_6: combined output after reduction, mb tiles
            uint32_t combined_cb = mb * out_tile_sz;
            // c_4 + c_7: mirror output staging (only with OutputMode::Full)
            uint32_t mirror_cbs = mb * mirror_out_overhead;

            if (input_cbs + intermed_cb + output_cb + combined_cb + mirror_cbs <= L1_BUDGET)
                return mb;
        }
        return 0;
    };

    uint32_t best_kb = 0, best_mb = 0, best_num_m_blocks = UINT32_MAX;
    for (uint32_t kb = 1; kb <= std::min(K_half, 8u); kb++) {
        if (K_half % kb != 0)
            continue;
        uint32_t mb = find_max_mb(kb);
        if (mb == 0)
            continue;
        uint32_t n = tt::div_up(Mpc, mb);
        if (n > best_num_m_blocks)
            break;
        best_num_m_blocks = n;
        best_kb = kb;
        best_mb = mb;
    }
    TT_FATAL(
        best_mb > 0,
        "Cannot fit mcast gram matmul in L1 (Mpc={}, K_half={}, L1_BUDGET={} bytes, mirror_out_overhead={} bytes)",
        Mpc,
        K_half,
        L1_BUDGET,
        mirror_out_overhead);

    const uint32_t K_block_tiles = best_kb;
    const uint32_t M_block = best_mb;
    const uint32_t N_block = M_block;
    const uint32_t num_m_blocks = best_num_m_blocks;
    const uint32_t num_n_blocks = tt::div_up(Mpc, N_block);
    const uint32_t block_sz = K_block_tiles * M_block;
    const uint32_t cb_size = input_cb_num_blocks * block_sz;
    const uint32_t num_tiles = M_block * K_tiles;  // tiles per sender per (m_sub, n_sub) pass
    // Output DRAM stride in tiles — use padded M (tile-aligned) to match DRAM layout
    const uint32_t padded_out_tiles = M_tiles;
    const uint32_t recv_tiles = K_half * M_block;  // tiles per receiver per (m_sub, n_sub) pass

    const uint32_t send_out_cb = (uint32_t)tt::CBIndex::c_5;
    const uint32_t c5_tiles = M_block * N_block;

    // Every CB lives on the square grid and on the helper column.
    desc.cbs.push_back(make_cb_descriptor(all_cores, (uint32_t)tt::CBIndex::c_0, tile_format, tile_sz, cb_size));
    desc.cbs.push_back(make_cb_descriptor(all_cores, (uint32_t)tt::CBIndex::c_1, tile_format, tile_sz, cb_size));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, (uint32_t)tt::CBIndex::c_2, intermed_format, intermed_tile_sz, M_block * N_block));
    if (mirror_active) {
        // c_4: mirror buffer for transposed tiles (M_block tiles, streamed col by col)
        desc.cbs.push_back(
            make_cb_descriptor(all_cores, (uint32_t)tt::CBIndex::c_4, out_tile_format, out_tile_sz, M_block));
        // c_7: staging for transpose (mb tiles BF16, batched per-column)
        desc.cbs.push_back(
            make_cb_descriptor(all_cores, (uint32_t)tt::CBIndex::c_7, out_tile_format, out_tile_sz, M_block));
    }
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, (uint32_t)tt::CBIndex::c_5, out_tile_format, out_tile_sz, c5_tiles));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, (uint32_t)tt::CBIndex::c_6, out_tile_format, out_tile_sz, M_block));

    // Semaphores on all cores (helpers need row sems for multicast reception)
    desc.semaphores.push_back(make_semaphore_descriptor(kRowSenderSemId, all_cores, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kRowReceiverSemId, all_cores, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kRowSenderSem2Id, all_cores, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kRowReceiverSem2Id, all_cores, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kColSenderSemId, all_cores, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kColReceiverSemId, all_cores, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kColSenderSem2Id, all_cores, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kColReceiverSem2Id, all_cores, INVALID));
    desc.semaphores.push_back(make_semaphore_descriptor(kReduceSemId, all_cores, 0U));
    desc.semaphores.push_back(make_semaphore_descriptor(kReduceAckSemId, all_cores, 0U));

    const auto noc_1 = detail::preferred_noc_for_dram_write(device->arch());
    const auto noc_0 = detail::preferred_noc_for_dram_read(device->arch());
    const DataMovementConfigDescriptor risc_0_config{.processor = DataMovementProcessor::RISCV_0, .noc = noc_0};
    const DataMovementConfigDescriptor risc_1_config{.processor = DataMovementProcessor::RISCV_1, .noc = noc_1};

    auto* const in_buffer = input.buffer();
    auto* const out_buffer = output.buffer();

    const std::string base = "tt-train/sources/ttml/metal/ops/k_split_gram_matmul/device/kernels/";
    const std::string sender_path = base + "mcast_sender.cpp";
    const std::string receiver_path = base + "mcast_receiver.cpp";
    const std::string receiver_writer_path = base + "mcast_receiver_writer.cpp";
    const std::string compute_matmul_path = base + "compute_matmul.cpp";
    const std::string helper_dram_reader_path = base + "helper_dram_reader.cpp";

    // === Core classification ===
    // Row senders: x=0
    std::vector<tt::tt_metal::CoreCoord> row_senders;  // (0, y>0) — regular sender
    for (uint32_t y = 1; y < grid_dim; y++) row_senders.push_back({0, y});
    // (0, 0) is sender_writer — handled separately

    // Col senders: y=0 (all x)
    std::vector<tt::tt_metal::CoreCoord> col_senders;
    col_senders.reserve(grid_dim);
    for (uint32_t x = 0; x < grid_dim; x++) col_senders.push_back({x, 0});

    // Row receivers (RISCV_0, x>0): ALL use receiver_writer (including y=0 edge)
    std::vector<tt::tt_metal::CoreCoord> row_lower_recv, row_upper_recv;
    for (uint32_t y = 0; y < grid_dim; y++) {
        for (uint32_t x = 1; x < grid_dim; x++) {
            if (x <= y) {
                row_lower_recv.push_back({x, y});
            } else {
                row_upper_recv.push_back({x, y});  // includes y=0 edge
            }
        }
    }

    // Col receivers (RISCV_1, y>0): interior = plain receiver, edge (x=0) = receiver_writer
    std::vector<tt::tt_metal::CoreCoord> col_lower_recv_interior, col_lower_recv_edge, col_upper_recv;
    for (uint32_t x = 0; x < grid_dim; x++) {
        for (uint32_t y = 1; y < grid_dim; y++) {
            if (x == 0) {
                col_lower_recv_edge.push_back({0, y});  // receiver_writer for output
            } else if (y >= x) {
                col_lower_recv_interior.push_back({x, y});
            } else {
                col_upper_recv.push_back({x, y});
            }
        }
    }

    // Kernels are pushed in the order of the k*Kernel handles.

    // === Sender kernels ===
    // Row sender+reduce for (0,0): RISCV_0/NOC_0, sends partial to helper (grid_dim, 0)
    {
        std::vector<uint32_t> ct = {
            num_tiles,
            tile_sz,
            kRowSenderSemId,
            kRowReceiverSemId,
            kRowSenderSem2Id,
            kRowReceiverSem2Id,
            (uint32_t)tt::CBIndex::c_0,
            block_sz,
            cb_size,
            send_out_cb,
            out_tile_sz,
            (uint32_t)tt::CBIndex::c_5,
            kReduceSemId,
            num_m_blocks,
            M_block,
            num_n_blocks,
            kReduceAckSemId};
        TensorAccessorArgs(*in_buffer).append_to(ct);
        desc.kernels.push_back(make_kernel_descriptor(
            sender_path, make_core_range_set({{0, 0}}), ct, {{"SENDER_REDUCE_SEND", "1"}}, risc_0_config));
    }

    // Row sender for (0, y>0): RISCV_0/NOC_0, CB c_0
    {
        std::vector<uint32_t> ct = {
            num_tiles,
            tile_sz,
            kRowSenderSemId,
            kRowReceiverSemId,
            kRowSenderSem2Id,
            kRowReceiverSem2Id,
            (uint32_t)tt::CBIndex::c_0,
            block_sz,
            cb_size,
            num_m_blocks,
            num_n_blocks};
        TensorAccessorArgs(*in_buffer).append_to(ct);
        desc.kernels.push_back(
            make_kernel_descriptor(sender_path, make_core_range_set(row_senders), ct, {}, risc_0_config));
    }

    // Col sender: RISCV_1/NOC_1, CB c_1
    {
        std::vector<uint32_t> ct = {
            num_tiles,
            tile_sz,
            kColSenderSemId,
            kColReceiverSemId,
            kColSenderSem2Id,
            kColReceiverSem2Id,
            (uint32_t)tt::CBIndex::c_1,
            block_sz,
            cb_size,
            num_m_blocks,
            num_n_blocks};
        TensorAccessorArgs(*in_buffer).append_to(ct);
        desc.kernels.push_back(
            make_kernel_descriptor(sender_path, make_core_range_set(col_senders), ct, {}, risc_1_config));
    }

    // === Receiver kernels ===
    // Row lower receivers split: off-diagonal (REDUCE_SEND) and diagonal (REDUCE_SEND to helper)
    // Both have same CT args structure, but compute defines differ (set per core group below)
    auto make_recv_send_ct = [&](uint32_t sender_sem_id, uint32_t receiver_sem_id) {
        return std::vector<uint32_t>{
            recv_tiles,
            tile_sz,
            sender_sem_id,
            receiver_sem_id,
            (uint32_t)tt::CBIndex::c_0,
            block_sz,
            send_out_cb,
            out_tile_sz,
            Mpc,
            (uint32_t)tt::CBIndex::c_5,
            kReduceSemId,
            num_m_blocks,
            M_block,
            num_n_blocks,
            kReduceAckSemId};
    };

    auto make_recv_write_ct = [&](uint32_t sender_sem_id, uint32_t receiver_sem_id) {
        std::vector<uint32_t> ct = {
            recv_tiles,
            tile_sz,
            sender_sem_id,
            receiver_sem_id,
            (uint32_t)tt::CBIndex::c_0,
            block_sz,
            (uint32_t)tt::CBIndex::c_6,
            out_tile_sz,
            Mpc,  // cb_out = c_6 (combined)
            padded_out_tiles,
            (uint32_t)tt::CBIndex::c_5,
            kReduceSemId,
            num_m_blocks,
            M_block,
            num_n_blocks,
            kReduceAckSemId};
        TensorAccessorArgs(*out_buffer).append_to(ct);
        return ct;
    };

    // Lower off-diagonal (x < y, x > 0): REDUCE_SEND — send transposed partial to upper
    std::vector<tt::tt_metal::CoreCoord> row_lower_offdiag, row_lower_diag;
    for (auto& c : row_lower_recv) {
        if (c.x < c.y)
            row_lower_offdiag.push_back(c);
        else
            row_lower_diag.push_back(c);  // x == y, diagonal
    }

    // Lower off-diagonal: REDUCE_SEND
    desc.kernels.push_back(make_kernel_descriptor(
        receiver_writer_path,
        make_core_range_set(row_lower_offdiag),
        make_recv_send_ct(kRowSenderSemId, kRowReceiverSemId),
        {{"REDUCE_SEND", "1"}},
        risc_0_config));

    // Lower diagonal (x == y, y > 0): REDUCE_SEND to helper
    desc.kernels.push_back(make_kernel_descriptor(
        receiver_writer_path,
        make_core_range_set(row_lower_diag),
        make_recv_send_ct(kRowSenderSemId, kRowReceiverSemId),
        {{"REDUCE_SEND", "1"}},
        risc_0_config));

    // Row upper receivers (x > y): REDUCE_RECV — receive partner's partial, write combined
    auto upper_recv_defines = std::map<std::string, std::string>{{"REDUCE_RECV", "1"}};
    if (mirror_active)
        upper_recv_defines["MIRROR_OUTPUT"] = "1";
    desc.kernels.push_back(make_kernel_descriptor(
        receiver_writer_path,
        make_core_range_set(row_upper_recv),
        make_recv_write_ct(kRowSenderSem2Id, kRowReceiverSem2Id),
        upper_recv_defines,
        risc_0_config));

    // Col lower receivers (interior, x>0, y≥x): plain receiver
    desc.kernels.push_back(make_kernel_descriptor(
        receiver_path,
        make_core_range_set(col_lower_recv_interior),
        {recv_tiles,
         tile_sz,
         kColSenderSemId,
         kColReceiverSemId,
         (uint32_t)tt::CBIndex::c_1,
         block_sz,
         num_m_blocks,
         num_n_blocks},
        {},
        risc_1_config));

    // Col lower receivers (edge, x=0, y>0): REDUCE_SEND to upper core (y, 0)
    desc.kernels.push_back(make_kernel_descriptor(
        receiver_writer_path,
        make_core_range_set(col_lower_recv_edge),
        {recv_tiles,
         tile_sz,
         kColSenderSemId,
         kColReceiverSemId,
         (uint32_t)tt::CBIndex::c_1,
         block_sz,
         send_out_cb,
         out_tile_sz,
         Mpc,
         (uint32_t)tt::CBIndex::c_5,
         kReduceSemId,
         num_m_blocks,
         M_block,
         num_n_blocks,
         kReduceAckSemId},
        {{"REDUCE_SEND", "1"}},
        risc_1_config));

    // Col upper receivers (y<x, y>0): plain receiver
    desc.kernels.push_back(make_kernel_descriptor(
        receiver_path,
        make_core_range_set(col_upper_recv),
        {recv_tiles,
         tile_sz,
         kColSenderSem2Id,
         kColReceiverSem2Id,
         (uint32_t)tt::CBIndex::c_1,
         block_sz,
         num_m_blocks,
         num_n_blocks},
        {},
        risc_1_config));

    // === Diagonal helper kernels (x=grid_dim column) ===
    // Helper row receiver: REDUCE_RECV on RISCV_0 (receives diagonal's partial)
    desc.kernels.push_back(make_kernel_descriptor(
        receiver_path,
        helper_cores,
        {recv_tiles,
         tile_sz,
         kRowSenderSem2Id,
         kRowReceiverSem2Id,
         (uint32_t)tt::CBIndex::c_0,
         block_sz,
         (uint32_t)tt::CBIndex::c_5,
         kReduceSemId,
         num_m_blocks,
         M_block,
         num_n_blocks,
         kReduceAckSemId},
        {{"REDUCE_RECV", "1"}},
        risc_0_config));

    // Helper DRAM reader: reads odd K-columns, writes combined output (c_6) to DRAM
    std::vector<uint32_t> ct_dram = {
        recv_tiles,
        tile_sz,
        (uint32_t)tt::CBIndex::c_1,
        block_sz,
        (uint32_t)tt::CBIndex::c_6,
        out_tile_sz,
        Mpc,
        padded_out_tiles,
        num_m_blocks,
        M_block,
        num_n_blocks};
    TensorAccessorArgs(*in_buffer).append_to(ct_dram);
    TensorAccessorArgs(*out_buffer).append_to(ct_dram);
    auto dram_defines = std::map<std::string, std::string>{};
    if (mirror_active)
        dram_defines["MIRROR_OUTPUT"] = "1";
    desc.kernels.push_back(
        make_kernel_descriptor(helper_dram_reader_path, helper_cores, ct_dram, dram_defines, risc_1_config));

    // === Compute ===
    // Different defines per core group for reduction mode
    const std::vector<uint32_t> compute_ct = {
        K_half, K_block_tiles, Mpc, subblock_w, M_block, subblock_h, N_block, num_n_blocks};
    auto make_compute = [&](const CoreRangeSet& cores, const std::map<std::string, std::string>& defines) {
        return make_compute_kernel_descriptor(
            cores,
            compute_ct,
            defines,
            compute_matmul_path,
            /*fp32_dest_acc_en=*/true,
            /*unpack_to_dest_mode=*/{},
            attrs.math_fidelity,
            /*math_approx_mode=*/false);
    };

    // Lower triangle + diagonal: REDUCE_SENDER_TRANSPOSE. Diagonal cores send transposed
    // too — their even-K partial is symmetric, so the transposed send matches the helper's
    // row-major consume order under the senders' column-major iteration.
    std::vector<tt::tt_metal::CoreCoord> sender_transpose_cores;
    for (uint32_t y = 0; y < grid_dim; y++)
        for (uint32_t x = 0; x <= y; x++) sender_transpose_cores.push_back({x, y});
    desc.kernels.push_back(
        make_compute(make_core_range_set(sender_transpose_cores), {{"REDUCE_SENDER_TRANSPOSE", "1"}}));

    // Upper: REDUCE_ACCUMULATOR
    std::vector<tt::tt_metal::CoreCoord> accum_cores;
    for (uint32_t y = 0; y < grid_dim; y++)
        for (uint32_t x = y + 1; x < grid_dim; x++) accum_cores.push_back({x, y});
    auto accum_defines = std::map<std::string, std::string>{{"REDUCE_ACCUMULATOR", "1"}};
    if (mirror_active)
        accum_defines["MIRROR_OUTPUT"] = "1";
    desc.kernels.push_back(make_compute(make_core_range_set(accum_cores), accum_defines));

    // Helpers: REDUCE_ACCUMULATOR (no mirror — diagonal blocks are self-symmetric)
    desc.kernels.push_back(make_compute(helper_cores, {{"REDUCE_ACCUMULATOR", "1"}}));

    // === Runtime args ===
    // On a cache hit override_runtime_arguments() rewrites the input and output addresses (the k*AddrIdx slots).

    // --- Row sender+reduce at (0,0): sends partial to helper (grid_dim, 0) ---
    {
        auto row_lower_start_p = device->worker_core_from_logical_core({0, 0});
        auto row_lower_end_p = row_lower_start_p;
        auto row_upper_start_p = device->worker_core_from_logical_core({1, 0});
        auto row_upper_end_p = device->worker_core_from_logical_core({upper_x_end, 0});
        auto helper_p = device->worker_core_from_logical_core({grid_dim, 0});
        desc.kernels[kRowSenderReduceKernel].emplace_runtime_args(
            tt::tt_metal::CoreCoord{0, 0},
            {in_buffer,
             (uint32_t)row_lower_start_p.x,
             (uint32_t)row_lower_start_p.y,
             (uint32_t)row_lower_end_p.x,
             (uint32_t)row_lower_end_p.y,
             1u,  // lower_num_dests (diagonal only at row 0)
             (uint32_t)row_upper_start_p.x,
             (uint32_t)row_upper_start_p.y,
             (uint32_t)row_upper_end_p.x,
             (uint32_t)row_upper_end_p.y,
             upper_x_end,  // upper_num_dests
             0u,           // injector_keeps_odd (row 0 keeps even)
             1u,           // lower_loopback (diagonal is self)
             0u,           // tile_offset_row (grid row 0)
             Mpc,
             K_tiles,
             M_tiles,
             K_tiles,
             (uint32_t)helper_p.x,    // partner_noc_x (helper core)
             (uint32_t)helper_p.y});  // partner_noc_y
    }

    // --- Row sender at (0, y>0) ---
    for (uint32_t y = 1; y < grid_dim; y++) {
        tt::tt_metal::CoreCoord sender_core{0, y};
        uint32_t row_lower_num = y + 1;
        auto row_lower_start_p = device->worker_core_from_logical_core({0, y});
        auto row_lower_end_p = device->worker_core_from_logical_core({y, y});
        uint32_t row_upper_num = (upper_x_end > y) ? (upper_x_end - y) : 0;
        tt::tt_metal::CoreCoord row_upper_start_log =
            (row_upper_num > 0) ? tt::tt_metal::CoreCoord{y + 1, y} : tt::tt_metal::CoreCoord{0, y};
        tt::tt_metal::CoreCoord row_upper_end_log =
            (row_upper_num > 0) ? tt::tt_metal::CoreCoord{upper_x_end, y} : tt::tt_metal::CoreCoord{0, y};
        auto row_upper_start_p = device->worker_core_from_logical_core(row_upper_start_log);
        auto row_upper_end_p = device->worker_core_from_logical_core(row_upper_end_log);
        desc.kernels[kRowSenderKernel].emplace_runtime_args(
            sender_core,
            {in_buffer,
             (uint32_t)row_lower_start_p.x,
             (uint32_t)row_lower_start_p.y,
             (uint32_t)row_lower_end_p.x,
             (uint32_t)row_lower_end_p.y,
             row_lower_num,  // lower_num_dests
             (uint32_t)row_upper_start_p.x,
             (uint32_t)row_upper_start_p.y,
             (uint32_t)row_upper_end_p.x,
             (uint32_t)row_upper_end_p.y,
             row_upper_num,  // upper_num_dests
             0u,             // injector_keeps_odd (row senders keep even)
             1u,             // lower_loopback (sender is part of lower mcast)
             y * Mpc,        // tile_offset_row
             Mpc,
             K_tiles,
             M_tiles,
             K_tiles});
    }

    // --- Col sender at (x, 0) ---
    for (uint32_t x = 0; x < grid_dim; x++) {
        tt::tt_metal::CoreCoord sender_core{x, 0};
        uint32_t col_lower_num = grid_dim - x;
        tt::tt_metal::CoreCoord col_lower_start_log{x, x}, col_lower_end_log{x, grid_dim - 1};
        if (x == 0)
            col_lower_start_log = {0, 0};
        auto col_lower_start_p = device->worker_core_from_logical_core(col_lower_start_log);
        auto col_lower_end_p = device->worker_core_from_logical_core(col_lower_end_log);
        std::swap(col_lower_start_p, col_lower_end_p);  // NOC_1: swap
        uint32_t col_upper_num = (x >= 2) ? (x - 1) : 0;
        tt::tt_metal::CoreCoord col_upper_start_log =
            (col_upper_num > 0) ? tt::tt_metal::CoreCoord{x, 1} : tt::tt_metal::CoreCoord{x, 0};
        tt::tt_metal::CoreCoord col_upper_end_log =
            (col_upper_num > 0) ? tt::tt_metal::CoreCoord{x, x - 1} : tt::tt_metal::CoreCoord{x, 0};
        auto col_upper_start_p = device->worker_core_from_logical_core(col_upper_start_log);
        auto col_upper_end_p = device->worker_core_from_logical_core(col_upper_end_log);
        std::swap(col_upper_start_p, col_upper_end_p);  // NOC_1: swap
        uint32_t keeps_odd = (x > 0) ? 1u : 0u;
        uint32_t loopback = (x == 0) ? 1u : 0u;
        desc.kernels[kColSenderKernel].emplace_runtime_args(
            sender_core,
            {in_buffer,
             (uint32_t)col_lower_start_p.x,
             (uint32_t)col_lower_start_p.y,
             (uint32_t)col_lower_end_p.x,
             (uint32_t)col_lower_end_p.y,
             col_lower_num,
             (uint32_t)col_upper_start_p.x,
             (uint32_t)col_upper_start_p.y,
             (uint32_t)col_upper_end_p.x,
             (uint32_t)col_upper_end_p.y,
             col_upper_num,
             keeps_odd,
             loopback,
             x * Mpc,  // tile_offset_row
             Mpc,
             K_tiles,
             M_tiles,
             K_tiles});
    }

    // --- Row receiver runtime args (x>0, all y) ---
    for (uint32_t y = 0; y < grid_dim; y++) {
        auto row_sender_p = device->worker_core_from_logical_core({0, y});
        for (uint32_t x = 1; x < grid_dim; x++) {
            tt::tt_metal::CoreCoord core{x, y};
            if (x < y) {
                // Lower off-diagonal: REDUCE_SEND to upper core (y, x)
                auto partner_p = device->worker_core_from_logical_core({y, x});
                desc.kernels[kRowLowerOffdiagKernel].emplace_runtime_args(
                    core,
                    {(uint32_t)row_sender_p.x, (uint32_t)row_sender_p.y, (uint32_t)partner_p.x, (uint32_t)partner_p.y});
            } else if (x == y) {
                // Lower diagonal: REDUCE_SEND to helper (grid_dim, y)
                auto partner_p = device->worker_core_from_logical_core({grid_dim, y});
                desc.kernels[kRowLowerDiagKernel].emplace_runtime_args(
                    core,
                    {(uint32_t)row_sender_p.x, (uint32_t)row_sender_p.y, (uint32_t)partner_p.x, (uint32_t)partner_p.y});
            } else {
                // Upper: REDUCE_RECV, write combined output to DRAM
                uint32_t M_start_tile = y * Mpc;
                uint32_t N_start_tile = x * Mpc;
                KernelDescriptor::RTArgList rt;
                rt.push_back((uint32_t)row_sender_p.x);
                rt.push_back((uint32_t)row_sender_p.y);
                rt.push_back(out_buffer);
                rt.push_back(M_start_tile);
                rt.push_back(N_start_tile);
                rt.push_back(M_tiles);
                if (mirror_active) {
                    rt.push_back(N_start_tile);  // mirror_M_start = N_start (swapped)
                    rt.push_back(M_start_tile);  // mirror_N_start = M_start (swapped)
                }
                // Reduce partner (lower core) — target of the consumption credit
                auto partner_p = device->worker_core_from_logical_core({y, x});
                rt.push_back((uint32_t)partner_p.x);
                rt.push_back((uint32_t)partner_p.y);
                desc.kernels[kRowUpperRecvKernel].emplace_runtime_args(core, rt);
            }
        }
    }

    // --- Col receiver runtime args (y>0) ---
    for (uint32_t x = 0; x < grid_dim; x++) {
        auto col_sender_p = device->worker_core_from_logical_core({x, 0});
        for (uint32_t y = 1; y < grid_dim; y++) {
            tt::tt_metal::CoreCoord core{x, y};
            if (x == 0) {
                // Edge lower (0, y>0): REDUCE_SEND to upper core (y, 0)
                auto partner_p = device->worker_core_from_logical_core({y, 0});
                desc.kernels[kColLowerRecvEdgeKernel].emplace_runtime_args(
                    core,
                    {(uint32_t)col_sender_p.x, (uint32_t)col_sender_p.y, (uint32_t)partner_p.x, (uint32_t)partner_p.y});
            } else if (y >= x) {
                desc.kernels[kColLowerRecvKernel].emplace_runtime_args(
                    core, {(uint32_t)col_sender_p.x, (uint32_t)col_sender_p.y});
            } else {
                desc.kernels[kColUpperRecvKernel].emplace_runtime_args(
                    core, {(uint32_t)col_sender_p.x, (uint32_t)col_sender_p.y});
            }
        }
    }

    // --- Helper runtime args ---
    for (uint32_t y = 0; y < grid_dim; y++) {
        tt::tt_metal::CoreCoord helper_core{grid_dim, y};
        auto row_sender_p = device->worker_core_from_logical_core({0, y});

        // RISCV_0: row receiver + REDUCE_RECV (receives diagonal's partial).
        // Reduce partner is the diagonal core (y, y) — target of the consumption credit.
        auto diag_p = device->worker_core_from_logical_core({y, y});
        desc.kernels[kHelperRecvKernel].emplace_runtime_args(
            helper_core, {(uint32_t)row_sender_p.x, (uint32_t)row_sender_p.y, (uint32_t)diag_p.x, (uint32_t)diag_p.y});

        // RISCV_1: DRAM reader + output writer for diagonal block G[y,y]
        desc.kernels[kHelperDramReaderKernel].emplace_runtime_args(
            helper_core,
            {in_buffer,
             y * Mpc,  // tile_offset_row
             K_tiles,
             out_buffer,
             y * Mpc,  // M_start_tile (diagonal)
             y * Mpc,  // N_start_tile (diagonal)
             M_tiles,
             K_tiles});
    }

    return desc;
}

void KSplitGramMatmulProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_coordinate*/) {
    const uint32_t in_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t out_addr = output.buffer()->address();
    const uint32_t grid_dim = gram_grid_dim(*tensor_args.input_tensor.device());

    // The input address, on the row sender+reduce at (0,0), the row senders (0, y>0) and the col senders (x, 0)
    auto& row_sender_reduce_rt = tt::tt_metal::GetRuntimeArgs(program, kRowSenderReduceKernel);
    row_sender_reduce_rt[0][0][kSenderInAddrIdx] = in_addr;
    auto& row_sender_rt = tt::tt_metal::GetRuntimeArgs(program, kRowSenderKernel);
    for (uint32_t y = 1; y < grid_dim; y++) {
        row_sender_rt[0][y][kSenderInAddrIdx] = in_addr;
    }
    auto& col_sender_rt = tt::tt_metal::GetRuntimeArgs(program, kColSenderKernel);
    for (uint32_t x = 0; x < grid_dim; x++) {
        col_sender_rt[x][0][kSenderInAddrIdx] = in_addr;
    }

    // The output address, on the row upper receivers (x > y), which write the combined output
    auto& row_upper_recv_rt = tt::tt_metal::GetRuntimeArgs(program, kRowUpperRecvKernel);
    for (uint32_t y = 0; y < grid_dim; y++) {
        for (uint32_t x = y + 1; x < grid_dim; x++) {
            row_upper_recv_rt[x][y][kUpperRecvOutAddrIdx] = out_addr;
        }
    }

    // Both addresses, on the diagonal helpers' DRAM reader (x = grid_dim)
    auto& helper_dram_reader_rt = tt::tt_metal::GetRuntimeArgs(program, kHelperDramReaderKernel);
    for (uint32_t y = 0; y < grid_dim; y++) {
        helper_dram_reader_rt[grid_dim][y][kHelperInAddrIdx] = in_addr;
        helper_dram_reader_rt[grid_dim][y][kHelperOutAddrIdx] = out_addr;
    }
}

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
