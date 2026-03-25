// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "k_split_gram_matmul_program_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace ttml::metal::ops::k_split_gram_matmul::device {

namespace {
tt::tt_metal::CoreRangeSet make_crs(const std::vector<tt::tt_metal::CoreCoord>& cores) {
    std::set<tt::tt_metal::CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.insert(tt::tt_metal::CoreRange(c, c));
    }
    return tt::tt_metal::CoreRangeSet(ranges);
}
}  // namespace

KSplitGramMatmulProgramFactory::cached_program_t KSplitGramMatmulProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    Program program = CreateProgram();
    const auto& input = tensor_args.input_tensor;
    auto* device = input.device();

    auto device_grid = device->compute_with_storage_grid_size();
    // Largest square grid with room for a helper column at x=grid_dim
    uint32_t grid_dim = static_cast<uint32_t>(std::min(device_grid.x - 1, device_grid.y));

    uint32_t logical_M_tiles = input.logical_shape()[-2] / tt::constants::TILE_HEIGHT;
    uint32_t logical_K_tiles = input.logical_shape()[-1] / tt::constants::TILE_WIDTH;

    uint32_t padded_M_tiles = tt::round_up(logical_M_tiles, grid_dim);
    uint32_t Mpc = padded_M_tiles / grid_dim;

    auto tile_format = tt::DataFormat::Float16_b;
    auto tile_sz = tt::tile_size(tile_format);
    uint32_t K_tiles = logical_K_tiles;
    uint32_t K_half = K_tiles / 2;

    auto full_grid = tt::tt_metal::CoreRange({0, 0}, {grid_dim - 1, grid_dim - 1});

    // Diagonal helper column: x=grid_dim cores compute odd-K partial for diagonal blocks.
    tt::tt_metal::CoreRange helper_range({grid_dim, 0}, {grid_dim, grid_dim - 1});
    auto all_cores = tt::tt_metal::CoreRangeSet(std::set<tt::tt_metal::CoreRange>{full_grid, helper_range});
    // Upper x-bound for row multicast (includes helper column)
    uint32_t upper_x_end = grid_dim;

    uint32_t subblock_h = 2;
    uint32_t subblock_w = std::min(Mpc, 2u);

    auto intermed_format = tt::DataFormat::Float32;
    auto intermed_tile_sz = tt::tile_size(intermed_format);
    auto out_tile_format = tt::DataFormat::Float16_b;
    auto out_tile_sz = tt::tile_size(out_tile_format);

    bool mirror = (attrs.output_mode == ttml::metal::OutputMode::Full);

    // Joint optimization of K_block_tiles and M_block with N_block = M_block streaming.
    // Mirror mode adds c_4 (mb tiles) + c_7 (mb tiles) = +2*mb*out_sz to linear term.
    const uint32_t L1_BUDGET =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    uint32_t mirror_out_overhead = mirror ? 2 * out_tile_sz : 0;  // per-mb extra for c_4 + c_7

    uint32_t best_subs = UINT32_MAX;
    uint32_t best_kb = 0, best_mb = 0, best_db = 1;

    for (uint32_t kb = std::min(K_half, 8u); kb >= 1; kb--) {
        if (K_half % kb != 0)
            continue;
        for (uint32_t db : {2u, 1u}) {
            uint32_t lo = 1, hi = Mpc, mb = 0;
            while (lo <= hi) {
                uint32_t mid = (lo + hi) / 2;
                uint32_t l1 = mid * mid * intermed_tile_sz +
                              mid * (2 * Mpc * out_tile_sz + out_tile_sz + mirror_out_overhead + 2 * db * kb * tile_sz);
                if (l1 <= L1_BUDGET) {
                    mb = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            if (mb == 0)
                continue;
            uint32_t subs = (Mpc + mb - 1) / mb;
            bool better = (subs < best_subs) || (subs == best_subs && db > best_db) ||
                          (subs == best_subs && db == best_db && kb > best_kb);
            if (better) {
                best_subs = subs;
                best_kb = kb;
                best_mb = mb;
                best_db = db;
            }
        }
    }
    TT_FATAL(best_mb > 0, "Cannot fit mcast gram matmul CBs in L1");

    // Per-nsb reduction: eliminates c_3, shrinks c_5 to mb*nb=mb².
    uint32_t pernsb_best_subs = UINT32_MAX;
    uint32_t pernsb_best_kb = 0, pernsb_best_mb = 0, pernsb_best_db = 1;

    for (uint32_t kb = std::min(K_half, 8u); kb >= 1; kb--) {
        if (K_half % kb != 0)
            continue;
        for (uint32_t db : {2u, 1u}) {
            uint32_t lo = 1, hi = Mpc, mb = 0;
            while (lo <= hi) {
                uint32_t mid = (lo + hi) / 2;
                uint32_t l1 = mid * mid * (intermed_tile_sz + out_tile_sz) +
                              mid * (out_tile_sz + mirror_out_overhead + 2 * db * kb * tile_sz);
                if (l1 <= L1_BUDGET) {
                    mb = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            if (mb == 0)
                continue;
            uint32_t subs = (Mpc + mb - 1) / mb;
            bool better = (subs < pernsb_best_subs) || (subs == pernsb_best_subs && db > pernsb_best_db) ||
                          (subs == pernsb_best_subs && db == pernsb_best_db && kb > pernsb_best_kb);
            if (better) {
                pernsb_best_subs = subs;
                pernsb_best_kb = kb;
                pernsb_best_mb = mb;
                pernsb_best_db = db;
            }
        }
    }

    bool use_per_nsb_reduction = (pernsb_best_subs == 1 && best_subs > 1);
    if (use_per_nsb_reduction) {
        best_subs = pernsb_best_subs;
        best_kb = pernsb_best_kb;
        best_mb = pernsb_best_mb;
        best_db = pernsb_best_db;
    }

    best_subs = (Mpc + best_mb - 1) / best_mb;
    use_per_nsb_reduction = use_per_nsb_reduction && (best_subs == 1);

    uint32_t K_block_tiles = best_kb;
    uint32_t M_block = best_mb;
    uint32_t N_block = M_block;
    uint32_t db_factor = best_db;
    uint32_t M_num_subblocks = best_subs;
    uint32_t N_num_subblocks = (Mpc + N_block - 1) / N_block;
    bool mirror_active = mirror;
    uint32_t block_sz = K_block_tiles * M_block;
    uint32_t cb_size = db_factor * block_sz;
    uint32_t num_tiles = M_block * K_tiles;  // tiles per sender per nsb pass
    // Output tensor is logical [M, M] — width in tiles matches logical_M_tiles
    uint32_t padded_out_tiles = logical_M_tiles;
    uint32_t recv_tiles = K_half * M_block;  // tiles per receiver per nsb pass

    // CB for sender output: c_3 (per-msb) or c_5 (per-nsb)
    uint32_t send_out_cb = use_per_nsb_reduction ? (uint32_t)tt::CBIndex::c_5 : (uint32_t)tt::CBIndex::c_3;
    uint32_t c5_tiles = use_per_nsb_reduction ? M_block * N_block : Mpc * M_block;

    auto create_all_cbs = [&](const tt::tt_metal::CoreRange& range) {
        create_circular_buffer(program, range, (uint32_t)tt::CBIndex::c_0, tile_format, tile_sz, cb_size);
        create_circular_buffer(program, range, (uint32_t)tt::CBIndex::c_1, tile_format, tile_sz, cb_size);
        create_circular_buffer(
            program, range, (uint32_t)tt::CBIndex::c_2, intermed_format, intermed_tile_sz, M_block * N_block);
        if (!use_per_nsb_reduction) {
            create_circular_buffer(
                program, range, (uint32_t)tt::CBIndex::c_3, out_tile_format, out_tile_sz, Mpc * M_block);
        }
        if (mirror_active) {
            // c_4: mirror buffer for transposed tiles (M_block tiles, streamed col by col)
            create_circular_buffer(program, range, (uint32_t)tt::CBIndex::c_4, out_tile_format, out_tile_sz, M_block);
            // c_7: staging for transpose (mb tiles BF16, batched per-column)
            create_circular_buffer(program, range, (uint32_t)tt::CBIndex::c_7, out_tile_format, out_tile_sz, M_block);
        }
        create_circular_buffer(program, range, (uint32_t)tt::CBIndex::c_5, out_tile_format, out_tile_sz, c5_tiles);
        create_circular_buffer(program, range, (uint32_t)tt::CBIndex::c_6, out_tile_format, out_tile_sz, M_block);
    };
    create_all_cbs(full_grid);
    {  // helpers
        create_all_cbs(helper_range);
    }

    // Semaphores on all cores (helpers need row sems for multicast reception)
    auto row_sender_sem = CreateSemaphore(program, all_cores, INVALID);
    auto row_receiver_sem = CreateSemaphore(program, all_cores, INVALID);
    auto row_sender_sem2 = CreateSemaphore(program, all_cores, INVALID);
    auto row_receiver_sem2 = CreateSemaphore(program, all_cores, INVALID);
    auto col_sender_sem = CreateSemaphore(program, all_cores, INVALID);
    auto col_receiver_sem = CreateSemaphore(program, all_cores, INVALID);
    auto col_sender_sem2 = CreateSemaphore(program, all_cores, INVALID);
    auto col_receiver_sem2 = CreateSemaphore(program, all_cores, INVALID);
    auto reduce_sem = CreateSemaphore(program, all_cores, 0);

    auto noc_1 = detail::preferred_noc_for_dram_write(device->arch());
    auto noc_0 = detail::preferred_noc_for_dram_read(device->arch());
    auto risc_1 = DataMovementProcessor::RISCV_1;
    auto risc_0 = DataMovementProcessor::RISCV_0;

    uint32_t in_addr = input.buffer()->address();

    uint32_t out_addr = output.buffer()->address();

    const std::string base = "tt-train/sources/ttml/metal/ops/k_split_gram_matmul/device/kernels/";
    const std::string sender_path = base + "reader_mcast_sender.cpp";
    const std::string receiver_path = base + "reader_mcast_receiver.cpp";
    const std::string receiver_writer_path = base + "reader_mcast_receiver_writer.cpp";
    const std::string compute_matmul_path = base + "compute_matmul.cpp";
    const std::string dram_reader_col_path = base + "dram_reader_col.cpp";

    // === Core classification ===
    // Row senders: x=0
    std::vector<tt::tt_metal::CoreCoord> row_senders;  // (0, y>0) — regular sender
    for (uint32_t y = 1; y < grid_dim; y++) row_senders.push_back({0, y});
    // (0, 0) is sender_writer — handled separately

    // Col senders: y=0 (all x)
    std::vector<tt::tt_metal::CoreCoord> col_senders;
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

    // === Sender kernels ===
    // Row sender+reduce for (0,0): RISCV_0/NOC_0, sends partial to helper (10,0)
    KernelHandle row_sender_reduce_kid;
    {
        std::vector<uint32_t> ct = {
            num_tiles,
            tile_sz,
            row_sender_sem,
            row_receiver_sem,
            row_sender_sem2,
            row_receiver_sem2,
            (uint32_t)tt::CBIndex::c_0,
            block_sz,
            cb_size,
            send_out_cb,
            out_tile_sz,
            (uint32_t)tt::CBIndex::c_5,
            reduce_sem,
            M_num_subblocks,
            M_block,
            N_num_subblocks};
        TensorAccessorArgs(*input.buffer()).append_to(ct);
        row_sender_reduce_kid = CreateKernel(
            program,
            sender_path,
            make_crs({{0, 0}}),
            DataMovementConfig{
                .processor = risc_0, .noc = noc_0, .compile_args = ct, .defines = {{"SENDER_REDUCE_SEND", "1"}}});
    }

    // Row sender for (0, y>0): RISCV_0/NOC_0, CB c_0
    KernelHandle row_sender_kid = 0;
    if (!row_senders.empty()) {
        std::vector<uint32_t> ct = {
            num_tiles,
            tile_sz,
            row_sender_sem,
            row_receiver_sem,
            row_sender_sem2,
            row_receiver_sem2,
            (uint32_t)tt::CBIndex::c_0,
            block_sz,
            cb_size,
            M_num_subblocks,
            N_num_subblocks};
        TensorAccessorArgs(*input.buffer()).append_to(ct);
        row_sender_kid = CreateKernel(
            program,
            sender_path,
            make_crs(row_senders),
            DataMovementConfig{.processor = risc_0, .noc = noc_0, .compile_args = ct});
    }

    // Col sender: RISCV_1/NOC_1, CB c_1
    KernelHandle col_sender_kid;
    {
        std::vector<uint32_t> ct = {
            num_tiles,
            tile_sz,
            col_sender_sem,
            col_receiver_sem,
            col_sender_sem2,
            col_receiver_sem2,
            (uint32_t)tt::CBIndex::c_1,
            block_sz,
            cb_size,
            M_num_subblocks,
            N_num_subblocks};
        TensorAccessorArgs(*input.buffer()).append_to(ct);
        col_sender_kid = CreateKernel(
            program,
            sender_path,
            make_crs(col_senders),
            DataMovementConfig{.processor = risc_1, .noc = noc_1, .compile_args = ct});
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
            reduce_sem,
            M_num_subblocks,
            M_block,
            N_num_subblocks};
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
            reduce_sem,
            M_num_subblocks,
            M_block,
            N_num_subblocks};
        TensorAccessorArgs(*output.buffer()).append_to(ct);
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

    KernelHandle row_lower_offdiag_kid = 0, row_lower_diag_kid = 0;
    KernelHandle row_upper_recv_kid = 0;
    KernelHandle col_lower_recv_kid = 0, col_lower_recv_edge_kid = 0, col_upper_recv_kid = 0;

    if (!row_lower_offdiag.empty()) {
        auto ct = make_recv_send_ct(row_sender_sem, row_receiver_sem);
        row_lower_offdiag_kid = CreateKernel(
            program,
            receiver_writer_path,
            make_crs(row_lower_offdiag),
            DataMovementConfig{
                .processor = risc_0, .noc = noc_0, .compile_args = ct, .defines = {{"REDUCE_SEND", "1"}}});
    }

    // Lower diagonal (x == y, y > 0): REDUCE_SEND — send partial to helper (no transpose)
    if (!row_lower_diag.empty()) {
        auto ct = make_recv_send_ct(row_sender_sem, row_receiver_sem);
        row_lower_diag_kid = CreateKernel(
            program,
            receiver_writer_path,
            make_crs(row_lower_diag),
            DataMovementConfig{
                .processor = risc_0, .noc = noc_0, .compile_args = ct, .defines = {{"REDUCE_SEND", "1"}}});
    }

    // Row upper receivers (x > y): REDUCE_RECV — receive partner's partial, write combined
    if (!row_upper_recv.empty()) {
        auto ct = make_recv_write_ct(row_sender_sem2, row_receiver_sem2);
        auto recv_defines = std::map<std::string, std::string>{{"REDUCE_RECV", "1"}};
        if (use_per_nsb_reduction)
            recv_defines["PER_NSB_REDUCTION"] = "1";
        if (mirror_active)
            recv_defines["MIRROR_OUTPUT"] = "1";
        row_upper_recv_kid = CreateKernel(
            program,
            receiver_writer_path,
            make_crs(row_upper_recv),
            DataMovementConfig{.processor = risc_0, .noc = noc_0, .compile_args = ct, .defines = recv_defines});
    }

    // Col lower receivers (interior, x>0, y≥x): plain receiver
    if (!col_lower_recv_interior.empty()) {
        std::vector<uint32_t> ct = {
            recv_tiles,
            tile_sz,
            col_sender_sem,
            col_receiver_sem,
            (uint32_t)tt::CBIndex::c_1,
            block_sz,
            M_num_subblocks,
            N_num_subblocks};
        col_lower_recv_kid = CreateKernel(
            program,
            receiver_path,
            make_crs(col_lower_recv_interior),
            DataMovementConfig{.processor = risc_1, .noc = noc_1, .compile_args = ct});
    }

    // Col lower receivers (edge, x=0, y>0): REDUCE_SEND to upper core (y, 0)
    if (!col_lower_recv_edge.empty()) {
        std::vector<uint32_t> ct = {
            recv_tiles,
            tile_sz,
            col_sender_sem,
            col_receiver_sem,
            (uint32_t)tt::CBIndex::c_1,
            block_sz,
            send_out_cb,
            out_tile_sz,
            Mpc,
            (uint32_t)tt::CBIndex::c_5,
            reduce_sem,
            M_num_subblocks,
            M_block,
            N_num_subblocks};
        col_lower_recv_edge_kid = CreateKernel(
            program,
            receiver_writer_path,
            make_crs(col_lower_recv_edge),
            DataMovementConfig{
                .processor = risc_1, .noc = noc_1, .compile_args = ct, .defines = {{"REDUCE_SEND", "1"}}});
    }

    // Col upper receivers (y<x, y>0): plain receiver
    if (!col_upper_recv.empty()) {
        std::vector<uint32_t> ct = {
            recv_tiles,
            tile_sz,
            col_sender_sem2,
            col_receiver_sem2,
            (uint32_t)tt::CBIndex::c_1,
            block_sz,
            M_num_subblocks,
            N_num_subblocks};
        col_upper_recv_kid = CreateKernel(
            program,
            receiver_path,
            make_crs(col_upper_recv),
            DataMovementConfig{.processor = risc_1, .noc = noc_1, .compile_args = ct});
    }

    // === Diagonal helper kernels (x=grid_dim column) ===
    KernelHandle helper_recv_kid = 0, helper_dram_reader_kid = 0;
    {  // helpers
        // Helper row receiver: REDUCE_RECV on RISCV_0 (receives diagonal's partial)
        std::vector<uint32_t> ct_recv = {
            recv_tiles,
            tile_sz,
            row_sender_sem2,
            row_receiver_sem2,
            (uint32_t)tt::CBIndex::c_0,
            block_sz,
            (uint32_t)tt::CBIndex::c_5,
            reduce_sem,
            Mpc,
            M_num_subblocks,
            M_block,
            N_num_subblocks};
        helper_recv_kid = CreateKernel(
            program,
            receiver_path,
            helper_range,
            DataMovementConfig{
                .processor = risc_0, .noc = noc_0, .compile_args = ct_recv, .defines = {{"REDUCE_RECV", "1"}}});

        // Helper DRAM reader: reads odd col tiles, then writes combined output (c_6) to DRAM
        std::vector<uint32_t> ct_dram = {
            recv_tiles,
            tile_sz,
            (uint32_t)tt::CBIndex::c_1,
            block_sz,
            (uint32_t)tt::CBIndex::c_6,
            out_tile_sz,
            Mpc,
            padded_out_tiles,
            M_num_subblocks,
            M_block,
            N_num_subblocks};
        TensorAccessorArgs(*input.buffer()).append_to(ct_dram);
        TensorAccessorArgs(*output.buffer()).append_to(ct_dram);
        auto dram_defines = std::map<std::string, std::string>{};
        if (use_per_nsb_reduction)
            dram_defines["PER_NSB_REDUCTION"] = "1";
        if (mirror_active)
            dram_defines["MIRROR_OUTPUT"] = "1";
        helper_dram_reader_kid = CreateKernel(
            program,
            dram_reader_col_path,
            helper_range,
            DataMovementConfig{.processor = risc_1, .noc = noc_1, .compile_args = ct_dram, .defines = dram_defines});
    }

    // === Compute ===
    // Different defines per core group for reduction mode
    auto compute_cfg = [&](std::map<std::string, std::string> defines = {}) {
        if (use_per_nsb_reduction)
            defines["PER_NSB_REDUCTION"] = "1";
        return ComputeConfig{
            .fp32_dest_acc_en = true,
            .compile_args = {K_half, K_block_tiles, Mpc, subblock_w, M_block, subblock_h, N_block, N_num_subblocks},
            .defines = defines};
    };

    // Lower off-diagonal: REDUCE_SENDER_TRANSPOSE
    std::vector<tt::tt_metal::CoreCoord> sender_transpose_cores;
    for (uint32_t y = 1; y < grid_dim; y++)
        for (uint32_t x = 0; x < y; x++) sender_transpose_cores.push_back({x, y});
    if (!sender_transpose_cores.empty()) {
        CreateKernel(
            program,
            compute_matmul_path,
            make_crs(sender_transpose_cores),
            compute_cfg({{"REDUCE_SENDER_TRANSPOSE", "1"}}));
    }

    // Diagonal: REDUCE_SENDER
    std::vector<tt::tt_metal::CoreCoord> sender_diag_cores;
    for (uint32_t d = 0; d < grid_dim; d++) sender_diag_cores.push_back({d, d});
    CreateKernel(program, compute_matmul_path, make_crs(sender_diag_cores), compute_cfg({{"REDUCE_SENDER", "1"}}));

    // Upper: REDUCE_ACCUMULATOR (or diagnostic variant)
    std::string accum_define = "REDUCE_ACCUMULATOR";

    std::vector<tt::tt_metal::CoreCoord> accum_cores;
    for (uint32_t y = 0; y < grid_dim; y++)
        for (uint32_t x = y + 1; x < grid_dim; x++) accum_cores.push_back({x, y});
    if (!accum_cores.empty()) {
        auto defines = std::map<std::string, std::string>{{accum_define, "1"}};
        if (mirror_active)
            defines["MIRROR_OUTPUT"] = "1";
        CreateKernel(program, compute_matmul_path, make_crs(accum_cores), compute_cfg(defines));
    }

    // Helpers: REDUCE_ACCUMULATOR (no mirror — diagonal blocks are self-symmetric)
    {  // helpers
        CreateKernel(program, compute_matmul_path, helper_range, compute_cfg({{accum_define, "1"}}));
    }

    // === Runtime args ===

    // --- Row sender+reduce at (0,0): sends partial to helper (grid_dim, 0) ---
    {
        auto row_lower_start_p = device->worker_core_from_logical_core({0, 0});
        auto row_lower_end_p = row_lower_start_p;
        auto row_upper_start_p = device->worker_core_from_logical_core({1, 0});
        auto row_upper_end_p = device->worker_core_from_logical_core({upper_x_end, 0});
        auto helper_p = device->worker_core_from_logical_core({grid_dim, 0});
        SetRuntimeArgs(
            program,
            row_sender_reduce_kid,
            tt::tt_metal::CoreCoord{0, 0},
            {in_addr,
             (uint32_t)row_lower_start_p.x,
             (uint32_t)row_lower_start_p.y,
             (uint32_t)row_lower_end_p.x,
             (uint32_t)row_lower_end_p.y,
             1u,
             (uint32_t)row_upper_start_p.x,
             (uint32_t)row_upper_start_p.y,
             (uint32_t)row_upper_end_p.x,
             (uint32_t)row_upper_end_p.y,
             upper_x_end,  // num upper dests
             0u,
             1u,
             0u,  // tile_offset_row = 0 (grid row 0)
             Mpc,
             K_tiles,
             logical_M_tiles,
             logical_K_tiles,
             (uint32_t)helper_p.x,
             (uint32_t)helper_p.y});
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
        SetRuntimeArgs(
            program,
            row_sender_kid,
            sender_core,
            {in_addr,
             (uint32_t)row_lower_start_p.x,
             (uint32_t)row_lower_start_p.y,
             (uint32_t)row_lower_end_p.x,
             (uint32_t)row_lower_end_p.y,
             row_lower_num,
             (uint32_t)row_upper_start_p.x,
             (uint32_t)row_upper_start_p.y,
             (uint32_t)row_upper_end_p.x,
             (uint32_t)row_upper_end_p.y,
             row_upper_num,
             0u,
             1u,
             y * Mpc,  // tile_offset_row (not flat)
             Mpc,
             K_tiles,
             logical_M_tiles,
             logical_K_tiles});
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
        SetRuntimeArgs(
            program,
            col_sender_kid,
            sender_core,
            {in_addr,
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
             logical_M_tiles,
             logical_K_tiles});
    }

    // --- Row receiver runtime args (x>0, all y) ---
    for (uint32_t y = 0; y < grid_dim; y++) {
        auto row_sender_p = device->worker_core_from_logical_core({0, y});
        for (uint32_t x = 1; x < grid_dim; x++) {
            tt::tt_metal::CoreCoord core{x, y};
            if (x < y) {
                // Lower off-diagonal: REDUCE_SEND to upper core (y, x)
                auto partner_p = device->worker_core_from_logical_core({y, x});
                SetRuntimeArgs(
                    program,
                    row_lower_offdiag_kid,
                    core,
                    {(uint32_t)row_sender_p.x, (uint32_t)row_sender_p.y, (uint32_t)partner_p.x, (uint32_t)partner_p.y});
            } else if (x == y) {
                // Lower diagonal: REDUCE_SEND to helper (grid_dim, y)
                auto partner_p = device->worker_core_from_logical_core({grid_dim, y});
                SetRuntimeArgs(
                    program,
                    row_lower_diag_kid,
                    core,
                    {(uint32_t)row_sender_p.x, (uint32_t)row_sender_p.y, (uint32_t)partner_p.x, (uint32_t)partner_p.y});
            } else {
                // Upper: REDUCE_RECV, write combined output to DRAM
                uint32_t M_start_tile = y * Mpc;
                uint32_t N_start_tile = x * Mpc;
                std::vector<uint32_t> rt = {
                    (uint32_t)row_sender_p.x,
                    (uint32_t)row_sender_p.y,
                    out_addr,
                    M_start_tile,
                    N_start_tile,
                    logical_M_tiles};
                if (mirror_active) {
                    rt.push_back(N_start_tile);  // mirror_M_start = N_start (swapped)
                    rt.push_back(M_start_tile);  // mirror_N_start = M_start (swapped)
                }
                SetRuntimeArgs(program, row_upper_recv_kid, core, rt);
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
                SetRuntimeArgs(
                    program,
                    col_lower_recv_edge_kid,
                    core,
                    {(uint32_t)col_sender_p.x, (uint32_t)col_sender_p.y, (uint32_t)partner_p.x, (uint32_t)partner_p.y});
            } else if (y >= x) {
                SetRuntimeArgs(program, col_lower_recv_kid, core, {(uint32_t)col_sender_p.x, (uint32_t)col_sender_p.y});
            } else {
                SetRuntimeArgs(program, col_upper_recv_kid, core, {(uint32_t)col_sender_p.x, (uint32_t)col_sender_p.y});
            }
        }
    }

    // --- Helper runtime args ---
    {  // helpers
        for (uint32_t y = 0; y < grid_dim; y++) {
            tt::tt_metal::CoreCoord helper_core{grid_dim, y};
            auto row_sender_p = device->worker_core_from_logical_core({0, y});

            // RISCV_0: row receiver + REDUCE_RECV (receives diagonal's partial)
            SetRuntimeArgs(program, helper_recv_kid, helper_core, {(uint32_t)row_sender_p.x, (uint32_t)row_sender_p.y});

            // RISCV_1: DRAM reader + output writer for diagonal block G[y,y]
            SetRuntimeArgs(
                program,
                helper_dram_reader_kid,
                helper_core,
                {in_addr,
                 y * Mpc,  // tile_offset_row
                 Mpc,
                 K_tiles,
                 out_addr,
                 y * Mpc,
                 y * Mpc,
                 logical_M_tiles,
                 logical_K_tiles});  // M_start = y*Mpc, N_start = y*Mpc (diagonal)
        }
    }

    return {std::move(program), shared_variables_t{}};
}

void KSplitGramMatmulProgramFactory::override_runtime_arguments(
    cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
}

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
