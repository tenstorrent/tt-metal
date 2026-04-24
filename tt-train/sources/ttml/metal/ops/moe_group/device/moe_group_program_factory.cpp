// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_group_program_factory.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kCombinedKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_group/device/kernels/dataflow/moe_group_combined_kernel.cpp";
constexpr auto kWorkerWriterPath =
    "tt-train/sources/ttml/metal/ops/moe_group/device/kernels/dataflow/moe_group_worker_writer.cpp";
constexpr auto kTilizeKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_group/device/kernels/compute/moe_group_tilize_kernel.cpp";

// CB indices
constexpr uint32_t kCbSrc0 = tt::CBIndex::c_0;
constexpr uint32_t kCbOut = tt::CBIndex::c_2;
constexpr uint32_t kCbScan = tt::CBIndex::c_3;
constexpr uint32_t kCbPlan = tt::CBIndex::c_4;
constexpr uint32_t kCbOffset = tt::CBIndex::c_5;

constexpr uint32_t kTargetChunkBytes = 128U * 1024U;

uint32_t pick_num_chunks(uint32_t h) {
    uint32_t row_bytes = h * 2U;
    uint32_t strip_bytes = 32U * row_bytes;
    uint32_t nc = (strip_bytes + kTargetChunkBytes - 1U) / kTargetChunkBytes;
    if (nc == 0U)
        nc = 1U;
    uint32_t Wt = h / tt::constants::TILE_WIDTH;
    if (nc > Wt)
        nc = Wt;
    return nc;
}

}  // namespace

namespace ttml::metal::ops::moe_group::device {

MoeGroupProgramFactory::cached_program_t MoeGroupProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& outputs) {
    auto* device = args.dispatched.device();
    auto& grouped = std::get<0>(outputs);
    auto& counts = std::get<1>(outputs);
    auto& offsets = std::get<2>(outputs);
    auto& plan = std::get<3>(outputs);

    tt::tt_metal::Program program{};

    const uint32_t h = attrs.h;
    const uint32_t e_local = attrs.e_local;
    const uint32_t k = attrs.k;
    const uint32_t total_rows = attrs.d * attrs.b * attrs.s;
    const uint32_t t_cap = attrs.t_cap;

    const uint32_t num_chunks = pick_num_chunks(h);
    const uint32_t Wt = h / tt::constants::TILE_WIDTH;
    const uint32_t tiles_per_chunk = (Wt + num_chunks - 1U) / num_chunks;
    const uint32_t hidden_chunk_bytes = tiles_per_chunk * tt::constants::TILE_WIDTH * 2U;
    const uint32_t last_chunk_tiles = Wt - (num_chunks - 1U) * tiles_per_chunk;
    const uint32_t last_chunk_bytes = last_chunk_tiles * tt::constants::TILE_WIDTH * 2U;

    // -------------------------------------------------------------------------
    // Core assignment: ALL cores run the combined scan+reader kernel.
    // No more separate scan core.
    // -------------------------------------------------------------------------
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_grid.y;

    tt::tt_metal::CoreCoord lead_coord{0, 0};
    tt::tt_metal::CoreRange full_range{
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{compute_grid.x - 1U, compute_grid.y - 1U}};
    tt::tt_metal::CoreRangeSet all_cores{full_range};

    uint32_t total_tiles = t_cap / tt::constants::TILE_HEIGHT;
    // Use the CoreCoord overload (like layernorm_bw) — routes through
    // num_cores_to_corerangeset, avoiding the sub-grid placement bug in
    // the CoreRangeSet overload that throws "Failed to assign all N
    // requested cores" when total_tiles % grid_size lands on specific
    // remainders.
    auto [num_workers, worker_all, worker_group_1, worker_group_2, tiles_group_1, tiles_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_tiles);

    uint32_t num_total_cores = num_workers;  // every core does scan + worker

    // Per-core scan slice
    uint32_t scan_slice_size = (total_rows + num_total_cores - 1U) / num_total_cores;

    // -------------------------------------------------------------------------
    // Circular buffers
    // -------------------------------------------------------------------------
    uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    tt::tt_metal::CircularBufferConfig cb_src0_cfg =
        tt::tt_metal::CircularBufferConfig(
            2U * tiles_per_chunk * bf16_tile_bytes, {{kCbSrc0, tt::DataFormat::Float16_b}})
            .set_page_size(kCbSrc0, bf16_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_src0_cfg);

    tt::tt_metal::CircularBufferConfig cb_out_cfg =
        tt::tt_metal::CircularBufferConfig(
            2U * tiles_per_chunk * bf16_tile_bytes, {{kCbOut, tt::DataFormat::Float16_b}})
            .set_page_size(kCbOut, bf16_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_out_cfg);

    constexpr uint32_t kPlanCbBytes = 32U * sizeof(uint32_t);
    tt::tt_metal::CircularBufferConfig cb_plan_cfg =
        tt::tt_metal::CircularBufferConfig(kPlanCbBytes, {{kCbPlan, tt::DataFormat::UInt32}})
            .set_page_size(kCbPlan, kPlanCbBytes);
    CreateCircularBuffer(program, all_cores, cb_plan_cfg);

    uint32_t offset_cb_bytes = (((e_local + 1U) * sizeof(uint32_t) + 31U) / 32U) * 32U;
    tt::tt_metal::CircularBufferConfig cb_offset_cfg =
        tt::tt_metal::CircularBufferConfig(offset_cb_bytes, {{kCbOffset, tt::DataFormat::UInt32}})
            .set_page_size(kCbOffset, offset_cb_bytes);
    CreateCircularBuffer(program, all_cores, cb_offset_cfg);

    // cb_scan: every core's scratch for scan + shared tables (only meaningful on lead).
    // Layout: [stage 32][leids_buf 32][counts e_local*4][offsets (e_local+1)*4]
    //         [cursors e_local*4]
    //         [shared_local_counts num_total_cores * e_local * 4]
    //         [shared_per_core_start num_total_cores * e_local * 4]
    //         [md_block (BLOCK_ROWS or slice_size)*32 + 32]
    //         [plan_stage e_local * 32 * 4][fill e_local * 4]
    constexpr uint32_t kBlockRows = 1024U;
    uint32_t slice_block_rows = scan_slice_size < kBlockRows ? scan_slice_size : kBlockRows;
    if (slice_block_rows == 0U)
        slice_block_rows = 1U;

    uint32_t overhead_bytes = 128U + 32U + (3U * e_local + 1U) * sizeof(uint32_t) + 64U;
    // Each shared table has num_total_cores slots; slot size =
    // round_up_to_align(e_local) uint32s (smallest multiple of the arch's L1
    // alignment that fits e_local). Arch-specific via HAL (16 B on WH/BH today,
    // may change on future parts).
    const uint32_t l1_align_u32 = tt::tt_metal::hal::get_l1_alignment() / sizeof(uint32_t);
    uint32_t shared_slot_u32 = ((e_local + l1_align_u32 - 1U) / l1_align_u32) * l1_align_u32;
    if (shared_slot_u32 < l1_align_u32)
        shared_slot_u32 = l1_align_u32;
    uint32_t kSharedSlotBytes = shared_slot_u32 * sizeof(uint32_t);
    uint32_t shared_table_bytes = num_total_cores * kSharedSlotBytes;
    uint32_t two_shared_tables = 2U * shared_table_bytes;
    uint32_t md_block_bytes = slice_block_rows * 32U + 32U;
    uint32_t plan_stage_bytes = ((e_local * 32U * sizeof(uint32_t) + 31U) / 32U) * 32U;
    uint32_t fill_bytes = ((e_local * sizeof(uint32_t) + 31U) / 32U) * 32U;
    uint32_t scan_scratch_bytes = overhead_bytes + two_shared_tables + md_block_bytes + plan_stage_bytes + fill_bytes;
    scan_scratch_bytes = ((scan_scratch_bytes + 31U) / 32U) * 32U;

    tt::tt_metal::CircularBufferConfig cb_scan_cfg =
        tt::tt_metal::CircularBufferConfig(scan_scratch_bytes, {{kCbScan, tt::DataFormat::UInt32}})
            .set_page_size(kCbScan, scan_scratch_bytes);
    CreateCircularBuffer(program, all_cores, cb_scan_cfg);

    // Compute address of shared tables in cb_scan (offset within scratch).
    // Layout: stage(128B), leids_buf(32B), counts, offsets, cursors.
    // MUST be 32B-aligned for cross-core NOC writes to land correctly.
    uint32_t shared_tables_offset_raw =
        128U + 32U + e_local * sizeof(uint32_t) + (e_local + 1U) * sizeof(uint32_t) + e_local * sizeof(uint32_t);
    uint32_t shared_tables_offset = (shared_tables_offset_raw + 31U) & ~31U;

    // Phase semaphores
    uint32_t scan_phase1_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0U);
    uint32_t scan_phase2_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0U);
    uint32_t scan_phase3_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0U);
    uint32_t plan_ready_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0U);

    // -------------------------------------------------------------------------
    // Buffer pointers
    // -------------------------------------------------------------------------
    auto* metadata_buf = args.metadata.buffer();
    auto* plan_buf = plan.buffer();
    auto* counts_buf = counts.buffer();
    auto* offsets_buf = offsets.buffer();
    auto* leids_buf = args.local_expert_ids.buffer();
    auto* dispatched_buf = args.dispatched.buffer();
    auto* grouped_buf = grouped.buffer();

    // -------------------------------------------------------------------------
    // NOC coords used as CT args by the combined kernel
    // -------------------------------------------------------------------------
    auto lead_virt = device->worker_core_from_logical_core(lead_coord);

    // Bounding NOC rectangle of the full worker grid — lead uses one multicast
    // per broadcast (phase2_sem + plan_ready_sem) via
    // mcast_sender_signal_receivers_loopback.
    auto mcast_tl = device->worker_core_from_logical_core({0, 0});
    auto mcast_br = device->worker_core_from_logical_core({compute_grid.x - 1U, compute_grid.y - 1U});
    uint32_t mcast_sx = std::min(mcast_tl.x, mcast_br.x);
    uint32_t mcast_ex = std::max(mcast_tl.x, mcast_br.x);
    uint32_t mcast_sy = std::min(mcast_tl.y, mcast_br.y);
    uint32_t mcast_ey = std::max(mcast_tl.y, mcast_br.y);
    uint32_t mcast_num_dests_including_self = compute_grid.x * compute_grid.y;

    // -------------------------------------------------------------------------
    // Combined NCRISC kernel CT args
    // -------------------------------------------------------------------------
    std::vector<uint32_t> combined_ct_args = {
        h,
        num_chunks,
        hidden_chunk_bytes,
        tiles_per_chunk,
        last_chunk_bytes,
        total_rows,
        k,
        e_local,
        t_cap,
        num_total_cores,
        shared_slot_u32,
        l1_align_u32,
        // Globally-constant values previously passed as RT args — moved to CT
        // so every worker's kernel binary has them baked in and the per-core
        // RT-arg list shrinks.
        lead_virt.x,                      // 12
        lead_virt.y,                      // 13
        scan_phase1_sem_id,               // 14
        scan_phase2_sem_id,               // 15
        scan_phase3_sem_id,               // 16
        plan_ready_sem_id,                // 17
        shared_tables_offset,             // 18
        mcast_sx,                         // 19
        mcast_sy,                         // 20
        mcast_ex,                         // 21
        mcast_ey,                         // 22
        mcast_num_dests_including_self};  // 23
    tt::tt_metal::TensorAccessorArgs(plan_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(dispatched_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(metadata_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(counts_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(offsets_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(leids_buf).append_to(combined_ct_args);

    auto combined_kernel_g1 = create_reader_kernel(program, worker_group_1, combined_ct_args, {}, kCombinedKernelPath);
    tt::tt_metal::KernelHandle combined_kernel_g2{};
    if (!worker_group_2.ranges().empty()) {
        combined_kernel_g2 = create_reader_kernel(program, worker_group_2, combined_ct_args, {}, kCombinedKernelPath);
    }

    // -------------------------------------------------------------------------
    // Worker writer kernel (BRISC) — unchanged
    // -------------------------------------------------------------------------
    std::vector<uint32_t> writer_ct_args = {
        num_chunks,
        tiles_per_chunk,
        Wt,
        e_local,
        last_chunk_tiles,
        num_total_cores,     // 5 — stride (= num_workers)
        plan_ready_sem_id};  // 6
    tt::tt_metal::TensorAccessorArgs(grouped_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(offsets_buf).append_to(writer_ct_args);

    auto writer_kernel_g1 = create_writer_kernel(program, worker_group_1, writer_ct_args, {}, kWorkerWriterPath);
    tt::tt_metal::KernelHandle writer_kernel_g2{};
    if (!worker_group_2.ranges().empty()) {
        writer_kernel_g2 = create_writer_kernel(program, worker_group_2, writer_ct_args, {}, kWorkerWriterPath);
    }

    // -------------------------------------------------------------------------
    // Compute (tilize) kernel — unchanged
    // -------------------------------------------------------------------------
    [[maybe_unused]] auto compute_g1 = create_compute_kernel(
        program,
        worker_group_1,
        {kCbSrc0, kCbOut, tiles_group_1, tiles_per_chunk, num_chunks},
        {},
        kTilizeKernelPath,
        false);
    if (!worker_group_2.ranges().empty()) {
        [[maybe_unused]] auto compute_g2 = create_compute_kernel(
            program,
            worker_group_2,
            {kCbSrc0, kCbOut, tiles_group_2, tiles_per_chunk, num_chunks},
            {},
            kTilizeKernelPath,
            false);
    }

    // -------------------------------------------------------------------------
    // Per-core RT args. Need to know each core's scan slice + worker assignment +
    // (for lead core) every other core's NOC XY for broadcasting phase signals.
    // -------------------------------------------------------------------------
    auto worker_cores_vec = tt::tt_metal::corerange_to_cores(worker_all, num_workers, /*row_wise=*/true);

    // Compute shared tables L1 addr (we need actual L1 address on lead's cb_scan).
    // We'll let the kernel compute it from get_write_ptr(cb_scan) + shared_tables_offset.
    // So we just pass the offset; the kernel does the addition.
    // BUT the kernel needs the L1 ABSOLUTE address on the LEAD core for cross-core NOC.
    // get_write_ptr is the same on all cores for the same CB allocation, so the offset
    // is the same. Pass offset; kernel computes lead-side addr same way.
    // Actually for cross-core NOC: it's lead_x/y + L1_addr. The L1_addr on lead's cb_scan
    // is what the kernel sees as get_write_ptr(cb_scan) + shared_tables_offset.
    // Since CB allocation is the same on all cores in the same CoreRangeSet, that
    // local address is identical → kernel can compute it itself. We pass the offset.

    uint32_t worker_idx = 0;
    for (const auto& core : worker_cores_vec) {
        uint32_t tiles_this_core = worker_group_1.contains(core) ? tiles_group_1 : tiles_group_2;

        // Scan slice for this core (worker_idx == my_core_idx for scan purposes).
        uint32_t my_slice_start = worker_idx * scan_slice_size;
        uint32_t my_slice_end = my_slice_start + scan_slice_size;
        if (my_slice_end > total_rows)
            my_slice_end = total_rows;
        if (my_slice_start > total_rows)
            my_slice_start = total_rows;

        // Next core NOC XY for tail-flush chaining (last core gets 0,0).
        uint32_t next_x = 0, next_y = 0;
        if (worker_idx + 1U < num_total_cores) {
            auto next_virt = device->worker_core_from_logical_core(worker_cores_vec[worker_idx + 1U]);
            next_x = next_virt.x;
            next_y = next_virt.y;
        }

        // RT args are now only the things that vary per-core (or per-run):
        // tensor buffer addresses and per-core identity/slice bounds.
        // Everything previously here that was globally-constant (sem ids,
        // lead NOC, mcast rect, shared_tables_offset, worker_stride) moved
        // to CT args in combined_ct_args above.
        std::vector<uint32_t> reader_rt = {
            plan_buf->address(),        // 0
            dispatched_buf->address(),  // 1
            worker_idx,                 // 2 my_worker_start (interleaved tile_row)
            tiles_this_core,            // 3 my_worker_count
            metadata_buf->address(),    // 4
            counts_buf->address(),      // 5
            offsets_buf->address(),     // 6
            leids_buf->address(),       // 7
            worker_idx,                 // 8 my_core_idx
            my_slice_start,             // 9
            my_slice_end,               // 10
            next_x,                     // 11 chain: next core NOC for tail flush
            next_y,                     // 12
        };

        // Writer RT args (stride / plan_ready_sem_id are globally-constant →
        // CT args now; only per-core + buffer addrs stay RT).
        std::vector<uint32_t> writer_rt = {
            grouped_buf->address(),  // 0
            worker_idx,              // 1 my_start
            tiles_this_core,         // 2 my_count
            offsets_buf->address(),  // 3
        };

        bool is_g1 = worker_group_1.contains(core);
        SetRuntimeArgs(program, is_g1 ? combined_kernel_g1 : combined_kernel_g2, core, reader_rt);
        SetRuntimeArgs(program, is_g1 ? writer_kernel_g1 : writer_kernel_g2, core, writer_rt);
        worker_idx++;
    }

    return cached_program_t{
        std::move(program),
        {/* num_cores       = */ num_workers,
         /* scan_coord      = */ lead_coord,
         /* scan_kernel     = */ combined_kernel_g1,  // reuse field for combined kernel g1
         /* reader_kernel_g1 = */ combined_kernel_g1,
         /* reader_kernel_g2 = */ combined_kernel_g2,
         /* writer_kernel_g1 = */ writer_kernel_g1,
         /* writer_kernel_g2 = */ writer_kernel_g2,
         /* worker_all      = */ worker_all,
         /* worker_group_1  = */ worker_group_1,
         /* worker_group_2  = */ worker_group_2,
         /* num_cores_y     = */ num_cores_y}};
}

void MoeGroupProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;

    auto& grouped = std::get<0>(outputs);
    auto& counts = std::get<1>(outputs);
    auto& offsets = std::get<2>(outputs);
    auto& plan = std::get<3>(outputs);

    auto* metadata_buf = tensor_args.metadata.buffer();
    auto* plan_buf = plan.buffer();
    auto* counts_buf = counts.buffer();
    auto* offsets_buf = offsets.buffer();
    auto* leids_buf = tensor_args.local_expert_ids.buffer();
    auto* dispatched_buf = tensor_args.dispatched.buffer();
    auto* grouped_buf = grouped.buffer();

    auto worker_cores_vec = tt::tt_metal::corerange_to_cores(sv.worker_all, sv.num_cores, /*row_wise=*/true);
    for (const auto& core : worker_cores_vec) {
        bool is_g1 = sv.worker_group_1.contains(core);

        auto& reader_rt = GetRuntimeArgs(program, is_g1 ? sv.reader_kernel_g1 : sv.reader_kernel_g2)[core.x][core.y];
        reader_rt[0] = plan_buf->address();
        reader_rt[1] = dispatched_buf->address();
        reader_rt[4] = metadata_buf->address();
        reader_rt[5] = counts_buf->address();
        reader_rt[6] = offsets_buf->address();
        reader_rt[7] = leids_buf->address();

        auto& writer_rt = GetRuntimeArgs(program, is_g1 ? sv.writer_kernel_g1 : sv.writer_kernel_g2)[core.x][core.y];
        writer_rt[0] = grouped_buf->address();
        writer_rt[3] = offsets_buf->address();
    }
}

}  // namespace ttml::metal::ops::moe_group::device
