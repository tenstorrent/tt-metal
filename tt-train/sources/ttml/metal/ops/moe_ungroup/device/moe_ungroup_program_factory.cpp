// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ungroup_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/dataflow/moe_ungroup_reader.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/dataflow/moe_ungroup_rmw_writer.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/compute/moe_ungroup_untilize_kernel.cpp";

// CB indices (must match the kernels' constexpr declarations)
constexpr uint32_t kCbSrc0 = tt::CBIndex::c_0;
constexpr uint32_t kCbOut = tt::CBIndex::c_2;
constexpr uint32_t kCbReaderScratch = tt::CBIndex::c_3;  // reader offsets + per-expert caches
constexpr uint32_t kCbScratch = tt::CBIndex::c_4;        // writer's scratch (zero buf, offsets, plan, w)
constexpr uint32_t kCbW = tt::CBIndex::c_5;              // COL-broadcast weight tile (only col 0 populated, w[r])
constexpr uint32_t kCbExistingRm = tt::CBIndex::c_6;    // row-major existing rows from ungrouped (writer fills)
constexpr uint32_t kCbExistingTile = tt::CBIndex::c_7;  // tilized existing (compute internal)
constexpr uint32_t kCbCombined = tt::CBIndex::c_8;      // mul+add output tiles (untilize input)
constexpr uint32_t kCbCtrl = tt::CBIndex::c_10;         // NCRISC->compute: per-core active-block count

constexpr uint32_t kTargetChunkBytes = 128U * 1024U;

// Pick the fewest fixed-size chunks whose per-chunk L1 strip stays under the
// target when possible. The kernels always exchange tiles_per_chunk CB pages;
// the reader zero-fills padded tile pages in the final chunk when Wt is not an
// even multiple of tiles_per_chunk.
uint32_t pick_num_chunks(uint32_t h) {
    // ceil(h / TILE_WIDTH) — last tile may be partial when h isn't tile-aligned.
    const uint32_t Wt = tt::round_up(h, tt::constants::TILE_WIDTH) / tt::constants::TILE_WIDTH;
    if (Wt == 0U) {
        return 1U;
    }
    const uint32_t tile_row_bytes =
        tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH * 2U;  // = 2 KiB per tile-column
    uint32_t tpc_cap = kTargetChunkBytes / tile_row_bytes;
    tpc_cap = std::max(tpc_cap, 1U);
    return (Wt + tpc_cap - 1U) / tpc_cap;
}

}  // namespace

namespace ttml::metal::ops::moe_ungroup::device {

MoeUngroupProgramFactory::cached_program_t MoeUngroupProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& output) {
    auto* device = args.expert_out.device();
    tt::tt_metal::Program program{};

    const uint32_t h = attrs.h;
    const uint32_t e_local = attrs.e_local;
    const uint32_t total_rows = attrs.d * attrs.b * attrs.s;

    const uint32_t num_chunks = pick_num_chunks(h);
    // ceil(h / TILE_WIDTH); the last tile is padded with zeros if h % 32 != 0.
    const uint32_t Wt = tt::round_up(h, tt::constants::TILE_WIDTH) / tt::constants::TILE_WIDTH;
    const uint32_t tiles_per_chunk = tt::round_up(Wt, num_chunks) / num_chunks;
    const uint32_t hidden_chunk_bytes = tiles_per_chunk * tt::constants::TILE_WIDTH * 2U;
    // Valid bytes in the last chunk reflect the actual H, not the tile-rounded
    // width: ungrouped is row-major bf16 with stride h*2, so reading/writing
    // more than (h - prior_chunks_width) * 2 bytes would cross into the next
    // row in DRAM. Reader/writer pad the unused L1 tail with zeros.
    const uint32_t prior_chunks_bytes = (num_chunks - 1U) * hidden_chunk_bytes;
    const uint32_t last_chunk_bytes = h * 2U - prior_chunks_bytes;

    const auto compute_grid = device->compute_with_storage_grid_size();
    const tt::tt_metal::CoreCoord lead_coord{0, 0};

    // Run on the full compute grid (mirrors moe_group). Inactive cores no-op
    // through the per-expert loops in lockstep, participating in barriers and
    // CB plumbing so the pipeline stays balanced.
    tt::tt_metal::CoreRangeSet worker_all{tt::tt_metal::CoreRange{
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{compute_grid.x - 1U, compute_grid.y - 1U}}};
    tt::tt_metal::CoreRangeSet worker_group_1 = worker_all;
    tt::tt_metal::CoreRangeSet worker_group_2{};
    const uint32_t num_workers = compute_grid.x * compute_grid.y;
    const uint32_t num_total_cores = num_workers;

    // -------------------------------------------------------------------------
    // Circular buffers
    // -------------------------------------------------------------------------
    const uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    // cb_src0: TILE bf16, double-buffered
    create_circular_buffer(
        program, worker_all, kCbSrc0, tt::DataFormat::Float16_b, bf16_tile_bytes, 2U * tiles_per_chunk);

    // cb_out: row-major bf16 produced by untilize, double-buffered
    create_circular_buffer(
        program, worker_all, kCbOut, tt::DataFormat::Float16_b, bf16_tile_bytes, 2U * tiles_per_chunk);

    // cb_reader_scratch: reader scratch holding offsets DMA + per-expert caches:
    //   offsets_l1 (e_local+1) u32  +  tr_start_per_expert e_local u32
    //                              +  my_real_count_per_expert e_local u32
    // Backing the caches in L1 keeps NCRISC stack usage bounded for large
    // e_local (e.g. 300+) where stack arrays would otherwise overflow.
    const uint32_t kL1_ALIGN = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t cb_reader_scratch_bytes = tt::round_up((3U * e_local + 1U) * sizeof(uint32_t), kL1_ALIGN);
    create_circular_buffer_bytes(
        program, worker_all, kCbReaderScratch, tt::DataFormat::UInt32, cb_reader_scratch_bytes);

    // cb_w: COL-broadcast weight tile (TILE bf16). BRISC writer builds this each
    // chunk, populating only column 0 with w_tile[r,0] = bf16(w[r]); compute
    // COL-broadcasts it against cb_src0 before untilizing. Capacity: 2
    // (double-buffer for pipelining).
    create_circular_buffer(program, worker_all, kCbW, tt::DataFormat::Float16_b, bf16_tile_bytes, 2U);

    // cb_existing_rm: row-major existing rows from ungrouped DRAM (writer fills,
    // compute tilizes). Asymmetric pages: one page per ROW (hidden_chunk_bytes).
    // Double-buffered by chunk so BRISC can DMA chunk N+1 while compute
    // tilizes/mul/add/untilizes chunk N.
    constexpr uint32_t cb_existing_rm_pages = 2U * tt::constants::TILE_HEIGHT;  // two chunks, one page per row
    create_circular_buffer_bytes(
        program,
        worker_all,
        kCbExistingRm,
        tt::DataFormat::Float16_b,
        cb_existing_rm_pages * hidden_chunk_bytes,
        hidden_chunk_bytes);

    // Compute-private intermediates stay single-buffered: they are produced
    // and consumed by the same compute kernel, so extra depth would spend L1
    // without adding reader/writer overlap.
    // cb_existing_tile: compute's tilize output, fed into mul+add. One TILE_HEIGHT-row
    // block produces tiles_per_chunk tiles.
    create_circular_buffer(
        program, worker_all, kCbExistingTile, tt::DataFormat::Float16_b, bf16_tile_bytes, tiles_per_chunk);

    // cb_combined: compute's add output, untilize input.
    create_circular_buffer(
        program, worker_all, kCbCombined, tt::DataFormat::Float16_b, bf16_tile_bytes, tiles_per_chunk);

    // cb_ctrl: NCRISC reader publishes per-core active-block count once at
    // startup; compute reads it to size its outer loop. 16B page (one uint32
    // padded to L1 alignment), single-page CB.
    constexpr uint32_t cb_ctrl_bytes = 16U;
    create_circular_buffer_bytes(program, worker_all, kCbCtrl, tt::DataFormat::UInt32, cb_ctrl_bytes);

    // cb_scratch: writer's scratch — zero buf + offsets + plan + grouped_scores
    // slice + w + rmw_buf. With moe_group emitting grouped_scores per row, we
    // no longer keep metadata/scores/leids buffers here; the writer reads a
    // 32-entry slice of grouped_scores per tile-row directly into w_buf.
    const uint32_t dram_align_bytes = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t offsets_page_bytes = tt::round_up((e_local + 1U) * sizeof(uint32_t), dram_align_bytes);
    uint32_t scratch_bytes = 0U;
    scratch_bytes += tt::round_up(h * 2U, offsets_page_bytes);         // zero_buf, aligns offsets_buf start
    scratch_bytes += offsets_page_bytes;                               // offsets_buf, matches TensorAccessor page
    scratch_bytes += tt::round_up(32U * sizeof(uint32_t), kL1_ALIGN);  // plan_buf
    scratch_bytes += tt::round_up(32U * sizeof(uint16_t), kL1_ALIGN);  // w_buf (bf16,
                                                                       //   directly read from grouped_scores)
    scratch_bytes = tt::round_up(scratch_bytes, kL1_ALIGN);
    create_circular_buffer_bytes(program, worker_all, kCbScratch, tt::DataFormat::UInt32, scratch_bytes);

    // -------------------------------------------------------------------------
    // Semaphores
    //   up_sem / down_sem    — NCRISC-side cross-core mcast barrier (NOC_0).
    //   brisc_done_sem       — local BRISC->NCRISC handshake (BRISC sets, NCRISC polls).
    //   brisc_release_sem    — local NCRISC->BRISC handshake (NCRISC sets, BRISC polls).
    // Both pairs are reused for the prezero barrier and every inter-expert
    // barrier (each call resets the sems to 0 before the next handshake).
    // -------------------------------------------------------------------------
    const uint32_t up_sem_id = tt::tt_metal::CreateSemaphore(program, worker_all, 0U);
    const uint32_t down_sem_id = tt::tt_metal::CreateSemaphore(program, worker_all, 0U);
    const uint32_t brisc_done_sem_id = tt::tt_metal::CreateSemaphore(program, worker_all, 0U);
    const uint32_t brisc_release_sem_id = tt::tt_metal::CreateSemaphore(program, worker_all, 0U);

    // -------------------------------------------------------------------------
    // Buffer pointers
    // -------------------------------------------------------------------------
    auto* expert_out_buf = args.expert_out.buffer();
    auto* ungrouped_buf = output.buffer();
    auto* plan_buf = args.plan.buffer();
    auto* offsets_buf = args.offsets.buffer();
    auto* grouped_scores_buf = args.grouped_scores.buffer();

    // -------------------------------------------------------------------------
    // Mcast rectangle (covers full worker grid).
    // -------------------------------------------------------------------------
    const auto lead_virt = device->worker_core_from_logical_core(lead_coord);
    const auto mcast_tl = device->worker_core_from_logical_core({0, 0});
    const auto mcast_br = device->worker_core_from_logical_core({compute_grid.x - 1U, compute_grid.y - 1U});
    const uint32_t mcast_sx = std::min(mcast_tl.x, mcast_br.x);
    const uint32_t mcast_ex = std::max(mcast_tl.x, mcast_br.x);
    const uint32_t mcast_sy = std::min(mcast_tl.y, mcast_br.y);
    const uint32_t mcast_ey = std::max(mcast_tl.y, mcast_br.y);
    const uint32_t mcast_num_dests_incl_self = compute_grid.x * compute_grid.y;

    // -------------------------------------------------------------------------
    // Reader CT args
    // -------------------------------------------------------------------------
    std::vector<uint32_t> reader_ct_args = {
        h,                          // 0
        num_chunks,                 // 1
        tiles_per_chunk,            // 2
        e_local,                    // 3
        num_total_cores,            // 4
        lead_virt.x,                // 5
        lead_virt.y,                // 6
        up_sem_id,                  // 7
        down_sem_id,                // 8
        brisc_done_sem_id,          // 9
        brisc_release_sem_id,       // 10
        mcast_sx,                   // 11
        mcast_sy,                   // 12
        mcast_ex,                   // 13
        mcast_ey,                   // 14
        mcast_num_dests_incl_self,  // 15
        kCbCtrl,                    // 16
        Wt,                         // 17
    };
    tt::tt_metal::TensorAccessorArgs(expert_out_buf).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(offsets_buf).append_to(reader_ct_args);

    auto reader_kernel_g1 = create_reader_kernel(program, worker_group_1, reader_ct_args, {}, kReaderKernelPath);
    tt::tt_metal::KernelHandle reader_kernel_g2{};
    if (!worker_group_2.ranges().empty()) {
        reader_kernel_g2 = create_reader_kernel(program, worker_group_2, reader_ct_args, {}, kReaderKernelPath);
    }

    // -------------------------------------------------------------------------
    // Writer CT args
    // -------------------------------------------------------------------------
    std::vector<uint32_t> writer_ct_args = {
        h,                     // 0
        num_chunks,            // 1
        hidden_chunk_bytes,    // 2
        tiles_per_chunk,       // 3
        last_chunk_bytes,      // 4
        total_rows,            // 5
        e_local,               // 6
        num_total_cores,       // 7
        brisc_done_sem_id,     // 8
        brisc_release_sem_id,  // 9
        kL1_ALIGN,             // 10
    };
    tt::tt_metal::TensorAccessorArgs(ungrouped_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(plan_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(offsets_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(grouped_scores_buf).append_to(writer_ct_args);

    auto writer_kernel_g1 = create_writer_kernel(program, worker_group_1, writer_ct_args, {}, kWriterKernelPath);
    tt::tt_metal::KernelHandle writer_kernel_g2{};
    if (!worker_group_2.ranges().empty()) {
        writer_kernel_g2 = create_writer_kernel(program, worker_group_2, writer_ct_args, {}, kWriterKernelPath);
    }

    // -------------------------------------------------------------------------
    // Compute kernel — tilizes existing rows, does broadcast-scaled accumulation
    // in DST (mul + add), then untilizes the combined rows for BRISC writeback.
    // -------------------------------------------------------------------------
    [[maybe_unused]] auto compute_g1 = create_compute_kernel(
        program,
        worker_group_1,
        {kCbSrc0, kCbOut, tiles_per_chunk, kCbW, kCbExistingRm, kCbExistingTile, kCbCombined, kCbCtrl},
        {},
        kComputeKernelPath,
        false);
    if (!worker_group_2.ranges().empty()) {
        [[maybe_unused]] auto compute_g2 = create_compute_kernel(
            program,
            worker_group_2,
            {kCbSrc0, kCbOut, tiles_per_chunk, kCbW, kCbExistingRm, kCbExistingTile, kCbCombined, kCbCtrl},
            {},
            kComputeKernelPath,
            false);
    }

    // -------------------------------------------------------------------------
    // Per-core RT args
    // -------------------------------------------------------------------------
    auto worker_cores_vec = tt::tt_metal::corerange_to_cores(worker_all, num_workers, /*row_wise=*/true);
    uint32_t worker_idx = 0;
    for (const auto& core : worker_cores_vec) {
        bool is_g1 = worker_group_1.contains(core);

        std::vector<uint32_t> reader_rt = {
            expert_out_buf->address(),  // 0
            offsets_buf->address(),     // 1
            worker_idx,                 // 2 my_core_idx
        };
        std::vector<uint32_t> writer_rt = {
            ungrouped_buf->address(),       // 0
            plan_buf->address(),            // 1
            offsets_buf->address(),         // 2
            grouped_scores_buf->address(),  // 3
            worker_idx,                     // 4 my_core_idx
        };

        SetRuntimeArgs(program, is_g1 ? reader_kernel_g1 : reader_kernel_g2, core, reader_rt);
        SetRuntimeArgs(program, is_g1 ? writer_kernel_g1 : writer_kernel_g2, core, writer_rt);
        worker_idx++;
    }

    return cached_program_t{
        std::move(program),
        {/* num_cores       = */ num_workers,
         /* reader_kernel_g1 = */ reader_kernel_g1,
         /* reader_kernel_g2 = */ reader_kernel_g2,
         /* writer_kernel_g1 = */ writer_kernel_g1,
         /* writer_kernel_g2 = */ writer_kernel_g2,
         /* worker_all      = */ worker_all,
         /* worker_group_1  = */ worker_group_1,
         /* worker_group_2  = */ worker_group_2}};
}

void MoeUngroupProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;

    auto* expert_out_buf = tensor_args.expert_out.buffer();
    auto* ungrouped_buf = output.buffer();
    auto* plan_buf = tensor_args.plan.buffer();
    auto* offsets_buf = tensor_args.offsets.buffer();
    auto* grouped_scores_buf = tensor_args.grouped_scores.buffer();

    auto worker_cores_vec = tt::tt_metal::corerange_to_cores(sv.worker_all, sv.num_cores, /*row_wise=*/true);
    for (const auto& core : worker_cores_vec) {
        bool is_g1 = sv.worker_group_1.contains(core);

        auto& reader_rt = GetRuntimeArgs(program, is_g1 ? sv.reader_kernel_g1 : sv.reader_kernel_g2)[core.x][core.y];
        reader_rt[0] = expert_out_buf->address();
        reader_rt[1] = offsets_buf->address();

        auto& writer_rt = GetRuntimeArgs(program, is_g1 ? sv.writer_kernel_g1 : sv.writer_kernel_g2)[core.x][core.y];
        writer_rt[0] = ungrouped_buf->address();
        writer_rt[1] = plan_buf->address();
        writer_rt[2] = offsets_buf->address();
        writer_rt[3] = grouped_scores_buf->address();
    }
}

}  // namespace ttml::metal::ops::moe_ungroup::device
