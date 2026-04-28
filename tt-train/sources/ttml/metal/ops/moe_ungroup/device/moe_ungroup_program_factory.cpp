// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ungroup_program_factory.hpp"

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
    "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/dataflow/moe_ungroup_writer.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/compute/moe_ungroup_untilize_kernel.cpp";

// CB indices (must match the kernels' constexpr declarations)
constexpr uint32_t kCbSrc0 = tt::CBIndex::c_0;
constexpr uint32_t kCbOut = tt::CBIndex::c_2;
constexpr uint32_t kCbZero = tt::CBIndex::c_3;          // reader's zero/offsets scratch
constexpr uint32_t kCbScratch = tt::CBIndex::c_4;       // writer's scratch (zero buf, plan, md, sc, w, rmw)
constexpr uint32_t kCbW = tt::CBIndex::c_5;             // weight tile (32×32 broadcast w[r])
constexpr uint32_t kCbExistingRm = tt::CBIndex::c_6;    // row-major existing rows from ungrouped (writer fills)
constexpr uint32_t kCbExistingTile = tt::CBIndex::c_7;  // tilized existing (compute internal)
constexpr uint32_t kCbCombined = tt::CBIndex::c_8;      // mul+add output tiles (untilize input)
constexpr uint32_t kCbScaled = tt::CBIndex::c_9;        // scaled-only tiles (mul output, add input)
constexpr uint32_t kCbCtrl = tt::CBIndex::c_10;         // NCRISC->compute: per-core active-block count

constexpr uint32_t kTargetChunkBytes = 128U * 1024U;

uint32_t pick_num_chunks(uint32_t h) {
    uint32_t row_bytes = h * 2U;
    uint32_t strip_bytes = 32U * row_bytes;
    uint32_t nc = (strip_bytes + kTargetChunkBytes - 1U) / kTargetChunkBytes;
    if (nc == 0U) {
        nc = 1U;
    }
    uint32_t Wt = h / tt::constants::TILE_WIDTH;
    if (nc > Wt) {
        nc = Wt;
    }
    return nc;
}

}  // namespace

namespace ttml::metal::ops::moe_ungroup::device {

MoeUngroupProgramFactory::cached_program_t MoeUngroupProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& output) {
    auto* device = args.expert_out.device();
    tt::tt_metal::Program program{};

    const uint32_t h = attrs.h;
    const uint32_t e_local = attrs.e_local;
    const uint32_t k = attrs.k;
    const uint32_t total_rows = attrs.d * attrs.b * attrs.s;

    const uint32_t num_chunks = pick_num_chunks(h);
    const uint32_t Wt = h / tt::constants::TILE_WIDTH;
    const uint32_t tiles_per_chunk = (Wt + num_chunks - 1U) / num_chunks;
    const uint32_t hidden_chunk_bytes = tiles_per_chunk * tt::constants::TILE_WIDTH * 2U;
    const uint32_t last_chunk_tiles = Wt - (num_chunks - 1U) * tiles_per_chunk;
    const uint32_t last_chunk_bytes = last_chunk_tiles * tt::constants::TILE_WIDTH * 2U;

    auto compute_grid = device->compute_with_storage_grid_size();
    tt::tt_metal::CoreCoord lead_coord{0, 0};

    // Run on the full compute grid (mirrors moe_group). Inactive cores no-op
    // through the per-expert loops in lockstep, participating in barriers and
    // CB plumbing so the pipeline stays balanced.
    tt::tt_metal::CoreRangeSet worker_all{tt::tt_metal::CoreRange{
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{compute_grid.x - 1U, compute_grid.y - 1U}}};
    tt::tt_metal::CoreRangeSet worker_group_1 = worker_all;
    tt::tt_metal::CoreRangeSet worker_group_2{};
    uint32_t num_workers = compute_grid.x * compute_grid.y;
    uint32_t num_total_cores = num_workers;

    // -------------------------------------------------------------------------
    // Circular buffers
    // -------------------------------------------------------------------------
    uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    // cb_src0: TILE bf16, double-buffered
    tt::tt_metal::CircularBufferConfig cb_src0_cfg =
        tt::tt_metal::CircularBufferConfig(
            2U * tiles_per_chunk * bf16_tile_bytes, {{kCbSrc0, tt::DataFormat::Float16_b}})
            .set_page_size(kCbSrc0, bf16_tile_bytes);
    CreateCircularBuffer(program, worker_all, cb_src0_cfg);

    // cb_out: row-major bf16 produced by untilize, double-buffered
    tt::tt_metal::CircularBufferConfig cb_out_cfg =
        tt::tt_metal::CircularBufferConfig(
            2U * tiles_per_chunk * bf16_tile_bytes, {{kCbOut, tt::DataFormat::Float16_b}})
            .set_page_size(kCbOut, bf16_tile_bytes);
    CreateCircularBuffer(program, worker_all, cb_out_cfg);

    // cb_zero: small reader scratch for offsets (32B aligned, holds (E_local+1)*4 + slack)
    uint32_t cb_zero_bytes = ((h * 2U + 31U) & ~31U);  // up to h bf16 zeros (also used to hold offsets)
    tt::tt_metal::CircularBufferConfig cb_zero_cfg =
        tt::tt_metal::CircularBufferConfig(cb_zero_bytes, {{kCbZero, tt::DataFormat::UInt32}})
            .set_page_size(kCbZero, cb_zero_bytes);
    CreateCircularBuffer(program, worker_all, cb_zero_cfg);

    // cb_w: 32×32 broadcast weight tile (TILE bf16). BRISC writer builds this
    // each chunk where w_tile[r,c] = bf16(w[r]); compute multiplies cb_src0
    // against it before untilizing. Capacity: 2 (double-buffer for pipelining).
    tt::tt_metal::CircularBufferConfig cb_w_cfg =
        tt::tt_metal::CircularBufferConfig(2U * bf16_tile_bytes, {{kCbW, tt::DataFormat::Float16_b}})
            .set_page_size(kCbW, bf16_tile_bytes);
    CreateCircularBuffer(program, worker_all, cb_w_cfg);

    // cb_existing_rm: row-major existing rows from ungrouped DRAM (writer fills,
    // compute tilizes). Asymmetric pages: one page per ROW (hidden_chunk_bytes).
    // 32 pages per chunk (one per tile-row's row), so tilize sees the data as
    // row-major with each row = block_width_tiles*32 cols contiguous.
    uint32_t cb_existing_rm_pages = 32U;  // 32 rows per chunk
    tt::tt_metal::CircularBufferConfig cb_existing_rm_cfg =
        tt::tt_metal::CircularBufferConfig(
            cb_existing_rm_pages * hidden_chunk_bytes, {{kCbExistingRm, tt::DataFormat::Float16_b}})
            .set_page_size(kCbExistingRm, hidden_chunk_bytes);
    CreateCircularBuffer(program, worker_all, cb_existing_rm_cfg);

    // cb_existing_tile: compute's tilize output, fed into mul+add.
    tt::tt_metal::CircularBufferConfig cb_existing_tile_cfg =
        tt::tt_metal::CircularBufferConfig(
            tiles_per_chunk * bf16_tile_bytes, {{kCbExistingTile, tt::DataFormat::Float16_b}})
            .set_page_size(kCbExistingTile, bf16_tile_bytes);
    CreateCircularBuffer(program, worker_all, cb_existing_tile_cfg);

    // cb_combined: compute's add output, untilize input.
    tt::tt_metal::CircularBufferConfig cb_combined_cfg =
        tt::tt_metal::CircularBufferConfig(
            tiles_per_chunk * bf16_tile_bytes, {{kCbCombined, tt::DataFormat::Float16_b}})
            .set_page_size(kCbCombined, bf16_tile_bytes);
    CreateCircularBuffer(program, worker_all, cb_combined_cfg);

    // cb_scaled: compute's mul output, add input.
    tt::tt_metal::CircularBufferConfig cb_scaled_cfg =
        tt::tt_metal::CircularBufferConfig(tiles_per_chunk * bf16_tile_bytes, {{kCbScaled, tt::DataFormat::Float16_b}})
            .set_page_size(kCbScaled, bf16_tile_bytes);
    CreateCircularBuffer(program, worker_all, cb_scaled_cfg);

    // cb_ctrl: NCRISC reader publishes per-core active-block count once at
    // startup; compute reads it to size its outer loop. 16B page (one uint32
    // padded to L1 alignment), single-page CB.
    constexpr uint32_t cb_ctrl_bytes = 16U;
    tt::tt_metal::CircularBufferConfig cb_ctrl_cfg =
        tt::tt_metal::CircularBufferConfig(cb_ctrl_bytes, {{kCbCtrl, tt::DataFormat::UInt32}})
            .set_page_size(kCbCtrl, cb_ctrl_bytes);
    CreateCircularBuffer(program, worker_all, cb_ctrl_cfg);

    // cb_scratch: writer's big scratch — zero buf + offsets + counts + leids + plan + md + sc + w + rmw_buf
    const uint32_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    auto round_up = [l1_align](uint32_t x) { return ((x + l1_align - 1U) / l1_align) * l1_align; };
    uint32_t scratch_bytes = 0U;
    scratch_bytes += round_up(h * 2U);                             // zero_buf
    scratch_bytes += round_up((e_local + 1U) * sizeof(uint32_t));  // offsets_buf
    scratch_bytes += round_up(e_local * sizeof(uint32_t));         // counts_buf
    scratch_bytes += round_up(e_local * sizeof(uint16_t));         // leids_buf
    scratch_bytes += round_up(32U * sizeof(uint32_t));             // plan_buf
    // md_buf and sc_buf: 32 rows * aligned page each.
    uint32_t md_aligned = round_up(k * sizeof(uint16_t));
    scratch_bytes += round_up(32U * md_aligned);     // md_buf
    scratch_bytes += round_up(32U * md_aligned);     // sc_buf (same row stride as md)
    scratch_bytes += round_up(32U * sizeof(float));  // w_buf
    // stage_buf: 32 contiguous slots of hidden_chunk_bytes — required by the
    // OPT 1 barrier-coalesced writer. Worst-case size: 32 * 128KB / 32 = 128KB
    // (chunk size is capped at kTargetChunkBytes).
    scratch_bytes += round_up(32U * hidden_chunk_bytes);  // stage_buf
    scratch_bytes = round_up(scratch_bytes);

    tt::tt_metal::CircularBufferConfig cb_scratch_cfg =
        tt::tt_metal::CircularBufferConfig(scratch_bytes, {{kCbScratch, tt::DataFormat::UInt32}})
            .set_page_size(kCbScratch, scratch_bytes);
    CreateCircularBuffer(program, worker_all, cb_scratch_cfg);

    // -------------------------------------------------------------------------
    // Semaphores
    //   up_sem / down_sem    — NCRISC-side cross-core mcast barrier (NOC_0).
    //   brisc_done_sem       — local BRISC->NCRISC handshake (BRISC sets, NCRISC polls).
    //   brisc_release_sem    — local NCRISC->BRISC handshake (NCRISC sets, BRISC polls).
    // Both pairs are reused for the prezero barrier and every inter-expert
    // barrier (each call resets the sems to 0 before the next handshake).
    // -------------------------------------------------------------------------
    uint32_t up_sem_id = tt::tt_metal::CreateSemaphore(program, worker_all, 0U);
    uint32_t down_sem_id = tt::tt_metal::CreateSemaphore(program, worker_all, 0U);
    uint32_t brisc_done_sem_id = tt::tt_metal::CreateSemaphore(program, worker_all, 0U);
    uint32_t brisc_release_sem_id = tt::tt_metal::CreateSemaphore(program, worker_all, 0U);

    // -------------------------------------------------------------------------
    // Buffer pointers
    // -------------------------------------------------------------------------
    auto* expert_out_buf = args.expert_out.buffer();
    auto* ungrouped_buf = output.buffer();
    auto* plan_buf = args.plan.buffer();
    auto* offsets_buf = args.offsets.buffer();
    auto* counts_buf = args.counts.buffer();
    auto* metadata_buf = args.metadata.buffer();
    auto* scores_buf = args.scores.buffer();
    auto* leids_buf = args.local_expert_ids.buffer();

    // -------------------------------------------------------------------------
    // Mcast rectangle (covers full worker grid).
    // -------------------------------------------------------------------------
    auto lead_virt = device->worker_core_from_logical_core(lead_coord);
    auto mcast_tl = device->worker_core_from_logical_core({0, 0});
    auto mcast_br = device->worker_core_from_logical_core({compute_grid.x - 1U, compute_grid.y - 1U});
    uint32_t mcast_sx = std::min(mcast_tl.x, mcast_br.x);
    uint32_t mcast_ex = std::max(mcast_tl.x, mcast_br.x);
    uint32_t mcast_sy = std::min(mcast_tl.y, mcast_br.y);
    uint32_t mcast_ey = std::max(mcast_tl.y, mcast_br.y);
    uint32_t mcast_num_dests_incl_self = compute_grid.x * compute_grid.y;

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
        k,                     // 6
        e_local,               // 7
        num_total_cores,       // 8
        brisc_done_sem_id,     // 9
        brisc_release_sem_id,  // 10
    };
    tt::tt_metal::TensorAccessorArgs(ungrouped_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(plan_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(offsets_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(counts_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(metadata_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(scores_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(leids_buf).append_to(writer_ct_args);

    auto writer_kernel_g1 = create_writer_kernel(program, worker_group_1, writer_ct_args, {}, kWriterKernelPath);
    tt::tt_metal::KernelHandle writer_kernel_g2{};
    if (!worker_group_2.ranges().empty()) {
        writer_kernel_g2 = create_writer_kernel(program, worker_group_2, writer_ct_args, {}, kWriterKernelPath);
    }

    // -------------------------------------------------------------------------
    // Compute kernel — does scaled untilize: mul cb_src * cb_w → cb_inter,
    // then untilize cb_inter → cb_out. Removes the per-element scalar multiply
    // from BRISC; writer only does scalar add for RMW after this.
    // -------------------------------------------------------------------------
    [[maybe_unused]] auto compute_g1 = create_compute_kernel(
        program,
        worker_group_1,
        {kCbSrc0, kCbOut, tiles_per_chunk, kCbW, kCbExistingRm, kCbExistingTile, kCbCombined, kCbScaled, kCbCtrl},
        {},
        kComputeKernelPath,
        false);
    if (!worker_group_2.ranges().empty()) {
        [[maybe_unused]] auto compute_g2 = create_compute_kernel(
            program,
            worker_group_2,
            {kCbSrc0, kCbOut, tiles_per_chunk, kCbW, kCbExistingRm, kCbExistingTile, kCbCombined, kCbScaled, kCbCtrl},
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
            ungrouped_buf->address(),  // 0
            plan_buf->address(),       // 1
            offsets_buf->address(),    // 2
            counts_buf->address(),     // 3
            metadata_buf->address(),   // 4
            scores_buf->address(),     // 5
            leids_buf->address(),      // 6
            worker_idx,                // 7 my_core_idx
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
    auto* counts_buf = tensor_args.counts.buffer();
    auto* metadata_buf = tensor_args.metadata.buffer();
    auto* scores_buf = tensor_args.scores.buffer();
    auto* leids_buf = tensor_args.local_expert_ids.buffer();

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
        writer_rt[3] = counts_buf->address();
        writer_rt[4] = metadata_buf->address();
        writer_rt[5] = scores_buf->address();
        writer_rt[6] = leids_buf->address();
    }
}

}  // namespace ttml::metal::ops::moe_ungroup::device
