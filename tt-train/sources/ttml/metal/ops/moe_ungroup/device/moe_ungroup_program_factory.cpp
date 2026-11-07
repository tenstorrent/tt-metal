// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ungroup_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "moe_ungroup_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/dataflow/moe_ungroup_reader.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/dataflow/moe_ungroup_rmw_writer.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/compute/moe_ungroup_untilize_kernel.cpp";

// Kernel push order: reader, writer, then the single compute kernel on the full grid
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader runtime args
constexpr uint32_t kExpertOutAddrIdx = 0U;
constexpr uint32_t kReaderOffsetsAddrIdx = 1U;
// writer runtime args
constexpr uint32_t kUngroupedAddrIdx = 0U;
constexpr uint32_t kPlanAddrIdx = 1U;
constexpr uint32_t kWriterOffsetsAddrIdx = 2U;
constexpr uint32_t kGroupedScoresAddrIdx = 3U;

// Semaphore ids on the full grid. The kernels take them as compile-time args (reader CT 7..10, writer CT 8..9).
constexpr uint32_t kUpSemId = 0U;
constexpr uint32_t kDownSemId = 1U;
constexpr uint32_t kBriscDoneSemId = 2U;
constexpr uint32_t kBriscReleaseSemId = 3U;

// CB indices (must match the kernels' constexpr declarations)
constexpr uint32_t kCbSrc0 = tt::CBIndex::c_0;
constexpr uint32_t kCbOut = tt::CBIndex::c_2;
constexpr uint32_t kCbReaderScratch = tt::CBIndex::c_3;  // reader offsets + per-expert caches
constexpr uint32_t kCbScratch = tt::CBIndex::c_4;        // writer's scratch (zero buf, offsets, plan, w)
constexpr uint32_t kCbW = tt::CBIndex::c_5;              // COL-broadcast weight tile (only col 0 populated, w[r])
constexpr uint32_t kCbExistingRm = tt::CBIndex::c_6;     // row-major existing rows from ungrouped (writer fills)
constexpr uint32_t kCbExistingTile = tt::CBIndex::c_7;   // tilized existing (compute internal)
constexpr uint32_t kCbCombined = tt::CBIndex::c_8;       // mul+add output tiles (untilize input)
constexpr uint32_t kCbCtrl = tt::CBIndex::c_10;          // NCRISC->compute: per-core active-block count

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

tt::tt_metal::ProgramDescriptor MoeUngroupProgramFactory::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& output) {
    auto* device = args.expert_out.device();
    tt::tt_metal::ProgramDescriptor desc;

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
    const uint32_t num_workers = compute_grid.x * compute_grid.y;
    const uint32_t num_total_cores = num_workers;

    // -------------------------------------------------------------------------
    // Circular buffers
    // -------------------------------------------------------------------------
    const uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    // cb_src0: TILE bf16, double-buffered
    desc.cbs.push_back(
        make_cb_descriptor(worker_all, kCbSrc0, tt::DataFormat::Float16_b, bf16_tile_bytes, 2U * tiles_per_chunk));

    // cb_out: row-major bf16 produced by untilize, double-buffered
    desc.cbs.push_back(
        make_cb_descriptor(worker_all, kCbOut, tt::DataFormat::Float16_b, bf16_tile_bytes, 2U * tiles_per_chunk));

    // cb_reader_scratch: reader scratch holding offsets DMA + per-expert caches:
    //   offsets_l1 (e_local+1) u32  +  tr_start_per_expert e_local u32
    //                              +  my_real_count_per_expert e_local u32
    // Backing the caches in L1 keeps NCRISC stack usage bounded for large
    // e_local (e.g. 300+) where stack arrays would otherwise overflow.
    const uint32_t kL1_ALIGN = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t cb_reader_scratch_bytes = tt::round_up((3U * e_local + 1U) * sizeof(uint32_t), kL1_ALIGN);
    desc.cbs.push_back(
        make_cb_descriptor_bytes(worker_all, kCbReaderScratch, tt::DataFormat::UInt32, cb_reader_scratch_bytes));

    // cb_w: COL-broadcast weight tile (TILE bf16). BRISC writer builds this each
    // chunk, populating only column 0 with w_tile[r,0] = bf16(w[r]); compute
    // COL-broadcasts it against cb_src0 before untilizing. Capacity: 2
    // (double-buffer for pipelining).
    desc.cbs.push_back(make_cb_descriptor(worker_all, kCbW, tt::DataFormat::Float16_b, bf16_tile_bytes, 2U));

    // cb_existing_rm: row-major existing rows from ungrouped DRAM (writer fills,
    // compute tilizes). Asymmetric pages: one page per ROW (hidden_chunk_bytes).
    // Double-buffered by chunk so BRISC can DMA chunk N+1 while compute
    // tilizes/mul/add/untilizes chunk N.
    constexpr uint32_t cb_existing_rm_pages = 2U * tt::constants::TILE_HEIGHT;  // two chunks, one page per row
    desc.cbs.push_back(make_cb_descriptor_bytes(
        worker_all,
        kCbExistingRm,
        tt::DataFormat::Float16_b,
        cb_existing_rm_pages * hidden_chunk_bytes,
        /*page_size_bytes=*/hidden_chunk_bytes));

    // Compute-private intermediates stay single-buffered: they are produced
    // and consumed by the same compute kernel, so extra depth would spend L1
    // without adding reader/writer overlap.
    // cb_existing_tile: compute's tilize output, fed into mul+add. One TILE_HEIGHT-row
    // block produces tiles_per_chunk tiles.
    desc.cbs.push_back(
        make_cb_descriptor(worker_all, kCbExistingTile, tt::DataFormat::Float16_b, bf16_tile_bytes, tiles_per_chunk));

    // cb_combined: compute's add output, untilize input.
    desc.cbs.push_back(
        make_cb_descriptor(worker_all, kCbCombined, tt::DataFormat::Float16_b, bf16_tile_bytes, tiles_per_chunk));

    // cb_ctrl: NCRISC reader publishes per-core active-block count once at
    // startup; compute reads it to size its outer loop. 16B page (one uint32
    // padded to L1 alignment), single-page CB.
    constexpr uint32_t cb_ctrl_bytes = 16U;
    desc.cbs.push_back(make_cb_descriptor_bytes(worker_all, kCbCtrl, tt::DataFormat::UInt32, cb_ctrl_bytes));

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
    desc.cbs.push_back(make_cb_descriptor_bytes(worker_all, kCbScratch, tt::DataFormat::UInt32, scratch_bytes));

    // -------------------------------------------------------------------------
    // Semaphores
    //   up_sem / down_sem    — NCRISC-side cross-core mcast barrier (NOC_0).
    //   brisc_done_sem       — local BRISC->NCRISC handshake (BRISC sets, NCRISC polls).
    //   brisc_release_sem    — local NCRISC->BRISC handshake (NCRISC sets, BRISC polls).
    // Both pairs are reused for the prezero barrier and every inter-expert
    // barrier (each call resets the sems to 0 before the next handshake).
    // -------------------------------------------------------------------------
    desc.semaphores.push_back(make_semaphore_descriptor(kUpSemId, worker_all, /*initial_value=*/0U));
    desc.semaphores.push_back(make_semaphore_descriptor(kDownSemId, worker_all, /*initial_value=*/0U));
    desc.semaphores.push_back(make_semaphore_descriptor(kBriscDoneSemId, worker_all, /*initial_value=*/0U));
    desc.semaphores.push_back(make_semaphore_descriptor(kBriscReleaseSemId, worker_all, /*initial_value=*/0U));

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
        kUpSemId,                   // 7
        kDownSemId,                 // 8
        kBriscDoneSemId,            // 9
        kBriscReleaseSemId,         // 10
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

    desc.kernels.push_back(  // kReaderKernelIdx
        make_reader_kernel_descriptor(worker_all, reader_ct_args, {}, kReaderKernelPath));

    // -------------------------------------------------------------------------
    // Writer CT args
    // -------------------------------------------------------------------------
    std::vector<uint32_t> writer_ct_args = {
        h,                   // 0
        num_chunks,          // 1
        hidden_chunk_bytes,  // 2
        tiles_per_chunk,     // 3
        last_chunk_bytes,    // 4
        total_rows,          // 5
        e_local,             // 6
        num_total_cores,     // 7
        kBriscDoneSemId,     // 8
        kBriscReleaseSemId,  // 9
        kL1_ALIGN,           // 10
    };
    tt::tt_metal::TensorAccessorArgs(ungrouped_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(plan_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(offsets_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(grouped_scores_buf).append_to(writer_ct_args);

    desc.kernels.push_back(  // kWriterKernelIdx
        make_writer_kernel_descriptor(worker_all, writer_ct_args, {}, kWriterKernelPath));

    // -------------------------------------------------------------------------
    // Compute kernel — tilizes existing rows, does broadcast-scaled accumulation
    // in DST (mul + add), then untilizes the combined rows for BRISC writeback.
    // -------------------------------------------------------------------------
    desc.kernels.push_back(make_compute_kernel_descriptor(
        worker_all,
        {kCbSrc0, kCbOut, tiles_per_chunk, kCbW, kCbExistingRm, kCbExistingTile, kCbCombined, kCbCtrl},
        {},
        kComputeKernelPath,
        /*fp32_dest_acc_en=*/false));

    // -------------------------------------------------------------------------
    // Per-core RT args
    // -------------------------------------------------------------------------
    // Every grid core gets args, since inactive cores still join the barriers. Slots as named at the top of the file;
    // override_runtime_arguments() rewrites the addresses on every cache hit. my_core_idx is the core's row-major
    // position in the grid (0 is the barrier lead at {0, 0}), so it depends only on the device.
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    auto worker_cores_vec = tt::tt_metal::corerange_to_cores(worker_all, num_workers, /*row_wise=*/true);
    uint32_t worker_idx = 0;
    for (const auto& core : worker_cores_vec) {
        reader.emplace_runtime_args(
            core,
            {
                expert_out_buf,  // 0
                offsets_buf,     // 1
                worker_idx,      // 2 my_core_idx
            });
        writer.emplace_runtime_args(
            core,
            {
                ungrouped_buf,       // 0
                plan_buf,            // 1
                offsets_buf,         // 2
                grouped_scores_buf,  // 3
                worker_idx,          // 4 my_core_idx
            });
        worker_idx++;
    }

    return desc;
}

void MoeUngroupProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // create_descriptor() gives args to every core of the compute grid.
    const auto compute_grid = args.expert_out.device()->compute_with_storage_grid_size();

    const uint32_t expert_out_addr = args.expert_out.buffer()->address();
    const uint32_t ungrouped_addr = output.buffer()->address();
    const uint32_t plan_addr = args.plan.buffer()->address();
    const uint32_t offsets_addr = args.offsets.buffer()->address();
    const uint32_t grouped_scores_addr = args.grouped_scores.buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for (uint32_t x = 0; x < compute_grid.x; ++x) {
        for (uint32_t y = 0; y < compute_grid.y; ++y) {
            auto& reader_args = reader_runtime_args[x][y];
            reader_args[kExpertOutAddrIdx] = expert_out_addr;
            reader_args[kReaderOffsetsAddrIdx] = offsets_addr;

            auto& writer_args = writer_runtime_args[x][y];
            writer_args[kUngroupedAddrIdx] = ungrouped_addr;
            writer_args[kPlanAddrIdx] = plan_addr;
            writer_args[kWriterOffsetsAddrIdx] = offsets_addr;
            writer_args[kGroupedScoresAddrIdx] = grouped_scores_addr;
        }
    }
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(MoeUngroupDeviceOperation, MoeUngroupProgramFactory);

}  // namespace ttml::metal::ops::moe_ungroup::device
