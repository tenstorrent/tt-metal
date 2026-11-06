// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_group_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "moe_group_device_operation.hpp"

namespace {

constexpr auto kCombinedKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_group/device/kernels/dataflow/moe_group_combined_kernel.cpp";
constexpr auto kWorkerWriterPath =
    "tt-train/sources/ttml/metal/ops/moe_group/device/kernels/dataflow/moe_group_worker_writer.cpp";
constexpr auto kTilizeKernelPath =
    "tt-train/sources/ttml/metal/ops/moe_group/device/kernels/compute/moe_group_tilize_kernel.cpp";

// Kernel handles are the push order into ProgramDescriptor::kernels, which keeps the legacy creation order: combined
// group 1, combined group 2, writer group 1, writer group 2, compute group 1, compute group 2, each group-2 kernel only
// when group 2 has cores. Without group 2 the writer is handle 1.
constexpr uint32_t kCombinedGroup1KernelIdx = 0U;
constexpr uint32_t kCombinedGroup2KernelIdx = 1U;
constexpr uint32_t kWriterGroup1KernelIdx = 2U;
constexpr uint32_t kWriterGroup2KernelIdx = 3U;
constexpr uint32_t kWriterKernelIdxWithoutGroup2 = 1U;

// combined kernel runtime args
constexpr uint32_t kPlanAddrIdx = 0U;
constexpr uint32_t kDispatchedAddrIdx = 1U;
constexpr uint32_t kMetadataAddrIdx = 4U;
constexpr uint32_t kCountsAddrIdx = 5U;
constexpr uint32_t kCombinedOffsetsAddrIdx = 6U;
constexpr uint32_t kLeidsAddrIdx = 7U;
constexpr uint32_t kScoresAddrIdx = 11U;
constexpr uint32_t kGroupedScoresAddrIdx = 12U;
constexpr uint32_t kKSlotAddrIdx = 13U;
// writer runtime args
constexpr uint32_t kGroupedAddrIdx = 0U;
constexpr uint32_t kWriterOffsetsAddrIdx = 3U;

// CB indices
constexpr uint32_t kCbSrc0 = tt::CBIndex::c_0;
constexpr uint32_t kCbOut = tt::CBIndex::c_2;
constexpr uint32_t kCbScan = tt::CBIndex::c_3;
constexpr uint32_t kCbPlan = tt::CBIndex::c_4;
constexpr uint32_t kCbOffset = tt::CBIndex::c_5;
constexpr uint32_t kCbCtrl = tt::CBIndex::c_6;  // NCRISC->compute: per-core active block count

// Phase semaphore ids, in the order legacy CreateSemaphore handed them out; the kernels get them as compile-time args.
constexpr uint32_t kScanPhase1SemId = 0U;
constexpr uint32_t kScanPhase2SemId = 1U;
constexpr uint32_t kScanPhase3SemId = 2U;
constexpr uint32_t kPlanReadySemId = 3U;

constexpr uint32_t kTargetChunkBytes = 128U * 1024U;

// L1 alignment for CB pages and cross-core NOC writes. NOT the tile width
// (which happens to also be 32 elements) — different unit, same number.
constexpr uint32_t kL1_ALIGN = 32U;

uint32_t pick_num_chunks(uint32_t h) {
    const uint32_t row_bytes = h * 2U;
    const uint32_t strip_bytes = 32U * row_bytes;
    uint32_t nc = (strip_bytes + kTargetChunkBytes - 1U) / kTargetChunkBytes;
    if (nc == 0U)
        nc = 1U;
    // ceil(h / TILE_WIDTH) — last tile may be partial when h isn't tile-aligned.
    const uint32_t Wt = tt::round_up(h, tt::constants::TILE_WIDTH) / tt::constants::TILE_WIDTH;
    if (Wt > 0U && nc > Wt)
        nc = Wt;
    if (nc == 0U)
        nc = 1U;
    return nc;
}

}  // namespace

namespace ttml::metal::ops::moe_group::device {

namespace {

// The grouped output's tile rows (t_cap / TILE_HEIGHT) over the device's compute grid. create_descriptor() pins it
// against split_work_to_cores, and override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const operation_attributes_t& attrs, const tensor_args_t& args) {
    return compute_work_split(
        args.dispatched.device()->compute_with_storage_grid_size(), attrs.t_cap / tt::constants::TILE_HEIGHT);
}

}  // namespace

tt::tt_metal::ProgramDescriptor MoeGroupProgramFactory::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& outputs) {
    auto* device = args.dispatched.device();
    auto& grouped = std::get<0>(outputs);
    auto& grouped_scores = std::get<1>(outputs);
    auto& k_slot = std::get<2>(outputs);
    auto& counts = std::get<3>(outputs);
    auto& offsets = std::get<4>(outputs);
    auto& plan = std::get<5>(outputs);

    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t h = attrs.h;
    const uint32_t e_local = attrs.e_local;
    const uint32_t k = attrs.k;
    const uint32_t total_rows = attrs.d * attrs.b * attrs.s;
    const uint32_t t_cap = attrs.t_cap;

    const uint32_t num_chunks = pick_num_chunks(h);
    // ceil(h / TILE_WIDTH); the last tile is padded with zeros if h % 32 != 0.
    const uint32_t Wt = tt::round_up(h, tt::constants::TILE_WIDTH) / tt::constants::TILE_WIDTH;
    const uint32_t tiles_per_chunk = tt::round_up(Wt, num_chunks) / num_chunks;
    const uint32_t hidden_chunk_bytes = tiles_per_chunk * tt::constants::TILE_WIDTH * 2U;
    const uint32_t last_chunk_tiles = Wt - (num_chunks - 1U) * tiles_per_chunk;
    // Valid bytes in the last chunk reflect the actual H, not the tile-rounded
    // width: dispatched is row-major bf16 with stride h*2, so reading more than
    // (h - prior_chunks_width) * 2 bytes would cross into the next row. The
    // reader pads the unused L1 tail to hidden_chunk_bytes with zeros.
    const uint32_t prior_chunks_bytes = (num_chunks - 1U) * hidden_chunk_bytes;
    const uint32_t last_chunk_bytes = h * 2U - prior_chunks_bytes;

    // -------------------------------------------------------------------------
    // Core assignment: ALL cores run the combined scan+reader kernel.
    // No more separate scan core.
    // -------------------------------------------------------------------------
    const auto compute_grid = device->compute_with_storage_grid_size();

    const tt::tt_metal::CoreCoord lead_coord{0, 0};
    const tt::tt_metal::CoreRange full_range{
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{compute_grid.x - 1U, compute_grid.y - 1U}};
    const tt::tt_metal::CoreRangeSet all_cores{full_range};

    const uint32_t total_tiles = t_cap / tt::constants::TILE_HEIGHT;
    // Use the CoreCoord overload (like layernorm_bw) — routes through
    // num_cores_to_corerangeset, avoiding the sub-grid placement bug in
    // the CoreRangeSet overload that throws "Failed to assign all N
    // requested cores" when total_tiles % grid_size lands on specific
    // remainders.
    auto [num_workers, worker_all, worker_group_1, worker_group_2, tiles_group_1, tiles_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_tiles);
    const WorkSplit split = make_work_split(attrs, args);
    check_work_split(split, num_workers, worker_group_1, worker_group_2, tiles_group_1, tiles_group_2);

    const uint32_t num_total_cores = num_workers;  // every core does scan + worker

    // Per-core scan slice
    const uint32_t scan_slice_size = (total_rows + num_total_cores - 1U) / num_total_cores;

    // -------------------------------------------------------------------------
    // Circular buffers
    // -------------------------------------------------------------------------
    const uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kCbSrc0, tt::DataFormat::Float16_b, bf16_tile_bytes, 2U * tiles_per_chunk));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kCbOut, tt::DataFormat::Float16_b, bf16_tile_bytes, 2U * tiles_per_chunk));

    constexpr uint32_t kPlanCbBytes = 32U * sizeof(uint32_t);
    desc.cbs.push_back(make_cb_descriptor_bytes(all_cores, kCbPlan, tt::DataFormat::UInt32, kPlanCbBytes));

    const uint32_t offset_cb_bytes = tt::round_up((e_local + 1U) * sizeof(uint32_t), kL1_ALIGN);
    desc.cbs.push_back(make_cb_descriptor_bytes(all_cores, kCbOffset, tt::DataFormat::UInt32, offset_cb_bytes));

    // cb_ctrl: NCRISC reader publishes per-core active-block count after the
    // worker phase reads offsets[E_local]; compute reads it to size the bulk
    // tilize call. 16B page (one uint32 padded to L1 alignment).
    constexpr uint32_t cb_ctrl_bytes = 16U;
    desc.cbs.push_back(make_cb_descriptor_bytes(all_cores, kCbCtrl, tt::DataFormat::UInt32, cb_ctrl_bytes));

    // cb_scan: every core's scratch for scan + shared tables (only meaningful on lead).
    // Layout: [stage 128B][leids_buf 32B][counts e_local*4][offsets (e_local+1)*4]
    //         [cursors e_local*4]
    //         [shared_local_counts num_total_cores * e_local * 4]
    //         [shared_per_core_start num_total_cores * e_local * 4]
    //         [md_block (BLOCK_ROWS or slice_size)*32 + 32]
    //         [plan_stage e_local * PLAN_CHUNK * 4][fill e_local * 4]
    constexpr uint32_t kBlockRows = 1024U;
    uint32_t slice_block_rows = scan_slice_size < kBlockRows ? scan_slice_size : kBlockRows;
    if (slice_block_rows == 0U)
        slice_block_rows = 1U;

    // Named sizes for the cb_scan layout.
    const uint32_t dram_align_bytes = tt::tt_metal::hal::get_dram_alignment();
    // Stage scratch must hold the offsets DMA write of size off_page_bytes
    // = round_up((e_local+1)*4, kL1_ALIGN). Pin a 128 B floor so leids_buf
    // stays at a stable offset for small e_local.
    const uint32_t off_page_bytes_host = tt::round_up((e_local + 1U) * sizeof(uint32_t), kL1_ALIGN);
    const uint32_t kStageBytes = std::max<uint32_t>(off_page_bytes_host, 128U);
    // leids_buf must match the kernel TensorAccessor's aligned page for the
    // leids tensor. On BH this is DRAM-aligned (64B), and using only the L1
    // alignment places shared_local_counts on top of the private counts array.
    const uint32_t leids_aligned_page_host = tt::round_up(e_local * sizeof(uint16_t), dram_align_bytes);
    const uint32_t kLeidsBufBytes = std::max<uint32_t>(leids_aligned_page_host, 32U);
    constexpr uint32_t kPlanChunk = 32U;  // plan pre-fill burst size (entries per chunk)
    // Metadata / scores aligned page = round_up(K * sizeof(uint16), DRAM_ALIGNMENT). DRAM alignment
    // is arch-specific (32 B on WH, 64 B on BH) and must match the TensorAccessor's AlignedPageSize
    // computed kernel-side — otherwise cb_scan is undersized and md_block / sc_block writes overflow
    // into plan_stage / gs_stage / ks_stage, corrupting the NOC staging buffers that publish
    // offsets/counts/plan to DRAM.
    const uint32_t kMdAlignedPage = tt::round_up(k * sizeof(uint16_t), dram_align_bytes);
    constexpr uint32_t kOverheadSlack = 64U;  // safety pad between sections (covers alignment carry-over)
    // counts(e_local) + offsets(e_local+1) + cursors(e_local) = 3*e_local + 1 uint32 entries.
    constexpr uint32_t kHeaderU32PerExpert = 3U;
    const uint32_t header_bytes = (kHeaderU32PerExpert * e_local + 1U) * sizeof(uint32_t);

    const uint32_t overhead_bytes = kStageBytes + kLeidsBufBytes + header_bytes + kOverheadSlack;
    // Each shared table has num_total_cores slots; slot size =
    // round_up_to_align(e_local) uint32s (smallest multiple of the arch's L1
    // alignment that fits e_local). Arch-specific via HAL (16 B on WH/BH today,
    // may change on future parts).
    const uint32_t l1_align_u32 = tt::tt_metal::hal::get_l1_alignment() / sizeof(uint32_t);
    uint32_t shared_slot_u32 = tt::round_up(e_local, l1_align_u32);
    if (shared_slot_u32 < l1_align_u32)
        shared_slot_u32 = l1_align_u32;
    const uint32_t kSharedSlotBytes = shared_slot_u32 * sizeof(uint32_t);
    const uint32_t shared_table_bytes = num_total_cores * kSharedSlotBytes;
    const uint32_t two_shared_tables = 2U * shared_table_bytes;
    const uint32_t md_block_bytes = slice_block_rows * kMdAlignedPage + kMdAlignedPage;
    // sc_block holds scores in lock-step with md_block. scores is bf16 with the
    // same K count per row, so the aligned page size matches md_aligned_page.
    const uint32_t sc_block_bytes = slice_block_rows * kMdAlignedPage + kMdAlignedPage;
    const uint32_t plan_stage_bytes = tt::round_up(e_local * kPlanChunk * sizeof(uint32_t), kL1_ALIGN);
    // gs_stage / ks_stage each hold e_local * kPlanChunk uint16-sized entries
    // (bf16 grouped_scores / uint16 k_slot).
    const uint32_t gs_stage_bytes = tt::round_up(e_local * kPlanChunk * sizeof(uint16_t), kL1_ALIGN);
    const uint32_t ks_stage_bytes = tt::round_up(e_local * kPlanChunk * sizeof(uint16_t), kL1_ALIGN);
    const uint32_t fill_bytes = tt::round_up(e_local * sizeof(uint32_t), kL1_ALIGN);
    uint32_t scan_scratch_bytes = overhead_bytes + two_shared_tables + md_block_bytes + sc_block_bytes +
                                  plan_stage_bytes + gs_stage_bytes + ks_stage_bytes + fill_bytes;
    scan_scratch_bytes = tt::round_up(scan_scratch_bytes, kL1_ALIGN);
    desc.cbs.push_back(make_cb_descriptor_bytes(all_cores, kCbScan, tt::DataFormat::UInt32, scan_scratch_bytes));

    // Compute address of shared tables in cb_scan (offset within scratch).
    // Layout: stage(kStageBytes), leids_buf(kLeidsBufBytes), counts, offsets, cursors.
    // MUST be kL1_ALIGN-aligned for cross-core NOC writes to land correctly.
    const uint32_t shared_tables_offset_raw = kStageBytes + kLeidsBufBytes + header_bytes;
    const uint32_t shared_tables_offset = tt::round_up(shared_tables_offset_raw, kL1_ALIGN);

    // Phase semaphores
    desc.semaphores.push_back(make_semaphore_descriptor(kScanPhase1SemId, all_cores, 0U));
    desc.semaphores.push_back(make_semaphore_descriptor(kScanPhase2SemId, all_cores, 0U));
    desc.semaphores.push_back(make_semaphore_descriptor(kScanPhase3SemId, all_cores, 0U));
    desc.semaphores.push_back(make_semaphore_descriptor(kPlanReadySemId, all_cores, 0U));

    // -------------------------------------------------------------------------
    // Buffer pointers
    // -------------------------------------------------------------------------
    auto* metadata_buf = args.metadata.buffer();
    auto* scores_buf = args.scores.buffer();
    auto* plan_buf = plan.buffer();
    auto* grouped_scores_buf = grouped_scores.buffer();
    auto* k_slot_buf = k_slot.buffer();
    auto* counts_buf = counts.buffer();
    auto* offsets_buf = offsets.buffer();
    auto* leids_buf = args.local_expert_ids.buffer();
    auto* dispatched_buf = args.dispatched.buffer();
    auto* grouped_buf = grouped.buffer();

    // -------------------------------------------------------------------------
    // NOC coords used as CT args by the combined kernel
    // -------------------------------------------------------------------------
    const auto lead_virt = device->worker_core_from_logical_core(lead_coord);

    // Bounding NOC rectangle of the full worker grid — lead uses one multicast
    // per broadcast (phase2_sem + plan_ready_sem) via
    // mcast_sender_signal_receivers_loopback.
    const auto mcast_tl = device->worker_core_from_logical_core({0, 0});
    const auto mcast_br = device->worker_core_from_logical_core({compute_grid.x - 1U, compute_grid.y - 1U});
    const uint32_t mcast_sx = std::min(mcast_tl.x, mcast_br.x);
    const uint32_t mcast_ex = std::max(mcast_tl.x, mcast_br.x);
    const uint32_t mcast_sy = std::min(mcast_tl.y, mcast_br.y);
    const uint32_t mcast_ey = std::max(mcast_tl.y, mcast_br.y);
    const uint32_t mcast_num_dests_including_self = compute_grid.x * compute_grid.y;

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
        lead_virt.x,                     // 12
        lead_virt.y,                     // 13
        kScanPhase1SemId,                // 14
        kScanPhase2SemId,                // 15
        kScanPhase3SemId,                // 16
        kPlanReadySemId,                 // 17
        shared_tables_offset,            // 18
        mcast_sx,                        // 19
        mcast_sy,                        // 20
        mcast_ex,                        // 21
        mcast_ey,                        // 22
        mcast_num_dests_including_self,  // 23
        kCbCtrl};                        // 24
    tt::tt_metal::TensorAccessorArgs(plan_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(dispatched_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(metadata_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(counts_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(offsets_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(leids_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(scores_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(grouped_scores_buf).append_to(combined_ct_args);
    tt::tt_metal::TensorAccessorArgs(k_slot_buf).append_to(combined_ct_args);

    auto combined_kernel_g1 = make_reader_kernel_descriptor(worker_group_1, combined_ct_args, {}, kCombinedKernelPath);
    auto combined_kernel_g2 = make_reader_kernel_descriptor(worker_group_2, combined_ct_args, {}, kCombinedKernelPath);

    // -------------------------------------------------------------------------
    // Worker writer kernel (BRISC) — unchanged
    // -------------------------------------------------------------------------
    std::vector<uint32_t> writer_ct_args = {
        num_chunks,
        tiles_per_chunk,
        Wt,
        e_local,
        last_chunk_tiles,
        num_total_cores,   // 5 — stride (= num_workers)
        kPlanReadySemId};  // 6
    tt::tt_metal::TensorAccessorArgs(grouped_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(offsets_buf).append_to(writer_ct_args);

    auto writer_kernel_g1 = make_writer_kernel_descriptor(worker_group_1, writer_ct_args, {}, kWorkerWriterPath);
    auto writer_kernel_g2 = make_writer_kernel_descriptor(worker_group_2, writer_ct_args, {}, kWorkerWriterPath);

    // -------------------------------------------------------------------------
    // Compute (tilize) kernel — unchanged
    // -------------------------------------------------------------------------
    auto compute_g1 = make_compute_kernel_descriptor(
        worker_group_1,
        {kCbSrc0, kCbOut, tiles_group_1, tiles_per_chunk, num_chunks, kCbCtrl},
        {},
        kTilizeKernelPath,
        false);
    auto compute_g2 = make_compute_kernel_descriptor(
        worker_group_2,
        {kCbSrc0, kCbOut, tiles_group_2, tiles_per_chunk, num_chunks, kCbCtrl},
        {},
        kTilizeKernelPath,
        false);

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

    // Strided tile-row layout: tile_row = my_worker_start + step * num_workers.
    // We use a UNIFORM my_worker_count = tiles_group_1 (the worst case) for ALL
    // cores so the runtime my_active_count formula in the kernel correctly
    // covers every active tile-row regardless of group_1/group_2 split. The
    // kernel's runtime cb_ctrl protocol clips the loop to the actual active
    // count per core, so cores that "would" have done tiles_group_2 worth
    // simply do fewer iterations at runtime.
    uint32_t worker_idx = 0;
    for (const auto& core : worker_cores_vec) {
        uint32_t tiles_this_core = tiles_group_1;

        // Scan slice for this core (worker_idx == my_core_idx for scan purposes).
        uint32_t my_slice_start = worker_idx * scan_slice_size;
        uint32_t my_slice_end = my_slice_start + scan_slice_size;
        if (my_slice_end > total_rows)
            my_slice_end = total_rows;
        if (my_slice_start > total_rows)
            my_slice_start = total_rows;

        // RT args are now only the things that vary per-core (or per-run):
        // tensor buffer addresses and per-core identity/slice bounds.
        // Everything previously here that was globally-constant (sem ids,
        // lead NOC, mcast rect, shared_tables_offset, worker_stride) moved
        // to CT args in combined_ct_args above. Slots as named at the top of
        // the file; override_runtime_arguments() rewrites the addresses on
        // every cache hit. The per-core values follow from the hashed
        // attributes.
        const bool is_g1 = worker_group_1.contains(core);
        auto& combined_kernel = is_g1 ? combined_kernel_g1 : combined_kernel_g2;
        combined_kernel.emplace_runtime_args(
            core,
            {
                plan_buf,            // 0
                dispatched_buf,      // 1
                worker_idx,          // 2 my_worker_start (interleaved tile_row)
                tiles_this_core,     // 3 my_worker_count
                metadata_buf,        // 4
                counts_buf,          // 5
                offsets_buf,         // 6
                leids_buf,           // 7
                worker_idx,          // 8 my_core_idx
                my_slice_start,      // 9
                my_slice_end,        // 10
                scores_buf,          // 11
                grouped_scores_buf,  // 12
                k_slot_buf,          // 13
            });

        auto& writer_kernel = is_g1 ? writer_kernel_g1 : writer_kernel_g2;
        writer_kernel.emplace_runtime_args(
            core,
            {
                grouped_buf,      // 0
                worker_idx,       // 1 my_start
                tiles_this_core,  // 2 my_count
                offsets_buf,      // 3
            });
        worker_idx++;
    }

    // Kernels in legacy creation order, which gives the handles named at the top of the file: combined, writer and
    // compute, each for group 1 then group 2. The group 2 kernels are skipped when the tiles split evenly
    // (worker_group_2 empty).
    for (auto* kernel :
         {&combined_kernel_g1, &combined_kernel_g2, &writer_kernel_g1, &writer_kernel_g2, &compute_g1, &compute_g2}) {
        if (!kernel->core_ranges.ranges().empty()) {
            desc.kernels.push_back(std::move(*kernel));
        }
    }

    return desc;
}

void MoeGroupProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& attrs,
    const tensor_args_t& args,
    tensor_return_value_t& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(attrs, args);

    const uint32_t plan_addr = std::get<5>(outputs).buffer()->address();
    const uint32_t dispatched_addr = args.dispatched.buffer()->address();
    const uint32_t metadata_addr = args.metadata.buffer()->address();
    const uint32_t counts_addr = std::get<3>(outputs).buffer()->address();
    const uint32_t offsets_addr = std::get<4>(outputs).buffer()->address();
    const uint32_t leids_addr = args.local_expert_ids.buffer()->address();
    const uint32_t scores_addr = args.scores.buffer()->address();
    const uint32_t grouped_scores_addr = std::get<1>(outputs).buffer()->address();
    const uint32_t k_slot_addr = std::get<2>(outputs).buffer()->address();
    const uint32_t grouped_addr = std::get<0>(outputs).buffer()->address();

    auto& combined_group_1_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kCombinedGroup1KernelIdx);
    auto& writer_group_1_runtime_args = tt::tt_metal::GetRuntimeArgs(
        program, split.has_group_2() ? kWriterGroup1KernelIdx : kWriterKernelIdxWithoutGroup2);
    // Without group 2 there are no group-2 kernels and every core is in group 1, so the aliases are never indexed.
    auto& combined_group_2_runtime_args = split.has_group_2()
                                              ? tt::tt_metal::GetRuntimeArgs(program, kCombinedGroup2KernelIdx)
                                              : combined_group_1_runtime_args;
    auto& writer_group_2_runtime_args = split.has_group_2()
                                            ? tt::tt_metal::GetRuntimeArgs(program, kWriterGroup2KernelIdx)
                                            : writer_group_1_runtime_args;

    // The cores and groups of create_descriptor()'s row-wise walk over worker_all, visited column-major instead: only
    // addresses are written here, so the order does not matter.
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& combined_args =
            (work.in_group_1 ? combined_group_1_runtime_args : combined_group_2_runtime_args)[core.x][core.y];
        combined_args[kPlanAddrIdx] = plan_addr;
        combined_args[kDispatchedAddrIdx] = dispatched_addr;
        combined_args[kMetadataAddrIdx] = metadata_addr;
        combined_args[kCountsAddrIdx] = counts_addr;
        combined_args[kCombinedOffsetsAddrIdx] = offsets_addr;
        combined_args[kLeidsAddrIdx] = leids_addr;
        combined_args[kScoresAddrIdx] = scores_addr;
        combined_args[kGroupedScoresAddrIdx] = grouped_scores_addr;
        combined_args[kKSlotAddrIdx] = k_slot_addr;

        auto& writer_args =
            (work.in_group_1 ? writer_group_1_runtime_args : writer_group_2_runtime_args)[core.x][core.y];
        writer_args[kGroupedAddrIdx] = grouped_addr;
        writer_args[kWriterOffsetsAddrIdx] = offsets_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(MoeGroupDeviceOperation, MoeGroupProgramFactory);

}  // namespace ttml::metal::ops::moe_group::device
