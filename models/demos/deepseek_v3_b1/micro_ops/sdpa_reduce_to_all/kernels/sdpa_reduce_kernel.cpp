// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Unified SDPA Reduce-to-All Kernel
//
// This kernel handles both worker and forwarder cores via compile-time dispatch.
// The core role is determined by the `is_worker` compile-time arg:
// - is_worker=1: Worker core (reader/writer/compute for SDPA reduction)
// - is_worker=0: Forwarder core (non-blocking fabric packet forwarding)
//
// Worker Core:
//   - NCRISC: Reader - prepares local and neighbor MS/L data for compute
//   - BRISC: Writer - sends local/R1 data to neighbors, scatters output
//   - TRISC: Compute - streaming SDPA tail reduction (R1 + R2)
//
// Forwarder Core:
//   - NCRISC: BWD direction - forwards fabric packets
//   - BRISC: FWD direction - forwards fabric packets
//   - TRISC: No-op

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/sdpa_reduce_worker.hpp"
#include "../../../unified_kernels/sdpa_reduce_forwarder.hpp"

void kernel_main() {
    constexpr bool is_worker = get_named_compile_time_arg_val("is_worker") == 1;

#if defined(COMPILE_FOR_NCRISC)
    // ========================================================================
    // NCRISC: Worker has reader logic, Forwarder has BWD forwarding logic
    // ========================================================================
    if constexpr (is_worker) {
        using Worker = deepseek_b1_ops::SdpaReduceWorker;

        using ReaderCTArgs = Worker::ReaderCTArgs<
            get_named_compile_time_arg_val("cb_local_l"),
            get_named_compile_time_arg_val("cb_local_ms"),
            get_named_compile_time_arg_val("cb_r1_neighbor_l"),
            get_named_compile_time_arg_val("cb_r1_neighbor_ms"),
            get_named_compile_time_arg_val("cb_r2_neighbor_l"),
            get_named_compile_time_arg_val("cb_r2_neighbor_ms"),
            get_named_compile_time_arg_val("ms_tile_size_bytes"),
            get_named_compile_time_arg_val("l_chunk_size_bytes"),
            get_named_compile_time_arg_val("num_l_chunks"),
            get_named_compile_time_arg_val("tiles_per_l_chunk"),
            get_named_compile_time_arg_val("position_enabled"),
            get_named_compile_time_arg_val("per_device_chunk_size")>;

        // Dummy WriterCT and ComputeCT - not used by NCRISC but needed for Op template
        using WriterCTArgs = Worker::WriterCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;
        using ComputeCTArgs = Worker::ComputeCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

        uint32_t per_core_rta_arg_idx = 0;
        Worker::ReaderArgs reader_args{
            .r1_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_recv_buffer_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_recv_buffer_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
        };
        Worker::Op<ReaderCTArgs, WriterCTArgs, ComputeCTArgs> op;
        op(reader_args);
    } else {
        using Fwd = deepseek_b1_ops::SdpaReduceForwarder;

        using FwdCTArgs = Fwd::CTArgs<
            get_named_compile_time_arg_val("fwd_slots_per_round"),
            get_named_compile_time_arg_val("fwd_slot_size"),
            get_named_compile_time_arg_val("fwd_r2_buffer_offset")>;

        uint32_t per_core_rta_arg_idx = 0;
        Fwd::ForwarderArgs fwd_args{
            .buffer_base = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .buffer_offset = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
            .r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
        };
        Fwd::Op<FwdCTArgs> op;
        op(fwd_args);
    }

#elif defined(COMPILE_FOR_BRISC)
    // ========================================================================
    // BRISC: Worker has writer logic, Forwarder has FWD forwarding logic
    // ========================================================================
    if constexpr (is_worker) {
        using Worker = deepseek_b1_ops::SdpaReduceWorker;

        // Dummy ReaderCT and ComputeCT - not used by BRISC but needed for Op template
        using ReaderCTArgs = Worker::ReaderCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

        using WriterCTArgs = Worker::WriterCTArgs<
            get_named_compile_time_arg_val("cb_local_l"),
            get_named_compile_time_arg_val("cb_local_ms"),
            get_named_compile_time_arg_val("cb_r1_result_l"),
            get_named_compile_time_arg_val("cb_r1_result_ms"),
            get_named_compile_time_arg_val("cb_packet_slot"),
            get_named_compile_time_arg_val("l1_alignment"),
            get_named_compile_time_arg_val("page_size_bytes"),
            get_named_compile_time_arg_val("slot_size"),
            get_named_compile_time_arg_val("ms_tile_size_bytes"),
            get_named_compile_time_arg_val("l_chunk_size_bytes"),
            get_named_compile_time_arg_val("num_l_chunks"),
            get_named_compile_time_arg_val("tiles_per_l_chunk"),
            get_named_compile_time_arg_val("cb_l_out"),
            get_named_compile_time_arg_val("scatter_num_tiles"),
            get_named_compile_time_arg_val("scatter_src_tile_size"),
            get_named_compile_time_arg_val("scatter_dst_tile_size"),
            get_named_compile_time_arg_val("scatter_face_size"),
            get_named_compile_time_arg_val("scatter_row_face_size"),
            get_named_compile_time_arg_val("scatter_num_rows")>;

        using ComputeCTArgs = Worker::ComputeCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

        uint32_t per_core_rta_arg_idx = 0;
        Worker::WriterArgs writer_args{
            .r1_dst_mesh_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_dst_chip_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_neighbor_dst_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_dst_mesh_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_dst_chip_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_neighbor_dst_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .current_core_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .current_core_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .fwd_core_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .fwd_core_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_fwd_slot_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
            .r1_base_slot_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_fwd_slot_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
            .r2_base_slot_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .scatter_dest_l1_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .scatter_dest_coords_addr = get_arg_addr(per_core_rta_arg_idx++),
            // scatter_arrival_enabled=0, so this is not used
            .scatter_arrival_sem_addr = 0,
        };
        Worker::Op<ReaderCTArgs, WriterCTArgs, ComputeCTArgs> op;
        op(writer_args);
    } else {
        using Fwd = deepseek_b1_ops::SdpaReduceForwarder;

        using FwdCTArgs = Fwd::CTArgs<
            get_named_compile_time_arg_val("fwd_slots_per_round"),
            get_named_compile_time_arg_val("fwd_slot_size"),
            get_named_compile_time_arg_val("fwd_r2_buffer_offset")>;

        uint32_t per_core_rta_arg_idx = 0;
        Fwd::ForwarderArgs fwd_args{
            .buffer_base = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .buffer_offset = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
            .r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
        };
        Fwd::Op<FwdCTArgs> op;
        op(fwd_args);
    }

#elif defined(COMPILE_FOR_TRISC)
    // ========================================================================
    // TRISC: Only worker has compute logic; forwarder TRISC is no-op
    // ========================================================================
    if constexpr (is_worker) {
        using Worker = deepseek_b1_ops::SdpaReduceWorker;

        // Dummy ReaderCT and WriterCT - not used by TRISC but needed for Op template
        using ReaderCTArgs = Worker::ReaderCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;
        using WriterCTArgs = Worker::WriterCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

        using ComputeCTArgs = Worker::ComputeCTArgs<
            get_named_compile_time_arg_val("cb_local_l"),
            get_named_compile_time_arg_val("cb_local_ms"),
            get_named_compile_time_arg_val("cb_r1_neighbor_l"),
            get_named_compile_time_arg_val("cb_r1_neighbor_ms"),
            get_named_compile_time_arg_val("cb_r1_result_l"),
            get_named_compile_time_arg_val("cb_r1_result_ms"),
            get_named_compile_time_arg_val("cb_r2_neighbor_l"),
            get_named_compile_time_arg_val("cb_r2_neighbor_ms"),
            get_named_compile_time_arg_val("cb_l_out"),
            get_named_compile_time_arg_val("cb_ms_out"),
            get_named_compile_time_arg_val("scale_fp32"),
            get_named_compile_time_arg_val("tiles_per_l_chunk"),
            get_named_compile_time_arg_val("num_l_chunks"),
            get_named_compile_time_arg_val("position_enabled"),
            get_named_compile_time_arg_val("per_device_chunk_size"),
            get_named_compile_time_arg_val("final_reduction")>;

        // Initialize compute engine for unified kernel
        deepseek_compute_kernel_init();

        Worker::ComputeArgs compute_args{};
        Worker::Op<ReaderCTArgs, WriterCTArgs, ComputeCTArgs> op;
        op(compute_args);
    }
    // else: forwarder TRISC is no-op
#endif
}
