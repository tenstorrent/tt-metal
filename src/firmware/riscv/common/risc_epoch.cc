
#include "risc_epoch.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#ifndef ERISC
#include "context.h"
#endif

void run_epoch(
    void (*risc_epoch_load)(uint64_t), void (*risc_kernels_load)(uint64_t), void (*init_ncrisc_streams)(),
    bool skip_initial_epoch_dram_load, uint64_t dram_next_epoch_ptr, bool& skip_kernels, uint32_t& epoch_empty_check_cnt,
#ifdef RISC_GSYNC_ENABLED
    volatile uint32_t &gsync_epoch, volatile uint32_t &epochs_in_progress,
#endif
    uint32_t &num_dram_input_streams, uint32_t &num_dram_output_streams, uint32_t &num_active_streams, uint32_t &num_active_dram_queues, uint32_t &num_dram_prefetch_streams,
    dram_q_state_t *dram_q_state, dram_input_stream_state_t *dram_input_stream_state, dram_output_stream_state_t *dram_output_stream_state, active_stream_info_t *active_stream_info,
    volatile epoch_stream_info_t* *dram_prefetch_epoch_stream_info, volatile active_stream_info_t* *dram_prefetch_active_stream_info
) {
#ifdef PERF_DUMP
    risc::record_timestamp_at_offset(risc::perf_event::EPOCH, risc::EPOCH_START_OFFSET);
#endif
    if (!skip_initial_epoch_dram_load) {
#ifndef PERF_DUMP
        RISC_POST_STATUS(0x10000002);
#endif
        risc_epoch_load(dram_next_epoch_ptr);
#ifndef ERISC
        skip_kernels = EPOCH_INFO_PTR->skip_kernels;
        if (!skip_kernels)
          risc_kernels_load(dram_next_epoch_ptr);
#endif
    }

    EPOCH_INFO_PTR->end_program = 0;
	if (EPOCH_INFO_PTR->overlay_valid) {
#ifdef PERF_DUMP
        call_with_cpu_flush((void *)risc::init_perf_dram_state);
#endif
#ifdef ERISC
        init_ncrisc_streams();
#else
        call_with_cpu_flush((void *)init_ncrisc_streams);
#endif

#ifdef ERISC
        risc_dram_stream_handler_init_l1(
            0,
#ifdef RISC_GSYNC_ENABLED
            gsync_epoch, epochs_in_progress,
#endif
            num_dram_input_streams, num_dram_output_streams, num_active_streams, num_active_dram_queues, num_dram_prefetch_streams,
            dram_q_state, dram_input_stream_state, dram_output_stream_state, active_stream_info,
            dram_prefetch_epoch_stream_info, dram_prefetch_active_stream_info
        );
#else
        call_with_cpu_flush_args((void *)risc_dram_stream_handler_init_l1,
#ifdef RISC_GSYNC_ENABLED
          (void *) &gsync_epoch, (void *) &epochs_in_progress,
#endif
          (void *) &num_dram_input_streams, (void *) &num_dram_output_streams, (void *) &num_active_streams, (void *) &num_active_dram_queues, (void *) &num_dram_prefetch_streams,
          (void *) dram_q_state, (void *) dram_input_stream_state, (void *) dram_output_stream_state, (void *) active_stream_info,
          (void *) dram_prefetch_epoch_stream_info, (void *) dram_prefetch_active_stream_info
        );
#endif
#ifndef ERISC
        if (!skip_kernels)
          deassert_trisc_reset();
#endif
        risc_dram_stream_handler_loop(
#ifdef RISC_GSYNC_ENABLED
          gsync_epoch, epochs_in_progress,
#endif
          num_dram_input_streams, num_dram_output_streams, num_active_streams, num_active_dram_queues, num_dram_prefetch_streams,
          dram_q_state, dram_input_stream_state, dram_output_stream_state, active_stream_info,
          dram_prefetch_epoch_stream_info, dram_prefetch_active_stream_info
        );
#ifndef ERISC
        if (!skip_kernels)
          assert_trisc_reset();
#endif
        epoch_empty_check_cnt = 0;
#ifdef PERF_DUMP
        risc::record_timestamp_at_offset(risc::perf_event::EPOCH, risc::EPOCH_END_OFFSET);
        call_with_cpu_flush((void *)risc::record_perf_dump_end);
#endif
    }
}

void run_dram_queue_update(
    void * pFunction, volatile uint32_t *noc_read_scratch_buf, uint64_t& my_q_table_offset, uint32_t& my_q_rd_ptr, uint64_t& dram_next_epoch_ptr, uint8_t& loading_noc
) {
    epoch_queue::IOQueueUpdateCmdInfo queue_update_info;
    risc_get_epoch_update_info(queue_update_info, noc_read_scratch_buf, my_q_table_offset, my_q_rd_ptr, dram_next_epoch_ptr);
      
    if (queue_update_info.num_buffers > 1) {
        uint64_t dram_addr_offset;
        uint32_t dram_coord_x;
        uint32_t dram_coord_y;
        risc_get_noc_addr_from_dram_ptr_l1((volatile uint32_t *)(&(queue_update_info.queue_header_addr)), dram_addr_offset, dram_coord_x, dram_coord_y);
        uint64_t queue_addr_ptr = NOC_XY_ADDR(NOC_X(dram_coord_x), NOC_Y(dram_coord_y), dram_addr_offset);
        ncrisc_noc_fast_read_any_len_l1(loading_noc, NCRISC_RD_CMD_BUF, queue_addr_ptr, 
#ifdef ERISC
                                        eth_l1_mem::address_map::OVERLAY_BLOB_BASE, 
#else
                                        l1_mem::address_map::OVERLAY_BLOB_BASE, 
#endif
                                        queue_update_info.num_buffers*8);
    }

#ifdef ERISC
    uint32_t header_addr = eth_l1_mem::address_map::OVERLAY_BLOB_BASE + MAX_DRAM_QUEUES_TO_UPDATE*8 + MAX_DRAM_QUEUES_TO_UPDATE*sizeof(dram_io_state_t);
#else
    uint32_t header_addr = l1_mem::address_map::OVERLAY_BLOB_BASE + MAX_DRAM_QUEUES_TO_UPDATE*8 + MAX_DRAM_QUEUES_TO_UPDATE*sizeof(dram_io_state_t);
#endif   
    volatile uint32_t *header_addr_ptr = (volatile uint32_t *)header_addr;
    header_addr_ptr[0] = queue_update_info.header[0];
    header_addr_ptr[1] = queue_update_info.header[1];
    header_addr_ptr[2] = queue_update_info.header[2];
    header_addr_ptr[3] = queue_update_info.header[3];
    header_addr_ptr[4] = queue_update_info.header[4];
    header_addr_ptr[5] = 0;
    header_addr_ptr[6] = 0;
    header_addr_ptr[7] = 0;

    uint8_t state[MAX_DRAM_QUEUES_TO_UPDATE-1];
    for (uint32_t i = 0; i < queue_update_info.num_buffers; i++) {
      state[i] = 0;
    }

    bool all_done = false;
    while (!all_done) {
        all_done = true;

        while (!ncrisc_noc_reads_flushed_l1(loading_noc));
        while (!ncrisc_noc_nonposted_writes_flushed_l1(loading_noc));

        for (int k = 0; k < queue_update_info.num_buffers; k++) {
            if (state[k] == 4)
                continue;

            volatile uint32_t *queue_addr;
    #ifdef ERISC
            volatile uint64_t* queue_addr_l1 = (volatile uint64_t* )eth_l1_mem::address_map::OVERLAY_BLOB_BASE;
    #else
            volatile uint64_t* queue_addr_l1 = (volatile uint64_t* )l1_mem::address_map::OVERLAY_BLOB_BASE;
    #endif

            if (queue_update_info.num_buffers > 1) {
                queue_addr = (volatile uint32_t *)&queue_addr_l1[k];
            } else {
                queue_addr = (volatile uint32_t *)&(queue_update_info.queue_header_addr);
            }

            uint64_t dram_addr_offset;
            uint32_t dram_coord_x;
            uint32_t dram_coord_y;
            risc_get_noc_addr_from_dram_ptr_l1(queue_addr, dram_addr_offset, dram_coord_x, dram_coord_y);
            uint64_t queue_addr_ptr = NOC_XY_ADDR(NOC_X(dram_coord_x), NOC_Y(dram_coord_y), dram_addr_offset);

            uint32_t l1_ptr_addr = ((uint32_t)queue_addr_l1) + MAX_DRAM_QUEUES_TO_UPDATE*8 + k*sizeof(dram_io_state_t);
            volatile dram_io_state_t *l1_ptrs = (volatile dram_io_state_t *)l1_ptr_addr;

            if (state[k] == 0 || state[k] == 2) {
                RISC_POST_STATUS(0x11D00001);
                while (!ncrisc_noc_fast_read_ok_l1(loading_noc, NCRISC_RD_CMD_BUF));
                ncrisc_noc_fast_read_l1(loading_noc, NCRISC_RD_CMD_BUF, queue_addr_ptr, (uint32_t)l1_ptr_addr, DRAM_HEADER_SIZE);
                if (state[k] == 2)
                    state[k] = 3;
                else
                    state[k] = 1;
                all_done = false;
            } else if (state[k] == 1) {
                RISC_POST_STATUS(0x11D00002);
                uint32_t total_readers = queue_update_info.num_readers;
                bool has_multi_readers = total_readers > 1;
                uint32_t reader_index = queue_update_info.reader_index;
                uint32_t rd_stride = l1_ptrs->rd_queue_update_stride;

                if (!has_multi_readers || reader_index == rd_stride) {
                    if (!has_multi_readers || reader_index == total_readers-1) {
                        if (queue_update_info.update_mask == 0xff) {
                            while (!ncrisc_noc_fast_write_ok_l1(loading_noc, NCRISC_WR_REG_CMD_BUF));
                            ncrisc_noc_fast_write_l1(loading_noc, NCRISC_WR_REG_CMD_BUF, (uint32_t)(&(header_addr_ptr[0])), queue_addr_ptr, DRAM_HEADER_SIZE,
                                                     DRAM_PTR_UPDATE_VC, false, false, 1);
                        } else {
                            for (int m = 0; m < 8; m++) {
                                bool update_word = (queue_update_info.update_mask >> m) & 0x1;
                                if (update_word) {
                                    while (!ncrisc_noc_fast_write_ok_l1(loading_noc, NCRISC_WR_REG_CMD_BUF));
                                    ncrisc_noc_fast_write_l1(loading_noc, NCRISC_WR_REG_CMD_BUF, (uint32_t)(&(header_addr_ptr[m])), queue_addr_ptr + 4*m, 4,
                                                        DRAM_PTR_UPDATE_VC, false, false, 1);
                                }
                            }
                        }
                    }

                    if (has_multi_readers) {
                        if (reader_index == total_readers-1) {
                            l1_ptrs->wr_queue_update_stride = DRAM_STRIDE_WRAP_BIT + 0;
                            l1_ptrs->rd_queue_update_stride = DRAM_STRIDE_WRAP_BIT + 0;
                        } else {
                            l1_ptrs->wr_queue_update_stride = reader_index + 1;
                            l1_ptrs->rd_queue_update_stride = reader_index + 1;
                        }
                        // Reg poll loop, flushed immediately
                        while (!ncrisc_noc_fast_write_ok_l1(loading_noc, NCRISC_WR_REG_CMD_BUF));
                        ncrisc_noc_fast_write_l1(loading_noc, NCRISC_WR_REG_CMD_BUF, (uint32_t)(&(l1_ptrs->wr_queue_update_stride)), queue_addr_ptr+DRAM_BUF_QUEUE_UPDATE_STRIDE_OFFSET, 2,
                                              DRAM_PTR_UPDATE_VC, false, false, 1);
                    }

                    state[k] = 2;
                    all_done = false;
                } else {
                    state[k] = 0;
                    all_done = false;
                    continue;
                }
            } else if (state[k] == 3) {
                RISC_POST_STATUS(0x11D00003);
                uint32_t total_readers = queue_update_info.num_readers;
                bool has_multi_readers = total_readers > 1;
                uint32_t reader_index = queue_update_info.reader_index;
                uint32_t rd_stride = l1_ptrs->rd_queue_update_stride;

                if (!has_multi_readers || (reader_index + DRAM_STRIDE_WRAP_BIT) == rd_stride) {
                    if (has_multi_readers) {
                        if (reader_index == total_readers-1) {
                            l1_ptrs->wr_queue_update_stride = 0;
                            l1_ptrs->rd_queue_update_stride = 0;
                        } else {
                            l1_ptrs->wr_queue_update_stride = DRAM_STRIDE_WRAP_BIT + reader_index + 1;
                            l1_ptrs->rd_queue_update_stride = DRAM_STRIDE_WRAP_BIT + reader_index + 1;
                        }
                        // Reg poll loop, flushed immediately
                        while (!ncrisc_noc_fast_write_ok_l1(loading_noc, NCRISC_WR_REG_CMD_BUF));
                        ncrisc_noc_fast_write_l1(loading_noc, NCRISC_WR_REG_CMD_BUF, (uint32_t)(&(l1_ptrs->wr_queue_update_stride)), queue_addr_ptr+DRAM_BUF_QUEUE_UPDATE_STRIDE_OFFSET, 2,
                                              DRAM_PTR_UPDATE_VC, false, false, 1);
                    }

                    state[k] = 4;
                } else {
                    state[k] = 2;
                    all_done = false;
                    continue;
                }
            }
        }

    }

    while (!ncrisc_noc_reads_flushed_l1(loading_noc));
    while (!ncrisc_noc_nonposted_writes_flushed_l1(loading_noc));
}
