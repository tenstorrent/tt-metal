// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_custom_mm_compressed_common.h"
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_custom_mm_compressed_runtime.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// DRAMStreamingMatmulCompressed micro-op
//
// Computes: output[1,N] = in0[1,K] @ decompress(in1_compressed[K,N])
// with in1 streamed from DRAM in variable-size subblocks.
//
// CB States:
//   NCRISC: Streams compressed in1 from DRAM with variable-size reads per
//           subblock. Block sizes read from L1 metadata tensor.
//   BRISC: No-op
//   TRISC: Compressed matmul with per-tile format reconfig, subblock-K
//          streaming with partial accumulation.
// ============================================================================
struct DRAMStreamingMatmulCompressed {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC) - DRAM streaming with variable block sizes
    template <
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t in1_tensor_addr_,
        uint32_t subblock_k_,
        uint32_t per_core_n_,
        uint32_t out_num_tiles_,
        uint32_t num_subblocks_k_,
        uint32_t bank_id_,
        uint32_t vc_,
        uint32_t meta_l1_addr_,
        uint32_t cb_in1_size_bytes_,
        uint32_t noc_max_page_size_>
    struct ReaderCTArgs {
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t in1_tensor_addr = in1_tensor_addr_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t per_core_n = per_core_n_;
        static constexpr uint32_t out_num_tiles = out_num_tiles_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
        static constexpr uint32_t bank_id = bank_id_;
        static constexpr uint32_t vc = vc_;
        static constexpr uint32_t meta_l1_addr = meta_l1_addr_;
        static constexpr uint32_t cb_in1_size_bytes = cb_in1_size_bytes_;
        static constexpr uint32_t noc_max_page_size = noc_max_page_size_;
    };

    // Writer CTArgs (BRISC) - no-op
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC) - compressed matmul with streaming
    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t subblock_k_,
        uint32_t per_core_n_,
        uint32_t num_subblocks_k_,
        uint32_t fmt_l1_addr_>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t per_core_n = per_core_n_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
        static constexpr uint32_t fmt_l1_addr = fmt_l1_addr_;
    };

    // ========================================================================
    // Op - the actual operation
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool PopIn0 = true>
    class Op {
    public:
        void operator()() {
            if constexpr (IsActiveCore) {
                impl();
            }
        }

    private:
        void impl() {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC: Stream compressed in1 from DRAM with variable block sizes
            //
            // Metadata tensor layout (per core, uint32 array in L1):
            //   For each subblock: block_size_bytes (1 uint32)
            //   Total entries = num_subblocks_k * per_core_n
            //   Order: for n in per_core_n: for sb_k in num_subblocks_k
            //
            // Each subblock is read in chunks of <= noc_max_page_size.
            // ================================================================
            constexpr uint32_t num_iterations = CTArgs::num_subblocks_k * CTArgs::per_core_n;
            constexpr uint32_t dram_bank_id = CTArgs::bank_id;
            constexpr uint32_t vc = CTArgs::vc;
            constexpr uint32_t max_page_size = CTArgs::noc_max_page_size;

            const volatile uint32_t* meta_ptr = reinterpret_cast<const volatile uint32_t*>(CTArgs::meta_l1_addr);

            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

            uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, CTArgs::in1_tensor_addr);
            uint32_t l1_write_addr_in1;
            uint32_t dram_read_offset = 0;

            // Triple-buffering with transaction IDs
            constexpr uint32_t num_buffers = 3;
            constexpr uint32_t extra_blocks_in_flight = 1;
            uint32_t num_free_blocks_in_buffer = num_buffers;
            uint32_t curr_block_trid = 1;
            uint32_t block_trid_to_wait = 1;

            // Reserve initial space
            cb_reserve_back(CTArgs::cb_in1, CTArgs::subblock_k * (extra_blocks_in_flight + 1));
            l1_write_addr_in1 = get_write_ptr(CTArgs::cb_in1);

            uint32_t cb_in1_base = l1_write_addr_in1;
            uint32_t cb_in1_end = cb_in1_base + CTArgs::cb_in1_size_bytes;

            // Each CB slot is max_subblock_bytes, matching CB's page_size-based accounting.
            // Actual compressed data may be smaller, but we snap to slot boundaries.
            constexpr uint32_t max_subblock_bytes = CTArgs::cb_in1_size_bytes / num_buffers;

            for (uint32_t iter = 0; iter < num_iterations; ++iter) {
                uint32_t block_size = meta_ptr[iter];
                uint32_t slot_start = l1_write_addr_in1;

                noc_async_read_set_trid(curr_block_trid);

                uint32_t remaining = block_size;
                while (remaining > 0) {
                    uint32_t chunk = (remaining > max_page_size) ? max_page_size : remaining;
                    noc_async_read_one_packet_set_state<true>(in1_base_addr, chunk, vc);
                    noc_async_read_one_packet_with_state_with_trid(
                        in1_base_addr, dram_read_offset, l1_write_addr_in1, curr_block_trid);
                    dram_read_offset += chunk;
                    l1_write_addr_in1 += chunk;
                    remaining -= chunk;
                }

                // Advance to next CB slot boundary (CB thinks each tile is max_tile_size)
                l1_write_addr_in1 = slot_start + max_subblock_bytes;

                if (num_free_blocks_in_buffer == num_buffers - extra_blocks_in_flight) {
                    noc_async_read_barrier_with_trid(block_trid_to_wait);
                    cb_push_back(CTArgs::cb_in1, CTArgs::subblock_k);
                    block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
                    cb_reserve_back(CTArgs::cb_in1, CTArgs::subblock_k * (extra_blocks_in_flight + 1));
                } else {
                    num_free_blocks_in_buffer -= 1;
                }

                curr_block_trid = (curr_block_trid == num_buffers) ? 1 : (curr_block_trid + 1);

                // Wrap at CB boundary
                if (l1_write_addr_in1 >= cb_in1_end) {
                    l1_write_addr_in1 = cb_in1_base;
                }
            }

            // Push remaining blocks
            for (uint32_t i = 0; i < extra_blocks_in_flight; ++i) {
                noc_async_read_barrier_with_trid(block_trid_to_wait);
                cb_push_back(CTArgs::cb_in1, CTArgs::subblock_k);
                block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Compressed matmul with subblock-K streaming
            //
            // For each N output column:
            //   tile_regs_acquire()
            //   For each K subblock:
            //     cb_wait_front(cb_in1, subblock_k)  // wait for streamed data
            //     Run compressed matmul with per-tile format reconfig
            //     cb_pop_front(cb_in1, subblock_k)
            //   Pack output
            //
            // Format metadata: per-tile info in L1 at fmt_l1_addr.
            // Layout: per_core_n * num_subblocks_k groups of subblock_k tiles.
            // Each tile is [relative_offset:24 | fmt:8].
            // Relative offsets are within each subblock; the kernel adds addr_in1.
            // ================================================================
            constexpr uint32_t num_tiles_k = CTArgs::subblock_k * CTArgs::num_subblocks_k;
            constexpr bool split_acc = true;
            constexpr bool dense_packing = false;

            reconfig_data_format<false, true>(CTArgs::cb_in1, CTArgs::cb_in0);
            pack_reconfig_data_format<true>(CTArgs::cb_out);
            compressed::custom_mm_compressed_block_init_short<split_acc, dense_packing>(
                CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out);

            cb_wait_front(CTArgs::cb_in0, num_tiles_k);

            // Get in0 base address and tile size
            uint32_t addr_in0 = 0;
            uint32_t in0_face_r_dim = 0;
            uint32_t in0_tile_size = 0;
            UNPACK(({
                uint32_t in0_id = get_operand_id(CTArgs::cb_in0);
                addr_in0 = get_local_cb_interface(in0_id).fifo_rd_ptr - 1;
                in0_face_r_dim = get_operand_face_r_dim(in0_id);
                in0_tile_size = get_local_cb_interface(in0_id).fifo_page_size;
            }));

            // Per-tile metadata pointer
            const volatile uint32_t* fmt_base_ptr = reinterpret_cast<const volatile uint32_t*>(CTArgs::fmt_l1_addr);
            constexpr uint32_t tiles_per_subblock = CTArgs::subblock_k;

            uint32_t fmt_tile_offset = 0;

            for (uint32_t n = 0; n < CTArgs::per_core_n; n++) {
                cb_reserve_back(CTArgs::cb_out, 1);
                tile_regs_acquire();

                uint32_t addr_in0_subblock = addr_in0;

                for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k; sb_k++) {
                    cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);

                    // Get in1 read address from CB (changes per subblock due to triple-buffering)
                    uint32_t addr_in1 = 0;
                    UNPACK(({
                        uint32_t in1_id = get_operand_id(CTArgs::cb_in1);
                        addr_in1 = get_local_cb_interface(in1_id).fifo_rd_ptr - 1;
                    }));

                    const volatile uint32_t* fmt_ptr = fmt_base_ptr + fmt_tile_offset;
                    uint32_t fmt_addr = reinterpret_cast<uint32_t>(fmt_ptr);
                    if (sb_k < CTArgs::num_subblocks_k - 1) {
                        compressed::custom_mm_compressed_block_runtime<CTArgs::subblock_k, 1, false, true>(
                            fmt_addr, addr_in0_subblock, addr_in1, in0_face_r_dim, 0);
                    } else {
                        compressed::custom_mm_compressed_block_runtime<CTArgs::subblock_k, 1, true, true>(
                            fmt_addr, addr_in0_subblock, addr_in1, in0_face_r_dim, 0);
                    }

                    addr_in0_subblock += CTArgs::subblock_k * in0_tile_size;
                    fmt_tile_offset += tiles_per_subblock;
                    cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);
                }

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CTArgs::cb_out, 0);
                tile_regs_release();
                cb_push_back(CTArgs::cb_out, 1);
            }

            custom_mm_block_uninit<dense_packing>();

            if constexpr (PopIn0) {
                cb_pop_front(CTArgs::cb_in0, num_tiles_k);
            }
#endif
        }
    };  // class Op

};  // struct DRAMStreamingMatmulCompressed

}  // namespace deepseek_b1_ops
