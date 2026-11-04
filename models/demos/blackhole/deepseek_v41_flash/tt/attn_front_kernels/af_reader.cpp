// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// attention-front POC, NCRISC: stream the two weights of the front chain (wqkv, then wq_b) from this core's DRAM bank
// into the ONE ring cb_w (blaze DRAMStreamingMatmul reader, reused as is).  The second stream starts right after the
// first one is fully pushed, so while the compute side runs the q/kv norm and the qr exchange the ring fills with the
// first tiles of wq_b (the prefetch that hides the norm latency). CT args: 0 cb_w, 1 bank_id, 2 vc, 3 w1 addr, 4 w2
// addr, 5.. geometry (see af python).

#include "models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t bank_id = get_compile_time_arg_val(1);
    constexpr uint32_t vc = get_compile_time_arg_val(2);
    constexpr uint32_t w1_addr = get_compile_time_arg_val(3);
    constexpr uint32_t w2_addr = get_compile_time_arg_val(4);
    constexpr uint32_t p1_page = get_compile_time_arg_val(5);
    constexpr uint32_t p1_npages = get_compile_time_arg_val(6);
    constexpr uint32_t p1_sbk = get_compile_time_arg_val(7);
    constexpr uint32_t p1_pn = get_compile_time_arg_val(8);
    constexpr uint32_t p1_blk = get_compile_time_arg_val(9);
    constexpr uint32_t p1_nsub = get_compile_time_arg_val(10);
    constexpr uint32_t p2_page = get_compile_time_arg_val(11);
    constexpr uint32_t p2_npages = get_compile_time_arg_val(12);
    constexpr uint32_t p2_sbk = get_compile_time_arg_val(13);
    constexpr uint32_t p2_pn = get_compile_time_arg_val(14);
    constexpr uint32_t p2_blk = get_compile_time_arg_val(15);
    constexpr uint32_t p2_nsub = get_compile_time_arg_val(16);
    constexpr uint32_t nb1 = get_compile_time_arg_val(17);
    constexpr uint32_t nb2 = get_compile_time_arg_val(18);
    constexpr uint32_t stage = get_compile_time_arg_val(19);
    constexpr uint32_t cb_unused = 31;

    using MM1 = deepseek_b1_ops::DRAMStreamingMatmul::
        ReaderCTArgs<cb_w, cb_unused, w1_addr, p1_page, p1_npages, p1_sbk, p1_pn, p1_blk, p1_pn, p1_nsub, bank_id, vc>;
    using MM2 = deepseek_b1_ops::DRAMStreamingMatmul::
        ReaderCTArgs<cb_w, cb_unused, w2_addr, p2_page, p2_npages, p2_sbk, p2_pn, p2_blk, p2_pn, p2_nsub, bank_id, vc>;
    deepseek_b1_ops::DRAMStreamingMatmul::Op<MM1, true, true, false, 0, false, false, nb1> mm1;
    {
        DeviceZoneScopedN("N_MM1");
        mm1();
    }
    if constexpr (stage >= 3) {
        deepseek_b1_ops::DRAMStreamingMatmul::Op<MM2, true, true, false, 0, false, false, nb2> mm2;
        {
            DeviceZoneScopedN("N_MM2");
            mm2();
        }
    }
}
