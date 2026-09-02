// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Tensix side of the x280 echo demo.
//
// Protocol (see ../x280/fw.c): this kernel places a magic word in its own L1,
// posts a request {seq, my_x, my_y, l1_addr, value} into the x280 firmware's
// LIM mailbox with a plain NOC write, then polls its own L1 for the response.
// The x280 hart probe-reads the magic through a TLB window (reads are safe even
// if its view of our coordinates is wrong) and, on a match, window-writes
// value*3+7 into l1_addr+16 — the first x280-initiated NOC write into a Tensix.
void kernel_main() {
    uint32_t l1_buf = get_arg_val<uint32_t>(0);
    uint32_t dst_dram = get_arg_val<uint32_t>(1);
    uint32_t l2cpu_x = get_arg_val<uint32_t>(2);
    uint32_t l2cpu_y = get_arg_val<uint32_t>(3);
    uint32_t mbox = get_arg_val<uint32_t>(4);
    uint32_t seq = get_arg_val<uint32_t>(5);
    uint32_t value = get_arg_val<uint32_t>(6);
    uint32_t timeout_iters = get_arg_val<uint32_t>(7);

    constexpr uint32_t MAGIC = 0x5AFE0001;
    constexpr uint32_t out_size = 128;

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out_args, dst_dram, out_size);

    volatile tt_l1_ptr uint32_t* buf = (volatile tt_l1_ptr uint32_t*)l1_buf;
    buf[0] = MAGIC;  // probe target for the x280
    buf[4] = 0;      // +16: response slot the x280 writes

    // Request block, contiguous so one NOC write delivers it. The x280 wants raw
    // NOC0 coordinates: my_x/my_y hold this core's own NIU node id, which is
    // exactly that.
    volatile tt_l1_ptr uint32_t* req = (volatile tt_l1_ptr uint32_t*)(l1_buf + 64);
    req[0] = seq;
    req[1] = (uint32_t)my_x[0];
    req[2] = (uint32_t)my_y[0];
    req[3] = l1_buf;
    req[4] = value;
    noc_async_write(l1_buf + 64, get_noc_addr(l2cpu_x, l2cpu_y, mbox + 0x80), 20);
    noc_async_write_barrier();

    // Poll our own L1 for the x280's write. BH RISC L1 reads are cached, so the
    // cache must be invalidated each pass.
    uint32_t expected = value * 3 + 7;
    uint32_t found = 0;
    uint32_t it = 0;
    for (; it < timeout_iters; ++it) {
        invalidate_l1_cache();
        if (buf[4] == expected) {
            found = 1;
            break;
        }
    }

    // Pull the firmware's mailbox (heartbeat, state, traps, response block) for
    // diagnostics regardless of outcome.
    noc_async_read(get_noc_addr(l2cpu_x, l2cpu_y, mbox), l1_buf + 128, 96);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* out = (volatile tt_l1_ptr uint32_t*)(l1_buf + 256);
    out[0] = found;
    out[1] = buf[4];
    out[2] = it;
    out[3] = ((uint32_t)my_x[0]) | (((uint32_t)my_y[0]) << 16);
    volatile tt_l1_ptr uint32_t* mb = (volatile tt_l1_ptr uint32_t*)(l1_buf + 128);
    for (uint32_t i = 0; i < 24; i++) {
        out[4 + i] = mb[i];
    }
    noc_async_write_page(0, out0, l1_buf + 256);
    noc_async_write_barrier();
}
