// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Near partial combine, reader (one core per kv group g): the nsplit partial z, O and row-max tiles of group g.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t nsplit = get_named_compile_time_arg_val("nsplit");
    constexpr uint32_t cb_z = 0, cb_o = 1, cb_c = 2, dt = 4;
    const uint32_t zp = get_arg_val<uint32_t>(0);
    const uint32_t op = get_arg_val<uint32_t>(1);
    const uint32_t mp = get_arg_val<uint32_t>(2);
    const uint32_t g = get_arg_val<uint32_t>(3);

    cb_reserve_back(cb_z, nsplit);
    attnio::read_pages(zp, 4096, g * nsplit, nsplit, get_write_ptr(cb_z));
    cb_reserve_back(cb_o, nsplit * dt);
    attnio::read_pages(op, 4096, g * nsplit * dt, nsplit * dt, get_write_ptr(cb_o));
    cb_reserve_back(cb_c, nsplit);
    attnio::read_pages(mp, 4096, g * nsplit, nsplit, get_write_ptr(cb_c));
    noc_async_read_barrier();
    cb_push_back(cb_z, nsplit);
    cb_push_back(cb_o, nsplit * dt);
    cb_push_back(cb_c, nsplit);
}
