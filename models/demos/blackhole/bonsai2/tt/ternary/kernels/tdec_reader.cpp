// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear, reader (NCRISC): the even output-tile records of this core into c_1 (tdec_stream.hpp).
#include "tdec_stream.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_nt = get_arg_val<uint32_t>(1);
    const uint32_t num_nt = get_arg_val<uint32_t>(2);
    constexpr uint32_t ppr = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr auto w_args = TensorAccessorArgs<2>();
    const auto w = TensorAccessor(w_args, w_addr, 4096);
    stream_records<ppr, NB>(tt::CBIndex::c_1, w, first_nt, num_nt, 0);
}
