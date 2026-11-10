// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR: persistent sender reader through the engine-callable phase (mk_ar_send.hpp), MK_PS_SEND=hdr.
// Same compile args as mk_ps_reader.cpp.
#include <cstdint>
#define MKAR_S_CB_RS get_compile_time_arg_val(0)
#define MKAR_S_CB_AG get_compile_time_arg_val(1)
#define MKAR_S_IN_PAGE get_compile_time_arg_val(2)
#define MKAR_S_OUT_PAGE get_compile_time_arg_val(3)
#define MKAR_S_CHUNK_TILES get_compile_time_arg_val(4)
#define MKAR_S_BATCH get_compile_time_arg_val(5)
#define MKAR_S_SEM_AG_READY get_compile_time_arg_val(6)
#define MKAR_ITERS get_compile_time_arg_val(7)
#define MKAR_S_SEM_DONE get_compile_time_arg_val(8)
#define MKAR_S_NOUT get_compile_time_arg_val(9)
#define MKAR_S_PROG (get_compile_time_arg_val(10) != 0)
#define MKAR_S_SEM_READY get_compile_time_arg_val(11)
#define MKAR_S_SEM_GO get_compile_time_arg_val(12)
#define MKAR_S_ROWS get_compile_time_arg_val(13)
#define MKAR_S_INPK (get_compile_time_arg_val(15) != 0)
#define MKAR_S_READYP (get_compile_time_arg_val(16) != 0)
#define MKAR_S_READYP0 get_compile_time_arg_val(17)
#define MKAR_S_READYP1 get_compile_time_arg_val(18)
#define MKAR_S_READYP2 get_compile_time_arg_val(19)
#define MKAR_S_READYP3 get_compile_time_arg_val(20)
#define MKAR_S_READYP4 get_compile_time_arg_val(21)
#define MKAR_S_READYP5 get_compile_time_arg_val(22)
#define MKAR_S_READYP6 get_compile_time_arg_val(23)
#define MKAR_S_READYP7 get_compile_time_arg_val(24)
#include "mk_ar_send.hpp"

static mkar::SendReader rd;

void kernel_main() {
    rd.init(0, 0);
    for (uint32_t it = 0; it < MKAR_ITERS; ++it) {
        rd.iter(it);
    }
    rd.fini(MKAR_ITERS);
}
