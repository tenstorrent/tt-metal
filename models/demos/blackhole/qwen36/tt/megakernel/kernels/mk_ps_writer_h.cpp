// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR: persistent sender writer through the engine-callable phase (mk_ar_sendw.hpp), MK_PS_SEND=hdr.
// Same compile args as mk_ps_writer.cpp (HDR_ROT / SRED / AGD not supported here).
#include <cstdint>
#define MKAR_S_MY_CHIP get_compile_time_arg_val(0)
#define MKAR_S_CB_HDR get_compile_time_arg_val(1)
#define MKAR_S_CB_RS get_compile_time_arg_val(2)
#define MKAR_S_CB_AG get_compile_time_arg_val(3)
#define MKAR_S_IN_PAGE get_compile_time_arg_val(4)
#define MKAR_S_OUT_PAGE get_compile_time_arg_val(5)
#define MKAR_S_CHUNK_TILES get_compile_time_arg_val(6)
#define MKAR_S_IN_PPP get_compile_time_arg_val(7)
#define MKAR_S_OUT_PPP get_compile_time_arg_val(8)
#define MKAR_S_CB_RED get_compile_time_arg_val(9)
#define MKAR_ITERS get_compile_time_arg_val(10)
#define MKAR_S_PROG (get_compile_time_arg_val(11) != 0)
// reader-only macros (unused by the writer, defined so mk_ar_send.hpp compiles)
#define MKAR_S_BATCH 1
#define MKAR_S_SEM_AG_READY 0
#define MKAR_S_SEM_DONE 0
#define MKAR_S_SEM_READY 0
#define MKAR_S_SEM_GO 0
#define MKAR_S_NOUT 0
#define MKAR_S_ROWS 0
#define MKAR_S_INPK 0
#define MKAR_S_READYP 0
#define MKAR_S_READYP0 0
#define MKAR_S_READYP1 0
#define MKAR_S_READYP2 0
#define MKAR_S_READYP3 0
#define MKAR_S_READYP4 0
#define MKAR_S_READYP5 0
#define MKAR_S_READYP6 0
#define MKAR_S_READYP7 0
#include "mk_ar_sendw.hpp"

void kernel_main() {
    // local, not file-scope: FabricConnectionManager is not constant-initialisable (no dynamic init of statics); in the
    // engine the object lives in kernel_main's scope for the whole pass as well
    mkar::SendWriter wr;
    wr.init(0, 0);
    for (uint32_t it = 0; it < MKAR_ITERS; ++it) {
        wr.iter(it);
    }
    wr.fini();
}
