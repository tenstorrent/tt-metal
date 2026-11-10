// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR step 3 (oneshot_ar.py OneShotARP): persistent AR_RECV on the output (owner) cores, ITERS all-reduces in
// ONE program. The per-AR body is the engine-callable phase in mk_ar_recv.hpp (mkar::Recv); this kernel only maps the
// compile-time args onto its config macros and loops.
//   owner : residual -> CB, local partial -> slot my_chip, wait the ring - 1 reduce-scatter increments, hand the slots
//           to compute, wait its reduced chunk, tell this link's senders (AG ready), deliver x / Sum(x^2), DONE;
//   gather: wait the owner's all-gather increment, publish the chunk on the output CB, deliver, DONE.
#include <cstdint>
#define MKAR_CB_RED get_compile_time_arg_val(0)
#define MKAR_CB_OUT get_compile_time_arg_val(1)
#define MKAR_FUSE_RES (get_compile_time_arg_val(2) != 0)
#define MKAR_CB_RES get_compile_time_arg_val(3)
#define MKAR_RES_TILES get_compile_time_arg_val(4)
#define MKAR_CHUNK_TILES get_compile_time_arg_val(5)
#define MKAR_IN_PAGE get_compile_time_arg_val(6)
#define MKAR_RING get_compile_time_arg_val(7)
#define MKAR_SEM_AG_READY get_compile_time_arg_val(8)
#define MKAR_ITERS get_compile_time_arg_val(9)
#define MKAR_SEM_DONE get_compile_time_arg_val(10)
#define MKAR_PROG (get_compile_time_arg_val(11) != 0)
#define MKAR_SEM_READY get_compile_time_arg_val(12)
#define MKAR_ROWS get_compile_time_arg_val(13)
#define MKAR_DELIVER get_compile_time_arg_val(14)
#define MKAR_STATS (get_compile_time_arg_val(15) != 0)
#define MKAR_SRED (get_compile_time_arg_val(16) != 0)
#define MKAR_AGD (get_compile_time_arg_val(17) != 0)
#define MKAR_INPK (get_compile_time_arg_val(18) != 0)
#define MKAR_CB_PART get_compile_time_arg_val(19)
#define MKAR_CB_SCALER get_compile_time_arg_val(20)
#define MKAR_CB_STAGE get_compile_time_arg_val(21)
#if defined(MK_PS_GSEM) && defined(MK_PS_EPOCH)
#define MKAR_SEM_BASE get_compile_time_arg_val(22)
#endif
#include "mk_ar_recv.hpp"

static mkar::Recv r;  // file scope: the parsed AR state lives in .bss, not on the (small) NCRISC stack

void kernel_main() {
    if (!r.init(0, 0)) {
        return;
    }
    for (uint32_t it = 0; it < MKAR_ITERS; ++it) {
        r.iter(it);
    }
    r.fini();
}
