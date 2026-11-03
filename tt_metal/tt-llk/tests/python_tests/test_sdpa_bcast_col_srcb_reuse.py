# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers experimental LLK sdpa_bcast_col_srcb_reuse (+ unpack_A_sdpa) (tt-metal#47554 /
# tt-blaze#1971), promoted into tt_llk_blackhole/llk_lib/experimental/ on main by #53295. The compute kernel includes
# the canonical headers; the demo-fork shadow tree this test was first written against no longer exists.
#
# sdpa_bcast_col_srcb_reuse documented contract
# (llk_math_sdpa_bcast_col_srcb_reuse.h / llk_unpack_A_sdpa.h header banners, plus what the op was measured to do
# on p100a -- see helpers/sdpa_bcast_utils.py for the measurement):
#   - eltwise ADD/SUB/MUL of a per-tile operand (SrcA) with a *column* broadcast (SrcB), where the column source is
#     DEST reused as a source register (DEST -> SrcB via MOVD2B), reused across every SrcA row. This is the softmax
#     scale / normalize step; the MUL path additionally supports high fidelity.
#   - num_faces: the init helper LLK_ASSERTs {1, 2, 4}, but sdpa_bcast_col_srcb_reuse_configure_mop hard-asserts == 2.
#     Combined with the MOP writing 8 dest rows per face, the only shape it instantiates is an 8x32 tile (two 8x16
#     faces) -- the demo's tile as well ("Each tile is 8x32, which is the same as a full 16x16 face", sdpa.h:317).
#   - It is a TWO-operand-group op. The execute runs the MOP twice and every eltwise op carries CLR_A, so it
#     retires 2 * num_tiles * num_faces SrcA dvalids; the demo pairs it with two llk_unpack_A calls per math call
#     (compute_kernel_api/sdpa.h:56-57) and computes cb_l1 * P1 + cb_l2 * P2. Feeding one operand group stalls MATH
#     forever on the second MOP run -- that was the hang this test used to hit.
#   - num_tiles is the MOP's outer-loop count: one llk_unpack_A call walks that many tiles and the math writes
#     DEST[t] = (A_t op P1) + (A_{num_tiles + t} op P2); num_tiles=2 is what reaches the MOP's last_inner_loop slot.
#   - The two column sources are DEST[src_index] and DEST[src_index + 1]: the preamble MOVD2Bs DEST rows 0-7 into
#     SrcB rows 0-7 (P1) and DEST rows 64-71 -- the top of the next 32x32 dest tile -- into SrcB rows 8-15 (P2).
#   - Unpack side is the base llk_unpack_A execute paired with the _llk_unpack_A_sdpa_init_ MOP config, plus
#     _llk_unpack_A_sdpa_set_srcb_dummy_valid_() which injects the dummy SrcB SET_DVALID the math preamble's
#     STALLWAIT(SRCB_VLD) waits on before its MOVD2B. That helper must be issued BEFORE the operand unpacks.
#
# This advance test exercises the MUL (softmax-scale), ADD and SUB instantiations, LoFi, on 8x32 tiles, at
# num_tiles 1 and 2. Every op accumulates both MOP passes into dest, giving (A0 op P1) + (A1 op P2).
#
# Blackhole-only (@blackhole_only): the primitive headers live under the Blackhole experimental/ tree.

from conftest import blackhole_only
from helpers.device import BootMode
from helpers.llk_params import MathOperation
from helpers.param_config import parametrize
from helpers.sdpa_bcast_utils import (
    SDPA_BCAST_FORMATS,
    SDPA_BCAST_TILE_COUNTS,
    run_sdpa_bcast_col_srcb_reuse,
)


@blackhole_only
@parametrize(
    formats=SDPA_BCAST_FORMATS,
    mathop=[MathOperation.Elwmul, MathOperation.Elwadd, MathOperation.Elwsub],
    num_tiles=SDPA_BCAST_TILE_COUNTS,
)
def test_sdpa_bcast_col_srcb_reuse(
    formats,
    mathop,
    num_tiles,
    boot_mode=BootMode.DEFAULT,
):
    run_sdpa_bcast_col_srcb_reuse(
        "sources/sdpa_bcast_col_srcb_reuse_test.cpp",
        formats,
        mathop,
        num_tiles=num_tiles,
        boot_mode=boot_mode,
    )
