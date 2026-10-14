# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Wormhole perf runs without --speed-of-light: which tests share one INIT measurement build.

The INIT measurement build (counters.h LLK_PERF_INIT_ONLY) compiles the speed-of-light header in every run: formats and
parameters are constants, as tt-metal compiles a kernel, so INIT's code does not branch on them at run time. Built with
runtime formats and parameters, INIT takes 2-5x more branches, and a code change of k instructions moves it by k +- a few
cycles, depending on where those branch targets fall in the cold icache.

One such build per runtime configuration would add a compile per test. Instead, tests whose speed-of-light INIT takes the
same cycles share one: the sharing key is the test's speed-of-light header without the lines that declare
INIT_NEUTRAL[source] names, plus every other compilation input. The first test of a key builds the ELFs from its own
header, so each test measures a speed-of-light INIT whose cycles equal those of its own.

A name is listed only if leaving it out shares builds, and if, on the full Wormhole perf suite, every key group then
holds only tests whose own speed-of-light INIT builds execute the same INIT window in every thread (Tensix instruction
words in full, the same addresses and length; only RISC immediates may differ) and take the same cycles. A constant can
change the code without changing a branch: eltwise binary's input formats make two unpack config words coincide or not,
and GCC then materializes the value once or twice; so formats, tile sizes and the binary op are not neutral there.
Sources not listed share nothing, so a new test or parameter costs a build per configuration until its names are checked
the same way, and a change to the table needs that check again (test_test_config.py pins the checked table).
"""

import re
from hashlib import sha256

# Per test source, the header names whose values change only constants of the speed-of-light INIT, not its cycles: tile,
# block and matmul dims, tile counts and sizes, operand addresses, loop counts, input formats and some op variants.
INIT_NEUTRAL = {
    "eltwise_bcast_col_custom_perf.cpp": frozenset(
        {"CT_DIM", "TILE_CNT", "buffer_B", "buffer_Res"}
    ),
    "eltwise_binary_test.cpp": frozenset(
        {
            "ACC_TO_DEST",
            "BLOCK_CT_DIM",
            "BLOCK_RT_DIM",
            "FULL_CT_DIM",
            "FULL_RT_DIM",
            "INPUT_NUM_BLOCKS",
            "INPUT_NUM_TILES_IN_BLOCK",
            "NUM_BLOCKS",
            "NUM_TILES_IN_BLOCK",
            "OUTPUT_NUM_BLOCKS",
            "OUTPUT_NUM_TILES_IN_BLOCK",
            "REUSE_DEST_TYPE",
            "UNPACK_A_OUT",
            "UNPACK_B_OUT",
            "UNPACK_S_OUT",
            "buffer_A",
            "buffer_B",
            "buffer_Res",
        }
    ),
    "eltwise_unary_datacopy_test.cpp": frozenset(
        {
            "BLOCK_CT_DIM",
            "BLOCK_RT_DIM",
            "FULL_CT_DIM",
            "FULL_RT_DIM",
            "INPUT_NUM_BLOCKS",
            "INPUT_NUM_TILES_IN_BLOCK",
            "NUM_BLOCKS",
            "NUM_TILES_IN_BLOCK",
            "OUTPUT_NUM_BLOCKS",
            "OUTPUT_NUM_TILES_IN_BLOCK",
            "TILE_CNT",
            "TILE_SIZE_UNPACK_A",
            "TILE_SIZE_UNPACK_B",
            "UNPACK_A_IN",
            "UNPACK_A_OUT",
            "UNPACK_B_IN",
            "UNPACK_B_OUT",
            "UNPACK_S_IN",
            "UNPACK_S_OUT",
            "buffer_A",
            "buffer_B",
            "buffer_Res",
        }
    ),
    "eltwise_unary_sfpu_perf.cpp": frozenset(
        {
            "FAST_MODE",
            "STABLE_SORT",
            "TILE_SIZE_UNPACK_A",
            "TILE_SIZE_UNPACK_B",
            "UNPACK_A_IN",
            "UNPACK_B_IN",
            "UNPACK_S_IN",
            "buffer_A",
            "buffer_B",
            "buffer_Res",
        }
    ),
    "fast_tilize_test.cpp": frozenset(
        {"BLOCK_RT_DIM", "FULL_RT_DIM", "TILE_CNT", "buffer_B", "buffer_Res"}
    ),
    "math_matmul_test.cpp": frozenset(
        {
            "CT_DIM",
            "INPUT_NUM_BLOCKS",
            "INPUT_NUM_TILES_IN_BLOCK",
            "KT_DIM",
            "NUM_BLOCKS",
            "NUM_TILES_IN_BLOCK",
            "OUTPUT_NUM_BLOCKS",
            "OUTPUT_NUM_TILES_IN_BLOCK",
            "RT_DIM",
            "TILE_CNT",
            "UNPACK_TRANSPOSE_FACES",
            "UNPACK_TRANSPOSE_WITHIN_FACE",
            "buffer_B",
            "buffer_Res",
        }
    ),
    "math_transpose_perf.cpp": frozenset({"MATH_TRANSPOSE_FACES"}),
    "matmul_test.cpp": frozenset(
        {
            "CT_DIM",
            "KT_DIM",
            "RT_DIM",
            "TILE_CNT",
            "TILE_SIZE_UNPACK_A",
            "TILE_SIZE_UNPACK_B",
            "UNPACK_A_IN",
            "UNPACK_B_IN",
            "UNPACK_S_IN",
            "buffer_A",
            "buffer_B",
            "buffer_Res",
        }
    ),
    "pack_dest_bank_perf.cpp": frozenset({"LOOP_FACTOR"}),
    "pack_test.cpp": frozenset(
        {
            "BLOCK_CT_DIM",
            "BLOCK_RT_DIM",
            "FULL_CT_DIM",
            "FULL_RT_DIM",
            "INPUT_NUM_TILES_IN_BLOCK",
            "NUM_TILES_IN_BLOCK",
            "OUTPUT_NUM_TILES_IN_BLOCK",
            "TILE_CNT",
            "TILE_SIZE_UNPACK_A",
            "TILE_SIZE_UNPACK_B",
            "UNPACK_A_IN",
            "UNPACK_B_IN",
            "UNPACK_S_IN",
            "buffer_A",
            "buffer_B",
            "buffer_Res",
        }
    ),
    "pack_untilize_perf.cpp": frozenset(
        {"BLOCK_RT_DIM", "FULL_RT_DIM", "TILE_CNT", "buffer_B", "buffer_Res"}
    ),
    "reduce_perf.cpp": frozenset(
        {
            "TILE_SIZE_UNPACK_A",
            "TILE_SIZE_UNPACK_B",
            "UNPACK_A_IN",
            "UNPACK_B_IN",
            "UNPACK_S_IN",
            "buffer_A",
            "buffer_B",
            "buffer_Res",
        }
    ),
    "sfpu_binary_bcast_test.cpp": frozenset({"SFPU_BINARY_OPERATION"}),
    "sfpu_binary_test.cpp": frozenset({"APPROX_MODE"}),
    "sfpu_binop_scalar_perf.cpp": frozenset({"SFPU_BINOP_MODE"}),
    "sfpu_reduce_row_max_perf.cpp": frozenset({"LOOP_FACTOR"}),
    "sfpu_reduce_sdpa_perf.cpp": frozenset({"LOOP_FACTOR"}),
    "unpack_a_bcast_eltwise_perf.cpp": frozenset(
        {"ELTWISE_BINARY_OP", "SRCA_REUSE_COUNT", "TILE_CNT", "buffer_B", "buffer_Res"}
    ),
    "unpack_tilize_perf.cpp": frozenset(
        {"BLOCK_RT_DIM", "FULL_RT_DIM", "TILE_CNT", "buffer_B", "buffer_Res"}
    ),
    "unpack_transpose_perf.cpp": frozenset({"UNPACK_TRANSPOSE_WITHIN_FACE"}),
}

# the name a header line declares: "#define NAME", "constexpr <type> NAME = ...", "constexpr Operand NAME(...)"
_DECLARED = re.compile(r"^\s*(?:#define\s+(\w+)|constexpr\s+.*?\b(\w+)\s*(?:=|\(|\{))")


def declared_name(line: str):
    """The name a build.h line declares, or None"""
    m = _DECLARED.match(line)
    return (m.group(1) or m.group(2)) if m else None


def init_share_key(source: str, inputs: list, header: str) -> str:
    """Key of the INIT measurement build a test takes: its source (the driver), its other compilation inputs and its
    speed-of-light header without the lines that declare INIT_NEUTRAL[source] names"""
    neutral = INIT_NEUTRAL.get(source.rsplit("/", 1)[-1], frozenset())
    lines = [l for l in header.split("\n") if declared_name(l) not in neutral]
    return sha256(
        " | ".join([source, *inputs, "<<speed-of-light header>>", *lines]).encode()
    ).hexdigest()
