# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Wormhole perf runs: which rows report the mean over the TILE_LOOP start phases.

BRISC releases the three threads at the TILE_LOOP park on fixed wall clock cycles (brisc.cpp serve), so every launch
starts them the same number of cycles apart. Each release resets the thread's branch table LFSR (period 31), so the two
threads released first meet the pipeline start at fixed LFSR phases, and some kernels' TILE_LOOP depends on them: on the
full suite, rows of the sources and run types in AVERAGED change by more than 0.01 % from one start phase to another
(in the build mode listed: speed-of-light kernels branch less and depend on it less). Those rows run once per phase of
PHASE_DESIGNS[design] and report the mean, with the spread over the phases in the std column; every other row runs at
the canonical phase (0, 0), the start every launch had before.

A phase (a, b) adds a cycles to the first release gap and b to the second, so the first thread's LFSR phase moves by
a + b and the second's by b (mod 31). The 31 x 31 grid of every pair is the phase average by definition. Every row but
fast_tilize's depends on the two phases additively, so the 31 phases (k, k), which give each thread each of its phases
once, have the grid's mean to the cycle; fast_tilize's L1_TO_L1 depends on the pair, so it runs the grid.
"""

import struct

# The start phase BRISC reads before the TILE_LOOP release (brisc.cpp .Lrel_phase): the runtime arguments' last 16 bytes
# (RUNTIME_ARGS, 1 KiB in memory.wormhole.ld), RELEASE_PHASE_VALID and then the cycles added to the two gaps.
RELEASE_PHASE_OFFSET = 0x400 - 16
RELEASE_PHASE_VALID = 0x6A900000
LFSR_PERIOD = 31

CANONICAL_PHASE = (0, 0)

PHASE_DESIGNS = {
    "latin": tuple((k, k) for k in range(LFSR_PERIOD)),
    "grid": tuple((a, b) for b in range(LFSR_PERIOD) for a in range(LFSR_PERIOD)),
}

# (source, run type) -> (design without --speed-of-light, design with it), None for the canonical start. From the release
# phase screen: every row of the suite at the 61 phases (a, 0) and (0, b) in both build modes; a source and run type is
# listed for a mode when any of its rows spreads by more than 0.01 % in that mode.
AVERAGED = {
    ("eltwise_binary_test.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("eltwise_unary_datacopy_test.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("eltwise_unary_sfpu_perf.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("eltwise_unary_typecast_perf.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("fast_tilize_test.cpp", "L1_TO_L1"): ("grid", "grid"),
    ("math_matmul_test.cpp", "L1_TO_L1"): ("latin", None),
    ("math_transpose_perf.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("matmul_test.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("pack_test.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("pack_untilize_perf.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("reduce_test.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("sfpu_binary_bcast_test.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("sfpu_binary_test.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("unpack_tilize_perf.cpp", "L1_TO_L1"): ("latin", "latin"),
    ("eltwise_unary_datacopy_test.cpp", "UNPACK_ISOLATE"): ("latin", "latin"),
    ("eltwise_unary_sfpu_perf.cpp", "UNPACK_ISOLATE"): ("latin", "latin"),
    ("pack_test.cpp", "UNPACK_ISOLATE"): ("latin", "latin"),
    ("sfpu_binary_bcast_test.cpp", "UNPACK_ISOLATE"): ("latin", "latin"),
    ("sfpu_binary_test.cpp", "UNPACK_ISOLATE"): ("latin", "latin"),
    ("sfpu_binop_scalar_perf.cpp", "UNPACK_ISOLATE"): ("latin", "latin"),
    ("sfpu_ternary_perf.cpp", "UNPACK_ISOLATE"): ("latin", "latin"),
    ("unpack_tilize_perf.cpp", "UNPACK_ISOLATE"): ("latin", None),
}


def release_phases(test_name: str, run_type_name: str, speed_of_light: bool) -> tuple:
    """The start phases a row runs at: its design's in this build mode when averaged, else the canonical one."""
    source = str(test_name).rsplit("/", 1)[-1]
    design = AVERAGED.get((source, run_type_name), (None, None))[bool(speed_of_light)]
    return PHASE_DESIGNS[design] if design else (CANONICAL_PHASE,)


def with_release_phase(runtime_arguments: bytes, phase) -> bytes:
    """The runtime arguments, padded to RELEASE_PHASE_OFFSET, then the start phase of the launch."""
    if len(runtime_arguments) > RELEASE_PHASE_OFFSET:
        raise ValueError(
            f"{len(runtime_arguments)} B of runtime arguments overlap the release phase"
        )
    a, b = phase
    return runtime_arguments.ljust(RELEASE_PHASE_OFFSET, b"\0") + struct.pack(
        "<Iiii", RELEASE_PHASE_VALID, a, b, 0
    )
