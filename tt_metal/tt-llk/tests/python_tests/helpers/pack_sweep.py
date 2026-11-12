# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The arms of the pack sweeps (perf_pack_ctx.py, perf_pack_block_sweep*.py, perf_pack_mmsweep.py, the matmul pack sweeps) as
pack_api_sweep.h runs them: production's pack_block, pack init and reconfigure through pack_api_shim.h, one ELF per arm.

An arm keeps the sweeps' (code, form) id. code & 3 the state: 0 steady, 1 a pack init before every block, 2 outputs A and B
alternating with a reconfigure. code & 4: main's per-tile loop for both outputs. code & 8 (or 16): A takes W in a CB of two
blocks (or one), so llk_pack_w_take continues or programs the base by itself. form in states 0 and 1: 0 per tile, 1 block,
2 W, 3 main, 5 the form tables (production's own choice); in state 2 the forms of A and B: 0 T/T, 1 B/T, 2 B/B, 3 T/B, 4 W/T,
5 the tables for both.
"""

from .data_format_inference import data_formats, infer_pack_in
from .format_config import DataFormat
from .test_config import TestConfig
from .test_variant_parameters import PACK_API_SWEEP

SHIM_CONFIG = {
    "compile_time_formats": True,
    "kernel_flags": ["-flto=1", "-flto-partition=one"],
}
STATE2_FORMS = {
    0: ("T", "T"),
    1: ("B", "T"),
    2: ("B", "B"),
    3: ("T", "B"),
    4: ("W", "T"),
    5: ("0", "0"),
}
FORMS = {0: "T", 1: "B", 2: "W", 3: "M", 5: "0"}
# Arms the farm sweeps did not have: W after an init, A's W after a reconfigure, W in a CB of one block, the tables steady.
NEW_ARMS = [(1, 2), (2, 4), (16, 2), (0, 5)]


def arm_fields(arm):
    """(state, form of A, form of B, A's CB depth in blocks) of an arm id."""
    code, form = arm
    state = code & 3
    if code & 4:
        return state, "M", "M", 0
    if code & 8:
        return state, "W", "T", 2
    if code & 16:
        return state, "W", "T", 1
    if state == 2:
        a, b = STATE2_FORMS[form]
        return state, a, b, 0
    return state, FORMS[form], "T", 0


def sweep_templates(
    arm,
    tiles,
    faces=4,
    face_r=16,
    kt=0,
    matmul=False,
    sfpu=False,
    format_b=DataFormat.Float16_b,
    dst=0,
    fidelity=None,
):
    """The PACK_API_SWEEP template of an arm with tiles per block (pack_block<tiles>), the output tile and the compile-time DEST start;
    fidelity for a test without a MATH_FIDELITY template. It adds no CSV column: the arm stays in dst_index (state << 12 | form << 8).
    """
    state, a, b, cbd = arm_fields(arm)
    return [
        PACK_API_SWEEP(
            a,
            tiles,
            kt,
            matmul,
            sfpu,
            dst,
            state,
            b,
            cbd,
            format_b,
            faces,
            face_r,
            fidelity,
        )
    ]


def bind_pack_src_b(kwargs):
    """Output B's packer input in a state 2 arm, as the JIT's pack_src_format of a second output CB: A's DEST format packed to
    B's format with the test's inference (with a 32 bit DEST that is B's format, not A's). Returns kwargs.
    """
    templates = kwargs["templates"]
    from_output = any(getattr(t, "pack_src_from_output", False) for t in templates)
    for t in templates:
        if isinstance(t, PACK_API_SWEEP) and t.state == 2:
            f, acc = kwargs["formats"], kwargs["dest_acc"]
            to_dest = kwargs.get("unpack_to_dest", False)
            a = data_formats(
                input_format=f.input_format,
                output_format=f.output_format,
                is_fp32_dest_acc_en=acc,
                num_iterations=1,
                unpacking_to_dest=to_dest,
                chip_arch=TestConfig.CHIP_ARCH,
                input_format_B=getattr(f, "input_format_B", None),
                pack_src_from_output=from_output,
            )[0]
            t.pack_src_b = infer_pack_in(
                f.input_format,
                t.format_b,
                a.math,
                acc,
                to_dest,
                TestConfig.CHIP_ARCH,
                pack_src_from_output=from_output,
            )
    return kwargs
