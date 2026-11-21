# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproducer: Blackhole SETADC thread override is ORed with the target thread's TDMA-range instruction.

The face-compressed matmul unpack thread writes the pack thread's unpacker-0 Y counter with
SETADC + ThreadOverride. In hardware, the ADC decoder hands the low 24 bits of *every* valid
opcode-0x40..0x68 instruction (PACR, SETDMAREG, ...) to its thread's ADC context, and the three
threads' arguments are ORed per context. If the pack thread presents such an instruction in the
same cycle, the override SETADC writes the OR of both payloads.

The shipped test never overlaps the two (the pack thread starts after math is done). Here the pack
thread packs throw-away tiles during the unpack phase:
  * test_overlap:      the override collides with pack-thread instructions -> wrong output (xfail, strict).
  * test_overlap_hold: same load, the pack thread holds mutex::THREAD2_ADC across each whole pack call
                       -> output matches the golden.
"""

import pytest
from conftest import blackhole_only
from helpers.compressed_utils import assign_random, run_compressed
from helpers.param_config import parametrize
from test_matmul_face_compressed import (
    COMPRESSION_GRANULARITY,
    SUPPORTED_FORMATS,
    SUPPORTED_M,
    encode_meta,
    pack_b,
    promote_assignment,
)

SHAPES = [(1, 7168, 64), (1, 1536, 128)]


def _run(shape, formats, source):
    M, K, N = shape
    assignment = assign_random(K, N, formats, COMPRESSION_GRANULARITY)
    run_compressed(
        M,
        K,
        N,
        assignment,
        source,
        COMPRESSION_GRANULARITY,
        SUPPORTED_M,
        SUPPORTED_FORMATS,
        promote_assignment,
        pack_b,
        encode_meta,
    )


@blackhole_only
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="SETADC ThreadOverride is ORed with the pack thread's same-cycle TDMA-range instruction",
)
@parametrize(shape=SHAPES, formats=[("bfp4", "bfp2")])
def test_overlap(shape, formats):
    _run(shape, formats, "sources/matmul_face_compressed_adc_override_test.cpp")


@blackhole_only
@parametrize(shape=SHAPES, formats=[("bfp4", "bfp2")])
def test_overlap_hold(shape, formats):
    _run(shape, formats, "sources/matmul_face_compressed_adc_override_hold_test.cpp")
