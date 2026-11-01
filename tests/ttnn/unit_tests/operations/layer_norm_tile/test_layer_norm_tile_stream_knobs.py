# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the reader/writer stream knobs away from their shipped settings.

The knobs are module constants in layer_norm_tile_program_descriptor; each case overrides them so
the trid ring, grouped writer flush, bank stagger and split_reader paths stay covered even while
parked. (7, 1, 3200, W) gives several blocks per core with a ragged last block and in_depth = 2.
"""

import pytest

import ttnn.operations.layer_norm_tile.layer_norm_tile_program_descriptor as descriptor
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_layer_norm_tile import DRAM, L1, W, run_and_check

KNOBS = {
    "trivial": dict(READ_TRID_DEPTH=1, WRITE_CHUNKS_PER_FLUSH=1, BANK_STAGGER=False, RESIDUAL_ON_WRITER=False),
    "trid3_flush3_stagger": dict(
        READ_TRID_DEPTH=3, WRITE_CHUNKS_PER_FLUSH=3, BANK_STAGGER=True, RESIDUAL_ON_WRITER=False
    ),
    "split_trid3_flush3_stagger": dict(
        READ_TRID_DEPTH=3, WRITE_CHUNKS_PER_FLUSH=3, BANK_STAGGER=True, RESIDUAL_ON_WRITER=True
    ),
    "split_trid15_flush15": dict(READ_TRID_DEPTH=15, WRITE_CHUNKS_PER_FLUSH=15, RESIDUAL_ON_WRITER=True),
}

CASES = [
    pytest.param((8, 1, 512, W), DRAM, True, id="8x512_dram_residual"),
    pytest.param((8, 1, 512, W), L1, True, id="8x512_l1_residual"),
    pytest.param((7, 1, 3200, W), DRAM, True, id="7x3200_dram_residual_multi_block"),
    pytest.param((7, 1, 3200, W), DRAM, False, id="7x3200_dram_no_residual_multi_block"),
    pytest.param((3, 1, 500, W), DRAM, True, id="3x500_h_non_aligned"),
]


@pytest.mark.parametrize("knobs", list(KNOBS), ids=list(KNOBS))
@pytest.mark.parametrize("shape,buf,with_residual", CASES)
def test_stream_knobs(device, monkeypatch, knobs, shape, buf, with_residual):
    for name, value in KNOBS[knobs].items():
        monkeypatch.setattr(descriptor, name, value)
    run_and_check(device, shape, with_residual=with_residual, input_buf=buf, output_buf=buf)
