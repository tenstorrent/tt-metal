# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Blackhole: the Welford state-save helpers must keep their values when the packer's ZEROACC of
# the other Dst half executes in the same cycle as one of their stores. A sweep of NOPs between
# the input copy and the save slides the save across that ZEROACC. See
# sources/welford_state_save_zeroacc_test.cpp.

import os
import random
import time

import numpy as np
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, Tilize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_INDEX,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)


@pytest.fixture(autouse=True)
def fresh_inputs():
    """The plugin reseeds torch with one fixed seed per test; a lost store could then
    read back a previous variant's identical value. Reseed so every variant has fresh inputs.
    """
    seed = (time.time_ns() ^ os.getpid()) & 0x7FFFFFFF
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    yield


def _raw_idx(offset):
    """Tile indices (face order) written by one 32-bit SFPSTORE at raw Dst offset: 4 rows, even columns."""
    out = []
    for r in range(4):
        face, face_row = divmod(offset + r, 16)
        for c in range(0, 16, 2):
            out.append(face * 256 + face_row * 16 + c)
    return out


P0, P4, P8, P12 = (_raw_idx(o) for o in (0, 4, 8, 12))

HELPERS = {"layernorm_state": 0, "group_state_and_anchor": 1}
NOPS = list(range(0, 49)) + [64, 96, 128]


@pytest.mark.parametrize("nops", NOPS, ids=[f"nops{x}" for x in NOPS])
@pytest.mark.parametrize("helpers", list(HELPERS))
def test_welford_state_save_zeroacc(helpers, nops):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("The ZEROACC row-valid loss is Blackhole behaviour")
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)
    input_dimensions = [256, 256]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    n = tile_cnt_A
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=None,
        test_config_kwargs={
            "test_name": "sources/welford_state_save_zeroacc_test.cpp",
            "formats": formats,
            "templates": [
                generate_input_dim(input_dimensions, input_dimensions),
                TILIZE(Tilize.No),
            ],
            "runtimes": [
                DEST_INDEX(nops | (HELPERS[helpers] << 12)),
                TILE_COUNT(n),
                NUM_FACES(4),
                NUM_BLOCKS(n),
                NUM_TILES_IN_BLOCK(1),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=n,
                tile_count_B=tile_cnt_B,
                tile_count_res=3 * n,
                num_faces=4,
            ),
            "dest_acc": DestAccumulation.Yes,
            "unpack_to_dest": False,
        },
    )
    got = torch.tensor(configuration.run().result, dtype=torch.float32).reshape(
        3 * n, 1024
    )
    inp = (
        src_A.to(torch.bfloat16)
        .to(torch.float32)
        .flatten()[: n * 1024]
        .reshape(n, 1024)
    )

    lost = []
    for b in range(n):
        mean_t, m2_t, anchor_t, x = got[3 * b], got[3 * b + 1], got[3 * b + 2], inp[b]
        if helpers == "layernorm_state":
            want = {
                "mean": (mean_t[P0], x[P0]),
                "m2": (m2_t[P0], x[P4] + x[P8]),
                "anchor": (mean_t[P4], x[P12]),
            }
        else:
            want = {
                "mean": (mean_t[P0], x[P0]),
                "m2": (m2_t[P0], x[P4]),
                "anchor": (anchor_t[P0], x[P12]),
            }
        for name, (g, w) in want.items():
            if ((g == 0) & (w != 0)).any():
                lost.append((b, name))
            assert torch.allclose(
                g[g != 0], w[g != 0], rtol=1e-2, atol=1e-3
            ), f"block {b}: {name} has wrong non-zero values"
    assert (
        not lost
    ), f"{len(lost)} saved vectors read back as zero (block, value): {lost[:16]}"
