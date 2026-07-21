# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reconfig-escape / empty-``uninit`` sweep for the experimental LLKs (BH + WH B0).

Several experimental LLKs have an empty or no-op ``uninit`` even though their
``init`` clobbers persistent HW state (ADDR_MODs, cfg regs such as
``THCON_SEC0_REG2_Haloize_mode``). This is the #1 contract hole in the
experimental-LLK inventory: state could leak into the following op.

Each parametrized case runs a two-op sequence:

    run 0 (POLLUTER): the selected experimental op (init + one op + its empty
                      uninit). Its packed output is discarded to a scratch buffer.
    run 1 (VICTIM):   a canonical datacopy (A2D) of a fresh, known input tile,
                      packed to the result buffer and validated against its input.

The victim datacopy runs ONLY its own standard init (no extra reinit help). A
green result **pins** the invariant that the experimental op's empty ``uninit``
leaves no escape the canonical op's own init doesn't already re-establish:

  * The MATH polluters clobber ADDR_MODs; the datacopy init reprograms exactly
    the ADDR_MODs it consumes (ADDR_MOD_0/2/3), so those clobbers cannot corrupt
    the copy.
  * The UNPACK polluter (``sub_bcast_col``) RMWs ``Haloize_mode`` (transpose
    within face); the victim's ``_llk_unpack_A_init_`` rewrites ``Haloize_mode``,
    so the transpose bit cannot leak.

If any case ever fails, that is a genuine reconfig escape (a polluter left state
that the canonical op does not reset) — a real finding to document, not to mask.
"""

from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import MATH_FIDELITY, TemplateParameter
from helpers.utils import passed_test

# 32x32 tile: 4 faces of 16x16.
ELEMENTS_PER_TILE = 1024
TILE_DIMENSIONS = [32, 32]

# Experimental polluter selector (matches the POLLUTER switch in the CPP kernel).
# Scenario 2 runs the paired SDPA fused sub+bcast-col op, which exercises BOTH the
# eltwise_binary_custom (MATH) and unpack_AB_sub_bcast_col_custom (UNPACK) empty uninits
# at once — they only run correctly as a pair.
POLLUTERS = {
    0: "matmul_custom_no_mop",
    1: "reduce_block_max_row",
    2: "eltwise_binary_custom + unpack_AB_sub_bcast_col_custom (SDPA sub+bcast-col)",
}


@dataclass
class POLLUTER(TemplateParameter):
    """Selects which experimental op runs as the run-0 polluter."""

    polluter: int = 0

    def convert_to_cpp(self) -> str:
        # A #define (not a constexpr) so the CPP kernel can switch on it with #if.
        return f"#define POLLUTER {self.polluter}"


@parametrize(
    formats=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    # Both fidelities: the matmul_no_mop polluter programs different fidelity-phase
    # ADDR_MODs (ADDR_MOD_5/6) at HiFi vs LoFi, so sweeping catches a wider ADDR_MOD set.
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    polluter=list(POLLUTERS.keys()),
)
def test_experimental_reconfig_escape(formats, math_fidelity, polluter):
    if get_chip_architecture() == ChipArchitecture.QUASAR:
        pytest.skip(
            "experimental reconfig-escape sweep targets Blackhole and Wormhole B0"
        )

    # bf16 DEST throughout (dest_acc=No): the escape under test is the named
    # ADDR_MOD / Haloize contract hole, independent of the fp32 zero-flag path.
    dest_acc = DestAccumulation.No

    # buffer_A[0]/buffer_B[0] feed the discarded polluter; buffer_A[1] is the
    # fresh, known tile the victim datacopy reads (and must reproduce). Two A
    # tiles + one B tile are laid out contiguously by StimuliConfig.
    input_dimensions = [2 * TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
        spec_B=StimuliSpec.uniform(low=0.0, high=1.0),
    )

    # The victim copies the SECOND A tile (buffer_A[1]).
    victim_tile = src_A[ELEMENTS_PER_TILE : 2 * ELEMENTS_PER_TILE]
    datacopy_golden = get_golden_generator(DataCopyGolden)
    golden = datacopy_golden(
        victim_tile,
        formats.output_format,
        input_format=formats.input_format,
    )

    configuration = TestConfig(
        "sources/experimental_reconfig_escape_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            POLLUTER(polluter=polluter),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
            sfpu=False,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == ELEMENTS_PER_TILE
    ), f"Expected one {ELEMENTS_PER_TILE}-element output tile, got {len(res_from_L1)}"

    res_tensor = torch.tensor(res_from_L1, dtype=torch.float32)
    assert passed_test(golden, res_tensor, formats.output_format), (
        f"reconfig escape from experimental '{POLLUTERS[polluter]}' (empty uninit): "
        f"victim datacopy diverged from its input"
    )
