# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Chains of pack programs in one kernel (Blackhole): fast untilize, pack_rows or untilize followed by a block pack, per-tile and block packs in both orders.

Each step packs its own four tiles of the input to its own four result tiles and the next step runs with the pack state the previous one left, the way a
fused kernel does (sources/pack_state_chain_test.cpp). The block steps run the close of a fast untilize whose pack uninit was not called, which
llk_pack_block runs first: with asserts off (the production build) that is the whole contract, and these ids fail without it.
"""

import dataclasses

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.bfp_format_utils import QUANTIZE_TO_FORMAT
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    PackGolden,
    PackRowsGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    PACK_SRC_FROM_OUTPUT,
    DEST_SYNC,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from helpers.utils import passed_test

KINDS = {"PT": 0, "PB1": 1, "PBW": 2, "PBC": 3, "PBCW": 4, "ROWS": 5, "UNT": 6, "FU": 7, "PTC": 8}
NT = 4
ROWS_R = 50
TILE_ELEMS = 1024


@dataclasses.dataclass
class CHAIN(TemplateParameter):
    steps: tuple = ()
    fu_pack_uninit: bool = False

    def convert_to_cpp(self) -> str:
        ids = ", ".join(str(KINDS[s]) for s in self.steps)
        return (
            f"constexpr int NSTEPS = {len(self.steps)};\n"
            f"constexpr int CHAIN_STEPS[] = {{{ids}}};\n"
            f"constexpr bool FU_PACK_UNINIT = {str(self.fu_pack_uninit).lower()};\n"
            f"constexpr std::uint32_t ROWS_R = {ROWS_R};"
        )


F16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
F32 = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
B8 = InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Bfp8_b)
B8_F16 = InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float16_b)
B8_F32 = InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float32)

# (id, steps, pack uninit of the fast untilize called)
FU_CHAINS = [
    ("fu_pb1", ("FU", "PB1"), False),
    ("fu_pbw", ("FU", "PBW"), False),
    ("fu_pbc", ("FU", "PBC"), False),
    ("fu_pbcw", ("FU", "PBCW"), False),
    ("fu_ptc", ("FU", "PTC"), False),
    ("fu_pbw_pbw", ("FU", "PBW", "PBW"), False),
    ("pbw_fu_pbw", ("PBW", "FU", "PBW"), False),
    ("pbcw_fu_pbcw", ("PBCW", "FU", "PBCW"), False),
    ("fu_uninit_pbw", ("FU", "PBW"), True),
    ("fu_uninit_pb1", ("FU", "PB1"), True),
]
ROWS_CHAINS = [
    ("rows_pb1", ("ROWS", "PB1")),
    ("rows_pbw", ("ROWS", "PBW")),
    ("rows_pbc", ("ROWS", "PBC")),
    ("rows_pbcw", ("ROWS", "PBCW")),
    ("pbw_rows_pbw", ("PBW", "ROWS", "PBW")),
    ("rows_pt", ("ROWS", "PT")),
]
UNT_CHAINS = [
    ("unt_pt", ("UNT", "PT")),
    ("unt_pb1", ("UNT", "PB1")),
    ("unt_pbw", ("UNT", "PBW")),
    ("unt_pbc", ("UNT", "PBC")),
    ("pt_unt_pt", ("PT", "UNT", "PT")),
]
MIX_CHAINS = [
    ("pb1_pt", ("PB1", "PT")),
    ("pt_pb1", ("PT", "PB1")),
    ("pbw_pt", ("PBW", "PT")),
    ("pt_pbw", ("PT", "PBW")),
    ("pbc_pt", ("PBC", "PT")),
    ("pt_pbc", ("PT", "PBC")),
    ("pbcw_pt", ("PBCW", "PT")),
    ("pt_pbcw", ("PT", "PBCW")),
    ("pbw_pb1_pbw", ("PBW", "PB1", "PBW")),
    ("pbcw_pbc_pbcw", ("PBCW", "PBC", "PBCW")),
    ("pb1_pt_pbw", ("PB1", "PT", "PBW")),
]

PLAIN_FORMATS = [F16, F32]
BFP_FORMATS = [B8]


def _acc_modes(formats):
    if formats.output_format == DataFormat.Float32 or formats.input_format == DataFormat.Float32:
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


def _step_golden(kind, src, formats, step, dest_acc):
    """Expected datums of one step (the region of its NT result tiles) and the (offset, length) slices that count."""
    tiles = src[step * NT * TILE_ELEMS : (step + 1) * NT * TILE_ELEMS]
    base = step * NT * TILE_ELEMS
    if kind in ("UNT", "FU"):
        gen = get_golden_generator(UntilizeGolden)
        g = gen(tiles, formats.output_format, [32, 32 * NT], input_format=formats.input_format)
        return [(base, g)]
    if kind == "ROWS":
        gen = get_golden_generator(PackRowsGolden)
        g = gen(tiles, formats.output_format, [32, 32 * NT], num_rows_to_pack=ROWS_R, tile_count=NT)
        per_tile = ROWS_R * 16
        return [(base + t * TILE_ELEMS, g[t * per_tile : (t + 1) * per_tile]) for t in range(NT)]
    gen = get_golden_generator(PackGolden)
    g = gen(tiles, formats.output_format, input_dimensions=[32, 32 * NT])
    if formats.output_format in QUANTIZE_TO_FORMAT:
        g = QUANTIZE_TO_FORMAT[formats.output_format](g).to(format_dict[formats.output_format])
    return [(base, g)]


def run_chain(formats, dest_acc, dest_sync, steps, fu_pack_uninit):
    steps = tuple(steps)
    n = len(steps)
    dims = [32, 32 * NT * n]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    configuration = TestConfig(
        "sources/pack_state_chain_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            DEST_SYNC(dest_sync),
            PACK_SRC_FROM_OUTPUT(),
            CHAIN(steps, fu_pack_uninit),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            sfpu=False,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )
    res = torch.tensor(configuration.run().result, dtype=format_dict[formats.output_format])
    assert len(res) == tile_cnt_A * TILE_ELEMS, f"result has {len(res)} datums"
    bad = []
    for s, kind in enumerate(steps):
        for off, golden in _step_golden(kind, src_A, formats, s, dest_acc):
            got = res[off : off + len(golden)]
            if not passed_test(golden, got, formats.output_format, print_errors=False):
                bad.append(f"step {s} {kind}")
                break
    assert not bad, "wrong output in " + ", ".join(bad) + f" of chain {steps}"


@skip_for_wormhole
@parametrize(
    formats=[F16, F32, B8_F16, B8_F32],
    dest_acc=_acc_modes,
    dest_sync=[DestSync.Half, DestSync.Full],
    chain=[c[0] for c in FU_CHAINS],
)
def test_pack_state_chain_fast_untilize(formats, dest_acc, dest_sync, chain):
    """Fast untilize followed by block packs (and a block pack, fast untilize, block pack) without the pack uninit of the fast untilize."""
    _, steps, uninit = next(c for c in FU_CHAINS if c[0] == chain)
    run_chain(formats, dest_acc, dest_sync, steps, uninit)


@skip_for_wormhole
@parametrize(
    formats=[F16, F32, B8],
    dest_acc=_acc_modes,
    dest_sync=[DestSync.Half, DestSync.Full],
    chain=[c[0] for c in ROWS_CHAINS],
)
def test_pack_state_chain_rows(formats, dest_acc, dest_sync, chain):
    """pack_rows followed by block and per-tile packs."""
    _, steps = next(c for c in ROWS_CHAINS if c[0] == chain)
    if formats in BFP_FORMATS and any(s in ("PB1", "PBW") for s in steps):
        pytest.skip("block float outputs take the closed block")
    run_chain(formats, dest_acc, dest_sync, steps, False)


@skip_for_wormhole
@parametrize(
    formats=[F16, F32],
    dest_acc=_acc_modes,
    dest_sync=[DestSync.Half, DestSync.Full],
    chain=[c[0] for c in UNT_CHAINS],
)
def test_pack_state_chain_untilize(formats, dest_acc, dest_sync, chain):
    """pack_untilize followed by per-tile (the byte-wise destination write under full sync) and block packs."""
    _, steps = next(c for c in UNT_CHAINS if c[0] == chain)
    run_chain(formats, dest_acc, dest_sync, steps, False)


@skip_for_wormhole
@parametrize(
    formats=[F16, F32, B8],
    dest_acc=_acc_modes,
    dest_sync=[DestSync.Half, DestSync.Full],
    chain=[c[0] for c in MIX_CHAINS],
)
def test_pack_state_chain_mix(formats, dest_acc, dest_sync, chain):
    """Block and per-tile packs in both orders."""
    _, steps = next(c for c in MIX_CHAINS if c[0] == chain)
    if formats in BFP_FORMATS and any(s in ("PB1", "PBW") for s in steps):
        pytest.skip("block float outputs take the closed block")
    run_chain(formats, dest_acc, dest_sync, steps, False)
