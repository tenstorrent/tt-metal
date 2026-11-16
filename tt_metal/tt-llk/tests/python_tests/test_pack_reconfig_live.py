# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The compute API's constant-output pack reconfigure (pack_api_shim.h) of one output, before and after every other writer of the
packer fields it writes (Blackhole, sources/pack_reconfig_live_test.cpp). A reconfigure to the output whose configuration is live
writes nothing; every other writer must make the next one write again. Steps: R a reconfigure and four per-tile packs, U pack
untilize init and uninit, F fast untilize init and uninit, T fast tilize init and uninit, W pack_rows init and uninit, S the
run-time FP32 DEST switch off and on (all three threads), M the unpack and math reconfigure to another input and back, Z the
experimental compute API's W stride writes, I a default pack init with its strides. Every R step is checked against the golden.
"""

import dataclasses
import hashlib
import os

import torch
from conftest import skip_for_wormhole
from helpers.bfp_format_utils import QUANTIZE_TO_FORMAT
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import PackGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    PACK_SRC_FROM_OUTPUT,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from helpers.utils import passed_test

NT = 4
TILE_ELEMS = 1024


@dataclasses.dataclass
class RECONFIG_LIVE(TemplateParameter):
    steps: str = "R"

    def convert_to_cpp(self) -> str:
        chars = ", ".join(f"'{c}'" for c in self.steps)
        return (
            f"constexpr int RL_NSTEPS = {len(self.steps)};\n"
            f"constexpr char RL_STEPS[] = {{{chars}}};\n"
            f"constexpr std::uint32_t RL_NR = {self.steps.count('R')};"
        )


# a reconfigure, a writer, a reconfigure to the same output; RR and RMR skip the second reconfigure
CHAINS = [
    "RR",
    "RRR",
    "RUR",
    "RFR",
    "RTR",
    "RWR",
    "RSR",
    "RMR",
    "RZR",
    "RIR",
    "RURMR",
    "RTRWRFR",
    "RSRMRR",
]

# input -> output with a second pack format in the kernel (no skip-all reconfigure)
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float16_b),
]


def _acc_modes(formats):
    if formats.input_format == DataFormat.Float32:
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=_acc_modes,
    dest_sync=[DestSync.Half, DestSync.Full],
    chain=CHAINS,
)
def test_pack_reconfig_live(formats, dest_acc, dest_sync, chain):
    nr = chain.count("R")
    dims = [32, 32 * NT * nr]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    configuration = TestConfig(
        "sources/pack_reconfig_live_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            MATH_FIDELITY(MathFidelity.HiFi4),
            DEST_SYNC(dest_sync),
            PACK_SRC_FROM_OUTPUT(),
            RECONFIG_LIVE(chain),
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
    res = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    )
    golden = get_golden_generator(PackGolden)(
        src_A, formats.output_format, input_dimensions=dims
    )
    if formats.output_format in QUANTIZE_TO_FORMAT:
        golden = QUANTIZE_TO_FORMAT[formats.output_format](golden).to(
            format_dict[formats.output_format]
        )
    # LLK_RESULT_HASH_FILE: one line per id with the result's hash, to compare the bits of two trees
    if os.environ.get("LLK_RESULT_HASH_FILE"):
        with open(os.environ["LLK_RESULT_HASH_FILE"], "a") as f:
            h = hashlib.sha1(res.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
            f.write(
                f"{formats.input_format.name}>{formats.output_format.name} {dest_acc.name} {dest_sync.name} {chain} {h}\n"
            )
    bad = []
    for r in range(nr):
        lo, hi = r * NT * TILE_ELEMS, (r + 1) * NT * TILE_ELEMS
        if not passed_test(
            golden[lo:hi], res[lo:hi], formats.output_format, print_errors=False
        ):
            bad.append(r)
    assert not bad, f"wrong output after reconfigure steps {bad} of chain {chain}"
