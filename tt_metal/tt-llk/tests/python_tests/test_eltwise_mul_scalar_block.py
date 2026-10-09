# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One SrcB scalar reused across contiguous SrcA tiles, then replaced on re-entry."""

from dataclasses import dataclass

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import EltwiseBinaryGolden
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TemplateParameter,
)
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]


@dataclass
class SCALAR_BLOCK_ALIAS(TemplateParameter):
    """True runs the aliased standard scalar broadcast multiply instead of the block form."""

    alias: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool SCALAR_BLOCK_ALIAS = {'true' if self.alias else 'false'};"
        )


def _scalar_tiles(scalars):
    # Other SrcB lanes are poison: only B[0] is a scalar.
    tiles = torch.full((len(scalars), 1024), -17.0, dtype=torch.bfloat16)
    tiles[:, 0] = scalars
    return tiles


def _block_config(
    src, scalars, block_size, dst_index, dest_acc, dest_sync, math_fidelity, alias
):
    return TestConfig(
        "sources/eltwise_mul_scalar_block_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
        templates=[
            DEST_SYNC(dest_sync),
            MATH_FIDELITY(math_fidelity),
            SCALAR_BLOCK_ALIAS(alias),
        ],
        runtimes=[
            NUM_BLOCKS(len(scalars)),
            NUM_TILES_IN_BLOCK(block_size),
            DEST_INDEX(dst_index),
        ],
        variant_stimuli=StimuliConfig(
            src,
            DataFormat.Float16_b,
            _scalar_tiles(scalars).flatten(),
            DataFormat.Float16_b,
            DataFormat.Float32,
            tile_count_A=len(scalars) * block_size,
            tile_count_B=len(scalars),
            tile_count_res=len(scalars) * block_size,
        ),
        dest_acc=dest_acc,
    )


@parametrize(
    block_layout=[(1, 0), (3, 1), (4, 0)],
    dest_acc=list(DestAccumulation),
    dest_sync=list(DestSync),
)
def test_eltwise_mul_scalar_block(block_layout, dest_acc, dest_sync):
    block_size, dst_index = block_layout
    scalars = torch.tensor([1.5, -0.5, 0.0, 2.0], dtype=torch.bfloat16)
    count = len(scalars) * block_size * 1024
    # Exactly representable products allow an elementwise, zero-tolerance check
    # in both DEST widths. Other SrcB lanes are poison: only B[0] is a scalar.
    src = ((torch.arange(count) * 17 % 63 - 31).to(torch.float32) / 8).to(
        torch.bfloat16
    )
    golden = (
        src.reshape(len(scalars), -1).float() * scalars.float()[:, None]
    ).flatten()
    config = _block_config(
        src,
        scalars,
        block_size,
        dst_index,
        dest_acc,
        dest_sync,
        MathFidelity.LoFi,
        False,
    )
    result = torch.as_tensor(config.run().result).float().flatten()
    assert torch.equal(
        result, golden
    ), "Scalar block multiply lost a tile, reused an old scalar, or wrote the wrong DEST offset"


def _fidelity_golden(src, scalars, math_fidelity):
    """The sum of the fidelity-masked phase products, as the standard multiply computes at math_fidelity."""
    phases = {
        MathFidelity.LoFi: 1,
        MathFidelity.HiFi2: 2,
        MathFidelity.HiFi3: 3,
        MathFidelity.HiFi4: 4,
    }[math_fidelity]
    src_a = src.reshape(len(scalars), -1)
    src_b = scalars[:, None].expand_as(src_a).contiguous()
    masking = EltwiseBinaryGolden()
    result = torch.zeros(src_a.shape, dtype=torch.float32)
    for phase in range(phases):
        a_m, b_m = masking._apply_fidelity_masking(
            DataFormat.Float16_b, src_a, src_b, phase
        )
        result += a_m.float() * b_m.float()
    return result.flatten()


@parametrize(
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    dest_acc=list(DestAccumulation),
)
def test_eltwise_mul_scalar_block_fidelity(math_fidelity, dest_acc):
    """Above LoFi the block form equals the aliased standard multiply lane by lane and follows the fidelity-masked
    reference rather than the one-phase product."""
    block_size, dst_index, dest_sync = 4, 0, DestSync.Half
    # Scalars and operands with full mantissas, so every fidelity phase contributes.
    scalars = torch.tensor([1.0 / 3, -0.7, 1.01, 2.9], dtype=torch.bfloat16)
    torch.manual_seed(0)
    count = len(scalars) * block_size * 1024
    src = (torch.rand(count) * 4.0 - 2.0).to(torch.bfloat16)

    golden = _fidelity_golden(src, scalars, math_fidelity)
    one_phase = _fidelity_golden(src, scalars, MathFidelity.LoFi)
    assert not torch.equal(
        golden, one_phase
    ), "the stimuli do not depend on the fidelity"

    block_cfg = _block_config(
        src, scalars, block_size, dst_index, dest_acc, dest_sync, math_fidelity, False
    )
    alias_cfg = _block_config(
        src, scalars, block_size, dst_index, dest_acc, dest_sync, math_fidelity, True
    )
    # Build both before running either: under --compile-producer run() returns after the first build.
    block_cfg.prepare()
    alias_cfg.prepare()
    block = torch.as_tensor(block_cfg.run().result).float().flatten()
    alias = torch.as_tensor(alias_cfg.run().result).float().flatten()

    assert torch.equal(
        block, alias
    ), f"the block multiply at {math_fidelity.name} differs from the aliased scalar broadcast multiply"
    if dest_acc == DestAccumulation.Yes:
        # Every phase product of two bf16 operands is exact in a 32-bit DEST, so the sum is too.
        assert torch.equal(
            block, golden
        ), f"the block multiply at {math_fidelity.name} is not the fidelity-masked product"
    else:
        assert passed_test(golden, block, DataFormat.Float32)
    assert (block - golden).abs().max() < (block - one_phase).abs().max(), (
        f"the block multiply at {math_fidelity.name} is closer to the one-phase product "
        "than to the fidelity-masked one: the fidelity phases did not run"
    )
