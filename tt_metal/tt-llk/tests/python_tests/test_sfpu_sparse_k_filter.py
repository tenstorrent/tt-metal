# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Sparse-K index filter SFPU test. Covers experimental/ckernel_sfpu_sparse_k_filter.h.

Each Dest lane holds one bank-addressed index. The kernel keeps the indices whose
global-bank field matches this core's bank and rewrites them, zeroing the rest:
    enc = ((idx >> GLOBAL_BANK_SHIFT) & BANK_MASK) == MY_BANK
            ? ((idx & WITHIN_BANK_MASK) + 1) << OUT_SHIFT : 0

Four properties carry the contract, and the stimuli are built to hit them:
  - The +1 offset.
  - Bits above the bank field are ignored.
  - WITHIN_BANK_MASK strips the high bits.
  - OUT_SHIFT packs without overlap. With WITHIN_BANK_MASK = 0x3FFF the largest
    encode is 16384, so it fits in 16 bits and OUT_SHIFT=16 leaves the low
    half clear. Two can, therefore, be packed into one 32-bit integer.

Only even values of ITERATIONS are tested. On odd ITERATIONS, the kernel leaves
half of the rows untouched.
"""

import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import SPARSE_K_CONFIG, TILE_COUNT

SFPU_ROW_ELEMENTS = 32
FORMATS = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)


# (bank_mask, global_bank_shift, within_bank_mask).
FIELD_LAYOUTS = [
    (0x3F, 14, 0x3FFF),
    (0x07, 8, 0x00FF),
]

MY_BANKS = [0, 1, 63]
OUT_SHIFTS = [0, 16]
ITERATIONS = [32, 16, 8]


def _build_indices(layout, my_bank: int, torch_format):
    """A tile of indices covering every case the contract turns on.

    lane 0             local 0 in MY_BANK, the +1 sentinel
    lane 1             local mask in MY_BANK, the largest valid encode
    lane 2             local 0, other bank; must be 0, not 1
    lane 3             MY_BANK, small local, with a high bit above the field
    lane 4             MY_BANK, local mask, with a high bit above the field
    lanes 5..          random mix of banks and locals
    """
    bank_mask, shift, within_mask = layout
    other_bank = (my_bank + 1) & bank_mask
    # A bit above the bank field, which the in-place compare must ignore and the
    # local extraction must mask away.
    high_bit = 1 << (shift + 6 + 2)

    values = [
        (my_bank << shift) | 0,
        (my_bank << shift) | within_mask,
        (other_bank << shift) | 0,
        (my_bank << shift) | 5 | high_bit,
        (my_bank << shift) | within_mask | high_bit,
    ]

    generator = torch.Generator().manual_seed(0)
    remaining = ELEMENTS_PER_TILE - len(values)
    banks = torch.randint(
        0, bank_mask + 1, (remaining,), generator=generator, dtype=torch.int64
    )
    locals_ = torch.randint(
        0, within_mask + 1, (remaining,), generator=generator, dtype=torch.int64
    )
    # Bias a slice toward MY_BANK so matches are plentiful even for a wide mask.
    banks[: remaining // 4] = my_bank
    tail = ((banks << shift) | locals_).tolist()

    return torch.tensor(values + tail, dtype=torch_format)


def _sparse_k_golden(indices, layout, my_bank: int, out_shift: int, iterations: int):
    """Elementwise golden; rows at or past `iterations` keep their input value."""
    bank_mask, shift, within_mask = layout
    idx = indices.to(torch.int64)

    matches = ((idx >> shift) & bank_mask) == my_bank
    enc = (((idx & within_mask) + 1) << out_shift) * matches.to(torch.int64)

    golden = idx.clone()
    written = SFPU_ROW_ELEMENTS * iterations
    golden[:written] = enc[:written]
    return golden


def _run(layout, my_bank, out_shift, iterations):
    """Run one variant; returns (result, golden, indices) as int64 flat tiles."""
    torch_format = format_dict[FORMATS.input_format]

    src_A = _build_indices(layout, my_bank, torch_format)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    golden = _sparse_k_golden(src_A, layout, my_bank, out_shift, iterations)

    bank_mask, shift, within_mask = layout
    configuration = TestConfig(
        "sources/sfpu_sparse_k_filter_test.cpp",
        FORMATS,
        templates=[
            SPARSE_K_CONFIG(
                sparse_k_iterations=iterations,
                bank_mask=bank_mask,
                my_bank=my_bank,
                global_bank_shift=shift,
                within_bank_mask=within_mask,
                out_shift=out_shift,
            ),
        ],
        runtimes=[
            TILE_COUNT(1),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    res_from_L1 = configuration.run().result
    res = torch.tensor(res_from_L1, dtype=torch_format).to(torch.int64)
    return res, golden, src_A.to(torch.int64)


def _valid_my_banks(layout):
    """MY_BANK has to fit inside BANK_MASK, or the kernel matches no lane at all."""
    bank_mask = layout[0]
    return [bank for bank in MY_BANKS if bank <= bank_mask]


@skip_for_wormhole
@parametrize(
    layout=FIELD_LAYOUTS,
    my_bank=lambda layout: _valid_my_banks(layout),
    out_shift=OUT_SHIFTS,
    iterations=ITERATIONS,
)
def test_sfpu_sparse_k_filter(layout, my_bank, out_shift, iterations):
    res, golden, indices = _run(layout, my_bank, out_shift, iterations)

    # Integers, so this is exact. The golden keeps the input value in rows at or
    # past ITERATIONS, which is also the check that the kernel stops there.
    mismatch = (res != golden).nonzero().flatten()
    assert mismatch.numel() == 0, (
        f"sparse-K filter mismatch at flat offsets {mismatch[:16].tolist()}: "
        f"idx={[hex(v) for v in indices[mismatch[:8]].tolist()]} "
        f"golden={golden[mismatch[:8]].tolist()} got={res[mismatch[:8]].tolist()}"
    )
