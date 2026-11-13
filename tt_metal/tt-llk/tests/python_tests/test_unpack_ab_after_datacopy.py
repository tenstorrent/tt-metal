# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Datacopy (unpack_A) then ELWADD (unpack_AB) in one DST section, with math behind the unpacker.

Without the unpacker 1 drain in _llk_unpack_AB_init_, this fails only on the first run after a chip reset
(tt-smi -r), so it is a manual repro: on a warm card it passes either way."""

import torch
from tt_llk_harness import DataFormat, DestAccumulation, StimuliConfig
from tt_llk_harness import TestConfig as LLKTestConfig
from tt_llk_harness import blackhole_only, input_output_formats, params, tilize_block

TILE_COUNT = 24
DEST_TILES = 8
COPIES = 6


@blackhole_only
def test_unpack_ab_after_datacopy():
    generator = torch.Generator().manual_seed(4869)
    a = torch.randint(-256, 0, (TILE_COUNT, 32, 32), generator=generator).bfloat16()
    b = torch.randint(1, 65, (TILE_COUNT, 32, 32), generator=generator).bfloat16()

    def tilized(t):
        return tilize_block(
            t.reshape(-1), [TILE_COUNT * 32, 32], DataFormat.Float16_b
        ).flatten()

    src_a, src_b = tilized(a), tilized(b)
    formats = input_output_formats([DataFormat.Float16_b])[0]
    config = LLKTestConfig(
        "sources/unpack_ab_after_datacopy_test.cpp",
        formats,
        runtimes=[params.TILE_COUNT(TILE_COUNT)],
        variant_stimuli=StimuliConfig(
            src_a,
            formats.input_format,
            src_b,
            formats.input_format,
            formats.output_format,
            tile_count_A=TILE_COUNT,
            tile_count_B=TILE_COUNT,
            tile_count_res=TILE_COUNT,
        ),
        dest_acc=DestAccumulation.No,
    )

    sections = TILE_COUNT // DEST_TILES
    ta = src_a.reshape(sections, DEST_TILES, 1024)
    tb = src_b.reshape(sections, DEST_TILES, 1024)
    expected = torch.cat(
        [ta[:, :COPIES], (ta[:, COPIES:].float() + tb[:, COPIES:].float()).bfloat16()],
        dim=1,
    )
    actual = torch.as_tensor(config.run().result, dtype=torch.bfloat16).reshape(
        sections, DEST_TILES, 1024
    )
    wrong = [
        (section, slot)
        for section in range(sections)
        for slot in range(DEST_TILES)
        if not torch.equal(actual[section, slot], expected[section, slot])
    ]
    assert not wrong, f"(DST section, slot) tiles that differ from the golden: {wrong}"
