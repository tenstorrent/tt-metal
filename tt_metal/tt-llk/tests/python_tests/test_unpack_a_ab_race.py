# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Datacopy (unpack_A) then ELWADD (unpack_AB) in one DST section, with math behind the unpacker."""

import pytest
import torch
from tt_llk_harness import DataFormat, DestAccumulation, StimuliConfig
from tt_llk_harness import TestConfig as LLKTestConfig
from tt_llk_harness import blackhole_only, input_output_formats, params, tilize_block


@blackhole_only
@pytest.mark.parametrize(
    "source",
    [
        "sources/unpack_a_ab_race_test.cpp",
        "sources/unpack_a_ab_race_nodelay_test.cpp",
        "sources/unpack_a_ab_race_abstall_test.cpp",
    ],
)
def test_unpack_a_ab_race(source):
    generator = torch.Generator().manual_seed(4869)
    a = torch.randint(-256, 0, (24, 32, 32), generator=generator).bfloat16()
    b = torch.randint(0, 65, (24, 32, 32), generator=generator).bfloat16()

    def tilized(t):
        return tilize_block(
            t.reshape(-1), [24 * 32, 32], DataFormat.Float16_b
        ).flatten()

    src_a, src_b = tilized(a), tilized(b)
    formats = input_output_formats([DataFormat.Float16_b])[0]
    config = LLKTestConfig(
        source,
        formats,
        runtimes=[params.TILE_COUNT(24)],
        variant_stimuli=StimuliConfig(
            src_a,
            formats.input_format,
            src_b,
            formats.input_format,
            formats.output_format,
            tile_count_A=24,
            tile_count_B=24,
            tile_count_res=24,
        ),
        dest_acc=DestAccumulation.No,
    )
    ta, tb = src_a.reshape(3, 8, 1024), src_b.reshape(3, 8, 1024)
    expected = torch.cat(
        [ta[:, :6], (ta[:, 6:].float() + tb[:, 6:].float()).bfloat16()], dim=1
    )
    for run in range(2):
        actual = torch.as_tensor(config.run().result, dtype=torch.bfloat16).reshape(
            3, 8, 1024
        )
        bad = [
            (batch, slot, int((actual[batch, slot] != expected[batch, slot]).sum()))
            for batch in range(3)
            for slot in range(8)
            if not torch.equal(actual[batch, slot], expected[batch, slot])
        ]
        assert not bad, f"run {run}: (batch, slot, wrong values) {bad}"
