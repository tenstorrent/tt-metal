# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Resume-offset alignment arithmetic. Imports no ttnn and needs no device."""

import pytest

from models.common.model_capabilities import floor_to_alignment, resume_offset_alignment


class TestResumeOffsetAlignment:
    def test_shipping_pairs_resolve_to_the_declared_value(self):
        # tt_transformers declares 256, gemma4 declares 128, both over a 64-token
        # KV block. Powers of two, so the declaration already covers the block.
        assert resume_offset_alignment(64, 256) == 256
        assert resume_offset_alignment(64, 128) == 128

    def test_without_a_declaration_only_the_block_size_applies(self):
        # A model that does not support chunked prefill still gets prefix-cache
        # offsets, which are block-aligned by construction.
        assert resume_offset_alignment(64, None) == 64
        assert resume_offset_alignment(64, 0) == 64

    def test_a_declaration_the_block_size_does_not_divide_takes_the_lcm(self):
        # The point of using an LCM rather than a max: 96 satisfies a 96-token
        # q_chunk_size but not a 64-token block, and max() would return it.
        assert resume_offset_alignment(64, 96) == 192

    def test_a_declaration_below_the_block_size_does_not_weaken_it(self):
        assert resume_offset_alignment(128, 64) == 128


class TestFloorToAlignment:
    @pytest.mark.parametrize(
        "offset, expected",
        [
            # Block-aligned but not q_chunk-aligned. These are the offsets that
            # the vLLM scheduler produces and that the traced SDPA cannot honour;
            # answering from the wrong prefix instead of raising is what makes
            # getting this wrong expensive to find.
            (1088, 1024),
            (1600, 1536),
            (3136, 3072),
            # Not a multiple of anything.
            (777, 768),
            (1, 0),
            (2061, 2048),
            # Already aligned, must not move.
            (0, 0),
            (768, 768),
            (2048, 2048),
            (2560, 2560),
        ],
    )
    def test_offsets_floor_to_the_alignment(self, offset, expected):
        assert floor_to_alignment([offset], 256) == [expected]

    def test_a_mixed_batch_is_floored_per_user(self):
        assert floor_to_alignment([0, 1088, 777, 2560], 256) == [0, 1024, 768, 2560]

    def test_flooring_never_raises_the_offset(self):
        # Flooring is only sound because it recomputes tokens whose K/V is
        # rewritten identically. Raising an offset would skip tokens instead.
        offsets = list(range(0, 4096, 37))
        for before, after in zip(offsets, floor_to_alignment(offsets, 256)):
            assert after <= before
            assert before - after < 256
            assert after % 256 == 0

    def test_numpy_style_integers_are_accepted(self):
        # vLLM hands the generator numpy integers; floor division on those
        # returns numpy types that later bit_length() calls reject.
        class FakeNpInt(int):
            pass

        assert floor_to_alignment([FakeNpInt(1088)], 256) == [1024]
        assert all(type(v) is int for v in floor_to_alignment([FakeNpInt(1088)], 256))
