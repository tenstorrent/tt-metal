# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Memory-boundary semantics; real TT numerics are in padding_variants.py."""

import unittest
from unittest.mock import patch
import numpy as np

from models.experimental.nllb.tt import backend


class DecoderMemoryTests(unittest.TestCase):
    def test_only_wholly_masked_trailing_tiles_removed(self):
        model = backend.Backend.__new__(backend.Backend)
        # Includes noncontiguous masks, a final valid tile, and tile boundaries.
        for positions, expected in [
            ([0], 32),
            ([0, 31], 32),
            ([0, 32], 64),
            ([2, 47], 64),
            ([0, 64], 96),
            ([95], 96),
            ([], 96),
        ]:
            with self.subTest(positions=positions):
                memory = np.arange(96 * 4).reshape(1, 1, 96, 4)
                valid = np.zeros(96, dtype=bool)
                valid[positions] = True
                original_memory = memory.copy()
                original_mask = valid.copy()

                def sliced(value, start, end):
                    return value[tuple(slice(a, b) for a, b in zip(start, end))].copy()

                with patch.object(backend.ttnn, "slice", side_effect=sliced) as call:
                    result, mask = model.decoder_memory(memory, valid)
                assert result.shape == (1, 1, expected, 4)
                assert np.array_equal(result, memory[..., :expected, :])
                assert np.array_equal(mask, valid[:expected])
                assert np.array_equal(memory, original_memory)
                assert np.array_equal(valid, original_mask)
                assert call.call_count == int(expected < 96)
                if expected == 96:
                    assert result is memory and mask is valid
                # Valid coordinates and all interior holes are unchanged.
                assert np.array_equal(np.flatnonzero(mask), positions)
                assert not valid[expected:].any()


if __name__ == "__main__":
    unittest.main()
