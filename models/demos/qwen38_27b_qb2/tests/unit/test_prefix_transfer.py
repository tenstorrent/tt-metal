# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check physical-page planning without claiming device-transfer correctness."""

import unittest

from models.demos.qwen38_27b_qb2.tt.prefix_transfer import overlapping, page_windows


class PrefixTransferPlanningTests(unittest.TestCase):
    def test_coalesces_only_adjacent_physical_pages_with_a_bounded_window(self):
        windows = page_windows((8, 9, 10, 2, 4, 5, 6, 7), 16, limit=48)
        self.assertEqual(
            [(w.logical_offset, w.size, w.start, w.length) for w in windows],
            [(0, 48, 8, 3), (48, 16, 2, 1), (64, 48, 4, 3), (112, 16, 7, 1)],
        )

    def test_arbitrary_codec_chunks_cover_exact_logical_bytes(self):
        pages = (8, 9, 2, 5, 6)
        windows = page_windows(pages, 17, limit=40)
        physical = [bytes([i]) * 17 for i in range(12)]
        expected = b"".join(physical[p] for p in pages)
        for offset in range(len(expected)):
            for size in (1, min(7, len(expected) - offset), len(expected) - offset):
                actual = bytearray(size)
                for window, source, destination, length in overlapping(windows, offset, size):
                    data = b"".join(physical[p] for p in range(window.start, window.start + window.length))
                    actual[destination : destination + length] = data[source : source + length]
                self.assertEqual(bytes(actual), expected[offset : offset + size])

    def test_invalid_page_ownership_is_rejected(self):
        for pages in ((), (1, 1), (-1,), (True,), (1.5,)):
            with self.assertRaises(ValueError):
                page_windows(pages, 16)

    def test_invalid_byte_ranges_and_window_budgets_are_rejected(self):
        for page_bytes, limit in ((0, 10), (20, 10), (1.5, 10)):
            with self.assertRaises(ValueError):
                page_windows((0,), page_bytes, limit=limit)
        for offset, size in ((-1, 1), (0, 0), (15, 2), (16, 1)):
            with self.assertRaises(ValueError):
                list(overlapping(page_windows((0,), 16), offset, size))


if __name__ == "__main__":
    unittest.main()
