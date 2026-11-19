# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compile the device integer arithmetic as host C++; never import TTNN."""

import ast
import ctypes
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tt.device_seed import (
    _INTEGER_SOURCE,
    MASK32,
    MASK64,
    reset_words,
    seed_for_draw,
)


class DeviceSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="qwen-seed-cpu-")
        root = Path(cls.temp.name)
        source = root / "seed.cpp"
        source.write_text(
            "#include <cstdint>\n"
            + _INTEGER_SOURCE
            + '\nextern "C" uint32_t hash_seed(uint64_t seed, uint64_t counter) {\n'
            + "return request_seed_for_draw({uint32_t(seed), uint32_t(seed >> 32)},"
            + "{uint32_t(counter), uint32_t(counter >> 32)}); }\n"
            + 'extern "C" uint32_t advance(uint32_t* row, bool active) { return advance_request_seed(row,active); }\n'
        )
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(root / "seed.so")],
            check=True,
            timeout=30,
            capture_output=True,
        )
        cls.library = ctypes.CDLL(str(root / "seed.so"))
        cls.library.hash_seed.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        cls.library.hash_seed.restype = ctypes.c_uint32
        cls.library.advance.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_bool]
        cls.library.advance.restype = ctypes.c_uint32
        common_path = Path(__file__).resolve().parents[3] / "common/sampling/generator.py"
        function = next(
            node
            for node in ast.parse(common_path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_hash_request_seed_to_device_seed"
        )
        namespace = {"_UINT64_MASK": MASK64, "DEVICE_SEED_MAX": 1_000_000}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(common_path), "exec"), namespace)
        cls.common_hash = staticmethod(namespace[function.name])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def row(self, seed, counter=0, **kwargs):
        return (ctypes.c_uint32 * 16)(*reset_words(seed, counter, **kwargs))

    def test_exact_current_common_hash_and_cpp_limbs(self):
        rng = random.Random(4711)
        pairs = [
            (seed, counter) for seed in (0, 1, -1, 2**32, 2**63, 2**80 + 7) for counter in (0, 1, MASK32, MASK64)
        ]
        pairs += [(rng.getrandbits(80) - 2**79, rng.getrandbits(64)) for _ in range(10000)]
        for seed, counter in pairs:
            expected = self.common_hash(seed, counter, salt=0)
            self.assertEqual(seed_for_draw(seed, counter), expected)
            self.assertEqual(self.library.hash_seed(seed & MASK64, counter), expected)
            self.assertTrue(1 <= expected <= 1_000_000)

    def test_counter_carry_and_modular_wrap(self):
        row = self.row(-19, MASK32)
        self.assertEqual(self.library.advance(row, True), self.common_hash(-19, MASK32))
        self.assertEqual(list(row)[2:4], [0, 1])
        row = self.row(19, MASK64)
        self.assertEqual(self.library.advance(row, True), self.common_hash(19, MASK64))
        self.assertEqual(list(row)[2:4], [0, 0])

    def test_inactive_does_not_advance_or_initialize(self):
        for row in (self.row(3, 7), self.row(None, entropy_seed=1234)):
            previous = list(row)
            for _ in range(7):
                self.assertEqual(self.library.advance(row, False), MASK32)
                self.assertEqual(list(row), previous)

    def test_unseeded_initializes_once_then_skips(self):
        row = self.row(None, entropy_seed=123456)
        self.assertEqual(self.library.advance(row, True), 123456)
        self.assertEqual(row[4], 0)
        for _ in range(5):
            self.assertEqual(self.library.advance(row, True), MASK32)

    def test_seeded_history_survives_row_moves_and_pause(self):
        reference = self.row(2**60 + 31, 17)
        rows = [self.row(9, 3), self.row(2**60 + 31, 17), self.row(9, 3)]
        expected = [self.library.advance(reference, True) for _ in range(12)]
        actual = [self.library.advance(rows[1], True) for _ in range(4)]
        rows = [rows[source] for source in [1, 2, 0]]
        for _ in range(10):
            self.library.advance(rows[0], False)
            self.library.advance(rows[1], True)
        actual += [self.library.advance(rows[0], True) for _ in range(8)]
        self.assertEqual(actual, expected)

    def test_same_seed_has_no_slot_salt_and_high_bits_matter(self):
        first, second = self.row(37, 22), self.row(37, 22)
        self.assertEqual(
            [self.library.advance(first, True) for _ in range(20)],
            [self.library.advance(second, True) for _ in range(20)],
        )
        self.assertNotEqual(
            [seed_for_draw(37, position) for position in range(20)],
            [seed_for_draw(37 + 2**32, position) for position in range(20)],
        )

    def test_invalid_resets(self):
        for value in (-1, MASK64 + 1, 0.5):
            with self.assertRaises((TypeError, ValueError)):
                reset_words(1, value)
        for value in (0, 1_000_001):
            with self.assertRaises(ValueError):
                reset_words(None, 0, entropy_seed=value)


if __name__ == "__main__":
    unittest.main()
