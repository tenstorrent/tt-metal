# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compile the exact kernel selection body without TTNN or device access."""
import ctypes
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tt.token_feedback import _SELECT_SOURCE


class TokenFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="qwen-feedback-cpu-")
        root = Path(cls.temp.name)
        source = root / "select.cpp"
        source.write_text(
            "#include <stdint.h>\n"
            + _SELECT_SOURCE
            + '\nextern "C" void run(uint32_t* c, uint32_t* s, uint32_t* a) { select_feedback(c,s,a); }\n'
        )
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(root / "select.so")], check=True
        )
        cls.library = ctypes.CDLL(str(root / "select.so"))
        cls.array = ctypes.c_uint32 * 32

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_unsigned_selection_and_sources(self):
        rng = random.Random(7)
        edges = [0, 1, 248319, 2**24 + 1, 2**31 - 1, 2**31, 2**32 - 2, 2**32 - 1]
        for repeat in range(100):
            current = edges * 4 if repeat == 0 else [rng.getrandbits(32) for _ in range(32)]
            saved = [rng.getrandbits(32) for _ in range(32)]
            for mask in ([0] * 32, [1] * 32, [i % 2 for i in range(32)], [rng.randrange(2) for _ in range(32)]):
                c, s, a = self.array(*current), self.array(*saved), self.array(*mask)
                self.library.run(c, s, a)
                self.assertEqual(list(c), [current[i] if mask[i] else saved[i] for i in range(32)])
                self.assertEqual(list(s), saved)
                self.assertEqual(list(a), mask)

    def test_alias_and_changed_membership(self):
        c = self.array(*range(32))
        a = self.array(*([0, 1] * 16))
        self.library.run(c, c, a)
        self.assertEqual(list(c), list(range(32)))
        saved = self.array(*range(100, 132))
        self.library.run(c, saved, a)
        a[:] = [1 - x for x in a]
        self.library.run(c, saved, a)
        self.assertEqual(list(c), list(saved))


if __name__ == "__main__":
    unittest.main()
