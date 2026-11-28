# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import count_pytests

PARAMETRIZED_MODULE = """
import pytest

@pytest.mark.parametrize("x", [1, 2, 3])
def test_param(x):
    assert x
"""

NESTED_HELPER = """
def helper():
    def test_nested_never_collected():
        pass
    return test_nested_never_collected
"""

NON_TEST_CLASS = """
class Helper:
    def test_method_on_non_test_class(self):
        pass
"""

TEST_CLASS = """
class TestGroup:
    def test_one(self):
        pass

    def test_two(self):
        pass
"""

STAR_TEST_SUFFIX = """
def test_from_star_test_suffix():
    assert True

def test_second_from_star_test_suffix():
    assert True
"""


class CountPytestsTests(unittest.TestCase):
    def _count(self, files):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, body in files.items():
                (root / name).write_text(body, encoding="utf-8")
            return count_pytests.count_tests_in_directory(str(root))

    def test_counts_parametrize_expansion(self):
        self.assertEqual(self._count({"test_a.py": PARAMETRIZED_MODULE}), 3)

    def test_ignores_test_helper_nested_in_another_function(self):
        self.assertEqual(self._count({"test_a.py": NESTED_HELPER}), 0)

    def test_ignores_method_on_class_pytest_does_not_collect(self):
        self.assertEqual(self._count({"test_a.py": NON_TEST_CLASS}), 0)

    def test_counts_methods_of_Test_classes(self):
        self.assertEqual(self._count({"test_a.py": TEST_CLASS}), 2)

    def test_counts_files_with_test_suffix(self):
        self.assertEqual(self._count({"misc_test.py": STAR_TEST_SUFFIX}), 2)

    def test_does_not_double_count_a_file_matching_both_patterns(self):
        self.assertEqual(self._count({"test_test.py": PARAMETRIZED_MODULE}), 3)


if __name__ == "__main__":
    unittest.main()
