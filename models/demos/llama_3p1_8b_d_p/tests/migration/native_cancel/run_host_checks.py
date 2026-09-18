"""Standalone stdlib checks; native blocking is confined to this child process."""

import importlib.abc
import sys
import unittest
from pathlib import Path


class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch",
            "numpy",
            "ttnn",
            "_ttnn",
            "_ttnncpp",
            "tt_lib",
            "tt_d_gen",
            "transformers",
        }:
            raise AssertionError("native/model import forbidden in host checks: " + fullname)
        return None


def main():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    sys.meta_path.insert(0, BlockNative())
    suite = unittest.defaultTestLoader.discover(str(here), pattern="checks_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun == 46 else 1


if __name__ == "__main__":
    raise SystemExit(main())
