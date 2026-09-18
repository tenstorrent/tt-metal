"""Only stdlib tests. No Torch, native binaries or device operations."""
import importlib.abc
import sys
import unittest
from pathlib import Path


class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "ttnn", "tt_lib", "tt_d_gen", "transformers"}:
            raise RuntimeError("blocked native/model import: " + fullname)
        return None


sys.meta_path.insert(0, BlockNative())
suite = (
    unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
    if len(sys.argv) > 1
    else unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="checks_*.py")
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
