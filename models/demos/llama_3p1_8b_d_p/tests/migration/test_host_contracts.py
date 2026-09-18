"""Run standalone stdlib contracts in isolated processes during repository pytest."""

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = {"writer_boundaries": 17, "runtime_edges": 37, "native_ranges": 57, "native_cancel": 46, "native_capacity": 41}


def run_checks(scenario):
    import importlib.abc
    import unittest

    class BlockNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "numpy", "ttnn", "tt_lib", "tt_d_gen", "transformers"}:
                raise RuntimeError("native/model import forbidden in host checks: " + fullname)
            return None

    directory = HERE / scenario
    sys.path.insert(0, str(directory))
    sys.meta_path.insert(0, BlockNative())
    suite = unittest.defaultTestLoader.discover(str(directory), pattern="checks_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun == SUITES[scenario] else 1


def collect_writer():
    """Collect the real entry with inert modules; executing any device function fails."""
    import importlib.abc
    import types

    import pytest

    def forbidden(*args, **kwargs):
        raise AssertionError("device/model call during collect-only check")

    def stub(name, **attributes):
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
        return module

    stub("numpy")
    stub("torch")
    stub("loguru", logger=types.SimpleNamespace())
    stub("ttnn", __path__=[], FabricConfig=types.SimpleNamespace(FABRIC_1D_RING="ring"))
    stub("ttnn.device", is_blackhole=lambda: True)
    prefix = "models.demos.llama_3p1_8b_d_p.tt."
    stub(prefix + "config", MeshConfig=forbidden)
    stub(prefix + "kv_cache", allocate_kv_cache=forbidden, write_kv_chunk=forbidden)
    stub(prefix + "runners.kv_chunk_table", build_kv_chunk_address_table=forbidden)

    class BlockRealNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "numpy", "ttnn", "tt_lib", "tt_d_gen", "transformers"}:
                raise RuntimeError("real native import during stub collection: " + fullname)
            return None

    sys.meta_path.insert(0, BlockRealNative())
    seen = []

    class CheckCollectedHelpers:
        def pytest_collection_modifyitems(self, items):
            assert len(items) == 1
            module = items[0].module
            directory = HERE / "writer_boundaries"
            assert Path(module.page_io.__file__).resolve() == directory / "page_io.py"
            oracle = sys.modules[module.independent_tensor_location.__module__]
            assert Path(oracle.__file__).resolve() == directory / "kv_table_oracle.py"
            contract = sys.modules[module.PageKey.__module__]
            assert Path(contract.__file__).resolve() == directory / "writer_boundaries.py"
            seen.append(items[0].nodeid)

        def pytest_runtest_call(self, item):
            forbidden()

    result = pytest.main(
        [
            "--noconftest",
            "--collect-only",
            "--import-mode=importlib",
            "-o",
            "addopts=",
            "-q",
            str(HERE / "writer_boundaries/test_writer_boundaries_device.py"),
        ],
        plugins=[CheckCollectedHelpers()],
    )
    return 0 if result == 0 and len(seen) == 1 else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--suite", choices=tuple(SUITES))
    mode.add_argument("--collect-writer", action="store_true")
    args = parser.parse_args()
    raise SystemExit(collect_writer() if args.collect_writer else run_checks(args.suite))
else:
    import pytest

    # Local helper names and import blockers belong only to the child. Repository
    # collection must neither import a scenario nor alter later device-test imports.
    @pytest.mark.parametrize("scenario", tuple(SUITES))
    def test_host_contracts(scenario):
        env = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), "--suite", scenario],
            cwd=HERE / scenario,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        output = completed.stdout + completed.stderr
        print(output)
        assert completed.returncode == 0, output
        assert re.search(r"Ran " + str(SUITES[scenario]) + r" tests in ", output), output

    # Importlib collection must resolve the writer's actual sibling helpers from
    # outside that directory. All heavy modules are inert stubs in this child.
    def test_writer_device_collection_uses_package_helpers():
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--collect-writer"],
            cwd=HERE.parents[4],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        output = completed.stdout + completed.stderr
        print(output)
        assert completed.returncode == 0, output
        assert "1 test collected" in output, output
