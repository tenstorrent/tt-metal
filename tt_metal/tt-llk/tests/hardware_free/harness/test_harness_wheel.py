# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The harness wheel (``tests/harness_wheel/make_wheel.py``): what it holds and how it is named.

Whether the harness *works* once installed is checked by
``test_out_of_tree_contract.py``, which runs the out-of-tree fixture on the
built wheel. The tests here pin the packaging itself.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import io
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

import pytest

_BUILDER = Path(__file__).resolve().parents[2] / "harness_wheel" / "make_wheel.py"
_spec = importlib.util.spec_from_file_location("build_wheel", _BUILDER)
build_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_wheel)

VERSION = build_wheel.Version(number=(1, 2, 3), committed=0, commit=0xABC << 148)
DIST_INFO = f"tt_llk_harness-{VERSION}.dist-info"


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    return build_wheel.build_wheel(tmp_path_factory.mktemp("wheel"), VERSION)


@pytest.fixture(scope="module")
def files(wheel) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_wheel_holds_what_an_installed_harness_needs(wheel, files):
    assert wheel.name == f"tt_llk_harness-{VERSION}-py3-none-any.whl"
    prefix = build_wheel.LLK_HOME_PREFIX
    for expected in (
        "tt_llk_harness/__init__.py",
        "tt_llk_harness/plugin.py",
        "tt_llk_harness/VERSION",
        "tt_llk_harness/requirements.txt",
        "helpers/test_config.py",
        "helpers/stimuli_generator/__init__.py",
        "helpers/llk_pytest_plugin.py",
        "ttsim_error.py",
        f"{prefix}/tests/helpers/include/counters.h",
        f"{prefix}/tests/helpers/src/trisc.cpp",
        f"{prefix}/tests/helpers/ld/sections.ld",
        f"{prefix}/tools/include/perf_counters/types.h",
        f"{prefix}/tests/perf_split_costs.json",
        f"{prefix}/tools/python/tt_llk_perf/__init__.py",
        f"{DIST_INFO}/licenses/LICENSE",
    ):
        assert expected in files, f"{expected} is missing from the wheel"


def test_wheel_holds_nothing_it_must_not(files):
    """No top-level package another wheel owns (``tt_llk_perf`` is ttnn's), and no toolchain or pin of one:
    SFPI is released far more often than the harness changes, and its pin belongs to the LLK tree.
    """
    assert {PurePosixPath(name).parts[0] for name in files} == {
        "helpers",
        "tt_llk_harness",
        "ttsim_error.py",
        DIST_INFO,
    }
    toolchain_files = {
        "sfpi-version",
        "sfpi-info.sh",
        "sfpi.version",
        "install_sfpi.sh",
    }
    carried = [
        name
        for name in files
        if "sfpi" in PurePosixPath(name).parts
        or PurePosixPath(name).name in toolchain_files
    ]
    assert not carried, carried


def test_metadata_and_record_describe_the_wheel(files):
    metadata = BytesParser().parsebytes(files[f"{DIST_INFO}/METADATA"])
    assert metadata["Version"] == str(VERSION)
    assert metadata.get_all("Requires-Dist") == list(build_wheel.RUNTIME_REQUIREMENTS)

    record = io.StringIO(files[f"{DIST_INFO}/RECORD"].decode())
    assert {path for path, _hash, _size in csv.reader(record)} == set(files)


def test_same_sources_give_the_same_wheel(wheel, tmp_path):
    assert build_wheel.build_wheel(tmp_path, VERSION).read_bytes() == wheel.read_bytes()


def test_a_release_is_named_by_its_number_and_any_other_build_by_its_commit():
    """``X.Y.Z`` is what a suite pins; any other build orders by commit time and cannot pass for a release."""
    assert str(VERSION._replace(release=True)) == "1.2.3"
    assert str(VERSION) == "1.2.3+197001010000.gabc000000000"
    later = VERSION._replace(committed=60, commit=0, dirty=True)
    assert str(later) == "1.2.3+197001010001.g000000000000.dirty"


@pytest.mark.parametrize(
    "text", ["1.2", "1.2.3.4", "1.2.x", "v1.2.3", "1.02.3", "1.2.-3", "1.2.3+g0", ""]
)
def test_a_version_that_is_not_three_plain_numbers_is_refused(text):
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        build_wheel.parse_version_number(text)


def test_the_builder_and_the_harness_report_one_version():
    """The builder reads the VERSION file without importing the harness; both must say the same."""
    import tt_llk_harness

    number = build_wheel.default_version().number
    assert ".".join(map(str, number)) == tt_llk_harness.__version__


def test_a_release_is_not_built_from_a_tree_with_changes(monkeypatch):
    """A published version must be reproducible from its commit."""
    real_run = build_wheel.subprocess.run

    def run_with_a_change(argv, **options):
        result = real_run(argv, **options)
        if argv[:2] == ["git", "status"]:
            result.stdout = " M tests/python_tests/helpers/utils.py\n"
        return result

    monkeypatch.setattr(build_wheel.subprocess, "run", run_with_a_change)

    assert build_wheel.default_version().dirty
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="uncommitted changes"
    ):
        build_wheel.default_version(release=True)


# The harness module that imports an in-tree test module, so it works from a checkout only.
_IN_TREE_ONLY_IMPORTS = {"helpers/bf16_sweep.py": {"test_eltwise_unary_sfpu"}}


def _imported(source: bytes) -> set[str]:
    """Top-level names of the absolute imports of a module, wherever in it they are."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.partition(".")[0])
    return names


def test_wheel_requires_what_its_python_imports(files):
    """Every test here runs where all of ``tests/requirements.txt`` is installed, so a requirement the wheel
    does not declare breaks nothing until somebody installs the wheel on its own."""
    shipped = {
        *(package.name for package in build_wheel.PACKAGES.values()),
        *(module.stem for module in build_wheel.MODULES.values()),
        "tt_llk_perf",
    }
    requirements = build_wheel.RUNTIME_REQUIREMENTS
    declared = {module for modules in requirements.values() for module in modules}
    imported = set()
    for name, source in files.items():
        if name.endswith(".py"):
            imported |= _imported(source) - _IN_TREE_ONLY_IMPORTS.get(name, set())
    third_party = imported - set(sys.stdlib_module_names) - shipped

    assert sorted(third_party - declared) == [], "imported, but not required"
    assert sorted(declared - third_party) == [], "required, but no longer imported"
    # No import shows this one: the plugin's report fixtures take pytest-xdist's ``worker_id``.
    assert "pytest-xdist" in requirements
