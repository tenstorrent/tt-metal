# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The harness wheel (``tests/harness_wheel/make_wheel.py``): what it holds and how it is named.

Whether the harness *works* once installed is checked where the contract is:
``test_out_of_tree_contract.py`` runs the out-of-tree fixture on the built
wheel. The tests here pin the packaging itself.
"""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

_BUILDER = Path(__file__).resolve().parents[2] / "harness_wheel" / "make_wheel.py"
_spec = importlib.util.spec_from_file_location("build_wheel", _BUILDER)
build_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_wheel)

VERSION = "1.2.3+gtest"


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    return build_wheel.build_wheel(tmp_path_factory.mktemp("wheel"), VERSION)


@pytest.fixture(scope="module")
def names(wheel) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def test_wheel_is_named_for_its_version(wheel):
    assert wheel.name == f"tt_llk_harness-{VERSION}-py3-none-any.whl"


def test_wheel_holds_the_harness_packages(names):
    for expected in (
        "tt_llk_harness/__init__.py",
        "tt_llk_harness/plugin.py",
        "helpers/__init__.py",
        "helpers/test_config.py",
        "helpers/golden_generators.py",
        "helpers/stimuli_config.py",
        "helpers/stimuli_generator/__init__.py",
        "helpers/pack.py",
        "helpers/unpack.py",
        "helpers/llk_pytest_plugin.py",
        "ttsim_error.py",
    ):
        assert expected in names, f"{expected} is missing from the wheel"


def test_wheel_installs_no_package_another_wheel_owns(names):
    """``tt_llk_perf`` is a top-level package of the ttnn wheel; here it stays under ``llk_home``."""
    assert {name.split("/")[0] for name in names} == {
        "helpers",
        "tt_llk_harness",
        "ttsim_error.py",
        f"tt_llk_harness-{VERSION}.dist-info",
    }


def test_wheel_holds_the_tt_llk_files_the_harness_reads(names):
    prefix = build_wheel.LLK_HOME_PREFIX
    for expected in (
        f"{prefix}/tests/helpers/include/counters.h",
        f"{prefix}/tests/helpers/include/operand.h",
        f"{prefix}/tools/include/perf_counters/types.h",
        f"{prefix}/tests/perf_split_costs.json",
        f"{prefix}/tools/python/tt_llk_perf/__init__.py",
        f"tt_llk_harness-{VERSION}.dist-info/licenses/LICENSE",
        "tt_llk_harness/requirements.txt",
    ):
        assert expected in names, f"{expected} is missing from the wheel"


def test_wheel_holds_no_kernel_or_firmware_source(names):
    """The consumer supplies what gets compiled; the wheel is the harness only."""
    compiled = sorted(n for n in names if n.endswith((".cpp", ".ld", ".S", ".elf")))
    assert not compiled, compiled


def test_metadata_carries_the_version_and_the_requirements(wheel):
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(f"tt_llk_harness-{VERSION}.dist-info/METADATA").decode()
    assert f"Version: {VERSION}\n" in metadata
    pins = build_wheel.pinned_requirements()
    for name in build_wheel.RUNTIME_REQUIREMENTS:
        assert f"Requires-Dist: {name}\n" in metadata
        assert f'Requires-Dist: {pins[name]} ; extra == "pinned"\n' in metadata


def test_record_lists_every_file(wheel, names):
    with zipfile.ZipFile(wheel) as archive:
        record = archive.read(f"tt_llk_harness-{VERSION}.dist-info/RECORD").decode()
    recorded = {line.split(",")[0] for line in record.splitlines()}
    assert recorded == names


def test_same_sources_give_the_same_wheel(wheel, tmp_path):
    again = build_wheel.build_wheel(tmp_path, VERSION)
    assert again.read_bytes() == wheel.read_bytes()


def test_default_version_leads_with_the_api_version():
    """The wheel's version is read from the source; it must be the one the harness reports."""
    import tt_llk_harness

    assert build_wheel.api_version() == tt_llk_harness.HARNESS_API_VERSION
    major, minor = tt_llk_harness.HARNESS_API_VERSION
    assert build_wheel.default_version().startswith(f"{major}.{minor}.")


def test_a_version_that_is_not_a_wheel_version_is_rejected(tmp_path):
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        build_wheel.build_wheel(tmp_path, "nightly build")
