# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The harness wheel (``tests/harness_wheel/make_wheel.py``): what it holds and how it is named.

Whether the harness *works* once installed is checked where the contract is:
``test_out_of_tree_contract.py`` runs the out-of-tree fixture on the built
wheel. The tests here pin the packaging itself.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

_BUILDER = Path(__file__).resolve().parents[2] / "harness_wheel" / "make_wheel.py"
_spec = importlib.util.spec_from_file_location("build_wheel", _BUILDER)
build_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_wheel)

VERSION = build_wheel.Version(api=(1, 2), committed=0, commit=0xABC << 148)


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
        f"{prefix}/tests/helpers/src/trisc.cpp",
        f"{prefix}/tests/helpers/ld/sections.ld",
        f"{prefix}/tools/include/perf_counters/types.h",
        f"{prefix}/tests/perf_split_costs.json",
        f"{prefix}/tools/python/tt_llk_perf/__init__.py",
        f"tt_llk_harness-{VERSION}.dist-info/licenses/LICENSE",
        "tt_llk_harness/requirements.txt",
    ):
        assert expected in names, f"{expected} is missing from the wheel"


def test_version_is_api_then_commit_time_then_commit():
    assert str(VERSION) == "1.2.197001010000+gabc000000000"
    later = build_wheel.Version(api=(1, 2), committed=60, commit=0, dirty=True)
    assert str(later) == "1.2.197001010001+g000000000000.dirty"


def test_metadata_carries_the_version_and_the_requirements(wheel):
    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(
            archive.read(f"tt_llk_harness-{VERSION}.dist-info/METADATA")
        )
    assert metadata["Version"] == str(VERSION)
    assert metadata.get_all("Requires-Dist") == list(build_wheel.RUNTIME_REQUIREMENTS)


def test_record_lists_every_file(wheel, names):
    with zipfile.ZipFile(wheel) as archive:
        record = archive.read(f"tt_llk_harness-{VERSION}.dist-info/RECORD").decode()
    recorded = {path for path, _hash, _size in csv.reader(io.StringIO(record))}
    assert recorded == names


def test_same_sources_give_the_same_wheel(wheel, tmp_path):
    again = build_wheel.build_wheel(tmp_path, VERSION)
    assert again.read_bytes() == wheel.read_bytes()


def test_default_version_leads_with_the_api_version():
    """The wheel's version is read from the source; it must be the one the harness reports."""
    import tt_llk_harness

    assert build_wheel.api_version() == tt_llk_harness.HARNESS_API_VERSION
    assert build_wheel.default_version().api == tt_llk_harness.HARNESS_API_VERSION
