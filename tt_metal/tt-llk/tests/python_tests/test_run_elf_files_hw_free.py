# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free test of TestConfig.run_elf_files and its TRISC image cache.

The device calls are stubbed and recorded. The cache (LAST_LOADED_ELFS) must name a
variant only while all of that variant's ELFs are on the core: a variant with a
missing thread ELF is refused before the core is touched, and a load that fails part
way makes the next run load every thread again, so no core starts on the code a
previous variant left in a thread's memory.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import test_config as test_config_module
from helpers.chip_architecture import ChipArchitecture
from helpers.device import BootMode
from helpers.test_config import TestConfig

THREADS = ["unpack", "math", "pack"]


class _Core:
    """Records the device calls run_elf_files makes and fails the loads it is told to."""

    def __init__(self):
        self.events = []
        self.fail_loads = set()

    def load_elf(self, elf_file, risc_name, **kwargs):
        path = Path(elf_file)
        if not path.is_file():
            raise RuntimeError(f"ELF file {elf_file} does not exist.")
        if path.name in self.fail_loads:
            raise RuntimeError(f"load of {elf_file} failed")
        self.events.append(("load", risc_name, path.parent.parent.name))
        return 0

    def brisc_command(self, location, command, **kwargs):
        self.events.append(("brisc", command.name))

    def loads(self):
        return [event for event in self.events if event[0] == "load"]

    def starts(self):
        return [event[1] for event in self.events if event[0] == "brisc"]


@pytest.fixture(params=[ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE])
def core(request, monkeypatch, tmp_path):
    core = _Core()
    monkeypatch.setattr(test_config_module, "load_elf", core.load_elf)
    monkeypatch.setattr(test_config_module, "commit_brisc_command", core.brisc_command)
    monkeypatch.setattr(
        test_config_module, "commit_tensix_soft_reset", lambda *a, **k: None
    )
    monkeypatch.setattr(
        test_config_module, "write_words_to_device", lambda *a, **k: None
    )
    monkeypatch.setattr(TestConfig, "CHIP_ARCH", request.param, raising=False)
    monkeypatch.setattr(TestConfig, "ARTEFACTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(TestConfig, "KERNEL_COMPONENTS", THREADS)
    monkeypatch.setattr(TestConfig, "TENSIX_LOCATION", "0,0", raising=False)
    monkeypatch.setattr(TestConfig, "TRISC_START_ADDRS", [0, 0, 0], raising=False)
    monkeypatch.setattr(TestConfig.TEST_TARGET, "run_simulator", False)
    monkeypatch.setattr(TestConfig, "DEVICE_PRINT_ENABLED", False, raising=False)
    monkeypatch.setattr(TestConfig, "BRISC_ELF_LOADED", True)
    monkeypatch.setattr(TestConfig, "LAST_LOADED_ELFS", Path())
    core.root = tmp_path
    core.arch = request.param
    return core


def _variant(core, variant_id, threads=THREADS):
    elf_dir = core.root / "test" / variant_id / "elf"
    elf_dir.mkdir(parents=True)
    for thread in threads:
        (elf_dir / f"{thread}.elf").write_bytes(b"\x7fELF")
    return SimpleNamespace(
        boot_mode=BootMode.DEFAULT,
        requires_device_print=False,
        test_name="test",
        variant_id=variant_id,
    )


def _run(variant):
    TestConfig.run_elf_files(variant)


def test_complete_variant_is_loaded_once_and_started_every_run(core):
    variant = _variant(core, "a")
    _run(variant)
    _run(variant)

    assert core.loads() == [
        ("load", "trisc0", "a"),
        ("load", "trisc1", "a"),
        ("load", "trisc2", "a"),
    ]
    assert core.starts()[-1] == "START_TRISCS"
    assert TestConfig.LAST_LOADED_ELFS == core.root / "test" / "a" / "elf"


def test_variant_without_a_thread_elf_is_refused_on_every_node(core):
    # Two nodes of one variant whose pack thread did not compile
    variant = _variant(core, "a", threads=["unpack", "math"])
    for _ in range(2):
        with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
            FileNotFoundError, match="pack.elf"
        ):
            _run(variant)

    assert core.events == []
    assert TestConfig.LAST_LOADED_ELFS == Path()


def test_refused_variant_keeps_the_previous_variant_loaded(core):
    first = _variant(core, "a")
    _run(first)
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        FileNotFoundError
    ):
        _run(_variant(core, "b", threads=["unpack"]))
    _run(first)

    assert [event[2] for event in core.loads()] == ["a", "a", "a"]


def test_failed_load_makes_the_next_run_load_every_thread(core):
    variant = _variant(core, "a")
    core.fail_loads = {"pack.elf"}
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError
    ):
        _run(variant)
    assert TestConfig.LAST_LOADED_ELFS == Path()

    core.fail_loads = set()
    core.events.clear()
    _run(variant)
    assert [event[1] for event in core.loads()] == ["trisc0", "trisc1", "trisc2"]


def test_failed_load_of_another_variant_reloads_the_first(core):
    first = _variant(core, "a")
    _run(first)
    core.fail_loads = {"math.elf"}
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError
    ):
        _run(_variant(core, "b"))

    # trisc0 now holds b's code, so a's next node must load again
    core.fail_loads = set()
    core.events.clear()
    _run(first)
    assert core.loads() == [
        ("load", "trisc0", "a"),
        ("load", "trisc1", "a"),
        ("load", "trisc2", "a"),
    ]
