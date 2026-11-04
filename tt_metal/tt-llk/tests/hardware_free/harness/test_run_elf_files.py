# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free test: LAST_LOADED_ELFS names a variant only while all its ELFs are on the core."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import test_config as test_config_module
from helpers.chip_architecture import ChipArchitecture
from helpers.device import (
    KERNEL_COMPLETE,
    TRISC_CORES,
    BootMode,
    RiscCore,
    get_soft_reset_mask,
)
from helpers.llk_params import Mailboxes
from helpers.test_config import TestConfig

THREADS = ["unpack", "math", "pack"]


LAUNCH_DONE = 0x1000


class _Core:
    """Records the device calls run_elf_files makes and fails the loads and writes it is told to."""

    def __init__(self):
        self.events = []
        self.fail_loads = set()
        self.fail_writes = set()

    def load_elf(self, elf_file, risc_name, **kwargs):
        path = Path(elf_file)
        if not path.is_file():
            raise RuntimeError(f"ELF file {elf_file} does not exist.")
        if path.name in self.fail_loads:
            raise RuntimeError(f"load of {elf_file} failed")
        self.events.append(("load", risc_name, path.parent.parent.name))
        return 0

    def write_words(self, location, addr, values, **kwargs):
        if addr in self.fail_writes:
            raise TimeoutError(f"write to {addr:#x} timed out")
        self.events.append(("write", addr, tuple(values)))

    def release(self, value, cores, location=None, **kwargs):
        if value == 0:
            self.events.append(("release", tuple(core.name for core in cores)))

    def reset(self, value, location=None, **kwargs):
        self.events.append(("reset", value))

    def loads(self):
        return [event for event in self.events if event[0] == "load"]

    def starts(self):
        return [event[1] for event in self.events if event[0] == "release"]

    def start_address_writes(self):
        return [event for event in self.events if event[0] == "write" and event[1] == 0]


@pytest.fixture(params=[ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE])
def core(request, monkeypatch, tmp_path):
    core = _Core()
    monkeypatch.setattr(test_config_module, "load_elf", core.load_elf)
    monkeypatch.setattr(test_config_module, "set_tensix_soft_reset", core.release)
    monkeypatch.setattr(test_config_module, "commit_tensix_soft_reset", core.reset)
    monkeypatch.setattr(test_config_module, "write_words_to_device", core.write_words)
    monkeypatch.setattr(
        test_config_module.device_module, "launch_done_address", lambda: LAUNCH_DONE
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
    variant = SimpleNamespace(
        boot_mode=BootMode.DEFAULT,
        requires_device_print=False,
        test_name="test",
        variant_id=variant_id,
    )
    variant.resolved_boot_mode = lambda: TestConfig.resolved_boot_mode(variant)
    return variant


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
    assert core.starts() == [("BRISC",), ("BRISC",)]
    assert TestConfig.LAST_LOADED_ELFS == core.root / "test" / "a" / "elf"


def test_variant_without_a_thread_elf_is_refused_on_every_node(core):
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


@pytest.mark.parametrize("core", [ChipArchitecture.WORMHOLE], indirect=True)
def test_failed_start_address_write_keeps_the_variant_loaded(core):
    # Wormhole TRISCs clear their start addresses after reading them, so the host writes them for every launch
    variant = _variant(core, "a")
    core.fail_writes = {0}
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        TimeoutError
    ):
        _run(variant)
    assert TestConfig.LAST_LOADED_ELFS == core.root / "test" / "a" / "elf"

    core.fail_writes = set()
    core.events.clear()
    _run(variant)
    assert core.loads() == []
    assert len(core.start_address_writes()) == 1
    assert core.starts() == [("BRISC",)]


def test_launch_resets_loads_clears_launch_done_then_releases_brisc(core):
    _run(_variant(core, "a"))

    start_write = (
        [("write", 0, (0, 0, 0))] if core.arch == ChipArchitecture.WORMHOLE else []
    )
    assert core.events == [
        ("reset", 1),
        ("load", "trisc0", "a"),
        ("load", "trisc1", "a"),
        ("load", "trisc2", "a"),
        *start_write,
        ("write", LAUNCH_DONE, (0,)),
        ("release", ("BRISC",)),
    ]


def test_simulator_launch_has_no_launch_done_register(core, monkeypatch):
    monkeypatch.setattr(
        test_config_module.device_module, "launch_done_address", lambda: None
    )
    _run(_variant(core, "a"))

    assert [
        event
        for event in core.events
        if event[0] == "write" and event[1] == LAUNCH_DONE
    ] == []
    assert core.starts() == [("BRISC",)]


class _Signals:
    """The registers and mailboxes the host wait reads, played back in order (the last value repeats)."""

    def __init__(self, launch_done, soft_reset, mailboxes=(KERNEL_COMPLETE,) * 3):
        self.launch_done, self.soft_reset, self.mailboxes = (
            list(launch_done),
            list(soft_reset),
            mailboxes,
        )
        self.reads = []

    def _next(self, values):
        return values.pop(0) if len(values) > 1 else values[0]

    def read_word(self, location, addr, **kwargs):
        assert addr == LAUNCH_DONE
        self.reads.append("launch_done")
        return self._next(self.launch_done)

    def soft_reset_bits(self, location="0,0"):
        self.reads.append("soft_reset")
        return self._next(self.soft_reset)

    def read_mailboxes(self, location, addr, num_bytes):
        return np.array(self.mailboxes, dtype=np.uint32).tobytes()


@pytest.fixture
def signals(core, monkeypatch):
    def install(*args, **kwargs):
        sig = _Signals(*args, **kwargs)
        monkeypatch.setattr(test_config_module.device_module, "Mailboxes", Mailboxes)
        monkeypatch.setattr(test_config_module, "read_word_from_device", sig.read_word)
        monkeypatch.setattr(test_config_module, "soft_reset_bits", sig.soft_reset_bits)
        monkeypatch.setattr(test_config_module, "read_from_device", sig.read_mailboxes)
        monkeypatch.setattr(
            test_config_module, "handle_if_assert_hit", lambda *a, **k: None
        )
        return sig

    return install


def _wait(core, timeout=2):
    variant = _variant(core, "a")
    variant.temp_elfs = []
    TestConfig.wait_for_tensix_operations_finished(variant, timeout=timeout)


IN_RESET = get_soft_reset_mask([RiscCore.BRISC, *TRISC_CORES])


def test_wait_reads_soft_reset_only_after_launch_done(core, signals):
    sig = signals(launch_done=[0, 0, KERNEL_COMPLETE], soft_reset=[IN_RESET])
    _wait(core)

    assert sig.reads == ["launch_done", "launch_done", "launch_done", "soft_reset"]


def test_wait_polls_until_brisc_has_put_all_four_cores_in_reset(core, signals):
    sig = signals(launch_done=[KERNEL_COMPLETE], soft_reset=[0, 0, IN_RESET])
    _wait(core)

    assert sig.reads.count("soft_reset") == 3


def test_wait_times_out_when_brisc_never_puts_the_cores_in_reset(core, signals):
    signals(launch_done=[KERNEL_COMPLETE], soft_reset=[0])
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        TimeoutError, match="BRISC"
    ):
        _wait(core, timeout=0.2)


def test_wait_fails_when_brisc_resets_a_thread_that_did_not_finish(core, signals):
    signals(
        launch_done=[KERNEL_COMPLETE],
        soft_reset=[IN_RESET],
        mailboxes=(KERNEL_COMPLETE, 0, KERNEL_COMPLETE),
    )
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="Math"
    ):
        _wait(core)
