# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LLK harness backend: detours are poked straight into device L1.

This works because the harness loads each thread ELF once and reuses it, so a
word written into L1 survives until something reloads the image. Everything in
here is specific to that harness — `TestConfig`, ttexalens and the soft reset —
which is why it lives in its own module: importing it is what selects it, and a
Metal sweep must never pay for `helpers` or contend with metal over the device.
"""

from contextlib import contextmanager

import pytest
import scanner
from cave import Injector
from helpers.device import (
    LLKAssertException,
    commit_tensix_soft_reset,
    set_tensix_soft_reset,
)
from helpers.logger import logger
from helpers.test_config import TestConfig
from ttexalens.tt_exalens_lib import read_word_from_device, write_words_to_device

# pytest's outcome exceptions derive from BaseException, so they need naming
# explicitly or they would escape the per-variant handler and fail the whole item.
_FAILED = pytest.fail.Exception


@contextmanager
def quiet_harness():
    """Mute the harness's own logging for the duration of a variant.

    Variants are *meant* to fail, and the harness answers every mismatch with a
    colour dump of the offending tiles. A hundred of those is the whole console.
    The baseline pass runs outside this, so a genuinely broken test still says so.
    """
    logger.disable("helpers")
    try:
        yield
    finally:
        logger.enable("helpers")


class _Baseline:
    """What the clean pass told us about this test."""

    def __init__(self):
        self.config = None
        self.saw_run = False
        self.had_result = False
        self.saw_load = False

    def worth_sweeping(self) -> bool:
        # A test that never loaded ELFs has nothing to perturb, and one whose run()
        # produced no result has no golden to mismatch against.
        if self.config is None or not self.saw_load:
            return False
        return self.had_result or not self.saw_run


class LLKBackend:
    name = "llk"
    # The LLK harness builds one kernel per test case, so the case name already
    # identifies it and there is nothing extra to record.
    kernel = ""

    def __init__(self, max_delay: int):
        self.max_delay = max_delay
        self.baseline = _Baseline()
        self._injector = None

    # -- device plumbing ---------------------------------------------------

    def _injector_for_device(self) -> Injector:
        if self._injector is None:
            location = TestConfig.TENSIX_LOCATION
            self._injector = Injector(
                read_words=lambda addr, count: [
                    read_word_from_device(location, addr + 4 * i) for i in range(count)
                ],
                write_words=lambda addr, words: write_words_to_device(
                    location, addr, words
                ),
                max_delay=self.max_delay,
            )
        return self._injector

    def injector_for(self, thread: str) -> Injector:
        # All three ELFs live in one core's L1 at distinct addresses, so one
        # injector over one address space serves every thread.
        return self._injector_for_device()

    def restore(self) -> None:
        self._injector_for_device().restore()

    def _forget_kernel_image(self) -> None:
        """Make the harness re-flash the TRISCs, and stop trusting our view of L1.

        BRISC is deliberately left alone: it is mid command loop, and re-flashing it
        from under itself is what stops it answering at all.
        """
        TestConfig.LAST_LOADED_ELFS = None
        self._injector_for_device().forget()

    # -- baseline capture --------------------------------------------------

    def watch_baseline(self):
        """Wrap both loaders. Suites like SDPA never call run(), only run_elf_files()."""
        original_run, original_load = TestConfig.run, TestConfig.run_elf_files
        self.baseline = baseline = _Baseline()

        def wrapped_run(config_self, *args, **kwargs):
            baseline.config = baseline.config or config_self
            baseline.saw_run = True
            outcome = original_run(config_self, *args, **kwargs)
            baseline.had_result = (
                baseline.had_result or getattr(outcome, "result", None) is not None
            )
            return outcome

        def wrapped_load(config_self, *args, **kwargs):
            baseline.config = baseline.config or config_self
            baseline.saw_load = True
            return original_load(config_self, *args, **kwargs)

        def unwatch():
            TestConfig.run, TestConfig.run_elf_files = original_run, original_load

        TestConfig.run, TestConfig.run_elf_files = wrapped_run, wrapped_load
        return unwatch

    def ready(self) -> bool:
        return self.baseline.worth_sweeping()

    # -- the sweep ---------------------------------------------------------

    def scans(self, site_mode: str) -> dict:
        return {
            thread: scanner.scan(path, site_mode)
            for thread, path in zip(
                TestConfig.KERNEL_COMPONENTS, self.baseline.config.temp_elfs
            )
        }

    def quiet(self):
        return quiet_harness()

    def classify(self, err) -> tuple:
        if isinstance(err, TimeoutError):
            return "hang", str(err)
        if isinstance(err, LLKAssertException):
            return "assert", str(err)
        if isinstance(err, (AssertionError, _FAILED)):
            return "mismatch", str(err)
        return "error", f"{type(err).__name__}: {err}"

    def recover(self, replay) -> bool:
        """Soft reset after a hang. False means the device needs a manual reset."""
        location = TestConfig.TENSIX_LOCATION
        try:
            self._injector_for_device().restore()
        except Exception:
            pass
        # Soft reset takes BRISC down with it, so its image really is stale here.
        self._forget_kernel_image()
        TestConfig.BRISC_ELF_LOADED = False
        for _ in range(3):
            try:
                commit_tensix_soft_reset(1, location=location)
                break
            except TimeoutError:
                # commit_ polls for an exact readback, which a wedged core can miss.
                # Re-assert without polling and give it another go.
                set_tensix_soft_reset(1, location=location)
        else:
            return False
        # Reload a clean image (no nested sweep) so the next arm() sees the scan's words.
        try:
            with quiet_harness():
                replay()
        except Exception:
            pass
        self._injector_for_device().forget()
        return True

    def finish(self) -> None:
        # The cave bytes outlive the restore, so the next case must not be handed
        # an L1 image the harness still believes is pristine.
        self._forget_kernel_image()
