# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin that sweeps detours over an LLK test.

Every collected test runs once clean. If that pass loaded ELFs and had something
to compare, the same test body is then re-run once per planned variant with a
detour armed, and anything that stops passing is recorded.

Load it with `-p ttnop_plugin` and this directory on PYTHONPATH.
"""

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import heartbeat
import pytest
import report
import scanner
import sweep as sweep_module
import torch
from cave import DetourError, Injector
from helpers.device import LLKAssertException
from helpers.logger import logger
from helpers.test_config import TestConfig
from ttexalens.tt_exalens_lib import read_word_from_device, write_words_to_device

# pytest's outcome exceptions derive from BaseException, so they need naming
# explicitly or they would escape the per-variant handler and fail the whole item.
_SKIPPED = (pytest.skip.Exception, pytest.xfail.Exception)
_FAILED = pytest.fail.Exception

_writer = None
# Set once a hang has asked the supervisor for recovery, so the worker stops
# after the case it is reporting rather than in the middle of it.
_parked = False


def _hb() -> heartbeat.Writer:
    """Return this process's progress writer."""
    global _writer
    if _writer is None:
        _writer = heartbeat.Writer()
    return _writer


def _hang_closes_case(nodeid: str, variant: str, skip_family: bool = True) -> None:
    """Close a case out on a hang, and ask to be moved off the core it cost us."""
    global _parked
    # Do not retry a case that just hung a core. The pytest failure keeps the result.
    _hb().mark_done(nodeid)
    _hb().request_recovery(nodeid, variant, skip_family=skip_family)
    # Nobody is watching an unsupervised run, so there is no recovery coming and
    # nothing to wait for.
    _parked = _hb().enabled


def _park() -> None:
    """Stop taking cases and wait for the supervisor to kill this worker.

    Any new case on the hung core would fail without getting a real run. Wait
    here until the supervisor removes this worker.

    The DONE beat first, so the silence that follows is read as a worker with
    nothing left to do rather than one that stopped answering.
    """
    _hb().finish()
    while True:
        time.sleep(60)


@contextmanager
def quiet_harness():
    """Mute the harness's own logging for the duration of a run."""
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


def _test_kwargs(item) -> dict:
    """Fixture values for calling the test body directly. Fixed for the whole sweep."""
    names = getattr(item._fixtureinfo, "argnames", ()) or ()
    return {name: item.funcargs[name] for name in names if name in item.funcargs}


class Perturber:
    def __init__(self, config: sweep_module.Config):
        self.config = config
        self.verbose = os.environ.get("TTNOP_VERBOSE", "") not in ("", "0")
        self.baseline = _Baseline()
        self.scans = {}
        self._item = None
        self._kwargs = {}
        self._injector = None
        self._rng_state = None
        # Result of every run() the body just made, and the same list from the
        # clean pass to compare it against.
        self._results = []
        self._baseline_results = None
        self._comparable = True
        self._last_pcc = None

    def begin(self) -> None:
        """Reset the per-test state and remember the RNG the baseline will draw from.

        This runs after the seed fixture. Restoring this state gives every
        variant the same input as the baseline.
        """
        self.baseline = _Baseline()
        self._results = []
        self._baseline_results = None
        self._comparable = True
        self._last_pcc = None
        self._rng_state = torch.get_rng_state() if self.config.drift else None

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
                max_delay=self.config.max_delay,
            )
        return self._injector

    def _forget_kernel_image(self) -> None:
        """Make the harness re-flash the TRISCs, and stop trusting our view of L1.

        BRISC is left alone because it is already running its command loop.
        """
        TestConfig.LAST_LOADED_ELFS = None
        self._injector_for_device().forget()

    # -- baseline capture --------------------------------------------------

    def watch_baseline(self):
        """Wrap both loaders. Suites like SDPA never call run(), only run_elf_files().

        Keep the wrappers installed for the variants so their results can be
        compared with the clean pass.
        """
        original_run, original_load = TestConfig.run, TestConfig.run_elf_files
        baseline = self.baseline

        def wrapped_run(config_self, *args, **kwargs):
            baseline.config = baseline.config or config_self
            baseline.saw_run = True
            outcome = original_run(config_self, *args, **kwargs)
            baseline.had_result = (
                baseline.had_result or getattr(outcome, "result", None) is not None
            )
            if config_self._bit_exact_unsupported_reason() is not None:
                self._comparable = False
            self._results.append(getattr(outcome, "result", None))
            return outcome

        def wrapped_load(config_self, *args, **kwargs):
            baseline.config = baseline.config or config_self
            baseline.saw_load = True
            return original_load(config_self, *args, **kwargs)

        def unwatch():
            TestConfig.run, TestConfig.run_elf_files = original_run, original_load

        TestConfig.run, TestConfig.run_elf_files = wrapped_run, wrapped_load
        return unwatch

    def beat(self, variant=None) -> None:
        """Say what we are about to attempt, before we attempt it."""
        writer = _hb()
        if not writer.enabled:
            return
        case = self._item.nodeid if self._item is not None else ""
        if variant is None:
            writer.beat(case)
            return
        scan = self.scans.get(variant.thread)
        writer.beat(
            case,
            {
                "thread": variant.thread,
                "site_index": variant.site.index,
                "addr": variant.site.addr,
                "op": variant.site.op,
                "filler": variant.filler,
                "filler_word": variant.filler_word,
                "delay": variant.delay,
                "label": variant.label(),
                # The supervisor may need this after the worker has stopped.
                "elf": getattr(scan, "elf", "") if scan is not None else "",
            },
        )

    def _scanned_elf_dir(self) -> Path:
        return Path(self.baseline.config.temp_elfs[0]).resolve().parent

    def _reload_scanned_image(self) -> None:
        """Put the scanned kernel back in L1."""
        self._forget_kernel_image()
        config = self.baseline.config
        with quiet_harness():
            config.write_runtimes_to_L1()
            if config.variant_stimuli:
                config.variant_stimuli.write(TestConfig.TENSIX_LOCATION)
            config.run_elf_files()
            config.wait_for_tensix_operations_finished()
        self._injector_for_device().forget()

    def _prepare_arm(self) -> None:
        loaded = TestConfig.LAST_LOADED_ELFS
        if loaded is not None and Path(loaded).resolve() == self._scanned_elf_dir():
            return
        self._reload_scanned_image()

    def run(self, variant):
        self.beat(variant)
        # The ELF is already in L1, so arming only needs a few word writes.
        self._prepare_arm()
        self._injector_for_device().arm(
            variant.thread,
            self.scans[variant.thread],
            variant.site,
            variant.delay,
            variant.filler_word,
        )
        if self.verbose:
            print(f">> {variant.label()}", flush=True)
        # Rewind to the stimuli the baseline drew, so a difference in the output is
        # the delay and not the data.
        if self._rng_state is not None:
            torch.set_rng_state(self._rng_state)
        self._results = []
        try:
            # Calling item.runtest here would enter this hook again.
            with quiet_harness():
                self._item.obj(**self._kwargs)
            moved, pcc = self._output_moved()
            # Keep the worst score from all repeats. _record clears it afterward.
            if moved and pcc is not None:
                if self._last_pcc is None or pcc < self._last_pcc:
                    self._last_pcc = pcc
            return ("drift", moved) if moved else (None, "")
        except _SKIPPED:
            return None, ""
        except DetourError:
            raise
        except TimeoutError as err:
            return "hang", str(err)
        except LLKAssertException as err:
            return "assert", str(err)
        except (AssertionError, _FAILED) as err:
            return "mismatch", str(err)
        except Exception as err:
            return "error", f"{type(err).__name__}: {err}"

    def _output_moved(self):
        """How this run's output differs from the baseline's.

        Returns (message, pcc). Message is "" when nothing moved.
        """
        if not self._comparable or self._baseline_results is None:
            return "", None
        return sweep_module.describe_drift(self._baseline_results, self._results)

    def _prove_reproducible(self, item) -> None:
        """Run the body once more, same stimuli, no detour, and check it agrees."""
        if not self.config.drift or not self._comparable:
            return
        self._baseline_results, self._results = self._results, []
        torch.set_rng_state(self._rng_state)
        try:
            with quiet_harness():
                item.obj(**self._kwargs)
        # pytest outcomes derive from BaseException, so catch them explicitly.
        except (*_SKIPPED, _FAILED, Exception) as err:
            self._comparable, reason = False, f"{type(err).__name__}: {err}"
        else:
            reason, _ = self._output_moved()
        if reason:
            self._comparable = False
            print(
                f">> {item.nodeid}: not reproducible, drift off ({reason})", flush=True
            )

    def sweep(self, item) -> list:
        """Perturb every planned variant of one test. Returns (label, tags) per finding."""
        config = self.baseline.config
        self.scans = {
            thread: scanner.scan(path, self.config.site_mode)
            for thread, path in zip(TestConfig.KERNEL_COMPONENTS, config.temp_elfs)
        }
        # Check every cave fits before touching the device, so a geometry mistake
        # is a loud error up front rather than a run of bogus "failures".
        injector = self._injector_for_device()
        for scan in self.scans.values():
            injector.cave_for(scan)

        # z_state/reconfig checks TensixState in the body and has no result buffer.
        # risc_nop is the only filler that shifts a RISC cfg write.
        saved = self.config.filler, self.config.threads
        if "z_state/reconfig/" in item.nodeid:
            self.config.filler = "risc_nop"
        try:
            variants = sweep_module.plan(self.config, self.scans)
        finally:
            self.config.filler, self.config.threads = saved
        if not variants:
            return []
        if self.verbose:
            print(f"\n>> {item.nodeid}: {len(variants)} variant(s)", flush=True)

        self._item = item
        self._kwargs = _test_kwargs(item)
        self._prove_reproducible(item)
        try:
            return sweep_module.run(
                self.config,
                variants,
                self,
                lambda variant, fails, tags, error: self._record(
                    item, variant, fails, tags, error
                ),
            )
        finally:
            self._item = None
            self._kwargs = {}
            self._baseline_results = None
            self._results = []
            try:
                injector.restore()
            except Exception:
                # Keep the original device error instead of raising a restore error.
                pass
            # The cave bytes outlive the restore, so the next case must not be handed
            # an L1 image the harness still believes is pristine.
            self._forget_kernel_image()

    def _record(self, item, variant, fails, tags, error) -> None:
        scan = self.scans[variant.thread]
        report.append(
            self.config.report_dir,
            {
                "case": item.nodeid,
                "arch": self.config.arch,
                "site_mode": self.config.site_mode,
                "thread": variant.thread,
                "site_index": variant.site.index,
                "addr": variant.site.addr,
                "op": variant.site.op,
                "filler": variant.filler,
                "filler_word": variant.filler_word,
                "delay": variant.delay,
                # Plan position, so a log several workers appended to can still be
                # read in sweep order.
                "seq": variant.seq,
                "runs": self.config.repeats,
                "fails": fails,
                "tag": ",".join(sorted(tags)),
                # First line only: a mismatch drags the whole offending tensor
                # behind it, and none of that survives a rebuild anyway.
                "error": error.strip().splitlines()[0][:200] if error.strip() else "",
                "chain": list(report.source_chain(scan.elf, variant.site.addr)),
                **(
                    {
                        "pcc": round(self._last_pcc, 6),
                        "pcc_delta": round(1.0 - self._last_pcc, 6),
                    }
                    if self._last_pcc is not None
                    else {}
                ),
            },
        )
        # One variant's score must not be attributed to the next one.
        self._last_pcc = None


_perturber = None


def _get() -> Perturber:
    global _perturber
    if _perturber is None:
        _perturber = Perturber(sweep_module.Config.from_env())
    return _perturber


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    root = heartbeat.state_dir()
    if root is not None and item.nodeid in heartbeat.completed(root):
        pytest.skip("already recorded")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    perturber = _get()
    perturber.begin()
    unwatch = perturber.watch_baseline()
    _hb().beat(item.nodeid)
    try:
        with quiet_harness():
            outcome = yield

        # A test that was already red tells us nothing about timing.
        if outcome.excinfo is not None:
            error = outcome.excinfo[1]
            if isinstance(error, TimeoutError):
                _hang_closes_case(item.nodeid, str(error))
            else:
                _hb().mark_done(item.nodeid)
            return
        # Reconfig tests can be swept without a result buffer because the body
        # checks TensixState.
        if not perturber.baseline.worth_sweeping() and not (
            perturber.baseline.saw_load and "z_state/reconfig/" in item.nodeid
        ):
            _hb().mark_done(item.nodeid)
            return
        try:
            findings = perturber.sweep(item)
        except sweep_module.DeviceWedged as err:
            # Record the hang and let the supervisor clear the core.
            _hang_closes_case(item.nodeid, str(err))
            outcome.force_exception(
                AssertionError(f"hang: {err} wedged the core. Recovery requested")
            )
            return
        # A case that loses the device stays unfinished for the next attempt.
        _hb().mark_done(item.nodeid)
        # Drift stays in the report because the variant still passed its golden.
        failures = [label for label, tags in findings if tags - {"drift"}]
        if failures:
            # Fail the pytest case and name the first variant that broke it.
            head = (
                failures[0]
                if len(failures) == 1
                else f"{failures[0]} (+{len(failures) - 1} more)"
            )
            outcome.force_exception(
                AssertionError(f"{len(failures)} perturbation(s) failed: {head}")
            )
            # A mismatch can leave device state dirty, so move this worker too.
            _hang_closes_case(item.nodeid, failures[0], skip_family=False)
    finally:
        unwatch()


# The argument has to be named `report` for pytest to bind it
def pytest_runtest_logreport(report):
    if not (
        report.when == "call"
        or report.outcome == "failed"
        or (report.when == "setup" and report.outcome == "skipped")
    ):
        return
    _hb().record_result(
        report.nodeid,
        report.outcome,
        getattr(report, "duration", 0.0),
        str(report.longrepr or ""),
    )


def pytest_runtest_logfinish(nodeid, location):
    # Once per item, whichever way it ended, and after teardown has let go of the
    # device
    _hb().idle()
    # Here rather than where the hang was caught, so the case that asked for
    # recovery is fully reported before this worker stops answering for work.
    if _parked:
        _park()


def pytest_sessionfinish(session, exitstatus):
    # Drop out of the live set first: a worker that ran out of work is not a
    # worker that stopped answering, and the supervisor must not confuse them.
    _hb().finish()
    # Workers share the JSONL. Only the master process renders the report.
    if hasattr(session.config, "workerinput"):
        return
    config = sweep_module.Config.from_env()
    records = report.load(config.report_dir)
    if not records:
        return
    path = report.write_markdown(
        config.report_dir,
        report.environment(config.arch, config.site_mode, config.filler, config.drift),
    )
    drifted = sum(1 for record in records if "drift" in record["tag"])
    print(
        f"\n>> {len(records)} recorded variant(s) ({drifted} drift) -> {path}",
        flush=True,
    )
