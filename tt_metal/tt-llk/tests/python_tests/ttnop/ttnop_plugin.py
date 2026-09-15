# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin that sweeps detours over a test.

Every collected test runs once clean. If that pass looked sweepable, the same
test body is then re-run once per planned variant with a detour armed, and
anything that stops passing is recorded.

Where the detour lands is the backend's business: the LLK backend pokes device
L1, the Metal backend pokes tt-metal's host-side kernel image. Everything here —
the baseline pass, the variant loop, the recording — is common to both.

Load it with `-p ttnop_plugin` and this directory on PYTHONPATH.
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import heartbeat
import pytest
import report
import sweep as sweep_module
from cave import DetourError

# pytest's outcome exceptions derive from BaseException, so they need naming
# explicitly or they would escape the per-variant handler and fail the whole item.
_SKIPPED = (pytest.skip.Exception, pytest.xfail.Exception)
_FAILED = pytest.fail.Exception

_writer = None
# Set once a hang has asked the supervisor for recovery, so the worker stops
# after the case it is reporting rather than in the middle of it.
_parked = False


def _hb() -> heartbeat.Writer:
    """One progress writer per process; a no-op unless a supervisor is watching."""
    global _writer
    if _writer is None:
        _writer = heartbeat.Writer()
    return _writer


def _recovery_closes_case(
    nodeid: str, variant: str, result: str, skip_family: bool
) -> None:
    """Persist a failed case and ask to be moved off its untrusted core."""
    global _parked
    writer = _hb()
    # Save a result before done/recovery: the supervisor may kill us immediately.
    writer.record_result(nodeid, "failed", message=f"{result}: {variant}")
    writer.mark_done(nodeid)
    writer.request_recovery(nodeid, variant, skip_family=skip_family)
    # Nobody is watching an unsupervised run, so there is no recovery coming and
    # nothing to wait for.
    _parked = writer.enabled


def _hang_closes_case(nodeid: str, variant: str) -> None:
    """Close a hung case and skip siblings that would likely hang another core."""
    _recovery_closes_case(nodeid, variant, "hang", skip_family=True)


def _park() -> None:
    """Stop taking cases and wait for the supervisor to kill this worker.

    A hung core is not this worker's to fix, and every case it pulls off the
    queue meanwhile fails against that core in about a second — marked done for
    a run it never really got, so permanently red for a fault it never saw. That
    is how one hang turned seventy cases red. Sleeping instead costs only the
    seconds until the supervisor's next poll.

    The DONE beat first, so the silence that follows is read as a worker with
    nothing left to do rather than one that stopped answering.
    """
    _hb().finish()
    while True:
        time.sleep(60)


def _make_backend(config: sweep_module.Config):
    """Pick the backend. Importing it is what selects it.

    The LLK backend pulls in `helpers` and ttexalens, neither of which exists in a
    tt-metal-rooted pytest process; the Metal backend pulls in ctypes and the
    tt-metal kernel cache, which an LLK run has no business touching. Keeping both
    behind a lazy import is what lets one plugin serve both.
    """
    if config.metal:
        import metal

        return metal.MetalBackend(config.max_delay)
    import backend_llk

    return backend_llk.LLKBackend(config.max_delay)


def _test_kwargs(item) -> dict:
    """Fixture values for calling the test body directly. Fixed for the whole sweep."""
    names = getattr(item._fixtureinfo, "argnames", ()) or ()
    return {name: item.funcargs[name] for name in names if name in item.funcargs}


class Perturber:
    def __init__(self, config: sweep_module.Config):
        self.config = config
        self.verbose = os.environ.get("TTNOP_VERBOSE", "") not in ("", "0")
        self.backend = _make_backend(config)
        self.baseline = None
        self.scans = {}
        self._item = None
        self._kwargs = {}
        self._rng_state = None
        # Result of every run() the body just made, and the same list from the
        # clean pass to compare it against. LLK only; metal has no TestConfig.run.
        self._results = []
        self._baseline_results = None
        self._comparable = True
        self._last_pcc = None

    def begin(self) -> None:
        """Reset the per-test state and remember the RNG the baseline will draw from.

        Called from the hook before the body runs, which is after conftest's autouse
        seed fixture — so restoring this state hands every variant byte-for-byte the
        stimuli the baseline saw. Metal has no that seam: drift is off.
        """
        self.backend.reset_case()
        self.baseline = None
        self._results = []
        self._baseline_results = None
        self._last_pcc = None
        if self.config.metal:
            self._comparable = False
            self._rng_state = None
            return
        self._comparable = True
        self._rng_state = None
        if self.config.drift:
            import torch

            self._rng_state = torch.get_rng_state()

    # -- the runtime the sweep loop drives ---------------------------------

    def beat(self, variant=None) -> None:
        """Say what we are about to attempt, before we attempt it.

        Published ahead of the call because the call is what may never return:
        afterwards is too late to name the variant that wedged the card, and
        naming it is the whole finding.
        """
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
                # Kept so the supervisor can still resolve the inline chain: it
                # renders the finding after the run it belonged to is dead.
                "elf": getattr(scan, "elf", "") if scan is not None else "",
            },
        )

    def run(self, variant):
        self.beat(variant)
        # The image is already in place and the cores are idle, so arming is a couple
        # of word writes. Leaving the image alone is what keeps a 100-delay sweep
        # cheap: a reload per variant would cost three ELFs to buy the same one
        # instruction. LLK reloads only when the body left a different kernel than
        # we scanned (test_generalized_moe_gate_idx_offset).
        self.backend.prepare_arm()
        self.backend.injector_for(variant.thread).arm(
            variant.thread,
            self.scans[variant.thread],
            variant.site,
            variant.delay,
            variant.filler_word,
        )
        if self.verbose:
            print(f">> {variant.label()}", flush=True)
        # Rewind to the stimuli the baseline drew, so a difference in the output is
        # the delay and not the data. With drift off the RNG stream instead runs on
        # across variants, which samples different data per variant but leaves the
        # runs incomparable: some races only show up on later draws.
        if self._rng_state is not None:
            import torch

            torch.set_rng_state(self._rng_state)
        self._results = []
        try:
            # Call the body, not item.runtest: we are already inside pytest_runtest_call,
            # and re-entering that hook nests another full sweep.
            with self.backend.quiet():
                self._item.obj(**self._kwargs)
            moved, pcc = self._output_moved()
            # A depth run repeats a variant; keep the worst score across its runs
            # rather than whichever repeat happened to go last. _record clears it.
            if moved and pcc is not None:
                if self._last_pcc is None or pcc < self._last_pcc:
                    self._last_pcc = pcc
            return ("drift", moved) if moved else (None, "")
        except _SKIPPED:
            return None, ""
        except DetourError:
            raise
        except (Exception, _FAILED) as err:
            return self.backend.classify(err)

    def _output_moved(self):
        """How this run's output differs from the baseline's.

        Returns (message, pcc). Message is "" when nothing moved.
        """
        if not self._comparable or self._baseline_results is None:
            return "", None
        return sweep_module.describe_drift(self._baseline_results, self._results)

    def _prove_reproducible(self, item) -> None:
        """Run the body once more, same stimuli, no detour, and check it agrees.

        A case that cannot reproduce its own output is non-deterministic for reasons
        that have nothing to do with a delay, and every variant of it would otherwise
        be reported as drift. Metal has no TestConfig.run seam, so this is LLK only.
        """
        if self.config.metal or not self.config.drift or not self._comparable:
            return
        import torch

        self._baseline_results, self._results = self._results, []
        torch.set_rng_state(self._rng_state)
        try:
            with self.backend.quiet():
                item.obj(**self._kwargs)
        # pytest's outcomes derive from BaseException; a body that just passed and
        # now skips or fails is as unreproducible as one that raised, and letting
        # either escape would rewrite the case's result.
        except (*_SKIPPED, _FAILED, Exception) as err:
            self._comparable, reason = False, f"{type(err).__name__}: {err}"
        else:
            reason, _ = self._output_moved()
        if reason:
            self._comparable = False
            print(
                f">> {item.nodeid}: not reproducible, drift off ({reason})", flush=True
            )

    # -- driving one test --------------------------------------------------

    def sweep(self, item) -> list:
        """Perturb every planned variant of one test. Returns (label, tags) per finding."""
        self._item = item
        self._kwargs = _test_kwargs(item)
        self._prove_reproducible(item)
        # z_state/reconfig has no result buffer; the body asserts TensixState.
        # risc_nop is the only filler that shifts a RISC cfg write.
        saved = self.config.filler, self.config.threads
        if "z_state/reconfig/" in item.nodeid:
            self.config.filler = "risc_nop"
        try:
            self.scans = self.backend.scans(self.config.site_mode)
            # Check every cave fits before touching anything, so a geometry mistake is
            # a loud error up front rather than a run of bogus "failures".
            for thread, scan in self.scans.items():
                self.backend.injector_for(thread).cave_for(scan)

            variants = sweep_module.plan(self.config, self.scans)
        finally:
            self.config.filler, self.config.threads = saved
        if not variants:
            return []
        if self.verbose:
            print(f"\n>> {item.nodeid}: {len(variants)} variant(s)", flush=True)

        try:
            return sweep_module.run(
                self.config,
                variants,
                self,
                lambda variant, runs, fails, tags, error: self._record(
                    item, variant, runs, fails, tags, error
                ),
            )
        finally:
            self._item = None
            self._kwargs = {}
            self._baseline_results = None
            self._results = []
            try:
                self.backend.restore()
            except Exception:
                # Only reachable once the device stopped taking writes, which is
                # already being raised past us — do not mask it with the symptom.
                pass
            self.backend.finish()

    def _record(self, item, variant, runs, fails, tags, error) -> None:
        scan = self.scans[variant.thread]
        report.append(
            self.config.report_dir,
            {
                "case": item.nodeid,
                "arch": self.config.arch,
                "backend": self.backend.name,
                "kernel": self.backend.kernel,
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
                "runs": runs,
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
    # Siblings of a hang land on the done-log from the supervisor. Skip them
    # before fixtures open the device, or the next worker hits the same site
    # and we lose another core.
    root = heartbeat.state_dir()
    if root is not None and item.nodeid in heartbeat.completed(root):
        pytest.skip("already recorded")
    marker = item.get_closest_marker("xfail")
    if marker is not None:
        reason = marker.kwargs.get("reason", "pre-marked xfail")
        pytest.xfail(f"ttnop does not execute xfailed tests: {reason}")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    # A fixture can add an xfail marker during setup, after pytest_runtest_setup.
    marker = pyfuncitem.get_closest_marker("xfail")
    if marker is not None:
        reason = marker.kwargs.get("reason", "pre-marked xfail")
        pytest.xfail(f"ttnop does not execute xfailed tests: {reason}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    perturber = _get()
    perturber.begin()
    unwatch = perturber.backend.watch_baseline(perturber)
    _hb().beat(item.nodeid)
    try:
        # The baseline runs muted too: a red one is skipped rather than swept, so
        # its tile dump buys nothing here, and conftest still names the failure.
        # Through the backend because each one silences a different harness.
        with perturber.backend.quiet():
            outcome = yield

        # A test that was already red tells us nothing about timing.
        if outcome.excinfo is not None:
            error = outcome.excinfo[1]
            if isinstance(error, TimeoutError):
                _hang_closes_case(item.nodeid, str(error))
            return
        if not perturber.backend.ready(item.nodeid):
            return
        try:
            findings = perturber.sweep(item)
        except DetourError as err:
            # The clean test passed, but this ELF cannot hold the requested
            # detour.
            outcome.force_exception(pytest.skip.Exception(f"ttnop skipped: {err}"))
            return
        except sweep_module.DeviceWedged as err:
            # Record the hang and let the supervisor clear the core.
            _hang_closes_case(item.nodeid, str(err))
            outcome.force_exception(
                AssertionError(f"hang: {err} wedged the core. Recovery requested")
            )
            return
        # Drift is report-only: the variant still passed the test's own golden, so
        # the case stays green and the record lives in report.md.
        failures = [label for label, tags in findings if tags - {"drift"}]
        if failures:
            # Hang the finding on the case itself so a sweep reads like an ordinary
            # pytest run: the case goes red and names the variant that broke it.
            head = (
                failures[0]
                if len(failures) == 1
                else f"{failures[0]} (+{len(failures) - 1} more)"
            )
            outcome.force_exception(
                AssertionError(f"{len(failures)} perturbation(s) failed: {head}")
            )
            # A failed perturbation can leave dest or semaphores dirty. Move the
            # worker to a spare core, but keep sibling parameters queued because
            # they are different timing windows rather than the same hang.
            _recovery_closes_case(
                item.nodeid, failures[0], "perturbation", skip_family=False
            )
    finally:
        unwatch()


# The argument has to be named `report` for pytest to bind it, which shadows the
# report module for the body of this hook; nothing here needs that module.
def pytest_runtest_logreport(report):
    # The junit file is assembled from these lines rather than from pytest's own
    # --junit-xml, which lands only at session end; the supervisor kills the session
    # when a core wedges, so that file would be missing on exactly the runs worth
    # reading. Setup and teardown are only worth a line when they went wrong.
    if not (
        report.when == "call"
        or report.outcome == "failed"
        or (report.when == "setup" and report.outcome == "skipped")
    ):
        return
    result_outcome = (
        "xfailed"
        if report.outcome == "skipped" and getattr(report, "wasxfail", None)
        else report.outcome
    )
    _hb().record_result(
        report.nodeid,
        result_outcome,
        getattr(report, "duration", 0.0),
        str(report.longrepr or ""),
    )
    # Result first: a killed worker must never leave a done case missing from JUnit.
    _hb().mark_done(report.nodeid)


def pytest_runtest_logfinish(nodeid, location):
    # Once per item, whichever way it ended, and after teardown has let go of the
    # device. From here until the next case starts the worker owns nothing, so the
    # supervisor must not read the gap as a stall — at the tail of a sweep that
    # gap is however long the slowest worker still has left to run.
    _hb().idle()
    # Here rather than where the hang was caught, so the case that asked for
    # recovery is fully reported before this worker stops answering for work.
    if _parked:
        _park()


def pytest_sessionfinish(session, exitstatus):
    # Drop out of the live set first: a worker that ran out of work is not a
    # worker that stopped answering, and the supervisor must not confuse them.
    _hb().finish()
    # Workers each swept part of the suite into the shared JSONL; the master renders it.
    if hasattr(session.config, "workerinput"):
        return
    config = sweep_module.Config.from_env()
    records = report.load(config.report_dir)
    if not records:
        return
    path = report.write_markdown(
        config.report_dir,
        report.environment(
            config.arch,
            config.site_mode,
            config.filler,
            config.drift,
            "metal" if config.metal else "llk",
        ),
    )
    drifted = sum(1 for record in records if "drift" in record["tag"])
    print(
        f"\n>> {len(records)} recorded variant(s) ({drifted} drift) -> {path}",
        flush=True,
    )
