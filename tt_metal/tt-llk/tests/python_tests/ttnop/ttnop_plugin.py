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
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pytest
import report
import sweep as sweep_module
from cave import DetourError

# pytest's outcome exceptions derive from BaseException, so they need naming
# explicitly or they would escape the per-variant handler and fail the whole item.
_SKIPPED = (pytest.skip.Exception, pytest.xfail.Exception)
_FAILED = pytest.fail.Exception


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
        self.scans = {}
        self._item = None
        self._kwargs = {}

    # -- the runtime the sweep loop drives ---------------------------------

    def _replay(self):
        """Run the test body once, exactly as a variant would. Used to reload a
        clean image during recovery."""
        self._item.obj(**self._kwargs)

    def run(self, variant):
        # The image is already in place and the cores are idle, so arming is a couple
        # of word writes. Leaving the image alone is what keeps a 100-delay sweep
        # cheap: a reload per variant would cost three ELFs to buy the same one
        # instruction.
        self.backend.injector_for(variant.thread).arm(
            variant.thread,
            self.scans[variant.thread],
            variant.site,
            variant.delay,
            variant.filler_word,
        )
        if self.verbose:
            print(f">> {variant.label()}", flush=True)
        # Keep the baseline's RNG stream across variants (no re-seed); some races
        # only show up on later draws.
        try:
            # Call the body, not item.runtest: we are already inside pytest_runtest_call,
            # and re-entering that hook nests another full sweep.
            with self.backend.quiet():
                self._item.obj(**self._kwargs)
            return None, ""
        except _SKIPPED:
            return None, ""
        except DetourError:
            raise
        except (Exception, _FAILED) as err:
            return self.backend.classify(err)

    def recover(self) -> bool:
        return self.backend.recover(self._replay)

    # -- driving one test --------------------------------------------------

    def sweep(self, item) -> list:
        """Perturb every planned variant of one test. Returns a label per failure."""
        self._item = item
        self._kwargs = _test_kwargs(item)
        try:
            self.scans = self.backend.scans(self.config.site_mode)
            # Check every cave fits before touching anything, so a geometry mistake is
            # a loud error up front rather than a run of bogus "failures".
            for thread, scan in self.scans.items():
                self.backend.injector_for(thread).cave_for(scan)

            variants = sweep_module.plan(self.config, self.scans)
            if not variants:
                return []
            if self.verbose:
                print(f"\n>> {item.nodeid}: {len(variants)} variant(s)", flush=True)

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
            try:
                self.backend.restore()
            except Exception:
                # Only reachable once the device stopped taking writes, which is
                # already being raised past us — do not mask it with the symptom.
                pass
            self.backend.finish()

    def _record(self, item, variant, fails, tags, error) -> None:
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
                "runs": self.config.repeats,
                "fails": fails,
                "tag": ",".join(sorted(tags)),
                # First line only: a mismatch drags the whole offending tensor
                # behind it, and none of that survives a rebuild anyway.
                "error": error.strip().splitlines()[0][:200] if error.strip() else "",
                "chain": list(report.source_chain(scan.elf, variant.site.addr)),
            },
        )


_perturber = None


def _get() -> Perturber:
    global _perturber
    if _perturber is None:
        _perturber = Perturber(sweep_module.Config.from_env())
    return _perturber


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    perturber = _get()
    unwatch = perturber.backend.watch_baseline()
    try:
        outcome = yield
    finally:
        unwatch()

    # A test that was already red tells us nothing about timing.
    if outcome.excinfo is not None or not perturber.backend.ready():
        return
    failures = perturber.sweep(item)
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


def pytest_sessionfinish(session, exitstatus):
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
            "metal" if config.metal else "llk",
        ),
    )
    print(f"\n>> {len(records)} failing variant(s) -> {path}", flush=True)
