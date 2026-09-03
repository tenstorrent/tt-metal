"""galaxy-kit pytest plugin for BATCHED benchmark sessions (worker.py).

Loaded only by kit consumer sessions (`-p lk_batch_plugin`, PYTHONPATH set
by worker.py).  It never touches the device or the tests; it observes.

Why it exists — the harness alone cannot attribute a batched session:
  * the perf report fixture is MODULE-scoped, so every test of one module
    lands in one CSV (many ops share a module, e.g.
    perf_eltwise_unary_sfpu.py), and
  * combine_perf_reports() collapses rows with identical
    (sweep-params, marker) keys by AVERAGING — so N in-session reps of one
    node would silently become one mean row, changing what a measurement
    means.

So this plugin writes ONE post CSV per node OCCURRENCE (i.e. per rep),
computed from the frames that occurrence appended to the live report,
through the harness's OWN postprocess/collapse functions — the same code
path a solo session's output takes.

Honesty duties:
  ORDER    — the spec (LK_BATCH_SPEC json) lists the exact node order the
             worker requested; if pytest collects anything else the plugin
             aborts the session BEFORE any device work (the worker then
             falls back to solo sessions).
  CORR GATE — an arm's perf occurrences carry gate_seq = the seq of that
             arm's correctness occurrence; if that corr fails (or errors
             in setup), the gated occurrences are SKIPPED before touching
             the device, preserving "perf runs only after corr PASSED".
  BAIL-OUT — a TENSIX TIMEOUT can leave the chip hung; after one, nothing
             else runs in this session (occurrences report "bailed"; the
             worker re-runs them solo).  Pre-hang values stay valid.
  MANIFEST — one TSV row per occurrence (outcome, frame count, seconds),
             flushed as it goes, so a killed session still accounts for
             everything it ran.
"""
import json
import os
import time
from pathlib import Path

import pytest

_SPEC = None       # list of occurrence dicts, in seq order
_OUT = None        # output dir (LK_BATCH_OUT)
_MANIFEST = None   # open file handle
_PTR = 0           # index of the occurrence currently executing
_FAILED_CORR = set()   # seqs of failed corr occurrences
_GATED_SKIP = set()    # seqs skipped because their gate corr failed
_RAN_ITEMS = set()     # id(item) for items that already ran once
_CUR = {}              # scratch for the occurrence in flight
_BAIL = False          # a device hang happened: run nothing else here

MANIFEST_COLS = ("seq", "nodeid", "role", "op", "leg", "arm", "rep",
                 "outcome", "nframes", "secs")


def _load():
    global _SPEC, _OUT, _MANIFEST
    spec_path = os.environ.get("LK_BATCH_SPEC")
    out_dir = os.environ.get("LK_BATCH_OUT")
    if not spec_path or not out_dir:
        raise pytest.UsageError(
            "lk_batch_plugin loaded without LK_BATCH_SPEC/LK_BATCH_OUT")
    _SPEC = json.loads(Path(spec_path).read_text())["occurrences"]
    _OUT = Path(out_dir)
    _OUT.mkdir(parents=True, exist_ok=True)
    _MANIFEST = (_OUT / "manifest.tsv").open("w", buffering=1)
    _MANIFEST.write("\t".join(MANIFEST_COLS) + "\n")


def _emit(occ, outcome, nframes=0, secs=""):
    row = dict(occ, outcome=outcome, nframes=nframes, secs=secs)
    _MANIFEST.write(
        "\t".join(str(row.get(c, "")) for c in MANIFEST_COLS) + "\n")
    _MANIFEST.flush()


def pytest_configure(config):
    _load()


def pytest_collection_modifyitems(session, config, items):
    """Rebuild the run list to be EXACTLY the requested occurrence order.

    pytest dedupes identical node ids given as args (even with
    --keep-duplicates: session.items keeps one item per node id), so
    in-session reps cannot be expressed on the command line.  The worker
    therefore passes each node once and this hook expands the collected
    items into the spec's occurrence sequence, reusing the collected item
    object for every occurrence of its node (each run gets a fresh
    setup/call/teardown).  Anything missing or unexpected aborts BEFORE
    device work; the worker then falls back to solo sessions."""
    by_id = {}
    for it in items:
        by_id.setdefault(it.nodeid, it)
    want = [o["nodeid"] for o in _SPEC]
    missing = [n for n in want if n not in by_id]
    extra = [n for n in by_id if n not in set(want)]
    if missing or extra:
        _MANIFEST.write("-1\tORDER\t\t\t\t\t\tORDER-VIOLATION\t0\t\n")
        _MANIFEST.flush()
        raise pytest.UsageError(
            "lk_batch_plugin: collected nodes differ from the requested "
            "batch (order/content is a measurement invariant); "
            f"missing={missing[:3]!r} unexpected={extra[:3]!r}")
    items[:] = [by_id[n] for n in want]


def pytest_runtest_setup(item):
    occ = _SPEC[_PTR]
    if _BAIL:
        # a TENSIX TIMEOUT earlier in this session likely left the device
        # hung; values measured after it would be unattributable.  Run
        # nothing else here — the worker re-runs the rest solo (values
        # booked BEFORE the hang are pre-hang and stay valid).
        _CUR["bailed"] = True
        pytest.skip("LK-BAIL: device hang earlier in this session")
    gate = occ.get("gate_seq")
    if gate is not None and gate in _FAILED_CORR:
        _GATED_SKIP.add(occ["seq"])
        pytest.skip("LK-CORR-GATE: this arm's correctness node failed in "
                    "this session; perf refused")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _PTR
    occ = _SPEC[_PTR] if _PTR < len(_SPEC) else None
    if occ is None or item.nodeid != occ["nodeid"]:
        # cannot happen after the collection check; belt anyway
        _MANIFEST.write("-1\tORDER\t\t\t\t\t\tORDER-VIOLATION\t0\t\n")
        _MANIFEST.flush()
        raise pytest.UsageError("lk_batch_plugin: runtime order violation")
    if id(item) in _RAN_ITEMS:
        # A repeated occurrence reuses the collected item object.  pytest's
        # SetupState considers a same-item successor "already set up" and
        # skips re-filling fixtures, so tear everything down and reset the
        # item's fixture request (the pytest-rerunfailures mechanism).
        # This finalizes only FIXTURES (the module perf report dumps its
        # CSV; fresh ones are built for this run) — device connections
        # live in conftest module globals and are untouched, so repeated
        # runs keep the amortized session while each gets fresh fixtures.
        item.session._setupstate.teardown_exact(None)
        item._initrequest()
    _RAN_ITEMS.add(id(item))
    _CUR.clear()
    _CUR.update(occ=occ, outcome="unknown", report=None, n0=0,
                t0=time.time())
    yield
    outcome = _CUR["outcome"]
    if occ["seq"] in _GATED_SKIP:
        outcome = "gate-skipped"
    elif _CUR.get("bailed"):
        outcome = "bailed"
    nframes = 0
    if occ["role"] == "perf" and outcome == "passed":
        # a dump failure must cost only THIS occurrence (the worker re-runs
        # it solo), never abort the session mid-batch
        try:
            nframes = _dump_item_csv(occ)
        except Exception as e:
            nframes = 0
            outcome = f"dump-error:{type(e).__name__}"
        if nframes == 0 and outcome == "passed":
            outcome = "no-perf-rows"
    if occ["role"] == "corr" and outcome != "passed":
        _FAILED_CORR.add(occ["seq"])
    _emit(occ, outcome, nframes, f"{time.time() - _CUR['t0']:.1f}")
    _PTR += 1


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    # checkpoint the live module-scoped report before the test body runs
    report = item.funcargs.get("perf_report")
    if report is not None:
        _CUR["report"] = report
        _CUR["n0"] = len(report._frames)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    global _BAIL
    out = yield
    rep = out.get_result()
    if rep.when == "call":
        _CUR["outcome"] = rep.outcome  # passed / failed / skipped
        # TimeoutError = the harness's TENSIX TIMED OUT (a hung chip);
        # checked on the excinfo itself so hook ordering cannot hide it
        if rep.failed and call.excinfo is not None and \
                call.excinfo.errisinstance(TimeoutError):
            _BAIL = True
    elif rep.when == "setup" and rep.outcome != "passed":
        _CUR["outcome"] = "skipped" if rep.skipped else "setup-error"


def _dump_item_csv(occ):
    """Write this occurrence's frames as raw + post CSVs, through the
    harness's own pipeline (postprocess_tile_loop + duplicate-key collapse
    + full-column sort — exactly what a solo session's combined post CSV
    goes through)."""
    import pandas as pd
    from helpers.perf.core import _collapse_duplicate_keys, postprocess_tile_loop

    report = _CUR.get("report")
    if report is None:
        return 0
    new = [f for f in report._frames[_CUR["n0"]:] if not f.empty]
    if not new:
        return 0
    raw = pd.concat(new, ignore_index=True)
    label = f"batch-item-{occ['seq']:04d}"
    raw.to_csv(_OUT / f"{occ['seq']:04d}.raw.csv", index=False)
    post = postprocess_tile_loop(raw.copy())
    post = _collapse_duplicate_keys(post, label)
    post = post.sort_values(by=post.columns.tolist()).reset_index(drop=True)
    post.to_csv(_OUT / f"{occ['seq']:04d}.post.csv", index=False)
    return len(new)
