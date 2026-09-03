"""galaxy-kit pytest plugin — per-node perf demux for BATCHED consumer sessions.

Loaded only by the worker's opt-in batched mode (LK_BATCH=1) via
``-p lk_batch_plugin`` with LK_BATCH_DEMUX pointing at a fresh directory.
It does nothing at all when LK_BATCH_DEMUX is unset, and it never changes
what the harness measures — it only OBSERVES the perf frames each test node
appends to the module-scoped ``perf_report`` fixture and, right after each
passing node, replays the harness's OWN dump/post-process/combine code on
just that node's frames.  The per-node output is therefore built by the
identical code path a solo session uses, so a demuxed cell's CSVs are
byte-comparable to a solo session's cell (same chip, same measurements).

Per node the demux dir gains:
  <idx>__<sha>/temp_perf_data/            raw + post per-worker CSVs (consumed)
  <idx>__<sha>/out/perf_data/<module>/    combined CSVs exactly as a solo
                                          session's farm perf_data would hold
  index.tsv                               nodeid -> dirname mapping (only
                                          written AFTER the node's combine
                                          succeeded: presence == demux OK)
  outcomes.jsonl                          one record per pytest report phase
  errors.log                              any demux failure (fail-closed: the
                                          worker falls back to a solo session
                                          for that cell)

Everything is written incrementally and flushed, so a session that later
times out still leaves valid demuxed cells for every node that finished.
"""

import hashlib
import json
import os
import traceback
from pathlib import Path

import pytest

DEMUX = os.environ.get("LK_BATCH_DEMUX", "")

_seq = 0


def _root() -> Path:
    return Path(DEMUX)


def _append(name: str, text: str):
    with (_root() / name).open("a") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _log_err(nodeid: str, err):
    _append("errors.log", f"{nodeid}\t{err!r}\n")


def _node_dir(nodeid: str) -> Path:
    global _seq
    sha = hashlib.sha1(nodeid.encode()).hexdigest()[:16]
    d = _root() / f"{_seq:03d}__{sha}"
    _seq += 1
    return d


def _demux_node(item, frames):
    """Replay the harness's fixture-teardown + sessionfinish combine on just
    this node's frames, into a private per-node directory."""
    from helpers.perf.core import PerfReport, combine_perf_reports
    from helpers.test_config import TestConfig

    module_stem = Path(str(item.fspath)).stem
    ndir = _node_dir(item.nodeid)
    temp = ndir / "temp_perf_data"
    out = ndir / "out"
    temp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # Same funnel as the solo path: append() (unique-column gate + schema
    # registry) then the single-schema assertion the module fixture applies.
    rep = PerfReport()
    for f in frames:
        rep.append(f, label=item.nodeid)
    rep.assert_single_schema(context=f"{item.nodeid} (lk-batch demux)")

    # Mirror conftest perf_report teardown: raw dump, post_process, post dump.
    # dump_csv prefixes TestConfig.PERF_DATA_DIR; an absolute Path overrides it.
    rep.dump_csv((temp / f"{module_stem}.master.csv").resolve())
    rep.post_process()
    rep.dump_csv((temp / f"{module_stem}.master.post.csv").resolve())

    # Mirror conftest pytest_sessionfinish: the harness's own combine, with
    # its two path globals pointed at this node's private dirs.  Restored
    # immediately; nothing else runs concurrently (single-threaded session).
    saved = (TestConfig.PERF_DATA_DIR, TestConfig.LLK_ROOT)
    try:
        TestConfig.PERF_DATA_DIR = temp
        TestConfig.LLK_ROOT = out
        combine_perf_reports()
    finally:
        TestConfig.PERF_DATA_DIR, TestConfig.LLK_ROOT = saved

    # Only a node whose combine SUCCEEDED is indexed (fail-closed contract).
    posts = list((out / "perf_data").glob("*/*.post.csv"))
    if not posts:
        raise RuntimeError("no combined post CSV produced")
    _append("index.tsv", f"{item.nodeid}\t{ndir.name}\t{module_stem}\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not DEMUX:
        yield
        return
    rep = None
    try:
        rep = item.funcargs.get("perf_report")
    except Exception:
        rep = None
    if rep is None:
        try:
            rep = item._request.getfixturevalue("perf_report")
        except Exception:
            rep = None
    # Frame-boundary bookkeeping.  Some test modules call perf_report.frame()
    # inside the test body, which COLLAPSES the fixture's frame list and
    # blurs the per-frame boundaries.  We snapshot both the identity of every
    # pre-existing frame and the pre-existing ROW count:
    #   - prefix identities intact  -> this node's frames are exactly the
    #     newly appended ones (the common perf_* module case);
    #   - prefix collapsed          -> concat order still puts pre-existing
    #     rows first (frame()/append never reorder), so this node's rows are
    #     the row-slice past the old count — accepted ONLY when every frame
    #     carries one identical column tuple (the same single-schema
    #     invariant the module fixture itself enforces at teardown), since a
    #     union-concat of mixed schemas would NaN-pad the slice;
    #   - anything else             -> DEMUX-SKIP (fail-closed: the worker
    #     runs that cell as a solo default session).
    pre_ids = pre_rows = None
    if rep is not None:
        try:
            pre_ids = [id(f) for f in rep._frames]
            pre_rows = sum(int(m.sum()) for m in rep._masks)
        except Exception:
            pre_ids = pre_rows = None
    outcome = yield
    if rep is None or pre_ids is None:
        return
    if outcome.excinfo is not None:
        return  # failed call: no cell is emitted, the worker reruns it solo
    try:
        cur = list(rep._frames)
        pre = len(pre_ids)
        if [id(f) for f in cur[:pre]] == pre_ids:
            frames = [f for f in cur[pre:] if not f.empty]
        else:
            sigs = {tuple(f.columns) for f in cur if not f.empty}
            if len(sigs) != 1:
                _log_err(item.nodeid, "DEMUX-SKIP: frame() collapse with mixed schemas")
                return
            merged = rep.frame()  # the harness's own collapse (idempotent)
            sliced = merged.iloc[pre_rows:].reset_index(drop=True)
            frames = [] if sliced.empty else [sliced]
    except Exception as e:  # noqa: BLE001 — never break the session
        _log_err(item.nodeid, e)
        return
    if not frames:
        return  # non-perf node (e.g. correctness gate): outcomes.jsonl suffices
    try:
        _demux_node(item, frames)
    except Exception as e:  # noqa: BLE001 — fail closed, cell reruns solo
        _log_err(
            item.nodeid, "".join(traceback.format_exception_only(type(e), e)).strip()
        )


def pytest_runtest_logreport(report):
    if not DEMUX:
        return
    _append(
        "outcomes.jsonl",
        json.dumps(
            {"node": report.nodeid, "when": report.when, "outcome": report.outcome}
        )
        + "\n",
    )


def pytest_sessionfinish(session, exitstatus):
    if not DEMUX:
        return
    (_root() / "session.json").write_text(
        json.dumps({"exitstatus": int(exitstatus)}) + "\n"
    )
