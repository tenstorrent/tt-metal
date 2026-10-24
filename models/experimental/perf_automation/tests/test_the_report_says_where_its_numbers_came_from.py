"""A number that was never measured rendered exactly like one that was.

Three defects in this file share one shape: the tool knew it had no evidence and the page did not
say so.

  * the memory roof and the GB/s column both divide by a stage's read set, and a pinned measurement,
    this build's measurement and a checkpoint estimate all printed as a bare figure. A byte hook that
    watched a ttnn class this build never instantiates returned 0 for every stage of every run for
    two days, and nothing on the page disagreed -- the estimate fallback kept printing plausible
    numbers.
  * the fidelity ladder marks one rung "in use". Without stage signposts that verdict is taken over
    the WHOLE profile and stamped on every stack, and voxtral's stacks do not share a rung: 60.3% of
    matmul FLOPs are LoFi against 39.7% HiFi4, so encode and prefill were priced at LoFi's 702 TFLOPS
    while they run HiFi4 at 175.5 -- a ceiling 4x too generous.

The fallbacks themselves are right. Inventing a per-stage peak would be worse than sharing one.
Presenting either as though it were a measurement is the defect."""
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _src(rel):
    return (_PA / rel).read_text()


def test_a_stage_records_which_source_its_bytes_came_from():
    """Reported BY THE OWNER. A caller that re-ran the pinned->measured->estimate chain to find out
    which one answered would be a second opinion on the quantity this function exists to own -- so
    stage_read_bytes returns the source, and _stage_roofs asks for it."""
    body = _src("cc_optimize/summary.py")
    assert '"bytes_source": _b_src' in body, "the roof no longer carries where its bytes came from"
    assert "with_source: bool = False" in body, "the owner cannot report which source answered"
    assert "with_source=True," in body, "_stage_roofs does not ask the owner for the source"
    for token in ('_ret(_p, "pinned")', '_ret(_m, "measured")', '_ret(_e or 0, "estimate")'):
        assert token in body, "missing byte-source case: %s" % token
    i = body.index("def _stage_roofs")
    fn = body[i : body.index("\ndef ", i + 10)]
    assert "_pinned_stage_bytes(name" not in fn, "the chain was re-derived inline to label the number"


def test_the_memory_row_prints_the_source_when_it_is_not_pinned():
    body = _src("cc_optimize/summary.py")
    assert '"bytes: %s" % _bsrc' in body, "the GB/s row does not state its byte source"
    assert '_bsrc in ("", "pinned")' in body, "a pinned read set should render without a caveat"


def test_a_shared_peak_says_it_is_shared():
    body = _src("cc_optimize/summary.py")
    assert "_shared_peak" in body
    assert "one peak shared by every stack" in body, "the ladder still presents a fallback as a measurement"


def test_the_byte_anchor_is_written_where_the_unit_is_known():
    """run.py's writer is guarded on `if _unit:` BEFORE any trace reports one, so it never fires.
    _persist_throughput has the unit in hand and already pins the other two roof inputs."""
    body = _src("cc_optimize/perf_mcp.py")
    i = body.index("def _persist_throughput")
    j = body.index("\ndef ", i + 10)
    fn = body[i:j]
    assert "KIND_ACTIVE_BYTES" in fn, "the memory roof's numerator is still unpinned"
    assert "_is_real_unit(_u)" in fn, "the anchor must refuse a unit it cannot recover"
    assert "KIND_PEAK_FLOPS" in fn, "sanity: the compute roof was always pinned here"


def test_git_revert_puts_the_stage_marks_back():
    """The revert restores the whole model dir, and the marks are an uncommitted edit inside it."""
    body = _src("cc_optimize/perf_mcp.py")
    i = body.index("def git_revert")
    j = body.index("\ndef ", i + 10)
    fn = body[i:j]
    assert "_reinject_stage_marks()" in fn, "a revert still erases the marks and leaves them erased"
    assert "def _reinject_stage_marks" in body and "def _perf_test_paths" in body


def test_the_reinjection_reads_the_manifest_rather_than_guessing_filenames():
    body = _src("cc_optimize/perf_mcp.py")
    i = body.index("def _perf_test_paths")
    fn = body[i : body.index("\ndef ", i + 10)]
    assert "perf_test_resolved" in fn and "components" in fn
    for hardcoded in ("test_main_perf", "voxtral", "tests/e2e"):
        assert hardcoded not in fn, "the path list hardcodes %r" % hardcoded
