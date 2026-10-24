"""Padding to a fixed tuple is a ladder borrowed from whatever model it was written for.

On Voxtral's 1500-frame audio encoder every rung of (64,128,256,512) sits below the real length, so
the bucketing does nothing; on a short model the first rung already doubles the work."""
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from agent.before_loop import _seq_retry_candidates  # noqa: E402


def test_candidates_grow_from_the_models_own_sequence():
    """The borrowed tuple contributed NOTHING once current_seq passed its top rung -- a model at
    1500 got an empty fallback and the retry had nothing to try."""
    assert _seq_retry_candidates("", 1500), "a long sequence still gets no candidates"
    assert all(c > 1500 for c in _seq_retry_candidates("", 1500))
    assert all(c > 64 for c in _seq_retry_candidates("", 64))


def test_every_candidate_is_tile_aligned():
    """A sequence off the tile height cannot shard cleanly and reproduces the same failure class."""
    from agent.tp import TILE

    for cur in (64, 100, 128, 512, 1500, 3000):
        for c in _seq_retry_candidates("", cur):
            assert c % TILE == 0, "candidate %d for current=%d is not tile-aligned" % (c, cur)


def test_the_error_derived_branch_still_leads():
    """The scaling branch reads block_h/num_cores_r/Mt out of the error -- that is the precise
    answer and must come before the grown fallback."""
    err = "block_h (4) ... num_cores_r=8 ... Mt=16"
    got = _seq_retry_candidates(err, 128)
    assert got[0] == 256, "the shard-derived candidate no longer leads: %s" % (got,)


def _code_only(path):
    """Source with comments and docstrings stripped. The fix deliberately QUOTES the old tuple so the
    next reader knows what was wrong; prose must not read as the defect still being present."""
    import ast
    import io
    import tokenize

    src = Path(path).read_text()
    toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type != tokenize.COMMENT]
    stripped = tokenize.untokenize(toks)
    tree = ast.parse(stripped)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                stripped = stripped.replace(d, "")
    return stripped


def test_no_borrowed_ladder_remains_anywhere():
    assert "(256, 384, 512, 768)" not in _code_only(_PA / "agent" / "before_loop.py")
    guide = (_PA / "GUIDELINES" / "08_DECODE_PREFILL_AND_MULTIDEVICE.md").read_text()
    assert "BUCKETS = (64, 128, 256, 512)" not in guide, "the playbook still teaches a fixed ladder"
    assert "DERIVE THE LADDER FROM THIS MODEL" in guide
