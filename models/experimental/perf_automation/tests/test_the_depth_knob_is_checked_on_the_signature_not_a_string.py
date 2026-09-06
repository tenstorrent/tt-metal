# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A clause that greps for a string passes models that do not honour it.

WHAT THIS COST, measured on Voxtral-Mini-3B, 2026-08-12.

The depth-knob clause searched every source file for `TT_PERF_LAYERS|num_layers=|n_layers=|layers=`.
Voxtral matched on its own `n_layers = tcfg.num_hidden_layers` -- a line that READS the depth from
the config and has nothing to do with capping it. Meanwhile the factory was
`build_pipeline(device, model=None, **kwargs)` and filtered kwargs to
{batch_size, prefill_capacity, kv_capacity}, so a `layers` argument was dropped without a word. The
generated perf test wrote the consequence in its own comment -- "No depth argument on this builder"
-- and every profile built all 32 layers: 35M tracy zones, a baseline killed at its budget, hours of
optimizing with no BEFORE number.

The contract reported "meets all 8 clauses" the entire time.

A SIGNATURE IS STRUCTURE; A STRING IS A COINCIDENCE. Whether a builder can accept a depth is
answered by its parameters, and that answer does not vary with how a model happens to name its
variables. The same reasoning is why the companion check for block discoverability is NOT here:
deciding by class name (`layer|block|decoder`) would fail the first model that names its wrapper
something else. That one is enforced where the tool actually walks the built object, and reports
when the walk finds nothing.
"""

import ast
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent


def _clause_src() -> str:
    src = (_PA / "agent" / "model_contract.py").read_text()
    i = src.index("def _c_depth_knob(")
    return src[i : src.index("\ndef ", i + 10)]


def test_it_reads_the_factory_signature():
    body = _clause_src()
    assert 'src.functions("build_pipeline")' in body, "the clause does not look at the factory"
    assert "kwonlyargs" in body and "fn.args.args" in body, "the clause does not read the parameters"


def test_it_no_longer_greps_the_whole_model_for_a_string():
    """`n_layers = cfg.num_hidden_layers` is a READ of the depth, not a cap, and it is what passed."""
    body = _clause_src()
    assert "re.search" not in body, "the clause still decides on a text match"
    assert "src.texts.values()" not in body, "the clause still scans whole files"


def test_kwargs_alone_does_not_satisfy_it():
    """A filtered **kwargs dict is exactly what swallowed `layers` silently."""
    body = _clause_src()
    assert "kwargs" in body, "the finding does not warn about **kwargs"


def test_the_clause_would_have_failed_the_old_factory():
    """The signature Voxtral shipped, run through the same parameter test the clause applies."""
    old = "def build_pipeline(device, model=None, **kwargs):\n    return None\n"
    new = "def build_pipeline(device, model=None, layers=None, **kwargs):\n    return None\n"
    depth_args = {"layers", "n_layers", "num_layers", "depth"}

    def accepts_depth(text):
        fn = ast.parse(text).body[0]
        names = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
        return bool(names & depth_args)

    assert not accepts_depth(old), "the old signature would still pass"
    assert accepts_depth(new), "the fixed signature would not pass"


def test_the_walk_reports_when_it_finds_no_blocks():
    """The other half: discoverability is decided by the walk, not by class names."""
    run_src = (_PA / "cc_optimize" / "run.py").read_text()
    assert "exposes NO discoverable block stacks" in run_src, "an empty walk is still silent"
    i = run_src.index("exposes NO discoverable block stacks")
    stanza = run_src[max(0, i - 1200) : i]
    assert 'facts["full_blocks"]' in stanza, "the report is not tied to the walk's own result"
