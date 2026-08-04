# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests for the dry run: drift against the oracles, and the honesty rules.

The two oracle tests run in **subprocesses**: a dry run shadows `ttnn` (and, on an
interpreter without torch, `torch`) in ``sys.modules``, which must not leak into
any other test. Everything else here is pure analyzer and runs in-process.

    pytest models/tt_dit/tools/dit_analyzer/tests/test_dryrun.py
    python3 models/tt_dit/tools/dit_analyzer/tests/test_dryrun.py
"""

from __future__ import annotations

import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TOOLS)

from dit_analyzer import analyze_graph  # noqa: E402
from dit_analyzer.builder import GraphBuilder  # noqa: E402
from dit_analyzer.ir import Mesh  # noqa: E402
from dit_analyzer.report import render_report  # noqa: E402

SP, TP = 0, 1
MESH = Mesh(shape=(2, 4), axis_names=("sp", "tp"))


# -----------------------------------------------------------------------------
# drift: the dry run vs the hand-written graph for the same block
# -----------------------------------------------------------------------------
def _python(*argv: str) -> subprocess.CompletedProcess:
    """Run a fresh interpreter from tools/, so shim installs cannot leak here."""
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, cwd=TOOLS, timeout=900)


def _dryrun(*args: str) -> subprocess.CompletedProcess:
    return _python("-c", "import sys; from dit_analyzer.cli import main; sys.exit(main(sys.argv[1:]))", *args)


def test_ltx_ring_matches_oracle():
    """BH 4x8 / Ring: 6 provable duplicate gathers, matching examples/ltx.py."""
    proc = _dryrun("dryrun", "ltx_block", "--preset", "bh_4x8", "--check-oracle")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "unregistered ops: none" in proc.stdout, proc.stdout
    assert "identical multiset" in proc.stdout, proc.stdout
    assert "'duplicate_gather': 6" in proc.stdout, proc.stdout
    assert "PASS" in proc.stdout, proc.stdout


def test_ltx_linear_is_clean():
    """BH 2x4 / Linear: same source, nothing redundant."""
    proc = _dryrun("dryrun", "ltx_block", "--preset", "bh_2x4", "--check-oracle")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "identical multiset" in proc.stdout, proc.stdout
    assert "findings: none" in proc.stdout, proc.stdout
    assert "PASS" in proc.stdout, proc.stdout


def test_findings_name_the_model_call_site_not_the_library():
    """Blocker 44: lead with the frame that chose to gather, not the AGMM call."""
    proc = _dryrun("dryrun", "ltx_block", "--preset", "bh_4x8", "--check-oracle")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for line in proc.stdout.splitlines():
        if line.strip().startswith("models/tt_dit") and "via" not in line:
            assert "attention_ltx.py" in line, line
    assert "via models/tt_dit/layers/linear.py" in proc.stdout, proc.stdout


def test_dryrun_lists_its_substitutions():
    """A run is never quietly less faithful than it looks."""
    proc = _dryrun("dryrun", "ltx_block", "--preset", "bh_4x8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "torch:" in proc.stdout, proc.stdout


def test_arch_predicates_answer_from_the_mesh_under_test():
    """`is_blackhole()` reads `ttnn.get_arch_name()`, and the model keys chunk sizes
    and program configs off it -- a generic stub would answer False for every mesh."""
    proc = _python(
        "-c",
        "import sys; sys.path.insert(0, '.');"
        "from dit_analyzer.dryrun import install;"
        "install((4, 8), 'blackhole');"
        "import ttnn;"
        "print('arch', ttnn.get_arch_name(), ttnn.device.is_blackhole(), ttnn.device.is_wormhole_b0())",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "arch blackhole True False" in proc.stdout, proc.stdout


def test_host_env_does_not_import_models_before_the_shim():
    """tt_dit imports ttnn at module level: probing `models.*` too early would pull in
    real ttnn on any machine that has it, and then install() refuses to run."""
    proc = _python(
        "-c",
        "import sys; sys.path.insert(0, '.');"
        "from dit_analyzer.dryrun.hostenv import ensure_host_env;"
        "ensure_host_env();"
        "print('models imported early:', sorted(m for m in sys.modules if m.startswith('models')))",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "models imported early: []" in proc.stdout, proc.stdout


def test_targets_are_listed_without_a_target():
    proc = _dryrun("dryrun")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ltx_block" in proc.stdout and "--preset bh_4x8" in proc.stdout


# -----------------------------------------------------------------------------
# import shadowing stays inside its process
# -----------------------------------------------------------------------------
class _swapped_ttnn:
    """Temporarily put `module` at sys.modules['ttnn'], restoring whatever was there."""

    def __init__(self, module):
        self.module = module

    def __enter__(self):
        self.previous = sys.modules.get("ttnn")
        if self.module is None:
            sys.modules.pop("ttnn", None)
        else:
            sys.modules["ttnn"] = self.module
        return self

    def __exit__(self, *exc):
        if self.previous is None:
            sys.modules.pop("ttnn", None)
        else:
            sys.modules["ttnn"] = self.previous
        return False


def test_install_refuses_to_shadow_a_real_ttnn():
    """The shim must never displace real ttnn in a live process."""
    import types

    from dit_analyzer.dryrun.install import install

    real = types.ModuleType("ttnn")
    real.__file__ = "/opt/tt-metal/ttnn/__init__.py"
    with _swapped_ttnn(real):
        try:
            install((2, 4))
            raise AssertionError("install() shadowed a real ttnn")
        except RuntimeError as exc:
            assert "already imported" in str(exc), exc
    assert sys.modules.get("ttnn") is None or getattr(sys.modules["ttnn"], "__file__", "") != real.__file__


def test_assert_installed_rejects_a_missing_shim():
    """Nothing emits a graph without checking which module it recorded."""
    from dit_analyzer.dryrun.install import assert_installed

    with _swapped_ttnn(None):
        try:
            assert_installed()
            raise AssertionError("assert_installed() passed with no shim")
        except RuntimeError as exc:
            assert "not installed" in str(exc), exc


# -----------------------------------------------------------------------------
# withhold, don't guess
# -----------------------------------------------------------------------------
def _graph_with_unregistered_op(unregistered: bool):
    """A provably duplicate gather, optionally behind an op with no semantics.

    x is column-sharded on tp, gathered twice with nothing in between: the second
    gather is redundant. With ``unregistered=True`` the value passes through a call
    the shim has no rule for, so the shape it reports is an assumption.
    """
    b = GraphBuilder(name="withhold", mesh=MESH)
    x = b.input("x", [1, 512, 1024], shard={TP: 2})
    if unregistered:
        x = b.unregistered("ttnn.experimental.mesh_partition", [x], loc="models/tt_dit/models/transformers/fake.py:10")
    where = "models/tt_dit/models/transformers/fake.py:20"
    first = b.all_gather(x, dim=2, mesh_axis=TP, label="ag1", loc=where)
    second = b.all_gather(first, dim=2, mesh_axis=TP, label="ag2", loc=where)
    return b.finish([b.pointwise("silu", [second], label="out")])


def test_unregistered_op_withholds_the_finding():
    clean = analyze_graph(_graph_with_unregistered_op(unregistered=False))
    assert [f.rule for f in clean.findings] == ["unused_gather"], [f.rule for f in clean.findings]
    assert not clean.withheld

    blocked = analyze_graph(_graph_with_unregistered_op(unregistered=True))
    assert not blocked.findings, "a finding downstream of an unregistered op must not be reported"
    # Two, not one: the pessimistic definition of the unregistered op's output
    # (replicated, full regions) makes the *first* gather look redundant too. An
    # invented finding reading exactly like a real one is why these are withheld.
    assert len(blocked.withheld) == 2, blocked.withheld
    assert {w.finding.rule for w in blocked.withheld} == {"unused_gather"}
    assert blocked.missing_ops == ["ttnn.experimental.mesh_partition"], blocked.missing_ops


def test_unregistered_op_is_visible_in_the_report():
    report = analyze_graph(_graph_with_unregistered_op(unregistered=True))
    text = render_report(report)
    assert "findings blocked on op coverage" in text, text
    assert "ttnn.experimental.mesh_partition" in text
    assert "UNREGISTERED_OP" in text, "the diagnostic must say which op is missing"


def test_withheld_findings_name_their_source():
    report = analyze_graph(_graph_with_unregistered_op(unregistered=True))
    assert report.withheld[0].finding.loc, "a withheld finding still points at the source"


# -----------------------------------------------------------------------------
# source attribution
# -----------------------------------------------------------------------------
def test_attribution_prefers_model_code_over_library_code():
    from dit_analyzer.ir import Node

    node = Node(
        id="n",
        op="all_gather",
        loc="models/tt_dit/layers/linear.py:250",
        stack=[
            "models/tt_dit/layers/linear.py:250",
            "models/tt_dit/models/transformers/ltx/attention_ltx.py:428",
            "models/tt_dit/models/transformers/ltx/transformer_ltx.py:900",
        ],
    )
    assert node.call_site == "models/tt_dit/models/transformers/ltx/attention_ltx.py:428"
    assert node.attribution == [
        "models/tt_dit/models/transformers/ltx/attention_ltx.py:428",
        "models/tt_dit/layers/linear.py:250",
    ]


def test_attribution_falls_back_to_the_innermost_frame():
    from dit_analyzer.ir import Node

    library_only = Node(id="n", op="all_gather", loc="models/tt_dit/parallel/manager.py:501", stack=[])
    assert library_only.attribution == ["models/tt_dit/parallel/manager.py:501"]

    no_model_frame = Node(
        id="n",
        op="all_gather",
        loc="models/tt_dit/parallel/manager.py:501",
        stack=["models/tt_dit/parallel/manager.py:501", "models/tt_dit/layers/linear.py:250"],
    )
    assert no_model_frame.call_site == "models/tt_dit/layers/linear.py:250"


def test_stack_survives_a_json_round_trip():
    from dit_analyzer.ir import Graph

    graph = _graph_with_unregistered_op(unregistered=False)
    graph.nodes[0].stack = ["models/tt_dit/layers/linear.py:250"]
    assert Graph.from_json(graph.to_json()).nodes[0].stack == ["models/tt_dit/layers/linear.py:250"]


def _tests():
    return [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failed = 0
    for name, fn in _tests():
        try:
            fn()
            print("PASS %s" % name)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (name, str(exc)[:2000]))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s: %r" % (name, exc))
    print("\n%d/%d passed" % (len(_tests()) - failed, len(_tests())))
    sys.exit(1 if failed else 0)
