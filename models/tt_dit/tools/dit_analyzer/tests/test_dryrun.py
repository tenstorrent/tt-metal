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


def test_sd35_vae_resnet_is_clean():
    """SD3.5 VAE ResnetBlock from source: the conv/group_norm family, no oracle.

    There is no hand-written VAE oracle, so the check is the honest one available:
    the block runs with zero unregistered ops, no analyzer diagnostics (conv2d,
    group_norm and the NHWC<->N,1,HW,C reshape all tracked, blockers 14/17), and
    no findings -- both vae_all_gathers are load-bearing. The conv/group_norm
    *shapes* are the shim's belief until on-device conformance (phase 11).
    """
    proc = _dryrun("dryrun", "sd35_vae_resnet", "--preset", "bh_2x4", "--analyze")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "unregistered ops: none" in proc.stdout, proc.stdout
    assert "no diagnostics" in proc.stdout, proc.stdout
    assert "no redundancy findings" in proc.stdout, proc.stdout


def test_fused_kernel_table_drives_emission_and_inspection():
    from dit_analyzer.dryrun import ops
    from dit_analyzer.dryrun.fused import FUSED_KERNELS, looks_fused

    # the two builder-driven kernels are declared in the table with their hidden
    # collective and stage order -- the shim binds to the table, not vice versa
    agmm = FUSED_KERNELS["all_gather_minimal_matmul_async"]
    assert agmm.collective == "all_gather" and agmm.order == "gather_then_matmul" and agmm.chunked
    mmrs = FUSED_KERNELS["minimal_matmul_strided_reduce_scatter_async"]
    assert mmrs.collective == "reduce_scatter" and mmrs.order == "matmul_then_scatter"
    assert ops.EXPERIMENTAL_OPS["all_gather_minimal_matmul_async"] is ops.all_gather_minimal_matmul_async

    # looks_fused: a new/unregistered kernel that hides a collective is guessable
    assert looks_fused("ttnn.experimental.all_gather_minimal_matmul_v2_async") == "all_gather"
    assert looks_fused("ttnn.experimental.minimal_matmul_reduce_scatter_fused") == "reduce_scatter"
    assert looks_fused("ring_joint_scaled_dot_product_attention") == "all_gather"
    # ... but a plain collective or a plain compute op is not a hidden-collective kernel
    assert looks_fused("ttnn.experimental.all_gather_async") is None
    assert looks_fused("ttnn.matmul") is None
    assert looks_fused("ttnn.experimental.mesh_partition") is None


def test_ops_missing_flags_a_fused_looking_unregistered_op():
    import io
    from contextlib import redirect_stdout

    from dit_analyzer.builder import GraphBuilder
    from dit_analyzer.cli import _ops_coverage
    from dit_analyzer.ir import Mesh

    b = GraphBuilder(name="fused_miss", mesh=Mesh(shape=(2, 4), axis_names=("sp", "tp")))
    x = b.input("x", [1, 512, 1024], shard={1: 2})
    y = b.unregistered("ttnn.experimental.all_gather_minimal_matmul_v2_async", [x], loc="models/tt_dit/x.py:1")
    buf = io.StringIO()
    with redirect_stdout(buf):
        _ops_coverage(b.finish([y]), fail=False)
    out = buf.getvalue()
    assert "looks like a fused kernel hiding a all_gather" in out, out
    assert "blocker 18" in out


def test_ops_missing_stub_generator_emits_both_halves():
    import io
    from contextlib import redirect_stdout

    from dit_analyzer.cli import _ops_coverage

    graph = _graph_with_unregistered_op(unregistered=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        _ops_coverage(graph, fail=False, stub=True)
    out = buf.getvalue()
    # the op is named, and both registration halves are stubbed
    assert "ttnn.experimental.mesh_partition" in out
    assert "def mesh_partition(" in out  # shim shape rule, canonical name
    assert "recorder.emit(" in out
    assert "_mesh_partition_apply" in out and "_mesh_partition_demand" in out  # analyzer OpSpec
    assert 'register(OpSpec("mesh_partition"' in out
    assert "TODO" in out
    # and without --stub it points at the flag instead of dumping skeletons
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        _ops_coverage(graph, fail=False, stub=False)
    assert "--stub" in buf2.getvalue() and "def mesh_partition(" not in buf2.getvalue()


def test_sd35_block_matches_oracle():
    """SD3.5-large joint block from source: a second block, second oracle.

    Exercises the fused split_query_key_value_and_split_heads shim and the
    dit_rms_norm_unary_fused spec (both first used here, not by LTX), plus the
    38->40 padded-head shape math. Every collective is load-bearing -> 0 findings.
    """
    proc = _dryrun("dryrun", "sd35_block", "--preset", "bh_2x4", "--check-oracle")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "unregistered ops: none" in proc.stdout, proc.stdout
    assert "identical multiset" in proc.stdout, proc.stdout
    assert "findings: none" in proc.stdout, proc.stdout
    assert "PASS" in proc.stdout, proc.stdout
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


def test_weight_chunk_width_matches_torch_chunk():
    # to_qkv(chunks=n) splits weight columns with torch.chunk; the shim's block
    # width must be exactly torch's, ceil-to-leading not floor (blocker 12).
    from dit_analyzer.region import shard_chunk_size

    try:
        import torch
    except ImportError:
        return  # torch-less CI: shard_chunk_size is covered in test_dit_analyzer
    for n_cols, count in [(3072, 3), (1024, 2), (30, 4), (38, 4)]:
        widths = [t.shape[-1] for t in torch.empty(8, n_cols).chunk(count, dim=-1)]
        chunk = shard_chunk_size(n_cols, count)
        expected = [min(chunk, max(0, n_cols - i * chunk)) for i in range(count)]
        assert widths == [w for w in expected if w > 0], (n_cols, count, widths, expected)


def test_fused_swiglu_preprocessing_runs_on_meta_and_preserves_shape():
    # the roadmap's "run the real _prepare_torch_state on meta tensors": the
    # swiglu reorder is a pure reshape/permute, so it runs with no bytes and is
    # shape-preserving -- which is why total_shape already captures it and the
    # residual (blocker 12) is column identity, not shape.
    try:
        import torch

        from models.tt_dit.utils.tensor import prepare_for_fused_swiglu
    except ImportError:
        return
    w = torch.empty(4096, 2 * 4096, device="meta")  # packed [.., 2N] swiglu weight
    out = prepare_for_fused_swiglu(w, ndev=4)
    assert tuple(out.shape) == (4096, 2 * 4096)  # reordered, not reshaped
    assert out.is_meta  # no bytes were ever materialised


def test_checkpoint_flags_match_tt_dit_detection():
    from dit_analyzer.dryrun.checkpoint import LTX_ADALN_KEY, declared, ltx_flags

    inner = 4096
    # audio+video: to_gate_logits present -> has_gate; adaln rows 9*inner > 6*inner
    av = declared(
        keys=[LTX_ADALN_KEY, "model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight"],
        shapes={LTX_ADALN_KEY: (9 * inner, inner)},
    )
    assert ltx_flags(av, inner) == {"has_gate": True, "cross_attention_adaln": True}
    # a video-only checkpoint: no gate key, small adaln (6*inner, not > 6*inner)
    vo = declared(keys=[LTX_ADALN_KEY], shapes={LTX_ADALN_KEY: (6 * inner, inner)})
    assert ltx_flags(vo, inner) == {"has_gate": False, "cross_attention_adaln": False}
    # adaln weight absent -> tt_dit's fallback is True
    none = declared(keys=["model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight"])
    assert ltx_flags(none, inner) == {"has_gate": True, "cross_attention_adaln": True}
    # key present but shape unknown (index.json source) -> adaln reported unknown
    keys_only = declared(keys=[LTX_ADALN_KEY])
    assert ltx_flags(keys_only, inner)["cross_attention_adaln"] is None


def test_safetensors_header_reads_shapes_without_weight_bytes():
    import json
    import struct
    import tempfile

    from dit_analyzer.dryrun.checkpoint import from_safetensors_header

    # a minimal, valid safetensors file: <u64 header_len><json header><data>
    header = {
        "w.gate": {"dtype": "F32", "shape": [16, 4], "data_offsets": [0, 256]},
        "__metadata__": {"format": "pt"},
    }
    blob = json.dumps(header).encode()
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00" * 256)  # tensor data we must never read
        path = f.name
    index = from_safetensors_header(path)
    os.unlink(path)
    assert index.keys == ["w.gate"]  # __metadata__ excluded
    assert index.shape_of("w.gate") == (16, 4)
    assert index.any_key_contains("gate")
    assert "safetensors header" in index.source


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
