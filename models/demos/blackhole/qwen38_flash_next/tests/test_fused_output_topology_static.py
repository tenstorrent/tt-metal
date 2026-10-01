# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every fused launch declares its written buffers' mesh placements in its program meta, or is listed here with the
reason it does not yet; ``fp.run_program`` stamps the declared placements after the launch (no device).

``ttnn.generic_op`` leaves an output with the allocation's default topology (``PlacementShard(0)`` on the four-die
line) instead of the placement the chain's op would have given it; a collective or a mesh-contract check downstream
misreads it, and a 1x1 device test cannot see it (the MoE dense composite at layer 0 and the GDN slab form both passed
their single-die tests and failed on the line, 2026-09-26).  The convention: the placement is stated where the program
is built, beside the bytes and FLOPs the legibility package already requires, and one helper applies it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

FUSED = Path(fp.__file__).resolve().parent
MODULES = sorted(FUSED.glob("*/__init__.py")) + sorted(p for p in FUSED.glob("*.py") if p.name != "__init__.py")
# (module, variant expression) -> why this launch's outputs are stamped elsewhere or not stamped (a shrinking list:
# a launch that declares ``outputs`` must leave it)
UNDECLARED: dict[tuple[str, str], str] = {
    ("final_mixer", '"down"'): "opt-in kernel; its caller stamps (ttnn/final_mixer hook)",
    ("final_mixer", '"low_rank"'): "opt-in kernel; its caller stamps",
    ("final_mixer", '"gate"'): "opt-in kernel; its caller stamps",
    ("final_mixer", '"normalize_down"'): "opt-in kernel; its caller stamps",
    ("final_mixer", '"low_rank_gate"'): "opt-in kernel; its caller stamps",
    (
        "gdn_post_rows",
        '"cast_verify" if chunks == 1 else "cast"',
    ): "gdn_prefill_rows.restamp_written stamps from buffer_layouts after the launch",
    (
        "gdn_post_rows",
        '"norm" if history else "norm_verify"',
    ): "gdn_prefill_rows / gdn_rows_wrap stamp the gated tile (dim 3) after the launch",
    (
        "gdn_pre_rows",
        '"verify_rows" if chunks == 1 else "slab"',
    ): "gdn_prefill_rows.restamp_written stamps from buffer_layouts after the launch",
    (
        "gdn_step",
        '"step"',
    ): "the GDN module's outputs are read on the same die (the state slot and the gated tile the out-projection consumes)",
    ("gr_fold", '"sem_probe"'): "a probe read back to the host",
    ("gr_read", '"down_project"'): "a partial sum the caller hands to its gather (module._mark_partial)",
    ("gr_read", '"noc_probe"'): "a probe read back to the host",
    ("gr_write", '"write"'): "the residual written in place keeps its placement",
    ("greedy_tail", '"scan"'): "per-die scan rows the merge program reads on the same die",
    ("greedy_tail", '"merge"'): "the merged rows are gathered by the caller's collective, which stamps",
    ("greedy_tail", '"resolve"'): "the token row is read back to the host",
    (
        "moe_post",
        '"post" if sigmoid is None else "post_sigmoid"',
    ): "the MoE module stamps its layer output after the add",
    ("ple", '"stats"'): "the PLE module stamps from the residual's topology after the launch",
    ("ple", '"normalize"'): "the PLE module stamps from the residual's topology after the launch",
    ("ple", '"gate"'): "the PLE module stamps from the residual's topology after the launch",
    (
        "mtp_accept",
        '"accept"',
    ): "the decision buffers and the statistics row are read by the split verify's tail on the same die",
    (
        "ple",
        '"conv_inject" if inject else "conv"',
    ): "the PLE module stamps from the residual's topology after the launch",
    ("ple", '"stats_lanes"'): "the PLE lanes body stamps from the residual's topology after the launch",
    ("ple", '"normalize_lanes"'): "the PLE lanes body stamps from the residual's topology after the launch",
    ("ple", '"gate_lanes"'): "the PLE lanes body stamps from the residual's topology after the launch",
    ("ple", '"conv_lanes"'): "the PLE lanes body stamps from the residual's topology after the launch",
    ("position_derive", '"advance"'): "the position scalar is advanced in place and keeps its placement",
    ("qsa_block", '"rms_norm_rows"'): "a mirror probe of one chain op (tests)",
    ("qsa_block", '"rope64"'): "a mirror probe of one chain op (tests)",
    ("qsa_block", '"index_tail"'): "the QSA module retags the index rows (ttnn/qsa.py _retag_tensor) after the launch",
    ("qsa_block", '"main_tail"'): "the QSA module retags the sparse query (dim 1) after the launch",
    (
        "qsa_block",
        '"main_tail_rows"',
    ): "the verify rows form (qsa_rows program 2): the QSA module retags the sparse query (dim 1) after the launch",
    ("qsa_block", '"post_attention"'): "the out-projection's activation shard is consumed on the same die",
    ("qsa_block", '"widen_partial"'): "the widened partial is consumed by the out-projection on the same die",
    ("qsa_block", '"selection_row"'): "the sparse indices are consumed by sparse_sdpa on the same die",
    ("qsa_block", '"lane_score_rows"'): "the lane windows are consumed by the lanes' merge on the same die",
    ("qsa_block", '"score_merge"'): "the merged scores are consumed by the selection on the same die",
    ("qsa_block", '"noc_probe"'): "a probe read back to the host",
    (
        "router_tail",
        '"lanes" if lanes else "single_core"',
    ): "the MoE module stamps the scores and indices (fused.program.stamp_topology) after the launch",
    ("sampler_tail", '"sample"'): "the token is read back to the host",
    ("sparse_sdpa_tiled", '"flash"'): "the slab attention's output is consumed by the out-projection on the same die",
    ("untilize_rows", '"rows"'): "the untilized rows are consumed on the same die",
}


def _dotted(node) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return node.id if isinstance(node, ast.Name) else ""


def _metas():
    for path in MODULES:
        source = path.read_text()
        module = path.parent.name if path.name == "__init__.py" else path.stem
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and _dotted(node.func) == "fp.program_meta":
                variant = ast.get_source_segment(source, node.args[1]) if len(node.args) > 1 else "?"
                yield module, variant, {k.arg for k in node.keywords}


def test_every_launch_declares_its_output_placements_or_is_listed() -> None:
    seen = set()
    for module, variant, keywords in _metas():
        key = (module, variant)
        seen.add(key)
        if "outputs" in keywords:
            assert key not in UNDECLARED, f"{key} declares its outputs: drop it from UNDECLARED"
        else:
            assert key in UNDECLARED, f"{key} neither declares outputs= nor is listed with a reason"
    assert set(UNDECLARED) <= seen, sorted(set(UNDECLARED) - seen)
    declared = {k for k in seen if k not in UNDECLARED}
    assert {("gr_fold", '"stats_normalize_down_gather"'), ("moe_dense", '"composite"')} <= declared


def test_program_meta_carries_the_declared_placements(expect_error) -> None:
    a, b, like = object(), object(), SimpleNamespace(tensor_topology=lambda: "T")
    meta = fp.program_meta("gr_read", "probe", 1, outputs=((a, 3), (b, None), (a, like)))
    assert meta.outputs == ((a, 3), (b, None), (a, like))
    assert fp.program_meta("x", "y", 1).outputs == ()
    with expect_error(ValueError, match="placement"):
        fp.program_meta("x", "y", 1, outputs=((a, "3"),))


def test_run_program_stamps_the_declared_placements_after_the_launch(monkeypatch) -> None:
    """The helper's contract on host fakes: the first io tensor's mesh, a shard dim or replicated per output, a tensor
    placement copied, and the launch's returned handle stamped with the output whose buffer it shares."""

    class Topology:
        def __init__(self, shape, coords):
            self._shape, self._coords = shape, coords

        def distribution_shape(self):
            return self._shape

        def mesh_coords(self):
            return self._coords

    class Fake:
        def __init__(self, address, topology=None):
            self.address, self.topology = address, topology

        def tensor_topology(self):
            return self.topology

        def update_tensor_topology(self, topology):
            self.topology = topology

        def buffer_address(self):
            return self.address

    stamped = []
    monkeypatch.setattr(
        fp.ttnn,
        "TensorTopology",
        lambda shape, placements, coords: stamped.append((shape, tuple(type(p).__name__ for p in placements), coords))
        or ("TT", shape, tuple(type(p).__name__ for p in placements)),
    )
    monkeypatch.setattr(fp.ttnn, "PlacementReplicate", lambda: SimpleNamespace(), raising=False)
    monkeypatch.setattr(fp.ttnn, "PlacementShard", lambda d: SimpleNamespace(dim=d), raising=False)
    result_handle = Fake(200)
    monkeypatch.setattr(fp.ttnn, "generic_op", lambda io, d: result_handle)
    ref = Fake(100, Topology("1x4", "coords"))
    sharded, replicated, copied = Fake(200), Fake(300), Fake(400)
    like = Fake(500, "LIKE")
    meta = fp.program_meta("gr_read", "probe", 1, outputs=((sharded, 3), (replicated, None), (copied, like)))
    out = fp.run_program([ref, sharded, replicated, copied], "descriptor", meta=meta)
    assert out is result_handle and result_handle.topology == sharded.topology  # the shared buffer's stamp
    assert sharded.topology == ("TT", "1x4", ("SimpleNamespace", "SimpleNamespace")) and replicated.topology[0] == "TT"
    assert copied.topology == "LIKE"
    assert len(stamped) == 3 and all(s[0] == "1x4" and s[2] == "coords" for s in stamped)
