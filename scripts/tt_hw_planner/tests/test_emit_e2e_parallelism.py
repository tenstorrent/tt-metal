# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wiring the TP x DP decision into the emit-e2e EXECUTION half: the shared builder prompt (used by
BOTH the fsm and cc engines) must gain a chip-placement block that opens the selected mesh and maps
tensors, and must stay byte-identical when there is only one chip (non-breaking)."""
from scripts.tt_hw_planner.commands.emit_e2e import (
    _build_agent_prompt,
    _mesh_chip_count,
    _parallelism_prompt_block,
)
from scripts.tt_hw_planner.parallelism import ParallelConfig, select_parallelism


class _FakeReport:
    def __init__(self, grid, blocked):
        self.tp_grid = grid
        self._blocked = set(blocked)

    def has_blockers(self, tp=None):
        return tp in self._blocked


def test_mesh_chip_count_parsing():
    assert _mesh_chip_count(None) == 1
    assert _mesh_chip_count("") == 1
    assert _mesh_chip_count("2x2") == 4
    assert _mesh_chip_count("1x8") == 8
    assert _mesh_chip_count("garbage") == 1


def test_single_chip_prompt_is_unchanged():
    base = _build_agent_prompt(model_id="m/x", demo_dir="/tmp/x", pcc=0.95)
    noted = _build_agent_prompt(model_id="m/x", demo_dir="/tmp/x", pcc=0.95, parallel_note="")
    assert base == noted
    assert "CHIP PLACEMENT" not in base


def test_none_config_yields_empty_block():
    assert _parallelism_prompt_block(None) == ""
    assert _parallelism_prompt_block(ParallelConfig(tp=1, dp=1)) == ""


def test_multichip_block_consumes_selector():
    pc = select_parallelism(4, _FakeReport([1, 2, 4, 8, 32], blocked=[4]))
    assert (pc.tp, pc.dp) == (2, 2)
    block = _parallelism_prompt_block(pc)
    assert "TP=2 x DP=2" in block
    assert "open_mesh_device(ttnn.MeshShape(2, 2))" in block
    assert "ReplicateTensorToMesh" in block
    assert "ShardTensorToMesh" in block


def test_multichip_block_lands_in_builder_prompt():
    pc = select_parallelism(4, _FakeReport([1, 2, 4, 8, 32], blocked=[]))
    assert (pc.tp, pc.dp) == (4, 1)
    prompt = _build_agent_prompt(
        model_id="m/x", demo_dir="/tmp/x", pcc=0.9, parallel_note=_parallelism_prompt_block(pc)
    )
    assert "CHIP PLACEMENT" in prompt
    assert "TP=4 x DP=1" in prompt
