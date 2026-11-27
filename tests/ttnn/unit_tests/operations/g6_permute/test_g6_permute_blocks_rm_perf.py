# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Refinements 3 and 5 perf sweep: blocks_rm_gather and blocks_rm_bank_coalesced knobs on FLUX_74
[1, 64, 128, 32, 2, 2] (LOOSE_CASES[0]) and FLUX_73, DRAM and L1, against Refinement 1's plan and
Refinement 3's gather.

Run under `scripts/run_safe_pytest.sh --profile --no-py-zones <file>::<node id>`: each case is one
fresh-cache dispatch; read DEVICE KERNEL DURATION from the ops CSV. Every case also checks the output
bit for bit, so a knob setting that breaks the gather fails here too.
DO NOT DELETE.
"""

import importlib
from fractions import Fraction

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import g6_permute

op_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
regime_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute_blocks_rm_gather")

BLOCKS = (0, 1, 4, 2, 5, 3)
FLUX_74 = (1, 64, 128, 32, 2, 2)
FLUX_73 = (1, 32, 64, 32, 2, 2)

# (id, {knob: value}) — every knob is a module attribute of the regime.
# Pseudo-knob: limit the worker grid to this many rows (full width).
ROWS = "rows"
# Pseudo-knob: place the input (and so the output) in L1 instead of DRAM.
L1 = "l1"
# Refinement 1's shipped plan: the whole per-core assignment as one block, page order, one NoC.
R1 = {
    "READ_MODE": "gather",
    "READ_ALT_NOC_SHARE": Fraction(0),
    "BANK_MAJOR": False,
    "BLOCK_PAGES": None,
    "BLOCK_H_LIMIT": None,
    "INPUT_DEPTH": 2,
}
# Refinement 3's shipped plan: one read per page, both NoCs, bank-major runs.
R3 = {"READ_MODE": "gather"}
# The coalesced read on any placement.
CO = {"READ_MODE": "coalesced"}
VARIANTS = [
    ("default", {}),
    ("r1", R1),
    ("r3", R3),
    ("co_bp64", {**CO, "COALESCED_BLOCK_PAGES": 64}),
    ("co_bp128", {**CO, "COALESCED_BLOCK_PAGES": 128}),
    ("co_bp256", {**CO, "COALESCED_BLOCK_PAGES": 256}),
    ("co_bp256_sd1", {**CO, "COALESCED_BLOCK_PAGES": 256, "STAGING_DEPTH": 1}),
    ("co_bp256_noc0", {**CO, "COALESCED_BLOCK_PAGES": 256, "COALESCED_ALT_NOC_SHARE": Fraction(0)}),
    ("co_bp256_d3", {**CO, "COALESCED_BLOCK_PAGES": 256, "INPUT_DEPTH": 3}),
    ("co_bp128_d4", {**CO, "COALESCED_BLOCK_PAGES": 128, "INPUT_DEPTH": 4}),
    ("co_stagger", {**CO, "COALESCED_BANK_STAGGER": True}),
    ("co_bp1024_stagger", {**CO, "COALESCED_BLOCK_PAGES": 1024, "COALESCED_BANK_STAGGER": True}),
    ("co_bp4096_stagger", {**CO, "COALESCED_BLOCK_PAGES": 4096, "COALESCED_BANK_STAGGER": True}),
    ("co_bp512", {**CO, "COALESCED_BLOCK_PAGES": 512}),
    ("co_bp1024", {**CO, "COALESCED_BLOCK_PAGES": 1024}),
    ("co_bp2048", {**CO, "COALESCED_BLOCK_PAGES": 2048}),
    ("co_bp4096", {**CO, "COALESCED_BLOCK_PAGES": 4096}),
    ("co_sd1", {**CO, "STAGING_DEPTH": 1}),
    ("co_noc0", {**CO, "COALESCED_ALT_NOC_SHARE": Fraction(0)}),
    ("co_d4", {**CO, "INPUT_DEPTH": 4}),
    ("l1_r3", {L1: True, **R3}),
    ("l1_co_bp1024", {L1: True, **CO, "COALESCED_BLOCK_PAGES": 1024}),
    ("l1_co_bp4096", {L1: True, **CO, "COALESCED_BLOCK_PAGES": 4096}),
    ("bp64", {**R3, "BLOCK_PAGES": 64}),
    ("d4", {**R3, "INPUT_DEPTH": 4}),
    ("l1_default", {L1: True}),
    ("l1_r1", {L1: True, **R1}),
    ("l1_bp64", {L1: True, "BLOCK_PAGES": 64}),
    ("l1_d4", {L1: True, "INPUT_DEPTH": 4}),
]


@pytest.fixture
def fresh_cache():
    op_module._PROGRAM_CACHE.clear()
    yield
    op_module._PROGRAM_CACHE.clear()


def _run(shape, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(5)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    actual = ttnn.to_torch(g6_permute(ttnn_input, BLOCKS))
    expected = torch.permute(torch_input, BLOCKS).contiguous()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("knobs", [v[1] for v in VARIANTS], ids=[v[0] for v in VARIANTS])
@pytest.mark.parametrize("shape", [FLUX_74, FLUX_73], ids=["flux74", "flux73"])
def test_knob_sweep(shape, knobs, device, fresh_cache, monkeypatch):
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    for name, value in knobs.items():
        if name == L1:
            memory_config = ttnn.L1_MEMORY_CONFIG
        elif name == ROWS:
            full = device.compute_with_storage_grid_size()
            monkeypatch.setattr(regime_module, "worker_grid", lambda _device, r=value: ttnn.CoreCoord(full.x, r))
        else:
            monkeypatch.setattr(regime_module, name, value)
    _run(shape, device, memory_config)
