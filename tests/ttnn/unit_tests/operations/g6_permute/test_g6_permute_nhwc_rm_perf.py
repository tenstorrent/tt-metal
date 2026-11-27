# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 perf sweep: nhwc_rm_transpose unit width (BLOCK_W) on Panoptic p4 [1, 128, 256, 32],
L1 (production) and DRAM.

Run under `scripts/run_safe_pytest.sh --profile --no-py-zones <file>::<node id>`: each case is one
fresh-cache dispatch; read DEVICE KERNEL DURATION from the ops CSV, in test order. Every case also
checks the output bit for bit.
DO NOT DELETE.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import g6_permute, g6_permute_plan

op_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
regime_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute_nhwc_rm_transpose")

NHWC = (0, 3, 1, 2)
PANOPTIC_P4 = (1, 128, 256, 32)
MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}

# (id, {knob: value}); BLOCK_W None is the cost-model choice.
VARIANTS = [
    ("auto", {}),
    ("w256", {"BLOCK_W": 256}),
    ("w128", {"BLOCK_W": 128}),
    ("w64", {"BLOCK_W": 64}),
    ("w32", {"BLOCK_W": 32}),
    ("w16", {"BLOCK_W": 16}),
]


@pytest.fixture
def fresh_cache():
    op_module._PROGRAM_CACHE.clear()
    yield
    op_module._PROGRAM_CACHE.clear()


@pytest.mark.parametrize("knobs", [v[1] for v in VARIANTS], ids=[v[0] for v in VARIANTS])
@pytest.mark.parametrize("buffer", ["l1", "dram"])
def test_block_w_sweep(buffer, knobs, device, fresh_cache, monkeypatch):
    for name, value in knobs.items():
        monkeypatch.setattr(regime_module, name, value)
    torch.manual_seed(5)
    torch_input = torch.randn(PANOPTIC_P4, dtype=torch.float32).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=MEMORY_CONFIG[buffer],
    )
    plan = g6_permute_plan(ttnn_input, NHWC)
    print(
        f"PLAN {buffer} {knobs}: block_w={plan.block_w} tail_w={plan.tail_w} units={plan.tensor_units} "
        f"cores={plan.num_cores} units/core={plan.units_per_core_g1}/{plan.units_per_core_g2} "
        f"block_units={plan.block_units} depth={plan.input_depth}"
    )
    actual = ttnn.to_torch(g6_permute(ttnn_input, NHWC))
    expected = torch.permute(torch_input, NHWC).contiguous()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
