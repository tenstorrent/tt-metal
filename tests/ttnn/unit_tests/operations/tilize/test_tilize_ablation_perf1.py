# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf 1 ablation harness — PERF ONLY, no correctness asserts.

DO NOT DELETE. Ablation removes a stage's PAYLOAD and keeps its
synchronization, so the output is WRONG BY DESIGN and this file must never
assert on values.

Unlike Refinement 6's harness this needs no kernel edits: the payload switches
are `#ifdef TILIZE_ABLATE_{READS,WRITES,COMPUTE}` in the three kernels, plumbed
as kernel `defines` from `TILIZE_ABLATE` by
`tilize_program_descriptor._ablation_defines`.

Stages OVERLAP (the reader's NoC reads run against the TRISCs' tilize and the
writer's stores), so a stage removed ALONE under-counts itself — the surviving
partner fills the gap. Peel CUMULATIVELY:

    for A in "" compute compute,reads compute,reads,writes; do
        TILIZE_ABLATE=$A scripts/run_safe_pytest.sh --profile \
            tests/ttnn/unit_tests/operations/tilize/test_tilize_ablation_perf1.py
    done

The last rung — every stage stubbed at once — is the only run that licenses a
claim about the op as a whole ("most of this is overhead"); two separate
single-stage runs that are each flat say the stages are BALANCED, not that
either is irreducible.
"""

import os

import pytest
import torch

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

CASES = [
    ((1, 1, 32, 16384), "attention"),  # the flagged perf focus
    ((1, 1, 2048, 2048), "square_large"),
    ((1, 1, 16384, 32), "tall_narrow"),
]


@pytest.mark.parametrize("shape,label", CASES, ids=[c[1] for c in CASES])
def test_ablation(device, shape, label):
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    print(
        f"\n[ablate {os.environ.get('TILIZE_ABLATE', '<none>')} {label}] {shape}: "
        f"bw={plan.block_width_tiles} chunks={plan.num_w_chunks} rows/blk="
        f"{plan.tensor_row_blocks // plan.num_row_groups} blocks={plan.num_blocks_total} "
        f"cores={len(plan.assignment)}/{grid.x * grid.y}"
    )
    ttnn.synchronize_device(device)
