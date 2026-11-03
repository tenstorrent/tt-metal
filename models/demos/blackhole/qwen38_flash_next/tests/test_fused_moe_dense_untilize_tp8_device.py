# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""moe_dense with and without the G4 untilize (``untilize=False``, used by the model when QWEN38_MOE_EXPERTS_MAX_ROWS > 2):
scores, indices, partial and sigmoid bit for bit on every die, plus traced time of both forms per row count.
Run with the TP=8 mesh and X47_ROWS (default 1,2,3,5,8)."""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_dense as md
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import shared_expert as se

BF16 = ttnn.bfloat16
HIDDEN = se.HIDDEN
ROWS = [int(v) for v in os.environ.get("X47_ROWS", "1,2,3,5,8").split(",")]
PARAMS = {**DEVICE_PARAMS, "trace_region_size": 64 * 1024 * 1024}


def per_die(tensor):
    return [ttnn.to_torch(t).view(torch.int16) for t in ttnn.get_device_tensors(tensor)]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
def test_untilize_off_is_bitwise_and_faster(mesh_device):
    mesh = mesh_device
    results = Path(os.environ.get("X47_RESULTS", "/tmp/x47.jsonl"))
    shared = se.gate_configs(mesh)
    ok = True
    for rows in ROWS:
        gen = torch.Generator().manual_seed(4700 + rows)
        gate_t = torch.randn(HIDDEN, se.INTERMEDIATE, generator=gen) * 0.02
        up_t = torch.randn(HIDDEN, se.INTERMEDIATE, generator=gen) * 0.02
        scalar_t = torch.randn(HIDDEN, 1, generator=gen) * 0.02
        down_t = torch.randn(se.INTERMEDIATE, HIDDEN, generator=gen) * 0.02
        weights = se.upload_weights(mesh, gate_t, up_t, scalar_t, down_t)
        logits = ttnn.from_torch(
            torch.randn(1, 1, rows, rt.EXPERTS, generator=gen).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_dram = ttnn.from_torch(
            (torch.randn(1, 1, rows, HIDDEN, generator=gen) * 2).to(torch.bfloat16),
            dtype=BF16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        full_hidden = ttnn.to_memory_config(hidden_dram, shared["hidden_act_memory_config"])
        gate_up_scalar_ws = ttnn.linear(
            full_hidden,
            weights["gate_up_scalar"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=shared["gate_up_scalar_program_config"],
            compute_kernel_config=shared["compute_kernel_config"],
        )
        cfg = md.configs(mesh)
        scores = fp.allocate((1, 1, rows, md.TOP_K), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh)
        indices = fp.allocate((1, 1, rows, md.TOP_K), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh)

        def call(untilize):
            r = md.moe_dense(
                logits,
                gate_up_scalar_ws,
                weights["down"],
                full_hidden,
                scores=scores,
                indices=indices,
                untilize=untilize,
                **cfg,
            )
            return r

        on = call(True)
        on_vals = [per_die(t) for t in on[:4]]
        assert on[4] is not None
        off = call(False)
        assert off[4] is None
        bad = sum(int((a != b).sum()) for x, y in zip(on_vals, [per_die(t) for t in off[:4]]) for a, b in zip(x, y))
        row = dict(rows=rows, differing_words=bad)

        def timed(untilize):
            def f():
                r = call(untilize)
                for t in r[2:]:
                    if t is not None:
                        ttnn.deallocate(t)

            return round(traced_wall_us(mesh, f, replays=50)[0], 2)

        row["untilize_on_us"] = timed(True)
        row["untilize_off_us"] = timed(False)
        logger.info(json.dumps(row))
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        ok &= bad == 0
    assert ok
