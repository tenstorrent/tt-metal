# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_concat.py — do NOT regenerate this file
# from a capture.
#
# The captured case concatenates 15 x [1,1,32,8192] + 1 x [1,1,32,5376] on dim=-1
# into a [1,1,32,128256] (LM-head-width) L1 output = ~8.2 MB, which OOMs the L1 of
# the small-device configs this suite runs on (needs ~4.1 MB/bank across 2 banks vs
# a ~3.8 MB bank). Nothing about the concat path itself needs that width.
#
# This case keeps the same shape of the problem — several equal last-dim tiles plus
# one ragged tail, TILE bf16, dim=-1, L1-interleaved output — but at a tiny width
# (8 x 512 + 256 = 4352 -> 32x4352 output = ~0.27 MB), so it fits any grid while
# still exercising the multi-input last-dim concat.
#
# Unlike the other graph_ops tests, this one does NOT build its inputs through
# graph_case (which uploads them with ttnn.from_torch(layout=TILE)). On Quasar
# from_torch(TILE) routes to the mainline device tilize, which faults on these
# wide-short (1-tile-tall, 32-row) inputs -- Neo0TRISC2 MEM_READ_NO_RESPONSE in
# tilize_metal2.cpp -- so the concat under test never runs. See
# project_quasar_tilize_block_factory_unported. Instead we materialize each TILE
# input the way the model/debug tests do: a ROW_MAJOR upload + ttnn.experimental.
# quasar.tilize (the Quasar-safe tilize), which handles wide-short tensors. The op
# under test (ttnn.concat) is unchanged.
# ---------------------------------------------------------------------------
"""Small-input companion of ``ttnn.concat`` (multi-input last-dim concat that fits a small grid)."""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

_OP = ttnn.concat


def _t(width):
    return {"k": "t", "shape": [1, 1, 32, width], "dtype": "BFLOAT16", "layout": "TILE", "mem": None}


CASES = [
    {
        "id": "small_8x512+256_32_bf16_l1",
        "op": "ttnn.concat",
        "count": 1,
        "args": [
            {
                "k": "tlist",
                "tensors": [_t(512) for _ in range(8)] + [_t(256)],
            },
        ],
        "kwargs": {
            "dim": {"k": "lit", "v": -1},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 4352],
            },
        ],
    },
]


def _tile_bf16_dram(t_bf16, mesh_device):
    """RM upload + Quasar-safe tilize (mirrors the model/debug-test path; avoids from_torch(TILE))."""
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    qt = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qt or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_concat_small_grid(ttnn_mesh_device, reset_seeds, case):
    specs = case["args"][0]["tensors"]
    dim = case["kwargs"]["dim"]["v"]
    want_shape = tuple(case["outs"][0]["shape"])

    torch.manual_seed(0)
    parts = [torch.randn(*s["shape"], dtype=torch.bfloat16) for s in specs]
    tt = [_tile_bf16_dram(p, ttnn_mesh_device) for p in parts]

    out = ttnn.concat(tt, dim=dim, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(ttnn_mesh_device)

    assert tuple(out.shape) == want_shape, f"concat output shape {tuple(out.shape)} != {want_shape}"
    ref = torch.cat(parts, dim=dim)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=ttnn_mesh_device)
