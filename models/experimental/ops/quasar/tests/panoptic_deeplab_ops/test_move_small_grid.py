# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_move.py — do NOT regenerate this file from a capture.
#
# The suite's \`\`-m emulator\`\` subset caps the activation at 48 x 48 x 256 elements, the
# total input footprint at 16 MiB and an L1 shard grid at 8 cores. The captured Panoptic-DeepLab
# call is at 512 x 1024 resolution on a 20-core device, so it is outside that subset. The variants
# below keep the call and shrink the tensors; dtypes are bfloat16 directly (the captured
# bfloat8_b has no Quasar format), so no TTNN_GRAPH_OPS_BF8_AS_BF16 is needed.
#
# The captured calls move L1-sharded activations (block- and height-sharded, on 16 to 20 cores,
# up to [1, 1, 8192, 512]) to a new L1 address, so their shard grids need more cores than the
# 2-node device has. The variants keep the op, the layouts and the TILE format and shard the
# activation over a 2 x 1 grid ([[0, 0, 1, 0]]): HEIGHT_SHARDED [1, 1, 128, 64] with 64-row
# shards, and BLOCK_SHARDED [1, 1, 128, 128] with 128 x 64 shards. The output keeps the input's
# memory config. The golden is the identity.
#
# A sharded move only launches its copy kernel when the output lands at a different address than
# the input (ttnn/cpp/ttnn/operations/experimental/quasar/move/move.cpp, move_sharded): it frees the
# input, allocates the output, and returns at once when the allocator hands back the same address.
# On an unfragmented L1, where the input is the only buffer, that is always the case, so the move
# would pass without running the kernel. The test therefore allocates a scratch buffer with the input's
# spec BEFORE the input, so the input sits above it, and frees the scratch right before the move. The
# free hole below the input fits the output, which lands there, and the test asserts that the
# output address differs from the input address.
# (Same technique as tests/tt_eager/python_api_testing/unit_testing/misc/test_move_sharded.py.)
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.move``: sharded L1 moves on a 2 x 1 grid."""

import copy

import pytest

import ttnn
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.move

_GRID = [[0, 0, 1, 0]]


def _case(case_id, shape, layout, shard_shape):
    mem = {
        "layout": layout,
        "buffer": "L1",
        "shard": {"grid": copy.deepcopy(_GRID), "shape": list(shard_shape), "orientation": "ROW_MAJOR"},
    }
    return {
        "id": case_id,
        "op": "ttnn.move",
        "count": 1,
        "args": [{"k": "t", "shape": list(shape), "dtype": "BFLOAT16", "layout": "TILE", "mem": mem}],
        "kwargs": {},
        "outs": [
            {"dtype": "BFLOAT16", "k": "t", "layout": "TILE", "mem": copy.deepcopy(mem), "shape": list(shape)},
        ],
    }


CASES = [
    _case("00_128x64_bf16_hs-l1", [1, 1, 128, 64], "HEIGHT_SHARDED", [64, 64]),
    _case("01_128x128_bf16_bs-l1", [1, 1, 128, 128], "BLOCK_SHARDED", [128, 64]),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_move_small_grid(ttnn_mesh_device, reset_seeds, case):
    state = {}

    def allocate_scratch():
        # Same shape, dtype, layout and shard spec as the input, so the hole it leaves fits the output.
        state["scratch"], _ = G.build_tensor(case["args"][0], ttnn_mesh_device, case, case["op"], "scratch")

    def free_scratch_and_note_input(args, kwargs):
        state["scratch"].deallocate()
        state["input_address"] = args[0].buffer_address()

    try:
        out = G.run_case(
            _OP,
            case,
            ttnn_mesh_device,
            before_inputs=allocate_scratch,
            before_op=free_scratch_and_note_input,
        )
    finally:
        scratch = state.get("scratch")
        if scratch is not None and scratch.is_allocated():
            scratch.deallocate()

    assert out.buffer_address() != state["input_address"], (
        f"ttnn.move returned the input's address {state['input_address']:#x}: the allocator had no lower "
        "free block, so the copy kernel was not exercised"
    )
