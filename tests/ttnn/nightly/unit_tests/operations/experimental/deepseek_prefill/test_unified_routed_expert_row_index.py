# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""unified_routed_expert_moe's optional row-index / row-major / L1-end modes, single chip, gpt-oss expert dims.

The default mode reads a dispatched buffer: row r of an expert's region holds a copy of the token routed there
(a dispatch gather made it). With x_row_index the op reads row x_row_index[r] of the token-ordered input instead,
so no gathered copy is needed. With output_row_major the down matmul's last pack untilizes, so the output is
ROW_MAJOR bf16 rows instead of bf8 tiles; zero_output_row names one output row the writer sets to zeros (a spare
row a combine points other chips' slots at). With l1_end the op's circular buffers end below that L1 address,
so buffers another component keeps resident in L1 above it can stay allocated.

Each case runs the same routing through the default mode (the host gathers the dispatched buffer) and through the
new mode, and checks both against a torch reference of every expert's rows. The "ragged" routing has token counts
that end inside a tile (the last tile-row of each region is partly padding) and a token-ordered input with more rows
than the output.
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc

SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip"),
]

EMB = 2880  # gpt-oss hidden size
HIDDEN = 2880  # gpt-oss expert intermediate size
# (token-ordered input rows, tokens routed to each local expert)
ROUTINGS = {
    "aligned": (256, [96, 160, 64, 128]),
    "ragged": (1024, [97, 161, 33, 127]),
}
L1_HELD_BYTES = 176 * 1024  # per core, about what gpt-oss's DFlash keeps resident across a prefill
PCC = 0.97


def _reference(x, w, b, alpha=1.702, limit=7.0):
    """Clamped SwiGLU-OAI expert with gate / up / down biases (gpt-oss), fp32, HF (out, in) weights."""
    gate = F.linear(x, w["gate_proj"], b["gate_proj_bias"]).clamp(max=limit)
    up = F.linear(x, w["up_proj"], b["up_proj_bias"]).clamp(min=-limit, max=limit)
    return F.linear((up + 1.0) * gate * torch.sigmoid(alpha * gate), w["down_proj"], b["down_proj_bias"])


def _hold_l1(mesh_device, held_bytes):
    """An interleaved L1 tensor of about held_bytes per bank at the top of L1; returns (tensor, its address)."""
    info = ttnn._ttnn.reports.get_device_info(mesh_device)
    tile_bytes = ttnn.TILE_SIZE * ttnn.TILE_SIZE * 2
    tiles_per_bank = held_bytes // tile_bytes
    held = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, ttnn.TILE_SIZE * tiles_per_bank, ttnn.TILE_SIZE * info.l1_num_banks]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mesh_device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    )
    return held, held.buffer_address()


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("routing", list(ROUTINGS))
@pytest.mark.parametrize("hold_l1", [False, True], ids=["whole_l1", "l1_end"])
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_row_index_row_major_output(mesh_device, device_params, hold_l1, routing):
    torch.manual_seed(5)
    NUM_TOKENS, COUNTS = ROUTINGS[routing]
    E = len(COUNTS)
    MAX_TOKENS = -(-max(COUNTS) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE  # region stride, whole tiles
    SPARE = E * MAX_TOKENS  # the spare tile past every region
    ROWS = SPARE + ttnn.TILE_SIZE
    weights = [
        {
            "gate_proj": torch.randn(HIDDEN, EMB) * 0.08,
            "up_proj": torch.randn(HIDDEN, EMB) * 0.08,
            "down_proj": torch.randn(EMB, HIDDEN) * 0.05,
        }
        for _ in range(E)
    ]
    biases = [
        {
            "gate_proj_bias": torch.randn(HIDDEN) * 0.5,
            "up_proj_bias": torch.randn(HIDDEN) * 0.5,
            "down_proj_bias": torch.randn(EMB) * 0.5,
        }
        for _ in range(E)
    ]
    x = torch.randn(NUM_TOKENS, EMB).to(torch.bfloat16)

    # Routing: expert e takes COUNTS[e] distinct tokens into rows offsets[e] onward of its region. Rows outside the
    # occupied regions (slack and the spare tile) read token 0: finite, never part of an expert's rows.
    offsets = [e * MAX_TOKENS for e in range(E)]
    tok_of_row = torch.zeros(ROWS, dtype=torch.int64)
    for e in range(E):
        tok_of_row[offsets[e] : offsets[e] + COUNTS[e]] = torch.randperm(NUM_TOKENS)[: COUNTS[e]]
    with torch.no_grad():
        expected = [
            _reference(x[tok_of_row[offsets[e] : offsets[e] + COUNTS[e]]].float(), weights[e], biases[e])
            for e in range(E)
        ]

    def idx(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint32
        )

    def rm(t, dtype):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    table, counts, region_offsets = idx(list(range(E))), idx(COUNTS), idx(offsets)
    expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=E,
        global_expert_idx_table=table,
        emb_dim=EMB,
        hidden_dim=HIDDEN,
        max_tokens=MAX_TOKENS,
        torch_weights=weights,
        torch_biases=biases,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
    )
    common = dict(
        max_dispatched_tokens_per_expert=MAX_TOKENS,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
        gate_biases=expert.gate_biases,
        up_biases=expert.up_biases,
        down_biases=expert.down_biases,
    )
    args = (region_offsets, counts, table, expert.gate_projs, expert.up_projs, expert.down_projs)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    # Default mode: the host gathers the dispatched buffer, the op writes bf8 tiles.
    gathered = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        rm(x[tok_of_row], ttnn.bfloat16), *args, **common
    )
    gathered = ttnn.to_torch(gathered, mesh_composer=composer)[:ROWS].float()

    # New mode: rows read through x_row_index, ROW_MAJOR bf16 output with the spare row zeroed; with hold_l1 an L1
    # buffer stays allocated at the top of L1 and the op's circular buffers end below it.
    held, l1_end = _hold_l1(mesh_device, L1_HELD_BYTES) if hold_l1 else (None, 0)
    indexed = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        rm(x, ttnn.bfloat16),
        *args,
        **common,
        l1_end=l1_end,
        x_row_index=rm(tok_of_row.to(torch.int32).reshape(1, ROWS), ttnn.uint32),
        output_row_major=True,
        zero_output_row=SPARE,
    )
    assert indexed.layout == ttnn.ROW_MAJOR_LAYOUT and indexed.dtype == ttnn.bfloat16
    indexed = ttnn.to_torch(indexed, mesh_composer=composer)[:ROWS].float()
    if held is not None:
        ttnn.deallocate(held)

    for e in range(E):
        rows = slice(offsets[e], offsets[e] + COUNTS[e])
        for label, out in (("gathered", gathered), ("indexed", indexed)):
            region = out[rows]
            assert torch.isfinite(region).all(), f"{label}, expert {e}: non-finite output"
            passing, pcc = comp_pcc(expected[e], region, PCC)
            logger.info(f"{label} expert {e} ({COUNTS[e]} tokens): PCC {pcc}")
            assert passing, f"{label}, expert {e}: PCC {pcc} < {PCC}"
        # the same math on the same rows: only the output rounding (bf8 tiles vs bf16 rows) differs
        passing, pcc = comp_pcc(gathered[rows], indexed[rows], 0.99)
        assert passing, f"expert {e}: indexed vs gathered PCC {pcc}"
    assert torch.equal(indexed[SPARE], torch.zeros(EMB)), "zero_output_row was not zeroed"
