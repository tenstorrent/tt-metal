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
than the output. The "padded_capacity" routing reserves more rows per expert than the input has tokens. A last test
calls the indexed mode twice with fresh index / routing tensors and checks the program-cache hit reads the new ones.
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
# (token-ordered input rows, tokens routed to each local expert, rows reserved per expert or None for the smallest
# whole-tile capacity that holds the largest count)
ROUTINGS = {
    "aligned": (256, [96, 160, 64, 128], None),
    "ragged": (1024, [97, 161, 33, 127], None),
    "padded_capacity": (128, [80, 40], 256),
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


def _random_expert_weights(E):
    """gpt-oss-sized torch weights and biases for E experts."""
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
    return weights, biases


def _route(num_tokens, counts, offsets, rows):
    """Expert e takes counts[e] distinct tokens into rows offsets[e] onward of its region. Rows outside the occupied
    regions (slack and the spare tile) read token 0: finite, never part of an expert's rows."""
    tok_of_row = torch.zeros(rows, dtype=torch.int64)
    for e, count in enumerate(counts):
        tok_of_row[offsets[e] : offsets[e] + count] = torch.randperm(num_tokens)[:count]
    return tok_of_row


def _uint32(mesh_device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint32
    )


def _rm(mesh_device, t, dtype):
    return ttnn.from_torch(
        t,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_expert(mesh_device, table, weights, biases, max_tokens):
    return TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=len(weights),
        global_expert_idx_table=table,
        emb_dim=EMB,
        hidden_dim=HIDDEN,
        max_tokens=max_tokens,
        torch_weights=weights,
        torch_biases=biases,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
    )


def _check_regions(label, out, expected, offsets, counts):
    for e, count in enumerate(counts):
        region = out[offsets[e] : offsets[e] + count]
        assert torch.isfinite(region).all(), f"{label}, expert {e}: non-finite output"
        passing, pcc = comp_pcc(expected[e], region, PCC)
        logger.info(f"{label} expert {e} ({count} tokens): PCC {pcc}")
        assert passing, f"{label}, expert {e}: PCC {pcc} < {PCC}"


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("routing", list(ROUTINGS))
@pytest.mark.parametrize("hold_l1", [False, True], ids=["whole_l1", "l1_end"])
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_row_index_row_major_output(mesh_device, device_params, hold_l1, routing):
    torch.manual_seed(5)
    NUM_TOKENS, COUNTS, CAPACITY = ROUTINGS[routing]
    E = len(COUNTS)
    MAX_TOKENS = CAPACITY or -(-max(COUNTS) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE  # region stride, whole tiles
    SPARE = E * MAX_TOKENS  # the spare tile past every region
    ROWS = SPARE + ttnn.TILE_SIZE
    weights, biases = _random_expert_weights(E)
    x = torch.randn(NUM_TOKENS, EMB).to(torch.bfloat16)

    offsets = [e * MAX_TOKENS for e in range(E)]
    tok_of_row = _route(NUM_TOKENS, COUNTS, offsets, ROWS)
    with torch.no_grad():
        expected = [
            _reference(x[tok_of_row[offsets[e] : offsets[e] + COUNTS[e]]].float(), weights[e], biases[e])
            for e in range(E)
        ]

    table, counts, region_offsets = (
        _uint32(mesh_device, list(range(E))),
        _uint32(mesh_device, COUNTS),
        _uint32(mesh_device, offsets),
    )
    expert = _make_expert(mesh_device, table, weights, biases, MAX_TOKENS)
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
        _rm(mesh_device, x[tok_of_row], ttnn.bfloat16), *args, **common
    )
    gathered = ttnn.to_torch(gathered, mesh_composer=composer)[:ROWS].float()

    # New mode: rows read through x_row_index, ROW_MAJOR bf16 output with the spare row zeroed; with hold_l1 an L1
    # buffer stays allocated at the top of L1 and the op's circular buffers end below it.
    held, l1_end = _hold_l1(mesh_device, L1_HELD_BYTES) if hold_l1 else (None, 0)
    indexed = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        _rm(mesh_device, x, ttnn.bfloat16),
        *args,
        **common,
        l1_end=l1_end,
        x_row_index=_rm(mesh_device, tok_of_row.to(torch.int32).reshape(1, ROWS), ttnn.uint32),
        output_row_major=True,
        zero_output_row=SPARE,
    )
    assert indexed.layout == ttnn.ROW_MAJOR_LAYOUT and indexed.dtype == ttnn.bfloat16
    indexed = ttnn.to_torch(indexed, mesh_composer=composer)[:ROWS].float()
    if held is not None:
        ttnn.deallocate(held)

    _check_regions("gathered", gathered, expected, offsets, COUNTS)
    _check_regions("indexed", indexed, expected, offsets, COUNTS)
    for e in range(E):
        rows = slice(offsets[e], offsets[e] + COUNTS[e])
        # the same math on the same rows: only the output rounding (bf8 tiles vs bf16 rows) differs
        passing, pcc = comp_pcc(gathered[rows], indexed[rows], 0.99)
        assert passing, f"expert {e}: indexed vs gathered PCC {pcc}"
    assert torch.equal(indexed[SPARE], torch.zeros(EMB)), "zero_output_row was not zeroed"


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_row_index_cache_hit_fresh_routing(mesh_device, device_params):
    """Two indexed calls with the same specs but fresh x_row_index / counts / region-offset tensors and different
    routing. The second call is a program-cache hit, so it must read the second call's buffers (a stale index
    address would route the first call's tokens). The first call's tensors stay allocated, so the second call's
    tensors get different addresses."""
    torch.manual_seed(6)
    NUM_TOKENS = 256
    COUNTS_BY_CALL = ([96, 160, 64, 128], [128, 32, 160, 96])
    E = len(COUNTS_BY_CALL[0])
    MAX_TOKENS = 160
    SPARE = E * MAX_TOKENS
    ROWS = SPARE + ttnn.TILE_SIZE
    offsets = [e * MAX_TOKENS for e in range(E)]
    weights, biases = _random_expert_weights(E)
    x = torch.randn(NUM_TOKENS, EMB).to(torch.bfloat16)

    table = _uint32(mesh_device, list(range(E)))
    expert = _make_expert(mesh_device, table, weights, biases, MAX_TOKENS)
    x_tt = _rm(mesh_device, x, ttnn.bfloat16)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    kept = []  # every call's input tensors, alive until the end
    entries = []
    for call, counts in enumerate(COUNTS_BY_CALL):
        tok_of_row = _route(NUM_TOKENS, counts, offsets, ROWS)
        with torch.no_grad():
            expected = [
                _reference(x[tok_of_row[offsets[e] : offsets[e] + counts[e]]].float(), weights[e], biases[e])
                for e in range(E)
            ]
        inputs = (
            _uint32(mesh_device, offsets),
            _uint32(mesh_device, counts),
            _rm(mesh_device, tok_of_row.to(torch.int32).reshape(1, ROWS), ttnn.uint32),
        )
        kept.append(inputs)
        region_offsets, counts_tt, row_index = inputs
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x_tt,
            region_offsets,
            counts_tt,
            table,
            expert.gate_projs,
            expert.up_projs,
            expert.down_projs,
            max_dispatched_tokens_per_expert=MAX_TOKENS,
            activation=ttnn.RoutedExpertActivation.SwiGluOai,
            gate_biases=expert.gate_biases,
            up_biases=expert.up_biases,
            down_biases=expert.down_biases,
            x_row_index=row_index,
            output_row_major=True,
            zero_output_row=SPARE,
        )
        entries.append(mesh_device.num_program_cache_entries())
        out = ttnn.to_torch(out, mesh_composer=composer)[:ROWS].float()
        _check_regions(f"call {call}", out, expected, offsets, counts)
        assert torch.equal(out[SPARE], torch.zeros(EMB)), f"call {call}: zero_output_row was not zeroed"
    assert kept[0][2].buffer_address() != kept[1][2].buffer_address(), "the two x_row_index tensors share an address"
    assert entries[1] == entries[0], f"the second call compiled a new program ({entries[0]} -> {entries[1]} entries)"
