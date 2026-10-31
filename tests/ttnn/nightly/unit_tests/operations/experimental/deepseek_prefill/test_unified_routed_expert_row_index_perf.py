# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time gate for unified_routed_expert_moe's row-index / row-major-output modes, measured with the real-time
program profiler on one chip at gpt-oss-20b's routed-prefill shapes (2048 tokens, top-4 of 32 experts, 8 local).

The modes save work outside the op, so the test times the whole sequence a caller runs before and after:

- gathered (default modes): x (TILE) is untilized into L1, ttnn.embedding gathers the dispatched buffer from it (one
  copy of a token per (token, expert) pair; rows outside every expert region read token 0 from a per-core L1 copy
  via padding_idx), the op writes bf8 tiles, ttnn.experimental.deepseek_prefill.insert zeroes the spare tile the combine points
  other chips' slots at, and ttnn.to_layout untilizes the output into the rows the combine reads;
- indexed: x (TILE) is untilized into DRAM, the op reads it through x_row_index and writes ROW_MAJOR bf16 rows with
  the spare row zeroed (output_row_major, zero_output_row).

The gate is the ratio of the two device times, so it does not depend on the board's absolute speed; each case also
logs both times ("RT-CAL").

Needs a host-IOMMU runner, hence requires_host_iommu: on Blackhole the profiler's D2H socket uses 64-bit PCIe
addressing, which requires IOMMU with no hugepage fallback (realtime_profiler_manager.cpp).
"""

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip"),
]

EMB = HIDDEN = 2880  # gpt-oss hidden size and expert intermediate size
TOKENS = 2048
TOP_K = 4
EXPERTS = 32
LOCAL = 8  # experts on this chip
_ITERS = 3
# Indexed sequence / gathered sequence device time must not exceed this. Measured 2026-10-26 on one Blackhole chip of
# a QuietBox 2 (two runs): gathered 1221 / 1225 us, indexed 1122 / 1125 us, ratio 0.918 / 0.919.
_MAX_RATIO = 0.96


def _routing():
    """(tok_of_row [rows] int64, counts [LOCAL], offsets [LOCAL], stride, spare, rows) for a random top-4 routing: the
    local experts' pairs in token order inside their regions; every other row reads token 0."""
    indices = torch.stack([torch.randperm(EXPERTS)[:TOP_K] for _ in range(TOKENS)])
    tokens_of = [torch.nonzero((indices == e).any(dim=1)).flatten() for e in range(LOCAL)]
    counts = [len(t) for t in tokens_of]
    stride = -(-max(counts) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    offsets = [e * stride for e in range(LOCAL)]
    spare = LOCAL * stride
    rows = spare + ttnn.TILE_SIZE
    tok_of_row = torch.zeros(rows, dtype=torch.int64)
    for e in range(LOCAL):
        tok_of_row[offsets[e] : offsets[e] + counts[e]] = tokens_of[e]
    return tok_of_row, counts, offsets, stride, spare, rows


def _device_ns(mesh_device, run):
    """Total device time of the programs one call of run() dispatches (median-free: the sum over _ITERS calls / _ITERS)."""

    def run_all():
        for _ in range(_ITERS):
            run()

    _, per_program = profile_realtime_program_merged(mesh_device, run_all)
    return sum(entry["duration_ns"] for entry in per_program.values()) / _ITERS


@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_row_index_row_major_output_perf(mesh_device, device_params):
    require_realtime_profiler("unified_routed_expert_moe row-index perf checks")
    torch.manual_seed(11)
    tok_of_row, counts, offsets, stride, spare, rows = _routing()
    weights = [
        {
            "gate_proj": torch.randn(HIDDEN, EMB) * 0.08,
            "up_proj": torch.randn(HIDDEN, EMB) * 0.08,
            "down_proj": torch.randn(EMB, HIDDEN) * 0.05,
        }
        for _ in range(LOCAL)
    ]
    biases = [
        {
            "gate_proj_bias": torch.randn(HIDDEN) * 0.5,
            "up_proj_bias": torch.randn(HIDDEN) * 0.5,
            "down_proj_bias": torch.randn(EMB) * 0.5,
        }
        for _ in range(LOCAL)
    ]

    def dev(t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=layout,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def idx(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint32
        )

    table, tt_counts, tt_offsets = idx(list(range(LOCAL))), idx(counts), idx(offsets)
    expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=LOCAL,
        global_expert_idx_table=table,
        emb_dim=EMB,
        hidden_dim=HIDDEN,
        max_tokens=stride,
        torch_weights=weights,
        torch_biases=biases,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
    )
    args = (tt_offsets, tt_counts, table, expert.gate_projs, expert.up_projs, expert.down_projs)
    common = dict(
        max_dispatched_tokens_per_expert=stride,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
        gate_biases=expert.gate_biases,
        up_biases=expert.up_biases,
        down_biases=expert.down_biases,
    )

    x = dev(torch.randn(TOKENS, EMB), ttnn.bfloat16, ttnn.TILE_LAYOUT)  # the residual stream, as the MoE gets it
    row_index = dev(tok_of_row.to(torch.int32).reshape(1, rows), ttnn.uint32)
    # the spare tile's insert: zeros at region start `spare`, 32 rows, through local expert 0's table slot
    zeros = dev(torch.zeros(ttnn.TILE_SIZE, EMB), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    spare_start = dev(torch.full((1, 16), spare, dtype=torch.int32), ttnn.uint32)
    spare_count = dev(torch.full((1, 16), ttnn.TILE_SIZE, dtype=torch.int32), ttnn.uint32)
    spare_idx = dev(torch.zeros((1, 16), dtype=torch.int32), ttnn.uint32)

    def gathered():
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        dispatched = ttnn.embedding(row_index, x_rm, padding_idx=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_rm)
        dispatched = ttnn.reshape(dispatched, (rows, EMB))
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(dispatched, *args, **common)
        ttnn.deallocate(dispatched)
        out = ttnn.experimental.deepseek_prefill.insert(
            out, zeros, spare_start, spare_count, spare_idx, local_expert_id=0
        )
        rows_out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.deallocate(out)
        return rows_out

    def indexed():
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x_rm, *args, **common, x_row_index=row_index, output_row_major=True, zero_output_row=spare
        )
        ttnn.deallocate(x_rm)
        return out

    # warm up (compile) both, and check they agree on every expert's rows and the spare row
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    a = ttnn.to_torch(gathered(), mesh_composer=composer)[:rows].float()
    b = ttnn.to_torch(indexed(), mesh_composer=composer)[:rows].float()
    for e in range(LOCAL):
        r = slice(offsets[e], offsets[e] + counts[e])
        pcc = torch.corrcoef(torch.stack([a[r].flatten(), b[r].flatten()]))[0, 1].item()
        assert pcc > 0.99, f"expert {e}: indexed vs gathered PCC {pcc}"
    assert torch.equal(b[spare], torch.zeros(EMB)) and torch.equal(a[spare], torch.zeros(EMB))

    gathered_ns = _device_ns(mesh_device, gathered)
    indexed_ns = _device_ns(mesh_device, indexed)
    ratio = indexed_ns / gathered_ns
    logger.info(
        f"RT-CAL gathered {gathered_ns / 1e3:.1f} us, indexed {indexed_ns / 1e3:.1f} us, ratio {ratio:.3f} "
        f"({TOKENS} tokens, local expert counts {counts})"
    )
    assert (
        ratio <= _MAX_RATIO
    ), f"indexed sequence {indexed_ns:.0f} ns is {ratio:.3f} of the gathered {gathered_ns:.0f} ns"
