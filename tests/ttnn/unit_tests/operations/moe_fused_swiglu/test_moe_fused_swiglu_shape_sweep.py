# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shape coverage for the C++ op: three (emb, hidden) pairs x both activation formats.

Every cell here MUST run — an L1 refusal is a FAILURE, not a skip. That is the whole point of the
file: these three shapes all fit at the shipped ``M_BLOCK`` 8 on an 11x8 grid with bfp4 weights, and
this test is what keeps that true.

    scripts/run_safe_pytest.sh --run-all \
        tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_shape_sweep.py

SHAPE VOCABULARY. The op calls the activation width ``emb`` and the MoE intermediate ``hidden``;
``w_gate``/``w_up`` are ``(emb, hidden)`` and ``w_down`` is ``(hidden, emb)``. In matmul terms the
down projection is ``K=hidden, N=emb``, which is how these three are usually named:

    K=2k  N=6k    -> emb 6144, hidden 2048
    K=2k  N=7k    -> emb 7168, hidden 2048
    K=3k  N=3.5k  -> emb 3584, hidden 3072

MEASURED L1 per core against the 1 461 376 B budget (``get_max_worker_l1_unreserved_size()`` 1 532 032
minus ``L1_CB_RESERVE`` 70 656), bfp4 weights, 11x8:

    emb 6144 hidden 2048   1 283 648 bf16_rm / 1 235 584 bfp8_tile
    emb 7168 hidden 2048   1 389 120 bf16_rm / 1 332 864 bfp8_tile
    emb 3584 hidden 3072   1 444 288 bf16_rm / 1 416 704 bfp8_tile   <- tightest

Those figures come from ``Blocking::l1_bytes()``, which models both DRAM scratch CBs as one 64 B
page; the real allocation adds (counts_page - 64), typically ~960 B more. The 3584/3072 bf16 cell
therefore has roughly 16 KB of headroom, the least in the supported set, and is the first thing any
future CB growth breaks. If this file starts failing there, the CB that grew is the cause.

TIMING uses the REAL-TIME PROFILER (``tests/ttnn/profiling/realtime_profiler_utils``), not tracy: one
dispatch produces one device program record, with no ``--profile`` flag, no manifest, and no separate
parse step. Where it is unavailable (it needs an IOMMU-pinned sku) the correctness half still runs and
the duration column reports ``n/a`` — a missing profiler must not cost shape coverage.

CORRECTNESS is cheap by construction: ``x`` is a 256-row block REPEATED to capacity. Tokens are
independent, so output row i is determined by input row ``i % 256`` — one 256-row fp32 golden is
therefore valid at EVERY count, tails included, and PCC is asserted for all of them. The additional
bitwise span comparison catches a stale weight slot or a botched M-block transition, which PCC over
a mostly-correct output would absorb.
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import nd_shard_n_tiles, weight_memory_configs

TILE = 32
GRID = (11, 8)
CAPACITY = 5120

#: x repeats with this period, so the golden is 256 rows wide and every later span must match it.
BLOCK_ROWS = 256

#: The standard M sweep — real tokens routed to the local expert. 32 is one tile-row (the shortest
#: program the op can run) and 5120 == capacity is the `full` fill bucket.
COUNTS = [32, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 5120]

#: bfp4 weights lose ~4 mantissa bits, which is what sets this gate; it matches the C++ op tests.
PCC_GATE = 0.975

NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137

#: (label, emb, hidden) — see the module docstring for the K/N naming.
SHAPES = [
    ("K2k_N6k", 6144, 2048),
    ("K2k_N7k", 7168, 2048),
    ("K3k_N3p5k", 3584, 3072),
]

FORMATS = {
    "bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
}


def _aux_tensor(values, device):
    return ttnn.from_torch(
        values,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _counts_for(count, device):
    values = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    values[GLOBAL_EXPERT_ID] = count
    return _aux_tensor(values, device)


def _idx_table(device):
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    return _aux_tensor(idx, device)


def _golden(host_block, host_weights, rows):
    """silu(x @ Wg) * (x @ Wu) @ Wd over the first `rows` rows, in fp32."""
    x = host_block[0, 0, :rows].float()
    w_gate, w_up, w_down = (w.float() for w in host_weights)
    return (torch.nn.functional.silu(x @ w_gate) * (x @ w_up)) @ w_down


@pytest.mark.parametrize("input_format", list(FORMATS), ids=list(FORMATS))
@pytest.mark.parametrize("label, emb, hidden", SHAPES, ids=[s[0] for s in SHAPES])
def test_shape_sweep(device, label, emb, hidden, input_format):
    grid = device.compute_with_storage_grid_size()
    if GRID[0] > int(grid.x) or GRID[1] > int(grid.y):
        pytest.skip(f"device grid {grid.x}x{grid.y} is smaller than {GRID[0]}x{GRID[1]}")

    dtype, layout = FORMATS[input_format]
    torch.manual_seed(1234)
    host_block = torch.randn((1, 1, BLOCK_ROWS, emb), dtype=torch.bfloat16)
    host_x = host_block.repeat(1, 1, CAPACITY // BLOCK_ROWS, 1)
    host_weights = [torch.randn(shape, dtype=torch.bfloat16) for shape in ((emb, hidden), (emb, hidden), (hidden, emb))]

    gate_up_config, down_config = weight_memory_configs(device, emb, hidden, core_grid=GRID)
    tt_x = ttnn.from_torch(host_x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_weights = [
        ttnn.from_torch(weight, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)
        for weight, config in zip(host_weights, (gate_up_config, gate_up_config, down_config))
    ]
    # The coalesced reader path is only taken when it can prove a contiguous run; an interleaved
    # weight is silently correct but slower, so assert the reader's own predicate rather than trust
    # that the shard applied.
    shard_widths = [nd_shard_n_tiles(w) for w in tt_weights]
    assert all(w > 0 for w in shard_widths), f"{label}: weights are interleaved, expected ND-sharded: {shard_widths}"

    tt_idx = _idx_table(device)
    tt_counts = {count: _counts_for(count, device) for count in COUNTS}
    profiled = ttnn.device.IsProgramRealtimeProfilerActive()

    def dispatch(count):
        return moe_fused_swiglu(tt_x, *tt_weights, tt_counts[count], tt_idx, LOCAL_EXPERT_ID, core_grid=GRID)

    # ONE fp32 golden for the 256-row block. Every token is independent, so row i of the output is
    # determined by row i % BLOCK_ROWS of the input — which makes this golden valid at EVERY count,
    # tails included, for the price of a single 256-row host matmul.
    golden_block = _golden(host_block, host_weights, BLOCK_ROWS)

    try:
        # The FIRST dispatch is where the op decides whether this cell fits L1. A refusal here is the
        # failure this file exists to catch, so let it surface with the op's own byte counts.
        ttnn.deallocate(dispatch(COUNTS[0]))
        # Drain it before the profiler callback is armed, so its record cannot be collected as the
        # first measured one.
        ttnn.synchronize_device(device)

        durations = {}
        for count in COUNTS:
            if profiled:
                out, record = profile_realtime_program(device, lambda: dispatch(count))
                durations[count] = record["duration_ns"] / 1e3
            else:
                out, durations[count] = dispatch(count), None

            host_out = ttnn.to_torch(out)[0, 0, :count].clone()
            ttnn.deallocate(out)

            assert torch.isfinite(host_out).all(), f"{label}/{input_format} M={count}: non-finite output"
            # PCC over the WHOLE count, every count — tails and partial repeats included.
            expected = golden_block.repeat((count + BLOCK_ROWS - 1) // BLOCK_ROWS, 1)[:count]
            assert_with_pcc(expected, host_out.float(), PCC_GATE)
            # Bitwise, not just PCC: identical input rows must give identical output bytes, which
            # catches a stale weight slot or a botched M-block transition that PCC would absorb.
            #
            # COMPLETE spans only. BLOCK_ROWS is exactly one full M-block, and only blocks with the
            # same `m_eff` share a reduce-scatter slice plan and matmul sub-block height. A ragged
            # tail runs a smaller m_eff (M=384 -> m_eff 8 then 4), so its float accumulation ORDER
            # differs and bit-identity is not promised there — PCC above is what covers the tail.
            for start in range(BLOCK_ROWS, count - BLOCK_ROWS + 1, BLOCK_ROWS):
                assert torch.equal(host_out[:BLOCK_ROWS], host_out[start : start + BLOCK_ROWS]), (
                    f"{label}/{input_format} M={count}: rows [{start}, {start + BLOCK_ROWS}) differ "
                    f"from rows [0, {BLOCK_ROWS}) for byte-identical input"
                )

        header = f"{label}/{input_format} emb={emb} hidden={hidden} grid={GRID[0]}x{GRID[1]} shards={shard_widths}"
        logger.info(header)
        for count in COUNTS:
            us = durations[count]
            logger.info(f"  M={count:>5}  {'n/a (profiler inactive)' if us is None else f'{us:9.2f} us'}")
        if not profiled:
            logger.warning("real-time profiler inactive on this sku; correctness ran, durations not reported")
    finally:
        for tensor in (tt_x, tt_idx, *tt_weights, *tt_counts.values()):
            ttnn.deallocate(tensor)
