# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5: regime R3 `block_sharded` (A L1 block-sharded: rows over grid rows, K over grid
columns; each core walks K one A shard per K-block -- its own shard zero-copy, its row peers' shards
over a rotating-sender row multicast; B from DRAM; output in place in the requested L1 block shard).

Pins what the golden suite does not see:
  - both peer-exchange modes (BS_ROW_MCAST True / False) are bit-identical;
  - a ragged last grid row (padding rows in the last A / output shards), a grid column that owns no
    output columns, a ragged last K shard;
  - the M split of the shard (block_m < shard rows) and the N-split streamed-output fallback;
  - the op's default output placement when memory_config is None;
  - a single K shard (no exchange) and a single grid row (ASPP-pool-shaped M = 1).
Accuracy gate: helpers.TOLERANCES / LOW_FIDELITY_TOLERANCES via run_genify_matmul.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

_BF16, _BFP8, _FP32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
_LOFI = ttnn.MathFidelity.LoFi


def _run(a_shape, b_shape, *, epilogue="none", dtype=_BF16, weight_dtype=_BF16, acc=True, act_shard=None, **extras):
    def go(device):
        ex = dict(extras)
        if act_shard is not None:
            ex["act_shard"] = act_shard
        run_genify_matmul(
            (a_shape, b_shape),
            dtype=dtype,
            weight_dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            fp32_dest_acc_en=acc,
            placement="block_sharded",
            epilogue=epilogue,
            device=device,
            extras=ex,
        )

    return go


def _grid(x, y):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))])


CASES = {
    # Cartesian-shaped (auto EVEN block shards).
    "m1_8x1": _run((1, 2048), (2048, 1536)),
    "rows2_8x6_bias_relu_dest16": _run((1, 384, 1536), (1536, 1024), epilogue="bias_relu", acc=False),
    "rows8_8x8_swiglu": _run((1, 1, 2048, 256), (256, 512), epilogue="swiglu", dtype=_BFP8, weight_dtype=_BFP8),
    "rows8_8x8_addcmul_fp32": _run((1, 1, 2048, 256), (256, 512), epilogue="addcmul", dtype=_FP32),
    # PDL-shaped: ragged last grid row (224-row shards: 2048 = 9 x 224 + 32), bf8, LoFi, bias_relu.
    "pdl_res5_ragged_rows": _run(
        (1, 1, 2048, 1024),
        (1024, 512),
        epilogue="bias_relu",
        dtype=_BFP8,
        weight_dtype=_BFP8,
        act_shard=([224, 128], _grid(8, 10)),
        math_fidelity=_LOFI,
        bias_dtype=_BFP8,
    ),
    # ASPP pool: M = 1 on an 8 x 1 grid, bf16 A -> bf8 output.
    "pdl_aspp_pool": _run(
        (1, 1, 1, 2048),
        (2048, 256),
        epilogue="bias_relu",
        weight_dtype=_BFP8,
        act_shard=([32, 256], _grid(8, 1)),
        math_fidelity=_LOFI,
        bias_dtype=_BFP8,
        output_dtype=_BFP8,
    ),
    # Ragged last K shard (K = 288 = 2 x 128 + 32) and 3 grid columns for N = 64: column 2 owns no output.
    "ragged_k_idle_column": _run((1, 1, 64, 288), (288, 64), act_shard=([32, 128], _grid(3, 2))),
    # A single K shard: no exchange, the own shard is the whole K-walk.
    "single_k_shard": _run((1, 1, 128, 256), (256, 256), act_shard=([64, 256], _grid(1, 2))),
}


@pytest.mark.parametrize("name", list(CASES))
def test_block_sharded(name, device):
    CASES[name](device)


def _inputs(device, m, k, n, shard, grid, dtype=_BF16):
    torch.manual_seed(0)
    a = torch.randn(1, 1, m, k)
    b = torch.randn(k, n) / k**0.5
    a_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )
    ta = create_ttnn_input_tensor(a, device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(b, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return a, b, ta, tb


def test_unicast_and_mcast_exchange_bit_identical(device, monkeypatch):
    """The R3 stepping stone (unicast peer reads) and R3a (rotating row multicast) produce the same bits."""
    _, _, ta, tb = _inputs(device, 256, 512, 256, [128, 128], _grid(4, 2))
    outs = []
    for mcast in (False, True):
        monkeypatch.setattr(pd, "BS_ROW_MCAST", mcast)
        outs.append(ttnn.to_torch(genify_matmul(ta, tb)))
    assert torch.equal(outs[0], outs[1])


def test_default_output_placement(device):
    """memory_config=None: the op places the output L1 block-sharded on A's grid, A's shard height."""
    a, b, ta, tb = _inputs(device, 256, 512, 320, [128, 128], _grid(4, 2))
    out = genify_matmul(ta, tb)
    spec = out.memory_config().shard_spec
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    assert spec.grid == ta.memory_config().shard_spec.grid and list(spec.shape) == [128, 96]
    ref = a.float() @ b.float()
    got = ttnn.to_torch(out).float()
    assert torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1] > 0.999


@pytest.mark.parametrize("budget", ["m_split", "n_split"])
def test_block_splits(device, monkeypatch, budget):
    """Tight L1 budgets force the shard-row split (in-place output) and the N-split fallback (streamed
    output through the sharded accessor); both stay correct."""
    # Shards of 4 x 16 tiles on 2 x 2 cores; output shards 4 x 8 tiles. The N split only beats the row
    # split when B's K-block dominates (a wide K shard).
    a, b, ta, tb = _inputs(device, 256, 1024, 512, [128, 512], _grid(2, 2))
    geom = pd._geometry(ta, tb, ta, True)
    topo = pd.BlockShardedTopology(None, 2, 2, 4, 16, 8)
    full = pd._footprint(geom, 4, 8, 16, partials=True, out_in_place=True)
    if budget == "m_split":
        cap, want = pd._footprint(geom, 2, 8, 16, partials=True, out_in_place=True), (2, 8, True)
    else:
        cap, want = pd._footprint(geom, 4, 2, 16, partials=True), (4, 2, False)
    assert cap < full
    blk, in_place = pd.choose_block_sharded_blocking(geom, topo, cap)
    assert (blk.block_m_tiles, blk.block_n_tiles, in_place) == want
    monkeypatch.setattr(pd, "l1_cb_budget", lambda *_: cap)
    got = ttnn.to_torch(genify_matmul(ta, tb)).float()
    ref = a.float() @ b.float()
    assert torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1] > 0.9999
