# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 10: R2a / R3a weight multicast (SHARDED_B_MCAST). One injector per B group (R3: a grid column,
R2: all of A's shard cores) reads B once and multicasts it; the first block's bias / gate row rides one row
round on the same channel.

The multicast changes only where B's bytes come from, never the reduction order, so every case must be
BIT-IDENTICAL to the per-core-read stepping stone (SHARDED_B_MCAST False). Pins:
  - HS on PDL's 86-core non-rectangular set (two rectangles), bias_relu (row round);
  - HS addcmul (gate row round + per-core residual) and swiglu (no row);
  - R3 with B rounds of several K-blocks (SHARDED_B_ROUND_BYTES), a ragged last K shard and grid columns
    that own no output columns;
  - multi-block walks (L1-forced M split in place, N split streamed): B per block, rows per core after the
    first block;
  - an HS core set of four scattered cores (> 3 rectangles): the multicast cannot cover it, the op falls back;
  - R3's traffic gate (BS_B_MCAST_MIN_SAVED_BYTES; the other R3 cases run with it disabled).
"""

import math

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

_BF16, _BFP8 = ttnn.bfloat16, ttnn.bfloat8_b


def _rect(x, y):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))])


def _sharded(t, device, dtype, layout, grid, shard):
    mc = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))
    return create_ttnn_input_tensor(t, device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mc)


def _run_both(
    device, monkeypatch, m, k, n, grid, shard, layout, epilogue="none", dtype=_BFP8, budget=None, r3_gate=False
):
    torch.manual_seed(0)
    A = torch.randn(1, 1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = _sharded(A, device, dtype, layout, grid, shard)
    tb = create_ttnn_input_tensor(B, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    kwargs = {}
    n_out = n // 2 if epilogue == "swiglu" else n
    if epilogue in ("bias", "bias_relu"):
        kwargs["bias"] = create_ttnn_input_tensor(torch.randn(1, n), device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        if epilogue == "bias_relu":
            kwargs["activation"] = "relu"
    elif epilogue == "swiglu":
        kwargs["activation"] = "swiglu"
    elif epilogue == "addcmul":
        kwargs["gate"] = create_ttnn_input_tensor(torch.randn(1, n), device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        kwargs["residual"] = create_ttnn_input_tensor(
            torch.randn(1, 1, m, n), device, dtype=dtype, layout=ttnn.TILE_LAYOUT
        )
    if budget is not None:
        monkeypatch.setattr(pd, "l1_cb_budget", lambda *_: budget)
    if not r3_gate:  # exercise the R3 multicast on shapes too small to pass its traffic gate
        monkeypatch.setattr(pd, "BS_B_MCAST_MIN_SAVED_BYTES", 0)
    # Spies: what the descriptor decided on the multicast run (channel built? which blocking?).
    seen = {}
    real_channel, real_plan = pd._sharded_b_channel, pd.plan_block_sharded

    def spy_channel(*args, **kw):
        # Refinement 18: R2 builds one channel per B group (rectangle); "mcast" = some group multicasts.
        channel = real_channel(*args, **kw)
        seen["mcast"] = seen.get("mcast", False) or channel is not None
        seen["channels"] = seen.get("channels", 0) + int(channel is not None)
        return channel

    def spy_plan(*args, **kw):
        plan = real_plan(*args, **kw)
        seen["blk"], seen["in_place"] = plan[3], plan[4]
        return plan

    monkeypatch.setattr(pd, "_sharded_b_channel", spy_channel)
    monkeypatch.setattr(pd, "plan_block_sharded", spy_plan)
    outs = []
    for on in (False, True):
        monkeypatch.setattr(pd, "SHARDED_B_MCAST", on)
        outs.append(ttnn.to_torch(genify_matmul(ta, tb, **kwargs)))
    assert list(outs[1].shape)[-1] == n_out
    assert torch.equal(outs[0], outs[1]), f"max diff {(outs[0].float() - outs[1].float()).abs().max()}"
    seen["out"] = outs[1]
    return seen


_HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def test_hs_pdl_86_cores_bias_relu(device, monkeypatch):
    grid = ttnn.num_cores_to_corerangeset(86, device.compute_with_storage_grid_size(), True)
    assert _run_both(device, monkeypatch, 8192, 512, 128, grid, [96, 512], _HS, "bias_relu")["mcast"]


@pytest.mark.parametrize("epilogue", ["addcmul", "swiglu"])
def test_hs_epilogues(device, monkeypatch, epilogue):
    grid = ttnn.num_cores_to_corerangeset(12, device.compute_with_storage_grid_size(), True)
    assert _run_both(device, monkeypatch, 12 * 64, 256, 256, grid, [64, 256], _HS, epilogue, dtype=_BF16)["mcast"]


def test_bs_multi_kblock_rounds(device, monkeypatch):
    """PDL res5 reduce: 8 K-blocks of 4 x 2 tiles; B rounds carry several K-blocks."""
    seen = _run_both(device, monkeypatch, 2048, 1024, 512, _rect(8, 10), [224, 128], _BS, "bias_relu")
    assert seen["mcast"] and seen["blk"].num_k_blocks == 8


def test_bs_traffic_gate(device, monkeypatch):
    """R3 keeps the per-core B reads when the multicast would save less than BS_B_MCAST_MIN_SAVED_BYTES (PDL
    res4 reduce: 0.97 MB, measured slower with the multicast) and multicasts above it (res5 reduce, 5 MB)."""
    assert not _run_both(device, monkeypatch, 8192, 512, 256, _rect(8, 8), [1024, 64], _BS, "bias_relu", r3_gate=True)[
        "mcast"
    ]
    assert _run_both(device, monkeypatch, 2048, 1024, 512, _rect(8, 10), [224, 128], _BS, "bias_relu", r3_gate=True)[
        "mcast"
    ]


def test_bs_ragged_k_and_empty_columns(device, monkeypatch):
    """K = 10 tiles in shards of 3 (ragged last K-block, 4 grid columns), N = 2 tiles over 4 columns of 1:
    columns 2 and 3 own no output columns (their injectors read nothing)."""
    monkeypatch.setattr(pd, "SHARDED_B_ROUND_BYTES", 1)  # one K-block per round
    _run_both(device, monkeypatch, 256, 320, 64, _rect(4, 2), [128, 96], _BS, "bias")
    monkeypatch.setattr(pd, "SHARDED_B_ROUND_BYTES", 1 << 30)  # all K-blocks in one round
    _run_both(device, monkeypatch, 256, 320, 64, _rect(4, 2), [128, 96], _BS, "bias")


@pytest.mark.parametrize("case", ["hs_n_split", "bs_m_split", "bs_n_split"])
def test_multi_block_walk(device, monkeypatch, case):
    """L1-forced multi-block walks (shards of 4 x 16 tiles; R2 on 4 cores, R3 on 2 x 2): every block gets its
    B round(s); the first block's row comes by the row round, later rows per core. (An R2 M split is never
    chosen: splitting a single-K-block walk double-buffers B.)"""
    placement, split = case.split("_", 1)
    m, k, n = 4 * 128, 512, 256
    if placement == "hs":
        grid, shard, layout = (
            ttnn.num_cores_to_corerangeset(4, device.compute_with_storage_grid_size(), True),
            [128, k],
            _HS,
        )
    else:
        m, k, n = 256, 1024, 512
        grid, shard, layout = _rect(2, 2), [128, 512], _BS
    a_probe = _sharded(torch.zeros(1, 1, m, k), device, _BF16, layout, grid, shard)
    b_probe = create_ttnn_input_tensor(torch.zeros(k, n), device, dtype=_BF16, layout=ttnn.TILE_LAYOUT)
    geom = pd._geometry(a_probe, b_probe, a_probe, True, "bias", row=b_probe)
    if placement == "hs":
        cores = tuple(ttnn.corerange_to_cores(grid, None, True))
        topo = pd.BlockShardedTopology(None, 4, 1, 4, 16, 8, cores)
    else:
        topo = pd.BlockShardedTopology(None, 2, 2, 4, 16, 8)

    def wanted(cap):
        try:
            blk, in_place = pd.choose_block_sharded_blocking(geom, topo, cap)
        except RuntimeError:
            return False
        whole_m, whole_n = blk.block_m_tiles == 4, blk.block_n_tiles == topo.out_w_tiles
        return (not whole_m and whole_n and in_place) if split == "m_split" else (whole_m and not whole_n)

    # The largest budget (1 KB steps) that yields the wanted split.
    cap = next(c for c in range(1 << 20, 0, -1024) if wanted(c))
    seen = _run_both(device, monkeypatch, m, k, n, grid, shard, layout, "bias", dtype=_BF16, budget=cap)
    assert seen["mcast"] and wanted(cap)


def test_hs_scattered_cores_fall_back(device, monkeypatch):
    """Four scattered shard cores are four rectangles: the multicast cannot cover them (> 3), so the op keeps
    the per-core B reads (and still matches)."""
    cores = [ttnn.CoreCoord(x, 0) for x in (0, 2, 4, 6)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    assert not _run_both(device, monkeypatch, 4 * 64, 128, 128, grid, [64, 128], _HS, "bias", dtype=_BF16)["mcast"]


@pytest.mark.parametrize(
    "case",
    [
        # (cores, m, k, n, epilogue): PDL heads (1-tile B, 103 cores = 9 x 11 + 4), res3 expand (86 = 7 x 11 + 9,
        # bias row tail of 16 tiles), addcmul (gate row tail + per-core residual), swiglu (no row: no tail).
        (103, 103 * 64, 32, 32, "bias"),
        (86, 86 * 96, 128, 512, "bias_relu"),
        (24, 24 * 64, 256, 256, "addcmul"),
        (24, 24 * 64, 256, 256, "swiglu"),
    ],
    ids=["heads", "res3_expand", "addcmul", "swiglu"],
)
def test_hs_one_shot_rect_groups(device, monkeypatch, case):
    """Refinement 18: a single-round R2 walk runs one B group per rectangle of the core set (own injector each)
    on the handshake-free one-shot channel, the first row riding in cb_weight_tiles' tail. Bit-identical to the
    per-core reads; the plan is asserted through spies."""
    cores, m, k, n, epilogue = case
    grid = ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), True)
    width = device.compute_with_storage_grid_size().x
    rects = (1 if cores >= width else 0) + int(cores % width != 0)
    real_one_shot = pd.sharded_b_one_shot
    shots = []
    monkeypatch.setattr(pd, "sharded_b_one_shot", lambda *a: shots.append(real_one_shot(*a)) or shots[-1])
    seen = _run_both(device, monkeypatch, m, k, n, grid, [m // cores, k], _HS, epilogue, dtype=_BF16)
    assert seen["channels"] == rects
    one_shot, tail = shots[-1]
    assert one_shot and (tail > 0) == (epilogue != "swiglu")


@pytest.mark.parametrize(
    "case",
    [
        # (m, k, n, grid x, grid y, shard, epilogue, channel): PDL res5 reduce (multicast B, rounds of 2 K-blocks),
        # ASPP projection (per-core B, 87 KB A rounds), addcmul on the res5 grid (per-core residual on NCRISC).
        (2048, 1024, 512, 8, 10, [224, 128], "bias_relu", True),
        (2048, 1280, 256, 8, 4, [512, 160], "bias_relu", False),
        (2048, 1024, 512, 8, 10, [224, 128], "addcmul", True),
    ],
    ids=["res5_reduce", "aspp_proj", "addcmul"],
)
def test_bs_split_roles(device, monkeypatch, case):
    """Refinement 18 (BS_B_ON_WRITER): an in-place R3 walk runs its B half (B rounds / reads, the bias / gate row)
    on BRISC as a second reader instance (GM_BS_ROLE 2) and its A half on NCRISC (GM_BS_ROLE 1), with no writer.
    Bit-identical to the single-reader walk."""
    m, k, n, gx, gy, shard, epilogue, channel = case
    real = pd.create_block_sharded_program_descriptor
    roles = []

    def spy(*a, **kw):
        d = real(*a, **kw)
        roles.append({v for kern in d.kernels for name, v in kern.defines if name == "GM_BS_ROLE"})
        return d

    monkeypatch.setattr(pd, "create_block_sharded_program_descriptor", spy)
    outs = []
    for split in (False, True):
        monkeypatch.setattr(pd, "BS_B_ON_WRITER", split)
        seen = _run_both(device, monkeypatch, m, k, n, _rect(gx, gy), shard, _BS, epilogue, r3_gate=True)
        assert seen["mcast"] == channel
        outs.append(seen["out"])
    assert {"1", "2"} in roles  # the split descriptor carries both halves
    assert torch.equal(outs[0], outs[1]), f"max diff {(outs[0].float() - outs[1].float()).abs().max()}"
