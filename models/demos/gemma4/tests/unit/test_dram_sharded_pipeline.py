# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The DRAM-sharded decode kernel's multi-core pipeline rows (tt/decode_matmul.py ``DRAM_SHARDED_PIPELINE_ROWS``).

Host: the row resolver keeps today's single-reader kernel on Wormhole, on any grid of <= 64 worker cores, under the
off switch and for models without rows. Device (one Blackhole chip, skipped elsewhere): at the shipped per-chip
shapes the pipelined ``DramShardedLinear`` is at least as close to the fp32 golden as today's kernel, on both entry
points (interleaved activation with the kernel's reshard; activation already in ``act_memory_config``), and the two
entry points agree. ``GEMMA4_PIPELINE_TEST_ALL=1`` runs every row of both models (the devq validation); the CI
default is the two 31B tp4 shapes (~30 s warm).
"""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.decode_matmul import (
    DRAM_SHARDED_PIPELINE_ROWS,
    MIN_PIPELINE_GRID_CORES,
    PIPELINE_ENV,
    dram_sharded_pipeline_row,
    model_tag,
)
from models.demos.gemma4.tt.dram_sharded import TILE_SIZE, DramShardedLinear, can_dram_shard

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
HIDDEN = {"31B": 5376, "12B": 3840}
# (model, op, tp) -> (K, N) of the lead shape (the sliding layers for qkv / out_proj)
SHAPES = {
    ("31B", "qkv", 4): (5376, 4096),
    ("31B", "qkv", 8): (5376, 2048),
    ("31B", "out_proj", 4): (2048, 5376),
    ("31B", "out_proj", 8): (1024, 5376),
    ("31B", "gate_up", 4): (5376, 10752),
    ("31B", "gate_up", 8): (5376, 5376),
    ("31B", "down", 4): (5376, 5376),
    ("31B", "down", 8): (2688, 5376),
    ("12B", "qkv", 4): (3840, 2048),
    ("12B", "qkv", 8): (3840, 1024),
    ("12B", "out_proj", 4): (1024, 3840),
    ("12B", "out_proj", 8): (512, 3840),
    ("12B", "gate_up", 4): (3840, 7680),
    ("12B", "gate_up", 8): (3840, 3840),
    ("12B", "down", 4): (3840, 3840),
    ("12B", "down", 8): (1920, 3840),
}
CI_CASES = [("31B", "out_proj", 4), ("31B", "gate_up", 4)]


def _pcc(a, b):
    a = a.reshape(-1).double() - a.reshape(-1).double().mean()
    b = b.reshape(-1).double() - b.reshape(-1).double().mean()
    return float(a @ b / (a.norm() * b.norm()))


def test_pipeline_rows_host(monkeypatch):
    monkeypatch.delenv(PIPELINE_ENV, raising=False)
    assert model_tag(5376) == "31B" and model_tag(3840) == "12B" and model_tag(2816) is None
    row = DRAM_SHARDED_PIPELINE_ROWS[("31B", "out_proj", 4)]
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (13, 10), True, default="1") == row
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (12, 10), True, default="1") == row
    # Wormhole, the 8x8 grid and the off switch keep today's kernel; so does a model without rows
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (8, 8), False, default="1") is None
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (8, 8), True, default="1") is None
    assert dram_sharded_pipeline_row(2816, "out_proj", 4, (13, 10), True, default="1") is None
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (13, 10), True, default="0") is None
    monkeypatch.setenv(PIPELINE_ENV, "0")
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (13, 10), True, default="1") is None
    monkeypatch.setenv(PIPELINE_ENV, "1")
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (13, 10), True, default="0") == row
    # every row fits its grid rule: workers x 8 banks within the smallest pipelined grid
    for (model, op, tp), (workers, bw) in DRAM_SHARDED_PIPELINE_ROWS.items():
        assert workers * 8 <= MIN_PIPELINE_GRID_CORES and bw >= 1 and (model, op, tp) in SHAPES


def _cases():
    keys = list(SHAPES) if os.environ.get("GEMMA4_PIPELINE_TEST_ALL") == "1" else CI_CASES
    return [pytest.param(k, id=f"{k[0]}-{k[1]}-tp{k[2]}") for k in keys]


@pytest.mark.parametrize("case", _cases())
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_pipeline_matches_single_reader(mesh_device, device_params, case, monkeypatch):
    model, op, tp = case
    grid = mesh_device.compute_with_storage_grid_size()
    blackhole = "blackhole" in ttnn.get_arch_name()
    if not blackhole or grid.x * grid.y < MIN_PIPELINE_GRID_CORES:
        pytest.skip("the pipeline rows apply to Blackhole grids of more than 64 worker cores")
    row = dram_sharded_pipeline_row(HIDDEN[model], op, tp, (grid.x, grid.y), blackhole, default="1")
    if row is None:
        pytest.skip(f"no pipeline row for {case}")
    k, n = SHAPES[case]
    if not can_dram_shard(k, n, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{k}, {n}] has no DRAM-sharded decode config on this grid")
    torch.manual_seed(7)
    w = (torch.randn(1, 1, k, n) * 0.02).to(torch.bfloat16)
    x = (torch.randn(1, 1, TILE_SIZE, k) * 2).to(torch.bfloat16)
    w_q = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
    golden = x.float().reshape(TILE_SIZE, k) @ w_q.reshape(k, n)
    monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1")
    base = DramShardedLinear(w, mesh_device, None, k=k, n=n, dtype=ttnn.bfloat8_b, cache_file_name=None)
    pipe = DramShardedLinear(
        w, mesh_device, None, k=k, n=n, dtype=ttnn.bfloat8_b, cache_file_name=None, pipeline_row=row
    )
    assert pipe.pipeline_row == row and base.pipeline_row is None
    assert pipe._decode_pc.use_multicore_pipeline and pipe._decode_pc.num_workers_per_dram_bank == row[0]
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    out_base = ttnn.to_torch(base(tx)).float().reshape(TILE_SIZE, n)
    out_pipe = ttnn.to_torch(pipe(tx)).float().reshape(TILE_SIZE, n)
    x_sh = ttnn.to_memory_config(tx, pipe.act_memory_config)
    out_sh = ttnn.to_torch(pipe._decode_sharded_rows(x_sh, ttnn.DRAM_MEMORY_CONFIG, None, TILE_SIZE)).float()
    out_sh = out_sh.reshape(TILE_SIZE, n)
    assert torch.isfinite(out_pipe).all() and torch.isfinite(out_sh).all()
    pcc_base, pcc_pipe, pcc_sh = _pcc(out_base, golden), _pcc(out_pipe, golden), _pcc(out_sh, golden)
    print(
        f"RESULT pipe_unit case={model}-{op}-tp{tp} K={k} N={n} row={row} pcc_base={pcc_base:.6f} pcc_pipe={pcc_pipe:.6f} pcc_sharded_entry={pcc_sh:.6f}"
    )
    assert pcc_pipe >= pcc_base - 1e-6, (pcc_pipe, pcc_base)
    assert pcc_sh >= pcc_base - 1e-6, (pcc_sh, pcc_base)
    assert _pcc(out_sh, out_pipe) > 0.99999
    for lin in (base, pipe):
        lin.weight.deallocate(True)
