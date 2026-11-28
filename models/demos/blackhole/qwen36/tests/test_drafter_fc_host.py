# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the drafter fc's explicit 1D program (tt/mtp.py drafter_fc_progcfg).

* At the TP=8 Qwen3.8 shape ([rows, 2*dim] x [2*dim, dim/tp], dim 5120) the program is a 1D mcast_in0 matmul on a
  grid of 20 working cores with 32-tile K blocks, one tile of rows and one output tile per core.
* The other decode programs built by create_matmul_1d_decode_progcfg keep their K blocks (cap 8).
* Qwen36MTP._fuse passes the program for decode calls of up to one tile of rows only; more rows (a reseed past
  B*T = 32), prefill and the single-device drafter keep the auto-configured matmul.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP, drafter_fc_progcfg
from models.tt_transformers.tt.common import Mode

DIM = 5120


@pytest.fixture
def progcfg_fields(monkeypatch):
    monkeypatch.setattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", lambda **kw: SimpleNamespace(**kw))


@pytest.mark.parametrize("grid_w, grid", [(11, (11, 2)), (8, (8, 3))], ids=["blackhole", "wormhole"])
def test_the_drafter_fc_program_at_the_tp8_shape(progcfg_fields, grid_w, grid):
    pc = drafter_fc_progcfg(DIM, 8, grid_w)
    k_tiles, n_tiles = 2 * DIM // 32, DIM // 8 // 32
    assert pc.compute_with_storage_grid_size == grid
    assert pc.in0_block_w == 32 and k_tiles % pc.in0_block_w == 0
    assert (pc.per_core_M, pc.per_core_N) == (1, 1)
    assert (pc.out_subblock_h, pc.out_subblock_w) == (1, 1)
    assert pc.mcast_in0 is True and pc.fuse_batch is True and pc.fused_activation is None
    # every one of the 20 output tiles has a core; the grid's other cores get no work
    assert n_tiles == 20 and grid[0] * grid[1] * pc.per_core_N >= n_tiles


def test_the_other_decode_programs_keep_their_k_blocks(progcfg_fields):
    pc = tpc.create_matmul_1d_decode_progcfg(32, DIM, 17408 // 8, num_cores=44, grid_w=11)
    assert pc.in0_block_w == 8  # 160 K tiles, cap 8
    capped = tpc.create_matmul_1d_decode_progcfg(32, DIM, 17408 // 8, num_cores=44, grid_w=11, max_in0_block_w=32)
    assert capped.in0_block_w == 32


def _bare_mtp(monkeypatch, rows, progcfg="fc-1d", num_devices=8):
    """A Qwen36MTP holding only what _fuse reads; ttnn.linear records its keyword arguments."""
    calls = []
    monkeypatch.setattr(ttnn, "concat", lambda ts, dim, **kw: SimpleNamespace(shape=(1, 1, rows, 2 * DIM)))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "TILE_SIZE", 32)
    monkeypatch.setattr(ttnn, "linear", lambda x, w, **kw: calls.append(kw) or "fused")
    mtp = Qwen36MTP.__new__(Qwen36MTP)
    mtp.num_devices = num_devices
    mtp.args = SimpleNamespace(get_norm_config=lambda key, mode: {})
    mtp.pre_fc_norm_embedding = mtp.pre_fc_norm_hidden = lambda x, mode, norm_config: "normed"
    mtp.fc = "fc"
    mtp._fc_compute_cfg = "hifi2" if num_devices > 1 else None
    mtp._fc_1d_progcfg = progcfg
    return mtp, calls


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_the_fc_takes_the_1d_program_for_one_tile_of_rows(monkeypatch, rows):
    mtp, calls = _bare_mtp(monkeypatch, rows)
    assert mtp._fuse("emb", "hidden", Mode.DECODE) == "fused"
    assert calls == [dict(memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config="hifi2", program_config="fc-1d")]


@pytest.mark.parametrize(
    "rows, mode, num_devices",
    [(33, Mode.DECODE, 8), (64, Mode.DECODE, 8), (32, Mode.PREFILL, 8), (2048, Mode.PREFILL, 8), (8, Mode.DECODE, 1)],
    ids=["reseed-33-rows", "reseed-64-rows", "prefill-32-rows", "prefill", "single-device"],
)
def test_other_fc_calls_keep_the_auto_program(monkeypatch, rows, mode, num_devices):
    mtp, calls = _bare_mtp(monkeypatch, rows, progcfg="fc-1d" if num_devices > 1 else None, num_devices=num_devices)
    mtp._fuse("emb", "hidden", mode)
    assert len(calls) == 1 and "program_config" not in calls[0]
    assert calls[0]["memory_config"] == ttnn.DRAM_MEMORY_CONFIG
