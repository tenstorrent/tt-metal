# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the RES40 out-projection of the TP=8 MTP verify and the TP=8 decode in-projection grids.

* DecodeResidualAllReduce.out_matmul (tt/decode_allreduce.py): a row-parallel out-projection of at most one tile of
  rows runs the 1D decode matmul on the 10x4 RES40 grid with 4 output tiles per core and writes the RES40 layout. Its
  K loop (in0_block_w, per_core_M, mcast_in0, no fused activation) is the one of the TP=8 attn_wo and gdn_out programs
  it replaces; only the N split over the cores differs. It refuses a sharded input, more than one tile of rows and a
  weight that does not end in dim.
* DecodeResidualAllReduce.all_reduce takes a RES40 partial as it is: no reshard and no early free.
* Qwen36ModelArgs at TP=8: the attention qkv decode program runs on 60 cores of a device-wide grid, the GDN qkvz on 84;
  K blocks unchanged. TP=4 keeps 64 cores 8 wide and 44 cores.
The attention and GDN call sites are in test_verify_attention_glue_host.py and test_decode_allreduce_host.py.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import decode_allreduce as dar_mod
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

DIM = 5120

# Qwen3.8-27B: the attributes Qwen36ModelArgs._init_tp_config reads.
_QWEN38 = dict(
    n_heads=24,
    n_kv_heads=4,
    head_dim=256,
    dim=DIM,
    hidden_dim=17408,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_num_value_heads=48,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    linear_q_dim=2048,
    linear_k_dim=2048,
    linear_v_dim=6144,
    max_batch_size=8,
    moe_num_experts=0,
)


class FT:
    def __init__(self, shape, name, mem="L1", sharded=False):
        self.shape, self.name, self._mem, self._sharded = tuple(shape), name, mem, sharded
        self.dtype = ttnn.bfloat16

    def memory_config(self):
        return self._mem

    def is_sharded(self):
        return self._sharded

    def __repr__(self):
        return f"FT({self.name})"


@pytest.fixture
def progcfg_fields(monkeypatch):
    monkeypatch.setattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", lambda **kw: SimpleNamespace(**kw))


def _tp_args(tp, grid=(11, 10)):
    args = SimpleNamespace(num_devices=tp, **_QWEN38)
    mesh = SimpleNamespace(shape=(1, tp), compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]))
    Qwen36ModelArgs._init_tp_config(args, mesh)
    return args


def _dar():
    d = object.__new__(dar_mod.DecodeResidualAllReduce)
    d.dim, d.res_mem, d._out_progcfgs = DIM, "RES40", {}
    return d


@pytest.fixture
def linear(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: FT(shape, f"view({t.name})", t._mem))
    monkeypatch.setattr(ttnn, "linear", lambda x, w, **kw: calls.append((x, w, kw)) or FT(x.shape, "res40", "RES40"))
    return calls


@pytest.mark.parametrize("site, k_attr", [("attn_wo", "attn_out_dim_tp"), ("gdn_out", "gdn_value_dim_tp")])
def test_the_res40_out_projection_repeats_the_tp8_k_loop(progcfg_fields, linear, site, k_attr):
    args = _tp_args(8)
    old = getattr(args, f"{site}_decode_1d_progcfg")
    k = getattr(args, k_attr)
    x = FT((1, 32, k), "gated")
    out = _dar().with_residual(FT((1, 1, 32, DIM), "x", "RES40")).out_matmul(x, FT((k, DIM), "w"), "cfg")
    ((xin, w, kw),) = linear
    pc = kw["program_config"]
    assert out.name == "res40" and xin.shape == (1, 1, 32, k) and w.name == "w"
    assert kw["memory_config"] == "RES40" and kw["compute_kernel_config"] == "cfg"
    # RES40: 40 cores, 10 wide, 4 output tiles each = the [32, dim / 40] width shards
    assert pc.compute_with_storage_grid_size == dar_mod.RES40_GRID and pc.per_core_N * 32 * 40 == DIM
    # the K loop of the program it replaces
    for field in ("in0_block_w", "per_core_M", "mcast_in0", "fuse_batch", "fused_activation"):
        assert getattr(pc, field) == getattr(old, field), field
    assert k // 32 % pc.in0_block_w == 0 and pc.in0_block_w == 8
    assert pc.compute_with_storage_grid_size != old.compute_with_storage_grid_size


@pytest.mark.parametrize(
    "x, weight",
    [
        (FT((1, 32, 768), "sharded", sharded=True), FT((768, DIM), "w")),
        (FT((1, 64, 768), "two tiles of rows"), FT((768, DIM), "w")),
        (FT((1, 32, 768), "gated"), FT((768, DIM // 2), "half-width weight")),
    ],
)
def test_the_res40_out_projection_refuses_what_res40_cannot_hold(progcfg_fields, linear, x, weight):
    assert _dar().out_matmul(x, weight, "cfg") is None and linear == []


def test_the_all_reduce_takes_a_res40_partial_without_a_reshard(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, mem: calls.append(("to_memory_config", t)) or t)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: calls.append(("deallocate", t)))
    monkeypatch.setattr(
        ttnn.experimental,
        "all_reduce_async",
        lambda packed, buf, **kw: calls.append(("all_reduce_async", packed)) or FT(packed.shape, "sum", "RES40"),
    )
    d = _dar()
    d.buffer, d.mesh, d.topology, d.num_links = "buffer", "mesh", "ring", 2
    d.tt_ccl = SimpleNamespace(get_and_cycle_barrier_semaphore_handle=lambda axis: "sem")
    partial = FT((1, 1, 32, DIM), "res40_partial", "RES40")
    d.all_reduce(partial, FT((1, 1, 32, DIM), "x", "RES40"))
    assert calls == [("all_reduce_async", partial), ("deallocate", partial)]


def test_the_tp8_in_projections_take_wider_grids_with_the_same_k_blocks(progcfg_fields):
    tp8, tp4 = _tp_args(8), _tp_args(4)
    qkv, qkvz = tp8.attn_qkv_decode_1d_progcfg, tp8.gdn_qkvz_decode_1d_progcfg
    # 60 cores on a device-wide (11) grid and 84 cores: one output tile per core for the 64 and 65 tile outputs
    assert qkv.compute_with_storage_grid_size == (11, 6) and qkv.per_core_N == 1
    assert qkvz.compute_with_storage_grid_size == (11, 8) and qkvz.per_core_N == 1
    assert tp4.attn_qkv_decode_1d_progcfg.compute_with_storage_grid_size == (8, 8)
    assert tp4.gdn_qkvz_decode_1d_progcfg.compute_with_storage_grid_size == (11, 4)
    for pc in (qkv, qkvz, tp4.attn_qkv_decode_1d_progcfg, tp4.gdn_qkvz_decode_1d_progcfg):
        assert pc.in0_block_w == 8 and pc.per_core_M == 1 and pc.mcast_in0 and pc.fused_activation is None


def test_a_tp8_grid_too_narrow_for_84_cores_keeps_the_tp4_in_projection_grids(progcfg_fields):
    """84 cores need 11 rows of an 8-wide (Wormhole) grid, so TP=8 there keeps the grids that fit."""
    tp8 = _tp_args(8, grid=(8, 8))
    assert tp8.attn_qkv_decode_1d_progcfg.compute_with_storage_grid_size == (8, 8)
    assert tp8.gdn_qkvz_decode_1d_progcfg.compute_with_storage_grid_size == (8, 6)
