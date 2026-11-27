# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the DRAM-sharded TP=8 decode MLP (Qwen36ModelArgs.mlp_decode_ds, tt/mlp.py).

* Qwen36ModelArgs._init_tp_config at TP=8 on a grid of 84 or more cores with 8 DRAM banks turns mlp_decode_ds on: one
  packed [gate | up] DRAM-sharded matmul on the RES40 shards (46 cores x 3 output tiles), SiLU(gate) * up into the down
  input shards (34 cores x 2 tiles), DRAM-sharded down (54 cores x 3 tiles), both on the multicore pipeline, at the
  LoFi of the other dense MLP matmuls. TP=4, a narrow grid, 7 DRAM banks and MLP_DECODE_DS = False keep the 1D MLP.
* load_mlp_weights: the packed per-device [gate_d | up_d] weight and the DRAM-sharded down load under their own cache
  keys, w1 and w3 do not load. A layer without the prefill AGMM (the MoE shared expert) or with the short-prefill path
  (prefill_small_m_max) keeps w1 and w3.
* Qwen36MLP._forward_tp of at most one tile of rows: the op sequence on a RES40 input (no reshard), on another layout
  (one reshard to RES40) and without the RES40 all-reduce (the partial goes interleaved to tt_all_reduce).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import mlp as mlp_mod
from models.demos.blackhole.qwen36.tt import model_config as mc
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

DIM, HIDDEN = 5120, 17408

# Qwen3.8-27B: the attributes Qwen36ModelArgs._init_tp_config reads.
_QWEN38 = dict(
    n_heads=24,
    n_kv_heads=4,
    head_dim=256,
    dim=DIM,
    hidden_dim=HIDDEN,
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


@pytest.fixture
def configs(monkeypatch):
    """ttnn config constructors that keep their fields, and a core-range builder that keeps its arguments."""
    for name in (
        "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        "WormholeComputeKernelConfig",
    ):
        monkeypatch.setattr(ttnn, name, lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(
        ttnn, "MemoryConfig", lambda layout, buffer, spec=None: SimpleNamespace(layout=layout, spec=spec)
    )
    monkeypatch.setattr(ttnn, "ShardSpec", lambda grid, shape, orientation: SimpleNamespace(grid=grid, shape=shape))
    monkeypatch.setattr(ttnn, "num_cores_to_corerangeset", lambda n, grid, row_wise: ("cores", n, row_wise))
    monkeypatch.setattr(ttnn, "CoreRangeSet", lambda ranges: ("res40", tuple(ranges)))
    monkeypatch.setattr(ttnn, "CoreRange", lambda a, b: (a, b))
    monkeypatch.setattr(ttnn, "CoreCoord", lambda x, y: (x, y))
    monkeypatch.setattr(ttnn, "MathFidelity", SimpleNamespace(LoFi="LoFi", HiFi2="HiFi2", HiFi4="HiFi4"))


def _mesh(tp, grid=(11, 10), banks=8):
    return SimpleNamespace(
        shape=(1, tp),
        compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]),
        dram_grid_size=lambda: SimpleNamespace(x=banks, y=1),
    )


def _tp_args(tp=8, **mesh_kw):
    args = SimpleNamespace(num_devices=tp, **_QWEN38)
    Qwen36ModelArgs._init_tp_config(args, _mesh(tp, **mesh_kw))
    return args


def _shard(mem):
    return mem.spec.grid[1], mem.spec.shape


def test_tp8_runs_the_decode_mlp_on_dram_sharded_matmuls(configs):
    args = _tp_args()
    assert args.mlp_decode_ds
    gu, down = args.mlp_ds_gateup_progcfg, args.mlp_ds_down_progcfg
    for pc in (gu, down):
        assert pc.per_core_M == 1 and pc.in0_block_w == 2 and pc.fused_activation is None
        assert pc.num_workers_per_dram_bank == 4 and pc.use_multicore_pipeline
    # 2 x 2176 / 32 = 136 output tiles: 46 cores x 3; 5120 / 32 = 160: 54 cores x 3
    assert gu.per_core_N == 3 and _shard(args.mlp_ds_gateup_out_memcfg) == (46, (32, 96))
    assert down.per_core_N == 3 and _shard(args.mlp_ds_down_out_memcfg) == (54, (32, 96))
    # SiLU(gate) * up: 68 tiles on 34 cores x 2 (down's K block), from RES40: 40 shards of 128 columns (gate+up's)
    assert _shard(args.mlp_ds_down_in_memcfg) == (34, (32, 64))
    assert args.mlp_ds_res_memcfg.spec.grid == ("res40", (((0, 0), (9, 3)),)) and args.mlp_ds_res_memcfg.spec.shape == (
        32,
        128,
    )
    # DRAM width shards over the 8 banks: [dim, 2 x hidden / tp] and [hidden / tp, dim]
    assert args.mlp_ds_gateup_weight_memcfg.spec.shape == (DIM, 2 * HIDDEN // 8 // 8)
    assert args.mlp_ds_w2_weight_memcfg.spec.shape == (HIDDEN // 8, DIM // 8)
    ckc = args.mlp_ds_compute_kernel_config
    assert ckc.math_fidelity == "LoFi" and ckc.fp32_dest_acc_en and ckc.packer_l1_acc


@pytest.mark.parametrize(
    "tp, mesh_kw",
    [(4, {}), (8, {"grid": (8, 8)}), (8, {"banks": 7})],
    ids=["tp4", "grid-under-84-cores", "7-dram-banks"],
)
def test_other_meshes_keep_the_1d_decode_mlp(configs, tp, mesh_kw):
    assert not _tp_args(tp, **mesh_kw).mlp_decode_ds


def test_mlp_decode_ds_false_keeps_the_1d_decode_mlp(configs, monkeypatch):
    monkeypatch.setattr(mc, "MLP_DECODE_DS", False)
    assert not _tp_args().mlp_decode_ds


def test_the_fidelity_constant_sets_both_matmuls(configs, monkeypatch):
    monkeypatch.setattr(mc, "MLP_DS_MATH_FIDELITY", "HiFi4")
    assert _tp_args().mlp_ds_compute_kernel_config.math_fidelity == "HiFi4"


@pytest.fixture
def as_tensor(monkeypatch):
    calls = []

    def fake(t, **kw):
        calls.append(kw)
        return SimpleNamespace(name=Path(kw["cache_file_name"]).name, **kw)

    monkeypatch.setattr(ttnn, "as_tensor", fake)
    return calls


def _state():
    return {f"{n}_proj.weight": torch.zeros(1) for n in ("gate", "up", "down")}


def test_ds_weights_load_under_their_own_cache_keys(configs, as_tensor):
    args = _tp_args()
    w = mlp_mod.load_mlp_weights(_mesh(8), _state(), Path("/cache"), args=args)
    assert w.w1 is None and w.w3 is None
    assert w.w2.name == "mlp.down_proj.weight.ds.tp" and w.w2.memory_config is args.mlp_ds_w2_weight_memcfg
    assert w.w2.dtype == ttnn.bfloat8_b
    assert w.w_gu_ds.name == "mlp.gate_up.weight.ds_packed.tp"
    assert w.w_gu_ds.memory_config is args.mlp_ds_gateup_weight_memcfg and w.w_gu_ds.dtype == ttnn.bfloat4_b
    assert w.w_gate_up.name == "mlp.gate_up.weight.swiglu.tp"  # the prefill AGMM weight stays
    assert sorted(c["cache_file_name"].split("/")[-1] for c in as_tensor) == [
        "mlp.down_proj.weight.ds.tp",
        "mlp.gate_up.weight.ds_packed.tp",
        "mlp.gate_up.weight.swiglu.tp",
    ]


def test_the_packed_weight_holds_gate_then_up_per_device(configs, as_tensor):
    """Device d's columns of the packed [K, 2 x hidden] weight are [gate_d | up_d], each hidden / tp wide."""
    tp, k, hidden = 8, 4, 16
    gate = torch.arange(hidden * k, dtype=torch.float32).reshape(hidden, k)
    up = -gate - 1
    mlp_mod._build_gate_up_ds(gate, up, _mesh(tp), tp, "/c/x", "mem")
    packed = as_tensor[0]["preprocess"](gate).float()
    n = hidden // tp
    assert packed.shape == (k, 2 * hidden)
    for d in range(tp):
        block = packed[:, d * 2 * n : (d + 1) * 2 * n]
        assert torch.equal(block[:, :n], gate.T[:, d * n : (d + 1) * n])
        assert torch.equal(block[:, n:], up.T[:, d * n : (d + 1) * n])


@pytest.mark.parametrize(
    "use_gateup_agmm, small_m_max", [(False, 0), (True, 128)], ids=["no-prefill-agmm", "short-prefill-path"]
)
def test_layers_that_read_gate_and_up_keep_w1_and_w3(configs, as_tensor, use_gateup_agmm, small_m_max):
    args = _tp_args()
    args.prefill_small_m_max = small_m_max
    assert not mlp_mod.mlp_decode_ds(args, use_gateup_agmm)
    w = mlp_mod.load_mlp_weights(_mesh(8), _state(), Path("/cache"), args=args, use_gateup_agmm=use_gateup_agmm)
    assert w.w1 is not None and w.w3 is not None and w.w_gu_ds is None
    assert not any(".ds" in c["cache_file_name"] for c in as_tensor)


class FT:
    """A device tensor stand-in: shape, a name and a memory config."""

    def __init__(self, shape, name, mem="L1"):
        self.shape, self.name, self._mem = tuple(shape), name, mem

    def memory_config(self):
        return self._mem

    def __getitem__(self, idx):
        sl = idx[-1]
        width = (sl.stop or self.shape[-1]) - (sl.start or 0)
        return FT(self.shape[:-1] + (width,), f"{self.name}[{sl.start}:{sl.stop}]")

    def __repr__(self):
        return f"FT({self.name})"


@pytest.fixture
def ops(monkeypatch):
    """Records every ttnn op the decode MLP calls: (op, first input name, the kwargs that matter)."""
    calls = []

    def linear(x, w, **kw):
        calls.append(("linear", x.name, w, kw["program_config"], kw["memory_config"], kw["compute_kernel_config"]))
        return FT(x.shape[:-1] + (w.shape[-1],), f"mm({w.name})", kw["memory_config"])

    def to_memory_config(t, mem):
        calls.append(("to_memory_config", t.name, mem))
        return FT(t.shape, f"{mem}({t.name})", mem)

    def mul(a, b, **kw):
        calls.append(("mul", a.name, b.name, kw["input_tensor_a_activations"], kw["memory_config"]))
        return FT(a.shape, "hidden", kw["memory_config"])

    monkeypatch.setattr(ttnn, "linear", linear)
    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)
    monkeypatch.setattr(ttnn, "mul", mul)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: calls.append(("deallocate", t.name)))
    monkeypatch.setattr(ttnn, "L1_MEMORY_CONFIG", "L1")
    monkeypatch.setattr(ttnn, "DRAM_MEMORY_CONFIG", "DRAM")
    monkeypatch.setattr(ttnn, "UnaryOpType", SimpleNamespace(SILU="SILU"))
    return calls


def _mlp(args):
    m = object.__new__(mlp_mod.Qwen36MLP)
    m.device, m.args, m.tt_ccl, m.num_devices = "mesh", args, "ccl", 8
    m._mlp_1d_decode, m._dram_sharded, m._fuse_gateup_agmm, m._decode_ds = True, False, True, True
    m.compute_kernel_config_ds, m.compute_kernel_config_decode, m.compute_kernel_config = "ckc_ds", "ckc", "ckc"
    m.weights = mlp_mod.MLPWeights(
        w1=None,
        w3=None,
        w2=SimpleNamespace(name="w2_ds", shape=(HIDDEN // 8, DIM)),
        w_gate_up="w_gate_up",
        w_gu_ds=SimpleNamespace(name="w_gu_ds", shape=(DIM, 2 * HIDDEN // 8)),
    )
    args.mlp_ds_res_memcfg, args.mlp_ds_gateup_out_memcfg = "RES40", "GU_OUT"
    args.mlp_ds_down_in_memcfg, args.mlp_ds_down_out_memcfg = "DOWN_IN", "DOWN_OUT"
    args.mlp_ds_gateup_progcfg, args.mlp_ds_down_progcfg = "PC_GU", "PC_DOWN"
    return m


def _ds_ops(m, x_name):
    w = m.weights
    return [
        ("linear", x_name, w.w_gu_ds, "PC_GU", "GU_OUT", "ckc_ds"),
        ("to_memory_config", "mm(w_gu_ds)", "L1"),
        ("deallocate", "mm(w_gu_ds)"),
        ("mul", "L1(mm(w_gu_ds))[None:2176]", "L1(mm(w_gu_ds))[2176:None]", ["SILU"], "DOWN_IN"),
        ("deallocate", "L1(mm(w_gu_ds))[None:2176]"),
        ("deallocate", "L1(mm(w_gu_ds))[2176:None]"),
        ("deallocate", "L1(mm(w_gu_ds))"),
        ("linear", "hidden", w.w2, "PC_DOWN", "DOWN_OUT", "ckc_ds"),
        ("deallocate", "hidden"),
    ]


@pytest.mark.parametrize("rows", [4, 32])
def test_a_res40_input_goes_straight_to_the_packed_matmul_and_the_all_reduce(configs, ops, rows):
    m = _mlp(_tp_args())
    got = []
    dar = SimpleNamespace(all_reduce=lambda p: got.append(p) or "sum")
    assert m._forward_tp(FT((1, 1, rows, DIM), "x", "RES40"), dar) == "sum"
    assert ops == _ds_ops(m, "x")
    assert got[0].name == "mm(w2_ds)" and got[0].memory_config() == "DOWN_OUT"


def test_another_layout_is_resharded_to_res40_once(configs, ops):
    m = _mlp(_tp_args())
    m._forward_tp(FT((1, 1, 32, DIM), "x", "ACT_SHARD"), SimpleNamespace(all_reduce=lambda p: p))
    assert ops[0] == ("to_memory_config", "x", "RES40")
    assert ops[1:] == _ds_ops(m, "RES40(x)")[:1] + [("deallocate", "RES40(x)")] + _ds_ops(m, "RES40(x)")[1:]


def test_without_the_res40_all_reduce_the_partial_goes_interleaved_to_tt_all_reduce(configs, ops, monkeypatch):
    import models.tt_transformers.tt.ccl as ccl

    reduced = []
    monkeypatch.setattr(ccl, "tt_all_reduce", lambda p, *a, **kw: reduced.append(p) or "out")
    args = _tp_args()
    args.ccl_topology = lambda: "ring"
    m = _mlp(args)
    assert m._forward_tp(FT((1, 1, 32, DIM), "x", "DRAM")) == "out"
    assert ops[-2:] == [("to_memory_config", "mm(w2_ds)", "L1"), ("deallocate", "mm(w2_ds)")]
    assert reduced[0].name == "L1(mm(w2_ds))"
