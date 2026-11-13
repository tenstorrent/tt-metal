# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the replicated decode residual of the TP=8 MTP verify (tt/decode_allreduce.py).

* Which models and meshes take it (a (1, 8) mesh on a 1D fabric, dim 5120, dense MLP) and that the model and its layers
  share one instance per TT_CCL.
* DecodeResidualAllReduce.all_reduce: one all_reduce_async with one_shot and the residual, on the RES40 layout.
* Qwen36DecoderLayer.forward: a full-width verify input (attention spec_verify_mode, GDN verify) runs the local norms and
  hands each sublayer the residual-adding all-reduce, with no separate add; a fractured input keeps the old path; any
  other full-width input (plain decode, seed) is refused.
* TPGatedDeltaNet: the verify on vLLM KV cache blocks ends in that all-reduce.
* Qwen36Model: a verify of one tile of rows on vLLM KV cache blocks enters the replicated residual, and the verify tail
  takes its final norm and drafter feed from it.
The device numerics are in tests/ttnn/unit_tests/operations/ccl/test_all_reduce_async_{residual,one_shot}.py and
test_decode_allreduce.py.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import decode_allreduce as dar_mod
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

DIM = 5120
TP = 8


class FT:
    """Fake tensor: a shape, a dtype and a memory config."""

    def __init__(self, shape, name="t", mem=None, dtype=None):
        self.shape = tuple(shape)
        self.name = name
        self._mem = mem
        self.dtype = ttnn.bfloat16 if dtype is None else dtype

    def memory_config(self):
        return self._mem

    def __repr__(self):
        return f"FT({self.name}, {self.shape})"


@pytest.fixture
def fabric_1d(monkeypatch):
    fabric = SimpleNamespace(FABRIC_1D="1d", FABRIC_1D_RING="1d_ring", FABRIC_2D_TORUS_XY="2d_torus")
    monkeypatch.setattr(ttnn, "FabricConfig", fabric)
    current = {"value": fabric.FABRIC_1D}
    monkeypatch.setattr(ttnn, "get_fabric_config", lambda: current["value"])
    return current


def _mesh(shape=(1, TP), grid=(11, 10)):
    return SimpleNamespace(shape=shape, compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]))


@pytest.mark.parametrize(
    "change, reason",
    [
        ({}, None),
        ({"fabric": "1d_ring"}, None),
        ({"mesh": (1, 4)}, "mesh"),
        ({"mesh": (4, 8)}, "mesh"),
        ({"dim": 4096}, "hidden dim"),
        ({"fabric": "2d_torus"}, "1D fabric"),
        ({"moe": 256}, "MoE"),
        ({"grid": (8, 8)}, "compute grid"),
    ],
)
def test_only_the_tested_shape_takes_the_replicated_residual(fabric_1d, change, reason):
    fabric_1d["value"] = change.get("fabric", "1d")
    args = SimpleNamespace(dim=change.get("dim", DIM), moe_num_experts=change.get("moe", 0))
    got = dar_mod.decode_allreduce_unsupported(args, _mesh(change.get("mesh", (1, TP)), change.get("grid", (11, 10))))
    if reason is None:
        assert got is None
    else:
        assert reason in got


def test_the_model_and_its_layers_share_one_instance_per_tt_ccl(fabric_1d, monkeypatch):
    built = []
    monkeypatch.setattr(
        dar_mod, "DecodeResidualAllReduce", lambda mesh, args, ccl: built.append(ccl) or SimpleNamespace(num_links=2)
    )
    args = SimpleNamespace(dim=DIM)
    ccl_a, ccl_b = SimpleNamespace(mesh_device=_mesh()), SimpleNamespace(mesh_device=_mesh())
    first = dar_mod.get_decode_allreduce(args, ccl_a)
    assert first is not None and dar_mod.get_decode_allreduce(args, ccl_a) is first
    assert dar_mod.get_decode_allreduce(args, ccl_b) is not first
    assert built == [ccl_a, ccl_b]
    assert dar_mod.get_decode_allreduce(args, None) is None
    ccl_4 = SimpleNamespace(mesh_device=_mesh((1, 4)))
    assert dar_mod.get_decode_allreduce(args, ccl_4) is None and dar_mod.get_decode_allreduce(args, ccl_4) is None
    assert built == [ccl_a, ccl_b]


@pytest.fixture
def dar(monkeypatch):
    """A DecodeResidualAllReduce without a device, and a log of the ttnn calls its all_reduce makes."""
    calls = []
    monkeypatch.setattr(
        ttnn, "reshape", lambda t, shape: calls.append(("reshape", t)) or FT(shape, "view", t._mem, t.dtype)
    )
    monkeypatch.setattr(ttnn, "deallocate", lambda t: calls.append(("deallocate", t)))

    def to_mem(t, mem):
        calls.append(("to_memory_config", t, mem))
        return FT(t.shape, "packed", mem, t.dtype)

    def typecast(t, dtype):
        calls.append(("typecast", t, dtype))
        return FT(t.shape, "cast", t.memory_config(), dtype)

    def all_reduce_async(packed, buf, **kw):
        calls.append(("all_reduce_async", packed, buf, kw))
        return FT(packed.shape, "sum", kw["memory_config"])

    monkeypatch.setattr(ttnn, "to_memory_config", to_mem)
    monkeypatch.setattr(ttnn, "typecast", typecast)
    monkeypatch.setattr(ttnn.experimental, "all_reduce_async", all_reduce_async)
    d = object.__new__(dar_mod.DecodeResidualAllReduce)
    d.dim, d.res_mem, d.buffer, d.mesh, d.topology, d.num_links = DIM, "RES40", "buffer", "mesh", "ring", 2
    sems = iter(["sem0", "sem1", "sem0"])
    d.tt_ccl = SimpleNamespace(get_and_cycle_barrier_semaphore_handle=lambda axis: next(sems) + f"@{axis}")
    return d, calls


@pytest.mark.parametrize("rows", [4, 32])
@pytest.mark.parametrize("partial_dtype", ["bf16", "fp32"])
def test_all_reduce_adds_the_residual_in_one_one_shot_call(dar, rows, partial_dtype):
    d, calls = dar
    residual = FT((1, 1, rows, DIM), "x", "RES40")
    partial = FT((1, rows, DIM), "partial", "DRAM", ttnn.float32 if partial_dtype == "fp32" else None)
    out = d.with_residual(residual).all_reduce(partial)
    names = [c[0] for c in calls]
    expected = ["reshape", "to_memory_config", "deallocate"]
    if partial_dtype == "fp32":
        expected += ["typecast", "deallocate"]
    assert names == expected + ["all_reduce_async", "deallocate"]
    (_, packed, buf, kw) = calls[names.index("all_reduce_async")]
    assert packed.shape == (1, 1, rows, DIM) and packed.memory_config() == "RES40" and packed.dtype == ttnn.bfloat16
    assert buf == "buffer"
    assert kw["residual_tensor"] is residual and kw["one_shot"] is True
    assert kw["cluster_axis"] == 1 and kw["memory_config"] == "RES40" and kw["mesh_device"] == "mesh"
    assert kw["multi_device_global_semaphore"] == "sem0@1" and kw["num_links"] == 2 and kw["topology"] == "ring"
    assert calls[-1] == ("deallocate", packed) and out.name == "sum"


@pytest.mark.parametrize("residual", [FT((1, 1, 8, DIM), "x", "DRAM"), FT((1, 1, 16, DIM), "x", "RES40")])
def test_all_reduce_refuses_a_residual_outside_res40(dar, residual, expect_error):
    d, calls = dar
    with expect_error(ValueError, "RES40"):
        d.all_reduce(FT((1, 1, 8, DIM), "partial", "DRAM"), residual)
    assert not any(c[0] == "all_reduce_async" for c in calls)


class FakeDar:
    """Records the local norms and the residual each sublayer's all-reduce was bound to."""

    def __init__(self):
        self.norms, self.bound = [], []

    def rms_norm(self, x, norm):
        self.norms.append((x, norm))
        return FT(x.shape, f"normed({x.name})")

    def with_residual(self, residual):
        self.bound.append(residual)
        return ("bound", residual)


def _layer(is_full_attention, dar, calls):
    layer = object.__new__(Qwen36DecoderLayer)
    layer.num_devices = TP
    layer.is_full_attention = is_full_attention
    layer.args = SimpleNamespace(dim=DIM, get_norm_config=lambda kind, mode: {})
    layer._decode_ar = dar

    class Norm:
        def __init__(self, name):
            self.norm = f"{name}.norm"
            self.name = name

        def __call__(self, x, mode=None, norm_config=None):
            calls.append((self.name, x))
            return FT((1, 1, x.shape[-2], DIM), f"gathered({x.name})")

    layer.attention_norm, layer.ffn_norm = Norm("attention_norm"), Norm("ffn_norm")

    def mixer(name):
        def run(x, *a, **kw):
            calls.append((name, x, kw.get("decode_ar")))
            return FT((1, 1, x.shape[-2], kw["decode_ar"][1].shape[-1] if "decode_ar" in kw else DIM // TP), "mixed")

        return run

    layer.attention = SimpleNamespace(
        forward_decode=mixer("attention.forward_decode"),
        forward_verify_recurrent=mixer("attention.forward_verify_recurrent"),
        forward_seed_recurrent=mixer("attention.forward_seed_recurrent"),
    )

    def mlp(x, mode=None, decode_ar=None):
        calls.append(("mlp", x, decode_ar))
        return FT((1, 1, x.shape[-2], DIM if decode_ar is not None else DIM // TP), "mlp_out")

    layer.feed_forward = SimpleNamespace(forward=mlp)
    return layer


def _forward(layer, x, step):
    if step == "decode":
        return layer.forward(x, mode="decode")
    if step == "attn_verify":
        return layer.forward(x, mode="decode", alias_kv_write=True, spec_verify_mode=True, n_users=2)
    if step == "verify":
        return layer.forward(x, mode="prefill", valid_len=8, gdn_recurrent=True, decode_cfg=True, n_users=2)
    return layer.forward(x, mode="prefill", valid_len=2, gdn_recurrent=True, gdn_seed=True, decode_cfg=True, n_users=2)


@pytest.fixture
def no_add(monkeypatch):
    adds = []
    monkeypatch.setattr(ttnn, "add", lambda a, b, **kw: adds.append((a, b)) or FT(a.shape, "sum"))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    return adds


@pytest.mark.parametrize(
    "is_full_attention, step, mixer",
    [
        (True, "attn_verify", "attention.forward_decode"),
        (False, "verify", "attention.forward_verify_recurrent"),
    ],
)
def test_a_replicated_input_runs_both_sublayers_on_the_residual_all_reduce(no_add, is_full_attention, step, mixer):
    calls, dar = [], FakeDar()
    layer = _layer(is_full_attention, dar, calls)
    x = FT((1, 1, 8, DIM), "x")
    out = _forward(layer, x, step)
    (mix,) = [c for c in calls if c[0] == mixer]
    (ff,) = [c for c in calls if c[0] == "mlp"]
    h = dar.bound[1]
    assert mix[2] == ("bound", x), "the mixer's all-reduce adds the layer input"
    assert h.name == "mixed" and ff[2] == ("bound", h), "the MLP's all-reduce adds the mixer's output"
    assert [n for _, n in dar.norms] == ["attention_norm.norm", "ffn_norm.norm"]
    assert dar.norms[0][0] is x and dar.norms[1][0] is h
    assert not any(c[0] in ("attention_norm", "ffn_norm") for c in calls), "no gathering norm"
    assert no_add == [] and out.name == "mlp_out"


@pytest.mark.parametrize("step", ["decode", "verify", "seed"])
def test_a_fractured_input_keeps_the_gathering_norms_and_the_adds(no_add, step):
    calls, dar = [], FakeDar()
    layer = _layer(False, dar, calls)
    x = FT((1, 1, 8, DIM // TP), "x")
    _forward(layer, x, step)
    assert [c[0] for c in calls if c[0] in ("attention_norm", "ffn_norm")] == ["attention_norm", "ffn_norm"]
    assert all(c[2] is None for c in calls if len(c) == 3), "no module gets an all-reduce bound to a residual"
    assert len(no_add) == 2 and dar.norms == [] and dar.bound == []


@pytest.mark.parametrize("is_full_attention, step", [(True, "decode"), (False, "decode"), (False, "seed")])
def test_a_replicated_input_outside_the_verify_is_refused(no_add, expect_error, is_full_attention, step):
    layer = _layer(is_full_attention, FakeDar(), [])
    with expect_error(ValueError, "only in the MTP verify"):
        _forward(layer, FT((1, 1, 2, DIM), "x"), step)


@pytest.fixture
def gdn_layer(monkeypatch):
    from models.demos.blackhole.qwen36.tt.gdn import tp

    reduce_scatters = []
    monkeypatch.setattr(ttnn.experimental.kda, "gdn_spec_step", lambda *a, **k: "gated")
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: FT(shape, "view"))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(tp.tpc, "matmul_1d_decode", lambda *a, **k: FT((1, 1, 32, 64), "qkvzab"))
    monkeypatch.setattr(tp, "tt_all_reduce", lambda partial, *a, **k: reduce_scatters.append(partial) or "fractured")
    layer = SimpleNamespace(
        Nv=6, Nk=2, Dk=128, Dv=128, K=4, qkvz_dim_tp=64, scale=1.0, mesh=None, tt_ccl=None, cfg=None,
        tw=dict.fromkeys(["qkvz", "conv_taps_rows", "dt_bias_fp32", "neg_exp_A_fp32", "norm_w_1d", "out"]),
        args=SimpleNamespace(gdn_qkvz_decode_1d_progcfg=None, ccl_topology=lambda: None),
        state_blocks=SimpleNamespace(verify=SimpleNamespace(T=4, ctrl="ctrl", idx="idx"), rec_k="rk", rec_v="rv", hist="h"),
        _row_proj=lambda gated, w: FT((1, 8, DIM), "partial"),
    )  # fmt: skip
    layer._forward_verify_blocks = tp.TPGatedDeltaNet._forward_verify_blocks.__get__(layer)
    return tp, layer, reduce_scatters


class RecordingAllReduce:
    def __init__(self):
        self.partials = []

    def all_reduce(self, partial):
        self.partials.append(partial)
        return "replicated"


def test_the_gdn_verify_on_vllm_blocks_ends_in_the_residual_all_reduce(gdn_layer):
    tp, layer, reduce_scatters = gdn_layer
    ar = RecordingAllReduce()
    x = FT((1, 1, 8, DIM), "x")
    assert tp.TPGatedDeltaNet.forward_verify_recurrent(layer, x, 8, pre_gathered=True, n_users=2, decode_ar=ar) == (
        "replicated"
    )
    assert [p.shape for p in ar.partials] == [(1, 1, 8, DIM)] and reduce_scatters == []
    assert tp.TPGatedDeltaNet.forward_verify_recurrent(layer, x, 8, pre_gathered=True, n_users=2) == "fractured"
    assert len(reduce_scatters) == 1


def test_the_gdn_verify_without_vllm_blocks_refuses_the_residual_all_reduce(gdn_layer, expect_error):
    tp, layer, _ = gdn_layer
    layer.state_blocks = None
    x = FT((1, 1, 8, DIM), "x")
    with expect_error(ValueError, "vLLM KV cache blocks"):
        tp.TPGatedDeltaNet.forward_verify_recurrent(layer, x, 8, pre_gathered=True, n_users=2, decode_ar=object())


def _model(dar, blocks=(True, True)):
    model = object.__new__(Qwen36Model)
    model._decode_ar = dar
    model.args = SimpleNamespace(dim=DIM)
    model.num_devices = TP
    model.layers = [SimpleNamespace(is_full_attention=True)] + [
        SimpleNamespace(is_full_attention=False, attention=SimpleNamespace(state_blocks=object() if b else None))
        for b in blocks
    ]
    return model


def test_a_verify_enters_the_replicated_residual_only_with_one_tile_of_rows_on_vllm_blocks():
    dar = object()
    assert _model(dar)._residual_ar(32) is dar
    assert _model(dar)._residual_ar(33) is None
    assert _model(dar, blocks=(True, False))._residual_ar(8) is None
    assert _model(None)._residual_ar(8) is None


def test_the_verify_tail_takes_the_final_norm_and_drafter_feed_from_the_replicated_residual(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "deallocate", lambda t: calls.append(("deallocate", t)))
    monkeypatch.setattr(
        ttnn,
        "mesh_partition",
        lambda t, dim, cluster_axis, memory_config: calls.append(("partition", t, dim)) or "feed",
    )
    dar = SimpleNamespace(final_norm=lambda x, norm: calls.append(("final_norm", x, norm)) or FT(x.shape, "normed"))
    model = _model(dar)
    model.norm = SimpleNamespace(norm="final.norm")
    model._lm_head = lambda normed: calls.append(("lm_head", normed)) or "logits"
    x = FT((1, 1, 8, DIM), "x")
    assert model._spec_tail(x) == ("logits", "feed")
    names = [c[0] for c in calls]
    assert names == ["final_norm", "deallocate", "lm_head", "partition", "deallocate"]
    assert calls[0][1] is x and calls[0][2] == "final.norm" and calls[1][1] is x
    normed = calls[2][1]
    assert normed.name == "normed" and calls[3][1] is normed and calls[3][2] == 3 and calls[4][1] is normed
