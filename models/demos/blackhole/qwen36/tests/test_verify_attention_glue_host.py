# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the op sequence of the MTP verify's attention layers (TPAttention._forward_verify).

* A spec_verify_mode forward_decode of n_users x T rows takes _forward_verify: the q|k|v slice lands in L1 with no copy,
  cos / sin staged as [1, 1, R, rope_dim] are not reshaped, the RoPE pass-through slice lands in DRAM with no copy, k and
  v reach the KV write unpadded, the KV write is one paged_fused_update_cache(rows_per_user=T) for K and V (also at one
  user) on the per-row positions and the per-user page table, and the gate multiplies the attention output after the
  head concat in its flat layout.
* Which verifies take that path (_forward_verify_ok); the plain decode never does.
* Qwen36Model: the verify body slices no positions for the KV write; the verify stages cos / sin as [1, 1, R, rope_dim]
  with the per-row decode values.
The device comparison against the previous op sequence is in test_verify_attention_glue.py.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.attention import rope_tp
from models.demos.blackhole.qwen36.tt.attention import tp as tp_mod
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

NH, NKV, HD, RD, DIM = 3, 1, 256, 64, 5120
L1, DRAM = "L1", "DRAM"


class FT:
    """Fake tensor: a shape, a name and a memory config."""

    def __init__(self, shape, name, mem=None):
        self.shape = tuple(shape)
        self.name = name
        self.mem = mem

    def __repr__(self):
        return f"FT({self.name}, {self.shape}, {self.mem})"


@pytest.fixture
def ops(monkeypatch):
    """Replace the ttnn ops of the attention verify by recorders; returns the call log."""
    log = []

    def rec(op, out, *args, **kw):
        log.append((op, args, kw, out))
        return out

    def slice_(t, starts, ends, steps=None, memory_config=None):
        steps = steps or (1,) * len(starts)
        shape = tuple(-(-(e - s) // st) for s, e, st in zip(starts, ends, steps))
        return rec(
            "slice", FT(shape, f"slice({t.name})", memory_config or t.mem), t, starts, ends, steps, mem=memory_config
        )

    def reshape(t, shape, padded=None, memory_config=None):
        return rec("reshape", FT(shape, f"view({t.name})", t.mem), t, tuple(shape))

    def transpose(t, a, b):
        s = list(t.shape)
        s[a], s[b] = s[b], s[a]
        return rec("transpose", FT(s, f"T({t.name})", t.mem), t)

    def concat(ts, dim, memory_config=None):
        s = list(ts[0].shape)
        s[dim] = sum(t.shape[dim] for t in ts)
        return rec("concat", FT(s, "cat", memory_config or ts[0].mem), *ts)

    def heads(qkv3, num_heads, num_kv_heads, memory_config):
        R = qkv3.shape[-2]
        out = (FT((1, R, NH, HD), "q_sh"), FT((1, R, NKV, HD), "k_sh"), FT((1, R, NKV, HD), "v_sh"))
        return rec("nlp_create_qkv_heads_decode", out, qkv3)

    def same(op, name=None):
        return lambda t, *a, memory_config=None, **kw: rec(
            op, FT(t.shape, name or f"{op}({t.name})", memory_config), t, *a
        )

    monkeypatch.setattr(ttnn, "Shape", lambda s: tuple(s))
    monkeypatch.setattr(ttnn, "L1_MEMORY_CONFIG", L1)
    monkeypatch.setattr(ttnn, "DRAM_MEMORY_CONFIG", DRAM)
    monkeypatch.setattr(ttnn, "slice", slice_)
    monkeypatch.setattr(ttnn, "reshape", reshape)
    monkeypatch.setattr(ttnn, "transpose", transpose)
    monkeypatch.setattr(ttnn, "concat", concat)
    monkeypatch.setattr(ttnn, "pad", lambda t, *a, **kw: rec("pad", FT(t.shape, "pad"), t))
    monkeypatch.setattr(
        ttnn, "to_memory_config", lambda t, mem: rec("to_memory_config", FT(t.shape, f"mc({t.name})", mem), t, mem)
    )
    monkeypatch.setattr(ttnn, "sharded_to_interleaved", lambda t, mem: rec("s2i", FT(t.shape, t.name[0], mem), t))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("deallocate", (t,), {}, None)))
    monkeypatch.setattr(ttnn, "rms_norm", same("rms_norm"))
    monkeypatch.setattr(
        ttnn,
        "multiply",
        lambda a, b, memory_config=None: rec("multiply", FT(a.shape, f"mul({a.name})", memory_config), a, b),
    )
    monkeypatch.setattr(ttnn, "sigmoid", same("sigmoid"))
    monkeypatch.setattr(ttnn.experimental, "rotary_embedding_hf", same("rope", "roped"))
    monkeypatch.setattr(ttnn.experimental, "nlp_create_qkv_heads_decode", heads)
    monkeypatch.setattr(
        ttnn.experimental,
        "paged_update_cache",
        lambda c, x, update_idxs_tensor, page_table: rec("update", None, c, x, update_idxs_tensor, page_table),
    )
    monkeypatch.setattr(
        ttnn.experimental,
        "paged_fused_update_cache",
        lambda c1, x1, c2, x2, **kw: rec("fused_update", None, c1, x1, c2, x2, **kw),
    )
    monkeypatch.setattr(
        ttnn.transformer,
        "paged_scaled_dot_product_attention_decode",
        lambda q, k, v, **kw: rec("sdpa", FT(q.shape, "attn", L1), q, **kw),
    )
    monkeypatch.setattr(
        tp_mod.tpc,
        "matmul_1d_decode",
        lambda x, w, *a, **kw: rec("qkv_mm", FT((1, 1, x.shape[-2], (2 * NH + 2 * NKV) * HD), "qkv", DRAM), x),
    )
    monkeypatch.setattr(tp_mod, "KV_GROUP_WRITE_OK", True)
    return log


def _attention(n_users):
    a = object.__new__(tp_mod.TPAttention)
    a.mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
    a.args = SimpleNamespace(
        proj_1d_decode=True, attn_qkv_decode_1d_progcfg=None, kv_update_shard_cfg="kv_cfg", ccl_topology=lambda: None
    )
    a.tw = {"q_norm": "q_norm", "k_norm": "k_norm", "wo": "wo", "wqkv_fused": "wqkv"}
    a.NH, a.NKV, a.HD, a.rope_dim, a.scale, a.compute_cfg = NH, NKV, HD, RD, HD**-0.5, None
    a.B = n_users
    a.use_paged, a.paged_k, a.paged_v = True, "k_cache", "v_cache"
    a._fused_qkv = a._use_nlp_decode_heads = True
    a._fuse_agmm = False
    a._spec_sdpa_cfg_cache, a._kv_shard_cfg_cache = {}, {}
    a._kv_rows_shard_cfg = lambda first, n, rows: ("rows_cfg", first, n, rows)
    a._concat_heads_decode = lambda t, B: FT((1, B, NH * HD), f"concat({t.name})", L1)
    a._wo_proj = lambda t, w: FT((1, t.shape[-2], DIM), "wo_partial", DRAM)
    return a


class RecordingAllReduce:
    def __init__(self):
        self.partials = []

    def all_reduce(self, partial):
        self.partials.append(partial)
        return "replicated"


def _verify(a, n_users, T, cos_rows_first=True):
    R = n_users * T
    groups = a._spec_sdpa_plan(T, n_users)[1]
    rope_shape = (1, 1, R, RD) if cos_rows_first else (1, R, 1, RD)
    kw = dict(
        page_table=FT((R, 16), "pt_rows"),
        alias_kv_write=True,
        spec_verify_mode=True,
        spec_page_table=FT((groups, 16), "pt_groups"),
        spec_n_users=n_users,
        spec_user_page_table=FT((n_users, 16), "pt_users"),
        decode_ar=RecordingAllReduce(),
    )
    out = a.forward_decode(
        FT((1, 1, R, DIM), "x"), FT((R,), "cur_pos"), FT(rope_shape, "cos"), FT(rope_shape, "sin"), **kw
    )
    return out, kw


def _calls(log, op):
    return [c for c in log if c[0] == op]


@pytest.mark.parametrize("n_users, T", [(8, 4), (1, 4), (2, 8), (2, 12)])
def test_the_verify_attention_runs_without_the_copies_pads_and_reshapes(ops, n_users, T):
    a = _attention(n_users)
    out, kw = _verify(a, n_users, T)
    R = n_users * T
    assert out == "replicated" and [p.shape for p in kw["decode_ar"].partials] == [(1, 1, R, DIM)]
    assert _calls(ops, "pad") == []
    # q|k|v lands in L1 from the slice; the only copies left are the KV write's two shards (K and V)
    (qkv3,) = [c for c in _calls(ops, "slice") if c[1][0].name == "qkv" and c[1][2][-1] == (NH + 2 * NKV) * HD]
    assert qkv3[2]["mem"] == L1 and _calls(ops, "nlp_create_qkv_heads_decode")[0][1][0] is qkv3[3]
    copies = [c for c in _calls(ops, "to_memory_config") if c[1][0].mem != c[1][1]]  # same config: no device op
    assert len(copies) == 2
    # cos / sin are used as staged, and the RoPE pass-through slice writes DRAM itself (q and k)
    assert not any(c[1][0].name in ("cos", "sin") for c in _calls(ops, "reshape"))
    passes = [c for c in _calls(ops, "slice") if c[1][1][-1] == RD]
    assert len(passes) == 2 and all(c[2]["mem"] == DRAM for c in passes)
    # one KV write for K and V: each the unpadded [1, R, NKV, HD] viewed as [1, n_users, T*32, HD], sharded one user
    # per core (V on cores [0, n), K on [n, 2n)), all T rows per user at the per-row positions, per-user page table
    assert _calls(ops, "update") == []
    (upd,) = _calls(ops, "fused_update")
    assert upd[1][0] == "k_cache" and upd[1][2] == "v_cache"
    rows = T * 32
    for shard, cfg in ((upd[1][1], ("rows_cfg", n_users, n_users, rows)), (upd[1][3], ("rows_cfg", 0, n_users, rows))):
        (src,) = [c for c in copies if c[3] is shard]
        assert src[1][1] == cfg and src[1][0].name.startswith("view(") and src[1][0].shape == (1, n_users, rows, HD)
        (view,) = [c for c in _calls(ops, "reshape") if c[3] is src[1][0]]
        assert view[1][0].shape == (1, R, NKV, HD) and view[1][1] == (1, n_users, rows, HD)
    assert upd[2]["update_idxs_tensor"].name == "cur_pos" and upd[2]["update_idxs_tensor"].shape == (R,)
    assert upd[2]["page_table"] is kw["spec_user_page_table"] and upd[2]["rows_per_user"] == T
    assert not any(c[1][0].name == "cur_pos" for c in _calls(ops, "slice"))
    freed = [c[1][0] for c in ops if c[0] == "deallocate"]
    assert upd[1][1] in freed and upd[1][3] in freed
    assert ops.index(upd) < min(
        i for i, c in enumerate(ops) if c[0] == "deallocate" and c[1][0] in (upd[1][1], upd[1][3])
    )
    # fused spec SDPA on the group table
    (sdpa,) = _calls(ops, "sdpa")
    assert sdpa[2]["page_table_tensor"] is kw["spec_page_table"] and sdpa[2]["spec_multi_pos_tiles"] == T // (
        a._spec_sdpa_plan(T, n_users)[1] // n_users
    )
    # the gate: no head-layout reshape, sigmoid on its flat view, multiplied after the head concat
    gate_reshapes = [c for c in _calls(ops, "reshape") if c[1][0].name == "slice(qkv)"]
    assert [c[1][1] for c in gate_reshapes] == [(1, R, NH * HD)]
    (sig,) = _calls(ops, "sigmoid")
    assert sig[1][0] is gate_reshapes[0][3]
    gated = [c for c in _calls(ops, "multiply") if c[1][1] is sig[3]]
    assert len(gated) == 1 and gated[0][1][0].name == "concat(view(attn))"


@pytest.mark.parametrize("n_users, T", [(8, 4), (1, 4), (4, 8), (2, 12)])
def test_the_multirow_kv_write_puts_k_and_v_on_disjoint_cores_one_user_each(monkeypatch, n_users, T):
    monkeypatch.setattr(ttnn, "CoreCoord", lambda x, y: (x, y))
    monkeypatch.setattr(ttnn, "CoreRange", lambda a, b: (a, b))
    monkeypatch.setattr(ttnn, "CoreRangeSet", lambda ranges: sorted(ranges, key=lambda r: (r[0][1], r[0][0])))
    monkeypatch.setattr(ttnn, "ShardSpec", lambda cores, shape, orientation: (cores, tuple(shape)))
    monkeypatch.setattr(ttnn, "MemoryConfig", lambda layout, buffer_type, spec: spec)
    a = _attention(n_users)
    del a._kv_rows_shard_cfg  # the class method, not _attention's stand-in
    rows = T * 32
    v_cores, v_shape = a._kv_rows_shard_cfg(0, n_users, rows)
    k_cores, k_shape = a._kv_rows_shard_cfg(n_users, n_users, rows)
    assert v_shape == k_shape == (rows, HD)
    # row-wise over the 11-wide grid: shard u (user u) on core u of its set
    assert [r[0] for r in v_cores] == [(i % 11, i // 11) for i in range(n_users)]
    assert [r[0] for r in k_cores] == [(i % 11, i // 11) for i in range(n_users, 2 * n_users)]
    assert all(r[0] == r[1] for r in v_cores + k_cores) and not {r[0] for r in v_cores} & {r[0] for r in k_cores}
    assert a._kv_rows_shard_cfg(0, n_users, rows) is a._kv_rows_shard_cfg(0, n_users, rows)


def test_the_multirow_kv_write_refuses_mismatched_positions_and_a_full_grid(ops, expect_error):
    a = _attention(8)
    k, v = FT((1, 32, NKV, HD), "k"), FT((1, 32, NKV, HD), "v")
    with expect_error(ValueError, "positions for 8 x 4 rows"):
        a._write_kv_multirow(k, v, FT((8,), "cur_pos"), 8, 4, FT((8, 16), "pt_users"))
    del a._kv_rows_shard_cfg
    with expect_error(ValueError, "exceed the 11x10 grid"):
        a._kv_rows_shard_cfg(100, 16, 128)


def test_rope_reshapes_decode_layout_cos_sin_only(ops):
    x = FT((1, 8, NH, HD), "q", L1)
    rope_tp.apply_partial_rope_decode(x, FT((1, 1, 8, RD), "cos"), FT((1, 1, 8, RD), "sin"), NH, 8, RD)
    assert _calls(ops, "reshape") == []
    rope_tp.apply_partial_rope_decode(x, FT((1, 8, 1, RD), "cos"), FT((1, 8, 1, RD), "sin"), NH, 8, RD)
    assert [(c[1][0].name, c[1][1]) for c in _calls(ops, "reshape")] == [("cos", (1, 1, 8, RD)), ("sin", (1, 1, 8, RD))]
    assert not any(c[1][1] == DRAM and c[1][0].name.startswith("slice") for c in _calls(ops, "to_memory_config"))


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "no_paged",
        "not_aliased",
        "split_qkv",
        "no_strided_slice",
        "T1",
        "T3",
        "rows33",
        "no_user_pt",
        "user_pt_rows",
        "group_pt_rows",
        "grid_too_small",
    ],
)
def test_which_verifies_take_forward_verify(ops, monkeypatch, change):
    a = _attention(8)
    if change == "grid_too_small":
        # 8 users need 16 cores for the multi-row KV write's two shard sets
        a.mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=5, y=3))
    n, T = (8, 1) if change == "T1" else (11, 3) if change == "rows33" else (8, 3) if change == "T3" else (8, 4)
    R = n * T
    groups = (a._spec_sdpa_plan(T, n) or (None, 1))[1]
    args = dict(
        B=R,
        page_table=None if change == "no_paged" else FT((R, 4), "pt"),
        alias_kv_write=change != "not_aliased",
        spec_page_table=FT((groups + (change == "group_pt_rows"), 4), "pt_groups"),
        spec_n_users=n,
        spec_user_page_table=None if change == "no_user_pt" else FT((n + (change == "user_pt_rows"), 4), "pt_users"),
    )
    if change == "split_qkv":
        a._fused_qkv = False
    if change == "no_strided_slice":
        monkeypatch.setattr(tp_mod, "KV_GROUP_WRITE_OK", False)
    assert a._forward_verify_ok(**args) is (change == "none")


def test_the_plain_decode_and_the_reseed_keep_the_full_body(ops, expect_error):
    a = _attention(8)

    class FullBody(Exception):
        pass

    def qkv(x, **kw):
        raise FullBody(f"full body, _qkv options {sorted(kw)}")

    a._qkv = qkv
    a._forward_verify = lambda *args, **kw: pytest.fail("only spec_verify_mode takes _forward_verify")
    for kw in ({}, {"alias_kv_write": True, "spec_n_users": 8}):
        with expect_error(FullBody, r"full body, _qkv options \[\]"):
            a.forward_decode(
                FT((1, 1, 32, DIM), "x"), FT((32,), "pos"), "cos", "sin", page_table=FT((32, 4), "pt"), **kw
            )


def test_the_verify_body_slices_no_positions_for_the_kv_write(monkeypatch):
    log = []
    monkeypatch.setattr(tp_mod, "KV_GROUP_WRITE_OK", True)
    monkeypatch.setattr(
        ttnn, "slice", lambda t, s, e, st=None: log.append(("slice", t, s, e, st)) or FT((2,), f"pos{s[0]}")
    )
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("deallocate", t)))
    monkeypatch.setattr(ttnn, "reshape", lambda t, s: t)
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, m: t)
    monkeypatch.setattr(ttnn, "untilize", lambda t, **kw: "u")
    monkeypatch.setattr(ttnn, "argmax", lambda t, **kw: "ids")
    monkeypatch.setattr(ttnn, "typecast", lambda t, dtype: "kvpos")

    def layer(full):
        def forward(x, **kw):
            log.append(("layer", full, sorted(kw)))
            return FT(x.shape, "h")

        return SimpleNamespace(is_full_attention=full, forward=forward)

    model = object.__new__(Qwen36Model)
    model._vfy_T, model._vfy_step_buf = 4, "step"
    model.args = SimpleNamespace(gdn_chunk_size=128)
    model.rope_lookup = lambda rows, n: (FT((1, 1, n, 64), "cos"), FT((1, 1, n, 64), "sin"))
    model._vfy_kvpt_buf = model._vfy_kvpt_users = model._vfy_kvpt1_buf = None
    model._vfy_state_idx = model._vfy_conv_sel = None
    model.embd = lambda tok: FT((1, 1, 8, DIM), "x")
    model._residual_ar = lambda rows: None
    model._spec_tail_ids = lambda x, shard_ids: log.append(("tail", x)) or ("logits", "feed", None)
    model.layers = [layer(False), layer(True), layer(False), layer(True)]
    model._forward_verify_bucket_tp(n_users=2)

    slices = [c for c in log if c[0] == "slice"]
    assert [(c[1], c[2], c[3]) for c in slices] == [("step", (r, 0), (r + 1, 8)) for r in range(3)]
    layers = [c for c in log if c[0] == "layer"]
    assert len(layers) == 4 and not any("spec_pos_groups" in c[2] for c in layers)
    tail = next(i for i, c in enumerate(log) if c[0] == "tail")
    last_layer = max(i for i, c in enumerate(log) if c[0] == "layer")
    freed = [c[1] for c in log[last_layer:tail] if c[0] == "deallocate"]
    assert "kvpos" in freed and [f.name for f in freed if isinstance(f, FT)] == ["h", "cos", "sin"]


def test_the_verify_stages_the_decode_rotation_rows_on_the_sequence_axis():
    model = object.__new__(Qwen36Model)
    model.args = SimpleNamespace(rope_head_dim=RD, rope_theta=1e7)
    model.rope = SimpleNamespace(rope_delta=3)
    pos = torch.tensor([100, 101, 102, 103, 7, 8, 9, 10], dtype=torch.int32)
    dec_cos, dec_sin = model._rope_tp_cos_sin_decode_torch(pos)
    cos, sin = model._verify_cos_sin_torch(pos)
    assert tuple(dec_cos.shape) == (1, 8, 1, RD) and tuple(cos.shape) == tuple(sin.shape) == (1, 1, 8, RD)
    assert cos.dtype == sin.dtype == torch.bfloat16
    assert torch.equal(cos.reshape(-1), dec_cos.reshape(-1)) and torch.equal(sin.reshape(-1), dec_sin.reshape(-1))
    # A multimodal verify passes one M-RoPE delta per row; both layouts take the same deltas.
    deltas = [0] * 4 + [-5] * 4
    dec_cos, dec_sin = model._rope_tp_cos_sin_decode_torch(pos, deltas)
    cos, sin = model._verify_cos_sin_torch(pos, deltas)
    assert torch.equal(cos.reshape(-1), dec_cos.reshape(-1)) and torch.equal(sin.reshape(-1), dec_sin.reshape(-1))
    assert not torch.equal(cos, model._verify_cos_sin_torch(pos)[0])
