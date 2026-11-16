# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the row view of the verify's partial RoPE (rope_tp._partial_rope_decode_row_view).

* apply_partial_rope_decode on cos / sin [1, 1, R * 32, rope_dim] rotates the [1, R, heads, rope_dim] slice viewed as
  [1, 1, R * 32, rope_dim] with one prefill-mode rotary_embedding_hf: no transpose, no copy, no cos / sin reshape; the
  other cos / sin layouts keep the transposed path.
* The verify attention (TPAttention._forward_verify) runs the row view for q and k.
* Qwen36Model.rope_lookup(row_view=True) repeats each looked-up row over its tile with one matmul per table against a
  0/1 matrix built once per row count; without row_view it is unchanged.
* In torch, with the tiles' padding rows modelled, the row view hands every valid element the operands of the
  transposed path (cos / sin row of its own row, the same rotate-half partner), so the element-wise op gives the same
  bytes.
The device comparison is in test_rope_row_view.py.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_verify_attention_glue_host import (
    DIM,
    FT,
    HD,
    NH,
    NKV,
    RD,
    RecordingAllReduce,
    _attention,
    _calls,
    ops,
)
from models.demos.blackhole.qwen36.tt.attention import rope_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

__all__ = ["ops"]  # the op recorder fixture, shared with test_verify_attention_glue_host


@pytest.fixture
def rope_ops(ops, monkeypatch):
    """``ops`` with the reshape's padded shape and the rotary op's keywords recorded."""

    def reshape(t, shape, padded=None, memory_config=None):
        out = FT(shape, f"view({t.name})", t.mem)
        ops.append(("reshape", (t, tuple(shape)), {"padded": None if padded is None else tuple(padded)}, out))
        return out

    def rope(t, cos, sin, **kw):
        out = FT(t.shape, "roped", kw.get("memory_config"))
        ops.append(("rope", (t, cos, sin), kw, out))
        return out

    monkeypatch.setattr(ttnn, "reshape", reshape)
    monkeypatch.setattr(ttnn.experimental, "rotary_embedding_hf", rope)
    return ops


@pytest.mark.parametrize("R", [4, 32])
@pytest.mark.parametrize("heads", [NH, NKV])
def test_the_row_view_rotates_the_slice_viewed_as_rows_without_transposes(rope_ops, R, heads):
    x = FT((1, R, heads, HD), "q", "L1")
    cos, sin = FT((1, 1, R * 32, RD), "cos"), FT((1, 1, R * 32, RD), "sin")
    out = rope_tp.apply_partial_rope_decode(x, cos, sin, heads, R, RD)

    assert _calls(rope_ops, "transpose") == [] and _calls(rope_ops, "to_memory_config") == []
    first, second = _calls(rope_ops, "slice")
    assert first[1][0] is x and (first[1][1], first[1][2]) == ((0, 0, 0, 0), (1, R, heads, RD))
    assert second[1][0] is x and (second[1][1], second[1][2]) == ((0, 0, 0, RD), (1, R, heads, HD))
    assert second[2]["mem"] == "DRAM"
    view_in, view_out = _calls(rope_ops, "reshape")
    # into rows: the slice re-viewed as R tiles of 32 rows (logical = padded, so the op sees whole tiles)
    assert view_in[1][0] is first[3] and view_in[1][1] == view_in[2]["padded"] == (1, 1, R * 32, RD)
    (rope,) = _calls(rope_ops, "rope")
    assert rope[1] == (view_in[3], cos, sin)
    assert rope[2] == {"is_decode_mode": False, "memory_config": "DRAM"}
    # back to heads: the valid rows of each tile, padded to one tile per row
    assert view_out[1][0] is rope[3] and view_out[1][1] == (1, R, heads, RD)
    assert view_out[2]["padded"] == (1, R, 32, RD)
    (cat,) = _calls(rope_ops, "concat")
    assert cat[1] == (view_out[3], second[3]) and out is cat[3] and out.shape == (1, R, heads, HD)
    freed = [c[1][0] for c in rope_ops if c[0] == "deallocate"]
    assert freed == [view_in[3], view_out[3], second[3]]


@pytest.mark.parametrize("shape", [(1, 1, 8, RD), (1, 8, 1, RD), (1, 1, 8 * 16, RD)])
def test_other_cos_sin_layouts_keep_the_transposed_rope(rope_ops, shape):
    rope_tp.apply_partial_rope_decode(FT((1, 8, NH, HD), "q", "L1"), FT(shape, "cos"), FT(shape, "sin"), NH, 8, RD)
    assert len(_calls(rope_ops, "transpose")) == 2
    (rope,) = _calls(rope_ops, "rope")
    assert rope[1][0].shape == (1, NH, 8, RD)


@pytest.mark.parametrize("n_users, T", [(8, 4), (1, 4), (2, 12)])
def test_the_verify_attention_rotates_q_and_k_in_the_row_view(rope_ops, n_users, T):
    a = _attention(n_users)
    R = n_users * T
    groups = a._spec_sdpa_plan(T, n_users)[1]
    cos, sin = FT((1, 1, R * 32, RD), "cos"), FT((1, 1, R * 32, RD), "sin")
    out = a.forward_decode(
        FT((1, 1, R, DIM), "x"),
        FT((R,), "cur_pos"),
        cos,
        sin,
        page_table=FT((R, 16), "pt_rows"),
        alias_kv_write=True,
        spec_verify_mode=True,
        spec_page_table=FT((groups, 16), "pt_groups"),
        spec_n_users=n_users,
        spec_user_page_table=FT((n_users, 16), "pt_users"),
        decode_ar=RecordingAllReduce(),
    )
    assert out == "replicated"
    assert _calls(rope_ops, "transpose") == []
    ropes = _calls(rope_ops, "rope")
    assert [(r[1][0].shape, r[1][1], r[1][2]) for r in ropes] == [((1, 1, R * 32, RD), cos, sin)] * 2
    assert not any(c[1][0] in (cos, sin) for c in _calls(rope_ops, "reshape"))


def test_the_repeat_matrix_selects_each_row_for_its_tile_rows():
    for rows in (1, 4, 32):
        m = rope_tp.rope_row_repeat_matrix(rows)
        assert m.dtype == torch.bfloat16 and tuple(m.shape) == (rows * 32, rows)
        assert torch.equal(m.sum(dim=1), torch.ones(rows * 32, dtype=torch.bfloat16))
        x = torch.randn(rows, RD).to(torch.bfloat16)
        assert torch.equal((m.float() @ x.float()).to(torch.bfloat16), x.repeat_interleave(32, dim=0))


def _lookup_model(monkeypatch, log):
    monkeypatch.setattr(
        ttnn, "embedding", lambda idx, table, **kw: log.append(("embedding", table, kw)) or FT((1, 8, RD), f"e{table}")
    )
    monkeypatch.setattr(ttnn, "reshape", lambda t, s: log.append(("reshape", t.name, tuple(s))) or FT(s, t.name))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("deallocate", t.name)))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda dev: "replicate")
    monkeypatch.setattr(ttnn, "DRAM_MEMORY_CONFIG", "DRAM")

    def from_torch(t, **kw):
        log.append(("from_torch", tuple(t.shape), t.clone(), kw))
        return FT(tuple(t.shape), "repeat_mat")

    def matmul(a, b, memory_config=None, compute_kernel_config=None):
        log.append(("matmul", a.name, b.name, memory_config, compute_kernel_config))
        return FT((1, 1, a.shape[-2], b.shape[-1]), f"rows({b.name})")

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "matmul", matmul)
    model = object.__new__(Qwen36Model)
    model.args = SimpleNamespace(rope_head_dim=RD)
    model.device = "mesh"
    model.rope_rm_tables = lambda: ("cos_table", "sin_table")
    return model


def test_rope_lookup_without_the_row_view_is_unchanged(monkeypatch):
    log = []
    model = _lookup_model(monkeypatch, log)
    cos, sin = model.rope_lookup("idx", 8)
    assert (cos.name, cos.shape, sin.name, sin.shape) == ("ecos_table", (1, 1, 8, RD), "esin_table", (1, 1, 8, RD))
    assert [c[0] for c in log] == ["embedding", "reshape", "embedding", "reshape"]


@pytest.mark.parametrize("n", [4, 32])
def test_rope_lookup_row_view_repeats_each_row_with_one_matmul_per_table(monkeypatch, n):
    log = []
    model = _lookup_model(monkeypatch, log)
    cos, sin = model.rope_lookup("idx", n, row_view=True)
    assert (cos.name, cos.shape) == ("rows(ecos_table)", (1, 1, n * 32, RD))
    assert (sin.name, sin.shape) == ("rows(esin_table)", (1, 1, n * 32, RD))
    (built,) = [c for c in log if c[0] == "from_torch"]
    assert built[1] == (1, 1, n * 32, n)
    assert torch.equal(built[2].reshape(n * 32, n), rope_tp.rope_row_repeat_matrix(n))
    assert built[3]["layout"] == ttnn.TILE_LAYOUT and built[3]["device"] == "mesh"
    assert built[3]["mesh_mapper"] == "replicate" and built[3]["memory_config"] == "DRAM"
    mms = [c for c in log if c[0] == "matmul"]
    assert [(m[1], m[2], m[3]) for m in mms] == [
        ("repeat_mat", "ecos_table", "DRAM"),
        ("repeat_mat", "esin_table", "DRAM"),
    ]
    assert all(m[4] is rope_tp._ROW_REPEAT_CFG for m in mms)
    assert [c[1] for c in log if c[0] == "deallocate"] == ["ecos_table", "esin_table"]
    # The matrix is built once per row count and kept: a capture after the warm pass writes nothing.
    log.clear()
    model.rope_lookup("idx", n, row_view=True)
    assert not any(c[0] == "from_torch" for c in log)
    model.rope_lookup("idx", 2 * n, row_view=True)
    assert [c[1] for c in log if c[0] == "from_torch"] == [(1, 1, 2 * n * 32, 2 * n)]


def _bf16(t):
    return t.to(torch.bfloat16)


def _rotary_hf(x, cos, sin):
    """rotary_embedding_hf, element by element in bf16: x * cos + rotate_half(x) * sin, each product rounded to bf16 and
    the sum rounded to bf16 (the op's bf16 intermediate buffers). Same-shape operands, last dim the rotary width."""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return _bf16(_bf16(x.float() * cos.float()).float() + _bf16(rot.float() * sin.float()).float())


@pytest.mark.parametrize("R, heads", [(4, NH), (32, NH), (4, NKV), (32, NKV)])
def test_the_row_view_gives_each_valid_element_the_operands_of_the_transposed_path(R, heads):
    g = torch.Generator().manual_seed(R * 10 + heads)
    x = _bf16(torch.randn(1, R, heads, RD, generator=g))
    cos = _bf16(torch.rand(R, RD, generator=g) * 2 - 1)
    sin = _bf16(torch.rand(R, RD, generator=g) * 2 - 1)

    # Transposed path: [1, heads, R, rd], cos / sin [1, 1, R, rd] broadcast over the heads, then back.
    transposed = _rotary_hf(x.transpose(1, 2), cos.reshape(1, 1, R, RD), sin.reshape(1, 1, R, RD)).transpose(1, 2)

    # Row view: one tile of 32 rows per row of x (rows past `heads` hold padding), viewed as R * 32 rows; cos / sin
    # row r repeated over tile rows r * 32 .. r * 32 + 31 by the 0/1 matrix.
    tiles = torch.full((1, R, 32, RD), float("nan"), dtype=torch.bfloat16)
    tiles[:, :, :heads] = x
    m = rope_tp.rope_row_repeat_matrix(R).float()
    cos_rows, sin_rows = _bf16(m @ cos.float()), _bf16(m @ sin.float())
    rows = _rotary_hf(tiles.reshape(1, 1, R * 32, RD), cos_rows, sin_rows)
    row_view = rows.reshape(1, R, 32, RD)[:, :, :heads]

    assert torch.equal(row_view.view(torch.int16), transposed.view(torch.int16))
