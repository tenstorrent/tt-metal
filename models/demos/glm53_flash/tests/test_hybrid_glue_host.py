# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the hybrid decode glue work (2026-10-26, _agents/hybrid_glue/STATUS.md): no device is opened
(``import ttnn`` only; ``pytest --noconftest``).

* the ``GLM53_STREAM_EXPERTS`` default rule (unset / 0 / 1 x dense 0 / 1) in ``stream_experts_default``, in
  ``TtMoE.expert_cache_complete`` (a warm block-op cache is not a stream one: the layout is in the file names) and the
  ``TtRoutedExperts`` precedence (``stream`` and ``dense`` are alternatives);
* the H1 / H2 knobs (``GLM53_HYBRID_GATHER`` / ``GLM53_HYBRID_COMBINE``): plumbing, the decode-shape gate
  (``decode_glue_modes``: T < 32, no top-k lists, not inside a prefill scope), the per-row selector constant and the
  layout goldens: the compact gather holds the padded gather's logical rows, the all-reduce + selector combine equals
  the padded reduce-scatter in fp32, and the bf16 rounding-order analysis (the same 8 partials summed in two ring
  orders stay within a few ulps);
* ``GLMMesh.all_reduce_rows_rs_ag`` (the chunk rule, the op pair it issues) and the C2 input knob;
* B8a: ``dense_regime`` / ``SparseDecodeMeta.regime`` / the regime guard of ``update`` (a dense-skip graph refuses to be
  replayed past 2050), the graph order of ``_decode_unified`` (buffers first, the dense-skip branch before the index
  query), the attribute contract ``tt/spec_decode.py`` relies on;
* the roofline rows behind the STATUS table.

    pytest --noconftest models/demos/glm53_flash/tests/test_hybrid_glue_host.py -q
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.tt import dsa as D  # noqa: E402
from models.demos.glm53_flash.tt import mesh as MESH  # noqa: E402
from models.demos.glm53_flash.tt import moe as M  # noqa: E402
from models.demos.glm53_flash.tt import roofline as rl  # noqa: E402

H = 4096
MD = Path(__file__).resolve().parents[1]


class _Mesh:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols


class _Cfg:
    index_topk, index_kpool, index_select_pools = 2048, 4, 512


class _Shape:
    def __init__(self, dims):
        self._d = list(dims)

    def __getitem__(self, i):
        return self._d[i]

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)


class _Tensor:
    def __init__(self, dims):
        self.shape = _Shape(dims)

    def memory_config(self):
        return "mc"


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _bf16_ulp(x: torch.Tensor) -> float:
    """One bf16 ulp at the magnitude of ``x``'s largest element (8 significant bits)."""
    m = float(x.abs().max())
    return (2.0 ** torch.tensor(m).log2().floor()).item() * 2.0**-7 if m > 0 else 2.0**-133


# ------------------------------------------------------------------------------------ the stream default rule
def test_stream_experts_default_rule():
    for env, dense, want in [
        ({}, False, True),  # unset, sparse layout -> the stream op (the 2026-10-26 default)
        ({}, True, False),  # unset, dense layout -> dense keeps its path
        ({"GLM53_STREAM_EXPERTS": "0"}, False, False),  # explicit 0 -> the sparse block op
        ({"GLM53_STREAM_EXPERTS": "0"}, True, False),
        ({"GLM53_STREAM_EXPERTS": "1"}, False, True),
        ({"GLM53_STREAM_EXPERTS": "1"}, True, True),  # explicit 1 next to dense: the constructor's assert, as before
        ({"GLM53_STREAM_EXPERTS": ""}, False, False),  # any other value = not "1"
    ]:
        assert M.stream_experts_default(dense, env) is want, (env, dense)
    assert M.STREAM_EXPERTS_ENV == "GLM53_STREAM_EXPERTS"


def test_stream_default_in_cache_complete(monkeypatch, tmp_path):
    """A warm block-op cache (no ``_stream`` tag) must not pass for the stream default; the stream files pass."""
    mesh = _Mesh(8, 4)
    monkeypatch.delenv("GLM53_STREAM_EXPERTS", raising=False)
    monkeypatch.delenv("GLM53_DENSE_EXPERTS", raising=False)
    monkeypatch.delenv("GLM53_MOE_LAYOUT", raising=False)
    monkeypatch.delenv(M.W13_DTYPE_ENV, raising=False)
    block = M.TtRoutedExperts.cache_files(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b, False, False)
    stream = M.TtRoutedExperts.cache_files(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b, False, True)
    dense = M.TtRoutedExperts.cache_files(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b, True, False)
    assert set(block) == {"w13", "w2"} and set(stream) == {"w13", "w2"} and set(dense) == {"w13", "w2", "w13_up"}
    assert all("_stream" in p.name for p in stream.values()) and not any("_stream" in p.name for p in block.values())
    assert all("_dense" in p.name for p in dense.values())
    for p in block.values():
        p.write_bytes(b"x")  # a warm block-op cache
    assert not M.TtMoE.expert_cache_complete(tmp_path, mesh), "the stream default must not accept the block-op files"
    monkeypatch.setenv("GLM53_STREAM_EXPERTS", "0")
    assert M.TtMoE.expert_cache_complete(tmp_path, mesh), "GLM53_STREAM_EXPERTS=0 selects the block-op files"
    monkeypatch.delenv("GLM53_STREAM_EXPERTS")
    for p in stream.values():
        p.write_bytes(b"x")
    assert M.TtMoE.expert_cache_complete(tmp_path, mesh), "the stream files complete the default"
    assert M.TtMoE.expert_cache_complete(tmp_path, mesh, stream_experts=True)
    assert M.TtMoE.expert_cache_complete(tmp_path, mesh, stream_experts=False)
    # the dense layout keeps its own files under the unset rule
    monkeypatch.setenv("GLM53_DENSE_EXPERTS", "1")
    assert not M.TtMoE.expert_cache_complete(tmp_path, mesh)
    for p in dense.values():
        p.write_bytes(b"x")
    assert M.TtMoE.expert_cache_complete(tmp_path, mesh)


def test_prefill_experts_resolution_table():
    """stream x dense x GLM53_PREFILL_EXPERTS in {unset(auto), blocks, regions}: only dense forces blocks; the stream
    layout follows the knob like the block layout (the 2026-10-26 fix: a 64 K prefill through 32-token stream blocks
    took 139-144 s against 25.4 s on regions). Composed with the default flip: unset GLM53_STREAM_EXPERTS -> stream."""
    table = {}
    for mode in (None, "blocks", "regions", "auto", "dispatch"):
        env = {} if mode is None else {"GLM53_PREFILL_EXPERTS": mode}
        for dense in (False, True):
            for stream in (False, True):
                table[(mode, dense, stream)] = M.prefill_experts_for(dense, stream, env)
    for mode in (None, "auto"):
        assert all(
            table[(mode, d, s)] == "blocks" for d in (False, True) for s in (False, True)
        )  # bare layer: auto -> blocks
    for mode in ("blocks", "regions", "dispatch"):
        assert table[(mode, False, False)] == mode and table[(mode, False, True)] == mode  # block and stream alike
        assert table[(mode, True, False)] == "blocks" and table[(mode, True, True)] == "blocks"  # dense forces blocks
    # composed with the default flip: unset GLM53_STREAM_EXPERTS -> stream -> regions when the planner asked for it
    env = {"GLM53_PREFILL_EXPERTS": "regions", "GLM53_REGIONS_KEEP_STACKED": "1"}
    stream = M.stream_experts_default(False, env)
    assert stream and M.prefill_experts_for(False, stream, env) == "regions"
    assert M.STREAM_DISPATCH_MAX_T == 256
    src = inspect.getsource(M.TtRoutedExperts.__init__)
    # (2026-10-27: resolved once, before the stream branch, which skips the stacked upload in the one-copy configuration)
    assert "prefill_mode = prefill_experts_for(dense, stream)" in src and "self.prefill_experts = prefill_mode" in src
    assert 'if not (dense or stream) else "blocks"' not in src


def test_stream_regions_cache_and_source(monkeypatch, tmp_path):
    """stream + regions: the constructor slices the regions lists out of the BLOCK layout (loaded from the block-op
    cache, then released), so ``cache_complete`` needs both file sets; stream + blocks needs the stream files only."""
    mesh = _Mesh(8, 4)
    monkeypatch.delenv("GLM53_STREAM_EXPERTS", raising=False)
    monkeypatch.delenv("GLM53_DENSE_EXPERTS", raising=False)
    monkeypatch.delenv(M.W13_DTYPE_ENV, raising=False)
    block = M.TtRoutedExperts.cache_files(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b, False, False)
    stream = M.TtRoutedExperts.cache_files(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b, False, True)
    for p in stream.values():
        p.write_bytes(b"x")
    monkeypatch.setenv("GLM53_PREFILL_EXPERTS", "regions")
    assert not M.TtMoE.expert_cache_complete(tmp_path, mesh), "regions lists need the block layout files"
    assert M.TtRoutedExperts.cache_complete(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b, False, True, "blocks")
    for p in block.values():
        p.write_bytes(b"x")
    assert M.TtMoE.expert_cache_complete(tmp_path, mesh)
    monkeypatch.setenv("GLM53_PREFILL_EXPERTS", "blocks")
    assert M.TtMoE.expert_cache_complete(tmp_path, mesh)
    src = inspect.getsource(M.TtRoutedExperts._build_dispatch_path)
    assert "self._block_source is not None" in src and "ttnn.deallocate(w13_src)" in src
    assert "make_w13, self.expert_dtype, files_block" in src and "expect_shape=(1, E, H, 2 * self.I)" in src


def test_stream_serves_and_list_gathers(monkeypatch):
    """The stream family serves <= 32 gathered tokens (block op) and 33..256 (dispatch ops) when its tensors are
    present; the regions lists serve the rest, and the router's top-k lists are gathered only for them.
    GLM53_REGIONS_DECODE (2026-10-27): ``stream`` lets the family serve from the regions lists without the stacked
    copy, ``regions`` sends every shape to the regions kernel when the lists exist."""

    class _RE:
        stream, w13, stream_dispatch = True, object(), True
        _dispatch, _regions_src = None, None  # no regions lists: the knob cannot redirect
        _regions_serves = M.TtRoutedExperts._regions_serves

    monkeypatch.delenv("GLM53_REGIONS_DECODE", raising=False)
    serves = M.TtRoutedExperts._stream_serves
    assert serves(_RE(), 32) and serves(_RE(), 64) and serves(_RE(), 256)
    assert not serves(_RE(), 288) and not serves(_RE(), 48) and not serves(_RE(), 4096)
    dropped = _RE()
    dropped.w13 = None  # GLM53_REGIONS_KEEP_STACKED=0: the regions lists serve decode too
    assert not serves(dropped, 32)
    blk = _RE()
    blk.stream = False
    assert not serves(blk, 32)
    one_copy = _RE()
    one_copy.w13, one_copy._dispatch, one_copy._regions_src = None, object(), object()
    monkeypatch.setenv("GLM53_REGIONS_DECODE", "off")
    assert not serves(one_copy, 32)  # off: the earlier fallback
    monkeypatch.delenv("GLM53_REGIONS_DECODE")
    assert serves(one_copy, 32)  # the default (stream) since 2026-10-27
    monkeypatch.setenv("GLM53_REGIONS_DECODE", "stream")
    assert serves(one_copy, 32) and serves(one_copy, 256) and not serves(one_copy, 288)
    assert not serves(dropped, 32)  # stream without the lists (not bank-local / no regions path): no redirect
    both = _RE()
    both._dispatch, both._regions_src = object(), object()
    monkeypatch.setenv("GLM53_REGIONS_DECODE", "regions")
    assert not serves(both, 32) and not serves(both, 64)  # the A/B knob forces the regions kernel
    assert serves(_RE(), 32)  # ... only when the lists exist
    monkeypatch.delenv("GLM53_REGIONS_DECODE", raising=False)
    fwd = inspect.getsource(M.TtRoutedExperts.forward)
    assert "self._stream_serves(self.mesh.rows * T)" in fwd and "idx = w = None" in fwd
    loc = inspect.getsource(M.TtRoutedExperts._local_experts)
    assert loc.index("self._stream_serves(T) and T > TILE") < loc.index("self._dispatch is not None and (T > TILE")


def test_routed_experts_precedence_source():
    """``TtRoutedExperts`` refuses ``dense and stream`` and builds the stream path first (``if stream ... elif dense``);
    ``TtMoE.__init__`` / ``expert_cache_complete`` both resolve the knob through ``stream_experts_default``."""
    src = inspect.getsource(M.TtRoutedExperts.__init__)
    assert 'assert not (dense and stream), "dense and stream are alternatives"' in src
    assert src.index("if stream:") < src.index("elif dense:")
    assert "stream_experts_default(dense_experts)" in inspect.getsource(M.TtMoE.__init__)
    assert "stream_experts_default(dense_experts)" in inspect.getsource(M.TtMoE.expert_cache_complete)
    assert 'os.environ.get("GLM53_STREAM_EXPERTS", "0")' not in inspect.getsource(M)


# ------------------------------------------------------------------------------------ H1 / H2 knobs
def test_hybrid_knob_plumbing():
    assert (
        M.hybrid_gather_mode({}) == "compact" and M.hybrid_combine_mode({}) == "allreduce"
    )  # the defaults since 2026-10-29
    assert (
        M.hybrid_gather_mode({"GLM53_HYBRID_GATHER": "padded"}) == "padded"
        and M.hybrid_combine_mode({"GLM53_HYBRID_COMBINE": "rs"}) == "rs"
    )
    assert M.hybrid_gather_mode({"GLM53_HYBRID_GATHER": "compact"}) == "compact"
    for v in ("allreduce", "allreduce_ccl"):
        assert M.hybrid_combine_mode({"GLM53_HYBRID_COMBINE": v}) == v
    for fn, key in ((M.hybrid_gather_mode, "GLM53_HYBRID_GATHER"), (M.hybrid_combine_mode, "GLM53_HYBRID_COMBINE")):
        try:
            fn({key: "bogus"})
            raised = False
        except ValueError:
            raised = True
        assert raised, key


def test_decode_glue_modes_gate():
    on = {"GLM53_HYBRID_GATHER": "compact", "GLM53_HYBRID_COMBINE": "allreduce"}
    assert M.decode_glue_modes(8, 4, has_idx=False, in_prefill=False, env=on) == (True, "allreduce")
    assert M.decode_glue_modes(2, 4, has_idx=False, in_prefill=False, env=on) == (True, "allreduce")
    assert M.decode_glue_modes(8, 4, has_idx=False, in_prefill=False, env={}) == (True, "allreduce")  # the defaults
    off = {"GLM53_HYBRID_GATHER": "padded", "GLM53_HYBRID_COMBINE": "rs"}
    assert M.decode_glue_modes(8, 4, has_idx=False, in_prefill=False, env=off) == (
        False,
        "rs",
    )  # the pre-2026-10-29 path
    assert M.decode_glue_modes(8, 32, has_idx=False, in_prefill=False, env=on) == (False, "rs")  # tile-aligned T
    assert M.decode_glue_modes(
        8, 32, has_idx=False, in_prefill=False, env=dict(on, GLM53_HYBRID_GLUE_FULL_TILE="1")
    ) == (
        True,
        "allreduce",
    )  # the full tile opts in (the 32-row verify probe)
    assert M.decode_glue_modes(
        8, 64, has_idx=False, in_prefill=False, env=dict(on, GLM53_HYBRID_GLUE_FULL_TILE="1")
    ) == (
        False,
        "rs",
    )
    assert M.decode_glue_modes(8, 16, has_idx=False, in_prefill=True, env=on) == (False, "rs")  # a prefill slab
    assert M.decode_glue_modes(8, 4, has_idx=True, in_prefill=False, env=on) == (False, "rs")  # dispatch routing
    assert M.decode_glue_modes(1, 4, has_idx=False, in_prefill=False, env=on) == (False, "rs")  # one row
    assert M.decode_glue_modes(
        8, 4, has_idx=False, in_prefill=False, env={"GLM53_HYBRID_COMBINE": "allreduce_ccl"}
    ) == (
        True,  # the compact gather is the default
        "allreduce_ccl",
    )


def test_row_selector_constant():
    for rows, T in ((8, 4), (2, 4), (8, 1), (2, 2), (8, 3)):
        Tg = rows * T
        sel = M.row_selector_host(rows, T, Tg)
        assert tuple(sel.shape) == (rows, 1, T, Tg)
        assert float(sel.sum()) == rows * T  # one 1 per (row, user)
        total = torch.randn(Tg, 16)
        for r in range(rows):
            assert torch.equal(sel[r, 0] @ total, total[r * T : (r + 1) * T])
    # the 8x4 at 4 users per row: E_r picks rows 4r .. 4r+3 of the 32 gathered rows
    sel = M.row_selector_host(8, 4, 32)
    assert sel[5, 0, 2, 22] == 1 and sel[5, 0].sum() == 4


def test_hybrid_glue_goldens():
    """The compact gather = the padded gather's logical rows; all-reduce + selector == padded reduce-scatter in fp32."""
    rows, T, Hd = 8, 4, 64
    g = torch.Generator().manual_seed(3)
    x_rows = torch.randn(rows, T, Hd, generator=g).bfloat16()
    partials = torch.randn(rows, rows * T, Hd, generator=g).bfloat16()
    gathered, padded_rs, allreduce_sel = M.hybrid_glue_host(x_rows, partials, rows, T)
    # the padded gather: each row's [T, H] tile-padded to 32 rows, concatenated on the row dim, logical rows kept
    padded = torch.cat([torch.nn.functional.pad(x_rows[r], (0, 0, 0, 32 - T)) for r in range(rows)], dim=0)
    logical = torch.cat([padded[32 * r : 32 * r + T] for r in range(rows)], dim=0)
    assert torch.equal(gathered, logical)
    assert torch.equal(padded_rs, allreduce_sel)
    assert tuple(padded_rs.shape) == (rows, T, Hd)


def test_combine_rounding_order_analysis():
    """On device both combines add the same 8 bf16 partials with one bf16 rounding per ring hop; the padded
    reduce-scatter reduces row chunk r in chunk r's hop order, the hidden-dim reduce-scatter reduces hidden chunk c in
    chunk c's order. Model both as sequential bf16 sums in two different sender orders: they agree to a few ulps and
    both sit within a few ulps of the fp32 sum (the bound the device A/B asserts: <= 8 ulps, PCC > 0.9999)."""
    rows, n = 8, 4096
    g = torch.Generator().manual_seed(5)
    parts = (torch.randn(rows, n, generator=g) * 0.05).bfloat16()
    exact = parts.float().sum(0)

    def ring_sum(order):
        acc = parts[order[0]].float()
        for r in order[1:]:
            acc = (acc + parts[r].float()).bfloat16().float()  # one bf16 rounding per hop
        return acc

    a = ring_sum(list(range(rows)))  # chunk r = row chunk: senders in line order
    b = ring_sum(list(reversed(range(rows))))  # another chunk's order (the far end first)
    ulp = _bf16_ulp(exact)
    assert float((a - b).abs().max()) <= 8 * ulp
    assert float((a - exact).abs().max()) <= 8 * ulp and float((b - exact).abs().max()) <= 8 * ulp
    assert _pcc(a, b) > 0.9999 and _pcc(a, exact) > 0.9999
    assert not torch.equal(
        a, b
    )  # the orders DO differ in the last bits: the device A/B is PCC / ulps, not bit identity


def test_expert_output_is_compact_for_the_combine():
    """The stream op's block output is [1,1,32,H] (rows*T = 32 gathered tokens on the 8x4: already compact), its down
    writer zero-fills the columns of an idle core, and the row-glue mode functions are read per call (trace-safe)."""
    from models.demos.glm53_flash.custom_ops.expert_stream.op import KERNEL_DIR, ExpertStream

    src = inspect.getsource(ExpertStream.alloc)
    assert "ttnn.Shape([1, 1, TILE, self.H])" in src
    writer = (KERNEL_DIR / "writer_down.cpp").read_text()
    assert "no active expert: this core's output columns are zero" in writer
    assert "noc_async_full_barrier();" in writer
    fwd = inspect.getsource(M.TtRoutedExperts.forward)
    assert "decode_glue_modes(" in fwd and "_combine_rows_allreduce(" in fwd and "_gather_rows_compact(" in fwd
    assert "tilize_with_zero_padding" in inspect.getsource(M.TtRoutedExperts._gather_rows_compact)
    assert "compute_kernel_config=self.fp32" in inspect.getsource(M.TtRoutedExperts._combine_rows_allreduce)


# ------------------------------------------------------------------------------------ mesh.py
def test_all_reduce_rows_rs_ag_issues_rs_then_ag(monkeypatch):
    calls = []
    monkeypatch.setattr(MESH.GLMMesh, "reduce_scatter_rows", lambda self, x, dim, **kw: calls.append(("rs", dim)) or x)
    monkeypatch.setattr(MESH.GLMMesh, "all_gather_rows", lambda self, x, dim, **kw: calls.append(("ag", dim)) or x)
    monkeypatch.setattr(MESH.ttnn, "deallocate", lambda t: None)
    mesh = MESH.GLMMesh(None, 8, 4)
    x = _Tensor([1, 1, 32, H])
    assert mesh.all_reduce_rows_rs_ag(x) is x
    assert calls == [("rs", 3), ("ag", 3)]
    calls.clear()
    block = MESH.GLMMesh(None, 2, 4)
    block.all_reduce_rows_rs_ag(_Tensor([1, 1, 8, H]), dim=-1)
    assert calls == [("rs", 3), ("ag", 3)]
    # a chunk that is not tile-aligned over the rows is refused (the rs form never expands to a composite)
    try:
        MESH.GLMMesh(None, 8, 4).all_reduce_rows_rs_ag(_Tensor([1, 1, 32, 4096]), dim=2)
        raised = False
    except AssertionError:
        raised = True
    assert raised
    assert MESH.GLMMesh(None, 1, 1).all_reduce_rows_rs_ag(x) is x  # one row: identity, no ops
    assert calls == [("rs", 3), ("ag", 3)]


def test_tp_allreduce_input_knob(monkeypatch):
    mesh = MESH.GLMMesh(None, 8, 4)
    monkeypatch.delenv("GLM53_TP_ALLREDUCE_INPUT", raising=False)
    assert mesh.tp_allreduce_input_l1 is False
    monkeypatch.setenv("GLM53_TP_ALLREDUCE_INPUT", "l1")
    assert mesh.tp_allreduce_input_l1 is True
    with mesh.prefill_scope():
        assert mesh.tp_allreduce_input_l1 is False  # prefill slabs keep today's path
    assert mesh.tp_allreduce_input_l1 is True
    assert "to_memory_config(x, ttnn.L1_MEMORY_CONFIG)" in inspect.getsource(MESH.GLMMesh.all_reduce_cols)


def test_c2_call_sites_hand_the_same_placement():
    """Every decode call site of ``all_reduce_cols`` passes a DecodeLinear (``ttnn.linear``) output or an ``add`` of
    two: DRAM interleaved bf16 TILE [1,1,B,H] -- the input side of the attention and MoE all-reduces is identical."""
    for rel, needle in (
        ("tt/kda.py", "y = self.lin_o(o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)"),
        ("tt/dsa.py", "y = self.lin_o(o_heads, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)"),
        ("tt/moe.py", "y = ttnn.add(routed, shared, output_tensor=routed)"),
    ):
        src = (MD / rel).read_text()
        assert needle in src, rel
        after = src[src.index(needle) :]
        assert "all_reduce_cols(y)" in after[:600], rel
    from models.demos.glm53_flash.tt.matmul_configs import DecodeLinear

    src = inspect.getsource(DecodeLinear.__call__)
    assert "memory_config" not in src  # ttnn.linear's default output: DRAM interleaved, both sites alike


# ------------------------------------------------------------------------------------ B8a dense skip
def test_pool_gather_knob_and_dense_regime():
    assert (
        D.pool_gather_mode({}) == "gather" and D.pool_gather_mode({"GLM53_POOL_GATHER": "dense_skip"}) == "dense_skip"
    )
    try:
        D.pool_gather_mode({"GLM53_POOL_GATHER": "x"})
        raised = False
    except ValueError:
        raised = True
    assert raised
    cfg = _Cfg()
    assert D.dense_exact_max_pos(cfg) == 2050
    assert D.dense_regime([0, 7, 2050], cfg) and not D.dense_regime([0, 2051], cfg) and D.dense_regime([], cfg)
    # the meta's cur_pos equals the position for every dense-range user (the paged attention bound): the two forms
    # coincide at 2047..2050 where n == K already
    valid, tail, cur = D.SparseDecodeMeta.host_values([0, 100, 2046, 2047, 2048, 2049, 2050, 2051], cfg)
    assert cur.tolist()[:7] == [0, 100, 2046, 2047, 2048, 2049, 2050]
    assert int(cur[7]) == 2047 + 0  # the sparse form: P*K - 1 + t


def test_sparse_meta_regime_guard(monkeypatch):
    cfg = _Cfg()
    meta = D.SparseDecodeMeta(
        valid_len=[None, None], tail_pool=None, cur_pos=None, host=[5, 6], users_per_chip=2, all_dense=True
    )
    assert meta.regime == "dense" and meta.graph_regime is None
    # no device: the host side of update() runs with the uploads stubbed out
    monkeypatch.setattr(D.ttnn, "copy_host_to_device_tensor", lambda host, dst: None)
    monkeypatch.setattr(D.ttnn, "from_torch", lambda t, **kw: t)
    meta.update([6, 7], cfg)
    assert meta.host == [6, 7] and meta.all_dense and meta.regime == "dense"
    meta.graph_regime = "dense"  # the last graph built was the dense-skip one (a captured trace replays it)
    try:
        meta.update([2049, 2051], cfg)
        raised = False
    except RuntimeError as e:
        raised = "re-capture" in str(e).lower() or "Re-capture" in str(e)
    assert raised, "leaving the dense regime under a dense-skip graph must be refused"
    assert meta.host == [6, 7]  # nothing was refreshed
    meta.update([2049, 2050], cfg)  # still dense: fine
    assert meta.all_dense
    meta.clear_graph_regime()
    meta.update([2049, 2051], cfg)  # an eager flow after the reset
    assert not meta.all_dense and meta.regime == "sparse"
    meta.graph_regime = "sparse"
    meta.update([3, 4], cfg)  # a sparse graph is exact in both regimes: never refused
    assert meta.all_dense


def test_decode_unified_graph_order():
    """Buffers are allocated before the regime branch (no persistent allocation after a capture), the dense-skip
    branch precedes the index query, and the paged attention uses the real cache + page table with ``meta.cur_pos``."""
    src = inspect.getsource(D.TtDSA._decode_unified)
    i_buf, i_skip, i_q = (
        src.index(s) for s in ("_ensure_sparse_buffers(", 'pool_gather_mode == "dense_skip"', "queries_and_weights(")
    )
    assert i_buf < i_skip < i_q
    paged = inspect.getsource(D.TtDSA._attend_paged)
    assert (
        "cache.cache" in paged and "page_table_tensor=cache.page_table" in paged and "cur_pos_tensor=cur_pos" in paged
    )
    gather = inspect.getsource(D.TtDSA._gather_and_attend)
    assert "self._compact_buffer(B)" in gather and "cur_pos_tensor=meta.cur_pos" in gather
    assert "self.pool_gather_mode = pool_gather_mode()" in inspect.getsource(D.TtDSA.__init__)


def test_spec_decode_attribute_contract():
    """The names tt/spec_decode.py reads off TtDSA stay (DsaSpecState: lin_stem, stem_bounds, qa_norm, kva_norm, _q,
    _absorb_q, _o_lat_to_heads, _o_proj, _decode_sdpa_config, sdpa_compute, scale, L; mesh.all_reduce_cols)."""
    init = inspect.getsource(D.TtDSA.__init__)
    for name in ("lin_stem", "stem_bounds", "qa_norm", "kva_norm", "sdpa_compute", "scale", "L"):
        assert f"self.{name} =" in init, name
    for name in ("_q", "_absorb_q", "_o_lat_to_heads", "_o_proj", "_decode_sdpa_config"):
        assert callable(getattr(D.TtDSA, name)), name
    assert callable(MESH.GLMMesh.all_reduce_cols)
    spec = (MD / "tt/spec_decode.py").read_text()
    for name in ("lin_stem", "stem_bounds", "qa_norm", "kva_norm", "_absorb_q", "_o_lat_to_heads", "_o_proj"):
        assert f"attn.{name}" in spec, name


# ------------------------------------------------------------------------------------ the roofline rows of the table
def test_roofline_rows_behind_the_status_table():
    cfg = rl.load_config()
    want = {
        4: {
            "moe.gather_x": (175.2, 35.0),  # measured 2026-10-29 (today_8x4_dsa3 #448-458)
            "moe.gather_routing": (67.4, 15.0),
            "moe.rs_rows": (269.4, 50.0),
            "dsa.pool_gather": (136.4, 0.0),
        },
        32: {"moe.gather_x": (101.7, 101.7), "moe.rs_rows": (128.5, 128.5), "dsa.pool_gather": (1270.0, 0.0)},
    }
    for upr, rows in want.items():
        lay = rl.Layout(users_per_row=upr, expert_path="stream" if upr < 32 else "block")
        got = {r.key: r for r in rl.layer_rows(cfg, lay, "DSA+MoE")}
        for key, (measured, after) in rows.items():
            assert got[key].measured_us == pytest.approx(measured, rel=0.01), (upr, key)
            assert got[key].after_us == pytest.approx(after, abs=0.01), (upr, key)
        if upr == 4:
            assert got["moe.gather_x"].n_ops == 11 and got["moe.rs_rows"].n_ops == 32  # the measured op counts
            assert got["kda.all_reduce" if "kda.all_reduce" in got else "dsa.all_reduce"].after_us == 30.0
