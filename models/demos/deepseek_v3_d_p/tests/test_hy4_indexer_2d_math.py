# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the DSA indexer on the 2D tensor-parallel layout of the Hy4 SC1 unit (#65887, ``tt/mla/indexer_2d.py``).

* Emulating every chip of an ``sp x tp`` mesh with the SAME host layout the device loads
  (``TtIndexer2D.host_weights_2d`` + ``chip_shard``, what ``ShardTensor2dMesh`` puts on each chip) and the module's
  collectives in torch (all-to-alls over SP then TP for q; reduce-scatters for k and the head weights) gives each chip
  exactly what the shared RoPE / Hadamard / ring-score code takes: q ``[H, S / (sp * tp), D]`` with every head in
  checkpoint order for the SP-major query block ``s * tp + t``, the index key ``[S / sp, D]`` for SP block s, and the
  head weights for the same query block; and the top-k they score equals the unsharded indexer's;
* at Hy4's real dims on the SC1 mesh (SP 8 on axis 0 x TP 4 on axis 1) every indexer weight is sharded once over all 32
  chips with the per-chip shapes of the module docstring; only the two k-norm vectors are replicated;
* the cache: names come from the layout and never match TtIndexer's TP-only files, a layout stamp guards the shards,
  placeholders are never cached, and a kv_only build never vouches for the indexer.
"""

from __future__ import annotations

import types

import pytest
import torch

from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer
from models.demos.deepseek_v3_d_p.tt.mla.indexer_2d import TtIndexer2D
from models.demos.deepseek_v3_d_p.tt.mla.mla_2d import ttMLA2DTP

F32 = torch.float32


def _config(H=8, D=64, Q=128, hidden=512, **extra):
    return types.SimpleNamespace(
        index_n_heads=H, index_head_dim=D, q_lora_rank=Q, hidden_size=hidden, index_topk=16, **extra
    )


HY4 = _config(H=32, D=128, Q=2048, hidden=6144)
HY4_SP, HY4_TP = 8, 4


def _idx_host(cfg, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) / s[-1] ** 0.5  # noqa: E731
    H, D = cfg.index_n_heads, cfg.index_head_dim
    return {
        "indexer.wq_b": r(H * D, cfg.q_lora_rank),
        "indexer.wk": r(D, cfg.hidden_size),
        "indexer.weights_proj": r(H, cfg.hidden_size),
        "indexer.k_norm": 1 + 0.1 * torch.randn(D, generator=g),
        "indexer.k_norm_bias": 0.1 * torch.randn(D, generator=g),
    }


def _all_to_all(chips: dict, sp: int, tp: int, axis: str) -> dict:
    """``all_to_all_async_generic(in_dim=1 (heads), out_dim=2 (sequence))`` over ``axis``: chip (s, t) gathers the heads
    of its axis peers, in axis order, for its own sequence block of theirs."""
    n = sp if axis == "sp" else tp
    out = {}
    for (s, t), _ in chips.items():
        peers = [(p, t) for p in range(sp)] if axis == "sp" else [(s, p) for p in range(tp)]
        mine = s if axis == "sp" else t
        out[s, t] = torch.cat([chips[c].chunk(n, dim=1)[mine] for c in peers], dim=0)
    return out


def _emulate_2d(hn: torch.Tensor, qr: torch.Tensor, idx_host: dict, cfg, sp: int, tp: int, parts: int = 1):
    """The 2D indexer's projections, chip by chip. ``hn`` [S, hidden] is the layer input (the residual, gathered over
    SP: chip (s, t) holds hn[:, H_t]); ``qr`` [S, q_lora] the normalized q_a row (every chip). Returns per chip
    (q [H, S / (sp * tp), D], k [S / sp, D] pre-LayerNorm, head weights [S / (sp * tp), H], scale folded in)."""
    host = TtIndexer2D.host_weights_2d(idx_host, cfg, sp, tp)
    W = lambda name, s, t: TtIndexer2D.chip_shard(*host[name], sp, tp, s, t)[0, 0].to(F32)  # noqa: E731
    S, H, D = hn.shape[0], cfg.index_n_heads, cfg.index_head_dim
    hpc = H // (sp * tp)
    chips = [(s, t) for s in range(sp) for t in range(tp)]
    x = {t: hn.chunk(tp, dim=1)[t] for t in range(tp)}  # the SP all-gather of the residual
    x_rc = {(s, t): x[t].chunk(sp, dim=1)[s] for s, t in chips}  # mesh_partition over SP
    scale = H**-0.5 * D**-0.5

    # q: this chip's head(s), every token -> all-to-all over SP, then over TP (heads in, sequence out).
    # Moved as `parts` pieces per head (TtIndexer2D._index_q_heads: Hy4 moves 2 x 64, the MLA q_rope geometry), then
    # each head's pieces are put back side by side.
    width = D // parts
    q0 = {(s, t): (qr @ W("wq_b", s, t)).reshape(S, hpc * parts, width).transpose(0, 1) for s, t in chips}
    q = _all_to_all(_all_to_all(q0, sp, tp, "sp"), sp, tp, "tp")
    q = {c: v.reshape(H, parts, v.shape[1], width).permute(0, 2, 1, 3).reshape(H, v.shape[1], D) for c, v in q.items()}
    # k: reduce-scatter over SP (sequence), then over TP (cols) + all-gather over TP = a sum over TP.
    k_part = {c: x_rc[c] @ W("wk", *c) for c in chips}
    k_sp = {(s, t): sum(k_part[p, t] for p in range(sp)).chunk(sp, dim=0)[s] for s, t in chips}
    k = {(s, t): sum(k_sp[s, p] for p in range(tp)) for s, t in chips}
    # head weights: reduce-scatter over SP (sequence), then over TP (sequence).
    w_part = {c: x_rc[c] @ W("weights_proj", *c) * scale for c in chips}
    w_sp = {(s, t): sum(w_part[p, t] for p in range(sp)).chunk(sp, dim=0)[s] for s, t in chips}
    w = {(s, t): sum(w_sp[s, p] for p in range(tp)).chunk(tp, dim=0)[t] for s, t in chips}
    return q, k, w


def _scores(q, k, w):
    """DSA logits (no RoPE / Hadamard: those are the shared code, the same on both layouts): ``q`` [n, H, D] for query
    rows ``rows``, ``k`` [S, D], ``w`` [n, H] -> [n, S]."""
    return torch.einsum("nh,nhs->ns", w, torch.relu(torch.einsum("nhd,sd->nhs", q, k)))


@pytest.mark.parametrize("parts", [1, 2])
@pytest.mark.parametrize("sp,tp", [(2, 4), (4, 2), (2, 2), (8, 1), (1, 4)])
def test_2d_emulation_matches_the_unsharded_indexer(sp, tp, parts):
    cfg = _config(H=sp * tp * 2 if sp * tp < 8 else 8)
    S = 32 * sp * tp
    g = torch.Generator().manual_seed(7)
    hn = torch.randn(S, cfg.hidden_size, generator=g)
    qr = torch.randn(S, cfg.q_lora_rank, generator=g)
    idx_host = _idx_host(cfg)
    H, D = cfg.index_n_heads, cfg.index_head_dim
    q_ref = (qr @ idx_host["indexer.wq_b"].T).view(S, H, D)
    k_ref = hn @ idx_host["indexer.wk"].T
    w_ref = hn @ idx_host["indexer.weights_proj"].T * H**-0.5 * D**-0.5
    q, k, w = _emulate_2d(hn, qr, idx_host, cfg, sp, tp, parts)
    rows = S // (sp * tp)
    k_full = torch.cat([k[s, 0] for s in range(sp)])
    torch.testing.assert_close(k_full, k_ref, rtol=1e-4, atol=1e-4)
    for s, t in q:
        blk = slice((s * tp + t) * rows, (s * tp + t + 1) * rows)  # SP-major query block (the score op's order)
        torch.testing.assert_close(q[s, t], q_ref[blk].transpose(0, 1), rtol=1e-4, atol=1e-4)  # all heads, in order
        torch.testing.assert_close(w[s, t], w_ref[blk], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(k[s, t], k_ref[s * (S // sp) : (s + 1) * (S // sp)], rtol=1e-4, atol=1e-4)
        # The chip's top-k over its query block equals the unsharded indexer's.
        mine = _scores(q[s, t].transpose(0, 1), k_full, w[s, t])
        ref = _scores(q_ref[blk], k_ref, w_ref[blk])
        torch.testing.assert_close(mine, ref, rtol=1e-4, atol=1e-4)
        assert torch.equal(mine.topk(cfg.index_topk).indices, ref.topk(cfg.index_topk).indices)


def test_heads_arrive_in_checkpoint_order():
    """Chip (s, t)'s heads sit at position t * sp + s of the all-to-alls' gather order (ttMLA2DTP.core_head_order with
    one head per chip), so the 32 index heads arrive in checkpoint order and weights_proj keeps its columns."""
    H, sp, tp = 32, HY4_SP, HY4_TP
    owned = {(s, t): TtIndexer2D.heads_of_chip(H, sp, tp, s, t) for s in range(sp) for t in range(tp)}
    assert sorted(h for hs in owned.values() for h in hs) == list(range(H))
    assert owned[1, 2] == [2 * sp + 1]
    gathered = [h for t in range(tp) for s in range(sp) for h in owned[s, t]]
    assert gathered == list(range(H))
    # The attention's own head order after the same two all-to-alls, at one head per chip, is the same permutation.
    order = ttMLA2DTP.core_head_order(H, sp, tp)
    # core_head_order[p] is the linear index s * tp + t of the chip whose head lands at position p.
    assert [owned[c // tp, c % tp][0] for c in order] == gathered


# Per-chip shard shapes on the SC1 mesh, as the module docstring lists them.
HY4_CHIP_SHAPES = {"wq_b": (1, 1, 2048, 128), "wk": (1, 1, 192, 128), "weights_proj": (1, 1, 192, 32)}


def test_hy4_indexer_weights_are_sharded_once_over_the_sc1_mesh():
    sp, tp = HY4_SP, HY4_TP
    TtIndexer2D.check_dims(HY4, sp, tp)
    host = TtIndexer2D.host_weights_2d(None, HY4, sp, tp)
    per_chip_bytes = 0
    for name, shape in HY4_CHIP_SHAPES.items():
        t, sp_dim, tp_dim = host[name]
        assert sp_dim is not None and tp_dim is not None, f"{name} must be split over both mesh axes"
        assert tuple(TtIndexer2D.chip_shard(t, sp_dim, tp_dim, sp, tp, 0, 0).shape) == shape, name
        assert torch.Size(shape).numel() * sp * tp == t.numel(), f"{name} is replicated"
        bytes_per = 1088 / 1024 if name != "weights_proj" else 2  # bf8 tiles (1088 B / 1024 values), bf16
        per_chip_bytes += torch.Size(shape).numel() * bytes_per
    replicated = {k for k, (_, a, b) in host.items() if a is None and b is None}
    assert replicated == {"k_norm", "k_norm_bias"}
    # ~0.31 MB of indexer weights per chip per full layer (TtIndexer on the same mesh: ~9.2 MB, wq_b replicated).
    assert 0.30e6 < per_chip_bytes < 0.32e6


def test_dims_that_do_not_tile_are_refused(expect_error):
    with expect_error(ValueError, "does not tile"):
        TtIndexer2D.check_dims(HY4, 8, 8)  # 32 heads over 64 chips
    with expect_error(ValueError, "does not tile"):
        TtIndexer2D.check_dims(_config(H=32, D=128, hidden=6144 + 32), 8, 4)  # hidden / 32 chips not whole tiles


# ---------------------------------------------------------------------------------------------- caches
def test_cache_names_come_from_the_layout_and_never_match_tp_only_files():
    assert TtIndexer2D.cache_name((8, 4), 0, 1) == "idx2d_8x4_sp8ax0_tp4ax1"
    names = {TtIndexer2D.cache_name(m, a, 1 - a) for m in ((8, 4), (4, 8), (4, 4), (2, 4)) for a in (0, 1)}
    assert len(names) == 8
    files = [f.name for f in TtIndexer2D.cache_files("/c", 0, (8, 4), 0, 1)]
    assert all(f.startswith("layer_0.idx2d_8x4_sp8ax0_tp4ax1.") for f in files) and len(files) == 5
    # TtIndexer's TP-only files are layer_{i}.mla.indexer_*: a different prefix space, and vice versa.
    assert not any(".mla.indexer_" in f for f in files)
    assert not any(f.startswith(f"layer_0.{ttMLA2DTP.cache_name((8, 4), 0, 1)}.") for f in files)


class _FakeAsTensor:
    """``ttnn.as_tensor`` without a device: records each call and, as the real one does on a miss, writes the cache file
    it was named (whatever tensor it got)."""

    def __init__(self):
        self.calls = []

    def __call__(self, t, *, dtype, layout, cache_file_name=None, **_):
        from models.demos.deepseek_v3_d_p.utils.sharded_cache import as_tensor_file

        self.calls.append(cache_file_name)
        if cache_file_name is not None:
            as_tensor_file(cache_file_name, dtype, layout).touch()
        return t


MESH = (2, 4)
SMALL = _config(H=32, D=128, Q=128, hidden=256 * 4)


def _convert_on_cpu(monkeypatch):
    import ttnn

    fake = _FakeAsTensor()
    monkeypatch.setattr(ttnn, "as_tensor", fake)
    monkeypatch.setattr(ttnn, "ShardTensor2dMesh", lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda *a, **k: None)
    mesh = types.SimpleNamespace(shape=MESH)

    def convert(tmp_path, idx_host, kv_only=False, device=None):
        return TtIndexer2D._convert_and_cache_weights(
            idx_host, mesh, SMALL, 0, 0, 1, cache_path=tmp_path, device=device, kv_only=kv_only
        )

    return convert, fake


def _complete(tmp_path) -> bool:
    return TtIndexer2D.check_cache_complete(tmp_path, "layer_0.mla", mesh_shape=MESH, sp_axis=0, tp_axis=1)


def test_cache_stamp_guards_the_shards(tmp_path, monkeypatch, expect_error):
    convert, _ = _convert_on_cpu(monkeypatch)
    convert(tmp_path, TtIndexer.extract_host_weights({f"{k}.weight": v for k, v in _idx_host(SMALL).items()}))
    guard = TtIndexer2D.cache_guard(tmp_path, 0, MESH, 0, 1)
    assert guard.stamped() and _complete(tmp_path)
    assert len(list(tmp_path.glob("*.tensorbin"))) == 5
    # TtIndexer's TP-only shards do not make a 2D indexer complete, nor the other way round.
    assert not TtIndexer.check_cache_complete(tmp_path, "layer_0.mla")
    # Another format of this module (or another frame): refused.
    monkeypatch.setattr(TtIndexer2D, "CACHE_FORMAT", TtIndexer2D.CACHE_FORMAT + 1)
    assert not _complete(tmp_path)
    with expect_error(RuntimeError, "records layout"):
        TtIndexer2D.cache_guard(tmp_path, 0, MESH, 0, 1).check()
    monkeypatch.setattr(TtIndexer2D, "CACHE_FORMAT", TtIndexer2D.CACHE_FORMAT - 1)
    monkeypatch.setattr(TtIndexer2D, "INDEX_K_FRAME", "rope_last")
    assert not _complete(tmp_path)
    monkeypatch.undo()
    # Shards with no stamp: incomplete, and a load refuses them.
    guard.stamp.unlink()
    assert not _complete(tmp_path)
    convert, _ = _convert_on_cpu(monkeypatch)
    with expect_error(RuntimeError, "no layout stamp"):
        convert(tmp_path, {})
    with expect_error(ValueError, "needs mesh_shape"):
        TtIndexer2D.check_cache_complete(tmp_path, "layer_0.mla")


def test_cache_only_load_never_caches_placeholders(tmp_path, monkeypatch, expect_error):
    convert, fake = _convert_on_cpu(monkeypatch)
    for empty in ({}, None):
        with expect_error(RuntimeError, "no source weights"):
            convert(tmp_path, empty, device=object())
    assert fake.calls == [] and list(tmp_path.iterdir()) == [] and not _complete(tmp_path)
    real = _idx_host(SMALL)
    convert(tmp_path, real)  # cache build from real weights stamps
    assert _complete(tmp_path)
    files = sorted(tmp_path.glob("*.tensorbin"))
    files[0].unlink()
    fake.calls.clear()
    with expect_error(RuntimeError, "1 shard file"):
        convert(tmp_path, {}, device=object())
    assert fake.calls == [] and not files[0].exists()
    convert(tmp_path, real)
    assert files[0].exists() and _complete(tmp_path)


def test_kv_only_build_does_not_vouch_for_the_indexer(tmp_path, monkeypatch, expect_error):
    convert, fake = _convert_on_cpu(monkeypatch)
    real = _idx_host(SMALL)
    assert convert(tmp_path, real, kv_only=True) is None  # cache-only kv_only build on a cold cache: writes nothing
    assert fake.calls == [] and list(tmp_path.iterdir()) == [] and not _complete(tmp_path)
    with expect_error(RuntimeError, "no source weights"):
        convert(tmp_path, {}, kv_only=True, device=object())
    convert(tmp_path, real)  # a full build stamps
    stamp = TtIndexer2D.cache_guard(tmp_path, 0, MESH, 0, 1).stamp.read_text()
    fake.calls.clear()
    convert(tmp_path, real, kv_only=True)  # kv_only now uses the cache, and leaves the stamp as it was
    assert len(fake.calls) == 5 and all(c is not None for c in fake.calls)
    assert TtIndexer2D.cache_guard(tmp_path, 0, MESH, 0, 1).stamp.read_text() == stamp


def test_ttmla2dtp_checks_the_2d_indexer_cache(tmp_path, monkeypatch):
    """ttMLA2DTP's completeness check asks for the 2D indexer shards on a full-indexer layer (not TtIndexer's)."""
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

    prefix = f"layer_0.{ttMLA2DTP.cache_name(MESH, 0, 1)}"
    for name in ttMLA.weight_names(True, True):
        (tmp_path / f"{prefix}.{name}_dtype_BFLOAT8_B_layout_TILE.tensorbin").touch()
    ttMLA2DTP.cache_guard(tmp_path, 0, MESH, 0, 1).write()
    init_checker(tmp_path)
    layout = dict(mesh_shape=MESH, sp_axis=0, tp_axis=1)
    check = lambda: ttMLA2DTP.check_cache_complete(tmp_path, "layer_0.mla", True, True, True, **layout)  # noqa: E731
    assert not check()
    convert, _ = _convert_on_cpu(monkeypatch)
    convert(tmp_path, _idx_host(SMALL))
    init_checker(tmp_path)
    assert check()
    assert ttMLA2DTP.INDEXER_CLASS is TtIndexer2D and ttMLA.INDEXER_CLASS is TtIndexer
