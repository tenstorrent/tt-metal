# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the MoE roofline work (2026-10-26, _agents/moe_roofline/STATUS.md): no device is opened.

* C10: ``RouterTopk.golden_rows`` (the [x | w] pages, the uint16 id rows, the bf16 index tiles with IDX_PAD) against
  the tile golden; the EP placement table applied to the ids; the writer / reader kernel ABI (compile-time arg
  indices, accessor offsets) against ``RouterTopk.__call__``'s vectors.
* C11: ``GluSplit.golden`` vs torch; the work split and core ranges on the 13x10 and 8x8 grids; the kernel ABI.
* C12: the bfp4 / bfp8 rounding of expert-like weights (a host block-float model; relative error, PCC); ``routed_w13_dtype`` and the
  cache names (dtype in the name, the layout tag, the placement tag); the placeholder guard ``_nonzero``.
* (d): the stream planner on both grids (plans are valid ``ExpertDispatch`` kwargs: L1 fits, divisibility), the
  cost model's calibration points (the 2026-10-20 A/B fit), the knob plumbing ``plan_from_env``, the kernel ABI of
  the modified expert_stream kernels (prefetch / unit-pairs / rotation arg indices, accessor offsets).
* (e): the EP32 placement planner (balanced vs contiguous on a Zipf popularity and on the real layer-4 routing of
  the T256 dump when present), the per-chip bytes model, the column-skew model, replication, the perm table.

    pytest --noconftest models/demos/glm53_flash/tests/test_moe_roofline_host.py -q
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.expert_stream import plan as P  # noqa: E402
from models.demos.glm53_flash.custom_ops.expert_stream.op import KERNEL_DIR as STREAM_KERNELS  # noqa: E402
from models.demos.glm53_flash.custom_ops.glu_split.op import KERNEL_DIR as GLU_KERNELS  # noqa: E402
from models.demos.glm53_flash.custom_ops.glu_split.op import GluSplit, core_range_set, split_work  # noqa: E402
from models.demos.glm53_flash.custom_ops.router_topk.op import IDX_PAD  # noqa: E402
from models.demos.glm53_flash.custom_ops.router_topk.op import KERNEL_DIR as ROUTER_KERNELS  # noqa: E402
from models.demos.glm53_flash.custom_ops.router_topk.op import RouterTopk, bf16_bits, rm_page_bytes  # noqa: E402
from models.demos.glm53_flash.tests.utils import pcc  # noqa: E402
from models.demos.glm53_flash.tt import moe as M  # noqa: E402

E, K_PICKS, H, I = 288, 8, 4096, 2048
GRIDS = [(13, 10), (8, 8)]  # Blackhole p150 and the forced-8x8 case (generic-kernel rule)


class _Grid:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Mesh:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols


# ---------------------------------------------------------------------------------------------- helpers
def _ct_indices(src: str) -> dict[str, str]:
    pat = re.compile(r"constexpr\s+(?:uint32_t|bool|auto)\s+(\w+)\s*=\s*get_compile_time_arg_val\(([^)]*)\)")
    return {m.group(1): m.group(2).strip() for m in pat.finditer(src)}


def _accessor_offset(src: str) -> int:
    m = re.search(r"TensorAccessorArgs<(\d+)>\(\)", src)
    assert m, "no TensorAccessorArgs<N> in the kernel"
    return int(m.group(1))


def _real_routing():
    """The layer-4 real routing of the T256 dump through the real gate (host math), or None without the dump."""
    dump = Path("bringup/references/glm53_flash/T256/layer4_prefill.pt")
    if not dump.exists():
        return None
    try:
        from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_tensors, shard_headers

        snapshot = default_snapshot_dir()
        prefix = "model.language_model.layers.4.mlp.gate."
        t = load_tensors(snapshot, [prefix + "weight", prefix + "e_score_correction_bias"], shard_headers(snapshot))
    except Exception:  # no checkpoint on this host
        return None
    x = torch.load(dump)["x_in"][0].float().mean(1)  # [256, H]
    scores = torch.sigmoid(x @ t[prefix + "weight"].float().t())
    idx, _, _ = RouterTopk.golden(scores, t[prefix + "e_score_correction_bias"].float(), K_PICKS, 2.5)
    return idx


# ============================================================================================ C10: router rows
def test_router_golden_rows_layout():
    g = torch.Generator().manual_seed(1)
    T = 32
    x = (torch.randn(T, H, generator=g) * 0.5).bfloat16()
    scores = torch.sigmoid(torch.randn(T, E, generator=g))
    bias = torch.randn(E, generator=g) * 0.1
    idx, w, dense = RouterTopk.golden(scores, bias, K_PICKS, 2.5)
    xw, idx_rows, tiles = RouterTopk.golden_rows(x, scores, bias, K_PICKS, 2.5)
    assert xw.shape == (T, H + K_PICKS) and xw.dtype == torch.bfloat16
    assert torch.equal(xw[:, :H], x) and torch.equal(xw[:, H:], w.bfloat16())
    assert torch.equal(idx_rows, idx)
    # the index tiles hold the ids as bf16 numbers: exact up to 256, rounded to even above (257..287 are NOT exact
    # in bf16: the same rounding the device writer (bf16_rne) and today's typecast path apply; the single-chip EP
    # op only matches ids inside its own [base, base + 9) range, which stays exact, and the mesh path carries the
    # ids as uint16 rows). The golden rounds the same way.
    assert tiles.shape == (T, 32) and torch.equal(tiles[:, :K_PICKS].float(), idx.float().bfloat16().float())
    assert bool((tiles[:, K_PICKS:].float() == IDX_PAD).all())
    exact = [int(torch.tensor(float(e)).bfloat16().item()) == e for e in range(E)]
    assert all(exact[:257]) and not all(exact)
    # the kernel writes the page tail as bf16 of the fp32 weight (RNE): same rounding as torch
    assert rm_page_bytes(H + K_PICKS) == 8208 and rm_page_bytes(K_PICKS) == 16
    assert bf16_bits(IDX_PAD) == 0x4580


def test_router_golden_rows_perm():
    g = torch.Generator().manual_seed(2)
    T = 8
    x = torch.randn(T, H, generator=g).bfloat16()
    scores = torch.sigmoid(torch.randn(T, E, generator=g))
    bias = torch.zeros(E)
    chips = P.plan_ep_placement([1.0 / E] * E, 32, 32, 1, 1)
    perm = torch.tensor([s[0] for s in P.placement_perm(chips, E, 9)])
    idx, _, _ = RouterTopk.golden(scores, bias, K_PICKS, 2.5)
    _, idx_rows, tiles = RouterTopk.golden_rows(x, scores, bias, K_PICKS, 2.5, perm=perm)
    assert torch.equal(idx_rows, perm[idx])
    assert torch.equal(tiles[:, :K_PICKS].float(), perm[idx].float().bfloat16().float())
    # slot d*9 + i holds global expert chips[d][i]: the chip's experts are contiguous in slot space
    inv = torch.empty(E, dtype=torch.long)
    inv[perm] = torch.arange(E)
    for d in range(32):
        assert inv[d * 9 : (d + 1) * 9].tolist() == chips[d]


def test_router_kernel_abi():
    w = (ROUTER_KERNELS / "writer_router_topk.cpp").read_text()
    r = (ROUTER_KERNELS / "reader_router_topk.cpp").read_text()
    wi = _ct_indices(w)
    # the scalar block RouterTopk.__call__ packs before the 7 accessors (scores, bias, dense, w, idx, xw, perm)
    expect = {
        "CB_S": "0",
        "CB_B": "1",
        "CB_CH": "2",
        "CB_SCR": "3",
        "E": "4",
        "K": "5",
        "SCALE_BITS": "6",
        "W_MODE": "7",
        "IDX_MODE": "8",
        "DENSE_MODE": "9",
        "X_MODE": "10",
        "PERM_MODE": "11",
        "CB_XROW": "12",
        "HT": "13",
        "XW_PAGE_BYTES": "14",
        "IDX_PAD_BITS": "15",
    }
    for k, v in expect.items():
        assert wi[k] == v, (k, wi.get(k))
    assert _accessor_offset(w) == 16
    assert w.count("next_compile_time_args_offset()") == 6  # bias, dense, w, idx, xw, perm chain
    ri = _ct_indices(r)
    assert ri == {"CB_XS": "0", "CB_XROW": "1", "HT": "2", "K": "3"} and _accessor_offset(r) == 4
    # common runtime args: [scores, bias, dense, w, idx, T, xw, perm]
    for i, name in enumerate(
        ["scores_addr", "bias_addr", "dense_addr", "w_addr", "idx_addr", "T", "xw_addr", "perm_addr"]
    ):
        assert f"{name} = get_common_arg_val<uint32_t>({i})" in w, name
    assert "noc_async_full_barrier();" in w and "noc_async_full_barrier();" in r
    # the IDX_MODE 4 pad and the modes the op hands out
    assert RouterTopk.IDX_MODE_ROWS == 3 and RouterTopk.IDX_MODE_TILES == 4
    assert RouterTopk.SCRATCH_BYTES >= 1536 + 64 + 576


# ============================================================================================ C11: glu_split
def test_glu_split_golden_and_work():
    g = torch.Generator().manual_seed(3)
    gu = torch.randn(1, 1, 32, 1024, generator=g) * 6
    h = GluSplit.golden(gu)
    gate, up = gu[..., :512], gu[..., 512:]
    ref = torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
    assert torch.equal(h, ref)
    assert float(h.abs().max()) <= 100.0 + 1e-3
    for gx, gy in GRIDS:
        grid = _Grid(gx, gy)
        for Rt, It in ((1, 16), (1, 96), (8, 16), (512, 96)):
            work = split_work(Rt * It, grid)
            assert sum(c for _, _, c in work) == Rt * It and work[0][1] == 0
            assert len(work) == min(Rt * It, gx * gy)
            assert max(c for _, _, c in work) - min(c for _, _, c in work) <= 1
            ranges = core_range_set(len(work), grid).ranges()
            assert 1 <= len(ranges) <= 2
            assert len({(c.x, c.y) for c, _, _ in work}) == len(work)
    with pytest.raises(AssertionError):  # allow-pytest.raises: noconftest
        GluSplit(device=None, limit=7.0)


def test_glu_split_kernel_abi():
    r = _ct_indices((GLU_KERNELS / "reader_glu_split.cpp").read_text())
    w = _ct_indices((GLU_KERNELS / "writer_glu_split.cpp").read_text())
    c = _ct_indices((GLU_KERNELS / "compute_glu_split.cpp").read_text())
    assert r == {"CB_G": "0", "CB_U": "1", "IT": "2", "TILE_BYTES": "3"}
    assert _accessor_offset((GLU_KERNELS / "reader_glu_split.cpp").read_text()) == 4
    assert w == {"CB_OUT": "0", "TILE_BYTES": "1"}
    assert _accessor_offset((GLU_KERNELS / "writer_glu_split.cpp").read_text()) == 2
    assert c == {"CB_G": "0", "CB_U": "1", "CB_OUT": "2"}
    for name in ("reader_glu_split.cpp", "writer_glu_split.cpp"):
        assert "noc_async_full_barrier();" in (GLU_KERNELS / name).read_text()
    src = (GLU_KERNELS / "compute_glu_split.cpp").read_text()
    assert "clamped_silu_glu_tile(0, 1, 2)" in src and "pack_tile(2, CB_OUT)" in src


def test_shared_glu_knob(monkeypatch):
    from models.demos.glm53_flash.tt import mlp

    monkeypatch.delenv("GLM53_SHARED_GLU", raising=False)
    assert mlp.shared_glu_mode() == "composite"
    for v in ("fused", "fused_all"):
        monkeypatch.setenv("GLM53_SHARED_GLU", v)
        assert mlp.shared_glu_mode() == v
    monkeypatch.setenv("GLM53_SHARED_GLU", "nope")
    with pytest.raises(AssertionError):  # allow-pytest.raises: noconftest
        mlp.shared_glu_mode()


# ============================================================================================ C12: bfp4 knob
def bfp_round_emulated(w: torch.Tensor, mant_bits: int, block: int = 16) -> torch.Tensor:
    """Host model of the block-float formats (bfp8_b: 7 magnitude bits, bfp4_b: 3) on blocks of 16 values along the
    last dim sharing the block max's exponent: step = 2^(e_max + 1 - mant_bits), round to nearest even. The exact
    ttnn conversion (``custom_ops/expert_stream/op.py::bfp4_round``) instantiates the MetalContext (it opens the UMD
    cluster even for a host tensor), so the host test models it and ``tests/test_moe_roofline_mesh.py`` compares the
    real one on chip 0."""
    shape = w.shape
    x = w.float().reshape(-1, block)
    m = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    e = torch.floor(torch.log2(m))
    step = torch.pow(2.0, e + 1 - mant_bits)
    q = torch.round(x / step) * step
    lim = (2**mant_bits - 1) * step
    return torch.maximum(torch.minimum(q, lim), -lim).reshape(shape)


def test_bfp4_round_trip():
    g = torch.Generator().manual_seed(4)
    w = torch.randn(1024, 512, generator=g) * 0.02  # expert-like weights at their real scale
    q8, q4 = bfp_round_emulated(w, 7), bfp_round_emulated(w, 3)
    assert pcc(w, q8) > 0.9999 and pcc(w, q4) > 0.99
    rel8 = float((q8 - w).norm() / w.norm())
    rel4 = float((q4 - w).norm() / w.norm())
    assert rel8 < 0.01 and 0.03 < rel4 < 0.2, (rel8, rel4)
    assert torch.isfinite(q4).all() and torch.equal(bfp_round_emulated(q4, 3), q4)  # idempotent
    # the formats' tile bytes (what the roofline counts): 1088 vs 576 per 32x32 tile
    assert P.TILE_BYTES["bfp8"] == 1088 and P.TILE_BYTES["bfp4"] == 576
    gu8, dn8 = P.expert_bytes(H, I, H, "bfp8", "bfp8")
    gu4, _ = P.expert_bytes(H, I, H, "bfp4", "bfp8")
    assert (gu8, dn8) == (128 * 128 * 1088, 64 * 128 * 1088)
    assert gu4 == 128 * 128 * 576
    assert abs(P.floor_us(9 * (gu8 + dn8)) - 469.6) < 1 and abs(P.floor_us(9 * (gu4 + dn8)) - 322.6) < 1


def test_w13_dtype_knob_and_cache_names(monkeypatch, tmp_path):
    monkeypatch.delenv(M.W13_DTYPE_ENV, raising=False)
    assert M.routed_w13_dtype(ttnn.bfloat8_b) == ttnn.bfloat8_b
    monkeypatch.setenv(M.W13_DTYPE_ENV, "bfp4")
    assert M.routed_w13_dtype(ttnn.bfloat8_b) == ttnn.bfloat4_b
    monkeypatch.setenv(M.W13_DTYPE_ENV, "bogus")
    with pytest.raises(AssertionError):  # allow-pytest.raises: noconftest
        M.routed_w13_dtype(ttnn.bfloat8_b)
    monkeypatch.delenv(M.W13_DTYPE_ENV, raising=False)
    mesh = _Mesh(8, 4)
    f8 = M.TtEPExperts.cache_files(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b)
    f4 = M.TtEPExperts.cache_files(tmp_path, mesh, ttnn.bfloat4_b, ttnn.bfloat8_b)
    assert f8["w13"].name == "experts_w13_BFLOAT8_B_mesh8x4_ep.tensorbin"  # the warm cache's name (unchanged)
    assert f4["w13"].name == "experts_w13_BFLOAT4_B_mesh8x4_ep.tensorbin"
    assert f8["w2"] == f4["w2"]  # the down format is shared
    h8 = M.TtRoutedExperts.cache_files(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b, stream=True)
    assert h8["w13"].name == "experts_w13_BFLOAT8_B_mesh8x4_stream.tensorbin"
    assert M.EXPERT_CACHE_LAYOUT == ""  # today's names; bump when a layout changes under an unchanged name
    # a warm bfp8 cache does not pass for a bfp4 one: expert_cache_complete applies the override
    for f in f8.values():
        f.write_bytes(b"x")
    monkeypatch.setenv("GLM53_MOE_LAYOUT", "ep")
    assert M.TtMoE.expert_cache_complete(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b)
    monkeypatch.setenv(M.W13_DTYPE_ENV, "bfp4")
    assert not M.TtMoE.expert_cache_complete(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b)
    for f in f4.values():
        f.write_bytes(b"x")
    assert M.TtMoE.expert_cache_complete(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b)
    # the placement tag lands in the EP names only
    chips = P.plan_ep_placement([1.0 / E] * E, 32, 32, 1, 1)
    pf = tmp_path / "placement.json"
    P.save_placement(pf, chips)
    monkeypatch.setenv("GLM53_EP_PLACEMENT", str(pf))
    fp = M.TtEPExperts.cache_files(tmp_path, mesh, ttnn.bfloat4_b, ttnn.bfloat8_b)
    assert fp["w13"].name == f"experts_w13_BFLOAT4_B_mesh8x4_ep_p{P.placement_hash(chips)}.tensorbin"
    assert M.ep_placement_tag_from_env(32) == "_p" + P.placement_hash(chips) and M.ep_placement_tag_from_env(1) == ""
    assert not M.TtMoE.expert_cache_complete(tmp_path, mesh, ttnn.bfloat8_b, ttnn.bfloat8_b)  # new names: cold load
    got, tag = M.ep_placement_from_env(E, 32)
    assert got == chips and tag == "_p" + P.placement_hash(chips)
    with pytest.raises(AssertionError):  # allow-pytest.raises: noconftest
        M.ep_placement_from_env(E, 8)  # a 32-chip placement on a block


def test_placeholder_guard():
    t = torch.randn(4, 4)
    assert M._nonzero(t, "x") is t
    for bad in (torch.zeros(3, 3), torch.empty(0), torch.full((2, 2), float("nan"))):
        with pytest.raises(RuntimeError):  # allow-pytest.raises: noconftest
            M._nonzero(bad, "w13")


# ============================================================================================ (d): stream planner
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("w13", ["bfp8", "bfp4"])
def test_stream_planner_grids(grid, w13):
    for active, T in ((2, 32), (5, 32), (9, 256)):
        plan = P.plan_expert_stream(
            grid=grid,
            E_local=9,
            K=H,
            I=I,
            H=H,
            T=T,
            w13=w13,
            expected_active=active,
            tokens_per_expert=4 if T == 32 else 28,
        )
        kw = plan.op_kwargs()
        assert 1 <= kw["groups"] <= min(32, (grid[0] * grid[1]) // 8)
        assert 8 % kw["unit_pairs"] == 0 and kw["prefetch_pairs"] >= 1
        assert kw["packet_tiles"] * P.TILE_BYTES[w13] <= P.MAX_PACKET_BYTES and 128 % kw["packet_tiles"] == 0
        assert kw["down_readers"] == 8
        # the ring must fit L1 next to the other CBs
        assert (
            P.gate_up_l1_bytes(128, P.TILE_BYTES[w13], kw["prefetch_pairs"], T, 9, True)
            <= P.L1_BYTES - P.L1_RESERVED_BYTES
        )
        assert plan.expected_gate_up_us >= plan.expected_floor_us * 0.5 or active <= 2
        assert plan.expected_gate_up_us > 0 and plan.expected_down_us > 0
    # bfp4 pairs are half the size: a deeper prefetch fits
    p8 = P.max_prefetch_pairs(128, P.TILE_BYTES["bfp8"], 32, 9, True)
    p4 = P.max_prefetch_pairs(128, P.TILE_BYTES["bfp4"], 32, 9, True)
    assert p8 == 1 and p4 >= 2, (p8, p4)
    # the hybrid shape (I = 256: one pair per bank): the planner keeps the whole-expert assignment
    hp = P.plan_expert_stream(grid=grid, E_local=72, K=H, I=256, H=H, T=32, expected_active=28, tokens_per_expert=4)
    assert hp.unit_pairs == 1


def test_stream_planner_calibration_and_regimes():
    fit = P.fit_gather_convoy()
    assert 20 < fit["stream_plus_compute_us"] < 40 and 2 < fit["gather_us_per_core"] < 3.5, fit
    # few active experts: the pair-unit split with the rotation beats the per-expert assignment in the model
    few = {
        (units, u, rot): P.gate_up_time_us(
            kt=128,
            ppb=8,
            tile_bytes=1088,
            groups=12,
            units=units,
            unit_pairs=u,
            n_active=2,
            passes_per_expert=1,
            gather_rotate=rot,
            tokens=32,
        )[0]
        for units in (False, True)
        for u in (1, 2, 4, 8)
        for rot in (False, True)
    }
    assert abs(few[(False, 1, False)] - 288) < 15  # the measured per-expert point (T=32, 2 active): 288 us
    assert abs(few[(True, 1, False)] - 579) < 15  # the measured units point: 579 us
    assert few[(True, 1, True)] < few[(False, 1, False)] * 0.6  # the convoy fix makes the units mode win
    # all 9 active at 256 tokens: bandwidth-bound, the plateau decides
    t9, d9 = P.gate_up_time_us(
        kt=128,
        ppb=8,
        tile_bytes=1088,
        groups=12,
        units=False,
        unit_pairs=1,
        n_active=9,
        passes_per_expert=1,
        gather_rotate=False,
        tokens=256,
    )
    gu, _ = P.expert_bytes(H, I, H)
    assert t9 >= P.floor_us(9 * gu) and d9["ceiling_us"] == pytest.approx(P.floor_us(9 * gu, P.STREAM_PLATEAU_GBS))
    assert d9["cores"] == 72


def test_plan_from_env_knobs():
    assert P.plan_from_env({}, 9, I) is None
    assert P.plan_from_env({"GLM53_EXPERT_PLAN": "legacy"}, 9, I) is None
    p = P.plan_from_env(
        {
            "GLM53_EXPERT_PLAN": "explicit",
            "GLM53_STREAM_UNIT_PAIRS": "2",
            "GLM53_STREAM_UNITS": "1",
            "GLM53_STREAM_PREFETCH": "2",
            "GLM53_STREAM_GATHER_ROTATE": "1",
            "GLM53_STREAM_PACKET_TILES": "4",
        },
        9,
        I,
    )
    assert p.op_kwargs() == dict(
        groups=12, units=True, unit_pairs=2, prefetch_pairs=2, packet_tiles=4, down_readers=8, gather_rotate=True
    )
    r = P.plan_from_env(
        {"GLM53_EXPERT_PLAN": "roofline", "GLM53_EXPERT_PLAN_ACTIVE": "3", "GLM53_EXPERT_W13_DTYPE": "bfp4"}, 9, I
    )
    assert r.prefetch_pairs >= 2 and r.packet_tiles == 16
    r2 = P.plan_from_env(
        {"GLM53_EXPERT_PLAN": "roofline", "GLM53_EXPERT_PLAN_GRID": "8x8", "GLM53_STREAM_GROUPS": "6"}, 9, I
    )
    assert r2.groups == 6
    with pytest.raises(ValueError):  # allow-pytest.raises: noconftest
        P.plan_from_env({"GLM53_EXPERT_PLAN": "magic"}, 9, I)
    assert P.legacy_plan(9, I).op_kwargs() == dict(
        groups=12, units=False, unit_pairs=1, prefetch_pairs=1, packet_tiles=None, down_readers=8, gather_rotate=False
    )


def test_expert_stream_kernel_abi():
    """The modified kernels' compile-time arg positions against the vectors ExpertGateUpDispatch / ExpertDownDispatch
    pack (12 / 13 on the readers, 16 / 17 on the gate|up writers, 20 on the down writer, 7 on the units compute)."""
    rd = (STREAM_KERNELS / "reader_gate_up_dispatch.cpp").read_text()
    ru = (STREAM_KERNELS / "reader_gate_up_dispatch_units.cpp").read_text()
    wd = (STREAM_KERNELS / "writer_gate_up_dispatch.cpp").read_text()
    wu = (STREAM_KERNELS / "writer_gate_up_dispatch_units.cpp").read_text()
    dw = (STREAM_KERNELS / "writer_down_dispatch.cpp").read_text()
    cu = (STREAM_KERNELS / "compute_gate_up_dispatch_units.cpp").read_text()
    cd = (STREAM_KERNELS / "compute_gate_up_dispatch.cpp").read_text()
    assert _ct_indices(rd)["num_buffers"] == "12" and _accessor_offset(rd) == 14
    assert _ct_indices(ru)["num_buffers"] == "12" and _ct_indices(ru)["U"] == "13" and _accessor_offset(ru) == 14
    assert _ct_indices(wd)["GATHER_ROTATE"] == "16" and _accessor_offset(wd) == 18
    assert _ct_indices(wu)["GATHER_ROTATE"] == "16" and _ct_indices(wu)["U"] == "17" and _accessor_offset(wu) == 18
    assert _ct_indices(dw)["GATHER_ROTATE"] == "20" and _accessor_offset(dw) == 21
    assert _ct_indices(cu)["U"] == "7" and _ct_indices(cd)["ppb"] == "7"
    for name in (
        "reader_gate_up_dispatch",
        "reader_gate_up_dispatch_units",
        "writer_gate_up_dispatch",
        "writer_gate_up_dispatch_units",
        "writer_down_dispatch",
        "reader_down_dispatch",
        "reader_gate_up",
        "writer_gate_up",
        "reader_down",
        "writer_down",
    ):
        assert "noc_async_full_barrier();" in (STREAM_KERNELS / f"{name}.cpp").read_text(), name
    common = (STREAM_KERNELS / "dispatch_common.hpp").read_text()
    assert "for_each_unit_chunk" in common and "rotated_row" in common
    # the op packs the scalars in that order (source inspection of op.py: the comments carry the indices)
    src = (STREAM_KERNELS.parent / "op.py").read_text()
    assert "self.prefetch_pairs + 1,  # 12" in src and "self.unit_pairs,  # 13" in src
    assert "int(self.gather_rotate),  # 16" in src and "self.unit_pairs,  # 17" in src
    assert "int(self.gather_rotate),  # 20" in src


def test_for_each_unit_chunk_model():
    """Host model of dispatch_common.hpp's LPT over (expert, chunk) units: U = PPB reproduces the per-expert
    assignment, U = 1 the pair units, and every unit is assigned exactly once over the groups."""

    def lpt(passes, groups, chunks_per_expert):
        order = sorted([e for e, p in enumerate(passes) if p], key=lambda e: (-passes[e], e))
        load = [0] * groups
        out = {g: [] for g in range(groups)}
        for e in order:
            for c in range(chunks_per_expert):
                g = min(range(groups), key=lambda j: (load[j], j))
                load[g] += passes[e]
                out[g].append((e, c))
        return out

    passes = [1, 0, 2, 0, 0, 1, 0, 0, 1]
    whole = lpt(passes, 12, 1)
    units = lpt(passes, 12, 8)
    halves = lpt(passes, 12, 4)
    assert sum(len(v) for v in whole.values()) == 4 and sum(len(v) for v in units.values()) == 32
    assert sum(len(v) for v in halves.values()) == 16
    assert max(len(v) for v in units.values()) == 3 and max(len(v) for v in whole.values()) == 1
    for e in (0, 2, 5, 8):
        assert sorted(c for v in halves.values() for ee, c in v if ee == e) == [0, 1, 2, 3]
    # rotated gather order is a permutation of the rows for every start
    for n in (1, 5, 32):
        for rot in (0, 3, 31, 97):
            seq = [(j + rot % n) % n for j in range(n)]
            assert sorted(seq) == list(range(n))


# ============================================================================================ (e): EP32 skew
def _zipf_popularity(seed=5, alpha=1.1):
    g = torch.Generator().manual_seed(seed)
    pref = 1.0 / (torch.arange(E, dtype=torch.float) + 1.0) ** alpha
    pref = pref[torch.randperm(E, generator=g)]
    p = pref / pref.sum() * K_PICKS  # per-token pick probability, sum = k
    return p.clamp(max=0.999).tolist()


def test_placement_planner_balances_bytes():
    gu, dn = P.expert_bytes(H, I, H)
    pop = _zipf_popularity()
    for users in (4, 32):
        tokens = 8 * users
        cont = P.contiguous_placement(E, 32)
        bal = P.plan_ep_placement(pop, 32, tokens, gu, dn)
        assert sorted(e for c in bal for e in c) == list(range(E)) and all(len(c) == 9 for c in bal)
        sc = P.column_skew(cont, pop, 8, 4, users, gu, dn)
        sb = P.column_skew(bal, pop, 8, 4, users, gu, dn)
        assert sb["step_max_mb"] <= sc["step_max_mb"] * 0.9, (users, sc["step_max_mb"], sb["step_max_mb"])
        assert sb["step_max_mb"] <= sb["balanced_mb"] * 1.15  # within 15 % of the perfectly balanced read
        # the per-chip bytes model never exceeds the chip's full read
        assert max(sb["per_chip_mb"]) <= 9 * (gu + dn) / 1e6 + 1e-6
    # replication: the two hottest experts get a second copy, slots grow to 10
    rep = P.plan_ep_placement(pop, 32, 256, gu, dn, replicate=2)
    assert sum(len(c) for c in rep) == E + 2 and max(len(c) for c in rep) == 10
    perm = P.placement_perm(rep, E, 10)
    assert sum(len(s) for s in perm) == E + 2 and sum(len(s) == 2 for s in perm) == 2
    with pytest.raises(AssertionError):  # allow-pytest.raises: noconftest
        P.plan_ep_placement(pop, 32, 256, gu, dn, slots_per_chip=8)


def test_placement_perm_and_files(tmp_path):
    pop = _zipf_popularity(seed=9)
    chips = P.plan_ep_placement(pop, 8, 64, 1, 1)
    perm = P.placement_perm(chips, E, 36)
    assert all(len(s) == 1 for s in perm) and sorted(s[0] for s in perm) == list(range(E))
    for d, lst in enumerate(chips):
        for i, e in enumerate(lst):
            assert perm[e][0] == d * 36 + i
    f = tmp_path / "p.json"
    P.save_placement(f, chips, {"note": "zipf"})
    assert P.load_placement(f) == chips and json.loads(f.read_text())["note"] == "zipf"
    assert P.placement_hash(chips) == P.placement_hash(P.load_placement(f)) and len(P.placement_hash(chips)) == 8
    assert P.placement_hash(chips) != P.placement_hash(P.contiguous_placement(E, 8))


def test_simulated_skew_matches_profile_pattern():
    """Realized per-step skew on a Zipf routing: contiguous placement leaves some chips idle and others with most
    of their experts active (the 33-502 us spread of the profile); the balanced placement narrows the column max."""
    gu, dn = P.expert_bytes(H, I, H)
    pop = _zipf_popularity(seed=11, alpha=1.3)
    g = torch.Generator().manual_seed(12)
    probs = torch.tensor(pop)
    steps = [
        [torch.multinomial(probs, K_PICKS, replacement=False, generator=g).tolist() for _ in range(32)]
        for _ in range(16)
    ]
    cont_place = P.contiguous_placement(E, 32)
    bal_place = P.plan_ep_placement(P.popularity_from_routing(sum(steps, []), E), 32, 32, gu, dn)
    cont = [P.simulate_skew(cont_place, rows, 8, 4, 4, gu, dn) for rows in steps]
    bal = [P.simulate_skew(bal_place, rows, 8, 4, 4, gu, dn) for rows in steps]
    assert min(min(c["per_chip_active"]) for c in cont) <= 1 and max(max(c["per_chip_active"]) for c in cont) >= 4
    # one step's realized max is a random variable: compare the mean over the steps (the expected-bytes balance)
    mean_cont = sum(c["step_max_mb"] for c in cont) / len(cont)
    mean_bal = sum(b["step_max_mb"] for b in bal) / len(bal)
    assert mean_bal <= mean_cont, (mean_bal, mean_cont)
    assert abs(sum(P.popularity_from_routing(steps[0], E)) - K_PICKS) < 1e-9


def test_real_routing_placement():
    """The real layer-4 routing (T256 dump + the real gate): the measured popularity's skew and what the balanced
    placement buys at 4 and 32 users per row (logged; skipped without the checkpoint / dump)."""
    idx = _real_routing()
    if idx is None:
        pytest.skip("no T256 dump / checkpoint on this host")
    rows = idx.tolist()
    pop = P.popularity_from_routing(rows, E)
    gu, dn = P.expert_bytes(H, I, H)
    active = sum(p > 0 for p in pop)
    assert 60 <= active <= 200  # 112 of 288 active over the 256 tokens (expert_kernel_sweep.md section 11)
    for users in (4, 32):
        cont = P.simulate_skew(P.contiguous_placement(E, 32), rows, 8, 4, users, gu, dn)
        bal = P.simulate_skew(P.plan_ep_placement(pop, 32, 8 * users, gu, dn), rows, 8, 4, users, gu, dn)
        print(
            f"real routing {users} users/row: contiguous col-max {cont['step_max_mb']:.0f} MB (mean {cont['mean_mb']:.0f}), "
            f"balanced col-max {bal['step_max_mb']:.0f} MB (mean {bal['mean_mb']:.0f})"
        )
        assert bal["step_max_mb"] <= cont["step_max_mb"]


def test_layer_floors_table():
    """The per-layer floors the STATUS table quotes."""
    sh = P.LayerShape()
    h4 = P.layer_floor_us(sh, "hybrid", 4, active=28)
    h32 = P.layer_floor_us(sh, "hybrid", 32, active=31)
    e4 = P.layer_floor_us(sh, "ep", 4, active=3)
    e32 = P.layer_floor_us(sh, "ep", 32, active=9)
    assert abs(h4["bytes"] / 1e6 - 93.6) < 0.5 and abs(h32["bytes"] / 1e6 - 103.6) < 0.5
    assert abs(e4["bytes"] / 1e6 - 80.2) < 0.5 and abs(e32["bytes"] / 1e6 - 240.5) < 0.5
    assert abs(e32["floor_us"] - 469.6) < 1 and abs(h4["floor_us"] - 182.8) < 1
    e32_4 = P.layer_floor_us(sh, "ep", 32, active=9, w13="bfp4")
    assert abs(e32_4["floor_us"] - 322.6) < 1
