# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the Hy4-preview MoE / dense-FFN placement (#65890, #65923). No device.

* ``TtMlp2D``'s sharding plan (fused gate|up per intermediate shard, ``ShardTensor2dMesh`` dims, and the
  all-gather / all-reduce / reduce-scatter sequence) emulated chip by chip in torch equals the dense HF MLP, on the
  SC1 mesh both ways round and in the prefill and decode forms.
* The torch reference (``reference/hy4_moe.py``): router picks and weights, the clamp on routed experts only.
* The clamp the routed-expert kernel bakes in (``ClampedSiluGluConfigDsV4``) is Hy4's ``swiglu_limit``.
* Config / layout wiring: tp32 runs ``Hy4TP32Config`` (FFN_2D_TP), the dev layout keeps the 1D modules.
* 2D-TP cache files are distinct from the 1D modules' (neither loads as the other) and name the hidden's mesh axis
  (a cache from the old split, H on axis 0, never loads for SC1's H on axis 1).
* DRAM per chip of the FFN / MoE weights on SC1 against #65884's table.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference import hy4_moe as ref
from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config, Hy4TP32Config
from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout
from models.demos.deepseek_v3_d_p.tt.moe.tt_mlp_2d import (
    Mlp2DLayout,
    cache_guard,
    cache_names,
    ffn_2d_layout,
    ffn_2d_tp,
    fuse_gate_up,
    shard_dims,
)

REPO = Path(__file__).resolve().parents[4]


# ------------------------------------------------------------------------------------------------ 2D TP emulation
def _shard(t: torch.Tensor, dims: tuple, mesh: tuple) -> dict:
    """``ShardTensor2dMesh`` on host: ``{(r, c): shard}``; ``dims[a]`` is the tensor dim split over mesh axis a."""
    out = {}
    for r in range(mesh[0]):
        for c in range(mesh[1]):
            s = t
            for axis, idx in ((0, r), (1, c)):
                if dims[axis] is not None:
                    s = s.chunk(mesh[axis], dim=dims[axis])[idx]
            out[(r, c)] = s
    return out


def _along(axis: int, coord: tuple, mesh: tuple) -> list:
    """The coordinates sharing ``coord``'s other index, in order along ``axis`` (one CCL ring / line)."""
    return [(i, coord[1]) if axis == 0 else (coord[0], i) for i in range(mesh[axis])]


def _emulate_mlp_2d(x, w, mesh, hidden_axis, tokens_sharded, activation=ref.swiglu_mlp, limit=None):
    """Per-chip TtMlp2D.forward in torch. ``x`` ``[S, H]`` global; returns the global ``[S, H]`` the chips hold."""
    inter_axis = 1 - hidden_axis
    ni, nh = mesh[inter_axis], mesh[hidden_axis]
    inter = w["gate_proj"].shape[0]
    gate_up = _shard(fuse_gate_up(w["gate_proj"], w["up_proj"], ni), shard_dims(hidden_axis, "hidden"), mesh)
    down = _shard(w["down_proj"].T.contiguous(), shard_dims(hidden_axis, "inter"), mesh)
    # Input: tokens over inter_axis (prefill) or replicated (decode); hidden over hidden_axis.
    x_dims = [None, None]
    x_dims[hidden_axis] = -1
    if tokens_sharded:
        x_dims[inter_axis] = -2
    xs = _shard(x, tuple(x_dims), mesh)
    coords = list(xs)
    if tokens_sharded:  # all_gather(dim=-2, inter_axis)
        xs = {k: torch.cat([xs[j] for j in _along(inter_axis, k, mesh)], dim=-2) for k in coords}
    gu = {k: xs[k] @ gate_up[k] for k in coords}
    gu = {k: sum(gu[j] for j in _along(hidden_axis, k, mesh)) for k in coords}  # all_reduce(hidden_axis)
    act = {}
    for k in coords:
        g, u = gu[k][:, : inter // ni], gu[k][:, inter // ni :]
        act[k] = ref.clamped_swiglu(g, u, limit) if limit else torch.nn.functional.silu(g) * u
    y = {k: act[k] @ down[k] for k in coords}
    y = {k: sum(y[j] for j in _along(inter_axis, k, mesh)) for k in coords}  # reduce over inter_axis
    if tokens_sharded:  # ... scattered over the tokens: chip i of the inter axis keeps token block i
        y = {k: y[k].chunk(ni, dim=-2)[k[inter_axis]] for k in coords}
    # Reassemble the global [S, H] from chip (r, c) = token block (inter index), hidden block (hidden index).
    rows = []
    for t in range(ni if tokens_sharded else 1):
        row = []
        for h in range(nh):
            k = [0, 0]
            k[hidden_axis], k[inter_axis] = h, t
            row.append(y[tuple(k)])
        rows.append(torch.cat(row, dim=-1))
    out = torch.cat(rows, dim=-2)
    if not tokens_sharded:  # decode: every chip of an inter-axis line must hold the same result
        for k in coords:
            other = list(k)
            other[inter_axis] = (k[inter_axis] + 1) % ni
            torch.testing.assert_close(y[k], y[tuple(other)])
    return out


def _mlp_weights(hidden, inter, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return {
        "gate_proj": torch.randn(inter, hidden, generator=g) * scale / hidden**0.5,
        "up_proj": torch.randn(inter, hidden, generator=g) * scale / hidden**0.5,
        "down_proj": torch.randn(hidden, inter, generator=g) / inter**0.5,
    }


@pytest.mark.parametrize("mesh, hidden_axis", [((8, 4), 0), ((8, 4), 1), ((4, 8), 1), ((4, 8), 0)])
@pytest.mark.parametrize("tokens_sharded", [True, False], ids=["prefill", "decode"])
def test_mlp_2d_plan_matches_dense(mesh, hidden_axis, tokens_sharded):
    """The 2D-TP plan reproduces the dense MLP exactly (fp64) on the SC1 mesh, either axis carrying the hidden."""
    hidden, inter, tokens = 256, 512, 64
    w = {k: v.double() for k, v in _mlp_weights(hidden, inter).items()}
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    expected = (torch.nn.functional.silu(x @ w["gate_proj"].T) * (x @ w["up_proj"].T)) @ w["down_proj"].T
    got = _emulate_mlp_2d(x, w, mesh, hidden_axis, tokens_sharded)
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-10)


def test_mlp_2d_plan_clamped():
    """The same plan with the routed experts' clamped activation (TtMlp2D supports it; Hy4 uses it for neither)."""
    hidden, inter, lim = 128, 256, 10.0
    w = {k: v.double() for k, v in _mlp_weights(hidden, inter, scale=40.0).items()}
    x = torch.randn(32, hidden, dtype=torch.float64)
    g, u = x @ w["gate_proj"].T, x @ w["up_proj"].T
    assert (g > lim).any() and (u.abs() > lim).any(), "test input does not reach the clamp"
    expected = ref.clamped_swiglu(g, u, lim) @ w["down_proj"].T
    got = _emulate_mlp_2d(x, w, (8, 4), Hy4TP32Config.FFN_2D_HIDDEN_AXIS, True, limit=lim)
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-10)


def test_fuse_gate_up_blocks(expect_error):
    """Column block i of the fused weight is [gate_i | up_i] (what TtMlp2D.forward slices per chip)."""
    hidden, inter, n = 8, 12, 3
    gate = torch.arange(inter * hidden, dtype=torch.float32).reshape(inter, hidden)
    up = -gate
    fused = fuse_gate_up(gate, up, n)
    assert fused.shape == (hidden, 2 * inter)
    w = inter // n
    for i, block in enumerate(fused.chunk(n, dim=-1)):
        torch.testing.assert_close(block[:, :w], gate[i * w : (i + 1) * w].T)
        torch.testing.assert_close(block[:, w:], up[i * w : (i + 1) * w].T)
    with expect_error(ValueError, "does not split 5 ways"):
        fuse_gate_up(gate, up, 5)


def test_shard_dims():
    # Rows (-2) over the named logical axis, columns (-1) over the other mesh axis.
    assert shard_dims(0, "hidden") == (-2, -1)
    assert shard_dims(1, "hidden") == (-1, -2)
    assert shard_dims(0, "inter") == (-1, -2)
    assert shard_dims(1, "inter") == (-2, -1)


# ------------------------------------------------------------------------------------------------------ reference
def _tiny_moe(E=16, H=64, I=32, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        gate_w=torch.randn(E, H, generator=g) / H**0.5,
        bias=torch.randn(E, generator=g) * 0.1,
        gate_up=torch.randn(E, 2 * I, H, generator=g) * scale / H**0.5,
        down=torch.randn(E, H, I, generator=g) / I**0.5,
        shared=_mlp_weights(H, I, seed=seed + 1),
    )


def test_route_bias_picks_but_does_not_weight():
    """HYV4MoE: the bias changes WHICH experts are picked, never the weights (sigmoid of the raw logit,
    normalised over the k picks, x routed_scaling_factor)."""
    m = _tiny_moe()
    x = torch.randn(40, 64)
    top_i, top_w = ref.route(x, m["gate_w"], m["bias"], top_k=4, routed_scaling_factor=2.827)
    scores = torch.sigmoid(x @ m["gate_w"].T)
    expect_i = (scores + m["bias"]).topk(4, dim=-1)[1]
    assert torch.equal(top_i, expect_i)
    w = scores.gather(1, top_i)
    torch.testing.assert_close(top_w, w / w.sum(-1, keepdim=True) * 2.827)
    torch.testing.assert_close(top_w.sum(-1), torch.full((40,), 2.827))
    # A large bias on one expert forces it into every token's picks.
    bias = m["bias"].clone()
    bias[3] = 100.0
    forced, _ = ref.route(x, m["gate_w"], bias, top_k=4)
    assert (forced == 3).any(-1).all()


def test_route_groups_one_group_is_noop():
    """Hy4 has n_group = topk_group = 1: group-limited routing must reduce to plain top-k."""
    m = _tiny_moe()
    x = torch.randn(20, 64)
    a = ref.route(x, m["gate_w"], m["bias"], top_k=4, n_group=1, topk_group=1)
    b = ref.route(x, m["gate_w"], m["bias"], top_k=4)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert Hy4Config.NUM_EXPERT_GROUPS == 1 and Hy4Config.NUM_LIMITED_GROUPS == 1


def test_routed_moe_matches_per_token_loop_and_clamps():
    """``routed_moe`` equals a per-token loop over the picks, and its clamp is live at this input scale."""
    m = _tiny_moe(scale=60.0)
    x = torch.randn(24, 64)
    lim = 10.0
    top_i, top_w = ref.route(x, m["gate_w"], m["bias"], top_k=4, routed_scaling_factor=2.827)
    got = ref.routed_moe(x, top_i, top_w, m["gate_up"], m["down"], limit=lim)
    expect = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for s in range(4):
            e = int(top_i[t, s])
            g, u = (x[t] @ m["gate_up"][e].T).chunk(2)
            act = torch.nn.functional.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)
            expect[t] += top_w[t, s] * (act @ m["down"][e].T)
    # fp32, summed in a different order over activations ~10x the clamp: compare relative to the output scale.
    torch.testing.assert_close(got, expect, rtol=1e-4, atol=1e-4 * expect.abs().max().item())
    gate_frac, up_frac = ref.clamp_fractions(x, top_i, m["gate_up"], limit=lim)
    assert gate_frac > 0.05 and up_frac > 0.05
    unclamped = ref.routed_moe(x, top_i, top_w, m["gate_up"], m["down"], limit=float("inf"))
    assert not torch.allclose(unclamped, got)


def test_shared_and_dense_are_unclamped():
    """HYV4MLP (shared expert, dense layer 0) has no clamp; the Hy4 configs agree (1D and 2D modules alike)."""
    w = _mlp_weights(64, 32, scale=60.0)
    x = torch.randn(8, 64)
    g = x @ w["gate_proj"].T
    assert (g > 10).any()
    expect = (torch.nn.functional.silu(g) * (x @ w["up_proj"].T)) @ w["down_proj"].T
    torch.testing.assert_close(ref.swiglu_mlp(x, w["gate_proj"], w["up_proj"], w["down_proj"]), expect)
    for cfg in (Hy4Config, Hy4TP32Config):
        assert cfg.SHARED_EXPERT_ACTIVATION == "silu" and cfg.DENSE_FFN_ACTIVATION == "silu"
        assert cfg.ROUTED_EXPERT_ACTIVATION == "clamped_silu_glu" and cfg.SWIGLU_LIMIT == 10.0


def test_moe_block_is_routed_plus_shared():
    m = _tiny_moe()
    x = torch.randn(16, 64)
    out = ref.moe(x, top_k=4, limit=10.0, routed_scaling_factor=2.827, **m)
    torch.testing.assert_close(out["out"], out["routed"] + out["shared"])


# ------------------------------------------------------------------------------------- kernel / config / layout
def test_routed_kernel_clamp_is_hy4_swiglu_limit():
    """The unified routed-expert kernel bakes ClampedSiluGluConfigDsV4 at compile time; Hy4 reuses it, so its limit
    must equal Hy4's swiglu_limit (and the formula be silu(min(g, L)) * clamp(u, -L, L))."""
    header = REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_clamped_silu_glu.h"
    if not header.is_file():
        pytest.skip(f"{header} not in this checkout")
    text = header.read_text()
    m = re.search(r"struct ClampedSiluGluConfigDsV4\s*{\s*static constexpr float limit = ([0-9.]+)f;", text)
    assert m, "ClampedSiluGluConfigDsV4 limit not found"
    assert float(m.group(1)) == Hy4Config.SWIGLU_LIMIT
    assert "gate = sfpi::min(gate, limit);" in text and "up = sfpi::clamp(up, -limit, limit);" in text


def test_ffn_2d_tp_wiring(expect_error):
    assert not ffn_2d_tp(Hy4Config) and ffn_2d_tp(Hy4TP32Config)
    assert issubclass(Hy4TP32Config, Hy4Config)
    tp32 = hy4_layout.HY4_PREFILL_LAYOUTS[hy4_layout.TP32_LAYOUT]
    dev = hy4_layout.HY4_PREFILL_LAYOUTS[hy4_layout.DEV_LAYOUT]
    assert tp32.ffn_2d_tp and not dev.ffn_2d_tp
    assert "#65890" not in {issue for _, issue in tp32.blockers}
    # SC1: H over the 4-chip mesh axis 1 (the TP axis of PREFILL_SP=8 PREFILL_TP=4 PREFILL_SP_AXIS=0).
    lay = ffn_2d_layout(Hy4TP32Config)
    assert lay == Mlp2DLayout((8, 4), 1) and lay.hidden_axis == tp32.hidden_axis
    assert lay.mesh_shape[tp32.sp_axis] == tp32.sp == 8 and lay.mesh_shape[tp32.hidden_axis] == tp32.tp == 4
    with expect_error(ValueError, "does not set FFN_2D_TP"):
        ffn_2d_layout(Hy4Config)


def test_adapter_picks_model_config_by_layout(monkeypatch):
    from types import SimpleNamespace

    from models.demos.deepseek_v3_d_p.tt.runners.adapters.hy4_preview import Hy4Adapter

    adapter = Hy4Adapter()
    mesh = SimpleNamespace(sp_factor=8, tp_factor=4, sp_axis=0)  # both layouts' mesh; the layout env picks
    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, hy4_layout.DEV_LAYOUT)
    assert adapter.model_config_for(mesh) is Hy4Config
    monkeypatch.delenv(hy4_layout.LAYOUT_ENV)
    assert adapter.model_config_for(SimpleNamespace(sp_factor=2, tp_factor=4, sp_axis=0)) is Hy4Config  # test mesh
    # tp32 is still blocked by #65885 / #65887 / #65892; with those cleared it runs the 2D-TP config.
    tp32 = hy4_layout.HY4_PREFILL_LAYOUTS[hy4_layout.TP32_LAYOUT]
    monkeypatch.setitem(hy4_layout.HY4_PREFILL_LAYOUTS, hy4_layout.TP32_LAYOUT, dataclasses.replace(tp32, blockers=()))
    assert adapter.model_config_for(mesh) is Hy4TP32Config
    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, hy4_layout.TP32_LAYOUT)
    assert adapter.model_config_for(mesh) is Hy4TP32Config


def test_2d_cache_names_come_from_the_layout():
    sc1 = Mlp2DLayout((8, 4), 1)
    assert cache_names(sc1) == ("gate_up_proj_8x4_h4ax1_i8ax0", "down_proj_8x4_h4ax1_i8ax0")
    # The first SC1 split (H on the 8-chip axis 0) and every other mesh / axis assignment get other names.
    layouts = [Mlp2DLayout(m, a) for m in ((8, 4), (4, 8), (4, 4), (2, 4)) for a in (0, 1)]
    assert len({cache_names(x) for x in layouts}) == len(layouts)
    assert cache_names(Mlp2DLayout((8, 4), 0))[0] == "gate_up_proj_8x4_h8ax0_i4ax1"


def test_2d_cache_files_and_stamp(tmp_path, expect_error):
    from models.demos.deepseek_v3_d_p.tt.moe.tt_mlp_2d import TtMlp2D
    from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

    prefix = "layer_1.shared_expert"
    sc1 = Mlp2DLayout((8, 4), 1)

    def reset(*names):
        for f in tmp_path.iterdir():
            f.unlink()
        for n in names:
            (tmp_path / f"{prefix}.{n}_dtype_BFLOAT8_B_layout_TILE.tensorbin").touch()
        init_checker(tmp_path)

    reset(*cache_names(sc1))
    guard = cache_guard(tmp_path, prefix, sc1)
    # Shards without a stamp: not complete, and loading them refuses.
    assert not TtMlp2D.check_cache_complete(tmp_path, prefix, sc1)
    with expect_error(RuntimeError, "no layout stamp"):
        guard.check()
    guard.write()
    assert TtMlp2D.check_cache_complete(tmp_path, prefix, sc1)
    assert not TtMlp2D.check_cache_complete(tmp_path, prefix, Mlp2DLayout((8, 4), 0))
    assert not TtSharedExpert.check_cache_complete(tmp_path, prefix)
    # A stamp written for other content (here another dtype) under the same name refuses.
    import ttnn

    other = cache_guard(tmp_path, prefix, sc1, ttnn.bfloat4_b)
    assert other.name == guard.name and not TtMlp2D.check_cache_complete(tmp_path, prefix, sc1, ttnn.bfloat4_b)
    with expect_error(RuntimeError, "records layout"):
        other.check()
    # The first SC1 split's files (gate_up_proj_tp2d / down_proj_tp2d): never complete for SC1 now.
    reset("gate_up_proj_tp2d", "down_proj_tp2d")
    assert not TtMlp2D.check_cache_complete(tmp_path, prefix, sc1)
    # The 1D files: complete for TtSharedExpert only.
    reset("gate_proj", "up_proj", "down_proj")
    assert TtSharedExpert.check_cache_complete(tmp_path, prefix)
    assert not TtMlp2D.check_cache_complete(tmp_path, prefix, sc1)


# ---------------------------------------------------------------------------------------------- DRAM per chip
BYTES_PER_ELEMENT = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}  # BFP: 16-value blocks, 1 shared exp


def ffn_dram_per_chip(cfg=Hy4Config, chips=32, experts="bfp8", dense="bfp8", ffn_replication=1, tp=4) -> dict:
    """Bytes per chip of the FFN / MoE weights on SC1 (SP 8 x TP 4 on the 8x4 mesh). ``ffn_replication``: how many
    chips hold each shared-expert / dense-FFN shard (1 = 2D TP; 8 = TP 4 replicated over the 8 SP rows, the 1D
    modules)."""
    H, I_moe, I_dense = cfg.EMB_SIZE, cfg.MOE_INTERMEDIATE_SIZE, cfg.INTERMEDIATE_SIZE
    moe_layers = cfg.NUM_LAYERS - cfg.NUM_DENSE_LAYERS
    routed = moe_layers * cfg.NUM_ROUTED_EXPERTS * 3 * H * I_moe / chips * BYTES_PER_ELEMENT[experts]
    shared = moe_layers * cfg.NUM_SHARED_EXPERTS * 3 * H * I_moe * ffn_replication / chips * BYTES_PER_ELEMENT[dense]
    dense_ffn = cfg.NUM_DENSE_LAYERS * 3 * H * I_dense * ffn_replication / chips * BYTES_PER_ELEMENT[dense]
    # Router: bf16 [E, H], hidden over the 4-chip TP axis, replicated over the 8 SP rows (TtMoEGatePrefill).
    router = moe_layers * cfg.NUM_ROUTED_EXPERTS * H * 2.0 / tp
    return {"routed": routed, "shared": shared, "dense": dense_ffn, "router": router}


def test_ffn_dram_per_chip_sc1():
    """#65884 budgets 25.9 GB per chip for all weights with BFP8 experts (14.1 GB with BFP4). The routed experts are
    almost all of it; 2D TP cuts the shared expert + dense FFN 8x against the TP-only modules."""
    gb = 1e9
    bfp8 = ffn_dram_per_chip(experts="bfp8")
    bfp4 = ffn_dram_per_chip(experts="bfp4")
    one_d = ffn_dram_per_chip(experts="bfp8", ffn_replication=8)
    assert abs(bfp8["routed"] / gb - 24.71) < 0.01  # 77 x 256 x 3 x 6144 x 2048 / 32 at 1.0625 B
    assert abs(bfp4["routed"] / gb - 13.08) < 0.01
    assert abs((bfp8["shared"] + bfp8["dense"]) / gb - 0.108) < 0.001
    assert abs((one_d["shared"] + one_d["dense"]) / gb - 0.862) < 0.001
    assert sum(bfp8.values()) / gb < 25.9 and sum(bfp4.values()) / gb < 14.1


def test_router_bias_stays_fp32_for_hy4(tmp_path):
    """Hy4 keeps the checkpoint's fp32 e_score_correction_bias on device (MOE_GATE_BIAS_FP32); GLM-5.3 keeps bf16. The
    fp32 bias caches under its own name, so a bf16 bias cache never satisfies the fp32 check."""
    import ttnn
    from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import (
        TtMoEGateConfig,
        TtMoEGatePrefill,
        gate_bias_cache_name,
        gate_bias_fp32,
    )
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

    assert gate_bias_fp32(Hy4Config) and gate_bias_fp32(Hy4TP32Config) and not gate_bias_fp32(GLM53Config)
    assert TtMoEGateConfig.from_model_cfg(Hy4Config).bias_dtype == ttnn.float32
    assert TtMoEGateConfig.from_model_cfg(GLM53Config).bias_dtype == ttnn.bfloat16
    assert gate_bias_cache_name(True, True) == "e_score_correction_bias_centered_fp32"
    assert gate_bias_cache_name(True) == "e_score_correction_bias_centered"
    prefix = "layer_1.gate"
    for name in ("weight_dtype_BFLOAT16_layout_TILE", "e_score_correction_bias_centered_dtype_BFLOAT16_layout_TILE"):
        (tmp_path / f"{prefix}.{name}.tensorbin").touch()
    init_checker(tmp_path)
    assert TtMoEGatePrefill.check_cache_complete(tmp_path, prefix, bias_centered=True)
    assert not TtMoEGatePrefill.check_cache_complete(tmp_path, prefix, bias_centered=True, bias_fp32=True)
    (tmp_path / f"{prefix}.e_score_correction_bias_centered_fp32_dtype_FLOAT32_layout_TILE.tensorbin").touch()
    init_checker(tmp_path)
    assert TtMoEGatePrefill.check_cache_complete(tmp_path, prefix, bias_centered=True, bias_fp32=True)
    # The rounding fp32 avoids: a typical Hy4 bias magnitude in bf16 loses ~1e-4.
    b = torch.tensor([7.123456e-2, -3.31e-2, 1.0e-3])
    assert (b.to(torch.bfloat16).float() - b).abs().max() > 1e-5
