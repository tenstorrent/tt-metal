# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: device iHC (tt/ihc.py) vs the pure-torch ground truth (reference/ihc.py), tenstorrent/tt-metal#65891.

Each piece is checked independently so a failure localises immediately:
    expand | project | hc_pre | hc_post | head | the wrapped sublayer | a stack of layers | the
    checkpoint's own parameters,
on one chip, then on the 8x4 Galaxy with the hidden dimension split across a mesh axis: the 4-chip
axis (the SC1 layout) and the 8-chip axis (the same code on the other axis).

Precision note: the one lossy step is the projection matmul mixes = RMSNorm(X) @ fn^T, a
reduction over n*C = 24576 terms. In the real model it is squashed by the sigmoid and the small
scale (~0.01), so end-to-end pieces are checked at C6144 with that scale and at C256 with an
aggressive 1.0.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCHead, IHCWrap, ihc_expand
from models.experimental.hy4_preview.tests.common import GALAXY_DEVICE_PARAMS, GALAXY_MESH, check_pcc, hy4_tensors
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.ihc import TtIHCHead, TtIHCWrap
from models.experimental.hy4_preview.tt.ihc import ihc_expand as tt_ihc_expand

PCC = 0.999
PCC_PROJ = 0.998  # the projection matmul alone, before the sigmoid squashes its error

C_HY4 = IHCConfig().dim  # 6144
_E2E = [(256, 1.0), (C_HY4, 0.01)]
_E2E_IDS = ["C256-s1.0", f"C{C_HY4}-s0.01"]


def _params(cfg, scale_val, seed):
    """Shared parameters for reference and device. The two scales differ so a pre/post swap shows."""
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(cfg.mix_hc, cfg.n * cfg.dim, generator=g) * 0.02
    base = torch.randn(cfg.mix_hc, generator=g)  # non-zero to exercise the biases
    scale = torch.tensor([float(scale_val), 0.5 * float(scale_val)])
    return fn, base, scale


def _head_params(cfg, scale_val, seed):
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(cfg.n, cfg.n * cfg.dim, generator=g) * 0.02
    base = torch.randn(cfg.n, generator=g)
    return fn, base, torch.full((1,), float(scale_val))


def _ref(module, fn, base, scale):
    module.fn.data, module.base.data, module.scale.data = fn, base, scale
    return module.eval()


def _ref_mixes(cfg, fn, x):
    T = x.shape[1]
    xf = x.reshape(T, cfg.n * cfg.dim)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.norm_eps)
    return F.linear(xf, fn) * rsqrt


def _up(plan, t, mesh_mapper=None):
    return ttnn.from_torch(
        t.float().contiguous(),
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=ttnn.float32,
        mesh_mapper=mesh_mapper,
    )


def _up_x(plan, x):
    """Reference streams [1,T,n,C] -> the device packing, per chip [1,1,T,n*C/tp]: each chip's own
    hidden slice of every stream.

    The hidden split cuts the last dim contiguously, so the columns are reordered chip-major-then-
    stream. On one chip this is the row-major reshape.
    """
    _, T, n, C = x.shape
    tp = plan.hidden_tp
    packed = x.reshape(1, T, n, tp, C // tp).permute(0, 1, 3, 2, 4).reshape(1, 1, T, n * C)
    return _up(plan, packed, plan.shard_hidden(3))


def _up_hidden(plan, t):
    """[1,T,C] -> per chip [1,1,T,C/tp]."""
    _, T, C = t.shape
    return _up(plan, t.reshape(1, 1, T, C), plan.shard_hidden(3))


def _down_x(plan, t, n):
    """Device streams back in the reference order [1,T,n,C]."""
    d = plan.to_torch_hidden_concat(t, dim=-1)  # [1,1,T,tp*n*C/tp], stream-major within a chip
    T = d.shape[-2]
    return d.reshape(1, T, plan.hidden_tp, n, -1).permute(0, 1, 3, 2, 4).reshape(1, T, n, -1)


def _down_hidden(plan, t):
    """Device single stream back as [1,T,C]."""
    d = plan.to_torch_hidden_concat(t, dim=-1)
    return d.reshape(1, d.shape[-2], -1)


# ---------------------------------------------------------------------------------------------
# One chip
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
def test_ihc_expand(device, T):
    torch.manual_seed(0)
    plan = MeshPlan(device, hidden_axis=None)
    C, n = 256, 4
    h = torch.randn(1, T, C)
    d_out = tt_ihc_expand(_up_hidden(plan, h), n)
    check_pcc("expand", ihc_expand(h, n), _down_x(plan, d_out, n), PCC)


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C", [256, C_HY4], ids=["C256", f"C{C_HY4}"])
def test_project(device, T, C):
    torch.manual_seed(0)
    plan = MeshPlan(device, hidden_axis=None)
    cfg = IHCConfig(dim=C)
    fn, base, scale = _params(cfg, 1.0, seed=2)
    x = torch.randn(1, T, cfg.n, C)

    d_mixes = TtIHCWrap(plan, cfg, fn, base, scale).project(_up_x(plan, x))
    check_pcc("mixes", _ref_mixes(cfg, fn, x), plan.to_torch_replicated(d_mixes), PCC_PROJ)


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C, scale_val", _E2E, ids=_E2E_IDS)
def test_hc_pre(device, T, C, scale_val):
    torch.manual_seed(0)
    plan = MeshPlan(device, hidden_axis=None)
    cfg = IHCConfig(dim=C)
    fn, base, scale = _params(cfg, scale_val, seed=3)
    x = torch.randn(1, T, cfg.n, C)

    r_y, r_post = _ref(IHCWrap(cfg), fn, base, scale).hc_pre(x)  # [1,T,C], [1,T,n]
    d_y, d_post = TtIHCWrap(plan, cfg, fn, base, scale).hc_pre(_up_x(plan, x))
    check_pcc("y", r_y, _down_hidden(plan, d_y), PCC)
    check_pcc("post", r_post, plan.to_torch_replicated(d_post), PCC)


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C", [256, C_HY4], ids=["C256", f"C{C_HY4}"])
def test_hc_post(device, T, C):
    torch.manual_seed(0)
    plan = MeshPlan(device, hidden_axis=None)
    cfg = IHCConfig(dim=C)
    fn, base, scale = _params(cfg, 1.0, seed=4)
    x = torch.randn(1, T, cfg.n, C)
    f_out = torch.randn(1, T, C)
    post = cfg.magnitude * torch.sigmoid(torch.randn(1, T, cfg.n)) + cfg.eps

    r_out = _ref(IHCWrap(cfg), fn, base, scale).hc_post(f_out, x, post)  # [1,T,n,C]
    d_out = TtIHCWrap(plan, cfg, fn, base, scale).hc_post(
        _up_hidden(plan, f_out), _up_x(plan, x), _up(plan, post.reshape(1, 1, T, cfg.n))
    )
    check_pcc("out", r_out, _down_x(plan, d_out, cfg.n), PCC)


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C, scale_val", _E2E, ids=_E2E_IDS)
def test_hc_head(device, T, C, scale_val):
    torch.manual_seed(0)
    plan = MeshPlan(device, hidden_axis=None)
    cfg = IHCConfig(dim=C)
    fn, base, scale = _head_params(cfg, scale_val, seed=5)
    x = torch.randn(1, T, cfg.n, C)

    r_y = _ref(IHCHead(cfg), fn, base, scale)(x)  # [1,T,C]
    d_y = TtIHCHead(plan, cfg, fn, base, scale)(_up_x(plan, x))
    check_pcc("head_y", r_y, _down_hidden(plan, d_y), PCC)


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C, scale_val", _E2E, ids=_E2E_IDS)
@pytest.mark.parametrize("f_kind", ["identity", "linear"])
def test_ihc_wrap(device, T, C, scale_val, f_kind):
    """hc_pre -> F -> hc_post, device vs reference."""
    torch.manual_seed(0)
    plan = MeshPlan(device, hidden_axis=None)
    cfg = IHCConfig(dim=C)
    fn, base, scale = _params(cfg, scale_val, seed=7)
    x = torch.randn(1, T, cfg.n, C)

    if f_kind == "identity":  # isolates the iHC wiring from F's own arithmetic
        ref_f = dev_f = lambda z: z
    else:
        w = torch.randn(C, C, generator=torch.Generator().manual_seed(8)) * 0.02
        ref_f = lambda z: F.linear(z, w)  # z @ w.T
        w_t_tt = _up(plan, w.t())
        dev_f = lambda z: ttnn.matmul(z, w_t_tt)

    ref_out = _ref(IHCWrap(cfg), fn, base, scale)(x, ref_f)  # [1,T,n,C]
    d_out = TtIHCWrap(plan, cfg, fn, base, scale)(_up_x(plan, x), dev_f)
    check_pcc(f"wrap[{f_kind}] C={C} T={T}", ref_out, _down_x(plan, d_out, cfg.n), PCC)


@pytest.mark.parametrize("C, scale_val", _E2E, ids=_E2E_IDS)
@pytest.mark.parametrize("layers", [6], ids=["L6"])
def test_ihc_stack(device, C, scale_val, layers):
    """expand -> ``layers`` x (attention site, FFN site) -> head: the 4-stream residual across stacked layers.

    Layers 0-5 are the slice #65893 validates. Every sublayer is the same norm-preserving linear,
    so what accumulates is the iHC arithmetic and nothing else.
    """
    torch.manual_seed(0)
    plan = MeshPlan(device, hidden_axis=None)
    T = 32
    cfg = IHCConfig(dim=C)
    w = torch.randn(C, C, generator=torch.Generator().manual_seed(8)) * (0.5 / C**0.5)
    w_t_tt = _up(plan, w.t())
    ref_f = lambda z: F.linear(z, w)
    dev_f = lambda z: ttnn.matmul(z, w_t_tt)

    emb = torch.randn(1, T, C)
    r_x = ihc_expand(emb, cfg.n)
    d_x = tt_ihc_expand(_up_hidden(plan, emb), cfg.n)
    for layer in range(layers):
        for site in range(2):
            fn, base, scale = _params(cfg, scale_val, seed=100 + 2 * layer + site)
            r_x = _ref(IHCWrap(cfg), fn, base, scale)(r_x, ref_f)
            d_x = TtIHCWrap(plan, cfg, fn, base, scale)(d_x, dev_f)
        check_pcc(f"stack layer {layer}", r_x, _down_x(plan, d_x, cfg.n), PCC)

    fn, base, scale = _head_params(cfg, scale_val, seed=99)
    r_y = _ref(IHCHead(cfg), fn, base, scale)(r_x)
    d_y = TtIHCHead(plan, cfg, fn, base, scale)(d_x)
    check_pcc("stack head", r_y, _down_hidden(plan, d_y), PCC)


# ---------------------------------------------------------------------------------------------
# The checkpoint's own parameters
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer", range(6), ids=lambda layer: f"layer{layer}")
@pytest.mark.parametrize("site", ["hc_attn_layer", "hc_mlp_layer"], ids=["attn", "mlp"])
def test_ihc_wrap_real_weights(device, layer, site):
    """Layers 0-5 with the checkpoint's fn / base / scale, where the logits sit where training left
    them rather than where a random init puts them."""
    torch.manual_seed(0)
    plan = MeshPlan(device, hidden_axis=None)
    T, cfg = 32, IHCConfig()
    prefix = f"model.layers.{layer}.{site}.hc_pre."
    fn, base, scale = hy4_tensors([prefix + "hc_fn", prefix + "hc_base", prefix + "hc_scale"])
    x = torch.randn(1, T, cfg.n, cfg.dim)
    f_out = torch.randn(1, T, cfg.dim)

    ref = _ref(IHCWrap(cfg), fn, base, scale)
    r_y, r_post = ref.hc_pre(x)
    r_out = ref.hc_post(f_out, x, r_post)

    wrap = TtIHCWrap(plan, cfg, fn, base, scale)
    d_x = _up_x(plan, x)
    d_y, d_post = wrap.hc_pre(d_x)
    d_out = wrap.hc_post(_up_hidden(plan, f_out), d_x, d_post)
    check_pcc(f"real L{layer} {site} y", r_y, _down_hidden(plan, d_y), PCC)
    check_pcc(f"real L{layer} {site} post", r_post, plan.to_torch_replicated(d_post), PCC)
    check_pcc(f"real L{layer} {site} out", r_out, _down_x(plan, d_out, cfg.n), PCC)


def test_hc_head_real_weights(device):
    torch.manual_seed(0)
    plan = MeshPlan(device, hidden_axis=None)
    T, cfg = 32, IHCConfig()
    prefix = "model.hc_head.hc_head_"
    fn, base, scale = hy4_tensors([prefix + "fn", prefix + "base", prefix + "scale"])
    x = torch.randn(1, T, cfg.n, cfg.dim)

    r_y = _ref(IHCHead(cfg), fn, base, scale)(x)
    d_y = TtIHCHead(plan, cfg, fn, base, scale)(_up_x(plan, x))
    check_pcc("real head_y", r_y, _down_hidden(plan, d_y), PCC)


# ---------------------------------------------------------------------------------------------
# 8x4 Galaxy: hidden split across a mesh axis
# ---------------------------------------------------------------------------------------------
galaxy = [
    pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True),
    pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True),
    # 1: hidden over the 4-chip axis, replicated across the 8-chip axis (SC1). 0: the other axis.
    pytest.mark.parametrize("hidden_axis", [1, 0], ids=["axis1-4chips-sc1", "axis0-8chips"]),
]


def _galaxy(fn):
    for mark in reversed(galaxy):
        fn = mark(fn)
    return fn


@_galaxy
@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C", [256, C_HY4], ids=["C256", f"C{C_HY4}"])
def test_project_hidden_split(mesh_device, hidden_axis, T, C):
    """Each chip's matmul is a partial inner product and its squares a partial sum; one all-reduce
    carries both."""
    torch.manual_seed(0)
    plan = MeshPlan(mesh_device, hidden_axis=hidden_axis)
    cfg = IHCConfig(dim=C)
    fn, base, scale = _params(cfg, 1.0, seed=2)
    x = torch.randn(1, T, cfg.n, C)
    r_mixes = _ref_mixes(cfg, fn, x)

    d_mixes = TtIHCWrap(plan, cfg, fn, base, scale).project(_up_x(plan, x))
    shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(d_mixes)]
    # The all-reduce leaves the same result on every chip of a hidden group; the groups hold replicas.
    for i, s in enumerate(shards[1:], start=1):
        check_pcc(f"mixes chip {i} vs chip 0", shards[0], s, 0.99999)
    check_pcc(f"mixes C={C}", r_mixes, shards[0], PCC_PROJ)
    # PCC is scale-invariant, so it cannot see a wrong RMS divisor: a chip-local width would be off
    # by sqrt(tp) here. The bound is the device rsqrt's bias.
    ratio = (shards[0].float().norm() / r_mixes.float().norm()).item()
    assert abs(ratio - 1.0) < 5e-3, f"scale drift {ratio:.6f}: the RMS divisor must be the global width"


@_galaxy
@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C, scale_val", _E2E, ids=_E2E_IDS)
def test_ihc_wrap_hidden_split(mesh_device, hidden_axis, T, C, scale_val):
    """The wrapped sublayer with every stream split across chips. F is elementwise, so it is exact
    on a hidden slice and the check is on the iHC wiring across chips."""
    torch.manual_seed(0)
    plan = MeshPlan(mesh_device, hidden_axis=hidden_axis)
    cfg = IHCConfig(dim=C)
    fn, base, scale = _params(cfg, scale_val, seed=7)
    x = torch.randn(1, T, cfg.n, C)

    ref_out = _ref(IHCWrap(cfg), fn, base, scale)(x, lambda z: 0.5 * z)  # [1,T,n,C]
    d_out = TtIHCWrap(plan, cfg, fn, base, scale)(_up_x(plan, x), lambda z: ttnn.multiply(z, 0.5))
    check_pcc(f"wrap C={C} T={T}", ref_out, _down_x(plan, d_out, cfg.n), PCC)


@_galaxy
@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C", [256, C_HY4], ids=["C256", f"C{C_HY4}"])
def test_hc_head_hidden_split(mesh_device, hidden_axis, T, C):
    """The mixes are all-reduced, the weighted sum stays per chip, so each chip ends with its own
    hidden slice of the collapsed stream."""
    torch.manual_seed(0)
    plan = MeshPlan(mesh_device, hidden_axis=hidden_axis)
    cfg = IHCConfig(dim=C)
    # A unit scale, so the projection is most of each sigmoid's argument and a wrong reduction shows.
    fn, base, scale = _head_params(cfg, 1.0, seed=5)
    x = torch.randn(1, T, cfg.n, C)

    r_y = _ref(IHCHead(cfg), fn, base, scale)(x)  # [1,T,C]
    d_y = TtIHCHead(plan, cfg, fn, base, scale)(_up_x(plan, x))
    check_pcc(f"head_y C={C}", r_y, _down_hidden(plan, d_y), PCC)
