# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the device cases of test_ihc_site.py run against the page-level host model (host_model.py) in place
of the chip, for both SFPU lane layouts. What this guards is everything around the kernels: the page layouts
(layout.py), fn^T and the constant page, the decoders of the coefficient / statistic / debug pages, and the checks
of the device test themselves. The ttnn form is stood in for by the reference."""

import types

import pytest
import torch

import ttnn
from models.experimental.hy4_preview.fused.ihc_site import layout
from models.experimental.hy4_preview.fused.ihc_site.host_model import HostSite
from models.experimental.hy4_preview.fused.ihc_site.op import golden_post, golden_pre
from models.experimental.hy4_preview.fused.ihc_site.tests import test_ihc_site as dev
from models.experimental.hy4_preview.reference.ihc import IHCConfig


class _Project:
    def __init__(self, cfg, fn):
        self.cfg, self.fn = cfg, fn

    def project(self, x):  # [1, 1, T, n C] -> [1, 1, T, 2 n]
        flat = x[0, 0].double()
        rs = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.cfg.norm_eps)
        return ((flat @ self.fn.double().t()) * rs).float().reshape(1, 1, flat.shape[0], -1)


class HostTtForm:
    """Stands in for the ttnn form: the reference, in the shapes the ttnn modules return."""

    def __init__(self, device, cfg, fn, base, scale):
        self.cfg, self.params = cfg, (fn, base, scale)
        self.wrap = _Project(cfg, fn)

    def up(self, t):
        return t.float()

    def up_x(self, x):
        return x.reshape(1, 1, x.shape[0], -1).float()

    def up_rows(self, v):
        return v.reshape(1, 1, *v.shape).float()

    def pre(self, d_x):
        T = d_x.shape[2]
        y, _, post, stat = golden_pre(self.cfg, *self.params, d_x[0, 0].reshape(T, self.cfg.n, -1))
        return y.reshape(1, 1, T, -1), post.reshape(1, 1, T, -1), stat.reshape(1, 1, T, 1)

    def post(self, d_f, d_x, d_post):
        T = d_x.shape[2]
        out = golden_post(self.cfg, d_x[0, 0].reshape(T, self.cfg.n, -1), d_f[0, 0], d_post[0, 0])
        return out.reshape(1, 1, T, -1)

    def pre_outputs(self, x):
        y, pre, post, stat = golden_pre(self.cfg, *self.params, x)
        return {"1/rms": stat, "pre": pre, "post": post, "y": y}


MM32 = {"on": False}


@pytest.fixture(params=["evenodd", "half"])
def host(request, monkeypatch):
    """test_ihc_site with the host model for the chip and SFPU lanes laid out as ``request.param``."""
    lanes = request.param
    monkeypatch.setattr(layout, "SFPU_LANES", lanes)
    monkeypatch.setattr(dev, "MOCK", False)
    monkeypatch.setattr(dev, "IhcSite", lambda device, cfg, **kw: HostSite(None, cfg, **kw))
    monkeypatch.setattr(dev, "TtForm", HostTtForm)
    monkeypatch.setattr(dev, "_probe", lambda device: {"lanes": lanes, "unpack_high": True, "mm32": MM32["on"]})
    monkeypatch.setattr(
        dev,
        "ttnn",
        types.SimpleNamespace(
            to_torch=lambda t: t, deallocate=lambda t: None, float32=ttnn.float32, bfloat16=ttnn.bfloat16
        ),
    )
    return lanes


def test_layout_round_trips():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 4, 256, generator=g)
    v = torch.randn(5, 256, generator=g)
    assert torch.equal(layout.pages_to_streams(layout.streams_to_pages(x), 5), x)
    assert torch.equal(layout.pages_to_rows(layout.rows_to_pages(v), 5), v)
    # the rows pages are the bytes of an [8, C] tensor of [8, 32] tiles: tile t = (face 0: columns 32 t .. + 15,
    # face 1: columns 32 t + 16 .. + 31), 8 rows each
    tiles = layout._pad_rows(v).reshape(8, 8, 2, 16).permute(1, 2, 0, 3).reshape(-1)
    pages = layout.rows_to_pages(v)  # [32, 64]: 2 pages of 4 faces of [16, 16]
    flat = pages.reshape(2, 16, 2, 2, 16).permute(2, 0, 3, 1, 4).reshape(-1)  # page, face, row, column
    assert torch.equal(flat, tiles)
    for lanes in ("half", "evenodd"):
        pre, post = torch.rand(5, 4, generator=g), torch.rand(5, 4, generator=g)
        back = layout.coef_from_page(layout.coef_to_page(pre, post, lanes), 5, lanes)
        assert torch.equal(back[0], pre) and torch.equal(back[1], post)


def test_probe(host):
    dev.test_probe_lanes(None)


@pytest.mark.parametrize("T", [1, 8])
@pytest.mark.parametrize("C, K", [(256, 1), (256, 2), (512, 4)])
def test_pre(host, C, K, T):
    dev.test_pre(None, C, K, "random", T)


@pytest.mark.parametrize("f_dtype", [ttnn.float32, ttnn.bfloat16], ids=["F-fp32", "F-bf16"])
@pytest.mark.parametrize("C, K", [(256, 1), (256, 2)])
def test_post(host, C, K, f_dtype):
    dev.test_post(None, C, K, 8, f_dtype)


@pytest.mark.parametrize("y_dtype", [ttnn.float32, ttnn.bfloat16], ids=["y-fp32", "y-bf16"])
@pytest.mark.parametrize("C, K", [(256, 1), (256, 2)])
def test_post_pre(host, C, K, y_dtype):
    dev.test_post_pre(None, C, K, "random", 8, y_dtype)


def test_projection(host):
    dev.test_precision_projection(None, 256, 1)


def test_hy4_width(host):
    """The Hy4 width on 8 cores with a checkpoint site (skipped without the checkpoint)."""
    dev.test_pre(None, IHCConfig().dim, 8, (0, "hc_attn_layer"), 8)
    dev.test_post_pre(None, IHCConfig().dim, 8, "random", 8, ttnn.float32)


def test_layers(host):
    if host == "half":
        pytest.skip("one lane layout is enough for the 12-site chain")
    dev.test_layers(None, 8, "fpu")


def test_root_exchange(host):
    dev.test_root_exchange(None, 256, 2)


def test_mm32_projection(host, monkeypatch):
    """The fallback projection's fn^T layout (faces of streams 1 and 2 swapped) through pre and post + pre."""
    monkeypatch.setitem(MM32, "on", True)
    dev.test_pre(None, 256, 2, "random", 8)
    dev.test_post_pre(None, 256, 1, "random", 8, ttnn.float32)


@pytest.mark.parametrize("proj", ["fpu", "sfpu"])
def test_site_dram(host, proj):
    dev.test_site_dram(None, 256, 2, proj)


@pytest.mark.parametrize("C, K", [(256, 1), (256, 2), (512, 4)])
def test_sfpu_projection(host, C, K):
    """The fn page layout of the SFPU projection through pre and post + pre."""
    dev.test_precision(None, C, K, "random", "sfpu")
    dev.test_post_pre_sfpu_projection(None, C, K, ttnn.float32)


def test_precision_fpu(host):
    dev.test_precision(None, 256, 2, "random", "fpu")
