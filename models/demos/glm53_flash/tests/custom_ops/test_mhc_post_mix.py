# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit test of ``custom_ops/mhc_post_mix`` (the fused exact fp32 mHC residual update, a model-local
``ttnn.generic_op``) against its float64 golden: random fp32 inputs at the model's shapes (decode B = 1/4/32 all
share the padded T = 32 tile row, prefill T = 256), real layer-4 streams with real Sinkhorn coefficients
(``TtMHCWrap.hc_pre``) compared with float64 and with the composite's exact fp32 ``hc_post`` chain, both output
modes (streams, collapse), and a trace capture + replay. The mesh cases need the chips; gate them with
``GLM53_MESH_TESTS=1``.

    TT_VISIBLE_DEVICES=1 pytest models/demos/glm53_flash/tests/custom_ops/test_mhc_post_mix.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import KERNEL_DIR, MhcPostMix
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mhc import TtMHCWrap, mhc_kernel_config

DUMP = Path("bringup/references/glm53_flash/R4097/layer4_prefill.pt")
MESH = pytest.mark.skipif(os.environ.get("GLM53_MESH_TESTS") != "1", reason="needs the mesh (GLM53_MESH_TESTS=1)")
MESHES = [(1, 1), pytest.param((1, 4), marks=MESH), pytest.param((2, 4), marks=MESH)]


# 1: FPU column broadcasts + SFPU binary ops; 2: resident operands, x8 coefficient tiles, SFPU FMAs (when its
# kernels are present)
HAS_V2 = (KERNEL_DIR / "compute_mhc_post_mix_v2.cpp").exists()
VERSIONS = [1] + ([2] if HAS_V2 else [])


def _up(device, t: torch.Tensor, dtype=ttnn.float32) -> ttnn.Tensor:
    return ttnn.from_torch(t.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)


def expand_x8(post: torch.Tensor, comb: torch.Tensor, n: int):
    """The v2 kernel's pre-expanded coefficient layout: post_x8 [T, 32] with post[:, j] in columns 8j..8j+7,
    comb_x8 [T, n*32] with comb[:, i*n+j] in columns 32j + 8i .. +7 (tile j = stream j)."""
    T = post.shape[0]
    post_x8 = torch.zeros(T, 32, dtype=post.dtype)
    comb_x8 = torch.zeros(T, n * 32, dtype=comb.dtype)
    for j in range(n):
        post_x8[:, 8 * j : 8 * j + 8] = post[:, j : j + 1]
        for i in range(n):
            comb_x8[:, 32 * j + 8 * i : 32 * j + 8 * i + 8] = comb[:, i * n + j : i * n + j + 1]
    return post_x8, comb_x8


def _err(name: str, ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.double().flatten(), got.double().flatten()
    d = ref - got
    rel = (d.norm() / ref.norm()).item()
    logger.info(f"  {name}: rel rms {rel:.3e} max|d|/max|ref| {(d.abs().max() / ref.abs().max()).item():.3e}")
    return rel


def _random_inputs(T: int, n: int, C: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, n * C, generator=g)
    y = torch.randn(T, C, generator=g)
    post = 2 * torch.rand(T, n, generator=g)
    comb = torch.softmax(torch.randn(T, n, n, generator=g), -1).reshape(T, n * n)
    return x, y, post, comb


def _run(op: MhcPostMix, device, x, y, post, comb, n, collapse, y_dtype=ttnn.float32, x8=False):
    T, C = y.shape
    xd, yd = _up(device, x.reshape(1, 1, T, n * C)), _up(device, y.reshape(1, 1, T, C), y_dtype)
    if x8:
        post, comb = expand_x8(post, comb, n)
    out = op.allocate_output(xd, collapse)
    op(xd, yd, _up(device, post), _up(device, comb), out, collapse)
    assert tuple(out.shape) == (1, 1, T, C if collapse else n * C), out.shape
    return out


@pytest.mark.parametrize("mesh_device", MESHES, indirect=True)
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("T", [32, 256])
@pytest.mark.parametrize("collapse", [False, True])
def test_post_mix_random(mesh_device, version, T, collapse):
    n, C = 4, 4096
    x, y, post, comb = _random_inputs(T, n, C)
    op = MhcPostMix(n, version=version)
    out = _run(op, mesh_device, x, y, post, comb, n, collapse)
    ref = op.golden(x, y, post, comb, n, collapse)
    for i, dev_out in enumerate(ttnn.get_device_tensors(out)):
        rel = _err(f"v{version} random T={T} collapse={collapse} dev{i}", ref, ttnn.to_torch(dev_out)[0, 0])
        assert rel < 1e-6


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("T", [32, 256])
@pytest.mark.parametrize("collapse", [False, True])
@pytest.mark.parametrize("y_dtype,x8", [(ttnn.bfloat16, False), (ttnn.float32, True), (ttnn.bfloat16, True)])
@pytest.mark.skipif(not HAS_V2, reason="v2 kernels not present")
def test_post_mix_v2_layouts(mesh_device, T, collapse, y_dtype, x8):
    """v2 input variants: the bf16 sublayer output read directly (y bf16-exact, widened on unpack) and the
    pre-expanded x8 coefficient tiles (what the fused pre-block will emit) instead of the raw Sinkhorn tiles."""
    n, C = 4, 4096
    x, y, post, comb = _random_inputs(T, n, C, seed=3)
    if y_dtype == ttnn.bfloat16:
        y = y.to(torch.bfloat16).float()
    op = MhcPostMix(n, version=2)
    out = _run(op, mesh_device, x, y, post, comb, n, collapse, y_dtype=y_dtype, x8=x8)
    ref = op.golden(x, y, post, comb, n, collapse)
    rel = _err(f"v2 T={T} collapse={collapse} y={y_dtype} x8={x8}", ref, ttnn.to_torch(out)[0, 0])
    assert rel < 1e-6


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("B", [1, 4, 32])
def test_post_mix_decode_batches(mesh_device, version, B):
    """Decode batches B = 1/4/32 all run on one padded 32-row tile; rows past B are pad rows the model ignores."""
    n, C = 4, 4096
    x, y, post, comb = _random_inputs(32, n, C, seed=B)
    x[B:] = 0
    op = MhcPostMix(n, version=version)
    for collapse in (False, True):
        out = _run(op, mesh_device, x, y, post, comb, n, collapse)
        ref = op.golden(x, y, post, comb, n, collapse)[:B]
        rel = _err(f"v{version} B={B} collapse={collapse}", ref, ttnn.to_torch(out)[0, 0, :B])
        assert rel < 1e-6


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 << 20}], indirect=True)
@pytest.mark.parametrize("version", VERSIONS)
def test_post_mix_trace(mesh_device, version):
    """The op inside a trace: capture one call on persistent inputs, replay it on fresh input values."""
    n, C, T = 4, 4096, 32
    op = MhcPostMix(n, version=version)
    x, y, post, comb = _random_inputs(T, n, C, seed=7)
    xd, yd = _up(mesh_device, x.reshape(1, 1, T, n * C)), _up(mesh_device, y.reshape(1, 1, T, C))
    pd, cd = _up(mesh_device, post), _up(mesh_device, comb)
    out = op.allocate_output(xd, False)
    op(xd, yd, pd, cd, out, False)  # warm (compiles)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op(xd, yd, pd, cd, out, False)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    x2, y2, post2, comb2 = _random_inputs(T, n, C, seed=8)
    for src, t in ((x2.reshape(1, 1, T, n * C), xd), (y2.reshape(1, 1, T, C), yd), (post2, pd), (comb2, cd)):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(src.float().contiguous(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32), t
        )
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    rel = _err(f"v{version} trace replay", op.golden(x2, y2, post2, comb2, n, False), ttnn.to_torch(out)[0, 0])
    ttnn.release_trace(mesh_device, tid)
    assert rel < 1e-6


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("collapse", [False, True])
def test_post_mix_real(mesh_device, version, collapse):
    """Real layer-4 streams (dump rows + 1e-3 noise -> fp32, not bf16-exact), real coefficients from the
    Sinkhorn kernel via the composite ``TtMHCWrap``; compared with float64 and with the composite's
    own exact fp32 ``hc_post`` chain."""
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    kcfg = mhc_kernel_config(cfg)
    raw = load_layer_raw(snapshot, 4, shard_headers(snapshot))
    wrap = TtMHCWrap(
        mesh_device, kcfg, raw["hc_ffn_fn"].float(), raw["hc_ffn_base"].float(), raw["hc_ffn_scale"].float()
    )
    n, C, T = kcfg.n, kcfg.dim, 32
    g = torch.Generator().manual_seed(1)
    x = torch.load(DUMP)["x_in"][0, 300 : 300 + T].reshape(T, n * C).float()
    x = x + torch.randn(x.shape, generator=g) * 1e-3 * x.abs()
    y = (torch.load(DUMP)["x_in"][0, 900 : 900 + T, 0].float() * 0.5).to(torch.bfloat16).float()
    xd, yd = _up(mesh_device, x.reshape(1, 1, T, n * C)), _up(mesh_device, y.reshape(1, 1, T, C))
    _, post, comb = wrap.hc_pre(xd)  # [1,1,T,n], [1,1,T,n*n] views of the kernel outputs
    post2, comb2 = ttnn.reshape(post, [T, n]), ttnn.reshape(comb, [T, n * n])
    op = MhcPostMix(n, version=version)
    out = op.allocate_output(xd, collapse)
    y_in = yd if version == 1 else _up(mesh_device, y.reshape(1, 1, T, C), ttnn.bfloat16)  # v2 reads bf16 y
    op(xd, y_in, post2, comb2, out, collapse)
    post_t, comb_t = ttnn.to_torch(post)[0, 0].float(), ttnn.to_torch(comb)[0, 0].float()
    ref = op.golden(x, y, post_t, comb_t, n, collapse)
    got = ttnn.to_torch(out)[0, 0]
    rel = _err(f"v{version} real collapse={collapse} vs float64", ref, got)
    assert rel < 1e-6
    comp = ttnn.to_torch(wrap.hc_post(yd, xd, post, comb))[0, 0]
    if collapse:
        comp = comp.reshape(T, n, C).mean(1)
    rel_c = _err(f"v{version} real collapse={collapse} vs composite hc_post", comp, got)
    assert rel_c < 1e-6
