# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit test of ``custom_ops/mhc_pre_block`` (the fused exact fp32 mHC pre-block, one program on the grid) on real
layer-4 parameters: random streams and real dump rows, decode shapes B = 1/4/32 on the padded 32-row tile, against
the float64 HF math (hn within the bf16 output rounding; post / comb lane tiles vs the standalone Sinkhorn kernel
fed the float64 mixes, and vs float64), plus a trace capture + replay.

    GLM53_MHC_POST_KERNEL=2 TT_VISIBLE_DEVICES=1 pytest models/demos/glm53_flash/tests/custom_ops/test_mhc_pre_block.py -s
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.glm53_flash.custom_ops.mhc_pre_block.op import MhcPreBlock
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.custom_ops.test_mhc_post_mix import _err, _up, expand_x8
from models.demos.glm53_flash.tests.test_mhc import _streams
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mhc import W, _upload, build_consts, mhc_kernel_config


def _build(mesh_device, which="ffn", exact=False):
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    kcfg = mhc_kernel_config(cfg)
    raw = load_layer_raw(snapshot, 4, shard_headers(snapshot))
    fn, base, scale = raw[f"hc_{which}_fn"].float(), raw[f"hc_{which}_base"].float(), raw[f"hc_{which}_scale"].float()
    consts = _upload(mesh_device, build_consts(kcfg, scale, base), (8, W, W), ttnn.float32)
    g = torch.Generator().manual_seed(5)
    w_norm = 0.8 + 0.4 * torch.rand(kcfg.dim, generator=g)
    op = MhcPreBlock(
        mesh_device,
        fn,
        consts,
        w_norm.to(torch.bfloat16).float(),
        n=kcfg.n,
        C=kcfg.dim,
        sinkhorn_iters=kcfg.sinkhorn_iters,
        eps=kcfg.eps,
        norm_eps=kcfg.norm_eps,
        rms_eps=float(cfg.rms_norm_eps),
        exact=exact,
    )
    return op, kcfg, (fn, base, scale)


def _check(mesh_device, op, kcfg, params, x: torch.Tensor, rows: int, tag: str):
    fn, base, scale = params
    n, C = kcfg.n, kcfg.dim
    xd = _up(mesh_device, x.reshape(1, 1, 32, n * C))
    hn, post, comb = op.allocate_outputs(xd)
    op(xd, hn, post, comb)
    r_mixes, r_pre, r_post, r_comb, r_h, r_hn = op.golden(x, fn, base, scale)
    hn_t = ttnn.to_torch(hn)[0, 0].double()[:rows]
    post_t, comb_t = ttnn.to_torch(post).double()[:rows], ttnn.to_torch(comb).double()[:rows]
    # the lane-layout coefficient tiles vs float64 (the Sinkhorn kernel's own precision) ...
    post_l, comb_l = expand_x8(r_post[:rows].float(), r_comb[:rows].float(), n)
    e_post = _err(f"{tag} post lane vs f64", post_l, post_t)
    e_comb = _err(f"{tag} comb lane vs f64", comb_l, comb_t)
    # ... and vs the standalone Sinkhorn kernel on the float64 mixes (isolates the fused projection)
    md = _up(mesh_device, torch.nn.functional.pad(r_mixes[:rows].float(), (0, 8)).reshape(rows, 32))
    s_pre_d, s_post, s_comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(md, op.consts, n, op.iters, op.eps)
    s_post_l, s_comb_l = expand_x8(ttnn.to_torch(s_post)[:, :n].float(), ttnn.to_torch(s_comb)[:, : n * n].float(), n)
    e_post_k = _err(f"{tag} post lane vs Sinkhorn kernel on f64 mixes", s_post_l, post_t)
    e_comb_k = _err(f"{tag} comb lane vs Sinkhorn kernel on f64 mixes", s_comb_l, comb_t)
    # hn vs float64 (bf16 output) and, isolating the exact mix + norm from the Sinkhorn kernel's own rounding,
    # vs the bf16 rounding of the norm of h built from the kernel's pre on the float64 mixes
    e_hn = _err(f"{tag} hn vs f64", r_hn[:rows], hn_t)
    s_pre = ttnn.to_torch(s_pre_d)[:, :n].double()
    h_k = (s_pre[:, :, None] * x.double()[:rows].reshape(rows, n, C)).sum(1)
    w = ttnn.to_torch(op.w)[0, 0, 0].double()
    hn_k = h_k * torch.rsqrt(h_k.square().mean(-1, keepdim=True) + op.rms_eps) * w
    e_hn_k = _err(f"{tag} hn vs bf16(norm(h from the kernel's pre))", hn_k.to(torch.bfloat16).double(), hn_t)
    # the FPU projection: mixes are composite-class (~5e-5 vs f64), the Sinkhorn amplifies that ~4x; everything
    # after the coefficients is exact fp32 and the bf16 output differs from the reference's rounding in a few ulps
    assert e_post < 1e-2 and e_comb < 1e-2, "coefficients far from the HF math"
    assert e_post_k < 2e-3 and e_comb_k < 2e-3, "coefficients differ from the Sinkhorn kernel beyond the mixes rounding"
    assert e_hn < 5e-3 and e_hn_k < 3e-3, "hn beyond bf16 rounding"
    return hn, post, comb


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("B", [1, 4, 32])
def test_pre_block_real(mesh_device, exact, B):
    op, kcfg, params = _build(mesh_device, exact=exact)
    x = _streams(32, offset=100)  # 32 real rows; rows past B are pad rows the model ignores
    x[B:] = 0.1 * torch.randn(32 - B, x.shape[1], generator=torch.Generator().manual_seed(B))
    _check(mesh_device, op, kcfg, params, x, B, f"{'exact' if exact else 'fpu'} real B={B}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_pre_block_random(mesh_device):
    op, kcfg, params = _build(mesh_device, which="attn")
    g = torch.Generator().manual_seed(2)
    x = torch.randn(32, kcfg.n * kcfg.dim, generator=g)
    _check(mesh_device, op, kcfg, params, x, 32, "random")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16 << 20}], indirect=True)
def test_pre_block_trace(mesh_device):
    op, kcfg, params = _build(mesh_device)
    n, C = kcfg.n, kcfg.dim
    x = _streams(32, offset=300)
    xd = _up(mesh_device, x.reshape(1, 1, 32, n * C))
    hn, post, comb = op.allocate_outputs(xd)
    op(xd, hn, post, comb)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op(xd, hn, post, comb)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    x2 = _streams(32, offset=700)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(x2.reshape(1, 1, 32, n * C).contiguous(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32), xd
    )
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    fn, base, scale = params
    _, _, _, _, _, r_hn = op.golden(x2, fn, base, scale)
    rel = _err("trace replay hn vs bf16(f64)", r_hn.to(torch.bfloat16).double(), ttnn.to_torch(hn)[0, 0].double())
    ttnn.release_trace(mesh_device, tid)
    assert rel < 5e-3  # the float64 reference carries the Sinkhorn kernel's own ~2e-3 (see _check for the exact bars)
