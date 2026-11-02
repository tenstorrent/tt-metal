# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device A/B of the fused pre-block (``custom_ops/mhc_pre_block``) against the composite decode pre-block of
``TtMHCFast`` on EVERY chip of a mesh, at the real decode shapes and dtypes of the model (``x [1,1,32,n*C]`` fp32
TILE interleaved with B = 4 and 32 valid rows, bf16 ``hn``, fp32 lane-layout ``post`` / ``comb``), real layer-4
parameters and real stream rows (decode ledger A1; _agents/mhc_a1_audit/AUDIT.md):

* ``test_fused_vs_composite_every_chip``: per chip, fused ``hn`` vs the composite's ``hn`` (bf16 class) and the
  fused lane tiles vs the composite's raw Sinkhorn tiles expanded on the host; replicated inputs must give
  bit-identical outputs on every chip; the model's own integration (``TtMHCFast._pre`` with ``pre_block_on``)
  must reproduce the direct op call bit for bit (exercises the mesh-safe norm-weight read-back);
* ``test_two_instances_share_one_program``: two ops (two norm weights, two inputs) called alternately -- the
  program-cache hit between instances must re-derive every buffer address (the generic-op slow-path rebuild);
* ``test_trace_replay_every_chip``: one captured call replayed on new inputs equals the eager result per chip.

    # 1. one chip, the exact mesh shapes (default GLM53_MESH_SHAPE=1,1)
    TT_VISIBLE_DEVICES=27 pytest models/demos/glm53_flash/tests/custom_ops/test_mhc_pre_block_mesh.py -s
    # 2. the (2,4) block, all 8 chips
    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 \\
      TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto \\
      pytest models/demos/glm53_flash/tests/custom_ops/test_mhc_pre_block_mesh.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")
from models.demos.glm53_flash.custom_ops.mhc_pre_block.op import MhcPreBlock
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.custom_ops.test_mhc_post_mix import _err, expand_x8
from models.demos.glm53_flash.tests.test_mhc import DUMP, _streams
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mhc import TILE, TtMHCFast, mhc_kernel_config

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
LAYER = 4
# bars: hn is bf16 on both paths (ulp flips from the two projections' ~1e-4 rounding), the lane tiles come from the
# same Sinkhorn kernel on mixes that differ by the projection rounding (fused 5.9e-5 / composite 4.3e-5 vs f64)
HN_PCC, HN_REL = 0.9999, 5e-3
COEF_PCC, COEF_REL = 0.9999, 2e-3


def _setup(mesh_device):
    if not DUMP.exists():
        pytest.skip(f"stream dump {DUMP} not found")
    try:
        snapshot = default_snapshot_dir()
    except Exception as e:  # noqa: BLE001 (any resolution failure means no checkpoint on this host)
        pytest.skip(f"no checkpoint: {e}")
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    kcfg = mhc_kernel_config(cfg)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    fn, base, scale = raw["hc_ffn_fn"].float(), raw["hc_ffn_base"].float(), raw["hc_ffn_scale"].float()
    g = torch.Generator().manual_seed(5)
    w_norm = (0.8 + 0.4 * torch.rand(kcfg.dim, generator=g)).to(torch.bfloat16).float()
    # the composite decode pre-block (the reference): never the fused path, whatever GLM53_MHC_PRE_BLOCK says
    hc = TtMHCFast(mesh_device, kcfg, fn, base, scale)
    hc.pre_block_on = False
    w_dev = ttnn.from_torch(
        w_norm.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )  # as decoder_layer builds input_norm_w (replicated over the mesh)
    eps = float(cfg.rms_norm_eps)
    return cfg, kcfg, hc, (fn, base, scale), w_norm, w_dev, eps


def _fused_op(mesh_device, hc, params, w_norm, kcfg, eps, exact=False):
    fn, _, _ = params
    return MhcPreBlock(
        mesh_device,
        fn,
        hc.consts,
        w_norm,
        n=kcfg.n,
        C=kcfg.dim,
        sinkhorn_iters=kcfg.sinkhorn_iters,
        eps=kcfg.eps,
        norm_eps=kcfg.norm_eps,
        rms_eps=eps,
        exact=exact,
    )


def _rows(B: int, offset: int, nC: int) -> torch.Tensor:
    x = _streams(TILE, offset=offset)  # 32 real rows [32, n*C]
    if B < TILE:
        x[B:] = 0.1 * torch.randn(TILE - B, nC, generator=torch.Generator().manual_seed(B + offset))
    return x


def _up_replicated(mesh_device, x: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        x.reshape(1, 1, TILE, -1).contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _shards(t: ttnn.Tensor) -> list[torch.Tensor]:
    """Per-chip host copies (mesh row-major order), as ``[32, width]`` float tensors."""
    return [ttnn.to_torch(d).float().reshape(TILE, -1) for d in ttnn.get_device_tensors(t)]


def _composite_pre(hc, xd, w_dev, eps):
    h, post, comb, _x32, x_sm, owned = hc._pre(xd, norm=(w_dev, eps))
    if owned:
        ttnn.deallocate(x_sm)
    return h, post, comb


def _check_chip(tag: str, i: int, n: int, B: int, hn_c, post_c, comb_c, hn_f, post_f, comb_f):
    """One chip: the fused outputs against the composite's (host side)."""
    hn_pcc = pcc(hn_c[:B], hn_f[:B])
    hn_rel = _err(f"{tag} chip {i} hn fused vs composite", hn_c[:B], hn_f[:B])
    post_l, comb_l = expand_x8(post_c[:B, :n], comb_c[:B, : n * n], n)
    post_rel = _err(f"{tag} chip {i} post lane fused vs composite", post_l, post_f[:B])
    comb_rel = _err(f"{tag} chip {i} comb lane fused vs composite", comb_l, comb_f[:B])
    post_pcc, comb_pcc = pcc(post_l, post_f[:B]), pcc(comb_l, comb_f[:B])
    logger.info(f"{tag} chip {i}: hn pcc {hn_pcc:.7f} post pcc {post_pcc:.7f} comb pcc {comb_pcc:.7f}")
    assert torch.isfinite(hn_f).all() and torch.isfinite(post_f).all() and torch.isfinite(comb_f).all(), tag
    assert hn_pcc >= HN_PCC and hn_rel < HN_REL, (tag, i, hn_pcc, hn_rel)
    assert post_pcc >= COEF_PCC and post_rel < COEF_REL, (tag, i, post_pcc, post_rel)
    assert comb_pcc >= COEF_PCC and comb_rel < COEF_REL, (tag, i, comb_pcc, comb_rel)


@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=32 << 20)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
def test_fused_vs_composite_every_chip(mesh_device, B):
    cfg, kcfg, hc, params, w_norm, w_dev, eps = _setup(mesh_device)
    n, C = kcfg.n, kcfg.dim
    op = _fused_op(mesh_device, hc, params, w_norm, kcfg, eps)
    x = _rows(B, 100, n * C)
    xd = _up_replicated(mesh_device, x)
    # the composite reference
    h_c, post_c, comb_c = _composite_pre(hc, xd, w_dev, eps)
    # the fused op, called directly
    hn_f, post_f, comb_f = op.allocate_outputs(xd)
    op(xd, hn_f, post_f, comb_f)
    # ... and through the model's integration (TtMHCFast._pre -> _fused_pre_block: the mesh-safe weight read-back)
    hc_f = TtMHCFast(mesh_device, kcfg, params[0], params[1], params[2])
    hc_f.pre_block_on, hc_f._fn_raw = True, params[0].float()
    h_m, post_m, comb_m = _composite_pre(hc_f, xd, w_dev, eps)
    assert hc_f.pre_block is not None, "the fused path was not taken"
    ttnn.synchronize_device(mesh_device)

    hn_cs, post_cs, comb_cs = _shards(h_c), _shards(post_c), _shards(comb_c)
    hn_fs, post_fs, comb_fs = _shards(hn_f), _shards(post_f), _shards(comb_f)
    hn_ms, post_ms, comb_ms = _shards(h_m), _shards(post_m), _shards(comb_m)
    nchips = len(hn_fs)
    assert nchips == MESH[0] * MESH[1], (nchips, MESH)
    tag = f"B={B} mesh {MESH}"
    for i in range(nchips):
        _check_chip(tag, i, n, B, hn_cs[i], post_cs[i], comb_cs[i], hn_fs[i], post_fs[i], comb_fs[i])
        # the model's integration is the same program on the same inputs: bit-identical to the direct call
        assert torch.equal(hn_ms[i], hn_fs[i]), f"chip {i}: TtMHCFast._pre hn != direct op call"
        assert torch.equal(post_ms[i][:B], post_fs[i][:B]) and torch.equal(comb_ms[i][:B], comb_fs[i][:B]), i
    # replicated input: every chip must produce the same bits (the program, its addresses and its NoC tables are
    # the same on every chip; a chip-dependent difference here is the mesh hazard this test exists for)
    for i in range(1, nchips):
        assert torch.equal(hn_fs[i], hn_fs[0]), f"fused hn differs between chip 0 and chip {i}"
        assert torch.equal(post_fs[i], post_fs[0]) and torch.equal(comb_fs[i], comb_fs[0]), i
        assert torch.equal(hn_cs[i], hn_cs[0]), f"composite hn differs between chip 0 and chip {i} (reference)"
    # against the float64 math as well (the unit test's bars)
    _, _, _, _, _, r_hn = op.golden(x, params[0], params[1], params[2])
    rel = _err(f"{tag} fused hn vs f64", r_hn[:B], hn_fs[0][:B].double())
    assert rel < 5e-3


@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_two_instances_share_one_program(mesh_device):
    """Two ops with different weights and inputs, called alternately: every call after the first is a program-cache
    hit of the same program with other buffer addresses (the model has 90 instances)."""
    cfg, kcfg, hc, params, w_norm, w_dev, eps = _setup(mesh_device)
    n, C = kcfg.n, kcfg.dim
    B = 4
    g = torch.Generator().manual_seed(11)
    w2 = (0.5 + torch.rand(C, generator=g)).to(torch.bfloat16).float()
    w2_dev = ttnn.from_torch(w2.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    op1 = _fused_op(mesh_device, hc, params, w_norm, kcfg, eps)
    op2 = _fused_op(mesh_device, hc, params, w2, kcfg, eps)
    x1, x2 = _rows(B, 100, n * C), _rows(B, 300, n * C)
    xd1, xd2 = _up_replicated(mesh_device, x1), _up_replicated(mesh_device, x2)
    outs1 = op1.allocate_outputs(xd1)
    outs2 = op2.allocate_outputs(xd2)
    outs1b = op1.allocate_outputs(xd1)
    op1(xd1, *outs1)  # miss
    op2(xd2, *outs2)  # hit: other fn/w/mask/x/outputs
    op1(xd1, *outs1b)  # hit again, back to the first instance's buffers
    h1, p1, c1 = _composite_pre(hc, xd1, w_dev, eps)
    h2, p2, c2 = _composite_pre(hc, xd2, w2_dev, eps)
    ttnn.synchronize_device(mesh_device)
    for i, chip in enumerate(zip(_shards(outs1[0]), _shards(outs1b[0]))):
        assert torch.equal(chip[0], chip[1]), f"chip {i}: the second call of op1 differs from its first"
    for i, (hn_c, post_c, comb_c, hn_f, post_f, comb_f) in enumerate(
        zip(_shards(h1), _shards(p1), _shards(c1), _shards(outs1[0]), _shards(outs1[1]), _shards(outs1[2]))
    ):
        _check_chip("op1", i, n, B, hn_c, post_c, comb_c, hn_f, post_f, comb_f)
    for i, (hn_c, post_c, comb_c, hn_f, post_f, comb_f) in enumerate(
        zip(_shards(h2), _shards(p2), _shards(c2), _shards(outs2[0]), _shards(outs2[1]), _shards(outs2[2]))
    ):
        _check_chip("op2", i, n, B, hn_c, post_c, comb_c, hn_f, post_f, comb_f)


@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=32 << 20)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_trace_replay_every_chip(mesh_device):
    cfg, kcfg, hc, params, w_norm, w_dev, eps = _setup(mesh_device)
    n, C = kcfg.n, kcfg.dim
    B = 4
    op = _fused_op(mesh_device, hc, params, w_norm, kcfg, eps)
    x1, x2 = _rows(B, 100, n * C), _rows(B, 700, n * C)
    xd = _up_replicated(mesh_device, x1)
    hn, post, comb = op.allocate_outputs(xd)
    op(xd, hn, post, comb)  # program-cache warm-up with the exact signature
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op(xd, hn, post, comb)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    # eager reference on x2 (another op instance: the trace must not depend on the instance)
    op_e = _fused_op(mesh_device, hc, params, w_norm, kcfg, eps)
    xd2 = _up_replicated(mesh_device, x2)
    hn_e, post_e, comb_e = op_e.allocate_outputs(xd2)
    op_e(xd2, hn_e, post_e, comb_e)
    ttnn.synchronize_device(mesh_device)
    host2 = ttnn.from_torch(
        x2.reshape(1, 1, TILE, -1).contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.copy_host_to_device_tensor(host2, xd)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    for i, (a, b, pa, pb, ca, cb) in enumerate(
        zip(_shards(hn), _shards(hn_e), _shards(post), _shards(post_e), _shards(comb), _shards(comb_e))
    ):
        assert torch.equal(a, b), f"chip {i}: trace replay hn != eager"
        assert torch.equal(pa, pb) and torch.equal(ca, cb), f"chip {i}: trace replay post/comb != eager"
    ttnn.release_trace(mesh_device, tid)
    h_c, _, _ = _composite_pre(hc, xd2, w_dev, eps)
    for i, (hn_c, hn_f) in enumerate(zip(_shards(h_c), _shards(hn))):
        assert pcc(hn_c[:B], hn_f[:B]) >= HN_PCC, i
