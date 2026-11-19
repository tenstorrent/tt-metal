# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device A/Bs of the mHC roofline knobs (_agents/mhc_roofline/STATUS.md) -- NOT run on the host that wrote them; the
chips were owned by the validation queue. Every case builds the default path and the knob path side by side on the
same inputs (real layer-4 parameters and stream rows), at the model's decode shapes (``x [1,1,32,n*C]`` fp32 TILE,
B = 4 and 32 valid rows), and compares them to the bar the knob's numerics allow:

| knob (env / constructor) | path | bar vs the default |
|---|---|---|
| ``GLM53_MHC_POST_SPLIT=extras`` (``post_split``) | post-mix v2 streams + the delegated (position, stream) extras | bit-identical (every output tile is computed by exactly one core with the same operands) |
| ``GLM53_MHC_PRE_SPLIT=extras`` (``pre_split``) | fused pre-block, the two-column cores' second columns projected as extra stream tiles elsewhere | hn PCC >= 0.9999, rel rms < 1e-4; post / comb lanes rel < 1e-4 (same fp32 terms, summed in a different grouping) |
| ``GLM53_MHC_PRE_SINKHORN=fused`` (``pre_sinkhorn``) | fused pre-block, one dest pass per Sinkhorn normalisation | post / comb lanes rel < 2e-3, hn PCC >= 0.9999 (fp32 SFPU products instead of 19-bit FPU operands: a rounding change -> the logits gate below) |
| ``GLM53_MHC_KSUM=fast`` (``ksum_mode``) | composite decode pre-block (fused pre-block OFF), fast_reduce_nc K-sums | mixes rel < 2e-3 vs the exact chains (the 19-bit SrcA rounding of the partials, ~5e-4 modelled); coefficients rel < 2e-3 |
| ``GLM53_MHC_PREFILL_NORM`` | the per-call norm mode (``forward(prefill=True)``) | ``single`` inside a prefill scope == ``norm_mode="single"`` bit for bit; outside == ``sharded`` |

Plus: each knob's wrap captured in a trace and replayed on new inputs (== eager), a traced us-per-wrap table (ITERS
replays; the device-time split needs tracy), and the (2,4) block / e2e commands in ``main``'s docstring below.

    # chip 27 (one chip of the block; the exact mesh shapes)
    TT_VISIBLE_DEVICES=27 pytest models/demos/glm53_flash/tests/test_mhc_roofline_mesh.py -s
    # the (2,4) block, every chip bit-level
    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 \\
      TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto \\
      pytest models/demos/glm53_flash/tests/test_mhc_roofline_mesh.py -s
    # the block decoder test and the e2e chain with a knob (same env), e.g. the fused Sinkhorn + both splits:
    GLM53_MHC_PRE_BLOCK=1 GLM53_MHC_PRE_SINKHORN=fused GLM53_MHC_PRE_SPLIT=extras GLM53_MHC_POST_SPLIT=extras \\
      pytest models/demos/glm53_flash/tests/test_mesh_decoder.py -s            # layers 0/3/4 vs HF
    GLM53_MHC_PRE_BLOCK=1 GLM53_MHC_PRE_SINKHORN=fused GLM53_MHC_PRE_SPLIT=extras GLM53_MHC_POST_SPLIT=extras \\
      GLM53_MODEL_LAYERS=0-21 GLM53_MODEL_B=4 GLM53_MODEL_S=16 GLM53_MODEL_PCC_BAR=0.5 GLM53_MODEL_TRACE_MB=512 \\
      GLM53_MOE_LAYOUT=ep pytest models/demos/glm53_flash/tests/test_mesh_model.py -s -k chain   # vs the same without
    # the logits gate for the rounding changes (fused Sinkhorn, fast K-sums), 8x4:
    GLM53_DUMP=L45_NAT_S128 GLM53_MESH_SHAPE=8,4 GLM53_MODEL_LAYERS=0-44 GLM53_MODEL_B=32 GLM53_MHC_PRE_BLOCK=1 \\
      GLM53_MHC_PRE_SINKHORN=fused pytest models/demos/glm53_flash/tests/test_mesh_model_logits.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")
from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import MhcPostMix  # noqa: E402
from models.demos.glm53_flash.custom_ops.mhc_pre_block.op import MhcPreBlock  # noqa: E402
from models.demos.glm53_flash.reference.checkpoint import (  # noqa: E402
    default_snapshot_dir,
    load_layer_raw,
    shard_headers,
)
from models.demos.glm53_flash.tests.custom_ops.test_mhc_post_mix import _err, expand_x8  # noqa: E402
from models.demos.glm53_flash.tests.test_mhc import DUMP, _streams, torch_reference  # noqa: E402
from models.demos.glm53_flash.tests.utils import pcc  # noqa: E402
from models.demos.glm53_flash.tt.config import GLM53FlashConfig  # noqa: E402
from models.demos.glm53_flash.tt.mesh import mesh_device_params  # noqa: E402
from models.demos.glm53_flash.tt.mhc import TILE, TtMHCFast, mhc_kernel_config  # noqa: E402

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
LAYER = 4
ITERS = int(os.environ.get("GLM53_MHC_PERF_ITERS", "20"))
PCC_BAR = 0.9999


def _setup(mesh_device):
    if not DUMP.exists():
        pytest.skip(f"stream dump {DUMP} not found")
    try:
        snapshot = default_snapshot_dir()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no checkpoint: {e}")
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    kcfg = mhc_kernel_config(cfg)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    params = (raw["hc_ffn_fn"].float(), raw["hc_ffn_base"].float(), raw["hc_ffn_scale"].float())
    g = torch.Generator().manual_seed(5)
    w_norm = (0.8 + 0.4 * torch.rand(kcfg.dim, generator=g)).to(torch.bfloat16).float()
    w_dev = ttnn.from_torch(
        w_norm.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    return cfg, kcfg, params, w_norm, w_dev, float(cfg.rms_norm_eps)


def _rows(B: int, offset: int, nC: int) -> torch.Tensor:
    x = _streams(TILE, offset=offset)
    if B < TILE:
        x[B:] = 0.1 * torch.randn(TILE - B, nC, generator=torch.Generator().manual_seed(B + offset))
    return x


def _up(mesh_device, x: torch.Tensor, dtype=ttnn.float32) -> ttnn.Tensor:
    return ttnn.from_torch(
        x.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _shards(t: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float().reshape(TILE, -1) for d in ttnn.get_device_tensors(t)]


def _fast(mesh_device, kcfg, params, **kw) -> TtMHCFast:
    hc = TtMHCFast(mesh_device, kcfg, *params, **kw)
    hc.pre_block_on = kw.pop("pre_block_on", hc.pre_block_on)
    return hc


def _pre(hc, xd, w_dev, eps, **kw):
    """The model's own pre-block (fused or composite per ``hc.pre_block_on``) -> (hn, post, comb) host shards."""
    h, post, comb, _x32, x_sm, owned = hc._pre(xd, norm=(w_dev, eps), **kw)
    if owned:
        ttnn.deallocate(x_sm)
    out = (_shards(h), _shards(post), _shards(comb))
    for t in (h, post, comb):
        ttnn.deallocate(t)
    return out


def _same_on_every_chip(tag: str, shards: list[torch.Tensor]):
    for i, s in enumerate(shards[1:], 1):
        assert torch.equal(s, shards[0]), f"{tag}: chip {i} differs from chip 0"


# ------------------------------------------------------------------------------------------------ post-mix extras
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
@pytest.mark.parametrize("y_dtype", [ttnn.bfloat16, ttnn.float32])
def test_post_split_extras_bit_identical(mesh_device, B, y_dtype):
    """The extras split of the post-mix v2 kernel (streams mode): every output tile is produced by exactly one core
    from the same operands -> bit-identical to the tile split on every chip; collapse / h-mode never split."""
    cfg, kcfg, params, w_norm, w_dev, eps = _setup(mesh_device)
    n, C = kcfg.n, kcfg.dim
    x = _rows(B, 0, n * C)
    g = torch.Generator().manual_seed(B)
    y = torch.randn(TILE, C, generator=g)
    if y_dtype == ttnn.bfloat16:
        y = y.to(torch.bfloat16).float()
    post = 2 * torch.rand(TILE, n, generator=g)
    comb = torch.softmax(torch.randn(TILE, n, n, generator=g), -1).reshape(TILE, n * n)
    xd, yd = _up(mesh_device, x.reshape(1, 1, TILE, n * C)), _up(mesh_device, y.reshape(1, 1, TILE, C), y_dtype)
    pd, cd = _up(mesh_device, post), _up(mesh_device, comb)
    outs = {}
    for split in ("tile", "extras"):
        op = MhcPostMix(n, version=2, split=split)
        for collapse in (False, True):
            out = op.allocate_output(xd, collapse)
            op(xd, yd, pd, cd, out, collapse)
            outs[(split, collapse)] = _shards(out)
            ttnn.deallocate(out)
        # the lane-layout inputs (what the fused pre-block emits) too
        post_l, comb_l = expand_x8(post, comb, n)
        out = op.allocate_output(xd, False)
        op(xd, yd, _up(mesh_device, post_l), _up(mesh_device, comb_l), out, False)
        outs[(split, "lane")] = _shards(out)
        ttnn.deallocate(out)
    ref = op.golden(x, y, post, comb, n, False)
    for key in ((False,), (True,), ("lane",)):
        t, e = outs[("tile",) + key], outs[("extras",) + key]
        _same_on_every_chip(f"post split {key}", e)
        for i, (a, b) in enumerate(zip(t, e)):
            assert torch.equal(a, b), f"post extras vs tile differ on chip {i} ({key})"
    rel = _err(f"post extras B={B} y={y_dtype} vs float64", ref, outs[("extras", False)][0])
    assert rel < 1e-6


# ------------------------------------------------------------------------------------------------ pre-block knobs
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
@pytest.mark.parametrize(
    "knob,hn_rel,coef_rel",
    [
        (dict(split="extras"), 1e-4, 1e-4),  # the same fp32 terms grouped differently
        (dict(sinkhorn="fused"), 5e-3, 2e-3),  # fp32 SFPU products instead of 19-bit FPU operands
        (dict(sinkhorn="fused", split="extras"), 5e-3, 2e-3),
    ],
)
def test_pre_block_knobs_vs_default(mesh_device, B, knob, hn_rel, coef_rel):
    """``MhcPreBlock(sinkhorn=, split=)`` against the default program on the same inputs, every chip; the default
    against the composite path is covered by custom_ops/test_mhc_pre_block_mesh.py."""
    cfg, kcfg, params, w_norm, w_dev, eps = _setup(mesh_device)
    n, C = kcfg.n, kcfg.dim
    fn = params[0]
    hc = _fast(mesh_device, kcfg, params)  # for its consts
    x = _rows(B, 32, n * C)
    xd = _up(mesh_device, x.reshape(1, 1, TILE, n * C))
    common = dict(n=n, C=C, sinkhorn_iters=kcfg.sinkhorn_iters, eps=kcfg.eps, norm_eps=kcfg.norm_eps, rms_eps=eps)
    base = MhcPreBlock(mesh_device, fn, hc.consts, w_norm, **common)
    variant = MhcPreBlock(mesh_device, fn, hc.consts, w_norm, **common, **knob)
    if "split" in knob:
        assert variant.max_extra == (MhcPreBlock.MAX_EXTRA if variant.work_ext is not None else 0)
        logger.info(
            f"extras plan: {sum(len(e) for *_, e in variant.work_ext or [])} extra tiles, grid {variant.gx}x{variant.gy}"
        )
    res = {}
    for tag, op in (("base", base), ("knob", variant)):
        hn, post, comb = op.allocate_outputs(xd)
        op(xd, hn, post, comb)
        ttnn.synchronize_device(mesh_device)
        res[tag] = (_shards(hn), _shards(post), _shards(comb))
        for t in (hn, post, comb):
            ttnn.deallocate(t)
    _same_on_every_chip(f"{knob} hn", res["knob"][0])
    _, _, _, _, _, hn_ref = base.golden(x, fn, params[1], params[2])
    for i in range(len(res["base"][0])):
        hb, pb, cb = (r[i] for r in res["base"])
        hk, pk, ck = (r[i] for r in res["knob"])
        p = pcc(hb[:B], hk[:B])
        rel_h = _err(f"{knob} chip {i} hn knob vs base", hb[:B], hk[:B])
        rel_p = _err(f"{knob} chip {i} post lanes", pb[:B], pk[:B])
        rel_c = _err(f"{knob} chip {i} comb lanes", cb[:B], ck[:B])
        assert torch.isfinite(hk).all() and torch.isfinite(pk).all() and torch.isfinite(ck).all()
        assert p >= PCC_BAR and rel_h < hn_rel, (knob, i, p, rel_h)
        assert rel_p < coef_rel and rel_c < coef_rel, (knob, i, rel_p, rel_c)
        # and both against the float64 math (bf16 hn: ~4e-3 rel rms class)
        assert pcc(hn_ref[:B].float(), hk[:B]) >= PCC_BAR
    # the model's integration: TtMHCFast with the knobs -> the same op outputs through _pre
    hc_k = _fast(
        mesh_device, kcfg, params, pre_sinkhorn=knob.get("sinkhorn", "llk"), pre_split=knob.get("split", "cols")
    )
    hc_k.pre_block_on = True
    hn_m, post_m, comb_m = _pre(hc_k, xd, w_dev, eps)
    for i in range(len(hn_m)):
        assert torch.equal(hn_m[i], res["knob"][0][i]) and torch.equal(comb_m[i], res["knob"][2][i])


# ------------------------------------------------------------------------------------------------ A9 fast K-sums
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
def test_ksum_fast_vs_exact(mesh_device, B):
    """The composite decode pre-block (fused pre-block OFF) with ``ksum_mode="fast"`` (fast_reduce_nc, 2 ops) vs
    ``"exact"`` (ttnn.sum chains, 7 ops): the mixes within the 19-bit-partial bound (~5e-4 rel modelled, bar 2e-3),
    the coefficients and hn to PCC >= 0.9999; both vs the fp32 reference."""
    cfg, kcfg, params, w_norm, w_dev, eps = _setup(mesh_device)
    n, C = kcfg.n, kcfg.dim
    x = _rows(B, 64, n * C)
    xd = _up(mesh_device, x.reshape(1, 1, TILE, n * C))
    r_mixes, r_h, r_post, r_comb = torch_reference(x, *params, kcfg)
    out = {}
    for mode in ("exact", "fast"):
        hc = _fast(mesh_device, kcfg, params, ksum_mode=mode)
        hc.pre_block_on = False
        mixes = hc.project(xd)
        out[mode] = (_shards(mixes), _pre(hc, xd, w_dev, eps))
        ttnn.deallocate(mixes)
    mix_cols = (2 + n) * n
    for i in range(len(out["exact"][0])):
        me, mf = out["exact"][0][i][:B, :mix_cols], out["fast"][0][i][:B, :mix_cols]
        rel = _err(f"ksum chip {i} mixes fast vs exact", me, mf)
        assert rel < 2e-3, rel
        assert pcc(r_mixes[:B], mf) >= 0.999 and pcc(r_mixes[:B], me) >= 0.999
        he, hf = out["exact"][1][0][i], out["fast"][1][0][i]
        assert pcc(he[:B], hf[:B]) >= PCC_BAR
        ce, cf = out["exact"][1][2][i], out["fast"][1][2][i]
        assert _err(f"ksum chip {i} comb fast vs exact", ce[:B, : n * n], cf[:B, : n * n]) < 2e-3


# ------------------------------------------------------------------------------------------------ norm selection
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_prefill_scope_norm_mode(mesh_device):
    """``forward(prefill=True)`` runs the one-core norm (bit-identical to ``norm_mode="single"``); ``prefill=False``
    the configured sharded one (== ``norm_mode="sharded"``); the two differ only by bf16 ulp flips (PCC >= 0.9999)."""
    cfg, kcfg, params, w_norm, w_dev, eps = _setup(mesh_device)
    n, C = kcfg.n, kcfg.dim
    x = _rows(32, 96, n * C)
    xd = _up(mesh_device, x.reshape(1, 1, TILE, n * C))
    gain = ttnn.typecast(
        _up(mesh_device, 0.5 + torch.rand(1, 1, 1, C, generator=torch.Generator().manual_seed(2))), ttnn.bfloat16
    )
    hc = _fast(mesh_device, kcfg, params, norm_mode="sharded")
    hc.pre_block_on = False
    seen = {}

    def sub(key):
        def f(t):
            seen[key] = _shards(t)
            return ttnn.multiply(t, gain)

        return f

    for key, kw in (
        ("prefill", dict(prefill=True)),
        ("single", dict(norm_mode="single")),
        ("decode", dict(prefill=False)),
        ("sharded", dict(norm_mode="sharded")),
    ):
        y = hc.forward(xd, sub(key), sublayer_dtype=ttnn.bfloat16, norm=(w_dev, eps), **kw)
        ttnn.deallocate(y)
    for i in range(len(seen["prefill"])):
        assert torch.equal(seen["prefill"][i], seen["single"][i]), i
        assert torch.equal(seen["decode"][i], seen["sharded"][i]), i
        assert pcc(seen["prefill"][i], seen["decode"][i]) >= PCC_BAR


# ------------------------------------------------------------------------------------------------ trace + perf
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=64 << 20)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_knobs_traced_and_timed(mesh_device):
    """Every knob's full wrap (fused pre-block + trivial sublayer + post-mix) captured once and replayed on new inputs
    (== eager, per chip), then timed over ITERS replays: the us-per-wrap table of STATUS.md (wall time; the device
    split needs tracy: ``python -m tracy -r -p -v -m pytest ... -k traced``)."""
    cfg, kcfg, params, w_norm, w_dev, eps = _setup(mesh_device)
    n, C = kcfg.n, kcfg.dim
    gain = ttnn.typecast(
        _up(mesh_device, 0.5 + torch.rand(1, 1, 1, C, generator=torch.Generator().manual_seed(9))), ttnn.bfloat16
    )
    x1, x2 = _rows(32, 0, n * C), _rows(32, 32, n * C)
    xin = _up(mesh_device, x1.reshape(1, 1, TILE, n * C))  # persistent input: overwritten before the replay
    rows = []
    configs = [
        ("default (llk, cols, tile)", {}),
        ("post extras", dict(post_split="extras")),
        ("pre extras", dict(pre_split="extras")),
        ("pre fused Sinkhorn", dict(pre_sinkhorn="fused")),
        ("all three", dict(pre_sinkhorn="fused", pre_split="extras", post_split="extras")),
    ]
    for tag, kw in configs:
        hc = _fast(mesh_device, kcfg, params, **kw)
        hc.pre_block_on = True
        sub = lambda t: ttnn.multiply(t, gain)  # noqa: E731
        y = hc.forward(xin, sub, sublayer_dtype=ttnn.bfloat16, norm=(w_dev, eps))  # eager warm-up (builds the op)
        ttnn.deallocate(y)
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        y_tr = hc.forward(xin, sub, sublayer_dtype=ttnn.bfloat16, norm=(w_dev, eps))
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        # eager on x2, then the replay on x2 through the persistent input
        x2d = _up(mesh_device, x2.reshape(1, 1, TILE, n * C))
        y_e = hc.forward(x2d, sub, sublayer_dtype=ttnn.bfloat16, norm=(w_dev, eps))
        ref = _shards(y_e)
        ttnn.deallocate(y_e)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                x2.reshape(1, 1, TILE, n * C),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            xin,
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got = _shards(y_tr)
        for i, (a, b) in enumerate(zip(ref, got)):
            assert torch.equal(a, b), f"{tag}: trace replay != eager on chip {i}"
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / ITERS * 1e6
        ttnn.release_trace(mesh_device, tid)
        rows.append((tag, us))
        logger.info(f"PERF-MHC-WRAP {tag}: {us:.1f} us per traced wrap (mesh {MESH}, ITERS {ITERS})")
    base = rows[0][1]
    for tag, us in rows:
        logger.info(f"| {tag} | {us:.1f} us | {us - base:+.1f} |")
