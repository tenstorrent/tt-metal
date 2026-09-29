# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 hyper-connections (bead F8): lagged pre-mix chain and final collapse vs the reference.

Two chained blocks with cheap elementwise sublayers isolate the mHC arithmetic from attention and MoE. The
reference is the vendored ``Block.hc_mixes / hc_pre / hc_post`` with V4.1's lagged wiring and the model's
final ``hc_pre(h, last ffn_pre)``. Bars (G1): mHC >= 0.999, mHC projection >= 0.998; repeats bit-identical.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.mhc import TtV41HyperConnections, initial_pre_mix
from tests.ttnn.utils_for_testing import comp_pcc

MHC_PCC = 0.999
PROJECTION_PCC = 0.998
SEQ = 256
SNAPSHOT = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
)
REF = SimpleNamespace(
    norm_eps=C.RMS_NORM_EPS, hc_mult=C.HC_MULT, hc_sinkhorn_iters=C.HC_SINKHORN_ITERS, hc_eps=C.HC_EPS
)


def _synthetic_hc(gen):
    mix, width = (2 + C.HC_MULT) * C.HC_MULT, C.HC_MULT * C.EMB_SIZE
    return (
        torch.randn(mix, width, generator=gen) * width**-0.5,
        0.5 * torch.randn(mix, generator=gen),
        0.5 * torch.randn(3, generator=gen),
    )


def _real_hc(layer):
    from safetensors import safe_open

    index = __import__("json").loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for site in ("attn", "ffn"):
        names = [f"layers.{layer}.hc_{site}_{k}" for k in ("fn", "base", "scale")]
        with safe_open(SNAPSHOT / index[names[0]], "pt") as f:
            out[site] = tuple(f.get_tensor(n).float() for n in names)
    return out


def _ref_block(x, pre_mix, hc, attention, ffn):
    """V4.1 Block.forward's residual wiring with the given sublayers (x [S, n, D] bf16)."""
    residual = x
    a_pre, a_post, a_comb = v41.Block.hc_mixes(REF, x, hc["attn"][0], hc["attn"][2], hc["attn"][1])
    h = attention(v41.Block.hc_pre(REF, x, pre_mix))
    x = v41.Block.hc_post(REF, h, residual, a_post, a_comb)
    residual = x
    f_pre, f_post, f_comb = v41.Block.hc_mixes(REF, x, hc["ffn"][0], hc["ffn"][2], hc["ffn"][1])
    h = ffn(v41.Block.hc_pre(REF, x, a_pre))
    x = v41.Block.hc_post(REF, h, residual, f_post, f_comb)
    return x, f_pre, (a_pre, a_post, a_comb)


def _pack(x, tp):
    s, n, d = x.shape
    return x.reshape(s, n, tp, d // tp).permute(0, 2, 1, 3).reshape(1, 1, s, n * d)


def _unpack(t, n, tp):
    s, width = t.shape[-2], t.shape[-1]
    return t.reshape(s, tp, n, width // n // tp).permute(0, 2, 1, 3).reshape(s, n, -1)


def _pcc(a, b):
    return comp_pcc(a.float(), b.float(), 0.0)[1]


@pytest.mark.parametrize("weights", ["synthetic", "real_layers_2_3"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)
def test_v41_mhc_chain(mesh_device, device_params, weights):
    gen = torch.Generator().manual_seed(7)
    if weights == "synthetic":
        hcs = [{"attn": _synthetic_hc(gen), "ffn": _synthetic_hc(gen)} for _ in range(2)]
    else:
        if not (SNAPSHOT / "model.safetensors.index.json").is_file():
            pytest.skip("V4.1 checkpoint shards not downloaded")
        hcs = [_real_hc(2), _real_hc(3)]
    shape, (sp, tp), n = tuple(mesh_device.shape), tuple(mesh_device.shape), C.HC_MULT
    topology = per_axis_topology(device_params["fabric_config"])[1]

    # Sublayers: elementwise scales (distinct per site), identical on both sides.
    scales = [torch.randn(C.EMB_SIZE, generator=gen).to(torch.bfloat16) for _ in range(4)]
    x0 = torch.randn(SEQ, n, C.EMB_SIZE, generator=gen).to(torch.bfloat16)

    # reference: two blocks, then the model's final collapse
    pre = torch.zeros(SEQ, n)
    pre[:, 0] = 1.0
    x, ref_pre, ref_mixes = x0[None], pre[None], []
    ref_streams = []
    for b in range(2):
        x, ref_pre, mixes = _ref_block(
            x, ref_pre, hcs[b], lambda h, s=scales[2 * b]: h * s, lambda h, s=scales[2 * b + 1]: h * s
        )
        ref_mixes.append([m[0] for m in mixes])
        ref_streams.append(x[0])
    ref_final = v41.Block.hc_pre(REF, x, ref_pre)[0]
    ref_pre = ref_pre[0]
    ref_projection = torch.nn.functional.linear(x0.flatten(1).float(), hcs[0]["attn"][0]) * torch.rsqrt(
        x0.flatten(1).float().square().mean(-1, keepdim=True) + C.RMS_NORM_EPS
    )

    # device
    def tp_row(v):
        return ttnn.from_torch(
            v.reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(None, 3)),
        )

    tt_scales = [tp_row(s) for s in scales]
    blocks = [TtV41HyperConnections(mesh_device, C, hc["attn"], hc["ffn"], topology) for hc in hcs]
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))

    def run():
        x = ttnn.from_torch(
            _pack(x0.float(), tp),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
        )
        projection = down(blocks[0].attn_site.project(x))[0, 0, :, : ref_projection.shape[-1]]
        split = [down(t)[0, 0] for t in blocks[0].attn_site.split(x)]
        pre = initial_pre_mix(mesh_device, C, SEQ)
        streams = []
        for b, blk in enumerate(blocks):
            x, pre = blk(
                x,
                pre,
                attention=lambda h, s=tt_scales[2 * b]: ttnn.multiply(h, s),
                ffn=lambda h, s=tt_scales[2 * b + 1]: ttnn.multiply(h, s),
            )
            streams.append(_unpack(down(x)[0, 0], n, tp))
        final = down(blocks[-1].final_collapse(x, pre))[0, 0]
        return projection, split, streams, down(pre)[0, 0, :, :n], final

    first, second = run(), run()
    for a, b in zip(torch.utils._pytree.tree_leaves(first), torch.utils._pytree.tree_leaves(second)):
        assert torch.equal(a, b), "mHC chain is not bit-identical across repeats"
    projection, split, streams, pre_out, final = first

    tp_cols = lambda t, k: t[:, :k]  # split outputs are replicated across TP; keep one copy
    results = {
        "projection": _pcc(ref_projection, projection),
        "pre": _pcc(ref_mixes[0][0], tp_cols(split[0], n)),
        "post": _pcc(ref_mixes[0][1], tp_cols(split[1], n)),
        "comb": _pcc(ref_mixes[0][2].flatten(-2), tp_cols(split[2], n * n)),
        "block0_streams": _pcc(ref_streams[0], streams[0]),
        "block1_streams": _pcc(ref_streams[1], streams[1]),
        "next_pre_mix": _pcc(ref_pre, pre_out),
        "final_collapse": _pcc(ref_final, final),
    }
    print(f"mHC {weights}: {results}")
    assert results["projection"] >= PROJECTION_PCC, results
    for key, value in results.items():
        if key != "projection":
            assert value >= MHC_PCC, (key, results)
