# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Routed experts on the MiMo all-gather MoE ops (GLM_EXPERTS_MODE=ag, tt/experts_ag.py) vs the dispatch / combine
path (unified) on one MoE layer's golden (component rung, last dumped chunk), in both residual layouts:

- replicated: x / routing all S rows on every chip -> [S, H] on every chip (chip 0 read back);
- split: mesh row r's S/2 rows on its chips -> chip (r, c) rows r S/2 + c S/(2 C) .. (concatenated in row-major order).

Each mode's output is scored against the golden (PCC, rel L2, scale coefficient) and ag against unified; then the
split call is timed warm (wall, one sync per call). Asserts, against unified on the same golden (both run the spec's
expert dtype, so the bfp4 noise is common; at bfp4 neither reaches the 0.99 component PCC): finite, PCC >= unified's
- 0.002, rel L2 <= unified's + 0.01, scale coefficient within 3% of 1 (layer 4, bfp4: ag 0.98831 / 0.1597 / 1.0236 vs
unified 0.98840 / 0.1541 / 1.0029; the flat expert's bfp8 x and h and LoFi path keep ~2% of gain even with the fp32
down accumulation, 4.8% without).

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_experts_ag.py -s

GLM_AG_TEST_LAYER (default 4), GLM_AG_TEST_MODES (default "ag,unified"), GLM_AG_TEST_ITERS (default 5).
"""

import os
import time

import torch

from models.demos.common.bringup.testing.component import _step
from models.demos.common.bringup.testing.harness import component_golden, device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
LAYER = int(os.environ.get("GLM_AG_TEST_LAYER", "4"))
MODES = os.environ.get("GLM_AG_TEST_MODES", "ag,unified").split(",")
ITERS = int(os.environ.get("GLM_AG_TEST_ITERS", "5"))


def _stats(got, want):
    w = want.double().reshape(-1, want.shape[-1])
    g = got.double().reshape(w.shape)
    gc, wc = g - g.mean(), w - w.mean()
    return {
        "pcc": float((gc * wc).sum() / (gc.norm() * wc.norm())),
        "rel": float((g - w).norm() / w.norm()),
        "coef": float((g * w).sum() / (w * w).sum()),
        "row_rel": float(((g - w).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)).max()),
        "finite": bool(torch.isfinite(g).all()),
    }


def _fmt(st):
    return f"pcc={st['pcc']:.6f} rel={st['rel']:.5f} coef={st['coef']:.5f} worst_row_rel={st['row_rel']:.4f}" + (
        "" if st["finite"] else " NON-FINITE"
    )


@mesh_parametrize
def test_experts_ag(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host
    from models.demos.glm53_flash_d_p.tt.experts import build_experts

    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, "experts")
    gl = g.layer(c, LAYER)
    x, r = (gl[i].float() for i in st.inputs)
    want = gl[st.output].float()
    s, H = x.shape[-2], x.shape[-1]
    rows, cols = tuple(mesh_device.shape)
    hooks.apply_device_settings(S)
    loader, cfg = hooks._loader_cfg(S)
    max_chunk = max(hooks._chunks(S))

    xb = x.reshape(1, 1, s, H).to(torch.bfloat16)
    rb = r.reshape(1, 1, s, -1).to(torch.bfloat16)
    half = lambda t: ttnn.from_torch(  # noqa: E731  mesh row i holds rows [i S/2, (i+1) S/2) on all its chips
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None)),
    )
    split_in = (half(xb), half(rb))
    rep_in = (replicate(mesh_device, xb), replicate(mesh_device, rb))

    outs, times, fails = {}, {}, []
    for mode in MODES:
        os.environ["GLM_EXPERTS_MODE"] = mode
        t0 = time.time()
        mod = build_experts(mesh_device, loader, cfg, LAYER, max_chunk, weights_dtype=hooks.experts_dtype(S))
        print(f"[{mode}] built layer {LAYER} in {time.time() - t0:.1f} s", flush=True)

        yd = mod(*rep_in[:1], dense=rep_in[1])
        outs[(mode, "replicated")] = replicated_to_host(yd).reshape(s, H)
        ttnn.deallocate(yd)

        yd = mod(split_in[0], dense=split_in[1], split=True)
        parts = [ttnn.to_torch(t).reshape(-1, H) for t in ttnn.get_device_tensors(yd)]  # row-major chip order
        outs[(mode, "split")] = torch.cat(parts)
        ttnn.deallocate(yd)

        for _ in range(2):  # warm
            ttnn.deallocate(mod(split_in[0], dense=split_in[1], split=True))
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        for _ in range(ITERS):
            ttnn.deallocate(mod(split_in[0], dense=split_in[1], split=True))
        ttnn.synchronize_device(mesh_device)
        times[mode] = (time.time() - t0) / ITERS * 1e3
        del mod

    for (mode, lay), out in outs.items():
        stt = _stats(out, want)
        print(f"[{mode} {lay}] vs golden: {_fmt(stt)}", flush=True)
        if mode == "ag":
            if not stt["finite"]:
                fails.append(f"ag {lay}: non-finite output")
            if not abs(stt["coef"] - 1) <= 0.03:
                fails.append(f"ag {lay}: scale coefficient {stt['coef']:.5f} outside 1 +- 0.03")
            if ("unified", lay) in outs:
                base = _stats(outs[("unified", lay)], want)
                if not stt["pcc"] >= base["pcc"] - 0.002:
                    fails.append(f"ag {lay}: pcc {stt['pcc']:.6f} < unified {base['pcc']:.6f} - 0.002")
                if not stt["rel"] <= base["rel"] + 0.01:
                    fails.append(f"ag {lay}: rel {stt['rel']:.5f} > unified {base['rel']:.5f} + 0.01")
                print(f"[ag vs unified {lay}]: {_fmt(_stats(out, outs[('unified', lay)]))}", flush=True)
    for mode, ms in times.items():
        print(f"[{mode}] split call warm wall {ms:.2f} ms (chunk {s}, layer {LAYER})", flush=True)
    for t in split_in + rep_in:
        ttnn.deallocate(t)
    assert not fails, "; ".join(fails)
