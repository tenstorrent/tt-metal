# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Does the activation magnitude drift through the layers on the device (inflation / deflation), and where?

Runs a golden rung (default s4096: 2 x 2048 tokens; the golden has every step boundary of every layer) on the device
from position 0 and, for every layer, compares step outputs against the CPU golden. PCC is scale-invariant, so this
measures scale directly:
  gain       <dev, ref> / <ref, ref>   (1 = no scale bias; > 1 inflated along the true direction)
  norm       ||dev|| / ||ref||          (> gain when uncorrelated noise adds energy)
  rel        ||dev - ref|| / ||ref||
  tok p1/p50/p99  per-token ||dev|| / ||ref|| quantiles
  maxabs     max |dev| / max |ref|      (outlier channels)
  stream gains (4 mHC streams) for the residual boundaries (in / h_mid / out: rows 4 t + n).

Captured steps (device outputs in the split layout, gathered to the host): attn_collapse -> attn_in, attention ->
attn_out, attn_residual -> h_mid, ffn_collapse -> ffn_in, mlp / moe_add -> mlp_out, experts -> experts_out,
shared_expert -> shared_out, ffn_residual -> out. Results: generated/glm53_flash_d_p_lb/inflation_<rung>.json.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_inflation.py -s
"""

import json
import os
from pathlib import Path

import torch

import ttnn
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
OUT = Path(__file__).resolve().parents[4] / "generated" / "glm53_flash_d_p_lb"
CAPTURE = {  # device step -> golden boundary
    "attn_collapse": "attn_in",
    "attention": "attn_out",
    "attn_residual": "h_mid",
    "ffn_collapse": "ffn_in",
    "mlp": "mlp_out",
    "moe_add": "mlp_out",
    "experts": "experts_out",
    "shared_expert": "shared_out",
    "ffn_residual": "out",
}
STREAMED = {"in", "h_mid", "out"}


def stats(dev: torch.Tensor, ref: torch.Tensor, n_streams: int) -> dict:
    d, r = dev.double().reshape(ref.shape), ref.double()
    rr = (r * r).sum()
    out = {
        "gain": float((d * r).sum() / rr),
        "norm": float(d.norm() / r.norm()),
        "rel": float((d - r).norm() / r.norm()),
        "maxabs": float(d.abs().max() / r.abs().max()),
        "ref_rms": float(r.pow(2).mean().sqrt()),
    }
    tok = d.norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
    q = torch.quantile(tok.float(), torch.tensor([0.01, 0.5, 0.99]))
    out.update(tok_p1=float(q[0]), tok_p50=float(q[1]), tok_p99=float(q[2]))
    if n_streams:
        h = r.shape[-1]
        ds, rs = d.reshape(-1, n_streams, h), r.reshape(-1, n_streams, h)
        out["stream_gain"] = [float((ds[:, k] * rs[:, k]).sum() / (rs[:, k] ** 2).sum()) for k in range(n_streams)]
        out["stream_rms"] = [float(rs[:, k].pow(2).mean().sqrt()) for k in range(n_streams)]
    return out


@mesh_parametrize
def test_inflation(mesh_device):
    from models.demos.glm53_flash_d_p.tt.common import replicated_to_host, split_to_host

    rung_name = os.environ.get("GLM_INFL_RUNG", "s4096")
    rung = S.rung(rung_name)
    g = Golden.for_rung(S, rung_name)
    chunk, n_chunks = rung["chunk"], rung["seq"] // rung["chunk"]
    layers = S.layers()
    tokens = g.tokens()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    n = model.cfg.hc_mult
    nd = mesh_device.get_num_devices()
    hidden = model.cfg.hidden_size

    def to_host(t):
        if not isinstance(t, ttnn.Tensor):
            return None
        rows = t.shape[-2]
        if rows == chunk // nd:
            x = split_to_host(t)
        elif rows == chunk:
            x = replicated_to_host(t)
        else:
            return None  # a half-gathered boundary: not captured
        x = x.float()
        return x.reshape(-1, hidden)

    captured = {}

    def wrap(blk, name):
        fn = blk.overrides[name]

        def w(ctx, *args):
            y = fn(ctx, *args)
            hy = to_host(y)
            if hy is not None:
                captured[(blk.i, CAPTURE[name])] = hy
            return y

        blk.overrides[name] = w

    for blk in model.blocks.values():
        for name in list(blk.overrides):
            if name in CAPTURE:
                wrap(blk, name)

    results = {}
    for c in range(n_chunks):
        s0 = c * chunk
        h = model.embed(tokens[s0 : s0 + chunk])
        res_c = {}
        for i in layers:
            captured.clear()
            h2 = model.layer(i, h, s0, None)
            model.free(h)
            h = h2
            gl = g.layer(c, i)
            row = {}
            for (li, bname), dev in captured.items():
                if bname in gl:
                    row[bname] = stats(dev, gl[bname].float(), n if bname in STREAMED else 0)
            res_c[i] = row
            o = row.get("out", {})
            print(
                f"chunk {c} L{i:02d} out: gain {o.get('gain', float('nan')):.4f} norm {o.get('norm', float('nan')):.4f} "
                f"rel {o.get('rel', float('nan')):.4f} tok[{o.get('tok_p1', 0):.3f},{o.get('tok_p50', 0):.3f},"
                f"{o.get('tok_p99', 0):.3f}] maxabs {o.get('maxabs', 0):.3f} ref_rms {o.get('ref_rms', 0):.4f}",
                flush=True,
            )
        model.free(h)
        results[c] = res_c

    # summary over layers (last chunk): mean gain per boundary, split by block type
    last = results[n_chunks - 1]
    print("\nper-boundary gain over layers (last chunk): mean / min / max", flush=True)
    names = sorted({b for row in last.values() for b in row})
    for b in names:
        gs = [row[b]["gain"] for row in last.values() if b in row]
        rel = [row[b]["rel"] for row in last.values() if b in row]
        print(
            f"  {b:12s} gain {sum(gs) / len(gs):.4f} [{min(gs):.4f}, {max(gs):.4f}]  rel mean {sum(rel) / len(rel):.4f}"
            f" max {max(rel):.4f}",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"inflation_{rung_name}.json").write_text(json.dumps(results, indent=1))
