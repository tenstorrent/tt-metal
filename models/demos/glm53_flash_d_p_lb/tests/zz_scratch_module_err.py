# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch (accuracy investigation 2026-10-27): the error each module injects, per layer, at the last chunk of the
s56320 prompt. Every layer runs alone on the device from the golden's own input (chunk 10 `in`) and the golden state
at 51200 (teacher forcing: no error carried from the layers above); each captured step output is compared with the
golden boundary. The error a step injects ~ its rel minus its input's rel.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_module_err.py -s
"""

import os

import ttnn
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
CAPTURE = (
    "attn_norm",
    "attention",
    "attn_residual",
    "ffn_norm",
    "mlp",
    "experts",
    "shared_expert",
    "moe_add",
    "ffn_residual",
)
GOLD = ("attn_norm", "attn_out", "h_mid", "ffn_norm", "mlp_out", "experts_out", "shared_out", "out")


def _rel(d, r):
    d, r = d.double().reshape(r.shape), r.double()
    return float((d - r).norm() / r.norm())


@mesh_parametrize
def test_module_err(mesh_device):
    from models.demos.glm53_flash_d_p.tt.common import replicated_to_host, split_to_host

    g = Golden.for_rung(S, "s56320")
    tokens = g.tokens()
    seq, chunk = 56320, 5120
    last = seq - chunk
    layers = [int(v) for v in os.environ["GLM_ME_LAYERS"].split(",")] if os.environ.get("GLM_ME_LAYERS") else S.layers()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    nd = mesh_device.get_num_devices()
    hidden = model.cfg.hidden_size
    captured = {}

    def to_host(t):
        if not isinstance(t, ttnn.Tensor):
            return None
        rows = t.shape[-2]
        if rows == chunk // nd:
            x = split_to_host(t)
        elif rows == chunk:
            x = replicated_to_host(t)
        else:
            return None
        return x.float().reshape(-1, hidden)

    for blk in model.blocks.values():
        for name in list(blk.overrides):
            if name in CAPTURE:
                fn = blk.overrides[name]

                def w(ctx, *args, fn=fn, name=name):
                    y = fn(ctx, *args)
                    hy = to_host(y)
                    if hy is not None:
                        captured[name] = hy
                    return y

                blk.overrides[name] = w

    sums = {}
    for i in layers:
        gl = g.layer(10, i)
        model.blocks[i].load_state({k: v.float() for k, v in g.state(i, at=last).items()}, last)
        h = model.from_host(gl["in"].float())
        captured.clear()
        out = model.layer(i, h, last, None)
        captured["ffn_residual"] = model.to_host(out)
        model.free(h)
        model.free(out)
        row = []
        for name, dev in captured.items():
            # the closest golden boundary of the same shape (a block-mode MoE step may return routed + shared)
            best = min(
                ((b, _rel(dev, gl[b].float())) for b in GOLD if b in gl and gl[b].numel() == dev.numel()),
                key=lambda t: t[1],
                default=None,
            )
            if best is not None:
                row.append(f"{name}->{best[0]} {best[1]:.4f}")
                sums.setdefault((name, best[0]), []).append(best[1])
        print(f"[me] L{i:02d} " + "  ".join(row), flush=True)
    print("[me] mean over layers:", flush=True)
    for (name, b), v in sorted(sums.items()):
        print(f"[me]   {name:14s}->{b:12s} mean {sum(v) / len(v):.4f} max {max(v):.4f} (n {len(v)})", flush=True)
