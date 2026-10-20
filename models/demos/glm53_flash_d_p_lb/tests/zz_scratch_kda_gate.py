# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the KDA decay gate on the device vs fp32 math on the device's own inputs. ttKDA._compute_decay is one
ttnn.linear with a fused SIGMOID (gate = sigmoid(f_a x @ (W_b * scale) + b * scale); g = -5 gate per token and
channel). A slow head's gate is small (sigmoid far in its tail), and its steady-state carry ~ 1 / mean(g): a biased
tail of the sigmoid shows as a state too large or too small (zz_scratch_kda_drift: L2 head 56 1.52x). One chunk of
the prompt through layers GLM_KG_LAYERS (default 0-2); per layer and head: mean gate device / fp32-on-the-same-inputs
(and the same with the bf16-rounded fp32 result), the head's mean gate, the worst heads by the ratio.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_kda_gate.py -s
"""

import os

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


@mesh_parametrize
def test_kda_gate(mesh_device):
    import ttnn
    from models.demos.common.bringup.reference.golden import Golden
    from models.demos.deepseek_v3_d_p.tt.kda import kda as kda_mod

    g = Golden.for_rung(S, "s56320")
    tokens = g.tokens().to(torch.long)
    chunk = 5120
    layers = [int(v) for v in os.environ.get("GLM_KG_LAYERS", "0,1,2").split(",")]
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    R, C = tuple(mesh_device.shape)
    caps = []
    orig = kda_mod.ttKDA._compute_decay

    def per_dev(t):
        return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]

    def wrapped(self, projected, off):
        gate = orig(self, projected, off)
        w = self.weights
        caps.append(
            dict(
                layer=layers[len(caps)],  # (one call per layer, in order)
                x=per_dev(projected),
                off=off,
                gate=per_dev(gate),
                W=per_dev(w.decay_output_projection),
                b=per_dev(w.decay_bias_flat),
                heads=self.config.num_heads,
                kd=self.config.head_k_dim,
            )
        )
        return gate

    kda_mod.ttKDA._compute_decay = wrapped
    try:
        h = model.embed(tokens[:chunk])
        for i in layers:
            h2 = model.layer(i, h, 0, None)
            model.free(h)
            h = h2
        model.free(h)
    finally:
        kda_mod.ttKDA._compute_decay = orig
    for cap in caps:
        H, kd = cap["heads"], cap["kd"]
        rows = []  # (global head, ratio, ratio_bf16, mean ref)
        for d in range(R * C):
            c = d % C
            W, b = cap["W"][d].reshape(-1, H * kd), cap["b"][d].reshape(1, H * kd)
            x = cap["x"][d].reshape(-1, cap["x"][d].shape[-1])[:, cap["off"] : cap["off"] + W.shape[0]]
            ref = torch.sigmoid(x.double() @ W.double() + b.double())
            dev = cap["gate"][d].reshape(-1, H * kd).double()
            refb = ref.float().to(torch.bfloat16).double()
            if d // C != 0:  # (the mesh rows hold the other sequence half: the same heads, report row 0)
                continue
            for hh in range(H):
                sl = slice(hh * kd, (hh + 1) * kd)
                mr = ref[:, sl].mean()
                rows.append(
                    (
                        c * H + hh,
                        float(dev[:, sl].mean() / mr),
                        float(refb[:, sl].mean() / mr),
                        float(mr),
                        float(((dev[:, sl] - ref[:, sl]).abs() / ref[:, sl].clamp_min(1e-30)).median()),
                    )
                )
        rows.sort(key=lambda r: -abs(r[1] - 1))
        print(
            f"[kg] L{cap['layer']}: heads {len(rows)}; mean gate device / fp32: min {min(r[1] for r in rows):.4f} "
            f"max {max(r[1] for r in rows):.4f}",
            flush=True,
        )
        for gh, ratio, rb, mr, med in rows[:8]:
            print(
                f"[kg]   head {gh:3d}: device / fp32 {ratio:.4f} (bf16-rounded fp32 {rb:.4f}), mean gate {mr:.3e}, "
                f"median per-element rel err {med:.2e}",
                flush=True,
            )
        slow = sorted(rows, key=lambda r: r[3])[:6]
        print(
            "[kg]   slowest heads: " + ", ".join(f"h{r[0]} gate {r[3]:.2e} ratio {r[1]:.4f}" for r in slow), flush=True
        )
