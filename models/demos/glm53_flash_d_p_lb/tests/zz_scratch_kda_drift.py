# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the KDA recurrent state's drift from the reference over the long prompt, on the dense KDA layers 0-2 alone
(the embedding in: no MoE, no combine). The s56320 golden's prompt in 5120 chunks; at 51200 (golden snapshot) and at
56320 (golden final), per layer and head: rel L2 vs the golden, the norm ratio device / golden (a decay bias shows
as a ratio away from 1, noise as rel with ratio ~1). Also the KDA output (layer out) of the last chunk vs the golden.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_kda_drift.py -s
"""

import os

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@mesh_parametrize
def test_kda_drift(mesh_device):
    from models.demos.common.bringup.reference.golden import Golden

    g = Golden.for_rung(S, "s56320")
    tokens = g.tokens().to(torch.long)
    seq, chunk = 56320, 5120
    last = seq - chunk
    layers = [int(v) for v in os.environ.get("GLM_KD_LAYERS", "0,1,2").split(",")]
    if os.environ.get("GLM_KD_MASK"):  # (probe: chunk preparation's bf16 outputs; bit 5 final_decay, 1 kd, 2 q_decay)
        from models.demos.deepseek_v3_d_p.tt.kda import recurrence

        recurrence.KDA_PREP_OUTPUT_BF16_MASK = int(os.environ["GLM_KD_MASK"], 0)
        print(f"[kd] prep bf16 mask {recurrence.KDA_PREP_OUTPUT_BF16_MASK:#x}", flush=True)
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    snaps = {}
    for start in range(0, seq, chunk):
        if start == last:
            snaps[last] = {i: model.blocks[i].state_torch(last)["kda_recurrent"].clone() for i in layers}
        h = model.embed(tokens[start : start + chunk])
        for i in layers:
            h2 = model.layer(i, h, start, None)
            model.free(h)
            h = h2
            if start == last:
                go = g.layer(10, i)["out"].float()
                print(
                    f"[kd] L{i} last-chunk out rel vs golden {_rel(model.to_host(h).float().reshape(go.shape), go):.3e}"
                )
        model.free(h)
    snaps[seq] = {i: model.blocks[i].state_torch(seq)["kda_recurrent"].clone() for i in layers}
    for at in (last, seq):
        for i in layers:
            gv = (g.state(i, at=last) if at == last else g.state(i))["kda_recurrent"].float()
            dv = snaps[at][i].reshape(gv.shape)
            H = gv.shape[0]
            rel_h = [_rel(dv[h], gv[h]) for h in range(H)]
            ratio_h = [float(dv[h].norm() / gv[h].norm().clamp_min(1e-30)) for h in range(H)]
            print(
                f"[kd] at {at} L{i}: state rel {_rel(dv, gv):.3e}, norm ratio {float(dv.norm() / gv.norm()):.4f}; "
                f"per head rel min / median / max {min(rel_h):.2e} / {sorted(rel_h)[H // 2]:.2e} / {max(rel_h):.2e}",
                flush=True,
            )
            order = sorted(range(H), key=lambda h: -rel_h[h])[:6]
            print(
                "[kd]   worst heads: "
                + ", ".join(
                    f"h{h} rel {rel_h[h]:.2e} ratio {ratio_h[h]:.3f} |g| {float(gv[h].norm()):.2e}" for h in order
                ),
                flush=True,
            )
