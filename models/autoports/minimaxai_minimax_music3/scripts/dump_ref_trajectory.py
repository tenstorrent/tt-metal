# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-step fp32 CPU reference trajectory of the chunk loop (for the stage-05 drift log).

Runs the vendored torch DiT (``reference/flow_transformer_ref.py``) through the same chunk loop as
``tt/denoiser.py`` but with the torch model, starting from the golden noise / conditions of
``~/mm3-bringup/reference/chunks.pt``, and stores the latents after every Euler step. About 5 minutes per
chunk on the host CPU. Output (gitignored, ~21 MB): ``generated/ref_trajectory.pt``.

    source ~/mm3-bringup/common.sh && cd $MM3_WT && $MM3_PY $MM3_MODEL_DIR/scripts/dump_ref_trajectory.py
"""
import os
import time
from pathlib import Path

import torch

from models.autoports.minimaxai_minimax_music3.reference import flow_transformer_ref as REF
from models.autoports.minimaxai_minimax_music3.tt.denoiser import BLEND_EPS, DIT_CFG_SCALE, OVERLAP_LATENT_LENGTH
from models.autoports.minimaxai_minimax_music3.tt.scheduler import FlowMatchEulerScheduler

torch.set_num_threads(max(8, (os.cpu_count() or 8) - 4))
out = Path(__file__).resolve().parents[1] / "generated" / "ref_trajectory.pt"
g = torch.load(os.path.expanduser("~/mm3-bringup/reference/chunks.pt"))
model = REF.load_transformer()
records = []
prev_lat = prev_cond = None
with torch.no_grad():
    for k in range(len(g["chunk_starts"])):
        condition = g["conditions_raw"][k].clone()
        L = condition.shape[1]
        overlap = 0
        if prev_lat is not None:
            overlap = min(prev_lat.shape[-1], L)
            condition[:, :overlap] = prev_cond[:, :overlap]
        assert torch.equal(condition, g["conditions"][k]), "splice differs from the golden spliced condition"
        latents = g["noises"][k].clone()
        noise_prompt = latents[..., :overlap].clone() if overlap else None
        sched = FlowMatchEulerScheduler(30)
        cond_both = torch.cat([condition, torch.zeros_like(condition)], 0)
        steps = []
        t0 = time.time()
        for i, t in enumerate(sched.timesteps):
            if overlap:
                tv = float(t)
                latents[..., :overlap] = (1.0 - (1.0 - BLEND_EPS) * tv) * noise_prompt + tv * prev_lat[..., :overlap]
            v = model(latents.expand(2, -1, -1).contiguous(), t.reshape(1).expand(2), cond_both)
            v = v[1:2] + DIT_CFG_SCALE * (v[0:1] - v[1:2])
            latents = sched.step(v, t, latents)
            steps.append(latents.clone())
            print(f"chunk {k} step {i} t={float(t):.4f} {time.time() - t0:.0f}s", flush=True)
        if overlap:
            latents[..., :overlap] = prev_lat[..., :overlap]
        os_, oe = max(0, L - 2 * OVERLAP_LATENT_LENGTH), max(
            max(0, L - 2 * OVERLAP_LATENT_LENGTH), L - OVERLAP_LATENT_LENGTH
        )
        prev_lat, prev_cond = latents[..., os_:oe].clone(), condition[:, os_:oe].clone()
        err = (latents - g["latents"][k]).abs().max().item()
        print(f"chunk {k}: max abs diff vs golden final latent {err:.3e}", flush=True)
        records.append({"chunk": k, "steps": torch.stack(steps), "final": latents, "golden_max_abs_diff": err})
torch.save(records, out)
print("wrote", out)
