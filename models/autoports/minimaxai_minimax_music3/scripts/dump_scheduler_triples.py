# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dump (sample, velocity, t, prev_sample) tuples from diffusers' FlowMatchEulerDiscreteScheduler.

Run in the CPU reference venv (it has diffusers):

    source ~/mm3-bringup/common.sh && $MM3_REF_PY $MM3_MODEL_DIR/scripts/dump_scheduler_triples.py

Writes ``doc/flow_dit/pcc/scheduler_triples.pt`` (small), consumed by ``tests/test_flow_transformer.py``.
The schedule is the pipeline's: ``set_timesteps(sigmas=np.linspace(1, 1/N, N))`` with the checkpoint's
scheduler config (invert_sigmas=True, shift=1, num_train_timesteps=1).
"""
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

out = Path(__file__).resolve().parents[1] / "doc" / "flow_dit" / "pcc" / "scheduler_triples.pt"
out.parent.mkdir(parents=True, exist_ok=True)
weights = os.environ["MM3_WEIGHTS"]
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(weights, subfolder="scheduler")
print(sched.config)
g = torch.Generator().manual_seed(1234)
records = []
for n in (30, 8, 1):
    sched.set_timesteps(sigmas=np.linspace(1.0, 1.0 / n, n))
    sigmas, timesteps = sched.sigmas.clone(), sched.timesteps.clone()
    steps = []
    x = torch.randn(1, 128, 32, generator=g)
    for i, t in enumerate(timesteps):
        v = torch.randn(1, 128, 32, generator=g)
        prev = sched.step(v, t, x, return_dict=False)[0]
        if i < 3 or i == len(timesteps) - 1:
            steps.append(
                {"i": i, "t": t.clone(), "sample": x.clone(), "velocity": v.clone(), "prev_sample": prev.clone()}
            )
        x = prev
    records.append(
        {"num_inference_steps": n, "sigmas": sigmas, "timesteps": timesteps, "steps": steps, "final": x.clone()}
    )
torch.save({"config": dict(sched.config), "records": records}, out)
print("wrote", out, out.stat().st_size, "bytes")
