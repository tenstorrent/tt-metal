# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Timed loop of the window attention (tt/attention.py:TtWindowAttention) of one layer at the run's chunk, for the
sparse_sdpa dtype variants (settings WINDOW_SDPA_SCORES / _PROBS / _OUT).

Not a gate: a measurement for BREADCRUMBS / the settings' descriptions (FIX.2). Random inputs (q_resid normal, kv on
the e4m3 grid, bf16), a chunk at start = chunk (the carry read from the cache), 2 warm-up calls, then the mean wall
time of N_CALLS synchronized calls (the forward has no host sync, so this is the device time of one layer's attention
per chunk; the cache write included).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tt/perf_attention.py
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.bringup import hooks
from models.demos.deepseek_v41_flash_d_p.tt import attention as A
from models.demos.deepseek_v41_flash_d_p.tt.layout import row_split_to_device
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

N_CALLS = 5
LAYER = 0
# name -> setting overrides (the rest at their defaults)
VARIANTS = {
    "reference (FIX.2 default: fp32 scores, bf16 q / probs / out)": {},
    "DS41-31 (fp32 scores, TF32 q / probs, fp32 out; before FIX.2)": {
        "WINDOW_SDPA_PROBS": "fp32",
        "WINDOW_SDPA_OUT": "fp32",
    },
    "bf16 scores, bf16 out": {"WINDOW_SDPA_SCORES": "bf16"},
}


@pytest.mark.timeout(3600)
@mesh_parametrize
def test_time_attention(mesh_device, monkeypatch):
    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    start = chunk
    g = torch.Generator().manual_seed(0)
    q_resid = torch.randn(chunk, cfg.q_lora_rank, generator=g)
    kv = (torch.randn(chunk, cfg.head_dim, generator=g)).to(torch.float8_e4m3fn).float()
    tables = A.WindowTables(mesh_device, chunk)
    outs = {}
    for name, over in VARIANTS.items():
        for k in ("WINDOW_SDPA_SCORES", "WINDOW_SDPA_PROBS", "WINDOW_SDPA_OUT"):
            monkeypatch.delenv(settings.env_name(k), raising=False)
        for k, v in over.items():
            monkeypatch.setenv(settings.env_name(k), v)
        mod = A.build_attention(mesh_device, ck, cfg, LAYER, 2 * chunk, chunk, tables=tables)
        cache = A.TtWindowCache(mesh_device, tables, 2 * chunk)
        qd = row_split_to_device(mesh_device, q_resid)
        kd = row_split_to_device(mesh_device, kv, dtype=ttnn.bfloat16)
        for _ in range(2):
            ttnn.deallocate(mod(qd, kd, start, None, cache))
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        for _ in range(N_CALLS):
            ttnn.deallocate(mod(qd, kd, start, None, cache))
        ttnn.synchronize_device(mesh_device)
        dt = (time.time() - t0) / N_CALLS
        od = mod(qd, kd, start, None, cache)
        outs[name] = ttnn.to_torch(ttnn.get_device_tensors(od)[0]).float()
        print(f"ATTN_TIME {name}: layer={LAYER} chunk={chunk} per_call_ms={dt * 1e3:.2f}")
        for t in (od, qd, kd):
            ttnn.deallocate(t)
        cache.free()
        del mod
    ref = outs[next(iter(VARIANTS))]
    for name, o in outs.items():
        rel = ((o - ref).norm() / ref.norm()).item()
        print(f"ATTN_DIFF {name} vs the default (chip 0): rel {rel:.5f}")
