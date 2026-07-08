# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace capture/replay parity + coarse perf for the tt_pipeline streamed driver (plan §11.3).

For each of two standalone mesh setups (P150x4 / P150x8, each opening its OWN parent mesh):
  (1) replay() actions PCC >= 0.9999 vs the captured stream_euler output (self-consistency);
  (2) replay() PCC >= 0.99 vs the capture=False eager output (trace == eager);
  (3) report per-step wall-time (coarse; device-time claims are tracy-only per CLAUDE.md).
close() releases the loop trace + transports; Pipeline.release_all() in teardown.

8-way is the explicit 1-layer-stage checkpoint (M4) and is reachable ONLY here. Skips cleanly
without hardware or a checkpoint.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/ttuser/salnahari/pi05_weights/pi05_base"))
SEED = 42
_TRACE_REGION = 134_217_728


def _compute_pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


def _open_parent(n):
    if ttnn.get_num_devices() < n:
        pytest.skip(f"need >={n} chips, have {ttnn.get_num_devices()}")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, n), l1_small_size=24576, trace_region_size=_TRACE_REGION)


def _close_parent(parent):
    try:
        ttnn.close_mesh_device(parent)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.parametrize("n_submeshes", [4, 8])
@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(), reason=f"checkpoint not found at {CHECKPOINT_DIR}"
)
def test_trace_replay_parity(n_submeshes):
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_pipeline import StageDenoise
    from models.experimental.pi0_5.tt.tt_pipeline._d2d_pipeline import Pipeline

    num_steps = int(os.environ.get("PI05_PIPELINE_STEPS", "5"))
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR), num_denoising_steps=num_steps)
    weights = Pi0_5WeightLoader(str(CHECKPOINT_DIR)).categorized_weights
    ec = cfg.expert_config
    B = 1
    ah = cfg.action_horizon
    ah_pad = ((ah + 31) // 32) * 32
    prefix_len = 288

    torch.manual_seed(SEED)
    x_t_init = torch.zeros(B, ah_pad, cfg.action_dim)
    x_t_init[:, :ah, :] = torch.randn(B, ah, cfg.action_dim)
    prefix_kv_torch = [
        (
            torch.randn(B, ec.num_kv_heads, prefix_len, ec.head_dim) * 0.5,
            torch.randn(B, ec.num_kv_heads, prefix_len, ec.head_dim) * 0.5,
        )
        for _ in range(ec.depth)
    ]

    parent = _open_parent(n_submeshes)
    try:
        # eager (capture=False) reference
        stage_eager = StageDenoise(cfg, weights, None, parent_mesh=parent, n_submeshes=n_submeshes)
        eager_actions = stage_eager.sample_actions(
            x_t_init_torch=x_t_init,
            prefix_kv_cache_torch=prefix_kv_torch,
            num_steps=num_steps,
            prefix_len=prefix_len,
            action_horizon=ah,
            capture=False,
        )
        stage_eager.close()

        # captured + replay
        stage = StageDenoise(cfg, weights, None, parent_mesh=parent, n_submeshes=n_submeshes)
        captured = stage.sample_actions(
            x_t_init_torch=x_t_init,
            prefix_kv_cache_torch=prefix_kv_torch,
            num_steps=num_steps,
            prefix_len=prefix_len,
            action_horizon=ah,
            capture=True,
        )
        t0 = time.perf_counter()
        replayed = stage.replay()
        dt = time.perf_counter() - t0

        pcc_self = _compute_pcc(captured, replayed[:, :ah, :])
        pcc_eager = _compute_pcc(eager_actions[:, :ah, :], replayed[:, :ah, :])
        ms_per_step = dt * 1e3 / num_steps
        print(
            f"\n[n={n_submeshes}] replay self-PCC={pcc_self:.6f}  vs-eager PCC={pcc_eager:.6f}  "
            f"~{ms_per_step:.3f} ms/step (coarse wall-time)"
        )

        stage.close()
        Pipeline.release_all()
    finally:
        _close_parent(parent)

    assert pcc_self >= 0.9999, f"replay self-consistency PCC {pcc_self:.6f} < 0.9999"
    assert pcc_eager >= 0.99, f"trace-vs-eager PCC {pcc_eager:.6f} < 0.99"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
