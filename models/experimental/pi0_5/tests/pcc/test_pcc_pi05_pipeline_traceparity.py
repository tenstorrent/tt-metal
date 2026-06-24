# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional trace/replay parity WITHOUT tracy (plan iter-3 §8, SC3).

MANDATE: run as PLAIN pytest, NEVER under ``python -m tracy``, NO TT_SYMBIOTE_SIGNPOST_MODE
(tracy hangs on cross-chip negative-delta zones -- see PORT_NOTES "no-tracy mandate").

Validates: replay == eager == golden within PCC, on the 4-chip P150x4 streamed path.
  (1) eager  = stream_euler(capture=False)            -- full eager Euler loop (fresh driver)
  (2) replayed = stream_euler(capture=True)           -- a REPLAY output (capture + replay_loop
                                                          drain; per d2d_pipeline semantics
                                                          capture=True RETURNS the replayed x_t)
  Asserts: PCC(eager, replayed) >= 0.9999 (replay == eager),
           PCC(replayed, golden) >= 0.95 (target 0.99) via the shared _streamed_golden
           (replayed == golden), and PCC(eager, golden) >= 0.95.

ENVIRONMENT NOTE (PORT_NOTES "no-tracy mandate"): this build ships a real-time device profiler
that is ALWAYS-ON and emits "Skipping zone with end < start" on cross-chip negative-delta zones
during repeated trace REPLAY drains, hanging the process. A SECOND replay() drain in the same
process reproduces the iter-2 hang regardless of env. This test therefore validates trace parity
with a SINGLE capture+replay per driver (the e2e path), NOT a second standalone replay() drain;
the re-drain stability claim is documented-deferred (environmental, not a port defect).

Under the §9 reset harness. n=8 kept as a skip-on-<8-chips param.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

sys.path.insert(0, str(Path(__file__).parent))  # sibling helper modules (_fabric_harness, _streamed_golden)
from _fabric_harness import close_parent, open_parent_with_retry, reset_board  # noqa: E402
from _streamed_golden import compute_pcc, euler_golden  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/ttuser/salnahari/pi05_weights/pi05_base"))
SEED = 42
PCC_THRESHOLD = float(os.environ.get("PI05_PIPELINE_PCC", "0.95"))


@pytest.fixture(scope="module", autouse=True)
def _board_reset_session():
    reset_board()
    yield
    reset_board()


@pytest.mark.parametrize("n_submeshes", [4, 8])
@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(), reason=f"checkpoint not found at {CHECKPOINT_DIR}"
)
def test_trace_parity_vs_golden(n_submeshes):
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_pipeline import StageDenoise
    from models.experimental.pi0_5.tt.tt_pipeline._d2d_pipeline import Pipeline
    from models.experimental.pi0_5.tt.tt_pipeline.weight_adapt import suffix_reference

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

    sc = SuffixConfig(action_dim=cfg.action_dim, action_horizon=ah, expert_width=ec.width, pi05=True)
    suffix_ref = suffix_reference(weights["pi0_projections"], sc)
    golden = euler_golden(cfg, weights, suffix_ref, prefix_kv_torch, x_t_init, num_steps, ah)

    parent = open_parent_with_retry(n_submeshes)
    try:
        # eager
        stage_eager = StageDenoise(cfg, weights, None, parent_mesh=parent, n_submeshes=n_submeshes)
        eager = stage_eager.sample_actions(
            x_t_init_torch=x_t_init,
            prefix_kv_cache_torch=prefix_kv_torch,
            num_steps=num_steps,
            prefix_len=prefix_len,
            action_horizon=ah,
            capture=False,
        )
        stage_eager.close()

        # captured + replay: stream_euler(capture=True) RETURNS the replayed x_t (a replay output)
        stage = StageDenoise(cfg, weights, None, parent_mesh=parent, n_submeshes=n_submeshes)
        replayed = stage.sample_actions(
            x_t_init_torch=x_t_init,
            prefix_kv_cache_torch=prefix_kv_torch,
            num_steps=num_steps,
            prefix_len=prefix_len,
            action_horizon=ah,
            capture=True,
        )
        stage.close()
        Pipeline.release_all()
    finally:
        close_parent(parent)

    pcc_self = compute_pcc(eager[:, :ah, :], replayed[:, :ah, :])
    pcc_golden = compute_pcc(replayed[:, :ah, :], golden)
    pcc_eager_golden = compute_pcc(eager[:, :ah, :], golden)
    print(
        f"\n[n={n_submeshes}] trace parity: eager-vs-replay={pcc_self:.6f} "
        f"replay-vs-golden={pcc_golden:.6f} eager-vs-golden={pcc_eager_golden:.6f}"
    )

    assert pcc_self >= 0.9999, f"eager-vs-replay PCC {pcc_self:.6f} < 0.9999"
    assert pcc_golden >= PCC_THRESHOLD, f"replay-vs-golden PCC {pcc_golden:.6f} < {PCC_THRESHOLD}"
    assert pcc_eager_golden >= PCC_THRESHOLD, f"eager-vs-golden PCC {pcc_eager_golden:.6f} < {PCC_THRESHOLD}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
