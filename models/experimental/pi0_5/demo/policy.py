# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Headless policy surface for real-robot / server deployment.

The pi0.5 policy is already decoupled from the LIBERO simulator behind
`Pi0_5LiberoAdapter` (in `libero_sim/libero_rollout.py`): it takes two uint8
camera images + an 8-D robot state + a task string, does all preprocessing
(resize/pad→224, tokenize, normalize) and action denormalization internally,
and runs on TT kernels via the `ttnn` (single chip) or `ttnn_1x8` (8-chip mesh,
trace+2CQ) backends. The LIBERO env is only the obs source + action sink.

This module is the neutral entry point for driving that policy WITHOUT the sim.
`Pi0_5LiberoAdapter` is re-exported here (importing it does NOT pull in
LIBERO/robosuite/mujoco — those imports are lazy in `libero_rollout`), and
`build_policy()` opens the device/mesh + constructs the adapter, mirroring
`libero_rollout.main`'s backend setup.

Embodiment: the upstream `pi05_libero` checkpoint expects the LIBERO 7-DoF
delta-EE + gripper action space and 8-D state; a different robot needs its own
norm_stats + checkpoint.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

# Re-export the sim-agnostic policy wrapper. Headless-safe: libero_rollout's
# LIBERO/robosuite imports are lazy (inside make_libero_env), so importing this
# does not require the simulator to be installed.
from models.experimental.pi0_5.libero_sim.libero_rollout import Pi0_5LiberoAdapter  # noqa: F401

# 1×8-mesh-specific flags — NOT in pi05_production.env; they control how
# attention/MLP weights shard across the 8 chips. Must be set before the
# pipeline is constructed (same set libero_rollout.main + the perf test apply).
_TP8_FLAGS = {
    "PI0_TP": "8",
    "PI0_TP8_ATTN_HEADPAR": "1",
    "PI0_MLP_BS": "1",
    "PI0_MLP_FUSED_RS": "0",
}


@contextmanager
def build_policy(
    checkpoint: str,
    backend: str = "ttnn_1x8",
    action_horizon: Optional[int] = None,
    state_in_prompt: bool = False,
    device_id: int = 0,
    tokenizer_path: Optional[str] = None,
):
    """Open the device/mesh + build a ready `Pi0_5LiberoAdapter`, then clean up.

    Yields the adapter. Call `adapter.predict_chunk(agent_img, wrist_img, state,
    task_desc, num_denoising_steps=N)` per control step.

    Args:
        checkpoint: pi05_libero checkpoint dir (model.safetensors + config.json +
            assets/.../norm_stats.json).
        backend: "ttnn_1x8" (8-chip mesh, trace+2CQ, ~31 ms/chunk) | "ttnn"
            (single chip) | "pytorch" (CPU reference).
        action_horizon: chunk size; auto-read from the checkpoint's config.json
            when None (10 for upstream pi05_libero).
        state_in_prompt: False = upstream openpi (task-prompt only, QUANTILE norm);
            True = lerobot finetune (state discretized into the prompt, MEAN_STD).
        device_id: single-chip device id (backend="ttnn" only).
        tokenizer_path: PaliGemma SentencePiece model; defaults to $PI0_TOKENIZER_PATH.

    Usage:
        with build_policy(ckpt, backend="ttnn_1x8") as adapter:
            chunk = adapter.predict_chunk(agent_img, wrist_img, state, "pick up the cup")
    """
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.prod_env import apply_production_env_defaults

    # Production perf flags (fold path, matmul tuning, upstream masks, ...) must
    # be in the environment before the TTNN pipeline is built.
    apply_production_env_defaults()
    if backend == "ttnn_1x8":
        for _k, _v in _TP8_FLAGS.items():
            os.environ.setdefault(_k, _v)

    if action_horizon is None:
        action_horizon = action_horizon_from_checkpoint(checkpoint)

    extra = {"tokenizer_path": tokenizer_path} if tokenizer_path else {}
    ttnn_device = None
    mesh_ctx = None
    mesh_handles = None
    try:
        if backend == "ttnn":
            import ttnn

            ttnn_device = ttnn.open_device(device_id=device_id, l1_small_size=24576, trace_region_size=134_217_728)
        elif backend == "ttnn_1x8":
            from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp8_mesh

            # 256 MiB trace region: per-task trace re-capture exceeds 128 MiB on
            # some tasks. num_command_queues=2 → H2D upload overlaps CQ0 compute.
            mesh_ctx = open_prefill_tp8_mesh(
                tp=8,
                l1_small_size=24576,
                trace_region_size=256 * 1024 * 1024,
                num_command_queues=2,
            )
            mesh_handles = mesh_ctx.__enter__()
        elif backend != "pytorch":
            raise ValueError(f"Unknown backend: {backend!r}")

        adapter = Pi0_5LiberoAdapter(
            checkpoint,
            backend=backend,
            ttnn_device=ttnn_device,
            mesh_handles=mesh_handles,
            chunk_size=action_horizon,
            action_horizon=action_horizon,
            state_in_prompt=state_in_prompt,
            **extra,
        )
        yield adapter
    finally:
        if mesh_ctx is not None:
            mesh_ctx.__exit__(None, None, None)
        elif ttnn_device is not None:
            import ttnn

            ttnn.close_device(ttnn_device)
