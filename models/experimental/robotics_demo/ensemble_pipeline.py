# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Ensemble pipeline: SmolVLA (fast) + PI0 (precise) running concurrently
with intelligent action fusion.

SmolVLA provides rapid coarse proposals (~229ms), PI0 provides refined
precise control (~330ms). Three fusion strategies blend their outputs
for superior robotic control.
"""

import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class FusionStrategy(Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    TEMPORAL_BLEND = "temporal_blend"
    CONFIDENCE_GATE = "confidence_gate"


def fuse_actions_weighted(
    pi0_actions: np.ndarray,
    smolvla_actions: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Weighted average fusion.

    alpha weights PI0 (precision), (1-alpha) weights SmolVLA (speed).
    Default alpha=0.6 biases toward PI0's higher-quality predictions.
    """
    pi0 = np.asarray(pi0_actions, dtype=np.float32)
    smol = np.asarray(smolvla_actions, dtype=np.float32)

    min_horizon = min(pi0.shape[0], smol.shape[0])
    min_dim = min(pi0.shape[1], smol.shape[1])
    pi0 = pi0[:min_horizon, :min_dim]
    smol = smol[:min_horizon, :min_dim]

    return alpha * pi0 + (1 - alpha) * smol


def fuse_actions_temporal(
    pi0_actions: np.ndarray,
    smolvla_actions: np.ndarray,
    crossover_step: int = 10,
) -> np.ndarray:
    """
    Temporal blending: SmolVLA actions for the near future (fast reflexes),
    PI0 actions for later in the horizon (precise planning).

    Smooth sigmoid crossover around crossover_step.
    """
    pi0 = np.asarray(pi0_actions, dtype=np.float32)
    smol = np.asarray(smolvla_actions, dtype=np.float32)

    min_horizon = min(pi0.shape[0], smol.shape[0])
    min_dim = min(pi0.shape[1], smol.shape[1])
    pi0 = pi0[:min_horizon, :min_dim]
    smol = smol[:min_horizon, :min_dim]

    t = np.arange(min_horizon, dtype=np.float32)
    # Sigmoid: 0 at early steps (SmolVLA), 1 at late steps (PI0)
    pi0_weight = 1.0 / (1.0 + np.exp(-(t - crossover_step) / 2.0))
    pi0_weight = pi0_weight[:, np.newaxis]

    return pi0_weight * pi0 + (1.0 - pi0_weight) * smol


def fuse_actions_confidence(
    pi0_actions: np.ndarray,
    smolvla_actions: np.ndarray,
) -> np.ndarray:
    """
    Confidence-gated fusion: per-timestep, pick the model whose
    action vector has lower variance (higher confidence / more decisive).
    """
    pi0 = np.asarray(pi0_actions, dtype=np.float32)
    smol = np.asarray(smolvla_actions, dtype=np.float32)

    min_horizon = min(pi0.shape[0], smol.shape[0])
    min_dim = min(pi0.shape[1], smol.shape[1])
    pi0 = pi0[:min_horizon, :min_dim]
    smol = smol[:min_horizon, :min_dim]

    pi0_var = np.var(pi0, axis=1, keepdims=True)
    smol_var = np.var(smol, axis=1, keepdims=True)

    # Lower variance = more confident; use that model's actions
    use_pi0 = (pi0_var <= smol_var).astype(np.float32)
    return use_pi0 * pi0 + (1.0 - use_pi0) * smol


FUSION_FNS = {
    FusionStrategy.WEIGHTED_AVERAGE: fuse_actions_weighted,
    FusionStrategy.TEMPORAL_BLEND: fuse_actions_temporal,
    FusionStrategy.CONFIDENCE_GATE: fuse_actions_confidence,
}


class EnsemblePipeline:
    """
    Orchestrates concurrent PI0 + SmolVLA inference and fuses their outputs.

    SmolVLA runs on one chip (fast ~229ms), PI0 on another (precise ~330ms).
    Both receive the same observations; their actions are fused before
    being applied to the robot.
    """

    def __init__(
        self,
        pi0_model,
        smolvla_model,
        fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        alpha: float = 0.6,
        crossover_step: int = 10,
    ):
        self.pi0_model = pi0_model
        self.smolvla_model = smolvla_model
        self.fusion_strategy = fusion_strategy
        self.alpha = alpha
        self.crossover_step = crossover_step

        self._pi0_result = None
        self._smolvla_result = None
        self._pi0_time = 0.0
        self._smolvla_time = 0.0

    def run_concurrent_inference(
        self,
        pi0_inputs: Dict,
        smolvla_images,
        smolvla_instruction: str,
        num_inference_steps: int = 10,
        action_dim: int = 7,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run both models concurrently using threads, then fuse.

        Returns:
            (fused_actions, timing_info)
        """
        self._pi0_result = None
        self._smolvla_result = None

        def _run_pi0():
            t0 = time.time()
            with torch.no_grad():
                self._pi0_result = self.pi0_model.sample_actions(**pi0_inputs)
            self._pi0_time = (time.time() - t0) * 1000

        def _run_smolvla():
            t0 = time.time()
            with torch.no_grad():
                self._smolvla_result = self.smolvla_model.sample_actions(
                    images=smolvla_images,
                    instruction=smolvla_instruction,
                    num_inference_steps=num_inference_steps,
                    action_dim=action_dim,
                )
            self._smolvla_time = (time.time() - t0) * 1000

        t_pi0 = threading.Thread(target=_run_pi0)
        t_smol = threading.Thread(target=_run_smolvla)
        t_start = time.time()
        t_pi0.start()
        t_smol.start()
        t_pi0.join()
        t_smol.join()
        wall_time = (time.time() - t_start) * 1000

        pi0_actions = self._to_numpy(self._pi0_result)
        smolvla_actions = self._to_numpy(self._smolvla_result)

        fuse_fn = FUSION_FNS[self.fusion_strategy]
        kwargs = {}
        if self.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            kwargs["alpha"] = self.alpha
        elif self.fusion_strategy == FusionStrategy.TEMPORAL_BLEND:
            kwargs["crossover_step"] = self.crossover_step

        fused = fuse_fn(pi0_actions, smolvla_actions, **kwargs)

        timing = {
            "pi0_ms": self._pi0_time,
            "smolvla_ms": self._smolvla_time,
            "wall_ms": wall_time,
            "speedup_vs_sequential": (self._pi0_time + self._smolvla_time) / wall_time if wall_time > 0 else 0,
            "fusion_strategy": self.fusion_strategy.value,
        }
        return fused, timing

    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        if tensor is None:
            return np.zeros((50, 32), dtype=np.float32)
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "numpy"):
            arr = tensor.float().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return arr
