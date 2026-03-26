# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Real-time metrics collection for the robotics demo suite.

Tracks inference latency, control frequency, distance-to-target,
action smoothness, and scaling efficiency across multiple environments.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EnvironmentMetrics:
    """Per-environment metrics snapshot."""

    env_id: int = 0
    model_name: str = ""
    step: int = 0
    inference_time_ms: float = 0.0
    loop_time_ms: float = 0.0
    control_freq_hz: float = 0.0
    ee_position: Optional[np.ndarray] = None
    target_position: Optional[np.ndarray] = None
    distance_to_target: float = float("inf")
    action_norm: float = 0.0
    is_inference_step: bool = False


class MetricsCollector:
    """
    Thread-safe metrics collection and aggregation.

    Collects per-environment, per-step data and provides
    aggregate statistics for dashboard display.
    """

    def __init__(self, num_envs: int, history_len: int = 500):
        self.num_envs = num_envs
        self.history_len = history_len
        self._lock = threading.Lock()

        self._inference_times: Dict[int, deque] = {
            i: deque(maxlen=history_len) for i in range(num_envs)
        }
        self._loop_times: Dict[int, deque] = {
            i: deque(maxlen=history_len) for i in range(num_envs)
        }
        self._distances: Dict[int, deque] = {
            i: deque(maxlen=history_len) for i in range(num_envs)
        }
        self._action_norms: Dict[int, deque] = {
            i: deque(maxlen=history_len) for i in range(num_envs)
        }
        self._steps: Dict[int, int] = {i: 0 for i in range(num_envs)}
        self._model_names: Dict[int, str] = {}
        self._total_inferences: Dict[int, int] = {i: 0 for i in range(num_envs)}

        self._start_time = time.time()

    def record(self, metrics: EnvironmentMetrics):
        """Record a metrics snapshot for one environment step."""
        with self._lock:
            eid = metrics.env_id
            self._steps[eid] = metrics.step
            self._model_names[eid] = metrics.model_name
            self._loop_times[eid].append(metrics.loop_time_ms)
            self._distances[eid].append(metrics.distance_to_target)
            self._action_norms[eid].append(metrics.action_norm)

            if metrics.is_inference_step and metrics.inference_time_ms > 0:
                self._inference_times[eid].append(metrics.inference_time_ms)
                self._total_inferences[eid] += 1

    def get_env_summary(self, env_id: int) -> Dict:
        """Get summary stats for a single environment."""
        with self._lock:
            inf_times = list(self._inference_times[env_id])
            loop_times = list(self._loop_times[env_id])
            distances = list(self._distances[env_id])

            return {
                "env_id": env_id,
                "model_name": self._model_names.get(env_id, "unknown"),
                "step": self._steps[env_id],
                "total_inferences": self._total_inferences[env_id],
                "avg_inference_ms": float(np.mean(inf_times)) if inf_times else 0.0,
                "std_inference_ms": float(np.std(inf_times)) if inf_times else 0.0,
                "min_inference_ms": float(np.min(inf_times)) if inf_times else 0.0,
                "max_inference_ms": float(np.max(inf_times)) if inf_times else 0.0,
                "avg_loop_ms": float(np.mean(loop_times)) if loop_times else 0.0,
                "control_freq_hz": 1000.0 / np.mean(loop_times) if loop_times and np.mean(loop_times) > 0 else 0.0,
                "current_distance": distances[-1] if distances else float("inf"),
                "min_distance": float(np.min(distances)) if distances else float("inf"),
                "distance_history": distances[-50:],
                "inference_history": inf_times[-50:],
            }

    def get_global_summary(self) -> Dict:
        """Get aggregate summary across all environments."""
        elapsed = time.time() - self._start_time
        env_summaries = [self.get_env_summary(i) for i in range(self.num_envs)]

        total_inferences = sum(s["total_inferences"] for s in env_summaries)
        all_inf_times = []
        for s in env_summaries:
            all_inf_times.extend(s["inference_history"])

        return {
            "num_envs": self.num_envs,
            "elapsed_seconds": elapsed,
            "total_inferences": total_inferences,
            "aggregate_throughput_fps": total_inferences / elapsed if elapsed > 0 else 0.0,
            "avg_inference_ms": float(np.mean(all_inf_times)) if all_inf_times else 0.0,
            "per_env": env_summaries,
        }

    def get_scaling_efficiency(self) -> Dict:
        """
        Compute scaling efficiency: compare N-env throughput vs 1-env baseline.

        Returns dict with per-env FPS and scaling factor.
        """
        summaries = [self.get_env_summary(i) for i in range(self.num_envs)]
        per_env_fps = []
        for s in summaries:
            if s["avg_inference_ms"] > 0:
                per_env_fps.append(1000.0 / s["avg_inference_ms"])
            else:
                per_env_fps.append(0.0)

        single_fps = per_env_fps[0] if per_env_fps else 0.0
        total_fps = sum(per_env_fps)

        return {
            "per_env_fps": per_env_fps,
            "total_fps": total_fps,
            "single_chip_fps": single_fps,
            "scaling_factor": total_fps / single_fps if single_fps > 0 else 0.0,
            "ideal_scaling": float(self.num_envs),
            "efficiency_pct": (total_fps / (single_fps * self.num_envs) * 100)
            if single_fps > 0
            else 0.0,
        }


class LatencyTimer:
    """Context manager for timing code sections."""

    def __init__(self):
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self._start) * 1000.0
