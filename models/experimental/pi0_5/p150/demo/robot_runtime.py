# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-robot control loop for the pi0.5 TT policy.

`run_realrobot` mirrors `libero_sim/libero_rollout.py::run_episode` but sources
observations from a `RobotInterface` and applies actions through it, instead of
the LIBERO simulator. The TT policy (`Pi0_5LiberoAdapter.predict_chunk`) is used
unchanged.

Implement `RobotInterface` for your arm + cameras; `MockRobot` lets the whole
loop run with no hardware (for wiring / latency checks).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class RobotInterface(ABC):
    """Hardware boundary for the demo — implement for your arm + cameras.

    Contract (LIBERO-compatible embodiment: 7-DoF delta-EE + gripper):
      capture()      -> (agentview_img, wrist_img), each (H, W, 3) uint8 in the
                        SAME orientation the policy was trained on (upright). Any
                        camera-mount rotation is the implementation's job.
      get_state()    -> np.ndarray shape (8,) float32 =
                        [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)].
                        (In LIBERO this is built from robosuite obs; a real arm
                        computes the EEF axis-angle from its pose quaternion.)
      send_action(a) -> apply one 7-D action [Δpos(3), Δaxis_angle(3), gripper]
                        (raw robot space — already denormalized by the policy).
      reset()        -> move to a start pose / clear episode state (optional).
      is_done()      -> True to stop the loop early (task complete) (optional).
    """

    @abstractmethod
    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (agentview_img, wrist_img), each (H, W, 3) uint8."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the 8-D robot state [eef_pos(3), axis_angle(3), gripper(2)]."""

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        """Apply one 7-D action to the robot."""

    def reset(self) -> None:
        """Optional: move to a start pose / clear state before an episode."""

    def is_done(self) -> bool:
        """Optional: return True to end the loop early (task complete)."""
        return False


class MockRobot(RobotInterface):
    """No-hardware stub: serves fixed (or provided) frames + a fixed state and
    records the actions it is sent. Use to exercise the full policy loop and
    measure per-chunk latency without a robot.
    """

    def __init__(
        self,
        agent_img: Optional[np.ndarray] = None,
        wrist_img: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        image_size: int = 256,
    ):
        self._agent = agent_img if agent_img is not None else np.zeros((image_size, image_size, 3), np.uint8)
        self._wrist = wrist_img if wrist_img is not None else np.zeros((image_size, image_size, 3), np.uint8)
        self._state = state if state is not None else np.zeros(8, np.float32)
        self.sent_actions = []  # actions passed to send_action (for inspection)

    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._agent, self._wrist

    def get_state(self) -> np.ndarray:
        return self._state

    def send_action(self, action: np.ndarray) -> None:
        self.sent_actions.append(np.asarray(action, dtype=np.float64))


def run_realrobot(
    adapter,
    robot: RobotInterface,
    task_desc: str,
    *,
    max_steps: int = 1000,
    replan_steps: int = 5,
    num_denoising_steps: int = 5,
    enable_motion: bool = False,
    verbose: bool = True,
) -> dict:
    """Drive `robot` with the pi0.5 TT policy for one episode.

    Loop (mirrors run_episode): reset → while steps < max_steps:
        obs = capture() + get_state()
        chunk = adapter.predict_chunk(agent, wrist, state, task_desc, N)   # on TT
        apply the first `replan_steps` actions, then re-plan.
    Stops on robot.is_done() or max_steps.

    Safety: with `enable_motion=False` (default) actions are logged but NOT sent
    to the robot. Set `enable_motion=True` to command the arm.

    Note: for the ttnn_1x8 backend the FIRST predict_chunk for a task also
    captures the trace (slower); subsequent chunks replay it (~31 ms).
    """
    robot.reset()
    n_steps = 0
    pred_times = []
    stop_reason = "max_steps"

    while n_steps < max_steps:
        agent_img, wrist_img = robot.capture()
        state = np.asarray(robot.get_state(), dtype=np.float32)

        t0 = time.perf_counter()
        chunk = adapter.predict_chunk(
            agent_img,
            wrist_img,
            state,
            task_desc,
            num_denoising_steps=num_denoising_steps,
        )
        dt = time.perf_counter() - t0
        pred_times.append(dt)
        if verbose:
            act0 = np.array2string(np.asarray(chunk[0]), precision=3, suppress_small=True)
            tag = "SEND" if enable_motion else "log-only"
            print(
                f"  chunk {len(pred_times)} @ step {n_steps}: pred {dt * 1000:.0f}ms  act[0]={act0}  [{tag}]",
                flush=True,
            )

        for a in chunk[:replan_steps]:
            if enable_motion:
                robot.send_action(np.asarray(a, dtype=np.float64))
            n_steps += 1
            if robot.is_done():
                stop_reason = "done"
                break
            if n_steps >= max_steps:
                break
        if stop_reason == "done":
            break

    result = {
        "steps": n_steps,
        "stop_reason": stop_reason,
        "n_chunks": len(pred_times),
        "avg_chunk_pred_ms": float(np.mean(pred_times)) * 1000.0 if pred_times else 0.0,
    }
    if verbose:
        print(f"  done: {result}", flush=True)
    return result
