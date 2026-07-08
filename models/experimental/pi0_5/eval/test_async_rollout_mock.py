# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Validate the async rollout with a mock env and mock adapter.

Mock timing:
  - predict_chunk: sleeps for INFER_MS (simulating device compute)
  - env.step: sleeps for CONTROL_PERIOD_MS (simulating 20 Hz robot control)

This proves the overlap works and measures arm stall correctly without
needing LIBERO / MuJoCo installed.

Usage:
    python models/experimental/pi0_5/eval/test_async_rollout_mock.py
"""
import time
import numpy as np
from models.experimental.pi0_5.eval.async_rollout import (
    run_episode_async,
    run_episode_sync,
)

INFER_MS = 56.0  # simulate 10-step inference
CONTROL_MS = 50.0  # 20 Hz control
REPLAN = 10  # execute 10 actions per chunk
MAX_STEPS = 50  # short episode for testing
DENOISE_STEPS = 10


class MockAdapter:
    def predict_chunk(self, agent_img, wrist_img, state, task_desc, num_denoising_steps=10):
        time.sleep(INFER_MS / 1000.0)
        return np.zeros((50, 7), dtype=np.float64)


class MockEnv:
    def __init__(self):
        self._steps = 0
        self._dummy_obs = {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        }

    def reset(self):
        self._steps = 0
        return self._dummy_obs

    def set_init_state(self, state):
        return self._dummy_obs

    def step(self, action):
        time.sleep(CONTROL_MS / 1000.0)
        self._steps += 1
        done = False
        return self._dummy_obs, 0, done, {}


def main():
    env = MockEnv()
    adapter = MockAdapter()
    task_desc = "mock task"

    print(f"Config: infer={INFER_MS}ms, control={CONTROL_MS}ms, " f"replan={REPLAN}, max_steps={MAX_STEPS}")
    print(f"Expected per replan cycle:")
    print(
        f"  sync:  {INFER_MS + REPLAN * CONTROL_MS:.0f}ms "
        f"(infer {INFER_MS}ms + execute {REPLAN * CONTROL_MS:.0f}ms)"
    )
    print(f"  async: {max(INFER_MS, REPLAN * CONTROL_MS):.0f}ms " f"(max of infer, execute) — inference hidden")
    print()

    # --- ASYNC ---
    print("=" * 60)
    print("  ASYNC")
    print("=" * 60)
    t0 = time.perf_counter()
    r_async = run_episode_async(
        env,
        adapter,
        task_desc,
        num_denoising_steps=DENOISE_STEPS,
        max_steps=MAX_STEPS,
        replan_steps=REPLAN,
        num_steps_wait=0,
    )
    async_wall = (time.perf_counter() - t0) * 1000
    print(
        f"  wall={async_wall:.0f}ms  steps={r_async['steps']}  "
        f"chunks={r_async['n_chunks']}  arm_idle={r_async['arm_idle_ms']:.1f}ms"
    )

    # --- SYNC ---
    print()
    print("=" * 60)
    print("  SYNC")
    print("=" * 60)
    t0 = time.perf_counter()
    r_sync = run_episode_sync(
        env,
        adapter,
        task_desc,
        num_denoising_steps=DENOISE_STEPS,
        max_steps=MAX_STEPS,
        replan_steps=REPLAN,
        num_steps_wait=0,
    )
    sync_wall = (time.perf_counter() - t0) * 1000
    print(
        f"  wall={sync_wall:.0f}ms  steps={r_sync['steps']}  "
        f"chunks={r_sync['n_chunks']}  arm_idle={r_sync['arm_idle_ms']:.1f}ms"
    )

    # --- Comparison ---
    print()
    print("=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    n_chunks = r_async["n_chunks"]
    expected_sync = n_chunks * (INFER_MS + REPLAN * CONTROL_MS)
    expected_async = INFER_MS + n_chunks * REPLAN * CONTROL_MS  # bootstrap + overlapped
    print(f"  Async wall:     {async_wall:7.0f}ms  (expected ~{expected_async:.0f}ms)")
    print(f"  Sync wall:      {sync_wall:7.0f}ms  (expected ~{expected_sync:.0f}ms)")
    print(f"  Speedup:        {sync_wall/async_wall:.2f}x")
    print(f"  Async arm idle: {r_async['arm_idle_ms']:7.1f}ms")
    print(f"  Sync arm idle:  {r_sync['arm_idle_ms']:7.1f}ms")
    saved = r_sync["arm_idle_ms"] - r_async["arm_idle_ms"]
    print(f"  Arm idle saved: {saved:7.1f}ms ({saved/r_sync['arm_idle_ms']*100:.0f}%)")


if __name__ == "__main__":
    main()
