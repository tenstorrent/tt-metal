# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Async rollout: overlaps inference with action execution so the arm never stalls.

While the robot executes chunk N's actions (500 ms at 20 Hz, replan=10),
the device computes chunk N+1's inference (~37-56 ms) in a background thread.
The arm runs continuously — inference latency is fully hidden behind execution.

    ┌──────────────────────────────────────────────────────────────────┐
    │  arm: execute chunk 0     │  arm: execute chunk 1     │  ...    │
    │  device:  (idle)  │ infer │  device:  (idle)  │ infer │  ...    │
    │                   │chunk 1│                   │chunk 2│         │
    └──────────────────────────────────────────────────────────────────┘

Usage (same env as the sync rollout):

    PI05_CHECKPOINT_DIR=/.../pi05_libero_upstream PI0_UPSTREAM_MASKS=1 \\
    QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 \\
    PYTHONPATH=$PWD python_env/bin/python \\
      models/experimental/pi0_5/eval/async_rollout.py \\
      --backend ttnn --suite libero_spatial --task 0 --seeds 0

Comparison flags:
    --async       (default) overlapped inference + execution
    --sync        sequential baseline (same as run_episode in libero_rollout.py)

Reports per-episode: success, steps, avg chunk prediction time, arm idle time
(time the arm waited for inference when the action buffer ran dry).
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

import numpy as np


# ── obs → adapter inputs ────────────────────────────────────────────
def extract_obs(obs):
    agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    quat_xyzw = obs["robot0_eef_quat"].astype(np.float32)
    w = float(np.clip(quat_xyzw[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sinh = max(float(np.sqrt(max(1.0 - w * w, 0.0))), 1e-8)
    axis_angle = (quat_xyzw[:3] / sinh) * angle
    state = np.concatenate(
        [
            obs["robot0_eef_pos"].astype(np.float32),
            axis_angle.astype(np.float32),
            obs["robot0_gripper_qpos"].astype(np.float32),
        ]
    )
    return agent_img, wrist_img, state


# ── async episode runner ────────────────────────────────────────────
def run_episode_async(
    env,
    adapter,
    task_desc: str,
    num_denoising_steps: int,
    max_steps: int = 200,
    replan_steps: int = 10,
    num_steps_wait: int = 10,
    initial_state=None,
    seed: int = 0,
    record_frames: bool = False,
) -> dict:
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = (
            env.regenerate_obs_from_state(env.get_sim_state())
            if hasattr(env, "regenerate_obs_from_state")
            else env.reset()
        )
    frames = [] if record_frames else None
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        if record_frames:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

    success = False
    n_steps = 0
    inference_times = []
    arm_idle_ms = 0.0

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05_infer")

    # Bootstrap: first chunk must be synchronous (no prior actions to execute).
    agent_img, wrist_img, state = extract_obs(obs)
    t0 = time.perf_counter()
    current_chunk = adapter.predict_chunk(
        agent_img,
        wrist_img,
        state,
        task_desc,
        num_denoising_steps=num_denoising_steps,
    )
    dt = time.perf_counter() - t0
    inference_times.append(dt)
    print(f"      chunk 1 (bootstrap, sync): {dt:.3f}s", flush=True)

    chunk_idx = 1
    pending_future: Optional[Future] = None

    while n_steps < max_steps and not success:
        # Kick off NEXT chunk's inference in the background thread.
        # Uses the CURRENT obs (latest frame before we start executing).
        agent_img, wrist_img, state = extract_obs(obs)
        infer_start = time.perf_counter()
        pending_future = executor.submit(
            adapter.predict_chunk,
            agent_img,
            wrist_img,
            state,
            task_desc,
            num_denoising_steps,
        )

        # Execute CURRENT chunk's actions while inference runs in background.
        for a in current_chunk[:replan_steps]:
            obs, _, done, _ = env.step(a.astype(np.float64))
            n_steps += 1
            if record_frames:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if done:
                success = True
                break
            if n_steps >= max_steps:
                break

        # Collect the next chunk. If inference finished before execution,
        # this returns immediately. Otherwise we measure the stall.
        wait_start = time.perf_counter()
        next_chunk = pending_future.result()
        wait_end = time.perf_counter()

        infer_dt = wait_end - infer_start
        inference_times.append(infer_dt)
        chunk_idx += 1

        # Did the arm have to wait (stall)?
        exec_time = wait_start - infer_start  # time spent executing actions
        if infer_dt > exec_time:
            stall = (infer_dt - exec_time) * 1000.0
            arm_idle_ms += stall
            print(f"      chunk {chunk_idx}: {infer_dt:.3f}s " f"(ARM STALLED {stall:.1f}ms)", flush=True)
        else:
            print(
                f"      chunk {chunk_idx}: {infer_dt:.3f}s " f"(hidden, {(exec_time - infer_dt)*1000:.0f}ms spare)",
                flush=True,
            )

        current_chunk = next_chunk
        pending_future = None

    # Cancel any in-flight inference if we exited early (success/max_steps).
    if pending_future is not None and not pending_future.done():
        pending_future.cancel()
    executor.shutdown(wait=False)

    out = {
        "success": success,
        "steps": n_steps,
        "n_chunks": len(inference_times),
        "avg_chunk_pred_time": float(np.mean(inference_times)) if inference_times else 0.0,
        "arm_idle_ms": arm_idle_ms,
    }
    if record_frames:
        out["frames"] = frames
    return out


# ── sync baseline (same logic as libero_rollout.run_episode) ────────
def run_episode_sync(
    env,
    adapter,
    task_desc: str,
    num_denoising_steps: int,
    max_steps: int = 200,
    replan_steps: int = 10,
    num_steps_wait: int = 10,
    initial_state=None,
    seed: int = 0,
    record_frames: bool = False,
) -> dict:
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = (
            env.regenerate_obs_from_state(env.get_sim_state())
            if hasattr(env, "regenerate_obs_from_state")
            else env.reset()
        )
    frames = [] if record_frames else None
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        if record_frames:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

    success = False
    n_steps = 0
    inference_times = []

    while n_steps < max_steps:
        agent_img, wrist_img, state = extract_obs(obs)
        t0 = time.perf_counter()
        chunk = adapter.predict_chunk(
            agent_img,
            wrist_img,
            state,
            task_desc,
            num_denoising_steps=num_denoising_steps,
        )
        dt = time.perf_counter() - t0
        inference_times.append(dt)
        print(f"      chunk {len(inference_times)}: {dt:.3f}s (sync)", flush=True)
        for a in chunk[:replan_steps]:
            obs, _, done, _ = env.step(a.astype(np.float64))
            n_steps += 1
            if record_frames:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if done:
                success = True
                break
            if n_steps >= max_steps:
                break
        if success:
            break

    return {
        "success": success,
        "steps": n_steps,
        "n_chunks": len(inference_times),
        "avg_chunk_pred_time": float(np.mean(inference_times)) if inference_times else 0.0,
        "arm_idle_ms": sum(inference_times) * 1000.0,  # sync = arm always idle during inference
    }


# ── main ────────────────────────────────────────────────────────────
def main():
    from models.experimental.pi0_5.eval.libero_rollout import (
        Pi0_5LiberoAdapter,
        make_libero_env,
        SUITE_MAX_STEPS,
    )

    ap = argparse.ArgumentParser(description="pi0.5 async vs sync rollout comparison")
    ap.add_argument("--backend", default="ttnn", choices=["pytorch", "ttnn"])
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument(
        "--mode", default="both", choices=["async", "sync", "both"], help="Run async, sync, or both (for comparison)"
    )
    args = ap.parse_args()

    checkpoint_dir = os.environ.get(
        "PI05_CHECKPOINT_DIR",
        str(os.path.join(os.path.dirname(__file__), "..", "weights", "pi05_base")),
    )
    print(f"Loading adapter: backend={args.backend}, checkpoint={checkpoint_dir}")
    adapter = Pi0_5LiberoAdapter(checkpoint_dir, backend=args.backend)

    env, task, initial_states = make_libero_env(args.suite, args.task)
    task_desc = task.language
    max_steps = SUITE_MAX_STEPS.get(args.suite, 220)
    print(f"Task: {task_desc} | max_steps={max_steps} | replan={args.replan_steps} " f"| denoise={args.denoise_steps}")

    modes = ["async", "sync"] if args.mode == "both" else [args.mode]

    for mode in modes:
        runner = run_episode_async if mode == "async" else run_episode_sync
        print(f"\n{'='*72}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'='*72}")

        results = []
        for seed in args.seeds:
            init_st = initial_states[seed % len(initial_states)]
            print(f"\n  seed={seed}:")
            m = runner(
                env,
                adapter,
                task_desc,
                num_denoising_steps=args.denoise_steps,
                max_steps=max_steps,
                replan_steps=args.replan_steps,
                initial_state=init_st,
                seed=seed,
            )
            results.append(m)
            print(
                f"    success={m['success']}  steps={m['steps']}  chunks={m['n_chunks']}  "
                f"avg_pred={m['avg_chunk_pred_time']:.3f}s  arm_idle={m['arm_idle_ms']:.1f}ms"
            )

        n_success = sum(r["success"] for r in results)
        avg_pred = np.mean([r["avg_chunk_pred_time"] for r in results])
        total_idle = sum(r["arm_idle_ms"] for r in results)
        print(
            f"\n  {mode.upper()} SUMMARY: {n_success}/{len(results)} success, "
            f"avg_pred={avg_pred:.3f}s, total_arm_idle={total_idle:.1f}ms"
        )

    if len(modes) == 2:
        print(f"\n{'='*72}")
        print("  COMPARISON")
        print(f"{'='*72}")
        async_idle = sum(r["arm_idle_ms"] for r in results[: len(args.seeds)])
        sync_idle = sum(r["arm_idle_ms"] for r in results[len(args.seeds) :])
        print(f"  Async arm idle: {async_idle:.1f} ms total across {len(args.seeds)} episodes")
        print(f"  Sync  arm idle: {sync_idle:.1f} ms total across {len(args.seeds)} episodes")
        if sync_idle > 0:
            print(f"  Reduction: {(1 - async_idle/sync_idle)*100:.1f}%")


if __name__ == "__main__":
    main()
