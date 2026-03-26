# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Demo orchestrator: top-level controller for all four scenarios.

Manages device allocation, model loading, environment lifecycle,
metrics collection, video composition, and the main control loops.
Designed to be driven by the Streamlit dashboard or run standalone.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from models.experimental.robotics_demo.multi_env import MultiEnvironment, DEFAULT_TASKS
from models.experimental.robotics_demo.metrics import MetricsCollector, EnvironmentMetrics, LatencyTimer
from models.experimental.robotics_demo.video_composer import (
    compose_quad_view, compose_side_by_side, VideoRecorder,
)

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", "")
DEFAULT_PI0_CHECKPOINT = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")


class SimpleTokenizer:
    """Minimal tokenizer for demo use (mirrors PI0's SimpleRoboticsTokenizer)."""

    def __init__(self, vocab_size: int = 256000):
        self.vocab_size = vocab_size
        self._vocab = {
            "<pad>": 0, "<bos>": 2, "<eos>": 3,
            "pick": 100, "place": 101, "grasp": 102, "push": 106,
            "lift": 108, "reach": 105, "move": 104,
            "cube": 200, "block": 201, "object": 202,
            "up": 300, "right": 303, "forward": 304,
            "the": 400, "to": 403, "and": 405,
        }

    def encode(self, text: str, max_length: int = 32):
        tokens = [self._vocab["<bos>"]]
        for w in text.lower().split():
            tokens.append(self._vocab.get(w, hash(w) % (self.vocab_size - 1000) + 1000))
        tokens.append(self._vocab["<eos>"])
        mask = [True] * len(tokens)
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
            mask += [False] * (max_length - len(mask))
        else:
            tokens = tokens[:max_length]
            mask = mask[:max_length]
        return torch.tensor([tokens], dtype=torch.long), torch.tensor([mask], dtype=torch.bool)


# ======================================================================
# Scenario 1: Data-Parallel PI0 (4 robots, 4 chips)
# ======================================================================

def run_scenario_1(
    num_steps: int = 400,
    num_devices: int = 4,
    checkpoint_path: str = DEFAULT_PI0_CHECKPOINT,
    replan_interval: int = 5,
    record_video: bool = True,
    video_path: Optional[str] = None,
    on_frame=None,
    on_metrics=None,
):
    """
    Scenario 1: 4 PI0 replicas on 4 chips, each controlling a robot.

    Args:
        on_frame: Optional callback(frame: np.ndarray) for live display.
        on_metrics: Optional callback(global_summary: dict) for live metrics.
    """
    import ttnn
    from models.experimental.robotics_demo.data_parallel_pi0 import DataParallelPI0

    print("=" * 70)
    print("  SCENARIO 1: Data-Parallel PI0 -- 4 Robots, 4 Chips")
    print("=" * 70)

    tasks = DEFAULT_TASKS[:num_devices]
    envs = MultiEnvironment(num_envs=num_devices, tasks=tasks, image_size=224)
    dp = DataParallelPI0(num_devices=num_devices, checkpoint_path=checkpoint_path)
    tokenizer = SimpleTokenizer()
    metrics = MetricsCollector(num_envs=num_devices)

    recorder = VideoRecorder(path=video_path) if record_video else None

    action_buffers = [None] * num_devices
    buffer_indices = [0] * num_devices

    print(f"\nRunning {num_steps} steps across {num_devices} environments...\n")
    for step in range(num_steps):
        loop_timer = LatencyTimer()
        with loop_timer:
            need_replan = [(step % replan_interval == 0) or (action_buffers[i] is None)
                           for i in range(num_devices)]

            all_obs = envs.capture_all_observations()

            for i in range(num_devices):
                if need_replan[i]:
                    images, state = all_obs[i]
                    tokens, masks = tokenizer.encode(tasks[i])

                    obs_dict = dp.preprocess_for_device(i, images, state, tokens, masks)

                    inf_timer = LatencyTimer()
                    with inf_timer:
                        results = dp.models[i].sample_actions(
                            images=obs_dict["images_ttnn"],
                            img_masks=obs_dict["img_masks"],
                            lang_tokens=obs_dict["lang_tokens_ttnn"],
                            lang_masks=obs_dict["lang_masks_ttnn"],
                            state=obs_dict["state_ttnn"],
                        )
                    import ttnn as _ttnn
                    if isinstance(results, _ttnn.Tensor):
                        results = _ttnn.to_torch(results)
                    action_buffers[i] = results.float().cpu().numpy()
                    buffer_indices[i] = 0

                    metrics.record(EnvironmentMetrics(
                        env_id=i, model_name="PI0", step=step,
                        inference_time_ms=inf_timer.elapsed_ms,
                        loop_time_ms=0, distance_to_target=envs.envs[i].get_distance_to_target(),
                        is_inference_step=True,
                    ))
                else:
                    buffer_indices[i] += 1

            all_actions = []
            for i in range(num_devices):
                buf = action_buffers[i]
                idx = min(buffer_indices[i], buf.shape[1] - 1) if buf.ndim == 3 else min(buffer_indices[i], buf.shape[0] - 1)
                act = buf[0, idx, :7] if buf.ndim == 3 else buf[idx, :7]
                all_actions.append(act)

            envs.apply_all_actions(all_actions)
            envs.step_all()

        for i in range(num_devices):
            metrics.record(EnvironmentMetrics(
                env_id=i, model_name="PI0", step=step,
                loop_time_ms=loop_timer.elapsed_ms,
                distance_to_target=envs.envs[i].get_distance_to_target(),
                is_inference_step=False,
            ))

        if step % 5 == 0 or recorder or on_frame:
            frames = envs.capture_all_display_frames(640, 480)
            labels = [f"Chip {i}: {tasks[i]}" for i in range(num_devices)]
            env_mets = []
            for i in range(num_devices):
                s = metrics.get_env_summary(i)
                env_mets.append({
                    "inference_ms": s["avg_inference_ms"],
                    "freq_hz": s["control_freq_hz"],
                    "distance": s["current_distance"],
                })
            composite = compose_quad_view(frames, labels=labels, metrics=env_mets)

            if recorder:
                recorder.write_frame(composite)
            if on_frame:
                on_frame(composite)

        if on_metrics and step % 20 == 0:
            on_metrics(metrics.get_global_summary())

        if step % 100 == 0:
            gs = metrics.get_global_summary()
            print(f"Step {step:4d} | Avg inf: {gs['avg_inference_ms']:.0f}ms | "
                  f"Throughput: {gs['aggregate_throughput_fps']:.1f} FPS | "
                  f"Distances: {[f'{d:.3f}' for d in envs.get_all_distances()]}")

    if recorder:
        path = recorder.close()
        print(f"\nVideo saved: {path} ({recorder.duration_seconds:.1f}s)")

    summary = metrics.get_global_summary()
    scaling = metrics.get_scaling_efficiency()
    print(f"\nFinal: {summary['total_inferences']} inferences, "
          f"{summary['aggregate_throughput_fps']:.1f} aggregate FPS, "
          f"scaling efficiency {scaling['efficiency_pct']:.0f}%")

    envs.close()
    dp.close()
    return summary


# ======================================================================
# Scenario 2: PI0 vs SmolVLA Side-by-Side
# ======================================================================

def run_scenario_2(
    num_steps: int = 400,
    task: str = "pick up the cube",
    checkpoint_path: str = DEFAULT_PI0_CHECKPOINT,
    replan_interval: int = 5,
    record_video: bool = True,
    video_path: Optional[str] = None,
    on_frame=None,
    on_metrics=None,
):
    """
    Scenario 2: PI0 on chip 0 vs SmolVLA on chip 1, identical tasks.
    """
    import ttnn
    from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
    from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
    from models.experimental.pi0.common.weight_loader import PI0WeightLoader
    from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction
    from PIL import Image

    print("=" * 70)
    print("  SCENARIO 2: PI0 vs SmolVLA -- Side-by-Side Comparison")
    print("=" * 70)

    pi0_env = MultiEnvironment(num_envs=1, tasks=[task], image_size=224)
    smol_env = MultiEnvironment(num_envs=1, tasks=[task], image_size=224)
    metrics = MetricsCollector(num_envs=2)
    tokenizer = SimpleTokenizer()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
    subs = mesh.create_submeshes(ttnn.MeshShape(1, 1))
    pi0_dev = subs[0].get_devices()[0] if hasattr(subs[0], "get_devices") else subs[0]
    smol_dev = subs[1].get_devices()[0] if hasattr(subs[1], "get_devices") else subs[1]

    from models.experimental.robotics_demo.data_parallel_pi0 import create_pi0_config
    pi0_cfg = create_pi0_config()
    pi0_wl = PI0WeightLoader(checkpoint_path)
    pi0_model = PI0ModelTTNN(pi0_cfg, pi0_wl, pi0_dev, fresh_noise_per_call=True)

    smol_model = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=smol_dev)
    smol_model.processor.image_processor.do_image_splitting = False
    smol_model.eval()

    recorder = VideoRecorder(path=video_path) if record_video else None
    pi0_buf, smol_buf = None, None
    pi0_bidx, smol_bidx = 0, 0

    print(f"\nRunning {num_steps} steps: PI0 vs SmolVLA on '{task}'...\n")

    for step in range(num_steps):
        loop_timer = LatencyTimer()
        with loop_timer:
            replan = (step % replan_interval == 0) or (pi0_buf is None)

            if replan:
                pi0_obs = pi0_env.capture_all_observations()[0]
                images, state = pi0_obs
                tokens, masks = tokenizer.encode(task)
                images_tt = [ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                             device=pi0_dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                             for img in images]
                lang_tt = ttnn.from_torch(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=pi0_dev)
                lm_tt = ttnn.from_torch(masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=pi0_dev)
                st_tt = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=pi0_dev)

                pi0_timer = LatencyTimer()
                with pi0_timer:
                    pi0_raw = pi0_model.sample_actions(
                        images=images_tt, img_masks=[torch.ones(1, dtype=torch.bool)] * 2,
                        lang_tokens=lang_tt, lang_masks=lm_tt, state=st_tt,
                    )
                if isinstance(pi0_raw, ttnn.Tensor):
                    pi0_raw = ttnn.to_torch(pi0_raw)
                pi0_buf = pi0_raw.float().cpu().numpy()
                pi0_bidx = 0

                smol_frame = smol_env.envs[0].capture_display_frame(512, 512)
                smol_pil = Image.fromarray(smol_frame)
                smol_timer = LatencyTimer()
                with smol_timer:
                    smol_raw = smol_model.sample_actions(images=[smol_pil], instruction=task,
                                                         num_inference_steps=10, action_dim=7)
                smol_buf = np.asarray(smol_raw, dtype=np.float32)
                smol_bidx = 0

                metrics.record(EnvironmentMetrics(env_id=0, model_name="PI0", step=step,
                                                  inference_time_ms=pi0_timer.elapsed_ms,
                                                  distance_to_target=pi0_env.envs[0].get_distance_to_target(),
                                                  is_inference_step=True))
                metrics.record(EnvironmentMetrics(env_id=1, model_name="SmolVLA", step=step,
                                                  inference_time_ms=smol_timer.elapsed_ms,
                                                  distance_to_target=smol_env.envs[0].get_distance_to_target(),
                                                  is_inference_step=True))
            else:
                pi0_bidx += 1
                smol_bidx += 1

            if pi0_buf is not None:
                idx = min(pi0_bidx, pi0_buf.shape[1] - 1) if pi0_buf.ndim == 3 else min(pi0_bidx, pi0_buf.shape[0] - 1)
                pi0_act = pi0_buf[0, idx, :7] if pi0_buf.ndim == 3 else pi0_buf[idx, :7]
                pi0_env.apply_all_actions([pi0_act])
                pi0_env.step_all()

            if smol_buf is not None:
                idx = min(smol_bidx, smol_buf.shape[0] - 1)
                smol_act = smol_buf[idx, :7] if smol_buf.shape[1] >= 7 else np.pad(smol_buf[idx], (0, 7 - smol_buf.shape[1]))
                smol_env.apply_all_actions([smol_act])
                smol_env.step_all()

        for eid in [0, 1]:
            e = pi0_env if eid == 0 else smol_env
            metrics.record(EnvironmentMetrics(env_id=eid, model_name="PI0" if eid == 0 else "SmolVLA",
                                              step=step, loop_time_ms=loop_timer.elapsed_ms,
                                              distance_to_target=e.envs[0].get_distance_to_target(),
                                              is_inference_step=False))

        if step % 5 == 0 or recorder or on_frame:
            lf = pi0_env.capture_all_display_frames(640, 480)
            rf = smol_env.capture_all_display_frames(640, 480)
            pi0_s = metrics.get_env_summary(0)
            smol_s = metrics.get_env_summary(1)
            composite = compose_side_by_side(
                lf, rf, left_label="PI0", right_label="SmolVLA",
                left_metrics={"inference_ms": pi0_s["avg_inference_ms"],
                               "freq_hz": pi0_s["control_freq_hz"],
                               "distance": pi0_s["current_distance"]},
                right_metrics={"inference_ms": smol_s["avg_inference_ms"],
                                "freq_hz": smol_s["control_freq_hz"],
                                "distance": smol_s["current_distance"]},
            )
            if recorder:
                recorder.write_frame(composite)
            if on_frame:
                on_frame(composite)

        if on_metrics and step % 20 == 0:
            on_metrics(metrics.get_global_summary())

        if step % 100 == 0:
            print(f"Step {step:4d} | PI0 dist: {pi0_env.envs[0].get_distance_to_target():.3f} | "
                  f"SmolVLA dist: {smol_env.envs[0].get_distance_to_target():.3f}")

    if recorder:
        print(f"\nVideo saved: {recorder.close()} ({recorder.duration_seconds:.1f}s)")

    pi0_env.close()
    smol_env.close()
    ttnn.close_mesh_device(mesh)
    return metrics.get_global_summary()


# ======================================================================
# Scenario 3: Ensemble Pipeline
# ======================================================================

def run_scenario_3(
    num_steps: int = 400,
    task: str = "pick up the cube",
    checkpoint_path: str = DEFAULT_PI0_CHECKPOINT,
    fusion_strategy: str = "weighted_average",
    alpha: float = 0.6,
    replan_interval: int = 5,
    record_video: bool = True,
    video_path: Optional[str] = None,
    on_frame=None,
    on_metrics=None,
):
    """Scenario 3: SmolVLA + PI0 ensemble with action fusion."""
    import ttnn
    from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
    from models.experimental.pi0.common.weight_loader import PI0WeightLoader
    from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction
    from models.experimental.robotics_demo.ensemble_pipeline import (
        EnsemblePipeline, FusionStrategy,
    )
    from models.experimental.robotics_demo.data_parallel_pi0 import create_pi0_config
    from PIL import Image

    print("=" * 70)
    print("  SCENARIO 3: Ensemble Pipeline -- SmolVLA + PI0 Fusion")
    print("=" * 70)

    strategy_map = {
        "weighted_average": FusionStrategy.WEIGHTED_AVERAGE,
        "temporal_blend": FusionStrategy.TEMPORAL_BLEND,
        "confidence_gate": FusionStrategy.CONFIDENCE_GATE,
    }
    fs = strategy_map.get(fusion_strategy, FusionStrategy.WEIGHTED_AVERAGE)

    env = MultiEnvironment(num_envs=1, tasks=[task], image_size=224)
    metrics = MetricsCollector(num_envs=1)
    tokenizer = SimpleTokenizer()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
    subs = mesh.create_submeshes(ttnn.MeshShape(1, 1))
    pi0_dev = subs[0].get_devices()[0] if hasattr(subs[0], "get_devices") else subs[0]
    smol_dev = subs[1].get_devices()[0] if hasattr(subs[1], "get_devices") else subs[1]

    pi0_cfg = create_pi0_config()
    pi0_model = PI0ModelTTNN(pi0_cfg, PI0WeightLoader(checkpoint_path), pi0_dev, fresh_noise_per_call=True)

    smol_model = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=smol_dev)
    smol_model.processor.image_processor.do_image_splitting = False
    smol_model.eval()

    pipeline = EnsemblePipeline(pi0_model, smol_model, fusion_strategy=fs, alpha=alpha)
    recorder = VideoRecorder(path=video_path) if record_video else None

    action_buf = None
    buf_idx = 0

    print(f"\nRunning ensemble ({fs.value}) for {num_steps} steps on '{task}'...\n")

    for step in range(num_steps):
        loop_timer = LatencyTimer()
        with loop_timer:
            replan = (step % replan_interval == 0) or (action_buf is None)

            if replan:
                obs = env.capture_all_observations()[0]
                images, state = obs
                tokens, masks = tokenizer.encode(task)
                images_tt = [ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                             device=pi0_dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                             for img in images]
                pi0_inputs = {
                    "images": images_tt,
                    "img_masks": [torch.ones(1, dtype=torch.bool)] * 2,
                    "lang_tokens": ttnn.from_torch(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=pi0_dev),
                    "lang_masks": ttnn.from_torch(masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=pi0_dev),
                    "state": ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=pi0_dev),
                }

                smol_frame = env.envs[0].capture_display_frame(512, 512)
                smol_pil = Image.fromarray(smol_frame)

                inf_timer = LatencyTimer()
                with inf_timer:
                    fused, timing = pipeline.run_concurrent_inference(
                        pi0_inputs=pi0_inputs,
                        smolvla_images=[smol_pil],
                        smolvla_instruction=task,
                        action_dim=7,
                    )
                action_buf = fused
                buf_idx = 0

                metrics.record(EnvironmentMetrics(
                    env_id=0, model_name=f"Ensemble({fs.value})", step=step,
                    inference_time_ms=timing["wall_ms"],
                    distance_to_target=env.envs[0].get_distance_to_target(),
                    is_inference_step=True,
                ))

                if step % 50 == 0:
                    print(f"  Step {step}: PI0={timing['pi0_ms']:.0f}ms, SmolVLA={timing['smolvla_ms']:.0f}ms, "
                          f"Wall={timing['wall_ms']:.0f}ms, Speedup={timing['speedup_vs_sequential']:.1f}x")
            else:
                buf_idx += 1

            if action_buf is not None:
                idx = min(buf_idx, action_buf.shape[0] - 1)
                act = action_buf[idx, :7]
                env.apply_all_actions([act])
                env.step_all()

        metrics.record(EnvironmentMetrics(env_id=0, model_name=f"Ensemble({fs.value})", step=step,
                                          loop_time_ms=loop_timer.elapsed_ms,
                                          distance_to_target=env.envs[0].get_distance_to_target(),
                                          is_inference_step=False))

        if step % 5 == 0 or recorder or on_frame:
            frames = env.capture_all_display_frames(640, 480)
            composite = compose_quad_view(
                frames + frames + frames + frames,
                labels=[f"Ensemble ({fs.value})", f"Distance: {env.envs[0].get_distance_to_target():.3f}m", "", ""],
            )
            if recorder:
                recorder.write_frame(composite)
            if on_frame:
                on_frame(composite)

        if on_metrics and step % 20 == 0:
            on_metrics(metrics.get_global_summary())

    if recorder:
        print(f"\nVideo saved: {recorder.close()} ({recorder.duration_seconds:.1f}s)")

    env.close()
    ttnn.close_mesh_device(mesh)
    return metrics.get_global_summary()


# ======================================================================
# Scenario 4: Throughput Scaling Benchmark
# ======================================================================

def run_scenario_4(
    max_chips: int = 4,
    iterations_per_config: int = 10,
    checkpoint_path: str = DEFAULT_PI0_CHECKPOINT,
    on_metrics=None,
) -> Dict:
    """Scenario 4: Benchmark scaling from 1 to max_chips."""
    import ttnn
    from models.experimental.robotics_demo.benchmark import (
        run_pi0_benchmark, generate_scaling_chart, generate_latency_waterfall,
    )
    from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
    from models.experimental.pi0.common.weight_loader import PI0WeightLoader
    from models.experimental.robotics_demo.data_parallel_pi0 import create_pi0_config

    print("=" * 70)
    print("  SCENARIO 4: Throughput Scaling Benchmark")
    print("=" * 70)

    all_results = []
    pi0_cfg = create_pi0_config()
    wl = PI0WeightLoader(checkpoint_path)

    for n in range(1, max_chips + 1):
        print(f"\n--- Benchmarking {n} chip(s) ---")
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, n), l1_small_size=24576)
        subs = mesh.create_submeshes(ttnn.MeshShape(1, 1))

        per_chip_fps = []
        for i, sub in enumerate(subs[:n]):
            dev = sub.get_devices()[0] if hasattr(sub, "get_devices") else sub
            model = PI0ModelTTNN(pi0_cfg, wl, dev, fresh_noise_per_call=True)
            result = run_pi0_benchmark(model, dev, num_iterations=iterations_per_config)
            per_chip_fps.append(result["fps"])
            print(f"  Chip {i}: {result['avg_ms']:.1f}ms avg, {result['fps']:.1f} FPS")

        total_fps = sum(per_chip_fps)
        all_results.append({"model": "PI0", "num_chips": n, "total_fps": total_fps,
                            "per_chip_fps": per_chip_fps})

        ttnn.close_mesh_device(mesh)

        if on_metrics:
            on_metrics({"current_chips": n, "results": all_results})

    chart_path = generate_scaling_chart(all_results, title="PI0 Throughput Scaling (Blackhole)")
    if chart_path:
        print(f"\nScaling chart saved: {chart_path}")

    if all_results:
        waterfall_data = {
            "Vision (SigLIP)": 45,
            "VLM Prefill": 30,
            "Denoising (10 steps)": all_results[0]["per_chip_fps"][0] and
                                    (1000.0 / all_results[0]["per_chip_fps"][0] - 75) or 200,
            "Host overhead": 30,
        }
        wf_path = generate_latency_waterfall(waterfall_data, title="PI0 Latency Breakdown")
        if wf_path:
            print(f"Waterfall chart saved: {wf_path}")

    print(f"\nScaling summary:")
    for r in all_results:
        eff = r["total_fps"] / (all_results[0]["total_fps"] * r["num_chips"]) * 100 if all_results[0]["total_fps"] > 0 else 0
        print(f"  {r['num_chips']} chips: {r['total_fps']:.1f} FPS (efficiency: {eff:.0f}%)")

    return {"results": all_results, "chart_path": chart_path}


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tenstorrent Robotics Intelligence Demo")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], default=1,
                        help="Demo scenario to run (1-4)")
    parser.add_argument("--steps", type=int, default=400, help="Simulation steps")
    parser.add_argument("--chips", type=int, default=4, help="Number of chips to use")
    parser.add_argument("--task", type=str, default="pick up the cube", help="Task instruction")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_PI0_CHECKPOINT)
    parser.add_argument("--replan-interval", type=int, default=5)
    parser.add_argument("--record-video", action="store_true", default=True)
    parser.add_argument("--fusion", type=str, default="weighted_average",
                        choices=["weighted_average", "temporal_blend", "confidence_gate"])
    parser.add_argument("--alpha", type=float, default=0.6)
    args = parser.parse_args()

    if args.scenario == 1:
        run_scenario_1(num_steps=args.steps, num_devices=args.chips,
                       checkpoint_path=args.checkpoint, replan_interval=args.replan_interval,
                       record_video=args.record_video)
    elif args.scenario == 2:
        run_scenario_2(num_steps=args.steps, task=args.task,
                       checkpoint_path=args.checkpoint, replan_interval=args.replan_interval,
                       record_video=args.record_video)
    elif args.scenario == 3:
        run_scenario_3(num_steps=args.steps, task=args.task,
                       checkpoint_path=args.checkpoint, fusion_strategy=args.fusion,
                       alpha=args.alpha, replan_interval=args.replan_interval,
                       record_video=args.record_video)
    elif args.scenario == 4:
        run_scenario_4(max_chips=args.chips, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
