#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single-Chip Robotics Demo: PI0 or SmolVLA.

A simple, self-contained demo that runs one VLA model on one Tenstorrent
chip, controlling a Franka Panda robot in PyBullet simulation.
The user picks which model to run and provides a task instruction.

Usage:
    # PI0 (default)
    python run_demo.py --model pi0 --task "pick up the cube" --steps 300

    # SmolVLA
    python run_demo.py --model smolvla --task "pick up the cube" --steps 300

    # Headless with video recording
    xvfb-run -a python run_demo.py --model pi0 --record-video --steps 400

    # Demo mode (scripted motion, no TT hardware needed)
    python run_demo.py --demo-mode --steps 200
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.environ.get("TT_METAL_HOME",
    str(Path(__file__).parent.parent.parent.parent)))

from models.experimental.robotics_demo_single_chip.sim_env import FrankaCubeEnv
from models.experimental.robotics_demo_single_chip.tokenizer import SimpleTokenizer


def load_pi0_model(device, checkpoint_path):
    """Load PI0 model onto a single TT device."""
    import ttnn
    from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
    from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
    from models.experimental.pi0.common.weight_loader import PI0WeightLoader

    config = PI0ModelConfig(
        action_dim=32, action_horizon=50, state_dim=32,
        paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
        pi05=False,
    )
    config.siglip_config = SigLIPConfig(
        hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
        num_attention_heads=16, image_size=224, patch_size=14,
    )
    weight_loader = PI0WeightLoader(checkpoint_path)
    model = PI0ModelTTNN(config, weight_loader, device, fresh_noise_per_call=True)
    return model, config


def load_smolvla_model(device):
    """Load SmolVLA model onto a single TT device."""
    from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction
    model = SmolVLAForActionPrediction.from_pretrained(
        "lerobot/smolvla_base", ttnn_device=device)
    model.processor.image_processor.do_image_splitting = False
    model.eval()
    return model


class SingleChipDemo:
    """
    Run PI0 or SmolVLA in closed-loop control on one TT chip.

    The control loop:
      1. Capture camera images + robot state from PyBullet
      2. Preprocess into model-specific format
      3. Run inference on TT device
      4. Apply first 7 action dims to robot joints
      5. Step physics, record frame, repeat
    """

    def __init__(self, model_name, task, device_id=0, checkpoint_path=None,
                 replan_interval=5, use_delta=True, delta_scale=1.0,
                 max_velocity=0.5, image_size=224, seed=42, demo_mode=False):

        self.model_name = model_name
        self.task = task
        self.replan_interval = replan_interval
        self.use_delta = use_delta
        self.delta_scale = delta_scale
        self.max_velocity = max_velocity
        self.demo_mode = demo_mode
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        print("=" * 60)
        print(f"  Single-Chip Robotics Demo: {model_name.upper()}")
        print("=" * 60)

        # Environment
        print("\n[1/3] Initializing PyBullet environment...")
        self.env = FrankaCubeEnv(image_size=image_size)
        self.env.reset()
        print(f"  Robot: Franka Panda 7-DOF")
        print(f"  Cube at: {self.env.cube_position}")

        # Model
        self.device = None
        self.model = None
        self.tokenizer = SimpleTokenizer()

        if demo_mode:
            print("\n[2/3] Demo mode -- using scripted motion (no TT hardware)")
        else:
            import ttnn
            print(f"\n[2/3] Loading {model_name.upper()} on TT device {device_id}...")
            self.device = ttnn.open_device(device_id=device_id, l1_small_size=24576)

            if model_name == "pi0":
                if checkpoint_path is None:
                    tt_home = os.environ.get("TT_METAL_HOME", "")
                    checkpoint_path = os.path.join(tt_home,
                        "models/experimental/pi0/weights/pi0_base")
                self.model, self.config = load_pi0_model(self.device, checkpoint_path)
                print(f"  PI0 loaded (action_dim=32, horizon=50)")
            elif model_name == "smolvla":
                self.model = load_smolvla_model(self.device)
                print(f"  SmolVLA loaded (lerobot/smolvla_base)")
            else:
                raise ValueError(f"Unknown model: {model_name}. Use 'pi0' or 'smolvla'.")

        # Metrics
        self.inference_times = deque(maxlen=200)
        self.loop_times = deque(maxlen=200)

        print(f"\n[3/3] Ready.")
        print(f"  Model:    {model_name.upper()}")
        print(f"  Task:     {task}")
        print(f"  Replan:   every {replan_interval} steps")
        print(f"  Action:   {'delta' if use_delta else 'absolute'}")
        print(f"  Seed:     {seed}")

    def _infer_pi0(self, images, state):
        """Run PI0 inference and return raw action tensor."""
        import ttnn
        images_tt = [
            ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for img in images
        ]
        tokens, masks = self.tokenizer.encode(self.task)
        lang_tt = ttnn.from_torch(tokens, dtype=ttnn.uint32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        mask_tt = ttnn.from_torch(masks.float(), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=self.device)
        state_tt = ttnn.from_torch(state, dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=self.device)
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in images]

        with torch.no_grad():
            result = self.model.sample_actions(
                images=images_tt, img_masks=img_masks,
                lang_tokens=lang_tt, lang_masks=mask_tt, state=state_tt)

        if isinstance(result, ttnn.Tensor):
            result = ttnn.to_torch(result)
        return result.float().cpu().numpy()

    def _infer_smolvla(self):
        """Run SmolVLA inference and return action array."""
        pil_img = self.env.capture_pil_image()
        with torch.no_grad():
            actions = self.model.sample_actions(
                images=[pil_img], instruction=self.task,
                num_inference_steps=10, action_dim=7)
        return np.asarray(actions, dtype=np.float32)

    def _scripted_motion(self, step):
        """Generate demo motion targeting the cube via IK."""
        import pybullet as pb
        cube = self.env.get_cube_position()
        cx, cy, cz = cube
        t = step * 0.02
        phase = (step % 200) / 200.0

        if phase < 0.3:
            target = [cx, cy, cz + 0.18 - phase * 0.4]
        elif phase < 0.5:
            target = [cx, cy, cz + 0.04]
        elif phase < 0.8:
            target = [cx, cy, cz + 0.04 + (phase - 0.5) * 1.5]
        else:
            target = [0.3, 0.0, 0.55]

        ik = pb.calculateInverseKinematics(
            self.env.robot_id, 11, target,
            physicsClientId=self.env.physics_client)
        return np.array(ik[:7])

    def run(self, num_steps=300, record_video=False, video_path=None):
        """Main control loop."""
        recorder = None
        if record_video:
            try:
                import imageio
                if video_path is None:
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = f"single_chip_{self.model_name}_{ts}.mp4"
                recorder = imageio.get_writer(video_path, fps=20)
                print(f"\n  Recording video: {video_path}")
            except Exception as e:
                print(f"  Video recording unavailable: {e}")

        print(f"\n{'='*60}")
        print(f"  Running {num_steps} steps...")
        print(f"{'='*60}\n")

        action_buffer = None
        buf_idx = 0

        # Warm up (if using real model)
        if not self.demo_mode and self.model is not None:
            print("  Warming up model (first inference includes compilation)...")
            for i in range(2):
                t0 = time.time()
                if self.model_name == "pi0":
                    images, state = self.env.capture_observations()
                    self._infer_pi0(images, state)
                else:
                    self._infer_smolvla()
                if self.device is not None:
                    import ttnn
                    ttnn.synchronize_device(self.device)
                print(f"    Warmup {i+1}/2: {(time.time()-t0)*1000:.0f}ms")
            print("  Warmup complete.\n")

        for step in range(num_steps):
            loop_start = time.time()
            need_replan = (step % self.replan_interval == 0) or (action_buffer is None)

            if need_replan:
                inf_start = time.time()

                if self.demo_mode:
                    joints = self._scripted_motion(step)
                    action_buffer = joints.reshape(1, -1)
                    buf_idx = 0
                elif self.model_name == "pi0":
                    images, state = self.env.capture_observations()
                    raw = self._infer_pi0(images, state)
                    action_buffer = raw
                    buf_idx = 0
                elif self.model_name == "smolvla":
                    raw = self._infer_smolvla()
                    action_buffer = raw
                    buf_idx = 0

                inf_ms = (time.time() - inf_start) * 1000
                self.inference_times.append(inf_ms)
            else:
                buf_idx += 1

            # Extract action for this step
            if action_buffer is not None:
                if action_buffer.ndim == 3:
                    idx = min(buf_idx, action_buffer.shape[1] - 1)
                    act = action_buffer[0, idx, :7]
                elif action_buffer.ndim == 2:
                    idx = min(buf_idx, action_buffer.shape[0] - 1)
                    act = action_buffer[idx, :7]
                    if len(act) < 7:
                        act = np.pad(act, (0, 7 - len(act)))
                else:
                    act = action_buffer[:7]

                if self.demo_mode:
                    self.env.apply_actions(act, use_delta=False, max_velocity=8.0)
                else:
                    self.env.apply_actions(act, use_delta=self.use_delta,
                                           delta_scale=self.delta_scale,
                                           max_velocity=self.max_velocity)

            self.env.step()

            loop_ms = (time.time() - loop_start) * 1000
            self.loop_times.append(loop_ms)

            if recorder:
                frame = self.env.capture_display_frame(960, 540)
                try:
                    import cv2
                    dist = self.env.get_distance_to_target()
                    ee = self.env.get_ee_position()
                    cv2.putText(frame, f"{self.model_name.upper()} | {self.task}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 206, 201), 2)
                    avg_inf = np.mean(self.inference_times) if self.inference_times else 0
                    avg_loop = np.mean(self.loop_times) if self.loop_times else 0
                    hz = 1000 / avg_loop if avg_loop > 0 else 0
                    cv2.putText(frame, f"Inf: {avg_inf:.0f}ms | Loop: {avg_loop:.0f}ms | {hz:.1f} Hz",
                                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    cv2.putText(frame, f"EE: [{ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}] | Dist: {dist:.3f}m",
                                (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    cv2.putText(frame, f"Step {step}/{num_steps}",
                                (860, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                except ImportError:
                    pass
                recorder.append_data(frame)

            if step % 50 == 0:
                dist = self.env.get_distance_to_target()
                ee = self.env.get_ee_position()
                avg_inf = np.mean(self.inference_times) if self.inference_times else 0
                avg_loop = np.mean(self.loop_times) if self.loop_times else 0
                hz = 1000 / avg_loop if avg_loop > 0 else 0
                replan_tag = "INFER" if need_replan else "buf"
                print(f"  Step {step:4d} | EE=[{ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f}] "
                      f"| Dist: {dist:.3f}m | Inf: {avg_inf:.0f}ms | {hz:.1f} Hz | {replan_tag}")

        # Summary
        print(f"\n{'='*60}")
        print(f"  Done!")
        print(f"{'='*60}")
        print(f"  Model:           {self.model_name.upper()}")
        print(f"  Steps:           {num_steps}")
        n_inf = len(self.inference_times)
        n_buf = num_steps - n_inf
        print(f"  Inferences:      {n_inf} ({n_inf/num_steps*100:.0f}%)")
        print(f"  Buffered steps:  {n_buf} ({n_buf/num_steps*100:.0f}%)")
        if self.inference_times:
            arr = np.array(self.inference_times)
            print(f"  Avg inference:   {arr.mean():.1f} +/- {arr.std():.1f} ms")
            print(f"  Min / Max:       {arr.min():.1f} / {arr.max():.1f} ms")
        if self.loop_times:
            arr = np.array(self.loop_times)
            print(f"  Avg loop:        {arr.mean():.1f} ms")
            print(f"  Control freq:    {1000/arr.mean():.1f} Hz")
        print(f"  Final distance:  {self.env.get_distance_to_target():.3f}m")

        if recorder:
            recorder.close()
            print(f"\n  Video saved: {video_path}")

    def close(self):
        self.env.close()
        if self.device is not None:
            import ttnn
            ttnn.close_device(self.device)
            print("  TT device closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Single-Chip Robotics Demo: PI0 or SmolVLA")
    parser.add_argument("--model", type=str, default="pi0",
                        choices=["pi0", "smolvla"],
                        help="Which VLA model to run (default: pi0)")
    parser.add_argument("--task", type=str, default="pick up the cube",
                        help="Task instruction for the robot")
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of simulation steps")
    parser.add_argument("--device", type=int, default=0,
                        help="TT device ID (default: 0)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PI0 checkpoint path (default: auto-detect)")
    parser.add_argument("--replan-interval", type=int, default=5,
                        help="Steps between re-planning (default: 5)")
    parser.add_argument("--use-delta", action="store_true", default=True,
                        help="Delta action mode (default)")
    parser.add_argument("--use-absolute", action="store_true",
                        help="Absolute action mode")
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--max-velocity", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-video", action="store_true",
                        help="Record simulation to MP4")
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--demo-mode", action="store_true",
                        help="Scripted motion (no TT hardware needed)")
    args = parser.parse_args()

    if args.use_absolute:
        args.use_delta = False

    demo = SingleChipDemo(
        model_name=args.model,
        task=args.task,
        device_id=args.device,
        checkpoint_path=args.checkpoint,
        replan_interval=args.replan_interval,
        use_delta=args.use_delta,
        delta_scale=args.delta_scale,
        max_velocity=args.max_velocity,
        image_size=args.image_size,
        seed=args.seed,
        demo_mode=args.demo_mode,
    )

    try:
        demo.run(num_steps=args.steps,
                 record_video=args.record_video,
                 video_path=args.video_path)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
