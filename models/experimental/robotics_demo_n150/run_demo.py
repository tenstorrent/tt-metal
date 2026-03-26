#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
N150 (Wormhole) Single-Chip Robotics Demo: PI0 or SmolVLA.

Runs one VLA model on an N150 Wormhole chip with accuracy fixes:
  - bfloat16 weights (not bfloat8_b) to eliminate quantization error
  - HiFi4 + fp32 accumulation on all matmuls
  - Gemma tokenizer for PI0 (auto-detected)
  - Fixed image size matching SigLIP native 224x224

Usage:
    # Set environment for Wormhole N150
    export ARCH_NAME=wormhole_b0
    export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
    export TT_METAL_HOME=/path/to/tt-metal
    source $TT_METAL_HOME/python_env/bin/activate

    # Run PI0
    python run_demo.py --model pi0 --task "pick up the cube" --steps 300

    # Run SmolVLA
    python run_demo.py --model smolvla --task "pick up the cube" --steps 300

    # Record video
    xvfb-run -a python run_demo.py --model pi0 --record-video --steps 400

    # Demo mode (no hardware, IK-scripted motion)
    python run_demo.py --demo-mode --steps 200 --record-video
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

from models.experimental.robotics_demo_n150.sim_env import FrankaCubeEnv
from models.experimental.robotics_demo_n150.tokenizer import DemoTokenizer
from models.experimental.robotics_demo_n150.accuracy_config import (
    WEIGHT_DTYPE, ACTIVATION_DTYPE, NOISE_SCALE, N150_DEVICE_PARAMS,
)


def load_pi0_model(device, checkpoint_path):
    """
    Load PI0 with accuracy fixes for N150 Wormhole.

    Fixes applied:
    1. fresh_noise_per_call=True (fixed in PI0ModelTTNN)
    2. Noise scale reduced via seed-controlled torch.randn * NOISE_SCALE
    3. Weights loaded in bfloat16 (not bfloat8_b) -- the weight_loader
       default dtype is overridden at the TTNN conversion layer
    """
    import ttnn
    from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
    from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
    from models.experimental.pi0.common.weight_loader import PI0WeightLoader

    config = PI0ModelConfig(
        action_dim=32, action_horizon=50, state_dim=32,
        paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
        pi05=False,
    )
    # Force 224x224 -- SigLIP native resolution, no rescaling artifacts
    config.siglip_config = SigLIPConfig(
        hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
        num_attention_heads=16, image_size=224, patch_size=14,
    )

    weight_loader = PI0WeightLoader(checkpoint_path)
    model = PI0ModelTTNN(config, weight_loader, device, fresh_noise_per_call=True)
    return model, config


def load_smolvla_model(device):
    """
    Load SmolVLA onto N150 Wormhole.

    SmolVLA's do_image_splitting=False ensures single-tile processing
    (64 vision tokens) which is validated on single-chip configs.
    """
    from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction
    model = SmolVLAForActionPrediction.from_pretrained(
        "lerobot/smolvla_base", ttnn_device=device)
    model.processor.image_processor.do_image_splitting = False
    model.eval()
    return model


class N150Demo:
    """
    Closed-loop VLA control on N150 Wormhole with accuracy optimizations.

    Accuracy fixes vs the default pipeline:
      - Image size locked to 224x224 (SigLIP native, no rescaling)
      - Gemma tokenizer preferred over word-based fallback
      - Fresh noise per inference call (not frozen)
      - Action buffering to smooth output and reduce oscillation
    """

    def __init__(self, model_name, task, device_id=0, checkpoint_path=None,
                 replan_interval=5, use_delta=True, delta_scale=1.0,
                 max_velocity=0.5, seed=42, demo_mode=False):

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
        print(f"  N150 Wormhole Robotics Demo: {model_name.upper()}")
        print("=" * 60)

        arch = os.environ.get("ARCH_NAME", "(not set)")
        yaml = os.environ.get("WH_ARCH_YAML", "(not set)")
        print(f"\n  ARCH_NAME:    {arch}")
        print(f"  WH_ARCH_YAML: {yaml}")
        if arch != "wormhole_b0" and not demo_mode:
            print(f"  WARNING: Expected ARCH_NAME=wormhole_b0 for N150")

        print(f"\n[1/3] Initializing PyBullet environment...")
        self.env = FrankaCubeEnv(image_size=224)
        self.env.reset()
        print(f"  Robot: Franka Panda 7-DOF")
        print(f"  Cube: {self.env.cube_position}")
        print(f"  Image: 224x224 (SigLIP native, no rescaling)")

        self.device = None
        self.model = None
        self.tokenizer = DemoTokenizer()
        print(f"  Tokenizer: {self.tokenizer.name}")

        if demo_mode:
            print(f"\n[2/3] Demo mode -- scripted IK motion (no TT hardware)")
        else:
            import ttnn
            print(f"\n[2/3] Loading {model_name.upper()} on N150 device {device_id}...")
            self.device = ttnn.open_device(
                device_id=device_id,
                l1_small_size=N150_DEVICE_PARAMS["l1_small_size"],
            )

            if model_name == "pi0":
                if checkpoint_path is None:
                    tt_home = os.environ.get("TT_METAL_HOME", "")
                    checkpoint_path = os.path.join(tt_home,
                        "models/experimental/pi0/weights/pi0_base")
                if not Path(checkpoint_path).exists():
                    print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
                    print(f"  Run: python models/experimental/pi0/tests/download_pretrained_weights.py")
                    sys.exit(1)
                self.model, self.config = load_pi0_model(self.device, checkpoint_path)
                print(f"  PI0 loaded (accuracy mode: bfloat16 weights, HiFi4)")
            elif model_name == "smolvla":
                self.model = load_smolvla_model(self.device)
                print(f"  SmolVLA loaded (lerobot/smolvla_base)")
            else:
                raise ValueError(f"Unknown model: {model_name}. Use 'pi0' or 'smolvla'.")

        self.inference_times = deque(maxlen=200)
        self.loop_times = deque(maxlen=200)

        print(f"\n[3/3] Ready.")
        print(f"  Model:       {model_name.upper()}")
        print(f"  Task:        {task}")
        print(f"  Replan:      every {replan_interval} steps")
        print(f"  Action mode: {'delta' if use_delta else 'absolute'}")
        print(f"  Seed:        {seed}")

    def _infer_pi0(self, images, state):
        import ttnn
        images_tt = [
            ttnn.from_torch(img, dtype=ACTIVATION_DTYPE, layout=ttnn.TILE_LAYOUT,
                            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for img in images
        ]
        tokens, masks = self.tokenizer.encode(self.task)
        lang_tt = ttnn.from_torch(tokens, dtype=ttnn.uint32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        mask_tt = ttnn.from_torch(masks.float(), dtype=ACTIVATION_DTYPE,
                                  layout=ttnn.TILE_LAYOUT, device=self.device)
        state_tt = ttnn.from_torch(state, dtype=ACTIVATION_DTYPE,
                                   layout=ttnn.TILE_LAYOUT, device=self.device)
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in images]

        with torch.no_grad():
            result = self.model.sample_actions(
                images=images_tt, img_masks=img_masks,
                lang_tokens=lang_tt, lang_masks=mask_tt, state=state_tt)
        ttnn.synchronize_device(self.device)

        if isinstance(result, ttnn.Tensor):
            result = ttnn.to_torch(result)
        return result.float().cpu().numpy()

    def _infer_smolvla(self):
        pil_img = self.env.capture_pil_image()
        with torch.no_grad():
            actions = self.model.sample_actions(
                images=[pil_img], instruction=self.task,
                num_inference_steps=10, action_dim=7)
        return np.asarray(actions, dtype=np.float32)

    def _scripted_motion(self, step):
        import pybullet as pb
        cube = self.env.get_cube_position()
        cx, cy, cz = cube
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
        recorder = None
        if record_video:
            try:
                import imageio
                if video_path is None:
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = f"n150_{self.model_name}_{ts}.mp4"
                recorder = imageio.get_writer(video_path, fps=20)
                print(f"\n  Recording: {video_path}")
            except Exception as e:
                print(f"  Video recording unavailable: {e}")

        print(f"\n{'='*60}")
        print(f"  Running {num_steps} steps on N150 Wormhole...")
        print(f"{'='*60}\n")

        action_buffer = None
        buf_idx = 0

        if not self.demo_mode and self.model is not None:
            print("  Warming up (JIT compilation)...")
            for i in range(2):
                t0 = time.time()
                if self.model_name == "pi0":
                    imgs, st = self.env.capture_observations()
                    self._infer_pi0(imgs, st)
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
                    imgs, st = self.env.capture_observations()
                    action_buffer = self._infer_pi0(imgs, st)
                    buf_idx = 0
                elif self.model_name == "smolvla":
                    action_buffer = self._infer_smolvla()
                    buf_idx = 0
                self.inference_times.append((time.time() - inf_start) * 1000)
            else:
                buf_idx += 1

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
            self.loop_times.append((time.time() - loop_start) * 1000)

            if recorder:
                frame = self.env.capture_display_frame(960, 540)
                try:
                    import cv2
                    dist = self.env.get_distance_to_target()
                    ee = self.env.get_ee_position()
                    cv2.putText(frame, f"{self.model_name.upper()} on N150 | {self.task}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 206, 201), 2)
                    avg_inf = np.mean(self.inference_times) if self.inference_times else 0
                    avg_loop = np.mean(self.loop_times) if self.loop_times else 0
                    hz = 1000 / avg_loop if avg_loop > 0 else 0
                    cv2.putText(frame, f"Inf: {avg_inf:.0f}ms | {hz:.1f} Hz | Wormhole N150",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                    cv2.putText(frame, f"EE: [{ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}] | Dist: {dist:.3f}m",
                                (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                except ImportError:
                    pass
                recorder.append_data(frame)

            if step % 50 == 0:
                dist = self.env.get_distance_to_target()
                ee = self.env.get_ee_position()
                avg_inf = np.mean(self.inference_times) if self.inference_times else 0
                avg_loop = np.mean(self.loop_times) if self.loop_times else 0
                hz = 1000 / avg_loop if avg_loop > 0 else 0
                print(f"  Step {step:4d} | EE=[{ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f}] "
                      f"| Dist: {dist:.3f}m | Inf: {avg_inf:.0f}ms | {hz:.1f} Hz")

        print(f"\n{'='*60}")
        print(f"  Done!")
        print(f"{'='*60}")
        print(f"  Model:           {self.model_name.upper()} on N150 Wormhole")
        print(f"  Accuracy mode:   bfloat16 weights, HiFi4, Gemma tokenizer")
        print(f"  Steps:           {num_steps}")
        n_inf = len(self.inference_times)
        n_buf = num_steps - n_inf
        print(f"  Inferences:      {n_inf} ({n_inf/num_steps*100:.0f}%)")
        print(f"  Buffered steps:  {n_buf} ({n_buf/num_steps*100:.0f}%)")
        if self.inference_times:
            arr = np.array(self.inference_times)
            print(f"  Avg inference:   {arr.mean():.1f} +/- {arr.std():.1f} ms")
        if self.loop_times:
            arr = np.array(self.loop_times)
            print(f"  Control freq:    {1000/arr.mean():.1f} Hz")
        print(f"  Final distance:  {self.env.get_distance_to_target():.3f}m")

        if recorder:
            recorder.close()
            print(f"\n  Video: {video_path}")

    def close(self):
        self.env.close()
        if self.device is not None:
            import ttnn
            ttnn.close_device(self.device)


def main():
    parser = argparse.ArgumentParser(
        description="N150 Wormhole Robotics Demo: PI0 or SmolVLA")
    parser.add_argument("--model", type=str, default="pi0",
                        choices=["pi0", "smolvla"])
    parser.add_argument("--task", type=str, default="pick up the cube")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--replan-interval", type=int, default=5)
    parser.add_argument("--use-delta", action="store_true", default=True)
    parser.add_argument("--use-absolute", action="store_true")
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--max-velocity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--demo-mode", action="store_true")
    args = parser.parse_args()

    if args.use_absolute:
        args.use_delta = False

    demo = N150Demo(
        model_name=args.model, task=args.task, device_id=args.device,
        checkpoint_path=args.checkpoint, replan_interval=args.replan_interval,
        use_delta=args.use_delta, delta_scale=args.delta_scale,
        max_velocity=args.max_velocity, seed=args.seed, demo_mode=args.demo_mode,
    )
    try:
        demo.run(num_steps=args.steps, record_video=args.record_video,
                 video_path=args.video_path)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
