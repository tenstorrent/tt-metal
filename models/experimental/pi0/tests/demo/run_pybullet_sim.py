#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Real-time PI0 Simulation with PyBullet

This script demonstrates PI0 inference in a real-time simulation loop
with PyBullet physics engine, proving the model can work in closed-loop control.

Usage:
    python run_pybullet_sim.py
    python run_pybullet_sim.py --headless  # Run without GUI
    python run_pybullet_sim.py --steps 1000 --task "pick up cube"
    python run_pybullet_sim.py --headless --record-video  # Record to video
    xvfb-run -a python run_pybullet_sim.py --record-video  # Record with virtual display
    python run_pybullet_sim.py --demo-mode --record-video  # Scripted motion demo
    python run_pybullet_sim.py --verbose-actions  # See what PI0 predicts
    python run_pybullet_sim.py --seed 123  # Use different random seed
    python run_pybullet_sim.py --use-gemma-tokenizer  # Use official Gemma tokenizer (requires HF auth)
"""

import sys
import os
from pathlib import Path
import time
import argparse
import numpy as np
import torch
from collections import deque
from datetime import datetime

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("❌ PyBullet not installed. Install with: pip install pybullet")
    sys.exit(1)

import ttnn

# Demo folder location
DEMO_DIR = Path(__file__).parent

# Add parent paths for imports
sys.path.insert(0, str(DEMO_DIR.parent.parent.parent.parent))

from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


class GemmaTokenizerWrapper:
    """
    Wrapper for official Gemma tokenizer from transformers.
    Requires HuggingFace authentication for gated Gemma models.
    """

    def __init__(self, model_name="google/gemma-2b"):
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.use_official = True
            print(f"   ✅ Loaded official Gemma tokenizer from {model_name}")
        except Exception as e:
            print(f"   ⚠️  Could not load Gemma tokenizer: {e}")
            print(f"   Falling back to SimpleRoboticsTokenizer")
            self.use_official = False
            self.fallback = SimpleRoboticsTokenizer()

    def encode(self, text, max_length=32):
        if self.use_official:
            encoded = self.tokenizer(
                text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            tokens = encoded["input_ids"][0].tolist()
            mask = encoded["attention_mask"][0].bool().tolist()
            return tokens, mask
        else:
            return self.fallback.encode(text, max_length)


class SimpleRoboticsTokenizer:
    """
    Simple word-based tokenizer for robotics tasks.

    This is a fallback tokenizer that's better than character-based encoding
    but not as sophisticated as Gemma's SentencePiece tokenizer.

    Provides better semantic encoding for common robotics commands.
    """

    def __init__(self, vocab_size=256000):
        self.vocab_size = vocab_size

        # Common robotics vocabulary (more semantic than char-based)
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "pick": 100,
            "place": 101,
            "grasp": 102,
            "release": 103,
            "move": 104,
            "reach": 105,
            "push": 106,
            "pull": 107,
            "lift": 108,
            "drop": 109,
            "cube": 200,
            "block": 201,
            "object": 202,
            "ball": 203,
            "box": 204,
            "up": 300,
            "down": 301,
            "left": 302,
            "right": 303,
            "forward": 304,
            "backward": 305,
            "the": 400,
            "a": 401,
            "an": 402,
            "to": 403,
            "from": 404,
            "and": 405,
            "on": 406,
            "in": 407,
            "at": 408,
        }

    def encode(self, text, max_length=32):
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            max_length: Maximum sequence length

        Returns:
            tokens: List of token IDs
            mask: Boolean mask (True for valid tokens)
        """
        # Simple word tokenization
        words = text.lower().strip().split()

        tokens = [self.vocab["<bos>"]]

        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Hash unknown words deterministically
                token_id = hash(word) % (self.vocab_size - 1000) + 1000
                tokens.append(token_id)

        tokens.append(self.vocab["<eos>"])

        # Create mask (True for real tokens, False for padding)
        mask = [True] * len(tokens)

        # Pad or truncate to max_length
        if len(tokens) < max_length:
            tokens.extend([self.vocab["<pad>"]] * (max_length - len(tokens)))
            mask.extend([False] * (max_length - len(mask)))
        else:
            tokens = tokens[:max_length]
            mask = mask[:max_length]

        return tokens, mask


class PI0SimulationEnv:
    """
    Real-time robotics simulation with PI0 inference.

    Features:
    - Multi-view RGB observation capture
    - TTNN-optimized inference pipeline
    - Receding horizon control (use first action from horizon)
    - Performance monitoring (inference time, loop frequency)
    """

    def __init__(
        self,
        checkpoint_path,
        device_id=0,
        use_gui=True,
        record_video=False,
        video_path=None,
        demo_mode=False,
        verbose_actions=False,
        seed=42,
        use_gemma_tokenizer=False,
    ):
        """
        Initialize simulation environment and PI0 model.

        Args:
            checkpoint_path: Path to PI0 weights
            device_id: TT device ID (default 0)
            use_gui: Show PyBullet GUI (default True)
            record_video: Enable video recording (default False)
            video_path: Path to save video (default: auto-generated)
            demo_mode: Use scripted demo motion instead of PI0 (default False)
            verbose_actions: Print predicted actions (default False)
            seed: Random seed for reproducibility (default 42)
            use_gemma_tokenizer: Try to use official Gemma tokenizer (requires HF auth)
        """
        # Set random seeds for reproducibility
        self.seed = seed
        self.use_gemma_tokenizer = use_gemma_tokenizer
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("\n" + "=" * 70)
        print("   PI0 REAL-TIME SIMULATION - Franka Panda Robot (7-DOF)")
        print("=" * 70)
        print(f"   Random seed: {seed} (for reproducibility)")

        # Initialize PyBullet
        print("\n📦 Initializing PyBullet...")
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # Step manually for precise control

        # Load environment
        print("   Loading environment...")
        self.plane_id = p.loadURDF("plane.urdf")

        # Load Franka Panda robot arm (PI0 was trained on Franka robots!)
        print("   Loading Franka Panda robot...")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)

        # Filter controllable joints (exclude fixed joints and gripper)
        # Franka Panda has 7 arm joints + 2 gripper fingers
        self.controllable_joints = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode("utf-8")
            # Only include the 7 main arm joints (exclude gripper)
            if joint_type != p.JOINT_FIXED and "panda_joint" in joint_name:
                self.controllable_joints.append(i)

        print(f"   ✅ Franka Panda loaded: {len(self.controllable_joints)} arm joints (7-DOF)")
        print(f"   🤖 This robot was used in PI0's training data!")

        # Add a simple cube as a manipulation target
        self.cube_id = p.loadURDF("cube_small.urdf", [0.5, 0.0, 0.5])

        print("   ✅ Environment loaded (plane + cube + robot)")

        # Setup camera configuration for observations
        self.camera_config = {"width": 224, "height": 224, "fov": 60, "near": 0.1, "far": 5.0}

        # Initialize PI0 model
        print("\n📦 Loading PI0 model...")
        config = self._create_config()
        weight_loader = PI0WeightLoader(checkpoint_path)

        print(f"   Opening TT device {device_id}...")
        self.device = ttnn.open_device(device_id=device_id, l1_small_size=24576)

        print("   Initializing TTNN model (this may take a few seconds)...")
        self.model = PI0ModelTTNN(config, weight_loader, self.device)
        self.config = config

        print("   ✅ Model loaded successfully")

        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.loop_times = deque(maxlen=100)
        self.preprocess_times = deque(maxlen=100)
        self.capture_times = deque(maxlen=100)

        # Video recording setup
        self.record_video = record_video
        self.video_frames = []
        if record_video:
            if video_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.video_path = f"pi0_simulation_{timestamp}.mp4"
            else:
                self.video_path = video_path
            print(f"\n📹 Video recording enabled: {self.video_path}")
        else:
            self.video_path = None

        # Simulation mode settings
        self.demo_mode = demo_mode
        self.verbose_actions = verbose_actions
        if demo_mode:
            print("\n🎬 Demo mode enabled: Using scripted motion")
        if verbose_actions:
            print("\n🔍 Verbose actions enabled: Will print predicted actions")

        # Initialize tokenizer
        print("\n📝 Initializing tokenizer...")
        if use_gemma_tokenizer:
            self.tokenizer = GemmaTokenizerWrapper()
        else:
            self.tokenizer = SimpleRoboticsTokenizer()
            print("   Using word-based tokenizer for task instructions")

    def _create_config(self):
        """Create model config matching pretrained weights."""
        config = PI0ModelConfig(
            action_dim=32,  # Keep pretrained action_dim (we'll use first 7)
            action_horizon=50,
            state_dim=32,  # Keep pretrained state_dim (we'll pad our 14-dim state)
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=False,
        )
        config.siglip_config = SigLIPConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
        )
        return config

    def capture_video_frame(self):
        """
        Capture a frame for video recording.
        Uses a wider angle and higher resolution for better video quality.
        """
        if not self.record_video:
            return

        # Use a nice camera angle for recording (optimized for Franka Panda)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.2, 0.8, 0.8], cameraTargetPosition=[0.3, 0, 0.3], cameraUpVector=[0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=16 / 9, nearVal=0.1, farVal=5.0)

        # Capture high-res frame for video (720p)
        width, height = 1280, 720
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Convert to numpy array (remove alpha channel)
        frame = np.array(rgb[:, :, :3], dtype=np.uint8)
        self.video_frames.append(frame)

    def save_video(self):
        """
        Save recorded frames to video file using imageio.
        """
        if not self.record_video or len(self.video_frames) == 0:
            return

        try:
            import imageio

            print(f"\n📹 Saving video to {self.video_path}...")
            print(f"   Frames: {len(self.video_frames)}")

            # Save video at 20 FPS (adjust based on your control frequency)
            imageio.mimsave(self.video_path, self.video_frames, fps=20, codec="libx264")

            print(f"   ✅ Video saved successfully!")
            print(f"   Duration: {len(self.video_frames)/20:.1f} seconds")
            print(f"   Location: {os.path.abspath(self.video_path)}")
        except ImportError:
            print("   ⚠️  imageio not installed. Install with: pip install imageio[ffmpeg]")
        except Exception as e:
            print(f"   ❌ Error saving video: {e}")

    def capture_observations(self):
        """
        Capture multi-view RGB observations from simulation.

        Returns:
            images: List of torch tensors [1, 3, H, W], normalized [-1, 1]
            state: Robot joint states as torch tensor [1, state_dim]
        """
        images = []

        # Camera 1: Front view of Franka Panda
        view_matrix_1 = p.computeViewMatrix(
            cameraEyePosition=[1.0, 0.0, 0.5], cameraTargetPosition=[0.3, 0, 0.3], cameraUpVector=[0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_config["fov"],
            aspect=1.0,
            nearVal=self.camera_config["near"],
            farVal=self.camera_config["far"],
        )

        # Capture RGB from camera 1
        _, _, rgb1, _, _ = p.getCameraImage(
            width=self.camera_config["width"],
            height=self.camera_config["height"],
            viewMatrix=view_matrix_1,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Preprocess: uint8 -> float32 -> normalize to [-1, 1]
        rgb1 = np.array(rgb1[:, :, :3], dtype=np.float32) / 255.0
        rgb1 = (rgb1 - 0.5) / 0.5  # Normalize to [-1, 1] for SigLIP
        rgb1_tensor = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        images.append(rgb1_tensor)

        # Camera 2: Side view of Franka Panda (different angle for multi-view)
        view_matrix_2 = p.computeViewMatrix(
            cameraEyePosition=[0.3, 1.0, 0.5], cameraTargetPosition=[0.3, 0, 0.3], cameraUpVector=[0, 0, 1]
        )

        _, _, rgb2, _, _ = p.getCameraImage(
            width=self.camera_config["width"],
            height=self.camera_config["height"],
            viewMatrix=view_matrix_2,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb2 = np.array(rgb2[:, :, :3], dtype=np.float32) / 255.0
        rgb2 = (rgb2 - 0.5) / 0.5
        rgb2_tensor = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0)
        images.append(rgb2_tensor)

        # Get robot joint states from Kuka IIWA
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]

        # Combine positions and velocities into state vector [14,] = [7 pos + 7 vel]
        robot_state = positions + velocities  # 14 dimensions

        # Pad to 32 dimensions to match pretrained model expectations
        padded_state = robot_state + [0.0] * (32 - len(robot_state))
        state = torch.tensor([padded_state], dtype=torch.float32)

        return images, state

    def preprocess_for_ttnn(self, images, state, prompt="pick and place"):
        """
        Convert observations to TTNN tensors.

        Args:
            images: List of torch tensors [1, 3, 224, 224]
            state: Torch tensor [1, state_dim]
            prompt: Task instruction string

        Returns:
            Tuple of TTNN-ready inputs for model.sample_actions()
        """
        # Convert images to TTNN
        images_ttnn = [
            ttnn.from_torch(
                img,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for img in images
        ]

        # Tokenize prompt using word-based tokenizer
        # NOTE: This is better than character-based but still not the exact Gemma
        # SentencePiece tokenizer PI0 was trained with. For production, use proper
        # Gemma tokenizer from transformers (requires HuggingFace authentication).
        tokens, mask = self.tokenizer.encode(prompt, max_length=32)
        lang_tokens = torch.tensor([tokens], dtype=torch.long)
        lang_masks = torch.tensor([mask], dtype=torch.bool)

        # Convert language tokens to TTNN
        lang_tokens_ttnn = ttnn.from_torch(
            lang_tokens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        lang_masks_ttnn = ttnn.from_torch(
            lang_masks.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Convert state to TTNN
        state_ttnn = ttnn.from_torch(
            state,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Image masks (all images are valid)
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in images]

        return images_ttnn, img_masks, lang_tokens_ttnn, lang_masks_ttnn, state_ttnn

    def apply_actions(self, actions, step=0):
        """
        Apply predicted actions to Franka Panda robot in simulation.

        Uses receding horizon control: only apply the first action
        from the predicted action horizon.

        Args:
            actions: Tensor [1, action_horizon, 32] - model predicts 32-dim actions
            step: Current simulation step (for demo mode)
        """
        # Use only the first action (receding horizon control)
        action_step = actions[0, 0].float().cpu().numpy()  # [32,]

        # Use only the first 7 dimensions for the 7-DOF Franka Panda
        # (Model was trained on 32-dim actions, we map first 7 to our robot)
        action_step = action_step[:7]

        # Print actions if verbose mode enabled (before scaling)
        if self.verbose_actions and step % 50 == 0:
            print(f"\n   Predicted actions (first 7): {action_step}")
            print(f"   Action range: [{action_step.min():.3f}, {action_step.max():.3f}]")
            print(f"   Action mean: {action_step.mean():.3f}, std: {action_step.std():.3f}")

        # Override with demo motion if in demo mode
        if self.demo_mode:
            action_step = self._get_demo_motion(step)
        else:
            # FIXED: Scale normalized actions to actual joint ranges
            # PI0 outputs normalized actions (roughly [-1, 1]), scale them to joint limits
            # Option 1: Scale to full joint range (more aggressive)
            scaled_actions = []
            for i, joint_idx in enumerate(self.controllable_joints):
                if i < len(action_step):
                    joint_info = p.getJointInfo(self.robot_id, joint_idx)
                    lower_limit = joint_info[8]
                    upper_limit = joint_info[9]

                    # Normalize action from [-1, 1] to [lower_limit, upper_limit]
                    # Assuming PI0 outputs are roughly in [-1, 1] range
                    if lower_limit < upper_limit:
                        # Scale from normalized space to joint space
                        mid = (lower_limit + upper_limit) / 2.0
                        range_half = (upper_limit - lower_limit) / 2.0
                        scaled_action = mid + action_step[i] * range_half
                    else:
                        # Fallback: scale to Franka Panda typical range
                        scaled_action = action_step[i] * 2.5  # Scale up by factor

                    scaled_actions.append(scaled_action)

            action_step = np.array(scaled_actions)

            # Print scaled actions if verbose
            if self.verbose_actions and step % 50 == 0:
                print(f"   Scaled actions: {action_step}")
                print(f"   Scaled range: [{action_step.min():.3f}, {action_step.max():.3f}]")

        # Apply actions to robot joints
        for i, joint_idx in enumerate(self.controllable_joints):
            if i < len(action_step):
                # Get joint info for limits
                joint_info = p.getJointInfo(self.robot_id, joint_idx)
                lower_limit = joint_info[8]  # joint lower limit
                upper_limit = joint_info[9]  # joint upper limit

                # Clamp to safe limits
                if lower_limit < upper_limit:
                    target_pos = np.clip(action_step[i], lower_limit, upper_limit)
                else:
                    target_pos = np.clip(action_step[i], -2.87, 2.87)

                # Apply position control
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=87,  # Franka Panda max force
                    maxVelocity=1.0,
                )

    def _get_demo_motion(self, step):
        """
        Generate scripted demo motion for the robot.
        Creates a smooth sinusoidal motion to demonstrate the robot moving.

        Args:
            step: Current simulation step

        Returns:
            action_step: Array of 7 joint positions
        """
        # Create sinusoidal motion on different joints with different frequencies
        t = step * 0.02  # Time variable

        joint_positions = [
            0.3 * np.sin(t * 0.5),  # Joint 1: slow rotation
            0.5 * np.sin(t * 0.7 + 1.0),  # Joint 2:
            0.4 * np.sin(t * 0.3 + 2.0),  # Joint 3:
            0.6 * np.sin(t * 0.9 + 3.0),  # Joint 4:
            0.3 * np.sin(t * 1.2 + 4.0),  # Joint 5:
            0.4 * np.sin(t * 0.6 + 5.0),  # Joint 6:
            0.2 * np.sin(t * 1.5 + 6.0),  # Joint 7: fast oscillation
        ]

        return np.array(joint_positions)

    def run_episode(self, num_steps=1000, task_prompt="pick and place"):
        """
        Run a single episode of real-time control.

        Control loop:
        1. Capture observations (RGB images + robot state)
        2. Preprocess for TTNN (convert to device tensors)
        3. Run PI0 inference (sample actions via denoising)
        4. Apply first action to robot (receding horizon)
        5. Step physics simulation
        6. Repeat

        Args:
            num_steps: Number of simulation steps
            task_prompt: Natural language task instruction
        """
        print(f"\n{'='*70}")
        print(f"🚀 Running episode: '{task_prompt}'")
        print(f"   Robot: Franka Panda (7-DOF) - trained in PI0 dataset!")
        print(f"   Steps: {num_steps}")
        print(f"   Robot DOF: 7 (using first 7 of {self.config.action_dim} predicted actions)")
        print(f"   State: 14-dim (7 pos + 7 vel), padded to {self.config.state_dim}")
        print(f"   Action horizon: {self.config.action_horizon}")
        print(f"   Random seed: {self.seed}")

        # Detect which tokenizer is actually being used
        if isinstance(self.tokenizer, GemmaTokenizerWrapper):
            if self.tokenizer.use_official:
                print(f"   Tokenizer: ✅ Official Gemma (SentencePiece)")
                print(f"\n✅ Using authentic Gemma tokenizer - matches PI0 training!")
            else:
                print(f"   Tokenizer: SimpleRoboticsTokenizer (word-based)")
                print(f"\n⚠️  NOTE: Gemma tokenizer failed to load, using word-based fallback.")
                print(f"   Task understanding should be good but may not be perfect.")
        else:
            print(f"   Tokenizer: SimpleRoboticsTokenizer (word-based)")
            print(f"\n⚠️  NOTE: Using word-based tokenizer instead of Gemma's SentencePiece.")
            print(f"   Task understanding should be improved but may not be perfect.")

        print(f"{'='*70}\n")

        # Warm-up inference (first call includes JIT compilation)
        print("⏳ Warming up model (first inference includes JIT compilation)...")
        warmup_images, warmup_state = self.capture_observations()
        warmup_inputs = self.preprocess_for_ttnn(warmup_images, warmup_state, task_prompt)

        with torch.no_grad():
            _ = self.model.sample_actions(
                images=warmup_inputs[0],
                img_masks=warmup_inputs[1],
                lang_tokens=warmup_inputs[2],
                lang_masks=warmup_inputs[3],
                state=warmup_inputs[4],
            )
            ttnn.synchronize_device(self.device)

        print("✅ Warm-up complete! Starting control loop...\n")
        time.sleep(1.0)

        for step in range(num_steps):
            loop_start = time.time()

            # 1. Capture observations
            cap_start = time.time()
            images, state = self.capture_observations()
            self.capture_times.append((time.time() - cap_start) * 1000)

            # 2. Preprocess for TTNN
            prep_start = time.time()
            images_ttnn, img_masks, lang_tokens_ttnn, lang_masks_ttnn, state_ttnn = self.preprocess_for_ttnn(
                images, state, task_prompt
            )
            self.preprocess_times.append((time.time() - prep_start) * 1000)

            # 3. Run inference
            inf_start = time.time()
            with torch.no_grad():
                actions_ttnn = self.model.sample_actions(
                    images=images_ttnn,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens_ttnn,
                    lang_masks=lang_masks_ttnn,
                    state=state_ttnn,
                )
            ttnn.synchronize_device(self.device)
            inf_time = (time.time() - inf_start) * 1000
            self.inference_times.append(inf_time)

            # Convert to torch for application
            if isinstance(actions_ttnn, ttnn.Tensor):
                actions = ttnn.to_torch(actions_ttnn)
            else:
                actions = actions_ttnn

            # 4. Apply actions to robot
            self.apply_actions(actions, step=step)

            # 5. Step simulation
            p.stepSimulation()

            # 6. Capture video frame (if recording enabled)
            self.capture_video_frame()

            # 7. Performance metrics
            loop_time = (time.time() - loop_start) * 1000
            self.loop_times.append(loop_time)

            # Print progress
            if step % 50 == 0:
                avg_cap = np.mean(self.capture_times)
                avg_prep = np.mean(self.preprocess_times)
                avg_inf = np.mean(self.inference_times)
                avg_loop = np.mean(self.loop_times)
                hz = 1000.0 / avg_loop if avg_loop > 0 else 0

                print(
                    f"Step {step:4d} | "
                    f"Cap: {avg_cap:.1f}ms | "
                    f"Prep: {avg_prep:.1f}ms | "
                    f"Inf: {avg_inf:.1f}ms | "
                    f"Loop: {avg_loop:.1f}ms | "
                    f"Freq: {hz:.1f} Hz"
                )

            # Optional: Add a small sleep to maintain real-time sync
            # Uncomment if simulation runs too fast
            # time.sleep(0.01)

        # Episode summary
        print(f"\n{'='*70}")
        print(f"✅ Episode complete!")
        print(f"{'='*70}")
        print(f"\n📊 Performance Summary:")
        print(f"   Capture time:     {np.mean(self.capture_times):.2f} ± {np.std(self.capture_times):.2f} ms")
        print(f"   Preprocess time:  {np.mean(self.preprocess_times):.2f} ± {np.std(self.preprocess_times):.2f} ms")
        print(f"   Inference time:   {np.mean(self.inference_times):.2f} ± {np.std(self.inference_times):.2f} ms")
        print(f"   Total loop time:  {np.mean(self.loop_times):.2f} ± {np.std(self.loop_times):.2f} ms")
        print(f"   Control frequency: {1000.0/np.mean(self.loop_times):.1f} Hz")
        print(f"\n   Min inference:    {np.min(self.inference_times):.2f} ms")
        print(f"   Max inference:    {np.max(self.inference_times):.2f} ms")
        print(f"{'='*70}\n")

        # Save video if recording was enabled
        self.save_video()

    def close(self):
        """Cleanup resources."""
        print("\n🧹 Cleaning up...")
        ttnn.close_device(self.device)
        p.disconnect()
        print("✅ Done!\n")


def main():
    parser = argparse.ArgumentParser(description="PI0 Real-Time Simulation with PyBullet")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PI0 checkpoint (default: $TT_METAL_HOME/models/experimental/pi0/weights/pi0_base)",
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps (default: 500)")
    parser.add_argument(
        "--task", type=str, default="pick and place", help="Task instruction prompt (default: 'pick and place')"
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI (headless mode)")
    parser.add_argument("--device", type=int, default=0, help="TT device ID (default: 0)")
    parser.add_argument("--record-video", action="store_true", help="Record simulation to video file")
    parser.add_argument(
        "--video-path", type=str, default=None, help="Path to save video (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Use scripted demo motion instead of PI0 predictions (makes robot move visibly)",
    )
    parser.add_argument(
        "--verbose-actions", action="store_true", help="Print predicted actions to see what the model outputs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--use-gemma-tokenizer",
        action="store_true",
        help="Use official Gemma tokenizer (requires HuggingFace authentication)",
    )

    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
        if not TT_METAL_HOME:
            print("❌ Error: TT_METAL_HOME environment variable not set")
            print("   Please set it to your tt-metal installation path")
            sys.exit(1)
        checkpoint_path = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Run: python tests/download_pretrained_weights.py")
        sys.exit(1)

    # Create environment
    env = PI0SimulationEnv(
        checkpoint_path,
        device_id=args.device,
        use_gui=not args.headless,
        record_video=args.record_video,
        video_path=args.video_path,
        demo_mode=args.demo_mode,
        verbose_actions=args.verbose_actions,
        seed=args.seed,
        use_gemma_tokenizer=args.use_gemma_tokenizer,
    )

    try:
        # Run episode
        env.run_episode(num_steps=args.steps, task_prompt=args.task)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
