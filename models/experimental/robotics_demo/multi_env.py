# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Multi-environment PyBullet manager for the robotics demo suite.

Manages N independent PyBullet physics servers running headless,
each hosting a Franka Panda robot with a manipulation target.
Supports parallel observation capture, action application, and
composite frame rendering for live display and video recording.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError("PyBullet required: pip install pybullet")


DEFAULT_TASKS = [
    "pick up the cube",
    "push the block right",
    "lift the object",
    "reach the target",
]

CUBE_COLORS = [
    [1.0, 0.2, 0.2, 1.0],  # red
    [0.2, 0.5, 1.0, 1.0],  # blue
    [0.2, 0.9, 0.3, 1.0],  # green
    [1.0, 0.8, 0.1, 1.0],  # yellow
]


class SingleBulletEnv:
    """One headless PyBullet environment with a Franka Panda + cube."""

    def __init__(
        self,
        env_id: int = 0,
        cube_position: Optional[List[float]] = None,
        image_size: int = 224,
        seed: int = 42,
    ):
        self.env_id = env_id
        self.image_size = image_size

        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf", [0, 0, 0],
            useFixedBase=True, physicsClientId=self.physics_client,
        )

        if cube_position is None:
            cube_position = [0.5, 0.0, 0.025]
        self.cube_position = list(cube_position)
        self.cube_id = p.loadURDF(
            "cube_small.urdf", self.cube_position, physicsClientId=self.physics_client,
        )
        if env_id < len(CUBE_COLORS):
            p.changeVisualShape(
                self.cube_id, -1,
                rgbaColor=CUBE_COLORS[env_id],
                physicsClientId=self.physics_client,
            )

        self.controllable_joints = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            if info[2] != p.JOINT_FIXED and "panda_joint" in info[1].decode("utf-8"):
                self.controllable_joints.append(i)

        self.camera_config = {
            "width": image_size, "height": image_size,
            "fov": 60, "near": 0.1, "far": 5.0,
        }

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def capture_observations(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Return (list_of_image_tensors, state_tensor) for this env."""
        images = []
        proj = p.computeProjectionMatrixFOV(
            fov=self.camera_config["fov"], aspect=1.0,
            nearVal=self.camera_config["near"], farVal=self.camera_config["far"],
        )

        for eye, target in [
            ([1.0, 0.0, 0.5], [0.3, 0, 0.3]),
            ([0.3, 1.0, 0.5], [0.3, 0, 0.3]),
        ]:
            view = p.computeViewMatrix(
                cameraEyePosition=eye, cameraTargetPosition=target,
                cameraUpVector=[0, 0, 1],
            )
            _, _, rgba, _, _ = p.getCameraImage(
                width=self.camera_config["width"],
                height=self.camera_config["height"],
                viewMatrix=view, projectionMatrix=proj,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self.physics_client,
            )
            rgb = np.array(rgba[:, :, :3], dtype=np.float32) / 255.0
            rgb = (rgb - 0.5) / 0.5
            images.append(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0))

        states = p.getJointStates(self.robot_id, self.controllable_joints,
                                  physicsClientId=self.physics_client)
        pos = [s[0] for s in states]
        vel = [s[1] for s in states]
        padded = pos + vel + [0.0] * (32 - len(pos) - len(vel))
        state = torch.tensor([padded], dtype=torch.float32)
        return images, state

    # Per-environment camera angles so each quadrant looks distinct
    CAMERA_ANGLES = [
        {"eye": [1.3, 0.6, 0.9], "target": [0.3, 0.0, 0.3]},   # front-right
        {"eye": [0.2, 1.2, 0.8], "target": [0.3, 0.0, 0.3]},   # left-side
        {"eye": [1.0, -0.7, 1.0], "target": [0.3, 0.0, 0.25]}, # front-left high
        {"eye": [0.8, 0.9, 0.5], "target": [0.4, 0.0, 0.2]},   # low angle
    ]

    def capture_display_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Capture a pretty frame for live display / video (not model input)."""
        cam = self.CAMERA_ANGLES[self.env_id % len(self.CAMERA_ANGLES)]
        view = p.computeViewMatrix(
            cameraEyePosition=cam["eye"],
            cameraTargetPosition=cam["target"],
            cameraUpVector=[0, 0, 1],
        )
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.1, farVal=5.0)
        _, _, rgba, _, _ = p.getCameraImage(
            width=width, height=height,
            viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client,
        )
        return np.array(rgba[:, :, :3], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def apply_actions(
        self,
        actions: np.ndarray,
        use_delta: bool = True,
        delta_scale: float = 1.0,
        max_velocity: float = 5.0,
    ):
        """Apply 7-dim joint targets to the robot."""
        for i, jidx in enumerate(self.controllable_joints):
            if i >= len(actions):
                break
            info = p.getJointInfo(self.robot_id, jidx, physicsClientId=self.physics_client)
            lo, hi = info[8], info[9]

            if use_delta:
                cur = p.getJointState(self.robot_id, jidx, physicsClientId=self.physics_client)[0]
                target = cur + actions[i] * delta_scale
            else:
                mid = (lo + hi) / 2.0
                half = (hi - lo) / 2.0
                target = mid + actions[i] * half if lo < hi else actions[i] * 2.5

            if lo < hi:
                target = float(np.clip(target, lo, hi))
            else:
                target = float(np.clip(target, -2.87, 2.87))

            p.setJointMotorControl2(
                self.robot_id, jidx, p.POSITION_CONTROL,
                targetPosition=target, force=87, maxVelocity=max_velocity,
                physicsClientId=self.physics_client,
            )

    def step(self, num_substeps: int = 8):
        """Step physics multiple times so position-controlled joints converge."""
        for _ in range(num_substeps):
            p.stepSimulation(physicsClientId=self.physics_client)

    # ------------------------------------------------------------------
    # Spatial info
    # ------------------------------------------------------------------
    def get_ee_position(self) -> np.ndarray:
        state = p.getLinkState(self.robot_id, 11, physicsClientId=self.physics_client)
        return np.array(state[0])

    def get_cube_position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.physics_client)
        return np.array(pos)

    def get_distance_to_target(self) -> float:
        return float(np.linalg.norm(self.get_ee_position() - self.get_cube_position()))

    def close(self):
        p.disconnect(physicsClientId=self.physics_client)


class MultiEnvironment:
    """Manages N SingleBulletEnv instances for parallel simulation."""

    def __init__(
        self,
        num_envs: int = 4,
        tasks: Optional[List[str]] = None,
        image_size: int = 224,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.tasks = tasks or DEFAULT_TASKS[:num_envs]
        while len(self.tasks) < num_envs:
            self.tasks.append(self.tasks[-1])

        cube_offsets = [
            [0.5, 0.0, 0.025],
            [0.5, 0.1, 0.025],
            [0.4, -0.1, 0.025],
            [0.6, 0.0, 0.025],
        ]
        self.envs: List[SingleBulletEnv] = []
        for i in range(num_envs):
            cp = cube_offsets[i % len(cube_offsets)]
            self.envs.append(SingleBulletEnv(env_id=i, cube_position=cp,
                                             image_size=image_size, seed=seed + i))

    def capture_all_observations(self) -> List[Tuple[List[torch.Tensor], torch.Tensor]]:
        return [env.capture_observations() for env in self.envs]

    def capture_all_display_frames(self, width: int = 640, height: int = 480) -> List[np.ndarray]:
        return [env.capture_display_frame(width, height) for env in self.envs]

    def apply_all_actions(self, all_actions: List[np.ndarray], **kwargs):
        for env, act in zip(self.envs, all_actions):
            env.apply_actions(act, **kwargs)

    def step_all(self):
        for env in self.envs:
            env.step()

    def get_all_distances(self) -> List[float]:
        return [env.get_distance_to_target() for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()
