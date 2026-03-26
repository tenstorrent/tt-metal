# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single PyBullet environment for the single-chip demo.

Franka Panda robot with a manipulation cube, two cameras,
and helpers for observation capture, action application,
and video-quality frame rendering.
"""

import numpy as np
import torch
from typing import List, Tuple

import pybullet as p
import pybullet_data


class FrankaCubeEnv:
    """Headless Franka Panda + cube simulation."""

    def __init__(self, image_size: int = 224, cube_position=None):
        self.image_size = image_size
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)

        self.plane_id = p.loadURDF("plane.urdf",
                                   physicsClientId=self.physics_client)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0],
                                   useFixedBase=True,
                                   physicsClientId=self.physics_client)

        if cube_position is None:
            cube_position = [0.5, 0.0, 0.025]
        self.cube_position = list(cube_position)
        self.cube_id = p.loadURDF("cube_small.urdf", self.cube_position,
                                  physicsClientId=self.physics_client)
        p.changeVisualShape(self.cube_id, -1, rgbaColor=[1, 0.2, 0.2, 1],
                            physicsClientId=self.physics_client)

        self.controllable_joints = []
        for i in range(p.getNumJoints(self.robot_id,
                                       physicsClientId=self.physics_client)):
            info = p.getJointInfo(self.robot_id, i,
                                  physicsClientId=self.physics_client)
            if info[2] != p.JOINT_FIXED and "panda_joint" in info[1].decode():
                self.controllable_joints.append(i)

    def capture_observations(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Two-camera RGB images normalized for SigLIP, plus 32-dim state."""
        images = []
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0,
                                            nearVal=0.1, farVal=5.0)
        for eye, target in [([1.0, 0.0, 0.5], [0.3, 0, 0.3]),
                             ([0.3, 1.0, 0.5], [0.3, 0, 0.3])]:
            view = p.computeViewMatrix(cameraEyePosition=eye,
                                       cameraTargetPosition=target,
                                       cameraUpVector=[0, 0, 1])
            _, _, rgba, _, _ = p.getCameraImage(
                width=self.image_size, height=self.image_size,
                viewMatrix=view, projectionMatrix=proj,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self.physics_client)
            rgb = np.array(rgba[:, :, :3], dtype=np.float32) / 255.0
            rgb = (rgb - 0.5) / 0.5
            images.append(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0))

        states = p.getJointStates(self.robot_id, self.controllable_joints,
                                  physicsClientId=self.physics_client)
        pos = [s[0] for s in states]
        vel = [s[1] for s in states]
        padded = pos + vel + [0.0] * (32 - len(pos) - len(vel))
        return images, torch.tensor([padded], dtype=torch.float32)

    def capture_display_frame(self, width: int = 960, height: int = 540) -> np.ndarray:
        """High-quality render for the UI video panel."""
        view = p.computeViewMatrix(cameraEyePosition=[1.3, 0.6, 0.9],
                                   cameraTargetPosition=[0.3, 0, 0.3],
                                   cameraUpVector=[0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=width / height,
                                            nearVal=0.1, farVal=5.0)
        _, _, rgba, _, _ = p.getCameraImage(
            width=width, height=height, viewMatrix=view,
            projectionMatrix=proj, renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client)
        return np.array(rgba[:, :, :3], dtype=np.uint8)

    def capture_pil_image(self) -> "Image":
        """512x512 PIL image for SmolVLA's processor."""
        from PIL import Image as PILImage
        view = p.computeViewMatrix(cameraEyePosition=[1.0, 0.0, 0.5],
                                   cameraTargetPosition=[0.3, 0, 0.3],
                                   cameraUpVector=[0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0,
                                            nearVal=0.1, farVal=5.0)
        _, _, rgba, _, _ = p.getCameraImage(
            width=512, height=512, viewMatrix=view,
            projectionMatrix=proj, renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client)
        return PILImage.fromarray(np.array(rgba[:, :, :3], dtype=np.uint8))

    def apply_actions(self, actions: np.ndarray, use_delta: bool = True,
                      delta_scale: float = 1.0, max_velocity: float = 5.0):
        for i, jidx in enumerate(self.controllable_joints):
            if i >= len(actions):
                break
            info = p.getJointInfo(self.robot_id, jidx,
                                  physicsClientId=self.physics_client)
            lo, hi = info[8], info[9]
            if use_delta:
                cur = p.getJointState(self.robot_id, jidx,
                                      physicsClientId=self.physics_client)[0]
                target = cur + actions[i] * delta_scale
            else:
                mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
                target = mid + actions[i] * half if lo < hi else actions[i] * 2.5
            target = float(np.clip(target, lo if lo < hi else -2.87,
                                    hi if lo < hi else 2.87))
            p.setJointMotorControl2(
                self.robot_id, jidx, p.POSITION_CONTROL,
                targetPosition=target, force=87, maxVelocity=max_velocity,
                physicsClientId=self.physics_client)

    def step(self, substeps: int = 8):
        for _ in range(substeps):
            p.stepSimulation(physicsClientId=self.physics_client)

    def get_ee_position(self) -> np.ndarray:
        return np.array(p.getLinkState(self.robot_id, 11,
                        physicsClientId=self.physics_client)[0])

    def get_cube_position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(
            self.cube_id, physicsClientId=self.physics_client)
        return np.array(pos)

    def get_distance_to_target(self) -> float:
        return float(np.linalg.norm(self.get_ee_position() -
                                     self.get_cube_position()))

    def reset(self):
        """Reset robot to home position and cube to original location."""
        home = [0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.8]
        for i, jidx in enumerate(self.controllable_joints):
            p.resetJointState(self.robot_id, jidx, home[i],
                              physicsClientId=self.physics_client)
        p.resetBasePositionAndOrientation(
            self.cube_id, self.cube_position, [0, 0, 0, 1],
            physicsClientId=self.physics_client)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physics_client)

    def close(self):
        p.disconnect(physicsClientId=self.physics_client)
