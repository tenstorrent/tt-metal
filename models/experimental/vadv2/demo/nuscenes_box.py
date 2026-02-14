# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.eval.detection.constants import DETECTION_NAMES, ATTRIBUTE_NAMES


def color_map(data, cmap):
    """数值映射为颜色"""

    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = list(), 256 / cmo.N

    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i * k), int((i + 1) * k)):
            cs.append(c)
    cs = np.array(cs)
    data = np.uint8(255 * (data - dmin) / (dmax - dmin))

    return cs[data]


class CustomNuscenesBox:
    """Simple data class representing a 3d box including, label, score and velocity."""

    def __init__(
        self,
        center: List[float],
        size: List[float],
        orientation: Quaternion,
        fut_trajs: List[float],
        label: int = np.nan,
        score: float = np.nan,
        velocity: Tuple = (np.nan, np.nan, np.nan),
        name: str = None,
        token: str = None,
    ):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token
        self.fut_trajs = np.array(fut_trajs)

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = np.allclose(self.velocity, other.velocity) or (
            np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))
        )

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = (
            "label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], "
            "rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, "
            "vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}"
        )

        return repr_str.format(
            self.label,
            self.score,
            self.center[0],
            self.center[1],
            self.center[2],
            self.wlh[0],
            self.wlh[1],
            self.wlh[2],
            self.orientation.axis[0],
            self.orientation.axis[1],
            self.orientation.axis[2],
            self.orientation.degrees,
            self.orientation.radians,
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
            self.name,
            self.token,
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(
        self,
        axis: Axes,
        view: np.ndarray = np.eye(3),
        normalize: bool = False,
        colors: Tuple = ("b", "r", "k"),
        linewidth: float = 2,
        box_idx=None,
        alpha=0.5,
    ) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color, alpha):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth, alpha=alpha)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot(
                [corners.T[i][0], corners.T[i + 4][0]],
                [corners.T[i][1], corners.T[i + 4][1]],
                color=colors[2],
                linewidth=linewidth,
                alpha=alpha,
            )

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0], alpha)
        draw_rect(corners.T[4:], colors[1], alpha)

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot(
            [center_bottom[0], center_bottom_forward[0]],
            [center_bottom[1], center_bottom_forward[1]],
            color=colors[0],
            linewidth=linewidth,
            alpha=alpha,
        )
        if (
            box_idx is not None
            and center_bottom[0] > -35
            and center_bottom[1] > -35
            and center_bottom[0] < 35
            and center_bottom[1] < 35
        ):
            text = f"{box_idx}"
            axis.text(center_bottom[0], center_bottom[1], text, ha="left", fontsize=5)

    def render_fut_trajs(
        self, axis: Axes, color: str = "b", linewidth: float = 1, fut_ts: int = 6, mode_idx=None
    ) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """

        fut_coords = self.fut_trajs.reshape((-1, fut_ts, 2))
        if mode_idx is not None:
            fut_coords = fut_coords[[mode_idx]]
        alpha = 0.8
        for i in range(fut_coords.shape[0]):
            fut_coord = fut_coords[i]
            fut_coord = fut_coord.cumsum(axis=-2)
            fut_coord = fut_coord + self.center[:2]
            if np.abs(fut_coord[-1] - self.center[:2]).max() >= 10:
                if color == "g":
                    axis.scatter(fut_coord[-1, 0], fut_coord[-1, 1], c=color, marker="*", s=70, alpha=alpha)
                elif color == "b":
                    axis.scatter(fut_coord[-1, 0], fut_coord[-1, 1], c=color, marker="o", s=20, alpha=alpha)
                if (
                    mode_idx is None
                    and fut_coord[-1, 0] > -35
                    and fut_coord[-1, 1] > -35
                    and fut_coord[-1, 0] < 35
                    and fut_coord[-1, 1] < 35
                ):
                    text = f"{i}"
                    axis.text(fut_coord[-1, 0], fut_coord[-1, 1], text, ha="left", fontsize=5)
            axis.plot(
                [self.center[0], fut_coord[0, 0]],
                [self.center[1], fut_coord[0, 1]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
            for i in range(fut_coord.shape[0] - 1):
                axis.plot(
                    [fut_coord[i, 0], fut_coord[i + 1, 0]],
                    [fut_coord[i, 1], fut_coord[i + 1, 1]],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )

    def render_fut_trajs_grad_color(
        self,
        axis: Axes,
        linewidth: float = 1,
        linestyles="solid",
        cmap="viridis",
        fut_ts: int = 6,
        alpha: int = 0.8,
        mode_idx=None,
    ) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """

        fut_coords = self.fut_trajs.reshape((-1, fut_ts, 2))
        if mode_idx is not None:
            fut_coords = fut_coords[[mode_idx]]

        for i in range(fut_coords.shape[0]):
            fut_coord = fut_coords[i]
            fut_coord = fut_coord.cumsum(axis=-2)
            fut_coord = fut_coord + self.center[:2]
            fut_coord = np.concatenate((self.center[np.newaxis, :2], fut_coord), axis=0)
            fut_coord_segments = np.stack((fut_coord[:-1], fut_coord[1:]), axis=1)

            fut_vecs = None
            for j in range(fut_coord_segments.shape[0]):
                fut_vec_j = fut_coord_segments[j]
                x_linspace = np.linspace(fut_vec_j[0, 0], fut_vec_j[1, 0], 51)
                y_linspace = np.linspace(fut_vec_j[0, 1], fut_vec_j[1, 1], 51)
                xy = np.stack((x_linspace, y_linspace), axis=1)
                xy = np.stack((xy[:-1], xy[1:]), axis=1)
                if fut_vecs is None:
                    fut_vecs = xy
                else:
                    fut_vecs = np.concatenate((fut_vecs, xy), axis=0)

            y = np.sin(np.linspace(3 / 2 * np.pi, 5 / 2 * np.pi, 301))
            colors = color_map(y[:-1], cmap)
            line_segments = LineCollection(
                fut_vecs, colors=colors, linewidths=linewidth, linestyles=linestyles, cmap=cmap
            )

            # if mode_idx is None and abs(fut_coord[-1, 0]) < 35 and abs(fut_coord[-1, 1]) < 35:
            #     text = f'{i}'
            #     axis.text(fut_coord[-1, 0], fut_coord[-1, 1], text, ha='left', fontsize=5)

            axis.add_collection(line_segments)

    def render_fut_trajs_coords(self, axis: Axes, color: str = "b", linewidth: float = 1, fut_ts: int = 12) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """

        fut_coords = self.fut_trajs.reshape((-1, fut_ts, 2))
        alpha = 0.2 if color == "b" else 1
        for i in range(fut_coords.shape[0]):
            fut_coord = fut_coords[i]
            fut_coord = fut_coord + self.center[:2]
            if np.abs(fut_coord[-1] - self.center[:2]).max() >= 10:
                if color == "g":
                    axis.scatter(fut_coord[-1, 0], fut_coord[-1, 1], c=color, marker="*", s=70, alpha=alpha)
                elif color == "b":
                    axis.scatter(fut_coord[-1, 0], fut_coord[-1, 1], c=color, marker="o", s=20, alpha=alpha)
            axis.plot(
                [self.center[0], fut_coord[0, 0]],
                [self.center[1], fut_coord[0, 1]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
            for i in range(fut_coord.shape[0] - 1):
                axis.plot(
                    [fut_coord[i, 0], fut_coord[i + 1, 0]],
                    [fut_coord[i, 1], fut_coord[i + 1, 1]],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )

    def render_cv2(
        self,
        im: np.ndarray,
        view: np.ndarray = np.eye(3),
        normalize: bool = False,
        colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
        linewidth: int = 2,
    ) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im, (int(prev[0]), int(prev[1])), (int(corner[0]), int(corner[1])), color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(
                im,
                (int(corners.T[i][0]), int(corners.T[i][1])),
                (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                colors[2][::-1],
                linewidth,
            )

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(
            im,
            (int(center_bottom[0]), int(center_bottom[1])),
            (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
            colors[0][::-1],
            linewidth,
        )

    def copy(self) -> "CustomNuscenesBox":
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)


class CustomDetectionBox(EvalBox):
    """Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
        num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
        detection_name: str = "car",  # The class name used in the detection challenge.
        detection_score: float = -1.0,  # GT samples do not have a score.
        attribute_name: str = "",  # Box attribute. Each box can have at most 1 attribute.
        fut_trajs=None,
    ):  # future trajectories of a pred box, shape=[fut_ts*2].
        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, "Error: detection_name cannot be empty!"
        assert detection_name in DETECTION_NAMES, "Error: Unknown detection_name %s" % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == "", (
            "Error: Unknown attribute_name %s" % attribute_name
        )

        assert type(detection_score) == float, "Error: detection_score must be a float!"
        assert not np.any(np.isnan(detection_score)), "Error: detection_score may not be NaN!"

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name
        self.fut_trajs = fut_trajs

    def __eq__(self, other):
        return (
            self.sample_token == other.sample_token
            and self.translation == other.translation
            and self.size == other.size
            and self.rotation == other.rotation
            and self.velocity == other.velocity
            and self.ego_translation == other.ego_translation
            and self.num_pts == other.num_pts
            and self.detection_name == other.detection_name
            and self.detection_score == other.detection_score
            and self.attribute_name == other.attribute_name
            and self.fut_trajs == other.fut_trajs
        )

    def serialize(self) -> dict:
        """Serialize instance into json-friendly format."""
        return {
            "sample_token": self.sample_token,
            "translation": self.translation,
            "size": self.size,
            "rotation": self.rotation,
            "velocity": self.velocity,
            "ego_translation": self.ego_translation,
            "num_pts": self.num_pts,
            "detection_name": self.detection_name,
            "detection_score": self.detection_score,
            "attribute_name": self.attribute_name,
            "fut_trajs": self.fut_trajs,
        }

    @classmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized content."""
        return cls(
            sample_token=content["sample_token"],
            translation=tuple(content["translation"]),
            size=tuple(content["size"]),
            rotation=tuple(content["rotation"]),
            velocity=tuple(content["velocity"]),
            fut_trajs=tuple(content["fut_trajs"]),
            ego_translation=(0.0, 0.0, 0.0) if "ego_translation" not in content else tuple(content["ego_translation"]),
            num_pts=-1 if "num_pts" not in content else int(content["num_pts"]),
            detection_name=content["detection_name"],
            detection_score=-1.0 if "detection_score" not in content else float(content["detection_score"]),
            attribute_name=content["attribute_name"],
        )
