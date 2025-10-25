# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import cv2
import copy
from skimage.draw import polygon

ego_width, ego_length = 1.85, 4.084


class PlanningMetric:
    def __init__(self):
        super().__init__()
        self.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]

        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = ego_width
        self.H = ego_length

        self.category_index = {"human": [2, 3, 4, 5, 6, 7, 8], "vehicle": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx

    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor(
            [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long
        )

        return bev_resolution, bev_start_position, bev_dimension

    def get_label(self, gt_agent_boxes, gt_agent_feats):
        segmentation_np, pedestrian_np = self.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0)

        return segmentation, pedestrian

    def get_birds_eye_view_label(self, gt_agent_boxes, gt_agent_feats):
        T = 6
        segmentation = np.zeros((T, self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((T, self.bev_dimension[0], self.bev_dimension[1]))
        agent_num = gt_agent_feats.shape[1]

        gt_agent_boxes = gt_agent_boxes.tensor.cpu().numpy()  # (N, 9)
        gt_agent_feats = gt_agent_feats.cpu().numpy()

        gt_agent_fut_trajs = gt_agent_feats[..., : T * 2].reshape(-1, 6, 2)
        gt_agent_fut_mask = gt_agent_feats[..., T * 2 : T * 3].reshape(-1, 6)
        gt_agent_fut_yaw = gt_agent_feats[..., T * 3 + 10 : T * 4 + 10].reshape(-1, 6, 1)
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        gt_agent_boxes[:, 6:7] = -1 * (gt_agent_boxes[:, 6:7] + np.pi / 2)  # NOTE: convert yaw to lidar frame
        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]

        for t in range(T):
            for i in range(agent_num):
                if gt_agent_fut_mask[i][t] == 1:
                    # Filter out all non vehicle instances
                    category_index = int(gt_agent_feats[0, i][27])
                    agent_length, agent_width = gt_agent_boxes[i][4], gt_agent_boxes[i][3]
                    x_a = gt_agent_fut_trajs[i, t, 0]
                    y_a = gt_agent_fut_trajs[i, t, 1]
                    yaw_a = gt_agent_fut_yaw[i, t, 0]
                    param = [x_a, y_a, yaw_a, agent_length, agent_width]
                    if category_index in self.category_index["vehicle"]:
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(segmentation[t], [poly_region], 1.0)
                    if category_index in self.category_index["human"]:
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(pedestrian[t], [poly_region], 1.0)

        return segmentation, pedestrian

    def _get_poly_region_in_image(self, param):
        lidar2cv_rot = np.array([[1, 0], [0, -1]])
        x_a, y_a, yaw_a, agent_length, agent_width = param
        trans_a = np.array([[x_a, y_a]]).T
        rot_mat_a = np.array([[np.cos(yaw_a), -np.sin(yaw_a)], [np.sin(yaw_a), np.cos(yaw_a)]])
        agent_corner = np.array(
            [
                [agent_length / 2, -agent_length / 2, -agent_length / 2, agent_length / 2],
                [agent_width / 2, agent_width / 2, -agent_width / 2, -agent_width / 2],
            ]
        )  # (2,4)
        agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a  # (2,4)
        # convert to cv frame
        agent_corner_cv2 = (
            np.matmul(lidar2cv_rot, agent_corner_lidar)
            - self.bev_start_position[:2, None]
            + self.bev_resolution[:2, None] / 2.0
        ).T / self.bev_resolution[
            :2
        ]  # (4,2)
        agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)

        return agent_corner_cv2

    def evaluate_single_coll(self, traj, segmentation, input_gt):
        pts = np.array(
            [
                [-self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, -self.W / 2.0],
                [-self.H / 2.0 + 0.5, -self.W / 2.0],
            ]
        )
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        trajs_ = copy.deepcopy(trajs)
        trajs_[:, :, [0, 1]] = trajs_[:, :, [1, 0]]  # can also change original tensor
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc  # (n_future, 32, 2)

        r = (self.bev_dimension[0] - trajs_[:, :, 0]).astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:, :, 1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())

        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        B, n_future, _ = trajs.shape

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i], input_gt=True)

            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            xi = ((-self.bx[0] / 2 - yy) / self.dx[0]).long()
            yi = ((-self.bx[1] / 2 + xx) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            ).to(gt_box_coll.device)
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], xi[m1], yi[m1]].long()

            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i], input_gt=False).to(ti.device)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs):
        pred_len = trajs.shape[0]
        ade = float(
            sum(
                torch.sqrt((trajs[i, 0] - gt_trajs[i, 0]) ** 2 + (trajs[i, 1] - gt_trajs[i, 1]) ** 2)
                for i in range(pred_len)
            )
            / pred_len
        )

        return ade
