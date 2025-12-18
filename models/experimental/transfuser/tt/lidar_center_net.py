# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
import ttnn

from models.experimental.transfuser.reference.utils import (
    get_lidar_to_bevimage_transform,
)

from models.experimental.transfuser.reference.point_pillar import PointPillarNet


from models.experimental.transfuser.tt.transfuser_backbone import TtTransfuserBackbone
from models.experimental.transfuser.tt.head import TTLidarCenterNetHead


class LidarCenterNet(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        in_channels: input channels
    """

    def __init__(
        self,
        device,
        parameters,
        config,
        backbone,
        image_architecture="resnet34",
        lidar_architecture="resnet18",
        use_velocity=True,
        torch_model=None,
        use_fallback=False,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        self.use_target_point_image = config.use_target_point_image
        self.gru_concat_target_point = config.gru_concat_target_point
        self.use_point_pillars = config.use_point_pillars

        if self.use_point_pillars == True:
            self.point_pillar_net = PointPillarNet(
                config.num_input,
                config.num_features,
                min_x=config.min_x,
                max_x=config.max_x,
                min_y=config.min_y,
                max_y=config.max_y,
                pixels_per_meter=int(config.pixels_per_meter),
            )

        self.backbone = backbone

        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": True,
            "math_approx_mode": False,
        }
        assert backbone == "transFuser", "Only Transfuser supported for LidarCenterNet."
        # self._model = TransfuserBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(
        #     torch.device("cpu")
        # )
        self._model = TtTransfuserBackbone(
            device,
            parameters=parameters,
            stride=2,
            model_config=model_config,
            config=self.config,
            torch_model=torch_model,
            use_fallback=use_fallback,
        )

        channel = config.channel

        self.pred_bev = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, kernel_size=(1, 1), stride=1, padding=0, bias=True),
        ).to(torch.device("cpu"))

        # prediction heads
        # self.head = LidarCenterNetHead(channel, channel, 1, train_cfg=config).to(self.device)
        # Initialize TTNN model

        self.head = TTLidarCenterNetHead(
            device=device,
            parameters=parameters["head"],
            in_channel=channel,
            feat_channel=channel,
            num_classes=1,
            num_dir_bins=config.num_dir_bins,
        )
        self.i = 0

        # waypoints prediction
        self.join = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        ).to(torch.device("cpu"))

        self.decoder = nn.GRUCell(
            input_size=4 if self.gru_concat_target_point else 2,  # 2 represents x,y coordinate
            hidden_size=self.config.gru_hidden_size,
        ).to(torch.device("cpu"))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(self.config.gru_hidden_size, 3).to(torch.device("cpu"))

    def forward_gru(self, z, target_point):
        z = self.join(z)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

        target_point = target_point.clone()
        target_point[:, 1] *= -1

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            if self.gru_concat_target_point:
                x_in = torch.cat([x, target_point], dim=1)
            else:
                x_in = x

            z = self.decoder(x_in, z)
            dx = self.output(z)

            x = dx[:, :2] + x

            output_wp.append(x[:, :2])

        pred_wp = torch.stack(output_wp, dim=1)

        # pred the wapoints in the vehicle coordinate and we convert it to lidar coordinate here because the GT waypoints is in lidar coordinate
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - self.config.lidar_pos[0]

        pred_brake = None
        steer = None
        throttle = None
        brake = None

        return pred_wp, pred_brake, steer, throttle, brake

    def get_bbox_local_metric(self, bbox):
        x, y, w, h, yaw, speed, brake, confidence = bbox

        w = (
            w / self.config.bounding_box_divisor / self.config.pixels_per_meter
        )  # We multiplied by 2 when collecting the data, and multiplied by 8 when loading the labels.
        h = (
            h / self.config.bounding_box_divisor / self.config.pixels_per_meter
        )  # We multiplied by 2 when collecting the data, and multiplied by 8 when loading the labels.

        T = get_lidar_to_bevimage_transform()
        T_inv = np.linalg.inv(T)

        center = np.array([x, y, 1.0])

        center_old_coordinate_sys = T_inv @ center

        center_old_coordinate_sys = center_old_coordinate_sys + np.array(self.config.lidar_pos)

        # Convert to standard CARLA right hand coordinate system
        center_old_coordinate_sys[1] = -center_old_coordinate_sys[1]

        bbox = np.array([[-h, -w, 1], [-h, w, 1], [h, w, 1], [h, -w, 1], [0, 0, 1], [0, h * speed * 0.5, 1]])

        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

        for point_index in range(bbox.shape[0]):
            bbox[point_index] = R @ bbox[point_index]
            bbox[point_index] = bbox[point_index] + np.array(
                [center_old_coordinate_sys[0], center_old_coordinate_sys[1], 0]
            )

        return bbox, brake, confidence

    def forward_ego(self, tt_rgb, tt_lidar_bev, tt_velocity, target_point):
        features, _, fused_features = self._model(tt_rgb, tt_lidar_bev, tt_velocity, self.device)

        return features, fused_features
        # Validate output_fused_tensor
        tt_fused_torch = ttnn.to_torch(fused_features, device=self.device, dtype=torch.float32)

        pred_wp, _, _, _, _ = self.forward_gru(tt_fused_torch, target_point)

        return features, pred_wp
