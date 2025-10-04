# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import timm


class RegNet(nn.Module):
    """
    Encoder network for image input list.
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, architecture, normalize=True, out_features=512):
        super().__init__()
        assert architecture.startswith("regnet"), f"Only RegNet architecture supported, got: {architecture}"

        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=True)

        self.features.fc = None

        self.features.conv1 = self.features.stem.conv
        self.features.bn1 = self.features.stem.bn
        self.features.act1 = nn.Sequential()  # The Relu is part of the batch norm here.
        self.features.maxpool = nn.Sequential()
        self.features.layer1 = self.features.s1
        self.features.layer2 = self.features.s2
        self.features.layer3 = self.features.s3
        self.features.layer4 = self.features.s4
        self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.features.head = nn.Sequential()


class Stage(nn.Module):
    """
    RegNet stage
    """

    def __init__(self, config, stage_name, image_architecture="resnet34"):
        super().__init__()
        self.config = config
        self.stage_name = stage_name

        self.image_encoder = RegNet(
            architecture=image_architecture, normalize=True, out_features=self.config.perception_output_features
        )
        self.s = getattr(self.image_encoder.features, stage_name)
        print(self.s.b1)

    def forward(self, image):
        """
        Args:
            image: Image Input
        """
        image_features = self.s.b1(image)

        # image_features = self.s.b1.conv1(image)
        # image_features = self.s.b1.conv2(image_features)
        # image_features = self.s.b1.se.fc1(image_features)
        # image_features = self.s.b1.se.fc2(image_features)
        # image_features = self.s.b1.conv3(image_features)
        # image_features = self.s.b2(image_features)
        # image_features = self.s(image)

        return image_features
