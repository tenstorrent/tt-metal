# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from models.experimental.transfuser.reference.gpt import GPT
from torch.nn import functional as F
import timm


def normalize_imagenet(x):
    """Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


class ImageCNN(nn.Module):
    """
    Encoder network for image input list.
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, architecture, normalize=True, out_features=512):
        super().__init__()
        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=False)
        self.features.fc = None
        # Delete parts of the networks we don't want
        if architecture.startswith("regnet"):  # Rename modules so we can use the same code
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

        elif architecture.startswith("convnext"):
            self.features.conv1 = self.features.stem._modules["0"]
            self.features.bn1 = self.features.stem._modules["1"]
            self.features.act1 = nn.Sequential()  # Don't see any activatin function after the stem. Need to verify
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.stages._modules["0"]
            self.features.layer2 = self.features.stages._modules["1"]
            self.features.layer3 = self.features.stages._modules["2"]
            self.features.layer4 = self.features.stages._modules["3"]
            self.features.global_pool = self.features.head
            self.features.global_pool.flatten = nn.Sequential()
            self.features.global_pool.fc = nn.Sequential()
            self.features.head = nn.Sequential()
            # ConvNext don't have the 0th entry that res nets use.
            self.features.feature_info.append(self.features.feature_info[3])
            self.features.feature_info[3] = self.features.feature_info[2]
            self.features.feature_info[2] = self.features.feature_info[1]
            self.features.feature_info[1] = self.features.feature_info[0]

            # This layer norm is not pretrained anymore but that shouldn't matter since it is the last layer in the network.
            _tmp = self.features.global_pool.norm
            self.features.global_pool.norm = nn.LayerNorm((out_features, 1, 1), _tmp.eps, _tmp.elementwise_affine)


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        in_channels: input channels
    """

    def __init__(self, architecture, in_channels=2, out_features=512):
        super().__init__()

        self._model = timm.create_model(architecture, pretrained=False)
        self._model.fc = None

        if architecture.startswith("regnet"):  # Rename modules so we can use the same code
            self._model.conv1 = self._model.stem.conv
            self._model.bn1 = self._model.stem.bn
            self._model.act1 = nn.Sequential()
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self._model.head = nn.Sequential()

        elif architecture.startswith("convnext"):
            self._model.conv1 = self._model.stem._modules["0"]
            self._model.bn1 = self._model.stem._modules["1"]
            self._model.act1 = nn.Sequential()
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.stages._modules["0"]
            self._model.layer2 = self._model.stages._modules["1"]
            self._model.layer3 = self._model.stages._modules["2"]
            self._model.layer4 = self._model.stages._modules["3"]
            self._model.global_pool = self._model.head
            self._model.global_pool.flatten = nn.Sequential()
            self._model.global_pool.fc = nn.Sequential()
            self._model.head = nn.Sequential()
            _tmp = self._model.global_pool.norm
            self._model.global_pool.norm = nn.LayerNorm((out_features, 1, 1), _tmp.eps, _tmp.elementwise_affine)

        # Change the first conv layer so that it matches the amount of channels in the LiDAR
        # Timm might be able to do this automatically
        _tmp = self._model.conv1
        use_bias = _tmp.bias != None
        self._model.conv1 = nn.Conv2d(
            in_channels,
            out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size,
            stride=_tmp.stride,
            padding=_tmp.padding,
            bias=use_bias,
        )
        # Need to delete the old conv_layer to avoid unused parameters
        if architecture.startswith("convnext"):
            del self._model.stem._modules["0"]
        elif architecture.startswith("regnet"):
            del self._model.stem.conv
        torch.cuda.empty_cache()
        if use_bias:
            self._model.conv1.bias = _tmp.bias

        del _tmp


class TransfuserBackbone(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture="resnet34", lidar_architecture="resnet18", use_velocity=True):
        super().__init__()
        self.config = config

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))

        self.image_encoder = ImageCNN(
            architecture=image_architecture, normalize=True, out_features=self.config.perception_output_features
        )

        if config.use_point_pillars == True:
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len

        if self.config.use_target_point_image == True:
            in_channels += 1

        self.lidar_encoder = LidarEncoder(
            architecture=lidar_architecture,
            in_channels=in_channels,
            out_features=self.config.perception_output_features,
        )

        self.transformer1 = GPT(
            n_embd=self.image_encoder.features.feature_info[1]["num_chs"],
            n_head=config.n_head,
            block_exp=config.block_exp,
            n_layer=config.n_layer,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            config=config,
            use_velocity=use_velocity,
        )

        self.transformer2 = GPT(
            n_embd=self.image_encoder.features.feature_info[2]["num_chs"],
            n_head=config.n_head,
            block_exp=config.block_exp,
            n_layer=config.n_layer,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            config=config,
            use_velocity=use_velocity,
        )

        self.transformer3 = GPT(
            n_embd=self.image_encoder.features.feature_info[3]["num_chs"],
            n_head=config.n_head,
            block_exp=config.block_exp,
            n_layer=config.n_layer,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            config=config,
            use_velocity=use_velocity,
        )

        self.transformer4 = GPT(
            n_embd=self.image_encoder.features.feature_info[4]["num_chs"],
            n_head=config.n_head,
            block_exp=config.block_exp,
            n_layer=config.n_layer,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            config=config,
            use_velocity=use_velocity,
        )

        if self.image_encoder.features.feature_info[4]["num_chs"] != self.config.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(
                self.image_encoder.features.feature_info[4]["num_chs"], self.config.perception_output_features, (1, 1)
            )
            self.change_channel_conv_lidar = nn.Conv2d(
                self.image_encoder.features.feature_info[4]["num_chs"], self.config.perception_output_features, (1, 1)
            )
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()

        # FPN fusion
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        # top down
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode="bilinear", align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))

        # lateral
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))

        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity, return_intermediates=False):
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        """
        intermediates = {}
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        lidar_tensor = lidar

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)
        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)

        # Image fusion at (B, 72, 40, 176)
        # Lidar fusion at (B, 72, 64, 64)
        image_embd_layer1 = self.avgpool_img(image_features)
        lidar_embd_layer1 = self.avgpool_lidar(lidar_features)
        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, velocity)
        image_features_layer1 = F.interpolate(
            image_features_layer1,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer1 = F.interpolate(
            lidar_features_layer1,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        # Image fusion at (B, 216, 20, 88)
        # Image fusion at (B, 216, 32, 32)
        image_embd_layer2 = self.avgpool_img(image_features)
        lidar_embd_layer2 = self.avgpool_lidar(lidar_features)
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, velocity)
        image_features_layer2 = F.interpolate(
            image_features_layer2,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer2 = F.interpolate(
            lidar_features_layer2,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        # Image fusion at (B, 576, 10, 44)
        # Image fusion at (B, 576, 16, 16)
        image_embd_layer3 = self.avgpool_img(image_features)
        lidar_embd_layer3 = self.avgpool_lidar(lidar_features)
        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, velocity)
        image_features_layer3 = F.interpolate(
            image_features_layer3,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer3 = F.interpolate(
            lidar_features_layer3,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3
        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        # Image fusion at (B, 1512, 5, 22)
        # Image fusion at (B, 1512, 8, 8)
        image_embd_layer4 = self.avgpool_img(image_features)
        lidar_embd_layer4 = self.avgpool_lidar(lidar_features)
        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, velocity)
        image_features_layer4 = F.interpolate(
            image_features_layer4,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer4 = F.interpolate(
            lidar_features_layer4,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4

        # Downsamples channels to 512
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)

        x4 = lidar_features
        image_features_grid = image_features  # For auxilliary information

        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)

        fused_features = image_features + lidar_features
        features = self.top_down(x4)
        return features, image_features_grid, fused_features
