# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.transfuser.tt.gpt import TTGpt
from models.experimental.transfuser.tt.topdown import TtTopDown
from models.experimental.transfuser.tt.stages import Ttstages
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    AutoShardedStrategyConfiguration,
)


class TtTransfuserBackbone:
    def __init__(
        self,
        device,
        parameters,
        model_args,
        stride,
        model_config,
        config,
        torch_model=None,
    ) -> None:
        self.device = device
        self.config = config
        self.inplanes = 32
        self.dtype = ttnn.bfloat16

        def make_stage(params, model_args, *, planes, blocks, s, groups, stage_name, with_torch):
            return Ttstages._make_layer(
                device=self.device,
                parameters=params,
                model_args=model_args,
                planes=planes,
                blocks=blocks,
                stride=s,
                groups=groups,
                model_config=model_config,
                stage_name=stage_name,
                torch_model=(torch_model if with_torch else None),
            )

        # ---------- Parameter roots ----------
        img = parameters.image_encoder.features
        lidar = parameters.lidar_encoder._model

        # ---------- Stems ----------
        conv1_params = model_args["image_encoder"]["stem"]["conv"]
        conv1_config = self._create_conv_config(
            parameters=img.conv1,
            batch_size=conv1_params["batch_size"],
            input_height=conv1_params["input_height"],
            input_width=conv1_params["input_width"],
            in_channels=conv1_params["in_channels"],
            out_channels=conv1_params["out_channels"],
            stride=conv1_params["stride"],
            kernel_size=conv1_params["kernel_size"],
            padding=conv1_params["padding"],
            groups=conv1_params["groups"],
        )
        self.conv1 = TtConv2d(conv1_config, device=device)

        lidar_conv1_params = model_args["lidar_encoder"]["conv1"]
        lidar_conv1_config = self._create_conv_config(
            parameters=lidar.conv1,
            batch_size=lidar_conv1_params["batch_size"],
            input_height=lidar_conv1_params["input_height"],
            input_width=lidar_conv1_params["input_width"],
            in_channels=lidar_conv1_params["in_channels"],
            out_channels=lidar_conv1_params["out_channels"],
            stride=lidar_conv1_params["stride"],
            kernel_size=lidar_conv1_params["kernel_size"],
            padding=lidar_conv1_params["padding"],
            groups=lidar_conv1_params["groups"],
        )
        self.lidar_conv1 = TtConv2d(lidar_conv1_config, device=device)

        # ---------- Stage specs (shared for image & lidar) ----------
        # (name, planes, blocks, stride, groups)
        specs = [
            ("layer1", 72, 2, 2, 3),
            ("layer2", 216, 5, 2, 9),
            ("layer3", 576, 13, 2, 24),
            ("layer4", 1512, 1, 2, 63),
        ]

        # Build image stages and lidar stages (pure TT)
        for name, planes, blocks, s, groups in specs:
            setattr(
                self,
                f"image_{name}",
                make_stage(
                    getattr(img, name),
                    model_args["image_encoder"][f"s{name.replace('layer', '')}"],
                    planes=planes,
                    blocks=blocks,
                    s=s,
                    groups=groups,
                    stage_name=name,
                    with_torch=True,
                ),
            )
            setattr(
                self,
                f"lidar_{name}",
                make_stage(
                    getattr(lidar, name),
                    model_args["lidar_encoder"][f"s{name.replace('layer', '')}"],
                    planes=planes,
                    blocks=blocks,
                    s=s,
                    groups=groups,
                    stage_name=name,
                    with_torch=False,
                ),
            )

        # ---------- GPT compute config ----------
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # ---------- Transformers 1..4 ----------
        # n_embd aligns with the planes of layer1..layer4
        embds = {1: 72, 2: 216, 3: 576, 4: 1512}
        for i in (1, 2, 3, 4):
            setattr(
                self,
                f"transformer{i}",
                TTGpt(
                    device=self.device,
                    parameters=parameters[f"transformer{i}"],
                    n_head=config.n_head,
                    n_layer=config.n_layer,
                    use_velocity=config.use_velocity,
                    img_vert_anchors=config.img_vert_anchors,
                    img_horz_anchors=config.img_horz_anchors,
                    lidar_vert_anchors=config.lidar_vert_anchors,
                    lidar_horz_anchors=config.lidar_horz_anchors,
                    seq_len=config.seq_len,
                    n_embd=embds[i],
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=compute_kernel_config,
                ),
            )

        # ---------- Optional channel adapters ----------
        if self.config.perception_output_features != 1512:
            conv1x1_params = model_args["change_channel_conv_image"]
            conv1x1_config = self._create_conv_config(
                parameters=parameters.change_channel_conv_image,
                batch_size=conv1x1_params["batch_size"],
                input_height=conv1x1_params["input_height"],
                input_width=conv1x1_params["input_width"],
                in_channels=conv1x1_params["in_channels"],
                out_channels=conv1x1_params["out_channels"],
                stride=conv1x1_params["stride"],
                kernel_size=conv1x1_params["kernel_size"],
                padding=conv1x1_params["padding"],
                groups=conv1x1_params["groups"],
                activation=None,
            )
            lidar_conv1x1_params = model_args["change_channel_conv_lidar"]
            lidar_conv1x1_config = self._create_conv_config(
                parameters=parameters.change_channel_conv_lidar,
                batch_size=lidar_conv1x1_params["batch_size"],
                input_height=lidar_conv1x1_params["input_height"],
                input_width=lidar_conv1x1_params["input_width"],
                in_channels=lidar_conv1x1_params["in_channels"],
                out_channels=lidar_conv1x1_params["out_channels"],
                stride=lidar_conv1x1_params["stride"],
                kernel_size=lidar_conv1x1_params["kernel_size"],
                padding=lidar_conv1x1_params["padding"],
                groups=lidar_conv1x1_params["groups"],
                activation=None,
            )
            self.change_channel_conv_image = TtConv2d(conv1x1_config, device=device)
            self.change_channel_conv_lidar = TtConv2d(lidar_conv1x1_config, device=device)

        # ---------- Top-down head ----------
        self.top_down = TtTopDown(
            device=device,
            parameters=parameters,
            perception_output_features=config.perception_output_features,
            bev_features_channels=config.bev_features_chanels,
            bev_upsample_factor=config.bev_upsample_factor,
        )

    def _create_conv_config(
        self,
        parameters,
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
        stride,
        kernel_size,
        padding,
        groups,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    ):
        # Convert weights to float32 format (required by tt_cnn builder)
        weight = parameters.weight
        if isinstance(weight, ttnn.Tensor):
            weight = ttnn.from_torch(ttnn.to_torch(weight), dtype=self.dtype)

        # Convert bias to shape (1, 1, 1, out_channels) in float32
        bias = None
        if "bias" in parameters and parameters.bias is not None:
            bias_torch = ttnn.to_torch(parameters.bias).reshape(1, 1, 1, -1)
            bias = ttnn.from_torch(bias_torch, dtype=self.dtype)

        # Convert stride to list format (required by ttnn.conv2d)
        if isinstance(stride, int):
            stride_list = [stride, stride]
        elif isinstance(stride, tuple) and len(stride) == 2:
            stride_list = list(stride)
        else:
            stride_list = stride

        # Convert padding to list format (required by ttnn.conv2d)
        if isinstance(padding, int):
            padding_list = [padding, padding]
        elif isinstance(padding, tuple) and len(padding) == 2:
            padding_list = list(padding)
        elif isinstance(padding, tuple) and len(padding) == 4:
            padding_list = list(padding)
        else:
            padding_list = padding

        # Select math fidelity based on block (HiFi4 for block 2 for better accuracy)
        math_fidelity = ttnn.MathFidelity.HiFi4

        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=kernel_size,
            stride=stride_list,
            padding=padding_list,
            groups=groups,
            weight=weight,
            bias=bias,
            activation=activation,
            activation_dtype=self.dtype,
            weights_dtype=self.dtype,
            output_dtype=self.dtype,
            sharding_strategy=AutoShardedStrategyConfiguration(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=True,
            deallocate_activation=True,
            enable_act_double_buffer=False,
        )

    def normalize_imagenet_ttnn(self, x):
        """Normalize input images according to ImageNet standards using TTNN operations."""
        # Convert from [0,255] to [0,1]
        x = ttnn.multiply(x, 1.0 / 255.0)

        # Create normalization constants as tensors
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        mean = ttnn.from_torch(
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        std_inv = ttnn.from_torch(
            torch.tensor([1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]).reshape(1, 1, 1, 3),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Normalize all channels at once (no slice/concat needed)
        x = ttnn.subtract(x, mean)
        x = ttnn.multiply(x, std_inv)

        return x

    def __call__(self, image_x, lidar_x, velocity, device):
        def _avgpool_to_L1(x, shape, out_hw):
            return ttnn.sharded_to_interleaved(
                ttnn.adaptive_avg_pool2d(
                    input_tensor=x,
                    batch_size=shape[0],
                    input_h=shape[1],
                    input_w=shape[2],
                    channels=shape[3],
                    output_size=out_hw,
                ),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        def _avgpool_to_L1_lidar_layer4(x, shape, out_hw):
            return ttnn.sharded_to_interleaved(
                ttnn.adaptive_avg_pool2d(
                    input_tensor=x,
                    batch_size=shape[0],
                    input_h=shape[1],
                    input_w=shape[2],
                    channels=shape[3],
                    output_size=out_hw,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                ),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        def _interp_bilinear_rm_dram_pad_slice_back_to_tile(feat, pad_c, scale, slice_to):
            # ROW_MAJOR + DRAM, optional pad(channel), upsample, optional slice, back to TILE
            out = ttnn.to_layout(feat, ttnn.ROW_MAJOR_LAYOUT)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            if pad_c and pad_c > 0:
                out = ttnn.pad(out, padding=((0, 0), (0, 0), (0, 0), (0, pad_c)), value=0.0)
            out = ttnn.upsample(out, scale_factor=scale, mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if slice_to is not None:
                out = ttnn.slice(out, [0, 0, 0, 0], slice_to)
            return ttnn.to_layout(out, ttnn.TILE_LAYOUT)

        # ---------- input normalize (unchanged) ----------
        image_x = self.normalize_imagenet_ttnn(image_x)

        # image_encoder_conv1
        image_out = self.conv1(image_x)
        # lidar_encoder_conv1
        lidar_out = self.lidar_conv1(lidar_x)
        # image_encoder_layer1
        for block in self.image_layer1:
            image_out = block(image_out, device)
        ttnn.ReadDeviceProfiler(device)
        # lidar_encoder_layer1
        for block in self.lidar_layer1:
            lidar_out = block(lidar_out, device)
        ttnn.ReadDeviceProfiler(device)
        # Layer1 avgpool - image
        image_embd_layer1 = _avgpool_to_L1(
            image_out, (1, 40, 176, 72), [self.config.img_vert_anchors, self.config.img_horz_anchors]
        )
        # Layer1 avgpool - lidar
        lidar_embd_layer1 = _avgpool_to_L1(
            lidar_out, (1, 64, 64, 72), [self.config.lidar_vert_anchors, self.config.lidar_horz_anchors]
        )
        # Layer1 transformer
        image_features_layer1, lidar_features_layer1 = self.transformer1(
            image_embd_layer1, lidar_embd_layer1, velocity, 72
        )
        image_features_layer1 = ttnn.permute(image_features_layer1, (0, 2, 3, 1))
        lidar_features_layer1 = ttnn.permute(lidar_features_layer1, (0, 2, 3, 1))
        # Layer1 bilinear interpolation - image
        image_features_layer1 = _interp_bilinear_rm_dram_pad_slice_back_to_tile(
            image_features_layer1, pad_c=24, scale=(8, 8), slice_to=[1, 40, 176, 72]
        )
        # Layer1 bilinear interpolation - lidar
        lidar_features_layer1 = _interp_bilinear_rm_dram_pad_slice_back_to_tile(
            lidar_features_layer1, pad_c=24, scale=(8, 8), slice_to=[1, 64, 64, 72]
        )
        # Layer1 Add
        image_out = ttnn.sharded_to_interleaved(image_out, ttnn.DRAM_MEMORY_CONFIG)
        lidar_out = ttnn.sharded_to_interleaved(lidar_out, ttnn.DRAM_MEMORY_CONFIG)
        image_out = ttnn.reshape(image_out, image_features_layer1.shape)
        lidar_out = ttnn.reshape(lidar_out, lidar_features_layer1.shape)
        image_features = ttnn.add(image_out, image_features_layer1)
        lidar_features = ttnn.add(lidar_out, lidar_features_layer1)
        # image_encoder_layer2
        for block in self.image_layer2:
            image_features = block(image_features, device)
        ttnn.ReadDeviceProfiler(device)
        # lidar_encoder_layer2
        for block in self.lidar_layer2:
            lidar_features = block(lidar_features, device)
        ttnn.ReadDeviceProfiler(device)
        # Layer2 avgpool - image
        image_embd_layer2 = _avgpool_to_L1(
            image_features, (1, 20, 88, 216), [self.config.img_vert_anchors, self.config.img_horz_anchors]
        )
        # Layer2 avgpool - lidar
        lidar_embd_layer2 = _avgpool_to_L1(
            lidar_features, (1, 32, 32, 216), [self.config.lidar_vert_anchors, self.config.lidar_horz_anchors]
        )
        # layer2 transformer
        image_features_layer2, lidar_features_layer2 = self.transformer2(
            image_embd_layer2, lidar_embd_layer2, velocity, 216
        )
        image_features_layer2 = ttnn.permute(image_features_layer2, (0, 2, 3, 1))
        lidar_features_layer2 = ttnn.permute(lidar_features_layer2, (0, 2, 3, 1))
        # Layer2 bilinear interpolation - image
        image_features_layer2 = _interp_bilinear_rm_dram_pad_slice_back_to_tile(
            image_features_layer2, pad_c=8, scale=(4, 4), slice_to=[1, 20, 88, 216]
        )
        # Layer2 bilinear interpolation - lidar
        lidar_features_layer2 = _interp_bilinear_rm_dram_pad_slice_back_to_tile(
            lidar_features_layer2, pad_c=8, scale=(4, 4), slice_to=[1, 32, 32, 216]
        )
        # layer2 Add
        image_features = ttnn.reshape(image_features, image_features_layer2.shape)
        lidar_features = ttnn.reshape(lidar_features, lidar_features_layer2.shape)
        if image_features.memory_config().buffer_type != ttnn.BufferType.DRAM:
            image_features = ttnn.to_memory_config(image_features, ttnn.DRAM_MEMORY_CONFIG)
        image_features = ttnn.add(image_features, image_features_layer2)
        lidar_features = ttnn.add(lidar_features, lidar_features_layer2)
        # image_encoder_layer3
        for block in self.image_layer3:
            image_features = block(image_features, device)
        ttnn.ReadDeviceProfiler(device)
        # lidar_encoder_layer3
        for block in self.lidar_layer3:
            lidar_features = block(lidar_features, device)
        ttnn.ReadDeviceProfiler(device)
        # Layer3 avgpool - image
        image_embd_layer3 = _avgpool_to_L1(
            image_features, (1, 10, 44, 576), [self.config.img_vert_anchors, self.config.img_horz_anchors]
        )
        # Layer3 avgpool - lidar
        lidar_embd_layer3 = _avgpool_to_L1(
            lidar_features, (1, 16, 16, 576), [self.config.lidar_vert_anchors, self.config.lidar_horz_anchors]
        )
        # layer3 transformer
        image_features_layer3, lidar_features_layer3 = self.transformer3(
            image_embd_layer3, lidar_embd_layer3, velocity, 576
        )
        image_features_layer3 = ttnn.permute(image_features_layer3, (0, 2, 3, 1))
        lidar_features_layer3 = ttnn.permute(lidar_features_layer3, (0, 2, 3, 1))
        # Layer3 bilinear interpolation - image
        image_features_layer3 = _interp_bilinear_rm_dram_pad_slice_back_to_tile(
            image_features_layer3, pad_c=0, scale=(2, 2), slice_to=None
        )
        # Layer3 bilinear interpolation - lidar
        lidar_features_layer3 = _interp_bilinear_rm_dram_pad_slice_back_to_tile(
            lidar_features_layer3, pad_c=0, scale=(2, 2), slice_to=None
        )
        # layer3 Add
        image_features = ttnn.reshape(image_features, image_features_layer3.shape)
        lidar_features = ttnn.reshape(lidar_features, lidar_features_layer3.shape)
        image_features = ttnn.add(image_features, image_features_layer3)
        lidar_features = ttnn.add(lidar_features, lidar_features_layer3)
        # image_encoder_layer4
        for block in self.image_layer4:
            image_features = block(image_features, device)
        ttnn.ReadDeviceProfiler(device)
        # lidar_encoder_layer4
        for block in self.lidar_layer4:
            lidar_features = block(lidar_features, device)
        ttnn.ReadDeviceProfiler(device)
        # Layer4 avgpool - image
        image_embd_layer4 = _avgpool_to_L1(
            image_features, (1, 5, 22, 1512), [self.config.img_vert_anchors, self.config.img_horz_anchors]
        )
        # Layer4 avgpool - lidar
        lidar_embd_layer4 = _avgpool_to_L1_lidar_layer4(
            lidar_features, (1, 8, 8, 1512), [self.config.lidar_vert_anchors, self.config.lidar_horz_anchors]
        )
        # layer4 transformer
        image_features_layer4, lidar_features_layer4 = self.transformer4(
            image_embd_layer4, lidar_embd_layer4, velocity, 1512
        )
        image_features_layer4 = ttnn.permute(image_features_layer4, (0, 2, 3, 1))
        lidar_features_layer4 = ttnn.permute(lidar_features_layer4, (0, 2, 3, 1))
        # Layer4 bilinear interpolation - image
        image_features_layer4 = _interp_bilinear_rm_dram_pad_slice_back_to_tile(
            image_features_layer4, pad_c=24, scale=(1, 1), slice_to=[1, 5, 22, 1512]
        )
        # Layer4 bilinear interpolation - lidar
        lidar_features_layer4 = _interp_bilinear_rm_dram_pad_slice_back_to_tile(
            lidar_features_layer4, pad_c=24, scale=(1, 1), slice_to=[1, 8, 8, 1512]
        )
        # layer4 Add
        image_features = ttnn.reshape(image_features, image_features_layer4.shape)
        lidar_features = ttnn.reshape(lidar_features, lidar_features_layer4.shape)
        image_features = ttnn.add(image_features, image_features_layer4)
        lidar_features = ttnn.add(lidar_features, lidar_features_layer4)
        # Downsamples channels to 512
        image_features, (iH, iW) = self.change_channel_conv_image(image_features, return_output_dim=True)
        lidar_features, (lH, lW) = self.change_channel_conv_lidar(lidar_features, return_output_dim=True)
        x4 = lidar_features  # Save for FPN
        image_features_grid = image_features  # For auxiliary information
        shape_ = (1, iH, iW, image_features.shape[-1])
        shape_l = (1, lH, lW, lidar_features.shape[-1])
        image_features_grid = ttnn.reshape(image_features_grid, shape_)
        x4 = ttnn.reshape(x4, shape_l)
        # Global average pooling
        image_features = ttnn.global_avg_pool2d(image_features)
        lidar_features = ttnn.global_avg_pool2d(lidar_features)
        # Flatten
        image_features = ttnn.reshape(image_features, (1, self.config.perception_output_features))
        lidar_features = ttnn.reshape(lidar_features, (1, self.config.perception_output_features))
        # fused_features
        fused_features = ttnn.add(image_features, lidar_features)
        # FPN top-down
        p2, p3, p4, p5 = self.top_down(x4)
        features = (p2, p3, p4, p5)
        return features, image_features_grid, fused_features
