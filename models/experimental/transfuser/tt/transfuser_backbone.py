# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.transfuser.tt.utils import TTConv2D
from typing import List
from loguru import logger
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck
from models.experimental.transfuser.tt.gpt import TTGpt


class TtTransfuserBackbone:
    def __init__(
        self,
        device,
        parameters,
        stride,
        model_config,
        config,
    ) -> None:
        self.device = device
        self.config = config
        self.inplanes = 32
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.image_encoder.features.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
        )
        self.lidar_conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.lidar_encoder._model.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
            fp32_dest_acc_en=model_config.get("fp32_dest_acc_en", True),
            packer_l1_acc=model_config.get("packer_l1_acc", True),
            math_approx_mode=model_config.get("math_approx_mode", False),
        )
        # Layer1 for both encoders
        self.image_layer1 = self._make_layer(
            parameters=parameters.image_encoder.features.layer1,
            planes=72,
            blocks=2,  # no of bottlenecks
            stride=2,
            groups=3,  # conv2
            model_config=model_config,
            stage_name="layer1",
        )

        self.lidar_layer1 = self._make_layer(
            parameters=parameters.lidar_encoder._model.layer1,
            planes=72,
            blocks=2,
            stride=2,
            groups=3,
            model_config=model_config,
            stage_name="layer1",
        )

        # Layer2 for both encoders
        self.image_layer2 = self._make_layer(
            parameters=parameters.image_encoder.features.layer2,
            planes=216,
            blocks=5,
            stride=2,
            groups=9,  # conv2
            model_config=model_config,
            stage_name="layer2",
        )

        self.lidar_layer2 = self._make_layer(
            parameters=parameters.lidar_encoder._model.layer2,
            planes=216,
            blocks=5,
            stride=2,
            groups=9,
            model_config=model_config,
            stage_name="layer2",
        )

        # Layer3 for both encoders
        self.image_layer3 = self._make_layer(
            parameters=parameters.image_encoder.features.layer3,
            planes=576,
            blocks=13,
            stride=2,
            groups=24,  # conv2
            model_config=model_config,
            stage_name="layer3",
        )

        self.lidar_layer3 = self._make_layer(
            parameters=parameters.lidar_encoder._model.layer3,
            planes=576,
            blocks=13,
            stride=2,
            groups=24,
            model_config=model_config,
            stage_name="layer3",
        )

        # High accuracy compute kernel config
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Layer4 for both encoders
        self.image_layer4 = self._make_layer(
            parameters=parameters.image_encoder.features.layer4,
            planes=1512,
            blocks=1,
            stride=2,
            groups=63,  # conv2
            model_config=model_config,
            stage_name="layer4",
        )

        self.lidar_layer4 = self._make_layer(
            parameters=parameters.lidar_encoder._model.layer4,
            planes=1512,
            blocks=1,
            stride=2,
            groups=63,
            model_config=model_config,
            stage_name="layer4",
        )

        self.transformer1 = TTGpt(
            device=self.device,
            parameters=parameters["transformer1"],
            n_head=config.n_head,
            n_layer=config.n_layer,
            use_velocity=config.use_velocity,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            n_embd=72,  # layer1 output channels
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        self.transformer2 = TTGpt(
            device=self.device,
            parameters=parameters["transformer2"],
            n_head=config.n_head,
            n_layer=config.n_layer,
            use_velocity=config.use_velocity,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            n_embd=216,  # layer2 output channels
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        self.transformer3 = TTGpt(
            device=self.device,
            parameters=parameters["transformer3"],
            n_head=config.n_head,
            n_layer=config.n_layer,
            use_velocity=config.use_velocity,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            n_embd=576,  # layer3 output channels
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # compute_kernel_config=compute_kernel_config,
        )
        self.transformer4 = TTGpt(
            device=self.device,
            parameters=parameters["transformer4"],
            n_head=config.n_head,
            n_layer=config.n_layer,
            use_velocity=config.use_velocity,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            n_embd=1512,  # layer4 output channels
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # compute_kernel_config=compute_kernel_config,
        )

    # def _make_layer(
    #     self,
    #     parameters,
    #     planes: int,
    #     blocks: int,
    #     stride: int,
    #     groups: int = 1,
    #     model_config=None,
    #     stage_name=None,
    # ) -> List[TTRegNetBottleneck]:
    #     layers = []

    #     # Determine shard layout based on stage name
    #     if stage_name == "layer1":
    #         shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    #     elif stage_name == "layer2":
    #         shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    #     elif stage_name == "layer3":
    #         shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    #     elif stage_name == "layer4":
    #         shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    #     else:
    #         # Default to HEIGHT_SHARDED for backward compatibility
    #         shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    #     # First block (may have downsample)
    #     downsample = stride != 1 or self.inplanes != planes
    #     layers.append(
    #         TTRegNetBottleneck(
    #             parameters=parameters["b1"],
    #             model_config=model_config,
    #             stride=stride,
    #             downsample=downsample,
    #             groups=groups,
    #             shard_layout=shard_layout,
    #         )
    #     )
    #     self.inplanes = planes

    #     # Remaining blocks
    #     for block_num in range(1, blocks):
    #         block_name = f"b{block_num + 1}"
    #         layers.append(
    #             TTRegNetBottleneck(
    #                 parameters=parameters[block_name],
    #                 model_config=model_config,
    #                 stride=1,
    #                 downsample=False,
    #                 groups=groups,
    #                 shard_layout=shard_layout,
    #             )
    #         )

    #     return layers

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        groups: int = 1,
        model_config=None,
        stage_name=None,
    ) -> List[TTRegNetBottleneck]:
        """
        parameters:
        - Either a root dict that contains {layer1, layer2, ...} each with {b1,b2,...}
        - Or a stage dict that directly contains {b1,b2,...}
        stage_name:
        - Required if 'parameters' is the root dict (so we can pick the stage).
        - Ignored if 'parameters' already looks like a stage dict.
        """

        # ---- Resolve which stage dict to use ----
        def _resolve_stage_dict(params, stage_key):
            # If it already looks like a stage dict (has b1), just use it
            if isinstance(params, dict) and any(k.startswith("b") for k in params.keys()):
                return params
            # Otherwise expect a root dict with the stage_name present
            if not isinstance(params, dict) or stage_key not in params:
                available = list(params.keys()) if isinstance(params, dict) else []
                raise KeyError(
                    f"Expected a stage dict for '{stage_key}' or a root dict containing it. " f"Got keys: {available}"
                )
            return params[stage_key]

        stage_params = _resolve_stage_dict(parameters, stage_name)

        # ---- Choose shard layout per stage ----
        if stage_name in ("layer1", "layer2"):
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif stage_name in ("layer3", "layer4"):
            shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            # Default to HEIGHT_SHARDED
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

        # ---- Validate available blocks ----
        # Expected names: b1, b2, ..., b{blocks}
        available_block_names = sorted(
            [k for k in stage_params.keys() if k.startswith("b")],
            key=lambda s: int(s[1:]) if s[1:].isdigit() else 0,
        )

        # If fewer blocks than requested, raise a descriptive error
        if len(available_block_names) < blocks:
            raise KeyError(
                f"Requested {blocks} blocks for {stage_name}, but only found blocks: "
                f"{available_block_names}. "
                f"Did you pass parameters for the wrong stage (e.g., layer1 for layer2)?"
            )

        layers = []

        # ---- First block (may have downsample) ----
        downsample = stride != 1 or self.inplanes != planes
        layers.append(
            TTRegNetBottleneck(
                parameters=stage_params["b1"],
                model_config=model_config,
                stride=stride,
                downsample=downsample,
                groups=groups,
                shard_layout=shard_layout,
            )
        )
        self.inplanes = planes

        # ---- Remaining blocks (stride=1, no downsample) ----
        # Build exactly the number requested, in order b2..b{blocks}
        for idx in range(2, blocks + 1):
            bname = f"b{idx}"
            if bname not in stage_params:
                # Extra guard (should have been caught above)
                raise KeyError(f"Missing block '{bname}' in {stage_name}. " f"Available: {available_block_names}")
            layers.append(
                TTRegNetBottleneck(
                    parameters=stage_params[bname],
                    model_config=model_config,
                    stride=1,
                    downsample=False,
                    groups=groups,
                    shard_layout=shard_layout,
                )
            )

        return layers

    def normalize_imagenet_ttnn(self, x):
        """Normalize input images according to ImageNet standards using TTNN operations.
        Expects input in NHWC format
        """
        # First divide by 255.0 to convert from [0,255] to [0,1]
        x = ttnn.multiply(x, 1.0 / 255.0)

        # For NHWC format: [batch, height, width, channels]
        # Slice along the channel dimension (dim=3)
        x_r = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], 1])  # Red channel
        x_g = ttnn.slice(x, [0, 0, 0, 1], [x.shape[0], x.shape[1], x.shape[2], 2])  # Green channel
        x_b = ttnn.slice(x, [0, 0, 0, 2], [x.shape[0], x.shape[1], x.shape[2], 3])  # Blue channel

        # Normalize each channel: (x - mean) / std
        x_r = ttnn.subtract(x_r, 0.485)
        x_r = ttnn.multiply(x_r, 1.0 / 0.229)

        x_g = ttnn.subtract(x_g, 0.456)
        x_g = ttnn.multiply(x_g, 1.0 / 0.224)

        x_b = ttnn.subtract(x_b, 0.406)
        x_b = ttnn.multiply(x_b, 1.0 / 0.225)

        # Concatenate along channel dimension (dim=3 for NHWC)
        x = ttnn.concat([x_r, x_g, x_b], dim=3)

        return x

    def __call__(self, image_x, lidar_x, velocity, device):
        # Process image input
        logger.info(f"image_encoder_conv1")
        image_x = self.normalize_imagenet_ttnn(image_x)
        image_out, image_shape = self.conv1(device, image_x, image_x.shape)
        # Reshape to spatial dimensions: 80 * 352 = 28160
        # out = ttnn.reshape(out, (1, 80, 352, 32))
        # out = ttnn.permute(out, (0, 3, 1, 2))
        logger.info(f"lidar_encoder_conv1")
        # Process lidar input
        lidar_out, lidar_shape = self.lidar_conv1(device, lidar_x, lidar_x.shape)
        print("..........................................")
        print(lidar_shape)
        print(image_shape)

        logger.info(f"image_encoder_layer1")
        # image_out = ttnn.reshape(image_out, (1, 80, 352, 32))
        image_out = ttnn.reshape(image_out, image_shape)
        # Process layer1 blocks
        for block in self.image_layer1:
            image_out = block(image_out, device)

        logger.info(f"lidar_encoder_layer1")
        lidar_out = ttnn.reshape(lidar_out, lidar_shape)
        # lidar_out = ttnn.reshape(lidar_out, (1, 128, 128, 32))
        for block in self.lidar_layer1:
            lidar_out = block(lidar_out, device)

        logger.info(f"img_avgpool")

        image_h = image_out.shape[1]
        image_w = image_out.shape[2]
        image_c = image_out.shape[3]

        image_features_flat = ttnn.reshape(image_out, (1, 1, image_out.shape[0] * image_h * image_w, image_c))
        image_embd_layer1 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_features_flat,
            batch_size=image_out.shape[0],
            input_h=image_h,
            input_w=image_w,
            channels=image_c,
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )
        logger.info(f"lidar_avgpool")
        lidar_h = lidar_out.shape[1]
        lidar_w = lidar_out.shape[2]
        lidar_c = lidar_out.shape[3]

        lidar_features_flat = ttnn.reshape(lidar_out, (1, 1, lidar_out.shape[0] * lidar_h * lidar_w, lidar_c))

        lidar_embd_layer1 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_features_flat,
            batch_size=lidar_out.shape[0],
            input_h=lidar_h,
            input_w=lidar_w,
            channels=lidar_c,
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
        )
        logger.info(f"Layer1 transformer")

        image_embd_layer1 = ttnn.to_memory_config(image_embd_layer1, ttnn.DRAM_MEMORY_CONFIG)
        image_embd_layer1 = ttnn.to_layout(image_embd_layer1, ttnn.TILE_LAYOUT)

        lidar_embd_layer1 = ttnn.to_memory_config(lidar_embd_layer1, ttnn.DRAM_MEMORY_CONFIG)
        lidar_embd_layer1 = ttnn.to_layout(lidar_embd_layer1, ttnn.TILE_LAYOUT)

        image_features_layer1, lidar_features_layer1 = self.transformer1(
            image_embd_layer1, lidar_embd_layer1, velocity, 72
        )
        image_features_layer1 = ttnn.permute(image_features_layer1, (0, 2, 3, 1))
        lidar_features_layer1 = ttnn.permute(lidar_features_layer1, (0, 2, 3, 1))

        logger.info(f"Layer1 image and lidar interpolation- bilinear")
        logger.info(f"bilinear_image")
        image_features_layer1 = ttnn.to_layout(image_features_layer1, ttnn.ROW_MAJOR_LAYOUT)
        image_features_layer1 = ttnn.to_memory_config(image_features_layer1, ttnn.DRAM_MEMORY_CONFIG)
        image_features_layer1 = ttnn.pad(
            image_features_layer1, padding=((0, 0), (0, 0), (0, 0), (0, 24)), value=0.0  # Pad 24 channels (96 - 72)
        )
        image_features_layer1 = ttnn.upsample(
            image_features_layer1, scale_factor=(8, 8), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # Slice back to original 72 channels
        image_features_layer1 = ttnn.slice(image_features_layer1, [0, 0, 0, 0], [1, 40, 176, 72])
        image_features_layer1 = ttnn.to_layout(image_features_layer1, ttnn.TILE_LAYOUT)

        logger.info(f"bilinear_lidar")
        lidar_features_layer1 = ttnn.to_layout(lidar_features_layer1, ttnn.ROW_MAJOR_LAYOUT)
        lidar_features_layer1 = ttnn.to_memory_config(lidar_features_layer1, ttnn.DRAM_MEMORY_CONFIG)
        lidar_features_layer1 = ttnn.pad(lidar_features_layer1, padding=((0, 0), (0, 0), (0, 0), (0, 24)), value=0.0)
        lidar_features_layer1 = ttnn.upsample(
            lidar_features_layer1, scale_factor=(8, 8), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # Slice back to original 72 channels
        lidar_features_layer1 = ttnn.slice(lidar_features_layer1, [0, 0, 0, 0], [1, 64, 64, 72])
        lidar_features_layer1 = ttnn.to_layout(lidar_features_layer1, ttnn.TILE_LAYOUT)

        logger.info("Image and lidar - add")
        image_features = ttnn.add(image_out, image_features_layer1)
        lidar_features = ttnn.add(lidar_out, lidar_features_layer1)

        logger.info(f"image_encoder_layer2")
        for block in self.image_layer2:
            image_features = block(image_features, device)
        logger.info(f"lidar_encoder_layer2")
        for block in self.lidar_layer2:
            lidar_features = block(lidar_features, device)

        logger.info(f"img2_avgpool")
        image_h = image_features.shape[1]
        image_w = image_features.shape[2]
        image_c = image_features.shape[3]

        image_features_flat = ttnn.reshape(image_features, (1, 1, image_features.shape[0] * image_h * image_w, image_c))
        image_embd_layer2 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_features_flat,
            batch_size=image_features.shape[0],
            input_h=image_h,
            input_w=image_w,
            channels=image_c,
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )
        logger.info(f"lidar2_avgpool")
        lidar_h = lidar_features.shape[1]
        lidar_w = lidar_features.shape[2]
        lidar_c = lidar_features.shape[3]
        lidar_features_flat = ttnn.reshape(lidar_features, (1, 1, lidar_features.shape[0] * lidar_h * lidar_w, lidar_c))
        lidar_embd_layer2 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_features_flat,
            batch_size=lidar_features.shape[0],
            input_h=lidar_h,
            input_w=lidar_w,
            channels=lidar_c,
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
        )

        logger.info(f"layer2 transformer")

        image_embd_layer2 = ttnn.to_memory_config(image_embd_layer2, ttnn.DRAM_MEMORY_CONFIG)
        image_embd_layer2 = ttnn.to_layout(image_embd_layer2, ttnn.TILE_LAYOUT)

        lidar_embd_layer2 = ttnn.to_memory_config(lidar_embd_layer2, ttnn.DRAM_MEMORY_CONFIG)
        lidar_embd_layer2 = ttnn.to_layout(lidar_embd_layer2, ttnn.TILE_LAYOUT)

        image_features_layer2, lidar_features_layer2 = self.transformer2(
            image_embd_layer2, lidar_embd_layer2, velocity, 216
        )
        image_features_layer2 = ttnn.permute(image_features_layer2, (0, 2, 3, 1))
        lidar_features_layer2 = ttnn.permute(lidar_features_layer2, (0, 2, 3, 1))

        logger.info(f"Layer2 image and lidar interpolation- bilinear")
        logger.info(f"bilinear_image")
        image_features_layer2 = ttnn.to_layout(image_features_layer2, ttnn.ROW_MAJOR_LAYOUT)
        image_features_layer2 = ttnn.to_memory_config(image_features_layer2, ttnn.DRAM_MEMORY_CONFIG)
        image_features_layer2 = ttnn.pad(image_features_layer2, padding=((0, 0), (0, 0), (0, 0), (0, 8)), value=0.0)
        image_features_layer2 = ttnn.upsample(
            image_features_layer2, scale_factor=(4, 4), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # Slice back to original 216 channels
        image_features_layer2 = ttnn.slice(image_features_layer2, [0, 0, 0, 0], [1, 20, 88, 216])
        image_features_layer2 = ttnn.to_layout(image_features_layer2, ttnn.TILE_LAYOUT)

        logger.info(f"bilinear_lidar")
        lidar_features_layer2 = ttnn.to_layout(lidar_features_layer2, ttnn.ROW_MAJOR_LAYOUT)
        lidar_features_layer2 = ttnn.to_memory_config(lidar_features_layer2, ttnn.DRAM_MEMORY_CONFIG)
        lidar_features_layer2 = ttnn.pad(lidar_features_layer2, padding=((0, 0), (0, 0), (0, 0), (0, 8)), value=0.0)
        lidar_features_layer2 = ttnn.upsample(
            lidar_features_layer2, scale_factor=(4, 4), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # Slice back to original 216 channels
        lidar_features_layer2 = ttnn.slice(lidar_features_layer2, [0, 0, 0, 0], [1, 32, 32, 216])
        lidar_features_layer2 = ttnn.to_layout(lidar_features_layer2, ttnn.TILE_LAYOUT)

        logger.info("layer2 Image and lidar - add")
        image_features = ttnn.add(image_features, image_features_layer2)
        lidar_features = ttnn.add(lidar_features, lidar_features_layer2)

        logger.info(f"image_encoder_layer3")
        for block in self.image_layer3:
            image_features = block(image_features, device)

        logger.info(f"lidar_encoder_layer3")
        for block in self.lidar_layer3:
            lidar_features = block(lidar_features, device)

        logger.info(f"img3_avgpool")
        image_h = image_features.shape[1]
        image_w = image_features.shape[2]
        image_c = image_features.shape[3]
        image_features_flat = ttnn.reshape(image_features, (1, 1, image_features.shape[0] * image_h * image_w, image_c))
        image_embd_layer3 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_features_flat,
            batch_size=image_features.shape[0],
            input_h=image_h,
            input_w=image_w,
            channels=image_c,
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )
        logger.info(f"lidar3_avgpool")
        lidar_h = lidar_features.shape[1]
        lidar_w = lidar_features.shape[2]
        lidar_c = lidar_features.shape[3]
        lidar_features_flat = ttnn.reshape(lidar_features, (1, 1, lidar_features.shape[0] * lidar_h * lidar_w, lidar_c))
        lidar_embd_layer3 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_features_flat,
            batch_size=lidar_features.shape[0],
            input_h=lidar_h,
            input_w=lidar_w,
            channels=lidar_c,
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
        )

        logger.info(f"layer3 transformer")

        image_embd_layer3 = ttnn.to_memory_config(image_embd_layer3, ttnn.DRAM_MEMORY_CONFIG)
        image_embd_layer3 = ttnn.to_layout(image_embd_layer3, ttnn.TILE_LAYOUT)

        lidar_embd_layer3 = ttnn.to_memory_config(lidar_embd_layer3, ttnn.DRAM_MEMORY_CONFIG)
        lidar_embd_layer3 = ttnn.to_layout(lidar_embd_layer3, ttnn.TILE_LAYOUT)

        image_features_layer3, lidar_features_layer3 = self.transformer3(
            image_embd_layer3, lidar_embd_layer3, velocity, 576
        )
        image_features_layer3 = ttnn.permute(image_features_layer3, (0, 2, 3, 1))
        lidar_features_layer3 = ttnn.permute(lidar_features_layer3, (0, 2, 3, 1))

        logger.info(f"Layer3 image and lidar interpolation- bilinear")
        logger.info(f"bilinear_image")
        image_features_layer3 = ttnn.to_layout(image_features_layer3, ttnn.ROW_MAJOR_LAYOUT)
        image_features_layer3 = ttnn.to_memory_config(image_features_layer3, ttnn.DRAM_MEMORY_CONFIG)
        image_features_layer3 = ttnn.upsample(
            image_features_layer3, scale_factor=(2, 2), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        image_features_layer3 = ttnn.to_layout(image_features_layer3, ttnn.TILE_LAYOUT)

        logger.info(f"bilinear_lidar")
        lidar_features_layer3 = ttnn.to_layout(lidar_features_layer3, ttnn.ROW_MAJOR_LAYOUT)
        lidar_features_layer3 = ttnn.to_memory_config(lidar_features_layer3, ttnn.DRAM_MEMORY_CONFIG)
        lidar_features_layer3 = ttnn.upsample(
            lidar_features_layer3, scale_factor=(2, 2), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        lidar_features_layer3 = ttnn.to_layout(lidar_features_layer3, ttnn.TILE_LAYOUT)

        logger.info("layer3 Image and lidar - add")
        image_features = ttnn.add(image_features, image_features_layer3)
        lidar_features = ttnn.add(lidar_features, lidar_features_layer3)

        logger.info(f"image_encoder_layer4")
        for block in self.image_layer4:
            image_features = block(image_features, device)

        logger.info(f"lidar_encoder_layer4")
        for block in self.lidar_layer4:
            lidar_features = block(lidar_features, device)

        logger.info(f"img4_avgpool")
        image_h = image_features.shape[1]
        image_w = image_features.shape[2]
        image_c = image_features.shape[3]
        image_features_flat = ttnn.reshape(image_features, (1, 1, image_features.shape[0] * image_h * image_w, image_c))
        image_embd_layer4 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_features_flat,
            batch_size=image_features.shape[0],
            input_h=image_h,
            input_w=image_w,
            channels=image_c,
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )
        logger.info(f"lidar4_avgpool")
        lidar_h = lidar_features.shape[1]
        lidar_w = lidar_features.shape[2]
        lidar_c = lidar_features.shape[3]
        lidar_features_flat = ttnn.reshape(lidar_features, (1, 1, lidar_features.shape[0] * lidar_h * lidar_w, lidar_c))
        lidar_embd_layer4 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_features_flat,
            batch_size=lidar_features.shape[0],
            input_h=lidar_h,
            input_w=lidar_w,
            channels=lidar_c,
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
        )

        return image_embd_layer4, lidar_embd_layer4
        logger.info(f"layer4 transformer")

        image_embd_layer4 = ttnn.to_memory_config(image_embd_layer4, ttnn.DRAM_MEMORY_CONFIG)
        image_embd_layer4 = ttnn.to_layout(image_embd_layer4, ttnn.TILE_LAYOUT)

        lidar_embd_layer4 = ttnn.to_memory_config(lidar_embd_layer4, ttnn.DRAM_MEMORY_CONFIG)
        lidar_embd_layer4 = ttnn.to_layout(lidar_embd_layer4, ttnn.TILE_LAYOUT)

        image_features_layer4, lidar_features_layer4 = self.transformer4(
            image_embd_layer4, lidar_embd_layer4, velocity, 1512
        )
        image_features_layer4 = ttnn.permute(image_features_layer4, (0, 2, 3, 1))
        lidar_features_layer4 = ttnn.permute(lidar_features_layer4, (0, 2, 3, 1))

        logger.info(f"Layer4 image and lidar interpolation- bilinear")
        logger.info(f"bilinear_image")
        image_features_layer4 = ttnn.to_layout(image_features_layer4, ttnn.ROW_MAJOR_LAYOUT)
        image_features_layer4 = ttnn.to_memory_config(image_features_layer4, ttnn.DRAM_MEMORY_CONFIG)
        image_features_layer4 = ttnn.pad(image_features_layer4, padding=((0, 0), (0, 0), (0, 0), (0, 24)), value=0.0)
        image_features_layer4 = ttnn.upsample(
            image_features_layer4, scale_factor=(2, 2), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        image_features_layer4 = ttnn.slice(image_features_layer4, [0, 0, 0, 0], [1, 5, 22, 1512])
        image_features_layer4 = ttnn.to_layout(image_features_layer4, ttnn.TILE_LAYOUT)

        logger.info(f"bilinear_lidar")
        lidar_features_layer4 = ttnn.to_layout(lidar_features_layer4, ttnn.ROW_MAJOR_LAYOUT)
        lidar_features_layer4 = ttnn.to_memory_config(lidar_features_layer4, ttnn.DRAM_MEMORY_CONFIG)
        lidar_features_layer4 = ttnn.pad(lidar_features_layer4, padding=((0, 0), (0, 0), (0, 0), (0, 24)), value=0.0)
        lidar_features_layer4 = ttnn.upsample(
            lidar_features_layer4, scale_factor=(2, 2), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        lidar_features_layer4 = ttnn.slice(lidar_features_layer4, [0, 0, 0, 0], [1, 8, 8, 1512])
        lidar_features_layer4 = ttnn.to_layout(lidar_features_layer4, ttnn.TILE_LAYOUT)

        logger.info("layer4 Image and lidar - add")
        image_features = ttnn.add(image_features, image_features_layer4)
        lidar_features = ttnn.add(lidar_features, lidar_features_layer4)
        return image_features, lidar_features
