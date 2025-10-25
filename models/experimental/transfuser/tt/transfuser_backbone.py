# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from models.experimental.transfuser.tt.utils import TTConv2D
from loguru import logger
from models.experimental.transfuser.tt.gpt import TTGpt
from models.experimental.transfuser.tt.topdown import TtTopDown
from models.experimental.transfuser.tt.stages import Ttstages


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
        self.image_layer1 = Ttstages._make_layer(
            parameters=parameters.image_encoder.features.layer1,
            planes=72,
            blocks=2,  # no of bottlenecks
            stride=2,
            groups=3,  # conv2
            model_config=model_config,
            stage_name="layer1",
        )

        self.lidar_layer1 = Ttstages._make_layer(
            parameters=parameters.lidar_encoder._model.layer1,
            planes=72,
            blocks=2,
            stride=2,
            groups=3,
            model_config=model_config,
            stage_name="layer1",
        )

        # Layer2 for both encoders
        self.image_layer2 = Ttstages._make_layer(
            parameters=parameters.image_encoder.features.layer2,
            planes=216,
            blocks=5,
            stride=2,
            groups=9,  # conv2
            model_config=model_config,
            stage_name="layer2",
        )

        self.lidar_layer2 = Ttstages._make_layer(
            parameters=parameters.lidar_encoder._model.layer2,
            planes=216,
            blocks=5,
            stride=2,
            groups=9,
            model_config=model_config,
            stage_name="layer2",
        )

        # Layer3 for both encoders
        self.image_layer3 = Ttstages._make_layer(
            parameters=parameters.image_encoder.features.layer3,
            planes=576,
            blocks=13,
            stride=2,
            groups=24,  # conv2
            model_config=model_config,
            stage_name="layer3",
        )

        self.lidar_layer3 = Ttstages._make_layer(
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
        self.image_layer4 = Ttstages._make_layer(
            parameters=parameters.image_encoder.features.layer4,
            planes=1512,
            blocks=1,
            stride=2,
            groups=63,  # conv2
            model_config=model_config,
            stage_name="layer4",
        )

        self.lidar_layer4 = Ttstages._make_layer(
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
            compute_kernel_config=compute_kernel_config,
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
            compute_kernel_config=compute_kernel_config,
        )

        if self.config.perception_output_features != 1512:
            self.change_channel_conv_image = TTConv2D(
                kernel_size=1,
                stride=1,
                padding=0,
                parameters=parameters.change_channel_conv_image,
                kernel_fidelity=model_config,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            self.change_channel_conv_lidar = TTConv2D(
                kernel_size=1,
                stride=1,
                padding=0,
                parameters=parameters.change_channel_conv_lidar,
                kernel_fidelity=model_config,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        self.top_down = TtTopDown(
            device=device,
            parameters=parameters,
            perception_output_features=config.perception_output_features,
            bev_features_channels=config.bev_features_chanels,
            bev_upsample_factor=config.bev_upsample_factor,
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
        # Process image input
        image_x = self.normalize_imagenet_ttnn(image_x)
        logger.info(f"image_encoder_conv1")
        image_out, image_shape = self.conv1(device, image_x, image_x.shape)
        logger.info(f"lidar_encoder_conv1")
        # Process lidar input
        lidar_out, lidar_shape = self.lidar_conv1(device, lidar_x, lidar_x.shape)
        logger.info(f"image_encoder_layer1")
        # Process layer1 blocks
        for block in self.image_layer1:
            image_out, image_shape = block(image_out, device, image_shape)
        logger.info(f"lidar_encoder_layer1")
        for block in self.lidar_layer1:
            lidar_out, lidar_shape = block(lidar_out, device, lidar_shape)
        logger.info(f"img_avgpool")
        image_embd_layer1 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_out,
            batch_size=image_shape[0],
            input_h=image_shape[1],
            input_w=image_shape[2],
            channels=image_shape[3],
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )
        logger.info(f"lidar_avgpool")
        lidar_embd_layer1 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_out,
            batch_size=lidar_shape[0],
            input_h=lidar_shape[1],
            input_w=lidar_shape[2],
            channels=lidar_shape[3],
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
        )
        logger.info(f"Layer1 transformer")
        image_embd_layer1 = ttnn.sharded_to_interleaved(image_embd_layer1, memory_config=ttnn.L1_MEMORY_CONFIG)
        lidar_embd_layer1 = ttnn.sharded_to_interleaved(lidar_embd_layer1, memory_config=ttnn.L1_MEMORY_CONFIG)

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
        image_out = ttnn.reshape(image_out, image_features_layer1.shape)
        lidar_out = ttnn.reshape(lidar_out, lidar_features_layer1.shape)
        image_features = ttnn.add(image_out, image_features_layer1)
        lidar_features = ttnn.add(lidar_out, lidar_features_layer1)
        image_shape = image_features.shape
        lidar_shape = lidar_features.shape
        logger.info(f"image_encoder_layer2")
        for block in self.image_layer2:
            image_features, image_shape = block(image_features, device, image_shape)
        logger.info(f"lidar_encoder_layer2")
        for block in self.lidar_layer2:
            lidar_features, lidar_shape = block(lidar_features, device, lidar_shape)
        logger.info(f"img2_avgpool")
        image_embd_layer2 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_features,
            batch_size=image_shape[0],
            input_h=image_shape[1],
            input_w=image_shape[2],
            channels=image_shape[3],
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )

        logger.info(f"lidar2_avgpool")
        lidar_embd_layer2 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_features,
            batch_size=lidar_shape[0],
            input_h=lidar_shape[1],
            input_w=lidar_shape[2],
            channels=lidar_shape[3],
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
        )

        logger.info(f"layer2 transformer")
        image_embd_layer2 = ttnn.sharded_to_interleaved(image_embd_layer2, memory_config=ttnn.L1_MEMORY_CONFIG)
        lidar_embd_layer2 = ttnn.sharded_to_interleaved(lidar_embd_layer2, memory_config=ttnn.L1_MEMORY_CONFIG)
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
        image_features = ttnn.reshape(image_features, image_features_layer2.shape)
        lidar_features = ttnn.reshape(lidar_features, lidar_features_layer2.shape)
        image_features = ttnn.add(image_features, image_features_layer2)
        lidar_features = ttnn.add(lidar_features, lidar_features_layer2)
        image_shape = image_features.shape
        lidar_shape = lidar_features.shape
        logger.info(f"image_encoder_layer3")
        for block in self.image_layer3:
            image_features, image_shape = block(image_features, device, image_shape)

        logger.info(f"lidar_encoder_layer3")
        for block in self.lidar_layer3:
            lidar_features, lidar_shape = block(lidar_features, device, lidar_shape)

        logger.info(f"img3_avgpool")
        image_embd_layer3 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_features,
            batch_size=image_shape[0],
            input_h=image_shape[1],
            input_w=image_shape[2],
            channels=image_shape[3],
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )

        logger.info(f"lidar3_avgpool")
        lidar_embd_layer3 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_features,
            batch_size=lidar_shape[0],
            input_h=lidar_shape[1],
            input_w=lidar_shape[2],
            channels=lidar_shape[3],
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
        )

        logger.info(f"layer3 transformer")
        image_embd_layer3 = ttnn.sharded_to_interleaved(image_embd_layer3, memory_config=ttnn.L1_MEMORY_CONFIG)
        lidar_embd_layer3 = ttnn.sharded_to_interleaved(lidar_embd_layer3, memory_config=ttnn.L1_MEMORY_CONFIG)

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
        image_features = ttnn.reshape(image_features, image_features_layer3.shape)
        lidar_features = ttnn.reshape(lidar_features, lidar_features_layer3.shape)
        image_features = ttnn.add(image_features, image_features_layer3)
        lidar_features = ttnn.add(lidar_features, lidar_features_layer3)
        image_shape = image_features.shape
        lidar_shape = lidar_features.shape
        logger.info(f"image_encoder_layer4")
        for block in self.image_layer4:
            image_features, image_shape = block(image_features, device, image_shape)

        logger.info(f"lidar_encoder_layer4")
        for block in self.lidar_layer4:
            lidar_features, lidar_shape = block(lidar_features, device, lidar_shape)

        logger.info(f"img4_avgpool")
        image_embd_layer4 = ttnn.adaptive_avg_pool2d(
            input_tensor=image_features,
            batch_size=image_shape[0],
            input_h=image_shape[1],
            input_w=image_shape[2],
            channels=image_shape[3],
            output_size=[self.config.img_vert_anchors, self.config.img_horz_anchors],
        )

        logger.info(f"lidar4_avgpool")
        lidar_embd_layer4 = ttnn.adaptive_avg_pool2d(
            input_tensor=lidar_features,
            batch_size=lidar_shape[0],
            input_h=lidar_shape[1],
            input_w=lidar_shape[2],
            channels=lidar_shape[3],
            output_size=[self.config.lidar_vert_anchors, self.config.lidar_horz_anchors],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        image_embd_layer4 = ttnn.sharded_to_interleaved(image_embd_layer4, memory_config=ttnn.L1_MEMORY_CONFIG)
        lidar_embd_layer4 = ttnn.sharded_to_interleaved(lidar_embd_layer4, memory_config=ttnn.L1_MEMORY_CONFIG)

        logger.info(f"layer4 transformer")
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
            image_features_layer4, scale_factor=(1, 1), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        image_features_layer4 = ttnn.slice(image_features_layer4, [0, 0, 0, 0], [1, 5, 22, 1512])
        image_features_layer4 = ttnn.to_layout(image_features_layer4, ttnn.TILE_LAYOUT)

        logger.info(f"bilinear_lidar")
        lidar_features_layer4 = ttnn.to_layout(lidar_features_layer4, ttnn.ROW_MAJOR_LAYOUT)
        lidar_features_layer4 = ttnn.to_memory_config(lidar_features_layer4, ttnn.DRAM_MEMORY_CONFIG)
        lidar_features_layer4 = ttnn.pad(lidar_features_layer4, padding=((0, 0), (0, 0), (0, 0), (0, 24)), value=0.0)
        lidar_features_layer4 = ttnn.upsample(
            lidar_features_layer4, scale_factor=(1, 1), mode="bilinear", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        lidar_features_layer4 = ttnn.slice(lidar_features_layer4, [0, 0, 0, 0], [1, 8, 8, 1512])
        lidar_features_layer4 = ttnn.to_layout(lidar_features_layer4, ttnn.TILE_LAYOUT)

        logger.info("layer4 Image and lidar - add")
        image_features = ttnn.reshape(image_features, image_features_layer4.shape)
        lidar_features = ttnn.reshape(lidar_features, lidar_features_layer4.shape)
        image_features = ttnn.add(image_features, image_features_layer4)
        lidar_features = ttnn.add(lidar_features, lidar_features_layer4)

        logger.info("Downsamples channels to 512")
        # Downsamples channels to 512
        image_features, shape_ = self.change_channel_conv_image(device, image_features, image_features.shape)
        lidar_features, shape_l = self.change_channel_conv_lidar(device, lidar_features, lidar_features.shape)
        x4 = lidar_features  # Save for FPN
        image_features_grid = image_features  # For auxiliary information
        # Reshape before feeding to FPN
        image_features_grid = ttnn.reshape(image_features_grid, shape_)
        x4 = ttnn.reshape(x4, shape_l)

        logger.info("Global average pooling")
        # Global average pooling
        image_features = ttnn.global_avg_pool2d(image_features)
        lidar_features = ttnn.global_avg_pool2d(lidar_features)

        logger.info("Flatten")
        # Flatten
        image_features = ttnn.reshape(image_features, (1, self.config.perception_output_features))
        lidar_features = ttnn.reshape(lidar_features, (1, self.config.perception_output_features))

        logger.info("fused_features")
        fused_features = ttnn.add(image_features, lidar_features)

        logger.info("FPN top-down")
        p2, p3, p4, p5 = self.top_down(x4)
        features = (p2, p3, p4, p5)

        return features, image_features_grid, fused_features
