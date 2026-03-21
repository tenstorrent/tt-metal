# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image

# from torchvision.models import MobileNet_V3_Small_Weights
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_cnn.tt.pipeline import get_memory_config_for_persistent_dram_tensor
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor
from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel
from models.experimental.atss_swin_l_dyhead.reference.model import build_atss_model


class ATSSPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
        resolution=(640, 640),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        input_path=None,
    ):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.resolution = resolution
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.model_location_generator = model_location_generator
        self.torch_input_tensor = torch_input_tensor

        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer
        self.real_input_path = input_path

        self.torch_model = build_atss_model()

        # Create input tensor
        if self.real_input_path and os.path.exists(self.real_input_path):
            img = Image.open(self.real_input_path).convert("RGB")
            preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                ]
            )
            input_tensor = preprocess(img)
            self.torch_input_tensor = input_tensor.unsqueeze(0)

        else:
            self.torch_input_tensor = torch.randn(
                (self.batch_size, 3, self.resolution[0], self.resolution[1]), dtype=torch.float32
            )

        # Preprocess model parameters
        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        ATSS_CKPT_PATH = "models/experimental/atss_swin_l_dyhead/weights/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth"

        self.ttnn_model = TtATSSModel.from_checkpoint(
            ATSS_CKPT_PATH,
            device=self.device,
            input_h=resolution[0],
            input_w=resolution[1],
            inputs_mesh_mapper=self.inputs_mesh_mapper,
            output_mesh_composer=self.outputs_mesh_composer,
        )

        self.torch_output = self.torch_model.forward(self.torch_input_tensor)
        self.torch_input_tensor = self.torch_input_tensor.permute(0, 2, 3, 1)

    def setup_dram_interleaved_input(self, torch_input_tensor=None, mesh_mapper=None):
        # Inputs to MobileNetV3 need to be in ttnn.DRAM_MEMORY_CONFIG for supporting DRAM sliced Conv2d
        mesh_mapper = self.inputs_mesh_mapper if mesh_mapper is None else mesh_mapper
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        return tt_inputs_host, ttnn.DRAM_MEMORY_CONFIG

    def setup_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, pad_channels=16):
        # Inputs to MobileNetV3 need to be in ttnn.L1_MEMORY_CONFIG for supporting L1 sharded input
        mesh_mapper = self.inputs_mesh_mapper if mesh_mapper is None else mesh_mapper
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        original_channels = torch_input_tensor.shape[-1]
        if pad_channels and original_channels < pad_channels:
            torch_input_tensor = torch.nn.functional.pad(
                torch_input_tensor, (0, pad_channels - original_channels), value=0
            )

        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        # ttnn tensor shape reflects per-device shape when using ShardTensorToMesh
        batch = tt_inputs_host.shape[0]
        height = tt_inputs_host.shape[1]
        width = tt_inputs_host.shape[2]
        channels = tt_inputs_host.shape[3]

        tt_inputs_host = ttnn.reshape(tt_inputs_host, (1, 1, batch * height * width, channels))

        dram_input_mem_config = get_memory_config_for_persistent_dram_tensor(
            tt_inputs_host.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
        )

        input_l1_core_grid = ttnn.CoreGrid(x=8, y=8)
        height_dim = tt_inputs_host.shape[-2]

        if height_dim % input_l1_core_grid.num_cores != 0:
            num_cores = input_l1_core_grid.num_cores
            while height_dim % num_cores != 0 and num_cores > 1:
                num_cores -= 1
            y = min(8, num_cores)
            while num_cores % y != 0:
                y -= 1
            x = num_cores // y
            input_l1_core_grid = ttnn.CoreGrid(x=x, y=y)

        l1_input_mem_config = ttnn.create_sharded_memory_config(
            shape=(height_dim // input_l1_core_grid.num_cores, channels),
            core_grid=input_l1_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return tt_inputs_host, dram_input_mem_config, l1_input_mem_config, channels

    def run(self):
        self.tt_output = self.ttnn_model.forward(self.input_tensor)

    def validate(self, tt_output=None):
        logger.info("Starting Full Validation against reference...")
        # Validate output tensor
        tt_output = self.tt_output if tt_output is None else tt_output

        self._PCC_THRESH = 0.98
        self.pcc_passed = self.pcc_message = []

        # Unpack TTNN outputs from the last run
        ttnn_cls, ttnn_reg, ttnn_cent = self.tt_output

        # Get Reference outputs
        ref_cls, ref_reg, ref_cent = self.torch_output

        ttnn_cls_rp, ttnn_reg_rp, ttnn_cent_rp = [], [], []

        # 1. Validate Head Outputs (5 levels of FPN)
        for i in range(5):
            # Classification
            N, C, H, W = ref_cls[i].shape
            actual_cls = ttnn_cls[i].reshape(N, H, W, C).permute(0, 3, 1, 2)
            ttnn_cls_rp.append(actual_cls)
            passing_cls, pcc_cls = comp_pcc(ref_cls[i], actual_cls, 0.96)

            # Regression
            N, C, H, W = ref_reg[i].shape
            actual_reg = ttnn_reg[i].reshape(N, H, W, C).permute(0, 3, 1, 2)
            ttnn_reg_rp.append(actual_reg)
            passing_reg, pcc_reg = comp_pcc(ref_reg[i], actual_reg, 0.96)

            # Centerness
            N, C, H, W = ref_cent[i].shape
            actual_cent = ttnn_cent[i].reshape(N, H, W, C).permute(0, 3, 1, 2)
            ttnn_cent_rp.append(actual_cent)
            passing_cent, pcc_cent = comp_pcc(ref_cent[i], actual_cent, 0.96)

            logger.info(f"Level {i} | Cls PCC: {pcc_cls:.6f} | Reg PCC: {pcc_reg:.6f} | Cent PCC: {pcc_cent:.6f}")
            assert passing_cls and passing_reg and passing_cent, f"Level {i} failed PCC validation"

        # Post-processing Comparison
        from models.experimental.atss_swin_l_dyhead.reference.postprocess import atss_postprocess

        ref_results = atss_postprocess(ref_cls, ref_reg, ref_cent, img_shape=self.resolution, score_thr=0.05)
        ttnn_results = atss_postprocess(
            ttnn_cls_rp, ttnn_reg_rp, ttnn_cent_rp, img_shape=self.resolution, score_thr=0.05
        )

        ref_n = ref_results["bboxes"].shape[0]
        ttnn_n = ttnn_results["bboxes"].shape[0]
        logger.info(f"Detections -> Ref: {ref_n}, TTNN: {ttnn_n}")

        if ref_n > 0 and ttnn_n > 0:
            n_common = min(ref_n, ttnn_n)
            bbox_pass, bbox_pcc = comp_pcc(ref_results["bboxes"][:n_common], ttnn_results["bboxes"][:n_common], 0.90)
            score_pass, score_pcc = comp_pcc(ref_results["scores"][:n_common], ttnn_results["scores"][:n_common], 0.90)
            logger.info(f"BBox PCC (top {n_common}): {bbox_pcc:.6f} | Score PCC: {score_pcc:.6f}")
            assert bbox_pass and score_pass, "Post-processing PCC below threshold"

    def dealloc_output(self):
        ttnn.deallocate(self.tt_output)
