# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import torch.nn.functional as F

import os
import ttnn
from models.experimental.functional_yolov9c.demo.common import (
    YOLOV9_BOXES_PCC,
    YOLOV9_CONFS_PCC,
    get_model_result,
    load_torch_model,
)

# from models.experimental.functional_yolov9c.tt.model_postprocessing import gen_yolov4_boxes_confs, get_region_boxes
from models.experimental.yolo_evaluation.yolo_evaluation_utils import LoadImages, preprocess, postprocess
from models.experimental.functional_yolov9c.demo.demo_utils import load_coco_class_names

from models.experimental.functional_yolov9c.tt.model_preprocessing import (
    create_yolov9c_model_parameters,
)
from models.experimental.functional_yolov9c.tt.ttnn_yolov9c import YoloV9
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


class YOLOv9PerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
        resolution=(640, 640),
    ):
        torch.manual_seed(0)
        self.resolution = resolution
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_location_generator = model_location_generator

        self.torch_model = load_torch_model()

        save_dir = f"models/experimental/functional_yolov9c/demo/runs"
        source = f"models/sample_data/huggingface_cat_image.jpg"
        model_type = f"torch_model"
        dataset = LoadImages(path=source)
        model_save_dir = os.path.join(save_dir, model_type)
        os.makedirs(model_save_dir, exist_ok=True)
        names = load_coco_class_names()

        for batch in dataset:
            paths, im0s, s = batch
            self.torch_input_tensor = preprocess(im0s, res=self.resolution[0])
            self.torch_output_tensor = self.torch_model(self.torch_input_tensor)

        torch_results = postprocess(self.torch_output_tensor, self.torch_input_tensor, im0s, batch, names)[0]
        self.torch_ref_boxes, self.torch_ref_confs = torch_results["boxes"]["xyxy"], torch_results["boxes"]["conf"]

        self.parameters = create_yolov9c_model_parameters(self.torch_model, self.torch_input_tensor, device=self.device)

        self.ttnn_yolov9c_model = YoloV9(self.device, self.parameters)

    def run(self):
        self.output_tensor = self.ttnn_yolov9c_model(self.input_tensor)

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape

        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        torch_input_tensor = F.pad(torch_input_tensor, (0, 29))
        input_mem_config = ttnn.create_sharded_memory_config(
            [6400, 32],
            core_grid=device.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self._setup_l1_sharded_input(device)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x * dram_grid_size.y),
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        result_boxes, result_confs = get_model_result(output_tensor, self.resolution)

        self.pcc_passed, self.pcc_message = assert_with_pcc(self.ref_boxes, result_boxes, pcc=YOLOV9_BOXES_PCC)
        logger.info(
            f"Yolov9c - Bboxes. batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

        self.pcc_passed, self.pcc_message = assert_with_pcc(self.ref_confs, result_confs, pcc=YOLOV9_CONFS_PCC)
        logger.info(
            f"Yolov9c - Confs. batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor[0])
        ttnn.deallocate(self.output_tensor[1])
