# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import time

import requests
import torch
from loguru import logger
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ttnn.model_preprocessing import ParameterDict, ParameterList, preprocess_model_parameters

import ttnn
from models.demos.segformer.reference.segformer_for_semantic_segmentation import (
    SegformerForSemanticSegmentationReference,
)
from models.demos.segformer.tt.ttnn_segformer_for_semantic_segmentation import TtSegformerForSemanticSegmentation
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.integration_tests.segformer.test_segformer_decode_head import (
    create_custom_preprocessor as create_custom_preprocessor_decode_head,
)
from tests.ttnn.integration_tests.segformer.test_segformer_model import (
    create_custom_preprocessor as create_custom_preprocessor_model,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerForSemanticSegmentationReference):
            parameters["segformer"] = {}
            segformer_preprocess = create_custom_preprocessor_model(device)
            parameters["segformer"] = segformer_preprocess(model.segformer, None, None)
            parameters["decode_head"] = {}
            deocde_preprocess = create_custom_preprocessor_decode_head(device)
            parameters["decode_head"] = deocde_preprocess(model.decode_head, None, None)

        return parameters

    return custom_preprocessor


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["sr", "proj", "dwconv", "linear_fuse", "classifier"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


def load_segformer_torch_model(device, model_location_generator=None):
    torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config = torch_model.config
    reference_model = SegformerForSemanticSegmentationReference(config=config)
    state_dict = torch_model.state_dict()
    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )
    parameters = move_to_device(parameters, device)

    for i in range(4):
        parameters["decode_head"]["linear_c"][i]["proj"]["weight"] = ttnn.to_device(
            parameters["decode_head"]["linear_c"][i]["proj"]["weight"], device=device
        )
        parameters["decode_head"]["linear_c"][i]["proj"]["bias"] = ttnn.to_device(
            parameters["decode_head"]["linear_c"][i]["proj"]["bias"], device=device
        )

    return reference_model, config, parameters


class SegformerTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_location_generator = model_location_generator
        reference_model, config, self.parameters = load_segformer_torch_model(device)
        self.ttnn_segformer_model = TtSegformerForSemanticSegmentation(config, self.parameters)

        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        self.inputs = processor(images=image, return_tensors="pt")
        self.torch_output_tensor = reference_model(self.inputs.pixel_values)
        input_pixels_permuted = torch.permute(self.inputs.pixel_values, (0, 2, 3, 1))
        self.input_tensor = ttnn.from_torch(input_pixels_permuted, ttnn.bfloat16)

    def run(self):
        self.output_tensor = self.ttnn_segformer_model(
            self.device,
            self.input_tensor,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            parameters=self.parameters,
        )

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        num_devices = device.get_num_devices()

        torch_input_tensor = self.inputs.pixel_values
        n, c, h, w = torch_input_tensor.shape
        padded_c = 16

        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_h = (n * w * h + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})

        shard_spec_input = ttnn.ShardSpec(shard_grid, (shard_h, c), ttnn.ShardOrientation.ROW_MAJOR)
        shard_spec_padded = ttnn.ShardSpec(shard_grid, (shard_h, padded_c), ttnn.ShardOrientation.ROW_MAJOR)

        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec_input
        )
        padded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec_padded
        )

        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        return tt_inputs_host, [input_mem_config, padded_mem_config], [[n, h, w, c], [n, h, w, padded_c]]

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_configs, _ = self.setup_l1_sharded_input(device)
        input_mem_config = input_mem_configs[0]
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
                3,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor.logits if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        h = w = int(math.sqrt(output_tensor.shape[-1]))
        final_output_tensor = torch.reshape(output_tensor, (output_tensor.shape[0], output_tensor.shape[1], h, w))
        valid_pcc = 0.985
        self.pcc_passed, self.pcc_message = assert_with_pcc(
            self.torch_output_tensor.logits, final_output_tensor, pcc=valid_pcc
        )

        logger.info(f"Segformer , PCC={self.pcc_message}")

    def dealloc_input(self):
        ttnn.deallocate(self.input_tensor)

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor.logits)


class SegformerBare:
    def __init__(self, device, batch_size, act_dtype, weight_dtype, model_location_generator=None):
        self.device = device

        self.test_infra = SegformerTestInfra(device, batch_size, act_dtype, weight_dtype, model_location_generator=None)

        self.tt_inputs_host, input_mem_configs, input_shapes = self.test_infra.setup_l1_sharded_input(device)

        self.input_mem_config = input_mem_configs[0]
        self.padded_mem_config = input_mem_configs[1]

        self.input_shape = input_shapes[0]
        self.padded_shape = input_shapes[1]

    def copy_input_to_device(self):
        tt_inputs_l1 = self.tt_inputs_host.to(self.device, self.input_mem_config)
        self.test_infra.input_tensor = ttnn.pad(
            tt_inputs_l1, self.padded_shape, [0, 0, 0, 0], 0, memory_config=self.padded_mem_config
        )

    def compile(self):
        start_jit = time.time()
        self.copy_input_to_device()
        self.test_infra.run()
        end_jit = time.time()
        self.test_infra.validate()
        self.test_infra.dealloc_output()

        self.jit_time = end_jit - start_jit

    def cache(self):
        self.copy_input_to_device()
        self.test_infra.run()
        self.test_infra.validate()
        self.test_infra.dealloc_output()

    def optimized_inference(self):
        start = time.time()
        self.copy_input_to_device()
        self.test_infra.run()
        end = time.time()
        self.test_infra.validate()
        self.test_infra.dealloc_output()

        self.inference_time = end - start


class SegformerTrace2CQ:
    def __init__(self, device, batch_size, act_dtype, weight_dtype, model_location_generator=None):
        self.device = device

        self.test_infra = SegformerTestInfra(device, batch_size, act_dtype, weight_dtype, model_location_generator=None)

        self.tt_inputs_host, input_mem_configs, input_shapes = self.test_infra.setup_l1_sharded_input(device)

        self.input_mem_config = input_mem_configs[0]
        self.padded_mem_config = input_mem_configs[1]

        self.input_shape = input_shapes[0]
        self.padded_shape = input_shapes[1]

        self.tt_inputs_unpadded = ttnn.allocate_tensor_on_device(
            self.input_shape, self.tt_inputs_host.dtype, self.tt_inputs_host.layout, device, self.input_mem_config
        )

        self.tt_outputs_host = []

    def copy_input_to_device(self, input_consumed):
        ttnn.wait_for_event(1, input_consumed)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs_unpadded, cq_id=1)
        input_copied = ttnn.record_event(self.device, 1)

        ttnn.wait_for_event(0, input_copied)
        self.test_infra.input_tensor = ttnn.pad(
            self.tt_inputs_unpadded,
            self.padded_shape,
            [0, 0, 0, 0],
            0,
            memory_config=self.padded_mem_config,
            queue_id=0,
        )
        input_consumed = ttnn.record_event(self.device, 0)
        return input_consumed

    def copy_output_to_host(self, output_consumed):
        ttnn.wait_for_event(0, output_consumed)
        tt_output_buffer = ttnn.to_memory_config(self.test_infra.output_tensor.logits, ttnn.L1_MEMORY_CONFIG)
        output_produced = ttnn.record_event(self.device, 0)

        ttnn.wait_for_event(1, output_produced)
        self.tt_outputs_host.append(tt_output_buffer.cpu(blocking=False, cq_id=1))
        ttnn.deallocate(tt_output_buffer)

        output_consumed = ttnn.record_event(self.device, 1)
        return output_consumed

    def validate_outputs(self):
        for iter in range(0, len(self.tt_outputs_host)):
            self.test_infra.validate(self.tt_outputs_host[iter])

    def compile(self):
        jit_start = time.time()

        # write input to L1 and pad
        can_copy = ttnn.record_event(self.device, 0)
        copy_done = self.copy_input_to_device(can_copy)

        # run model
        ttnn.wait_for_event(0, copy_done)
        self.test_infra.run()
        can_read = ttnn.record_event(self.device, 0)

        # read output
        read_done = self.copy_output_to_host(can_read)

        jit_end = time.time()

        self.jit_time = jit_end - jit_start

        self.test_infra.dealloc_input()
        self.test_infra.dealloc_output()

        return read_done

    def trace_capture(self, compile_finished):
        copy_done = self.copy_input_to_device(compile_finished)

        ttnn.wait_for_event(0, copy_done)
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.test_infra.run()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        can_read = ttnn.record_event(self.device, 0)

        read_done = self.copy_output_to_host(can_read)

        self.test_infra.dealloc_input()

        return read_done

    def trace_execute(self, capture_finished, num_iterations=10):
        start = time.time()

        # copy of first input
        copy_done = self.copy_input_to_device(capture_finished)

        # reads from previous steps should have finished
        read_done = ttnn.record_event(self.device, 0)

        for iter in range(0, num_iterations - 1):
            # start executing trace after data has been read
            ttnn.wait_for_event(0, read_done)
            ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

            # deallocate old and copy new input
            self.test_infra.dealloc_input()
            copy_done = self.copy_input_to_device(copy_done)

            # read output
            read_done = self.copy_output_to_host(read_done)

        # last iteration without an extra input copy
        ttnn.wait_for_event(0, read_done)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

        # read output
        self.copy_output_to_host(read_done)

        ttnn.synchronize_device(self.device)

        end = time.time()

        self.inference_time = (end - start) / num_iterations

    def trace_release(self, tid):
        if tid is not None:
            ttnn.release_trace(self.device, tid)
