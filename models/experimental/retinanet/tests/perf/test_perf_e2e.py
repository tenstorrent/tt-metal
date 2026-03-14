# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn
from loguru import logger
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from ttnn.model_preprocessing import preprocess_model_parameters

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.retinanet.tt.tt_retinanet import TTRetinaNet
from models.experimental.retinanet.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.retinanet.tests.pcc.test_resnet50_fpn import infer_ttnn_module_args as infer_module_args
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)
from ttnn.model_preprocessing import infer_ttnn_module_args


def get_mesh_mappers(device):
    """Helper function to get mesh mappers based on number of devices"""
    if device.get_num_devices() != 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = None
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


def create_retinanet_pipeline_model(ttnn_model, device, input_height, input_width, in_channels, padded_channels):
    """Create a wrapper function for the RetinaNet model to use with pipeline"""

    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"
        unfolded_input = ttnn.reshape(l1_input_tensor, (1, input_height, input_width, padded_channels))
        unfolded_input = ttnn.to_memory_config(unfolded_input, ttnn.L1_MEMORY_CONFIG)
        if padded_channels > in_channels:
            unfolded_input = unfolded_input[:, :, :, :in_channels]
        output_dict = ttnn_model(unfolded_input, device)
        return [
            output_dict["0"],
            output_dict["1"],
            output_dict["2"],
            output_dict["p6"],
            output_dict["p7"],
            output_dict["regression"],
            output_dict["classification"],
        ]

    return run


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 8000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, expected_compile_time, expected_throughput_fps",
    [(1, 3, 512, 512, 30.0, 40)],
)
def test_retinanet_perf_e2e(
    num_iterations,
    batch_size,
    in_channels,
    input_height,
    input_width,
    expected_compile_time,
    expected_throughput_fps,
    device,
    model_location_generator,
):
    """E2E performance test for RetinaNet with pipeline config (use_trace=False, num_command_queues=2)

    Note: Input is padded from 3 to 16 channels for 16B alignment, then sliced back to 3 channels
    before passing to the model. use_trace=False because RetinaNet uses fallback_ops.group_norm.
    """

    torch.manual_seed(42)

    inputs_mesh_mapper, weights_mesh_mapper, _ = get_mesh_mappers(device)

    # Load full RetinaNet model
    retinanet = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    retinanet.eval()

    # Torch input preprocessing
    preprocess = transforms.Compose(
        [
            transforms.CenterCrop((input_height, input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open("models/experimental/retinanet/resources/dog_800x800.jpg").convert("RGB")
    torch_input_tensor = preprocess(img).unsqueeze(0)

    # Get backbone features for reference
    backbone = retinanet.backbone.body
    fpn_model = retinanet.backbone.fpn
    torch_fpn_input_tensor = backbone(torch_input_tensor)

    # Get model args
    conv_args = infer_ttnn_module_args(
        model=retinanet.backbone,
        run_model=lambda model: retinanet.backbone(torch_input_tensor),
        device=device,
    )
    fpn_args = infer_module_args(
        model=fpn_model, run_model=lambda model: fpn_model(torch_fpn_input_tensor), device=device
    )

    model_args = {}
    model_args["stem"] = {}
    model_args["stem"]["conv1"] = conv_args["body"]["conv1"]
    model_args["stem"]["maxpool"] = conv_args["body"]["maxpool"]

    model_args["fpn"] = {}
    model_args["fpn"]["inner_blocks"] = {}
    model_args["fpn"]["inner_blocks"][0] = fpn_args["fpn"]["fpn"]["inner_blocks"][0]["fpn"]["inner_blocks"][0][0]
    model_args["fpn"]["inner_blocks"][1] = fpn_args["fpn"]["fpn"]["inner_blocks"][1]["fpn"]["inner_blocks"][1][0]
    model_args["fpn"]["inner_blocks"][2] = fpn_args["fpn"]["fpn"]["inner_blocks"][2]["fpn"]["inner_blocks"][2][0]

    model_args["fpn"]["layer_blocks"] = {}
    model_args["fpn"]["layer_blocks"][0] = fpn_args["fpn"]["fpn"]["layer_blocks"][0]["fpn"]["layer_blocks"][0][0]
    model_args["fpn"]["layer_blocks"][1] = fpn_args["fpn"]["fpn"]["layer_blocks"][1]["fpn"]["layer_blocks"][1][0]
    model_args["fpn"]["layer_blocks"][2] = fpn_args["fpn"]["fpn"]["layer_blocks"][2]["fpn"]["layer_blocks"][2][0]

    model_args["fpn"]["extra_blocks"] = {}
    model_args["fpn"]["extra_blocks"]["p6"] = fpn_args["fpn"]["fpn"]["extra_blocks"]["fpn"]["extra_blocks"]["p6"]
    model_args["fpn"]["extra_blocks"]["p7"] = fpn_args["fpn"]["fpn"]["extra_blocks"]["fpn"]["extra_blocks"]["p7"]

    model_args["layer1"] = conv_args["body"]["layer1"]
    model_args["layer2"] = conv_args["body"]["layer2"]
    model_args["layer3"] = conv_args["body"]["layer3"]
    model_args["layer4"] = conv_args["body"]["layer4"]

    # Model configuration
    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    }

    # Preprocess parameters for FULL RetinaNet model
    parameters = preprocess_model_parameters(
        initialize_model=lambda: retinanet,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )

    torch_input_nhwc = torch_input_tensor.permute(0, 2, 3, 1)
    padded_channels = 16
    torch_input_padded = torch.nn.functional.pad(
        torch_input_nhwc.reshape(1, 1, input_height * input_width, in_channels),
        (0, padded_channels - in_channels),
        value=0,
    )
    tt_host_tensor = ttnn.from_torch(
        torch_input_padded,
        dtype=ttnn.bfloat16,
        mesh_mapper=inputs_mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    ttnn_model = TTRetinaNet(parameters=parameters, model_config=model_config, device=device, model_args=model_args)

    run_model = create_retinanet_pipeline_model(
        ttnn_model, device, input_height, input_width, in_channels, padded_channels
    )

    input_dram_mem_config = get_memory_config_for_persistent_dram_tensor(
        tt_host_tensor.shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED, device.dram_grid_size()
    )

    input_l1_core_grid = ttnn.CoreGrid(x=8, y=8)
    input_l1_mem_config = ttnn.create_sharded_memory_config(
        shape=(tt_host_tensor.shape[2] // input_l1_core_grid.num_cores, tt_host_tensor.shape[-1]),
        core_grid=input_l1_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipeline = create_pipeline_from_config(
        config,
        run_model,
        device,
        dram_input_memory_config=input_dram_mem_config,
        l1_input_memory_config=input_l1_mem_config,
    )

    host_inputs = [tt_host_tensor] * num_iterations

    start = time.time()
    pipeline.compile(tt_host_tensor)
    end = time.time()
    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    start = time.time()
    outputs = pipeline.enqueue(host_inputs).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end - start):.2f} fps")

    prep_perf_report(
        model_name="retinanet-2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}",
    )
