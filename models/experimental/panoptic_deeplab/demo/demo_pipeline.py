# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Tuple, List
import pytest
import cv2
import torch
from loguru import logger
import ttnn
from models.experimental.panoptic_deeplab.reference.pytorch_model import PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS
from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.common import (
    get_panoptic_deeplab_config,
    PDL_L1_SMALL_SIZE,
)
from models.experimental.panoptic_deeplab.tt.model_configs import ModelOptimisations
from models.experimental.panoptic_deeplab.demo.demo_utils import (
    preprocess_image,
    create_panoptic_visualization,
    create_deeplab_v3plus_visualization,
    save_predictions,
    preprocess_input_params,
)
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
)
from models.experimental.panoptic_deeplab.tt.tt_custom_pipeline import (
    CustomTracedModelExecutor,
    create_pipeline_from_config,
)
from models.common.utility_functions import profiler
from tests.ttnn.utils_for_testing import check_with_pcc


def create_model_wrapper(ttnn_model: TtPanopticDeepLab):
    """
    Create a model wrapper function for use with pipeline/executor.

    The wrapper takes an L1 input tensor and calls the model's forward method.
    The executor expects the model to accept L1 tensors and return device tensors.

    Args:
        ttnn_model: The TtPanopticDeepLab model instance

    Returns:
        A callable function that takes an L1 input tensor and returns model outputs
    """

    def model_forward(l1_input_tensor: ttnn.Tensor):
        """
        Forward pass wrapper for pipeline executor.

        Args:
            l1_input_tensor: Input tensor in L1 memory (expected by executor)

        Returns:
            Tuple of (semantic_logits, center_heatmap, offset_map)
            For DEEPLAB_V3_PLUS, returns only semantic_logits (not a tuple with None)
        """
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"

        # Call model forward
        semantic_logits, center_heatmap, offset_map, _ = ttnn_model.forward(l1_input_tensor, return_features=False)

        # For DEEPLAB_V3_PLUS, center_heatmap and offset_map are None
        # Return only semantic_logits to avoid None handling issues in executor
        if ttnn_model.model_category == DEEPLAB_V3_PLUS:
            return semantic_logits
        else:
            # Return as tuple for PANOPTIC_DEEPLAB
            return (semantic_logits, center_heatmap, offset_map)

    return model_forward


def create_host_input_tensors(
    device: ttnn.Device, image_paths: List[str], target_size: Tuple[int, int]
) -> Tuple[list, ttnn.MemoryConfig, ttnn.MemoryConfig]:
    """
    Create host input tensors for multiple images.

    Args:
        device: TTNN device
        image_paths: List of paths to input images
        target_size: Target size as (height, width)

    Returns:
        Tuple of (list of host input tensors, dram_memory_config, l1_memory_config)
    """
    host_inputs = []
    dram_memory_config = None
    l1_memory_config = None

    SHARD_WIDTH = 8

    for i, image_path in enumerate(image_paths):
        # Preprocess image to get NCHW tensor [1, C, H, W]
        torch_input = preprocess_image(image_path, target_size)

        assert len(torch_input.shape) == 4, f"Expected input tensor to be rank 4 (was {len(torch_input.shape)})"

        C = torch_input.shape[1]
        H = torch_input.shape[2]
        W = torch_input.shape[3]
        HW = H * W

        # Pad channels to SHARD_WIDTH (8) if needed
        # Padding format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        # For channel dimension (dim=1), we pad at the end: pad_front=0, pad_back=SHARD_WIDTH-C
        torch_input = torch.nn.functional.pad(torch_input, (0, 0, 0, 0, 0, SHARD_WIDTH - C), mode="constant", value=0)

        # Convert to channel last (NHWC): [1, H, W, SHARD_WIDTH]
        torch_input = torch_input.permute(0, 2, 3, 1)

        # Create memory configs on first iteration
        if i == 0:
            # CustomTracedModelExecutor doesn't use DRAM memory config - it transfers directly to L1
            dram_memory_config = None

            # Create L1 sharded memory config (height sharding across full grid)
            core_range_set = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device.core_grid.x - 1, device.core_grid.y - 1))}
            )
            num_cores = device.core_grid.x * device.core_grid.y
            shard_height = (1 * HW + num_cores - 1) // num_cores

            sharded_memory_config = ttnn.create_sharded_memory_config_(
                shape=(shard_height, SHARD_WIDTH),
                core_grid=core_range_set,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Use the L1 sharding config for the pipeline
            l1_memory_config = ttnn.MemoryConfig(
                sharded_memory_config.memory_layout,
                ttnn.BufferType.L1,
                sharded_memory_config.shard_spec,
            )

        # Convert to TTNN host tensor (not on device)
        # Use ROW_MAJOR_LAYOUT and bfloat16 dtype to match the original preprocessing
        host_input = ttnn.from_torch(
            torch_input,
            device=None,  # Host tensor, not on device
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        host_inputs.append(host_input)

    return host_inputs, dram_memory_config, l1_memory_config


def run_panoptic_deeplab_demo_pipeline(
    device: ttnn.Device,
    image_paths: List[str],
    weights_path: str,
    output_dir: str = "panoptic_deeplab_predictions",
    target_size: Tuple[int, int] = (512, 1024),
    center_threshold: float = 0.05,
    model_category=DEEPLAB_V3_PLUS,
):
    """
    Run Panoptic DeepLab inference on multiple images using pipeline with a custom implementation of TracedModelExecutor.

    Args:
        device: TTNN device
        image_paths: List of paths to input images
        weights_path: Path to model weights (.pkl file)
        output_dir: Directory to save outputs
        target_size: Input size as (height, width)
        center_threshold: Center threshold for panoptic segmentation
        model_category: Model category (PANOPTIC_DEEPLAB or DEEPLAB_V3_PLUS)
    """
    logger.info(f"Running Panoptic DeepLab pipeline demo on {len(image_paths)} images")
    logger.info(f"Target size: {target_size}")

    # Get model configuration
    config = get_panoptic_deeplab_config()
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    project_channels = config["project_channels"]
    decoder_channels = config["decoder_channels"]
    sem_seg_head_channels = config["sem_seg_head_channels"]
    ins_embed_head_channels = config["ins_embed_head_channels"]
    common_stride = config["common_stride"]

    # Load original images for visualization
    original_images = []
    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.warning(f"Could not load image from {image_path}, skipping...")
            continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image, (int(target_size[1]), int(target_size[0])))
        original_images.append((image_path, original_image))

    if len(original_images) == 0:
        logger.error("No valid images found")
        return

    try:
        # Load PyTorch model with weights
        logger.info("Loading PyTorch model...")
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            train_size=target_size,
            weights_path=weights_path,
            model_category=model_category,
        )
        pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
        pytorch_model.eval()

        # Create TTNN parameters
        logger.info("Creating TTNN parameters...")
        ttnn_parameters = create_panoptic_deeplab_parameters(
            pytorch_model, device, input_height=int(target_size[0]), input_width=int(target_size[1]), batch_size=1
        )

        # Apply Conv+BatchNorm fusion
        logger.info("Applying Conv+BatchNorm fusion to parameters...")
        fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)

        # Create model configurations
        model_configs = ModelOptimisations(
            conv_act_dtype=ttnn.bfloat8_b,
            conv_w_dtype=ttnn.bfloat8_b,
        )
        model_configs.setup_resnet_backbone()
        model_configs.setup_aspp()
        model_configs.setup_decoder()
        model_configs.setup_heads()

        # Create TTNN model
        ttnn_model = TtPanopticDeepLab(
            device=device,
            parameters=fused_parameters,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            train_size=target_size,
            model_configs=model_configs,
            model_category=model_category,
        )

    except FileNotFoundError:
        logger.error(f"Weights file not found: {weights_path}")
        logger.error("Please download the Panoptic DeepLab weights and place them at the specified path.")
        return

    # Create model wrapper for pipeline
    model_wrapper = create_model_wrapper(ttnn_model)

    # Create host input tensors
    logger.info("Preprocessing images for TTNN model...")
    host_inputs, dram_memory_config, l1_memory_config = create_host_input_tensors(
        device, [path for path, _ in original_images], target_size
    )

    num_inputs = len(host_inputs)

    # Create pipeline using TracedModelExecutor
    logger.info("Creating pipeline with TracedModelExecutor...")
    pipeline_config = PipelineConfig(
        use_trace=True,
        num_command_queues=1,
        all_transfers_on_separate_command_queue=False,
    )

    pipeline_args = {
        "config": pipeline_config,
        "model": model_wrapper,
        "device": device,
        "dram_input_memory_config": dram_memory_config,
        "l1_input_memory_config": l1_memory_config,
    }

    pipe = create_pipeline_from_config(**pipeline_args)

    # Verify executor type
    assert isinstance(
        pipe.executor, CustomTracedModelExecutor
    ), f"Expected CustomTracedModelExecutor, got {type(pipe.executor).__name__}"

    # Compile pipeline
    logger.info("Compiling pipeline...")
    pipe.compile(host_inputs[0])

    # Run pipeline with timing
    logger.info(f"Running pipeline for {num_inputs} images...")
    timing_key = f"pipeline_execution_{model_category}"

    profiler.clear()
    profiler.enable()
    profiler.start(timing_key)
    outputs = pipe.enqueue(host_inputs).pop_all()
    profiler.end(timing_key, PERF_CNT=num_inputs)

    # Store timing results for later printing (after PCC)
    avg_execution_time = profiler.get(timing_key)
    total_execution_time = avg_execution_time * num_inputs
    avg_execution_time_us = avg_execution_time * 1e6  # Convert to microseconds
    samples_per_second = 1.0 / avg_execution_time if avg_execution_time > 0 else 0

    # Generate reference outputs from PyTorch (after pipeline execution)
    logger.info("Generating reference outputs from PyTorch model...")
    reference_outputs = []
    for host_input in host_inputs:
        # Convert host input back to torch for reference
        torch_input = ttnn.to_torch(host_input)
        # Input is in NHWC format [1, H, W, C] where C=8 (3 original + 5 padding)
        # We need to remove padding and convert to NCHW for PyTorch model
        assert torch_input.shape[0] == 1, f"Expected batch size 1, got {torch_input.shape[0]}"
        # Remove padding: take only first 3 channels
        torch_input = torch_input[:, :, :, :3]  # [1, H, W, 3]
        # Convert NHWC -> NCHW
        torch_input = torch_input.permute(0, 3, 1, 2)  # [1, 3, H, W]

        with torch.no_grad():
            pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(torch_input)
        reference_outputs.append((pytorch_semantic, pytorch_center, pytorch_offset))

    # Validate outputs
    assert len(outputs) == len(reference_outputs), f"Expected {len(reference_outputs)} outputs, got {len(outputs)}"
    assert len(outputs) == num_inputs, f"Expected {num_inputs} outputs, got {len(outputs)}"

    # Validate outputs with PCC checks
    logger.info("Validating outputs with PCC checks...")
    logger.info("=" * 80)
    logger.info(f"PCC Values for Individual Images ({model_category}):")
    logger.info("=" * 80)
    all_passed = []
    semantic_pcc_values = []
    center_pcc_values = []
    offset_pcc_values = []

    def get_pcc_value(pytorch_output, ttnn_output, to_channel_first=False, output_channels=None, exp_pcc=0.999):
        """Helper function to get PCC value without logging."""
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        if to_channel_first:
            ttnn_output_torch = ttnn_output_torch.permute(0, 3, 1, 2)  # NHWC to NCHW

        if output_channels is not None:
            ttnn_output_torch = ttnn_output_torch[:, :output_channels, :, :]

        passed, pcc = check_with_pcc(pytorch_output, ttnn_output_torch, exp_pcc)
        return passed, float(pcc)

    for i, (ttnn_output, ref_tuple) in enumerate(zip(outputs, reference_outputs)):
        # Get image name for this output
        image_path, _ = original_images[i]
        image_name = os.path.basename(image_path)
        pytorch_semantic, pytorch_center, pytorch_offset = ref_tuple

        # Handle different output formats based on model category
        # Note: Pipeline converts tuple outputs to lists, so we check for both
        if model_category == DEEPLAB_V3_PLUS:
            # For DEEPLAB_V3_PLUS, output is a single tensor (semantic_logits only)
            ttnn_semantic = ttnn_output
            assert isinstance(ttnn_semantic, ttnn.Tensor), f"Semantic output {i} should be ttnn.Tensor"
            assert ttnn_semantic.storage_type() == ttnn.StorageType.HOST, f"Semantic output {i} should be on host"
        else:
            # For PANOPTIC_DEEPLAB, output is a tuple/list of 3 tensors
            # Pipeline converts tuples to lists, so we accept both
            assert isinstance(
                ttnn_output, (tuple, list)
            ), f"Output {i} should be a tuple or list for PANOPTIC_DEEPLAB, got {type(ttnn_output)}"
            assert len(ttnn_output) == 3, f"Output {i} should have 3 elements, got {len(ttnn_output)}"
            ttnn_semantic, ttnn_center, ttnn_offset = ttnn_output

            # Validate output structure
            assert isinstance(ttnn_semantic, ttnn.Tensor), f"Semantic output {i} should be ttnn.Tensor"
            assert ttnn_semantic.storage_type() == ttnn.StorageType.HOST, f"Semantic output {i} should be on host"
            assert isinstance(ttnn_center, ttnn.Tensor), f"Center output {i} should be ttnn.Tensor"
            assert ttnn_center.storage_type() == ttnn.StorageType.HOST, f"Center output {i} should be on host"
            assert isinstance(ttnn_offset, ttnn.Tensor), f"Offset output {i} should be ttnn.Tensor"
            assert ttnn_offset.storage_type() == ttnn.StorageType.HOST, f"Offset output {i} should be on host"

        # Check semantic output
        passed, pcc = get_pcc_value(
            pytorch_semantic,
            ttnn_semantic,
            to_channel_first=False,
            output_channels=ttnn_model.semantic_head.get_output_channels_for_slicing(),
            exp_pcc=0.95,
        )
        all_passed.append(passed)
        semantic_pcc_values.append(pcc)

        # Print PCC for this image
        logger.info(f"Image {i+1}/{num_inputs}: {image_name}")
        logger.info(f"  Semantic PCC: {pcc:.6f} {'✅' if passed else '❌'}")

        # Check instance outputs (only for PANOPTIC_DEEPLAB)
        if model_category == PANOPTIC_DEEPLAB:
            passed_center, pcc_center = get_pcc_value(
                pytorch_center,
                ttnn_center,
                to_channel_first=False,
                output_channels=ttnn_model.instance_head.get_center_output_channels_for_slicing(),
                exp_pcc=0.784,
            )
            all_passed.append(passed_center)
            center_pcc_values.append(pcc_center)

            passed_offset, pcc_offset = get_pcc_value(
                pytorch_offset,
                ttnn_offset,
                to_channel_first=False,
                output_channels=ttnn_model.instance_head.get_offset_output_channels_for_slicing(),
                exp_pcc=0.965,
            )
            all_passed.append(passed_offset)
            offset_pcc_values.append(pcc_offset)

            logger.info(f"  Center PCC: {pcc_center:.6f} {'✅' if passed_center else '❌'}")
            logger.info(f"  Offset PCC: {pcc_offset:.6f} {'✅' if passed_offset else '❌'}")

        logger.info("")  # Empty line between images

    # Cleanup
    pipe.cleanup()

    logger.info("=" * 80)

    # Print timing results after PCC statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Timing Results for {model_category}:")
    logger.info("=" * 80)
    logger.info(f"  Average execution time: {avg_execution_time_us:.2f} μs")
    logger.info(f"  Average samples per second: {samples_per_second:.2f}")
    logger.info(f"  Total execution time: {total_execution_time * 1e6:.2f} μs ({total_execution_time * 1e3:.2f} ms)")
    logger.info(f"  Number of samples: {num_inputs}")
    logger.info("=" * 80)
    logger.info("")

    # Log PCC check results (but don't fail - this is a demo)
    if not all(all_passed):
        logger.warning(f"Some outputs did not pass PCC check. Results: {all_passed}")
    else:
        logger.info(f"✅ All PCC checks passed for {model_category}!")

    # Process results
    logger.info("Processing results...")
    for i, (ttnn_output, (image_path, original_image)) in enumerate(zip(outputs, original_images)):
        # Handle different output formats based on model category
        if model_category == DEEPLAB_V3_PLUS:
            # For DEEPLAB_V3_PLUS, output is a single tensor (semantic_logits only)
            ttnn_semantic_logits = ttnn_output
        else:
            # For PANOPTIC_DEEPLAB, output is a tuple/list of 3 tensors
            assert isinstance(ttnn_output, (tuple, list)), f"Output {i} should be a tuple or list for PANOPTIC_DEEPLAB"
            assert len(ttnn_output) == 3, f"Output {i} should have 3 elements, got {len(ttnn_output)}"
            ttnn_semantic_logits, ttnn_center_logits, ttnn_offset_logits = ttnn_output

        # Handle semantic output - convert from NHWC to NCHW and slice padding if needed
        ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic_logits)
        semantic_original_channels = ttnn_model.semantic_head.get_output_channels_for_slicing()
        if semantic_original_channels is not None:
            ttnn_semantic_torch = ttnn_semantic_torch[:, :semantic_original_channels, :, :]

        # Convert to numpy in HWC format for visualization
        semantic_np_ttnn = ttnn_semantic_torch.float().squeeze(0).permute(1, 2, 0).numpy()

        if model_category == PANOPTIC_DEEPLAB:
            # Handle center output
            ttnn_center_torch = ttnn.to_torch(ttnn_center_logits)
            center_original_channels = ttnn_model.instance_head.get_center_output_channels_for_slicing()
            if center_original_channels is not None:
                ttnn_center_torch = ttnn_center_torch[:, :center_original_channels, :, :]

            # Handle offset output
            ttnn_offset_torch = ttnn.to_torch(ttnn_offset_logits)
            offset_original_channels = ttnn_model.instance_head.get_offset_output_channels_for_slicing()
            if offset_original_channels is not None:
                ttnn_offset_torch = ttnn_offset_torch[:, :offset_original_channels, :, :]

            center_np_ttnn = ttnn_center_torch.float().squeeze(0).permute(1, 2, 0).numpy()
            offset_np_ttnn = ttnn_offset_torch.float().squeeze(0).permute(1, 2, 0).numpy()

            panoptic_vis_ttnn, panoptic_info_ttnn = create_panoptic_visualization(
                semantic_np_ttnn,
                center_np_ttnn,
                offset_np_ttnn,
                original_image,
                center_threshold=center_threshold,
                score_threshold=center_threshold,
                stuff_area=1,
                top_k=1000,
                nms_kernel=11,
            )
        else:
            import time

            start_time = time.time()
            panoptic_vis_ttnn, panoptic_info_ttnn = create_deeplab_v3plus_visualization(
                semantic_np_ttnn,
                original_image=original_image,
            )
            end_time = time.time()
            visualization_duration_us = (end_time - start_time) * 1e6
            logger.info(f"create_deeplab_v3plus_visualization() duration: {visualization_duration_us:.2f} μs")

        # Save results
        image_name = os.path.basename(image_path)
        ttnn_output_dir = os.path.join(output_dir, "ttnn_output")
        save_predictions(ttnn_output_dir, image_name, original_image, panoptic_vis_ttnn)

        logger.info(f"Processed image {i+1}/{len(outputs)}: {image_name}")

    logger.info(f"Demo completed! Results saved to {output_dir}")
    logger.info(f"Processed {len(outputs)} images using pipeline with TracedModelExecutor")


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE, "trace_region_size": 2000000}], indirect=True
)
@pytest.mark.parametrize(
    "output_dir",
    [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../demo_outputs")),
    ],
)
@pytest.mark.parametrize("model_category", [PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS])
def test_panoptic_deeplab_demo_pipeline(device, output_dir, model_category, model_location_generator):
    skip_if_not_blackhole_20_cores(device)
    images, weights_path, output_dir = preprocess_input_params(
        output_dir, model_category, current_dir=__file__, model_location_generator=model_location_generator
    )
    # Process all images using pipeline
    run_panoptic_deeplab_demo_pipeline(device, images, weights_path, output_dir, model_category=model_category)
