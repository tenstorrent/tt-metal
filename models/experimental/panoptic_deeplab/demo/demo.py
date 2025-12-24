# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Tuple, List, Union, Optional
import pytest
from loguru import logger
import ttnn
from models.experimental.panoptic_deeplab.reference.pytorch_model import PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS
from models.experimental.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    get_panoptic_deeplab_config,
    create_pytorch_model,
    create_ttnn_model,
    create_model_wrapper,
    create_host_input_tensors_from_torch,
    generate_reference_outputs,
    process_ttnn_outputs_for_visualization,
    load_and_preprocess_image_for_visualization,
    extract_outputs_from_pipeline_result,
    create_visualization_from_outputs,
    validate_outputs_with_pcc,
    convert_pytorch_outputs_to_numpy,
)
from models.experimental.panoptic_deeplab.demo.demo_utils import (
    preprocess_image,
    save_predictions,
    preprocess_input_params,
    skip_if_not_blackhole_20_or_110_cores,
)
from models.tt_cnn.tt.pipeline import PipelineConfig
from models.experimental.panoptic_deeplab.tt.tt_custom_pipeline import (
    create_pipeline_from_config,
)
from models.common.utility_functions import profiler


# Default PCC validation thresholds for demo
# These thresholds are used when validation_thresholds is not provided
DEMO_VALIDATION_THRESHOLDS = {
    "semantic_exp_pcc": 0.951,
    "semantic_exp_abs_err": 1.952,
    "semantic_exp_rel_err": 0.180,
    "center_exp_pcc": 0.949,
    "center_exp_abs_err": 0.008,
    "center_exp_rel_err": 22.349,
    "offset_exp_pcc": 0.966,
    "offset_exp_abs_err": 12.160,
    "offset_exp_rel_err": 2.405,
}


def create_host_input_tensors_from_images(
    device: ttnn.Device, image_paths: List[str], target_size: Tuple[int, int]
) -> Tuple[List[ttnn.Tensor], ttnn.MemoryConfig, ttnn.MemoryConfig, List[str]]:
    """
    Create host input tensors for multiple images.

    Args:
        device: TTNN device
        image_paths: List of paths to input images
        target_size: Target size as (height, width)

    Returns:
        Tuple of (list of host input tensors, dram_memory_config, l1_memory_config, list of successfully processed image paths)

    Raises:
        RuntimeError: If no valid images could be processed
    """
    torch_inputs = []
    successful_paths = []
    for image_path in image_paths:
        try:
            # Preprocess image to get NCHW tensor [1, C, H, W]
            torch_input = preprocess_image(image_path, target_size)
            torch_inputs.append(torch_input)
            successful_paths.append(image_path)
        except (ValueError, Exception) as e:
            logger.warning(f"Could not preprocess image from {image_path}: {e}, skipping...")
            continue

    if len(torch_inputs) == 0:
        raise RuntimeError("No valid images could be preprocessed")

    host_inputs, dram_memory_config, l1_memory_config = create_host_input_tensors_from_torch(device, torch_inputs)
    return host_inputs, dram_memory_config, l1_memory_config, successful_paths


def run_panoptic_deeplab_demo(
    device: ttnn.Device,
    weights_path: str,
    image_paths: Optional[Union[str, List[str]]] = None,
    host_inputs: Optional[List[ttnn.Tensor]] = None,
    dram_memory_config: Optional[ttnn.MemoryConfig] = None,
    l1_memory_config: Optional[ttnn.MemoryConfig] = None,
    output_dir: str = "panoptic_deeplab_predictions",
    target_size: Tuple[int, int] = (512, 1024),
    center_threshold: float = 0.05,
    model_category: str = PANOPTIC_DEEPLAB,
    use_trace: bool = True,
    num_command_queues: int = 1,
    all_transfers_on_separate_command_queue: bool = False,
    generate_visualization: bool = True,
    save_outputs: bool = True,
    return_outputs: bool = False,
    validation_thresholds: Optional[dict] = None,
    layer_name_prefix: str = "",
) -> Optional[Tuple[List, List, List[bool], dict]]:
    """
    Run Panoptic DeepLab inference using pipeline framework.

    Uses pipeline framework with either ModelExecutor (use_trace=False) or
    CustomTracedModelExecutor (use_trace=True).

    Args:
        device: TTNN device
        weights_path: Path to model weights (.pkl file)
        image_paths: Single image path (str) or list of image paths (List[str]).
                     Required if host_inputs is None.
        host_inputs: Pre-created host input tensors. If provided, image_paths is ignored.
        dram_memory_config: Pre-created DRAM memory config. If None, will be extracted from inputs.
        l1_memory_config: Pre-created L1 memory config. If None, will be extracted from inputs.
        output_dir: Directory to save outputs (only used if save_outputs=True)
        target_size: Input size as (height, width). Only used if image_paths is provided.
        center_threshold: Center threshold for panoptic segmentation
        model_category: Model category (PANOPTIC_DEEPLAB or DEEPLAB_V3_PLUS)
        use_trace: If True, use CustomTracedModelExecutor; if False, use ModelExecutor
        num_command_queues: Number of command queues for pipeline
        all_transfers_on_separate_command_queue: Whether to use separate command queue for transfers
        generate_visualization: Whether to generate visualizations
        save_outputs: Whether to save outputs to disk
        return_outputs: Whether to return outputs and reference outputs
        validation_thresholds: Optional dict with custom validation thresholds:
            {
                'semantic_exp_pcc': float,
                'semantic_exp_abs_err': float,
                'semantic_exp_rel_err': float,
                'center_exp_pcc': float,
                'center_exp_abs_err': float,
                'center_exp_rel_err': float,
                'offset_exp_pcc': float,
                'offset_exp_abs_err': float,
                'offset_exp_rel_err': float,
            }
        layer_name_prefix: Prefix for layer names in validation messages

    Returns:
        If return_outputs=True, returns tuple of (outputs, reference_outputs, validation_results, timing_info),
        where:
        - outputs: List of TTNN output tensors
        - reference_outputs: List of PyTorch reference outputs
        - validation_results: List of boolean values indicating pass/fail for each validation check
        - timing_info: Dictionary with timing information:
            {
                'avg_execution_time_us': float,  # Average execution time in microseconds
                'samples_per_second': float,     # Throughput in samples per second
                'total_execution_time_us': float, # Total execution time in microseconds
                'num_inputs': int                 # Number of inputs processed
            }
        Otherwise returns None.
    """
    # Validate inputs
    if host_inputs is None and image_paths is None:
        raise ValueError("Either image_paths or host_inputs must be provided")

    # Get model configuration
    model_config = get_panoptic_deeplab_config()
    num_classes = model_config["num_classes"]
    project_channels = model_config["project_channels"]
    decoder_channels = model_config["decoder_channels"]
    sem_seg_head_channels = model_config["sem_seg_head_channels"]
    ins_embed_head_channels = model_config["ins_embed_head_channels"]
    common_stride = model_config["common_stride"]

    # Handle input creation
    original_images = []
    if host_inputs is None:
        # Normalize image_paths to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        logger.info(f"Running Panoptic DeepLab demo on {len(image_paths)} image(s)")
        logger.info(f"Target size: {target_size}")

        # Load original images for visualization (if needed)
        if generate_visualization:
            for image_path in image_paths:
                try:
                    original_image = load_and_preprocess_image_for_visualization(image_path, target_size)
                    original_images.append((image_path, original_image))
                except (ValueError, Exception) as e:
                    logger.warning(f"Could not load image from {image_path}: {e}, skipping...")
                    continue

            if len(original_images) == 0:
                logger.warning("No valid images found for visualization")
        else:
            # Still need to track paths for validation messages
            original_images = [(path, None) for path in image_paths]

        # Create host input tensors from images
        logger.info("Preprocessing images for TTNN model...")
        (
            host_inputs,
            dram_memory_config,
            l1_memory_config,
            successful_image_paths,
        ) = create_host_input_tensors_from_images(device, image_paths, target_size)

        # Filter original_images to match successfully processed images
        if generate_visualization:
            original_images = [(path, img) for path, img in original_images if path in successful_image_paths]
        else:
            original_images = [(path, None) for path in successful_image_paths]
    else:
        # Use pre-created inputs
        logger.info(f"Using {len(host_inputs)} pre-created input tensor(s)")
        if dram_memory_config is None or l1_memory_config is None:
            # Extract memory configs from first input
            sharded_memory_config = host_inputs[0].memory_config()
            if dram_memory_config is None:
                dram_memory_config = None  # CustomTracedModelExecutor doesn't use DRAM
            if l1_memory_config is None:
                l1_memory_config = ttnn.MemoryConfig(
                    sharded_memory_config.memory_layout,
                    ttnn.BufferType.L1,
                    sharded_memory_config.shard_spec,
                )
        # Create dummy original_images for indexing
        original_images = [(f"input_{i}", None) for i in range(len(host_inputs))]

    if len(host_inputs) == 0:
        logger.error("No valid inputs found")
        if return_outputs:
            return [], [], [], {}
        return None

    logger.info(f"Model category: {model_category}")
    executor_name = "CustomTracedModelExecutor" if use_trace else "ModelExecutor"
    logger.info(f"Executor type: {executor_name}")

    try:
        # Load PyTorch model with weights
        pytorch_model = create_pytorch_model(
            weights_path=weights_path,
            model_category=model_category,
            target_size=target_size,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
        )

        # Create TTNN model (handles parameter creation, fusion, model configs, and model initialization)
        ttnn_model = create_ttnn_model(
            device=device,
            pytorch_model=pytorch_model,
            target_size=target_size,
            batch_size=1,
            model_category=model_category,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
        )

    except FileNotFoundError:
        logger.error(f"Weights file not found: {weights_path}")
        logger.error("Please download the Panoptic DeepLab weights and place them at the specified path.")
        if return_outputs:
            return [], [], [], {}
        return None

    # Pipeline execution path
    # Create model wrapper for pipeline
    model_wrapper = create_model_wrapper(ttnn_model)

    num_inputs = len(host_inputs)

    # Create pipeline
    logger.info(f"Creating pipeline with {executor_name}...")
    pipeline_config = PipelineConfig(
        use_trace=use_trace,
        num_command_queues=num_command_queues,
        all_transfers_on_separate_command_queue=all_transfers_on_separate_command_queue,
    )

    pipeline_args = {
        "config": pipeline_config,
        "model": model_wrapper,
        "device": device,
        "dram_input_memory_config": dram_memory_config,
        "l1_input_memory_config": l1_memory_config,
    }

    pipe = create_pipeline_from_config(**pipeline_args)

    # Compile pipeline
    logger.info("Compiling pipeline...")
    pipe.compile(host_inputs[0])

    # Run pipeline with timing
    logger.info(f"Running pipeline for {num_inputs} inputs...")
    timing_key = f"pipeline_execution_{executor_name}_{model_category}"

    profiler.clear()
    profiler.enable()
    profiler.start(timing_key)
    outputs = pipe.enqueue(host_inputs).pop_all()
    profiler.end(timing_key, PERF_CNT=num_inputs)

    # Store timing results
    avg_execution_time = profiler.get(timing_key)
    total_execution_time = avg_execution_time * num_inputs
    avg_execution_time_us = avg_execution_time * 1e6
    samples_per_second = 1.0 / avg_execution_time if avg_execution_time > 0 else 0

    # Generate reference outputs from PyTorch
    reference_outputs = generate_reference_outputs(pytorch_model, host_inputs)

    # Validate outputs with PCC and relative error checks
    logger.info("Validating outputs with PCC and relative error checks...")
    logger.info("=" * 80)
    logger.info(f"Validation Results ({model_category}):")
    logger.info("=" * 80)
    all_passed = []

    # Prepare validation thresholds - use provided thresholds or fall back to defaults
    if validation_thresholds:
        val_kwargs = validation_thresholds.copy()
    else:
        val_kwargs = DEMO_VALIDATION_THRESHOLDS.copy()

    for i, (ttnn_output, ref_tuple) in enumerate(zip(outputs, reference_outputs)):
        image_path, _ = original_images[i] if i < len(original_images) else (f"input_{i}", None)
        image_name = os.path.basename(image_path) if image_path else f"input_{i}"
        pytorch_semantic, pytorch_center, pytorch_offset = ref_tuple

        # Extract outputs from pipeline result
        ttnn_semantic, ttnn_center, ttnn_offset = extract_outputs_from_pipeline_result(
            ttnn_output, model_category, output_index=i
        )

        logger.info(f"Input {i+1}/{num_inputs}: {image_name}")

        # Validate outputs with PCC
        passed_list = validate_outputs_with_pcc(
            ttnn_model=ttnn_model,
            model_category=model_category,
            pytorch_semantic=pytorch_semantic,
            ttnn_semantic=ttnn_semantic,
            pytorch_center=pytorch_center,
            ttnn_center=ttnn_center,
            pytorch_offset=pytorch_offset,
            ttnn_offset=ttnn_offset,
            layer_name_prefix=f"{layer_name_prefix}{image_name}_",
            **val_kwargs,
        )
        all_passed.extend(passed_list)

        logger.info("")  # Empty line between inputs

    # Cleanup
    pipe.cleanup()

    logger.info("=" * 80)
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Timing Results for {model_category} ({executor_name}):")
    logger.info("=" * 80)
    logger.info(f"  Average execution time: {avg_execution_time_us:.2f} μs")
    logger.info(f"  Average samples per second: {samples_per_second:.2f}")
    logger.info(f"  Total execution time: {total_execution_time * 1e6:.2f} μs ({total_execution_time * 1e3:.2f} ms)")
    logger.info(f"  Number of samples: {num_inputs}")
    logger.info("=" * 80)
    logger.info("")

    # Log PCC and relative error check results
    if not all(all_passed):
        logger.warning(f"Some outputs did not pass PCC or relative error check. Results: {all_passed}")
    else:
        logger.info(f"✅ All PCC and relative error checks passed for {model_category}!")

    # Process results for visualization (if enabled)
    if generate_visualization:
        logger.info("Processing results...")
        for i, (ttnn_output, (image_path, original_image)) in enumerate(zip(outputs, original_images)):
            if original_image is None:
                continue  # Skip if no original image available

            # Extract outputs from pipeline result
            ttnn_semantic_logits, ttnn_center_logits, ttnn_offset_logits = extract_outputs_from_pipeline_result(
                ttnn_output, model_category, output_index=i
            )

            # Process TTNN outputs for visualization
            semantic_np_ttnn, center_np_ttnn, offset_np_ttnn = process_ttnn_outputs_for_visualization(
                ttnn_semantic_logits,
                ttnn_model,
                ttnn_center_logits,
                ttnn_offset_logits,
            )

            # Create visualization
            panoptic_vis_ttnn, panoptic_info_ttnn = create_visualization_from_outputs(
                semantic_np_ttnn,
                original_image,
                model_category,
                center_np=center_np_ttnn,
                offset_np=offset_np_ttnn,
                center_threshold=center_threshold,
            )

            # Save results (if enabled)
            if save_outputs:
                image_name = os.path.basename(image_path)
                ttnn_output_dir = os.path.join(output_dir, "ttnn_output")
                save_predictions(ttnn_output_dir, image_name, original_image, panoptic_vis_ttnn)

            logger.info(f"Processed TTNN image {i+1}/{len(outputs)}: {os.path.basename(image_path)}")

        # Process and save PyTorch reference outputs for visualization
        logger.info("Processing PyTorch reference outputs for visualization...")
        for i, (ref_tuple, (image_path, original_image)) in enumerate(zip(reference_outputs, original_images)):
            if original_image is None:
                continue  # Skip if no original image available

            pytorch_semantic, pytorch_center, pytorch_offset = ref_tuple

            # Convert PyTorch outputs to numpy format for visualization
            semantic_np_pytorch, center_np_pytorch, offset_np_pytorch = convert_pytorch_outputs_to_numpy(
                pytorch_semantic,
                pytorch_center,
                pytorch_offset,
            )

            # Create visualization
            panoptic_vis_pytorch, panoptic_info_pytorch = create_visualization_from_outputs(
                semantic_np_pytorch,
                original_image,
                model_category,
                center_np=center_np_pytorch,
                offset_np=offset_np_pytorch,
                center_threshold=center_threshold,
            )

            # Save results (if enabled)
            if save_outputs:
                image_name = os.path.basename(image_path)
                pytorch_output_dir = os.path.join(output_dir, "pytorch_output")
                save_predictions(pytorch_output_dir, image_name, original_image, panoptic_vis_pytorch)

            logger.info(f"Processed PyTorch image {i+1}/{len(reference_outputs)}: {os.path.basename(image_path)}")

        if save_outputs:
            logger.info(f"Demo completed! Results saved to {output_dir}")
    else:
        logger.info(f"Demo completed! Processed {len(outputs)} inputs using pipeline with {executor_name}")

    if return_outputs:
        timing_info = {
            "avg_execution_time_us": avg_execution_time_us,
            "samples_per_second": samples_per_second,
            "total_execution_time_us": total_execution_time * 1e6,
            "num_inputs": num_inputs,
        }
        return outputs, reference_outputs, all_passed, timing_info
    return None


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": PDL_L1_SMALL_SIZE, "trace_region_size": 2000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    "output_dir",
    [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../demo_outputs")),
    ],
)
@pytest.mark.parametrize("model_category", [PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS])
@pytest.mark.parametrize("use_trace", [False, True], ids=["ModelExecutor", "CustomTracedModelExecutor"])
def test_panoptic_deeplab_demo_pipeline(device, output_dir, model_category, use_trace, model_location_generator):
    """Test pipeline execution with both executor types for both model categories."""
    skip_if_not_blackhole_20_or_110_cores(device)
    images, weights_path, output_dir = preprocess_input_params(
        output_dir, model_category, current_dir=__file__, model_location_generator=model_location_generator
    )
    run_panoptic_deeplab_demo(
        device=device,
        weights_path=weights_path,
        image_paths=images,
        output_dir=output_dir,
        model_category=model_category,
        use_trace=use_trace,
        generate_visualization=True,
        save_outputs=True,
    )
