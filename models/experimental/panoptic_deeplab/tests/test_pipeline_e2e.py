# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for Panoptic DeepLab using TTNN builder, executor, and pipeline.

This module tests the complete Panoptic DeepLab model using the pipeline/executor
framework from models/tt_cnn/tt/, enabling testing across different executor
configurations (traced, multi-CQ, pipelined I/O, etc.).

Note: The model wrapper handles different output formats:
- For PANOPTIC_DEEPLAB: Returns tuple (semantic_logits, center_heatmap, offset_map)
- For DEEPLAB_V3_PLUS: Returns single tensor (semantic_logits only)
This avoids None handling issues in the executor's output schema creation.
"""

import pytest
import torch
import ttnn
from dataclasses import dataclass
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_model import PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS
from models.experimental.panoptic_deeplab.tt.model_configs import ModelOptimisations
from models.experimental.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    get_panoptic_deeplab_weights_path,
    get_panoptic_deeplab_config,
    create_pytorch_model,
    create_ttnn_model,
    create_model_wrapper,
    create_host_input_tensors_random,
    generate_reference_outputs,
)
from models.common.utility_functions import profiler
from models.experimental.panoptic_deeplab.tests.pcc.common import check_ttnn_output
from models.tt_cnn.tt.executor import (
    ModelExecutor,
)
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
)
from models.experimental.panoptic_deeplab.tt.tt_custom_pipeline import (
    CustomTracedModelExecutor,
    create_pipeline_from_config,
)
from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores


@dataclass
class ExecutorTestConfig:
    """Configuration for testing different executor types."""

    name: str
    use_trace: bool
    num_command_queues: int
    all_transfers_on_separate_command_queue: bool = False
    requires_output_config: bool = False
    requires_minimum_inputs: int = 1
    expected_executor_type: type = None


EXECUTOR_CONFIGS = [
    ExecutorTestConfig(
        name="ModelExecutor",
        use_trace=False,
        num_command_queues=1,
        all_transfers_on_separate_command_queue=False,
        requires_minimum_inputs=1,
        expected_executor_type=ModelExecutor,
    ),
    ExecutorTestConfig(
        name="TracedModelExecutor",
        use_trace=True,
        num_command_queues=1,
        all_transfers_on_separate_command_queue=False,
        requires_minimum_inputs=1,
        expected_executor_type=CustomTracedModelExecutor,
    ),
]


@pytest.mark.parametrize("executor_config", EXECUTOR_CONFIGS, ids=lambda cfg: cfg.name)
@pytest.mark.parametrize(
    "model_category",
    [PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS],
    ids=["panoptic_deeplab", "deeplab_v3_plus"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": PDL_L1_SMALL_SIZE,
            "trace_region_size": 2000000,
            "num_command_queues": 1,
        }
    ],
    indirect=True,
)
def test_panoptic_deeplab_pipeline_e2e(
    device, executor_config: ExecutorTestConfig, model_category: str, model_location_generator
):
    """
    End-to-end test for Panoptic DeepLab using pipeline/executor framework.

    Tests the complete model with different executor configurations, validates
    outputs against PyTorch reference, and measures execution timing.
    Uses 100 random inputs in a single round.
    """
    skip_if_not_blackhole_20_cores(device)
    torch.manual_seed(0)

    # Get model configuration
    model_config = get_panoptic_deeplab_config()
    batch_size = model_config["batch_size"]
    num_classes = model_config["num_classes"]
    input_height, input_width = model_config["train_size"]

    # Get weights path
    complete_weights_path = get_panoptic_deeplab_weights_path(model_location_generator, __file__)

    try:
        # Create PyTorch reference model
        pytorch_model = create_pytorch_model(
            weights_path=complete_weights_path,
            model_category=model_category,
            target_size=model_config["train_size"],
            num_classes=num_classes,
            common_stride=model_config["common_stride"],
            project_channels=model_config["project_channels"],
            decoder_channels=model_config["decoder_channels"],
            sem_seg_head_channels=model_config["sem_seg_head_channels"],
            ins_embed_head_channels=model_config["ins_embed_head_channels"],
        )

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
        ttnn_model = create_ttnn_model(
            device=device,
            pytorch_model=pytorch_model,
            target_size=model_config["train_size"],
            batch_size=batch_size,
            model_category=model_category,
            num_classes=num_classes,
            common_stride=model_config["common_stride"],
            project_channels=model_config["project_channels"],
            decoder_channels=model_config["decoder_channels"],
            sem_seg_head_channels=model_config["sem_seg_head_channels"],
            ins_embed_head_channels=model_config["ins_embed_head_channels"],
            model_configs=model_configs,
        )

        # Create model wrapper for pipeline
        model_wrapper = create_model_wrapper(ttnn_model)

        # Create 100 random input tensors
        num_inputs = 100
        logger.info(f"Creating {num_inputs} random input tensors...")
        host_inputs, dram_memory_config, l1_memory_config = create_host_input_tensors_random(
            device, batch_size, input_height, input_width, num_inputs
        )

        # Create pipeline using extracted memory configs
        pipeline_config = PipelineConfig(
            use_trace=executor_config.use_trace,
            num_command_queues=executor_config.num_command_queues,
            all_transfers_on_separate_command_queue=executor_config.all_transfers_on_separate_command_queue,
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
            pipe.executor, executor_config.expected_executor_type
        ), f"Expected {executor_config.expected_executor_type.__name__}, got {type(pipe.executor).__name__}"

        # Compile pipeline
        logger.info(f"Compiling pipeline with {executor_config.name}...")
        pipe.compile(host_inputs[0])

        # Run pipeline with timing
        logger.info(f"Running pipeline with {executor_config.name} for {num_inputs} inputs...")
        timing_key = f"pipeline_execution_{executor_config.name}_{model_category}"

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
        reference_outputs = generate_reference_outputs(pytorch_model, host_inputs)

        # Validate outputs
        assert len(outputs) == len(reference_outputs), f"Expected {len(reference_outputs)} outputs, got {len(outputs)}"
        assert len(outputs) == num_inputs, f"Expected {num_inputs} outputs, got {len(outputs)}"

        # Validate outputs with PCC and relative error checks
        logger.info("Validating outputs with PCC and relative error checks...")
        all_passed = []

        for i, (ttnn_output, ref_tuple) in enumerate(zip(outputs, reference_outputs)):
            pytorch_semantic, pytorch_center, pytorch_offset = ref_tuple

            # Handle different output formats based on model category
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

            # Check semantic output with PCC and relative errors
            passed = check_ttnn_output(
                layer_name=f"semantic_{i}",
                pytorch_output=pytorch_semantic,
                ttnn_output=ttnn_semantic,
                to_channel_first=False,
                output_channels=ttnn_model.semantic_head.get_output_channels_for_slicing(),
                exp_pcc=0.985,
                exp_abs_err=1.340,
                exp_rel_err=0.480,
            )
            all_passed.append(passed)

            # Check instance outputs (only for PANOPTIC_DEEPLAB)
            if model_category == PANOPTIC_DEEPLAB:
                passed_center = check_ttnn_output(
                    layer_name=f"center_{i}",
                    pytorch_output=pytorch_center,
                    ttnn_output=ttnn_center,
                    to_channel_first=False,
                    output_channels=ttnn_model.instance_head.get_center_output_channels_for_slicing(),
                    exp_pcc=0.784,
                    exp_abs_err=0.001,
                    exp_rel_err=2.710,
                )
                all_passed.append(passed_center)

                passed_offset = check_ttnn_output(
                    layer_name=f"offset_{i}",
                    pytorch_output=pytorch_offset,
                    ttnn_output=ttnn_offset,
                    to_channel_first=False,
                    output_channels=ttnn_model.instance_head.get_offset_output_channels_for_slicing(),
                    exp_pcc=0.985,
                    exp_abs_err=11.480,
                    exp_rel_err=1.120,
                )
                all_passed.append(passed_offset)

        # Cleanup
        pipe.cleanup()

        # Print timing results
        logger.info(
            f"Timing results for {executor_config.name} with {model_category}: "
            f"Average execution time: {avg_execution_time_us:.2f} μs, "
            f"Average samples per second: {samples_per_second:.2f}"
        )

        # Fail test if any PCC or relative error checks failed
        assert all(all_passed), f"Some outputs did not pass PCC or relative error check. Results: {all_passed}"
        logger.info(f"✅ All PCC and relative error tests passed for {executor_config.name} with {model_category}!")

    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")
