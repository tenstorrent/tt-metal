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
from models.experimental.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    get_panoptic_deeplab_weights_path,
    get_panoptic_deeplab_config,
    create_host_input_tensors_random,
)
from models.experimental.panoptic_deeplab.demo.demo import run_panoptic_deeplab_demo
from models.tt_cnn.tt.executor import (
    ModelExecutor,
)
from models.experimental.panoptic_deeplab.tt.tt_custom_pipeline import (
    CustomTracedModelExecutor,
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


# Custom PCC validation thresholds for pipeline e2e tests
# These thresholds are calibrated for TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=4,3 configuration
VALIDATION_THRESHOLDS = {
    "semantic_exp_pcc": 0.985,
    "semantic_exp_abs_err": 1.340,
    "semantic_exp_rel_err": 0.480,
    "center_exp_pcc": 0.781,
    "center_exp_abs_err": 0.001,
    "center_exp_rel_err": 2.710,
    "offset_exp_pcc": 0.984,
    "offset_exp_abs_err": 11.480,
    "offset_exp_rel_err": 1.850,
}


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
    input_height, input_width = model_config["train_size"]

    # Get weights path
    complete_weights_path = get_panoptic_deeplab_weights_path(model_location_generator, __file__)

    try:
        # Create 100 random input tensors
        num_inputs = 100
        logger.info(f"Creating {num_inputs} random input tensors...")
        host_inputs, dram_memory_config, l1_memory_config = create_host_input_tensors_random(
            device, batch_size, input_height, input_width, num_inputs
        )

        # Verify executor type by creating a temporary pipeline
        # This is test-specific validation that we need to keep
        from models.experimental.panoptic_deeplab.tt.common import (
            create_pytorch_model,
            create_ttnn_model,
            create_model_wrapper,
        )
        from models.experimental.panoptic_deeplab.tt.model_configs import ModelOptimisations
        from models.tt_cnn.tt.pipeline import PipelineConfig
        from models.experimental.panoptic_deeplab.tt.tt_custom_pipeline import create_pipeline_from_config

        pytorch_model = create_pytorch_model(
            weights_path=complete_weights_path,
            model_category=model_category,
            target_size=model_config["train_size"],
            num_classes=model_config["num_classes"],
            common_stride=model_config["common_stride"],
            project_channels=model_config["project_channels"],
            decoder_channels=model_config["decoder_channels"],
            sem_seg_head_channels=model_config["sem_seg_head_channels"],
            ins_embed_head_channels=model_config["ins_embed_head_channels"],
        )

        model_configs = ModelOptimisations(
            conv_act_dtype=ttnn.bfloat8_b,
            conv_w_dtype=ttnn.bfloat8_b,
        )
        model_configs.setup_resnet_backbone()
        model_configs.setup_aspp()
        model_configs.setup_decoder()
        model_configs.setup_heads()

        ttnn_model = create_ttnn_model(
            device=device,
            pytorch_model=pytorch_model,
            target_size=model_config["train_size"],
            batch_size=batch_size,
            model_category=model_category,
            num_classes=model_config["num_classes"],
            common_stride=model_config["common_stride"],
            project_channels=model_config["project_channels"],
            decoder_channels=model_config["decoder_channels"],
            sem_seg_head_channels=model_config["sem_seg_head_channels"],
            ins_embed_head_channels=model_config["ins_embed_head_channels"],
            model_configs=model_configs,
        )

        model_wrapper = create_model_wrapper(ttnn_model)
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
        pipe.cleanup()

        # Use run_panoptic_deeplab_demo for the actual execution
        result = run_panoptic_deeplab_demo(
            device=device,
            weights_path=complete_weights_path,
            host_inputs=host_inputs,
            dram_memory_config=dram_memory_config,
            l1_memory_config=l1_memory_config,
            model_category=model_category,
            use_trace=executor_config.use_trace,
            num_command_queues=executor_config.num_command_queues,
            all_transfers_on_separate_command_queue=executor_config.all_transfers_on_separate_command_queue,
            generate_visualization=False,
            save_outputs=False,
            return_outputs=True,
            validation_thresholds=VALIDATION_THRESHOLDS,
            layer_name_prefix="",
        )

        # Verify outputs were returned
        assert result is not None, "run_panoptic_deeplab_demo should return outputs when return_outputs=True"
        outputs, reference_outputs, validation_results, timing_info = result

        # Validate outputs
        assert len(outputs) == len(reference_outputs), f"Expected {len(reference_outputs)} outputs, got {len(outputs)}"
        assert len(outputs) == num_inputs, f"Expected {num_inputs} outputs, got {len(outputs)}"

        # Fail test if any PCC or relative error checks failed
        assert all(
            validation_results
        ), f"Some outputs did not pass PCC or relative error check. Results: {validation_results}"

        # Log timing information
        logger.info(
            f"Timing results for {executor_config.name} with {model_category}: "
            f"Average execution time: {timing_info['avg_execution_time_us']:.2f} μs, "
            f"Average samples per second: {timing_info['samples_per_second']:.2f}"
        )

        logger.info(f"✅ All tests passed for {executor_config.name} with {model_category}!")

    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")
