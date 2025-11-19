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

Known Issue: There is a bug in models/tt_cnn/tt/builder.py TtConv2d._apply_channel_slicing()
where the tuple (weight, bias) returned from conv2d with return_weights_and_bias=True
is incorrectly assigned to self.weight_slices[i], causing subsequent calls to pass
a tuple as weight_tensor. This needs to be fixed in the builder code.
"""

import pytest
import torch
import ttnn
from dataclasses import dataclass
from typing import Callable, Tuple
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_model import PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS
from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.model_configs import ModelOptimisations
from models.experimental.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    get_panoptic_deeplab_weights_path,
    get_panoptic_deeplab_config,
    preprocess_nchw_input_tensor,
)
from models.experimental.panoptic_deeplab.tests.pcc.common import check_ttnn_output
from models.tt_cnn.tt.executor import (
    ModelExecutor,
    MultiCQTracedModelOverlappedInputExecutor,
    TracedModelExecutor,
)
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
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
        expected_executor_type=TracedModelExecutor,
    ),
    ExecutorTestConfig(
        name="MultiCQTracedModelOverlappedInputExecutor",
        use_trace=True,
        num_command_queues=2,
        all_transfers_on_separate_command_queue=False,
        requires_minimum_inputs=1,
        expected_executor_type=MultiCQTracedModelOverlappedInputExecutor,
    ),
    # Note: MultiCQTracedModelPipelinedIOExecutor doesn't support tuple outputs,
    # so we skip it for now. If needed, we'd need to modify the model wrapper
    # to return a single tensor or modify the executor to support tuples.
]


def create_model_wrapper(ttnn_model: TtPanopticDeepLab) -> Callable:
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
    device: ttnn.Device, batch_size: int, input_height: int, input_width: int, num_inputs: int
) -> Tuple[list, ttnn.MemoryConfig, ttnn.MemoryConfig]:
    """
    Create host input tensors for Panoptic DeepLab.

    Uses interleaved DRAM to avoid core grid constraints (especially for traced executors),
    and converts to L1 using to_memory_config which handles the interleaved-to-sharded conversion.

    Args:
        device: TTNN device
        batch_size: Batch size (should be 1 for Panoptic DeepLab)
        input_height: Input image height
        input_width: Input image width
        num_inputs: Number of input tensors to create

    Returns:
        Tuple of (list of host input tensors, dram_memory_config, l1_memory_config)
        - dram_memory_config: Interleaved DRAM (no core constraints)
        - l1_memory_config: Sharded L1 with full grid (original sharding from preprocessed tensor)
    """
    host_inputs = []
    dram_memory_config = None
    l1_memory_config = None

    for i in range(num_inputs):
        # Create random input in NCHW format
        torch_input = torch.randn(batch_size, 3, input_height, input_width, dtype=torch.bfloat16)
        # Preprocess to TTNN format (height-sharded on device)
        ttnn_input = preprocess_nchw_input_tensor(device, torch_input)

        # Extract memory config from first tensor
        if i == 0:
            # Get the L1 memory config from the preprocessed tensor (full grid sharding)
            original_mem_config = ttnn_input.memory_config()
            original_shard_spec = original_mem_config.shard_spec

            # Always use interleaved DRAM (no core constraints)
            # This avoids the logical grid constraint issue with traced executors
            # The executor will use to_memory_config to convert from interleaved DRAM to sharded L1
            dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

            # Use the original L1 sharding (full grid) - to_memory_config will handle conversion
            l1_memory_config = ttnn.MemoryConfig(
                original_mem_config.memory_layout,
                ttnn.BufferType.L1,
                original_shard_spec,
            )

        # Convert to host tensor for pipeline
        host_input = ttnn_input.cpu()
        host_inputs.append(host_input)

    return host_inputs, dram_memory_config, l1_memory_config


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
            "num_command_queues": 2,  # Required for multi-CQ executors
        }
    ],
    indirect=True,
)
def test_panoptic_deeplab_pipeline_e2e(
    device, executor_config: ExecutorTestConfig, model_category: str, model_location_generator
):
    """
    End-to-end test for Panoptic DeepLab using pipeline/executor framework.

    Tests the complete model with different executor configurations and validates
    outputs against PyTorch reference.
    """
    skip_if_not_blackhole_20_cores(device)
    torch.manual_seed(0)

    # Get model configuration
    config = get_panoptic_deeplab_config()
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    input_height, input_width = config["train_size"]

    # Get weights path
    complete_weights_path = get_panoptic_deeplab_weights_path(model_location_generator, __file__)

    try:
        # Create PyTorch reference model
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=num_classes,
            common_stride=config["common_stride"],
            project_channels=config["project_channels"],
            decoder_channels=config["decoder_channels"],
            sem_seg_head_channels=config["sem_seg_head_channels"],
            ins_embed_head_channels=config["ins_embed_head_channels"],
            train_size=config["train_size"],
            weights_path=complete_weights_path,
            model_category=model_category,
        )
        pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
        pytorch_model.eval()

        # Create TTNN parameters
        ttnn_parameters = create_panoptic_deeplab_parameters(
            pytorch_model, device, input_height=input_height, input_width=input_width, batch_size=batch_size
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
            common_stride=config["common_stride"],
            project_channels=config["project_channels"],
            decoder_channels=config["decoder_channels"],
            sem_seg_head_channels=config["sem_seg_head_channels"],
            ins_embed_head_channels=config["ins_embed_head_channels"],
            train_size=config["train_size"],
            model_configs=model_configs,
            model_category=model_category,
        )

        # Create model wrapper for pipeline
        model_wrapper = create_model_wrapper(ttnn_model)

        # Determine number of inputs based on executor requirements
        num_inputs = max(executor_config.requires_minimum_inputs, 3)  # Test with at least 3 inputs

        # Create host input tensors and extract memory configs
        # Uses interleaved DRAM to avoid core grid constraints, converts to L1 via to_memory_config
        host_inputs, dram_memory_config, l1_memory_config = create_host_input_tensors(
            device, batch_size, input_height, input_width, num_inputs
        )

        # Create pipeline using extracted memory configs
        config = PipelineConfig(
            use_trace=executor_config.use_trace,
            num_command_queues=executor_config.num_command_queues,
            all_transfers_on_separate_command_queue=executor_config.all_transfers_on_separate_command_queue,
        )

        pipeline_args = {
            "config": config,
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

        # Generate reference outputs from PyTorch
        reference_outputs = []
        for host_input in host_inputs:
            # Convert host input back to torch for reference
            torch_input = ttnn.to_torch(host_input)
            # Input is in NHWC format [1, H, W, C] where C=8 (3 original + 5 padding)
            # We need to remove padding and convert to NCHW for PyTorch model
            assert torch_input.shape[0] == 1, f"Expected batch size 1, got {torch_input.shape[0]}"
            h, w, c = torch_input.shape[1], torch_input.shape[2], torch_input.shape[3]
            # Remove padding: take only first 3 channels
            torch_input = torch_input[:, :, :, :3]  # [1, H, W, 3]
            # Convert NHWC -> NCHW
            torch_input = torch_input.permute(0, 3, 1, 2)  # [1, 3, H, W]

            with torch.no_grad():
                pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(torch_input)
            reference_outputs.append((pytorch_semantic, pytorch_center, pytorch_offset))

        # Run pipeline
        logger.info(f"Running pipeline with {executor_config.name}...")
        outputs = pipe.enqueue(host_inputs).pop_all()

        # Validate outputs
        assert len(outputs) == len(reference_outputs), f"Expected {len(reference_outputs)} outputs, got {len(outputs)}"

        all_passed = []
        for i, (ttnn_output, ref_tuple) in enumerate(zip(outputs, reference_outputs)):
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
            all_passed.append(
                check_ttnn_output(
                    f"Semantic_{i}",
                    pytorch_semantic,
                    ttnn_semantic,
                    to_channel_first=False,
                    output_channels=ttnn_model.semantic_head.get_output_channels_for_slicing(),
                    exp_pcc=0.986,
                )
            )

            # Check instance outputs (only for PANOPTIC_DEEPLAB)
            if model_category == PANOPTIC_DEEPLAB:
                all_passed.append(
                    check_ttnn_output(
                        f"Center_{i}",
                        pytorch_center,
                        ttnn_center,
                        to_channel_first=False,
                        output_channels=ttnn_model.instance_head.get_center_output_channels_for_slicing(),
                        exp_pcc=0.805,
                    )
                )
                all_passed.append(
                    check_ttnn_output(
                        f"Offset_{i}",
                        pytorch_offset,
                        ttnn_offset,
                        to_channel_first=False,
                        output_channels=ttnn_model.instance_head.get_offset_output_channels_for_slicing(),
                        exp_pcc=0.989,
                    )
                )

        # Cleanup
        pipe.cleanup()

        # Fail test if any PCC checks failed
        assert all(all_passed), f"Some outputs did not pass PCC check. Results: {all_passed}"
        logger.info(f"✅ All PCC tests passed for {executor_config.name} with {model_category}!")

    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")


@pytest.mark.parametrize("executor_config", EXECUTOR_CONFIGS[:2], ids=lambda cfg: cfg.name)  # Test first 2 executors
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": PDL_L1_SMALL_SIZE,
            "trace_region_size": 2000000,
            "num_command_queues": 2,  # Required for multi-CQ executors
        }
    ],
    indirect=True,
)
def test_panoptic_deeplab_pipeline_multiple_rounds(
    device, executor_config: ExecutorTestConfig, model_location_generator
):
    """
    Test multiple execution rounds with the pipeline to ensure stability.
    """
    skip_if_not_blackhole_20_cores(device)
    torch.manual_seed(0)

    config = get_panoptic_deeplab_config()
    batch_size = config["batch_size"]
    input_height, input_width = config["train_size"]

    complete_weights_path = get_panoptic_deeplab_weights_path(model_location_generator, __file__)

    try:
        # Setup model (same as main test)
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=config["num_classes"],
            common_stride=config["common_stride"],
            project_channels=config["project_channels"],
            decoder_channels=config["decoder_channels"],
            sem_seg_head_channels=config["sem_seg_head_channels"],
            ins_embed_head_channels=config["ins_embed_head_channels"],
            train_size=config["train_size"],
            weights_path=complete_weights_path,
            model_category=PANOPTIC_DEEPLAB,
        )
        pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
        pytorch_model.eval()

        ttnn_parameters = create_panoptic_deeplab_parameters(
            pytorch_model, device, input_height=input_height, input_width=input_width, batch_size=batch_size
        )
        fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)

        model_configs = ModelOptimisations(conv_act_dtype=ttnn.bfloat8_b, conv_w_dtype=ttnn.bfloat8_b)
        model_configs.setup_resnet_backbone()
        model_configs.setup_aspp()
        model_configs.setup_decoder()
        model_configs.setup_heads()

        ttnn_model = TtPanopticDeepLab(
            device=device,
            parameters=fused_parameters,
            num_classes=config["num_classes"],
            common_stride=config["common_stride"],
            project_channels=config["project_channels"],
            decoder_channels=config["decoder_channels"],
            sem_seg_head_channels=config["sem_seg_head_channels"],
            ins_embed_head_channels=config["ins_embed_head_channels"],
            train_size=config["train_size"],
            model_configs=model_configs,
            model_category=PANOPTIC_DEEPLAB,
        )

        model_wrapper = create_model_wrapper(ttnn_model)

        # Create pipeline
        num_inputs_per_round = 2
        host_inputs_sample, dram_memory_config, l1_memory_config = create_host_input_tensors(
            device, batch_size, input_height, input_width, 1
        )

        config = PipelineConfig(
            use_trace=executor_config.use_trace,
            num_command_queues=executor_config.num_command_queues,
            all_transfers_on_separate_command_queue=executor_config.all_transfers_on_separate_command_queue,
        )

        pipeline_args = {
            "config": config,
            "model": model_wrapper,
            "device": device,
            "dram_input_memory_config": dram_memory_config,
            "l1_input_memory_config": l1_memory_config,
        }

        pipe = create_pipeline_from_config(**pipeline_args)
        pipe.compile(host_inputs_sample[0])

        # Run multiple rounds
        total_outputs = []
        for round_num in range(3):
            logger.info(f"Running round {round_num + 1}/3...")
            round_inputs, _, _ = create_host_input_tensors(
                device, batch_size, input_height, input_width, num_inputs_per_round
            )
            round_outputs = pipe.enqueue(round_inputs).pop_all()
            total_outputs.extend(round_outputs)

        pipe.cleanup()

        # Verify we got expected number of outputs
        assert len(total_outputs) == 3 * num_inputs_per_round, f"Expected {3 * num_inputs_per_round} outputs"
        logger.info(f"✅ Multiple rounds test passed for {executor_config.name}!")

    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")
