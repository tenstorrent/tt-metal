# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import time
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# TTNN implementation
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")


def create_pi0_pipeline_model(ttnn_model, device, inputs):
    """Wrapper to adapt PI0 model inputsfor pipeline interface."""

    def run(pipeline_input):
        with torch.no_grad():
            output_dict = ttnn_model.sample_actions(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
        return output_dict

    return run


def create_config() -> PI0ModelConfig:
    """Create PI0 model configuration."""
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
    )

    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )

    return config


def create_test_inputs(config: PI0ModelConfig, device, batch_size: int = 1):
    """Create test inputs as TTNN device tensors."""
    image_size = config.siglip_config.image_size

    images_torch = [torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32) for _ in range(2)]
    img_masks_torch = [torch.ones(batch_size, dtype=torch.bool) for _ in range(2)]
    lang_tokens_torch = torch.randint(0, 256000, (batch_size, 32))
    lang_masks_torch = torch.ones(batch_size, 32, dtype=torch.bool)
    state_torch = torch.randn(batch_size, config.state_dim, dtype=torch.float32)

    images = [
        ttnn.from_torch(
            img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for img in images_torch
    ]

    img_masks = [
        ttnn.from_torch(
            mask.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for mask in img_masks_torch
    ]

    lang_tokens = ttnn.from_torch(lang_tokens_torch, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    lang_masks = ttnn.from_torch(lang_masks_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    state = ttnn.from_torch(state_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
    }


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 40000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch_size, expected_compile_time, expected_throughput_fps",
    [(1, 2, 3)],
)
def test_perf_pi0_ttnn(device, num_iterations, batch_size, expected_compile_time, expected_throughput_fps):
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    # Create config and inputs
    config = create_config()
    inputs = create_test_inputs(config, device, batch_size=batch_size)

    # Load weights
    weight_loader = PI0WeightLoader(str(checkpoint_path))

    # Initialize models
    model_ttnn = PI0ModelTTNN(config, weight_loader, device)

    image_shape = inputs["images"][0].shape
    dram_grid_size = device.dram_grid_size()

    # Calculate physical 2D dimensions for WIDTH_SHARDED layout
    width = image_shape[-1]
    volume = image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3]
    physical_height = volume // width
    max_cores = dram_grid_size.x

    # Find optimal cores ensuring tile-aligned shards (multiple of 32)
    dram_cores = 1
    for cores in range(max_cores, 0, -1):
        if width % cores == 0 and (width // cores) % 32 == 0:
            dram_cores = cores
            break

    # Create sharded memory configs for DRAM and L1
    shard_width = width // dram_cores
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))}),
        [physical_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    l1_input_memory_config = ttnn.create_sharded_memory_config(
        shape=(physical_height, shard_width),
        core_grid=ttnn.CoreGrid(y=1, x=dram_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    run_model = create_pi0_pipeline_model(model_ttnn, device, inputs)

    # Create pipeline configuration with 2CQ + Trace enabled
    config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipeline = create_pipeline_from_config(
        config,
        run_model,
        device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    # Pipeline API accepts single tensor; wrapper uses actual PI0 inputs from closure
    image_host = inputs["images"][0].cpu()
    host_inputs = [image_host] * num_iterations

    start = time.time()
    pipeline.compile(image_host)
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
        model_name="pi0-2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}",
    )
