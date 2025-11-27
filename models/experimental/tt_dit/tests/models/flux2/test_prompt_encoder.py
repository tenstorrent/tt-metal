# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

import ttnn

from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.flux2.prompt_encoder import PromptEncoder
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 8), id="1x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_encode(mesh_device: ttnn.MeshDevice) -> None:
    tp_axis = 1
    sequence_length = 64
    checkpoint_name = "black-forest-labs/FLUX.2-dev"
    prompts = ["A painting", "A futuristic city skyline at sunset during winter"]

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
    )

    tt_encoder = PromptEncoder(
        checkpoint_name=checkpoint_name,
        use_torch_encoder=False,
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    torch_encoder = PromptEncoder(
        checkpoint_name=checkpoint_name,
        use_torch_encoder=True,
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    logger.info("running Torch model...")
    torch_embeds, torch_mask = torch_encoder.encode(prompts, num_images_per_prompt=2, sequence_length=sequence_length)

    logger.info("running TT model...")
    tt_embeds, tt_mask = tt_encoder.encode(prompts, num_images_per_prompt=2, sequence_length=sequence_length)

    assert torch_mask.equal(tt_mask)
    print(torch_mask.sum(dim=1))

    assert_quality(torch_embeds, tt_embeds, pcc=0.9997, relative_rmse=0.04)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 8), id="1x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_upsample(mesh_device: ttnn.MeshDevice) -> None:
    tp_axis = 1
    checkpoint_name = "black-forest-labs/FLUX.2-dev"
    prompts = ["A futuristic city skyline at sunset"]
    temperature = 0.15

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
    )

    tt_encoder = PromptEncoder(
        checkpoint_name=checkpoint_name,
        use_torch_encoder=False,
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    logger.info("running TT model...")
    tt_output = tt_encoder.upsample(prompts, max_length=224, temperature=temperature)
    print(tt_output)
