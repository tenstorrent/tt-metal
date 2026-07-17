# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

import ttnn

from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.flux2.prompt_encoder import PromptEncoder, _format_input
from ....pipelines.flux2.system_messages import SYSTEM_MESSAGE_UPSAMPLING_I2I
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((4, 8), id="4x8"),
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
    tt_encoder.load_weights()

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

    assert_quality(torch_embeds, tt_embeds, pcc=0.9997, relative_rmse=0.04)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((4, 8), id="4x8"),
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

    tt_encoder.load_weights()

    logger.info("running TT model...")
    tt_output = tt_encoder.upsample(prompts, max_length=224, temperature=temperature)
    print(tt_output)


def test_format_input_i2i(tt_dit_cache_dir) -> None:
    from diffusers.pipelines.flux2.pipeline_flux2 import format_input as diffusers_format_input
    from PIL import Image

    prompts = ["Make the sky purple at sunset"]
    image = Image.new("RGB", (64, 64), color=(1, 2, 3))

    ref = diffusers_format_input(prompts, system_message=SYSTEM_MESSAGE_UPSAMPLING_I2I, images=[[image]])
    ours = _format_input(prompts, system_message=SYSTEM_MESSAGE_UPSAMPLING_I2I, images=[image])

    assert len(ref) == len(ours) == 1
    assert len(ref[0]) == len(ours[0])
    for ref_msg, our_msg in zip(ref[0], ours[0], strict=True):
        assert ref_msg["role"] == our_msg["role"]
        ref_types = [part["type"] for part in ref_msg["content"]]
        our_types = [part["type"] for part in our_msg["content"]]
        assert ref_types == our_types
        for ref_part, our_part in zip(ref_msg["content"], our_msg["content"], strict=True):
            if ref_part["type"] == "text":
                assert ref_part["text"] == our_part["text"]


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((4, 8), id="4x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_upsample_i2i(tt_dit_cache_dir, mesh_device: ttnn.MeshDevice) -> None:
    from PIL import Image

    tp_axis = 1
    checkpoint_name = "black-forest-labs/FLUX.2-dev"
    prompts = ["Make the sky purple at sunset"]
    temperature = 0.15
    image = Image.new("RGB", (256, 256), color=(10, 20, 30))

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
    tt_encoder.load_weights()

    logger.info("running TT model (img2img upsample)...")
    tt_output = tt_encoder.upsample(
        prompts,
        max_length=224,
        temperature=temperature,
        images=[image],
    )

    assert len(tt_output) == 1
    assert isinstance(tt_output[0], str)
    assert len(tt_output[0]) > 0
