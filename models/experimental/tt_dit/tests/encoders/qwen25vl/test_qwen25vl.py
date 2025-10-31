# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers as reference
import pytest
import torch
import ttnn
from loguru import logger

from ....encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "num_links"),
    [
        pytest.param((2, 4), (2, 4), 1, 1, id="2x4sp0tp1"),
        pytest.param((4, 8), (4, 8), 1, 4, id="4x8sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("prompts", "num_images_per_prompt"),
    [
        (["hello", "meow"], 4),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_qwen25vl(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    tp_axis: int,
    num_links: int,
    prompts: list[str],
    num_images_per_prompt: int,
) -> None:
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    tp_factor = tuple(submesh_device.shape)[tp_axis]

    checkpoint = "Qwen/Qwen-Image"
    max_sequence_length = 512

    torch_pipeline = reference.QwenImagePipeline.from_pretrained(checkpoint)
    assert isinstance(torch_pipeline, reference.QwenImagePipeline)

    ccl_manager = CCLManager(
        mesh_device=submesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    tt_encoder_pair = Qwen25VlTokenizerEncoderPair(
        checkpoint,
        tokenizer_subfolder="tokenizer",
        encoder_subfolder="text_encoder",
        max_sequence_length=max_sequence_length,
        max_batch_size=len(prompts) * num_images_per_prompt,
        device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        ),
        use_torch=False,
    )

    logger.info("running torch model...")
    with torch.no_grad():
        embeds, mask = torch_pipeline.encode_prompt(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    logger.info("running TT model...")
    tt_embeds, tt_mask = tt_encoder_pair.encode(prompts, num_images_per_prompt=num_images_per_prompt)

    assert_quality(embeds, tt_embeds, pcc=0, relative_rmse=1)
    assert_quality(mask, tt_mask, pcc=0, relative_rmse=1)
