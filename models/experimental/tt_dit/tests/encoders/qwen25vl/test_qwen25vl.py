# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers.pipelines.qwenimage.pipeline_qwenimage as reference
import pytest
import torch
import ttnn
from loguru import logger

from ....encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"),
    [
        pytest.param((1, 1), (1, 1), id="1x1"),
        # pytest.param((2, 4), (2, 4), id="2x4sp0tp1"),
        # pytest.param((4, 8), (4, 8), id="4x8sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "prompts",
    [
        ["hello", "meow"],
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
    prompts: list[str],
) -> None:
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    # There is a bug in the HF implementation where the prompt_embeds_mask is incorrectly repeated
    # if num_images_per_prompt != 1.
    # https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L262
    # is
    # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    # but should be
    # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
    num_images_per_prompt = 1

    pipeline_checkpoint = "Qwen/Qwen-Image"
    text_model_checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_sequence_length = 512

    torch_pipeline = reference.QwenImagePipeline.from_pretrained(pipeline_checkpoint)
    assert isinstance(torch_pipeline, reference.QwenImagePipeline)

    tt_encoder_pair = Qwen25VlTokenizerEncoderPair(
        text_model_checkpoint,
        max_sequence_length=max_sequence_length,
        max_batch_size=len(prompts) * num_images_per_prompt,
        device=submesh_device,
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

    assert_quality(embeds, tt_embeds, pcc=1, relative_rmse=0)
    assert_quality(mask, tt_mask, pcc=1, relative_rmse=0)
