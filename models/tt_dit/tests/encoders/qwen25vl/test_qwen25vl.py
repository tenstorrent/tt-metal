# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Inference through `Qwen25VlEncoder`, with and without an image, against `transformers`.

`Qwen25VlEncoder` keeps only the language model; `STATE_CONVERSION` drops every `model.visual.*`
key. An image runs through the separate `Qwen25VlVisionModel`, whose tokens the encoder splices in
at the `<|image_pad|>` rows.
"""

from __future__ import annotations

import diffusers.pipelines.qwenimage.pipeline_qwenimage
import PIL.Image
import pytest
import torch
import transformers
from loguru import logger

import ttnn
from models.tt_dit.encoders.mrope import mrope_position_ids
from models.tt_dit.encoders.qwen25vl import Qwen25VlCheckpoint
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwenimage.text_encoder import (
    EDIT_FORMAT,
    EDIT_PLUS_FORMAT,
    PROMPT_DROP_IDX,
    PROMPT_TEMPLATE,
    PromptFormat,
    TextEncoder,
)
from models.tt_dit.tests.encoders.qwen25vl.test_qwen25vl_vision import EDIT_CHECKPOINT, photo
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params_req_exact_devices

CHECKPOINT = "Qwen/Qwen-Image"
SUBFOLDER = "text_encoder"
SEQUENCE_LENGTH_BUCKETS = (128, 256, 512)


@pytest.mark.parametrize(
    ("mesh_device", "tp", "fsdp"),
    [
        pytest.param((1, 2), (2, 1), None, id="1x2"),
        pytest.param((1, 4), (4, 1), None, id="1x4"),
        pytest.param((1, 8), (8, 1), None, id="1x8"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "masked",
    [
        pytest.param(True, id="masked"),
        pytest.param(False, id="unmasked"),
    ],
)
def test_text_only_forward(
    *, mesh_device: ttnn.MeshDevice, tp: tuple[int, int], fsdp: tuple[int, int] | None, masked: bool
) -> None:
    """A prompt of token ids alone reproduces the reference hidden states, with no image input."""
    torch.manual_seed(0)

    batch_size = 1
    sequence_length = 512

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig.from_tuples(tp=tp, sp=None, fsdp=fsdp)

    torch_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(CHECKPOINT, subfolder=SUBFOLDER)
    text_config = torch_model.config.text_config

    model = Qwen25VlCheckpoint(CHECKPOINT, subfolder=SUBFOLDER).build(
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    tokens = torch.randint(0, text_config.vocab_size, [batch_size, sequence_length])
    lengths = torch.randint(sequence_length // 4, 3 * sequence_length // 4, [batch_size])
    mask = torch.arange(sequence_length).flip([0]) < lengths.unsqueeze(1) if masked else None

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_mask = tensor.from_torch(mask, device=mesh_device) if mask is not None else None

    logger.info("running ttnn model...")
    tt_hidden_states = model.forward(
        tt_tokens,
        mask=tt_mask,
        skip_final_linear=True,
        output_hidden_states=True,
    )
    tt_hidden_states_torch = [tensor.to_torch(t) for t in tt_hidden_states]

    logger.info("running torch model...")
    with torch.no_grad():
        out = torch_model.forward(
            input_ids=tokens,
            attention_mask=mask if mask is not None else torch.ones_like(tokens),
            output_hidden_states=True,
        )
    assert not isinstance(out, tuple)
    hidden_states = list(out.hidden_states or [])

    if mask is not None:
        # Masked positions at the start of the sequence hold undefined values from a softmax over
        # all -inf, and we number padding positions differently from transformers, so compare only
        # the real tokens.
        _, _, d = hidden_states[0].shape
        hidden_states = [t.masked_select(mask.unsqueeze(-1)).view([-1, d]) for t in hidden_states]
        tt_hidden_states_torch = [t.masked_select(mask.unsqueeze(-1)).view([-1, d]) for t in tt_hidden_states_torch]

    assert len(hidden_states) == len(tt_hidden_states_torch)

    # Error accumulates with depth, so the last hidden state sets the bound for all four: it
    # measures 99.63 % / 8.6 % in the worst configuration, four layers up 99.95 % / 3.1 %.
    for x, tt_x in zip(hidden_states[-4:], tt_hidden_states_torch[-4:], strict=True):
        assert_quality(x, tt_x, pcc=0.996, relative_rmse=0.09)


# Qwen-Image-Edit's template, with the image before the prompt.
EDIT_PROMPT_TEMPLATE = EDIT_FORMAT.template.format(f"{EDIT_FORMAT.image_prompt}{{}}")


@pytest.mark.parametrize(
    ("mesh_device", "tp"),
    [
        pytest.param((1, 4), (4, 1), id="1x4"),
        pytest.param((1, 8), (8, 1), id="1x8"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("size", [1024, 384])
def test_image_prompt_forward(*, mesh_device: ttnn.MeshDevice, tp: tuple[int, int], size: int) -> None:
    """An edit prompt with a photo, through the vision tower and the encoder, against `transformers`.

    1024 x 1024 is the size Qwen-Image-Edit passes to the encoder, 384 x 384 the size of each
    condition image of Qwen-Image-Edit-2509 and 2511.
    """
    processor = transformers.Qwen2VLProcessor.from_pretrained(EDIT_CHECKPOINT, subfolder="processor")
    inputs = processor(
        text=[EDIT_PROMPT_TEMPLATE.format("Make the dog wear a red scarf")],
        images=[photo(size)],
        return_tensors="pt",
    )
    tokens, pixel_values, image_grid_thw = inputs.input_ids, inputs.pixel_values, inputs.image_grid_thw

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig.from_tuples(tp=tp, sp=None, fsdp=None)

    checkpoint = Qwen25VlCheckpoint(CHECKPOINT, subfolder=SUBFOLDER)
    tower = checkpoint.build_vision(device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager)
    model = checkpoint.build(device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager)

    vision_inputs = tower.prepare_inputs(pixel_values, image_grid_thw)
    vision_embeds = tower.forward(
        tensor.from_torch(vision_inputs.patches, device=mesh_device),
        rope=(
            tensor.from_torch(vision_inputs.cos, device=mesh_device),
            tensor.from_torch(vision_inputs.sin, device=mesh_device),
        ),
        reverse_index=tensor.from_torch(
            vision_inputs.reverse_index.unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        cu_seqlens=vision_inputs.cu_seqlens,
        cu_window_seqlens=vision_inputs.cu_window_seqlens,
    )

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_mask = (tokens == image_token_id).int()
    position_ids = mrope_position_ids(
        vision_mask, image_grid_thw=image_grid_thw, spatial_merge_size=tower.spatial_merge_size
    )

    logger.info("running ttnn model...")
    tt_hidden = model.forward(
        tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32),
        positions=tensor.from_torch(position_ids.float(), device=mesh_device, dtype=ttnn.float32),
        vision_embeds=vision_embeds,
        vision_mask=tensor.from_torch(vision_mask, device=mesh_device),
        skip_final_linear=True,
    )
    tt_hidden = tensor.to_torch(tt_hidden)

    logger.info("running torch model...")
    # A bfloat16 reference would be about as far from float32 as the device is; see the bounds below.
    torch_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CHECKPOINT, subfolder=SUBFOLDER, dtype=torch.float32
    )
    with torch.no_grad():
        out = torch_model.forward(
            input_ids=tokens,
            attention_mask=inputs.attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            # Without it, transformers 5 falls back to sequential positions for the image tokens
            # instead of the multimodal rope positions; diffusers' edit pipelines leave it out.
            mm_token_type_ids=inputs.mm_token_type_ids,
            output_hidden_states=True,
        )
    hidden = out.hidden_states[-1]

    # The pipeline drops the template prefix, so compare the image and prompt tokens on their own too.
    image_rows = vision_mask[0].bool()
    # The image rows are sensitive to rounding: transformers itself in bfloat16 reaches 96.9 % (384²)
    # and 92.3 % (1024²) on them. This measured 96.6-97.5 % and 89.8-90.6 %, text rows 99.8 %.
    image_pcc, all_pcc = {384: (0.96, 0.975), 1024: (0.88, 0.89)}[size]
    assert_quality(hidden[0, image_rows], tt_hidden[0, image_rows], pcc=image_pcc)
    assert_quality(hidden[0, ~image_rows], tt_hidden[0, ~image_rows], pcc=0.995)
    assert_quality(hidden, tt_hidden, pcc=all_pcc)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"),
    [
        pytest.param((2, 2), (1, 2), id="2x2_1x2"),
        pytest.param((2, 4), (1, 4), id="2x4_1x4"),
        pytest.param((4, 8), (1, 4), id="4x8_1x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({**line_params_req_exact_devices, "trace_region_size": 31000000}, id="line")],
    indirect=True,
)
@pytest.mark.parametrize("traced", [pytest.param(True, id="traced")])
def test_qwen25vl_encoder_pair(*, mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], traced: bool) -> None:
    """The QwenImage text encoder wrapper against the prompt encoding of the diffusers pipeline."""
    # There is a bug in the HF implementation where the prompt_embeds_mask is incorrectly repeated
    # if num_images_per_prompt != 1.
    # https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L262
    # is
    # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    # but should be
    # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
    num_images_per_prompt = 1
    prompts = ["", "Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot"]

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    # Only the tokenizer and text encoder are needed for prompt encoding.
    torch_pipeline = diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline.from_pretrained(
        CHECKPOINT, transformer=None, vae=None
    )
    assert torch_pipeline.prompt_template_encode == PROMPT_TEMPLATE
    assert torch_pipeline.prompt_template_encode_start_idx == PROMPT_DROP_IDX

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=submesh_shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device=submesh_device, num_links=1, topology=ttnn.Topology.Linear)

    text_encoder = TextEncoder(
        checkpoint_name=CHECKPOINT,
        device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        sequence_length_buckets=SEQUENCE_LENGTH_BUCKETS,
        use_torch=False,
    )

    logger.info("running TT model...")
    [tt_embeds] = text_encoder.encode_cfg(
        prompts,
        prompts,
        num_images_per_prompt=num_images_per_prompt,
        cfg_enabled=False,
        batch_passes=False,
        traced=traced,
    )
    length = tt_embeds.shape[1]
    assert length in SEQUENCE_LENGTH_BUCKETS

    logger.info("running torch model...")
    with torch.no_grad():
        embeds, mask = torch_pipeline.encode_prompt(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max(SEQUENCE_LENGTH_BUCKETS),
        )
        embeds = torch.nn.functional.pad(embeds, [0, 0, 0, length - embeds.shape[1]], value=0)
        mask = torch.nn.functional.pad(mask, [0, length - mask.shape[1]], value=0)

    assert torch.all(tt_embeds[mask == 0] == 0), "padded positions must be zero"
    assert_quality(embeds, tt_embeds, pcc=0.988, relative_rmse=0.15)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"),
    [
        pytest.param((2, 2), (1, 2), id="2x2_1x2"),
        pytest.param((2, 4), (1, 4), id="2x4_1x4"),
        pytest.param((4, 8), (1, 4), id="4x8_1x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [line_params_req_exact_devices], indirect=True)
@pytest.mark.parametrize(
    ("checkpoint", "prompt_format", "image_sizes", "buckets", "pcc"),
    [
        # Qwen-Image-Edit shows the image at the size of its latents. The image rows are sensitive
        # to rounding at 1024² (see `test_image_prompt_forward`).
        pytest.param(EDIT_CHECKPOINT, EDIT_FORMAT, [(1024, 1024)], (1408, 1536, 1920), 0.88, id="edit"),
        # Edit-2509 and 2511 show each image at about 384². Torch at bfloat16 reaches 0.945 against
        # float32 on this input, and 0.90 on the one above.
        pytest.param(
            "Qwen/Qwen-Image-Edit-2509", EDIT_PLUS_FORMAT, [(384, 384), (512, 288)], (256, 512, 1024), 0.93, id="plus"
        ),
    ],
)
def test_edit_text_encoder(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    checkpoint: str,
    prompt_format: PromptFormat,
    image_sizes: list[tuple[int, int]],
    buckets: tuple[int, ...],
    pcc: float,
) -> None:
    """The edit prompt encoding of the QwenImage text encoder wrapper against `transformers`.

    The reference is what the diffusers edit pipelines compute, but with the multimodal rope
    positions, which they lose with transformers 5 (they don't pass `mm_token_type_ids`).
    """
    prompt = "Make the dog wear a red scarf"
    # The second image is the photo mirrored, so that the two differ.
    images = [photo(max(size)).resize(size) for size in image_sizes]
    images[1:] = [im.transpose(PIL.Image.Transpose.FLIP_LEFT_RIGHT) for im in images[1:]]

    reference_pipeline = (
        diffusers.QwenImageEditPipeline if prompt_format is EDIT_FORMAT else diffusers.QwenImageEditPlusPipeline
    )(scheduler=None, vae=None, text_encoder=None, tokenizer=None, processor=None, transformer=None)
    expected_template = EDIT_PROMPT_TEMPLATE if prompt_format is EDIT_FORMAT else prompt_format.template
    assert reference_pipeline.prompt_template_encode == expected_template
    assert reference_pipeline.prompt_template_encode_start_idx == prompt_format.drop_idx

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=submesh_shape[1], mesh_axis=1))
    ccl_manager = CCLManager(mesh_device=submesh_device, num_links=1, topology=ttnn.Topology.Linear)
    text_encoder = TextEncoder(
        checkpoint_name=checkpoint,
        device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        sequence_length_buckets=buckets,
        use_torch=False,
        prompt_format=prompt_format,
    )

    logger.info("running TT model...")
    [tt_embeds] = text_encoder.encode_cfg(
        [prompt],
        [prompt],
        num_images_per_prompt=1,
        cfg_enabled=False,
        batch_passes=False,
        traced=False,
        images=images,
    )
    length = text_encoder.prompt_length(prompt, images=images)
    assert tt_embeds.shape[1] == min(b for b in buckets if b >= length)
    assert torch.all(tt_embeds[:, length:] == 0), "padded positions must be zero"

    logger.info("running torch model...")
    processor = transformers.Qwen2VLProcessor.from_pretrained(checkpoint, subfolder="processor")
    announcement = "".join(prompt_format.image_prompt.format(i + 1) for i in range(len(images)))
    inputs = processor(text=[prompt_format.template.format(announcement + prompt)], images=images, return_tensors="pt")
    torch_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint, subfolder=SUBFOLDER, dtype=torch.float32
    )
    with torch.no_grad():
        out = torch_model.forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            mm_token_type_ids=inputs.mm_token_type_ids,
            output_hidden_states=True,
        )
    embeds = out.hidden_states[-1][:, prompt_format.drop_idx :]
    assert embeds.shape[1] == length

    image_rows = inputs.mm_token_type_ids[0, prompt_format.drop_idx :].bool()
    assert_quality(embeds[0, ~image_rows], tt_embeds[0, :length][~image_rows], pcc=0.99)
    assert_quality(embeds, tt_embeds[:, :length], pcc=pcc)
