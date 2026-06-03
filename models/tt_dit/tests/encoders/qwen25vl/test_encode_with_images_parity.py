# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end regression test for Qwen25VlTokenizerEncoderPair.encode_with_images.

Runs the torch reference Qwen2_5_VLForConditionalGeneration and our device
implementation on the same (text, image) inputs, then checks per-region PCC on
the last hidden state after applying the ``attention_mask``-based masking that
both paths perform. The two regions are:

- **Text tokens**: positions with ``input_ids != image_token_id``. These go
  through token embedding + LLM only; they should match the torch reference at
  high PCC (comparable to ``test_qwen25vl_text_encoder`` which reports 0.952
  masked / 0.991 unmasked).

- **Image tokens**: positions with ``input_ids == image_token_id``. These are
  replaced by vision-encoder features before the LLM, so they reflect the
  composition of 32-block ViT bf16 accumulation + 28-block LLM drift, and are
  known to land well below the text PCC. We enforce a regression floor here
  rather than a quality threshold; see the Risk #1 note in
  ``plans/device_host_removal_*.md``.

The whole-tensor PCC is also reported for reference but not asserted, because
it is dominated by the image-token region and so conflates the two concerns.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import transformers
from loguru import logger
from PIL import Image
from transformers import Qwen2VLProcessor

import ttnn

from ....encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager

# Same template the pipeline uses for edit prompts (includes the image_pad tokens
# so the processor emits the grid_thw needed by the vision tower).
PROMPT_TEMPLATE_EDIT = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
    "<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _make_formatted_prompt(user_text: str) -> str:
    image_prefix = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    return PROMPT_TEMPLATE_EDIT.format(image_prefix + user_text)


def _make_condition_image(size: int = 384, seed: int = 0) -> Image.Image:
    """Load a real reference image if ``PARITY_IMAGE`` is set; otherwise random pixels.

    A trained ViT behaves very differently on natural images than on random noise,
    so parity measurements should use a real image to avoid OOD drift.
    """
    env_path = os.environ.get("PARITY_IMAGE")
    if env_path and Path(env_path).exists():
        img = Image.open(env_path).convert("RGB").resize((size, size), Image.LANCZOS)
        return img
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.mark.parametrize(
    "mesh_device, submesh_shape",
    [
        pytest.param((4, 8), (1, 4), id="4x8_1x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_encode_with_images_parity(*, mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int]) -> None:
    torch.manual_seed(0)

    checkpoint = "Qwen/Qwen-Image-Edit-2511"
    tp_axis = 1
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=submesh_device.shape[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device=submesh_device, num_links=1, topology=ttnn.Topology.Linear)

    logger.info("loading VL processor...")
    vl_processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    logger.info("building TT encoder pair (with vision encoder)...")
    encoder_pair = Qwen25VlTokenizerEncoderPair(
        checkpoint,
        tokenizer_subfolder="tokenizer",
        encoder_subfolder="text_encoder",
        device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        use_torch=False,
        is_fsdp=True,
        build_vision_encoder=True,
    )

    logger.info("loading torch VL reference...")
    torch_vl_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint, subfolder="text_encoder"
    )
    torch_vl_model.eval()

    formatted_prompts = [_make_formatted_prompt("A dog running in the grass.")]
    images = [_make_condition_image(size=384, seed=123)]

    logger.info("running torch reference...")
    with torch.no_grad():
        model_inputs = vl_processor(
            text=formatted_prompts,
            images=images * len(formatted_prompts),
            padding=True,
            return_tensors="pt",
        )
        outputs = torch_vl_model(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values.to(torch_vl_model.dtype),
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )
        ref_hidden = outputs.hidden_states[-1].to(torch.float32)
        ref_mask = model_inputs.attention_mask
        # encode_with_images multiplies the last hidden state by the attention mask
        # (per-position 0/1 values) before returning, so apply the same transform
        # to the reference to make the two tensors directly comparable.
        ref_hidden = ref_hidden * ref_mask.to(torch.float32).unsqueeze(-1)

    image_token_id = int(torch_vl_model.config.image_token_id)
    input_ids_cpu = model_inputs.input_ids
    del torch_vl_model

    logger.info("running device encode_with_images...")
    tt_hidden, tt_mask = encoder_pair.encode_with_images(
        formatted_prompts,
        images * len(formatted_prompts),
        num_images_per_prompt=1,
    )

    logger.info(f"ref_hidden.shape={tuple(ref_hidden.shape)}  tt_hidden.shape={tuple(tt_hidden.shape)}")
    assert (
        tt_hidden.shape == ref_hidden.shape
    ), f"shape mismatch: ref={tuple(ref_hidden.shape)} tt={tuple(tt_hidden.shape)}"
    assert torch.equal(tt_mask, ref_mask), "attention mask mismatch between torch ref and device encoder"

    tt_hidden_f32 = tt_hidden.to(torch.float32)

    # Per-region PCC diagnostics so we can tell whether drift is concentrated in the
    # image-token positions (vision-encoder splice issue) or spread across text tokens
    # (LLM bf16 accumulation).
    image_pos_mask = input_ids_cpu == image_token_id
    text_pos_mask = (~image_pos_mask) & ref_mask.bool()

    def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.detach().flatten().to(torch.float64)
        b = b.detach().flatten().to(torch.float64)
        cov = torch.cov(torch.stack([a, b])).numpy()
        std_a = cov[0, 0] ** 0.5
        std_b = cov[1, 1] ** 0.5
        if std_a == 0 or std_b == 0:
            return float("nan")
        return float(cov[0, 1] / (std_a * std_b))

    assert image_pos_mask.any(), "test input must include image tokens"
    assert text_pos_mask.any(), "test input must include non-image text tokens"

    img_pcc = _pcc(ref_hidden[image_pos_mask], tt_hidden_f32[image_pos_mask])
    txt_pcc = _pcc(ref_hidden[text_pos_mask], tt_hidden_f32[text_pos_mask])
    overall_pcc = _pcc(ref_hidden[ref_mask.bool()], tt_hidden_f32[ref_mask.bool()])
    n_img = int(image_pos_mask.sum().item())
    n_txt = int(text_pos_mask.sum().item())
    logger.info(
        f"per-region PCC: image-tokens={img_pcc * 100:.2f}%  text-tokens={txt_pcc * 100:.2f}%  "
        f"overall={overall_pcc * 100:.2f}%"
    )
    logger.info(f"image tokens: {n_img}   text tokens: {n_txt}")

    # Regression floors. Text tokens should match the LLM-only parity bound well.
    # Image tokens are drift-prone due to bf16 accumulation through the vision ViT;
    # the floor here is set below the currently observed value so a future change
    # that meaningfully regresses vision-encoder precision will fail the test.
    TEXT_PCC_FLOOR = 0.90
    IMAGE_PCC_FLOOR = 0.35

    assert (
        txt_pcc >= TEXT_PCC_FLOOR
    ), f"text-token PCC regressed below floor: {txt_pcc * 100:.2f}% < {TEXT_PCC_FLOOR * 100:.2f}%"
    assert img_pcc >= IMAGE_PCC_FLOOR, (
        f"image-token PCC regressed below floor: {img_pcc * 100:.2f}% < {IMAGE_PCC_FLOOR * 100:.2f}%. "
        f"Likely cause: vision-encoder precision loss. Either revert the change that caused this "
        f"or lower the floor if the regression is intentional and the downstream pipeline gate still passes."
    )
