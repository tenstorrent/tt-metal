# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Generation for FIBO-vlm through `Vlm`, from text, from an image, and from both."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import transformers
from loguru import logger
from PIL import Image

import ttnn
from models.common.modules.tt_ccl import default_topology
from models.tt_dit.encoders.qwen3vl.model_qwen3vl_v2 import mrope_position_ids
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.fibo.vlm import Vlm
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

CHECKPOINT = "briaai/FIBO-vlm"
MAX_NEW_TOKENS = 32
JSON_PREFIX = '{"short_description":'
IMAGE_PATH = Path(__file__).resolve().parents[4] / "demos" / "multimodal" / "gemma3" / "dog.jpg"


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("prompt", "crop"),
    [
        # `<generate>`, from the prompt alone.
        pytest.param("A red bicycle leaning against a stone wall at sunset.", None, id="text"),
        # `<inspire>`, from the whole 512 x 512 image: 32 x 32 patches, which the tower pads to its
        # limit of 56 x 56.
        pytest.param(None, (0, 0, 512, 512), id="inspire"),
        # `<refine>`, from a crop of the image with editing instructions: 24 x 34 patches, not an
        # even number of tiles.
        pytest.param("Make it a winter scene.", (0, 0, 512, 360), id="refine_cropped"),
    ],
)
def test_generation(
    *, mesh_device: ttnn.MeshDevice, prompt: str | None, crop: tuple[int, int, int, int] | None
) -> None:
    tp_axis = 1

    ccl_manager = CCLManager(mesh_device, topology=default_topology(mesh_device) or ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
    )

    vlm = Vlm(
        checkpoint_name=CHECKPOINT,
        device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        prompt_length=1024,
        cache_length=2048,
    )

    image = Image.open(IMAGE_PATH).convert("RGB").crop(crop) if crop is not None else None
    inputs = vlm._tokenize(prompt, image)
    prompt_len = inputs.tokens.shape[1]
    max_length = prompt_len + MAX_NEW_TOKENS
    image_inputs = (
        {"pixel_values": inputs.pixel_values, "image_grid_thw": inputs.image_grid_thw} if image is not None else {}
    )

    torch_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.float32)

    logger.info("running torch model...")
    with torch.no_grad():
        reference = torch_model.generate(
            input_ids=inputs.tokens,
            attention_mask=torch.ones_like(inputs.tokens),
            mm_token_type_ids=inputs.token_types,
            **image_inputs,
            max_length=max_length,
            do_sample=False,
        )
        # Teacher-forced input is the reference sequence minus its last token; every position from
        # the end of the prompt onward is then predicting a generated token.
        forced = reference[:, :-1]
        forced_token_types = torch.nn.functional.pad(inputs.token_types, [0, forced.shape[1] - prompt_len])
        out = torch_model.forward(
            input_ids=forced,
            attention_mask=torch.ones_like(forced),
            mm_token_type_ids=forced_token_types,
            **image_inputs,
            output_hidden_states=True,
        )
        visual = (
            torch_model.model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw, return_dict=True)
            if image is not None
            else None
        )
    logger.info(f"torch: {vlm._tokenizer.decode(reference[0, prompt_len:])!r}")

    generated_positions = slice(prompt_len - 1, forced.shape[1])

    vision_args = vlm._vision_args(inputs)
    prefill_vision_args = {}
    if visual is not None:
        logger.info("comparing ttnn vision tower...")
        # The tower's output holds the image's rows first, then those of the padding.
        image_rows = visual.pooler_output.shape[0]
        assert_quality(
            visual.pooler_output.float(),
            tensor.to_torch(vision_args["vision_embeds"], mesh_axes=[None, None])[:image_rows],
            pcc=0.99,
        )
        assert len(vision_args["deepstack_embeds"]) == len(visual.deepstack_features)
        for feature, tt_feature in zip(visual.deepstack_features, vision_args["deepstack_embeds"], strict=True):
            assert_quality(feature.float(), tensor.to_torch(tt_feature, mesh_axes=[None, None])[:image_rows], pcc=0.99)

        # The image enters the prefill as it enters `generate`, over the whole forced sequence.
        positions = mrope_position_ids(
            forced_token_types,
            image_grid_thw=inputs.image_grid_thw,
            spatial_merge_size=vlm._vision_tower.spatial_merge_size,
        )
        prefill_vision_args = {
            "positions": tensor.from_torch(positions.float(), device=mesh_device, dtype=ttnn.float32),
            "vision_embeds": vision_args["vision_embeds"],
            "vision_mask": tensor.from_torch(forced_token_types.bool(), device=mesh_device),
            "deepstack_embeds": vision_args["deepstack_embeds"],
        }

    logger.info("running ttnn model, teacher-forced prefill...")
    tt_hidden_states = vlm._encoder.forward(
        tensor.from_torch(forced, device=mesh_device, dtype=ttnn.uint32),
        **prefill_vision_args,
        skip_final_linear=True,
        output_hidden_states=True,
    )
    hidden_states = list(out.hidden_states or [])
    tt_hidden_states_torch = [tensor.to_torch(t) for t in tt_hidden_states]
    assert len(hidden_states) == len(tt_hidden_states_torch)

    for x, tt_x in zip(hidden_states[-4:], tt_hidden_states_torch[-4:], strict=True):
        assert_quality(x[:, generated_positions], tt_x[:, generated_positions], pcc=0.9994, relative_rmse=0.04)

    logger.info("running ttnn model, teacher-forced decode...")
    # `guide` feeds the reference's own tokens back in, so each step's logits line up with the
    # reference logits at the generated positions.
    tt_out = vlm._encoder.generate(
        inputs.tokens,
        mask=None,
        eos_tokens=None,
        max_length=reference.shape[1],
        guide=reference,
        return_logits=True,
        **vision_args,
    )
    assert tt_out.logits is not None

    assert_quality(out.logits[:, generated_positions].float(), tt_out.logits, ccc=0.997, relative_rmse=0.08)

    logger.info("running ttnn model, free-running decode after warm-up...")
    vlm.warm_up(traced=False)
    programs = mesh_device.num_program_cache_entries()
    output = vlm.generate_raw(prompt, image=image, seed=0, traced=False, max_length=max_length)
    logger.info(f"ttnn:  {output.text!r}")

    # Warm-up compiled every program generation takes, for prompts and images of every size.
    assert mesh_device.num_program_cache_entries() == programs

    # Decode did its job if it produced the JSON FIBO consumes.
    assert output.text.startswith(JSON_PREFIX)
    assert output.prompt_tokens == prompt_len
