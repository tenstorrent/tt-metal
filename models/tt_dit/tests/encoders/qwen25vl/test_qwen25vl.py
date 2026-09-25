# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Text-only inference through `Qwen25VlEncoder`.

The checkpoint is a vision-language model, but `Qwen25VlEncoder` keeps only the language model:
`STATE_CONVERSION` drops every `model.visual.*` key, so there is no vision tower to run and the
encoder's interface takes token ids alone. These tests pin both halves of that -- the weights that
survive conversion, and the numerics of a text-only forward against `transformers`.
"""

from __future__ import annotations

import diffusers.pipelines.qwenimage.pipeline_qwenimage
import pytest
import torch
import transformers
from loguru import logger

import ttnn
from models.tt_dit.encoders.qwen25vl import Qwen25VlCheckpoint
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwenimage.text_encoder import (
    PROMPT_DROP_IDX,
    PROMPT_TEMPLATE,
    SEQUENCE_LENGTH,
    TextEncoder,
)
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params_req_exact_devices

CHECKPOINT = "Qwen/Qwen-Image"
SUBFOLDER = "text_encoder"


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
        use_torch=False,
    )

    logger.info("running torch model...")
    with torch.no_grad():
        embeds, mask = torch_pipeline.encode_prompt(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=SEQUENCE_LENGTH,
        )
        embeds = torch.nn.functional.pad(embeds, [0, 0, 0, SEQUENCE_LENGTH - embeds.shape[1]], value=0)
        mask = torch.nn.functional.pad(mask, [0, SEQUENCE_LENGTH - mask.shape[1]], value=0)

    logger.info("running TT model...")
    tt_embeds, tt_mask = text_encoder.encode_cfg(
        prompts,
        prompts,
        num_images_per_prompt=num_images_per_prompt,
        cfg_enabled=False,
        traced=traced,
    )
    assert torch.equal(mask, tt_mask)
    assert_quality(embeds, tt_embeds, pcc=0.988, relative_rmse=0.15)
