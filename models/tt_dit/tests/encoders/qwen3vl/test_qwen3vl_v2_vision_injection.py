# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""How vision reaches the Qwen3-VL decoder, against transformers on a small random model: the
tower's merged tokens replace the `<|image_pad|>` row embeddings before the block stack, and its
deepstack features are added to those rows after each of the first few decoder layers."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import transformers
from loguru import logger

import ttnn
from models.tt_dit.encoders.qwen3vl.model_qwen3vl_v2 import Qwen3VlEncoder, mrope_position_ids
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

VOCAB_SIZE = 256
HIDDEN = 128
SPATIAL_MERGE_SIZE = 2
# A prompt of text around one image, at a length that is not a multiple of the tile size.
TEXT_BEFORE = 5
IMAGE_GRID = (1, 8, 4)  # 8 tokens after the 2x2 merge
TEXT_AFTER = 37
SEQ = TEXT_BEFORE + IMAGE_GRID[1] * IMAGE_GRID[2] // SPATIAL_MERGE_SIZE**2 + TEXT_AFTER
EXTRA = 8  # tokens generated after the prompt

MESH = [
    pytest.param((1, 1), (1, 0), {"l1_small_size": 32768}, id="1x1"),
    pytest.param((1, 4), (4, 1), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}, id="1x4"),
]


def _reference_text_model(layers: int) -> transformers.PreTrainedModel:
    config = transformers.Qwen3VLTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
        },
    )
    torch.manual_seed(0)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel._from_config(config).eval()


def _encoder(
    reference: transformers.PreTrainedModel, mesh_device: ttnn.MeshDevice, tp: tuple[int, int], *, head: bool = False
) -> Qwen3VlEncoder:
    """The encoder of `reference`, with a language-model head tied to the token embedding if `head`."""
    parallel_config = EncoderParallelConfig.from_tuples(tp=tp, sp=None)
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    config = dataclasses.replace(Qwen3VlEncoder.config_from_hf(reference.config), final_linear=head)
    encoder = Qwen3VlEncoder(config, device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager)
    state = {f"model.language_model.{k}": v for k, v in reference.state_dict().items()}
    if head:
        state["lm_head.weight"] = state["model.language_model.embed_tokens.weight"]
    encoder.load_torch_state_dict(Qwen3VlEncoder.convert_state(state))
    return encoder


def _prompt() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token ids, the vision-row mask and the multimodal rope positions of the prompt."""
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB_SIZE, [1, SEQ])
    num_image_tokens = IMAGE_GRID[1] * IMAGE_GRID[2] // SPATIAL_MERGE_SIZE**2
    mask = torch.zeros([1, SEQ], dtype=torch.bool)
    mask[:, TEXT_BEFORE : TEXT_BEFORE + num_image_tokens] = True
    position_ids = mrope_position_ids(
        mask.long(), image_grid_thw=torch.tensor([IMAGE_GRID]), spatial_merge_size=SPATIAL_MERGE_SIZE
    )
    return ids, mask, position_ids


@pytest.mark.parametrize(("mesh_device", "tp", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_text_only_path_is_unchanged(*, mesh_device: ttnn.MeshDevice, tp: tuple[int, int]) -> None:
    reference = _reference_text_model(2)
    encoder = _encoder(reference, mesh_device, tp)
    ids, _, _ = _prompt()
    tt_ids = tensor.from_torch(ids, device=mesh_device, dtype=ttnn.uint32)

    plain = encoder.forward(tt_ids, skip_final_linear=True)
    explicit_none = encoder.forward(
        tt_ids, vision_embeds=None, vision_mask=None, deepstack_embeds=(), skip_final_linear=True
    )
    assert torch.equal(tensor.to_torch(plain), tensor.to_torch(explicit_none))


@pytest.mark.parametrize(("mesh_device", "tp", "device_params"), MESH[:1], indirect=["mesh_device", "device_params"])
def test_vision_arguments_must_be_paired(*, mesh_device: ttnn.MeshDevice, tp: tuple[int, int], expect_error) -> None:
    reference = _reference_text_model(1)
    encoder = _encoder(reference, mesh_device, tp)
    ids, mask, _ = _prompt()
    tt_ids = tensor.from_torch(ids, device=mesh_device, dtype=ttnn.uint32)
    tt_mask = tensor.from_torch(mask, device=mesh_device)
    embeds = tensor.from_torch(torch.zeros(int(mask.sum()), HIDDEN), device=mesh_device)

    with expect_error(ValueError, "must be passed together"):
        encoder.forward(tt_ids, vision_embeds=embeds, skip_final_linear=True)
    with expect_error(ValueError, "must be passed together"):
        encoder.forward(tt_ids, vision_mask=tt_mask, skip_final_linear=True)
    with expect_error(ValueError, "needs vision_mask"):
        encoder.forward(tt_ids, deepstack_embeds=[embeds], skip_final_linear=True)


@pytest.mark.parametrize(("mesh_device", "tp", "device_params"), MESH, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("num_deepstack", [pytest.param(0, id="no_deepstack"), pytest.param(2, id="deepstack2")])
def test_vision_injection_matches_reference(
    *, mesh_device: ttnn.MeshDevice, tp: tuple[int, int], num_deepstack: int
) -> None:
    reference = _reference_text_model(4)
    encoder = _encoder(reference, mesh_device, tp)
    ids, mask, position_ids = _prompt()
    num_image_tokens = int(mask.sum())

    torch.manual_seed(2)
    vision_embeds = torch.randn(num_image_tokens, HIDDEN)
    deepstack_embeds = [torch.randn(num_image_tokens, HIDDEN) for _ in range(num_deepstack)]

    logger.info("running torch model...")
    with torch.no_grad():
        inputs_embeds = reference.embed_tokens(ids)
        inputs_embeds[mask] = vision_embeds
        out = reference.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=mask,
            deepstack_visual_embeds=deepstack_embeds or None,
        )
    expected = out.last_hidden_state

    logger.info("running ttnn model...")
    tt_out = encoder.forward(
        tensor.from_torch(ids, device=mesh_device, dtype=ttnn.uint32),
        positions=tensor.from_torch(position_ids.float(), device=mesh_device, dtype=ttnn.float32),
        vision_embeds=tensor.from_torch(vision_embeds, device=mesh_device),
        vision_mask=tensor.from_torch(mask, device=mesh_device),
        deepstack_embeds=[tensor.from_torch(e, device=mesh_device) for e in deepstack_embeds],
        skip_final_linear=True,
    )
    actual = tensor.to_torch(tt_out)

    assert actual.shape == expected.shape
    assert_quality(expected, actual, pcc=0.995)

    # Vision rows must have landed exactly where the reference put them: a text-only forward differs
    # from the injected one everywhere from the first vision row onward, and nowhere before it.
    text_only = tensor.to_torch(
        encoder.forward(tensor.from_torch(ids, device=mesh_device, dtype=ttnn.uint32), skip_final_linear=True)
    )
    assert torch.equal(text_only[:, :TEXT_BEFORE], actual[:, :TEXT_BEFORE])
    assert not torch.allclose(text_only[:, TEXT_BEFORE:], actual[:, TEXT_BEFORE:], atol=1e-2)


@pytest.mark.parametrize(("mesh_device", "tp", "device_params"), MESH[:1], indirect=["mesh_device", "device_params"])
def test_generation_after_image_matches_reference(*, mesh_device: ttnn.MeshDevice, tp: tuple[int, int]) -> None:
    """Teacher-forced decoding after an image prompt.

    The generated text continues from the largest prompt position, which the image leaves behind
    the sequence length, so the reference is the model over the whole sequence with its positions.
    """
    reference = _reference_text_model(4)
    encoder = _encoder(reference, mesh_device, tp, head=True)
    ids, mask, _ = _prompt()
    num_image_tokens = int(mask.sum())

    torch.manual_seed(3)
    extra = torch.randint(0, VOCAB_SIZE, [1, EXTRA])
    full_ids = torch.cat([ids, extra], dim=1)
    full_mask = torch.cat([mask, torch.zeros_like(extra, dtype=torch.bool)], dim=1)
    position_ids = mrope_position_ids(
        full_mask.long(), image_grid_thw=torch.tensor([IMAGE_GRID]), spatial_merge_size=SPATIAL_MERGE_SIZE
    )
    vision_embeds = torch.randn(num_image_tokens, HIDDEN)
    deepstack_embeds = [torch.randn(num_image_tokens, HIDDEN) for _ in range(2)]

    logger.info("running torch model...")
    with torch.no_grad():
        inputs_embeds = reference.embed_tokens(full_ids)
        inputs_embeds[full_mask] = vision_embeds
        hidden = reference.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=full_mask,
            deepstack_visual_embeds=deepstack_embeds,
        ).last_hidden_state
        expected = hidden[:, SEQ - 1 : SEQ + EXTRA - 1] @ reference.embed_tokens.weight.T

    logger.info("running ttnn model...")
    out = encoder.generate(
        ids,
        mask=None,
        max_length=SEQ + EXTRA,
        eos_tokens=None,
        guide=full_ids,
        return_logits=True,
        positions=position_ids[..., :SEQ],
        vision_embeds=tensor.from_torch(vision_embeds, device=mesh_device),
        vision_mask=mask,
        deepstack_embeds=[tensor.from_torch(e, device=mesh_device) for e in deepstack_embeds],
    )

    assert torch.equal(out.tokens, full_ids)
    assert_quality(expected, out.logits, pcc=0.995)
