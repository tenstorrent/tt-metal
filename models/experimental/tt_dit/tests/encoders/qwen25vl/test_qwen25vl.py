# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers.pipelines.qwenimage.pipeline_qwenimage as reference
import pytest
import torch
import ttnn
from loguru import logger
from transformers import Qwen2_5_VLForConditionalGeneration

from ....encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from ....encoders.qwen25vl.model_qwen25vl import Qwen25VlTextEncoder
from ....utils import tensor
from ....utils.check import assert_quality


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
# TODO: test with and without attention mask
def test_qwen25vl_text_encoder(
    mesh_device: ttnn.MeshDevice,
) -> None:
    torch.manual_seed(0)

    batch_size = 1
    sequence_length = 512

    torch_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen-Image", subfolder="text_encoder")
    torch_text_model = torch_model.model.language_model

    remove_layers = 0
    del torch_text_model.layers[-remove_layers:]

    model = Qwen25VlTextEncoder(
        vocab_size=torch_model.config.vocab_size,
        hidden_size=torch_model.config.hidden_size,
        intermediate_size=torch_model.config.intermediate_size,
        hidden_act=torch_model.config.hidden_act,
        num_hidden_layers=torch_model.config.num_hidden_layers - remove_layers,
        num_attention_heads=torch_model.config.num_attention_heads,
        num_key_value_heads=torch_model.config.num_key_value_heads,
        rms_norm_eps=torch_model.config.rms_norm_eps,
        rope_theta=torch_model.config.rope_theta,
        mrope_section=torch_model.config.rope_scaling["mrope_section"],
        device=mesh_device,
    )
    model.load_torch_state_dict(torch_text_model.state_dict())

    tokens = torch.randint(0, torch_model.config.vocab_size, [batch_size, sequence_length])
    # attention_mask = torch.randint(0, 2, [batch_size, sequence_length])
    attention_mask = None  # TODO: enable

    # causal_attention_mask = attention_mask[:, None, :].expand(-1, sequence_length, -1).tril()
    position_ids, _ = torch_model.model.get_rope_index(tokens, attention_mask=attention_mask)

    head_dim = torch_model.config.hidden_size // torch_model.config.num_attention_heads
    inv_freq = torch_model.config.rope_theta ** (
        -torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim
    )
    # In contrast to other models, Qwen2_5_VL has different position ids for the grids
    # So we expand the inv_freq to shape (3, ...)
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    mrope_section = torch_model.config.rope_scaling["mrope_section"] * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device) if attention_mask is not None else None
    tt_pos_embeds_cos = tensor.from_torch(cos, device=mesh_device)
    tt_pos_embeds_sin = tensor.from_torch(sin, device=mesh_device)

    logger.info("running ttnn model...")
    tt_hidden_states = model.forward(
        tt_tokens,
        attention_mask=tt_attention_mask,
        # position_ids=position_ids,
        pos_embeds=(tt_pos_embeds_cos, tt_pos_embeds_sin),
    )
    tt_prompt_embeds = tt_hidden_states[-1]
    tt_prompt_embeds_torch = tensor.to_torch(tt_prompt_embeds)

    logger.info("running torch model...")
    with torch.no_grad():
        out = torch_model.forward(
            tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = out.hidden_states[-1]

    assert len(out.hidden_states) == len(tt_hidden_states)
    assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.9999, relative_rmse=0.015)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"),
    [
        pytest.param((1, 2), (1, 2), id="1x2"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "prompts",
    [
        ["Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot"],
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_qwen25vl_encoder_pair(
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
