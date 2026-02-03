# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers.pipelines.qwenimage.pipeline_qwenimage
import pytest
import torch
import transformers
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
import ttnn
from loguru import logger

from ....encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from ....encoders.qwen25vl.model_qwen25vl import (
    Qwen25VlAttention,
    Qwen25VlContext,
    Qwen25VlTextEncoder,
    create_rope_tensors,
    prepare_attention_bias,
)
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 2), id="1x2"),
        pytest.param((1, 8), id="1x8"),
    ],
    indirect=True,
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
def test_qwen25vl_attention(*, mesh_device: ttnn.MeshDevice, masked: bool) -> None:
    torch.manual_seed(0)

    batch_size = 10
    sequence_length = 512
    tp_axis = 1

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    parent_torch_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image", subfolder="text_encoder"
    )
    torch_model = parent_torch_model.model.language_model.layers[0].self_attn
    assert isinstance(torch_model, transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention)

    model = Qwen25VlAttention(
        hidden_size=torch_model.config.hidden_size,
        num_heads=torch_model.config.num_attention_heads,
        num_key_value_heads=torch_model.config.num_key_value_heads,
        ctx=Qwen25VlContext(mesh_device, tp_axis, ccl_manager),
    )
    model.load_torch_state_dict(torch_model.state_dict())

    sequence = torch.randn([batch_size, sequence_length, torch_model.config.hidden_size])
    m = torch.randint(0, sequence_length + 1, [batch_size])
    attention_mask = torch.arange(sequence_length) < m.unsqueeze(1) if masked else None
    cos, sin = create_rope_tensors(
        batch_size,
        sequence_length,
        attention_mask,
        head_dim=torch_model.config.hidden_size // torch_model.config.num_attention_heads,
        rope_theta=torch_model.config.rope_theta,
        mrope_section=torch_model.config.rope_scaling["mrope_section"],
    )

    tt_sequence = tensor.from_torch(sequence, device=mesh_device)
    tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device) if attention_mask is not None else None
    tt_pos_embeds_cos = tensor.from_torch(cos, device=mesh_device)
    tt_pos_embeds_sin = tensor.from_torch(sin, device=mesh_device)

    tt_attention_bias = prepare_attention_bias(tt_attention_mask) if tt_attention_mask is not None else None

    logger.info("running ttnn model...")
    tt_out = model.forward(
        tt_sequence,
        attention_bias=tt_attention_bias,
        pos_embeds=(tt_pos_embeds_cos, tt_pos_embeds_sin),
    )
    tt_out_torch = tensor.to_torch(tt_out)

    logger.info("running torch model...")
    position_ids, _ = parent_torch_model.model.get_rope_index(input_ids=sequence, attention_mask=attention_mask)
    position_embeddings = torch_model.rotary_emb(sequence, position_ids)
    if attention_mask is not None:
        causal_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        causal_attention_mask = causal_attention_mask.expand([-1, -1, sequence_length, -1])
        causal_attention_mask = causal_attention_mask.tril()
        causal_attention_mask = causal_attention_mask.bool()
    else:
        causal_attention_mask = None

    with torch.no_grad():
        out, _ = torch_model.forward(
            sequence,
            attention_mask=causal_attention_mask,
            position_embeddings=position_embeddings,
        )

    assert_quality(out, tt_out_torch, pcc=0.982, relative_rmse=0.19)


@pytest.mark.parametrize(
    ("mesh_device", "batch_size", "skip_layers"),
    [
        pytest.param((1, 2), 1, 0, id="1x2"),
        pytest.param((1, 4), 1, 0, id="1x4"),
        pytest.param((1, 8), 1, 0, id="1x8"),
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
def test_qwen25vl_text_encoder(
    *, mesh_device: ttnn.MeshDevice, batch_size: int, skip_layers: int, masked: bool
) -> None:
    torch.manual_seed(0)

    sequence_length = 512
    tp_axis = 1

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
    )

    torch_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image", subfolder="text_encoder"
    )
    torch_text_model = torch_model.model.language_model

    mid = len(torch_text_model.layers) // 2
    del torch_text_model.layers[mid - skip_layers // 2 : mid - (-skip_layers // 2)]

    model = Qwen25VlTextEncoder(
        vocab_size=torch_model.config.vocab_size,
        hidden_size=torch_model.config.hidden_size,
        intermediate_size=torch_model.config.intermediate_size,
        hidden_act=torch_model.config.hidden_act,
        num_hidden_layers=torch_model.config.num_hidden_layers - skip_layers,
        num_attention_heads=torch_model.config.num_attention_heads,
        num_key_value_heads=torch_model.config.num_key_value_heads,
        rms_norm_eps=torch_model.config.rms_norm_eps,
        rope_theta=torch_model.config.rope_theta,
        mrope_section=torch_model.config.rope_scaling["mrope_section"],
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    model.load_torch_state_dict(torch_text_model.state_dict())

    tokens = torch.randint(0, torch_model.config.vocab_size, [batch_size, sequence_length])
    m = torch.randint(0, sequence_length + 1, [batch_size])
    attention_mask = torch.arange(sequence_length) < m.unsqueeze(1) if masked else None
    cos, sin = model.create_rope_tensors(batch_size, sequence_length, attention_mask)

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device) if attention_mask is not None else None
    tt_pos_embeds_cos = tensor.from_torch(cos, device=mesh_device)
    tt_pos_embeds_sin = tensor.from_torch(sin, device=mesh_device)

    logger.info("running ttnn model...")
    tt_hidden_states = model.forward(
        tt_tokens,
        attention_mask=tt_attention_mask,
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

    if masked:
        assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.952, relative_rmse=0.31)
    else:
        assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.991, relative_rmse=0.14)


@pytest.mark.parametrize(
    "mesh_device , submesh_shape",
    [[(2, 4), (1, 4)], [(4, 8), (1, 4)]],
    ids=["2x4_1x4", "4x8_1x4"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "prompts",
    [
        [
            "",
            "Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot",
        ],
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_qwen25vl_encoder_pair(
    *, mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], prompts: list[str]
) -> None:
    # There is a bug in the HF implementation where the prompt_embeds_mask is incorrectly repeated
    # if num_images_per_prompt != 1.
    # https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L262
    # is
    # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    # but should be
    # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
    num_images_per_prompt = 1
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    checkpoint = "Qwen/Qwen-Image"

    torch_pipeline = diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline.from_pretrained(checkpoint)

    template = torch_pipeline.prompt_template_encode
    start_idx = torch_pipeline.prompt_template_encode_start_idx
    sequence_length = 512

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=submesh_shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device=submesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_encoder_pair = Qwen25VlTokenizerEncoderPair(
        checkpoint,
        tokenizer_subfolder="tokenizer",
        encoder_subfolder="text_encoder",
        use_torch=False,
        device=submesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    logger.info("running torch model...")
    with torch.no_grad():
        embeds, mask = torch_pipeline.encode_prompt(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=sequence_length,
        )
        embeds = torch.nn.functional.pad(embeds, [0, 0, 0, sequence_length - embeds.shape[1]], value=0)
        mask = torch.nn.functional.pad(mask, [0, sequence_length - mask.shape[1]], value=0)

    logger.info("running TT model...")
    formatted_prompts = [template.format(e) for e in prompts]
    tt_embeds, tt_mask = tt_encoder_pair.encode(
        formatted_prompts,
        num_images_per_prompt=num_images_per_prompt,
        sequence_length=sequence_length + start_idx,
    )
    tt_embeds = tt_embeds[:, start_idx:]
    tt_mask = tt_mask[:, start_idx:]
    tt_embeds *= tt_mask.unsqueeze(-1)

    assert torch.allclose(mask, tt_mask)
    assert_quality(embeds, tt_embeds, pcc=0.983, relative_rmse=0.19)
