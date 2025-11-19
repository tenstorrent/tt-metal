# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from loguru import logger
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from ....encoders.qwen25vl.model_qwen25vl import Qwen25VlTextEncoder
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize("prompts", [["Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot"]])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_qwen25vl(
    mesh_device: ttnn.MeshDevice,
    prompts: list[str],
) -> None:
    checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_sequence_length = 512

    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompts = [template.format(e) for e in prompts]

    tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint)

    tp_axis = 1
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
    )

    torch_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen-Image", subfolder="text_encoder")
    torch_text_model = torch_model.model.language_model

    model = Qwen25VlTextEncoder(
        vocab_size=torch_model.config.vocab_size,
        hidden_size=torch_model.config.hidden_size,
        intermediate_size=torch_model.config.intermediate_size,
        hidden_act=torch_model.config.hidden_act,
        num_hidden_layers=torch_model.config.num_hidden_layers,
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

    tokenizer_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        max_length=max_sequence_length,
        truncation=True,
    )

    logger.info("running torch model...")
    with torch.no_grad():
        output = torch_model.forward(
            tokenizer_out.input_ids, attention_mask=tokenizer_out.attention_mask, output_hidden_states=True
        )
    hidden_states = output.hidden_states[-1].to("cpu")

    logger.info("running ttnn model...")
    attention_mask = None  # TODO

    cos, sin = model.create_rope_tensors(len(prompts), tokenizer_out.input_ids.shape[1], attention_mask)

    tt_tokens = tensor.from_torch(tokenizer_out.input_ids, device=mesh_device, dtype=ttnn.uint32)
    tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device) if attention_mask is not None else None
    tt_pos_embeds_cos = tensor.from_torch(cos, device=mesh_device)
    tt_pos_embeds_sin = tensor.from_torch(sin, device=mesh_device)

    tt_hidden_states = model.forward(
        tt_tokens,
        attention_mask=tt_attention_mask,
        pos_embeds=(tt_pos_embeds_cos, tt_pos_embeds_sin),
    )
    tt_prompt_embeds = tt_hidden_states[-1]
    tt_prompt_embeds_torch = tensor.to_torch(tt_prompt_embeds)

    assert_quality(hidden_states, tt_prompt_embeds_torch, pcc=0.98, relative_rmse=0.2)
