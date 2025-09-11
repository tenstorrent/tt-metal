# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[6]))

import torch
import pytest
import ttnn
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from models.experimental.tt_dit.encoders.clip.model_clip import (
    CLIPStack,
    CLIPConfig,
    TextEmbeddings,
    create_4d_causal_attention_mask,
)
from models.experimental.tt_dit.parallel.manager import CCLManager
from models.experimental.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.experimental.tt_dit.utils.check import assert_quality


@pytest.mark.parametrize(
    "model_name",
    [
        "large",
    ],
)
@pytest.mark.parametrize(
    "clip_path, tokenizer_path, expected_pcc",
    [
        ("text_encoder", "tokenizer", 0.99),
        ("text_encoder_2", "tokenizer_2", 0.985),
    ],
    ids=["encoder_1", "encoder_2"],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["t3k"], indirect=True)
@pytest.mark.parametrize("submesh_shape", [(1, 4), (2, 2)], ids=["1x4", "2x2"])
@pytest.mark.parametrize(
    "device_params, topology",
    [[{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear]],
    indirect=["device_params"],
)
def test_clip_stack_all_layers(
    *,
    mesh_device: ttnn.Device,
    submesh_shape: ttnn.MeshShape,
    model_name: str,
    clip_path: str,
    tokenizer_path: str,
    expected_pcc: float,
    topology: ttnn.Topology,
) -> None:
    parent_mesh_shape = tuple(mesh_device.shape)
    if any(x[0] < x[1] for x in zip(parent_mesh_shape, submesh_shape)):
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")
    encoder_submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    print(f"Running on submesh {encoder_submesh.shape} of parent mesh {mesh_device.shape}")

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=encoder_submesh.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(
        mesh_device=encoder_submesh,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )

    model_name_checkpoint = f"stabilityai/stable-diffusion-3.5-{model_name}"

    hf_model = CLIPTextModelWithProjection.from_pretrained(
        model_name_checkpoint, subfolder=clip_path, local_files_only=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_name_checkpoint, subfolder=tokenizer_path, local_files_only=True)

    hf_model.eval()

    logger.info("=== HuggingFace CLIP Config ===")
    logger.info(f"vocab_size: {hf_model.config.vocab_size}")
    logger.info(f"hidden_size: {hf_model.config.hidden_size}")
    logger.info(f"intermediate_size: {hf_model.config.intermediate_size}")
    logger.info(f"num_attention_heads: {hf_model.config.num_attention_heads}")
    logger.info(f"num_hidden_layers: {hf_model.config.num_hidden_layers}")

    test_text = "A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"

    hf_inputs = tokenizer(test_text, padding=True, truncation=True, max_length=77, return_tensors="pt")
    tt_prompt = ttnn.from_torch(
        hf_inputs.input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=encoder_submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(encoder_submesh),
    )

    # === TT-DiT CLIPStack (all encoder layers) ====
    config = CLIPConfig(
        vocab_size=hf_model.config.vocab_size,
        embed_dim=hf_model.config.hidden_size,
        ff_dim=hf_model.config.intermediate_size,
        num_heads=hf_model.config.num_attention_heads,
        num_hidden_layers=hf_model.config.num_hidden_layers,
        max_prompt_length=77,
        layer_norm_eps=hf_model.config.layer_norm_eps,
        attention_dropout=hf_model.config.attention_dropout,
        hidden_act=hf_model.config.hidden_act,
    )

    logger.info(f"Testing CLIPStack with {config.num_hidden_layers} encoder layers")

    tt_embedding = TextEmbeddings(config, encoder_submesh)
    embeddings_state_dict = {}
    for key, value in hf_model.state_dict().items():
        if key.startswith("text_model.embeddings."):
            new_key = key.replace("text_model.embeddings.", "")
            embeddings_state_dict[new_key] = value
    tt_embedding.load_state_dict(embeddings_state_dict)

    hidden_states = tt_embedding(tt_prompt, encoder_submesh)

    causal_attention_mask = create_4d_causal_attention_mask(
        tt_prompt.shape, encoder_submesh, dtype=hidden_states.get_dtype()
    )

    tt_clip_stack = CLIPStack(config, encoder_submesh, ccl_manager, parallel_config)

    encoder_state_dict = {}
    for key, value in hf_model.state_dict().items():
        if key.startswith("text_model.encoder."):
            new_key = key.replace("text_model.encoder.", "")
            encoder_state_dict[new_key] = value
    tt_clip_stack.load_state_dict(encoder_state_dict)

    tt_start_time = time.time()
    tt_all_hidden_states = tt_clip_stack(hidden_states, causal_attention_mask, ccl_manager, parallel_config)
    tt_end_time = time.time()
    tt_execution_time = tt_end_time - tt_start_time

    expected_outputs = config.num_hidden_layers + 1  # input + all encoder layers

    # get HF reference outputs
    with torch.no_grad():
        hf_start_time = time.time()

        # get HF embeddings
        hf_token_embeddings = hf_model.text_model.embeddings.token_embedding(hf_inputs.input_ids)
        seq_len = hf_inputs.input_ids.shape[-1]
        position_ids = torch.arange(seq_len).expand((1, -1))
        hf_position_embeddings = hf_model.text_model.embeddings.position_embedding(position_ids)
        hf_hidden_states = hf_token_embeddings + hf_position_embeddings

        # create causal attention mask for HF
        batch_size, tgt_len = hf_inputs.input_ids.shape
        hf_causal_mask = torch.full((tgt_len, tgt_len), float("-inf"))
        mask_cond = torch.arange(hf_causal_mask.size(-1))
        hf_causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(hf_causal_mask.size(-1), 1), 0)
        hf_causal_mask = hf_causal_mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len)

        # run through HF encoder layers manually to match CLIPStack behavior
        hf_all_hidden_states = [hf_hidden_states]  # input embeddings
        current_hidden_states = hf_hidden_states

        for layer_idx in range(hf_model.config.num_hidden_layers):
            layer_output = hf_model.text_model.encoder.layers[layer_idx](
                current_hidden_states, None, hf_causal_mask  # hidden_states, attention_mask, causal_attention_mask
            )[
                0
            ]  # HF returns tuple (hidden_states, attentions)
            hf_all_hidden_states.append(layer_output)
            current_hidden_states = layer_output

        hf_end_time = time.time()
        hf_execution_time = hf_end_time - hf_start_time

    # convert mesh tensor to torch tensor for pcc (final layer output)
    # since weights are replicated, can get the tensor from any single device
    tt_final_layer_output = ttnn.to_torch(ttnn.get_device_tensors(tt_all_hidden_states[-2])[0])
    hf_final_layer_output = hf_all_hidden_states[-1]

    logger.info(f"TT CLIPStack (all {config.num_hidden_layers} layers) execution time: {tt_execution_time:.4f} seconds")
    logger.info(
        f"HF encoder (all {hf_model.config.num_hidden_layers} layers) execution time: {hf_execution_time:.4f} seconds"
    )

    assert hf_final_layer_output.shape == tt_final_layer_output.shape

    assert_quality(hf_final_layer_output, tt_final_layer_output, pcc=expected_pcc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
