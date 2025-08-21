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

from models.experimental.tt_dit.encoders.clip.model_clip import TextEmbeddings, CLIPConfig
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
def test_clip_text_embeddings(
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

    # === TT-DiT TextEmbeddings ====
    config = CLIPConfig(
        vocab_size=hf_model.config.vocab_size,
        embed_dim=hf_model.config.hidden_size,
        ff_dim=hf_model.config.intermediate_size,
        num_heads=hf_model.config.num_attention_heads,
        num_hidden_layers=hf_model.config.num_hidden_layers,
        max_prompt_length=77,
        layer_norm_eps=hf_model.config.layer_norm_eps,
        attention_dropout=hf_model.config.attention_dropout,
        hidden_act="quick_gelu",
    )

    tt_embedding = TextEmbeddings(config, encoder_submesh)
    embeddings_state_dict = {}
    for key, value in hf_model.state_dict().items():
        if key.startswith("text_model.embeddings."):
            new_key = key.replace("text_model.embeddings.", "")
            embeddings_state_dict[new_key] = value
    tt_embedding.load_state_dict(embeddings_state_dict)

    # time TT model inference only
    tt_start_time = time.time()
    tt_embeddings_output = tt_embedding(tt_prompt, encoder_submesh)
    tt_end_time = time.time()
    tt_execution_time = tt_end_time - tt_start_time

    # get HF outputs
    with torch.no_grad():
        hf_start_time = time.time()

        hf_token_embeddings = hf_model.text_model.embeddings.token_embedding(hf_inputs.input_ids)
        seq_len = hf_inputs.input_ids.shape[-1]
        position_ids = torch.arange(seq_len).expand((1, -1))
        hf_position_embeddings = hf_model.text_model.embeddings.position_embedding(position_ids)
        hf_embeddings_output = hf_token_embeddings + hf_position_embeddings

        hf_end_time = time.time()
        hf_execution_time = hf_end_time - hf_start_time

    # convert mesh tensor to torch tensor for pcc
    # since weights are replicated, can get the tensor from any single device
    tt_embeddings_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_embeddings_output)[0])

    logger.info(f"TT embeddings execution time: {tt_execution_time:.4f} seconds")
    logger.info(f"HF embeddings execution time: {hf_execution_time:.4f} seconds")

    assert hf_embeddings_output.shape == tt_embeddings_output_torch.shape

    assert_quality(hf_embeddings_output, tt_embeddings_output_torch, pcc=expected_pcc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
