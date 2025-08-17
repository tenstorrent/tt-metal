# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[5]))

import torch
import pytest
import ttnn
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from models.experimental.tt_dit.encoders.clip.model_clip import CLIPEncoder, CLIPConfig
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
def test_clip_encoder(
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
    encoder_parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=encoder_submesh.shape[1], mesh_axis=1),
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

    # test text
    test_text = "A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"

    # tokenize
    hf_inputs = tokenizer(test_text, padding=True, truncation=True, max_length=77, return_tensors="pt")
    tt_prompt = ttnn.from_torch(
        hf_inputs.input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=encoder_submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(encoder_submesh),
    )

    # map HF config to our CLIPConfig
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

    CLIP = CLIPEncoder(config, encoder_submesh)
    CLIP.load_state_dict(hf_model.state_dict())
    tt_embeddings = CLIP(tt_prompt, encoder_submesh, with_projection=True)

    with torch.no_grad():
        hf_embeddings = hf_model.text_model.embeddings(hf_inputs.input_ids)

    # Convert mesh tensor to torch tensor for pcc
    # Since weights are replicated, we can get the tensor from any single device
    tt_embeddings_single_device = ttnn.get_device_tensors(tt_embeddings)[0]
    tt_embeddings_torch = ttnn.to_torch(tt_embeddings_single_device)

    logger.info(f"TT text encoder embeddings shape: {tt_embeddings_torch.shape}")
    logger.info(f"HF text encoder embeddings shape: {hf_embeddings.shape}")

    assert tt_embeddings_torch.shape == hf_embeddings.shape
    assert_quality(tt_embeddings_torch, hf_embeddings, pcc=expected_pcc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
