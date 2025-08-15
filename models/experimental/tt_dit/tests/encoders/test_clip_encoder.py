# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import torch
import ttnn
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
import pytest

from ...encoders.clip.model_clip import CLIPTextModel
from ...encoders.clip.config_clip import CLIPTextConfig
from ...parallel.config import EncoderParallelManager


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
    encoder_parallel_manager = EncoderParallelManager(encoder_submesh, topology, mesh_axis=1, num_links=1)
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
    logger.info(f"max_position_embeddings: {hf_model.config.max_position_embeddings}")

    # create tt model like sd3.5 test pattern
    logger.info("Creating TT CLIP model...")

    config = CLIPTextConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        intermediate_size=hf_model.config.intermediate_size,
        num_attention_heads=hf_model.config.num_attention_heads,
        num_hidden_layers=hf_model.config.num_hidden_layers,
        max_prompt_length=77,  # clip standard
        layer_norm_eps=hf_model.config.layer_norm_eps,
        attention_dropout=hf_model.config.attention_dropout,
        hidden_act=hf_model.config.hidden_act,
    )

    tt_model = CLIPTextModel(
        config=config, mesh_device=encoder_submesh, with_projection=True, parallel_manager=encoder_parallel_manager
    )

    # load weights
    logger.info("Loading HuggingFace weights into TT model...")
    hf_state_dict = hf_model.state_dict()

    # print all the keys of hf_state_dict
    logger.info(f"HF state dict keys: {hf_state_dict.keys()}")

    tt_model.load_state_dict(hf_state_dict)

    # test text
    test_text = "A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"

    # tokenize
    hf_inputs = tokenizer(test_text, padding=True, truncation=True, max_length=77, return_tensors="pt")
    tt_tokens = ttnn.from_torch(
        hf_inputs.input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=encoder_submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(encoder_submesh),
    )

    # test embeddings first
    logger.info("Testing TT embeddings...")
    start_time = time.time()

    tt_embeddings = tt_model.embeddings(tt_tokens, device=encoder_submesh)
    causal_attention_mask = tt_model.embeddings.causal_attention_mask
    logger.info(f"TT embeddings runtime: {time.time() - start_time}")

    # test hf embeddings for comparison
    logger.info("Testing HF embeddings...")
    with torch.no_grad():
        hf_embeddings = hf_model.text_model.embeddings(hf_inputs.input_ids)

    # convert tt embeddings to torch for comparison
    tt_embeddings_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_embeddings)[0])

    # basic embeddings validation
    logger.info(f"HF embeddings shape: {hf_embeddings.shape}")
    logger.info(f"TT embeddings shape: {tt_embeddings_torch.shape}")

    assert (
        hf_embeddings.shape == tt_embeddings_torch.shape
    ), f"Embeddings shape mismatch: {hf_embeddings.shape} vs {tt_embeddings_torch.shape}"

    logger.info("Testing TT transformer layer with attention and MLP...")

    start_time = time.time()
    with torch.no_grad():
        hf_output = hf_model(**hf_inputs, output_hidden_states=True)
        sequence_output = hf_output.hidden_states[-2]
        pooled_output = hf_output.text_embeds
    logger.info(f"text encoder CPU runtime: {time.time() - start_time}")

    # debug
    logger.info(f"HF text encoder sequence output shape: {sequence_output.shape}")
    logger.info(f"HF text encoder pooled output shape: {pooled_output.shape}")
    logger.info(f"HF text encoder sequence output mean: {sequence_output.mean():.6f}, std: {sequence_output.std():.6f}")
    logger.info(f"HF text encoder pooled output mean: {pooled_output.mean():.6f}, std: {pooled_output.std():.6f}")

    logger.info("compiling text encoder...")

    # breakpoint()

    # pass through encoder with causal mask
    encoder_outputs = tt_model.encoder(
        tt_embeddings,
        mesh_device=encoder_submesh,
        causal_attention_mask=causal_attention_mask,
        parallel_manager=encoder_parallel_manager,
    )

    tt_sequence_output, tt_projected_output = encoder_outputs

    logger.info(f"text encoder TT-NN runtime: {time.time() - start_time}")
    logger.info("text encoder done...")

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    test_clip_encoder()
