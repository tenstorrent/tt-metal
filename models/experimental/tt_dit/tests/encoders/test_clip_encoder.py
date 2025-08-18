# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[5]))

import torch
import pytest
import ttnn
import time
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from models.experimental.tt_dit.encoders.clip.model_clip import CLIPEncoder, CLIPConfig
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
    test_start_time = time.time()

    parent_mesh_shape = tuple(mesh_device.shape)
    if any(x[0] < x[1] for x in zip(parent_mesh_shape, submesh_shape)):
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")

    # Setup timing
    setup_start_time = time.time()
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
    setup_time = time.time() - setup_start_time
    logger.info(f"Setup time: {setup_time:.3f}s")

    # Model loading timing
    model_loading_start_time = time.time()
    model_name_checkpoint = f"stabilityai/stable-diffusion-3.5-{model_name}"

    hf_model = CLIPTextModelWithProjection.from_pretrained(
        model_name_checkpoint, subfolder=clip_path, local_files_only=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_name_checkpoint, subfolder=tokenizer_path, local_files_only=True)

    hf_model.eval()
    model_loading_time = time.time() - model_loading_start_time
    logger.info(f"Model loading time: {model_loading_time:.3f}s")

    logger.info("=== HuggingFace CLIP Config ===")
    logger.info(f"vocab_size: {hf_model.config.vocab_size}")
    logger.info(f"hidden_size: {hf_model.config.hidden_size}")
    logger.info(f"intermediate_size: {hf_model.config.intermediate_size}")
    logger.info(f"num_attention_heads: {hf_model.config.num_attention_heads}")
    logger.info(f"num_hidden_layers: {hf_model.config.num_hidden_layers}")

    # Print weights dictionary keys
    # weights_dict = hf_model.state_dict()
    # logger.info("=== Weights Dictionary Keys ===")
    # for key in weights_dict.keys():
    #     logger.info(f"  {key}")
    # logger.info(f"Total number of weight keys: {len(weights_dict)}")

    # Input preparation timing
    input_prep_start_time = time.time()
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

    eos_token_id = hf_model.config.eos_token_id
    input_prep_time = time.time() - input_prep_start_time
    logger.info(f"Input preparation time: {input_prep_time:.3f}s")

    # TT model setup timing
    tt_setup_start_time = time.time()
    # === USING tt-dit CLIP: ====
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

    tt_clip = CLIPEncoder(config, encoder_submesh, ccl_manager, parallel_config, eos_token_id)
    tt_clip.load_state_dict(hf_model.state_dict())  # load weights
    tt_setup_time = time.time() - tt_setup_start_time
    logger.info(f"TT model setup time: {tt_setup_time:.3f}s")

    # Inference timing
    inference_start_time = time.time()
    tt_clip_output = tt_clip(tt_prompt, encoder_submesh, with_projection=True)

    # =====

    with torch.no_grad():
        # Run HF model to capture self-attention output
        hf_output = hf_model.text_model(hf_inputs.input_ids)
    inference_time = time.time() - inference_start_time
    logger.info(f"Inference time: {inference_time:.3f}s")

    # Output processing and comparison timing
    comparison_start_time = time.time()
    # Extract outputs from TT encoder
    # tt_clip_output is a tuple: (all_hidden_states, projected_output)
    tt_all_hidden_states, tt_projected_output = tt_clip_output

    # Get the final hidden states (last layer) from the list of all hidden states
    tt_final_hidden_states = tt_all_hidden_states[-1]

    # Convert mesh tensor to torch tensor for pcc
    # Since weights are replicated, we can get the tensor from any single device
    tt_output_single_device = ttnn.get_device_tensors(tt_final_hidden_states)[0]
    tt_output = ttnn.to_torch(tt_output_single_device)

    hf_final_hidden_states = hf_output.last_hidden_state

    logger.info(f"TT text encoder final hidden states shape: {tt_output.shape}")
    logger.info(f"HF text encoder final hidden states shape: {hf_final_hidden_states.shape}")

    assert tt_output.shape == hf_final_hidden_states.shape
    assert_quality(tt_output, hf_final_hidden_states, pcc=expected_pcc)
    comparison_time = time.time() - comparison_start_time
    logger.info(f"Output comparison time: {comparison_time:.3f}s")

    # Total test time
    total_test_time = time.time() - test_start_time
    logger.info(f"=== TIMING SUMMARY ===")
    logger.info(f"Setup: {setup_time:.3f}s")
    logger.info(f"Model loading: {model_loading_time:.3f}s")
    logger.info(f"Input preparation: {input_prep_time:.3f}s")
    logger.info(f"TT model setup: {tt_setup_time:.3f}s")
    logger.info(f"Inference: {inference_time:.3f}s")
    logger.info(f"Output comparison: {comparison_time:.3f}s")
    logger.info(f"Total test time: {total_test_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
