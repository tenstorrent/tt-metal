# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.tt_dit.encoders.clip.model_clip import CLIPConfig, CLIPEncoder
from models.experimental.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.experimental.tt_dit.parallel.manager import CCLManager
from models.experimental.tt_dit.utils.check import assert_quality
from transformers import CLIPTextModel, CLIPTokenizer


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
        ("text_encoder_2", "tokenizer_2", 0.984),
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

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=encoder_submesh.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(
        mesh_device=encoder_submesh,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )

    model_name_checkpoint = f"stabilityai/stable-diffusion-3.5-{model_name}"

    hf_model = CLIPTextModel.from_pretrained(model_name_checkpoint, subfolder=clip_path, local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(model_name_checkpoint, subfolder=tokenizer_path, local_files_only=True)

    hf_model.eval()

    logger.info("=== HuggingFace CLIP Config ===")
    logger.info(f"vocab_size: {hf_model.config.vocab_size}")
    logger.info(f"hidden_size: {hf_model.config.hidden_size}")
    logger.info(f"intermediate_size: {hf_model.config.intermediate_size}")
    logger.info(f"num_attention_heads: {hf_model.config.num_attention_heads}")
    logger.info(f"num_hidden_layers: {hf_model.config.num_hidden_layers}")

    # Test prompt. Cannot use randn tensor due to specific HF eos token id
    test_text = "A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"

    hf_inputs = tokenizer(test_text, padding=True, truncation=True, max_length=77, return_tensors="pt")
    tt_prompt = ttnn.from_torch(
        hf_inputs.input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=encoder_submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(encoder_submesh),
    )

    eos_token_id = hf_model.config.eos_token_id
    logger.info(f"EOS token id: {eos_token_id}")  # eos token id: 2

    logger.info(f"Activation function: {hf_model.config.hidden_act}")  # quick_gelu

    # === TT-DiT CLIP ====
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

    tt_clip = CLIPEncoder(config, encoder_submesh, ccl_manager, parallel_config, eos_token_id)
    tt_clip.load_state_dict(hf_model.state_dict())

    # times TT model inference only
    tt_start_time = time.time()
    tt_sequence_output, tt_pooled_output, tt_normalized_output = tt_clip(
        tt_prompt, encoder_submesh, return_normalized_state=True
    )
    tt_end_time = time.time()
    tt_execution_time = tt_end_time - tt_start_time

    # get HF reference outputs
    with torch.no_grad():
        hf_start_time = time.time()
        hf_output = hf_model(hf_inputs.input_ids, output_hidden_states=True)
        hf_end_time = time.time()
        hf_execution_time = hf_end_time - hf_start_time

    hf_sequence_output = hf_output.hidden_states[-1]
    hf_pooled_output = hf_output.pooler_output
    hf_normalized_output = hf_output.last_hidden_state

    # convert mesh tensor to torch tensor for pcc
    # since weights are replicated, can get the tensor from any single device
    tt_sequence_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_sequence_output[-1])[0])
    tt_pooled_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled_output)[0])
    tt_normalized_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_normalized_output)[0])

    logger.info(f"TT model execution time: {tt_execution_time:.4f} seconds")
    logger.info(f"HF model execution time: {hf_execution_time:.4f} seconds")

    assert hf_sequence_output.shape == tt_sequence_output_torch.shape
    assert hf_pooled_output.shape == tt_pooled_output_torch.shape
    assert hf_normalized_output.shape == tt_normalized_output_torch.shape

    assert_quality(hf_sequence_output, tt_sequence_output_torch, pcc=expected_pcc)
    assert_quality(hf_pooled_output, tt_pooled_output_torch, pcc=expected_pcc)
    assert_quality(hf_normalized_output, tt_normalized_output_torch, pcc=expected_pcc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
