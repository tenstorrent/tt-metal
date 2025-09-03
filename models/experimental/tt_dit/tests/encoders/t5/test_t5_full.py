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
from transformers.models.t5.modeling_t5 import T5EncoderModel
from models.experimental.tt_dit.parallel.manager import CCLManager
from models.experimental.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor

from models.experimental.tt_dit.encoders.t5.model_t5 import T5Config, T5Encoder
from models.experimental.tt_dit.utils.check import assert_quality


@pytest.mark.parametrize(
    "model_name",
    [
        "large",
    ],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["t3k"], indirect=True)
@pytest.mark.parametrize("submesh_shape", [(1, 4), (2, 2)], ids=["1x4", "2x2"])
@pytest.mark.parametrize(
    "device_params, topology",
    [[{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear]],
    indirect=["device_params"],
)
def test_t5_encoder(
    *,
    mesh_device: ttnn.Device,
    submesh_shape: ttnn.MeshShape,
    model_name: str,
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

    hf_model = T5EncoderModel.from_pretrained(model_name_checkpoint, subfolder="text_encoder_3", local_files_only=False)

    hf_model.eval()

    logger.info("=== HuggingFace T5 Config ===")
    logger.info(f"vocab_size: {hf_model.config.vocab_size}")
    logger.info(f"hidden_size: {hf_model.config.d_model}")
    logger.info(f"intermediate_size: {hf_model.config.d_ff}")
    logger.info(f"d_kv: {hf_model.config.d_kv}")
    logger.info(f"num_attention_heads: {hf_model.config.num_heads}")
    logger.info(f"num_hidden_layers: {hf_model.config.num_layers}")
    logger.info(f"relative_attention_num_buckets: {hf_model.config.relative_attention_num_buckets}")
    logger.info(f"relative_attention_max_distance: {hf_model.config.relative_attention_max_distance}")
    logger.info(f"layer_norm_epsilon: {hf_model.config.layer_norm_epsilon}")

    max_prompt_length = 256
    torch.manual_seed(0)
    tokens = torch.randint(hf_model.config.vocab_size, [1, max_prompt_length])

    tt_prompt = ttnn.from_torch(
        tokens,
        layout=ttnn.TILE_LAYOUT,
        device=encoder_submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(encoder_submesh),
    )

    # See weight state dict key names
    # logger.info(f"print huggingface state dict keys: {hf_model.state_dict().keys()}")

    # === TT-DiT T5 ====
    config = T5Config(
        vocab_size=hf_model.config.vocab_size,
        embed_dim=hf_model.config.d_model,
        ff_dim=hf_model.config.d_ff,
        kv_dim=hf_model.config.d_kv,
        num_heads=hf_model.config.num_heads,
        num_hidden_layers=hf_model.config.num_layers,
        max_prompt_length=max_prompt_length,
        layer_norm_eps=hf_model.config.layer_norm_epsilon,
        relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
        relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
    )

    tt_encoder = T5Encoder(config, encoder_submesh, ccl_manager, parallel_config)
    tt_encoder.load_state_dict(hf_model.state_dict())

    # time TT model inference only
    tt_start_time = time.time()
    tt_output = tt_encoder(tt_prompt, encoder_submesh)

    tt_end_time = time.time()
    tt_execution_time = tt_end_time - tt_start_time

    # get HF reference outputs
    with torch.no_grad():
        hf_start_time = time.time()

        hf_outputs = hf_model(tokens).last_hidden_state

        hf_end_time = time.time()
        hf_execution_time = hf_end_time - hf_start_time

    # convert mesh tensor to torch tensor for pcc
    # since weights are replicated, can get the tensor from any single device
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output[-1])[0])

    logger.info(f"TT encoder execution time: {tt_execution_time:.4f} seconds")
    logger.info(f"HF encoder execution time: {hf_execution_time:.4f} seconds")

    assert hf_outputs.shape == tt_output_torch.shape

    assert_quality(hf_outputs, tt_output_torch, pcc=0.93)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
