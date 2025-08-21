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

from models.experimental.tt_dit.encoders.t5.model_t5 import RelativeTextEmbeddings, T5Config, T5Attention
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
def test_t5_embeddings(
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

    hf_model = T5EncoderModel.from_pretrained(model_name_checkpoint, subfolder="text_encoder_3", local_files_only=True)

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

    # create input
    max_prompt_length = 256
    torch.manual_seed(0)
    tokens = torch.randint(hf_model.config.vocab_size, [1, max_prompt_length])

    # convert to tt tensor
    tt_prompt = ttnn.from_torch(
        tokens,
        layout=ttnn.TILE_LAYOUT,
        device=encoder_submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(encoder_submesh),
    )

    logger.info(f"print huggingface state dict keys: {hf_model.state_dict().keys()}")

    # === USING tt-dit T5 ====
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

    tt_embedding = RelativeTextEmbeddings(config, encoder_submesh, ccl_manager, parallel_config)
    tt_self_attn = T5Attention(
        config=config, mesh_device=encoder_submesh, ccl_manager=ccl_manager, parallel_config=parallel_config
    )
    # load only the embeddings part of the state dict
    embeddings_state_dict = {}
    for key, value in hf_model.state_dict().items():
        if key.startswith("encoder.embed_tokens.") or key.startswith(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias."
        ):
            embeddings_state_dict[key] = value

    tt_embedding.load_state_dict(embeddings_state_dict)

    # load only the self attn part of the state dict
    self_attn_state_dict = {}
    for key, value in hf_model.state_dict().items():
        if key.startswith("encoder.block.0.layer.0.SelfAttention."):
            self_attn_state_dict[key] = value
    tt_self_attn.load_state_dict(self_attn_state_dict)

    # time TT model inference only
    tt_start_time = time.time()
    tt_embeddings_output, tt_position_bias = tt_embedding(tt_prompt, encoder_submesh)

    # Debug intermediate values
    logger.info("=== Checking intermediate values ===")
    q = tt_self_attn.q_proj(tt_embeddings_output)
    k = tt_self_attn.k_proj(tt_embeddings_output)
    v = tt_self_attn.v_proj(tt_embeddings_output)

    # Log Q,K,V stats
    q_torch = ttnn.to_torch(ttnn.get_device_tensors(q)[0])
    k_torch = ttnn.to_torch(ttnn.get_device_tensors(k)[0])
    v_torch = ttnn.to_torch(ttnn.get_device_tensors(v)[0])
    logger.info(
        f"Q tensor stats - min: {q_torch.min()}, max: {q_torch.max()}, mean: {q_torch.mean()}, shape: {q_torch.shape}"
    )
    logger.info(
        f"K tensor stats - min: {k_torch.min()}, max: {k_torch.max()}, mean: {k_torch.mean()}, shape: {k_torch.shape}"
    )
    logger.info(
        f"V tensor stats - min: {v_torch.min()}, max: {v_torch.max()}, mean: {v_torch.mean()}, shape: {v_torch.shape}"
    )

    # Debug attention computation
    qkv = ttnn.concat([q, k, v], dim=-1)
    q_split, k_split, v_split = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=tt_self_attn.num_heads // tt_self_attn.parallel_config.tensor_parallel.factor, transpose_key=True
    )

    # Log split heads stats
    q_split_torch = ttnn.to_torch(ttnn.get_device_tensors(q_split)[0])
    k_split_torch = ttnn.to_torch(ttnn.get_device_tensors(k_split)[0])
    v_split_torch = ttnn.to_torch(ttnn.get_device_tensors(v_split)[0])
    logger.info(f"Q split stats - shape: {q_split_torch.shape}")
    logger.info(f"K split stats - shape: {k_split_torch.shape}")
    logger.info(f"V split stats - shape: {v_split_torch.shape}")

    scores = ttnn.matmul(q_split, k_split)
    scores_torch = ttnn.to_torch(ttnn.get_device_tensors(scores)[0])
    logger.info(
        f"Attention scores stats - min: {scores_torch.min()}, max: {scores_torch.max()}, mean: {scores_torch.mean()}, shape: {scores_torch.shape}"
    )

    scores = scores + tt_position_bias
    scores_bias_torch = ttnn.to_torch(ttnn.get_device_tensors(scores)[0])
    logger.info(
        f"Scores+bias stats - min: {scores_bias_torch.min()}, max: {scores_bias_torch.max()}, mean: {scores_bias_torch.mean()}, shape: {scores_bias_torch.shape}"
    )

    attn_weights = ttnn.softmax(scores, dim=-1)
    weights_torch = ttnn.to_torch(ttnn.get_device_tensors(attn_weights)[0])
    logger.info(
        f"Attention weights stats - min: {weights_torch.min()}, max: {weights_torch.max()}, mean: {weights_torch.mean()}, shape: {weights_torch.shape}"
    )

    attn_output = ttnn.matmul(attn_weights, v_split)
    attn_out_torch = ttnn.to_torch(ttnn.get_device_tensors(attn_output)[0])
    logger.info(
        f"Attention output stats - min: {attn_out_torch.min()}, max: {attn_out_torch.max()}, mean: {attn_out_torch.mean()}, shape: {attn_out_torch.shape}"
    )

    tt_self_attn_output = tt_self_attn(tt_embeddings_output, tt_position_bias, ccl_manager, parallel_config)

    tt_end_time = time.time()
    tt_execution_time = tt_end_time - tt_start_time

    # === get HF embeddings for comparison ===
    with torch.no_grad():
        # time HF model execution
        hf_start_time = time.time()

        # get HF embeddings manually
        hf_token_embeddings = hf_model.encoder.embed_tokens(tokens).detach()

        # breakpoint()
        # get HF self attn manually
        hf_self_attn_output = (
            hf_model.encoder.block[0].layer[0].SelfAttention(hf_token_embeddings)[0].detach()
        )  # Get first element of tuple and detach

        hf_end_time = time.time()
        hf_execution_time = hf_end_time - hf_start_time

    # convert mesh tensor to torch tensor for pcc
    # since weights are replicated, can get the tensor from any single device
    tt_self_attn_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_self_attn_output)[0])
    # breakpoint()
    # Log shapes for debugging

    logger.info(f"HF self attention output shape: {hf_self_attn_output.shape}")
    logger.info(f"TT self attention output shape: {tt_self_attn_output_torch.shape}")

    logger.info(f"TT embeddings execution time: {tt_execution_time:.4f} seconds")
    logger.info(f"HF embeddings execution time: {hf_execution_time:.4f} seconds")

    # Verify shapes match at each step
    assert (
        tt_embeddings_output.shape == hf_token_embeddings.shape
    ), f"Embedding shapes don't match: TT {tt_embeddings_output.shape} vs HF {hf_token_embeddings.shape}"
    assert (
        hf_self_attn_output.shape == tt_self_attn_output_torch.shape
    ), f"Self attention output shapes don't match: HF {hf_self_attn_output.shape} vs TT {tt_self_attn_output_torch.shape}"

    # Compare outputs with quality checks
    assert_quality(hf_self_attn_output, tt_self_attn_output_torch, pcc=0.95)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
