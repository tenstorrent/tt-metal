# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger
from transformers.models.umt5.configuration_umt5 import UMT5Config as HF_UMT5Config
from transformers.models.umt5.modeling_umt5 import UMT5EncoderModel
from transformers import AutoTokenizer
import ttnn
from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config as TT_UMT5Config, UMT5Encoder as TT_UMT5Encoder
from models.tt_dit.encoders.t5.model_t5 import (
    TokenEmbeddings as TT_UMT5TokenEmbeddings,
    RelativePositionEmbeddings as TT_UMT5RelativePositionEmbeddings,
)
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.perf.benchmarking_utils import BenchmarkProfiler


@pytest.mark.parametrize(
    "mesh_device,submesh_shape", [[(2, 4), (1, 4)], [(4, 8), (1, 8)]], ids=["t3k", "glx"], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "device_params, topology",
    [[{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear]],
    indirect=["device_params"],
)
def test_umt5_embeddings(
    *,
    mesh_device: ttnn.Device,
    submesh_shape: ttnn.MeshShape,
    topology: ttnn.Topology,
) -> None:
    parent_mesh_shape = tuple(mesh_device.shape)
    if any(x[0] < x[1] for x in zip(parent_mesh_shape, submesh_shape)):
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")
    encoder_submesh = mesh_device  # mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    print(f"Running on submesh {encoder_submesh.shape} of parent mesh {mesh_device.shape}")

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=encoder_submesh.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(
        mesh_device=encoder_submesh,
        num_links=4,
        topology=ttnn.Topology.Linear,
    )

    model_name_checkpoint = f"Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    hf_model = UMT5EncoderModel.from_pretrained(model_name_checkpoint, subfolder="text_encoder", local_files_only=True)

    hf_model.eval()

    logger.info("=== HuggingFace UMT5 Config ===")
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

    # === TT-DiT UMT5 ====
    config = TT_UMT5Config(
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

    state_dict = hf_model.state_dict()

    tt_token_embed = TT_UMT5TokenEmbeddings(config, encoder_submesh)
    tt_token_embed.load_torch_state_dict({"weight": state_dict["encoder.embed_tokens.weight"]})
    tt_relative_position_embed = TT_UMT5RelativePositionEmbeddings(
        config, encoder_submesh, ccl_manager, parallel_config
    )
    tt_relative_position_embed.load_torch_state_dict(
        {"weight": state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]}
    )

    tt_start_time = time.time()
    tt_embeddings_output = tt_token_embed(tt_prompt)
    tt_position_bias = tt_relative_position_embed(tt_embeddings_output.shape[1])
    tt_end_time = time.time()
    tt_execution_time = tt_end_time - tt_start_time

    with torch.no_grad():
        hf_start_time = time.time()

        hf_token_embeddings = hf_model.encoder.embed_tokens(tokens)

        hf_position_bias = (
            hf_model.encoder.block[0]
            .layer[0]
            .SelfAttention.compute_bias(
                hf_token_embeddings.size(1), hf_token_embeddings.size(1), device=hf_token_embeddings.device
            )
        )

        hf_end_time = time.time()
        hf_execution_time = hf_end_time - hf_start_time

    # convert mesh tensor to torch tensor for pcc
    # since weights are replicated, can get the tensor from any single device
    tt_embeddings_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_embeddings_output)[0])
    mesh_shape = list(encoder_submesh.shape)
    mesh_shape[1 - parallel_config.tensor_parallel.mesh_axis] = 1

    tt_position_bias_torch = ttnn.to_torch(
        tt_position_bias,
        mesh_composer=ttnn.create_mesh_composer(
            encoder_submesh, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(mesh_shape))
        ),  # [0,1] is the mesh dimensions to concatenate. Set replicated dimensions to 1.
    )

    logger.info(f"TT embeddings execution time: {tt_execution_time:.4f} seconds")
    logger.info(f"HF embeddings execution time: {hf_execution_time:.4f} seconds")

    assert_quality(hf_token_embeddings, tt_embeddings_output_torch, pcc=0.99)
    assert_quality(hf_position_bias, tt_position_bias_torch, pcc=0.99)


@pytest.mark.parametrize(
    "dit_unit_test",
    [{"1": True, "0": False}.get(os.environ.get("DIT_UNIT_TEST"), False)],
)
@pytest.mark.parametrize(
    "mesh_device,submesh_shape, num_links",
    [[(2, 4), (1, 4), 1], [(4, 8), (1, 8), 4]],
    ids=["t3k", "glx"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [[{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear]],
    indirect=["device_params"],
)
def test_umt5_encoder(
    *,
    mesh_device: ttnn.Device,
    submesh_shape: ttnn.MeshShape,
    num_links: int,
    topology: ttnn.Topology,
    dit_unit_test: bool,
) -> None:
    parent_mesh_shape = tuple(mesh_device.shape)
    if any(x[0] < x[1] for x in zip(parent_mesh_shape, submesh_shape)):
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")
    encoder_submesh = mesh_device  # mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    print(f"Running on submesh {encoder_submesh.shape} of parent mesh {mesh_device.shape}")

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=encoder_submesh.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(
        mesh_device=encoder_submesh,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    model_name_checkpoint = f"Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    if dit_unit_test:
        # Config for t5-xxl with minimal layers
        hf_config = HF_UMT5Config(
            d_model=4096,
            d_ff=10240,
            num_heads=64,
            num_layers=2,
            num_decoder_layers=2,
            feed_forward_proj="gated-gelu",
            output_past=True,
        )
        hf_model = UMT5EncoderModel(hf_config)
    else:
        hf_model = UMT5EncoderModel.from_pretrained(
            model_name_checkpoint, subfolder="text_encoder", local_files_only=False
        )
    hf_model.eval()

    logger.info("=== HuggingFace UMT5 Config ===")
    logger.info(f"vocab_size: {hf_model.config.vocab_size}")
    logger.info(f"hidden_size: {hf_model.config.d_model}")
    logger.info(f"intermediate_size: {hf_model.config.d_ff}")
    logger.info(f"d_kv: {hf_model.config.d_kv}")
    logger.info(f"num_attention_heads: {hf_model.config.num_heads}")
    logger.info(f"num_hidden_layers: {hf_model.config.num_layers}")
    logger.info(f"relative_attention_num_buckets: {hf_model.config.relative_attention_num_buckets}")
    logger.info(f"relative_attention_max_distance: {hf_model.config.relative_attention_max_distance}")
    logger.info(f"layer_norm_epsilon: {hf_model.config.layer_norm_epsilon}")

    max_prompt_length = 512
    torch.manual_seed(0)
    prompt = [
        "A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze.",
        "Negative prompt: A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name_checkpoint, subfolder="tokenizer", trust_remote_code=True)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_prompt_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

    tt_prompt = ttnn.from_torch(
        text_input_ids,
        layout=ttnn.TILE_LAYOUT,
        device=encoder_submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(encoder_submesh),
    )

    # === TT-DiT UMT5 ====
    config = TT_UMT5Config(
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

    tt_encoder = TT_UMT5Encoder(config, encoder_submesh, ccl_manager, parallel_config)
    tt_encoder.load_torch_state_dict(hf_model.state_dict())

    # time TT model inference only
    # with warmup
    tt_output = tt_encoder(tt_prompt)
    benchmark_profiler = BenchmarkProfiler()
    num_runs = 10
    for i in range(num_runs):
        with benchmark_profiler("tt_umt5_encoder", i):
            tt_output = tt_encoder(tt_prompt)[-1]

    # get HF reference outputs
    with torch.no_grad():
        hf_outputs = hf_model(text_input_ids).last_hidden_state

    # # convert mesh tensor to torch tensor for pcc
    # # since weights are replicated, can get the tensor from any single device
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    tt_execution_time_b = benchmark_profiler.get_duration_average("tt_umt5_encoder")
    logger.info(f"TT encoder execution time benchmark: {tt_execution_time_b:.4f} seconds")
    assert_quality(hf_outputs, tt_output_torch, pcc=0.99)
