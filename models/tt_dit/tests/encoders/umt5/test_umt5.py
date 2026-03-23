# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer
from transformers.models.umt5.configuration_umt5 import UMT5Config as HF_UMT5Config
from transformers.models.umt5.modeling_umt5 import UMT5EncoderModel

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....encoders.t5.model_t5 import RelativePositionEmbeddings as TT_UMT5RelativePositionEmbeddings
from ....encoders.umt5.model_umt5 import UMT5Config as TT_UMT5Config
from ....encoders.umt5.model_umt5 import UMT5Encoder as TT_UMT5Encoder
from ....layers.embeddings import Embedding
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import cache
from ....utils.check import assert_quality

# Shared device configuration for UMT5 tests to reduce code duplication and ensure we are using consistent device configurations.


def umt5_device_config(func):
    """Decorator to apply standard UMT5 device/mesh parametrization to tests."""
    func = pytest.mark.parametrize(
        "dit_unit_test",
        [{"1": True, "0": False}.get(os.environ.get("DIT_UNIT_TEST"), False)],
    )(func)
    func = pytest.mark.parametrize(
        "mesh_device, num_links, max_execution_time",
        [[(2, 4), 1, 0.13], [(4, 8), 4, 0.1], [(4, 8), 2, 0.1]],
        ids=["t3k", "wh_glx", "bh_glx"],
        indirect=["mesh_device", "num_links"],
    )(func)
    func = pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    )(func)
    return func


@pytest.fixture
def num_links(request):
    return request.param


@pytest.fixture
def parallel_config_and_ccl_manager(mesh_device, num_links):
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )
    return parallel_config, ccl_manager


def get_random_weights_model() -> UMT5EncoderModel:
    hf_config = HF_UMT5Config(
        d_model=4096,
        d_ff=10240,
        num_heads=64,
        num_layers=24,
        num_decoder_layers=0,
        relative_attention_max_distance=128,
        relative_attention_num_buckets=32,
        feed_forward_proj="gated-gelu",
        output_past=True,
    )
    return UMT5EncoderModel(hf_config)


@umt5_device_config
def test_umt5_embeddings(
    *,
    mesh_device: ttnn.Device,
    parallel_config_and_ccl_manager: tuple,
    dit_unit_test: bool,
    max_execution_time: float,
) -> None:
    torch.manual_seed(0)

    parallel_config, ccl_manager = parallel_config_and_ccl_manager

    if dit_unit_test:
        hf_model = get_random_weights_model()
    else:
        model_name_checkpoint = f"Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        hf_model = UMT5EncoderModel.from_pretrained(
            model_name_checkpoint, subfolder="text_encoder", local_files_only=True
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
    tokens = torch.randint(hf_model.config.vocab_size, [1, max_prompt_length])

    tt_prompt = ttnn.from_torch(
        tokens,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
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

    tt_token_embed = Embedding(config.vocab_size, config.embed_dim, device=mesh_device)
    tt_token_embed.load_torch_state_dict({"weight": state_dict["encoder.embed_tokens.weight"]})
    tt_relative_position_embed = TT_UMT5RelativePositionEmbeddings(config, mesh_device, ccl_manager, parallel_config)
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
    mesh_shape = list(mesh_device.shape)
    mesh_shape[1 - parallel_config.tensor_parallel.mesh_axis] = 1

    tt_position_bias_torch = ttnn.to_torch(
        tt_position_bias,
        mesh_composer=ttnn.create_mesh_composer(
            mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(mesh_shape))
        ),  # [0,1] is the mesh dimensions to concatenate. Set replicated dimensions to 1.
    )

    logger.info(f"TT embeddings execution time: {tt_execution_time:.4f} seconds")
    logger.info(f"HF embeddings execution time: {hf_execution_time:.4f} seconds")

    assert_quality(hf_token_embeddings, tt_embeddings_output_torch, pcc=(1.0 - 1e-5), relative_rmse=0.005)
    assert_quality(hf_position_bias, tt_position_bias_torch, pcc=(1.0 - 1e-5), relative_rmse=0.005)


@umt5_device_config
def test_umt5_encoder(
    *,
    mesh_device: ttnn.Device,
    parallel_config_and_ccl_manager: tuple,
    dit_unit_test: bool,
    max_execution_time: float,
) -> None:
    torch.manual_seed(0)

    parallel_config, ccl_manager = parallel_config_and_ccl_manager

    model_name_checkpoint = f"Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    if dit_unit_test:
        hf_model = get_random_weights_model()
    else:
        hf_model = UMT5EncoderModel.from_pretrained(
            model_name_checkpoint, subfolder="text_encoder", local_files_only=False
        )
    hf_model.eval()

    max_prompt_length = 512

    prompt = [
        "A Roman general standing on a battlefield at dawn, torn red cape blowing in the wind, distant soldiers forming ranks, painterly brushwork in the style of Caravaggio, chiaroscuro lighting, epic composition",
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    ]
    total_prompts = len(prompt)
    num_devices = mesh_device.shape[1 - parallel_config.tensor_parallel.mesh_axis]
    prompt += [" "] * ((num_devices - (total_prompts % num_devices)) % num_devices)
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

    dims = [None, None]
    DP_axis = 1 - parallel_config.tensor_parallel.mesh_axis
    dims[DP_axis] = 0
    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)

    tt_prompt = ttnn.from_torch(
        text_input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
    )

    tt_mask = ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
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

    tt_encoder = TT_UMT5Encoder(config, mesh_device, ccl_manager, parallel_config)
    cache.load_model(
        tt_encoder,
        model_name=model_name_checkpoint,
        subfolder="text_encoder",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        get_torch_state_dict=lambda: hf_model.state_dict(),
    )

    # warmup
    tt_output = tt_encoder(tt_prompt, attention_mask=tt_mask)
    ttnn.synchronize_device(mesh_device)  # wait for all operations to complete
    benchmark_profiler = BenchmarkProfiler()
    num_runs = 10
    for i in range(num_runs):
        with benchmark_profiler("tt_umt5_encoder", i):
            tt_output = tt_encoder(tt_prompt, attention_mask=tt_mask)[-1]
            tt_output = ccl_manager.all_gather(tt_output, dim=0, mesh_axis=DP_axis, use_hyperparams=True)
            # wait for all operations to complete to get correct dispatch + execution time
            ttnn.synchronize_device(mesh_device)

    # get HF reference outputs
    with torch.no_grad():
        hf_outputs = hf_model(text_input_ids, attention_mask=mask).last_hidden_state

    # # convert mesh tensor to torch tensor for pcc
    # # since weights are replicated, can get the tensor from any single device
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    tt_execution_time = benchmark_profiler.get_duration_average("tt_umt5_encoder")
    logger.info(f"TT encoder execution time benchmark: {tt_execution_time:.4f} seconds")
    assert_quality(hf_outputs[:total_prompts], tt_output_torch[:total_prompts], pcc=0.99, relative_rmse=0.06)
    # assert (
    #    tt_execution_time < max_execution_time
    # ), f"TT Encoder execution time {tt_execution_time:.4f} seconds is greater than the max {max_execution_time} seconds"
