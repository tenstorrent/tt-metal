# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
import ttnn
from loguru import logger
from transformers.models.t5.modeling_t5 import T5EncoderModel

from ..reference.t5_encoder import T5Config, T5Encoder
from ..tt.t5_encoder import TtT5Encoder, TtT5EncoderParameters
from ..tt.utils import assert_quality
from ..tt.parallel_config import EncoderParallelManager


@pytest.mark.parametrize(
    "model_name",
    [
        "large",
    ],
)
@pytest.mark.parametrize("mesh_device", [(2, 4), (8, 4)], ids=["t3k", "tg"], indirect=True)
@pytest.mark.parametrize("submesh_shape", [(1, 4), (2, 2), (4, 4)], ids=["1x4", "2x2", "4x4"])
@pytest.mark.parametrize(
    "device_params, topology",
    [[{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear]],
    indirect=["device_params"],
)
def test_t5_encoder(*, mesh_device: ttnn.Device, submesh_shape: ttnn.MeshShape, model_name: str, topology) -> None:
    parent_mesh_shape = tuple(mesh_device.shape)
    if any(x[0] < x[1] for x in zip(parent_mesh_shape, submesh_shape)):
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")
    encoder_submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    print(f"Running on submesh {encoder_submesh.shape} of parent mesh {mesh_device.shape}")
    parallel_manager = EncoderParallelManager(encoder_submesh, topology, mesh_axis=1, num_links=1)

    hf_model = T5EncoderModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="text_encoder_3", local_files_only=True
    )

    with torch.device("meta"):
        torch_model = T5Encoder(
            T5Config(
                vocab_size=hf_model.config.vocab_size,
                d_model=hf_model.config.d_model,
                d_ff=hf_model.config.d_ff,
                d_kv=hf_model.config.d_kv,
                num_layers=hf_model.config.num_layers,
                num_heads=hf_model.config.num_heads,
                relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
                relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
                layer_norm_epsilon=hf_model.config.layer_norm_epsilon,
            )
        )
    torch_model.load_state_dict(hf_model.state_dict(), assign=True)
    torch_model.eval()

    start_time = time.time()
    parameters = TtT5EncoderParameters.from_torch(
        torch_model.state_dict(), device=encoder_submesh, dtype=ttnn.bfloat16, parallel_manager=parallel_manager
    )
    tt_model = TtT5Encoder(
        parameters,
        num_heads=hf_model.config.num_heads,
        relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
        relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
        layer_norm_epsilon=hf_model.config.layer_norm_epsilon,
    )
    logger.info(f"model creation time: {time.time() - start_time}")

    torch.manual_seed(0)
    tokens = torch.randint(hf_model.config.vocab_size, [1, 256])

    tt_tokens_host = ttnn.from_torch(
        tokens, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(encoder_submesh)
    )

    start_time = time.time()
    with torch.no_grad():
        output = torch_model(tokens)
    logger.info(f"CPU runtime: {time.time() - start_time}")

    tt_tokens = tt_tokens_host.to(encoder_submesh)

    logger.info("compiling...")
    tt_model(tt_tokens, encoder_submesh, parallel_manager)

    logger.info("executing...")
    start_time = time.time()
    tt_output = tt_model(tt_tokens, encoder_submesh, parallel_manager)
    logger.info(f"TT-NN runtime: {time.time() - start_time}")
    logger.info("done...")

    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])

    assert output.shape == tt_output_torch.shape
    assert_quality(output, tt_output_torch, pcc=0.945)
