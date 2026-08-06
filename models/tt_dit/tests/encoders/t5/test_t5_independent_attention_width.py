# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from transformers import T5Config as HFT5Config
from transformers import T5EncoderModel
from transformers.models.umt5.configuration_umt5 import UMT5Config as HFUMT5Config
from transformers.models.umt5.modeling_umt5 import UMT5EncoderModel

import ttnn
from models.tt_dit.encoders.t5.model_t5 import T5Config, T5Encoder
from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality


def test_t5_config_keeps_attention_width_independent_from_model_width():
    hunyuan = T5Config(embed_dim=1472, ff_dim=3584, kv_dim=64, num_heads=6, num_hidden_layers=12)
    umt5 = UMT5Config(embed_dim=4096, ff_dim=10240, kv_dim=64, num_heads=64, num_hidden_layers=24)

    assert hunyuan.attention_inner_dim == 384
    assert hunyuan.attention_inner_dim != hunyuan.embed_dim
    assert umt5.attention_inner_dim == umt5.embed_dim == 4096
    assert hunyuan.use_relative_position_bias == [True] + [False] * 11
    assert umt5.use_relative_position_bias == [True] * 24


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["hunyuan_byt5", "umt5_width_regression"])
def test_t5_encoder_random_weight_pcc_independent_attention_width(mesh_device, variant):
    """PCC the new unequal-width path and the established equal-width UMT5 path."""
    torch.manual_seed(0)
    if variant == "hunyuan_byt5":
        hf_config = HFT5Config(
            vocab_size=1510,
            d_model=1472,
            d_ff=3584,
            d_kv=64,
            num_heads=6,
            num_layers=12,
            num_decoder_layers=0,
            feed_forward_proj="gated-gelu",
            dense_act_fn="gelu_new",
            is_encoder_decoder=False,
            tie_word_embeddings=False,
        )
        hf_model = T5EncoderModel(hf_config)
        tt_config = T5Config(
            vocab_size=1510,
            embed_dim=1472,
            ff_dim=3584,
            kv_dim=64,
            num_heads=6,
            num_hidden_layers=12,
            max_prompt_length=256,
        )
        tt_type = T5Encoder
    else:
        # Keep the production UMT5 widths while limiting layers/sequence so this
        # focused regression does not duplicate the existing 24-layer perf test.
        hf_config = HFUMT5Config(
            vocab_size=256,
            d_model=4096,
            d_ff=10240,
            d_kv=64,
            num_heads=64,
            num_layers=2,
            num_decoder_layers=0,
            feed_forward_proj="gated-gelu",
            is_encoder_decoder=False,
        )
        hf_model = UMT5EncoderModel(hf_config)
        tt_config = UMT5Config(
            vocab_size=256,
            embed_dim=4096,
            ff_dim=10240,
            kv_dim=64,
            num_heads=64,
            num_hidden_layers=2,
            max_prompt_length=32,
        )
        tt_type = UMT5Encoder

    hf_model.eval()
    sequence_length = tt_config.max_prompt_length
    tokens = torch.randint(0, hf_config.vocab_size, (1, sequence_length))
    mask = torch.ones_like(tokens, dtype=torch.bfloat16)
    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=2, mesh_axis=1))
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    tt_model = tt_type(tt_config, mesh_device, ccl_manager, parallel_config)
    tt_model.load_torch_state_dict(hf_model.state_dict(), strict=True)
    tt_tokens = ttnn.from_torch(
        tokens,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_mask = ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    with torch.no_grad():
        reference = hf_model(tokens, attention_mask=mask.float()).last_hidden_state
    actual = ttnn.to_torch(ttnn.get_device_tensors(tt_model(tt_tokens, attention_mask=tt_mask)[-1])[0])

    assert actual.shape == reference.shape
    assert_quality(reference, actual, pcc=0.99)


@pytest.mark.skipif(
    os.environ.get("HY_RUN_REAL_BYT5") != "1",
    reason="set HY_RUN_REAL_BYT5=1 only on healthy idle hardware with a complete local checkpoint",
)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_hunyuan_byt5_real_weight_pcc_offline(mesh_device):
    """Opt-in real-weight gate; local_files_only prevents accidental downloads."""
    from models.demos.hf_eager.hunyuanvideo_1_5.tt.byt5_encoder import TTByT5EncoderAdapter

    repo = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
    try:
        hf_model = T5EncoderModel.from_pretrained(repo, subfolder="text_encoder_2", local_files_only=True)
    except OSError as error:
        pytest.skip(f"complete local byT5 checkpoint is unavailable: {error}")
    hf_model.eval()
    tokens = torch.randint(0, hf_model.config.vocab_size, (1, 256))
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        reference = hf_model(tokens, attention_mask=mask.float()).last_hidden_state

    adapter = TTByT5EncoderAdapter(hf_model, mesh_device)
    try:
        actual = adapter(tokens, attention_mask=mask.float())[0]
    finally:
        adapter.deallocate_weights()
    assert_quality(reference, actual, pcc=0.99)
