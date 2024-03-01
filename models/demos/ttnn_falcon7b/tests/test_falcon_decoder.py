# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.demos.ttnn_falcon7b.tt.falcon_decoder import TtFalconDecoderLayer
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


torch.manual_seed(0)


def run_test_FalconDecoder_inference(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    layer_num,
    pcc,
    model_config,
):
    configuration = transformers.FalconConfig.from_pretrained(model_version)
    model = transformers.models.falcon.modeling_falcon.FalconDecoderLayer(configuration).eval()

    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    user_id = 0

    # Generate input, attention_mask, and kv_cache --------------------------------------
    # TODO: Generate attention_mask on device
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert batch == 1, "For prefill, batch must be 1!"
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        decoder_input = (torch.rand(batch, q_len, configuration.hidden_size) * 2) - 1
        attention_mask_bool = torch.zeros(batch, 71, q_len, kv_len, dtype=int)  # TODO(cfjchu): change 71 to 1
        layer_past = None

        tt_decoder_input = ttnn.from_torch(
            decoder_input.unsqueeze(1), dtype=model_config["DEFAULT_DTYPE"], layout=ttnn.TILE_LAYOUT, device=device
        )
        tt_attention_mask = ttnn.from_torch(
            attention_mask_bool, dtype=model_config["DEFAULT_DTYPE"], layout=ttnn.TILE_LAYOUT, device=device
        )
        tt_k_cache = torch.zeros(batch, configuration.max_position_embeddings, head_dim)
        tt_v_cache = torch.zeros(batch, configuration.max_position_embeddings, head_dim)

        tt_k_cache = ttnn.from_torch(
            tt_k_cache.unsqueeze(1), device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
        )
        tt_v_cache = ttnn.from_torch(
            tt_v_cache.unsqueeze(1), device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
        )
        tt_layer_past = (tt_k_cache, tt_v_cache)
        position_ids = None

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        decoder_input = (torch.rand(batch, q_len, configuration.hidden_size) * 2) - 1

        attention_mask_bool = torch.zeros(batch, 71, q_len, kv_len, dtype=int)  # TODO(cfjchu): change 71 to 1
        attention_mask_bool[:, :, :, -1] = True

        tt_attention_mask = ttnn.from_torch(
            attention_mask_bool, dtype=model_config["DEFAULT_DTYPE"], layout=ttnn.TILE_LAYOUT, device=device
        )

        k_cache = torch.rand(batch, kv_cache_len, head_dim)
        v_cache = torch.rand(batch, kv_cache_len, head_dim)
        layer_past = (k_cache, v_cache)

        tt_decoder_input = ttnn.from_torch(
            decoder_input.unsqueeze(1).transpose(0, 2),
            dtype=model_config["DEFAULT_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        kv_len_padded = (kv_len + 31) // 32 * 32
        attention_mask_bool_padded = torch.cat(
            (
                attention_mask_bool,
                torch.ones(batch, 71, q_len, kv_len_padded - kv_len, dtype=int),
            ),
            dim=-1,
        )
        tt_attention_mask = ttnn.from_torch(
            (attention_mask_bool_padded.transpose(0, 2) * -100000).expand(
                -1, configuration.num_attention_heads, -1, -1
            ),
            dtype=model_config["DEFAULT_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        k_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
        v_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
        layer_past = (k_cache, v_cache)

        tt_k_cache = torch.zeros(batch, configuration.max_position_embeddings, head_dim)
        tt_v_cache = torch.zeros(batch, configuration.max_position_embeddings, head_dim)
        tt_k_cache[:, :kv_cache_len, :] = k_cache.squeeze(1)
        tt_v_cache[:, :kv_cache_len, :] = v_cache.squeeze(1)
        tt_k_cache = ttnn.from_torch(
            tt_k_cache.unsqueeze(1), device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
        )
        tt_v_cache = ttnn.from_torch(
            tt_v_cache.unsqueeze(1), device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
        )
        tt_layer_past = (tt_k_cache, tt_v_cache)
        position_ids = torch.LongTensor([kv_cache_len])

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    pytorch_out, pytorch_layer_present = model(
        hidden_states=decoder_input,
        alibi=None,
        attention_mask=attention_mask_bool,
        position_ids=position_ids,
        layer_past=layer_past,
        use_cache=True,
    )

    tt_cache_path = get_tt_cache_path(model_version)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=create_custom_preprocessor(model_config, tt_cache_path=tt_cache_path, device=device),
    )
    tt_FalconDecoder_model = TtFalconDecoderLayer(
        device,
        configuration,
        max_position_embeddings,
        model_config,
        parameters,
    )

    tt_out, tt_layer_present = tt_FalconDecoder_model(
        hidden_states=tt_decoder_input,
        llm_mode=llm_mode,
        alibi=None,
        attention_mask=tt_attention_mask,
        user_id=user_id,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
        use_cache=True,
    )
    tt_out = ttnn.to_torch(tt_out).squeeze(1)

    tt_layer_present = (
        ttnn.to_torch(tt_layer_present[0]).squeeze(1),
        ttnn.to_torch(tt_layer_present[1]).squeeze(1),
    )
    if llm_mode == "decode":
        tt_out = tt_out.transpose(0, 1)
    tt_layer_present = (
        tt_layer_present[0][:, :kv_len, :],
        tt_layer_present[1][:, :kv_len, :],
    )

    assert_with_pcc(pytorch_out, tt_out.to(pytorch_out.dtype), pcc)
    assert_with_pcc(pytorch_layer_present[0].squeeze(1), tt_layer_present[0].to(pytorch_layer_present[0].dtype), pcc)
    assert_with_pcc(pytorch_layer_present[1].squeeze(1), tt_layer_present[1].to(pytorch_layer_present[1].dtype), pcc)


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128", "decode_batch32"],
)
@pytest.mark.parametrize(
    "model_version, layer_num, pcc",
    (("tiiuae/falcon-7b-instruct", 0, 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_FalconDecoder_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    layer_num,
    pcc,
    model_config_str,
    device,
):
    model_config = get_model_config(model_config_str)

    run_test_FalconDecoder_inference(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        layer_num,
        pcc,
        model_config,
    )
