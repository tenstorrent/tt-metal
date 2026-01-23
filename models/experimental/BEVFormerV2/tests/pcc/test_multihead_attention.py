# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.BEVFormerV2.reference.multihead_attention import MultiheadAttention
from models.experimental.BEVFormerV2.tt.ttnn_multihead_attention import TtMultiheadAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.experimental.BEVFormerV2.common import download_bevformerv2_weights


from models.experimental.BEVFormerV2.tests.pcc.custom_preprocessors import (
    custom_preprocessor_multihead_attention as custom_preprocessor,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_multihead_attention(
    device,
    reset_seeds,
    model_location_generator,
):
    embed_dims = 256
    num_heads = 8
    batch_first = False

    torch_model = MultiheadAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        batch_first=batch_first,
    )

    weights_path = download_bevformerv2_weights()
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    mha_state = {}
    for key, value in state_dict.items():
        if "decoder.layers.0.attentions.0" in key:
            new_key = key.replace("pts_bbox_head.transformer.decoder.layers.0.attentions.0.", "")
            mha_state[new_key] = value

    torch_model.load_state_dict(mha_state, strict=False)
    torch_model.eval()

    query = torch.randn(900, 1, 256)
    key = torch.randn(900, 1, 256)
    value = torch.randn(900, 1, 256)
    query_pos = torch.randn(900, 1, 256)

    torch_output = torch_model(
        query,
        key,
        value,
        query_pos=query_pos,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    ttnn_model = TtMultiheadAttention(
        params=parameters.multihead_attention,
        device=device,
        embed_dims=embed_dims,
        num_heads=num_heads,
        batch_first=batch_first,
    )

    query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    key = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    query_pos = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)

    ttnn_output = ttnn_model(
        query,
        key,
        value,
        query_pos=query_pos,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(pcc_message)
