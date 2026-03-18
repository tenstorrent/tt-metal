#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import profiler
from models.demos.rvc.torch_impl.vc.hubert import HubertModel as TorchHubertModel
from models.demos.rvc.torch_impl.vc.hubert import HubertPretrainingConfig, HubertPretrainingTask
from models.demos.rvc.tt_impl.vc.hubert import HubertModel as TTHubertModel
from tests.ttnn.utils_for_testing import assert_with_pcc


def _get_test_hubert_cfg(layer_norm_first: bool) -> dict:
    return {
        "label_rate": 50,
        "extractor_mode": "default",
        "encoder_layers": 2,
        "encoder_embed_dim": 512,
        "encoder_ffn_embed_dim": 2048,
        "encoder_attention_heads": 8,
        "activation_fn": "gelu",
        "layer_type": "transformer",
        "final_dim": 256,
        "untie_final_proj": True,
        "layer_norm_first": layer_norm_first,
        "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        "conv_bias": False,
        "logit_temp": 0.1,
        "feature_grad_mult": 0.1,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "required_seq_len_multiple": 2,
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("layer_norm_first", [False, True])
def test_hubert_model(device, layer_norm_first):
    torch.manual_seed(0)
    profiler.clear()

    cfg = _get_test_hubert_cfg(layer_norm_first=layer_norm_first)
    task = HubertPretrainingTask(HubertPretrainingConfig())

    torch_model = TorchHubertModel(cfg, task.cfg).eval()
    tt_model = TTHubertModel(device=device, cfg=cfg, task_cfg=task.cfg)
    tt_model.load_state_dict(parameters=torch_model.state_dict())

    batch_size = 1
    input_length = 4096
    output_layer = cfg["encoder_layers"]

    torch_source = torch.randn(batch_size, input_length, dtype=torch.float32)
    tt_source = ttnn.from_torch(torch_source.unsqueeze(-1), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    torch_output = torch_model(torch_source, output_layer=output_layer)
    tt_output = tt_model(tt_source, output_layer=output_layer)

    assert tuple(tt_output.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output, pcc=0.99)

    torch_final_proj = torch_model.final_proj(torch_output)
    tt_final_proj = tt_model.final_proj(tt_output)

    assert tuple(tt_final_proj.shape) == tuple(torch_final_proj.shape)
    assert_with_pcc(torch_final_proj, tt_final_proj, pcc=0.95)

    tt_model.deallocate()
