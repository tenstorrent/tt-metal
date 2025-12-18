import pytest
import torch
from transformers import CLIPModel

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.clip_vit.tt.tt_clip_text import TtCLIPMLP
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size, seq_len, pcc, layer_idx", [(1, 77, 0.95, 0), (32, 77, 0.95, 0)])
def test_clip_text_mlp(batch_size, seq_len, pcc, layer_idx):
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model.eval()

    config = torch_model.config.text_config

    device = None
    try:
        device = ttnn.open_device(device_id=0)

        torch_mlp = torch_model.text_model.encoder.layers[layer_idx].mlp
        ttnn_mlp = TtCLIPMLP(config, torch_mlp, device)

        torch_input = torch.randn(batch_size, seq_len, config.hidden_size)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        with torch.no_grad():
            torch_output = torch_mlp(torch_input)

        ttnn_output = ttnn_mlp(ttnn_input)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=pcc)
        assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)
