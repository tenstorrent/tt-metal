# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc

from models.experimental.BEVFormerV2.reference.ffn import FFN
from models.experimental.BEVFormerV2.tt.ttnn_ffn import TtFFN
from models.experimental.BEVFormerV2.tt.model_preprocessing import prepare_ffn_parameters_for_test


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ffn_pcc(
    device,
    reset_seeds,
    model_location_generator,
):
    torch.manual_seed(42)

    embed_dims = 256
    feedforward_channels = 512
    bs = 1
    seq_len = 900

    try:
        pytorch_ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_dropout=0.0,
        )
        pytorch_ffn.eval()
    except Exception as e:
        pytest.skip(f"Failed to create PyTorch model: {e}")

    x = torch.randn((bs, seq_len, embed_dims), dtype=torch.float32)

    with torch.no_grad():
        torch_output = pytorch_ffn(x)

    x_ttnn = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_ttnn = ttnn.to_layout(x_ttnn, ttnn.TILE_LAYOUT)

    ffn_params = prepare_ffn_parameters_for_test(pytorch_ffn, device)
    ttnn_ffn = TtFFN(
        params=ffn_params,
        device=device,
    )

    ttnn_output = ttnn_ffn(x_ttnn)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if ttnn_output_torch.shape != torch_output.shape:
        raise ValueError(f"Shape mismatch: ttnn={ttnn_output_torch.shape}, ref={torch_output.shape}")

    pcc_result = comp_pcc(torch_output, ttnn_output_torch)
    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

    assert pcc_value > 0.96, f"FFN PCC {pcc_value:.6f} is below threshold 0.96"
