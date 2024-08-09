# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import SegformerModel
import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_selfoutput import TtSegformerSelfOutput
from models.experimental.functional_segformer.reference.segformer_selfoutput import SegformerSelfOutput


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, block_i, self_output_i",
    [
        (1, 16384, 32, 0, 0),
        (1, 16384, 32, 0, 1),
        (1, 4096, 64, 1, 0),
        (1, 4096, 64, 1, 1),
        (1, 1024, 160, 2, 0),
        (1, 1024, 160, 2, 1),
        (1, 256, 256, 3, 0),
        (1, 256, 256, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_selfoutput(device, block_i, self_output_i, batch_size, seq_len, hidden_size, reset_seeds):
    torch_input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    torch_model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config = torch_model.config
    torch_model = torch_model.encoder.block[block_i][self_output_i].attention.output

    reference_model = SegformerSelfOutput(config=config, hidden_size=hidden_size)
    sd = torch_model.state_dict()
    reference_model.load_state_dict(sd)
    reference_model.eval()

    torch_output = reference_model(torch_input_tensor, None)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(torch_input_tensor, None),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtSegformerSelfOutput()

    ttnn_output = ttnn_model(ttnn_input_tensor, parameters=parameters)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
