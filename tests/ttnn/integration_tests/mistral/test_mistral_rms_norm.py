# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn

import json
from pathlib import Path
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.reference.model import Transformer
from models.experimental.mistral.reference.model import RMSNorm
from models.experimental.functional_mistral.tt.ttnn_functional_rms_norm import rms_norm
from models.experimental.functional_mistral.tt.mistral_utility import custom_preprocessor
from models.utility_functions import skip_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
def test_mistral_rms_norm_inference(model_location_generator, device, reset_seeds):
    model_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    transformer = Transformer.from_folder(Path(model_path), n_layers=1, max_batch_size=1, is_whole_model=False)
    state_dict = torch.load(model_path / "consolidated.00.pth")
    state_dict = {k[24:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention_norm."))}
    with open(model_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    model_args.max_batch_size = 1
    model_args.n_layers = 1
    dim = 4096

    model_args.max_batch_size = 1
    ref_model = transformer.layers[0].attention_norm
    reference_model = RMSNorm(dim=dim)
    reference_model.load_state_dict(state_dict)

    input = torch.rand(1, 11, 4096)
    reference_ouput = reference_model(input)

    ttnn_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )
    output = rms_norm(model_args, ttnn_input, parameters=parameters.weight)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(reference_ouput, output.to(reference_ouput.dtype), 0.99)
