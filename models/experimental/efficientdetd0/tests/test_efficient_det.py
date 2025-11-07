# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.tt.efficient_det import TtEfficientDetBackbone

import pytest

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    create_custom_mesh_preprocessor,
)
from ttnn.model_preprocessing import infer_ttnn_module_args
from models.demos.utils.common_demo_utils import get_mesh_mappers

torch.manual_seed(0)


@pytest.mark.parametrize(
    "batch, channels, height, width",
    [
        (1, 3, 512, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_efficient_det(batch, channels, height, width, device):
    torch_model = EfficientDetBackbone(
        num_classes=80,
        compound_coef=0,
        load_weights=False,
    )
    torch_model.eval()
    # torch_model.to(device)
    # Run PyTorch forward pass
    torch_inputs = torch.randn(batch, channels, height, width)
    with torch.no_grad():
        torch_outputs = torch_model(torch_inputs)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(
            # mesh_mapper=weights_mesh_mapper,
            input_tensor=torch_inputs,
            device=device,
        ),
        device=device,
    )
    print(".........................................................")
    print(f"parameters: {parameters=}")
    print(".........................................................")
    print(".........................................................")
    # Infer module arguments for all Conv2d layers
    module_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda torch_model: torch_model(torch_inputs), device=None
    )

    # Create TTNN BiFPN model
    ttnn_model = TtEfficientDetBackbone(
        device=device,
        parameters=parameters,
        conv_params=module_args,
        num_classes=80,
        compound_coef=0,
    )
    pytest.skip("Skipping efficient det test")
