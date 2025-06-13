# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from ttnn.device import is_wormhole_b0

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import (
    verify_with_pcc,
    UNET_FULL_MODEL_PCC,
    UNET_FULL_MODEL_PCC_BH,
    UNET_L1_SMALL_REGION_SIZE,
)


def run_unet_model(batch, groups, device, iterations=1):
    torch_input, ttnn_input = create_unet_input_tensors(
        batch,
        groups,
        channel_order="first",
        pad=False,
        fold=True,
        device=device,
        memory_config=unet_shallow_ttnn.UNet.input_sharded_memory_config,
    )
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_output_tensor = model(torch_input)
    output_tensor = ttnn_model(ttnn_input, move_input_tensor_to_device=False, deallocate_input_activation=True).cpu()

    B, C, H, W = torch_output_tensor.shape
    ttnn_output_tensor = ttnn.to_torch(output_tensor).reshape(B, C, H, W)
    verify_with_pcc(
        torch_output_tensor,
        ttnn_output_tensor,
        UNET_FULL_MODEL_PCC if is_wormhole_b0(device) else UNET_FULL_MODEL_PCC_BH,
    )

    for _ in range(iterations - 1):
        _, ttnn_input = create_unet_input_tensors(
            batch,
            groups,
            channel_order="first",
            pad=False,
            fold=True,
            device=device,
            memory_config=unet_shallow_ttnn.UNet.input_sharded_memory_config,
        )
        ttnn_model(ttnn_input, move_input_tensor_to_device=False, deallocate_input_activation=True).cpu()
        ttnn.DumpDeviceProfiler(device)


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("groups", [4])
@pytest.mark.parametrize("device_params", [{"l1_small_size": UNET_L1_SMALL_REGION_SIZE}], indirect=True)
def test_unet_model(batch, groups, device, use_program_cache, reset_seeds):
    if (
        not is_wormhole_b0(device)
        and device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y != 110
    ):
        pytest.skip(f"Shallow UNet only support 110 cores on BH (was {device.compute_with_storage_grid_size()})")
    run_unet_model(batch, groups, device)
