# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def prepare_grid_sample_grid_pytorch(grid, input_shape):
    """
    Implementacija "grid_prepare-a" u python-u
    Input_shape je ulaz feature slike u NHWC formatu (na primer [1, 48, 160, 256])
    Grid je ono sto bi inace bio ulaz u grid_sample (tj ulaz u pytorch grid sample)
    To bi iz reference implementacije bili ovi,

    top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]])
    btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]])
    top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]])
    btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]]),

    gde je svaki dimenzije (1, 25281, 7, 2)

    Izlaz ovoga je preparovan grid, velicine (1, 25281, 7, 6)

    Prve dve vrednosti predstavljuju gornji levi pixel koji treba odgovarajucem output sticku za bilinearnu interpolaciju,
    a sledecih 6 vrednosti predstavljaju tezine za bilinearnu interpolaciju.

    U samom testu pise kako sam ja ovo iskoristio
    """
    batch_size, input_h, input_w, channels = input_shape
    grid_n, grid_h, grid_w, _ = grid.shape

    # Extract x and y coordinates
    x_coord = grid[..., 0]  # Shape: (N, H_out, W_out)
    y_coord = grid[..., 1]  # Shape: (N, H_out, W_out)

    # Scale factors for coordinate transformation (align_corners=False)
    height_scale = float(input_h) * 0.5
    height_offset = height_scale - 0.5
    width_scale = float(input_w) * 0.5
    width_offset = width_scale - 0.5

    # Transform to image coordinates
    h_coord_image = y_coord * height_scale + height_offset
    w_coord_image = x_coord * width_scale + width_offset

    # Get corner pixel coordinates (floor operation)
    h0 = torch.floor(h_coord_image).to(torch.int32)
    w0 = torch.floor(w_coord_image).to(torch.int32)
    h1 = h0 + 1
    w1 = w0 + 1

    # Boundary checks
    h0_valid = (h0 >= 0) & (h0 < input_h)
    h1_valid = (h1 >= 0) & (h1 < input_h)
    w0_valid = (w0 >= 0) & (w0 < input_w)
    w1_valid = (w1 >= 0) & (w1 < input_w)

    # Calculate interpolation weights
    h_frac = h_coord_image - h0.float()
    w_frac = w_coord_image - w0.float()
    h_frac_inv = 1.0 - h_frac
    w_frac_inv = 1.0 - w_frac

    # Compute bilinear weights with boundary conditions
    weight_nw = torch.where(h0_valid & w0_valid, h_frac_inv * w_frac_inv, torch.tensor(0.0))
    weight_ne = torch.where(h0_valid & w1_valid, h_frac_inv * w_frac, torch.tensor(0.0))
    weight_sw = torch.where(h1_valid & w0_valid, h_frac * w_frac_inv, torch.tensor(0.0))
    weight_se = torch.where(h1_valid & w1_valid, h_frac * w_frac, torch.tensor(0.0))

    # Clamp coordinates to 16-bit range
    h0_clamped = torch.clamp(h0, -32768, 32767).to(torch.int16)
    w0_clamped = torch.clamp(w0, -32768, 32767).to(torch.int16)

    # Convert int16 bit representation to bfloat16 (reinterpret bits, not values)
    # First convert int16 to uint16, then reinterpret as bfloat16
    h0_bits = h0_clamped.view(torch.uint16)
    w0_bits = w0_clamped.view(torch.uint16)

    # Create bfloat16 tensors with the same bit pattern
    # We need to create a tensor where the bfloat16 bits match the uint16 bits
    h0_as_bf16 = torch.zeros_like(h0_bits, dtype=torch.bfloat16)
    w0_as_bf16 = torch.zeros_like(w0_bits, dtype=torch.bfloat16)

    # Copy the bit pattern by viewing as bytes and reconstructing
    h0_as_bf16.view(torch.uint16).copy_(h0_bits)
    w0_as_bf16.view(torch.uint16).copy_(w0_bits)

    # Stack results into output tensor
    output = torch.stack(
        [
            h0_as_bf16,  # North-west height coordinate (as bit pattern)
            w0_as_bf16,  # North-west width coordinate (as bit pattern)
            weight_nw,  # Weight for north-west pixel
            weight_ne,  # Weight for north-east pixel
            weight_sw,  # Weight for south-west pixel
            weight_se,  # Weight for south-east pixel
        ],
        dim=-1,
    )

    # Return as float32, conversion to bfloat16 done later
    return output.float()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_shape_nchw, base_grid_shape, channel_extent_factor",
    [
        ((1, 256, 48, 160), (1, 25281, 7, 2), 7),
    ],
)
def test_pytorch_precomputed_grid_channel_extending(device, input_shape_nchw, base_grid_shape, channel_extent_factor):
    """
    Sta je ovde fora. Imamo dimenzije input slike, kao i dimenzije grida.
    Grid je dimenzije ove (1, 25281, 7, 2), to je bas taj grid sa kojim runu-je pytorch

    Sto se tice onoga sto radimo mi, malo je tricky.
    Prvo, uradi se preapre_grid_sample_grid_pytorch, koji pretvara grid iz (1, 25281, 7, 2) u (1, 25281, 7, 6), ovaj grid nazovimo A
    Posle toga se radi ovaj view, tj reshape tog grida, tako da on bude (1, 25281, 7/7, 6*7) = (1, 25281, 1, 42)
    Ovo je pozajmljivanje od pretposlednje dimenzije poslednjoj, ono sluzi kao oznaka da shape izlaza iz ttnn.grid_sample bude
    shape-a (1, 25281, 1, 256*7), a i dobar je za perf
    Posle toga se iz pytorch-a taj grid prebaci u bfloat16, pa se prebaci na device
    Sa istim tim gridom se na kraju uradi grid sample

    Bitna stvar je sledeca. Ovaj grid a, u svojih poslednjih 4 elementa (tj A[..., 2:6]) sadrzi neke weightove.
    To su obicni brojevi, koji se mnoze sa tim stickovima pa to ide na redukciju.
    Kada bi uzeo A[..., 2:6] i pomnozio sa area-om (ili podelio ne secam se kako), dobio bi unapred te neke weightove, pa umesto da deljenje ide na kraju
    ide u okviru grid_sample-a tehnicki. Nadam se da bi ovo popravilo pcc, mada nisam siguran, svakako je to deljenjem malim brojevima uvek nezgodno
    Dole sam ti napisao primer sta bi uradio ako bi iskoristio area-u
    """
    torch.manual_seed(0)

    batch_size, channels, height, width = input_shape_nchw
    input_shape_nhwc = [batch_size, height, width, channels]
    grid_n, grid_h, grid_w, grid_coords = base_grid_shape

    # Step 1: Get the normal pytorch grid in fp32
    torch_grid = torch.rand(base_grid_shape, dtype=torch.float32) * 2.0 - 1.0

    # Step 2: Preprocess it in python
    pytorch_precomputed = prepare_grid_sample_grid_pytorch(torch_grid, input_shape_nhwc)

    """
    pytorch_precomputed[2:6] = pytorch_precomputed[2:6]*area, ako je area odgovarajucih dimenzija, nisam siguran
    """

    # Step 3: Reshape it so that the grid, instead of being 1, H_out, W_out, 6, make it into 1, H_out, W_out/channel_extent_factor, 6*channel_extent_factor
    new_grid_w = grid_w // channel_extent_factor
    final_last_dim = 6 * channel_extent_factor
    pytorch_reshaped = pytorch_precomputed.view(batch_size, grid_h, new_grid_w, final_last_dim)

    # Step 4: Convert that grid to bfloat16
    pytorch_reshaped_bf16 = pytorch_reshaped.to(torch.bfloat16)

    # Step 5: Send it to ttnn on device
    ttnn_grid_device = ttnn.from_torch(pytorch_reshaped_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Create input tensor
    torch_input_nchw = torch.randn(input_shape_nchw, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Step 6: Run ttnn grid sample
    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_device, use_precomputed_grid=True)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Expected output using PyTorch grid_sample
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Reshape expected output for channel extending
    torch_expected_reshaped = (
        torch_output_nhwc.view(batch_size, grid_h, new_grid_w, channel_extent_factor, channels)
        .contiguous()
        .view(batch_size, grid_h, new_grid_w, channels * channel_extent_factor)
    )

    # Step 7: Compare to torch with pcc
    pcc_passed, pcc_message = assert_with_pcc(torch_expected_reshaped, ttnn_output_torch, pcc=0.99)
    assert pcc_passed, f"PCC test failed: {pcc_message}"
