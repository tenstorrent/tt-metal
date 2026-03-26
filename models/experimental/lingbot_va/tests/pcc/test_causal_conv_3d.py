# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC: diffusers WanCausalConv3d vs TTNN WanCausalConv3d (encoder conv_in)."""

import torch
import ttnn
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from models.common.metrics import compute_pcc
from models.tt_dit.models.vae.vae_wan2_1 import WanCausalConv3d as WanCausalConv3dTTNN
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import aligned_channels

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/vae"
BATCH_SIZE = 1
PCC_THRESHOLD = 0.99
H = 256
W = 320


def test_wan_causal_conv3d():
    autoencoder_klwan = AutoencoderKLWan.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")
    autoencoder_klwan.eval()

    conv_in_layer = autoencoder_klwan.encoder.conv_in
    actual_in_channels = conv_in_layer.in_channels
    actual_out_channels = conv_in_layer.out_channels
    actual_kernel_size = conv_in_layer.kernel_size
    actual_stride = conv_in_layer.stride
    actual_padding = conv_in_layer.padding

    temporal_kernel = actual_kernel_size[0] if isinstance(actual_kernel_size, tuple) else actual_kernel_size
    T = max(temporal_kernel, 4)

    wan_causal_conv3d_weights = conv_in_layer.state_dict()

    wan_causal_conv3d = WanCausalConv3d(
        actual_in_channels,
        actual_out_channels,
        actual_kernel_size[0] if isinstance(actual_kernel_size, tuple) else actual_kernel_size,
        actual_stride[0] if isinstance(actual_stride, tuple) else actual_stride,
        actual_padding[0] if isinstance(actual_padding, tuple) else actual_padding,
    )
    wan_causal_conv3d.load_state_dict(wan_causal_conv3d_weights)
    wan_causal_conv3d.eval()

    torch.manual_seed(42)
    input_tensor = torch.randn(BATCH_SIZE, actual_in_channels, T, H, W, dtype=torch.float32)
    input_tensor = input_tensor * 2.0 - 1.0

    with torch.no_grad():
        wan_causal_conv3d_out = wan_causal_conv3d(input_tensor)

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=1, mesh_axis=0),
            width_parallel=ParallelFactor(factor=1, mesh_axis=1),
        )

        wan_causal_conv3d_ttnn = WanCausalConv3dTTNN(
            actual_in_channels,
            actual_out_channels,
            kernel_size=actual_kernel_size[0] if isinstance(actual_kernel_size, tuple) else actual_kernel_size,
            stride=actual_stride[0] if isinstance(actual_stride, tuple) else actual_stride,
            padding=actual_padding[0] if isinstance(actual_padding, tuple) else actual_padding,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        wan_causal_conv3d_ttnn.load_torch_state_dict(wan_causal_conv3d_weights)

        tt_input_tensor = ttnn.from_torch(
            input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )
        tt_input_tensor = ttnn.permute(tt_input_tensor, (0, 2, 3, 4, 1))

        actual_C = tt_input_tensor.shape[-1]
        aligned_C = aligned_channels(actual_C)
        if aligned_C != actual_C:
            pad_amount = aligned_C - actual_C
            tt_input_tensor = ttnn.pad(tt_input_tensor, [(0, 0), (0, 0), (0, 0), (0, 0), (0, pad_amount)], 0.0)

        tt_out = ttnn.permute(wan_causal_conv3d_ttnn(tt_input_tensor, logical_h=H), (0, 4, 1, 2, 3))
        tt_torch = ttnn.to_torch(tt_out)

        pcc = compute_pcc(wan_causal_conv3d_out, tt_torch)
        assert pcc >= PCC_THRESHOLD, f"PCC {pcc} < {PCC_THRESHOLD}"
    finally:
        ttnn.close_mesh_device(mesh_device)
