# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test gradient-guided bf16 rounding on a deep CNN.

Verifies that PCC between the float32 reference output and the ttnn bf16
output improves after applying rounding correction.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from tools.quantization.gradient_guided_rounding import gradient_guided_bf16_rounding

NUM_LAYERS = 26
IN_CHANNELS = 64
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 64
HEIGHT = 32
WIDTH = 32
BATCH_SIZE = 1
KERNEL_SIZE = (3, 3)
STRIDE = (1, 1)
PADDING = (1, 1)

LAYER_NAMES = tuple(f"conv{i}" for i in range(NUM_LAYERS))


class TorchCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channels = [IN_CHANNELS] + [HIDDEN_CHANNELS] * (NUM_LAYERS - 1) + [OUT_CHANNELS]
        for i, name in enumerate(LAYER_NAMES):
            setattr(self, name, torch.nn.Conv2d(channels[i], channels[i + 1], 3, padding=1))

    def forward(self, x):
        for i, name in enumerate(LAYER_NAMES):
            x = getattr(self, name)(x)
            if i < NUM_LAYERS - 1:
                x = torch.relu(x)
        return x


class TtnnCNN:
    """Minimal ttnn mirror of TorchCNN using ttnn.conv2d + ttnn.relu."""

    def __init__(self, torch_model: TorchCNN, device):
        self.device = device
        channels = [IN_CHANNELS] + [HIDDEN_CHANNELS] * (NUM_LAYERS - 1) + [OUT_CHANNELS]
        self.in_channels = [channels[i] for i in range(NUM_LAYERS)]
        self.out_channels = [channels[i + 1] for i in range(NUM_LAYERS)]

        for name in LAYER_NAMES:
            layer = getattr(torch_model, name)
            weight = ttnn.from_torch(layer.weight.data, dtype=ttnn.bfloat16)
            bias = ttnn.from_torch(layer.bias.data.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)
            setattr(self, f"{name}_weight", weight)
            setattr(self, f"{name}_bias", bias)

    def __call__(self, x):
        h, w = HEIGHT, WIDTH
        for i, name in enumerate(LAYER_NAMES):
            weight = getattr(self, f"{name}_weight")
            bias = getattr(self, f"{name}_bias")

            [x, [h, w]] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=bias,
                in_channels=self.in_channels[i],
                out_channels=self.out_channels[i],
                device=self.device,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
                batch_size=BATCH_SIZE,
                input_height=h,
                input_width=w,
                return_output_dim=True,
            )
            if i < NUM_LAYERS - 1:
                x = ttnn.relu(x)
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return x, h, w


def _build_weight_mapping(ttnn_model, device):
    """
    Build a WeightMapping list for the CNN layers.

    Conv weights have the same OIHW layout in both torch and ttnn, so no
    transpose is needed.  Bias is stored as [1,1,1,O] in ttnn vs [O] in torch.
    """
    mapping = []
    for name in LAYER_NAMES:

        def _make_weight_pair(ln):
            def getter():
                return ttnn.to_torch(getattr(ttnn_model, f"{ln}_weight")).to(torch.bfloat16)

            def setter(t):
                setattr(
                    ttnn_model,
                    f"{ln}_weight",
                    ttnn.from_torch(t, dtype=ttnn.bfloat16),
                )

            return f"{ln}.weight", getter, setter

        def _make_bias_pair(ln):
            def getter():
                b = ttnn.to_torch(getattr(ttnn_model, f"{ln}_bias"))
                return b.flatten().to(torch.bfloat16)

            def setter(t):
                setattr(
                    ttnn_model,
                    f"{ln}_bias",
                    ttnn.from_torch(t.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                )

            return f"{ln}.bias", getter, setter

        mapping.append(_make_weight_pair(name))
        mapping.append(_make_bias_pair(name))
    return mapping


def _ttnn_forward(ttnn_model, device, dummy_input_nchw):
    """
    Run the ttnn CNN model and return the result as a float32 NCHW torch tensor.
    """
    x_nhwc = dummy_input_nchw.permute(0, 2, 3, 1).contiguous()
    tt_input = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_output, h, w = ttnn_model(tt_input)

    out = ttnn.to_torch(ttnn.from_device(tt_output))
    out = out.reshape(BATCH_SIZE, h, w, -1)[:, :, :, :OUT_CHANNELS]
    out = out.permute(0, 3, 1, 2).contiguous().float()
    return out


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("num_iterations", [1, 5])
def test_cnn_pcc_improves_after_rounding(device, num_iterations):
    torch.manual_seed(42)
    torch_model = TorchCNN().eval()
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)

    with torch.no_grad():
        ref_output = torch_model(dummy_input)

    ttnn_model = TtnnCNN(torch_model, device)

    before_output = _ttnn_forward(ttnn_model, device, dummy_input)
    _, pcc_before = comp_pcc(ref_output, before_output)

    weight_mapping = _build_weight_mapping(ttnn_model, device)

    gradient_guided_bf16_rounding(
        reference_module=torch_model,
        target_forward_fn=lambda x: _ttnn_forward(ttnn_model, device, x),
        dummy_input=dummy_input,
        weight_mapping=weight_mapping,
        num_iterations=num_iterations,
    )

    after_output = _ttnn_forward(ttnn_model, device, dummy_input)
    _, pcc_after = comp_pcc(ref_output, after_output)

    print(f"PCC before={pcc_before:.6f}, after={pcc_after:.6f}")
    assert pcc_after > pcc_before, f"PCC did not improve: before={pcc_before:.6f}, after={pcc_after:.6f}"
