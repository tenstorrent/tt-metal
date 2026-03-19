# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test gradient-guided bf16 rounding on a deep MLP.

Verifies that PCC between the float32 reference output and the ttnn bf16
output improves after applying rounding correction.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
from tools.quantization.gradient_guided_rounding import gradient_guided_bf16_rounding

NUM_LAYERS = 16
HIDDEN = 256
IN_FEATURES = 256
OUT_FEATURES = 256
BATCH_SIZE = 32
MATH_FIDELITIES = [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]

LAYER_NAMES = tuple(f"fc{i}" for i in range(NUM_LAYERS))


class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dims = [IN_FEATURES] + [HIDDEN] * (NUM_LAYERS - 1) + [OUT_FEATURES]
        for i, name in enumerate(LAYER_NAMES):
            setattr(self, name, torch.nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for i, name in enumerate(LAYER_NAMES):
            x = getattr(self, name)(x)
            if i < NUM_LAYERS - 1:
                x = torch.relu(x)
        return x


class TtnnMLP:
    """Minimal ttnn mirror of TorchMLP using ttnn.linear + ttnn.relu."""

    def __init__(self, torch_model: TorchMLP, device, math_fidelity):
        self.device = device
        ComputeConfigClass = (
            ttnn.types.BlackholeComputeKernelConfig if ttnn.device.is_blackhole() else ttnn.WormholeComputeKernelConfig
        )
        self.compute_kernel_config = ComputeConfigClass(math_fidelity=math_fidelity, math_approx_mode=True)
        for name in LAYER_NAMES:
            layer = getattr(torch_model, name)
            weight = preprocess_linear_weight(layer.weight.data, dtype=ttnn.bfloat16)
            bias = preprocess_linear_bias(layer.bias.data, dtype=ttnn.bfloat16)
            setattr(self, f"{name}_weight", ttnn.to_device(weight, device))
            setattr(self, f"{name}_bias", ttnn.to_device(bias, device))

    def __call__(self, x):
        for i, name in enumerate(LAYER_NAMES):
            x = ttnn.linear(
                x,
                getattr(self, f"{name}_weight"),
                bias=getattr(self, f"{name}_bias"),
                compute_kernel_config=self.compute_kernel_config,
            )
            if i < NUM_LAYERS - 1:
                x = ttnn.relu(x)
        return x


def _build_weight_mapping(ttnn_model, device):
    """
    Build a WeightMapping list that bridges torch parameter shapes and
    ttnn storage conventions (transposed weight, reshaped bias).
    """
    mapping = []
    for name in LAYER_NAMES:

        def _make_weight_pair(ln):
            def getter():
                w = ttnn.to_torch(getattr(ttnn_model, f"{ln}_weight"))
                return w.T.contiguous().to(torch.bfloat16)

            def setter(t):
                setattr(
                    ttnn_model,
                    f"{ln}_weight",
                    ttnn.from_torch(
                        t.T.contiguous(),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    ),
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
                    ttnn.from_torch(
                        t.reshape(1, -1),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    ),
                )

            return f"{ln}.bias", getter, setter

        mapping.append(_make_weight_pair(name))
        mapping.append(_make_bias_pair(name))
    return mapping


def _ttnn_forward(ttnn_model, device, dummy_input):
    """Run ttnn model and return result as float32 torch tensor."""
    tt_input = ttnn.from_torch(
        dummy_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_output = ttnn_model(tt_input)
    return ttnn.to_torch(tt_output).float()


@pytest.mark.parametrize("math_fidelity", MATH_FIDELITIES)
@pytest.mark.parametrize("num_iterations", [1, 5])
def test_mlp_pcc_improves_after_rounding(device, num_iterations, math_fidelity, record_pcc_result):
    torch.manual_seed(42)
    torch_model = TorchMLP().eval()
    dummy_input = torch.randn(BATCH_SIZE, IN_FEATURES)

    with torch.no_grad():
        ref_output = torch_model(dummy_input)

    ttnn_model = TtnnMLP(torch_model, device, math_fidelity)

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

    record_pcc_result(
        test="MLP",
        fidelity=math_fidelity,
        iters=num_iterations,
        pcc_before=pcc_before,
        pcc_after=pcc_after,
    )
    assert pcc_after > pcc_before, f"PCC did not improve: before={pcc_before:.6f}, after={pcc_after:.6f}"
