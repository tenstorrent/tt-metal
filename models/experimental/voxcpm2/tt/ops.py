# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared device operations. Torch is used only for checkpoint preparation."""

import ttnn


def upload(value, device, dtype):
    return ttnn.from_torch(
        value.detach().cpu().contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class TtLinear:
    def __init__(self, state, prefix, device, dtype):
        weight = state[prefix + ".weight"]
        if weight.ndim != 2:
            raise ValueError(f"{prefix}: expected a matrix weight")
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = upload(weight.T, device, dtype)
        bias = state.get(prefix + ".bias")
        self.bias = None if bias is None else upload(bias.reshape(1, -1), device, dtype)
        self.compute = compute_config(device)

    def __call__(self, hidden):
        if hidden.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear input width {hidden.shape[-1]} != {self.in_features}"
            )
        return ttnn.linear(
            hidden,
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class TtRMSNorm:
    def __init__(self, state, prefix, epsilon, device, dtype):
        self.weight = upload(state[prefix + ".weight"].reshape(1, -1), device, dtype)
        self.epsilon = epsilon
        self.compute = compute_config(device)

    def __call__(self, hidden):
        return ttnn.rms_norm(
            hidden,
            weight=self.weight,
            epsilon=self.epsilon,
            compute_kernel_config=self.compute,
        )
