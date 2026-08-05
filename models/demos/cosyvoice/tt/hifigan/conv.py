# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Conv1d for the HiFT vocoder: weight_norm folded at load, channels-last on device.

Two conventions this module fixes for the whole vocoder, because getting them
wrong in one place and right in another is the classic source of a silent
regression:

**Channels-last everywhere inside HiFT.** ttnn.conv1d takes `[N, L, C]` while the
reference works in `[N, C, L]`. Permuting at every layer boundary would cost a
transpose per conv on tensors that reach 512 channels at audio rate. So tensors
stay `[N, L, C]` from `conv_pre` to `conv_post` and are permuted exactly twice --
once on the way in, once on the way out.

**weight_norm folded on host.** Every conv in HiFT is wrapped in torch's
weight_norm, i.e. `w = g * v/||v||`. The norm is constant once the weights are
frozen, so computing it per inference is pure overhead. `from_torch_conv1d`
collapses it at construction.
"""
from __future__ import annotations

import torch

import ttnn


def accurate_compute_config(device):
    """High-fidelity compute config for the vocoder's convolutions.

    TTNN defaults to `MathFidelity.LoFi` with `fp32_dest_acc_en=False`. That is the
    right trade for most models, but HiFT is ~40 convolutions deep with a residual
    accumulating through all of them, and the errors compound: the full vocoder
    scored **PCC 0.98954** at the defaults, just under the 0.99 gate, with a
    provably correct graph.

    HiFi4 plus fp32 destination accumulation is the standard lever for exactly
    this -- depth-accumulated bfloat16 drift, not a wrong computation.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def fold_weight_norm(weight_v: torch.Tensor, weight_g: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """w = g * v / ||v||, with the norm taken over every axis except `dim`."""
    norm_dims = [d for d in range(weight_v.dim()) if d != dim]
    norm = weight_v.norm(2, dim=norm_dims, keepdim=True)
    return weight_g * weight_v / norm.clamp_min(1e-12)


def extract_conv_weights(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pull (weight, bias) out of a Conv1d whether or not weight_norm is applied.

    Handles both spellings torch has shipped: the legacy `weight_v`/`weight_g`
    attributes, and the newer `parametrizations.weight` container. Falling back to
    `.weight` covers the plain case.
    """
    if hasattr(module, "parametrizations") and "weight" in getattr(module, "parametrizations", {}):
        # torch.nn.utils.parametrizations.weight_norm
        p = module.parametrizations.weight
        w = fold_weight_norm(p.original1, p.original0, dim=p[0].dim)
    elif hasattr(module, "weight_v") and hasattr(module, "weight_g"):
        # legacy torch.nn.utils.weight_norm
        w = fold_weight_norm(module.weight_v, module.weight_g, dim=0)
    else:
        w = module.weight
    b = getattr(module, "bias", None)
    return w.detach().float(), (b.detach().float() if b is not None else None)


class TtConv1d:
    """A single Conv1d on device. Input and output are both `[N, L, C]`."""

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # [out_ch, in_ch/groups, k]
        bias: torch.Tensor | None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        high_fidelity: bool = True,
    ):
        assert weight.dim() == 3, f"expected [out_ch, in_ch/groups, k], got {tuple(weight.shape)}"
        self.device = device
        self.out_channels, self.in_per_group, self.kernel_size = weight.shape
        self.in_channels = self.in_per_group * groups
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        self.dtype = dtype

        self.weight = ttnn.from_torch(weight, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bias = None
        if bias is not None:
            # conv bias wants a 4-D [1, 1, 1, out_ch] row
            self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.conv_config = ttnn.Conv1dConfig(weights_dtype=weights_dtype, deallocate_activation=False)
        self.compute_config = accurate_compute_config(device) if high_fidelity else None

    @classmethod
    def from_module(cls, device, module: torch.nn.Module, **kw):
        """Build from a torch Conv1d, folding weight_norm if present."""
        w, b = extract_conv_weights(module)
        return cls(
            device,
            w,
            b,
            stride=int(module.stride[0]),
            padding=int(module.padding[0]) if isinstance(module.padding, (tuple, list)) else int(module.padding),
            dilation=int(module.dilation[0]),
            groups=int(module.groups),
            **kw,
        )

    def __call__(self, x, input_length: int, batch_size: int = 1):
        out, out_length = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_length=input_length,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            dtype=self.dtype,
            return_output_dim=True,
        )
        # ttnn.conv1d returns the flattened conv layout, not [N, L, C]. Restoring
        # the documented shape here rather than at each call site is what makes
        # `ttnn.permute(x, (0, 2, 1))` and the residual adds downstream legal --
        # otherwise the rank mismatch only surfaces at the first permute, far from
        # the conv that produced it.
        out = ttnn.reshape(out, (batch_size, out_length, self.out_channels))
        return out, out_length

    @staticmethod
    def out_length(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
        return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
