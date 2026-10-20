# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""ConvTranspose1d for HiFT's upsampling stages.

TTNN has `conv1d`, `conv2d` and `conv_transpose2d` but no native 1-D transpose, so
the 1-D transpose is expressed as a 2-D one with `H = 1` -- the same trick
`istft.py` uses for overlap-add, verified there at `H=1, in_ch=16, k=16, stride=4`.

CosyVoice2's HiFT upsamples in 3 stages, `upsample_rates [8, 5, 3]` with
`upsample_kernel_sizes [16, 11, 7]` -- unlike CosyVoice1's 2 stages `[8, 8]`/
`[16, 16]`, but nothing here is stage-count-specific: kernel size, stride and
padding all come from the weight/module handed in.
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn

from ..geometry_cache import GeometryWeightCache
from .conv import (
    AGREEMENT_TOLERANCE,
    accurate_compute_config,
    config_tensors_in_dram,
    extract_conv_weights,
    pick_most_accurate,
    relative_error,
    safe_compute_config,
)


class TtConvTranspose1d:
    """Transposed 1-D convolution via conv_transpose2d at H=1.

    Input and output are channels-last `[N, L, C]`, matching conv.py.
    """

    _warned = False

    def __init__(
        self,
        device,
        weight,
        bias,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        high_fidelity: bool = True,
    ):
        # torch ConvTranspose1d weight is [in_ch, out_ch/groups, k]; TTNN's
        # conv_transpose2d wants (C, O/G, K_H, K_W), so the extra H axis is 1.
        assert weight.dim() == 3, f"expected [in_ch, out_ch/groups, k], got {tuple(weight.shape)}"
        self.device = device
        self.in_channels, self.out_per_group, self.kernel_size = weight.shape
        self.out_channels = self.out_per_group * groups
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        self.dtype = dtype

        self.weight = ttnn.from_torch(weight.unsqueeze(2), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Per-geometry caches, threshold-evicted on real DRAM pressure -- see
        # geometry_cache.py and TtConv1d's identical pattern (same class, same
        # ownership-transfer reasoning applies here unchanged).
        self._prep_cache = GeometryWeightCache(device)
        # Host copies, used only as the tie-break reference in `_verify_and_resolve`.
        self._host_weight = weight.detach().float().clone()
        self._host_bias = bias.detach().float().clone() if bias is not None else None
        # prepare_conv_transpose2d_weights asserts conv_config.weights_dtype.has_value(),
        # so the config cannot be left to the op's default here.
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype, config_tensors_in_dram=config_tensors_in_dram()
        )
        self.bias = None
        if bias is not None:
            self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.compute_config = accurate_compute_config(device) if high_fidelity else None
        self._safe_compute_config = safe_compute_config(device) if high_fidelity else None
        # accurate_compute_config (HiFi4 + fp32_dest_acc_en=True + packer_l1_acc=True)
        # measured wrong on ttnn.conv1d at some shapes on this hardware -- see its
        # docstring in conv.py. Not yet independently confirmed at a wrong shape here
        # (isolated conv_transpose2d microbenchmarks passed at every kernel size
        # CosyVoice2's 3-stage topology introduces), but the failure mode is silent by
        # nature and conv_transpose2d shares the same compute-config plumbing as
        # conv1d, so this is verified on the same terms rather than assumed safe.
        # Same reasoning for `prepare_conv_transpose2d_weights` vs.
        # tenstorrent/tt-metal#55545 (filed against `prepare_conv_weights`, but the
        # same hoisted-weight-prep pattern on the same hardware) -- not
        # independently confirmed broken here either, but `_verify_and_resolve`
        # checks both axes together regardless, same as TtConv1d.
        # Maps (length, batch_size) -> the (weight, bias, compute_config) triple
        # that measured correct.
        self._verified_config = GeometryWeightCache(device)

    @classmethod
    def from_module(cls, device, module, **kw):
        w, b = extract_conv_weights(module)
        return cls(
            device,
            w,
            b,
            stride=int(module.stride[0]),
            padding=int(module.padding[0]),
            dilation=int(module.dilation[0]),
            groups=int(module.groups),
            **kw,
        )

    def out_length(self, length: int) -> int:
        return (length - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1

    def _prepared(self, nhwc, length: int, batch_size: int):
        key = (length, batch_size)
        cached = self._prep_cache.get(key)
        if cached is not None:
            return cached
        kw = dict(
            input_memory_config=nhwc.memory_config(),
            input_layout=nhwc.layout,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=length,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, self.dilation),
            groups=self.groups,
            device=self.device,
            input_dtype=self.dtype,
            conv_config=self.conv_config,
        )
        try:
            w = ttnn.prepare_conv_transpose2d_weights(
                weight_tensor=self.weight, weights_format="IOHW", has_bias=self.bias is not None, **kw
            )
            b = ttnn.prepare_conv_transpose2d_bias(bias_tensor=self.bias, **kw) if self.bias is not None else None
        except Exception as e:  # noqa: BLE001
            if not TtConvTranspose1d._warned:
                TtConvTranspose1d._warned = True
                logger.warning(f"prepare_conv_transpose2d_weights unavailable, stays untraceable: {str(e)[:200]}")
            w, b = self.weight, self.bias
        owned = [t for t in (w, b) if t is not None and t is not self.weight and t is not self.bias]
        self._prep_cache.put(key, owned, (w, b))
        return w, b

    def _conv_transpose(self, nhwc, weight, bias, length: int, batch_size: int, compute_config):
        out = ttnn.conv_transpose2d(
            input_tensor=nhwc,
            weight_tensor=weight,
            bias_tensor=bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=length,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, self.dilation),
            groups=self.groups,
            conv_config=self.conv_config,
            compute_config=compute_config,
            dtype=self.dtype,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    def _to_host(self, t, batch_size: int, out_length: int):
        return ttnn.to_torch(t).float().reshape(batch_size, out_length, self.out_channels).double()

    def _host_reference(self, nhwc, length: int, batch_size: int):
        """float64 torch conv_transpose1d of the same input with the same
        (un-quantised) weights, `[B, L_out, C_out]`. Only run to arbitrate a disagreement."""
        x_cf = ttnn.to_torch(nhwc).float().reshape(batch_size, length, self.in_channels).transpose(1, 2).double()
        bias = self._host_bias.double() if self._host_bias is not None else None
        ref = torch.nn.functional.conv_transpose1d(
            x_cf,
            self._host_weight.double(),
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return ref.transpose(1, 2)

    def _verify_and_resolve(self, nhwc, weight, bias, length: int, batch_size: int, out, key):
        """First call for this geometry only -- see TtConv1d._verify_and_resolve,
        same reasoning (a single raw-weight + safe-config reference, catching
        either the compute-config axis or the weight-prep axis at once) and
        same mechanism, adapted for conv_transpose2d: agreement is a relative L2
        distance within `AGREEMENT_TOLERANCE`, and a disagreement is arbitrated
        by a float64 host conv_transpose1d between prepared+accurate,
        raw+safe and raw+accurate, not by assuming the safe reference is right."""
        ref = self._conv_transpose(nhwc, self.weight, self.bias, length, batch_size, self._safe_compute_config)
        lo = self.out_length(length)
        fast_host = self._to_host(out, batch_size, lo)
        ref_host = self._to_host(ref, batch_size, lo)
        if relative_error(fast_host, ref_host) <= AGREEMENT_TOLERANCE:
            ttnn.deallocate(ref)
            self._resolve(key, (weight, bias, self.compute_config))
            return out

        raw_accurate = self._conv_transpose(nhwc, self.weight, self.bias, length, batch_size, self.compute_config)
        candidates = {
            "prepared weight + accurate config": (out, (weight, bias, self.compute_config)),
            "raw weight + safe config": (ref, (self.weight, self.bias, self._safe_compute_config)),
            "raw weight + accurate config": (raw_accurate, (self.weight, self.bias, self.compute_config)),
        }
        host = {
            "prepared weight + accurate config": fast_host,
            "raw weight + safe config": ref_host,
            "raw weight + accurate config": self._to_host(raw_accurate, batch_size, lo),
        }
        best, errors = pick_most_accurate(host, self._host_reference(nhwc, length, batch_size))
        logger.warning(
            f"fast conv path disagrees with the raw-weight/safe-config reference at "
            f"ConvTranspose1d({self.in_channels}->{self.out_channels}, k={self.kernel_size}, s={self.stride}) "
            f"length {length}; relative error vs float64 host conv: "
            + ", ".join(f"{n} {e:.4g}" for n, e in errors.items())
            + f" -> using {best}"
        )
        chosen_tensor, resolved = candidates[best]
        self._resolve(key, resolved)
        for name, (t, _) in candidates.items():
            if name != best:
                ttnn.deallocate(t)
        return chosen_tensor

    def _resolve(self, key, resolved: tuple) -> None:
        """See TtConv1d._resolve -- identical ownership-transfer reasoning: if the
        winning weight/bias are `_prep_cache[key]`'s prepared tensors, ownership moves to
        `_verified_config` (`discard`); otherwise any prepared tensors for this key are
        now unused and `pop` frees them for real."""
        weight, bias, _ = resolved
        prepared = self._prep_cache.get(key)
        if prepared is not None and prepared[0] is weight and prepared[1] is bias:
            self._prep_cache.discard(key)
        else:
            self._prep_cache.pop(key)
        owned = [t for t in (weight, bias) if t is not None and t is not self.weight and t is not self.bias]
        self._verified_config.put(key, owned, resolved)

    def __call__(self, x, length: int, batch_size: int = 1):
        """x: ttnn [B, L, C_in] -> (ttnn [B, L_out, C_out], L_out)."""
        nhwc = ttnn.reshape(x, (batch_size, 1, length, self.in_channels))
        key = (length, batch_size)
        verified = None if self.compute_config is None else self._verified_config.get(key)
        if self.compute_config is None:
            weight, bias = self._prepared(nhwc, length, batch_size)
            out = self._conv_transpose(nhwc, weight, bias, length, batch_size, None)
        elif verified is not None:
            weight, bias, compute_config = verified
            out = self._conv_transpose(nhwc, weight, bias, length, batch_size, compute_config)
        else:
            weight, bias = self._prepared(nhwc, length, batch_size)
            out = self._conv_transpose(nhwc, weight, bias, length, batch_size, self.compute_config)
            out = self._verify_and_resolve(nhwc, weight, bias, length, batch_size, out, key)
        lo = self.out_length(length)
        return ttnn.reshape(out, (batch_size, lo, self.out_channels)), lo

    def release_caches(self) -> None:
        """See TtConv1d.release_caches -- same purpose, same "not needed for normal
        per-utterance operation" note."""
        self._prep_cache.clear()
        self._verified_config.clear()
