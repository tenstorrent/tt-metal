# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Conv1d for the HiFT vocoder: weight_norm folded at load, channels-last on device.

Two conventions this module fixes for the whole vocoder:

**Channels-last everywhere inside HiFT.** ttnn.conv1d takes `[N, L, C]` while
upstream's `cosyvoice.hifigan.generator` works in `[N, C, L]`. Permuting at every
layer boundary would cost a transpose per conv on tensors that reach 512 channels
at audio rate, so tensors stay `[N, L, C]` from `conv_pre` to `conv_post` and are
permuted exactly twice -- once on the way in, once on the way out.

**weight_norm folded on host.** Every conv in HiFT is wrapped in torch's
weight_norm, i.e. `w = g * v/||v||`. The norm is constant once the weights are
frozen, so computing it per inference is pure overhead; `extract_conv_weights`
collapses it at construction.
"""

from __future__ import annotations

from loguru import logger

import ttnn


def accurate_compute_config(device):
    """High-fidelity compute config for the vocoder's convolutions.

    HiFT is ~40 convolutions deep with a residual accumulating through all of
    them, and bf16 drift compounds over that depth. HiFi4 plus fp32 destination
    accumulation is the standard lever for depth-accumulated bfloat16 drift, not
    a wrong computation -- see the CosyVoice1 TTNN bring-up
    (tenstorrent/tt-metal#52540), which measured the full vocoder just under a
    0.99 PCC gate at LoFi/no-fp32-accum on a provably correct graph, and fixed
    it with this config.

    **This exact config (HiFi4 + fp32_dest_acc_en=True + packer_l1_acc=True) has
    since been found to silently corrupt `ttnn.conv1d` at some shapes on this
    build/hardware** -- measured PCC 0.0011 (should be ~0.9999) at
    `in_channels=18, kernel=16, stride=8, padding=4`, the `source_downs` shape
    CosyVoice2's 3-stage HiFT introduces. `HiFi4` with `fp32_dest_acc_en=False`
    (same `packer_l1_acc=True`) measured correct at the same shape (PCC 0.9999),
    so the disagreement is specifically the `fp32_dest_acc_en` + `packer_l1_acc`
    combination, not HiFi4 itself. This is the same *class* of issue
    `TtConv1d._prepared`'s docstring already documents for `prepare_conv_weights`
    on Wormhole (a disagreement up to 1e37 at some input lengths) -- a different
    symptom, same lesson: an "accurate" compute config is not safe to trust
    unverified at an unfamiliar shape on this hardware. `TtConv1d` verifies this
    config against `safe_compute_config` once per geometry and falls back if
    they disagree; see `TtConv1d._verify_and_resolve`. `TtConvTranspose1d` in
    upsample.py verifies the same way, on the same suspicion.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def safe_compute_config(device):
    """HiFi4 without the `fp32_dest_acc_en` + `packer_l1_acc` combination that
    `accurate_compute_config` documents as unsafe at some conv1d shapes. Still
    HiFi4 (not LoFi), so this keeps most of the fidelity benefit; it is the
    fallback `TtConv1d` verifies against and switches to per-geometry.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def fold_weight_norm(weight_v, weight_g, dim: int = 0):
    """w = g * v / ||v||, with the norm taken over every axis except `dim`."""
    norm_dims = [d for d in range(weight_v.dim()) if d != dim]
    norm = weight_v.norm(2, dim=norm_dims, keepdim=True)
    return weight_g * weight_v / norm.clamp_min(1e-12)


def extract_conv_weights(module):
    """Pull (weight, bias) out of a Conv1d/ConvTranspose1d whether or not
    weight_norm is applied.

    Handles both spellings torch has shipped: the legacy `weight_v`/`weight_g`
    attributes, and the newer `parametrizations.weight` container. Falls back to
    `.weight` for a plain (unwrapped) conv.
    """
    if hasattr(module, "parametrizations") and "weight" in getattr(module, "parametrizations", {}):
        p = module.parametrizations.weight
        w = fold_weight_norm(p.original1, p.original0, dim=p[0].dim)
    elif hasattr(module, "weight_v") and hasattr(module, "weight_g"):
        w = fold_weight_norm(module.weight_v, module.weight_g, dim=0)
    else:
        w = module.weight
    b = getattr(module, "bias", None)
    return w.detach().float(), (b.detach().float() if b is not None else None)


class TtConv1d:
    """A single Conv1d on device. Input and output are both `[N, L, C]`."""

    _warned = False  # one warning per process, not per convolution

    def __init__(
        self,
        device,
        weight,  # [out_ch, in_ch/groups, k]
        bias,
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

        # OIHW with H=1, which is what prepare_conv_weights wants for a 1-D conv.
        self._weight_4d = ttnn.from_torch(weight.unsqueeze(2), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.weight = ttnn.from_torch(weight, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._prep_cache: dict = {}
        self.bias = None
        if bias is not None:
            self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.conv_config = ttnn.Conv1dConfig(weights_dtype=weights_dtype, deallocate_activation=False)
        self.compute_config = accurate_compute_config(device) if high_fidelity else None
        self._safe_compute_config = safe_compute_config(device) if high_fidelity else None
        # Verify accurate_compute_config against safe_compute_config once per
        # geometry rather than trust it -- see accurate_compute_config's
        # docstring for the measured disagreement this guards against. Maps
        # (input_length, batch_size) -> the compute_config that measured correct
        # for that geometry, so a disagreement is resolved once, not re-checked
        # (and not silently re-broken) on every subsequent call.
        self._verified_config: dict = {}

    @classmethod
    def from_module(cls, device, module, **kw):
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

    def _prepared(self, x, input_length: int, batch_size: int):
        """Pre-tilized, device-resident weights, cached per input geometry.

        `ttnn.conv1d` will accept a host-layout weight and prepare it internally,
        but it does that on every call -- pure per-call overhead on a weight that
        never changes, and it makes the op untraceable (a trace forbids host
        traffic). `prepare_conv_weights` hoists the transform out once per
        geometry; falls back to the unprepared path (correct, just untraceable
        and slower) if unavailable on this build.
        """
        key = (input_length, batch_size)
        if key in self._prep_cache:
            return self._prep_cache[key]
        kw = dict(
            input_memory_config=x.memory_config(),
            input_layout=x.layout,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,  # conv1d is conv2d at H=1
            input_width=input_length,
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
            w = ttnn.prepare_conv_weights(
                weight_tensor=self._weight_4d, weights_format="OIHW", has_bias=self.bias is not None, **kw
            )
            b = ttnn.prepare_conv_bias(bias_tensor=self.bias, **kw) if self.bias is not None else None
        except Exception as e:  # noqa: BLE001
            if not TtConv1d._warned:
                TtConv1d._warned = True
                logger.warning(f"prepare_conv_weights unavailable, convs stay untraceable: {str(e)[:200]}")
            w, b = self.weight, self.bias
        self._prep_cache[key] = (w, b)
        return w, b

    def _conv(self, x, weight, bias, input_length: int, batch_size: int, compute_config):
        return ttnn.conv1d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
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
            compute_config=compute_config,
            dtype=self.dtype,
            return_output_dim=True,
        )

    def _verify_and_resolve(self, x, weight, bias, input_length: int, batch_size: int, out, key):
        """First call for this geometry only: compare accurate_compute_config's
        result (`out`, already computed) against safe_compute_config's. Caches
        whichever config measured correct in `_verified_config[key]`, so every
        later call at this geometry goes straight to the right config in one
        conv -- this only runs once per geometry, not once per call.
        """
        ref, _ = self._conv(x, weight, bias, input_length, batch_size, self._safe_compute_config)
        a = float(ttnn.to_torch(out).float().abs().max())
        b = float(ttnn.to_torch(ref).float().abs().max())
        ok = a == a and abs(a - b) <= 0.02 * max(b, 1e-9)  # a != a catches NaN/inf
        if ok:
            ttnn.deallocate(ref)
            self._verified_config[key] = self.compute_config
            return out
        logger.warning(
            f"accurate_compute_config disagrees with safe_compute_config at Conv1d("
            f"{self.in_channels}->{self.out_channels}, k={self.kernel_size}, s={self.stride}) "
            f"length {input_length}: max|out| {a:.4g} vs {b:.4g}; using the safe config for this geometry"
        )
        self._verified_config[key] = self._safe_compute_config
        ttnn.deallocate(out)
        return ref

    def __call__(self, x, input_length: int, batch_size: int = 1):
        weight, bias = self._prepared(x, input_length, batch_size)
        key = (input_length, batch_size)
        if self.compute_config is None:
            out, out_length = self._conv(x, weight, bias, input_length, batch_size, None)
        elif key in self._verified_config:
            out, out_length = self._conv(x, weight, bias, input_length, batch_size, self._verified_config[key])
        else:
            out, out_length = self._conv(x, weight, bias, input_length, batch_size, self.compute_config)
            out = self._verify_and_resolve(x, weight, bias, input_length, batch_size, out, key)
        # ttnn.conv1d yields the flattened conv layout, not [N, L, C]. Restoring
        # the documented shape here (not at each call site) is what makes the
        # residual adds and permutes downstream legal.
        out = ttnn.reshape(out, (batch_size, out_length, self.out_channels))
        return out, out_length

    @staticmethod
    def out_length(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
        return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
