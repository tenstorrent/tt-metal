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

import os

import torch
from loguru import logger

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

    `COSYVOICE_FIDELITY` overrides it (`LoFi`/`HiFi2`/`HiFi3`/`HiFi4`) and
    `COSYVOICE_FP32_ACC=0` drops fp32 accumulation, so the accuracy/throughput trade can
    be measured rather than assumed. Fidelity is a *compute* lever: HiFi4 runs four
    passes where LoFi runs one, so it should matter on a compute-bound stage and not on
    a dispatch-bound one. Those two live in the same model here, which makes it a clean
    test -- see PERF.md for what it measured.
    """
    name = os.environ.get("COSYVOICE_FIDELITY", "HiFi4")
    fidelity = getattr(ttnn.MathFidelity, name, ttnn.MathFidelity.HiFi4)
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=os.environ.get("COSYVOICE_FP32_ACC", "1") != "0",
        packer_l1_acc=True,
    )


def prepare_weights_default(device) -> bool:
    """Whether to hoist conv weight preparation out of the op. Off on Wormhole.

    `ttnn.prepare_conv_weights` disagrees with the op's own preparation on Wormhole at
    some input lengths -- see `TtConv1d._prepared` for the measurements and
    `scripts/repro_conv1d_wormhole.py` for a standalone case. The disagreement reaches
    `1e37`, which is what breaks the streamed vocoder there.

    Defaulted by architecture rather than left to the caller because the failure is
    silent: the wrong answer is a number, not an exception, and it only appears at some
    lengths. `COSYVOICE_CONV_PREPARE` overrides in either direction so the A/B stays
    measurable -- and so the default can be dropped in one line once upstream is fixed.

    Applied to the **vocoder only**. The flow estimator uses the same `TtConv1d` and is
    captured in a trace, which unprepared weights make impossible; disabling it there took
    the flow stage from 0.683 to 1.723 s on n300.

    The vocoder is now traced too (`TtHiFTGenerator.decode`), which would have been a
    problem if this returned False outright -- but on Wormhole the generator arms
    `_verify_prepared` instead of dropping preparation, so the weights stay prepared and
    the geometry check is a one-off host read. Capture waits for a geometry's *second*
    sighting, by which time that read has already happened, so the two never overlap.
    """
    override = os.environ.get("COSYVOICE_CONV_PREPARE")
    if override is not None:
        return override != "0"
    try:
        return device.arch() != ttnn.Arch.WORMHOLE_B0
    except Exception:  # noqa: BLE001 -- a mesh or mock device without .arch()
        return True


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

    _warned = False  # one warning per process, not per convolution

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
        # OIHW with H=1, which is what prepare_conv_weights wants for a 1-D conv.
        self._weight_4d = ttnn.from_torch(weight.unsqueeze(2), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._prep_cache: dict = {}
        self.bias = None
        if bias is not None:
            # conv bias wants a 4-D [1, 1, 1, out_ch] row
            self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.conv_config = ttnn.Conv1dConfig(weights_dtype=weights_dtype, deallocate_activation=False)
        self.compute_config = accurate_compute_config(device) if high_fidelity else None
        self._verify = False  # set by the vocoder on Wormhole; see _verify_prepared
        self._verified: set = set()
        # On by default. The vocoder turns it off on Wormhole -- see
        # `prepare_weights_default` and `TtHiFTGenerator.__init__`. Deliberately *not*
        # defaulted by architecture here: this class is also the flow estimator's
        # convolution, and those run inside a captured trace, which is the one thing
        # unprepared weights make impossible. Setting the default here cost the flow
        # stage 0.683 -> 1.723 s on n300 before that was noticed.
        self._prepare = True

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

    def _prepared(self, x, input_length: int, batch_size: int):
        """Pre-tilized, device-resident weights, cached per input geometry.

        `ttnn.conv1d` will happily take a PyTorch-layout weight and sort it out
        internally -- but it does that **on every call**, and the preparation is a
        host-side layout transform, so it moves data across the command queue every
        time. Two consequences:

        * it is pure per-call overhead on a weight that never changes;
        * **it makes the op impossible to trace.** A trace records device commands
          and forbids host traffic in either direction, so capture fails with
          `Writes are not supported during trace capture` for a host weight, or
          `Reads are not supported` for a device one -- the op reads it back to
          prepare it. Measured both ways; device residency alone does not help.

        `prepare_conv_weights` hoists the transform out, and the op then has nothing
        to transfer. Output is bit-identical (`max|d| 0.000e+00`), and a bare conv1d
        captures cleanly once its weights are prepared this way.

        The prepared layout depends on the input geometry -- the sharding scheme
        follows `input_length` -- so the cache is keyed on it.

        **`bit-identical` is true on Blackhole and false on Wormhole.** At some input
        lengths the two paths disagree -- by a few percent at `input_length` 8321 and by
        `1e37` at 8193, for the vocoder's `Conv1d(128 -> 128, k=11, pad=5)`. The `1e37`
        is what breaks streaming there: `sin()` of it is `inf`, the vocoder's magnitude
        spectrum rails at its `1e2` clip, and the waveform saturates. Blackhole agrees
        exactly at every length tested.

        `COSYVOICE_CONV_PREPARE=0` takes the op's own preparation instead, which measured
        correct at every length on both parts, at ~1 ms per call. That is now a genuine
        trade rather than a free one: the vocoder is traced, so turning preparation off
        makes `TtHiFTGenerator.decode` uncapturable and costs the 3.2x that tracing is
        worth. Wormhole keeps preparation and verifies each geometry once instead --
        see `prepare_weights_default`. Revisit when the upstream defect is fixed;
        `scripts/probe_prepared_weights.py` is the check.
        """
        if not self._prepare:
            return self.weight, self.bias
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
            # Fall back to letting the op prepare them itself: correct, just slower
            # and not traceable. Logged rather than swallowed -- a silent fallback
            # here looks exactly like a working fast path from the outside, and the
            # only symptom is a trace capture that fails somewhere else entirely.
            if not TtConv1d._warned:
                TtConv1d._warned = True
                logger.warning(f"prepare_conv_weights unavailable, convs stay untraceable: {str(e)[:200]}")
            w, b = self.weight, self.bias
        self._prep_cache[key] = (w, b)
        return w, b

    def _verify_prepared(self, x, out, input_length: int, batch_size: int):
        """Check this geometry's prepared weight against the op's own, once.

        Disabling preparation everywhere on Wormhole is correct and costs the vocoder
        `0.084 -> 0.181 s` -- but only some geometries are affected, so most of that is
        paid for nothing. Running the convolution both ways the first time a geometry is
        seen, and keeping the prepared weight only where the two agree, is precise instead:
        one extra call per (length, batch) per conv, amortised to nothing over an
        utterance, against ~1 ms on every call thereafter.

        The comparison is `max|out|` rather than a full PCC because the failure is not
        subtle where it matters -- the observed disagreements run from 6x to 1e37 -- and a
        full comparison would mean a second device-to-host read of the whole activation.
        """
        key = (input_length, batch_size)
        self._verified.add(key)
        ref, _ = self._conv(x, self.weight, self.bias, input_length, batch_size)
        a = float(ttnn.to_torch(out).float().abs().max())
        b = float(ttnn.to_torch(ref).float().abs().max())
        ok = a == a and abs(a - b) <= 0.02 * max(b, 1e-9)  # a != a catches NaN/inf
        if ok:
            ttnn.deallocate(ref)
            return out
        self._prep_cache[key] = (self.weight, self.bias)
        logger.warning(
            f"prepare_conv_weights disagrees at Conv1d({self.in_channels}->{self.out_channels}, "
            f"k={self.kernel_size}) length {input_length}: max|out| {a:.4g} vs {b:.4g}; "
            "using the op's own preparation for this geometry"
        )
        ttnn.deallocate(out)
        return ref

    def _conv(self, x, weight, bias, input_length: int, batch_size: int):
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
            compute_config=self.compute_config,
            dtype=self.dtype,
            return_output_dim=True,
        )

    def __call__(self, x, input_length: int, batch_size: int = 1):
        weight, bias = self._prepared(x, input_length, batch_size)
        out, out_length = ttnn.conv1d(
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
            compute_config=self.compute_config,
            dtype=self.dtype,
            return_output_dim=True,
        )
        if self._verify and weight is not self.weight and (input_length, batch_size) not in self._verified:
            out = self._verify_prepared(x, out, input_length, batch_size)
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
