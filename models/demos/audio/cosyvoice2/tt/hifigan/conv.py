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

import os

import torch
from loguru import logger

import ttnn

from ..geometry_cache import GeometryWeightCache


def config_tensors_in_dram() -> bool:
    """Whether conv/halo config tensors go to DRAM instead of the L1_SMALL bank.

    Every ttnn conv-family op (conv, conv_transpose, pool, upsample) keeps small config tensors
    (sliding-window / halo indices) in the L1_SMALL bank and NEVER frees them -- upstream calls this
    by design (tenstorrent/tt-metal#33316: "allocate enough of it for the whole model in advance").
    They are per input geometry, so every new utterance length adds another set and a bank sized for
    one utterance (64 KB here) is exhausted after two or three different lengths. Placing them in
    DRAM (`Conv*Config.config_tensors_in_dram`, PR #27753 / #33328) removes the reservation.

    On by default for every conv in the port (HiFT convs, transposed convs, STFT/iSTFT, the flow encoder
    and estimator); `COSYVOICE2_CONV_CONFIG_IN_DRAM=0` turns it off. Read at construction time, so set it
    before the models are built. Measured: L1_SMALL stays at 0.00 KB across seven different utterance lengths
    with no cache clearing, and outputs are bit-identical with it on or off (max |diff| 0.0 on all six conv
    types).
    """
    return os.environ.get("COSYVOICE2_CONV_CONFIG_IN_DRAM", "1") == "1"


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
    combination, not HiFi4 itself. This is the same *class* of issue as
    tenstorrent/tt-metal#55545, which documents `ttnn.prepare_conv_weights`
    disagreeing with the op's own internal preparation on Wormhole (up to
    1e37 at some input lengths, confirmed to still reproduce on this build
    at this same Conv1d(128->128, k=11) architecture) -- a different
    symptom, same lesson: neither an "accurate" compute config nor a
    "prepared" weight is safe to trust unverified at an unfamiliar shape on
    this hardware. `TtConv1d._verify_and_resolve` checks both together
    (fast path vs. a raw-weight + safe-config reference) once per geometry
    and falls back to whichever measured correct. `TtConvTranspose1d` in
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


# The fast path (prepared weight + accurate_compute_config) and the raw-weight +
# safe_compute_config reference are both *correct* on a healthy geometry, but not
# identical: measured at Conv1d(128->128, k=11) L=18560 against a float64 conv, the
# accurate config is ~0.4% off and the "safe" one ~2.7% off, so the two differ by a
# few percent with nothing wrong. The silent defects this guards against are gross
# (PCC 0.0011, values ~1e37, a ~30x scale error) -- so the two only count as
# disagreeing past this relative L2 distance, and only then is a float64 host
# reference computed to decide who is right (see `pick_most_accurate`).
AGREEMENT_TOLERANCE = 0.05


def relative_error(got, want) -> float:
    """||got - want|| / ||want|| in float64; NaN/inf (or an all-zero `want`) count as +inf."""
    got, want = got.double(), want.double()
    denom = float(want.norm())
    err = float((got - want).norm())
    if not (err == err) or denom == 0.0 or denom != denom:
        return float("inf")
    return err / denom


def pick_most_accurate(host_candidates: dict, truth) -> tuple:
    """Name of the candidate closest to `truth`, plus every candidate's relative error.

    The resolver used to treat the raw-weight + safe-config output as ground truth
    and switch to it on any disagreement -- which chose the *less* accurate result
    whenever the fast path was the right one (see `AGREEMENT_TOLERANCE`).
    """
    errors = {name: relative_error(t, truth) for name, t in host_candidates.items()}
    return min(errors, key=errors.get), errors


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
        padding: int | tuple[int, int] = 0,
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
        # Per-geometry prepared-weight cache, threshold-evicted on real DRAM pressure --
        # see geometry_cache.py's module docstring for why (this is what was silently
        # unbounded before, and streaming will need the identical mechanism for its own
        # chunk-shape churn).
        self._prep_cache = GeometryWeightCache(device)
        # Host copies, used only as the tie-break reference in `_verify_and_resolve`.
        self._host_weight = weight.detach().float().clone()
        self._host_bias = bias.detach().float().clone() if bias is not None else None
        self.bias = None
        if bias is not None:
            self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype,
            deallocate_activation=False,
            config_tensors_in_dram=config_tensors_in_dram(),
        )
        self.compute_config = accurate_compute_config(device) if high_fidelity else None
        self._safe_compute_config = safe_compute_config(device) if high_fidelity else None
        # Verify the fast path (prepared weight + accurate_compute_config)
        # against a maximally-conservative reference (raw weight +
        # safe_compute_config) once per geometry rather than trust it -- see
        # `_verify_and_resolve`'s docstring for the two independent, silent
        # defects this catches. Maps (input_length, batch_size) -> the
        # (weight, bias, compute_config) triple that measured correct for
        # that geometry, so a disagreement is resolved once, not re-checked
        # (and not silently re-broken) on every subsequent call. Same
        # threshold-evicted cache class as `_prep_cache`, but a SEPARATE
        # instance: the two can have different lifetimes for the same key
        # (see `_verify_and_resolve`'s ownership-transfer note).
        self._verified_config = GeometryWeightCache(device)

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
        cached = self._prep_cache.get(key)
        if cached is not None:
            return cached
        # `prepare_conv_weights`/`prepare_conv_bias` are conv2d-level ops, so unlike `ttnn.conv1d` itself
        # (which accepts a conv1d-shaped `int` or `(pad_left, pad_right)` and does this translation
        # internally -- see `conv1d.cpp`) they need the conv2d padding spelled out: `(pad_height, pad_width)`
        # for symmetric, `(pad_top, pad_bottom, pad_left, pad_right)` for asymmetric. `pad_height` is always
        # 0 (conv1d is conv2d at H=1).
        pad2d = (0, self.padding) if isinstance(self.padding, int) else (0, 0, *self.padding)
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
            padding=pad2d,
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
        # Only tensors genuinely prepared FOR THIS GEOMETRY are this cache slot's to free --
        # the fallback (w, b) = (self.weight, self.bias) are the conv's own permanent raw
        # weights, shared across every geometry, and must never be deallocated by a
        # per-geometry eviction.
        owned = [t for t in (w, b) if t is not None and t is not self.weight and t is not self.bias]
        self._prep_cache.put(key, owned, (w, b))
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

    def _pad_pair(self) -> tuple[int, int]:
        return (self.padding, self.padding) if isinstance(self.padding, int) else tuple(self.padding)

    def _out_length(self, input_length: int) -> int:
        pad_left, pad_right = self._pad_pair()
        return (input_length + pad_left + pad_right - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

    def _to_host(self, t, batch_size: int, out_length: int):
        return ttnn.to_torch(t).float().reshape(batch_size, out_length, self.out_channels).double()

    def _host_reference(self, x, input_length: int, batch_size: int):
        """float64 torch conv of the same input with the same (un-quantised) weights,
        `[B, L_out, C_out]`. Only ever run to arbitrate a disagreement."""
        x_cf = ttnn.to_torch(x).float().reshape(batch_size, input_length, self.in_channels).transpose(1, 2).double()
        bias = self._host_bias.double() if self._host_bias is not None else None
        pad_left, pad_right = self._pad_pair()
        if pad_left == pad_right:
            x_pad, conv_padding = x_cf, pad_left  # torch.nn.functional.conv1d's own padding covers this case
        else:
            x_pad, conv_padding = torch.nn.functional.pad(x_cf, (pad_left, pad_right)), 0
        ref = torch.nn.functional.conv1d(
            x_pad,
            self._host_weight.double(),
            bias,
            stride=self.stride,
            padding=conv_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return ref.transpose(1, 2)

    def _verify_and_resolve(self, x, weight, bias, input_length: int, batch_size: int, out, key):
        """First call for this geometry only: compare the fast path's
        already-computed result (`out` -- prepared weight + accurate
        compute config) against ONE maximally-conservative reference
        computed together: raw, unprepared weight + safe_compute_config.

        A single combined reference rather than two separate per-axis
        verifications, because this catches either of two independent,
        silent defects at once, and correctness only needs to know "is the
        fast path right for this geometry," not which knob would have been
        to blame:

        * `accurate_compute_config`'s `fp32_dest_acc_en` + `packer_l1_acc`
          combination -- see `accurate_compute_config`'s docstring (found at
          CosyVoice2's `source_downs` shape, PCC 0.0011 vs. expected ~0.9999).
        * `ttnn.prepare_conv_weights` silently disagreeing with the op's own
          internal weight preparation on Wormhole -- up to `1e37` at some
          input lengths (tenstorrent/tt-metal#55545). Confirmed to still
          reproduce on this build at this exact `Conv1d(128->128, k=11)`
          architecture (CosyVoice2's HiFT source resblocks): disagreement at
          `L=9217`, using the reference report's own repro script unmodified
          -- NOT the same lengths that report's build found bad (8193/8321/
          8577/8705, all fine here). The affected lengths are build/ttnn-
          version-specific, which is exactly why this is a per-geometry
          runtime check rather than a fixed exclusion list: no one-time test
          of "our real utterance lengths are fine today" is a durable
          guarantee against a defect this sparse and silent.

        **Agreement** means a relative L2 distance within `AGREEMENT_TOLERANCE`
        (not `max|out|`, which a corruption can preserve): the fast path is kept,
        at no extra cost beyond the one reference conv. **Disagreement** does NOT
        mean the reference is right -- the raw-weight + safe-config reference is
        itself the *less* accurate of the two on a healthy geometry (~2.7% vs
        ~0.4% error at `Conv1d(128->128, k=11)` L=18560, where prepared and raw
        weights were bit-identical), and switching to it on the old `max|out|`
        within-2% check degraded accuracy on three resblock convs of the real
        464-frame utterance. So on a disagreement a float64 host conv arbitrates
        between three candidates -- prepared+accurate, raw+safe, raw+accurate (the
        right answer when only the prepared weight is at fault) -- and the closest
        to it wins.

        Caches whichever `(weight, bias, compute_config)` triple won in
        `_verified_config[key]`, so every later call at this geometry goes
        straight to it in one conv -- this only runs once per geometry, not once
        per call. (`self.compute_config is None`, i.e. `high_fidelity=False`,
        skips this entirely and is not covered by this check -- not exercised by
        the real vocoder, which always builds `high_fidelity=True`.)
        """
        ref, out_length = self._conv(x, self.weight, self.bias, input_length, batch_size, self._safe_compute_config)
        fast_host = self._to_host(out, batch_size, out_length)
        ref_host = self._to_host(ref, batch_size, out_length)
        if relative_error(fast_host, ref_host) <= AGREEMENT_TOLERANCE:
            ttnn.deallocate(ref)
            self._resolve(key, (weight, bias, self.compute_config))
            return out

        raw_accurate, _ = self._conv(x, self.weight, self.bias, input_length, batch_size, self.compute_config)
        candidates = {
            "prepared weight + accurate config": (out, (weight, bias, self.compute_config)),
            "raw weight + safe config": (ref, (self.weight, self.bias, self._safe_compute_config)),
            "raw weight + accurate config": (raw_accurate, (self.weight, self.bias, self.compute_config)),
        }
        host = {
            "prepared weight + accurate config": fast_host,
            "raw weight + safe config": ref_host,
            "raw weight + accurate config": self._to_host(raw_accurate, batch_size, out_length),
        }
        best, errors = pick_most_accurate(host, self._host_reference(x, input_length, batch_size))
        logger.warning(
            f"fast conv path disagrees with the raw-weight/safe-config reference at "
            f"Conv1d({self.in_channels}->{self.out_channels}, k={self.kernel_size}, s={self.stride}) "
            f"length {input_length}; relative error vs float64 host conv: "
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
        """Records the winning (weight, bias, compute_config) for `key` in
        `_verified_config`, and settles `_prep_cache`'s bookkeeping for the same key:
        if the winning weight/bias ARE `_prep_cache[key]`'s prepared tensors, ownership
        transfers to `_verified_config` (`discard`, not `pop` -- a `pop` here would
        deallocate tensors `_verified_config` is about to reference, a double-free the
        moment ITS entry is later evicted). Otherwise (raw weight won, or nothing was
        ever prepared) any prepared tensors for this key are now unused -- `pop` frees
        them for real."""
        weight, bias, _ = resolved
        prepared = self._prep_cache.get(key)
        # Identity, not equality: ttnn.Tensor's `==` is elementwise (matching torch's own
        # tensor semantics), not a single bool -- `is` is the only correct way to ask "is
        # this the SAME tensor _prep_cache already owns," which is all that matters here.
        if prepared is not None and prepared[0] is weight and prepared[1] is bias:
            self._prep_cache.discard(key)
        else:
            self._prep_cache.pop(key)
        owned = [t for t in (weight, bias) if t is not None and t is not self.weight and t is not self.bias]
        self._verified_config.put(key, owned, resolved)

    def __call__(self, x, input_length: int, batch_size: int = 1):
        key = (input_length, batch_size)
        verified = None if self.compute_config is None else self._verified_config.get(key)
        if self.compute_config is None:
            weight, bias = self._prepared(x, input_length, batch_size)
            out, out_length = self._conv(x, weight, bias, input_length, batch_size, None)
        elif verified is not None:
            weight, bias, compute_config = verified
            out, out_length = self._conv(x, weight, bias, input_length, batch_size, compute_config)
        else:
            weight, bias = self._prepared(x, input_length, batch_size)
            out, out_length = self._conv(x, weight, bias, input_length, batch_size, self.compute_config)
            out = self._verify_and_resolve(x, weight, bias, input_length, batch_size, out, key)
        # ttnn.conv1d yields the flattened conv layout, not [N, L, C]. Restoring
        # the documented shape here (not at each call site) is what makes the
        # residual adds and permutes downstream legal.
        out = ttnn.reshape(out, (batch_size, out_length, self.out_channels))
        return out, out_length

    def release_caches(self) -> None:
        """Explicitly frees every geometry this instance has prepared/verified, all at
        once -- the threshold-based eviction in `GeometryWeightCache` already keeps DRAM
        bounded automatically, so this is for a natural session boundary (e.g. between
        sessions, not between utterances), not something normal per-utterance operation
        needs to call."""
        self._prep_cache.clear()
        self._verified_config.clear()

    @staticmethod
    def out_length(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
        return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
