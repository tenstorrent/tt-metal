# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro ``AdaIN1d`` from ``reference/istftnet.py``.

``AdaIN1d`` applies ``InstanceNorm1d`` (with optional affine) along the time axis, then scales and
shifts per channel using ``(1 + gamma(s)) * y + beta(s)`` from a linear map of the style vector.

Activations use **NLC** layout ``[B, L, C]`` (length ``L``, channels ``C``). PyTorch appears only in
the ``preprocess_tt_*`` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

import ttnn

from .tt_matmul_memory import activation_interleaved_mc, maybe_reshard_to_caller, style_linear_plan

# ``ttnn.layer_norm`` on ``[B*C, L]`` matches ``InstanceNorm1d`` for most Kokoro shapes, but trained
# ``decoder.encode.norm2`` (C=1024, bf16, L≈96) still needs the decomposed path for PCC > 0.99.
_LEGACY_INSTANCE_NORM_CHANNELS = frozenset({1024})

# Fold InstanceNorm affine (w, b) into AdaIN coef/shift on generator BF16 paths. Coef/shift math
# runs in FP32 and is cast once to the activation dtype before ``addcmul``.
_ENABLE_AFFINE_FOLD_BF16 = True


def _cast_to_dtype(
    tensor: Optional[ttnn.Tensor],
    dtype,
    *,
    memory_config: ttnn.MemoryConfig,
    deallocate_source: bool = True,
) -> Optional[ttnn.Tensor]:
    if tensor is None or tensor.dtype == dtype:
        return tensor
    casted = ttnn.typecast(tensor, dtype, memory_config=memory_config)
    if deallocate_source:
        ttnn.deallocate(tensor)
    return casted


def _use_affine_fold(activation_dtype) -> bool:
    if activation_dtype == ttnn.float32:
        return True
    return _ENABLE_AFFINE_FOLD_BF16


def _fold_adain_coef_shift_fp32(
    *,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    inst_weight: Optional[ttnn.Tensor],
    inst_bias: Optional[ttnn.Tensor],
    memory_config: ttnn.MemoryConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Fold ``(1+gamma)*w`` and ``beta + (1+gamma)*b`` in FP32 for stable BF16 AdaIN."""
    gamma = _cast_to_dtype(gamma, ttnn.float32, memory_config=memory_config)
    beta = _cast_to_dtype(beta, ttnn.float32, memory_config=memory_config)
    in_w = _cast_to_dtype(inst_weight, ttnn.float32, memory_config=memory_config, deallocate_source=False)
    in_b = _cast_to_dtype(inst_bias, ttnn.float32, memory_config=memory_config, deallocate_source=False)

    coef = ttnn.add(gamma, 1.0, memory_config=memory_config)
    ttnn.deallocate(gamma)
    if in_b is not None:
        tmp_b = ttnn.multiply(coef, in_b, memory_config=memory_config)
        shift = ttnn.add(beta, tmp_b, memory_config=memory_config)
        ttnn.deallocate(tmp_b)
        ttnn.deallocate(beta)
    else:
        shift = beta
    if in_w is not None:
        scaled_coef = ttnn.multiply(coef, in_w, memory_config=memory_config)
        ttnn.deallocate(coef)
        coef = scaled_coef
    return coef, shift


@dataclass(frozen=True)
class TTInstanceNorm1dParams:
    """Affine parameters for instance norm over ``L`` in ``[B, L, C]`` (``None`` = identity scale/shift)."""

    weight: Optional[ttnn.Tensor]
    bias: Optional[ttnn.Tensor]
    eps: float


def _tt_instance_norm_1d_legacy_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: TTInstanceNorm1dParams,
    memory_config: ttnn.MemoryConfig,
    apply_affine: bool = True,
) -> ttnn.Tensor:
    mean = ttnn.mean(x_nlc, dim=1, keepdim=True, memory_config=memory_config)
    xc = ttnn.subtract(x_nlc, mean, memory_config=memory_config)
    var = ttnn.mean(ttnn.pow(xc, 2), dim=1, keepdim=True, memory_config=memory_config)
    inv_std = ttnn.rsqrt(ttnn.add(var, params.eps, memory_config=memory_config), memory_config=memory_config)
    y = ttnn.multiply(xc, inv_std, memory_config=memory_config)

    if apply_affine and params.weight is not None:
        y = ttnn.multiply(y, params.weight, memory_config=memory_config)
    if apply_affine and params.bias is not None:
        y = ttnn.add(y, params.bias, memory_config=memory_config)
    return y


def _tt_instance_norm_1d_fused_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: TTInstanceNorm1dParams,
    memory_config: ttnn.MemoryConfig,
    compute_kernel_config=None,
    apply_affine: bool = True,
) -> ttnn.Tensor:
    b = int(x_nlc.shape[0])
    l = int(x_nlc.shape[1])
    c = int(x_nlc.shape[2])

    x_bcl = ttnn.permute(x_nlc, (0, 2, 1), memory_config=memory_config)
    x_flat = ttnn.reshape(x_bcl, [b * c, l], memory_config=memory_config)
    ttnn.deallocate(x_bcl)

    ln_kwargs = {"epsilon": params.eps, "memory_config": memory_config}
    if compute_kernel_config is not None:
        ln_kwargs["compute_kernel_config"] = compute_kernel_config
    y_flat = ttnn.layer_norm(x_flat, **ln_kwargs)
    ttnn.deallocate(x_flat)

    y_bcl = ttnn.reshape(y_flat, [b, c, l], memory_config=memory_config)
    ttnn.deallocate(y_flat)
    y = ttnn.permute(y_bcl, (0, 2, 1), memory_config=memory_config)
    ttnn.deallocate(y_bcl)

    if apply_affine and params.weight is not None:
        y = ttnn.multiply(y, params.weight, memory_config=memory_config)
    if apply_affine and params.bias is not None:
        y = ttnn.add(y, params.bias, memory_config=memory_config)
    return y


def tt_instance_norm_1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: TTInstanceNorm1dParams,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    compute_kernel_config=None,
    apply_affine: bool = True,
) -> ttnn.Tensor:
    """
    Instance norm over the length dimension (dim ``1``) for each batch row and channel.

    Matches ``nn.InstanceNorm1d`` on inputs ``[B, C, L]`` after conversion to NLC.

    ``apply_affine=False`` returns the unscaled normalized activation and skips the per-channel
    ``weight``/``bias`` step; :class:`TTAdaIN1d` uses this to fold the affine into its own
    ``(1+gamma)``/``beta`` coefficients (avoiding two full-length elementwise ops).
    """
    c = int(x_nlc.shape[2])
    l = int(x_nlc.shape[1])
    use_fused = c not in _LEGACY_INSTANCE_NORM_CHANNELS and (l % 32 == 0)
    if not use_fused:
        return _tt_instance_norm_1d_legacy_nlc(
            x_nlc=x_nlc, params=params, memory_config=memory_config, apply_affine=apply_affine
        )
    return _tt_instance_norm_1d_fused_nlc(
        x_nlc=x_nlc,
        params=params,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
        apply_affine=apply_affine,
    )


@dataclass(frozen=True)
class TTAdaIN1dParams:
    """Device weights for :class:`TTAdaIN1d`."""

    fc_weight: ttnn.Tensor
    fc_bias: ttnn.Tensor
    instancenorm: TTInstanceNorm1dParams
    num_features: int
    # When True, ``fc_weight``/``fc_bias`` already encode ``coef``/``shift`` (the InstanceNorm affine
    # and the ``(1+gamma)`` factor pre-folded in at preprocess), and ``instancenorm`` carries no
    # affine — so :meth:`TTAdaIN1d.forward` is a single ``addcmul`` over the raw-normalized activation
    # (no per-forward per-channel fold ops). See :func:`preprocess_tt_adain_1d`.
    affine_folded: bool = False


def preprocess_tt_instance_norm_1d(
    inn: nn.InstanceNorm1d,
    device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTInstanceNorm1dParams:
    """Upload ``nn.InstanceNorm1d`` affine parameters for :func:`tt_instance_norm_1d_nlc`."""
    eps = float(inn.eps)
    if inn.affine and inn.weight is not None and inn.bias is not None:
        inf_w = ttnn.from_torch(
            inn.weight.detach().cpu().reshape(1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inf_b = ttnn.from_torch(
            inn.bias.detach().cpu().reshape(1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        inf_w = None
        inf_b = None
    return TTInstanceNorm1dParams(weight=inf_w, bias=inf_b, eps=eps)


def preprocess_tt_adain_1d(
    module: nn.Module,
    device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTAdaIN1dParams:
    """Upload PyTorch ``AdaIN1d`` (``.norm`` + ``.fc``) for :class:`TTAdaIN1d`.

    The InstanceNorm affine (``w``, ``b``) and the AdaIN ``(1 + gamma)`` factor are **pre-folded**
    into the style projection ``fc`` at upload time. AdaIN is

        out = (1 + gamma) * (w * norm(x) + b) + beta = coef * norm(x) + shift

    where ``coef`` and ``shift`` are affine in the style vector ``s`` (``gamma`` = ``fc``-rows[:C]·s,
    ``beta`` = ``fc``-rows[C:]·s; ``w``, ``b`` are static). Absorbing ``w``, ``b`` and the ``+1`` into
    the ``fc`` weight/bias (computed in float64 here) makes the per-forward AdaIN a single ``addcmul``
    over the raw-normalized activation, removing the four per-channel fp32 fold ops that
    :func:`_fold_adain_coef_shift_fp32` ran every call. Row layout is preserved (coef in rows [:C],
    shift in rows [C:]) so :meth:`TTAdaIN1d.forward` still chunks ``fc(s)`` into two halves. The fold
    is algebraically exact (fp64 preprocess), so PCC is unchanged.
    """
    fc = module.fc
    inn: nn.InstanceNorm1d = module.norm
    c = int(inn.num_features)

    fc_w = fc.weight.detach().cpu().double()  # [2C, style_dim]  (rows [:C]=gamma, [C:]=beta)
    fc_b = fc.bias.detach().cpu().double()  # [2C]
    w_gamma, w_beta = fc_w[:c], fc_w[c:]
    b_gamma, b_beta = fc_b[:c], fc_b[c:]
    if inn.affine and inn.weight is not None and inn.bias is not None:
        w_in = inn.weight.detach().cpu().double()  # [C]
        b_in = inn.bias.detach().cpu().double()  # [C]
    else:
        w_in = torch.ones(c, dtype=torch.float64)
        b_in = torch.zeros(c, dtype=torch.float64)

    # coef  = (1+gamma)*w = (w ⊙ w_gamma)·s + w*(1+b_gamma)
    w_coef = w_in[:, None] * w_gamma
    b_coef = w_in * (1.0 + b_gamma)
    # shift = beta + (1+gamma)*b = (w_beta + b ⊙ w_gamma)·s + (b_beta + b*(1+b_gamma))
    w_shift = w_beta + b_in[:, None] * w_gamma
    b_shift = b_beta + b_in * (1.0 + b_gamma)

    fc_w_folded = torch.cat([w_coef, w_shift], dim=0).float()  # [2C, style_dim]
    fc_b_folded = torch.cat([b_coef, b_shift], dim=0).float()  # [2C]

    fc_w = ttnn.from_torch(
        fc_w_folded,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    fc_b = ttnn.from_torch(
        fc_b_folded.reshape(1, 1, 1, -1),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Affine is folded into ``fc``; keep only ``eps`` so instance norm returns the raw normalized activation.
    inst = TTInstanceNorm1dParams(weight=None, bias=None, eps=float(inn.eps))

    return TTAdaIN1dParams(
        fc_weight=fc_w,
        fc_bias=fc_b,
        instancenorm=inst,
        num_features=c,
        affine_folded=True,
    )


class TTAdaIN1d:
    """Adaptive instance norm: ``(1 + gamma(s)) * InstanceNorm(x) + beta(s)`` on NLC tensors."""

    __slots__ = ("params",)

    def __init__(self, params: TTAdaIN1dParams) -> None:
        self.params = params

    def _project_style(
        self,
        style_bs: ttnn.Tensor,
        c: int,
        *,
        compute_kernel_config,
        memory_config: ttnn.MemoryConfig,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``fc(s)`` → two ``[b, 1, c]`` halves (rows [:c], rows [c:]) in TILE layout."""
        p = self.params
        b = int(style_bs.shape[0])
        style_out_mc, style_reshard = style_linear_plan(b, int(p.fc_weight.shape[-1]), 2 * c)
        linear_mc = style_out_mc if style_out_mc is not None else activation_interleaved_mc(memory_config)
        h = ttnn.linear(
            style_bs,
            p.fc_weight,
            bias=p.fc_bias,
            transpose_b=True,
            memory_config=linear_mc,
            compute_kernel_config=compute_kernel_config,
        )
        if style_reshard:
            h = maybe_reshard_to_caller(h, memory_config)
        while len(h.shape) > 2:
            h = ttnn.squeeze(h, 0)
        b = int(h.shape[0])
        h = ttnn.reshape(h, [b, 1, 2 * c], memory_config=memory_config)
        first, second = ttnn.chunk(h, 2, dim=-1)
        ttnn.deallocate(h)
        return first, second

    def forward(
        self,
        x_nlc: ttnn.Tensor,
        style_bs: ttnn.Tensor,
        *,
        compute_kernel_config,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        fuse_leaky_slope: Optional[float] = None,
    ) -> ttnn.Tensor:
        """``fuse_leaky_slope`` applies a ``LeakyReLU`` to the output. On the bf16 path it is fused
        into the final ``add`` (no separate Unary kernel); on the fp32 path it runs as one op."""
        p = self.params
        c = p.num_features

        if p.affine_folded:
            # ``fc`` already yields (coef, shift): AdaIN is one addcmul over the raw-normalized y.
            # No InstanceNorm affine and no per-channel fold ops (see :func:`preprocess_tt_adain_1d`).
            y = tt_instance_norm_1d_nlc(
                x_nlc=x_nlc,
                params=p.instancenorm,
                memory_config=memory_config,
                compute_kernel_config=compute_kernel_config,
                apply_affine=False,
            )
            coef, shift = self._project_style(
                style_bs, c, compute_kernel_config=compute_kernel_config, memory_config=memory_config
            )
            act_dtype = y.dtype
            if coef.dtype != act_dtype:
                coef = _cast_to_dtype(coef, act_dtype, memory_config=memory_config)
                shift = _cast_to_dtype(shift, act_dtype, memory_config=memory_config)
            out = ttnn.addcmul(shift, y, coef, memory_config=memory_config)
            ttnn.deallocate(y)
            ttnn.deallocate(coef)
            ttnn.deallocate(shift)
            if fuse_leaky_slope is not None:
                out = ttnn.leaky_relu(out, negative_slope=fuse_leaky_slope, memory_config=memory_config)
            return out

        # ---- legacy runtime-fold path (params constructed without pre-folding) ----
        fold_affine = _use_affine_fold(x_nlc.dtype)
        y = tt_instance_norm_1d_nlc(
            x_nlc=x_nlc,
            params=p.instancenorm,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            apply_affine=not fold_affine,
        )
        gamma, beta = self._project_style(
            style_bs, c, compute_kernel_config=compute_kernel_config, memory_config=memory_config
        )
        act_dtype = y.dtype

        if fold_affine:
            coef, shift = _fold_adain_coef_shift_fp32(
                gamma=gamma,
                beta=beta,
                inst_weight=p.instancenorm.weight,
                inst_bias=p.instancenorm.bias,
                memory_config=memory_config,
            )
            if act_dtype != ttnn.float32:
                coef = _cast_to_dtype(coef, act_dtype, memory_config=memory_config)
                shift = _cast_to_dtype(shift, act_dtype, memory_config=memory_config)

            out = ttnn.addcmul(shift, y, coef, memory_config=memory_config)
            ttnn.deallocate(y)
            ttnn.deallocate(coef)
            ttnn.deallocate(shift)
            if fuse_leaky_slope is not None:
                out = ttnn.leaky_relu(out, negative_slope=fuse_leaky_slope, memory_config=memory_config)
            return out

        scale = ttnn.add(gamma, 1.0, memory_config=memory_config)
        ttnn.deallocate(gamma)
        y = ttnn.multiply(y, scale, memory_config=memory_config)
        ttnn.deallocate(scale)
        out = ttnn.add(y, beta, memory_config=memory_config)
        ttnn.deallocate(y)
        ttnn.deallocate(beta)
        if fuse_leaky_slope is not None:
            out = ttnn.leaky_relu(out, negative_slope=fuse_leaky_slope, memory_config=memory_config)
        return out

    __call__ = forward
