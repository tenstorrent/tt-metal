# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host-side Chronos-2 state_dict -> TtChronosWeights conversion + InstanceNorm/Patch input prep.

reference : reference/chronos2/model.py (_prepare_patched_context / _prepare_patched_future)
"""

from __future__ import annotations

from typing import cast

import torch
from einops import rearrange, repeat


def preprocess_model_parameters(
    state_dict, device=None, *, eps: float = 1e-6, rope_theta: float = 10000.0, head_dim: int | None = None
):
    """Map a Chronos-2 state_dict onto host-side ``TtChronosWeights``.

    Expected keys (HF ``Chronos2Model`` format)::

        shared.weight
        input_patch_embedding.{hidden_layer,output_layer,residual_layer}.{weight,bias}
        encoder.block.{i}.layer.0.{layer_norm.weight, self_attention.{q,k,v,o}.weight}
        encoder.block.{i}.layer.1.{layer_norm.weight, self_attention.{q,k,v,o}.weight}
        encoder.block.{i}.layer.2.{mlp.wi.weight, mlp.wo.weight, layer_norm.weight}
        encoder.final_layer_norm.weight
        output_patch_embedding.{hidden_layer,output_layer,residual_layer}.{weight,bias}

    ``device`` is accepted for signature compatibility and ignored: device
    tensors are created once inside each ``Tt*`` module ``__init__``. ``eps``
    is the RMSNorm epsilon (not stored in the checkpoint; matches the
    reference default ``layer_norm_epsilon=1e-6``). ``rope_theta`` rebuilds the
    RoPE ``inv_freq`` buffer, which the reference registers with
    ``persistent=False`` and is therefore absent from the state_dict
    (formula mirrors ``Chronos2RotaryEmbedding.compute_default_rope_parameters``).
    ``head_dim`` (d_kv) cannot be inferred from shapes alone, so the caller
    must pass it (e.g. ``model.config.d_kv``); 64 for the released checkpoint.
    """
    # Lazy to avoid a module cycle (tt/model.py imports helpers from this file).
    from models.experimental.chronos_forecast.tt.encoder import TtEncoderWeights
    from models.experimental.chronos_forecast.tt.encoder_block import TtEncoderBlockWeights
    from models.experimental.chronos_forecast.tt.group_attention import TtGroupAttentionWeights
    from models.experimental.chronos_forecast.tt.model import TtChronosWeights
    from models.experimental.chronos_forecast.tt.residual_block import TtResidualBlockWeights
    from models.experimental.chronos_forecast.tt.time_attention import TtTimeAttentionWeights

    def _t(key: str) -> torch.Tensor:
        try:
            return state_dict[key].detach().clone()
        except KeyError:
            raise KeyError(f"preprocess_model_parameters: missing key {key!r}") from None

    def _residual(prefix: str) -> TtResidualBlockWeights:
        return TtResidualBlockWeights(
            hidden_weight=_t(f"{prefix}.hidden_layer.weight"),
            hidden_bias=_t(f"{prefix}.hidden_layer.bias"),
            output_weight=_t(f"{prefix}.output_layer.weight"),
            output_bias=_t(f"{prefix}.output_layer.bias"),
            residual_weight=_t(f"{prefix}.residual_layer.weight"),
            residual_bias=_t(f"{prefix}.residual_layer.bias"),
        )

    block_ids = sorted({int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.block.")})
    if not block_ids:
        raise KeyError("preprocess_model_parameters: no encoder.block.{i} keys found")
    if block_ids != list(range(len(block_ids))):
        raise KeyError(f"preprocess_model_parameters: non-contiguous blocks {block_ids}")

    blocks = []
    if head_dim is None:
        raise ValueError("preprocess_model_parameters: pass head_dim=model.config.d_kv (64 for amazon/chronos-2)")
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    for i in block_ids:
        t = f"encoder.block.{i}.layer.0"
        q = _t(f"{t}.self_attention.q.weight")
        if q.shape[0] % head_dim:
            raise ValueError(f"preprocess_model_parameters: inner {q.shape[0]} not divisible by head_dim {head_dim}")
        num_heads = q.shape[0] // head_dim
        time = TtTimeAttentionWeights(
            wqkv=torch.cat([q, _t(f"{t}.self_attention.k.weight"), _t(f"{t}.self_attention.v.weight")], dim=0),
            wo=_t(f"{t}.self_attention.o.weight"),
            rms_weight=_t(f"{t}.layer_norm.weight"),
            inv_freq=inv_freq.clone(),
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
        )
        g = f"encoder.block.{i}.layer.1"
        gq = _t(f"{g}.self_attention.q.weight")
        group = TtGroupAttentionWeights(
            wqkv=torch.cat([gq, _t(f"{g}.self_attention.k.weight"), _t(f"{g}.self_attention.v.weight")], dim=0),
            wo=_t(f"{g}.self_attention.o.weight"),
            rms_weight=_t(f"{g}.layer_norm.weight"),
            num_heads=gq.shape[0] // head_dim,
            head_dim=head_dim,
            eps=eps,
        )
        f = f"encoder.block.{i}.layer.2"
        blocks.append(
            TtEncoderBlockWeights(
                time=time,
                group=group,
                ff_wi=_t(f"{f}.mlp.wi.weight"),
                ff_wo=_t(f"{f}.mlp.wo.weight"),
                ff_rms_weight=_t(f"{f}.layer_norm.weight"),
                ff_eps=eps,
            )
        )

    # reg_token_id lives in the HF config, not the state_dict; the reference
    # sets it to 1 whenever use_reg_token is on (dummy + released checkpoints).
    return TtChronosWeights(
        input_embed=_residual("input_patch_embedding"),
        encoder=TtEncoderWeights(
            blocks=tuple(blocks),
            final_rms_weight=_t("encoder.final_layer_norm.weight"),
            final_eps=eps,
        ),
        output_embed=_residual("output_patch_embedding"),
        shared_weight=_t("shared.weight"),
        reg_token_id=1,
    )


def instance_norm(
    x: torch.Tensor,
    loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    eps: float = 1e-5,
    use_arcsinh: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Standardize along the last dim (Amazon Chronos InstanceNorm math)."""
    orig_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    if loc_scale is None:
        loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
        scale = torch.where(scale == 0, torch.as_tensor(eps, dtype=scale.dtype, device=scale.device), scale)
    else:
        loc, scale = loc_scale
        loc = loc.to(dtype=torch.float32)
        scale = scale.to(dtype=torch.float32)

    scaled_x = (x - loc) / scale
    if use_arcsinh:
        scaled_x = torch.arcsinh(scaled_x)
    return scaled_x.to(orig_dtype), (loc, scale)


def instance_norm_inverse(
    x: torch.Tensor,
    loc_scale: tuple[torch.Tensor, torch.Tensor],
    *,
    use_arcsinh: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Undo ``instance_norm`` with the stored loc/scale."""
    x = x.to(dtype=torch.float32)
    loc, scale = loc_scale
    loc = loc.to(dtype=torch.float32)
    scale = scale.to(dtype=torch.float32)
    if use_arcsinh:
        x = torch.sinh(x)
    x = x * scale + loc
    return x if output_dtype is None else x.to(output_dtype)


def patch(
    x: torch.Tensor,
    patch_size: int,
    patch_stride: int | None = None,
) -> torch.Tensor:
    """Window the last dim (Amazon Chronos ``Patch.forward``)."""
    if patch_stride is None:
        patch_stride = patch_size
    length = x.shape[-1]

    if length % patch_size != 0:
        padding_size = (
            *x.shape[:-1],
            patch_size - (length % patch_size),
        )
        padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device)
        x = torch.concat((padding, x), dim=-1)

    x = x.unfold(dimension=-1, size=patch_size, step=patch_stride)
    return x


def prepare_patched_context(
    context: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    *,
    patch_size: int,
    patch_stride: int | None = None,
    context_length: int | None = None,
    time_encoding_scale: int | None = None,
    apply_instance_norm: bool = True,
    use_arcsinh: bool = False,
    instance_norm_eps: float = 1e-5,
    loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """InstanceNorm (optional) + Patch + time encoding. Amazon ``_prepare_patched_context``."""
    if patch_stride is None:
        patch_stride = patch_size
    context_mask = (
        context_mask.to(context.dtype)
        if context_mask is not None
        else torch.isnan(context).logical_not().to(context.dtype)
    )

    batch_size, seq_len = context.shape
    if context_length is not None and seq_len > context_length:
        context = context[..., -context_length:]
        context_mask = context_mask[..., -context_length:]

    if apply_instance_norm:
        context, loc_scale = instance_norm(context, loc_scale, eps=instance_norm_eps, use_arcsinh=use_arcsinh)
    elif loc_scale is None:
        raise ValueError("loc_scale is required when apply_instance_norm is False")

    context = context.to(dtype=torch.float32)
    context_mask = context_mask.to(dtype=torch.float32)

    patched_context = patch(context, patch_size=patch_size, patch_stride=patch_stride)
    patched_mask = torch.nan_to_num(patch(context_mask, patch_size=patch_size, patch_stride=patch_stride), nan=0.0)
    patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

    attention_mask = patched_mask.sum(dim=-1) > 0
    num_context_patches = attention_mask.shape[-1]

    final_context_length = num_context_patches * patch_size
    scale = time_encoding_scale if time_encoding_scale is not None else context_length
    if scale is None:
        scale = final_context_length
    context_time_enc = torch.arange(start=-final_context_length, end=0, device=context.device, dtype=torch.float32)
    context_time_enc = (
        repeat(
            context_time_enc,
            "(n p) -> b n p",
            b=batch_size,
            n=num_context_patches,
            p=patch_size,
        )
        .div(cast(int, scale))
        .to(dtype=torch.float32)
    )

    patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)
    return patched_context, attention_mask, loc_scale


def prepare_patched_future(
    future_covariates: torch.Tensor | None,
    loc_scale: tuple[torch.Tensor, torch.Tensor],
    *,
    num_output_patches: int,
    output_patch_size: int,
    batch_size: int,
    time_encoding_scale: int,
    future_covariates_mask: torch.Tensor | None = None,
    use_arcsinh: bool = False,
    instance_norm_eps: float = 1e-5,
    apply_instance_norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Amazon ``_prepare_patched_future`` (zero-pad, rearrange, time encoding)."""
    if future_covariates is not None:
        if apply_instance_norm:
            future_covariates, _ = instance_norm(
                future_covariates, loc_scale, eps=instance_norm_eps, use_arcsinh=use_arcsinh
            )
        future_covariates = future_covariates.to(dtype=torch.float32)

        if future_covariates_mask is None:
            future_covariates_mask = torch.isnan(future_covariates).logical_not().to(future_covariates.dtype)

        future_covariates = torch.where(future_covariates_mask > 0.0, future_covariates, 0.0)

        if torch.isnan(future_covariates).any():
            raise ValueError(
                "future_covariates contains NaN values at indices not masked by future_covariates_mask. "
                "Input the correct future_covariates_mask or omit it to automatically infer the mask based on NaN values."
            )

        if num_output_patches * output_patch_size > future_covariates.shape[-1]:
            padding_shape = (
                *future_covariates.shape[:-1],
                num_output_patches * output_patch_size - future_covariates.shape[-1],
            )
            future_covariates = torch.cat([future_covariates, torch.zeros(padding_shape).to(future_covariates)], dim=-1)
            future_covariates_mask = torch.cat(
                [future_covariates_mask, torch.zeros(padding_shape).to(future_covariates_mask)], dim=-1
            )

        patched_future_covariates = rearrange(
            future_covariates, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size
        )
        patched_future_covariates_mask = rearrange(
            future_covariates_mask, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size
        )
    else:
        patched_future_covariates = torch.zeros(batch_size, num_output_patches, output_patch_size, dtype=torch.float32)
        patched_future_covariates_mask = torch.zeros(
            batch_size, num_output_patches, output_patch_size, dtype=torch.float32
        )

    final_future_length = num_output_patches * output_patch_size
    future_time_enc = torch.arange(start=0, end=final_future_length, dtype=torch.float32)
    future_time_enc = (
        repeat(
            future_time_enc,
            "(n p) -> b n p",
            b=batch_size,
            n=num_output_patches,
            p=output_patch_size,
        )
        .div(cast(int, time_encoding_scale))
        .to(dtype=torch.float32)
    )

    patched_future = torch.cat([future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1)
    return patched_future, patched_future_covariates_mask
