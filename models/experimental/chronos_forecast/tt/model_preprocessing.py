# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host-side Chronos-2 input packing + state_dict -> TtChronosWeights conversion.

reference : third_party/chronos-forecasting (Amazon preprocess) + reference/chronos2/model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat

ArrayLike = torch.Tensor | np.ndarray | Sequence


@dataclass(frozen=True)
class Chronos2PackedInputs:
    """Tensors ready for Chronos2Model.forward (host, float32).

    Shapes:
        context: (B, T)
        future_covariates: (B, H)
        group_ids: (B,)
    B is the sum of (n_targets + n_covariates) over series in the batch.
    """

    context: torch.Tensor
    future_covariates: torch.Tensor
    group_ids: torch.Tensor
    target_idx_ranges: list[tuple[int, int]]
    loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass(frozen=True)
class Chronos2PatchedInputs:
    """Patched tensors after InstanceNorm + Patch + time encoding.

    Shapes:
        patched_context: (B, num_context_patches, 3 * patch_size)
        attention_mask: (B, num_context_patches)
        patched_future: (B, num_output_patches, 3 * output_patch_size)
    """

    patched_context: torch.Tensor
    attention_mask: torch.Tensor
    patched_future: torch.Tensor
    loc_scale: tuple[torch.Tensor, torch.Tensor]
    group_ids: torch.Tensor
    target_idx_ranges: list[tuple[int, int]]


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


def prepare_chronos2_inputs(
    target: ArrayLike | Sequence[ArrayLike],
    *,
    prediction_length: int | None = None,
    past_covariates: ArrayLike | Sequence[ArrayLike] | None = None,
    future_covariates: ArrayLike | Sequence[ArrayLike] | None = None,
    use_target_encoding: bool = True,
    apply_instance_norm: bool = False,
    use_arcsinh: bool = False,
    instance_norm_eps: float = 1e-5,
) -> Chronos2PackedInputs:
    """Pack targets/covariates. target (T,)/(D,T) or per-series list; past (M,T);
    future (M_f,H) suffix of past; H required if future omitted. Categorical:
    target-encoded (1 target) else ordinal. Norm off by default (Amazon parity)."""
    target_is_batch = _is_nested_series_list(target)
    targets = _as_series_list(target, name="target")
    n_series = len(targets)
    past_per_series = _parse_covariates(past_covariates, n_series, target_is_batch, name="past_covariates")
    future_per_series = _parse_covariates(future_covariates, n_series, target_is_batch, name="future_covariates")

    target_2ds = [_as_variate_time(t, name="target") for t in targets]
    n_targets = target_2ds[0].shape[0]
    if any(t.shape[0] != n_targets for t in target_2ds):
        raise ValueError("all series must have the same number of target rows")

    series_lengths = [int(t.shape[-1]) for t in target_2ds]
    n_cov, n_future_cov, horizon = _covariate_layout(
        past_per_series, future_per_series, prediction_length, series_lengths
    )
    n_past_only = n_cov - n_future_cov

    encoded_past_cols, encoded_future_cols = _encode_covariate_columns(
        target_2ds=target_2ds,
        past_per_series=past_per_series,
        future_per_series=future_per_series,
        n_cov=n_cov,
        n_future_cov=n_future_cov,
        horizon=horizon,
        use_target_encoding=use_target_encoding,
    )

    context_parts: list[torch.Tensor] = []
    future_parts: list[torch.Tensor] = []
    group_ids_parts: list[torch.Tensor] = []
    target_idx_ranges: list[tuple[int, int]] = []
    row_cursor = 0

    for group_id, target_2d in enumerate(target_2ds):
        past_rows = [torch.from_numpy(encoded_past_cols[j][group_id].copy()) for j in range(n_cov)]
        future_rows = [torch.from_numpy(encoded_future_cols[j][group_id].copy()) for j in range(n_cov)]
        context = target_2d.to(dtype=torch.float32)
        if past_rows:
            context = torch.cat([context] + [row.unsqueeze(0) for row in past_rows], dim=0)
        nan_head = torch.full(
            (n_targets + n_past_only, horizon),
            float("nan"),
            dtype=torch.float32,
        )
        known_future_rows = future_rows[n_past_only:]
        if not known_future_rows:
            packed_future = nan_head
        else:
            packed_future = torch.cat(
                [nan_head] + [row.unsqueeze(0) for row in known_future_rows],
                dim=0,
            )

        n_rows = context.shape[0]
        context_parts.append(context)
        future_parts.append(packed_future)
        group_ids_parts.append(torch.full((n_rows,), group_id, dtype=torch.long))
        target_idx_ranges.append((row_cursor, row_cursor + n_targets))
        row_cursor += n_rows

    packed = Chronos2PackedInputs(
        context=_left_pad_and_cat_2d(context_parts),
        future_covariates=torch.cat(future_parts, dim=0),
        group_ids=torch.cat(group_ids_parts, dim=0),
        target_idx_ranges=target_idx_ranges,
    )
    if apply_instance_norm:
        packed = normalize_chronos2_inputs(packed, eps=instance_norm_eps, use_arcsinh=use_arcsinh)
    return packed


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


def normalize_chronos2_inputs(
    packed: Chronos2PackedInputs,
    *,
    eps: float = 1e-5,
    use_arcsinh: bool = False,
) -> Chronos2PackedInputs:
    """Apply InstanceNorm to packed V, then the same loc/scale to W."""
    context, loc_scale = instance_norm(packed.context, eps=eps, use_arcsinh=use_arcsinh)
    future, _ = instance_norm(packed.future_covariates, loc_scale, eps=eps, use_arcsinh=use_arcsinh)
    return Chronos2PackedInputs(
        context=context,
        future_covariates=future,
        group_ids=packed.group_ids,
        target_idx_ranges=packed.target_idx_ranges,
        loc_scale=loc_scale,
    )


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


def patch_chronos2_inputs(
    packed: Chronos2PackedInputs,
    *,
    patch_size: int = 16,
    patch_stride: int | None = None,
    output_patch_size: int | None = None,
    num_output_patches: int | None = None,
    context_length: int = 8192,
    time_encoding_scale: int | None = None,
    use_arcsinh: bool = False,
    instance_norm_eps: float = 1e-5,
) -> Chronos2PatchedInputs:
    """Patch packed V/W the way ``Chronos2Model.encode`` prepares embeddings."""
    if patch_stride is None:
        patch_stride = patch_size
    if output_patch_size is None:
        output_patch_size = patch_size
    if time_encoding_scale is None:
        time_encoding_scale = context_length
    horizon = packed.future_covariates.shape[-1]
    if num_output_patches is None:
        num_output_patches = max(1, (horizon + output_patch_size - 1) // output_patch_size)

    already_normed = packed.loc_scale is not None
    patched_context, attention_mask, loc_scale = prepare_patched_context(
        packed.context,
        patch_size=patch_size,
        patch_stride=patch_stride,
        context_length=context_length,
        time_encoding_scale=time_encoding_scale,
        apply_instance_norm=not already_normed,
        use_arcsinh=use_arcsinh,
        instance_norm_eps=instance_norm_eps,
        loc_scale=packed.loc_scale,
    )
    patched_future, _ = prepare_patched_future(
        packed.future_covariates,
        loc_scale,
        num_output_patches=num_output_patches,
        output_patch_size=output_patch_size,
        batch_size=packed.context.shape[0],
        time_encoding_scale=time_encoding_scale,
        use_arcsinh=use_arcsinh,
        instance_norm_eps=instance_norm_eps,
        apply_instance_norm=not already_normed,
    )
    return Chronos2PatchedInputs(
        patched_context=patched_context,
        attention_mask=attention_mask,
        patched_future=patched_future,
        loc_scale=loc_scale,
        group_ids=packed.group_ids,
        target_idx_ranges=packed.target_idx_ranges,
    )


def target_encode(
    id_codes: np.ndarray,
    cat_codes: np.ndarray,
    target: np.ndarray,
    n_items: int,
    n_categories: int,
    future_id_codes: np.ndarray | None = None,
    future_cat_codes: np.ndarray | None = None,
    smooth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-item smoothed target means: (smooth*item_mean + cat_sum)/(smooth + cat_count)."""
    mask = np.isfinite(target)
    target_masked = np.where(mask, target, 0.0)

    item_sums = np.bincount(id_codes, weights=target_masked * mask, minlength=n_items)
    item_counts = np.bincount(id_codes, weights=mask.astype(float), minlength=n_items)
    item_means = np.divide(item_sums, item_counts, out=np.zeros(n_items), where=item_counts > 0)

    n_slots = n_categories + 1
    combined_codes = id_codes * n_slots + cat_codes
    sums = np.bincount(combined_codes, weights=target_masked * mask, minlength=n_items * n_slots)
    counts = np.bincount(combined_codes, weights=mask.astype(float), minlength=n_items * n_slots)

    lookup = (smooth * np.repeat(item_means, n_slots) + sums) / (smooth + counts)
    encoded_past = lookup[combined_codes].astype(np.float32)

    encoded_future = None
    if future_id_codes is not None and future_cat_codes is not None:
        encoded_future = lookup[future_id_codes * n_slots + future_cat_codes].astype(np.float32)

    return encoded_past, encoded_future


def encode_categorical_covariate(
    past: ArrayLike,
    *,
    target: ArrayLike,
    future: ArrayLike | None = None,
    id_codes: np.ndarray | None = None,
    future_id_codes: np.ndarray | None = None,
    n_series: int = 1,
    use_target_encoding: bool = True,
    n_targets: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode one categorical covariate column to float32 past / optional future."""
    past_arr = np.asarray(past)
    target_arr = np.asarray(target, dtype=np.float32)
    if target_arr.ndim == 1:
        target_arr = target_arr.reshape(1, -1)
        n_targets = 1
    elif target_arr.shape[0] != n_targets:
        n_targets = int(target_arr.shape[0])

    if id_codes is None:
        id_codes = np.zeros(len(past_arr), dtype=np.intp)
    future_arr = None if future is None else np.asarray(future)
    if future_arr is not None and future_id_codes is None:
        future_id_codes = np.zeros(len(future_arr), dtype=np.intp)
    return _encode_categorical(
        past=past_arr,
        future=future_arr,
        target=target_arr,
        id_codes=id_codes,
        future_id_codes=future_id_codes,
        n_series=n_series,
        do_target_encode=bool(use_target_encoding and n_targets == 1),
    )


def _encode_categorical(
    past: np.ndarray,
    future: np.ndarray | None,
    target: np.ndarray,
    id_codes: np.ndarray,
    future_id_codes: np.ndarray | None,
    n_series: int,
    do_target_encode: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    past_s = pd.Series(past).astype("category")
    n_real = len(past_s.dtype.categories)
    nan_slot = n_real
    n_categories = n_real + 1

    past_codes = past_s.cat.codes.to_numpy(dtype=np.intp)
    past_codes = np.where(past_codes < 0, nan_slot, past_codes)

    future_codes = None
    if future is not None:
        future_s = pd.Series(future)
        codes = past_s.dtype.categories.get_indexer(future_s).astype(np.intp)
        future_codes = np.where(codes < 0, np.where(future_s.isna().to_numpy(), nan_slot, n_categories), codes)

    if do_target_encode:
        return target_encode(
            id_codes=id_codes,
            cat_codes=past_codes,
            target=target[0],
            n_items=n_series,
            n_categories=n_categories,
            future_id_codes=future_id_codes if future_codes is not None else None,
            future_cat_codes=future_codes,
        )

    enc_past = past_codes.astype(np.float32)
    enc_future = None
    if future_codes is not None:
        enc_future = np.where(future_codes == n_categories, np.nan, future_codes).astype(np.float32)
    return enc_past, enc_future


def _encode_covariate_columns(
    *,
    target_2ds: list[torch.Tensor],
    past_per_series: list[list[np.ndarray]],
    future_per_series: list[list[np.ndarray]],
    n_cov: int,
    n_future_cov: int,
    horizon: int,
    use_target_encoding: bool,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    n_series = len(target_2ds)
    n_targets = target_2ds[0].shape[0]
    series_lengths = [int(t.shape[-1]) for t in target_2ds]
    stacked_target = np.concatenate([t.numpy() for t in target_2ds], axis=1)
    id_codes = np.repeat(np.arange(n_series), series_lengths)
    future_id_codes = np.repeat(np.arange(n_series), horizon)
    nan_future = np.full(n_series * horizon, np.nan, dtype=np.float32)
    do_target_encode = bool(use_target_encoding and n_targets == 1)

    encoded_past_cols: list[list[np.ndarray]] = []
    encoded_future_cols: list[list[np.ndarray]] = []
    for col in range(n_cov):
        stacked_past = np.concatenate([rows[col] for rows in past_per_series])
        is_known_future = col >= n_cov - n_future_cov
        stacked_future = None
        if is_known_future:
            stacked_future = np.concatenate([rows[col - (n_cov - n_future_cov)] for rows in future_per_series])

        if _is_categorical(stacked_past):
            enc_past, enc_future = _encode_categorical(
                past=stacked_past,
                future=stacked_future,
                target=stacked_target,
                id_codes=id_codes,
                future_id_codes=future_id_codes if stacked_future is not None else None,
                n_series=n_series,
                do_target_encode=do_target_encode,
            )
        else:
            enc_past = np.asarray(stacked_past, dtype=np.float32)
            enc_future = np.asarray(stacked_future, dtype=np.float32) if stacked_future is not None else None

        if enc_future is None:
            enc_future = nan_future
        encoded_past_cols.append(_split_by_lengths(enc_past, series_lengths))
        encoded_future_cols.append(_split_by_lengths(enc_future, [horizon] * n_series))

    return encoded_past_cols, encoded_future_cols


def _split_by_lengths(values: np.ndarray, lengths: Sequence[int]) -> list[np.ndarray]:
    out = []
    start = 0
    for length in lengths:
        out.append(np.asarray(values[start : start + length], dtype=np.float32))
        start += length
    return out


def _covariate_layout(
    past_per_series: list[list[np.ndarray]],
    future_per_series: list[list[np.ndarray]],
    prediction_length: int | None,
    series_lengths: list[int],
) -> tuple[int, int, int]:
    n_covs = {len(rows) for rows in past_per_series}
    n_futs = {len(rows) for rows in future_per_series}
    if len(n_covs) != 1 or len(n_futs) != 1:
        raise ValueError("all series must have the same number of past and future covariate rows")
    n_cov = n_covs.pop()
    n_future_cov = n_futs.pop()
    if n_future_cov > 0 and n_cov == 0:
        raise ValueError("future_covariates requires past_covariates (known-future vars must have history)")
    if n_future_cov > n_cov:
        raise ValueError(
            "future_covariates rows must be a suffix of past_covariates rows, " f"got M_f={n_future_cov} > M={n_cov}"
        )

    for rows, t_len in zip(past_per_series, series_lengths):
        for row in rows:
            if len(row) != t_len:
                raise ValueError(f"past_covariates time length must match target, got {len(row)} vs {t_len}")

    inferred: int | None = None
    for rows in future_per_series:
        for row in rows:
            h = len(row)
            if inferred is None:
                inferred = h
            elif inferred != h:
                raise ValueError(f"all future_covariates must share horizon H, got {h} vs {inferred}")

    if prediction_length is None:
        if inferred is None:
            raise ValueError("prediction_length is required when future_covariates is omitted")
        horizon = inferred
    else:
        if prediction_length <= 0:
            raise ValueError(f"prediction_length must be positive, got {prediction_length}")
        if inferred is not None and inferred != prediction_length:
            raise ValueError(
                "prediction_length must match future_covariates length, " f"got {prediction_length} vs {inferred}"
            )
        horizon = prediction_length
    return n_cov, n_future_cov, horizon


def _parse_covariates(
    value: ArrayLike | Sequence[ArrayLike] | None,
    n_series: int,
    target_is_batch: bool,
    *,
    name: str,
) -> list[list[np.ndarray]]:
    if value is None:
        return [[] for _ in range(n_series)]
    if target_is_batch:
        if not _is_nested_series_list(value) and n_series != 1:
            raise ValueError(f"{name} must be a list when target contains {n_series} series")
        items = list(value) if _is_nested_series_list(value) else [value]
        if len(items) != n_series:
            raise ValueError(f"{name} list length must match number of target series, got {len(items)} vs {n_series}")
        return [_covariate_rows(item, name=name) if item is not None else [] for item in items]
    return [_covariate_rows(value, name=name)]


def _covariate_rows(payload: ArrayLike, *, name: str) -> list[np.ndarray]:
    if _is_row_list(payload):
        rows = [np.asarray(row) for row in payload]
    else:
        arr = _as_numpy(payload)
        if arr.ndim == 1:
            rows = [arr]
        elif arr.ndim == 2:
            rows = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError(f"{name} must have shape (T,) or (n_variates, T), got {tuple(arr.shape)}")
    for row in rows:
        if row.ndim != 1:
            raise ValueError(f"{name} rows must be 1-d, got {tuple(row.shape)}")
    return rows


def _is_row_list(value: object) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray, str, bytes)) or value is None:
        return False
    if not isinstance(value, Sequence) or len(value) == 0:
        return False
    first = value[0]
    return isinstance(first, (torch.Tensor, np.ndarray, Sequence)) and not isinstance(first, (str, bytes, int, float))


def _is_categorical(values: np.ndarray) -> bool:
    arr = np.asarray(values)
    if arr.dtype == np.bool_ or arr.dtype.kind in ("O", "U", "S"):
        return True
    return False


def _as_series_list(value: ArrayLike | Sequence[ArrayLike], *, name: str) -> list[torch.Tensor]:
    if _is_nested_series_list(value):
        series = [_to_float_tensor(item, name=name) for item in value]
        if not series:
            raise ValueError(f"{name} is empty. Provide at least one series.")
        return series
    return [_to_float_tensor(value, name=name)]


def _is_nested_series_list(value: object) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return False
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence):
        return False
    if len(value) == 0:
        return True
    first = value[0]
    return isinstance(first, (torch.Tensor, Sequence)) and not isinstance(first, (str, bytes, int, float))


def _to_float_tensor(value: ArrayLike, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(value, dtype=np.float32))
    if tensor.ndim == 0:
        raise ValueError(f"{name} must be at least 1-d, got scalar")
    if tensor.ndim > 2:
        raise ValueError(f"{name} must have shape (T,) or (n_variates, T), got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32)


def _as_variate_time(value: ArrayLike, *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else _to_float_tensor(value, name=name)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"{name} must have shape (T,) or (n_variates, T), got {tuple(tensor.shape)}")


def _as_numpy(value: ArrayLike) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _left_pad_and_cat_2d(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        n_variates, length = tensor.shape
        if length < max_len:
            pad = torch.full((n_variates, max_len - length), float("nan"), dtype=tensor.dtype)
            tensor = torch.cat([pad, tensor], dim=-1)
        padded.append(tensor)
    return torch.cat(padded, dim=0)
