# tt/tst_embedding.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN port of value/lag/temporal/static feature embedding.
# Mirrors HuggingFace TimeSeriesTransformerModel.create_network_inputs.
#
# FIX (this revision): ttnn.concat in this tt-metal build requires ALL
# input tensors to be in TILE_LAYOUT -- it does not auto-promote ROW_MAJOR
# inputs. The first version of this port built static_real_features /
# static_cat_features lookups in ROW_MAJOR (matching ttnn.embedding's
# input-index requirement) and fed them into ttnn.concat alongside
# TILE-layout tensors, which fails hard:
#   TT_FATAL: ttnn.concat: expected all input tensors to be in tile layout
# Fix: every tensor is forced to TILE_LAYOUT immediately before any
# ttnn.concat call, via an explicit _ensure_tile() helper. ROW_MAJOR is
# kept ONLY where required (ttnn.embedding's index input, ttnn.slice on
# pre-tile-layout source tensors), and converted back to TILE right after.

import ttnn

LAGS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37]
D_MODEL = 26


def _ensure_tile(x):
    """Force a tensor to TILE_LAYOUT if it isn't already -- required before
    any ttnn.concat call, since this op does not accept mixed layouts."""
    if x.layout != ttnn.TILE_LAYOUT:
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    return x


def _ttnn_abs(x):
    return ttnn.abs(x)


def _ttnn_log1p(x):
    return ttnn.log1p(x)


def _ttnn_log(x):
    return ttnn.log(x)


def _mean_scaler(device, context, observed_context):
    """
    context: ttnn [B, T, 1]. observed_context: ttnn [B, T, 1] (0/1 mask).
    Returns (scaled_context, loc, scale), all ttnn [B, 1, 1], TILE_LAYOUT.
    Matches torch reference:
        num = (context.abs() * observed).sum(dim=1, keepdim=True)
        den = observed.sum(dim=1, keepdim=True).clamp(min=1.0)
        scale = (num / den).clamp(min=1e-10)
        loc = zeros_like(scale)
        return (context - loc) / scale, loc, scale
    """
    context = _ensure_tile(context)
    observed_context = _ensure_tile(observed_context)

    abs_context = _ttnn_abs(context)
    weighted = ttnn.multiply(abs_context, observed_context)
    num = ttnn.sum(weighted, dim=1, keepdim=True)
    den = ttnn.sum(observed_context, dim=1, keepdim=True)
    den = ttnn.clamp(den, min=1.0)
    scale = ttnn.divide(num, den)
    scale = ttnn.clamp(scale, min=1e-10)
    loc = ttnn.zeros_like(scale)
    scaled = ttnn.divide(ttnn.subtract(context, loc), scale)
    return scaled, loc, scale


def _get_lagged_subsequences(device, sequence, lags, subseq_len, shift=0):
    """
    sequence: ttnn [B, full_len]. Returns ttnn [B, subseq_len, len(lags)],
    TILE_LAYOUT.

    Lag offsets are static Python ints (compile-time constants from the
    LAGS list), so this is expressed as repeated ttnn.slice + ttnn.concat,
    NOT ttnn.embedding -- embedding is index->row lookup for data-dependent
    indices; here the indices (start/end per lag) are fixed at trace time,
    making slicing the correct primitive.
    """
    B = sequence.shape[0]
    full_len = sequence.shape[1]
    pieces = []
    for lag in lags:
        idx = lag - shift
        start = full_len - idx - subseq_len
        end = full_len - idx if idx > 0 else full_len
        piece = ttnn.slice(sequence, slice_start=[0, start], slice_end=[B, end])
        # [B, subseq_len] -> [B, subseq_len, 1] for concat along last dim
        piece = ttnn.reshape(piece, (B, subseq_len, 1))
        piece = _ensure_tile(piece)
        pieces.append(piece)
    return ttnn.concat(pieces, dim=-1)


def _build_static_feat(device, loc, scale, static_real_features, static_cat_features, cat_embedder_weight):
    """
    loc, scale: ttnn [B, 1, 1]. static_real_features: ttnn [B, num_static_real] or None.
    static_cat_features: ttnn [B, num_static_cat] uint32/int32 or None.
    cat_embedder_weight: ttnn [vocab_size, embedding_dim].
    Returns ttnn [B, static_dim], TILE_LAYOUT.
    """
    B = loc.shape[0]
    squeezed_loc = ttnn.reshape(loc, (B, 1))
    squeezed_scale = ttnn.reshape(scale, (B, 1))
    squeezed_loc = _ensure_tile(squeezed_loc)
    squeezed_scale = _ensure_tile(squeezed_scale)

    log_abs_loc = _ttnn_log1p(_ttnn_abs(squeezed_loc))
    log_scale = _ttnn_log(squeezed_scale)
    log_abs_loc = _ensure_tile(log_abs_loc)
    log_scale = _ensure_tile(log_scale)
    static_feat = ttnn.concat([log_abs_loc, log_scale], dim=1)

    if static_real_features is not None:
        static_real_features = _ensure_tile(static_real_features)
        static_feat = ttnn.concat([static_real_features, static_feat], dim=1)

    if static_cat_features is not None:
        # static_cat_features[:, 0] -- first categorical column, matching
        # the torch reference's static_cat_features[:, 0].long()
        # ttnn.embedding requires ROW_MAJOR index input -- do NOT convert
        # cat_col to tile before calling ttnn.embedding.
        cat_col = ttnn.slice(static_cat_features, slice_start=[0, 0], slice_end=[B, 1])
        cat_col = ttnn.reshape(cat_col, (B, 1))
        cat_col = ttnn.to_layout(cat_col, ttnn.ROW_MAJOR_LAYOUT)
        emb = ttnn.embedding(cat_col, cat_embedder_weight)  # [B, 1, embedding_dim]
        emb = ttnn.reshape(emb, (B, emb.shape[-1]))
        emb = _ensure_tile(emb)  # convert back to TILE before concat
        static_feat = ttnn.concat([emb, static_feat], dim=1)

    return static_feat


def prepare_encoder_input(
    device,
    past_values,
    past_time_features,
    past_observed_mask,
    static_cat_features,
    static_real_features,
    cat_embedder_weight,
    value_proj_weight,
    pos_emb_weight,
    context_length=24,
):
    """
    All tensor args are ttnn tensors. past_values: [B, past_len].
    past_time_features: [B, past_len, num_time_features].
    past_observed_mask: [B, past_len].
    static_cat_features: [B, num_static_cat] or None.
    static_real_features: [B, num_static_real] or None.
    cat_embedder_weight: [vocab, embedding_dim].
    value_proj_weight: [in_features, D_MODEL] (pre-transposed at load time
        to ttnn.linear's [in,out] convention -- see tst_model.py's
        load_weights()).
    pos_emb_weight: [max_positions, D_MODEL].

    Returns (emb, loc, scale) -- all ttnn tensors, TILE_LAYOUT.
    """
    B = past_values.shape[0]
    past_len = past_values.shape[1]

    context = ttnn.slice(past_values, slice_start=[0, past_len - context_length], slice_end=[B, past_len])
    context = ttnn.reshape(context, (B, context_length, 1))
    observed_context = ttnn.slice(
        past_observed_mask, slice_start=[0, past_len - context_length], slice_end=[B, past_len]
    )
    observed_context = ttnn.reshape(observed_context, (B, context_length, 1))

    _, loc, scale = _mean_scaler(device, context, observed_context)

    past_values_3d = ttnn.reshape(past_values, (B, past_len, 1))
    past_values_3d = _ensure_tile(past_values_3d)
    inputs_scaled = ttnn.divide(ttnn.subtract(past_values_3d, loc), scale)
    inputs_scaled = ttnn.reshape(inputs_scaled, (B, past_len))

    lagged = _get_lagged_subsequences(device, inputs_scaled, LAGS, context_length)
    reshaped_lagged = ttnn.reshape(lagged, (B, context_length, len(LAGS)))
    reshaped_lagged = _ensure_tile(reshaped_lagged)

    time_feat = ttnn.slice(
        past_time_features,
        slice_start=[0, past_len - context_length, 0],
        slice_end=[B, past_len, past_time_features.shape[-1]],
    )
    time_feat = _ensure_tile(time_feat)

    static_feat = _build_static_feat(device, loc, scale, static_real_features, static_cat_features, cat_embedder_weight)
    static_dim = static_feat.shape[-1]
    expanded_static = ttnn.reshape(static_feat, (B, 1, static_dim))
    expanded_static = ttnn.repeat(expanded_static, (1, context_length, 1))
    expanded_static = _ensure_tile(expanded_static)

    features = ttnn.concat([expanded_static, time_feat], dim=-1)
    transformer_inputs = ttnn.concat([reshaped_lagged, features], dim=-1)

    emb = ttnn.linear(transformer_inputs, value_proj_weight)

    positions = ttnn.slice(pos_emb_weight, slice_start=[0, 0], slice_end=[context_length, D_MODEL])
    positions = ttnn.reshape(positions, (1, context_length, D_MODEL))
    positions = _ensure_tile(positions)
    emb = ttnn.add(emb, positions)

    return emb, loc, scale


def prepare_decoder_input(
    device,
    future_values,
    future_time_features,
    past_values,
    loc,
    scale,
    static_cat_features,
    static_real_features,
    cat_embedder_weight,
    value_proj_weight,
    pos_emb_weight,
    context_length=24,
    shift=0,
):
    """
    All tensor args are ttnn tensors. future_values: [B, pred_len].
    future_time_features: [B, pred_len, num_time_features].
    past_values: [B, past_len]. loc, scale: [B, 1, 1] from
    prepare_encoder_input.

    Returns emb: ttnn [B, pred_len, D_MODEL], TILE_LAYOUT.
    """
    B = future_values.shape[0]
    pred_len = future_values.shape[1]
    past_len = past_values.shape[1]

    full_seq = ttnn.concat([_ensure_tile(past_values), _ensure_tile(future_values)], dim=1)
    full_len = past_len + pred_len
    full_seq_3d = ttnn.reshape(full_seq, (B, full_len, 1))
    full_seq_3d = _ensure_tile(full_seq_3d)
    inputs_scaled = ttnn.divide(ttnn.subtract(full_seq_3d, loc), scale)
    inputs_scaled = ttnn.reshape(inputs_scaled, (B, full_len))

    lagged = _get_lagged_subsequences(device, inputs_scaled, LAGS, pred_len, shift=shift)
    reshaped_lagged = ttnn.reshape(lagged, (B, pred_len, len(LAGS)))
    reshaped_lagged = _ensure_tile(reshaped_lagged)

    time_feat = _ensure_tile(future_time_features)

    static_feat = _build_static_feat(device, loc, scale, static_real_features, static_cat_features, cat_embedder_weight)
    static_dim = static_feat.shape[-1]
    expanded_static = ttnn.reshape(static_feat, (B, 1, static_dim))
    expanded_static = ttnn.repeat(expanded_static, (1, pred_len, 1))
    expanded_static = _ensure_tile(expanded_static)

    features = ttnn.concat([expanded_static, time_feat], dim=-1)
    transformer_inputs = ttnn.concat([reshaped_lagged, features], dim=-1)

    emb = ttnn.linear(transformer_inputs, value_proj_weight)

    positions = ttnn.slice(
        pos_emb_weight,
        slice_start=[context_length, 0],
        slice_end=[context_length + pred_len, D_MODEL],
    )
    positions = ttnn.reshape(positions, (1, pred_len, D_MODEL))
    positions = _ensure_tile(positions)
    emb = ttnn.add(emb, positions)

    return emb
