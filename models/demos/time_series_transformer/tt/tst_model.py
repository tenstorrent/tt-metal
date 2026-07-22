# tt/tst_model.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
Forward-pass and generation entry points for the Time Series Transformer
TTNN port: encoder/decoder layer stacks, and three generation paths --
generate() (untraced reference), generate_traced() (single fused trace,
no KV-cache; kept as a correctness gate, not a performance path), and
teacher_forced_nll() (exact NLL under teacher forcing). Weight loading
lives in tst_weights.py. Distribution dispatch lives in tst_distribution.py.
The KV-cache + per-layer-trace inference path lives in
tst_model_cached_additions.py, not here.
"""

import torch
import torch.nn.functional as F

import ttnn

from .attention import build_causal_mask, precompute_cross_attn_kv
from .tst_config import CONTEXT_LENGTH, D_MODEL, LAGS, NUM_PARALLEL_SAMPLES, PADDED_WIDTH, PREDICTION_LENGTH
from .tst_decoder_layer import tst_decoder_layer
from .tst_distribution import (
    _distribution_head,
    _sample_next_step,
    negative_binomial_params,
    nll_negative_binomial,
    nll_normal,
    nll_student_t,
    normal_params,
    student_t_params,
)
from .tst_embedding import prepare_decoder_input, prepare_encoder_input
from .tst_encoder_layer import tst_encoder_layer
from .tst_io import (
    _apply_layernorm_ttnn,
    _future_time_to_ttnn,
    _future_vals_k_to_ttnn,
    _inputs_to_ttnn,
    _past_values_repeated_to_ttnn,
)


def run_encoder(device, encoder_input, weights, apply_layernorm=False):
    h = encoder_input
    if apply_layernorm:
        h = _apply_layernorm_ttnn(h, weights["encoder_layernorm_ttnn"])
    pad = PADDED_WIDTH - h.shape[-1]
    if pad > 0:
        h = ttnn.pad(h, padding=[(0, 0), (0, 0), (0, pad)], value=0.0)
    hidden = h
    for i in range(2):
        hidden = tst_encoder_layer(hidden, weights, layer_idx=i)
    return hidden


def run_decoder_step(
    device, decoder_input, encoder_hidden, weights, causal_mask, apply_layernorm=False, precomputed_kv=None
):
    h = decoder_input
    if apply_layernorm:
        h = _apply_layernorm_ttnn(h, weights["decoder_layernorm_ttnn"])
    pad = PADDED_WIDTH - h.shape[-1]
    if pad > 0:
        h = ttnn.pad(h, padding=[(0, 0), (0, 0), (0, pad)], value=0.0)
    hidden = h
    for i in range(2):
        hidden = tst_decoder_layer(
            hidden, encoder_hidden, weights, layer_idx=i, causal_mask=causal_mask, precomputed_kv=precomputed_kv
        )
    return hidden


@torch.no_grad()
def generate(
    device,
    weights,
    past_values,
    past_time_features,
    future_time_features,
    past_observed_mask,
    static_categorical_features,
    static_real_features,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    num_parallel_samples=NUM_PARALLEL_SAMPLES,
):
    B = past_values.shape[0]
    S = num_parallel_samples
    device_cpu = past_values.device
    dt = weights.get("dist_type", "student_t")

    pv_tt, pt_tt, pm_tt, sc_tt, sr_tt = _inputs_to_ttnn(
        device,
        past_values,
        past_time_features,
        past_observed_mask,
        static_categorical_features,
        static_real_features,
    )
    enc_emb, loc, scale = prepare_encoder_input(
        device,
        past_values=pv_tt,
        past_time_features=pt_tt,
        past_observed_mask=pm_tt,
        static_cat_features=sc_tt,
        static_real_features=sr_tt,
        cat_embedder_weight=weights["cat_embedder"],
        value_proj_weight=weights["encoder_value_proj"],
        pos_emb_weight=weights["encoder_pos_emb"],
        context_length=context_length,
    )
    enc_emb = _apply_layernorm_ttnn(enc_emb, weights["encoder_layernorm_ttnn"])
    encoder_hidden = run_encoder(device, enc_emb, weights)

    loc_t = ttnn.to_torch(loc).float()
    scale_t = ttnn.to_torch(scale).float()
    repeated_loc_t = loc_t.repeat_interleave(S, dim=0)
    repeated_scale_t = scale_t.repeat_interleave(S, dim=0)
    _sc = repeated_scale_t.squeeze(-1).squeeze(-1)
    _lc = repeated_loc_t.squeeze(-1).squeeze(-1)

    repeated_past_raw = past_values.repeat_interleave(S, dim=0).float()
    repeated_past_raw_tt = _past_values_repeated_to_ttnn(device, repeated_past_raw)
    repeated_future_time = future_time_features.repeat_interleave(S, dim=0)
    repeated_future_time_tt = _future_time_to_ttnn(device, repeated_future_time)
    repeated_static_cat = static_categorical_features.repeat_interleave(S, dim=0)
    repeated_static_cat_tt = ttnn.from_torch(
        repeated_static_cat.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    repeated_static_real = static_real_features.repeat_interleave(S, dim=0)
    repeated_static_real_tt = ttnn.from_torch(
        repeated_static_real.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    repeated_loc_tt = ttnn.from_torch(repeated_loc_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    repeated_scale_tt = ttnn.from_torch(
        repeated_scale_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    enc_hidden_t = ttnn.to_torch(encoder_hidden).float()
    enc_hidden_rep = ttnn.from_torch(
        enc_hidden_t.repeat_interleave(S, dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    future_samples = []
    for k in range(prediction_length):
        if k == 0:
            future_vals_k = torch.zeros(B * S, 1, device=device_cpu)
        else:
            prev_raw = torch.stack(future_samples, dim=1)
            future_vals_k = torch.cat([prev_raw, torch.zeros(B * S, 1, device=device_cpu)], dim=1)

        future_vals_k_tt = _future_vals_k_to_ttnn(device, future_vals_k)
        future_time_k_tt = ttnn.slice(
            repeated_future_time_tt,
            slice_start=[0, 0, 0],
            slice_end=[B * S, k + 1, repeated_future_time_tt.shape[-1]],
        )
        dec_emb_k = prepare_decoder_input(
            device,
            future_values=future_vals_k_tt,
            future_time_features=future_time_k_tt,
            past_values=repeated_past_raw_tt,
            loc=repeated_loc_tt,
            scale=repeated_scale_tt,
            static_cat_features=repeated_static_cat_tt,
            static_real_features=repeated_static_real_tt,
            cat_embedder_weight=weights["cat_embedder"],
            value_proj_weight=weights["decoder_value_proj"],
            pos_emb_weight=weights["decoder_pos_emb"],
            context_length=context_length,
        )
        dec_emb_k = _apply_layernorm_ttnn(dec_emb_k, weights["decoder_layernorm_ttnn"])
        mask_k = build_causal_mask(device, k + 1, batch_size=dec_emb_k.shape[0])
        dec_out = run_decoder_step(device, dec_emb_k, enc_hidden_rep, weights, causal_mask=mask_k)
        dec_out_torch = ttnn.to_torch(dec_out).float()[..., :D_MODEL]

        params = _distribution_head(dec_out_torch[:, -1:, :], weights)
        next_sample = _sample_next_step(params, dt, _lc, _sc)
        future_samples.append(next_sample)

    samples = torch.stack(future_samples, dim=1)
    return samples.reshape(B, S, prediction_length)


def _run_decoder_layers_fixed(hidden_ttnn, weights, causal_mask, precomputed_kv):
    """2-layer decoder stack on fixed-shape tensors -- safe to trace."""
    hidden = hidden_ttnn
    for layer_idx in range(2):
        hidden = tst_decoder_layer(
            hidden_states=hidden,
            encoder_hidden_states=None,
            weights=weights,
            layer_idx=layer_idx,
            causal_mask=causal_mask,
            precomputed_kv=precomputed_kv[layer_idx],
        )
    return hidden


class TracedDecoderContext:
    """
    Holds a decoder trace captured once and replayed many times across
    autoregressive steps.

    Trace capture is separated from per-call replay because capturing and
    releasing a trace on every generate_traced() call left the allocator in
    an inconsistent state: release_trace() does not guarantee the device has
    reclaimed the previous trace's resources before the next
    begin_trace_capture()'s allocations run, and a second call in the same
    process would hang.

    Build once via build_traced_decoder_context(), replay via
    run_traced_generation() (or generate_traced(traced_ctx=...)) as many
    times as needed, and call release() (or use as a context manager) when
    done.
    """

    def __init__(
        self,
        device,
        trace_id,
        captured_dec_input,
        traced_out,
        causal_mask_full,
        precomputed_kv,
        enc_hidden_rep,
        repeated_loc_tt,
        repeated_scale_tt,
        repeated_past_raw_tt,
        repeated_future_time_tt,
        repeated_static_cat_tt,
        repeated_static_real_tt,
        _lc,
        _sc,
        BS,
        T_max,
        dist_type,
    ):
        self.device = device
        self.trace_id = trace_id
        self.captured_dec_input = captured_dec_input
        self.traced_out = traced_out
        self.causal_mask_full = causal_mask_full
        self.precomputed_kv = precomputed_kv
        self.enc_hidden_rep = enc_hidden_rep
        self.repeated_loc_tt = repeated_loc_tt
        self.repeated_scale_tt = repeated_scale_tt
        self.repeated_past_raw_tt = repeated_past_raw_tt
        self.repeated_future_time_tt = repeated_future_time_tt
        self.repeated_static_cat_tt = repeated_static_cat_tt
        self.repeated_static_real_tt = repeated_static_real_tt
        self._lc = _lc
        self._sc = _sc
        self.BS = BS
        self.T_max = T_max
        self.dist_type = dist_type
        self._released = False

    def release(self):
        if not self._released:
            ttnn.release_trace(self.device, self.trace_id)
            self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def build_traced_decoder_context(
    device,
    weights,
    past_values,
    past_time_features,
    future_time_features,
    past_observed_mask,
    static_categorical_features,
    static_real_features,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    num_parallel_samples=NUM_PARALLEL_SAMPLES,
):
    """
    One-time setup for traced autoregressive decoding: runs the encoder,
    precomputes cross-attention KV, allocates the fixed-shape device buffer
    the trace reads from, and captures the decoder trace once.

    Returns a TracedDecoderContext for use with run_traced_generation() or
    generate_traced(traced_ctx=...).
    """
    B = past_values.shape[0]
    S = num_parallel_samples
    BS = B * S
    T_max = prediction_length
    dt = weights.get("dist_type", "student_t")

    pv_tt, pt_tt, pm_tt, sc_tt, sr_tt = _inputs_to_ttnn(
        device,
        past_values,
        past_time_features,
        past_observed_mask,
        static_categorical_features,
        static_real_features,
    )
    enc_emb, loc, scale = prepare_encoder_input(
        device,
        past_values=pv_tt,
        past_time_features=pt_tt,
        past_observed_mask=pm_tt,
        static_cat_features=sc_tt,
        static_real_features=sr_tt,
        cat_embedder_weight=weights["cat_embedder"],
        value_proj_weight=weights["encoder_value_proj"],
        pos_emb_weight=weights["encoder_pos_emb"],
        context_length=context_length,
    )
    enc_emb = _apply_layernorm_ttnn(enc_emb, weights["encoder_layernorm_ttnn"])
    encoder_hidden = run_encoder(device, enc_emb, weights)

    loc_t = ttnn.to_torch(loc).float()
    scale_t = ttnn.to_torch(scale).float()
    repeated_loc_t = loc_t.repeat_interleave(S, dim=0)
    repeated_scale_t = scale_t.repeat_interleave(S, dim=0)
    _sc = repeated_scale_t.squeeze(-1).squeeze(-1)
    _lc = repeated_loc_t.squeeze(-1).squeeze(-1)

    repeated_past_raw = past_values.repeat_interleave(S, dim=0).float()
    repeated_past_raw_tt = _past_values_repeated_to_ttnn(device, repeated_past_raw)
    repeated_future_time = future_time_features.repeat_interleave(S, dim=0)
    repeated_future_time_tt = _future_time_to_ttnn(device, repeated_future_time)
    repeated_static_cat = static_categorical_features.repeat_interleave(S, dim=0)
    repeated_static_cat_tt = ttnn.from_torch(
        repeated_static_cat.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    repeated_static_real = static_real_features.repeat_interleave(S, dim=0)
    repeated_static_real_tt = ttnn.from_torch(
        repeated_static_real.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    repeated_loc_tt = ttnn.from_torch(repeated_loc_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    repeated_scale_tt = ttnn.from_torch(
        repeated_scale_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    enc_hidden_t = ttnn.to_torch(encoder_hidden).float()
    enc_hidden_rep = ttnn.from_torch(
        enc_hidden_t.repeat_interleave(S, dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    precomputed_kv = []
    for layer_idx in range(2):
        w_cross = weights[f"decoder.layers.{layer_idx}"]["encoder_attn"]
        k_pre, v_pre = precompute_cross_attn_kv(enc_hidden_rep, w_cross)
        precomputed_kv.append((k_pre, v_pre))

    causal_mask_full = build_causal_mask(device, T_max, batch_size=BS)

    # Must be allocated before begin_trace_capture so it isn't captured as
    # a trace-internal allocation.
    captured_dec_input = ttnn.from_torch(
        torch.zeros(BS, T_max, PADDED_WIDTH, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Warmup run to compile kernels before capture (untraced).
    _ = _run_decoder_layers_fixed(captured_dec_input, weights, causal_mask_full, precomputed_kv)
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_out = _run_decoder_layers_fixed(captured_dec_input, weights, causal_mask_full, precomputed_kv)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    return TracedDecoderContext(
        device=device,
        trace_id=trace_id,
        captured_dec_input=captured_dec_input,
        traced_out=traced_out,
        causal_mask_full=causal_mask_full,
        precomputed_kv=precomputed_kv,
        enc_hidden_rep=enc_hidden_rep,
        repeated_loc_tt=repeated_loc_tt,
        repeated_scale_tt=repeated_scale_tt,
        repeated_past_raw_tt=repeated_past_raw_tt,
        repeated_future_time_tt=repeated_future_time_tt,
        repeated_static_cat_tt=repeated_static_cat_tt,
        repeated_static_real_tt=repeated_static_real_tt,
        _lc=_lc,
        _sc=_sc,
        BS=BS,
        T_max=T_max,
        dist_type=dt,
    )


def _get_lagged_cpu(seq_2d, lags, subseq_len):
    """
    Vectorized CPU lag extraction using torch index ops instead of a Python
    double-for-loop: O(len(lags)) Python-level iterations regardless of
    subseq_len or BS.

    seq_2d: [BS, full_len] float32. Returns [BS, subseq_len, len(lags)] float32.
    """
    BS, full_len = seq_2d.shape
    out = seq_2d.new_zeros(BS, subseq_len, len(lags))
    t_idx = torch.arange(subseq_len, dtype=torch.long)
    for li, lag in enumerate(lags):
        src = full_len - subseq_len - lag + t_idx  # [subseq_len]
        valid = (src >= 0) & (src < full_len)
        safe_src = src.clamp(0, full_len - 1)
        out[:, :, li] = seq_2d[:, safe_src] * valid.float()
    return out


def _build_static_feat_cpu(loc_cpu, scale_cpu, static_real_cpu, static_cat_cpu, cat_emb_w_cpu):
    """CPU mirror of _build_static_feat. Returns [BS, static_dim] float32."""
    lc = loc_cpu.squeeze(-1).squeeze(-1)
    sc = scale_cpu.squeeze(-1).squeeze(-1)
    parts = [
        torch.log(torch.abs(lc) + 1.0).unsqueeze(-1),
        torch.log(sc + 1.0).unsqueeze(-1),
    ]
    if static_real_cpu is not None and static_real_cpu.numel() > 0:
        parts.append(static_real_cpu.float())
    if static_cat_cpu is not None:
        for col in range(static_cat_cpu.shape[1]):
            parts.append(cat_emb_w_cpu[static_cat_cpu[:, col].long()])
    return torch.cat(parts, dim=-1)


def _prepare_dec_step_cpu(
    k,
    future_samples_so_far,
    past_values_cpu,
    future_time_cpu,
    static_feat_cpu,
    loc_cpu,
    scale_cpu,
    value_proj_cpu,
    dec_ln_w_cpu,
    dec_ln_b_cpu,
    pos_emb_cpu,
    context_length,
    T_max,
):
    """
    Full per-step decoder embedding on CPU. Returns [BS, T_max, PADDED_WIDTH]
    bfloat16 CPU tensor.

    Per tt-metal trace docs, no device ops may run between execute_trace
    replays, so lag extraction, value projection, positional embedding, and
    decoder input layernorm all run here on CPU instead. See
    tech_reports/AdvancedPerformanceOptimizationsForModels/
    AdvancedPerformanceOptimizationsForModels.md.

    Core decoder compute (self-attn, cross-attn, FFN, internal layernorms)
    is unchanged inside the trace on-device.
    """
    BS = past_values_cpu.shape[0]
    pred_len_k = k + 1

    if k == 0:
        future_vals_k = torch.zeros(BS, 1)
    else:
        prev = torch.stack(future_samples_so_far, dim=1)
        future_vals_k = torch.cat([prev, torch.zeros(BS, 1)], dim=1)

    full_seq = torch.cat([past_values_cpu, future_vals_k], dim=1)
    full_seq_scaled = (full_seq - loc_cpu.squeeze(-1)) / scale_cpu.squeeze(-1)

    lagged = _get_lagged_cpu(full_seq_scaled, LAGS, pred_len_k)

    time_feat_k = future_time_cpu[:, :pred_len_k, :]
    expanded_static = static_feat_cpu.unsqueeze(1).expand(BS, pred_len_k, static_feat_cpu.shape[-1])
    features = torch.cat([expanded_static, time_feat_k], dim=-1)
    transformer_inputs = torch.cat([lagged, features], dim=-1)

    emb = transformer_inputs @ value_proj_cpu

    pos = pos_emb_cpu[context_length : context_length + pred_len_k]
    emb = emb + pos.unsqueeze(0)

    emb = F.layer_norm(emb.float(), [D_MODEL], weight=dec_ln_w_cpu.float(), bias=dec_ln_b_cpu.float())

    pad_seq = T_max - pred_len_k
    pad_feat = PADDED_WIDTH - D_MODEL
    if pad_seq > 0:
        emb = F.pad(emb, (0, 0, 0, pad_seq))
    if pad_feat > 0:
        emb = F.pad(emb, (0, pad_feat))

    return emb.to(torch.bfloat16)


def run_traced_generation(ctx, weights, context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH):
    """
    Autoregressive sampling loop against a pre-captured TracedDecoderContext.

    Per-step device ops are only: copy_host_to_device_tensor -> execute_trace
    -> .cpu() readback. All embedding prep runs on CPU before
    copy_host_to_device_tensor, per the tt-metal trace requirement that no
    other device ops run between execute_trace replays.

    Core decoder compute (self-attn, cross-attn, FFN, layernorm) stays
    inside the trace on-device, unchanged.
    """
    device = ctx.device
    BS = ctx.BS
    T_max = ctx.T_max
    dt = ctx.dist_type

    past_values_cpu = ttnn.to_torch(ctx.repeated_past_raw_tt).float()
    future_time_cpu = ttnn.to_torch(ctx.repeated_future_time_tt).float()
    loc_cpu = ttnn.to_torch(ctx.repeated_loc_tt).float()
    scale_cpu = ttnn.to_torch(ctx.repeated_scale_tt).float()
    static_cat_cpu = ttnn.to_torch(ctx.repeated_static_cat_tt).long()
    static_real_cpu = ttnn.to_torch(ctx.repeated_static_real_tt).float()
    cat_emb_w_cpu = ttnn.to_torch(weights["cat_embedder"]).float()
    value_proj_cpu = ttnn.to_torch(weights["decoder_value_proj"]).float()
    pos_emb_cpu = ttnn.to_torch(weights["decoder_pos_emb"]).float()
    dec_ln = weights["decoder_layernorm_ttnn"]
    dec_ln_w_cpu = ttnn.to_torch(dec_ln["weight"]).float().squeeze()
    dec_ln_b_cpu = ttnn.to_torch(dec_ln["bias"]).float().squeeze()

    # Static features constant across all steps — compute once
    static_feat_cpu = _build_static_feat_cpu(loc_cpu, scale_cpu, static_real_cpu, static_cat_cpu, cat_emb_w_cpu)

    future_samples = []

    for k in range(prediction_length):
        step_input = _prepare_dec_step_cpu(
            k,
            future_samples,
            past_values_cpu,
            future_time_cpu,
            static_feat_cpu,
            loc_cpu,
            scale_cpu,
            value_proj_cpu,
            dec_ln_w_cpu,
            dec_ln_b_cpu,
            pos_emb_cpu,
            context_length,
            T_max,
        )  # [BS, T_max, PADDED_WIDTH] bfloat16 CPU

        step_host_tt = ttnn.from_torch(step_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)
        ttnn.copy_host_to_device_tensor(step_host_tt, ctx.captured_dec_input)
        ttnn.execute_trace(device, ctx.trace_id, cq_id=0, blocking=True)
        dec_out_torch = ttnn.to_torch(ctx.traced_out).float()[..., :D_MODEL]

        params = _distribution_head(dec_out_torch[:, k : k + 1, :], weights)
        next_sample = _sample_next_step(params, dt, ctx._lc, ctx._sc)
        future_samples.append(next_sample)

    return torch.stack(future_samples, dim=1)


@torch.no_grad()
def generate_traced(
    device,
    weights,
    past_values,
    past_time_features,
    future_time_features,
    past_observed_mask,
    static_categorical_features,
    static_real_features,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    num_parallel_samples=NUM_PARALLEL_SAMPLES,
    traced_ctx=None,
):
    """
    generate() with a TTNN trace over the decoder stack, no KV-cache.

    DEPRECATION STATUS: this is a correctness gate for test_tst_e2e_traced.py
    only -- never benchmarked against Stage 1 targets. See
    ../CHANGELOG.md "Decode path retirement plan" for the removal condition.
    Do not add new features here; add them to generate_traced_cached() in
    tst_model_cached_additions.py instead.

    Called without traced_ctx, builds and releases its own
    TracedDecoderContext internally — signature and behavior match
    generate() for a single call. Pass a pre-built TracedDecoderContext via
    traced_ctx to reuse an already-captured trace across multiple calls
    (see TracedDecoderContext).
    """
    B = past_values.shape[0]
    S = num_parallel_samples

    owns_ctx = traced_ctx is None
    if owns_ctx:
        traced_ctx = build_traced_decoder_context(
            device,
            weights,
            past_values,
            past_time_features,
            future_time_features,
            past_observed_mask,
            static_categorical_features,
            static_real_features,
            context_length=context_length,
            prediction_length=prediction_length,
            num_parallel_samples=num_parallel_samples,
        )

    try:
        samples = run_traced_generation(
            traced_ctx,
            weights,
            context_length=context_length,
            prediction_length=prediction_length,
        )
    finally:
        if owns_ctx:
            traced_ctx.release()

    return samples.reshape(B, S, prediction_length)


@torch.no_grad()
def teacher_forced_nll(
    device,
    weights,
    past_values,
    past_time_features,
    future_time_features,
    future_values,
    past_observed_mask,
    static_categorical_features,
    static_real_features,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
):
    B = past_values.shape[0]

    pv_tt, pt_tt, pm_tt, sc_tt, sr_tt = _inputs_to_ttnn(
        device,
        past_values,
        past_time_features,
        past_observed_mask,
        static_categorical_features,
        static_real_features,
    )
    enc_emb, loc, scale = prepare_encoder_input(
        device,
        past_values=pv_tt,
        past_time_features=pt_tt,
        past_observed_mask=pm_tt,
        static_cat_features=sc_tt,
        static_real_features=sr_tt,
        cat_embedder_weight=weights["cat_embedder"],
        value_proj_weight=weights["encoder_value_proj"],
        pos_emb_weight=weights["encoder_pos_emb"],
        context_length=context_length,
    )
    enc_emb = _apply_layernorm_ttnn(enc_emb, weights["encoder_layernorm_ttnn"])
    encoder_hidden = run_encoder(device, enc_emb, weights)

    future_values_tt = ttnn.from_torch(
        future_values.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    future_time_tt = _future_time_to_ttnn(device, future_time_features)
    dec_emb = prepare_decoder_input(
        device,
        future_values=future_values_tt,
        future_time_features=future_time_tt,
        past_values=pv_tt,
        loc=loc,
        scale=scale,
        static_cat_features=sc_tt,
        static_real_features=sr_tt,
        cat_embedder_weight=weights["cat_embedder"],
        value_proj_weight=weights["decoder_value_proj"],
        pos_emb_weight=weights["decoder_pos_emb"],
        context_length=context_length,
    )
    dec_emb = _apply_layernorm_ttnn(dec_emb, weights["decoder_layernorm_ttnn"])

    causal_mask = build_causal_mask(device, prediction_length, batch_size=dec_emb.shape[0])
    dec_out = run_decoder_step(device, dec_emb, encoder_hidden, weights, causal_mask=causal_mask)
    dec_out_torch = ttnn.to_torch(dec_out).float()[..., :D_MODEL]

    dh = weights["dist_head"]
    dt = weights.get("dist_type", "student_t")
    scale_t = ttnn.to_torch(scale).float()
    loc_t = ttnn.to_torch(loc).float()
    _sc = scale_t.squeeze(-1)
    _lc = loc_t.squeeze(-1)

    if dt == "normal":
        loc_d, scale_d = normal_params(dec_out_torch, dh["w0"], dh["b0"], dh["w1"], dh["b1"])
        targets_norm = (future_values.float() - _lc) / _sc
        nll = nll_normal(loc_d, scale_d, targets_norm)
        return nll.mean().item() + _sc.log().mean().item()
    elif dt == "negative_binomial":
        total_count, logits = negative_binomial_params(dec_out_torch, dh["w0"], dh["b0"], dh["w1"], dh["b1"])
        logits_scaled = logits + _sc.log()
        nll = nll_negative_binomial(total_count, logits_scaled, future_values.float())
        return nll.mean().item()
    else:  # student_t
        df, loc_d, scale_d = student_t_params(dec_out_torch, dh["w0"], dh["b0"], dh["w1"], dh["b1"], dh["w2"], dh["b2"])
        targets_norm = (future_values.float() - _lc) / _sc
        nll = nll_student_t(df, loc_d, scale_d, targets_norm)
        return nll.mean().item() + _sc.log().mean().item()
