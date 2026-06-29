# tt/tst_model.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import torch
import torch.nn.functional as F

import ttnn

from .tst_attention import build_causal_mask, precompute_cross_attn_kv
from .tst_decoder_layer import tst_decoder_layer
from .tst_distribution import (
    negative_binomial_params,
    nll_negative_binomial,
    nll_normal,
    nll_student_t,
    normal_params,
    sample_negative_binomial,
    sample_normal,
    sample_student_t,
    student_t_params,
)
from .tst_embedding import prepare_decoder_input, prepare_encoder_input
from .tst_encoder_layer import tst_encoder_layer

D_MODEL = 26
NUM_HEADS = 2
HEAD_DIM_TRUE = 13
HEAD_DIM_PADDED = 32
PADDED_WIDTH = NUM_HEADS * HEAD_DIM_PADDED  # 64

CONTEXT_LENGTH = 24
PREDICTION_LENGTH = 24
NUM_PARALLEL_SAMPLES = 100


def _pad_weight_per_head(W_out_in):
    W_in_out = W_out_in.T
    head_cols = W_in_out.split(HEAD_DIM_TRUE, dim=1)
    padded_heads = [F.pad(h, (0, HEAD_DIM_PADDED - HEAD_DIM_TRUE)) for h in head_cols]
    return torch.cat(padded_heads, dim=1)


def _pad_bias_per_head(b_out):
    head_chunks = b_out.split(HEAD_DIM_TRUE, dim=0)
    padded_heads = [F.pad(h, (0, HEAD_DIM_PADDED - HEAD_DIM_TRUE)) for h in head_chunks]
    return torch.cat(padded_heads, dim=0)


def _pad_input_per_head(W_in_out):
    head_rows = W_in_out.split(HEAD_DIM_TRUE, dim=0)
    padded_heads = [F.pad(h, (0, 0, 0, HEAD_DIM_PADDED - HEAD_DIM_TRUE)) for h in head_rows]
    return torch.cat(padded_heads, dim=0)


def _pad_input_dim(W_in_out, target_in=D_MODEL):
    pad_rows = PADDED_WIDTH - target_in
    if pad_rows <= 0:
        return W_in_out
    return F.pad(W_in_out, (0, 0, 0, pad_rows))


def _to_ttnn(t, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.bfloat16, layout=layout, device=device)


def _to_ttnn_int(t, device, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t.contiguous().to(torch.int32), dtype=ttnn.uint32, layout=layout, device=device)


def _build_fused_qkv(state, prefix, device):
    def padded_in_out(name):
        W = state[f"{prefix}.{name}.weight"].float()
        b = state[f"{prefix}.{name}.bias"].float()
        W_padded = _pad_weight_per_head(W)
        W_padded = _pad_input_dim(W_padded)
        b_padded = _pad_bias_per_head(b)
        return W_padded, b_padded

    Wq, bq = padded_in_out("q_proj")
    Wk, bk = padded_in_out("k_proj")
    Wv, bv = padded_in_out("v_proj")
    fused_w = torch.cat([Wq, Wk, Wv], dim=1)
    fused_b = torch.cat([bq, bk, bv], dim=0)
    return _to_ttnn(fused_w, device), _to_ttnn(fused_b, device)


def _build_cross_attn_weights(state, prefix, device):
    Wq = state[f"{prefix}.q_proj.weight"].float()
    bq = state[f"{prefix}.q_proj.bias"].float()
    Wq_padded = _pad_input_dim(_pad_weight_per_head(Wq))
    bq_padded = _pad_bias_per_head(bq)

    Wk = state[f"{prefix}.k_proj.weight"].float()
    bk = state[f"{prefix}.k_proj.bias"].float()
    Wv = state[f"{prefix}.v_proj.weight"].float()
    bv = state[f"{prefix}.v_proj.bias"].float()
    Wk_padded = _pad_input_dim(_pad_weight_per_head(Wk))
    Wv_padded = _pad_input_dim(_pad_weight_per_head(Wv))
    bk_padded = _pad_bias_per_head(bk)
    bv_padded = _pad_bias_per_head(bv)

    fused_kv_w = torch.cat([Wk_padded, Wv_padded], dim=1)
    fused_kv_b = torch.cat([bk_padded, bv_padded], dim=0)
    return (
        _to_ttnn(Wq_padded, device),
        _to_ttnn(bq_padded, device),
        _to_ttnn(fused_kv_w, device),
        _to_ttnn(fused_kv_b, device),
    )


def _build_out_proj(state, prefix, device):
    W = state[f"{prefix}.out_proj.weight"].float()
    b = state[f"{prefix}.out_proj.bias"].float()
    W_t = W.T
    W_input_padded = _pad_input_per_head(W_t)
    W_padded = F.pad(W_input_padded, (0, PADDED_WIDTH - D_MODEL))
    b_padded = F.pad(b, (0, PADDED_WIDTH - D_MODEL))
    return _to_ttnn(W_padded, device), _to_ttnn(b_padded, device)


def _build_ffn(state, prefix, device, ffn_dim):
    W1 = state[f"{prefix}.fc1.weight"].float()
    b1 = state[f"{prefix}.fc1.bias"].float()
    W1_padded = F.pad(W1.T, (0, 0, 0, PADDED_WIDTH - D_MODEL))

    W2 = state[f"{prefix}.fc2.weight"].float()
    b2 = state[f"{prefix}.fc2.bias"].float()
    W2_padded = F.pad(W2.T, (0, PADDED_WIDTH - D_MODEL))
    b2_padded = F.pad(b2, (0, PADDED_WIDTH - D_MODEL))
    return (
        _to_ttnn(W1_padded, device),
        _to_ttnn(b1, device),
        _to_ttnn(W2_padded, device),
        _to_ttnn(b2_padded, device),
    )


def _build_layer_norm(state, prefix, device):
    w = state[f"{prefix}.weight"].float()
    b = state[f"{prefix}.bias"].float()
    return _to_ttnn(w, device), _to_ttnn(b, device)


def _build_layer_norm_dict(state, prefix, device):
    w, b = _build_layer_norm(state, prefix, device)
    return {"weight": w, "bias": b}


def load_weights(hf_model, device):
    state = hf_model.state_dict()
    cfg = hf_model.config
    weights = {}

    for i in range(cfg.encoder_layers):
        prefix = f"model.encoder.layers.{i}"
        qkv_w, qkv_b = _build_fused_qkv(state, f"{prefix}.self_attn", device)
        out_w, out_b = _build_out_proj(state, f"{prefix}.self_attn", device)
        fc1_w, fc1_b, fc2_w, fc2_b = _build_ffn(state, prefix, device, cfg.encoder_ffn_dim)
        ln1_w, ln1_b = _build_layer_norm(state, f"{prefix}.self_attn_layer_norm", device)
        ln2_w, ln2_b = _build_layer_norm(state, f"{prefix}.final_layer_norm", device)
        weights[f"encoder.layers.{i}"] = {
            "qkv_weight": qkv_w,
            "qkv_bias": qkv_b,
            "out_proj_weight": out_w,
            "out_proj_bias": out_b,
            "fc1_weight": fc1_w,
            "fc1_bias": fc1_b,
            "fc2_weight": fc2_w,
            "fc2_bias": fc2_b,
            "self_attn_layer_norm_weight": ln1_w,
            "self_attn_layer_norm_bias": ln1_b,
            "final_layer_norm_weight": ln2_w,
            "final_layer_norm_bias": ln2_b,
        }

    for i in range(cfg.decoder_layers):
        prefix = f"model.decoder.layers.{i}"
        self_qkv_w, self_qkv_b = _build_fused_qkv(state, f"{prefix}.self_attn", device)
        self_out_w, self_out_b = _build_out_proj(state, f"{prefix}.self_attn", device)
        cross_q_w, cross_q_b, cross_kv_w, cross_kv_b = _build_cross_attn_weights(
            state, f"{prefix}.encoder_attn", device
        )
        cross_out_w, cross_out_b = _build_out_proj(state, f"{prefix}.encoder_attn", device)
        fc1_w, fc1_b, fc2_w, fc2_b = _build_ffn(state, prefix, device, cfg.decoder_ffn_dim)
        ln1_w, ln1_b = _build_layer_norm(state, f"{prefix}.self_attn_layer_norm", device)
        ln2_w, ln2_b = _build_layer_norm(state, f"{prefix}.encoder_attn_layer_norm", device)
        ln3_w, ln3_b = _build_layer_norm(state, f"{prefix}.final_layer_norm", device)
        weights[f"decoder.layers.{i}"] = {
            "self_attn": {
                "qkv_weight": self_qkv_w,
                "qkv_bias": self_qkv_b,
                "out_proj_weight": self_out_w,
                "out_proj_bias": self_out_b,
            },
            "encoder_attn": {
                "q_proj_weight": cross_q_w,
                "q_proj_bias": cross_q_b,
                "kv_weight": cross_kv_w,
                "kv_bias": cross_kv_b,
                "out_proj_weight": cross_out_w,
                "out_proj_bias": cross_out_b,
            },
            "fc1_weight": fc1_w,
            "fc1_bias": fc1_b,
            "fc2_weight": fc2_w,
            "fc2_bias": fc2_b,
            "self_attn_layer_norm_weight": ln1_w,
            "self_attn_layer_norm_bias": ln1_b,
            "encoder_attn_layer_norm_weight": ln2_w,
            "encoder_attn_layer_norm_bias": ln2_b,
            "final_layer_norm_weight": ln3_w,
            "final_layer_norm_bias": ln3_b,
        }

    weights["encoder_layernorm"] = _build_layer_norm_dict(state, "model.encoder.layernorm_embedding", device)
    weights["decoder_layernorm"] = _build_layer_norm_dict(state, "model.decoder.layernorm_embedding", device)

    enc_value_proj_t = state["model.encoder.value_embedding.value_projection.weight"].float().T.contiguous()
    dec_value_proj_t = state["model.decoder.value_embedding.value_projection.weight"].float().T.contiguous()
    weights["encoder_value_proj"] = _to_ttnn(enc_value_proj_t, device)
    weights["decoder_value_proj"] = _to_ttnn(dec_value_proj_t, device)
    weights["encoder_pos_emb"] = _to_ttnn(state["model.encoder.embed_positions.weight"].float(), device)
    weights["decoder_pos_emb"] = _to_ttnn(state["model.decoder.embed_positions.weight"].float(), device)
    weights["cat_embedder"] = _to_ttnn(state["model.embedder.embedders.0.weight"].float(), device)

    weights["dist_head"] = {
        "w0": state["parameter_projection.proj.0.weight"].float(),
        "b0": state["parameter_projection.proj.0.bias"].float(),
        "w1": state["parameter_projection.proj.1.weight"].float(),
        "b1": state["parameter_projection.proj.1.bias"].float(),
        "w2": state["parameter_projection.proj.2.weight"].float(),
        "b2": state["parameter_projection.proj.2.bias"].float(),
    }
    weights["dist_type"] = getattr(cfg, "distribution_output", "student_t")

    weights["encoder_layernorm_f32"] = {
        "weight": state["model.encoder.layernorm_embedding.weight"].float(),
        "bias": state["model.encoder.layernorm_embedding.bias"].float(),
    }
    weights["decoder_layernorm_f32"] = {
        "weight": state["model.decoder.layernorm_embedding.weight"].float(),
        "bias": state["model.decoder.layernorm_embedding.bias"].float(),
    }
    weights["encoder_layernorm_ttnn"] = {
        "weight": _to_ttnn(state["model.encoder.layernorm_embedding.weight"].float(), device),
        "bias": _to_ttnn(state["model.encoder.layernorm_embedding.bias"].float(), device),
    }
    weights["decoder_layernorm_ttnn"] = {
        "weight": _to_ttnn(state["model.decoder.layernorm_embedding.weight"].float(), device),
        "bias": _to_ttnn(state["model.decoder.layernorm_embedding.bias"].float(), device),
    }
    return weights


def _apply_layernorm_ttnn(x, ln_weights, orig_dim=D_MODEL):
    return ttnn.layer_norm(x, weight=ln_weights["weight"], bias=ln_weights["bias"])


def _distribution_head(hidden, weights):
    """hidden: torch [B, T, D_MODEL] -> distribution params tuple."""
    dh = weights["dist_head"]
    dt = weights.get("dist_type", "student_t")
    if dt == "normal":
        return normal_params(hidden, dh["w0"], dh["b0"], dh["w1"], dh["b1"])
    elif dt == "negative_binomial":
        return negative_binomial_params(hidden, dh["w0"], dh["b0"], dh["w1"], dh["b1"])
    else:
        return student_t_params(hidden, dh["w0"], dh["b0"], dh["w1"], dh["b1"], dh["w2"], dh["b2"])


def _sample_next_step(params, dist_type, _lc, _sc):
    """
    Shared sampling logic for generate() and generate_traced().
    Applies the second squeeze needed to collapse the leftover seq-len-1
    axis from slicing hidden as [:, -1:, :] before sampling.
    """
    if dist_type == "normal":
        loc_d, scale_d = params
        raw_loc = _lc + _sc * loc_d.squeeze(-1)
        raw_scale = _sc * scale_d.squeeze(-1)
        return sample_normal(raw_loc, raw_scale)
    elif dist_type == "negative_binomial":
        total_count, logits = params
        total_count = total_count.squeeze(-1)
        logits = logits.squeeze(-1)
        logits_scaled = logits + _sc.log()
        return sample_negative_binomial(total_count, logits_scaled)
    else:  # student_t
        df, loc_d, scale_d = params
        raw_loc = _lc + _sc * loc_d.squeeze(-1)
        raw_scale = _sc * scale_d.squeeze(-1)
        return sample_student_t(df.squeeze(-1), raw_loc, raw_scale)


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


def _inputs_to_ttnn(
    device, past_values, past_time_features, past_observed_mask, static_categorical_features, static_real_features
):
    pv = ttnn.from_torch(past_values.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    pt = ttnn.from_torch(past_time_features.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    pm = ttnn.from_torch(past_observed_mask.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    sc = ttnn.from_torch(
        static_categorical_features.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    sr = ttnn.from_torch(static_real_features.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return pv, pt, pm, sc, sr


def _future_time_to_ttnn(device, future_time_features):
    return ttnn.from_torch(
        future_time_features.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _past_values_repeated_to_ttnn(device, past_values_raw):
    return ttnn.from_torch(past_values_raw.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _future_vals_k_to_ttnn(device, future_vals_k):
    return ttnn.from_torch(future_vals_k.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


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
        mask_k = build_causal_mask(device, k + 1)
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


# ---------------------------------------------------------------------------
# Trace-once, replay-many decoder context.
#
# BUG THIS FIXES: the previous generate_traced() called begin_trace_capture()
# / end_trace_capture() / release_trace() ANEW on every single invocation,
# and also reallocated captured_dec_input, enc_hidden_rep, and
# precomputed_kv from scratch each call. The first call worked because the
# device started clean; the second call in the same process measurably
# hung indefinitely (reproduced on a freshly tt-smi-reset device -- not an
# environment artifact). The allocator's own warning
# ("Allocating device buffers is unsafe due to the existence of an active
# trace") is the proximate mechanism: release_trace() does not guarantee
# the device has fully reclaimed the previous trace's resources before the
# next begin_trace_capture()'s fresh allocations are issued.
#
# FIX: separate the one-time setup (encoder pass, KV precompute, buffer
# allocation, single trace capture) from the per-call autoregressive
# replay. A TracedDecoderContext is built once per (device, weights,
# input batch, S) combination and can be reused across many generate
# calls. generate_traced() keeps its original signature and behavior for
# a single call (it builds and tears down its own context internally if
# none is supplied), so existing call sites and tests are unaffected.
# ---------------------------------------------------------------------------
class TracedDecoderContext:
    """Holds everything needed to replay the captured decoder trace.

    Build once via build_traced_decoder_context(); call
    replay_traced_decoder(ctx, dec_emb_k, k) repeatedly; release via
    ctx.release() when fully done (or rely on generate_traced() to do
    this automatically when it owns the context).
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
    the trace reads from, and captures the decoder trace ONCE.

    Returns a TracedDecoderContext that can be fed into
    replay_traced_decoder() repeatedly (e.g. once per autoregressive step,
    across many generate() calls) without recapturing the trace.
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

    causal_mask_full = build_causal_mask(device, T_max)

    # Pre-allocate device buffer that the trace reads from.
    # Must be allocated BEFORE begin_trace_capture so it isn't captured
    # as a trace-internal allocation.
    captured_dec_input = ttnn.from_torch(
        torch.zeros(BS, T_max, PADDED_WIDTH, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Warmup run to compile kernels before capture (untraced).
    _ = _run_decoder_layers_fixed(captured_dec_input, weights, causal_mask_full, precomputed_kv)
    ttnn.synchronize_device(device)

    # Capture the trace EXACTLY ONCE. This is the fix: previously this
    # happened inside generate_traced() and therefore ran every call.
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
    Vectorized CPU lag extraction. seq_2d: [BS, full_len] float32.
    Returns [BS, subseq_len, len(lags)] float32.
    Replaces the Python double-for-loop with torch index ops -- O(len(lags))
    Python iterations regardless of subseq_len or BS.
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
    Full per-step decoder embedding on CPU.
    Returns [BS, T_max, PADDED_WIDTH] bfloat16 CPU tensor.

    Per tt-metal trace docs, no device ops may run between execute_trace
    replays. This moves lag extraction, value projection, positional
    embedding, and decoder input layernorm to CPU.
    Ref: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/
         AdvancedPerformanceOptimizationsForModels/
         AdvancedPerformanceOptimizationsForModels.md

    Core decoder compute (self-attn, cross-attn, FFN, internal layernorms)
    is unchanged inside the trace on-device.
    """
    import torch.nn.functional as F

    BS = past_values_cpu.shape[0]
    pred_len_k = k + 1

    if k == 0:
        future_vals_k = torch.zeros(BS, 1)
    else:
        prev = torch.stack(future_samples_so_far, dim=1)
        future_vals_k = torch.cat([prev, torch.zeros(BS, 1)], dim=1)

    full_seq = torch.cat([past_values_cpu, future_vals_k], dim=1)
    full_seq_scaled = (full_seq - loc_cpu.squeeze(-1)) / scale_cpu.squeeze(-1)

    _LAGS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37]
    lagged = _get_lagged_cpu(full_seq_scaled, _LAGS, pred_len_k)

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

    Per-step device ops are ONLY:
        copy_host_to_device_tensor -> execute_trace -> .cpu() readback
    All embedding prep runs on CPU before copy_host_to_device_tensor,
    per the tt-metal trace requirement that no other device ops run between
    execute_trace replays.

    Core decoder compute (self-attn, cross-attn, FFN, layernorm) stays
    inside the trace on-device unchanged.
    """
    device = ctx.device
    BS = ctx.BS
    T_max = ctx.T_max
    dt = ctx.dist_type

    # Pull CPU copies once before the loop
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
        # 1. CPU: full embedding prep for step k
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

        # 2. Device: write -> trace -> read  (ONLY device ops per step)
        step_host_tt = ttnn.from_torch(step_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)
        ttnn.copy_host_to_device_tensor(step_host_tt, ctx.captured_dec_input)
        ttnn.execute_trace(device, ctx.trace_id, cq_id=0, blocking=True)
        dec_out_torch = ttnn.to_torch(ctx.traced_out).float()[..., :D_MODEL]

        # 3. Host: distribution head + sample
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
    generate() with a TTNN trace over the decoder stack.

    Signature and single-call behavior are unchanged from before: call it
    exactly as before and it builds, uses, and releases its own trace
    context internally.

    NEW: pass a pre-built TracedDecoderContext via `traced_ctx` (from
    build_traced_decoder_context()) to reuse an already-captured trace
    across multiple calls -- this is the fix for the hang that occurred
    when calling generate_traced() repeatedly in the same process, since
    the trace is no longer captured and released on every call.
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

    causal_mask = build_causal_mask(device, prediction_length)
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
