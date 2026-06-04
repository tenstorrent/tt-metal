# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import torch
import torch.nn.functional as F
import ttnn
from .tst_encoder_layer import tst_encoder_layer
from .tst_decoder_layer import tst_decoder_layer
from .ttnn_utils import layer_norm_padded
from .tst_embedding import prepare_encoder_input, prepare_decoder_input
from .tst_distribution import student_t_params, sample_student_t

D_MODEL = 26
CONTEXT_LENGTH = 24
PREDICTION_LENGTH = 24
NUM_PARALLEL_SAMPLES = 100


def load_weights(hf_model, device):
    state = hf_model.state_dict()
    weights = {}

    def to_ttnn(key, transpose=False):
        t = state[key].float()
        if transpose:
            t = t.T.contiguous()
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def to_ttnn_1d(key):
        t = state[key].float()
        if t.shape[-1] % 32 != 0:
            pad_size = 32 - (t.shape[-1] % 32)
            t = F.pad(t, (0, pad_size))
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def to_ttnn_bias(key):
        t = state[key].float()
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    for i in range(2):
        prefix = f"model.encoder.layers.{i}"
        out_prefix = f"encoder.layers.{i}"
        weights[out_prefix] = {
            "self_attn.q_proj.weight": to_ttnn(f"{prefix}.self_attn.q_proj.weight"),
            "self_attn.k_proj.weight": to_ttnn(f"{prefix}.self_attn.k_proj.weight"),
            "self_attn.v_proj.weight": to_ttnn(f"{prefix}.self_attn.v_proj.weight"),
            "self_attn.out_proj.weight": to_ttnn(f"{prefix}.self_attn.out_proj.weight"),
            "self_attn.q_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.q_proj.bias"),
            "self_attn.k_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.k_proj.bias"),
            "self_attn.v_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.v_proj.bias"),
            "self_attn.out_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.out_proj.bias"),
            "self_attn_layer_norm.weight": to_ttnn_1d(f"{prefix}.self_attn_layer_norm.weight"),
            "self_attn_layer_norm.bias": to_ttnn_1d(f"{prefix}.self_attn_layer_norm.bias"),
            "fc1.weight": to_ttnn(f"{prefix}.fc1.weight"),
            "fc1.bias": to_ttnn_bias(f"{prefix}.fc1.bias"),
            "fc2.weight": to_ttnn(f"{prefix}.fc2.weight"),
            "fc2.bias": to_ttnn_bias(f"{prefix}.fc2.bias"),
            "final_layer_norm.weight": to_ttnn_1d(f"{prefix}.final_layer_norm.weight"),
            "final_layer_norm.bias": to_ttnn_1d(f"{prefix}.final_layer_norm.bias"),
        }

    for i in range(2):
        prefix = f"model.decoder.layers.{i}"
        out_prefix = f"decoder.layers.{i}"
        weights[out_prefix] = {
            "self_attn.q_proj.weight": to_ttnn(f"{prefix}.self_attn.q_proj.weight"),
            "self_attn.k_proj.weight": to_ttnn(f"{prefix}.self_attn.k_proj.weight"),
            "self_attn.v_proj.weight": to_ttnn(f"{prefix}.self_attn.v_proj.weight"),
            "self_attn.out_proj.weight": to_ttnn(f"{prefix}.self_attn.out_proj.weight"),
            "self_attn.q_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.q_proj.bias"),
            "self_attn.k_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.k_proj.bias"),
            "self_attn.v_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.v_proj.bias"),
            "self_attn.out_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.out_proj.bias"),
            "self_attn_layer_norm.weight": to_ttnn_1d(f"{prefix}.self_attn_layer_norm.weight"),
            "self_attn_layer_norm.bias": to_ttnn_1d(f"{prefix}.self_attn_layer_norm.bias"),
            "encoder_attn.q_proj.weight": to_ttnn(f"{prefix}.encoder_attn.q_proj.weight"),
            "encoder_attn.k_proj.weight": to_ttnn(f"{prefix}.encoder_attn.k_proj.weight"),
            "encoder_attn.v_proj.weight": to_ttnn(f"{prefix}.encoder_attn.v_proj.weight"),
            "encoder_attn.out_proj.weight": to_ttnn(f"{prefix}.encoder_attn.out_proj.weight"),
            "encoder_attn.q_proj.bias": to_ttnn_bias(f"{prefix}.encoder_attn.q_proj.bias"),
            "encoder_attn.k_proj.bias": to_ttnn_bias(f"{prefix}.encoder_attn.k_proj.bias"),
            "encoder_attn.v_proj.bias": to_ttnn_bias(f"{prefix}.encoder_attn.v_proj.bias"),
            "encoder_attn.out_proj.bias": to_ttnn_bias(f"{prefix}.encoder_attn.out_proj.bias"),
            "encoder_attn_layer_norm.weight": to_ttnn_1d(f"{prefix}.encoder_attn_layer_norm.weight"),
            "encoder_attn_layer_norm.bias": to_ttnn_1d(f"{prefix}.encoder_attn_layer_norm.bias"),
            "fc1.weight": to_ttnn(f"{prefix}.fc1.weight"),
            "fc1.bias": to_ttnn_bias(f"{prefix}.fc1.bias"),
            "fc2.weight": to_ttnn(f"{prefix}.fc2.weight"),
            "fc2.bias": to_ttnn_bias(f"{prefix}.fc2.bias"),
            "final_layer_norm.weight": to_ttnn_1d(f"{prefix}.final_layer_norm.weight"),
            "final_layer_norm.bias": to_ttnn_1d(f"{prefix}.final_layer_norm.bias"),
        }

    weights["encoder_layernorm"] = {
        "weight": to_ttnn_1d("model.encoder.layernorm_embedding.weight"),
        "bias":   to_ttnn_1d("model.encoder.layernorm_embedding.bias"),
    }
    weights["decoder_layernorm"] = {
        "weight": to_ttnn_1d("model.decoder.layernorm_embedding.weight"),
        "bias":   to_ttnn_1d("model.decoder.layernorm_embedding.bias"),
    }

    weights["encoder_value_proj"] = state["model.encoder.value_embedding.value_projection.weight"].float()
    weights["decoder_value_proj"] = state["model.decoder.value_embedding.value_projection.weight"].float()
    weights["encoder_pos_emb"]    = state["model.encoder.embed_positions.weight"].float()
    weights["decoder_pos_emb"]    = state["model.decoder.embed_positions.weight"].float()
    weights["cat_embedder"]       = state["model.embedder.embedders.0.weight"].float()

    weights["dist_head"] = {
        "w0": state["parameter_projection.proj.0.weight"].float(),
        "b0": state["parameter_projection.proj.0.bias"].float(),
        "w1": state["parameter_projection.proj.1.weight"].float(),
        "b1": state["parameter_projection.proj.1.bias"].float(),
        "w2": state["parameter_projection.proj.2.weight"].float(),
        "b2": state["parameter_projection.proj.2.bias"].float(),
    }

    weights["encoder_layernorm_f32"] = {
        "weight": state["model.encoder.layernorm_embedding.weight"].float(),
        "bias":   state["model.encoder.layernorm_embedding.bias"].float(),
    }
    weights["decoder_layernorm_f32"] = {
        "weight": state["model.decoder.layernorm_embedding.weight"].float(),
        "bias":   state["model.decoder.layernorm_embedding.bias"].float(),
    }

    return weights


def _apply_layernorm_f32(x, ln_weights):
    return F.layer_norm(x, [x.shape[-1]], ln_weights["weight"], ln_weights["bias"])


def _distribution_head(hidden, weights):
    """hidden: [B, T, D_MODEL] -> (df, loc, scale) each [B, T]"""
    from .tst_distribution import student_t_params
    dh = weights["dist_head"]
    return student_t_params(hidden, dh["w0"], dh["b0"], dh["w1"], dh["b1"], dh["w2"], dh["b2"])


def run_encoder(device, encoder_input, weights, apply_layernorm=False):
    h = encoder_input.float()
    if apply_layernorm:
        h = _apply_layernorm_f32(h, weights["encoder_layernorm_f32"])
    hidden = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for i in range(2):
        hidden = tst_encoder_layer(device, hidden, weights, layer_idx=i)
    return hidden


def run_decoder_step(device, decoder_input, encoder_hidden, weights, apply_layernorm=False):
    h = decoder_input.float()
    if apply_layernorm:
        h = _apply_layernorm_f32(h, weights["decoder_layernorm_f32"])
    hidden = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for i in range(2):
        hidden = tst_decoder_layer(device, hidden, encoder_hidden, weights, layer_idx=i)
    return hidden


@torch.no_grad()
def generate(
    device, weights,
    past_values, past_time_features, future_time_features,
    past_observed_mask, static_categorical_features, static_real_features,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    num_parallel_samples=NUM_PARALLEL_SAMPLES,
):
    """Generate samples. Uses prepare_decoder_input (value embedding path) with
    correct autoregressive feedback of sampled values."""
    from .tst_embedding import prepare_encoder_input, prepare_decoder_input

    B = past_values.shape[0]
    S = num_parallel_samples
    device_cpu = past_values.device

    # --- Encoder ---
    enc_emb, loc, scale = prepare_encoder_input(
        past_values=past_values,
        past_time_features=past_time_features,
        past_observed_mask=past_observed_mask,
        static_cat_features=static_categorical_features,
        static_real_features=static_real_features,
        cat_embedder_weight=weights["cat_embedder"],
        value_proj_weight=weights["encoder_value_proj"],
        pos_emb_weight=weights["encoder_pos_emb"],
        context_length=context_length,
    )
    enc_emb = _apply_layernorm_f32(enc_emb, weights["encoder_layernorm_f32"])
    encoder_hidden = run_encoder(device, enc_emb, weights)  # [B, T, D_pad]

    # --- Repeat for parallel samples ---
    repeated_loc   = loc.repeat_interleave(S, dim=0)    # [B*S, 1, 1]
    repeated_scale = scale.repeat_interleave(S, dim=0)  # [B*S, 1, 1]
    _sc = repeated_scale.squeeze(-1).squeeze(-1)        # [B*S]
    _lc = repeated_loc.squeeze(-1).squeeze(-1)          # [B*S]

    # Normalized past values for lag extraction
    repeated_past_values_norm = (
        past_values.repeat_interleave(S, dim=0).float() - _lc.unsqueeze(-1)
    ) / _sc.unsqueeze(-1)  # [B*S, past_len]

    repeated_future_time = future_time_features.repeat_interleave(S, dim=0)
    repeated_static_cat  = static_categorical_features.repeat_interleave(S, dim=0)
    repeated_static_real = static_real_features.repeat_interleave(S, dim=0)

    enc_hidden_t = ttnn.to_torch(encoder_hidden).float()
    enc_hidden_rep = ttnn.from_torch(
        enc_hidden_t.repeat_interleave(S, dim=0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )  # [B*S, T, D_pad]

    future_samples = []  # raw-space samples

    for k in range(prediction_length):
        # Build future_values for decoder embedding:
        # positions 0..k-1 = previous raw samples (will be re-normalized inside prepare_decoder_input)
        # position k = zero placeholder
        if k == 0:
            future_vals_k = torch.zeros(B * S, 1, device=device_cpu)
        else:
            prev_raw = torch.stack(future_samples, dim=1)  # [B*S, k] raw space
            future_vals_k = torch.cat(
                [prev_raw, torch.zeros(B * S, 1, device=device_cpu)], dim=1
            )  # [B*S, k+1]

        # Unscale past to raw space for prepare_decoder_input (it will re-normalize internally)
        repeated_past_raw = repeated_past_values_norm * _sc.unsqueeze(-1) + _lc.unsqueeze(-1)

        dec_emb_k = prepare_decoder_input(
            future_values=future_vals_k,
            future_time_features=repeated_future_time[:, :k+1, :],
            past_values=repeated_past_raw,
            loc=repeated_loc,
            scale=repeated_scale,
            static_cat_features=repeated_static_cat,
            static_real_features=repeated_static_real,
            cat_embedder_weight=weights["cat_embedder"],
            value_proj_weight=weights["decoder_value_proj"],
            pos_emb_weight=weights["decoder_pos_emb"],
            context_length=context_length,
        )
        dec_emb_k = _apply_layernorm_f32(dec_emb_k, weights["decoder_layernorm_f32"])

        dec_out = run_decoder_step(device, dec_emb_k, enc_hidden_rep, weights, apply_layernorm=False)
        dec_out_torch = ttnn.to_torch(dec_out).float()[..., :D_MODEL]  # [B*S, k+1, D]

        df_k, loc_d_k, scale_d_k = _distribution_head(dec_out_torch[:, -1:, :], weights)

        raw_loc_k   = _lc + _sc * loc_d_k.squeeze(-1)
        raw_scale_k = _sc * scale_d_k.squeeze(-1)

        from torch.distributions import StudentT
        next_sample = StudentT(df=df_k.squeeze(-1), loc=raw_loc_k, scale=raw_scale_k).sample()
        future_samples.append(next_sample)

        # Update normalized past with new sample
        next_norm = (next_sample - _lc) / _sc
    samples = torch.stack(future_samples, dim=1)  # [B*S, pred_len]
    return samples.reshape(B, S, prediction_length)

