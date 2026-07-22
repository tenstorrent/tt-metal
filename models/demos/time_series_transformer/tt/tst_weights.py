# tt/tst_weights.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
HF state_dict -> padded TTNN weights. Owns per-head padding, fused QKV
construction, and the full load_weights() entry point. No forward-pass
logic here — see tst_model.py for encoder/decoder execution.
"""

import torch
import torch.nn.functional as F

import ttnn

from .tst_config import D_MODEL, HEAD_DIM_PADDED, HEAD_DIM_TRUE, PADDED_WIDTH


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


def _to_ttnn(t, device, layout=ttnn.TILE_LAYOUT, memory_config=None):
    return ttnn.from_torch(
        t.contiguous().float(), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config
    )


def _to_ttnn_int(t, device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=None):
    return ttnn.from_torch(
        t.contiguous().to(torch.int32), dtype=ttnn.uint32, layout=layout, device=device, memory_config=memory_config
    )


def _build_fused_qkv(state, prefix, device, memory_config=None):
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
    return _to_ttnn(fused_w, device, memory_config=memory_config), _to_ttnn(
        fused_b, device, memory_config=memory_config
    )


def _build_cross_attn_weights(state, prefix, device, memory_config=None):
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
        _to_ttnn(Wq_padded, device, memory_config=memory_config),
        _to_ttnn(bq_padded, device, memory_config=memory_config),
        _to_ttnn(fused_kv_w, device, memory_config=memory_config),
        _to_ttnn(fused_kv_b, device, memory_config=memory_config),
    )


def _build_out_proj(state, prefix, device, memory_config=None):
    W = state[f"{prefix}.out_proj.weight"].float()
    b = state[f"{prefix}.out_proj.bias"].float()
    W_t = W.T
    W_input_padded = _pad_input_per_head(W_t)
    W_padded = F.pad(W_input_padded, (0, PADDED_WIDTH - D_MODEL))
    b_padded = F.pad(b, (0, PADDED_WIDTH - D_MODEL))
    return _to_ttnn(W_padded, device, memory_config=memory_config), _to_ttnn(
        b_padded, device, memory_config=memory_config
    )


def _build_ffn(state, prefix, device, ffn_dim, memory_config=None):
    W1 = state[f"{prefix}.fc1.weight"].float()
    b1 = state[f"{prefix}.fc1.bias"].float()
    W1_padded = F.pad(W1.T, (0, 0, 0, PADDED_WIDTH - D_MODEL))

    W2 = state[f"{prefix}.fc2.weight"].float()
    b2 = state[f"{prefix}.fc2.bias"].float()
    W2_padded = F.pad(W2.T, (0, PADDED_WIDTH - D_MODEL))
    b2_padded = F.pad(b2, (0, PADDED_WIDTH - D_MODEL))
    return (
        _to_ttnn(W1_padded, device, memory_config=memory_config),
        _to_ttnn(b1, device, memory_config=memory_config),
        _to_ttnn(W2_padded, device, memory_config=memory_config),
        _to_ttnn(b2_padded, device, memory_config=memory_config),
    )


def _build_layer_norm(state, prefix, device, memory_config=None):
    w = state[f"{prefix}.weight"].float()
    b = state[f"{prefix}.bias"].float()
    return _to_ttnn(w, device, memory_config=memory_config), _to_ttnn(b, device, memory_config=memory_config)


def _build_layer_norm_dict(state, prefix, device):
    w, b = _build_layer_norm(state, prefix, device)
    return {"weight": w, "bias": b}


def load_weights(hf_model, device, hot_path_memory_config=None):
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
        self_qkv_w, self_qkv_b = _build_fused_qkv(
            state, f"{prefix}.self_attn", device, memory_config=hot_path_memory_config
        )
        self_out_w, self_out_b = _build_out_proj(
            state, f"{prefix}.self_attn", device, memory_config=hot_path_memory_config
        )
        cross_q_w, cross_q_b, cross_kv_w, cross_kv_b = _build_cross_attn_weights(
            state, f"{prefix}.encoder_attn", device, memory_config=hot_path_memory_config
        )
        cross_out_w, cross_out_b = _build_out_proj(
            state, f"{prefix}.encoder_attn", device, memory_config=hot_path_memory_config
        )
        fc1_w, fc1_b, fc2_w, fc2_b = _build_ffn(
            state, prefix, device, cfg.decoder_ffn_dim, memory_config=hot_path_memory_config
        )
        ln1_w, ln1_b = _build_layer_norm(
            state, f"{prefix}.self_attn_layer_norm", device, memory_config=hot_path_memory_config
        )
        ln2_w, ln2_b = _build_layer_norm(
            state, f"{prefix}.encoder_attn_layer_norm", device, memory_config=hot_path_memory_config
        )
        ln3_w, ln3_b = _build_layer_norm(
            state, f"{prefix}.final_layer_norm", device, memory_config=hot_path_memory_config
        )
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

    weights["dist_head_ttnn"] = {
        "w0": _to_ttnn(state["parameter_projection.proj.0.weight"].float().T.contiguous(), device),
        "b0": _to_ttnn(state["parameter_projection.proj.0.bias"].float(), device),
        "w1": _to_ttnn(state["parameter_projection.proj.1.weight"].float().T.contiguous(), device),
        "b1": _to_ttnn(state["parameter_projection.proj.1.bias"].float(), device),
        "w2": _to_ttnn(state["parameter_projection.proj.2.weight"].float().T.contiguous(), device),
        "b2": _to_ttnn(state["parameter_projection.proj.2.bias"].float(), device),
    }

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
