# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Reference validation: run the HF DeepSeek-V3 model on CPU and compare
hidden-state outputs against the TT pipeline, position by position.

Usage from the demo CLI:
    python cli.py --prompt "who are you" --validate --hf-model-path /path/to/DeepSeek-V3

The reference runs embedding + N decoder layers (no final RMSNorm) to match
the pipeline output when the LM head stage has been replaced by passthrough.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

HIDDEN_SIZE = 7168
FIRST_K_DENSE_REPLACE = 3


def generate_reference_hidden_states(
    token_ids: list[int],
    hf_model_path: str | Path,
    num_layers: int = 12,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
) -> list[torch.Tensor]:
    """
    Run HF reference DeepSeek-V3 through embedding + *num_layers* decoder layers.

    Returns per-position hidden states **before** the final RMSNorm, as a list
    of ``(hidden_size,)`` bf16 tensors (one per input token).

    Layer-ID overrides mirror the pipeline CLI flags:
      - dense_layer_id_override: all dense layers (0..2) use this layer's weights
      - moe_layer_id_override:   all MoE layers  (3+)  use this layer's weights
    """
    from transformers import AutoConfig
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3DecoderLayer
    from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
    from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

    hf_model_path = Path(hf_model_path)

    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    config.num_hidden_layers = num_layers

    lazy_sd = LazyStateDict(hf_model_path)

    # --- embedding ---
    logger.info("Loading embedding weights …")
    embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    embed_tokens.weight.data = lazy_sd["model.embed_tokens.weight"].to(torch.bfloat16)
    embed_tokens = embed_tokens.eval()

    # --- decoder layers ---
    # Cache dequantized weight dicts so that layers sharing the same weight_layer_id
    # are dequantized only once (each nn.Module still gets its own parameter copy).
    _deq_cache: dict[int, dict[str, torch.Tensor]] = {}

    layers: list[DeepseekV3DecoderLayer] = []
    for i in range(num_layers):
        is_moe = i >= config.first_k_dense_replace and i % config.moe_layer_freq == 0

        if is_moe and moe_layer_id_override is not None:
            weight_layer_id = moe_layer_id_override
        elif not is_moe and dense_layer_id_override is not None:
            weight_layer_id = dense_layer_id_override
        else:
            weight_layer_id = i

        logger.info(
            "Loading decoder layer {} (weights from layer {}, {}) …",
            i,
            weight_layer_id,
            "MoE" if is_moe else "Dense",
        )

        layer = DeepseekV3DecoderLayer(config, layer_idx=i).eval().to(torch.bfloat16)

        if weight_layer_id not in _deq_cache:
            layer_sd = sub_state_dict(lazy_sd, f"model.layers.{weight_layer_id}.")
            _deq_cache[weight_layer_id] = dequantize_state_dict(layer_sd, config)

        layer.load_state_dict(_deq_cache[weight_layer_id], strict=True)
        layers.append(layer)

    # --- forward pass ---
    logger.info("Running reference forward ({} tokens, {} layers) …", len(token_ids), num_layers)
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        hidden_states = embed_tokens(input_ids)  # (1, S, H)

        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        attention_mask = _prepare_4d_causal_attention_mask(None, (batch_size, seq_length), hidden_states, 0)

        for i, layer in enumerate(layers):
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_output[0]
            logger.debug(
                "Layer {} output: mean={:.6f}, std={:.6f}",
                i,
                hidden_states.mean().item(),
                hidden_states.std().item(),
            )

    # (1, S, H) → list of S tensors each (H,)
    per_position = [hidden_states[0, t, :] for t in range(seq_length)]
    logger.info("Reference forward complete — {} position outputs.", len(per_position))
    return per_position


def host_lm_head_sample(
    hidden_state: torch.Tensor,
    hf_model_path: str | Path,
    tokenizer=None,
    *,
    hidden_dim: int = HIDDEN_SIZE,
) -> dict:
    """
    Run final RMSNorm + LM head matmul + argmax on a single hidden state on host.

    *hidden_state* is a raw pipeline output tensor (may need bfloat16 reinterpretation).
    Returns a dict with the sampled token ID, the decoded text, and top-5 logits.
    """
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

    hf_model_path = Path(hf_model_path)
    lazy_sd = LazyStateDict(hf_model_path)

    pipe_flat = hidden_state.flatten()
    if pipe_flat.numel() != hidden_dim:
        pipe_flat = pipe_flat.view(torch.bfloat16)
    h = pipe_flat[:hidden_dim].to(torch.float32)

    norm_weight = lazy_sd["model.norm.weight"].to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h_normed = h * torch.rsqrt(variance + 1e-6) * norm_weight

    lm_head_weight = lazy_sd["lm_head.weight"].to(torch.float32)  # (vocab, hidden)
    logits = h_normed @ lm_head_weight.T  # (vocab,)

    top5_vals, top5_ids = torch.topk(logits, 5)
    sampled_id = int(top5_ids[0].item())

    result = {
        "token_id": sampled_id,
        "top5_ids": top5_ids.tolist(),
        "top5_logits": top5_vals.tolist(),
    }

    if tokenizer is not None:
        result["token_text"] = tokenizer.decode([sampled_id])
        result["top5_texts"] = [tokenizer.decode([tid]) for tid in top5_ids.tolist()]

    logger.info(
        "Host LM head sample: token_id={} top5_ids={} top5_logits=[{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]",
        sampled_id,
        top5_ids.tolist(),
        *top5_vals.tolist(),
    )
    if tokenizer is not None:
        logger.info("Decoded: '{}' | Top-5: {}", result["token_text"], result["top5_texts"])

    return result


def compare_hidden_states(
    pipeline_output: torch.Tensor,
    reference_output: torch.Tensor,
    position_idx: int,
    *,
    hidden_dim: int = HIDDEN_SIZE,
) -> dict[str, float]:
    """
    Compare a single pipeline output against the reference.

    The pipeline D2H socket writes bfloat16 activations, but the host output
    buffer may declare a different dtype (e.g. float32). When the element count
    doesn't match ``hidden_dim``, the raw bytes are reinterpreted as bfloat16.

    Returns a dict with PCC, cosine similarity, and max abs error.
    """
    pipe_flat = pipeline_output.flatten()
    if pipe_flat.numel() != hidden_dim:
        pipe_flat = pipe_flat.view(torch.bfloat16)
    pipe = pipe_flat[:hidden_dim].to(torch.float32)
    ref = reference_output.flatten()[:hidden_dim].to(torch.float32)

    cos_sim = torch.nn.functional.cosine_similarity(pipe.unsqueeze(0), ref.unsqueeze(0)).item()

    stacked = torch.stack([pipe, ref])
    pcc = torch.corrcoef(stacked)[0, 1].item()

    max_abs_err = (pipe - ref).abs().max().item()

    pipe_mag = torch.norm(pipe).item()
    ref_mag = torch.norm(ref).item()

    logger.info(
        "Position {:>3d} — PCC={:.6f}  CosSim={:.6f}  MaxAbsErr={:.6f}  PipeMag={:.6f}  RefMag={:.6f}",
        position_idx,
        pcc,
        cos_sim,
        max_abs_err,
        pipe_mag,
        ref_mag,
    )
    logger.info("Position {:>3d} — Pipeline[:64]: {}", position_idx, pipe[:64].tolist())
    logger.info("Position {:>3d} — Reference[:64]: {}", position_idx, ref[:64].tolist())
    return {"pcc": pcc, "cos_sim": cos_sim, "max_abs_err": max_abs_err, "pipe_mag": pipe_mag, "ref_mag": ref_mag}
