# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttml -> tt-transformers weight-export bridge for Llama and Qwen3.

:func:`weights_ref_hf_dict` exports a ttml model's parameters as an HF-keyed dict of
on-device ``ttnn.Tensor`` handles, the wire format consumed by tt-transformers'
``Transformer.update_weights(hf_state_dict, hf_rope=False)``.

Qwen3: both stacks store Q/K rows in the same interleaved (Meta-permute-equivalent)
layout:

  - ttml's ``load_weights_from_hf`` runs :func:`unpermute_proj_rows` on
    ``q_proj`` / ``k_proj`` and :func:`unpermute_norm_weights` on ``q_norm`` /
    ``k_norm`` (see ``ttml.models.qwen3.weights``).
  - tt-transformers' ``convert_hf_qkv_to_meta_format`` runs :func:`reverse_permute`
    on ``q_proj`` / ``k_proj`` and :func:`reverse_permute_1d` on ``q_norm`` /
    ``k_norm`` (see ``models/tt_transformers/tt/load_checkpoints.py``).

Both produce the same interleaved rows ``[r0, i0, r1, i1, ...]`` per head. So a
ttml Qwen3 parameter can be handed straight to tt-transformers'
``Transformer.update_weights(hf_rope=False)`` under its HF key -- no additional
permutation. The only shape rewrite is splitting ttml's fused ``kv_proj`` back
into HF's separate ``k_proj`` / ``v_proj``.

The Qwen3 dict is layout-identical to the Llama one (bf16, TILE, DRAM-interleaved,
replicated, HF Linear shapes wrapped in two leading unit dims), so the same weight
transport accepts it without changes.
"""

from __future__ import annotations

from typing import Any, Optional

import ttnn

from ttml.models import WeightTyingType
from ttml.models.llama import Llama
from ttml.models.qwen3 import Qwen3


def weights_ref_hf_dict(model) -> dict[str, ttnn.Tensor]:
    """HF-keyed on-device export of a ttml Llama or Qwen3 for tt-transformers'
    ``Transformer.update_weights(hf_state_dict, hf_rope=False)``."""
    if isinstance(model, Qwen3):
        return _qwen3_weights_ref_hf_dict(model)
    if isinstance(model, Llama):
        return _llama_weights_ref_hf_dict(model)
    raise TypeError(f"weights_ref_hf_dict supports ttml Llama and Qwen3 models, got {type(model).__name__}")


def _llama_weights_ref_hf_dict(model) -> dict[str, ttnn.Tensor]:
    """Export this ttml model's parameters as an HF-keyed dict of on-device
    ``ttnn.Tensor`` handles, shaped for tt-transformers'
    ``Transformer.update_weights(hf_state_dict, hf_rope=False)`` (HF
    safetensors dot-keys; HF shapes wrapped in two leading unit dims;
    bf16, TILE, DRAM-interleaved, replicated).

    Q/K row order: both ttml and TTT store Meta-permuted rows for
    Llama-3.2-1B, so the consumer uses ``hf_rope=False`` (no permutation).

    Tied embeddings: with ``weight_tying=Enabled``, ``embed_tokens`` and
    ``lm_head`` point at the same handle; safe because the consumer
    ``ttnn.copy``s into a separate destination and never aliases the source.

    Most values are live handles into ttml's parameter store; do not mutate
    ttml's parameters between this call and ``update_weights``. The K/V split
    is the exception: ttml fuses K and V into one ``kv_linear/weight``
    (K rows first, then V), so we expose them via two ``ttnn.slice`` calls
    (newly allocated, ~64 MB total for Llama-3.2-1B-Instruct).

    Single-device assumption: parameters must be replicated across the mesh
    (no DDP/TP shard mapper). The grpo single-device config satisfies this;
    DDP/TP would need a host-side per-parameter concat first.
    """
    cfg = model.config
    assert cfg.weight_tying == WeightTyingType.Enabled, (
        "weights_ref_hf_dict requires weight_tying=Enabled (Llama-3.2-1B/-Instruct "
        f"tie embed_tokens and lm_head). Got weight_tying={cfg.weight_tying!r}."
    )

    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    H = cfg.hidden_size
    head_dim = H // n_heads
    kv_dim = n_kv * head_dim

    params = model.parameters()

    def get(name: str) -> ttnn.Tensor:
        if name not in params:
            raise RuntimeError(
                f"ttml parameter {name!r} not found; available keys (first 10): " f"{sorted(params.keys())[:10]}"
            )
        return params[name].get_value()

    out: dict[str, ttnn.Tensor] = {}

    # Tied: same handle exposed under both HF keys.
    fc = get("Llama/fc/weight")
    out["model.embed_tokens.weight"] = fc
    out["lm_head.weight"] = fc
    out["model.norm.weight"] = get("Llama/ln_fc/gamma")

    for i in range(len(model.blocks)):
        p = f"Llama/blocks/{i}"

        out[f"model.layers.{i}.input_layernorm.weight"] = get(f"{p}/attention_norm/gamma")
        out[f"model.layers.{i}.post_attention_layernorm.weight"] = get(f"{p}/mlp_norm/gamma")

        out[f"model.layers.{i}.self_attn.q_proj.weight"] = get(f"{p}/attention/q_linear/weight")
        out[f"model.layers.{i}.self_attn.o_proj.weight"] = get(f"{p}/attention/out_linear/weight")

        kv = get(f"{p}/attention/kv_linear/weight")
        kv_shape = tuple(kv.shape)
        assert kv_shape == (1, 1, 2 * kv_dim, H), (
            f"kv_linear shape mismatch at layer {i}: got {kv_shape}, " f"expected (1, 1, {2 * kv_dim}, {H})"
        )
        out[f"model.layers.{i}.self_attn.k_proj.weight"] = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, kv_dim, H])
        out[f"model.layers.{i}.self_attn.v_proj.weight"] = ttnn.slice(kv, [0, 0, kv_dim, 0], [1, 1, 2 * kv_dim, H])

        out[f"model.layers.{i}.mlp.gate_proj.weight"] = get(f"{p}/mlp/w1/weight")
        out[f"model.layers.{i}.mlp.up_proj.weight"] = get(f"{p}/mlp/w3/weight")
        out[f"model.layers.{i}.mlp.down_proj.weight"] = get(f"{p}/mlp/w2/weight")

    return out


def _qwen3_weights_ref_hf_dict(qwen3_model: Any, tie_word_embeddings: Optional[bool] = None) -> dict[str, ttnn.Tensor]:
    """Export a ttml Qwen3 model's parameters as an HF-keyed dict of on-device
    ``ttnn.Tensor`` handles, shaped for tt-transformers'
    ``Transformer.update_weights(hf_state_dict, hf_rope=False)``.

    Row layouts (already matching between ttml and tt-transformers):

    - ``q_proj`` / ``k_proj``: interleaved ``[r0, i0, r1, i1, ...]`` per head
      -- ttml's ``unpermute_proj_rows`` produces the same output as
      tt-transformers' ``reverse_permute``.
    - ``q_norm`` / ``k_norm`` (Qwen3 QK-Norm gammas): interleaved on head_dim --
      ttml's ``unpermute_norm_weights`` matches tt-transformers'
      ``reverse_permute_1d``.

    Fused ``kv_proj`` (ttml, K rows then V rows) is split via two ``ttnn.slice``
    calls into HF's ``k_proj`` (rows ``[0, kv_dim)``) and ``v_proj`` (rows
    ``[kv_dim, 2*kv_dim)``). Those two slices are newly allocated (a copy of
    ``kv_proj``'s data on each call); every other exported handle is a live
    view into the ttml parameter store -- do not mutate ttml's parameters
    between this call and ``update_weights``.

    Tied embeddings (Qwen3-0.6B): ``model.embed_tokens.weight`` and
    ``lm_head.weight`` expose the same underlying ``fc/weight`` handle; the
    consumer ``ttnn.copy``s into a separate destination and never aliases.

    Single-device / replicated only: with DDP the params stay replicated on the
    mesh, which is fine; a TP shard mapper on the parameters is not.

    Args:
        qwen3_model: A ttml ``Qwen3`` instance (from ``ttml.models.qwen3``).
        tie_word_embeddings: Override for the tying flag. If ``None``, read
            from ``qwen3_model.config.weight_tying``.

    Returns:
        HF-keyed dict of live on-device ``ttnn.Tensor`` handles.
    """
    cfg = qwen3_model.config

    if tie_word_embeddings is None:
        tie = cfg.weight_tying == WeightTyingType.Enabled
    else:
        tie = bool(tie_word_embeddings)

    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    kv_dim = n_kv * head_dim
    H = cfg.hidden_size

    params = qwen3_model.parameters()
    # ttml's ``load_weights_from_hf`` reads the root prefix the same way; keep parity.
    any_key = next(iter(params))
    root_prefix = any_key.split("/")[0]

    def get(name: str) -> ttnn.Tensor:
        if name not in params:
            raise RuntimeError(
                f"ttml parameter {name!r} not found; available keys (first 10): {sorted(params.keys())[:10]}"
            )
        return params[name].get_value()

    out: dict[str, ttnn.Tensor] = {}

    fc = get(f"{root_prefix}/fc/weight")
    if tie:
        out["model.embed_tokens.weight"] = fc
        out["lm_head.weight"] = fc
    else:
        out["model.embed_tokens.weight"] = get(f"{root_prefix}/tok_emb/weight")
        out["lm_head.weight"] = fc

    out["model.norm.weight"] = get(f"{root_prefix}/ln_fc/weight")

    for i in range(cfg.num_hidden_layers):
        tp = f"{root_prefix}/blocks/{i}"
        hp = f"model.layers.{i}"

        out[f"{hp}.self_attn.q_proj.weight"] = get(f"{tp}/self_attn/q_proj/weight")
        out[f"{hp}.self_attn.o_proj.weight"] = get(f"{tp}/self_attn/o_proj/weight")
        out[f"{hp}.self_attn.q_norm.weight"] = get(f"{tp}/self_attn/q_norm/weight")
        out[f"{hp}.self_attn.k_norm.weight"] = get(f"{tp}/self_attn/k_norm/weight")

        kv = get(f"{tp}/self_attn/kv_proj/weight")
        kv_shape = tuple(kv.shape)
        assert kv_shape == (
            1,
            1,
            2 * kv_dim,
            H,
        ), f"kv_proj shape mismatch at layer {i}: got {kv_shape}, expected (1, 1, {2 * kv_dim}, {H})"
        out[f"{hp}.self_attn.k_proj.weight"] = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, kv_dim, H])
        out[f"{hp}.self_attn.v_proj.weight"] = ttnn.slice(kv, [0, 0, kv_dim, 0], [1, 1, 2 * kv_dim, H])

        out[f"{hp}.input_layernorm.weight"] = get(f"{tp}/input_layernorm/weight")
        out[f"{hp}.post_attention_layernorm.weight"] = get(f"{tp}/post_attention_layernorm/weight")

        out[f"{hp}.mlp.gate_proj.weight"] = get(f"{tp}/mlp/gate_proj/weight")
        out[f"{hp}.mlp.up_proj.weight"] = get(f"{tp}/mlp/up_proj/weight")
        out[f"{hp}.mlp.down_proj.weight"] = get(f"{tp}/mlp/down_proj/weight")

    return out
