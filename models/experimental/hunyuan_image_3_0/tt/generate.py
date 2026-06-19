# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# TTNN text generation for the HunyuanImage-3.0 Instruct path (recaption / think /
# img_ratio sub-stages of `generate_image`). The base port is diffusion-only; this
# adds the missing token-sampling loop.
#
# The sampling math (stage transitions, temperature / top-k / top-p / repetition
# penalty, the decode loop) is provider-standard and inherently host-side, so it lives
# in the `ref/generate.py` golden (PCC/bit-exact-gated against upstream + HF). This
# module RE-EXPORTS that golden unchanged — so the device path and the reference share
# one source of truth — and contributes only `make_backbone_logits_fn`, the adapter
# that wires the resident TTNN backbone + LM head as the loop's `forward_logits_fn`.

# Single source of truth: the host sampling/stage logic is the ref golden.
from models.experimental.hunyuan_image_3_0.ref.generate import (  # noqa: F401
    SamplingConfig,
    StageTransitionLogitsProcessor,
    apply_repetition_penalty,
    generate_text,
    sample_next_token,
    top_k_top_p_filter,
)

# Back-compat aliases for the private helper names used by earlier tests.
_apply_repetition_penalty = apply_repetition_penalty
_top_k_top_p_filter = top_k_top_p_filter

__all__ = [
    "SamplingConfig",
    "StageTransitionLogitsProcessor",
    "apply_repetition_penalty",
    "top_k_top_p_filter",
    "sample_next_token",
    "generate_text",
    "make_backbone_logits_fn",
]


def make_backbone_logits_fn(model, lm_head, device, *, attention_mask_fn=None):
    """Adapter: wrap the resident backbone + LM head as a `forward_logits_fn`.

    Each call uploads the current ids, re-forwards the full sequence through the
    backbone (causal, text-only — no `image_infos`), applies the LM head to the LAST
    position only, and returns next-token logits [B, V] on host.

    NOTE (perf): this recomputes the whole prefix every step (O(S^2)) because the TTNN
    backbone has no KV cache yet. Correct but slow; a static-cache incremental decode
    (mirroring `HunyuanStaticCache`) is the future optimization. `model` must be built
    with `embed_state_dict` (forward by input_ids) and `apply_final_norm=True` so the
    returned hidden states are already ln_f-normed for the head.

    Args:
        attention_mask_fn: optional `S -> ttnn additive mask [B,1,S,S]`. If None, the
                           backbone uses its built-in causal SDPA.
    """
    import ttnn

    def forward_logits_fn(ids):
        # `ids` is the loop's host token sequence ([B, S] int tensor); from_torch casts
        # it to the uint32 ROW_MAJOR ids the backbone embeds.
        S = ids.shape[1]
        ids_tt = ttnn.from_torch(
            ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask = attention_mask_fn(S) if attention_mask_fn is not None else None
        hidden = model.forward(input_ids=ids_tt, seq_len=S, image_infos=None, attention_mask=mask)
        logits_tt = lm_head(hidden, last_token_only=True)  # [B, 1, V]
        logits = ttnn.to_torch(logits_tt).float().squeeze(1)  # [B, V]
        ttnn.deallocate(logits_tt)
        ttnn.deallocate(hidden)
        ttnn.deallocate(ids_tt)
        return logits

    return forward_logits_fn
