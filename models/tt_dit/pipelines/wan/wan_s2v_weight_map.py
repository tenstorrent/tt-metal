# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Translate a Wan2.2-S2V-14B native state_dict into tt_dit's hierarchy.

The reference checkpoint (``Wan-AI/Wan2.2-S2V-14B``) is published with the
``WanModel_S2V`` module names from ``wan/modules/s2v/model_s2v.py``, NOT in
Diffusers' ``WanTransformer3DModel`` layout. tt_dit's
``WanS2VTransformer3DModel`` is built on top of the Diffusers layout, so we
need a key rename pass before ``load_torch_state_dict`` will accept the
checkpoint.

Only key names change here — tensor shapes are left as-is. The receiving
tt_dit modules' ``_prepare_torch_state`` methods handle:

  * splitting / TP-aware head-interleaved fusing of q/k/v into ``to_qkv``
    (or ``k/v`` into ``to_kv``) — done by ``WanAttention._prepare_torch_state``,
  * transposing Linear weights from HF ``[out, in]`` to ttnn ``[in, out]`` —
    done by ``Linear._prepare_torch_state`` (and friends),
  * reshaping ``Conv1d`` weight to the 5-D form ``ttnn.experimental.conv3d``
    expects — done by ``CausalConv1d._prepare_torch_state``,
  * permuting/reshaping ``LayerNorm`` weight for distributed sharding —
    done by ``DistributedLayerNorm._prepare_torch_state``,
  * the ``ffn.0 → ffn.net.0.proj`` / ``ffn.2 → ffn.net.2`` Diffusers ↔ raw
    layout dance — done by ``WanTransformerBlock._prepare_torch_state``.

So this mapper's only job is to produce keys in the *Diffusers* naming
convention (e.g. ``blocks.{i}.attn1.to_q.weight``, ``ffn.net.0.proj.weight``,
``condition_embedder.time_embedder.linear_1.weight``). Every reference
module is loaded into an on-device tt_dit module — no keys are excluded.
"""

from __future__ import annotations

import re

import torch

# Top-level renames that don't depend on a block index.
_FLAT_RENAMES: dict[str, str] = {
    "head.head.weight": "proj_out.weight",
    "head.head.bias": "proj_out.bias",
    "head.modulation": "scale_shift_table",
    "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
    "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
    "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
    "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
    "time_projection.1.weight": "condition_embedder.time_proj.weight",
    "time_projection.1.bias": "condition_embedder.time_proj.bias",
    "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
    "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
    "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
    "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    "casual_audio_encoder.weights": "audio_encoder.weights",
    # cond_encoder is an on-device WanPatchEmbed; its `_prepare_torch_state`
    # consumes the raw Conv3d ``weight``/``bias`` and reshapes/permutes them.
    "cond_encoder.weight": "cond_encoder.weight",
    "cond_encoder.bias": "cond_encoder.bias",
    # trainable_cond_mask is an on-device Parameter shape [3, dim].
    "trainable_cond_mask.weight": "trainable_cond_mask",
}


# Inside a transformer block, the ref ↔ tt name mapping. The ``self_attn`` /
# ``cross_attn`` and ``norm{1,2,3}`` renames are listed explicitly because the
# index-shuffling of the norms is the one place where it's easy to get wrong.
#
# Ref layout (wan/modules/model.py:WanAttentionBlock):
#   norm1 (no-affine) → self_attn → norm3 (with-affine) → cross_attn
#                                  → norm2 (no-affine) → ffn
#
# tt layout (transformer_wan.py:WanTransformerBlock):
#   norm1 (no-affine) → attn1     → norm2 (with-affine) → attn2
#                                  → norm3 (no-affine) → ffn
#
# So ``ref.norm3 ↔ tt.norm2`` (the cross-attention pre-norm). The other two
# layernorms have ``elementwise_affine=False`` and therefore no state to map,
# but we still rename them for cleanliness if they're ever materialized.

_ATTN_SUFFIX_RENAMES: dict[str, str] = {
    "q.weight": "to_q.weight",
    "q.bias": "to_q.bias",
    "k.weight": "to_k.weight",
    "k.bias": "to_k.bias",
    "v.weight": "to_v.weight",
    "v.bias": "to_v.bias",
    "o.weight": "to_out.0.weight",
    "o.bias": "to_out.0.bias",
    "norm_q.weight": "norm_q.weight",
    "norm_k.weight": "norm_k.weight",
}

# blocks.{i}.<piece>.<rest>  e.g.  blocks.7.self_attn.q.weight
_BLOCKS_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
# audio_injector.injector.{i}.<rest>  e.g.  audio_injector.injector.3.q.weight
_AI_INJECTOR_RE = re.compile(r"^audio_injector\.injector\.(\d+)\.(.+)$")
# audio_injector.injector_adain_layers.{i}.<rest>
_AI_ADAIN_RE = re.compile(r"^audio_injector\.injector_adain_layers\.(\d+)\.(.+)$")


def _translate_attn_suffix(piece: str, rest: str) -> str | None:
    """Translate a ``self_attn.*`` / ``cross_attn.*`` suffix to its tt_dit name.

    ``piece`` is either ``self_attn`` or ``cross_attn``. ``rest`` is the
    remainder (e.g. ``q.weight``, ``norm_k.weight``, ``o.bias``).
    """
    target_attn = "attn1" if piece == "self_attn" else "attn2"
    if rest in _ATTN_SUFFIX_RENAMES:
        return f"{target_attn}.{_ATTN_SUFFIX_RENAMES[rest]}"
    return None  # unrecognized; caller decides what to do


def _translate_block_key(idx: int, rest: str) -> str | None:
    """Map a per-block key suffix (after ``blocks.{idx}.``) to its tt_dit name."""
    # Modulation parameter → scale_shift_table (the block's _prepare_torch_state
    # unsqueezes to [1, 1, 6, dim]).
    if rest == "modulation":
        return f"blocks.{idx}.scale_shift_table"

    # The cross-attention pre-norm: ref.norm3 ↔ tt.norm2.
    if rest in ("norm3.weight", "norm3.bias"):
        return f"blocks.{idx}.norm2.{rest.split('.', 1)[1]}"

    # FFN: ref ffn.{0,2}.{w,b} ↔ Diffusers ffn.net.{0.proj, 2}.{w,b}. The block's
    # _prepare_torch_state then renames to ffn.{ff1,ff2}.
    if rest.startswith("ffn.0."):
        return f"blocks.{idx}.ffn.net.0.proj.{rest.split('.', 2)[2]}"
    if rest.startswith("ffn.2."):
        return f"blocks.{idx}.ffn.net.2.{rest.split('.', 2)[2]}"

    # Self-/cross-attention substates.
    if rest.startswith("self_attn.") or rest.startswith("cross_attn."):
        piece, sub = rest.split(".", 1)
        tt_suffix = _translate_attn_suffix(piece, sub)
        if tt_suffix is None:
            return None
        return f"blocks.{idx}.{tt_suffix}"

    return None


def _translate_audio_injector_key(idx: int, rest: str) -> str | None:
    """Map ``audio_injector.injector.{idx}.<rest>`` → tt_dit name.

    The injector is a ``ModuleList`` of cross-attention layers. The mapping is
    the same as for cross_attn inside a transformer block (q/k/v/o → to_q/...).
    """
    if rest in _ATTN_SUFFIX_RENAMES:
        return f"audio_injector.injector.{idx}.{_ATTN_SUFFIX_RENAMES[rest]}"
    return None


def translate_s2v_state_dict(ref_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate a Wan2.2-S2V-14B native state_dict to tt_dit's naming.

    Args:
        ref_state_dict: ``{key: tensor}`` keyed by the reference repo's native
            module names (output of :func:`load_s2v_state_dict`).

    Returns:
        A new dict keyed by tt_dit module names. CPU-shadow keys are excluded.
        Tensors are NOT copied or cast — they're the same Python objects as in
        the input.

    Raises:
        KeyError: if a key in the input doesn't match any known translation
            rule. We fail loud rather than silently dropping unknown keys so a
            new reference version doesn't slip past unnoticed.
    """
    out: dict[str, torch.Tensor] = {}

    for key, tensor in ref_state_dict.items():
        # 1. Flat renames.
        if key in _FLAT_RENAMES:
            out[_FLAT_RENAMES[key]] = tensor
            continue

        # 3. patch_embedding.{weight,bias} pass through.
        if key in ("patch_embedding.weight", "patch_embedding.bias"):
            out[key] = tensor
            continue

        # 3b. frame_packer.{proj,proj_2x,proj_4x}.{weight,bias} pass through.
        # FramePackMotionerWan's WanPatchEmbed children consume the raw
        # Conv3d weight/bias via their own ``_prepare_torch_state``.
        if key.startswith("frame_packer."):
            out[key] = tensor
            continue

        # 4. casual_audio_encoder.encoder.* → audio_encoder.encoder.* (no
        #    further suffix changes; the CausalConv1d / final_linear /
        #    padding_tokens substates all have matching names in tt_dit).
        if key.startswith("casual_audio_encoder.encoder."):
            out["audio_encoder.encoder." + key[len("casual_audio_encoder.encoder.") :]] = tensor
            continue

        # 5. blocks.{i}.<rest>
        m = _BLOCKS_RE.match(key)
        if m is not None:
            idx, rest = int(m.group(1)), m.group(2)
            translated = _translate_block_key(idx, rest)
            if translated is None:
                msg = f"unrecognized block key suffix: {key!r}"
                raise KeyError(msg)
            out[translated] = tensor
            continue

        # 6. audio_injector.injector.{i}.<rest>
        m = _AI_INJECTOR_RE.match(key)
        if m is not None:
            idx, rest = int(m.group(1)), m.group(2)
            translated = _translate_audio_injector_key(idx, rest)
            if translated is None:
                msg = f"unrecognized audio_injector.injector key suffix: {key!r}"
                raise KeyError(msg)
            out[translated] = tensor
            continue

        # 7. audio_injector.injector_adain_layers.{i}.linear.{weight,bias} —
        #    name matches tt_dit; just copy through.
        m = _AI_ADAIN_RE.match(key)
        if m is not None:
            idx, rest = int(m.group(1)), m.group(2)
            if rest in ("linear.weight", "linear.bias"):
                out[key] = tensor
                continue
            msg = f"unrecognized audio_injector.injector_adain_layers key: {key!r}"
            raise KeyError(msg)

        msg = f"no translation rule for reference key {key!r}"
        raise KeyError(msg)

    return out
