# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types

from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig


def build_drafter_rope_hf_config(cfg: DFlashDrafterConfig, max_seq_len: int) -> types.SimpleNamespace:
    """SimpleNamespace shaped like the ``hf_config`` that ``rope.get_cos_sin_matrix`` consumes.

    GQA: sets ``qk_rope_head_dim = cfg.head_dim`` (128) so the full head is rotated (Qwen3), unlike the
    MLA target which rotates only the 64-dim pe slice. MLA drafter: ``cfg.qk_rope_head_dim`` (64), the
    k_pe slice, in the same Meta-interleaved pairing TorchSpec / blaze's ``mla_golden.rope_interleaved`` use,
    so no weight permutation is needed (see ``cfg.rope_dim``).
    """
    return types.SimpleNamespace(
        qk_rope_head_dim=cfg.rope_dim,
        max_seq_len=max_seq_len,
        rope_theta=float(cfg.rope_theta),
        rope_scaling={
            "factor": cfg.rope_factor,
            "original_max_position_embeddings": cfg.rope_orig_max_pos,
            "beta_fast": cfg.rope_beta_fast,
            "beta_slow": cfg.rope_beta_slow,
            "mscale": cfg.rope_mscale,
            "mscale_all_dim": cfg.rope_mscale_all_dim,
        },
    )


# Model-level norm on the projected context feature: DFlash/DFlash2 call it ``hidden_norm``, Inferact's
# DSpark ``context_norm`` (cf. blaze ``weights/dflash/mla.py:CONTEXT_NORM_KEYS``). Loaded under the first
# name and stored as ``hidden_norm.weight``, the key the device drafter reads.
CONTEXT_NORM_KEYS = ("hidden_norm.weight", "context_norm.weight")

# Per-draft-layer attention weights the prefill context path reads, by attention mode.
_TAIL_PROJ = {
    "gqa": ("k_proj", "v_proj", "k_norm"),
    "mla": ("kv_a_proj_with_mqa", "kv_a_layernorm"),
}


def load_drafter_state_dict(path: str, *, build_kv_tail: bool = True, attention_mode: str = "gqa") -> dict:
    """Load exactly the prefill-subset weights the device drafter consumes from
    ``$DFLASH_HF_MODEL/model.safetensors`` (see ``TtDFlashDrafter._load_weights``):

      * ``fc.weight`` — always (every pipeline rank slices its owned column blocks out of it);
      * ``hidden_norm.weight`` (or ``context_norm.weight``, renamed) + per-draft-layer tail weights — only when
        ``build_kv_tail`` (the last rank, which builds the KV tail). Non-tail ranks accumulate + forward
        the FC partial and skip the tail, so only ``fc.weight`` is read into host RAM. The tail is
        ``self_attn.{k_proj,v_proj,k_norm}`` for GQA and ``self_attn.{kv_a_proj_with_mqa,kv_a_layernorm}``
        for MLA. Raw checkpoint weights: unlike blaze's decode loader, nothing is gamma-folded (prefill has
        no input_layernorm on the context path).
    """
    tail_proj = _TAIL_PROJ[attention_mode]
    import os

    from safetensors import safe_open

    st = os.path.join(path, "model.safetensors")
    if not os.path.exists(st):
        raise FileNotFoundError(
            f"drafter weights not found: {st} (set DFLASH_HF_MODEL to a dir with config.json + model.safetensors)"
        )

    def _wanted(k: str) -> bool:
        if k == "fc.weight":
            return True
        if not build_kv_tail:
            return False
        if k in CONTEXT_NORM_KEYS:
            return True
        return k.startswith("layers.") and k.endswith(".weight") and any(f".self_attn.{p}." in k for p in tail_proj)

    sd: dict = {}
    with safe_open(st, framework="pt") as f:
        for k in f.keys():
            if _wanted(k):
                sd[k] = f.get_tensor(k)
    assert "fc.weight" in sd, f"drafter checkpoint {st} missing fc.weight"
    if build_kv_tail:
        norms = [k for k in CONTEXT_NORM_KEYS if k in sd]
        assert len(norms) == 1, f"drafter checkpoint {st} must carry exactly one of {CONTEXT_NORM_KEYS}, got {norms}"
        sd["hidden_norm.weight"] = sd.pop(norms[0])
    return sd
