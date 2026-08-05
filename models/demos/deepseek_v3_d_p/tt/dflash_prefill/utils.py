# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Free-function helpers for the DFlash drafter prefill path (Kimi-K2.6-DFlash).

Kept out of ``tt_dflash_drafter.py`` so that module stays about the ``TtDFlashDrafter`` class:
  * ``build_drafter_rope_hf_config`` — shape the drafter's scalar rope params into the ``hf_config`` that
    ``rope.get_cos_sin_matrix`` consumes (used by ``TtDFlashDrafter``);
  * ``load_drafter_state_dict`` — read exactly the prefill-subset weights each pipeline rank needs from
    the HF checkpoint (used by the prefill runtime).
See issue #49586.
"""

from __future__ import annotations

import types

from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig


def build_drafter_rope_hf_config(cfg: DFlashDrafterConfig, max_seq_len: int) -> types.SimpleNamespace:
    """SimpleNamespace shaped like the ``hf_config`` that ``rope.get_cos_sin_matrix`` consumes.

    Crucially sets ``qk_rope_head_dim = cfg.head_dim`` (128) so the full head is rotated (Qwen3),
    unlike the MLA target which rotates only the 64-dim pe slice. Feed this with ``interleave=False``
    (half-split / rotate_half) to match the drafter's native Qwen3 weights.
    """
    return types.SimpleNamespace(
        qk_rope_head_dim=cfg.head_dim,
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


def load_drafter_state_dict(path: str, *, build_kv_tail: bool = True) -> dict:
    """Load exactly the prefill-subset weights the device drafter consumes from
    ``$DFLASH_HF_MODEL/model.safetensors`` (see ``TtDFlashDrafter._load_weights``):

      * ``fc.weight`` — always (every pipeline rank slices its owned column blocks out of it);
      * ``hidden_norm.weight`` + per-draft-layer ``self_attn.{k_proj,v_proj,k_norm}.weight`` — only when
        ``build_kv_tail`` (the last rank, which builds the KV tail). Non-tail ranks accumulate + forward
        the FC partial and skip the tail, so only ``fc.weight`` is read into host RAM.

    The drafter's decode-only weights (q_proj/o_proj/mlp/embeddings/lm_head) are never read here. Uses
    ``safe_open`` so unwanted tensors stay on disk. Mirrors the test conftest's safetensors reader so the
    device drafter and the HF reference consume identical weights."""
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
        if k == "hidden_norm.weight":
            return True
        return (
            k.startswith("layers.")
            and k.endswith(".weight")
            and any(f".self_attn.{p}." in k for p in ("k_proj", "v_proj", "k_norm"))
        )

    sd: dict = {}
    with safe_open(st, framework="pt") as f:
        for k in f.keys():
            if _wanted(k):
                sd[k] = f.get_tensor(k)
    assert "fc.weight" in sd, f"drafter checkpoint {st} missing fc.weight"
    return sd
