# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for the Kimi-K2.6-DFlash *drafter*.

Values are from the HF checkout ``Kimi-K2.6-DFlash/config.json`` (architecture
``DFlashDraftModel``, a Qwen3-style GQA model). Every default here is Kimi-K2.6-DFlash-specific
(dims, target-layer taps, rope) — do NOT reuse for another drafter without re-deriving from its
own ``config.json``. See issue #49586.
"""

from __future__ import annotations

import types
from dataclasses import dataclass


@dataclass(frozen=True)
class DFlashDrafterConfig:
    hidden_size: int = 7168
    head_dim: int = 128
    num_attention_heads: int = 64
    num_key_value_heads: int = 8  # GQA
    num_hidden_layers: int = 6  # draft layers
    num_target_layers: int = 61  # verifier layers
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02  # std for random-weight tests (config.json initializer_range)
    block_size: int = 8  # speculative block (decode-time)
    context_len: int = 4096  # spec-decode KV window
    mask_token_id: int = 163838
    # residual-stream taps of the 61-layer verifier (0-indexed layer OUTPUTS) whose hiddens
    # are concatenated (in this order) into the FC context feature.
    target_layer_ids: tuple[int, ...] = (1, 12, 24, 35, 47, 58)
    # deepseek_yarn rope (config.json rope_parameters) — identical params to the Kimi target,
    # but applied to the FULL head_dim (128) in Qwen3 half-split style, not the MLA 64-dim pe.
    rope_theta: float = 50000.0
    rope_factor: float = 64.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_orig_max_pos: int = 4096
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 8 * 128 = 1024

    @property
    def target_feature_size(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size  # 6 * 7168 = 43008

    @classmethod
    def from_pretrained(cls, path: str) -> "DFlashDrafterConfig":
        """Build the device drafter config from a HF checkpoint dir (``$DFLASH_HF_MODEL``): reads
        ``config.json`` for dims + rope params (+ ``dflash_config.target_layer_ids``). The device rope is
        built from these scalar params via :func:`build_drafter_rope_hf_config` (NOT the transformers
        ROPE_INIT_FUNCTIONS), so only the numeric rope params are extracted here — no rope-type remapping is
        needed on the device side. Mirrors the test conftest's ``_drafter_cfg_from_hf`` so the production
        runtime, the standalone test, and the HF reference all consume identical dims/rope."""
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(path, trust_remote_code=True)
        rs = dict(getattr(c, "rope_scaling", None) or getattr(c, "rope_parameters", None) or {})
        dfc = dict(getattr(c, "dflash_config", None) or {})
        d = cls()  # defaults fill anything the checkpoint config omits
        return cls(
            hidden_size=c.hidden_size,
            head_dim=getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads,
            num_hidden_layers=c.num_hidden_layers,
            rms_norm_eps=c.rms_norm_eps,
            target_layer_ids=tuple(dfc.get("target_layer_ids", d.target_layer_ids)),
            rope_theta=float(rs.get("rope_theta") or getattr(c, "rope_theta", None) or d.rope_theta),
            rope_factor=float(rs.get("factor", d.rope_factor)),
            rope_beta_fast=float(rs.get("beta_fast", d.rope_beta_fast)),
            rope_beta_slow=float(rs.get("beta_slow", d.rope_beta_slow)),
            rope_orig_max_pos=int(rs.get("original_max_position_embeddings", d.rope_orig_max_pos)),
            rope_mscale=float(rs.get("mscale", d.rope_mscale)),
            rope_mscale_all_dim=float(rs.get("mscale_all_dim", d.rope_mscale_all_dim)),
        )


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
