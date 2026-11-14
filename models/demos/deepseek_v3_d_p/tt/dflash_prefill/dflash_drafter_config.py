# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for the DFlash *drafters* whose context KV the prefill path builds.

Two attention modes:

* ``"gqa"`` (default): Kimi-K2.6/K2.7-DFlash and RadixArk's Kimi-K3-DSpark. Values are from the HF
  checkout ``Kimi-K2.6-DFlash/config.json`` (architecture ``DFlashDraftModel``, a Qwen3-style GQA model).
  Prefill writes separate K and V caches per kv-head. See issue #49586.
* ``"mla"``: ``lightseekorg/kimi-k3-dflash2`` (``DFlash2DraftModel``, ``attention_mode: mla``) and
  Inferact's ``Kimi-K3-DSpark`` (``K3DSparkModel``), the two drafters tt-blaze decodes with MLA
  (``blaze/models/dflash/k3_mla_blaze_config.py``). Prefill writes ONE latent cache row per token,
  ``[kv_a_layernorm(c_kv) | rope(k_pe)]`` (512 + 64 = 576), which serves as both K and V.

Every default here is checkpoint-specific (dims, target-layer taps, rope) — do NOT reuse for another
drafter without re-deriving from its own ``config.json``. :func:`kimi_k3_dflash2_mla_config` is the
K3 DFlash2 preset (tests, and the fallback for keys an MLA checkpoint config omits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

ATTENTION_MODES = ("gqa", "mla")

# HF architecture names of the MLA drafters (see the module docstring).
MLA_ARCHITECTURES = ("DFlash2DraftModel", "K3DSparkModel")


def yarn_get_mscale(scale: float, mscale: float) -> float:
    """DeepSeek / TorchSpec YaRN amplitude ``0.1 * mscale * ln(scale) + 1`` (1 when scale <= 1)."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@dataclass(frozen=True)
class DFlashDrafterConfig:
    hidden_size: int = 7168
    head_dim: int = 128
    num_attention_heads: int = 64
    num_key_value_heads: int = 8  # GQA
    num_hidden_layers: int = 6  # draft layers
    # Verifier layers this drafter attaches to. None = the checkpoint config did not declare it: this field
    # exists ONLY to be cross-checked against the loaded verifier, so it must never default to a value that
    # would satisfy that check (61 is Kimi-K2.x's own depth, i.e. exactly the answer being verified). None is
    # falsy, so the build guard (`assert num_target_layers`) rejects a checkpoint that omits the key.
    num_target_layers: int | None = None
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
    rope_convention: str = "interleaved"
    # Kimi-K3-DSpark (architectures DSparkDraftModel): the DFlash context-KV path plus its yarn cos/sin amplitude.
    is_dspark: bool = False
    # "gqa" (K/V caches per kv-head) or "mla" (one 576-wide latent cache). See the module docstring.
    attention_mode: str = "gqa"
    # MLA only: c_kv width and the rotated k_pe width of kv_a_proj_with_mqa's output.
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64

    def __post_init__(self):
        if self.attention_mode not in ATTENTION_MODES:
            raise ValueError(f"attention_mode {self.attention_mode!r} must be one of {ATTENTION_MODES}")

    @property
    def is_mla(self) -> bool:
        return self.attention_mode == "mla"

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 8 * 128 = 1024

    @property
    def latent_dim(self) -> int:
        """MLA cache row width: normed c_kv then roped k_pe (512 + 64 = 576)."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def rope_dim(self) -> int:
        """Width the rope table rotates: the whole GQA head (Qwen3), or MLA's k_pe slice only."""
        return self.qk_rope_head_dim if self.is_mla else self.head_dim

    @property
    def rope_yarn_amplitude(self) -> float:
        """``m(factor, mscale) / m(factor, mscale_all_dim)``: the cos/sin scale TorchSpec's YaRN (and blaze's
        ``mla_golden.yarn_inv_freq``) bakes in. ``mla/rope.py``'s tables are pure rotations, so a non-unit
        amplitude has to be folded into the weights that feed RoPE (RoPE is linear). 1.0 for both K3 MLA
        drafters (mscale == mscale_all_dim == 1)."""
        return yarn_get_mscale(self.rope_factor, self.rope_mscale) / yarn_get_mscale(
            self.rope_factor, self.rope_mscale_all_dim
        )

    @property
    def target_feature_size(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size  # 6 * 7168 = 43008

    @staticmethod
    def hf_attention_mode(c) -> str:
        """``"mla"`` when the HF config declares an MLA drafter, else ``"gqa"``.

        Checked in order: an explicit ``attention_mode`` (top level or under ``dflash_config``), a known MLA
        architecture name, then the presence of ``kv_lora_rank`` (only MLA configs carry it)."""
        dfc = dict(getattr(c, "dflash_config", None) or {})
        mode = getattr(c, "attention_mode", None) or dfc.get("attention_mode")
        if mode is not None:
            mode = str(mode).lower()
            if mode not in ATTENTION_MODES:
                raise ValueError(f"drafter config attention_mode {mode!r} is not one of {ATTENTION_MODES}")
            return mode
        if any(a in MLA_ARCHITECTURES for a in (getattr(c, "architectures", None) or ())):
            return "mla"
        return "mla" if getattr(c, "kv_lora_rank", None) is not None else "gqa"

    @classmethod
    def from_hf_config(cls, c) -> "DFlashDrafterConfig":
        """Build the device drafter config from an already-loaded HF config object."""
        if cls.hf_attention_mode(c) == "mla":
            return cls._from_hf_config_mla(c)
        rs = dict(getattr(c, "rope_scaling", None) or getattr(c, "rope_parameters", None) or {})
        dfc = dict(getattr(c, "dflash_config", None) or {})
        d = cls()  # defaults fill anything the checkpoint config omits
        return cls(
            hidden_size=c.hidden_size,
            head_dim=getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads,
            num_hidden_layers=c.num_hidden_layers,
            # Top-level key in the drafter's config.json (61 for Kimi-K2.x), NOT under dflash_config. It is
            # the drafter's own declaration of which verifier it attaches to, so the runtime cross-checks it
            # against the verifier's num_hidden_layers before building. Absent -> None (the fail-closed default).
            num_target_layers=(int(v) if (v := getattr(c, "num_target_layers", None)) is not None else None),
            rms_norm_eps=c.rms_norm_eps,
            target_layer_ids=tuple(dfc.get("target_layer_ids", d.target_layer_ids)),
            rope_theta=float(rs.get("rope_theta") or getattr(c, "rope_theta", None) or d.rope_theta),
            rope_factor=float(rs.get("factor", d.rope_factor)),
            rope_beta_fast=float(rs.get("beta_fast", d.rope_beta_fast)),
            rope_beta_slow=float(rs.get("beta_slow", d.rope_beta_slow)),
            rope_orig_max_pos=int(rs.get("original_max_position_embeddings", d.rope_orig_max_pos)),
            rope_mscale=float(rs.get("mscale", d.rope_mscale)),
            rope_mscale_all_dim=float(rs.get("mscale_all_dim", d.rope_mscale_all_dim)),
            is_dspark="DSparkDraftModel" in (getattr(c, "architectures", None) or ()),
        )

    @classmethod
    def _from_hf_config_mla(cls, c) -> "DFlashDrafterConfig":
        """MLA drafter (K3 DFlash2 / Inferact DSpark). NOT yet checked against a real ``config.json``: keys
        are read where TorchSpec's configs put them, and the K3 DFlash2 preset fills anything absent EXCEPT
        ``num_target_layers``, which stays fail-closed (None) so the runtime's verifier cross-check rejects
        a checkpoint that does not declare its target."""
        rs = dict(getattr(c, "rope_scaling", None) or getattr(c, "rope_parameters", None) or {})
        dfc = dict(getattr(c, "dflash_config", None) or {})
        d = kimi_k3_dflash2_mla_config()

        def get(name, default):
            v = getattr(c, name, None)
            return default if v is None else v

        taps = dfc.get("target_layer_ids", getattr(c, "target_layer_ids", None))
        return cls(
            attention_mode="mla",
            hidden_size=int(get("hidden_size", d.hidden_size)),
            head_dim=int(get("v_head_dim", d.head_dim)),
            num_attention_heads=int(get("num_attention_heads", d.num_attention_heads)),
            # MLA caches ONE latent head; num_key_value_heads (if present) describes the un-absorbed form.
            num_key_value_heads=1,
            num_hidden_layers=int(get("num_hidden_layers", d.num_hidden_layers)),
            num_target_layers=(int(v) if (v := getattr(c, "num_target_layers", None)) is not None else None),
            rms_norm_eps=float(get("rms_norm_eps", d.rms_norm_eps)),
            block_size=int(dfc.get("block_size", get("block_size", d.block_size))),
            mask_token_id=int(dfc.get("mask_token_id", get("mask_token_id", d.mask_token_id))),
            target_layer_ids=tuple(taps) if taps is not None else d.target_layer_ids,
            rope_theta=float(rs.get("rope_theta") or getattr(c, "rope_theta", None) or d.rope_theta),
            rope_factor=float(rs.get("factor", d.rope_factor)),
            rope_beta_fast=float(rs.get("beta_fast", d.rope_beta_fast)),
            rope_beta_slow=float(rs.get("beta_slow", d.rope_beta_slow)),
            rope_orig_max_pos=int(rs.get("original_max_position_embeddings", d.rope_orig_max_pos)),
            rope_mscale=float(rs.get("mscale", d.rope_mscale)),
            rope_mscale_all_dim=float(rs.get("mscale_all_dim", d.rope_mscale_all_dim)),
            kv_lora_rank=int(get("kv_lora_rank", d.kv_lora_rank)),
            qk_rope_head_dim=int(get("qk_rope_head_dim", d.qk_rope_head_dim)),
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "DFlashDrafterConfig":
        """Build the device drafter config from a HF checkpoint dir (``$DFLASH_HF_MODEL``): reads
        ``config.json`` via ``AutoConfig`` and extracts dims + rope params through :meth:`from_hf_config`."""
        from transformers import AutoConfig

        return cls.from_hf_config(AutoConfig.from_pretrained(path, trust_remote_code=True))


def kimi_k3_dflash2_mla_config(**overrides) -> DFlashDrafterConfig:
    """``lightseekorg/kimi-k3-dflash2`` as tt-blaze describes it (``kimi_k3_dflash2_mla.model_config.json`` and
    ``k3_mla_blaze_config.DFLASH2_MLA_TARGET_LAYER_IDS``, blaze commit f9ac6439eb): five MLA layers, taps
    19/37/66/78/90 of the 93-layer Kimi-K3, block 8, YaRN theta 50000 / factor 32 / original 32768 / mscale 1.

    ``num_target_layers`` is 93 here because this is a PRESET (tests, synthetic weights); the checkpoint path
    (:meth:`DFlashDrafterConfig.from_hf_config`) keeps it fail-closed. K3.1 shares K3's 93-layer stack, so the
    same drafter attaches to both; that equivalence is assumed, not verified."""
    cfg = DFlashDrafterConfig(
        attention_mode="mla",
        hidden_size=7168,
        head_dim=128,  # v_head_dim; unused by the prefill context path
        num_attention_heads=64,
        num_key_value_heads=1,
        num_hidden_layers=5,
        num_target_layers=93,
        rms_norm_eps=1e-5,
        block_size=8,
        context_len=4096,  # sliding window of draft layers 0-3 (decode-side); prefill keeps full history
        mask_token_id=163592,
        target_layer_ids=(19, 37, 66, 78, 90),
        rope_theta=50000.0,
        rope_factor=32.0,
        rope_beta_fast=32.0,
        rope_beta_slow=1.0,
        rope_orig_max_pos=32768,
        rope_mscale=1.0,
        rope_mscale_all_dim=1.0,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    return replace(cfg, **overrides) if overrides else cfg
