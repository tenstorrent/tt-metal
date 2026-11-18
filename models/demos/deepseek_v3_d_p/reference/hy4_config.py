# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Hy4-preview (tencent/Hy4-preview, HF ``model_type: hy_v4``) model configuration.

Hy4 shares GLM-5.3's attention and MoE dimensions (hidden 6144, 78 layers, 64 heads, q_lora 2048,
kv_lora 512, qk 192 + 64, v 256, DSA indexer 32 x 128 top-2048, 256 experts top-8 at 2048 plus one
shared expert), so ``Hy4Config`` SUBCLASSES ``GLM53Config`` and overrides only the deltas. Every
override below is one row of the GLM-5.3 -> Hy4 delta table; every constant it does not override is
inherited from GLM-5.3 on purpose.

Deltas that are plain configuration (handled here, consumed by the existing TT prefill stack):

  * explicit indexer map: ``indexer_types`` lists 21 ``full`` layers (0, 1, 5, 9, ..., 77) and 57
    ``shared`` layers (config.json; the tracking issue #65884 says the same, 21 / 57).
  * one dense layer (layer 0) with an 18432 FFN, instead of GLM's three 12288 dense layers.
  * vocab 120,832; ``routed_scaling_factor`` 2.827; ``swiglu_limit`` 10 (clamped SiLU-GLU on the
    routed experts only; the shared expert and dense layer 0 are HF ``HYV4MLP``, plain SiLU-GLU);
    plain RoPE with theta 1e7 (no YaRN).

Deltas wired to an existing TT path but owned by a sibling issue for numerical sign-off:

  * ``gated_mla`` (elementwise sigmoid output gate, ``linear_gate`` 6144 -> 16384): routed through
    ttMLA's existing sigmoid output gate (``mla_use_output_gate`` / ``g_proj``, built for Kimi-K3).
    Equivalence with HF ``hy_v4`` is #65921's acceptance test.

Deltas the TT prefill stack does not implement yet (the adapter refuses to serve them unless the
bring-up override is set; see ``Hy4Adapter``):

  * ``learnable_sink``: a learnable per-head attention sink (#65921).
  * iHC hyper-connections, ``hc_mult`` 4, fp32 residual streams (#65891).
  * the MTP layer (#65925): the checkpoint stores it as ``model.mtp_layers.0.*``, not the
    DeepSeek ``model.layers.78.*`` the GLM MTP path reads.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, Mapping, Optional

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config

HF_MODEL_TYPE = "hy_v4"

# Hy4's ``indexer_types`` exactly as config.json lists it ("F" = full, "S" = shared). Full layers:
# 0, 1, then every fourth layer from 5 to 77 -> 21 full, 57 shared. Kept as an explicit pattern (not
# GLM's freq/offset rule) because the checkpoint carries indexer weights for exactly these layers.
_HY4_INDEX_PATTERN = "FF" + "SSSF" * 19


class Hy4Config(GLM53Config):
    """Hy4-preview model dimensions: GLM-5.3 plus the deltas below."""

    # --- Model architecture deltas -------------------------------------------------------------
    NUM_LAYERS = 78  # same as GLM-5.3; restated because the indexer pattern length depends on it
    NUM_DENSE_LAYERS = 1  # mlp_layer_types: ["dense", "sparse" x 77]  (GLM-5.3: 3)
    INTERMEDIATE_SIZE = 18432  # dense layer-0 FFN  (GLM-5.3: 12288)
    VOCAB_SIZE = 120832  # (GLM-5.3: 154880)
    MAX_POSITION_EMBEDDINGS = 1048576
    ROPE_THETA = 10_000_000  # rope_parameters.rope_theta, rope_type "default" -> no YaRN  (GLM-5.3: 8e6)

    # --- MoE deltas ----------------------------------------------------------------------------
    ROUTE_SCALE = 2.827  # routed_scaling_factor  (GLM-5.3: 2.5)
    # swiglu_limit 10 clamps the ROUTED experts only: HF HYV4MoE computes silu(min(gate, 10)) *
    # clamp(up, -10, 10) per expert, while the shared expert and the dense layer-0 FFN are HYV4MLP with no
    # clamp (reference/hy4/model.py on hy4-preview/prefill-golden-refs). The routed-expert kernel bakes the
    # limit at compile time (ClampedSiluGluConfigDsV4, limit 10), so SWIGLU_LIMIT must stay 10; the shared
    # expert and TtFfn would read it only under "clamped_silu_glu", which Hy4 does not name for them.
    SWIGLU_LIMIT = 10.0
    ROUTED_EXPERT_ACTIVATION = "clamped_silu_glu"
    SHARED_EXPERT_ACTIVATION = "silu"
    DENSE_FFN_ACTIVATION = "silu"
    # moe_fused_swiglu has no ClampedSiluGlu, so the routed-expert hybrid split (GLM: 320) must stay off:
    # every expert runs through unified_routed_expert_moe.
    ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD = None

    # --- Indexer delta -------------------------------------------------------------------------
    INDEX_TOPK_PATTERN = _HY4_INDEX_PATTERN
    # The same map expressed as GLM's freq/offset rule (full if max(i - offset + 1, 0) % freq == 0). Only
    # kept so a consumer that re-derives the map from these two numbers gets the same answer; the
    # explicit pattern above is authoritative and test_hy4_config checks the two agree.
    INDEX_TOPK_FREQ = 4
    INDEX_SKIP_TOPK_OFFSET = 2
    # GLM-5.3's indexer RoPE is interleaved; Hy4's is unconfirmed. #65922 checks it against HF hy_v4.
    INDEX_ROPE_INTERLEAVE = True

    # --- Attention deltas ----------------------------------------------------------------------
    # gated_mla, gating_type "elementwise": sigmoid(x @ linear_gate^T) * attn_out before o_proj.
    USE_OUTPUT_GATE = True
    # learnable_sink: one learnable logit per head joins the softmax. Not implemented in sparse MLA yet.
    USE_LEARNABLE_SINK = True

    # --- iHC hyper-connections (not implemented in the prefill stack yet; #65891) --------------
    ENABLE_IHC = True
    HC_MULT = 4
    HC_EPS = 1e-6
    HC_MAGNITUDE = 2.0

    # --- MTP (#65925) --------------------------------------------------------------------------
    NUM_MTP_LAYERS = 1

    @classmethod
    def indexer_types(cls, num_layers: Optional[int] = None) -> list[str]:
        """Per-layer indexer mode (``"full"`` / ``"shared"``) from the explicit pattern, length ``num_layers``."""
        n = cls.NUM_LAYERS if num_layers is None else num_layers
        m = {"F": "full", "S": "shared"}
        return [m[c] for c in cls.INDEX_TOPK_PATTERN][:n]

    @classmethod
    def full_indexer_layers(cls) -> list[int]:
        return [i for i, t in enumerate(cls.indexer_types()) if t == "full"]

    @classmethod
    def mlp_layer_types(cls) -> list[str]:
        return ["dense"] * cls.NUM_DENSE_LAYERS + ["sparse"] * (cls.NUM_LAYERS - cls.NUM_DENSE_LAYERS)


# ---------------------------------------------------------------------------------------------------
# config.json parsing
# ---------------------------------------------------------------------------------------------------

# config.json key -> Hy4Config constant. The TT stack sizes buffers, weight caches and the KV cache from
# the CONSTANTS (the adapter's ``model_config``), so a checkpoint whose config.json disagrees with any of
# these must fail at load, not produce a mis-sized model.
_JSON_TO_CONSTANT = {
    "hidden_size": "EMB_SIZE",
    "num_hidden_layers": "NUM_LAYERS",
    "intermediate_size": "INTERMEDIATE_SIZE",
    "moe_intermediate_size": "MOE_INTERMEDIATE_SIZE",
    "vocab_size": "VOCAB_SIZE",
    "max_position_embeddings": "MAX_POSITION_EMBEDDINGS",
    "num_attention_heads": "NUM_ATTENTION_HEADS",
    "q_lora_rank": "Q_LORA_RANK",
    "kv_lora_rank": "KV_LORA_RANK",
    "qk_nope_head_dim": "QK_NOPE_HEAD_DIM",
    "qk_rope_head_dim": "QK_ROPE_HEAD_DIM",
    "v_head_dim": "V_HEAD_DIM",
    "n_routed_experts": "NUM_ROUTED_EXPERTS",
    "num_experts_per_tok": "NUM_EXPERTS_PER_TOKEN",
    "n_shared_experts": "NUM_SHARED_EXPERTS",
    "n_group": "NUM_EXPERT_GROUPS",
    "topk_group": "NUM_LIMITED_GROUPS",
    "index_topk": "INDEX_TOPK",
    "index_head_dim": "INDEX_HEAD_DIM",
    "index_n_heads": "INDEX_N_HEADS",
    "rms_norm_eps": "RMS_NORM_EPS",
    "routed_scaling_factor": "ROUTE_SCALE",
    "swiglu_limit": "SWIGLU_LIMIT",
    "gated_mla": "USE_OUTPUT_GATE",
    "learnable_sink": "USE_LEARNABLE_SINK",
    "enable_ihc": "ENABLE_IHC",
    "hc_mult": "HC_MULT",
    "hc_eps": "HC_EPS",
    "hc_magnitude": "HC_MAGNITUDE",
    "num_nextn_predict_layers": "NUM_MTP_LAYERS",
}


def load_hy4_config_json(path: str | Path) -> dict:
    """Read ``config.json`` from a checkpoint directory (or the file itself)."""
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    return json.loads(p.read_text())


def rope_theta_from_config(cfg: Mapping[str, Any]) -> float:
    """theta from ``rope_parameters`` (transformers >= 5) or a top-level ``rope_theta``. Rejects any RoPE
    type other than plain ``default``: the Hy4 adapter builds no YaRN tables."""
    rp = cfg.get("rope_parameters") or {}
    rope_type = rp.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(f"Hy4 prefill supports rope_type 'default' only, config.json has {rope_type!r}")
    theta = rp.get("rope_theta", cfg.get("rope_theta"))
    if theta is None:
        raise ValueError("config.json carries no rope_theta")
    return float(theta)


def validate_hy4_config_dict(cfg: Mapping[str, Any], model_cfg: type = Hy4Config) -> list[str]:
    """Every disagreement between a ``hy_v4`` config.json and ``model_cfg``'s constants, as messages.

    Empty when the checkpoint matches what the TT stack is built for."""
    problems: list[str] = []
    if cfg.get("model_type") != HF_MODEL_TYPE:
        problems.append(f"model_type {cfg.get('model_type')!r} != {HF_MODEL_TYPE!r}")
    for key, const in _JSON_TO_CONSTANT.items():
        if key not in cfg:
            problems.append(f"config.json has no {key!r} (expected {const}={getattr(model_cfg, const)!r})")
            continue
        want = getattr(model_cfg, const)
        got = cfg[key]
        same = (
            abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
            if isinstance(want, float)
            else got == want
        )
        if not same:
            problems.append(f"{key}={got!r} but {model_cfg.__name__}.{const}={want!r}")
    try:
        theta = rope_theta_from_config(cfg)
        if theta != float(model_cfg.ROPE_THETA):
            problems.append(f"rope_theta={theta!r} but {model_cfg.__name__}.ROPE_THETA={model_cfg.ROPE_THETA!r}")
    except ValueError as e:
        problems.append(str(e))
    if cfg.get("indexer_types") != model_cfg.indexer_types():
        problems.append("indexer_types differs from Hy4Config.INDEX_TOPK_PATTERN")
    mlp_types = cfg.get("mlp_layer_types")
    if mlp_types is not None and list(mlp_types) != model_cfg.mlp_layer_types():
        problems.append(
            f"mlp_layer_types is not {model_cfg.NUM_DENSE_LAYERS} dense layer(s) followed by sparse ones: the "
            "prefill block picks dense vs MoE by NUM_DENSE_LAYERS (a dense prefix)"
        )
    if cfg.get("gating_type", "elementwise") != "elementwise":
        problems.append(f"gating_type {cfg.get('gating_type')!r}: only the elementwise output gate is wired")
    if cfg.get("use_mla", True) is not True or cfg.get("use_dsa", True) is not True:
        problems.append("use_mla / use_dsa must both be true")
    return problems


def hy4_hf_config(
    max_seq: int = 8192,
    config_dict: Optional[Mapping[str, Any]] = None,
    model_cfg: type = Hy4Config,
):
    """HF-attribute-style config the TT prefill stack reads, for Hy4-preview.

    ``hy_v4`` is not AutoConfig-loadable on our transformers pin, so this mirrors ``glm_5_3_hf_config``
    (same field set) plus the Hy4 fields. With ``config_dict`` (a parsed checkpoint config.json) the
    checkpoint is validated against ``model_cfg`` first and a mismatch raises; the per-layer maps
    (``indexer_types``) are then taken from it. Without it, everything comes from the constants.

    ``rope_scaling.factor=1.0`` keeps YaRN off: Hy4 uses plain RoPE at theta 1e7.
    """
    if config_dict is not None:
        problems = validate_hy4_config_dict(config_dict, model_cfg)
        if problems:
            raise ValueError("Hy4 config.json does not match Hy4Config:\n  " + "\n  ".join(problems))
        indexer_types = list(config_dict["indexer_types"])
        rope_theta = rope_theta_from_config(config_dict)
    else:
        indexer_types = model_cfg.indexer_types()
        rope_theta = float(model_cfg.ROPE_THETA)

    return types.SimpleNamespace(
        model_type=HF_MODEL_TYPE,
        vocab_size=model_cfg.VOCAB_SIZE,
        hidden_size=model_cfg.EMB_SIZE,
        intermediate_size=model_cfg.INTERMEDIATE_SIZE,  # dense layer-0 FFN = 18432
        moe_intermediate_size=model_cfg.MOE_INTERMEDIATE_SIZE,
        num_attention_heads=model_cfg.NUM_ATTENTION_HEADS,
        # MLA: one latent shared by every head. config.json's num_key_value_heads (8) is not an MLA dimension
        # and no prefill consumer reads it, so keep GLM's convention (= heads).
        num_key_value_heads=model_cfg.NUM_ATTENTION_HEADS,
        kv_lora_rank=model_cfg.KV_LORA_RANK,
        q_lora_rank=model_cfg.Q_LORA_RANK,
        qk_nope_head_dim=model_cfg.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=model_cfg.QK_ROPE_HEAD_DIM,
        v_head_dim=model_cfg.V_HEAD_DIM,
        rms_norm_eps=model_cfg.RMS_NORM_EPS,
        max_seq_len=max_seq,
        rope_theta=rope_theta,
        attention_bias=False,
        rope_scaling={
            "factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": max_seq,
        },
        index_n_heads=model_cfg.INDEX_N_HEADS,
        index_head_dim=model_cfg.INDEX_HEAD_DIM,
        index_topk=model_cfg.INDEX_TOPK,
        index_rope_interleave=model_cfg.INDEX_ROPE_INTERLEAVE,
        sparse_mla_overlap_profile="auto",
        indexer_types=indexer_types,
        index_topk_freq=model_cfg.INDEX_TOPK_FREQ,
        index_skip_topk_offset=model_cfg.INDEX_SKIP_TOPK_OFFSET,
        first_k_dense_replace=model_cfg.NUM_DENSE_LAYERS,
        mlp_layer_types=model_cfg.mlp_layer_types(),
        n_routed_experts=model_cfg.NUM_ROUTED_EXPERTS,
        n_shared_experts=model_cfg.NUM_SHARED_EXPERTS,
        num_experts_per_tok=model_cfg.NUM_EXPERTS_PER_TOKEN,
        routed_scaling_factor=model_cfg.ROUTE_SCALE,
        swiglu_limit=model_cfg.SWIGLU_LIMIT,
        num_hidden_layers=model_cfg.NUM_LAYERS,
        num_nextn_predict_layers=model_cfg.NUM_MTP_LAYERS,
        index_share_for_mtp_iteration=True,
        # --- Hy4 attention deltas ---
        # Read by ttMLA (output gate) -- the Kimi-K3 gate path, weights cached as g_proj.
        mla_use_output_gate=bool(model_cfg.USE_OUTPUT_GATE),
        # Not read by any TT module yet (#65921); carried so the sparse MLA can key on it when it lands.
        mla_learnable_sink=bool(model_cfg.USE_LEARNABLE_SINK),
        # --- iHC (#65891); not read by the prefill stack yet ---
        enable_ihc=bool(model_cfg.ENABLE_IHC),
        hc_mult=model_cfg.HC_MULT,
        hc_eps=model_cfg.HC_EPS,
        hc_magnitude=model_cfg.HC_MAGNITUDE,
        # BF16 release by default; the MXFP8 release is dequantized by the Hy4 checkpoint reader
        # (tt/runners/hy4_checkpoint.py), which reads the scale tensors directly, so no block size is needed.
        quantization_config=None,
    )
