# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
"""
config_adapter.py — Kimi K2.5 → DSV3-compatible architecture params

Loads the HuggingFace config for moonshotai/Kimi-K2.5 (model_type: "kimi_k2" or "kimi_k25")
and produces a validated parameter dict compatible with the tt-metal DeepSeek V3 runtime.

Kimi K2.5 is architecturally near-identical to DeepSeek V3. The key deltas are:
  - 384 routed experts (vs 256)
  - 64 attention heads (vs 128)
  - Flat top-k routing: n_group=1, topk_group=1 (vs grouped n_group=8, topk_group=4)
  - first_k_dense_replace=1 (vs 3) — only layer 0 is a dense MLP
  - rope_theta=50000 (vs 10000)
  - YaRN factor=64.0 (vs DSV3's value)
  - rms_norm_eps=1e-5 (vs 1e-6 — important for accuracy)
  - vocab_size=163840 (vs ~129280)
  - routed_scaling_factor=2.827 (vs 2.5)
  - Weight quantization: INT4 group-32 symmetric (vs FP8) — expert weights only
  - No MTP layers (vs 1 in DSV3)

Usage:
    from models.demos.kimi_k25.utils.config_adapter import KimiK25Config, load_kimi_config

    # From HuggingFace (requires network / local weights):
    params = load_kimi_config("moonshotai/Kimi-K2.5")

    # From a local directory with config.json:
    params = load_kimi_config("/path/to/kimi-k2.5")

    # From hardcoded research fixture (no network required):
    params = KimiK25Config.from_fixture()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Known-good architecture constants (from research doc + HF config.json)
# Used for validation and as the fixture fallback.
# ---------------------------------------------------------------------------

_KIMI_K25_REFERENCE = {
    # Transformer backbone
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,  # MLA: kv heads == q heads before low-rank proj
    # MLA params (identical to DSV3)
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    # MoE routing
    "num_experts_per_tok": 8,           # top-k = 8
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,         # only layer 0 is dense
    "moe_layer_freq": 1,                # every layer is MoE (except first_k_dense)
    "n_group": 1,                       # flat routing — no expert grouping
    "topk_group": 1,
    "routed_scaling_factor": 2.827,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    # FFN sizes
    "moe_intermediate_size": 2048,
    "intermediate_size": 18432,         # dense layer intermediate size
    # Normalisation
    "rms_norm_eps": 1e-5,              # NOTE: 1e-5, not 1e-6 (DSV3)
    # Rotary embeddings
    "rope_theta": 50000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 64.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
    },
    # Context / vocab
    "max_position_embeddings": 131072,  # model max; YaRN extends to 256K
    "vocab_size": 163840,
    # Special tokens
    "bos_token_id": 1,
    "eos_token_id": 163585,             # generation_config prefers 163586; log warning
    # Weight quantization
    "quant_method": "compressed-tensors",
    "quant_type": "int4",               # W4A16 symmetric, group_size=32
    "quant_group_size": 32,
    "quant_targets": "routed_experts",  # only routed expert linears are quantized
    # Architecture deltas from DSV3
    "num_mtp_layers": 0,               # no multi-token prediction
    "activation_function": "silu",     # SwiGLU via split gate
    "hidden_act": "silu",
}

# Tolerances for float comparisons during validation
_FLOAT_FIELDS = {"routed_scaling_factor", "rms_norm_eps", "rope_theta"}
_FLOAT_TOL = 1e-6


# ---------------------------------------------------------------------------
# KimiK25Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class KimiK25Config:
    """Validated architecture parameters for Kimi K2.5.

    This is the single source of truth passed into the tt-metal model instantiation.
    Fields match the DSV3 config contract where applicable; Kimi-specific fields are
    added here and stripped before passing to DSV3 components that don't need them.
    """

    # ---- Transformer backbone ----
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_attention_heads: int = 64
    num_key_value_heads: int = 64

    # ---- MLA ----
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # ---- MoE routing ----
    num_experts_per_tok: int = 8
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"

    # ---- FFN ----
    moe_intermediate_size: int = 2048
    intermediate_size: int = 18432

    # ---- Normalisation ----
    rms_norm_eps: float = 1e-5

    # ---- Rotary ----
    rope_theta: float = 50000.0
    rope_scaling: Dict[str, Any] = field(default_factory=lambda: {
        "type": "yarn",
        "factor": 64.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
    })

    # ---- Context / vocab ----
    max_position_embeddings: int = 131072
    vocab_size: int = 163840
    bos_token_id: int = 1
    eos_token_id: int = 163585

    # ---- Quantization metadata ----
    quant_method: str = "compressed-tensors"
    quant_type: str = "int4"
    quant_group_size: int = 32
    quant_targets: str = "routed_experts"

    # ---- Architecture flags ----
    num_mtp_layers: int = 0
    hidden_act: str = "silu"

    # ---- Derived / computed ----
    head_dim: int = field(init=False)
    experts_per_device_tg: int = field(init=False)   # TG (32 devices): 384 / 32 = 12
    experts_per_device_dual: int = field(init=False)  # DUAL (64): 6
    experts_per_device_quad: int = field(init=False)  # QUAD (128): 3
    padded_vocab_size: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 192
        self.experts_per_device_tg = self.n_routed_experts // 32   # 12
        self.experts_per_device_dual = self.n_routed_experts // 64  # 6
        self.experts_per_device_quad = self.n_routed_experts // 128  # 3

        # Pad vocab to next multiple of 64 for efficient embedding tiling
        self.padded_vocab_size = ((self.vocab_size + 63) // 64) * 64  # 163840 already aligned

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_fixture(cls) -> "KimiK25Config":
        """Return a config built from the hardcoded research fixture.

        Use this when network access or local weights are unavailable.
        All values are sourced from the research doc (kimi-k2-5-galaxy-port.md)
        and verified against the HF config.json.
        """
        return cls()

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "KimiK25Config":
        """Build and validate a KimiK25Config from a HuggingFace AutoConfig object.

        Args:
            hf_config: A transformers.AutoConfig instance loaded from
                       ``moonshotai/Kimi-K2.5`` or a local directory.

        Returns:
            KimiK25Config with validated fields.

        Raises:
            ValueError: If a critical field mismatches the expected value.
            warnings.warn: For non-critical mismatches (e.g. eos_token_id discrepancy).
        """
        model_type = getattr(hf_config, "model_type", "unknown")
        if model_type not in ("kimi_k2", "kimi_k25", "deepseek_v3"):
            warnings.warn(
                f"Unexpected model_type='{model_type}'. "
                "Expected 'kimi_k2', 'kimi_k25', or 'deepseek_v3'. Proceeding anyway.",
                stacklevel=2,
            )

        # Extract the text_config sub-config if present (multimodal wrapper)
        text_cfg = getattr(hf_config, "text_config", hf_config)

        def get(attr: str, default: Any = None) -> Any:
            return getattr(text_cfg, attr, getattr(hf_config, attr, default))

        # --- Build config from HF fields ---
        rope_scaling_raw = get("rope_scaling", {})
        if hasattr(rope_scaling_raw, "to_dict"):
            rope_scaling_raw = rope_scaling_raw.to_dict()

        cfg = cls(
            hidden_size=get("hidden_size", 7168),
            num_hidden_layers=get("num_hidden_layers", 61),
            num_attention_heads=get("num_attention_heads", 64),
            num_key_value_heads=get("num_key_value_heads", 64),
            q_lora_rank=get("q_lora_rank", 1536),
            kv_lora_rank=get("kv_lora_rank", 512),
            qk_nope_head_dim=get("qk_nope_head_dim", 128),
            qk_rope_head_dim=get("qk_rope_head_dim", 64),
            v_head_dim=get("v_head_dim", 128),
            num_experts_per_tok=get("num_experts_per_tok", 8),
            n_routed_experts=get("n_routed_experts", 384),
            n_shared_experts=get("n_shared_experts", 1),
            first_k_dense_replace=get("first_k_dense_replace", 1),
            moe_layer_freq=get("moe_layer_freq", 1),
            n_group=get("n_group", 1),
            topk_group=get("topk_group", 1),
            routed_scaling_factor=float(get("routed_scaling_factor", 2.827)),
            norm_topk_prob=bool(get("norm_topk_prob", True)),
            scoring_func=get("scoring_func", "sigmoid"),
            topk_method=get("topk_method", "noaux_tc"),
            moe_intermediate_size=get("moe_intermediate_size", 2048),
            intermediate_size=get("intermediate_size", 18432),
            rms_norm_eps=float(get("rms_norm_eps", 1e-5)),
            rope_theta=float(get("rope_theta", 50000.0)),
            rope_scaling=rope_scaling_raw or _KIMI_K25_REFERENCE["rope_scaling"],
            max_position_embeddings=get("max_position_embeddings", 131072),
            vocab_size=get("vocab_size", 163840),
            bos_token_id=get("bos_token_id", 1),
            eos_token_id=get("eos_token_id", 163585),
            num_mtp_layers=get("num_mtp_layers", 0),
            hidden_act=get("hidden_act", "silu"),
        )

        # Extract quantization metadata from quantization_config if present
        quant_cfg = getattr(hf_config, "quantization_config", None) or \
                    getattr(text_cfg, "quantization_config", None)
        if quant_cfg is not None:
            cfg.quant_method = getattr(quant_cfg, "quant_type", "compressed-tensors")
            # compressed-tensors stores per-layer configs; we surface the summary
            # INT4 group-32 symmetric for routed experts is the known configuration

        cfg._validate()
        return cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate fields against known-good reference values.

        Critical mismatches raise ValueError.
        Non-critical mismatches emit warnings.
        """
        ref = _KIMI_K25_REFERENCE
        errors = []

        def _check(field_name: str, actual: Any, critical: bool = True) -> None:
            expected = ref.get(field_name)
            if expected is None:
                return
            if field_name in _FLOAT_FIELDS:
                ok = abs(float(actual) - float(expected)) < _FLOAT_TOL
            else:
                ok = actual == expected
            if not ok:
                msg = (
                    f"  {field_name}: got {actual!r}, expected {expected!r}"
                )
                if critical:
                    errors.append(msg)
                else:
                    warnings.warn(
                        f"KimiK25Config non-critical mismatch — {msg}",
                        stacklevel=3,
                    )

        # Critical fields — wrong value → bad model behaviour
        _check("hidden_size", self.hidden_size)
        _check("num_hidden_layers", self.num_hidden_layers)
        _check("num_attention_heads", self.num_attention_heads)
        _check("q_lora_rank", self.q_lora_rank)
        _check("kv_lora_rank", self.kv_lora_rank)
        _check("qk_nope_head_dim", self.qk_nope_head_dim)
        _check("qk_rope_head_dim", self.qk_rope_head_dim)
        _check("v_head_dim", self.v_head_dim)
        _check("n_routed_experts", self.n_routed_experts)
        _check("num_experts_per_tok", self.num_experts_per_tok)
        _check("first_k_dense_replace", self.first_k_dense_replace)
        _check("n_group", self.n_group)
        _check("topk_group", self.topk_group)
        _check("moe_intermediate_size", self.moe_intermediate_size)
        _check("intermediate_size", self.intermediate_size)
        _check("rms_norm_eps", self.rms_norm_eps)     # 1e-5 not 1e-6 — subtle accuracy risk
        _check("vocab_size", self.vocab_size)
        _check("num_mtp_layers", self.num_mtp_layers)

        # Non-critical — important but recoverable
        _check("routed_scaling_factor", self.routed_scaling_factor, critical=False)
        _check("rope_theta", self.rope_theta, critical=False)

        # EOS token ID discrepancy: config says 163585, generation_config says 163586
        if self.eos_token_id not in (163585, 163586):
            warnings.warn(
                f"KimiK25Config: unexpected eos_token_id={self.eos_token_id}. "
                "Expected 163585 (config.json) or 163586 (generation_config.json).",
                stacklevel=3,
            )
        elif self.eos_token_id == 163585:
            warnings.warn(
                "KimiK25Config: eos_token_id=163585 (from config.json). "
                "generation_config.json uses 163586. Using generation_config value is recommended.",
                stacklevel=3,
            )

        if errors:
            raise ValueError(
                "KimiK25Config validation failed — critical field mismatches:\n"
                + "\n".join(errors)
                + "\n\nCheck that the loaded model is moonshotai/Kimi-K2.5."
            )

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return all config fields as a plain dict."""
        import dataclasses
        return dataclasses.asdict(self)

    def dsv3_overrides(self) -> Dict[str, Any]:
        """Return the subset of fields that differ from DeepSeek V3 defaults.

        Use this to patch a DSV3 config object in place of creating a new one.
        These are the fields that must be changed when reusing DSV3 components.
        """
        return {
            "num_attention_heads": self.num_attention_heads,   # 64 (DSV3: 128)
            "n_routed_experts": self.n_routed_experts,         # 384 (DSV3: 256)
            "first_k_dense_replace": self.first_k_dense_replace,  # 1 (DSV3: 3)
            "n_group": self.n_group,                           # 1 (DSV3: 8)
            "topk_group": self.topk_group,                     # 1 (DSV3: 4)
            "routed_scaling_factor": self.routed_scaling_factor,  # 2.827 (DSV3: 2.5)
            "rope_theta": self.rope_theta,                     # 50000 (DSV3: 10000)
            "rope_scaling": self.rope_scaling,                 # YaRN factor=64 (DSV3: different)
            "rms_norm_eps": self.rms_norm_eps,                 # 1e-5 (DSV3: 1e-6)
            "vocab_size": self.vocab_size,                     # 163840 (DSV3: ~129280)
            "num_mtp_layers": self.num_mtp_layers,             # 0 (DSV3: 1)
        }

    def summary(self) -> str:
        """Return a human-readable summary for logging."""
        return (
            f"KimiK25Config(\n"
            f"  layers={self.num_hidden_layers}, heads={self.num_attention_heads}, "
            f"hidden={self.hidden_size}\n"
            f"  experts={self.n_routed_experts} (top-{self.num_experts_per_tok}, "
            f"n_group={self.n_group}, first_dense={self.first_k_dense_replace})\n"
            f"  vocab={self.vocab_size} (padded={self.padded_vocab_size})\n"
            f"  rope_theta={self.rope_theta}, rms_eps={self.rms_norm_eps}\n"
            f"  quant: {self.quant_type} group={self.quant_group_size} "
            f"targets={self.quant_targets}\n"
            f"  experts/device: TG={self.experts_per_device_tg}, "
            f"DUAL={self.experts_per_device_dual}, QUAD={self.experts_per_device_quad}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_kimi_config(
    model_name_or_path: str,
    use_fixture: bool = False,
    trust_remote_code: bool = True,
) -> KimiK25Config:
    """Load and validate a KimiK25Config from HuggingFace or local path.

    Args:
        model_name_or_path: HF model name (e.g. ``"moonshotai/Kimi-K2.5"``) or
                            local directory containing ``config.json``.
        use_fixture:        If True, skip HF loading and return the hardcoded fixture.
                            Useful for offline/CI environments.
        trust_remote_code:  Passed to ``AutoConfig.from_pretrained``.

    Returns:
        Validated KimiK25Config.
    """
    if use_fixture:
        cfg = KimiK25Config.from_fixture()
        import logging
        logging.getLogger(__name__).info("KimiK25Config: using hardcoded fixture (no HF download)")
        return cfg

    try:
        from transformers import AutoConfig
    except ImportError as e:
        raise ImportError(
            "transformers is required to load from HuggingFace. "
            "Install with: pip install transformers>=4.57.1\n"
            "Or use load_kimi_config(use_fixture=True) for offline mode."
        ) from e

    hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    return KimiK25Config.from_hf_config(hf_config)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate KimiK25Config")
    parser.add_argument(
        "--model-path",
        default="moonshotai/Kimi-K2.5",
        help="HF model name or local path. Use --fixture to skip HF download.",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use hardcoded fixture instead of loading from HF",
    )
    args = parser.parse_args()

    print(f"Loading Kimi K2.5 config from: {'fixture' if args.fixture else args.model_path}")

    try:
        cfg = load_kimi_config(args.model_path, use_fixture=args.fixture)
    except Exception as e:
        print(f"[FAIL] Config load/validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("[OK] Config loaded and validated successfully")
    print()
    print(cfg.summary())
    print()
    print("DSV3 overrides (fields that differ from DeepSeek V3):")
    for k, v in cfg.dsv3_overrides().items():
        print(f"  {k}: {v}")

    print()
    print("[PASS] KimiK25Config smoke test complete")
