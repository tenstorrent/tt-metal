# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma 26B-A4B-it model + generation config.

Hyperparameters for the DiffusionGemma text backbone (identical to the Gemma-4
26B-A4B MoE) and the discrete-diffusion generation procedure.

Provenance of each field is marked:
  * ``# verified`` — confirmed against the HF ``config.json`` / model card /
    vLLM blog during plan review (see ``plan.md`` §2).
  * ``# TODO(confirm)`` — not surfaced from a primary source yet; the value is a
    Gemma-lineage default or a plan-stated value to reconcile against the real
    ``config.json`` during the #47461 weight-mapping pass.

This is plain config — no torch / ttnn import — so it is importable without the
gated checkpoint, transformers ``diffusion_gemma``, or hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextConfig:
    """DiffusionGemma text backbone (== Gemma-4 26B-A4B MoE)."""

    # --- dimensions -------------------------------------------------------
    num_hidden_layers: int = 30  # verified
    hidden_size: int = 2816  # verified
    num_attention_heads: int = 16  # verified
    num_key_value_heads: int = 8  # verified
    head_dim: int = 256  # verified
    vocab_size: int = 262144  # verified

    # --- MoE --------------------------------------------------------------
    num_experts: int = 128  # verified
    num_experts_per_tok: int = 8  # verified (top-8)
    num_shared_experts: int = 1  # verified via model card ("+1 shared MLP")
    moe_intermediate_size: int = 704  # verified

    # --- attention -------------------------------------------------------
    sliding_window: int = 1024  # verified (sliding layers; full-attn interleaved)
    # Every 6th layer is full_attention: layer_types has full at [5,11,17,23,29]
    # for the 30-layer 26B-A4B (configs/gemma-4-26B-A4B-it/config.json).
    sliding_window_pattern: int = 6  # verified (derived from layer_types)
    # K=V tying applies to full-attn (global) layers ONLY; sliding/local layers
    # keep a real separate V (matters for the bidirectional local-window path,
    # #47462). See gemma4 tt/attention/__init__.py:34.
    attention_k_eq_v: bool = True  # verified (26B-A4B config.json: attention_k_eq_v=True)

    # --- RoPE (dual theta, per layer type) -------------------------------
    rope_theta_full: float = 1.0e6  # verified (full-attention layers)
    rope_theta_sliding: float = 1.0e4  # verified (sliding layers)

    # --- norms / softcap / activation ------------------------------------
    final_logit_softcapping: float = 30.0  # verified
    rms_norm_eps: float = 1.0e-6  # verified (26B-A4B config.json: 1e-06)
    hidden_activation: str = "gelu_pytorch_tanh"  # verified (config.json hidden_activation)

    # --- canvas ----------------------------------------------------------
    canvas_length: int = 256  # verified (block size)

    @classmethod
    def from_hf_config(cls, hf: dict) -> "TextConfig":
        """Build a TextConfig from an HF ``config.json`` dict (gemma4 family).

        Reads ``text_config`` if nested, maps known field names, and derives
        ``sliding_window_pattern`` from ``layer_types``. Absent keys fall back to
        the verified defaults above (e.g. 12B has no MoE fields). Used by the
        #47461 weight-mapping pass to keep this config in sync with the real
        checkpoint and to catch renamed/missing keys.
        """
        text = hf.get("text_config", hf)
        field_map = [
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "sliding_window",
            "rms_norm_eps",
            "final_logit_softcapping",
            "num_experts",
            "moe_intermediate_size",
            "attention_k_eq_v",
            "hidden_activation",
        ]
        kwargs = {k: text[k] for k in field_map if text.get(k) is not None}
        layer_types = text.get("layer_types")
        if layer_types:
            full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
            if len(full) >= 2:
                kwargs["sliding_window_pattern"] = full[1] - full[0]
        return cls(**kwargs)


@dataclass(frozen=True)
class DiffusionConfig:
    """Discrete-diffusion generation procedure (the net-new generation delta).

    Drives the per-step denoise loop (#47463) and the on-device sampling
    primitives (#47472). The reference implementation of these primitives lives
    in ``reference/sampling.py``.
    """

    canvas_length: int = 256  # verified
    max_denoise_steps: int = 48  # verified (model card: max 48 steps)

    # Linear temperature schedule across denoise steps.
    temperature_start: float = 0.8  # verified
    temperature_end: float = 0.4  # verified

    # Entropy-budget acceptance: accept most→least confident until accumulated
    # entropy exceeds this budget; renoise the rest.
    entropy_budget: float = 0.1  # verified (model card: entropy bound = 0.1)

    # Stop when the argmax canvas is stable for this many steps AND mean
    # per-token entropy is below the threshold (or the step cap is reached).
    entropy_stop_threshold: float = 0.1  # TODO(confirm)
    stable_steps_to_halt: int = 1  # TODO(confirm)

    # Noise model: rejected positions are renoised to RANDOM token ids (uniform
    # discrete diffusion), NOT a [MASK] token / absorbing state.
    use_random_token_noise: bool = True  # verified

    # Self-conditioning (extra gated-MLP weights beyond the backbone): active in
    # denoise, zeroed on encoder passes.
    use_self_conditioning: bool = True  # verified


@dataclass(frozen=True)
class VisionConfig:
    """SigLIP-family vision tower (Functional+ / multimodal, #47467).

    Not on the near-term text-first critical path.
    """

    num_hidden_layers: int = 27  # verified
    hidden_size: int = 1152  # verified
    patch_size: int = 16  # verified (config.json patch_size)
    soft_tokens_per_image: int = 280  # verified (config.json vision_soft_tokens_per_image)
    # Variable-resolution token budgets (#47467). "SigLIP" is an author-applied
    # family label; the config model_type is gemma4_vision.
    resolution_token_budgets: tuple[int, ...] = (70, 140, 280, 560, 1120)  # verified


@dataclass(frozen=True)
class DiffusionGemmaConfig:
    """Top-level config aggregating text / diffusion / vision."""

    text: TextConfig = field(default_factory=TextConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    # Max context = canvas_length * max_blocks (256 * 1024 = 256K).
    max_blocks: int = 1024  # verified (256K / 256)

    @property
    def max_context(self) -> int:
        return self.text.canvas_length * self.max_blocks
