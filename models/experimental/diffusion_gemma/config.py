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
    # Interleave pattern of sliding vs full-attention layers. Gemma-3 lineage
    # uses a fixed period (e.g. 5 sliding : 1 full). Reconcile against config.json.
    sliding_window_pattern: int = 6  # TODO(confirm)
    # K=V tying applies to full-attn (global) layers ONLY; sliding/local layers
    # keep a real separate V (matters for the bidirectional local-window path,
    # #47462). See gemma4 tt/attention/__init__.py:34.
    attention_k_eq_v: bool = True  # TODO(confirm) inherited from gemma4 lineage

    # --- RoPE (dual theta, per layer type) -------------------------------
    rope_theta_full: float = 1.0e6  # verified (full-attention layers)
    rope_theta_sliding: float = 1.0e4  # verified (sliding layers)

    # --- norms / softcap -------------------------------------------------
    final_logit_softcapping: float = 30.0  # verified
    rms_norm_eps: float = 1.0e-6  # TODO(confirm) Gemma default

    # --- canvas ----------------------------------------------------------
    canvas_length: int = 256  # verified (block size)


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
