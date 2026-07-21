# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for the Kimi-K2.6-DFlash *drafter* (block-diffusion speculative-decoding draft model).

Pure-Python (no ttnn/torch import) so both the device module
(``tt/speculative_decoding/dflash/tt_dflash_drafter.py``) and the torch-side HF PCC test
(``tests/speculative_decoding/dflash/test_dflash.py``) can import it.

Values are from the HF checkout ``Kimi-K2.6-DFlash/config.json`` (architecture
``DFlashDraftModel``, a Qwen3-style GQA model). See issue #49586.
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
