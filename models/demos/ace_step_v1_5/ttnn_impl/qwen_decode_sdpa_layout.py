# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE 5 Hz LM decode: align post-SDPA gather shard with unified residual grid.

After ``nlp_concat_heads_decode`` the stock path runs ``tt_all_gather`` into
``get_attn_gather_users_mem_config(DECODE)``. When that WIDTH spec differs from
``get_attn_wo_output_mem_config(DECODE)`` / residual grid, TTNN inserts extra
L1 reshards before ``o_proj``.

This extends the unified decode shard patch to ``get_attn_gather_users_mem_config``
(decode, no prefetcher). SDPA output must stay HEIGHT-sharded for
``nlp_concat_heads_decode`` — do **not** route ``get_attn_sdpa_output_mem_config`` to WIDTH.
"""

from __future__ import annotations

from typing import Any

from models.demos.ace_step_v1_5.ttnn_impl.qwen_decode_shard import _patch_decode_residual_shard_getter


def ace_step_patch_model_args_sdpa_gather_unified(model_args: Any) -> None:
    """Route ``get_attn_gather_users_mem_config(DECODE)`` through residual grid."""
    if getattr(model_args, "is_galaxy", False):
        return
    if hasattr(model_args, "get_attn_gather_users_mem_config"):
        _patch_decode_residual_shard_getter(
            model_args,
            "get_attn_gather_users_mem_config",
            prefetcher_index=2,
        )


__all__ = ["ace_step_patch_model_args_sdpa_gather_unified"]
