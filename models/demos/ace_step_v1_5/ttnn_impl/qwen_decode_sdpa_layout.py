# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE 5 Hz LM decode: post-SDPA gather shard layout.

After ``nlp_concat_heads_decode`` the stock path runs ``tt_all_gather`` into
``get_attn_gather_users_mem_config(DECODE)`` (typically ``[32, 32]`` on the users core
grid). Routing that gather output through ``get_residual_mem_config(DECODE)`` (``[32, 64]``)
made downstream ``o_proj`` matmul warn and ignore the wrong output mem config.

SDPA output must stay HEIGHT-sharded for ``nlp_concat_heads_decode`` — do **not** route
``get_attn_sdpa_output_mem_config`` to WIDTH.
"""

from __future__ import annotations

from typing import Any


def ace_step_patch_model_args_sdpa_gather_unified(model_args: Any) -> None:
    """No-op: keep stock ``get_attn_gather_users_mem_config(DECODE)`` shard spec."""
    del model_args


__all__ = ["ace_step_patch_model_args_sdpa_gather_unified"]
