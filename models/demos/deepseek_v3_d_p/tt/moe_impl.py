# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which MoE a prefill block builds: TtMoe or the MoE block's C++ ops (ttnn.experimental.moe_block).

GLM-5.3 defaults to the MoE block (tt/kimi_k3/moe_block_ffn.py): the gate / shared expert from the checkpoint, the
routed experts from the TtMoe tensorbin cache. Every other model stays on TtMoe (Kimi-K3 opts in with K3_MOE_IMPL).
Order: the caller's ``moe_impl`` ("block" / "ttmoe"), then $MOE_IMPL, then the model's default.

A mesh that runs the block needs room for its persistent words and semaphores in L1_SMALL (MOE_BLOCK_L1_SMALL) and a
fabric payload that holds one combine row (moe_block_fabric_payload). Paths that cannot give it that (the runner, MTP,
the synthetic-weight block tests, the TtMoe cache build) pass moe_impl="ttmoe".
"""

from __future__ import annotations

import os
from typing import Optional

BLOCK = "block"
TTMOE = "ttmoe"
_BLOCK_DEFAULT = ("GLM53Config",)  # model config classes whose MoE is the block by default

# L1_SMALL a mesh needs for the block's persistent words + packed semaphores (4608 B measured on GLM-5.3) on top
# of the model's own L1_SMALL users.
MOE_BLOCK_L1_SMALL = 5888


def moe_impl_of(model_cfg=None, moe_impl: Optional[str] = None) -> str:
    """"block" or "ttmoe" for a model config class (or its name)."""
    choice = moe_impl or os.environ.get("MOE_IMPL")
    if choice:
        assert choice in (BLOCK, TTMOE), f"MOE_IMPL / moe_impl must be '{BLOCK}' or '{TTMOE}', got {choice!r}"
        return choice
    name = model_cfg if isinstance(model_cfg, str) else getattr(model_cfg, "__name__", None)
    return BLOCK if name in _BLOCK_DEFAULT else TTMOE


def moe_block_enabled(model_cfg=None, moe_impl: Optional[str] = None) -> bool:
    """True when a prefill block of this model builds the MoE block instead of TtMoe."""
    return moe_impl_of(model_cfg, moe_impl) == BLOCK


def moe_block_fabric_payload(emb_dim: int) -> int:
    """Fabric payload bytes the block's combine needs: one row (emb_dim bf16) plus its trailer, 64-B aligned."""
    return max(7232, (2 * emb_dim + 32 + 63) // 64 * 64)
