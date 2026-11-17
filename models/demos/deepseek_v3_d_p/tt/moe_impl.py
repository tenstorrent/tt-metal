# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which MoE a prefill block builds: TtMoe or the MoE block's C++ ops (ttnn.experimental.moe_block).

A model whose config class is in _BLOCK_DEFAULT defaults to the MoE block (tt/kimi_k3/moe_block_ffn.py: the gate /
shared expert from the checkpoint, the routed experts from the TtMoe tensorbin cache); none does yet - every model
stays on TtMoe unless asked. Order: the caller's ``moe_impl`` ("block" / "ttmoe"), then $MOE_IMPL, then the
model's default.

A mesh that runs the block needs room for its persistent words and semaphores in L1_SMALL (moe_block_l1_small: the
words' slab, on top of the model's own L1_SMALL use) and a fabric payload that holds one combine row
(moe_block_fabric_payload); moe_block_mesh_params gives both for any model (the prefill runner's mesh open). Paths that cannot give it that (the synthetic-weight block tests, the TtMoe cache
build) pass moe_impl="ttmoe".
"""

from __future__ import annotations

import os
from typing import Optional

BLOCK = "block"
TTMOE = "ttmoe"
_BLOCK_DEFAULT: tuple = ()  # model config classes whose MoE is the block by default

# The model's own L1_SMALL use a bank next to the block (its other ops' semaphores and words, allocated before the
# block's slab), where the model's l1_small_size is not that figure:
# Every other model: its l1_small_size.
_MODEL_L1_SMALL: dict = {}
_L1_SMALL_ALIGN = 64  # (the allocator's alignment of the slab and of the model's last buffer)

# The persistent words' slab a core, where the model's word sets need less than MoEBlockConfig.words_slab_bytes:
_MODEL_WORDS_SLAB: dict = {}


def _model_name(model_cfg) -> Optional[str]:
    return model_cfg if isinstance(model_cfg, str) else getattr(model_cfg, "__name__", None)


def moe_block_words_slab(model_cfg=None) -> int:
    """The MoE block's persistent words' slab a core for this model (MoEBlockConfig.words_slab_bytes by default)."""
    import ttnn

    default = int(ttnn.experimental.moe_block.MoEBlockConfig().words_slab_bytes)
    return _MODEL_WORDS_SLAB.get(_model_name(model_cfg), default)


def moe_block_l1_small(model_cfg=None) -> int:
    """L1_SMALL the MoE block holds a bank: its persistent words' slab (moe_block_words_slab, sized so every word set
    of the block fits) plus the allocator's alignment."""
    return moe_block_words_slab(model_cfg) + _L1_SMALL_ALIGN


def model_l1_small_with_block(model_cfg, l1_small_size: int) -> int:
    """L1_SMALL a mesh of this model opens with when the MoE block is its MoE: the model's own use (its l1_small_size,
    or the measured figure where that is not one) plus the block's, at least l1_small_size."""
    own = _MODEL_L1_SMALL.get(_model_name(model_cfg), l1_small_size)
    return max(l1_small_size, own + moe_block_l1_small(model_cfg))


def moe_impl_of(model_cfg=None, moe_impl: Optional[str] = None) -> str:
    """ "block" or "ttmoe" for a model config class (or its name)."""
    name = model_cfg if isinstance(model_cfg, str) else getattr(model_cfg, "__name__", None)
    choice = (
        moe_impl or os.environ.get("MOE_IMPL")
    )
    if choice:
        assert choice in (BLOCK, TTMOE), f"MOE_IMPL / moe_impl must be '{BLOCK}' or '{TTMOE}', got {choice!r}"
        return choice
    return BLOCK if name in _BLOCK_DEFAULT else TTMOE


def moe_block_enabled(model_cfg=None, moe_impl: Optional[str] = None) -> bool:
    """True when a prefill block of this model builds the MoE block instead of TtMoe."""
    return moe_impl_of(model_cfg, moe_impl) == BLOCK


def moe_block_fabric_payload(emb_dim: int) -> int:
    """Fabric payload bytes the block's combine needs: one row (emb_dim bf16) plus its trailer, 64-B aligned."""
    return (2 * emb_dim + 32 + 63) // 64 * 64


def moe_block_mesh_params(
    model_cfg, fabric_payload: int, l1_small_size: int, moe_impl: Optional[str] = None
) -> tuple[int, int]:
    """(fabric payload, L1_SMALL) a mesh of this model opens with: the model's own, raised to what the MoE block needs
    when it is the model's MoE - one combine row of the routed width (Kimi-K3: the latent) and L1_SMALL room for the
    block's persistent words and semaphores. Unchanged on TtMoe. Model-generic: the prefill runner's mesh open."""
    if not moe_block_enabled(model_cfg, moe_impl):
        return fabric_payload, l1_small_size
    width = getattr(model_cfg, "ROUTED_EXPERT_HIDDEN_SIZE", None) or model_cfg.EMB_SIZE
    return max(fabric_payload, moe_block_fabric_payload(width)), model_l1_small_with_block(model_cfg, l1_small_size)
