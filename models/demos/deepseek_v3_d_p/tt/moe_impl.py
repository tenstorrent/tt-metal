# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which MoE a prefill block builds: TtMoe or the MoE block's C++ ops (ttnn.experimental.moe_block).

Kimi-K3 and Kimi-K3.1 run the MoE block (tt/kimi_k3/moe_block_ffn.py: the gate / shared expert and the latent pair
from the checkpoint, the routed experts from the TtMoe tensorbin cache). Every other model runs TtMoe. A caller's
``moe_impl`` ("block" / "ttmoe"), else $MOE_IMPL, overrides the model's default.

A mesh that runs the block opens with:
- L1_SMALL: the model's L1_SMALL_SIZE, its one figure, which holds the block's persistent words' slab
  (moe_block_words_slab) on top of the model's own ops' use;
- a fabric payload that holds one combine row (moe_block_fabric_payload_size / moe_block_device_params).
Paths that do not open their mesh for it (the synthetic-weight block tests, the TtMoe cache build) pass
moe_impl="ttmoe".
"""

from __future__ import annotations

import os
from typing import Optional

import ttnn

moe_block_utils = ttnn._ttnn.operations.experimental.moe_block

BLOCK = "block"
TTMOE = "ttmoe"
_BLOCK_DEFAULT = ("KimiK3Config", "KimiK31Config")  # model config classes whose MoE is the block by default

# The persistent words' slab a core, where the model's word sets need less than MoEBlockConfig.words_slab_bytes. The
# slab sits in L1_SMALL (the model's L1_SMALL_SIZE holds it), at the top of L1 above ring_mla's static CBs (MLA's
# chunked attention places them up to a fixed ceiling, 1564224): every byte here is a byte less for them, so the slab
# is the busiest core's need plus 16 B.
# - Kimi-K3 / K3.1 (the same MoE): 1712 B. Its busiest cores are the head's ports at chunk 5120 (4x4 SP-axis-1 stage
#   and 8x4): the mailbox, the x gather's and routing exchange's words, the padded-forward metadata (64 B aligned), the
#   combine barrier, the dispatch semaphores and the router iteration words - 1696 B.
_MODEL_WORDS_SLAB = {
    "KimiK3Config": 1712,
    "KimiK31Config": 1712,
}


def _model_name(model_cfg) -> Optional[str]:
    return model_cfg if isinstance(model_cfg, str) else getattr(model_cfg, "__name__", None)


def moe_block_words_slab(model_cfg=None) -> int:
    """The MoE block's persistent words' slab a core for this model (MoEBlockConfig.words_slab_bytes by default)."""
    default = int(moe_block_utils.MoEBlockConfig().words_slab_bytes)
    return _MODEL_WORDS_SLAB.get(_model_name(model_cfg), default)


def moe_impl_of(model_cfg=None, moe_impl: Optional[str] = None) -> str:
    """ "block" or "ttmoe" for a model config class (or its name)."""
    choice = moe_impl or os.environ.get("MOE_IMPL")
    if choice:
        assert choice in (BLOCK, TTMOE), f"MOE_IMPL / moe_impl must be '{BLOCK}' or '{TTMOE}', got {choice!r}"
        return choice
    return BLOCK if _model_name(model_cfg) in _BLOCK_DEFAULT else TTMOE


def moe_block_enabled(model_cfg=None, moe_impl: Optional[str] = None) -> bool:
    """True when a prefill block of this model builds the MoE block instead of TtMoe."""
    return moe_impl_of(model_cfg, moe_impl) == BLOCK


def moe_block_fabric_payload(emb_dim: int) -> int:
    """Fabric payload bytes the block's combine needs: one row (emb_dim bf16) plus its trailer, 64-B aligned."""
    return (2 * emb_dim + 32 + 63) // 64 * 64


def moe_block_fabric_payload_size(model_cfg, fabric_payload: int, moe_impl: Optional[str] = None) -> int:
    """The fabric payload a mesh of this model opens with: the model's own, raised to one combine row of the routed
    width (Kimi-K3: the latent) when the MoE block is the model's MoE."""
    if not moe_block_enabled(model_cfg, moe_impl):
        return fabric_payload
    width = getattr(model_cfg, "ROUTED_EXPERT_HIDDEN_SIZE", None) or model_cfg.EMB_SIZE
    return max(fabric_payload, moe_block_fabric_payload(width))


def moe_block_device_params(model_cfg, params: dict, moe_impl: Optional[str] = None) -> dict:
    """A device_params dict (a mesh fixture's) with its fabric router's payload raised to moe_block_fabric_payload_size.
    L1_SMALL and the fabric config stay the caller's."""
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config

    router = params.get("fabric_router_config")
    own = router.max_packet_payload_size_bytes if router is not None else 0
    payload = moe_block_fabric_payload_size(model_cfg, own, moe_impl)
    if payload == own:
        return params
    return {**params, "fabric_router_config": create_fabric_router_config(payload)}
