# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which MoE a prefill block builds: TtMoe or the MoE block's C++ ops (ttnn.experimental.moe_block).

GLM-5.3 and Kimi-K3 default to the MoE block (tt/kimi_k3/moe_block_ffn.py): the gate / shared expert (K3: and the
latent pair) from the checkpoint, the routed experts from the TtMoe tensorbin cache. Every other model stays on TtMoe.
Order: the caller's ``moe_impl`` ("block" / "ttmoe"), then $MOE_IMPL (Kimi-K3: $K3_MOE_IMPL first), then the model's
default.

A mesh that runs the block needs room for its persistent words and semaphores in L1_SMALL (MOE_BLOCK_L1_SMALL) and a
fabric payload that holds one combine row (moe_block_fabric_payload); moe_block_mesh_params gives both for any model
(the prefill runner's mesh open). Paths that cannot give it that (the synthetic-weight block tests, the TtMoe cache
build) pass moe_impl="ttmoe".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

BLOCK = "block"
TTMOE = "ttmoe"
_BLOCK_DEFAULT = ("GLM53Config", "KimiK3Config")  # model config classes whose MoE is the block by default

# L1_SMALL a mesh needs for the block's persistent words + packed semaphores (4608 B measured on GLM-5.3, 5376 B on
# Kimi-K3 before the words were packed) on top of the model's own L1_SMALL users.
MOE_BLOCK_L1_SMALL = 5888
# Kimi-K3: the block's persistent words are packed into one L1_SMALL slab (MoEBlockConfig.words_slab_bytes, 2048 B a
# core; the 18 word sets of a 5120-token chunk use 1904 B of it - their per-core maximum - where one tensor a set took
# their 4256 B sum on every core). Measured in the K3 L24 chunked test (torus XY, CI env): the model's own L1_SMALL 832 B
# before the block, 2880 B with it -> 2880 + 64 B margin. L1_SMALL is carved from the top of L1, and MLA's chunked
# attention (ring_mla) puts its static CBs up to a fixed L1 ceiling, so less is better (5408 clashed by 256 B on the
# SC4 DSpark runner, 5888 by 448 B). KimiK3Config.L1_SMALL_SIZE (4096, inherited, not measured) still floors the mesh
# open (moe_block_mesh_params takes the max).
_MOE_BLOCK_L1_SMALL_BY_MODEL = {"KimiK3Config": 2944}


def moe_block_l1_small(model_cfg) -> int:
    """L1_SMALL a mesh of this model opens with when the MoE block is its MoE."""
    name = model_cfg if isinstance(model_cfg, str) else getattr(model_cfg, "__name__", None)
    return _MOE_BLOCK_L1_SMALL_BY_MODEL.get(name, MOE_BLOCK_L1_SMALL)


def moe_impl_of(model_cfg=None, moe_impl: Optional[str] = None) -> str:
    """ "block" or "ttmoe" for a model config class (or its name)."""
    name = model_cfg if isinstance(model_cfg, str) else getattr(model_cfg, "__name__", None)
    choice = (
        moe_impl or (os.environ.get("K3_MOE_IMPL") if name == "KimiK3Config" else None) or os.environ.get("MOE_IMPL")
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
    return max(7232, (2 * emb_dim + 32 + 63) // 64 * 64)


def moe_block_mesh_params(
    model_cfg, fabric_payload: int, l1_small_size: int, moe_impl: Optional[str] = None
) -> tuple[int, int]:
    """(fabric payload, L1_SMALL) a mesh of this model opens with: the model's own, raised to what the MoE block needs
    when it is the model's MoE - one combine row of the routed width (Kimi-K3: the latent) and L1_SMALL room for the
    block's persistent words and semaphores. Unchanged on TtMoe. Model-generic: the prefill runner's mesh open."""
    if not moe_block_enabled(model_cfg, moe_impl):
        return fabric_payload, l1_small_size
    width = getattr(model_cfg, "ROUTED_EXPERT_HIDDEN_SIZE", None) or model_cfg.EMB_SIZE
    return max(fabric_payload, moe_block_fabric_payload(width)), max(l1_small_size, moe_block_l1_small(model_cfg))


def k3_moe_block_device_params(params: dict, moe_impl: Optional[str] = None) -> dict:
    """A Kimi-K3 mesh's device_params with what the MoE block needs when it is K3's MoE: a fabric payload holding one
    combine row of the latent (2 x 3584 + 32 B), L1_SMALL room for its persistent words and semaphores (at least
    MOE_BLOCK_L1_SMALL) and the XY torus fabric (a pinned TT_MESH_GRAPH_DESC_PATH must be a torus-XY one: on the 2D
    mesh fabric the block's first call hangs). Unchanged on TtMoe."""
    if not moe_block_enabled("KimiK3Config", moe_impl):
        return params
    import ttnn
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config

    out = dict(params)
    mgd = os.environ.get("TT_MESH_GRAPH_DESC_PATH", "")
    assert not mgd or "torus_xy" in Path(mgd).name, (
        f"Kimi-K3 on the MoE block runs on the XY torus (on the 2D mesh fabric its first call hangs): "
        f"TT_MESH_GRAPH_DESC_PATH={mgd} is not a torus-XY descriptor - use single_bh_galaxy_torus_xy_graph_descriptor"
        f".textproto, or K3_MOE_IMPL=ttmoe"
    )
    out["fabric_config"] = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    payload = moe_block_fabric_payload(KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE)
    router = out.get("fabric_router_config")
    if router is None or router.max_packet_payload_size_bytes < payload:
        out["fabric_router_config"] = create_fabric_router_config(payload)
    out["l1_small_size"] = max(int(out.get("l1_small_size", 0)), moe_block_l1_small("KimiK3Config"))
    return out
