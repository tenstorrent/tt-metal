# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""HybridGlm4MLA: TTNNModule wrapping the agentic three-phase MLA attention.

Combines tt-symbiote's module lifecycle (from_torch, preprocess_weights,
move_weights_to_device) with the agentic attention decode path (kv_cache_update,
q_projection, flash_mla_and_output) including:
- Fused KV branch kernel (GLMKVCacheBranch) for batch=1
- Compressed KVPE cache (576-dim, BF8)
- Head-parallel kv_b2 path
- Optional Q sharding
- Configurable MLA scale mode
"""

from __future__ import annotations

from typing import Any, Optional

import ttnn
from models.demos.glm4_moe_lite.tt.attention_decode import flash_mla_and_output, kv_cache_update, q_projection
from models.demos.glm4_moe_lite.tt.decoder_layer_tt import _shard_kvpe_update_tensor
from models.demos.glm4_moe_lite.tt.layer_weights import DecoderLayerTTWeights
from models.demos.glm4_moe_lite_hybrid.core.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite_hybrid.core.module import TTNNModule
from models.demos.glm4_moe_lite_hybrid.core.runtime_config import Glm4RuntimeConfig


class HybridGlm4MLA(TTNNModule):
    """TTNN-accelerated Multi-Latent Attention for GLM-4.7-Flash.

    Wraps the agentic three-phase attention decode path in a TTNNModule with
    HuggingFace integration via from_torch(). Supports both prefill and decode.
    """

    def __init__(self):
        super().__init__()
        self.hparams: Optional[Glm4MoeLiteHParams] = None
        self.cfg: Optional[Glm4RuntimeConfig] = None
        self.weights: Optional[DecoderLayerTTWeights] = None
        self.layer_idx: int = 0
        self._rope_setup: Optional[dict] = None
        self._fused_kv_branch_fn: Any = None

    @classmethod
    def from_torch(cls, torch_attn, **kwargs) -> "HybridGlm4MLA":
        instance = cls()
        instance._fallback_torch_layer = torch_attn
        return instance

    def configure(
        self,
        hparams: Glm4MoeLiteHParams,
        cfg: Glm4RuntimeConfig,
        weights: DecoderLayerTTWeights,
        layer_idx: int,
    ) -> "HybridGlm4MLA":
        self.hparams = hparams
        self.cfg = cfg
        self.weights = weights
        self.layer_idx = layer_idx
        return self

    def setup_rope(
        self,
        cos_batch: ttnn.Tensor,
        sin_batch: ttnn.Tensor,
        trans_matrix: ttnn.Tensor,
    ) -> None:
        self._rope_setup = {
            "cos_batch": cos_batch,
            "sin_batch": sin_batch,
            "trans_matrix": trans_matrix,
        }

    def setup_fused_kv_branch(self, fused_kv_branch_fn: Any) -> None:
        self._fused_kv_branch_fn = fused_kv_branch_fn

    def forward_decode(
        self,
        x: ttnn.Tensor,
        *,
        batch: int,
        kvpe_cache: ttnn.Tensor,
        page_table_tt: ttnn.Tensor,
        tt_positions: ttnn.Tensor,
        positions_main_tt: ttnn.Tensor | None = None,
        positions_draft_tt: ttnn.Tensor | None = None,
        profile: dict[str, float] | None = None,
    ) -> ttnn.Tensor:
        """Run one decode step: KV update -> Q projection -> FlashMLA -> output."""
        assert self.hparams is not None and self.cfg is not None and self.weights is not None
        assert self._rope_setup is not None, "Call setup_rope() before forward_decode()"

        rope = self._rope_setup
        w = self.weights

        q_a = kv_cache_update(
            device=self.device,
            x=x,
            w=w,
            hparams=self.hparams,
            cfg=self.cfg,
            batch=batch,
            cos_batch=rope["cos_batch"],
            sin_batch=rope["sin_batch"],
            trans_matrix=rope["trans_matrix"],
            kvpe_cache=kvpe_cache,
            page_table_tt=page_table_tt,
            tt_positions=tt_positions,
            positions_main_tt=positions_main_tt,
            positions_draft_tt=positions_draft_tt,
            use_decode_rope=False,
            rope_decode_fn=None,
            shard_kvpe_fn=_shard_kvpe_update_tensor,
            fused_kv_branch_fn=self._fused_kv_branch_fn,
            profile=profile,
        )

        q_kvpe = q_projection(
            device=self.device,
            x=x,
            w=w,
            hparams=self.hparams,
            cfg=self.cfg,
            batch=batch,
            cos_batch=rope["cos_batch"],
            sin_batch=rope["sin_batch"],
            trans_matrix=rope["trans_matrix"],
            q_a_from_kv=q_a,
            use_decode_rope=False,
            rope_decode_fn=None,
            profile=profile,
        )

        attn_out = flash_mla_and_output(
            device=self.device,
            q_kvpe=q_kvpe,
            w=w,
            hparams=self.hparams,
            cfg=self.cfg,
            batch=batch,
            kvpe_cache=kvpe_cache,
            page_table_tt=page_table_tt,
            tt_positions=tt_positions,
            profile=profile,
        )

        return attn_out

    def forward(self, *args, **kwargs):
        return self.forward_decode(*args, **kwargs)
