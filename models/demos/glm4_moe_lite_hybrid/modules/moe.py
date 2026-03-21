# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""HybridGlm4MoE: TTNNModule wrapping the agentic MoE implementation.

Combines tt-symbiote's module lifecycle with the agentic MoE paths:
- Router: sigmoid + bias + topk with BF16 centering trick
- 4 expert paths: sparse (default), dense decode, dense prefill, packed prefill
- Optional fused_persistent_moe_decode
- Dispatch: reduce (all-reduce) or a2a (all-to-all)
- Shared expert (dense SwiGLU) + routed experts with fused reduce option
"""

from __future__ import annotations

from typing import Any, Optional

import ttnn
from models.demos.glm4_moe_lite.tt.layer_weights import DecoderLayerTTWeights, MoELayerTTWeights
from models.demos.glm4_moe_lite.tt.mlp_decode import dense_mlp_forward, moe_mlp_forward
from models.demos.glm4_moe_lite.tt.moe_tt import (
    Glm4MoeLiteMoERuntime,
    create_moe_runtime,
    moe_dense_experts_forward_decode_tt,
    moe_dense_experts_forward_prefill_tt,
    moe_packed_experts_forward_prefill_tt,
    moe_sparse_experts_forward_tt,
    moe_topk_cpu_reference,
    moe_topk_tt,
)
from models.demos.glm4_moe_lite_hybrid.core.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite_hybrid.core.module import TTNNModule
from models.demos.glm4_moe_lite_hybrid.core.runtime_config import Glm4RuntimeConfig


class HybridGlm4MoERouter(TTNNModule):
    """MoE top-k router with BF16 bias centering trick.

    Wraps moe_topk_tt() from the agentic implementation.
    Supports fallback to CPU reference via cfg.moe_router_impl.
    """

    def __init__(self):
        super().__init__()
        self.hparams: Optional[Glm4MoeLiteHParams] = None
        self.cfg: Optional[Glm4RuntimeConfig] = None
        self.moe_weights: Optional[MoELayerTTWeights] = None

    def configure(
        self,
        hparams: Glm4MoeLiteHParams,
        cfg: Glm4RuntimeConfig,
        moe_weights: MoELayerTTWeights,
    ) -> "HybridGlm4MoERouter":
        self.hparams = hparams
        self.cfg = cfg
        self.moe_weights = moe_weights
        return self

    def forward(
        self,
        x: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Route tokens to experts.

        Returns:
            (topk_weights, topk_indices) both [1,1,T,K] TILE
        """
        assert self.hparams is not None and self.cfg is not None
        assert self.moe_weights is not None

        if self.cfg.moe_router_impl == "cpu":
            return moe_topk_cpu_reference(
                device=self.device,
                x=x,
                moe_w=self.moe_weights,
                hparams=self.hparams,
            )
        return moe_topk_tt(
            x=x,
            moe_w=self.moe_weights,
            hparams=self.hparams,
            compute_kernel_config=self.cfg.mlp_compute_kernel_config(),
        )


class HybridGlm4MoEExperts(TTNNModule):
    """MoE expert execution with 4 selectable paths.

    Wraps the agentic expert implementations:
    1. sparse (default): ttnn.scatter -> moe_expert_token_remap -> sparse_matmul
    2. dense_decode: host-driven per-expert (debug/correctness)
    3. dense_prefill: all tokens through all local experts
    4. packed_prefill: gather -> matmul -> scatter
    """

    def __init__(self):
        super().__init__()
        self.hparams: Optional[Glm4MoeLiteHParams] = None
        self.cfg: Optional[Glm4RuntimeConfig] = None
        self.moe_weights: Optional[MoELayerTTWeights] = None
        self.moe_runtime: Optional[Glm4MoeLiteMoERuntime] = None

    def configure(
        self,
        hparams: Glm4MoeLiteHParams,
        cfg: Glm4RuntimeConfig,
        moe_weights: MoELayerTTWeights,
        moe_runtime: Glm4MoeLiteMoERuntime,
    ) -> "HybridGlm4MoEExperts":
        self.hparams = hparams
        self.cfg = cfg
        self.moe_weights = moe_weights
        self.moe_runtime = moe_runtime
        return self

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        topk_indices: ttnn.Tensor,
        topk_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Execute routed experts on the selected tokens.

        Args:
            hidden_states: [1,1,T,H] TILE
            topk_indices: [1,1,T,K] TILE uint16
            topk_weights: [1,1,T,K] TILE bf16

        Returns:
            routed_output: [1,1,T,H] TILE
        """
        assert self.cfg is not None and self.hparams is not None
        assert self.moe_weights is not None and self.moe_runtime is not None

        tokens = int(hidden_states.shape[2])
        moe_decode_mc = getattr(self.moe_runtime, "decode_memory_config", ttnn.DRAM_MEMORY_CONFIG)
        mlp_ckc = self.cfg.mlp_compute_kernel_config()

        use_dense_decode = self.cfg.moe_experts_impl in {"dense_decode", "dense-decode"} and tokens == 1
        _PACKED_PREFILL_MIN_TOKENS = 33
        use_packed_prefill = self.cfg.moe_packed_prefill and tokens >= _PACKED_PREFILL_MIN_TOKENS
        use_dense_prefill = self.cfg.moe_dense_prefill and not use_packed_prefill and tokens >= 33

        expert_kwargs = dict(
            device=self.device,
            hidden_states=hidden_states,
            topk_expert_indices=topk_indices,
            topk_expert_weights=topk_weights,
            moe_w=self.moe_weights,
            memory_config=moe_decode_mc,
        )

        if use_dense_decode:
            if tokens > 1:
                return moe_dense_experts_forward_prefill_tt(
                    **expert_kwargs,
                    hparams=self.hparams,
                    compute_kernel_config=mlp_ckc,
                    skip_defensive_clones=self.cfg.skip_defensive_clones,
                )
            return moe_dense_experts_forward_decode_tt(
                **expert_kwargs,
                hparams=self.hparams,
                compute_kernel_config=mlp_ckc,
                skip_defensive_clones=self.cfg.skip_defensive_clones,
            )
        elif use_packed_prefill:
            return moe_packed_experts_forward_prefill_tt(
                **expert_kwargs,
                hparams=self.hparams,
                compute_kernel_config=mlp_ckc,
                skip_defensive_clones=self.cfg.skip_defensive_clones,
            )
        elif use_dense_prefill:
            return moe_dense_experts_forward_prefill_tt(
                **expert_kwargs,
                hparams=self.hparams,
                compute_kernel_config=mlp_ckc,
                skip_defensive_clones=self.cfg.skip_defensive_clones,
            )
        else:
            return moe_sparse_experts_forward_tt(
                **expert_kwargs,
                rt=self.moe_runtime,
                skip_defensive_clones=self.cfg.skip_defensive_clones,
                skip_final_reduce=self.cfg.fuse_mlp_moe_reduce and self.cfg.tp_enabled,
            )


class HybridGlm4MoEMLP(TTNNModule):
    """Full MoE MLP: shared expert (dense SwiGLU) + routed experts.

    Wraps both dense_mlp_forward() for layer 0 and moe_mlp_forward() for MoE
    layers, with all the agentic optimizations (DRAM sharding, fused gate+up,
    fused MoE+shared reduce).
    """

    def __init__(self):
        super().__init__()
        self.hparams: Optional[Glm4MoeLiteHParams] = None
        self.cfg: Optional[Glm4RuntimeConfig] = None
        self.weights: Optional[DecoderLayerTTWeights] = None
        self.moe_runtime: Optional[Glm4MoeLiteMoERuntime] = None
        self.is_dense_layer: bool = True

    @classmethod
    def from_torch(cls, torch_mlp, **kwargs) -> "HybridGlm4MoEMLP":
        instance = cls()
        instance._fallback_torch_layer = torch_mlp
        return instance

    def configure(
        self,
        hparams: Glm4MoeLiteHParams,
        cfg: Glm4RuntimeConfig,
        weights: DecoderLayerTTWeights,
        moe_runtime: Optional[Glm4MoeLiteMoERuntime] = None,
        is_dense_layer: bool = True,
    ) -> "HybridGlm4MoEMLP":
        self.hparams = hparams
        self.cfg = cfg
        self.weights = weights
        self.moe_runtime = moe_runtime
        self.is_dense_layer = is_dense_layer
        return self

    def forward(self, x: ttnn.Tensor, **kwargs) -> ttnn.Tensor:
        """Run MLP forward: dense SwiGLU or shared+routed MoE.

        Args:
            x: [1,1,B,hidden] TILE

        Returns:
            mlp_out: [1,1,B,hidden] TILE
        """
        assert self.hparams is not None and self.cfg is not None and self.weights is not None

        if self.is_dense_layer or self.weights.moe is None:
            return dense_mlp_forward(
                x,
                self.weights,
                device=self.device,
                cfg=self.cfg,
            )
        else:
            assert self.moe_runtime is not None, "MoE runtime required for MoE layers"
            return moe_mlp_forward(
                x,
                self.weights,
                device=self.device,
                cfg=self.cfg,
                hparams=self.hparams,
                moe_runtime=self.moe_runtime,
            )


class HybridGlm4MoERuntimeManager:
    """Manages MoE runtime creation and caching per device.

    Creates the Glm4MoeLiteMoERuntime (expert mapping tensors, program configs,
    sparsity block size) once and reuses across layers.
    """

    def __init__(self):
        self._runtime: Optional[Glm4MoeLiteMoERuntime] = None

    def get_or_create(
        self,
        device: Any,
        hparams: Glm4MoeLiteHParams,
    ) -> Glm4MoeLiteMoERuntime:
        if self._runtime is None:
            self._runtime = create_moe_runtime(device=device, hparams=hparams)
        return self._runtime

    @property
    def runtime(self) -> Optional[Glm4MoeLiteMoERuntime]:
        return self._runtime
