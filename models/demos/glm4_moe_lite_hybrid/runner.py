# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""HybridGlm4Runner: top-level runner for GLM-4.7-Flash.

Loads a HuggingFace model, replaces modules with TTNN equivalents, manages
weight conversion and device placement, and exposes prefill/decode/generate APIs.

Usage:
    runner = HybridGlm4Runner.from_pretrained(
        "zai-org/GLM-4.7-Flash",
        device=mesh_device,
    )
    output_ids = runner.generate(input_ids, max_new_tokens=128)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

import ttnn
from models.demos.glm4_moe_lite.tt.decoder_layer_tt import (
    prepare_decode_rope_and_positions_tt,
    run_decoder_layer_decode_one_step_update_cache_tt,
)
from models.demos.glm4_moe_lite.tt.layer_weights import DecoderLayerTTWeights, convert_decoder_layer_weights
from models.demos.glm4_moe_lite.tt.tt_embedding import convert_embedding_weight_to_tt, run_tt_embedding
from models.demos.glm4_moe_lite.tt.weights import load_glm_lazy_state_dict
from models.demos.glm4_moe_lite_hybrid.core.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite_hybrid.core.module import TTNNModule
from models.demos.glm4_moe_lite_hybrid.core.module_replacement import register_module_replacement
from models.demos.glm4_moe_lite_hybrid.core.runtime_config import Glm4RuntimeConfig
from models.demos.glm4_moe_lite_hybrid.modules.decode_trace import DecodeTraceSamplingState
from models.demos.glm4_moe_lite_hybrid.modules.kvpe_cache import CompressedKVPECache, CompressedKVPECacheConfig
from models.demos.glm4_moe_lite_hybrid.modules.moe import HybridGlm4MoERuntimeManager


def _is_mesh_device(device: Any) -> bool:
    return device.__class__.__name__ == "MeshDevice"


@dataclass
class HybridRunnerConfig:
    """Configuration for the hybrid runner."""

    snapshot_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    max_seq_len: int = 8192
    max_batch_size: int = 32
    max_num_blocks: int = 2048
    kv_cache_block_size: int = 64
    enable_moe: bool = True
    enable_mtp: bool = False
    enable_fused_kv_branch: bool = False


class HybridGlm4Runner:
    """Top-level runner for the hybrid GLM-4.7-Flash implementation.

    Orchestrates:
    1. HuggingFace model loading
    2. Module replacement with TTNN equivalents
    3. Weight conversion and device placement
    4. Compressed KVPE cache creation
    5. MoE runtime initialization
    6. Prefill and decode execution
    7. Optional MTP speculative decoding
    """

    def __init__(
        self,
        device: Any,
        hparams: Glm4MoeLiteHParams,
        cfg: Glm4RuntimeConfig,
        runner_config: HybridRunnerConfig,
    ):
        self.device = device
        self.hparams = hparams
        self.cfg = cfg
        self.runner_config = runner_config

        self.kvpe_cache: Optional[CompressedKVPECache] = None
        self.moe_runtime_mgr = HybridGlm4MoERuntimeManager()
        self.layer_weights: list[Optional[DecoderLayerTTWeights]] = [None] * int(hparams.num_hidden_layers)
        self.embed_weight: Optional[ttnn.Tensor] = None
        self.lm_head_weight: Optional[ttnn.Tensor] = None
        self.final_norm: Any = None

        self._trace_states: dict[int, DecodeTraceSamplingState] = {}
        self._rope: Optional[dict[str, ttnn.Tensor]] = None
        self._replaced_modules: dict[str, TTNNModule] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Any,
        runner_config: Optional[HybridRunnerConfig] = None,
    ) -> "HybridGlm4Runner":
        """Load a HuggingFace model and create the hybrid runner.

        This is the primary entry point. It:
        1. Loads the HF config and creates Glm4MoeLiteHParams
        2. Parses runtime config from env vars
        3. Creates the runner instance
        4. Optionally loads and converts weights
        """
        from transformers import AutoConfig

        if runner_config is None:
            runner_config = HybridRunnerConfig()

        hf_config = AutoConfig.from_pretrained(model_name_or_path)
        hparams = Glm4MoeLiteHParams.from_hf_config(hf_config)
        hparams.validate()

        cfg = Glm4RuntimeConfig.from_env(device=device)

        runner = cls(
            device=device,
            hparams=hparams,
            cfg=cfg,
            runner_config=runner_config,
        )
        return runner

    def load_weights(self, snapshot_dir: Optional[str] = None) -> None:
        """Load and convert all model weights from a snapshot directory."""
        snap = snapshot_dir or self.runner_config.snapshot_dir
        if snap is None:
            raise ValueError("No snapshot_dir provided. Set runner_config.snapshot_dir or pass it directly.")

        state = load_glm_lazy_state_dict(snap)
        cache_dir = self.runner_config.cache_dir

        for layer_idx in range(int(self.hparams.num_hidden_layers)):
            is_dense = layer_idx < int(self.hparams.first_k_dense_replace)
            self.layer_weights[layer_idx] = convert_decoder_layer_weights(
                device=self.device,
                state=state,
                layer_idx=layer_idx,
                hparams=self.hparams,
                cache_dir=Path(cache_dir) if cache_dir else None,
                enable_moe=self.runner_config.enable_moe and not is_dense,
                skip_fused_kv_branch=not self.runner_config.enable_fused_kv_branch,
            )

        self.embed_weight = convert_embedding_weight_to_tt(
            device=self.device,
            state=state,
        )

    def init_kvpe_cache(self, batch_size: int = 1) -> CompressedKVPECache:
        """Initialize the compressed KVPE paged attention cache."""
        config = CompressedKVPECacheConfig(
            block_size=self.runner_config.kv_cache_block_size,
            max_num_blocks=self.runner_config.max_num_blocks,
            num_layers=int(self.hparams.num_hidden_layers),
        )
        self.kvpe_cache = CompressedKVPECache(self.hparams, config)
        self.kvpe_cache.to_device(self.device, batch_size=batch_size)
        return self.kvpe_cache

    def init_moe_runtime(self) -> None:
        """Initialize the MoE runtime (expert mapping, program configs)."""
        self.moe_runtime_mgr.get_or_create(self.device, self.hparams)

    def decode_step(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        batch: int,
        profile: dict[str, float] | None = None,
    ) -> ttnn.Tensor:
        """Run one full decode step through all layers.

        Args:
            token_ids: [B] int32 token IDs
            positions: [B] int32 position indices
            batch: batch size

        Returns:
            hidden_states after all layers: [1,1,B,hidden]
        """
        assert self.kvpe_cache is not None, "Call init_kvpe_cache() first"
        assert self.embed_weight is not None, "Call load_weights() first"

        x = run_tt_embedding(
            device=self.device,
            token_ids=token_ids,
            tt_weight=self.embed_weight,
        )
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        hidden = int(self.hparams.hidden_size)
        x = ttnn.reshape(x, (1, 1, batch, hidden))

        tt_positions, cos_batch, sin_batch = prepare_decode_rope_and_positions_tt(
            device=self.device,
            rope=self._rope,
            positions=positions,
        )

        for layer_idx in range(int(self.hparams.num_hidden_layers)):
            w = self.layer_weights[layer_idx]
            assert w is not None

            x = run_decoder_layer_decode_one_step_update_cache_tt(
                device=self.device,
                x_embed_tok=x,
                tt_positions=tt_positions,
                page_table_tt=self.kvpe_cache.page_table,
                kvpe_cache=self.kvpe_cache.get_cache(layer_idx),
                cos_batch=cos_batch,
                sin_batch=sin_batch,
                trans_matrix=self._rope["trans_matrix"],
                cos_decode=None,
                sin_decode=None,
                trans_decode=None,
                rope_sharded_cfg=None,
                w=w,
                hparams=self.hparams,
                moe_runtime=self.moe_runtime_mgr.runtime,
                profile=profile,
                use_decode_rope=False,
            )

        return x

    def get_module_replacement_map(self) -> dict:
        """Build the HuggingFace module -> TTNN module replacement mapping.

        This is the bridge between HF's model tree and the hybrid TTNN modules.
        Call register_module_replacement() with this map on a loaded HF model.
        """
        return {
            # Subclasses should populate this with their specific HF class mappings.
            # Example:
            # Glm4MoeLiteAttention: HybridGlm4MLA,
            # Glm4MoeLiteMLP: HybridGlm4MoEMLP,
        }

    def register_replacements(self, model) -> dict[str, TTNNModule]:
        """Register TTNN module replacements on a HuggingFace model.

        Args:
            model: HuggingFace CausalLM model

        Returns:
            Dict of module_name -> TTNNModule for all replaced modules
        """
        class_map = self.get_module_replacement_map()
        if not class_map:
            return {}
        self._replaced_modules = register_module_replacement(
            model,
            class_map,
            model_config={"hparams": self.hparams, "cfg": self.cfg},
        )
        for name, module in self._replaced_modules.items():
            module.set_device_recursive(self.device)
        return self._replaced_modules

    def setup_all_modules(self) -> None:
        """Preprocess and move weights for all replaced TTNN modules."""
        for module in self._replaced_modules.values():
            module.preprocess_weights()
            module.move_weights_to_device()

    @property
    def perf_summary(self) -> dict:
        """Return a summary dict suitable for logging performance metrics."""
        return {
            "model": "GLM-4.7-Flash",
            "implementation": "hybrid",
            "num_layers": int(self.hparams.num_hidden_layers),
            "hidden_size": int(self.hparams.hidden_size),
            "num_experts": int(self.hparams.n_routed_experts),
            "experts_per_tok": int(self.hparams.num_experts_per_tok),
            "kv_cache_type": "compressed_kvpe",
            "kv_cache_dim": int(self.hparams.kv_lora_rank + self.hparams.qk_rope_head_dim),
            "moe_experts_impl": self.cfg.moe_experts_impl,
            "dram_sharded_mlp": self.cfg.dram_sharded_mlp,
            "dram_sharded_attn": self.cfg.dram_sharded_attn,
            "tp_enabled": self.cfg.tp_enabled,
            "tp_size": self.cfg.tp_size,
        }
