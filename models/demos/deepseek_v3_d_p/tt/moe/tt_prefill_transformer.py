# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TtPrefillTransformer — multi-layer prefill model for DeepSeek V3.

Composes: embed -> [block x N] -> norm -> lm_head -> sample

Equivalent to the reference Transformer class (models/demos/deepseek_v3/reference/deepseek/model.py:419)
but targeting the TT prefill path with SP+TP parallelism.
"""

import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tracy import signpost
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker


class TtPrefillTransformer(LightweightModule):
    """
    Multi-layer prefill transformer for DeepSeek V3.

    Architecture: embed -> [TtPrefillBlock x num_layers] -> norm -> lm_head -> sample

    State dict keys:
        embed_weight:   torch.Tensor [vocab_size, emb_dim]
        norm_weight:    torch.Tensor [emb_dim]
        layers:         list[dict] — per-layer state dicts for TtPrefillBlock
    """

    @staticmethod
    def check_cache_complete(
        cache_path: Path | None,
        num_layers: int,
        experts_per_chip: int = 8,
        first_k_dense: int = 3,
    ) -> bool:
        """
        Top-level cache completeness check for the full transformer.

        Checks embedding, all blocks (norms + MLA + FFN/MoE), and final norm.
        Replaces the monolithic check_ttnn_cache_complete from cache_utils.py.

        Args:
            cache_path: Path to TTNN weight cache directory
            num_layers: Number of transformer layers
            experts_per_chip: Number of routed experts per chip (default: 8)
            first_k_dense: Number of initial dense (non-MoE) layers (default: 3)

        Returns:
            True if all expected cache files exist, False otherwise
        """
        if not cache_path or not cache_path.exists():
            logger.debug(f"TTNN cache path does not exist: {cache_path}")
            return False

        # Initialize fast cache checker for this directory
        init_checker(cache_path)

        # Embedding
        if not TtParallelEmbedding.check_cache_complete(cache_path):
            return False

        # Per-layer blocks
        for layer_idx in range(num_layers):
            is_dense = layer_idx < first_k_dense
            if not TtPrefillBlock.check_cache_complete(cache_path, layer_idx, is_dense, experts_per_chip):
                return False

        # Final norm
        if not TtDistributedRmsNorm.check_cache_complete(cache_path, "norm"):
            return False

        # LM head
        if not TtLMHead.check_cache_complete(cache_path):
            return False

        logger.info(f"TTNN cache complete at {cache_path} ({num_layers} layers)")
        return True

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: PretrainedConfig,
        state_dict: dict,
        num_layers: int,
        seq_len: int,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        sp_axis: int = 0,
        tp_axis: int = 1,
        is_balanced: bool = False,
        capacity_factor: int = 2,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path: Optional[Path] = None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.seq_len = seq_len

        # Log environment variables that define reference output cache and TTNN weights cache.
        # This is to prevent accidental cache creation at unusal places and fill disk space.
        TT_DS_PREFILL_TTNN_CACHE = os.getenv("TT_DS_PREFILL_TTNN_CACHE", None)
        TT_DS_PREFILL_HOST_REF_CACHE = os.getenv("TT_DS_PREFILL_HOST_REF_CACHE", None)

        logger.debug(f"{TT_DS_PREFILL_TTNN_CACHE=}")
        logger.debug(f"{TT_DS_PREFILL_HOST_REF_CACHE=}")
        if TT_DS_PREFILL_TTNN_CACHE is None:
            logger.error(f"TT_DS_PREFILL_TTNN_CACHE environment variable is not set; export TT_DS_PREFILL_TTNN_CACHE=")
            import pytest

            pytest.fail(f"{TT_DS_PREFILL_TTNN_CACHE=}")
        if TT_DS_PREFILL_HOST_REF_CACHE is None:
            logger.error(
                f"TT_DS_PREFILL_HOST_REF_CACHE environment variable is not set; export TT_DS_PREFILL_HOST_REF_CACHE="
            )
            import pytest

            pytest.fail(f"{TT_DS_PREFILL_HOST_REF_CACHE=}")

        logger.info(f"Building TtPrefillTransformer with {num_layers} layers, seq_len={seq_len}")

        # --- Embedding ---
        self.embed = TtParallelEmbedding(
            mesh_device=mesh_device,
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            torch_weight=state_dict.get("embed_weight"),  # None if cache exists
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            weight_cache_path=weight_cache_path,
        )

        # --- Transformer layers ---
        self.layers = []
        for i in range(num_layers):
            logger.info(f"Building layer {i}/{num_layers}...")
            # Get layer weights or empty dict if loading from cache
            layer_state = state_dict["layers"][i] if state_dict.get("layers") else {}
            layer = TtPrefillBlock(
                mesh_device=mesh_device,
                config=config,
                state_dict=layer_state,
                layer_idx=i,
                seq_len=seq_len,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                is_balanced=is_balanced,
                capacity_factor=capacity_factor,
                gate_fallback_mode=gate_fallback_mode,
                routed_expert_activations_dtype=routed_expert_activations_dtype,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                shared_expert_activations_dtype=shared_expert_activations_dtype,
                shared_expert_weights_dtype=shared_expert_weights_dtype,
                weight_cache_path=weight_cache_path,
            )
            self.layers.append(layer)

        # --- Final norm ---
        self.norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=config.hidden_size,
            torch_weight=state_dict.get("norm_weight"),  # None if cache exists
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix="norm",
        )

        # --- RoPE (computed once, reused across all layers) ---
        self.rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

        # --- LM Head ---
        self.lm_head = TtLMHead(
            mesh_device=mesh_device,
            emb_dim=config.hidden_size,
            vocab_size=config.vocab_size,
            torch_weight=state_dict.get("lm_head_weight"),  # None if cache exists
            num_links=num_links,
            topology=topology,
            is_balanced=is_balanced,
            weight_cache_path=weight_cache_path,
        )

        logger.info(f"TtPrefillTransformer construction complete ({num_layers} layers)")

    def _to_host(self, tt_tensor):
        """Bring SP+TP sharded tensor to host as [1, seq, emb] bfloat16."""
        host = ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        ).to(torch.bfloat16)
        if len(tt_tensor.shape) == 4:
            host = host.squeeze(0)
        return host

    def forward(
        self,
        token_ids: ttnn.Tensor,
        kvpe_cache: ttnn.Tensor,
        return_intermediates: bool = False,
        read_profiler: bool = False,
        temperature: float | list[float] = 0.0,
    ):
        """
        Forward pass: embed -> [block x N] -> norm -> lm_head.

        Args:
            token_ids: [1, 1, seq_len_per_chip] uint32, SP-sharded
            kvpe_cache: externally created KVPE cache [num_layers, 1, seq_len_local, head_dim];
                        each layer writes to its own slot via cache_layer_idx
            return_intermediates: if True, sync + snapshot to host after each stage
            read_profiler: if True, read TTNN profiler after each layer to avoid profiler buffer overflows
            temperature: Temperature for sampling. Can be a single float or list of floats.
                        If list, returns first temperature result but stores all in intermediates.

        Returns:
            Tuple of (first_token_id, first_token_prob, intermediates_dict or None)
            - first_token_id: sampled token ID (for first temperature if list provided)
            - first_token_prob: probability of sampled token (for first temperature if list provided)
            - intermediates: dict with keys like "embed", "layer_0", "norm", "lm_head", "first_token"
                            where "first_token" is a list of results for each temperature
                            (None if return_intermediates=False)
        """
        rope_tensors = self.rope_setup.get_rope_tensors(self.seq_len)
        intermediates = {} if return_intermediates else None

        h = self.embed(token_ids)  # [1, seq_per_chip, emb_dim/tp]
        h = ttnn.unsqueeze_to_4D(h)  # [1, 1, seq_per_chip, emb_dim/tp]

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates["embed"] = self._to_host(h)

        for i, layer in enumerate(self.layers):
            signpost(f"forward_layer_{i}_start")
            h, _ = layer(h, rope_tensors, kvpe_cache, cache_layer_idx=i)
            signpost(f"forward_layer_{i}_end")
            if return_intermediates:
                ttnn.synchronize_device(self.mesh_device)
                intermediates[f"layer_{i}"] = self._to_host(h)
            if read_profiler:
                ttnn.ReadDeviceProfiler(self.mesh_device)

        h = self.norm(h)

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates["norm"] = self._to_host(h)

        # LM Head: compute logits and sample first token
        global_token_id = self.seq_len - 1  # Last token; assuming padding front/left
        logits, (device_id, token_offset) = self.lm_head(h, global_token_id)

        logits_host = self.lm_head.logit_to_host(logits)
        assert (
            logits_host.shape[-1] == self.lm_head.vocab_size
        ), f"Expected full vocab {self.lm_head.vocab_size}, got {logits_host.shape[-1]} — TP concat may be broken"
        first_token_logits = self.lm_head.select_first_token(logits_host, device_id, token_offset)

        # Handle temperature as float or list
        temperatures = temperature if isinstance(temperature, list) else [temperature]

        # Sample for all temperatures
        sweep_results = []
        for temp in temperatures:
            token_id, token_prob, top5 = self._sample_token(first_token_logits.clone(), temp)
            sweep_results.append(
                {
                    "token_id": token_id,
                    "probability": token_prob,
                    "temperature": temp,
                    "device_id": device_id,
                    "token_offset": token_offset,
                    "top5": top5,
                }
            )

        # Return values are for first temperature
        first_token_id = sweep_results[0]["token_id"]
        first_token_prob = sweep_results[0]["probability"]

        logger.debug(f"[TtPrefillTransformer.forward] {logits.shape}")
        logger.debug(f"[TtPrefillTransformer.forward] {logits_host.shape}")
        logger.debug(f"[TtPrefillTransformer.forward] {first_token_logits.shape}")
        logger.debug(
            f"[TtPrefillTransformer.forward] {first_token_id=}, {first_token_prob=:.4f} {device_id=}, {token_offset=}"
        )

        if return_intermediates:
            intermediates["lm_head"] = logits_host
            intermediates["first_token"] = sweep_results

        return first_token_id, first_token_prob, intermediates

    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0) -> tuple[int, float, list]:
        """
        Sample token from logits with temperature scaling.

        Uses Gumbel-softmax trick for sampling (same as DeepSeek reference).
        https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/generate.py

        Args:
            logits: Logits tensor for a single token position
            temperature: Temperature for scaling (0.0 = argmax)

        Returns:
            Tuple of (sampled_token_id, probability, top5_list)
            where top5_list is [{token_id, probability}, ...]
        """
        probs = torch.softmax(logits.float(), dim=-1)

        # Get top-5 tokens (unscaled)
        top5_probs, top5_ids = torch.topk(probs.flatten(), k=5)
        top5 = [{"token_id": tid.item(), "probability": tprob.item()} for tid, tprob in zip(top5_ids, top5_probs)]

        if temperature <= 0:
            # Deterministic argmax — no Gumbel noise
            sampled_id = probs.argmax(dim=-1)
            prob = probs.flatten()[sampled_id.item()].item()
            return sampled_id.item(), prob, top5

        logits = logits / temperature
        probs = torch.softmax(logits.float(), dim=-1)

        # Recompute top-5 with temperature-scaled probs
        top5_probs, top5_ids = torch.topk(probs.flatten(), k=5)
        top5 = [{"token_id": tid.item(), "probability": tprob.item()} for tid, tprob in zip(top5_ids, top5_probs)]

        # Gumbel-softmax trick for sampling (use non-in-place to preserve probs)
        gumbel = probs / torch.empty_like(probs).exponential_(1)
        sampled_id = gumbel.argmax(dim=-1)
        prob = probs.flatten()[sampled_id.item()].item()
        return sampled_id.item(), prob, top5
