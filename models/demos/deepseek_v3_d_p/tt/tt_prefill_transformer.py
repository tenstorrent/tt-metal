# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TtPrefillTransformer — multi-layer prefill model for DeepSeek V3.

Composes: embed -> [block x N] -> norm -> lm_head -> sample

Equivalent to the reference Transformer class (models/demos/deepseek_v3/reference/deepseek/model.py:419)
but targeting the TT prefill path with SP+TP parallelism.
"""

from pathlib import Path
from typing import Callable, Optional, Union

import torch
from loguru import logger
from tracy import signpost
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reverse_reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
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
        model_cfg: type,
        state_dict: dict,
        num_layers: int,
        seq_len: int,
        dispatch_buffer_capacity_factor: int = 2,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        sp_axis: int = 0,
        tp_axis: int = 1,
        is_balanced: bool = False,
        padding_side: str = "right",
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path: Optional[Path] = None,
        lm_head_is_column_parallel: bool = False,
        is_chunked: bool = False,
        slot_num: int = 1,
        max_seq_len: Optional[int] = None,
        kv_only_last_layer: bool = False,
        routing_use_l1_small_for_semaphores: bool = False,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.seq_len = seq_len
        self.padding_side = padding_side
        self.is_chunked = is_chunked
        self.num_layers = num_layers
        self.kv_only_last_layer = kv_only_last_layer

        if not state_dict and not (weight_cache_path and weight_cache_path.exists()):
            raise ValueError(
                "TtPrefillTransformer requires weights: pass a non-empty state_dict "
                f"or a weight_cache_path to an existing cache (got {weight_cache_path=})."
            )

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
        # When `kv_only_last_layer`, the last block is constructed with
        # `kv_only=True`: only attn_norm + the KV branch of MLA are built.
        self.layers = []
        for i in range(self.num_layers):
            is_last = i == self.num_layers - 1
            # Get layer weights or empty dict if loading from cache
            layer_state = state_dict["layers"][i] if state_dict.get("layers") else {}
            layer = TtPrefillBlock(
                mesh_device=mesh_device,
                config=config,
                model_cfg=model_cfg,
                state_dict=layer_state,
                layer_idx=i,
                seq_len=seq_len,
                dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                is_balanced=is_balanced,
                gate_fallback_mode=gate_fallback_mode,
                routed_expert_activations_dtype=routed_expert_activations_dtype,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                shared_expert_activations_dtype=shared_expert_activations_dtype,
                shared_expert_weights_dtype=shared_expert_weights_dtype,
                weight_cache_path=weight_cache_path,
                is_chunked=is_chunked,
                slot_num=slot_num,
                layer_num=num_layers,
                max_seq_len=max_seq_len,
                kv_only=kv_only_last_layer and is_last,
                routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
            )
            self.layers.append(layer)

        # --- Final norm + LM Head ---
        # Both are skipped when `kv_only_last_layer`: there is no output token
        # to sample. Migration handles downstream signaling.
        if not kv_only_last_layer:
            self.norm = TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=config.hidden_size,
                torch_weight=state_dict.get("norm_weight"),  # None if cache exists
                epsilon=config.rms_norm_eps,
                cluster_axis=tp_axis,
                num_links=num_links,
                topology=topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix="norm",
            )
            self.lm_head = TtLMHead(
                mesh_device=mesh_device,
                emb_dim=config.hidden_size,
                vocab_size=config.vocab_size,
                torch_weight=state_dict.get("lm_head_weight"),  # None if cache exists
                num_links=num_links,
                topology=topology,
                is_balanced=is_balanced,
                weight_cache_path=weight_cache_path,
                is_column_parallel=lm_head_is_column_parallel,
            )

        # --- RoPE (computed once, reused across all layers) ---
        self.rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

        # Chunked prefill uses the KV-pad-aware indexed rotated path: whole-cache cos/sin/trans built
        # once here and reused for every chunk (only the runtime kv_actual offset varies). seq_len is
        # the per-chunk size and max_seq_len the full per-user cache length.
        self.indexed_rope = (
            self.rope_setup.get_rope_tensors_indexed(
                cache_seq_len_global=max_seq_len if max_seq_len is not None else seq_len,
                chunk_size_global=seq_len,
            )
            if is_chunked
            else None
        )

        self.is_balanced = is_balanced
        self.chunk_order = create_balanced_chunk_order(mesh_device.shape[sp_axis]) if is_balanced else None

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
        number_of_non_padded_tokens: int,
        return_intermediates: bool = False,
        read_profiler: bool = False,
        temperature: Union[float, list[float]] = 0.0,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
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
            on_layer_complete: optional callback invoked by MLA after fill_cache_for_user_().
                Called as on_layer_complete(layer_idx). Used for KV cache
                migration in disaggregated prefill/decode. When set, MLA also zeros
                the padding region of the cache before fill so migration sees valid KV
                + zero padding. When None, no migration or zeroing.

        Returns:
            Tuple of (first_token_id, first_token_prob, intermediates_dict or None)
            - first_token_id: sampled token ID (for first temperature if list provided)
            - first_token_prob: probability of sampled token (for first temperature if list provided)
            - intermediates: dict with keys like "embed", "layer_0", "norm", "lm_head", "first_token"
                            where "first_token" is a list of results for each temperature
                            (None if return_intermediates=False)
        """
        # Chunked prefill ([actual_start, actual_end) set) uses the prebuilt whole-cache indexed rope
        # and writes this chunk at the actual_start offset of user cache_user_id's slot; the single-shot
        # path builds per-call rope for this seq_len. The norm/lm_head/sample tail still runs and a token
        # is returned, but the chunked caller ignores it (the populated cache is the output).
        if actual_start is not None:
            assert self.is_chunked, "actual_start requires the transformer to be built with is_chunked=True"
            rope_tensors = self.indexed_rope
        else:
            rope_tensors = self.rope_setup.get_rope_tensors(self.seq_len)
        intermediates = {} if return_intermediates else None

        h = self.embed(token_ids)  # [1, seq_per_chip, emb_dim/tp]
        h = ttnn.unsqueeze_to_4D(h)  # [1, 1, seq_per_chip, emb_dim/tp]

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates["embed"] = self._to_host(h)

        for i, layer in enumerate(self.layers):
            signpost(f"forward_layer_{i}_start")
            h, _ = layer(
                h,
                rope_tensors,
                kvpe_cache,
                cache_layer_idx=i,
                return_intermediates=return_intermediates,
                on_layer_complete=on_layer_complete,
                actual_start=actual_start,
                actual_end=actual_end,
                cache_user_id=cache_user_id,
            )
            signpost(f"forward_layer_{i}_end")
            if self.kv_only_last_layer and i == len(self.layers) - 1:
                # Last layer was kv-only — KV cache filled, migration callback
                # fired, no hidden state flowing forward. Skip norm + lm_head +
                # sample; no first_token to produce.
                return None, None, intermediates
            if return_intermediates:
                ttnn.synchronize_device(self.mesh_device)
                intermediates[f"layer_{i}"] = self._to_host(h)
            if read_profiler:
                ttnn.ReadDeviceProfiler(self.mesh_device)

        h = self.norm(h)

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates["norm"] = self._to_host(h)

        # LM Head: extract logits for last real token
        logits_host, first_token_logits = self._lm_head_and_extract(h, number_of_non_padded_tokens)

        if return_intermediates:
            intermediates["lm_head"] = logits_host
            intermediates["logits"] = first_token_logits

        # Reorder intermediates if balanced. Skip reordering for logits and lm_head in zigzag mode.
        no_reorder_keys = {"logits", "lm_head"}
        if return_intermediates and self.is_balanced:
            for key, tensor in intermediates.items():
                if key in no_reorder_keys:
                    logger.debug(f"Skipping reordering for non-sequence intermediate {key}")
                    continue
                if isinstance(tensor, torch.Tensor):
                    logger.debug(f"Reordering intermediate {key} with shape {tensor.shape}")
                    intermediates[key] = reverse_reorder_tensor_chunks(tensor, self.chunk_order, seq_dim=-2)
                else:
                    logger.debug(f"Skipping reordering for intermediate {key} of type {type(tensor)}")

        # Sample token(s) from logits
        first_token_id, first_token_prob, sweep_results = self._sample(
            first_token_logits, number_of_non_padded_tokens, temperature
        )

        if return_intermediates:
            intermediates["first_token"] = sweep_results

        return first_token_id, first_token_prob, intermediates

    def _lm_head_and_extract(
        self,
        h: ttnn.Tensor,
        number_of_non_padded_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LM head and extract last-token logits. Topology-aware.

        Args:
            h: Hidden states after final norm
            number_of_non_padded_tokens: Count of real tokens in the sequence

        Returns:
            Tuple of (logits_host, first_token_logits)
        """
        if self.padding_side == "right":
            global_token_id = number_of_non_padded_tokens - 1
        else:  # "left"
            global_token_id = self.seq_len - 1

        logits, (device_id, token_offset) = self.lm_head(h, global_token_id)

        logits_host = self.lm_head.logit_to_host(logits, device_id)
        assert (
            logits_host.shape[-1] == self.lm_head.vocab_size
        ), f"Expected full vocab {self.lm_head.vocab_size}, got {logits_host.shape[-1]} — TP concat may be broken"
        first_token_logits = self.lm_head.select_first_token(logits_host, token_offset)

        logger.debug(f"[TtPrefillTransformer._extract] {logits.shape}")
        logger.debug(f"[TtPrefillTransformer._extract] {logits_host.shape}")
        logger.debug(f"[TtPrefillTransformer._extract] {first_token_logits.shape}")

        return logits_host, first_token_logits

    def _sample(
        self,
        first_token_logits: torch.Tensor,
        number_of_non_padded_tokens: int,
        temperature: Union[float, list[float]],
    ) -> tuple[int, float, list[dict]]:
        """Sample token(s) from extracted logits with temperature sweep.

        Args:
            first_token_logits: Logits for the last real token position
            number_of_non_padded_tokens: Count of real tokens (stored in results)
            temperature: Temperature for sampling (single float or list for sweep)

        Returns:
            Tuple of (first_token_id, first_token_prob, sweep_results)
        """
        temperatures = temperature if isinstance(temperature, list) else [temperature]

        sweep_results = []
        for temp in temperatures:
            token_id, token_prob, top5 = self._sample_token(first_token_logits.clone(), temp)
            sweep_results.append(
                {
                    "number_of_non_padded_tokens": number_of_non_padded_tokens,
                    "token_id": token_id,
                    "probability": token_prob,
                    "temperature": temp,
                    "top5": top5,
                }
            )

        first_token_id = sweep_results[0]["token_id"]
        first_token_prob = sweep_results[0]["probability"]

        logger.debug(f"[TtPrefillTransformer._sample] {first_token_id=}, {first_token_prob=:.4f}")

        return first_token_id, first_token_prob, sweep_results

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
