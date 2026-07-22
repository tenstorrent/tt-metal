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
from models.demos.deepseek_v3_d_p.tt.mla.indexer import resolve_has_indexer
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
        first_layer_idx: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        kv_only_last_layer: bool = False,
    ) -> bool:
        """
        Top-level cache completeness check for the full transformer.

        Checks embedding, all blocks (norms + MLA + FFN/MoE), and final norm.
        Replaces the monolithic check_ttnn_cache_complete from cache_utils.py.

        Args:
            cache_path: Path to TTNN weight cache directory
            num_layers: Number of transformer layers built by this instance
            experts_per_chip: Number of routed experts per chip (default: 8)
            first_k_dense: Number of initial dense (non-MoE) layers (default: 3)
            first_layer_idx: Global index of this instance's first layer. Non-zero
                for a pipeline-parallel rank owning a layer slice; block cache keys
                are global, so dense/MoE selection must use the global index.
            is_first_rank / is_last_rank: a pipeline-parallel rank builds the
                embedding only on the first rank and the final norm + LM head only
                on the last, so check only the weights it actually loads. Both True
                for single-rank.

        Returns:
            True if all expected cache files exist, False otherwise
        """
        if not cache_path or not cache_path.exists():
            logger.debug(f"TTNN cache path does not exist: {cache_path}")
            return False

        # Initialize fast cache checker for this directory
        init_checker(cache_path)

        # Embedding (first rank only)
        if is_first_rank and not TtParallelEmbedding.check_cache_complete(cache_path):
            return False

        # Per-layer blocks — cache keys are global, so index globally.
        for local_idx in range(num_layers):
            layer_idx = first_layer_idx + local_idx
            is_dense = layer_idx < first_k_dense
            if not TtPrefillBlock.check_cache_complete(cache_path, layer_idx, is_dense, experts_per_chip):
                return False

        # Final norm + LM head: only the last rank that emits a token loads these
        # (skipped for a kv_only last layer and for non-last pipeline ranks).
        if is_last_rank and not kv_only_last_layer:
            if not TtDistributedRmsNorm.check_cache_complete(cache_path, "norm"):
                return False
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
        slot_num: int = 1,
        max_seq_len: Optional[int] = None,
        kv_only_last_layer: bool = False,
        routing_use_l1_small_for_semaphores: bool = False,
        first_layer_idx: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.seq_len = seq_len
        self.padding_side = padding_side
        self.num_layers = num_layers
        self.kv_only_last_layer = kv_only_last_layer
        # Pipeline-parallel slicing. A rank owns layers [first_layer_idx, first_layer_idx+num_layers),
        # builds the embedding only on the first rank, and the norm + LM head only on the last rank that
        # also emits a token (is_last_rank and not kv_only_last_layer). All default so a single-rank
        # instance builds the whole model unchanged.
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        # GLM-5.2 indexer reuse: global per-layer full/shared map (None on models without it -> every
        # layer computes its own indexer, i.e. current behavior). first_layer_idx maps this rank's
        # local layer slice onto the global map.
        self.first_layer_idx = first_layer_idx
        self.indexer_types = getattr(config, "indexer_types", None)

        if not state_dict and not (weight_cache_path and weight_cache_path.exists()):
            raise ValueError(
                "TtPrefillTransformer requires weights: pass a non-empty state_dict "
                f"or a weight_cache_path to an existing cache (got {weight_cache_path=})."
            )

        logger.info(f"Building TtPrefillTransformer with {num_layers} layers, seq_len={seq_len}")

        # --- Embedding (first rank only) ---
        self.embed = (
            TtParallelEmbedding(
                mesh_device=mesh_device,
                vocab_size=config.vocab_size,
                emb_dim=config.hidden_size,
                torch_weight=state_dict.get("embed_weight"),  # None if cache exists
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                weight_cache_path=weight_cache_path,
            )
            if is_first_rank
            else None
        )

        # --- Transformer layers ---
        # layer_idx is the GLOBAL index (drives weight cache keys + dense/MoE selection);
        # cache_layer_idx in forward is the LOCAL slot. layer_num is this instance's slice
        # length so the block's flat KV slot (cache_user_id * layer_num + cache_layer_idx)
        # matches the per-rank cache sized to num_layers. With kv_only_last_layer, the last block is
        # built kv_only=True (only attn_norm + the KV branch of MLA).
        self.layers = []
        for local_idx in range(num_layers):
            layer_idx = first_layer_idx + local_idx
            is_last = local_idx == num_layers - 1
            logger.info(f"Building layer {local_idx}/{num_layers} (global idx {layer_idx})...")
            # Get layer weights or empty dict if loading from cache. state_dict, when
            # provided, holds this instance's slice (local indexing).
            layer_state = state_dict["layers"][local_idx] if state_dict.get("layers") else {}
            layer = TtPrefillBlock(
                mesh_device=mesh_device,
                config=config,
                model_cfg=model_cfg,
                state_dict=layer_state,
                layer_idx=layer_idx,
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
                slot_num=slot_num,
                layer_num=num_layers,
                max_seq_len=max_seq_len,
                kv_only=kv_only_last_layer and is_last,
                routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
            )
            self.layers.append(layer)

        # --- Final norm (last token-emitting rank only) ---
        # Built iff is_last_rank and not kv_only_last_layer: a kv_only last layer (chunked prefill)
        # emits no token, and non-last pipeline ranks forward the hidden state — both skip the tail.
        build_tail = is_last_rank and not kv_only_last_layer
        self.norm = (
            TtDistributedRmsNorm(
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
            if build_tail
            else None
        )

        # --- RoPE (computed once, reused across all layers) ---
        self.rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

        # Prefill is chunked-only: the KV-pad-aware indexed rotated path uses whole-cache cos/sin/trans
        # built once here and reused for every chunk (only the runtime kv_actual offset varies). seq_len
        # is the per-chunk size and max_seq_len the full per-user cache length. Both dense and sparse use
        # it (sparse folds its single full-seq chunk onto the same block-cyclic path).
        self._has_indexer = resolve_has_indexer(config)
        self.indexed_rope = self.rope_setup.get_rope_tensors_indexed(
            cache_seq_len_global=max_seq_len if max_seq_len is not None else seq_len,
            chunk_size_global=seq_len,
        )

        # --- LM Head (last token-emitting rank only) ---
        self.lm_head = (
            TtLMHead(
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
            if build_tail
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
        actual_isl: int,
        return_intermediates: bool = False,
        read_profiler: bool = False,
        temperature: Union[float, list[float]] = 0.0,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
        index_kv_cache: Optional[ttnn.Tensor] = None,
    ):
        """
        Forward pass: [embed] -> [block x N] -> [norm -> lm_head -> sample].

        Pipeline-parallel ranks run a slice of this: the embedding runs only on the
        first rank and the norm/LM-head/sample tail only on the last, so the input
        and output are dual-mode (see Args/Returns).

        Args:
            token_ids: on the first rank, [1, 1, seq_len_per_chip] uint32 SP-sharded
                token IDs to embed; on a non-first rank, the [1, 1, seq_per_chip,
                emb_dim/tp] hidden-state activation handed over from the previous rank.
            kvpe_cache: externally created KVPE cache [num_layers, 1, seq_len_local, head_dim];
                        each layer writes to its own slot via cache_layer_idx
            index_kv_cache: sparse-DSA (v3.2 / GLM) — the caller-owned, layer-stacked block-cyclic indexer
                        key cache [num_users * num_layers, 1, T, D_idx] (SP-sharded on the seq axis), same
                        ownership as kvpe_cache. Required for EVERY sparse forward — chunked AND single-shot
                        (folded onto the block-cyclic path); the indexer never self-allocates it. None only
                        for dense (non-sparse) variants.
            return_intermediates: if True, sync + snapshot to host after each stage
            read_profiler: if True, read TTNN profiler after each layer to avoid profiler buffer overflows
            temperature: Temperature for sampling. Can be a single float or list of floats.
                        If list, returns first temperature result but stores all in intermediates.
            on_layer_complete: optional callback fired by each block after MLA writes the chunk's KV.
                Called as on_layer_complete(layer_idx). Used for KV cache
                migration in disaggregated prefill/decode. When set, the block also zeros
                the cache pad window past actual_end before firing, so migration sees valid KV
                + zero padding. When None, no migration or zeroing.

        Returns:
            On a non-last rank: the hidden-state activation tensor to hand to the next
            rank (no token — the tail did not run).

            On the last rank (and single-rank): a tuple of
            (first_token_id, first_token_prob, intermediates_dict or None)
            - first_token_id: sampled token ID (for first temperature if list provided)
            - first_token_prob: probability of sampled token (for first temperature if list provided)
            - intermediates: dict with keys like "embed", "layer_0", "norm", "lm_head", "first_token"
                            where "first_token" is a list of results for each temperature
                            (None if return_intermediates=False)
        """
        # Prefill is chunked-only: both dense and sparse use the prebuilt whole-cache indexed rope,
        # writing this chunk at the actual_start offset of user cache_user_id's slot. Dense requires
        # actual_start; sparse folds its single full-seq chunk onto the same path (offset 0) and accepts
        # None. The norm/lm_head/sample tail still runs and a token is returned, but the chunked caller
        # ignores it (the populated cache is the output).
        assert (
            self._has_indexer or actual_start is not None
        ), "dense chunked prefill requires actual_start; sparse accepts None (folded to offset 0)"
        rope_tensors = self.indexed_rope
        intermediates = {} if return_intermediates else None

        if self.is_first_rank:
            h = self.embed(token_ids)  # [1, seq_per_chip, emb_dim/tp]
            h = ttnn.unsqueeze_to_4D(h)  # [1, 1, seq_per_chip, emb_dim/tp]
            if return_intermediates:
                ttnn.synchronize_device(self.mesh_device)
                intermediates["embed"] = self._to_host(h)
        else:
            # token_ids carries the upstream rank's hidden-state activation, already
            # [1, 1, seq_per_chip, emb_dim/tp]. No embedding on this rank.
            h = token_ids

        # GLM-5.2 reuse: hold the most recent "full" layer's top-k indices and inject them into the
        # following "shared" layers. reuse=False (no indexer_types) leaves the call + 2-tuple return
        # exactly as before.
        reuse = self.indexer_types is not None
        # reuse seeds from the first "full" layer within this forward; a stack starting on a "shared"
        # layer has no prior indices (pipeline-parallel would need them threaded in from the prior rank).
        if reuse:
            assert (
                self.indexer_types[self.first_layer_idx] == "full"
            ), f"first layer {self.first_layer_idx} must be 'full' to seed indexer reuse, got '{self.indexer_types[self.first_layer_idx]}'"
        indexer_indices = None
        for i, layer in enumerate(self.layers):
            signpost(f"forward_layer_{i}_start")
            mode = self.indexer_types[self.first_layer_idx + i] if reuse else "full"
            inject = indexer_indices if (reuse and mode == "shared") else None
            ret = layer(
                h,
                rope_tensors,
                kvpe_cache,
                cache_layer_idx=i,
                return_intermediates=return_intermediates,
                on_layer_complete=on_layer_complete,
                actual_start=actual_start,
                actual_end=actual_end,
                cache_user_id=cache_user_id,
                actual_isl=actual_isl,
                padding_side=self.padding_side,
                indexer_indices=inject,
                return_indexer_indices=reuse,
                index_kv_cache=index_kv_cache,
            )
            if reuse:
                h, _, new_idx = ret
                if mode == "full":
                    if indexer_indices is not None:
                        ttnn.deallocate(indexer_indices)
                    indexer_indices = new_idx
            else:
                h, _ = ret
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
        # GLM-5.2 reuse: free the last full layer's held top-k indices after the final layer.
        if reuse and indexer_indices is not None:
            ttnn.deallocate(indexer_indices)

        # Non-last pipeline ranks stop here: the layer slice's output activation is
        # handed to the next rank, which continues from this hidden state. The norm /
        # LM-head / sample tail (and its weights) live only on the last rank.
        if not self.is_last_rank:
            return h

        h = self.norm(h)

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates["norm"] = self._to_host(h)

        # LM Head: extract logits for last real token
        logits_host, first_token_logits = self._lm_head_and_extract(h, actual_isl)

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
        first_token_id, first_token_prob, sweep_results = self._sample(first_token_logits, actual_isl, temperature)

        if return_intermediates:
            intermediates["first_token"] = sweep_results

        return first_token_id, first_token_prob, intermediates

    def _lm_head_and_extract(
        self,
        h: ttnn.Tensor,
        actual_isl: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LM head and extract last-token logits. Topology-aware.

        Args:
            h: Hidden states after final norm
            actual_isl: Count of real tokens in the sequence

        Returns:
            Tuple of (logits_host, first_token_logits)
        """
        if self.padding_side == "right":
            global_token_id = actual_isl - 1
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
        actual_isl: int,
        temperature: Union[float, list[float]],
    ) -> tuple[int, float, list[dict]]:
        """Sample token(s) from extracted logits with temperature sweep.

        Args:
            first_token_logits: Logits for the last real token position
            actual_isl: Count of real tokens (stored in results)
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
                    "actual_isl": actual_isl,
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
